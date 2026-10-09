#!/usr/bin/env python3
"""
LoRA SFT for LFM2.5-1.2B-Instruct with Unsloth.

Model loading, LoRA config and trainer settings follow Liquid AI's official recipe
(Liquid4All/cookbook: finetuning/scripts/unsloth-sft-lfm2.5.py); only the data
source (our local JSONL) and local-only saving differ.

Output:
    models/<run-name>/adapter/          LoRA weights + tokenizer
    models/<run-name>/train_summary.json  losses, runtime, peak VRAM, config
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--train-file", type=Path, default=ROOT / "data" / "processed" / "train.jsonl")
    p.add_argument("--val-file", type=Path, default=ROOT / "data" / "processed" / "val.jsonl")
    p.add_argument("--run-name", required=True, help="e.g. smoke, fenrir-3k")
    p.add_argument("--limit", type=int, default=None, help="Use only the first N training rows.")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="Overrides --epochs when > 0.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def main() -> None:
    args = _parse()

    # Unsloth must be imported before transformers/trl so its patches apply.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    out_dir = ROOT / "models" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    data = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "val": str(args.val_file)},
    )
    train_ds, val_ds = data["train"], data["val"]
    if args.limit:
        train_ds = train_ds.select(range(min(args.limit, len(train_ds))))
        val_ds = val_ds.select(range(min(max(args.limit // 10, 20), len(val_ds))))

    def to_text(batch: dict) -> dict:
        texts = tokenizer.apply_chat_template(
            batch["messages"], tokenize=False, add_generation_prompt=False
        )
        # The trainer adds BOS itself; strip it here so it isn't doubled.
        return {"text": [t.removeprefix(tokenizer.bos_token) for t in texts]}

    train_ds = train_ds.map(to_text, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(to_text, batched=True, remove_columns=val_ds.column_names)
    print(f"train rows: {len(train_ds)} | val rows: {len(val_ds)}")

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    eval_every = max(1, total_steps // 5)

    config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=max(1, total_steps // 50),
        eval_strategy="steps",
        eval_steps=eval_every,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="no",
        seed=args.seed,
        report_to="wandb",
        run_name=args.run_name,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=config,
    )
    # Loss only on assistant tokens: the model learns to answer, not to echo questions.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print(f"Training ~{total_steps} optimizer steps (eval every {eval_every})...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    runtime_min = (time.time() - t0) / 60
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    try:
        final_eval = trainer.evaluate()
    except RuntimeError as e:
        print(f"⚠ Final evaluate() failed: {e}")
        final_eval = {"eval_loss": None}

    summary = {
        "run_name": args.run_name,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "optimizer_steps": result.global_step,
        "train_loss": result.training_loss,
        "final_eval_loss": final_eval.get("eval_loss"),
        "runtime_minutes": round(runtime_min, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    for k in ("train_loss", "final_eval_loss", "runtime_minutes", "peak_vram_gb"):
        print(f"{k:>18}: {summary[k]}")
    print(f"{'adapter':>18}: {adapter_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
