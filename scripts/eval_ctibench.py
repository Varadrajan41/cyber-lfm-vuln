#!/usr/bin/env python3
"""
Score a model (optionally + LoRA adapter) on CTIBench MCQA, RCM and VSP.

Scoring:
    MCQA  accuracy of the final answer letter vs GT
    RCM   accuracy of the last CWE ID in the response vs GT
    VSP   1 - MAD/10 over CVSS v3.1 base scores (Cisco Foundation-Sec paper, arXiv:2508.01059);
          an unparseable vector counts as the maximum deviation (10)

Output:
    results/<run-name>/ctibench_<task>.jsonl   every prompt, raw output, parsed answer
    results/<run-name>/ctibench_summary.json   headline numbers
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from cvss31 import base_score, extract_last_vector

ROOT = Path(__file__).resolve().parent.parent

TASKS = {
    "mcq": {"config": "cti-mcq", "max_new_tokens": 256},
    "rcm": {"config": "cti-rcm", "max_new_tokens": 512},
    "vsp": {"config": "cti-vsp", "max_new_tokens": 512},
}

CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)
MCQ_LAST_LINE_RE = re.compile(r"^\(?([ABCD])\)?[.):]?$")
MCQ_PHRASE_RE = re.compile(r"(?:answer|option)\s*(?:is|:)?\s*\**\s*\(?([ABCD])\b", re.IGNORECASE)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir from train_sft.py")
    p.add_argument("--run-name", required=True, help="Results folder name, e.g. baseline, fenrir-3k")
    p.add_argument("--tasks", default="mcq,rcm,vsp")
    p.add_argument("--limit", type=int, default=None, help="Only the first N items per task (quick test).")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def parse_mcq(text: str) -> str | None:
    lines = [l.strip().strip("*#` ").strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        m = MCQ_LAST_LINE_RE.match(lines[-1])
        if m:
            return m.group(1)
    hits = MCQ_PHRASE_RE.findall(text)
    return hits[-1].upper() if hits else None


def parse_rcm(text: str) -> str | None:
    hits = CWE_RE.findall(text)
    return hits[-1].upper() if hits else None


def load_model(model_id: str, adapter: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter) if adapter else model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int) -> tuple[list[str], list[bool]]:
    import torch
    from tqdm import tqdm

    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    # Sort by length so each batch has similar-length prompts (less padding, faster).
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outputs: list[str] = [""] * len(texts)
    truncated: list[bool] = [False] * len(texts)

    for start in tqdm(range(0, len(order), batch_size), desc="generating"):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            toks = new_tokens[j]
            n_real = int((toks != tokenizer.pad_token_id).sum())
            truncated[i] = n_real >= max_new_tokens
            outputs[i] = tokenizer.decode(toks, skip_special_tokens=True)
    return outputs, truncated


def score_task(task: str, rows: list[dict], outputs: list[str]) -> tuple[dict, list[dict]]:
    records = []
    if task == "vsp":
        deviations, unparsed = [], 0
        for row, out in zip(rows, outputs):
            gt_score = base_score(row["GT"])
            pred_vec = extract_last_vector(out)
            pred_score = base_score(pred_vec) if pred_vec else None
            if pred_score is None:
                unparsed += 1
                dev = 10.0
            else:
                dev = abs(pred_score - gt_score)
            deviations.append(dev)
            records.append({"gt": row["GT"], "gt_score": gt_score, "pred": pred_vec,
                            "pred_score": pred_score, "abs_dev": dev})
        mad = sum(deviations) / len(deviations)
        return {"score": round(1 - mad / 10, 4), "mad": round(mad, 4), "unparsed": unparsed,
                "n": len(rows)}, records

    parser = parse_mcq if task == "mcq" else parse_rcm
    correct, unparsed = 0, 0
    for row, out in zip(rows, outputs):
        pred = parser(out)
        gt = str(row["GT"]).strip().upper()
        ok = pred == gt
        correct += ok
        unparsed += pred is None
        records.append({"gt": gt, "pred": pred, "correct": ok})
    return {"score": round(correct / len(rows), 4), "unparsed": unparsed, "n": len(rows)}, records


def main() -> None:
    args = _parse()
    from datasets import load_dataset

    out_dir = ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model, args.adapter)

    summary = {"model": args.model, "adapter": str(args.adapter) if args.adapter else None,
               "decoding": "greedy, repetition_penalty=1.05", "limit": args.limit, "tasks": {}}

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        spec = TASKS[task]
        ds = load_dataset("AI4Sec/cti-bench", spec["config"], split="test")
        rows = [dict(r) for r in ds]
        if args.limit:
            rows = rows[:args.limit]

        print(f"\n== {task.upper()} ({len(rows)} items, max_new_tokens={spec['max_new_tokens']})")
        t0 = time.time()
        outputs, truncated = generate(model, tokenizer, [r["Prompt"] for r in rows],
                                      spec["max_new_tokens"], args.batch_size)
        metrics, records = score_task(task, rows, outputs)
        metrics["truncated"] = sum(truncated)
        metrics["minutes"] = round((time.time() - t0) / 60, 1)
        summary["tasks"][task] = metrics
        print(f"   {metrics}")

        with open(out_dir / f"ctibench_{task}.jsonl", "w") as f:
            for row, out, rec, trunc in zip(rows, outputs, records, truncated):
                f.write(json.dumps({"url": row.get("URL"), **rec, "truncated": trunc,
                                    "output": out}, ensure_ascii=False) + "\n")

    (out_dir / "ctibench_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
