#!/usr/bin/env python3
"""
Side-by-side answers: base model vs base + LoRA adapter, same prompts, same decoding.

    python scripts/chat.py --adapter models/fenrir-3k/adapter
    python scripts/chat.py --adapter models/fenrir-3k/adapter --prompt "What is CWE-79?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_PROMPTS = [
    # Multi-output vulnerability assessment (the task we're training for)
    "Assess this vulnerability:\n"
    "In the Linux kernel through 6.7.1, there is a use-after-free in cec_queue_msg_fh, "
    "related to drivers/media/cec/core/cec-adap.c and drivers/media/cec/core/cec-api.c.\n\n"
    "Provide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques.",

    "Assess this vulnerability:\n"
    "The WP JobHunt plugin for WordPress is vulnerable to unauthorized modification of data "
    "due to a missing capability check on the cs_update_application_status_callback function.\n\n"
    "Provide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques.",

    "Assess this vulnerability:\n"
    "IBM OpenPages with Watson 8.3 and 9.0 could provide weaker than expected security using "
    "Native authentication. An attacker with access to the database could exploit this weakness.\n\n"
    "Provide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques.",

    # General security question (checks broad knowledge survived)
    "What is the difference between authentication and authorization?",

    # Non-security (checks general capability)
    "Write a haiku about autumn.",
]


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--prompt", action="append", help="Repeatable. Defaults to a built-in set.")
    p.add_argument("--max-new-tokens", type=int, default=400)
    return p.parse_args()


def main() -> None:
    args = _parse()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter))
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, str(args.adapter)).eval()

    def answer(prompt: str) -> str:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.05)
        return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    for prompt in args.prompt or DEFAULT_PROMPTS:
        print("\n" + "#" * 80 + f"\nPROMPT: {prompt}\n" + "#" * 80)
        with model.disable_adapter():
            print("\n--- BASE ---\n" + answer(prompt))
        print("\n--- FINE-TUNED ---\n" + answer(prompt))


if __name__ == "__main__":
    main()
