#!/usr/bin/env python3
"""
Build multi-output vulnerability assessment training data.

Combines xamxte/cve-to-cwe (CWE + ATT&CK) with NVD CVSS vectors
into chat-format JSONL for LoRA SFT.

Each training example:
    User:  "Assess this vulnerability: <CVE description>
            Provide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques."
    Asst:  "CWE: CWE-79
            CVSS: CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:L
            ATT&CK: T1059.007"

Output:
    data/processed/train.jsonl
    data/processed/val.jsonl
    data/processed/prep_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
NGRAM = 10


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cvss-cache", type=Path, default=ROOT / "data" / "nvd_cache" / "cvss_vectors.json",
                   help="JSON file from fetch_nvd_cvss.py")
    p.add_argument("--tokenizer", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--n-train", type=int, default=50000,
                   help="Max training rows. 0 = all available.")
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--max-seq-length", type=int, default=1024,
                   help="Shorter than Fenrir — structured outputs are concise.")
    p.add_argument("--require-cvss", action="store_true", default=True,
                   help="Only include rows that have a CVSS vector.")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    return p.parse_args()


def _normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _ngrams(words: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _ctibench_blocklist() -> tuple[set[str], set[tuple[str, ...]]]:
    """CVE IDs and n-grams from CTIBench test sets for decontamination."""
    from datasets import load_dataset

    cves: set[str] = set()
    for cfg in ("cti-rcm", "cti-rcm-2021", "cti-vsp"):
        for row in load_dataset("AI4Sec/cti-bench", cfg, split="test"):
            cves.update(c.upper() for c in CVE_RE.findall(json.dumps(row)))

    mcq_grams: set[tuple[str, ...]] = set()
    for q in load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")["Question"]:
        mcq_grams |= _ngrams(_normalize(q))

    return cves, mcq_grams


def format_example(description: str, cwe_id: str, cvss_vector: str, attack_techniques: list[str]) -> dict:
    """Build a chat-format training example."""
    user_msg = (
        f"Assess this vulnerability:\n{description}\n\n"
        f"Provide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques."
    )

    attack_str = ", ".join(attack_techniques) if attack_techniques else "None identified"
    assistant_msg = (
        f"CWE: {cwe_id}\n"
        f"CVSS: {cvss_vector}\n"
        f"ATT&CK: {attack_str}"
    )

    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main() -> None:
    args = _parse()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if not args.cvss_cache.exists():
        raise SystemExit(f"CVSS cache not found: {args.cvss_cache}\nRun fetch_nvd_cvss.py first.")

    print("Loading CVSS cache...")
    cvss_map: dict[str, str | None] = json.loads(args.cvss_cache.read_text())
    has_cvss = sum(1 for v in cvss_map.values() if v is not None)
    print(f"  {len(cvss_map)} CVEs in cache, {has_cvss} with CVSS vectors")

    print("Loading CTIBench blocklist for decontamination...")
    blocked_cves, mcq_grams = _ctibench_blocklist()
    print(f"  {len(blocked_cves)} blocked CVE IDs, {len(mcq_grams)} MCQA n-grams")

    print("Loading xamxte/cve-to-cwe...")
    ds = load_dataset("xamxte/cve-to-cwe", split="train").shuffle(seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    need = args.n_train + args.n_val if args.n_train > 0 else len(ds)
    kept: list[dict] = []
    dropped: Counter[str] = Counter()
    seen_cves: set[str] = set()
    token_lengths: list[int] = []

    for row in ds:
        if args.n_train > 0 and len(kept) >= need:
            break

        cve_id = str(row["cve_id"]).strip()
        description = str(row["description"]).strip()
        cwe_id = str(row["cwe_id"]).strip()
        attack_raw = row.get("attack_techniques", [])
        if isinstance(attack_raw, str):
            try:
                attack_techniques = json.loads(attack_raw.replace("'", '"'))
            except json.JSONDecodeError:
                attack_techniques = []
        else:
            attack_techniques = list(attack_raw) if attack_raw else []

        if not description or not cwe_id:
            dropped["empty_field"] += 1
            continue

        if cve_id in seen_cves:
            dropped["duplicate_cve"] += 1
            continue

        cvss_vector = cvss_map.get(cve_id)
        if args.require_cvss and cvss_vector is None:
            dropped["no_cvss"] += 1
            continue

        if cve_id.upper() in blocked_cves:
            dropped["ctibench_cve_overlap"] += 1
            continue

        if _ngrams(_normalize(description)) & mcq_grams:
            dropped["ctibench_mcqa_overlap"] += 1
            continue

        example = format_example(description, cwe_id, cvss_vector, attack_techniques)

        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens > args.max_seq_length:
            dropped["too_long"] += 1
            continue

        seen_cves.add(cve_id)
        token_lengths.append(n_tokens)
        kept.append(example)

    if len(kept) < args.n_val + 100:
        raise SystemExit(f"Only {len(kept)} clean rows found, need at least {args.n_val + 100}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_count = min(args.n_val, len(kept) // 10)
    splits = {"val": kept[:val_count], "train": kept[val_count:]}

    for name, rows in splits.items():
        with open(args.out_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    token_lengths.sort()
    report = {
        "dataset": "xamxte/cve-to-cwe + NVD CVSS",
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "require_cvss": args.require_cvss,
        "rows_scanned": len(kept) + sum(dropped.values()),
        "kept": {k: len(v) for k, v in splits.items()},
        "dropped": dict(dropped),
        "tokens_per_example": {
            "min": token_lengths[0],
            "median": token_lengths[len(token_lengths) // 2],
            "p95": token_lengths[int(len(token_lengths) * 0.95)],
            "max": token_lengths[-1],
            "total_tokens": sum(token_lengths),
        },
    }
    (args.out_dir / "prep_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out_dir}/train.jsonl and val.jsonl")


if __name__ == "__main__":
    main()
