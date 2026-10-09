#!/usr/bin/env python3
"""
Comprehensive evaluation for paper-grade results.

Runs all CTIBench subtasks the model can handle (RCM, RCM-2021, VSP),
plus multi-output evaluation on held-out data (CWE accuracy, CVSS accuracy,
ATT&CK accuracy) with confidence intervals and per-class breakdown.

Output:
    results/<run-name>/full_eval.json          all metrics in one file
    results/<run-name>/ctibench_rcm.jsonl       per-item RCM results
    results/<run-name>/ctibench_rcm2021.jsonl   per-item RCM-2021 results
    results/<run-name>/ctibench_vsp.jsonl       per-item VSP results
    results/<run-name>/multi_output.jsonl       per-item multi-output results
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from cvss31 import base_score, extract_last_vector

ROOT = Path(__file__).resolve().parent.parent

CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)
ATTACK_RE = re.compile(r"T\d{4}(?:\.\d{3})?")
CVSS_RE = re.compile(
    r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]"
)

# CWE parent→child mappings for hierarchy-aware eval (most common ones from NVD)
CWE_HIERARCHY = {
    "CWE-119": {"CWE-120", "CWE-121", "CWE-122", "CWE-125", "CWE-787"},
    "CWE-20":  {"CWE-129", "CWE-1284"},
    "CWE-200": {"CWE-209", "CWE-532", "CWE-497"},
    "CWE-264": {"CWE-269", "CWE-266", "CWE-250"},
    "CWE-399": {"CWE-401", "CWE-404", "CWE-770", "CWE-772"},
    "CWE-310": {"CWE-327", "CWE-326", "CWE-330"},
    "CWE-189": {"CWE-190", "CWE-191", "CWE-681"},
    "CWE-787": {"CWE-121", "CWE-122"},
    "CWE-284": {"CWE-285", "CWE-862", "CWE-863"},
    "CWE-269": set(),
}


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% confidence interval (no scipy needed, accurate for n≥30)."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96  # z for 95% CI
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return (round(max(0, lo), 4), round(min(1, hi), 4))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% confidence interval (Wilson score method)."""
    return wilson_ci(k, n, alpha)


def hierarchy_match(pred: str, gt: str) -> bool:
    """Check if pred matches gt or is a parent/child of gt."""
    if pred == gt:
        return True
    if gt in CWE_HIERARCHY and pred in CWE_HIERARCHY[gt]:
        return True
    if pred in CWE_HIERARCHY and gt in CWE_HIERARCHY[pred]:
        return True
    return False


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--run-name", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-ctibench", action="store_true",
                   help="Skip CTIBench evals (if already done)")
    p.add_argument("--skip-multioutput", action="store_true",
                   help="Skip multi-output eval on held-out val set")
    return p.parse_args()


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


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int,
             batch_size: int) -> list[str]:
    import torch
    from tqdm import tqdm

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outputs: list[str] = [""] * len(texts)

    for start in tqdm(range(0, len(order), batch_size), desc="generating"):
        idx = order[start : start + batch_size]
        enc = tokenizer(
            [texts[i] for i in idx], return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for j, i in enumerate(idx):
            outputs[i] = tokenizer.decode(new_tokens[j], skip_special_tokens=True)
    return outputs


# ── CTIBench evaluations ─────────────────────────────────────────────────────


def eval_rcm(model, tokenizer, config: str, batch_size: int) -> dict:
    """Evaluate on CTIBench RCM or RCM-2021."""
    from datasets import load_dataset

    ds = load_dataset("AI4Sec/cti-bench", config, split="test")
    rows = [dict(r) for r in ds]
    prompts = [r["Prompt"] for r in rows]

    t0 = time.time()
    outputs = generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=batch_size)
    elapsed = time.time() - t0

    records = []
    correct, correct_hier, unparsed = 0, 0, 0
    per_class: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for row, out in zip(rows, outputs):
        gt = row["GT"].strip().upper()
        hits = CWE_RE.findall(out)
        pred = hits[-1].upper() if hits else None

        ok = pred == gt
        ok_hier = hierarchy_match(pred, gt) if pred else False
        correct += ok
        correct_hier += ok_hier
        unparsed += pred is None

        per_class[gt]["total"] += 1
        if ok:
            per_class[gt]["correct"] += 1

        records.append({"gt": gt, "pred": pred, "correct": ok,
                        "hier_correct": ok_hier, "output": out})

    n = len(rows)
    acc = round(correct / n, 4)
    acc_hier = round(correct_hier / n, 4)
    ci = clopper_pearson(correct, n)

    # Macro F1
    f1s = []
    for cwe, stats in per_class.items():
        tp = stats["correct"]
        fn = stats["total"] - tp
        fp = sum(1 for r in records if r["pred"] == cwe and r["gt"] != cwe)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)

    macro_f1 = round(sum(f1s) / len(f1s), 4) if f1s else 0

    # Top-10 CWE classes by support
    top_classes = sorted(per_class.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
    top_class_acc = {cwe: round(s["correct"] / s["total"], 3) if s["total"] > 0 else 0
                     for cwe, s in top_classes}

    return {
        "metrics": {
            "accuracy": acc,
            "accuracy_95ci": list(ci),
            "accuracy_hierarchy_aware": acc_hier,
            "macro_f1": macro_f1,
            "unparsed": unparsed,
            "n": n,
            "minutes": round(elapsed / 60, 1),
        },
        "per_class_top10": top_class_acc,
        "records": records,
    }


def eval_vsp(model, tokenizer, batch_size: int) -> dict:
    """Evaluate on CTIBench VSP (CVSS vector scoring)."""
    from datasets import load_dataset

    ds = load_dataset("AI4Sec/cti-bench", "cti-vsp", split="test")
    rows = [dict(r) for r in ds]

    # Use our multi-output prompt format instead of CTIBench's prompt
    prompts = [
        f"Assess this vulnerability:\n{r['Description']}\n\n"
        f"Provide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques."
        for r in rows
    ]

    t0 = time.time()
    outputs = generate(model, tokenizer, prompts, max_new_tokens=256, batch_size=batch_size)
    elapsed = time.time() - t0

    records = []
    deviations = []
    exact_match, unparsed = 0, 0
    per_metric_match = Counter()
    per_metric_total = Counter()

    for row, out in zip(rows, outputs):
        gt_vec = row["GT"]
        gt_score = base_score(gt_vec)
        pred_vec = extract_last_vector(out)
        pred_score = base_score(pred_vec) if pred_vec else None

        if pred_score is None:
            unparsed += 1
            dev = 10.0
        else:
            dev = abs(pred_score - gt_score)
            if pred_vec == gt_vec:
                exact_match += 1

            # Per-metric comparison
            gt_parts = dict(p.split(":", 1) for p in gt_vec.split("/")[1:])
            pred_parts = dict(p.split(":", 1) for p in pred_vec.split("/")[1:])
            for metric in ("AV", "AC", "PR", "UI", "S", "C", "I", "A"):
                per_metric_total[metric] += 1
                if gt_parts.get(metric) == pred_parts.get(metric):
                    per_metric_match[metric] += 1

        deviations.append(dev)
        records.append({
            "gt_vec": gt_vec, "gt_score": gt_score,
            "pred_vec": pred_vec, "pred_score": pred_score,
            "abs_dev": round(dev, 2), "output": out,
        })

    n = len(rows)
    mad = sum(deviations) / n
    vsp_score = 1 - mad / 10

    per_metric_acc = {m: round(per_metric_match[m] / per_metric_total[m], 4)
                      if per_metric_total[m] > 0 else 0
                      for m in ("AV", "AC", "PR", "UI", "S", "C", "I", "A")}

    return {
        "metrics": {
            "vsp_score": round(vsp_score, 4),
            "mad": round(mad, 4),
            "exact_vector_match": exact_match,
            "exact_vector_match_pct": round(exact_match / n, 4),
            "unparsed": unparsed,
            "n": n,
            "minutes": round(elapsed / 60, 1),
        },
        "per_metric_accuracy": per_metric_acc,
        "records": records,
    }


# ── Multi-output evaluation on held-out val set ─────────────────────────────


def eval_multioutput(model, tokenizer, batch_size: int) -> dict:
    """Evaluate CWE, CVSS, and ATT&CK accuracy on the held-out val set."""
    val_path = ROOT / "data" / "processed-50k" / "val.jsonl"
    if not val_path.exists():
        val_path = ROOT / "data" / "processed" / "val.jsonl"
    if not val_path.exists():
        print("  ⚠ No val.jsonl found, skipping multi-output eval")
        return {}

    rows = [json.loads(l) for l in val_path.read_text().splitlines() if l.strip()]
    print(f"  Multi-output eval on {len(rows)} val examples")

    # Extract ground truth from assistant messages
    gt_data = []
    prompts = []
    for row in rows:
        user_msg = row["messages"][0]["content"]
        asst_msg = row["messages"][1]["content"]

        gt_cwe = CWE_RE.search(asst_msg)
        gt_cvss = CVSS_RE.search(asst_msg)
        gt_attack = ATTACK_RE.findall(asst_msg)

        gt_data.append({
            "cwe": gt_cwe.group(0).upper() if gt_cwe else None,
            "cvss": gt_cvss.group(0) if gt_cvss else None,
            "attack": sorted(set(gt_attack)),
        })
        # Use just the user prompt (the vulnerability description part)
        prompts.append(user_msg)

    t0 = time.time()
    outputs = generate(model, tokenizer, prompts, max_new_tokens=256, batch_size=batch_size)
    elapsed = time.time() - t0

    records = []
    cwe_correct, cwe_hier_correct, cwe_total = 0, 0, 0
    cvss_exact, cvss_total, cvss_unparsed = 0, 0, 0
    cvss_deviations = []
    attack_precision_sum, attack_recall_sum, attack_total = 0.0, 0.0, 0

    for gt, out in zip(gt_data, outputs):
        pred_cwe_m = CWE_RE.findall(out)
        pred_cwe = pred_cwe_m[0].upper() if pred_cwe_m else None
        pred_cvss = extract_last_vector(out)
        pred_attack = sorted(set(ATTACK_RE.findall(out)))

        rec = {"gt_cwe": gt["cwe"], "pred_cwe": pred_cwe,
               "gt_cvss": gt["cvss"], "pred_cvss": pred_cvss,
               "gt_attack": gt["attack"], "pred_attack": pred_attack,
               "output": out}

        # CWE
        if gt["cwe"]:
            cwe_total += 1
            if pred_cwe == gt["cwe"]:
                cwe_correct += 1
            if pred_cwe and hierarchy_match(pred_cwe, gt["cwe"]):
                cwe_hier_correct += 1

        # CVSS
        if gt["cvss"]:
            cvss_total += 1
            if pred_cvss is None:
                cvss_unparsed += 1
                cvss_deviations.append(10.0)
            else:
                gt_score = base_score(gt["cvss"])
                pred_score = base_score(pred_cvss)
                if gt_score is not None and pred_score is not None:
                    cvss_deviations.append(abs(gt_score - pred_score))
                if pred_cvss == gt["cvss"]:
                    cvss_exact += 1
                rec["cvss_score_dev"] = round(abs((gt_score or 0) - (pred_score or 0)), 2)

        # ATT&CK
        if gt["attack"]:
            attack_total += 1
            gt_set = set(gt["attack"])
            pred_set = set(pred_attack)
            if pred_set:
                p = len(gt_set & pred_set) / len(pred_set)
                r = len(gt_set & pred_set) / len(gt_set)
                attack_precision_sum += p
                attack_recall_sum += r

        records.append(rec)

    n = len(rows)
    cvss_mad = sum(cvss_deviations) / len(cvss_deviations) if cvss_deviations else 0

    return {
        "metrics": {
            "n": n,
            "minutes": round(elapsed / 60, 1),
            "cwe_accuracy": round(cwe_correct / cwe_total, 4) if cwe_total else 0,
            "cwe_accuracy_hierarchy": round(cwe_hier_correct / cwe_total, 4) if cwe_total else 0,
            "cwe_95ci": list(clopper_pearson(cwe_correct, cwe_total)),
            "cvss_exact_match": round(cvss_exact / cvss_total, 4) if cvss_total else 0,
            "cvss_mad": round(cvss_mad, 4),
            "cvss_vsp_score": round(1 - cvss_mad / 10, 4) if cvss_deviations else 0,
            "cvss_unparsed": cvss_unparsed,
            "attack_precision": round(attack_precision_sum / attack_total, 4) if attack_total else 0,
            "attack_recall": round(attack_recall_sum / attack_total, 4) if attack_total else 0,
        },
        "records": records,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse()

    out_dir = ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.adapter)

    full_eval: dict = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "decoding": "greedy, repetition_penalty=1.05",
    }

    if not args.skip_ctibench:
        # CTIBench RCM (2023-2024)
        print("\n== CTIBench RCM (cti-rcm, 1000 items)")
        rcm = eval_rcm(model, tokenizer, "cti-rcm", args.batch_size)
        print(f"   accuracy: {rcm['metrics']['accuracy']}  "
              f"CI: {rcm['metrics']['accuracy_95ci']}  "
              f"hier: {rcm['metrics']['accuracy_hierarchy_aware']}  "
              f"macro_f1: {rcm['metrics']['macro_f1']}")
        full_eval["ctibench_rcm"] = rcm["metrics"]
        full_eval["ctibench_rcm_per_class_top10"] = rcm["per_class_top10"]
        with open(out_dir / "ctibench_rcm.jsonl", "w") as f:
            for r in rcm["records"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # CTIBench RCM-2021 (2011-2021)
        print("\n== CTIBench RCM-2021 (cti-rcm-2021, 1000 items)")
        rcm21 = eval_rcm(model, tokenizer, "cti-rcm-2021", args.batch_size)
        print(f"   accuracy: {rcm21['metrics']['accuracy']}  "
              f"CI: {rcm21['metrics']['accuracy_95ci']}  "
              f"hier: {rcm21['metrics']['accuracy_hierarchy_aware']}  "
              f"macro_f1: {rcm21['metrics']['macro_f1']}")
        full_eval["ctibench_rcm_2021"] = rcm21["metrics"]
        full_eval["ctibench_rcm_2021_per_class_top10"] = rcm21["per_class_top10"]
        with open(out_dir / "ctibench_rcm2021.jsonl", "w") as f:
            for r in rcm21["records"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # CTIBench VSP (CVSS scoring) — using our prompt format
        print("\n== CTIBench VSP (cti-vsp, 1000 items, our prompt format)")
        vsp = eval_vsp(model, tokenizer, args.batch_size)
        print(f"   VSP score: {vsp['metrics']['vsp_score']}  "
              f"MAD: {vsp['metrics']['mad']}  "
              f"exact: {vsp['metrics']['exact_vector_match_pct']}  "
              f"unparsed: {vsp['metrics']['unparsed']}")
        print(f"   per-metric acc: {vsp['per_metric_accuracy']}")
        full_eval["ctibench_vsp"] = vsp["metrics"]
        full_eval["ctibench_vsp_per_metric"] = vsp["per_metric_accuracy"]
        with open(out_dir / "ctibench_vsp.jsonl", "w") as f:
            for r in vsp["records"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not args.skip_multioutput:
        print("\n== Multi-output evaluation (held-out val set)")
        mo = eval_multioutput(model, tokenizer, args.batch_size)
        if mo:
            m = mo["metrics"]
            print(f"   CWE acc: {m['cwe_accuracy']}  CI: {m['cwe_95ci']}  "
                  f"hier: {m['cwe_accuracy_hierarchy']}")
            print(f"   CVSS exact: {m['cvss_exact_match']}  "
                  f"MAD: {m['cvss_mad']}  VSP: {m['cvss_vsp_score']}")
            print(f"   ATT&CK prec: {m['attack_precision']}  "
                  f"recall: {m['attack_recall']}")
            full_eval["multioutput"] = m
            with open(out_dir / "multi_output.jsonl", "w") as f:
                for r in mo["records"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (out_dir / "full_eval.json").write_text(json.dumps(full_eval, indent=2))
    print(f"\n{'='*60}")
    print(f"All results saved to {out_dir}/full_eval.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
