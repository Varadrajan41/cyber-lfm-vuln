# Paper Reference Guide — Multi-Output Vulnerability Assessment with LFM2.5-1.2B

> This document maps every artifact in `cyber-lfm-vuln/` to the arXiv paper
> section where it should be cited or discussed. An agent writing the paper
> should read this first.

---

## 1. Proposed Paper Metadata

- **Working title**: "Multi-Output Vulnerability Assessment with a Small Hybrid-Architecture Language Model"
- **Authors**: Varadrajan [last name], [affiliation]
- **arXiv categories**: cs.CR (primary), cs.CL (cross-list)
- **Key claim**: A 1.2B hybrid SSM+attention model (LFM2.5) fine-tuned with LoRA produces CWE ID + CVSS v3.1 vector + ATT&CK techniques in a single pass, matching 8B and 125M specialists on CTIBench RCM while also generating structured CVSS and ATT&CK outputs that prior work does not evaluate.

---

## 2. Paper Outline → Code/Data Mapping

### Abstract

Key numbers to include:
- 75.2% CTIBench RCM accuracy (95% CI: 72.4–77.8%) — matches RoBERTa-125M (75.6%) and Cisco Foundation-Sec-8B-R (75.3%)
- 70.7% on CTIBench RCM-2021
- 0.897 VSP score for CVSS prediction, 41.8% exact vector match
- 83.5% ATT&CK precision and recall
- 1.2B parameters, 43MB LoRA adapter, 98 min training on single RTX 4060 laptop GPU
- 50,000 training examples from enriched xamxte/cve-to-cwe dataset

Source: `results/vuln-full-paper/full_eval.json`

---

### §1 Introduction

**Motivation**: Existing CVE→CWE work (RoBERTa classifiers, Foundation-Sec-8B) treats CWE mapping as a standalone classification task. Real vulnerability triage requires CWE + CVSS + ATT&CK together. A single generative model doing all three is more practical.

**Contributions** (4 items):
1. Multi-output formulation (CWE + CVSS + ATT&CK) as a single generative task
2. First published fine-tune of Liquid Foundation Models (hybrid SSM+attention) for cybersecurity
3. Dataset enrichment: combined xamxte/cve-to-cwe with 177,298 NVD CVSS vectors
4. Comprehensive evaluation with confidence intervals, hierarchy-aware scoring, per-metric CVSS accuracy, and ATT&CK precision/recall

**No code files needed** — this is narrative.

---

### §2 Related Work

Papers to cite:
- **xamxte/cve-to-cwe paper** (Mosievskiy, 2026): arxiv 2603.14911 — RoBERTa-125M, 75.6% CTIBench RCM
- **CTIBench** (Alam et al., 2024): NeurIPS 2024, the benchmark we evaluate on
- **Foundation-Sec-8B** (Kassianik et al., 2025): Cisco, 75.3% on CTIBench RCM
- **Liquid Foundation Models** (Liquid AI, 2024): LFM architecture paper — hybrid SSM+attention
- **Luca's cyber-cve2cwe-0.6b**: HuggingFace model, Qwen3-0.6B, CWE-only output on same dataset
- **V2W-BERT** (Das et al., 2021): earlier CVE→CWE work, data leakage concerns
- **Sec-Gemini v1** (Google, 2025): ~86% on CTIBench, closed/proprietary
- **LoRA** (Hu et al., 2022): PEFT method we use
- **Unsloth**: training framework

**No code files needed** — this is narrative + citations.

---

### §3 Dataset Construction

#### §3.1 Source Data

- Base dataset: `xamxte/cve-to-cwe` (234,770 train rows, 205 CWE classes, 361 ATT&CK techniques)
- License: CC-BY-4.0

Code: `scripts/prepare_data.py` (lines 59–70 for decontamination, lines 75–95 for format_example)

#### §3.2 CVSS Vector Enrichment

- Bulk-fetched CVSS v3.1 vectors from NVD API v2.0 for all 234,770 CVE IDs
- Result: 177,298 CVEs with CVSS vectors, 57,472 without
- Stored in: `data/nvd_cache/cvss_vectors.json` (13 MB)

Code: `scripts/fetch_nvd_bulk.py` (the bulk pagination approach)
Also: `scripts/fetch_nvd_cvss.py` (the initial per-CVE approach, replaced by bulk)

#### §3.3 Training Format

Each example is a chat-format message pair:
```
User:  "Assess this vulnerability:\n{CVE description}\n\nProvide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques."
Asst:  "CWE: CWE-79\nCVSS: CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:L\nATT&CK: T1059.007"
```

Code: `scripts/prepare_data.py` (lines 75–95, `format_example` function)
Sample data: `data/processed-50k/train.jsonl` (first few lines show the format)

#### §3.4 Decontamination

- Removed all 2,022 CVE IDs appearing in CTIBench test sets (cti-rcm, cti-rcm-2021, cti-vsp)
- Removed 68 rows with 10-gram overlap against CTIBench MCQ questions
- Total dropped: 68 MCQA overlaps + 5 CVE overlaps + 334 too-long + 17,058 no-CVSS

Code: `scripts/prepare_data.py` (lines 59–70, `_ctibench_blocklist` function)
Report: `data/processed-50k/prep_report.json`

#### §3.5 Final Dataset Statistics

| Split | Rows | Median tokens | Total tokens |
|-------|------|---------------|--------------|
| Train | 50,000 | 155 | 8,984,161 |
| Val   | 2,000  | 155 | (included above) |

Source: `data/processed-50k/prep_report.json`

---

### §4 Method

#### §4.1 Base Model

- **LiquidAI/LFM2.5-1.2B-Instruct**: 1.18B parameters, hybrid SSM+attention architecture
- Liquid Foundation Models use a mix of state-space model (SSM) layers and standard attention layers
- Instruct-tuned variant with chat template

#### §4.2 LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 16 |
| Dropout | 0 |
| Target modules | q_proj, k_proj, v_proj, out_proj, in_proj, w1, w2, w3 |
| Trainable params | 11,108,352 (0.94% of 1.18B) |
| PEFT version | 0.19.1 |

Source: `models/vuln-full/adapter/adapter_config.json`

#### §4.3 Training Details

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 1 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Max sequence length | 1024 |
| Precision | bfloat16 |
| Seed | 3407 |
| Response masking | train_on_responses_only |
| Hardware | NVIDIA RTX 4060 Laptop (8 GB) |
| Training time | 98.5 minutes |
| Peak VRAM | 2.98 GB |
| Optimizer steps | 6,250 |
| Final train loss | 0.102 |
| Final eval loss | 0.076 |

Source: `models/vuln-full/train_summary.json`
Code: `scripts/train_sft.py`

#### §4.4 Decoding

- Greedy decoding (do_sample=False)
- repetition_penalty=1.05
- max_new_tokens varies by task (512 for RCM, 256 for VSP/multi-output)

Code: `scripts/eval_full.py` (generate function, lines ~130–155)

---

### §5 Experiments & Results

#### §5.1 CTIBench CWE Mapping (Table 1 — main results)

Source: `results/vuln-full-paper/full_eval.json` and `results/baseline-paper/full_eval.json`

**Table 1: CTIBench RCM results (strict exact CWE match)**

| Model | Params | cti-rcm (95% CI) | cti-rcm-2021 (95% CI) |
|-------|--------|-------------------|------------------------|
| Sec-Gemini v1 (Google) | closed | ~86% | — |
| Ours (LFM2.5 + LoRA) | 1.2B | 75.2% (72.4–77.8%) | 70.7% (67.8–73.4%) |
| RoBERTa-base (xamxte) | 125M | 75.6% (72.8–78.2%) | 71.8% |
| Foundation-Sec-8B-R (Cisco) | 8B | 75.3% (72.5–77.9%) | — |
| GPT-4 | ~1.7T | 72.0% | — |
| Base LFM2.5-1.2B (no FT) | 1.2B | 18.0% (15.7–20.5%) | 20.1% (17.7–22.7%) |

Note: Our CIs overlap with RoBERTa and Cisco — statistically indistinguishable.

**Hierarchy-aware evaluation:**
- cti-rcm: 75.2% strict → 80.6% hierarchy-aware (+5.4pp)
- cti-rcm-2021: 70.7% strict → 78.0% hierarchy-aware (+7.3pp)

**Per-class accuracy (top 10 by support, cti-rcm):**

| CWE | Accuracy |
|-----|----------|
| CWE-79 (XSS) | 98.7% |
| CWE-89 (SQLi) | 96.4% |
| CWE-352 (CSRF) | 100% |
| CWE-434 (Unrestricted Upload) | 100% |
| CWE-78 (OS Command Injection) | 100% |
| CWE-125 (Out-of-bounds Read) | 90.5% |
| CWE-416 (Use-After-Free) | 90.0% |
| CWE-862 (Missing Authorization) | 86.4% |
| CWE-22 (Path Traversal) | 82.8% |
| CWE-787 (Out-of-bounds Write) | 24.2% ← hierarchy issue |

Source: `results/vuln-full-paper/full_eval.json` → `ctibench_rcm_per_class_top10`

CWE-787 is the outlier (24.2%) for the same reason as in the RoBERTa paper: the model predicts specific children (CWE-121/CWE-122) while ground truth uses generic parent CWE-787.

Code: `scripts/eval_full.py` (eval_rcm function)

#### §5.2 CVSS Vector Scoring (Table 2 — unique to this paper)

**Table 2: CTIBench VSP — CVSS v3.1 vector prediction**

| Model | VSP Score | MAD | Exact Match | Unparsed |
|-------|-----------|-----|-------------|----------|
| Ours (LFM2.5 + LoRA) | 0.897 | 1.03 | 41.8% | 0/1000 |
| Base LFM2.5-1.2B | 0.002 | 9.98 | 0.0% | 998/1000 |

**Per-metric CVSS accuracy (Table 3):**

| AV | AC | PR | UI | S | C | I | A |
|----|----|----|----|----|---|---|---|
| 90.7% | 95.5% | 71.9% | 91.8% | 90.6% | 78.0% | 81.2% | 79.2% |

Source: `results/vuln-full-paper/full_eval.json` → `ctibench_vsp` and `ctibench_vsp_per_metric`
Code: `scripts/eval_full.py` (eval_vsp function)
Also: `scripts/cvss31.py` (CVSS v3.1 base score calculator, FIRST specification)

#### §5.3 Multi-Output Evaluation on Held-Out Data (Table 4)

**Table 4: Multi-output accuracy on 2,000 held-out validation examples**

| Output | Metric | Score |
|--------|--------|-------|
| CWE | Accuracy | 81.7% (CI: 80.0–83.4%) |
| CWE | Hierarchy-aware | 83.1% |
| CVSS | Exact vector match | 43.4% |
| CVSS | VSP score | 0.915 |
| CVSS | Mean absolute deviation | 0.85 |
| ATT&CK | Precision | 83.5% |
| ATT&CK | Recall | 83.5% |

Source: `results/vuln-full-paper/full_eval.json` → `multioutput`
Data: `data/processed-50k/val.jsonl` (the 2,000 held-out examples)
Code: `scripts/eval_full.py` (eval_multioutput function)

#### §5.4 Data Scaling Analysis (Table 5 / Figure)

| Training Rows | cti-rcm Accuracy | Train Loss | Training Time |
|---------------|------------------|------------|---------------|
| 2,000 | 68.8% | 0.147 | 14 min |
| 50,000 | 75.2% | 0.102 | 98 min |
| 100,000 | 75.0% | 0.091 | 194 min |

Source:
- `models/vuln-smoke-2k/train_summary.json` + `results/vuln-smoke-2k/ctibench_summary.json`
- `models/vuln-full/train_summary.json` + `results/vuln-full-paper/full_eval.json`
- `models/vuln-100k/train_summary.json` + `results/vuln-100k/ctibench_summary.json`

This shows diminishing returns: 2k→50k gives +6.4pp, but 50k→100k gives -0.2pp (within noise). The plateau at ~75% suggests a ceiling in the benchmark/task, not data quantity.

Possible figure: line chart of accuracy vs training rows (log scale x-axis).

---

### §6 Discussion

#### §6.1 CWE Hierarchy Granularity

Same issue as RoBERTa paper: CWE-787 accuracy is 24.2% because the model predicts CWE-121/CWE-122 (specific children). Hierarchy-aware: 80.6%.

Code: `scripts/eval_full.py` (hierarchy_match function and CWE_HIERARCHY dict)

#### §6.2 Multi-Output vs Single-Output

Key argument: RoBERTa (125M) and our model both achieve ~75% on CWE, but:
- RoBERTa outputs ONLY a CWE label
- Our model outputs CWE + CVSS + ATT&CK in one pass
- 0 unparsed CVSS vectors (vs 998/1000 for base model) proves the model learned structured output format

#### §6.3 Hybrid Architecture

LFM2.5 uses hybrid SSM+attention layers. This is the first cybersecurity fine-tune of this architecture family. The 2.98 GB peak VRAM is notable — lighter than transformer-only models of similar parameter count.

#### §6.4 Limitations

- Single CWE per CVE (some vulnerabilities have multiple)
- CVSS vectors trained from NVD labels which may disagree with vendor assessments
- ATT&CK mapping is AI-generated (from Claude Sonnet 4.6 in the source dataset)
- English only
- CWE hierarchy analysis covers only common parent-child pairs
- No comparison with other base models (e.g., Qwen, LLaMA) fine-tuned the same way

---

### §7 Conclusion

- 1.2B hybrid model matches 8B general-purpose and 125M specialist models on CWE mapping
- First multi-output vulnerability assessment (CWE + CVSS + ATT&CK) from a single small model
- 41.8% exact CVSS vector match with 0.897 VSP score
- 83.5% ATT&CK precision/recall
- 98 minutes training, 2.98 GB VRAM, single laptop GPU

---

### References Section

Use BibTeX. Key references:

```bibtex
@misc{mosievskiy2026roberta,
  title={Fine-tuning RoBERTa for CVE-to-CWE Classification: A 125M Parameter Model Competitive with LLMs},
  author={Mosievskiy, Nikita},
  year={2026},
  eprint={2603.14911},
  archivePrefix={arXiv}
}

@inproceedings{alam2024ctibench,
  title={CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence},
  author={Alam, M. T. and Bhusal, D. and Nguyen, L. and Rastogi, N.},
  booktitle={NeurIPS},
  year={2024}
}

@misc{kassianik2025foundationsec,
  title={Llama-3.1-FoundationAI-SecurityLLM-Base-8B Technical Report},
  author={Kassianik, P. and others},
  year={2025},
  eprint={2504.21039},
  archivePrefix={arXiv}
}

@misc{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and others},
  year={2022},
  eprint={2106.09685},
  archivePrefix={arXiv}
}
```

Also cite: LFM architecture paper (Liquid AI), Google Sec-Gemini blog post, NVD/CVSS/MITRE ATT&CK official sources.

---

## 3. File Index — What to include when releasing code

### Core scripts (include in paper's GitHub repo)

| File | Description | Paper section |
|------|-------------|---------------|
| `scripts/prepare_data.py` | Dataset construction + decontamination | §3 |
| `scripts/fetch_nvd_bulk.py` | CVSS vector enrichment from NVD API | §3.2 |
| `scripts/train_sft.py` | LoRA SFT training | §4 |
| `scripts/eval_full.py` | All evaluations (CTIBench + multi-output) | §5 |
| `scripts/eval_ctibench.py` | Simpler CTIBench-only eval | §5.1 |
| `scripts/cvss31.py` | CVSS v3.1 base score calculator | §5.2 |
| `scripts/chat.py` | Interactive comparison (base vs fine-tuned) | Appendix/demo |

### Results (include as supplementary or in paper tables)

| File | Description | Paper section |
|------|-------------|---------------|
| `results/vuln-full-paper/full_eval.json` | **All fine-tuned model metrics** | §5 (Tables 1-4) |
| `results/baseline-paper/full_eval.json` | **All baseline metrics** | §5 (Tables 1-2) |
| `results/vuln-full-paper/ctibench_rcm.jsonl` | Per-item RCM predictions | Appendix |
| `results/vuln-full-paper/ctibench_rcm2021.jsonl` | Per-item RCM-2021 predictions | Appendix |
| `results/vuln-full-paper/ctibench_vsp.jsonl` | Per-item VSP predictions | Appendix |
| `results/vuln-full-paper/multi_output.jsonl` | Per-item multi-output predictions | Appendix |

### Training artifacts (reference but don't include in paper)

| File | Description |
|------|-------------|
| `models/vuln-full/adapter/` | Trained LoRA adapter (43 MB) — upload to HuggingFace |
| `models/vuln-full/train_summary.json` | Training hyperparams + loss curve |
| `data/nvd_cache/cvss_vectors.json` | 177k CVSS vectors — upload as HF dataset |
| `data/processed-50k/prep_report.json` | Dataset statistics |
| `data/processed-50k/train.jsonl` | 50k training examples |
| `data/processed-50k/val.jsonl` | 2k validation examples |
| `logs/train-vuln-full.log` | Full training log |

### Scaling experiment data

| File | Description |
|------|-------------|
| `models/vuln-smoke-2k/train_summary.json` | 2k training summary |
| `results/vuln-smoke-2k/ctibench_summary.json` | 2k eval: 68.8% |
| `models/vuln-100k/train_summary.json` | 100k training summary |
| `results/vuln-100k/ctibench_summary.json` | 100k eval: 75.0% |

### W&B runs (link in paper)

| Run | W&B URL |
|-----|---------|
| vuln-smoke-2k | https://wandb.ai/varadrajan123-genzeon/cyber-lfm-vuln/runs/idghvw43 |
| vuln-full (50k) | https://wandb.ai/varadrajan123-genzeon/cyber-lfm-vuln/runs/aps0qhd8 |
| vuln-100k | https://wandb.ai/varadrajan123-genzeon/cyber-lfm-vuln/runs/nqg9cdx1 |

---

## 4. Figures to Create for the Paper

1. **Bar chart**: CTIBench RCM comparison (our model vs RoBERTa vs Cisco vs GPT-4 vs base) with error bars
2. **Line chart**: Data scaling curve (2k → 50k → 100k rows vs accuracy)
3. **Heatmap or bar chart**: Per-metric CVSS accuracy (AV, AC, PR, UI, S, C, I, A)
4. **Bar chart**: Per-CWE accuracy on top-10 classes
5. **Example figure**: Side-by-side base vs fine-tuned output on a sample CVE (from chat.py output)

---

## 5. Reproducibility Checklist

To reproduce all results from scratch:

```bash
# 1. Fetch CVSS vectors
NVD_API_KEY=xxx python scripts/fetch_nvd_bulk.py

# 2. Prepare 50k training data
python scripts/prepare_data.py --n-train 50000 --n-val 2000 --max-seq-length 1024

# 3. Train
LFM=$(ls -d ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/*/)
WANDB_PROJECT=cyber-lfm-vuln python scripts/train_sft.py \
  --model "$LFM" --run-name vuln-full --max-seq-length 1024 --epochs 1

# 4. Evaluate
python scripts/eval_full.py --adapter models/vuln-full/adapter --run-name vuln-full-paper

# 5. Baseline
python scripts/eval_full.py --run-name baseline-paper --skip-multioutput
```

Hardware: NVIDIA RTX 4060 Laptop GPU (8 GB), WSL2 Ubuntu, ~2 hours total.
