---
base_model: LiquidAI/LFM2.5-1.2B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- en
tags:
- cybersecurity
- vulnerability-assessment
- cve
- cwe
- cvss
- mitre-attack
- lora
- sft
- transformers
- unsloth
- liquid-ai
- hybrid-ssm
datasets:
- xamxte/cve-to-cwe
---

# LFM2.5-1.2B-CyberVuln

A LoRA adapter that turns [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) (a hybrid SSM+attention model) into a **multi-output vulnerability assessment** specialist. Given a CVE description, the model produces three outputs in a single pass:

- **CWE ID** — weakness classification (e.g., CWE-416)
- **CVSS v3.1 vector** — severity scoring (e.g., CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H)
- **MITRE ATT&CK techniques** — adversary tactics (e.g., T1190, T1068)

## Benchmark Results

### CTIBench CWE Mapping (CVE→CWE)

| Model | Params | cti-rcm (95% CI) | cti-rcm-2021 |
|-------|--------|-------------------|--------------|
| Sec-Gemini v1 (Google) | closed | ~86% | — |
| **This model** | **1.2B** | **75.2% (72.4–77.8%)** | **70.7%** |
| RoBERTa-base ([xamxte](https://arxiv.org/abs/2603.14911)) | 125M | 75.6% (72.8–78.2%) | 71.8% |
| Foundation-Sec-8B-R (Cisco) | 8B | 75.3% (72.5–77.9%) | — |
| GPT-4 | ~1.7T | 72.0% | — |
| Base LFM2.5-1.2B (no fine-tuning) | 1.2B | 18.0% | 20.1% |

### CVSS Vector Prediction (CTIBench VSP)

| Metric | Score |
|--------|-------|
| VSP Score | 0.897 |
| Mean Absolute Deviation | 1.03 |
| Exact vector match | 41.8% |
| Unparsed outputs | 0/1000 |

**Per-metric CVSS accuracy:**

| AV | AC | PR | UI | S | C | I | A |
|----|----|----|----|----|---|---|---|
| 90.7% | 95.5% | 71.9% | 91.8% | 90.6% | 78.0% | 81.2% | 79.2% |

### Multi-Output (held-out 2,000 examples)

| Output | Metric | Score |
|--------|--------|-------|
| CWE | Accuracy | 81.7% |
| CVSS | Exact match | 43.4% |
| CVSS | VSP score | 0.915 |
| ATT&CK | Precision | 83.5% |
| ATT&CK | Recall | 83.5% |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model + adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = PeftModel.from_pretrained(base_model, "Varadrajan/LFM2.5-1.2B-CyberVuln")
tokenizer = AutoTokenizer.from_pretrained("Varadrajan/LFM2.5-1.2B-CyberVuln")

# Build prompt
messages = [{"role": "user", "content": (
    "Assess this vulnerability:\n"
    "In the Linux kernel through 6.7.1, there is a use-after-free in "
    "cec_queue_msg_fh, related to drivers/media/cec/core/cec-adap.c "
    "and drivers/media/cec/core/cec-api.c.\n\n"
    "Provide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques."
)}]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True
).to(model.device)

output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Expected output:**
```
CWE: CWE-416
CVSS: CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H
ATT&CK: T1068
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | LiquidAI/LFM2.5-1.2B-Instruct |
| Method | LoRA (rank 16, alpha 16) |
| Target modules | q_proj, k_proj, v_proj, out_proj, in_proj, w1, w2, w3 |
| Trainable params | 11.1M (0.94% of 1.18B) |
| Training data | 50,000 examples from [xamxte/cve-to-cwe](https://huggingface.co/datasets/xamxte/cve-to-cwe) enriched with NVD CVSS vectors |
| Epochs | 1 |
| Effective batch size | 8 (2 × 4 grad accum) |
| Learning rate | 2e-4 (cosine schedule) |
| Optimizer | AdamW 8-bit |
| Max sequence length | 1024 |
| Precision | bfloat16 |
| Hardware | NVIDIA RTX 4060 Laptop GPU (8 GB) |
| Training time | 98 minutes |
| Peak VRAM | 2.98 GB |
| Train loss | 0.102 |
| Eval loss | 0.076 |
| Framework | Unsloth + Transformers + PEFT |

### Data Construction

1. Started with [xamxte/cve-to-cwe](https://huggingface.co/datasets/xamxte/cve-to-cwe) (234,770 CVE→CWE+ATT&CK mappings)
2. Bulk-fetched CVSS v3.1 vectors from NVD API v2.0 for all CVE IDs → 177,298 with CVSS
3. Decontaminated against all CTIBench test sets (removed 68 MCQA n-gram overlaps + 5 CVE ID overlaps)
4. Final: 50,000 train + 2,000 validation examples

### Data Scaling

| Training Rows | CTIBench RCM | Train Loss | Time |
|---------------|-------------|------------|------|
| 2,000 | 68.8% | 0.147 | 14 min |
| 50,000 | 75.2% | 0.102 | 98 min |
| 100,000 | 75.0% | 0.091 | 194 min |

Diminishing returns beyond 50k rows — the 50k model is the best efficiency/performance trade-off.

## What Makes This Different

Unlike existing CVE→CWE classifiers (RoBERTa, Foundation-Sec) which output only a CWE label, this model produces **three structured outputs in one generative pass**: CWE + CVSS + ATT&CK. This is closer to what a security analyst actually needs for vulnerability triage.

The base model is a **hybrid SSM+attention architecture** (Liquid Foundation Model), not a standard transformer. This is the first published cybersecurity fine-tune of this architecture family.

## Limitations

- Single CWE per CVE (some vulnerabilities involve multiple weakness types)
- CVSS vectors trained from NVD labels which may not match vendor assessments
- ATT&CK technique mappings are AI-generated (via Claude Sonnet 4.6 in the source dataset)
- English CVE descriptions only
- Outputs are advisory — validate against NVD/MITRE before operational use

## Citation

```bibtex
@misc{varadrajan2026cybervuln,
  title={LFM2.5-1.2B-CyberVuln: Multi-Output Vulnerability Assessment with a Hybrid Architecture Language Model},
  author={Varadrajan},
  year={2026},
  howpublished={\url{https://huggingface.co/Varadrajan/LFM2.5-1.2B-CyberVuln}}
}
```

## Acknowledgments

- [xamxte/cve-to-cwe](https://huggingface.co/datasets/xamxte/cve-to-cwe) dataset by Nikita Mosievskiy
- [LiquidAI](https://www.liquid.ai/) for the LFM2.5 base model
- [CTIBench](https://arxiv.org/abs/2406.07599) benchmark by Alam et al.
- [Unsloth](https://github.com/unslothai/unsloth) for efficient training
- NVD/NIST for CVSS vector data
