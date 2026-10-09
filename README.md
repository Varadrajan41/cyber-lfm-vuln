# Multi-Output Vulnerability Assessment with LFM2.5-1.2B

Fine-tune [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) (a hybrid SSM+attention model) for multi-output vulnerability assessment. Given a CVE description, the model produces **CWE ID + CVSS v3.1 vector + MITRE ATT&CK techniques** in a single pass.

## Results

| Model | Params | CTIBench RCM (95% CI) | CTIBench RCM-2021 |
|-------|--------|-----------------------|-------------------|
| Sec-Gemini v1 (Google) | closed | ~86% | — |
| **This model** | **1.2B** | **75.2% (72.4–77.8%)** | **70.7%** |
| RoBERTa-base (xamxte) | 125M | 75.6% (72.8–78.2%) | 71.8% |
| Foundation-Sec-8B-R (Cisco) | 8B | 75.3% (72.5–77.9%) | — |
| GPT-4 | ~1.7T | 72.0% | — |
| Base LFM2.5-1.2B (no FT) | 1.2B | 18.0% | 20.1% |

**CVSS vector prediction**: 0.897 VSP score, 41.8% exact vector match, 0 unparsed outputs.  
**ATT&CK techniques**: 83.5% precision, 83.5% recall on held-out data.

## Links

- **Model (LoRA adapter)**: [Varadrajan/LFM2.5-1.2B-CyberVuln](https://huggingface.co/Varadrajan/LFM2.5-1.2B-CyberVuln)
- **Dataset**: [Varadrajan/cve-vuln-assessment](https://huggingface.co/datasets/Varadrajan/cve-vuln-assessment)
- **Base model**: [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- **Source dataset**: [xamxte/cve-to-cwe](https://huggingface.co/datasets/xamxte/cve-to-cwe)
- **Benchmark**: [CTIBench (NeurIPS 2024)](https://arxiv.org/abs/2406.07599)

## Quick Start

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, "Varadrajan/LFM2.5-1.2B-CyberVuln")
tokenizer = AutoTokenizer.from_pretrained("Varadrajan/LFM2.5-1.2B-CyberVuln")

messages = [{"role": "user", "content": (
    "Assess this vulnerability:\n"
    "In the Linux kernel through 6.7.1, there is a use-after-free in "
    "cec_queue_msg_fh.\n\n"
    "Provide: CWE ID, CVSS v3.1 vector, MITRE ATT&CK techniques."
)}]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True).to(model.device)
output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# CWE: CWE-416
# CVSS: CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H
# ATT&CK: T1068
```

### Reproduce from scratch

```bash
# 1. Fetch CVSS vectors from NVD (or download from HF dataset)
NVD_API_KEY=your-key python scripts/fetch_nvd_bulk.py

# 2. Prepare training data
python scripts/prepare_data.py --n-train 50000 --n-val 2000 --max-seq-length 1024

# 3. Train (98 min on RTX 4060)
LFM=$(ls -d ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/*/)
WANDB_PROJECT=cyber-lfm-vuln python scripts/train_sft.py \
  --model "$LFM" --run-name vuln-full --max-seq-length 1024 --epochs 1

# 4. Evaluate
python scripts/eval_full.py --adapter models/vuln-full/adapter --run-name vuln-full-paper

# 5. Baseline comparison
python scripts/eval_full.py --run-name baseline-paper --skip-multioutput
```

## Project Structure

```
scripts/
├── prepare_data.py      # Dataset construction + decontamination
├── fetch_nvd_bulk.py    # Bulk CVSS fetch from NVD API v2.0
├── train_sft.py         # LoRA SFT training with Unsloth
├── eval_full.py         # Comprehensive eval (CTIBench + multi-output)
├── eval_ctibench.py     # CTIBench-only eval
├── cvss31.py            # CVSS v3.1 base score calculator
└── chat.py              # Interactive base vs fine-tuned comparison

results/
├── vuln-full-paper/     # Fine-tuned model eval results
│   └── full_eval.json   # All metrics
└── baseline-paper/      # Base model eval results
    └── full_eval.json

models/
└── vuln-full/
    ├── adapter/         # LoRA adapter (upload to HuggingFace)
    └── train_summary.json
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | LiquidAI/LFM2.5-1.2B-Instruct |
| LoRA rank | 16 (alpha=16, 11.1M trainable params) |
| Training rows | 50,000 |
| Epochs | 1 |
| Learning rate | 2e-4 (cosine) |
| Precision | bfloat16 |
| Hardware | NVIDIA RTX 4060 Laptop (8 GB) |
| Training time | 98 minutes |
| Peak VRAM | 2.98 GB |
| Train loss | 0.102 |
| Eval loss | 0.076 |

## Data Scaling

| Rows | CTIBench RCM | Time |
|------|-------------|------|
| 2k | 68.8% | 14 min |
| 50k | 75.2% | 98 min |
| 100k | 75.0% | 194 min |

## License

- Code: Apache-2.0
- Model adapter: Apache-2.0
- Training data: CC-BY-4.0 (following [xamxte/cve-to-cwe](https://huggingface.co/datasets/xamxte/cve-to-cwe))
