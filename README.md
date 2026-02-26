# SAIL+: Contextual Entailment and Efficient Distillation for Search-Augmented Language Models

**CMU Language Technologies Institute | 11-785 Introduction to Deep Learning**

> Improving search-augmented LLMs with smarter entailment filtering and efficient knowledge distillation — achieving near-SAIL-7B performance at a fraction of the compute cost.

---

## Overview

Large language models (LLMs) suffer from stale knowledge and struggle to reliably filter noisy web search results. This project builds on **SAIL-7B** (Search-Augmented Instruction Learning), a retrieval-augmented LLM grounded in web search, by addressing three core limitations:

| Limitation | Our Solution |
|---|---|
| Knowledge becomes outdated | Search-augmented generation with live retrieval |
| Weak entailment filtering (RoBERTa/DeBERTa binary classifier) | Upgraded to DeBERTa-large + Mistral-7B with chain-of-thought reasoning |
| 7B parameters — too large for edge/mobile deployment | Knowledge distillation into Qwen 0.5B and 1.5B student models |

---

## Key Contributions

### 1. Enhanced Entailment Filtering
Replaced SAIL's default binary classifier with two stronger entailment models to better filter noisy search results:
- **DeBERTa-large** — lightweight, high-accuracy classifier
- **Mistral-7B with chain-of-thought prompting** — reasoning-capable model for ambiguous/noisy contexts

### 2. Knowledge Distillation Framework
Distilled SAIL-7B's search filtering and reasoning capabilities into compact student models:
- **Qwen 0.5B** → 80.4% accuracy / 80.2% F1 on entailment classification
- **Qwen 1.5B** → 85.7% accuracy / 85.1% F1 on entailment classification
- **Qwen 1.5B + Qwen 3B Instruct pipeline** → 60.3% / 60.1% F1 on Climate-Fever (vs. SAIL-7B baseline of 62.4% / 62.2%), with far fewer total parameters

---

## Results at a Glance

### Fine-Tuning: Climate-Fever (Fact-Checking)

| Model | Accuracy | F1 |
|---|---|---|
| SAIL-7B (baseline) | 62.4% | 62.2% |
| SAIL-7B + DeBERTa filtering | 58.5% | 59.0% |
| SAIL-7B + Mistral-7B filtering | **64.5%** | 52.8% |

### Fine-Tuning: Hate Speech Detection

| Model | Accuracy | F1 |
|---|---|---|
| SAIL-7B (baseline) | 76.8% | 59.0% |
| SAIL-7B + DeBERTa filtering | 77.1% | 58.7% |
| SAIL-7B + Mistral-7B filtering | **80.2%** | **60.5%** |

### Knowledge Distillation: Entailment Classification

| Student Model | Accuracy | F1 |
|---|---|---|
| SAIL-7B (teacher baseline) | 62.4% | 62.2% |
| Distilled Qwen 0.5B + Qwen 3B Instruct | 59.2% | 59.0% |
| Distilled Qwen 1.5B + Qwen 3B Instruct | **60.3%** | **60.1%** |

The distilled pipeline achieves **comparable performance to SAIL-7B** while using significantly fewer total parameters, making it viable for edge and mobile deployment.

---

## Technical Stack

| Area | Technologies |
|---|---|
| Base Models | SAIL-7B (LLaMA-7B), Mistral-7B, Qwen 0.5B / 1.5B / 3B |
| Entailment Models | DeBERTa-v3-large, Mistral-7B |
| Fine-Tuning | LoRA (PEFT), DeepSpeed ZeRO-3, bf16 mixed precision |
| Distillation | KL Divergence + Cross-Entropy loss, custom `DistillationTrainer` |
| Retrieval | DuckDuckGo + BM25 (top-5 results per query) |
| Training Data | Alpaca-GPT4 (~52K instruction-response pairs) |
| Evaluation | Climate-Fever, Hate Speech Detection (HSD) |
| Frameworks | PyTorch, Hugging Face Transformers, PEFT, DeepSpeed, Weights & Biases |
| Hardware | NVIDIA A100 GPUs |

---

## Architecture

```
[User Query]
     │
     ▼
[Web Search: DuckDuckGo + BM25] ──► Top-5 Retrieved Passages
     │
     ▼
[Entailment Filter: DeBERTa-large or Mistral-7B]
  → Classify each passage as: Informative | Distracting
     │
     ▼
[Fine-tuned SAIL-7B with LoRA] ──► Final Response
```

**Distillation Pipeline:**
```
[SAIL-7B Teacher]
  → Generates logits + labels over entailment dataset
     │
     ▼
[Student: Qwen 0.5B / 1.5B]
  → Trained on KL(teacher || student) + Cross-Entropy
     │
     ▼
[Qwen 3B Instruct]
  → Takes distilled classifications + generates final answer
```

---

## Fine-Tuning Details

- **LoRA** applied to attention projection matrices (`Q_proj`, `V_proj`) only — base model frozen
- **DeepSpeed ZeRO-3** with 16-step gradient accumulation
- **Cosine LR decay** with 3% warm-up
- **Mixed precision**: bf16 on A100 GPUs
- **Dataset**: 11,200 train / 2,800 validation examples
- **Runtime**: ~6.24 GPU-hours | final training loss: 1.28 | throughput: 1.5 samples/s

---

## Distillation Details

- **Loss**: `L_total = 0.5 * L_KL + 0.5 * L_CE` with temperature T=2.0
- **Tokenizer alignment**: teacher logits decoded → re-encoded with student tokenizer to handle vocabulary mismatch
- **Training**: AdamW + cosine decay, FP16, batch size 4, 3 epochs, eval every 100 steps
- **Custom**: `DistillationTrainer` class extending Hugging Face `Trainer`

---

## Project Structure

```
├── baselines/
│   ├── climate-fever/           # SAIL-7B and LLaMA-7B baseline notebooks
│   └── hate-speech-detection/   # Baseline evaluation notebooks
├── datasets/
│   ├── SAIL_train.json                          # Base training data
│   ├── SAIL_train_Mistral_7B_entailment.json    # Mistral-filtered dataset
│   └── SAIL_train_deberta_entailment.json       # DeBERTa-filtered dataset
├── knowledge_distillation/
│   ├── kd_training.ipynb        # Distillation training notebook (Colab-ready)
│   ├── distillation_script.py   # Fine-tuning based distillation
│   ├── distillation_logits_script.py  # Logits-based distillation
│   ├── kd_dataset.json          # Distillation training data
│   └── teacher_outputs.json     # Pre-computed SAIL-7B teacher outputs
├── finetune.py                  # LoRA fine-tuning script
├── finetune.sh                  # Fine-tuning launch script
├── ds_config.json               # DeepSpeed ZeRO-3 config
├── convert_train_data.py        # Dataset conversion (base)
├── convert_train_data_Mistral-7B.py  # Mistral entailment filtering
├── convert_train_data_deberta.py     # DeBERTa entailment filtering
├── data_prep.sh                 # Data preparation pipeline
├── check_label.py               # Label verification utility
└── requirements.txt
```

---

## Pre-Trained Models

Download our fine-tuned and distilled models from Google Drive:

| Model | Link |
|---|---|
| SAIL-7B finetuned with DeBERTa filtering | [Download](https://drive.google.com/drive/folders/1XIuVksnJXRNewCcNNx8pIe1dv82Kq2-M?usp=sharing) |
| SAIL-7B finetuned with Mistral-7B filtering | [Download](https://drive.google.com/drive/folders/1hQJ78oXWjmPPXkKaUR7c1ygD8X2mMQVt?usp=sharing) |
| Distilled Qwen 0.5B | [Download](https://drive.google.com/drive/folders/1oLty-I3PwMEYWOf6P5IhF-5AIAZR__2p?usp=drive_link) |
| Distilled Qwen 1.5B | [Download](https://drive.google.com/drive/folders/1Gy4ev2wYwVuLOvLrbiEXlYAVSD984Dpn?usp=drive_link) |

---

## Quickstart

### Fine-Tuning

```bash
pip install -r requirements.txt
export WANDB_API_KEY=<YOUR_API_KEY>
bash finetune.sh
```

### Knowledge Distillation (Google Colab)

1. Fork this repository
2. Open `knowledge_distillation/kd_training.ipynb` in Colab
3. Set your GitHub credentials in the first cell
4. Run either:
   - **Logits-based distillation** (KL divergence from teacher logits)
   - **Fine-tuning based distillation** (supervised on teacher outputs)

---

## Evaluation Datasets

| Dataset | Task | Size | Labels |
|---|---|---|---|
| [Climate-Fever](https://www.sustainlab.co/climatefever) | Fact-checking (climate claims) | 1,535 claims / 7,675 evidence pairs | Supports, Refutes |
| [Hate Speech Detection](https://github.com/Vicomtech/hate-speech-dataset) | Fairness / content moderation | 24,783 tweets | Hate Speech, Offensive, Neutral |

---

## Team

**Haojia Sun · Annabelle Min · Bhuvanashree Murugadoss · Ananya Sane**

Language Technologies Institute, Carnegie Mellon University
`{haojias, annabelm, bmurugad, asane}@cs.cmu.edu`

---

## References

- Luo et al. (2023). *SAIL: Search-Augmented Instruction Learning*
- He et al. (2020). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*
- Jiang et al. (2023). *Mistral 7B*
- Bai et al. (2023). *Qwen Technical Report*
- Diggelmann et al. (2020). *Climate-Fever: A Dataset for Verification of Real-World Climate Claims*
- de Gibert et al. (2018). *Hate Speech Dataset from a White Supremacy Forum*
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*
