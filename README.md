# Build LLM from Scratch - Practice & Notes

This project contains my practice code and notes while studying the book **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka, available at [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

## Project Overview

This repository implements a GPT-like Large Language Model from scratch using PyTorch, covering the complete pipeline from tokenization to instruction fine-tuning. The project includes:

### 1. Core Components

| File | Description |
|------|-------------|
| `02_tokenizer.py` | BPE (Byte Pair Encoding) tokenizer implementation using tiktoken |
| `03_self_attention.py` | Causal self-attention mechanism with masking |
| `03_multihead_attention.py` | Multi-head attention implementation |
| `04_gpt_model.py` | Complete GPT model architecture with Transformer blocks |
| `05_load_pretrained_weights.py` | Loading OpenAI's GPT-2 pretrained weights |
| `05_train.py` | Training script for next-token prediction |

### 2. Fine-tuning Implementations

| File | Description |
|------|-------------|
| `06_classification_fine_tuning.py` | Text classification fine-tuning (SMS Spam Detection) |
| `07_instruction_fine_tuning.py` | Instruction fine-tuning for following commands |
| `07_evaluate_fine_tuned_llms.py` | LLM-based evaluation using Ollama API |
| `appendix_e_lora.ipynb` | Parameter-efficient fine-tuning with LoRA |

### 3. Training Visualizations

- `loss_plot.png` - Pretraining loss curves
- `fine-tuning-loss-plot.png` - Classification fine-tuning loss
- `fine-tuning-accuracy-plot.png` - Classification accuracy curves
- `lora-loss-plot.png` - LoRA fine-tuning loss curves

## Model Architecture

The implemented GPT model follows the original GPT-2 architecture:
GPTModel
├── Token Embedding (50257 × 768)
├── Position Embedding (1024 × 768)
├── Dropout (p=0.1)
├── Transformer Blocks × 12
│   └── Each Block:
│       ├── LayerNorm
│       ├── Multi-Head Attention
│       │   ├── W_query (768 × 768)
│       │   ├── W_key (768 × 768)
│       │   ├── W_value (768 × 768)
│       │   └── out_proj (768 × 768)
│       ├── Dropout
│       ├── Residual Connection
│       ├── LayerNorm
│       ├── FeedForward
│       │   ├── Linear (768 × 3072)
│       │   ├── GELU Activation
│       │   └── Linear (3072 × 768)
│       ├── Dropout
│       └── Residual Connection
├── Final LayerNorm
└── Output Head (768 × 50257)

Total Parameters: ~124M (GPT-2 small), ~355M (GPT-2 medium)

## Key Features

### 1. From Scratch Implementation
- Complete attention mechanism (causal masking, scaled dot-product)
- Layer normalization and residual connections
- GELU activation function
- Token and positional embeddings

### 2. Pretraining
- Next-token prediction on "The Verdict" text
- Cross-entropy loss optimization
- AdamW optimizer with weight decay

### 3. Fine-tuning Strategies

**Classification Fine-tuning:**
- Task: SMS Spam Detection (binary classification)
- Method: Replace output head, freeze most layers, train last Transformer block
- Accuracy: ~97% on test set

**Instruction Fine-tuning:**
- Task: Follow instructions and generate appropriate responses
- Method: Full model fine-tuning on instruction dataset
- Data format: Instruction + Input + Response

**LoRA Fine-tuning:**
- Task: Same as classification (SMS Spam Detection)
- Method: Low-Rank Adaptation with rank=16
- Trainable parameters: ~2.7M (only 2% of total)
- Efficiency: Faster training, lower memory usage

### 4. Evaluation
- LLM-based evaluation using Ollama (Gemma3/Llama3)
- Automated scoring on 0-100 scale
- Comparison between model outputs and reference answers

## Learning Progress

1. **Chapter 2**: Text tokenization with BPE
2. **Chapter 3**: Attention mechanisms (self-attention → multi-head attention)
3. **Chapter 4**: GPT model architecture implementation
4. **Chapter 5**: Pretraining on unlabeled data
5. **Chapter 6**: Classification fine-tuning
6. **Chapter 7**: Instruction fine-tuning and evaluation
7. **Appendix E**: Parameter-efficient fine-tuning with LoRA

## Requirements

- Python 3.8+
- PyTorch 2.0+
- tiktoken
- pandas
- matplotlib
- tqdm

## Usage

```bash
# Pretraining
python 05_train.py

# Classification fine-tuning
python 06_classification_fine_tuning.py

# Instruction fine-tuning
python 07_instruction_fine_tuning.py

# LoRA fine-tuning (Jupyter notebook)
jupyter notebook appendix_e_lora.ipynb
```

## References

- Book: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
- Official Repository: https://github.com/rasbt/LLMs-from-scratch
- Publisher: Manning Publications