# GPT-2 with Rotary Positional Embeddings (RoPE)

This repository contains a **custom modified version of the GPT-2 model** in PyTorch, based on the original HuggingFace Transformers implementation. The major change is that the standard sinusoidal positional encodings are replaced by **Rotary Positional Embeddings (RoPE)** inside the self-attention mechanism.

---

## ✅ What This Project Does

- Uses GPT2 architecture, but changes positional encoding to **RoPE**.
- Implementation is done by modifying the internal GPT2Attention module.
- We perform **few-shot sentiment classification** (prompt-based) on the **SST-2 dataset**.
- The model is initialized from scratch using `GPT2Config()` (no pretrained weights).
- We generate text with a few positive/negative examples and see if the model can guess sentiment of a new example.

---

## 🧠 Why Use RoPE?

Rotary embeddings preserve **relative positions** between tokens and allow better generalization to longer sequences. This technique is used in modern large models like GPT-NeoX, LLaMA, etc.

---

## 📁 Files in this Repository
```
- `modeling_gpt2.py` → Modified GPT-2 model with RoPE integrated into its attention mechanism.
- `rope.py` → Contains the `apply_rope` function used to inject rotary embeddings.
- `run_rope_gpt2.py` → Loads the dataset, builds a few-shot prompt, runs inference using the RoPE-enhanced GPT-2 model.
- `sst2_dataset.csv` → The dataset used for sentiment classification (CSV file with `review` and `sentiment` columns).
- `requirements.txt` → Dependencies for this project.
```
---

## 📊 Dataset (SST-2)

SST-2 (Stanford Sentiment Treebank v2) is a popular benchmark dataset used for sentiment classification. It contains short movie reviews labeled as either **positive** or **negative**.

In this project, only a **small subset of the SST-2 dataset** was used for experimentation, not the full training dataset. The CSV file used here has two columns:
- `review` → the movie review text
- `sentiment` → either "positive" or "negative"

This limited dataset was only used for few-shot inference prompts and not for full training or fine-tuning.

---
