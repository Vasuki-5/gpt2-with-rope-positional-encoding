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

