# GPT-2 with Rotary Positional Embedding (RoPE)

This repository contains a **custom modified version of HuggingFace's GPT-2** where the standard sinusoidal position embeddings are replaced with **Rotary Positional Embeddings (RoPE)** inside the self-attention mechanism.

This is NOT a wrapper around HuggingFace's GPT2Model — instead, we edit the internal Transformer code directly (GPT2Attention, GPT2MLP, etc.) and apply RoPE to the Query / Key activations.

---

## 🧠 What was changed

- Based on HuggingFace’s original GPT2 implementation (`transformers` library).
- Imported a custom `apply_rope` function (`from rope import apply_rope`).
- Applied the rotation to query/key inside `GPT2Attention` module.
- The rest of the transformer architecture (MLP, LayerNorm, FeedForward) remains unchanged.

---

## ✨ Why RoPE?

Rotary Positional Embedding injects positional information by rotating the hidden states in multi-dimensional space. It:
- supports extrapolation to longer sequence lengths
- preserves relative positions naturally
- is used in modern models like GPT-NeoX, LLaMA, etc.

---

## 📦 Dependencies

Add this to your `requirements.txt`:

