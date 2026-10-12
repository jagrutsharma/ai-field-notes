---
title: Build a Transformer, Part 1: Self-attention
platform: LinkedIn
status: posted
image: ../build-a-transformer/01-self-attention/heatmap.png
link: https://github.com/jagrutsharma/ai-field-notes/tree/main/build-a-transformer/01-self-attention
posted: 2026-10-11
url: https://lnkd.in/p/gQc-WnPa
---

I recently presented the classic Vaswani et al. paper, "Attention Is All You Need", to a reading group. The equations and flows were very interesting, covering a broad range of concepts. But I really wanted to get a handle on the shapes as they moved through a transformer architecture. So, I worked through the attention equations by hand and later in a notebook. I built a minimal, single-head attention block in PyTorch on a 3-word sentence, using d_model = 4, d_k = 2, d_v = 5, and printed the shape at every step. This really made things click.

Some salient points:
1) Q and K only need to match each other; d_k is independent of everything else.
2) The causal mask sets future positions to -inf, so softmax gives 0 (heatmap attached).
3) V carries its own width (d_v), independent of d_k.
4) W_o maps the output from d_v back to d_model, so it can be added to the input.

I plan to turn this into a series on building a transformer using the paper. I'll build it one piece at a time: multi-head attention, positional encodings, and eventually a small working language model.

Part 1 (runs in Colab): https://github.com/jagrutsharma/ai-field-notes/tree/main/build-a-transformer/01-self-attention

#DeepLearning #PyTorch #Transformers #NLP
