---
title: Why Softmax?
platform: LinkedIn
status: posted
images:
  - ../concepts/softmax/saturation.png
  - ../concepts/softmax/temperature.png
link: https://github.com/jagrutsharma/ai-field-notes/tree/main/concepts/softmax
posted: 2026-10-14
url: https://lnkd.in/p/g9ZWQcqV
---

While I was presenting the "Attention Is All You Need" paper in a reading group, softmax came up a few times. It was noted as a very elegant function. It is small, but it shows up everywhere: attention weights, classifier outputs, and LLM sampling. I really got curious and wanted to understand why it works so well. So I wrote a short notebook with one small demo for each property. It is a side note to my Build a Transformer series.

Some points that stood out:

1. It is a "soft" argmax. The biggest score still wins, but every score keeps some weight, so a small change in any score moves the output a little. That is what lets a model learn. Argmax only gives a yes or no.

2. Temperature is just dividing the scores by T before softmax. Small T gets close to argmax, large T gets close to equal weights. This is the same temperature setting used in LLM sampling (chart attached).

3. Adding the same number to every score doesn't change the result. Real implementations use this: they subtract the max score first, so exp never overflows.

4. Its weakness: when scores are big, softmax becomes almost one-hot and the gradient goes to nearly zero. In attention, a score is a dot product of d_k numbers, so it grows like √d_k. With d_k = 512, the gradient was 3e-11 on raw scores and 0.24 after dividing by √d_k (chart attached). That is why the paper scales by √d_k.

Notebook (runs in Colab): https://github.com/jagrutsharma/ai-field-notes/tree/main/concepts/softmax

#DeepLearning #PyTorch #Transformers #NLP
