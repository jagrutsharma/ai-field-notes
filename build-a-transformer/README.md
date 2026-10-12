# Build a Transformer

A step-by-step series that builds a Transformer from scratch in PyTorch, following [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Each part is one small, runnable notebook that adds one idea on top of the last. Tensor shapes are printed at every step, and every dimension is deliberately different, so it's always clear where each shape comes from.

## Parts

| Part | Topic | Paper section | Notebook | Status |
|---|---|---|---|---|
| 1 | [Single-head self-attention + causal mask](01-self-attention/) | 3.2.1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jagrutsharma/ai-field-notes/blob/main/build-a-transformer/01-self-attention/attention-demo.ipynb) | ✅ |
| 2 | Multi-head attention | 3.2.2 | | Planned |
| 3 | Positional encodings | 3.5 | | Planned |
| 4 | The full block: Add & Norm + feed-forward | 3.1, 3.3 | | Planned |
| 5 | A tiny character-level language model | 3.4, 5 | | Planned |
| 6 | Trained attention, visualized | Figs. 3–5 | | Planned |
| 7 | Bonus: the KV cache at inference time | | | Planned |

## Running locally

```bash
uv venv --python 3.12
uv pip install -r requirements.txt   # from the repo root
```

Or open any notebook in Colab with the badge; no setup needed.
