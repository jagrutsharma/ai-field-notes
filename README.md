# AI Field Notes

Hands-on notes, demos, and code from exploring AI, from papers to production.

Each folder is a small, self-contained entry: I build something to understand it, then write up what clicked.

## Series

| Series | What it covers | Progress |
|---|---|---|
| [Build a Transformer](build-a-transformer/) | A Transformer from scratch in PyTorch, one idea per part, following *Attention Is All You Need* (2017) | Part 1 of 7 |

## Concepts

Standalone deep dives into one idea each.

| Concept | What it covers |
|---|---|
| [Why Softmax?](concepts/softmax/) | The properties that make softmax so useful, and why attention divides by √d_k |

## Related work

- [inference-benchmark-lab](https://github.com/jagrutsharma/inference-benchmark-lab): single-GPU LLM inference optimization
- [multinode-llm-serving](https://github.com/jagrutsharma/multinode-llm-serving): distributed LLM inference on Kubernetes
