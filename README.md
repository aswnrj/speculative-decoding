# speculative-decoding

My from-scratch implementation of speculative decoding in PyTorch. I wanted to understand how it actually works, so I'm building it piece by piece.

A small model (Llama 3.2 1B) drafts a few tokens, and a bigger model (Llama 3.2 3B) checks them all at once. Tokens that pass are kept. At the first rejected token, the big model picks a replacement instead. You get the same output as running the big model by itself, just faster.

Papers I followed:
- https://arxiv.org/abs/2211.17192
- https://arxiv.org/abs/2302.01318

## Setup

```
pip install torch -r requirements.txt
huggingface-cli login
pytest tests/
```

You need a GPU to run the scripts in `scripts/`.

Still a work in progress.
