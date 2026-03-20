"""Autoregressive character generation (shared by CLI and evaluation scripts)."""

import torch
from src.sampler import greedy_decode, temperature_sample, topk_sample


@torch.no_grad()
def generate(model, prompt, stoi, itos, block_size, max_new_tokens=200,
             strategy="greedy", temperature=1.0, top_k=20, device="cpu"):
    context = [stoi.get(ch, 0) for ch in prompt.lower()]
    context = torch.tensor([context], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context_window = context[:, -block_size:]
        logits = model(context_window)
        next_logits = logits[0, -1]

        if strategy == "greedy":
            next_token = greedy_decode(next_logits)
        elif strategy == "temperature":
            next_token = temperature_sample(next_logits, temperature)
        elif strategy == "topk":
            next_token = topk_sample(next_logits, top_k, temperature)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        context = torch.cat([context, next_tensor], dim=1)

    return "".join(itos[int(i)] for i in context[0].tolist())
