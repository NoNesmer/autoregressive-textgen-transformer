import torch
import torch.nn.functional as F


def greedy_decode(logits):
    """Select the token with the highest logit."""
    return torch.argmax(logits).item()


def temperature_sample(logits, temperature=1.0):
    """Scale logits by temperature, then sample from the resulting distribution."""
    if temperature <= 0:
        return greedy_decode(logits)
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def topk_sample(logits, k=20, temperature=1.0):
    """Keep only the top-k logits, zero out the rest, then temperature-sample."""
    if k <= 0:
        k = logits.size(-1)
    scaled = logits / max(temperature, 1e-8)
    top_values, top_indices = torch.topk(scaled, min(k, scaled.size(-1)))
    filtered = torch.full_like(scaled, float("-inf"))
    filtered.scatter_(-1, top_indices, top_values)
    probs = F.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
