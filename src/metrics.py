"""Evaluation metrics: perplexity, distinct-n, repetition statistics."""

import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_perplexity(model, dataloader, device):
    """
    Perplexity = exp(average cross-entropy per token).
    Uses sum reduction then divides by total tokens for consistency with train.py.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        b, t, c = logits.shape
        nll = F.cross_entropy(
            logits.view(b * t, c),
            y.view(b * t),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += b * t
    if total_tokens == 0:
        return float("nan")
    avg_loss = total_nll / total_tokens
    return math.exp(avg_loss)


def compute_distinct_n(text, n):
    """Ratio of unique character n-grams to total n-grams. Returns 0 if text too short."""
    if n <= 0 or len(text) < n:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def compute_repetition_stats(text, n=4):
    """
    first_repeat_position: start index of first n-gram that appeared earlier (-1 if none).
    repeat_fraction: fraction of n-gram positions that are repeats (seen before).
    longest_repeated_substring: max length of a substring that occurs at least twice.
    """
    if len(text) < n:
        return {
            "first_repeat_position": -1,
            "repeat_fraction": 0.0,
            "longest_repeated_substring": 0,
        }

    seen = {}
    first_repeat_position = -1
    repeat_count = 0
    total = len(text) - n + 1

    for i in range(total):
        gram = text[i : i + n]
        if gram in seen:
            repeat_count += 1
            if first_repeat_position < 0:
                first_repeat_position = i
        else:
            seen[gram] = i

    repeat_fraction = repeat_count / total if total else 0.0

    # Longest contiguous substring that occurs at least twice (pairwise LCP)
    best = 0
    L = len(text)
    for i in range(L):
        for j in range(i + 1, L):
            k = 0
            while i + k < L and j + k < L and text[i + k] == text[j + k]:
                k += 1
            best = max(best, k)

    return {
        "first_repeat_position": first_repeat_position,
        "repeat_fraction": repeat_fraction,
        "longest_repeated_substring": best,
    }


def metrics_for_text(text, rep_n=4):
    """Bundle distinct-2, distinct-3, and repetition stats for one string."""
    rep = compute_repetition_stats(text, n=rep_n)
    return {
        "distinct_2": compute_distinct_n(text, 2),
        "distinct_3": compute_distinct_n(text, 3),
        "first_repeat_position": rep["first_repeat_position"],
        "repeat_fraction": rep["repeat_fraction"],
        "longest_repeated_substring": rep["longest_repeated_substring"],
    }
