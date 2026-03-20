"""
Quick multi-strategy demo for presentations (greedy + several temperatures + top-k).
"""

import os
import sys
import argparse

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import metrics_for_text
from src.inference import load_config, load_vocab, load_model
from src.generation import generate


def run_block(model, stoi, itos, block_size, device, label, **kwargs):
    text = generate(
        model, kwargs.pop("prompt"), stoi, itos,
        block_size=block_size,
        device=device,
        **kwargs,
    )
    m = metrics_for_text(text, rep_n=4)
    print(f"\n[{label}]")
    print(text[:400] + ("..." if len(text) > 400 else ""))
    print(
        f"Distinct-2: {m['distinct_2']:.3f}  |  Distinct-3: {m['distinct_3']:.3f}  |  "
        f"Repeat frac: {m['repeat_fraction']:.3f}  |  Longest repeat run: {m['longest_repeated_substring']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Demo: compare sampling strategies")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="to be or not to be")
    parser.add_argument("--max_tokens", type=int, default=250)
    args = parser.parse_args()

    config = load_config(args.config)
    stoi, itos = load_vocab()
    vocab_size = len(stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, vocab_size, args.checkpoint, device)
    bs = config["block_size"]

    print(f"Prompt: \"{args.prompt}\"")
    print("=" * 60)

    run_block(
        model, stoi, itos, bs, device, "Greedy",
        prompt=args.prompt, max_new_tokens=args.max_tokens, strategy="greedy",
    )
    for t in (0.7, 1.0, 1.3):
        run_block(
            model, stoi, itos, bs, device, f"Temperature = {t}",
            prompt=args.prompt, max_new_tokens=args.max_tokens,
            strategy="temperature", temperature=t,
        )
    for k in (5, 20, 50):
        run_block(
            model, stoi, itos, bs, device, f"Top-k = {k} (T=1.0)",
            prompt=args.prompt, max_new_tokens=args.max_tokens,
            strategy="topk", top_k=k, temperature=1.0,
        )


if __name__ == "__main__":
    main()
