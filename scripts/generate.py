import os
import sys
import argparse
import json
import yaml
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CharTransformer
from src.generation import generate

DATA_DIR = "data/processed"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_vocab():
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    itos = {int(v): k for k, v in stoi.items()}
    return stoi, itos


def load_model(config, vocab_size, checkpoint_path, device):
    model = CharTransformer(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        embed_dim=config["d_model"],
        num_heads=config["n_heads"],
        num_layers=config["n_layers"],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text with CharTransformer")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint .pt file")
    parser.add_argument("--prompt", type=str, default="to be or not to be",
                        help="Seed text for generation")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Number of characters to generate")
    parser.add_argument("--strategy", type=str, default="greedy",
                        choices=["greedy", "temperature", "topk"],
                        help="Decoding strategy")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (used with temperature/topk)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="K for top-k sampling")
    args = parser.parse_args()

    config = load_config(args.config)
    stoi, itos = load_vocab()
    vocab_size = len(stoi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, vocab_size, args.checkpoint, device)

    print(f"Strategy: {args.strategy}", end="")
    if args.strategy == "temperature":
        print(f"  (T={args.temperature})")
    elif args.strategy == "topk":
        print(f"  (k={args.top_k}, T={args.temperature})")
    else:
        print()

    print(f"Prompt: \"{args.prompt}\"")
    print("-" * 60)

    text = generate(
        model, args.prompt, stoi, itos,
        block_size=config["block_size"],
        max_new_tokens=args.max_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
