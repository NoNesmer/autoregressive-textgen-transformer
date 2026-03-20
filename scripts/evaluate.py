"""
Evaluate test perplexity and greedy-generation metrics on standard prompts.
"""

import os
import sys
import argparse
import json

import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CharDataset
from src.metrics import compute_perplexity, metrics_for_text
from src.prompts import PROMPTS
from src.inference import load_config, load_vocab, load_model, DATA_DIR
from src.generation import generate


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set and greedy samples")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint .pt")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="Directory for JSON outputs")
    parser.add_argument("--gen_length", type=int, default=512, help="Chars to generate per prompt")
    args = parser.parse_args()

    config = load_config(args.config)
    stoi, itos = load_vocab()
    vocab_size = len(stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config, vocab_size, args.checkpoint, device)

    # Test perplexity (non-overlapping chunks for speed)
    test_dataset = CharDataset(DATA_DIR, "test", config["block_size"], stride=None)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    ppl = compute_perplexity(model, test_loader, device)
    print(f"Test perplexity: {ppl:.4f}")
    print(f"Test cross-entropy (nats): {__import__('math').log(ppl):.4f}")

    block_size = config["block_size"]
    greedy_results = []

    print("\nGreedy generation metrics per prompt:")
    print("-" * 80)
    for prompt in PROMPTS:
        text = generate(
            model, prompt, stoi, itos,
            block_size=block_size,
            max_new_tokens=args.gen_length,
            strategy="greedy",
            temperature=1.0,
            top_k=20,
            device=device,
        )
        m = metrics_for_text(text, rep_n=4)
        greedy_results.append({
            "prompt": prompt,
            "text_preview": text[:200].replace("\n", "\\n"),
            "metrics": m,
        })
        print(f"  [{prompt[:40]}...]" if len(prompt) > 40 else f"  [{prompt}]")
        print(f"    distinct-2: {m['distinct_2']:.4f}  distinct-3: {m['distinct_3']:.4f}  "
              f"repeat_frac: {m['repeat_fraction']:.4f}  first_rep_pos: {m['first_repeat_position']}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "greedy_metrics.json")
    payload = {
        "test_perplexity": ppl,
        "gen_length": args.gen_length,
        "prompts": greedy_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
