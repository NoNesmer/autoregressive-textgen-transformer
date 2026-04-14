"""
Phase 3: fixed human-evaluation sample pack (baseline model only).

Writes:
  - output_dir/samples/heval_*.txt
  - output_dir/manifest.json  (metrics + decoding metadata; for correlate_human_eval.py)
  - output_dir/ratings.csv      (template with empty Likert columns)

Seeds match scripts/run_experiments.py so generations align with that grid
when using the same --seed, --gen_length, and checkpoint.
"""

import argparse
import csv
import json
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import PROMPTS
from src.metrics import metrics_for_text
from src.inference import load_config, load_vocab, load_model
from src.generation import generate

# 15 fixed rows: (sample_id, prompt_idx, strategy, ...). Matches run_experiments seeding.
# Rows 1–10: temperature sampling T=1.0, sample_index=0, all ten prompts.
# Rows 11–15: top-k sampling k=20, T=1.0, sample_index=0, first five prompts.
HUMAN_EVAL_MANIFEST = [
    *[(i + 1, i, "temperature", {"temperature": 1.0, "top_k": 20, "sample_index": 0}) for i in range(10)],
    *[(i + 11, i, "topk", {"top_k": 20, "temperature": 1.0, "sample_index": 0}) for i in range(5)],
]


def _set_seed_temperature(base_seed, prompt_idx, temp, s, cuda):
    torch.manual_seed(base_seed + prompt_idx * 1000 + int(temp * 100) + s)
    if cuda:
        torch.cuda.manual_seed_all(base_seed + prompt_idx * 1000 + s)


def _set_seed_topk(base_seed, prompt_idx, k, s, cuda):
    torch.manual_seed(base_seed + prompt_idx * 2000 + k * 10 + s)
    if cuda:
        torch.cuda.manual_seed_all(base_seed + prompt_idx * 2000 + k + s)


BIAS_DISCLOSURE = (
    "Self-rated by the author; possible confirmation bias. "
    "Automatic metrics (distinct-3, repeat fraction) are listed for analysis only."
)


def main():
    parser = argparse.ArgumentParser(description="Prepare Phase-3 human eval sample pack (baseline)")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/baseline/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="results/human_eval")
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="Must match run_experiments --seed for identical texts")
    args = parser.parse_args()

    config = load_config(args.config)
    stoi, itos = load_vocab()
    vocab_size = len(stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_size = config["block_size"]
    cuda = device.type == "cuda"

    model = load_model(config, vocab_size, args.checkpoint, device)

    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    manifest = []
    csv_rows = []

    for sample_id, prompt_idx, strategy, params in HUMAN_EVAL_MANIFEST:
        if prompt_idx >= len(PROMPTS):
            raise ValueError(f"prompt_idx {prompt_idx} out of range for PROMPTS (len={len(PROMPTS)})")
        prompt = PROMPTS[prompt_idx]
        s = params["sample_index"]

        if strategy == "temperature":
            temp = params["temperature"]
            top_k_arg = int(params.get("top_k", 20))
            _set_seed_temperature(args.seed, prompt_idx, temp, s, cuda)
            text = generate(
                model,
                prompt,
                stoi,
                itos,
                block_size=block_size,
                max_new_tokens=args.gen_length,
                strategy="temperature",
                temperature=temp,
                top_k=top_k_arg,
                device=device,
            )
            temp_used = float(temp)
            top_k_decode = top_k_arg
        elif strategy == "topk":
            k = int(params["top_k"])
            temp = float(params["temperature"])
            _set_seed_topk(args.seed, prompt_idx, k, s, cuda)
            text = generate(
                model,
                prompt,
                stoi,
                itos,
                block_size=block_size,
                max_new_tokens=args.gen_length,
                strategy="topk",
                top_k=k,
                temperature=temp,
                device=device,
            )
            temp_used = temp
            top_k_decode = k
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        m = metrics_for_text(text, rep_n=4)
        fname = f"heval_{sample_id:02d}_p{prompt_idx}_{strategy}.txt"
        rel_text = os.path.join("samples", fname)
        abs_text = os.path.join(samples_dir, fname)
        with open(abs_text, "w", encoding="utf-8") as f:
            f.write(text)

        manifest.append(
            {
                "sample_id": sample_id,
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "strategy": strategy,
                "temperature": temp_used,
                "top_k": top_k_decode,
                "sample_index": s,
                "gen_length": args.gen_length,
                "seed_base": args.seed,
                "text_file": rel_text.replace("\\", "/"),
                "metrics": m,
            }
        )

        csv_rows.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "strategy": strategy,
                "temperature": temp_used,
                "top_k": top_k_decode,
                "gen_length": args.gen_length,
                "text_file": rel_text.replace("\\", "/"),
                "distinct_2": f"{m['distinct_2']:.6f}",
                "distinct_3": f"{m['distinct_3']:.6f}",
                "repeat_fraction": f"{m['repeat_fraction']:.6f}",
                "first_repeat_position": m.get("first_repeat_position", -1),
                "bias_disclosure": BIAS_DISCLOSURE,
                "likert_coherence": "",
                "likert_diversity": "",
                "likert_overall": "",
                "notes": "",
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    ratings_path = os.path.join(args.output_dir, "ratings.csv")
    fieldnames = [
        "sample_id",
        "prompt",
        "strategy",
        "temperature",
        "top_k",
        "gen_length",
        "text_file",
        "distinct_2",
        "distinct_3",
        "repeat_fraction",
        "first_repeat_position",
        "bias_disclosure",
        "likert_coherence",
        "likert_diversity",
        "likert_overall",
        "notes",
    ]
    with open(ratings_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"Wrote {len(manifest)} samples to {samples_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Ratings template: {ratings_path}")
    print("Likert (1–5): likert_coherence = plausible Shakespeare-like continuation;")
    print("likert_diversity = subjectively non-repetitive / varied;")
    print("likert_overall = overall quality.")


if __name__ == "__main__":
    main()
