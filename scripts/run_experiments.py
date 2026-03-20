"""
Run sampling grid: temperature and top-k across standard prompts.
"""

import os
import sys
import argparse
import json
from collections import defaultdict

import torch
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import PROMPTS
from src.metrics import metrics_for_text
from src.inference import load_config, load_vocab, load_model
from src.generation import generate

TEMPERATURES = [0.5, 0.7, 1.0, 1.3, 1.5]
TOP_K_VALUES = [1, 5, 20, 50]
# Joint (temperature, top-k) grid for heatmap (top-k sampling with T > 0)
HEATMAP_TEMPERATURES = [0.7, 1.0, 1.3]
HEATMAP_TOP_K = [5, 20, 50]


def _temp_fname_tag(temp):
    """Safe filename fragment for a float temperature (avoid '.' in basename)."""
    return str(temp).replace(".", "p")


def mean_metrics(samples_metrics):
    if not samples_metrics:
        return {}
    keys = samples_metrics[0].keys()
    return {k: sum(m[k] for m in samples_metrics) / len(samples_metrics) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="Run temperature / top-k experiment grid")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/experiments")
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed for sampling")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Use only first N prompts (for quick tests)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Also run top-k sampling at multiple temperatures for heatmap_metrics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    stoi, itos = load_vocab()
    vocab_size = len(stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_size = config["block_size"]

    model = load_model(config, vocab_size, args.checkpoint, device)

    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    prompts = PROMPTS
    if args.max_prompts is not None:
        prompts = PROMPTS[: max(1, args.max_prompts)]

    raw_results = []
    # For aggregation: key -> list of per-prompt averaged metrics
    temp_agg = defaultdict(list)
    topk_agg = defaultdict(list)

    heatmap_steps = 0
    if args.heatmap:
        heatmap_steps = len(prompts) * len(HEATMAP_TEMPERATURES) * len(HEATMAP_TOP_K)

    total_steps = len(prompts) * (
        len(TEMPERATURES) * args.num_samples + len(TOP_K_VALUES) * args.num_samples
    ) + heatmap_steps
    pbar = tqdm(total=total_steps, desc="Experiments")

    for prompt_idx, prompt in enumerate(prompts):
        for temp in TEMPERATURES:
            sample_metrics = []
            for s in range(args.num_samples):
                torch.manual_seed(args.seed + prompt_idx * 1000 + int(temp * 100) + s)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + prompt_idx * 1000 + s)

                text = generate(
                    model, prompt, stoi, itos,
                    block_size=block_size,
                    max_new_tokens=args.gen_length,
                    strategy="temperature",
                    temperature=temp,
                    top_k=20,
                    device=device,
                )
                m = metrics_for_text(text, rep_n=4)
                sample_metrics.append(m)
                raw_results.append({
                    "prompt": prompt,
                    "strategy": "temperature",
                    "temperature": temp,
                    "sample_index": s,
                    "text": text,
                    "metrics": m,
                })
                fname = f"t{_temp_fname_tag(temp)}_p{prompt_idx}_s{s}.txt"
                with open(os.path.join(samples_dir, fname), "w", encoding="utf-8") as f:
                    f.write(text)
                pbar.update(1)

            avg_m = mean_metrics(sample_metrics)
            temp_agg[temp].append(avg_m)

        for k in TOP_K_VALUES:
            sample_metrics = []
            for s in range(args.num_samples):
                torch.manual_seed(args.seed + prompt_idx * 2000 + k * 10 + s)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + prompt_idx * 2000 + k + s)

                text = generate(
                    model, prompt, stoi, itos,
                    block_size=block_size,
                    max_new_tokens=args.gen_length,
                    strategy="topk",
                    top_k=k,
                    temperature=1.0,
                    device=device,
                )
                m = metrics_for_text(text, rep_n=4)
                sample_metrics.append(m)
                raw_results.append({
                    "prompt": prompt,
                    "strategy": "topk",
                    "top_k": k,
                    "sample_index": s,
                    "text": text,
                    "metrics": m,
                })
                fname = f"k{k}_p{prompt_idx}_s{s}.txt"
                with open(os.path.join(samples_dir, fname), "w", encoding="utf-8") as f:
                    f.write(text)
                pbar.update(1)

            avg_m = mean_metrics(sample_metrics)
            topk_agg[k].append(avg_m)

    # Optional: (T, k) heatmap using top-k sampling with temperature T
    heatmap_cell_metrics = {}
    if args.heatmap:
        for temp in HEATMAP_TEMPERATURES:
            heatmap_cell_metrics[str(temp)] = {}
            for k in HEATMAP_TOP_K:
                per_prompt = []
                for prompt_idx, prompt in enumerate(prompts):
                    torch.manual_seed(
                        args.seed + 9000 + int(temp * 100) + k + prompt_idx * 17
                    )
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(args.seed + 9000 + k + prompt_idx)

                    text = generate(
                        model, prompt, stoi, itos,
                        block_size=block_size,
                        max_new_tokens=args.gen_length,
                        strategy="topk",
                        top_k=k,
                        temperature=temp,
                        device=device,
                    )
                    m = metrics_for_text(text, rep_n=4)
                    per_prompt.append(m)
                    raw_results.append({
                        "prompt": prompt,
                        "strategy": "topk_heatmap",
                        "temperature": temp,
                        "top_k": k,
                        "sample_index": 0,
                        "text": text,
                        "metrics": m,
                    })
                    tag_t = _temp_fname_tag(temp)
                    fname = f"hm_t{tag_t}_k{k}_p{prompt_idx}.txt"
                    with open(os.path.join(samples_dir, fname), "w", encoding="utf-8") as f:
                        f.write(text)
                    pbar.update(1)

                heatmap_cell_metrics[str(temp)][str(k)] = mean_metrics(per_prompt)

    pbar.close()

    def aggregate_over_prompts(list_of_per_prompt_avgs):
        """list_of_per_prompt_avgs: list of dicts with same keys"""
        if not list_of_per_prompt_avgs:
            return {}
        keys = list_of_per_prompt_avgs[0].keys()
        return {key: sum(d[key] for d in list_of_per_prompt_avgs) / len(list_of_per_prompt_avgs)
                for key in keys}

    aggregated = {
        "temperature": {
            str(t): aggregate_over_prompts(temp_agg[t]) for t in TEMPERATURES
        },
        "top_k": {
            str(k): aggregate_over_prompts(topk_agg[k]) for k in TOP_K_VALUES
        },
    }

    raw_path = os.path.join(args.output_dir, "raw_results.json")
    # Raw file can be huge; save compact version without full text for secondary file
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)

    agg_path = os.path.join(args.output_dir, "aggregated_metrics.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    if args.heatmap and heatmap_cell_metrics:
        hm_path = os.path.join(args.output_dir, "heatmap_metrics.json")
        with open(hm_path, "w", encoding="utf-8") as f:
            json.dump(heatmap_cell_metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved heatmap:     {hm_path}")

    print(f"Saved raw results: {raw_path}")
    print(f"Saved aggregated:  {agg_path}")
    print(f"Sample texts dir:  {samples_dir}")


if __name__ == "__main__":
    main()
