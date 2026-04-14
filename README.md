# Autoregressive Character-Level Transformer

Character-level **GPT-style** causal Transformer on the Project Gutenberg Shakespeare corpus. The codebase supports a **baseline** and a **larger** model (`configs/baseline.yaml`, `configs/large.yaml`), systematic **temperature** and **top‑k** sampling experiments, optional **T × k** heatmaps, **baseline vs. large** overlay plots with **Spearman** correlations, and a small **human-evaluation** pilot (`scripts/prepare_human_eval.py`, `scripts/correlate_human_eval.py`). The written summary is `final_report.md` (figures under `docs/figures/`).

---

## Reproducibility (read this first)

**What “reproducible” means here**

1. **From the same checkpoints** (same `best_model.pt` files): anyone can **bit‑reproduce** evaluation JSON, experiment JSON, and plots **given the same PyTorch / CUDA stack**; sampling uses fixed seeds in `scripts/run_experiments.py` (`--seed`, default `42`) and in human-eval prep.
2. **Retraining from scratch**: `scripts/train.py` fixes **Python / NumPy / PyTorch** seeds and uses a **deterministic** `DataLoader` shuffle (`torch.Generator` + `seed` from YAML), and sets `torch.backends.cudnn.deterministic = True`. **Exact** weight match across **different GPUs or library versions** is still **not** guaranteed (CUDA numerics); you should expect **very close** loss curves and **the same qualitative** sampling tables.

**Checkpoints and git**

- **`*.pt` checkpoints are not ignored** — `results/checkpoints/**/best_model.pt` (and any other `.pt` you keep) can be committed so a **clone** reproduces evaluation and sampling without retraining.
- Per-epoch files `model_epoch_*.pt` are optional to keep or commit (larger); for hand-ins, committing **`best_model.pt` only** per run is usually enough.
- To reproduce **only from source** without using committed weights: follow **Pipeline (order matters)** below; training time depends on GPU (CPU is possible but slow).

**Recommended environment**

- **Python 3.10+** (3.12 used in development).
- Install: `pip install -r requirements.txt`
- For your own audit, run once and save: `pip freeze > requirements-lock.txt` and attach that file to submissions if the course allows it.

**Pipeline (order matters)**

Run from the **repository root** (`autoregressive-textgen-transformer/`).

```bash
# 1) Data (creates data/processed/*.bin + vocab.json)
python src/prepare_data.py

# 2) Train both models (writes checkpoints + logs + training_curve_*.png under results/)
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/large.yaml

# 3) Test perplexity + greedy prompt metrics (match final_report paths)
python scripts/evaluate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/evaluation --gen_length 512

python scripts/evaluate.py --config configs/large.yaml ^
  --checkpoint results/checkpoints/large/best_model.pt ^
  --output_dir results/evaluation_large --gen_length 512

# 4) Full sampling grids (3 samples × settings × 10 prompts; + heatmap)
python scripts/run_experiments.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/experiments --gen_length 512 --num_samples 3 --heatmap --seed 42

python scripts/run_experiments.py --config configs/large.yaml ^
  --checkpoint results/checkpoints/large/best_model.pt ^
  --output_dir results/experiments_large --gen_length 512 --num_samples 3 --heatmap --seed 42

# 5) Plots: per-model sweeps (explicit training log avoids wrong auto-pick)
python scripts/plot_results.py ^
  --aggregated results/experiments/aggregated_metrics.json ^
  --heatmap_metrics results/experiments/heatmap_metrics.json ^
  --plots_dir results/plots ^
  --summary_out results/experiments/summary_table.md ^
  --training_log results/logs/training_log_baseline.csv

python scripts/plot_results.py ^
  --aggregated results/experiments_large/aggregated_metrics.json ^
  --heatmap_metrics results/experiments_large/heatmap_metrics.json ^
  --plots_dir results/plots_large ^
  --summary_out results/experiments_large/summary_table.md ^
  --training_log results/logs/training_log_large.csv

# 6) Baseline vs large overlays + Spearman (writes results/plots_compare/)
python scripts/plot_results.py ^
  --aggregated results/experiments/aggregated_metrics.json ^
  --heatmap_metrics results/experiments/heatmap_metrics.json ^
  --compare results/experiments_large/aggregated_metrics.json ^
  --compare_heatmap_metrics results/experiments_large/heatmap_metrics.json ^
  --plots_dir results/plots_compare ^
  --summary_out results/plots_compare/summary_table_baseline.md ^
  --spearman_out results/plots_compare/spearman_compare.json ^
  --training_log results/logs/training_log_baseline.csv

# 7) Optional: copy key PNGs into docs/figures for final_report.md (manual copy, or):
#     copy results/plots_compare/*.png docs/figures/
#     copy results/plots/training_curve_baseline.png docs/figures/
#     copy results/plots/training_curve_large.png docs/figures/

# 8) Optional: human-eval pack (15 fixed baseline samples + ratings template)
python scripts/prepare_human_eval.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/human_eval --gen_length 512 --seed 42

# After filling Likert columns in results/human_eval/ratings.csv:
python scripts/correlate_human_eval.py ^
  --ratings results/human_eval/ratings.csv ^
  --out results/human_eval/correlations.json
```

On **macOS / Linux**, replace line-ending `^` with `\` and run the same commands in `bash` or `zsh`.

---

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `matplotlib`, `tqdm`, `pyyaml` (see `requirements.txt`).

---

## Data

If `data/raw/shakespeare.txt` is missing, `src/prepare_data.py` can fetch and normalize Project Gutenberg Shakespeare; it then builds the char vocabulary and saves tensors:

```bash
python src/prepare_data.py
```

**Outputs:** `data/processed/{train,val,test}.bin`, `data/processed/vocab.json`.  
**Note:** `data/processed/*.bin` is gitignored by default (large files); you must run this step on a fresh clone before training.

---

## Training

All configs include **`seed: 42`** (used by `scripts/train.py` for Python / NumPy / Torch and for shuffling). `run_name` in each YAML sets the checkpoint subdirectory name.

```bash
# Main baseline (see configs/baseline.yaml for batch_size, epochs, patience)
python scripts/train.py --config configs/baseline.yaml

# Larger capacity ablation
python scripts/train.py --config configs/large.yaml

# Quick smoke model
python scripts/train.py --config configs/mini.yaml

# Single-batch overfit sanity check (no full train)
python scripts/train.py --config configs/baseline.yaml --overfit_tiny
```

**Outputs**

| Artifact | Path |
|----------|------|
| Best checkpoint | `results/checkpoints/<run_name>/best_model.pt` |
| Per-epoch checkpoints | `results/checkpoints/<run_name>/model_epoch_*.pt` |
| CSV log | `results/logs/training_log_<run_name>.csv` |
| Learning curve (during training) | `results/plots/training_curve_<run_name>.png` |

`vocab_size` is always read from `vocab.json` at runtime (do not hard-code vocabulary size in YAML).

---

## Generation

```bash
python scripts/generate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --prompt "to be or not to be" --max_tokens 300 --strategy greedy

python scripts/generate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --strategy temperature --temperature 0.7 --max_tokens 300

python scripts/generate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --strategy topk --top_k 20 --temperature 1.0 --max_tokens 300
```

---

## Evaluation (test perplexity + greedy metrics)

```bash
python scripts/evaluate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/evaluation --gen_length 512

python scripts/evaluate.py --config configs/large.yaml ^
  --checkpoint results/checkpoints/large/best_model.pt ^
  --output_dir results/evaluation_large --gen_length 512
```

Writes `greedy_metrics.json` under each `--output_dir` (test perplexity + per-prompt greedy samples).

---

## Sampling experiments

Full temperature × top‑k grid over **10** prompts in `src/prompts.py`. Default **`--seed 42`** matches the human-eval sample pack.

```bash
python scripts/run_experiments.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/experiments --gen_length 512 --num_samples 3 --heatmap --seed 42

python scripts/run_experiments.py --config configs/large.yaml ^
  --checkpoint results/checkpoints/large/best_model.pt ^
  --output_dir results/experiments_large --gen_length 512 --num_samples 3 --heatmap --seed 42
```

**Outputs:** `aggregated_metrics.json`, `raw_results.json`, optional `heatmap_metrics.json`, `samples/*.txt`.

Quick test (subset of prompts):

```bash
python scripts/run_experiments.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/experiments_smoke --gen_length 512 --num_samples 3 --max_prompts 2
```

---

## Plotting

**Single run** (temperature / top‑k / heatmap + optional training curve):

```bash
python scripts/plot_results.py ^
  --aggregated results/experiments/aggregated_metrics.json ^
  --heatmap_metrics results/experiments/heatmap_metrics.json ^
  --plots_dir results/plots ^
  --summary_out results/experiments/summary_table.md ^
  --training_log results/logs/training_log_baseline.csv
```

**Baseline vs large** (ablation + compare sweeps + compare heatmaps + Spearman JSON):

```bash
python scripts/plot_results.py ^
  --aggregated results/experiments/aggregated_metrics.json ^
  --heatmap_metrics results/experiments/heatmap_metrics.json ^
  --compare results/experiments_large/aggregated_metrics.json ^
  --compare_heatmap_metrics results/experiments_large/heatmap_metrics.json ^
  --plots_dir results/plots_compare ^
  --spearman_out results/plots_compare/spearman_compare.json ^
  --training_log results/logs/training_log_baseline.csv
```

`--compare` is an alias for `--compare_large`. Always pass **`--training_log`** explicitly if you have multiple `results/logs/training_log_*.csv` files, otherwise the script may pick the wrong one when auto-detecting.

---

## Human evaluation (optional)

```bash
python scripts/prepare_human_eval.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/human_eval --gen_length 512 --seed 42
```

Edit Likert columns in `results/human_eval/ratings.csv`, then:

```bash
python scripts/correlate_human_eval.py ^
  --ratings results/human_eval/ratings.csv ^
  --out results/human_eval/correlations.json
```

---

## Demo & safety check

```bash
python scripts/demo.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --prompt "to be or not to be" --max_tokens 250

python scripts/safety_check.py
```

---

## Project layout

```
configs/              # baseline.yaml, large.yaml, mini.yaml, baseline_long.yaml
data/raw/             # optional: shakespeare.txt (often gitignored / downloaded)
data/processed/       # train/val/test .bin + vocab.json (bins often gitignored)
docs/figures/         # PNGs embedded in final_report.md (small; safe to commit)
scripts/              # train, generate, evaluate, run_experiments, plot_results,
                       # prepare_human_eval, correlate_human_eval, demo, safety_check
src/                  # model, dataset, prepare_data, metrics, generation, prompts, …
final_report.md       # Final write-up + references
results/              # Default output root (see .gitignore for large/binary patterns)
  checkpoints/        # .pt files tracked if committed (best_model.pt per run)
  logs/
  plots/               # baseline-only plot_results output (default name clash avoided)
  plots_large/         # large-only plot_results output (optional; avoids overwriting plots/)
  plots_compare/
  experiments/
  experiments_large/
  evaluation/
  evaluation_large/
  human_eval/
```

---

## Ethics & limitations

- **Data:** Public-domain Shakespeare via Project Gutenberg; not modern web text.  
- **Content:** Plays contain violence and archaic/offensive language; generated text can reflect that.  
- **Use:** Educational / research toy LM — **not** a safe general chatbot; no factual reliability.

---

## Acknowledgments

- Project Gutenberg ([eBook #100](https://www.gutenberg.org/ebooks/100))  
- Core papers cited in `final_report.md` §8 (Transformer, GPT-style LMs, neural text degeneration, etc.)
