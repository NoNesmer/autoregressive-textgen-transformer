# Autoregressive Character-Level Transformer

Character-level **GPT-style** transformer trained on Shakespeare (Project Gutenberg).  
Compares **greedy**, **temperature**, and **top-k** decoding for quality vs. diversity (course project).

## Setup

```bash
pip install -r requirements.txt
```

## Data

Downloads and preprocesses the complete works (if `data/raw/shakespeare.txt` is missing), strips standard Gutenberg markers when present, normalizes text, builds a char vocabulary, and saves tensors:

```bash
python src/prepare_data.py
```

Outputs: `data/processed/{train,val,test}.bin`, `data/processed/vocab.json`.

## Training

```bash
# Main baseline (~2h target on GPU with stride=8; see configs/baseline.yaml)
python scripts/train.py --config configs/baseline.yaml

# Quick mini run
python scripts/train.py --config configs/mini.yaml

# Sanity check (single batch overfit)
python scripts/train.py --config configs/baseline.yaml --overfit_tiny
```

Checkpoints: `results/checkpoints/<run_name>/best_model.pt`  
Logs: `results/logs/training_log_<run_name>.csv`  
Plots: `results/plots/training_curve_<run_name>.png`

`vocab_size` is always read from `vocab.json` (do not rely on a manual count in YAML).

## Generation

```bash
python scripts/generate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --prompt "to be or not to be" --max_tokens 300 --strategy greedy

# temperature / topk
python scripts/generate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --strategy temperature --temperature 0.7 --max_tokens 300

python scripts/generate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --strategy topk --top_k 20 --temperature 1.0 --max_tokens 300
```

*(On Unix shells, replace `^` with `\`.)*

## Evaluation (test perplexity + greedy metrics)

```bash
python scripts/evaluate.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/evaluation --gen_length 512
```

Writes `results/evaluation/greedy_metrics.json`.

## Sampling experiments & plots

Runs the full temperature × top-k grid over 10 standard prompts (see `src/prompts.py`):

```bash
python scripts/run_experiments.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/experiments --gen_length 512 --num_samples 3

# Optional: also writes heatmap_metrics.json (T × k joint top-k sampling) for heatmap_diversity.png
python scripts/run_experiments.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/experiments --gen_length 512 --num_samples 3 --heatmap
```

Optional quick test (subset of prompts):

```bash
python scripts/run_experiments.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --output_dir results/experiments --gen_length 512 --num_samples 3 ^
  --max_prompts 2
```

Then plot:

```bash
python scripts/plot_results.py ^
  --aggregated results/experiments/aggregated_metrics.json ^
  --plots_dir results/plots ^
  --summary_out results/experiments/summary_table.md ^
  --training_log results/logs/training_log_baseline.csv
```

Plots: `temperature_vs_diversity.png`, `temperature_vs_repetition.png`, `topk_vs_diversity.png`, `topk_vs_repetition.png`, optional `heatmap_diversity.png` (if `heatmap_metrics.json` exists), plus `training_curve.png` when a log path is found.

### Ablation (after training `configs/large.yaml`)

```bash
python scripts/train.py --config configs/large.yaml
python scripts/run_experiments.py --config configs/large.yaml ^
  --checkpoint results/checkpoints/large/best_model.pt ^
  --output_dir results/experiments_large

python scripts/plot_results.py ^
  --aggregated results/experiments/aggregated_metrics.json ^
  --compare_large results/experiments_large/aggregated_metrics.json ^
  --plots_dir results/plots
```

## Demo & safety check

```bash
python scripts/demo.py --config configs/baseline.yaml ^
  --checkpoint results/checkpoints/baseline/best_model.pt ^
  --prompt "to be or not to be" --max_tokens 250

python scripts/safety_check.py
```

## Project layout

```
configs/           # baseline.yaml, mini.yaml, large.yaml
data/raw/          # optional: shakespeare.txt (auto-downloaded)
data/processed/    # *.bin tensors + vocab.json
scripts/           # train, generate, evaluate, experiments, plots, demo, safety
src/               # model, dataset, prepare_data, metrics, generation, prompts
results/           # checkpoints, logs, plots, experiments (gitignored where large)
```

## Ethics & limitations

- **Data:** Public-domain Shakespeare via Project Gutenberg; not modern web text.  
- **Content:** Plays contain violence and archaic/offensive language; generated text can reflect that.  
- **Use:** Educational / research toy LM — **not** a safe general chatbot; no factual reliability.

## Acknowledgments

- Project Gutenberg (e.g. [eBook #100](https://www.gutenberg.org/ebooks/100))  
- Vaswani et al., *Attention Is All You Need*; Holtzman et al., *Neural Text Degeneration*
