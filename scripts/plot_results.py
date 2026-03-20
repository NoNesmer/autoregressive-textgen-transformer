"""
Plot experiment metrics and regenerate training curves from CSV logs.
"""

import os
import sys
import argparse
import json
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass


def load_agg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_temperature(agg, out_dir):
    temps = sorted(agg["temperature"].keys(), key=lambda x: float(x))
    d2 = [agg["temperature"][t]["distinct_2"] for t in temps]
    d3 = [agg["temperature"][t]["distinct_3"] for t in temps]
    rep = [agg["temperature"][t]["repeat_fraction"] for t in temps]
    tf = [float(t) for t in temps]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(tf, d2, "o-", label="Distinct-2")
    ax.plot(tf, d3, "s-", label="Distinct-3")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Distinct-n")
    ax.set_title("Temperature vs. diversity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temperature_vs_diversity.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(tf, rep, "o-", color="C2")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("4-gram repeat fraction")
    ax.set_title("Temperature vs. repetition")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temperature_vs_repetition.png"))
    plt.close(fig)


def plot_topk(agg, out_dir):
    ks = sorted(agg["top_k"].keys(), key=lambda x: int(x))
    d2 = [agg["top_k"][k]["distinct_2"] for k in ks]
    d3 = [agg["top_k"][k]["distinct_3"] for k in ks]
    rep = [agg["top_k"][k]["repeat_fraction"] for k in ks]
    k_int = [int(k) for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(k_int, d2, "o-", label="Distinct-2")
    ax.plot(k_int, d3, "s-", label="Distinct-3")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Distinct-n")
    ax.set_title("Top-k vs. diversity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "topk_vs_diversity.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(k_int, rep, "o-", color="C2")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("4-gram repeat fraction")
    ax.set_title("Top-k vs. repetition")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "topk_vs_repetition.png"))
    plt.close(fig)


def write_summary_table(agg, out_path):
    lines = [
        "| Strategy | Temperature | Top-k | Distinct-2 | Distinct-3 | Repeat Frac | First Rep Pos |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for t in sorted(agg["temperature"].keys(), key=lambda x: float(x)):
        m = agg["temperature"][t]
        fr = m.get("first_repeat_position", -1)
        lines.append(
            f"| temperature | {t} | — | {m['distinct_2']:.4f} | {m['distinct_3']:.4f} | "
            f"{m['repeat_fraction']:.4f} | {fr:.1f} |"
        )
    for k in sorted(agg["top_k"].keys(), key=lambda x: int(x)):
        m = agg["top_k"][k]
        fr = m.get("first_repeat_position", -1)
        lines.append(
            f"| top-k | — | {k} | {m['distinct_2']:.4f} | {m['distinct_3']:.4f} | "
            f"{m['repeat_fraction']:.4f} | {fr:.1f} |"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_heatmap(heatmap_path, out_dir):
    """2D grid: temperature (rows) x top-k (cols) -> distinct-3 from heatmap_metrics.json."""
    with open(heatmap_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    temps = sorted(data.keys(), key=lambda x: float(x))
    first_row = data[temps[0]]
    ks = sorted(first_row.keys(), key=lambda x: int(x))
    mat = [[data[t][k]["distinct_3"] for k in ks] for t in temps]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(range(len(temps)))
    ax.set_yticklabels([str(t) for t in temps])
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Temperature")
    ax.set_title("Distinct-3: top-k sampling with temperature (heatmap)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Distinct-3")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap_diversity.png"))
    plt.close(fig)


def plot_training_curve(csv_path, out_png):
    import csv
    epochs, train_l, val_l = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_l.append(float(row["train_loss"]))
            val_l.append(float(row["val_loss"]))
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(epochs, train_l, label="Train loss")
    ax.plot(epochs, val_l, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy")
    ax.set_title("Training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_ablation(agg_baseline, agg_large, out_dir, label_a="baseline", label_b="large"):
    temps = sorted(agg_baseline["temperature"].keys(), key=lambda x: float(x))
    tf = [float(t) for t in temps]
    d3_a = [agg_baseline["temperature"][t]["distinct_3"] for t in temps]
    d3_b = [agg_large["temperature"][t]["distinct_3"] for t in temps]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(tf, d3_a, "o--", label=f"{label_a} (distinct-3)")
    ax.plot(tf, d3_b, "s-", label=f"{label_b} (distinct-3)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Distinct-3")
    ax.set_title("Ablation: temperature vs diversity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_temperature_distinct3.png"))
    plt.close(fig)

    ks = sorted(agg_baseline["top_k"].keys(), key=lambda x: int(x))
    k_int = [int(k) for k in ks]
    d3_a = [agg_baseline["top_k"][k]["distinct_3"] for k in ks]
    d3_b = [agg_large["top_k"][k]["distinct_3"] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(k_int, d3_a, "o--", label=f"{label_a}")
    ax.plot(k_int, d3_b, "s-", label=f"{label_b}")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Distinct-3")
    ax.set_title("Ablation: top-k vs diversity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_topk_distinct3.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--aggregated", type=str,
                        default="results/experiments/aggregated_metrics.json",
                        help="Path to aggregated_metrics.json")
    parser.add_argument("--plots_dir", type=str, default="results/plots",
                        help="Directory for PNG outputs")
    parser.add_argument("--summary_out", type=str,
                        default="results/experiments/summary_table.md",
                        help="Markdown summary table path")
    parser.add_argument("--training_log", type=str, default="",
                        help="Optional CSV log (e.g. results/logs/training_log_baseline.csv)")
    parser.add_argument("--compare_large", type=str, default="",
                        help="Second aggregated_metrics.json for ablation (e.g. experiments_large)")
    parser.add_argument("--heatmap_metrics", type=str, default="",
                        help="Path to heatmap_metrics.json (default: sibling of --aggregated)")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    if os.path.isfile(args.aggregated):
        agg = load_agg(args.aggregated)
        plot_temperature(agg, args.plots_dir)
        plot_topk(agg, args.plots_dir)
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
        write_summary_table(agg, args.summary_out)
        print(f"Plots saved to {args.plots_dir}")
        print(f"Summary table: {args.summary_out}")
    else:
        print(f"Warning: aggregated file not found: {args.aggregated}")

    hm_path = args.heatmap_metrics
    if not hm_path and args.aggregated and os.path.isfile(args.aggregated):
        hm_path = os.path.join(os.path.dirname(args.aggregated), "heatmap_metrics.json")
    if hm_path and os.path.isfile(hm_path):
        plot_heatmap(hm_path, args.plots_dir)
        print(f"Heatmap saved from {hm_path}")

    if args.compare_large and os.path.isfile(args.compare_large):
        agg_b = load_agg(args.aggregated) if os.path.isfile(args.aggregated) else None
        agg_l = load_agg(args.compare_large)
        if agg_b:
            plot_ablation(agg_b, agg_l, args.plots_dir)
            print("Ablation plots saved.")

    if args.training_log and os.path.isfile(args.training_log):
        out = os.path.join(args.plots_dir, "training_curve.png")
        plot_training_curve(args.training_log, out)
        print(f"Training curve: {out}")
    else:
        logs = glob.glob("results/logs/training_log_*.csv")
        if logs:
            plot_training_curve(logs[0], os.path.join(args.plots_dir, "training_curve.png"))
            print(f"Training curve from {logs[0]}")


if __name__ == "__main__":
    main()
