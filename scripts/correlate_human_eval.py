"""
Phase 3: correlate filled human Likert ratings with automatic metrics.

Reads results/human_eval/ratings.csv (or --ratings), expects numeric columns:
  distinct_3, repeat_fraction, likert_coherence, likert_diversity, likert_overall

Writes JSON with Pearson and Spearman correlations (requires numpy).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    n = a.size
    if n == 0:
        return np.zeros_like(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _parse_float(cell):
    if cell is None or str(cell).strip() == "":
        return None
    return float(str(cell).strip())


def _parse_int_likert(cell):
    if cell is None or str(cell).strip() == "":
        return None
    v = int(float(str(cell).strip()))
    if v < 1 or v > 5:
        raise ValueError(f"Likert must be 1–5, got {v}")
    return v


def load_completed_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d3 = _parse_float(row.get("distinct_3"))
            rep = _parse_float(row.get("repeat_fraction"))
            lc = _parse_int_likert(row.get("likert_coherence"))
            ld = _parse_int_likert(row.get("likert_diversity"))
            lo = _parse_int_likert(row.get("likert_overall"))
            if d3 is None or rep is None or lc is None or ld is None or lo is None:
                continue
            rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "distinct_3": d3,
                    "repeat_fraction": rep,
                    "likert_coherence": lc,
                    "likert_diversity": ld,
                    "likert_overall": lo,
                }
            )
    return rows


def pairwise_correlations(metric_x, likert_keys, rows):
    x = np.array([r[metric_x] for r in rows], dtype=float)
    out = {}
    for lk in likert_keys:
        y = np.array([r[lk] for r in rows], dtype=float)
        out[lk] = {
            "pearson_r": pearson_r(x, y),
            "spearman_rho": spearman_rho(x, y),
            "n": int(len(rows)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Correlate human Likert ratings with automatic metrics")
    parser.add_argument("--ratings", type=str, default="results/human_eval/ratings.csv")
    parser.add_argument("--out", type=str, default="results/human_eval/correlations.json")
    args = parser.parse_args()

    if not os.path.isfile(args.ratings):
        print(f"Missing ratings file: {args.ratings}")
        sys.exit(1)

    rows = load_completed_rows(args.ratings)
    if len(rows) < 3:
        print(
            f"Need at least 3 fully rated rows (distinct_3, repeat_fraction, all three Likerts). "
            f"Found {len(rows)} complete rows."
        )
        sys.exit(2)

    likert_keys = ["likert_coherence", "likert_diversity", "likert_overall"]
    summary = {
        "n_complete_rows": len(rows),
        "vs_distinct_3": pairwise_correlations("distinct_3", likert_keys, rows),
        "vs_repeat_fraction": pairwise_correlations("repeat_fraction", likert_keys, rows),
    }

    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
