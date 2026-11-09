"""Step 2: evaluate SBERT on the constructed data-engineering dataset (40 pairs).

Saves per-pair cosine scores (used by step 4), a rank-based disagreement list, a bootstrap
confidence interval and a scatter plot.
Run:  python scripts/02_eval_constructed.py
"""
import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from common import (DATA, MINILM, PAPER_NLI_BASE, RESULTS, bootstrap_ci, encode_cosine,
                    rank_percentile, read_pairs, spearman)

MODELS = [("paper_nli_base", PAPER_NLI_BASE), ("minilm", MINILM)]


def main():
    rows = read_pairs(os.path.join(DATA, "constructed_pairs.csv"))
    ids = [int(r["pair_id"]) for r in rows]
    s1 = [r["sentence1"] for r in rows]
    s2 = [r["sentence2"] for r in rows]
    human = np.array([float(r["score_annotator1"]) for r in rows])
    print(f"Loaded {len(rows)} pairs")

    cos = {}
    for key, (label, model_id) in MODELS:
        try:
            cos[key] = encode_cosine(SentenceTransformer(model_id), s1, s2)
        except Exception as exc:
            print(f"!! {label} failed: {exc!r}")
            continue
        rho = spearman(cos[key], human)
        lo, hi = bootstrap_ci(cos[key], human)
        print(f"{label}: Spearman vs annotator 1 = {rho:.4f}   95% bootstrap CI [{lo:.3f}, {hi:.3f}]")

    if not cos:
        raise SystemExit("No model could be loaded.")

    out = os.path.join(RESULTS, "constructed_scores.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id"] + [f"cos_{k}" for k in cos])
        for i, pid in enumerate(ids):
            w.writerow([pid] + [round(float(cos[k][i]), 5) for k in cos])
    print(f"\nSaved {out}")

    # Rank-based disagreement (does not depend on how cosine is rescaled)
    key = "minilm" if "minilm" in cos else next(iter(cos))
    gap = np.abs(rank_percentile(human) - rank_percentile(cos[key]))
    order = np.argsort(-gap)[:8]
    print(f"\nLargest rank disagreements for {key} (human rank percentile vs model rank percentile):")
    for i in order:
        print(f"\n  pair {ids[i]}  gap={gap[i]:.2f}  human={human[i]:.1f}  cosine={cos[key][i]:.3f}")
        print(f"    S1: {s1[i]}\n    S2: {s2[i]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        ax.scatter(human, cos[key], s=14)
        ax.set_xlabel("Annotator 1 score (0-5)")
        ax.set_ylabel("Cosine similarity")
        ax.set_title(f"Constructed data, n={len(rows)}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "constructed_scatter.pdf"))
        print("Saved results/constructed_scatter.pdf")
    except ImportError:
        print("matplotlib not installed; skipping the scatter plot (pip install matplotlib)")


if __name__ == "__main__":
    main()
