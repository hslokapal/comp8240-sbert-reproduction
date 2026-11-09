"""Step 3: apply SBERT to an existing dataset the paper did not use: BIOSSES.

BIOSSES: 100 biomedical sentence pairs, each scored 0-4 (mean of five human experts).
Run:  python scripts/03_eval_biosses.py
"""
import csv
import os

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from common import MINILM, PAPER_NLI_BASE, RESULTS, bootstrap_ci, encode_cosine, spearman

MODELS = [PAPER_NLI_BASE, MINILM]


def load_biosses():
    last = None
    for name in ["tabilab/biosses", "biosses"]:
        try:
            return load_dataset(name, split="train")
        except Exception as exc:
            last = exc
    raise SystemExit(f"Could not load BIOSSES: {last!r}")


def main():
    ds = load_biosses()
    print("Columns:", ds.column_names, " pairs:", len(ds))
    s1, s2, gold = ds["sentence1"], ds["sentence2"], ds["score"]
    rows = []
    for label, model_id in MODELS:
        try:
            cos = encode_cosine(SentenceTransformer(model_id), s1, s2)
        except Exception as exc:
            print(f"!! {label} failed: {exc!r}")
            continue
        rho = spearman(cos, gold)
        lo, hi = bootstrap_ci(cos, gold)
        rows.append([label, model_id, round(rho, 4), round(lo, 3), round(hi, 3)])
        print(f"{label}: Spearman = {rho:.4f}   95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
    out = os.path.join(RESULTS, "biosses_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "hf_id", "spearman", "ci_low", "ci_high"])
        w.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
