"""Step 1: replicate the STS Benchmark result.

Evaluates the paper's own checkpoints and a later model on the STS Benchmark test set
(1,379 pairs), using the paper's protocol: Spearman correlation (x100) between the cosine
similarity of the two sentence embeddings and the gold similarity score.

Paper values are from Reimers & Gurevych (2019), Tables 1 and 2 (STSb test set).
Run:  python scripts/01_replicate_stsb.py
"""
import csv
import os

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from common import (MINILM, PAPER_NLI_BASE, PAPER_NLI_STSB_BASE, RESULTS, encode_cosine,
                    spearman, write_environment)

# (label, model id, value reported in the paper or None)
MODELS = [
    (*PAPER_NLI_BASE, 77.03),        # paper Table 1/2: SBERT-NLI-base
    (*PAPER_NLI_STSB_BASE, 85.35),   # paper Table 2: SBERT-NLI-STSb-base (mean of 10 seeds)
    (*MINILM, None),                 # later model, not evaluated in the paper
]


def main():
    env = write_environment()
    print("Environment:", *env, sep="\n  ")

    test = load_dataset("sentence-transformers/stsb", split="test")
    s1, s2, gold = test["sentence1"], test["sentence2"], test["score"]
    print(f"\nSTS Benchmark test pairs: {len(test)}")

    rows = []
    for label, model_id, paper in MODELS:
        try:
            model = SentenceTransformer(model_id)
            cos = encode_cosine(model, s1, s2)
            ours = spearman(cos, gold) * 100
        except Exception as exc:  # keep going if one old checkpoint fails to load
            print(f"\n!! {label} failed: {exc!r}")
            rows.append([label, model_id, paper, "FAILED", ""])
            continue
        diff = "" if paper is None else round(ours - paper, 2)
        rows.append([label, model_id, paper, round(ours, 2), diff])
        print(f"\n{label}\n  ours={ours:.2f}  paper={paper}  difference={diff}")

    out = os.path.join(RESULTS, "stsb_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "hf_id", "paper_spearman_x100", "ours_spearman_x100", "difference"])
        w.writerows(rows)
    print(f"\nSaved {out}")
    print("\nLaTeX rows for the report table:")
    for label, _, paper, ours, diff in rows:
        print(f"{label} & {paper if paper is not None else '--'} & {ours} & {diff if diff != '' else '--'} \\\\")


if __name__ == "__main__":
    main()
