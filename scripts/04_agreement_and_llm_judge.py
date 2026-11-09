"""Step 4: annotation agreement and the LLM-as-judge comparison.

Inputs (CSV files with columns pair_id,score):
  data/annotator2.csv   scores from the second human annotator (blind to annotator 1)
  data/llm_judge.csv    scores from the LLM judge (blind to all human scores)
Also needs data/constructed_pairs.csv and results/constructed_scores.csv (from step 2).
Either optional file may be missing; the script reports whatever it can.
Run:  python scripts/04_agreement_and_llm_judge.py
"""
import json
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

from common import DATA, RESULTS, read_pairs, spearman


def load_scores(path, ids):
    rows = read_pairs(path)
    got = {}
    for r in rows:
        v = (r.get("score") or "").strip()
        if v:
            got[int(r["pair_id"])] = float(v)
    missing = [i for i in ids if i not in got]
    if missing:
        raise SystemExit(f"{path}: missing scores for pair_ids {missing}")
    return np.array([got[i] for i in ids])


def agreement(a, b):
    ai, bi = np.rint(a).astype(int), np.rint(b).astype(int)
    return {
        "spearman": round(spearman(a, b), 4),
        "pearson": round(float(pearsonr(a, b)[0]), 4),
        "weighted_kappa_quadratic": round(float(cohen_kappa_score(ai, bi, weights="quadratic", labels=list(range(6)))), 4),
        "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 3),
        "within_1_point": round(float(np.mean(np.abs(a - b) <= 1.0)), 3),
    }


def main():
    base = read_pairs(os.path.join(DATA, "constructed_pairs.csv"))
    ids = [int(r["pair_id"]) for r in base]
    a1 = np.array([float(r["score_annotator1"]) for r in base])
    raters = {"annotator1": a1}

    for name, fname in [("annotator2", "annotator2.csv"), ("llm_judge", "llm_judge.csv")]:
        path = os.path.join(DATA, fname)
        if os.path.exists(path):
            raters[name] = load_scores(path, ids)
        else:
            print(f"(no {fname} yet; skipping {name})")

    # SBERT cosine scores from step 2
    sbert = None
    sp = os.path.join(RESULTS, "constructed_scores.csv")
    if os.path.exists(sp):
        m = {int(r["pair_id"]): float(r["cos_minilm"]) for r in read_pairs(sp) if "cos_minilm" in r}
        sbert = np.array([m[i] for i in ids])
        raters["sbert_minilm_cosine"] = sbert
    else:
        print("(no results/constructed_scores.csv; run step 2 first to include SBERT)")

    out = {}
    names = list(raters)
    print("\nPairwise agreement (Spearman rho / quadratic weighted kappa on 0-5 integers):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            if "sbert_minilm_cosine" in (x, y):
                # cosine is not on the 0-5 scale, so report rank correlation only
                res = {"spearman": round(spearman(raters[x], raters[y]), 4)}
            else:
                res = agreement(raters[x], raters[y])
            out[f"{x} vs {y}"] = res
            print(f"  {x} vs {y}: {res}")

    humans = [raters[k] for k in ("annotator1", "annotator2") if k in raters]
    if len(humans) == 2 and sbert is not None:
        consensus = np.mean(humans, axis=0)
        out["sbert vs human consensus (mean of A1,A2)"] = {"spearman": round(spearman(sbert, consensus), 4)}
        print(f"\n  SBERT vs mean of the two humans: rho = {spearman(sbert, consensus):.4f}")
    if "llm_judge" in raters and sbert is not None:
        d = np.abs(raters["llm_judge"] - a1)
        top = np.argsort(-d)[:5]
        print("\nPairs where the LLM judge and annotator 1 differ most (candidates for the discussion):")
        for k in top:
            r = base[k]
            print(f"  pair {ids[k]}: A1={a1[k]:.1f} LLM={raters['llm_judge'][k]:.1f}")
            print(f"    S1: {r['sentence1']}\n    S2: {r['sentence2']}")

    # Each SBERT model against each rater (rank correlation x100, ready for the report table)
    if os.path.exists(sp):
        cos_rows = read_pairs(sp)
        out["sbert_vs_raters"] = {}
        print("\nSBERT models against each rater (Spearman x 100):")
        for col in [c for c in cos_rows[0].keys() if c.startswith("cos_")]:
            m = {int(r["pair_id"]): float(r[col]) for r in cos_rows}
            vec = np.array([m[i] for i in ids])
            line = {}
            for rname in ("annotator1", "annotator2", "llm_judge"):
                if rname in raters:
                    line[rname] = round(spearman(vec, raters[rname]) * 100, 2)
            if len(humans) == 2:
                line["mean_of_two_humans"] = round(spearman(vec, np.mean(humans, axis=0)) * 100, 2)
            out["sbert_vs_raters"][col] = line
            print(f"  {col}: {line}")

    with open(os.path.join(RESULTS, "agreement.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/agreement.json")


if __name__ == "__main__":
    main()
