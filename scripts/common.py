"""Shared helpers for the COMP8240 SBERT reproduction scripts."""
import csv
import os
import platform
import sys
from importlib import metadata

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# Models used in the project.
# key -> (label, Hugging Face id)
PAPER_NLI_BASE = ("SBERT-NLI-base (paper checkpoint)", "sentence-transformers/bert-base-nli-mean-tokens")
PAPER_NLI_STSB_BASE = ("SBERT-NLI-STSb-base (paper checkpoint)", "sentence-transformers/bert-base-nli-stsb-mean-tokens")
MINILM = ("all-MiniLM-L6-v2 (later model)", "sentence-transformers/all-MiniLM-L6-v2")


def cosine_rows(e1, e2):
    """Row-wise cosine similarity between two embedding matrices."""
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    return np.sum(e1 * e2, axis=1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1))


def spearman(a, b):
    rho, _ = spearmanr(a, b)
    return float(rho)


def bootstrap_ci(a, b, n_boot=2000, seed=0):
    """95% percentile bootstrap confidence interval for Spearman's rho."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        if len(set(a[idx])) < 2 or len(set(b[idx])) < 2:
            continue
        vals.append(spearman(a[idx], b[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rank_percentile(x):
    """Rank of each value scaled to 0..1 (ties get average rank)."""
    x = np.asarray(x, dtype=float)
    return (rankdata(x) - 1) / (len(x) - 1)


def encode_cosine(model, s1, s2, batch_size=64):
    e1 = model.encode(list(s1), convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
    e2 = model.encode(list(s2), convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
    return cosine_rows(e1, e2)


def read_pairs(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_environment(path=None):
    """Record the software environment (the report's replication section needs this)."""
    path = path or os.path.join(RESULTS, "environment.txt")
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}", f"machine {platform.machine()}"]
    for pkg in ["sentence-transformers", "transformers", "torch", "numpy", "scipy",
                "scikit-learn", "datasets", "pandas", "matplotlib"]:
        try:
            lines.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{pkg} (not installed)")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines
