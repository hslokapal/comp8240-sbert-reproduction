# Reproducing Sentence-BERT and testing it on data-engineering text (COMP8240 project)

Author: Lokapal Haachagowdanahalli Shankarappa (Macquarie University, COMP8240, Session 2 2026).

This repository holds the code and data for my final report. The project reproduces
*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (Reimers & Gurevych, EMNLP-IJCNLP 2019,
https://arxiv.org/abs/1908.10084) and tests how well it generalises to new data.

## What is here

| Path | Contents |
|------|----------|
| `scripts/01_replicate_stsb.py` | Replicates the STS Benchmark result with the paper's own checkpoints and a later model |
| `scripts/02_eval_constructed.py` | Evaluates SBERT on my 40-pair constructed dataset; per-pair scores, bootstrap CI, scatter plot |
| `scripts/03_eval_biosses.py` | Applies SBERT to BIOSSES, an existing dataset the paper did not use |
| `scripts/04_agreement_and_llm_judge.py` | Inter-annotator agreement and the LLM-as-judge comparison |
| `scripts/common.py` | Shared helpers (cosine, Spearman, bootstrap CI, environment capture) |
| `data/constructed_pairs.csv` | The constructed dataset: 40 pairs with annotator-1 scores and a similarity band |
| `data/pairs_for_annotation.csv` | The same pairs, shuffled, scores blank (given to annotator 2 and the LLM judge) |
| `data/annotator2.csv`, `data/llm_judge.csv` | Scores from the second annotator and the LLM judge |
| `docs/annotation_guidelines.md` | 0-5 scale, annotator instructions, and the exact LLM-judge prompt |
| `results/` | Outputs written by the scripts (environment, scores, agreement) |
| `report/` | LaTeX source of the final report (add the compiled PDF here before submitting) |

## How to run

Tested with Python 3.10.

```bash
conda create -n comp8240 python=3.10 -y
conda activate comp8240
pip install -r requirements.txt

python scripts/01_replicate_stsb.py          # needs internet (downloads models and the dataset)
python scripts/02_eval_constructed.py
python scripts/03_eval_biosses.py
python scripts/04_agreement_and_llm_judge.py
```

Models and datasets are downloaded from the Hugging Face Hub on first run. The STS Benchmark and BIOSSES
are loaded from the Hub and are not redistributed here (BIOSSES is GPL-3.0). Expect each script to take
from a few seconds to a couple of minutes on a laptop CPU.

## Data provenance and AI use (please read)

The 40 constructed sentence pairs were **drafted with the help of a large language model** (Claude), written
in the style of real Azure Data Factory, Databricks, PySpark, Synapse and medallion-architecture questions.
They are not copied from any documentation or forum. The model also suggested initial similarity scores.
I reviewed all 40 pairs and changed three scores (pairs 14, 20, 26). Because annotator 1's scores began as LLM
suggestions, they are not fully independent of the LLM; this is why a second human annotator and a separate
LLM judge each scored the pairs blind. See the report for the agreement figures and limitations.
The unit's AI policy is "Open"; I am responsible for all content.

## Limitations

40 pairs and two raters is small; the dataset was designed with deliberately spread similarity bands, which
makes ranking easier than on natural data; results on it are indicative only.
