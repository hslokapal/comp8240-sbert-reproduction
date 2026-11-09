# Annotation guidelines and LLM-judge prompt

Used for the second human annotator and for the LLM judge on the 40-pair constructed dataset.
Both are given `data/pairs_for_annotation.csv` (scores blank, pairs in shuffled order) and
must NOT see annotator 1's scores or each other's scores.

## Scale (0-5, adapted from the SemEval STS guidelines)

| Score | Meaning |
|------|---------|
| 5 | The two sentences are completely equivalent in meaning |
| 4 | Mostly equivalent; some unimportant details differ |
| 3 | Roughly equivalent; some important information differs or is missing |
| 2 | Not equivalent, but share some details |
| 1 | Not equivalent, but on the same topic |
| 0 | Completely dissimilar (different topics) |

Rules: judge **meaning**, not shared words. Two questions that mention the same tool (for example
"Azure Data Factory") but ask for different things are NOT equivalent. Decimals (one place) are allowed.
Do not look anything up; use your own judgement.

## Instructions for the second human annotator

1. Open `pairs_for_annotation.csv` (Excel or Google Sheets). Fill the `score` column for all 40 rows.
2. Work alone and do not discuss the pairs with anyone until you have finished.
3. Save as `annotator2.csv` (CSV, columns `pair_id,score`).
4. Participation is voluntary. No personal information is collected. Your scores will be reported only
   as agreement statistics, and you can withdraw before submission. You will be acknowledged by role
   (for example "second annotator") unless you ask to be named.

## LLM-judge prompt (paste into a NEW chat with no earlier context)

Use a **different model family** from the one that drafted the pairs, to reduce circularity, and
record the model name, version and date in the report.

```
You are a careful annotator for semantic textual similarity (STS). The sentence pairs are
technical questions about data engineering. For each pair give a similarity score from 0 to 5
(one decimal place allowed) using this scale:
5 completely equivalent; 4 mostly equivalent, unimportant details differ; 3 roughly equivalent,
some important information differs; 2 not equivalent but share some details; 1 not equivalent
but on the same topic; 0 completely dissimilar.
Judge meaning, not shared words. Two questions that mention the same tool but ask for different
things are not equivalent.
Output ONLY a CSV with the header pair_id,score and one row per pair. No other text.

Pairs (pair_id | sentence1 | sentence2):
<paste the pair_id, sentence1 and sentence2 columns of pairs_for_annotation.csv>
```

Save the reply as `data/llm_judge.csv`.

Optional second step (for the discussion section): after running step 2 of the pipeline, take the
pair_ids with the largest model-vs-human gaps and ask the LLM, in the same chat, "In one sentence each,
explain why you scored these pairs as you did." Quote at most one or two of these explanations.


**Note added after scoring:** the second annotator was asked after scoring and agreed that their 40 scores, labelled only as "annotator 2" and with no personal details, could be published in the report appendix and in this repository (`data/annotator2.csv`).
