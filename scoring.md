# Evaluation: `score.py`

`score.py` compares a system's predictions with the MELD gold labels and reports **weighted F1** (primary metric), **macro F1**, accuracy, per-class precision/recall/F1 and a confusion matrix. File formats are defined in [interface.md](interface.md).

## Metric definitions

Let $\mathcal{C}$ be the $K = 7$ emotion classes (`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`), $y_i$ the gold label and $\hat{y}_i$ the predicted label of utterance $i$, for $i = 1, \dots, N$. For each class $c$:

$$
\mathrm{TP}_c = \sum_i [\,y_i = c \wedge \hat{y}_i = c\,], \qquad
n_c = \sum_i [\,y_i = c\,], \qquad
m_c = \sum_i [\,\hat{y}_i = c\,]
$$

($n_c$ is the class's gold support, $m_c$ how often the system predicted it.)

$$
P_c = \frac{\mathrm{TP}_c}{m_c}, \qquad
R_c = \frac{\mathrm{TP}_c}{n_c}, \qquad
F1_c = \frac{2\,P_c R_c}{P_c + R_c}
$$

If a denominator is zero (a class that is never predicted, or $P_c = R_c = 0$), the value is defined as 0. This is the conservative choice: a system that never predicts `fear` gets no credit for `fear`.

$$
\text{Weighted F1} = \sum_{c \in \mathcal{C}} \frac{n_c}{N}\, F1_c
\qquad
\text{Macro F1} = \frac{1}{K} \sum_{c \in \mathcal{C}} F1_c
\qquad
\text{Accuracy} = \frac{1}{N} \sum_i [\,y_i = \hat{y}_i\,]
$$

Macro F1 always averages over all seven classes, including any class with zero support in the evaluated subset. All scores are reported on a 0–100 scale.

## Why these metrics

- **Weighted F1 is the standard MELD metric.** The MELD paper (Poria et al., 2019) reports weighted F1 for its baselines, and the three papers in our literature review report MELD results the same way: EmoBERTa (Kim and Vossen, 2021) and CoMPM (Lee and Lee, 2022). Using it keeps our numbers comparable with published results.
- **Macro F1 is needed because MELD is imbalanced.** `neutral` is 48% of the test set while `fear` and `disgust` are about 2–3% each. Weighted F1 is dominated by the large classes: the all-`neutral` baseline below already gets 31.27 weighted F1 but only 9.28 macro F1. Macro F1 weighs every class equally (Sokolova and Lapalme, 2009), so gains or losses on rare emotions stay visible. This matters for our research question, since emoji features may help specific emotions more than others.
- **Per-class scores and the confusion matrix** are printed so error analysis does not need separate code.
- **Accuracy** is reported for reference only; on imbalanced data it says little beyond the majority rate.

## Design choices

- **Standard library only.** Same as `prepare_data.py`: the scorer runs anywhere Python 3 runs, with no package versions that could change results.
- **Strict input checks.** The script stops with an error instead of producing a score if any gold id has no prediction, a prediction id is not in the gold file, an id appears twice, a label is not one of the seven classes, or a prediction line has fields other than `id` and `label`. A silently wrong score is worse than no score.
- **Matching by `id`**, never by line order (see interface.md).
- **`--exclude-ids`** reports the same predictions on a subset. Following the mentor's feedback, every test result is reported both on the full test set and without the 11 suspected-overlap utterances in `eval/suspected_overlap_test_ids.json`. The prediction file must still cover the full test set, so one file serves both reports; every excluded id must exist in the gold file, so a wrong split or a typo is caught.
- **Verified against scikit-learn.** Implementing the formulas directly keeps the definitions explicit. On 200 random prediction sets over the real test set (uniform, mostly correct, never predicting some classes, all-`neutral`), weighted F1, macro F1, accuracy and every per-class precision/recall/F1 match `sklearn.metrics` (Pedregosa et al., 2011) with `zero_division=0` to within 1.1e-16. `tests/test_score.py` covers a hand-computed example and every rejected input.

## Usage

```bash
python score.py --gold data/processed/test.jsonl --pred output/simple-baseline/test.jsonl
python score.py --gold data/processed/test.jsonl --pred output/simple-baseline/test.jsonl \
    --exclude-ids eval/suspected_overlap_test_ids.json
python score.py --gold data/processed/dev.jsonl --pred output/simple-baseline/dev.jsonl --json output/simple-baseline/dev_scores.json
```

| Argument | Meaning |
|---|---|
| `--gold` | Gold JSONL (`data/processed/{dev,test}.jsonl`) |
| `--pred` | Prediction JSONL with `id` and `label` |
| `--exclude-ids` | Optional JSON file `{"ids": [...]}` of gold ids to leave out |
| `--json` | Optional path; writes all metrics, per-class scores and the confusion matrix as JSON |
| `--labels` | Label map, default `data/label_map.json` |

Exit code 0 on success, 1 on any input error (the message names the file and line).

### Example output

Predicting `neutral` for all 2,610 test utterances (what the simple baseline is expected to produce):

```text
gold: data/processed/test.jsonl  pred: output/simple-baseline/test.jsonl  n=2610

weighted_f1  31.27
macro_f1      9.28
accuracy     48.12

label      precision  recall      f1  support  predicted
neutral        48.12  100.00   64.98     1256       2610
surprise        0.00    0.00    0.00      281          0
fear            0.00    0.00    0.00       50          0
sadness         0.00    0.00    0.00      208          0
joy             0.00    0.00    0.00      402          0
disgust         0.00    0.00    0.00       68          0
anger           0.00    0.00    0.00      345          0

confusion matrix (rows = gold, columns = predicted)
           neutral surpris    fear sadness     joy disgust   anger
neutral       1256       0       0       0       0       0       0
surprise       281       0       0       0       0       0       0
fear            50       0       0       0       0       0       0
sadness        208       0       0       0       0       0       0
joy            402       0       0       0       0       0       0
disgust         68       0       0       0       0       0       0
anger          345       0       0       0       0       0       0
```

With `--exclude-ids eval/suspected_overlap_test_ids.json` the header reads `n=2599 (11 ids excluded via eval/suspected_overlap_test_ids.json)` and the scores are 31.45 weighted F1, 9.30 macro F1, 48.29 accuracy.

An incomplete prediction file is rejected:

```text
score.py: error: 2605 gold ids have no prediction (e.g. ['test:100:0', 'test:100:1', 'test:100:10'])
```

### Tests

```bash
python -m unittest discover tests
```

## References

- Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. [MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://aclanthology.org/P19-1050/). *ACL*, 527–536.
- Taewoon Kim and Piek Vossen. 2021. [EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa](https://arxiv.org/abs/2108.12009). *arXiv:2108.12009*.
- Joosung Lee and Wooin Lee. 2022. [CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking for Emotion Recognition in Conversation](https://aclanthology.org/2022.naacl-main.416/). *NAACL-HLT*, 5669–5679.
- Marina Sokolova and Guy Lapalme. 2009. A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4):427–437.
- Fabian Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12:2825–2830.
- Wikipedia: [F-score](https://en.wikipedia.org/wiki/F-score).
