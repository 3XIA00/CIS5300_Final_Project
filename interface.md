# Shared Interface for Milestone 2 and Later

Every system in this project (simple baseline, strong baseline, later extensions) writes predictions in one format, and `score.py` is the only place that turns predictions into scores. This file fixes that contract so the pieces can be built in parallel.

## 1. Prediction files

**Format:** UTF-8 JSON Lines, one object per utterance, exactly two fields:

```json
{"id": "test:0:0", "label": "surprise"}
{"id": "test:0:1", "label": "anger"}
```

| Field | Type | Rule |
|---|---|---|
| `id` | string | The `id` of the gold record, `split:dialogue_id:utterance_id`, copied unchanged from `data/processed/<split>.jsonl` |
| `label` | string | One of the seven names in `data/label_map.json`: `neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger` |

Rules:

- One line per gold utterance in the split: no missing, extra or duplicate `id`s. Line order does not matter.
- Labels are lowercase names, not integer ids.
- Extra fields are rejected rather than ignored, so a typo such as `"lable"` fails loudly instead of being silently skipped.

**Why this design**

- **`id` instead of line order.** Matching by line number breaks silently as soon as a model shuffles, batches or sorts its outputs; matching by `id` makes `score.py` independent of order and lets it detect missing or duplicated items. The `id` already exists in every processed record and is unique across splits (data.md).
- **Label names instead of integer ids.** Names are readable when you open a prediction file, and they remove the risk of two components using different integer orders. `label_map.json` stays the single source of truth for the label set.
- **JSONL, two fields only.** It is the same format as the gold data, it can be read and written with the standard library, and each line can be inspected on its own. Keeping only `id` and `label` means every system, from a one-line baseline to a fine-tuned model, can produce it the same way. Probabilities or logits, if needed for analysis, go in a separate file next to the predictions rather than into this contract.

## 2. Gold files

Gold labels are read directly from `data/processed/{train,dev,test}.jsonl`, using their `id` and `label` fields. No separate gold copy is created, so predictions are always scored against the checksummed Milestone 1 data.

The main protocol uses the official splits (`train.jsonl` for training). `train_clean.jsonl` is only a sensitivity check (data.md).

## 3. Suspected overlap subset

`eval/suspected_overlap_test_ids.json` lists the 11 test utterances that data.md flags as suspected same-source overlap with training data (`test:84:0–2`, `test:85:0–5`, `test:155:0–1`). Following the mentor's feedback, every test result is reported twice: on the full test set, and with these ids excluded (`score.py --exclude-ids`). No retraining is involved: the same predictions are scored on a smaller gold set.

**Why it lives in `eval/` and not `data/`:** `data/` is the Milestone 1 deliverable. Every file in it is covered by `data/manifest.json` checksums and mirrored in `meld_text_data.tar.gz`; adding a file there would make the archive and the folder disagree. Evaluation settings belong with the evaluation code.

## 4. Directory layout

```text
score.py                  # the only scorer
scoring.md                # metric definitions, citations, usage
simple-baseline.py        # majority-class baseline (course file name)
simple-baseline.md
strong_baseline/          # RoBERTa-base code (package; underscores so it can be imported)
strong-baseline.md
deepmoji/                 # frozen DeepMoji features: setup, extraction, checks (deepmoji.md)
deepmoji.md
eval/
  suspected_overlap_test_ids.json
output/
  <system>/<split>.jsonl          # predictions, e.g. output/simple-baseline/test.jsonl
  <system>/seed<N>/<split>.jsonl  # seeded systems, e.g. output/roberta-base/seed13/test.jsonl
tests/
  test_score.py
```

- `output/` is committed: the final deliverable must include test predictions next to the gold labels, and committed predictions let anyone re-run `score.py` without a GPU.
- Model checkpoints are **not** committed (`.gitignore`): RoBERTa-base checkpoints are about 500 MB each and can be regenerated from the code and seed.
- File names with hyphens (`simple-baseline.py`, `strong-baseline.md`) follow the course page; the strong-baseline code is a directory with an underscore because Python cannot import a package whose name contains a hyphen.

## 5. Changing this interface

Changes affect every component, so propose them in the group room first and update this file, `score.py` and its tests in the same pull request.
