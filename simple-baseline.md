# Simple Baseline: Majority-Class Prediction

For each MELD utterance, predict the emotion that occurs most often in the official training split. The script counts gold training labels and breaks ties by the lowest label ID in [the label map](data/label_map.json). It uses only the utterance IDs from the prediction split. On the official train split, the majority class is `neutral` (4,710 of 9,989 utterances).

## Run

From the repository root with Python 3:

```sh
python simple-baseline.py --split dev
python simple-baseline.py --split test
python score.py --gold data/processed/dev.jsonl --pred output/simple-baseline/dev.jsonl
python score.py --gold data/processed/test.jsonl --pred output/simple-baseline/test.jsonl
python score.py --gold data/processed/test.jsonl --pred output/simple-baseline/test.jsonl --exclude-ids eval/suspected_overlap_test_ids.json
```

The baseline uses only the Python standard library. The default split is `test`. You can also run `python /path/to/CIS5300_Final_Project/simple-baseline.py --split test` from another directory. The default input and output paths are relative to the repository root. If you pass `--train PATH` or `--output PATH`, a relative path is resolved from your current directory. [The notebook](simple-baseline.ipynb) runs the same commands in Jupyter or Colab.

The first two lines of `output/simple-baseline/test.jsonl` follow the [shared prediction interface](interface.md):

```jsonl
{"id": "test:0:0", "label": "neutral"}
{"id": "test:0:1", "label": "neutral"}
```

## Results

[The project scorer](scoring.md) reports scores on a 0–100 scale. The overlap check scores the same full test prediction file after excluding the 11 suspected same-source records listed in [the exclusion file](eval/suspected_overlap_test_ids.json).

| Evaluation split | Utterances | Weighted F1 | Macro F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dev | 1,109 | 25.23 | 8.50 | 42.38 |
| Full test | 2,610 | 31.27 | 9.28 | 48.12 |
| Test excluding 11 suspected overlaps | 2,599 | 31.45 | 9.30 | 48.29 |

On full test, the baseline correctly predicts all 1,256 `neutral` utterances. Every other class has an F1 of zero, which accounts for the low macro F1.
