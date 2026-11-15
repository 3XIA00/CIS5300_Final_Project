"""Linear probe: logistic regression on frozen DeepMoji features alone.

    python -m deepmoji.probe                               # features/deepmoji/pretrained
    python score.py --gold data/processed/test.jsonl --pred output/deepmoji-probe/test.jsonl

A feasibility check, not one of the six experimental conditions: if DeepMoji features carry emotion
information, this probe should clearly beat the majority baseline. Features are standardised with
statistics from train only. Predictions follow interface.md.
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from deepmoji.common import FEATURE_DIR, ROOT, read_jsonl


def load(feature_dir, split):
    features = np.load(feature_dir / f"{split}.npy")
    with open(feature_dir / f"{split}_ids.json", encoding="utf-8") as handle:
        ids = json.load(handle)
    gold = read_jsonl(ROOT / "data" / "processed" / f"{split}.jsonl")
    if ids != [r["id"] for r in gold]:
        raise SystemExit(f"{split}: feature rows are not aligned with data/processed/{split}.jsonl")
    return features, ids, [r["label"] for r in gold]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=str(FEATURE_DIR / "pretrained"))
    p.add_argument("--output-dir", default=str(ROOT / "output" / "deepmoji-probe"))
    args = p.parse_args(argv)
    feature_dir, out = ROOT / args.features, ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    x_train, _, y_train = load(feature_dir, "train")
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(max_iter=3000).fit(scaler.transform(x_train), y_train)
    for split in ("dev", "test"):
        x, ids, _ = load(feature_dir, split)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for i, label in zip(ids, classifier.predict(scaler.transform(x))):
                handle.write(json.dumps({"id": i, "label": str(label)}) + "\n")
        print(f"wrote {out / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
