#!/usr/bin/env python3
"""Predict the training-set majority emotion for every MELD utterance."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from score import DEFAULT_LABELS, ScoringError, load_id_labels, load_labels, read_jsonl

DATA = Path("data/processed")
OUTPUT = Path("output/simple-baseline")


def read_prediction_ids(path):
    """Read only the IDs needed to produce one prediction per utterance."""
    ids = {}
    for number, record in read_jsonl(path):
        item_id = record.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ScoringError(f"{path}:{number}: expected a nonempty string id")
        if item_id in ids:
            raise ScoringError(f"{path}:{number}: duplicate id {item_id}")
        ids[item_id] = None
    if not ids:
        raise ScoringError(f"{path}: no records")
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="test",
                        help="MELD split to predict (default: test)")
    parser.add_argument("--train", type=Path, default=DATA / "train.jsonl",
                        help="training JSONL (default: data/processed/train.jsonl)")
    parser.add_argument("--output", type=Path,
                        help="prediction JSONL (default: output/simple-baseline/<split>.jsonl)")
    args = parser.parse_args(argv)

    input_path = DATA / f"{args.split}.jsonl"
    output_path = args.output or OUTPUT / f"{args.split}.jsonl"

    try:
        labels = load_labels(DEFAULT_LABELS)
        train_labels = load_id_labels(args.train, labels, strict_fields=False)
        counts = Counter(train_labels.values())
        # max keeps the first label on a tie, so label-map order breaks ties.
        majority = max(labels, key=counts.__getitem__)
        ids = read_prediction_ids(input_path)
        if output_path.resolve() in {args.train.resolve(), input_path.resolve()}:
            raise ScoringError(f"{output_path}: output would overwrite an input file")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for item_id in ids:
                output.write(json.dumps({"id": item_id, "label": majority}) + "\n")
    except (ScoringError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"simple-baseline.py: error: {error}", file=sys.stderr)
        return 1

    print(f"Training majority: {majority} ({counts[majority]:,}/{len(train_labels):,})")
    print(f"Wrote {len(ids):,} predictions to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
