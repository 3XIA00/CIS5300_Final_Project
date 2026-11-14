#!/usr/bin/env python3
"""Score emotion predictions on MELD against the gold labels.

Usage:
    python score.py --gold data/processed/test.jsonl --pred output/<system>/test.jsonl
    python score.py --gold data/processed/test.jsonl --pred output/<system>/test.jsonl \
        --exclude-ids eval/suspected_overlap_test_ids.json

Prints weighted F1 (the standard MELD metric), macro F1, accuracy, per-class
precision/recall/F1 and a confusion matrix. All scores are on a 0-100 scale.
The file formats are defined in interface.md; the metrics in scoring.md.

Uses only the Python standard library.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = ROOT / "data" / "label_map.json"
PREDICTION_FIELDS = {"id", "label"}


class ScoringError(Exception):
    """Raised for any input that would make the score untrustworthy."""


def read_jsonl(path):
    """Yield (line_number, object) for each non-empty line of a JSON Lines file."""
    path = Path(path)
    if not path.is_file():
        raise ScoringError(f"{path}: file not found")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ScoringError(f"{path}:{number}: invalid JSON ({error.msg})") from None
            if not isinstance(record, dict):
                raise ScoringError(f"{path}:{number}: expected a JSON object")
            yield number, record


def load_labels(path):
    """Return label names in label-id order, so tables always list classes the same way."""
    with Path(path).open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def load_id_labels(path, labels, *, strict_fields):
    """Read {id: label} from a gold or prediction file, rejecting anything ambiguous."""
    known = set(labels)
    result = {}
    for number, record in read_jsonl(path):
        where = f"{path}:{number}"
        if strict_fields and set(record) != PREDICTION_FIELDS:
            raise ScoringError(f"{where}: expected exactly the fields id and label, got {sorted(record)}")
        if "id" not in record or "label" not in record:
            raise ScoringError(f"{where}: missing id or label")
        item_id, label = record["id"], record["label"]
        if not isinstance(item_id, str) or not isinstance(label, str):
            raise ScoringError(f"{where}: id and label must be strings")
        if label not in known:
            raise ScoringError(f"{where}: unknown label {label!r}; expected one of {labels}")
        if item_id in result:
            raise ScoringError(f"{where}: duplicate id {item_id}")
        result[item_id] = label
    if not result:
        raise ScoringError(f"{path}: no records")
    return result


def load_exclude_ids(path):
    """Read the ids to leave out, from a JSON object with an "ids" list."""
    with Path(path).open(encoding="utf-8") as handle:
        content = json.load(handle)
    ids = content.get("ids") if isinstance(content, dict) else None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ScoringError(f'{path}: expected a JSON object with an "ids" list of strings')
    if len(set(ids)) != len(ids):
        raise ScoringError(f"{path}: duplicate ids")
    return set(ids)


def align(gold, predictions, exclude=frozenset()):
    """Pair gold and predicted labels by id; the prediction file must cover the full gold set."""
    missing = sorted(gold.keys() - predictions.keys())
    extra = sorted(predictions.keys() - gold.keys())
    if missing or extra:
        details = []
        if missing:
            details.append(f"{len(missing)} gold ids have no prediction (e.g. {missing[:3]})")
        if extra:
            details.append(f"{len(extra)} predicted ids are not in the gold file (e.g. {extra[:3]})")
        raise ScoringError("; ".join(details))
    unknown = sorted(exclude - gold.keys())
    if unknown:
        raise ScoringError(f"{len(unknown)} excluded ids are not in the gold file (e.g. {unknown[:3]})")
    ids = sorted(i for i in gold if i not in exclude)
    return [gold[i] for i in ids], [predictions[i] for i in ids]


def compute_metrics(gold_labels, predicted_labels, labels):
    """Compute all metrics as fractions in [0, 1]; see scoring.md for the definitions."""
    if len(gold_labels) != len(predicted_labels) or not gold_labels:
        raise ScoringError("gold and predicted label lists must be non-empty and equally long")
    pairs = Counter(zip(gold_labels, predicted_labels))
    confusion = [[pairs[(g, p)] for p in labels] for g in labels]
    total = len(gold_labels)
    per_class = {}
    for k, label in enumerate(labels):
        tp = confusion[k][k]
        support = sum(confusion[k])                      # gold count of this class
        predicted = sum(row[k] for row in confusion)     # times the system chose it
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": support, "predicted": predicted}
    return {
        "n": total,
        "weighted_f1": sum(c["f1"] * c["support"] for c in per_class.values()) / total,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(labels),
        "accuracy": sum(confusion[k][k] for k in range(len(labels))) / total,
        "per_class": per_class,
        "labels": list(labels),
        "confusion_matrix": confusion,
    }


def format_report(metrics, header):
    pct = lambda x: f"{100 * x:6.2f}"
    lines = [header, "",
             f"weighted_f1 {pct(metrics['weighted_f1'])}",
             f"macro_f1    {pct(metrics['macro_f1'])}",
             f"accuracy    {pct(metrics['accuracy'])}", "",
             f"{'label':<10}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}{'predicted':>11}"]
    for label in metrics["labels"]:
        c = metrics["per_class"][label]
        lines.append(f"{label:<10}{pct(c['precision']):>10}{pct(c['recall']):>8}{pct(c['f1']):>8}"
                     f"{c['support']:>9}{c['predicted']:>11}")
    short = [label[:7] for label in metrics["labels"]]
    lines += ["", "confusion matrix (rows = gold, columns = predicted)",
              f"{'':<10}" + "".join(f"{s:>8}" for s in short)]
    for label, row in zip(metrics["labels"], metrics["confusion_matrix"]):
        lines.append(f"{label:<10}" + "".join(f"{v:>8}" for v in row))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True, help="gold JSONL, e.g. data/processed/test.jsonl")
    parser.add_argument("--pred", required=True, help="prediction JSONL with id and label (interface.md)")
    parser.add_argument("--exclude-ids", help='JSON file {"ids": [...]} of gold ids to leave out of the score')
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="label map (default: data/label_map.json)")
    parser.add_argument("--json", dest="json_out", help="also write all metrics to this JSON file")
    args = parser.parse_args(argv)

    try:
        labels = load_labels(args.labels)
        gold = load_id_labels(args.gold, labels, strict_fields=False)
        predictions = load_id_labels(args.pred, labels, strict_fields=True)
        exclude = load_exclude_ids(args.exclude_ids) if args.exclude_ids else set()
        gold_labels, predicted_labels = align(gold, predictions, exclude)
        metrics = compute_metrics(gold_labels, predicted_labels, labels)
    except (ScoringError, OSError, json.JSONDecodeError) as error:
        print(f"score.py: error: {error}", file=sys.stderr)
        return 1

    header = f"gold: {args.gold}  pred: {args.pred}  n={metrics['n']}"
    if exclude:
        header += f" ({len(exclude)} ids excluded via {args.exclude_ids})"
    print(format_report(metrics, header))
    if args.json_out:
        record = {"gold": args.gold, "pred": args.pred, "exclude_ids": args.exclude_ids,
                  "excluded": len(exclude), **metrics}
        Path(args.json_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
