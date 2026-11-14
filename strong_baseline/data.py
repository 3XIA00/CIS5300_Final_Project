"""Load MELD records and build model inputs.

This module uses only the standard library, so it can be tested without torch.

Input construction is isolated in `build_input` on purpose: the Milestone 2
baseline (condition B) uses only the current utterance, and Milestone 3 adds
dialogue history (conditions C, CR, CD) by changing only how inputs are built.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
LABEL_MAP = ROOT / "data" / "label_map.json"
SPLITS = ("train", "train_clean", "dev", "test")


def load_label_map(path=LABEL_MAP):
    """Return {label name: label id} from data/label_map.json."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def label_names(label_map):
    """Label names ordered by id, so index k of the classifier output is label k."""
    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def load_split(split, data_dir=DATA_DIR):
    """Read data/processed/<split>.jsonl as a list of dicts, in file order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = Path(data_dir) / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_dialogues(records):
    """Group records by dialogue and sort each dialogue by utterance_id.

    Utterance ids can have gaps (data.md), so order comes from sorting the
    observed turns, never from arithmetic on ids.
    """
    dialogues = defaultdict(list)
    for record in records:
        dialogues[record["dialogue_id"]].append(record)
    for turns in dialogues.values():
        turns.sort(key=lambda record: record["utterance_id"])
    return dict(dialogues)


def previous_turns(records, n_history):
    """Return {id: [up to n_history previous records of the same dialogue, oldest first]}.

    With n_history=0 (the Milestone 2 baseline) every list is empty.
    """
    if n_history < 0:
        raise ValueError("n_history must be >= 0")
    history = {}
    for turns in group_dialogues(records).values():
        for position, record in enumerate(turns):
            start = max(0, position - n_history)
            history[record["id"]] = turns[start:position] if n_history else []
    return history


def format_turn(record):
    """'Speaker: text', the speaker-marked format used by EmoBERTa."""
    return f"{record['speaker']}: {record['text']}"


def build_input(record, history=()):
    """Return (target_text, history_text_or_None) for one utterance.

    The target is always the current utterance. Condition B passes no history.
    With history, the caller tokenizes the pair with truncation="only_second"
    so the oldest history is dropped first and the target is kept whole.
    """
    target = format_turn(record)
    if not history:
        return target, None
    # Most recent turn first, so truncating the end of the second segment drops the oldest turns.
    return target, " </s> ".join(format_turn(turn) for turn in reversed(list(history)))
