"""Tests for strong_baseline.data (standard library only; no torch needed).

Run from the repository root: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strong_baseline import data  # noqa: E402


def rec(dialogue, utterance, speaker="A", text="hi", label="neutral"):
    return {"id": f"test:{dialogue}:{utterance}", "dialogue_id": dialogue, "utterance_id": utterance,
            "speaker": speaker, "text": text, "label": label}


class BuildInputTest(unittest.TestCase):
    def test_target_only_is_speaker_colon_text(self):
        self.assertEqual(data.build_input(rec(0, 0, "Ross", "We were on a break!")),
                         ("Ross: We were on a break!", None))

    def test_history_is_most_recent_first(self):
        history = [rec(0, 0, "A", "one"), rec(0, 1, "B", "two")]
        target, context = data.build_input(rec(0, 2, "C", "three"), history)
        self.assertEqual(target, "C: three")
        self.assertEqual(context, "B: two </s> A: one")


class PreviousTurnsTest(unittest.TestCase):
    def test_uses_observed_order_not_id_arithmetic(self):
        # utterance 1 is missing and records are shuffled
        records = [rec(5, 3), rec(5, 0), rec(5, 2), rec(7, 0)]
        history = data.previous_turns(records, 1)
        self.assertEqual([r["id"] for r in history["test:5:2"]], ["test:5:0"])
        self.assertEqual([r["id"] for r in history["test:5:3"]], ["test:5:2"])
        self.assertEqual(history["test:7:0"], [])

    def test_zero_history_is_empty_for_all(self):
        records = [rec(0, i) for i in range(4)]
        self.assertTrue(all(v == [] for v in data.previous_turns(records, 0).values()))


class RealDataTest(unittest.TestCase):
    def test_splits_and_labels_load(self):
        label_map = data.load_label_map()
        names = data.label_names(label_map)
        self.assertEqual(names[0], "neutral")
        self.assertEqual(len(names), 7)
        sizes = {split: len(data.load_split(split)) for split in ("train", "dev", "test")}
        self.assertEqual(sizes, {"train": 9989, "dev": 1109, "test": 2610})
        for record in data.load_split("dev"):
            self.assertIn(record["label"], label_map)


if __name__ == "__main__":
    unittest.main()
