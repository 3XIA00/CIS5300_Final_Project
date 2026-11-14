"""Tests for score.py. Run from the repository root: python -m unittest discover tests"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import score  # noqa: E402

LABELS = score.load_labels(ROOT / "data" / "label_map.json")


class ComputeMetricsTest(unittest.TestCase):
    def test_hand_computed_example(self):
        # gold: neutral x3, joy x1; pred: neutral, neutral, joy, joy
        # neutral: P=2/2 R=2/3 F1=0.8; joy: P=1/2 R=1/1 F1=2/3; other five classes: F1=0
        m = score.compute_metrics(["neutral", "neutral", "neutral", "joy"],
                                  ["neutral", "neutral", "joy", "joy"], LABELS)
        self.assertAlmostEqual(m["per_class"]["neutral"]["f1"], 0.8)
        self.assertAlmostEqual(m["per_class"]["joy"]["f1"], 2 / 3)
        self.assertAlmostEqual(m["weighted_f1"], (3 * 0.8 + 1 * 2 / 3) / 4)
        self.assertAlmostEqual(m["macro_f1"], (0.8 + 2 / 3) / 7)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertEqual(m["confusion_matrix"][LABELS.index("neutral")][LABELS.index("joy")], 1)

    def test_perfect_predictions(self):
        gold = LABELS * 2
        m = score.compute_metrics(gold, gold, LABELS)
        self.assertEqual((m["weighted_f1"], m["macro_f1"], m["accuracy"]), (1.0, 1.0, 1.0))

    def test_unpredicted_class_scores_zero_without_error(self):
        m = score.compute_metrics(["fear", "neutral"], ["neutral", "neutral"], LABELS)
        self.assertEqual(m["per_class"]["fear"]["f1"], 0.0)
        self.assertEqual(m["per_class"]["fear"]["precision"], 0.0)


class FileHandlingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gold = self.write("gold.jsonl", [
            {"id": "test:0:0", "label": "joy", "text": "Hi!"},
            {"id": "test:0:1", "label": "neutral", "text": "Okay."},
            {"id": "test:1:0", "label": "anger", "text": "No!"},
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, records):
        path = self.dir / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return path

    def run_score(self, pred_records, *extra):
        pred = self.write("pred.jsonl", pred_records)
        return score.main(["--gold", str(self.gold), "--pred", str(pred), *extra])

    def preds(self, *labels):
        ids = ["test:0:0", "test:0:1", "test:1:0"]
        return [{"id": i, "label": l} for i, l in zip(ids, labels)]

    def test_order_does_not_matter(self):
        records = self.preds("joy", "neutral", "anger")
        out = self.dir / "a.json"
        self.assertEqual(self.run_score(list(reversed(records)), "--json", str(out)), 0)
        self.assertEqual(json.loads(out.read_text())["accuracy"], 1.0)

    def test_rejects_missing_extra_duplicate_and_unknown(self):
        good = self.preds("joy", "neutral", "anger")
        bad_inputs = [
            good[:2],                                             # missing id
            good + [{"id": "test:9:9", "label": "joy"}],          # extra id
            good + [good[0]],                                     # duplicate id
            self.preds("joy", "neutral", "angry"),                # unknown label
            [dict(good[0], score=0.9)] + good[1:],                # extra field
            [{"id": "test:0:0", "lable": "joy"}] + good[1:],      # misspelled field
        ]
        for records in bad_inputs:
            with self.subTest(records=records):
                self.assertEqual(self.run_score(records), 1)

    def test_exclude_ids(self):
        exclude = self.dir / "exclude.json"
        exclude.write_text(json.dumps({"ids": ["test:1:0"]}), encoding="utf-8")
        out = self.dir / "m.json"
        # the only wrong prediction is on the excluded item
        self.assertEqual(self.run_score(self.preds("joy", "neutral", "joy"),
                                        "--exclude-ids", str(exclude), "--json", str(out)), 0)
        metrics = json.loads(out.read_text())
        self.assertEqual((metrics["n"], metrics["accuracy"]), (2, 1.0))

    def test_exclude_ids_must_exist_in_gold(self):
        exclude = self.dir / "exclude.json"
        exclude.write_text(json.dumps({"ids": ["dev:1:0"]}), encoding="utf-8")
        self.assertEqual(self.run_score(self.preds("joy", "neutral", "anger"),
                                        "--exclude-ids", str(exclude)), 1)

    def test_overlap_list_matches_real_test_ids(self):
        ids = score.load_exclude_ids(ROOT / "eval" / "suspected_overlap_test_ids.json")
        gold = score.load_id_labels(ROOT / "data" / "processed" / "test.jsonl", LABELS, strict_fields=False)
        self.assertEqual(len(ids), 11)
        self.assertTrue(ids <= gold.keys())


if __name__ == "__main__":
    unittest.main()
