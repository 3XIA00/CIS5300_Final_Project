"""Check score.compute_metrics against scikit-learn on the real test set.

Skipped automatically when scikit-learn is not installed. To run it without
installing anything globally (from the repository root):

    uv run --no-project --with scikit-learn python -m unittest discover tests
"""

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import score  # noqa: E402

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
except ImportError:  # pragma: no cover - depends on the environment
    f1_score = None


@unittest.skipIf(f1_score is None, "scikit-learn is not installed")
class MatchesSklearnTest(unittest.TestCase):
    TOLERANCE = 1e-12
    RUNS = 200

    @classmethod
    def setUpClass(cls):
        cls.labels = score.load_labels(ROOT / "data" / "label_map.json")
        gold = score.load_id_labels(ROOT / "data" / "processed" / "test.jsonl", cls.labels, strict_fields=False)
        cls.gold = [gold[i] for i in sorted(gold)]

    def predictions(self, seed):
        """Four kinds of prediction set, cycling with the seed."""
        rng = random.Random(seed)
        labels, gold = self.labels, self.gold
        kind = seed % 4
        if kind == 0:  # uniform random
            return [rng.choice(labels) for _ in gold]
        if kind == 1:  # mostly correct
            return [g if rng.random() < 0.6 else rng.choice(labels) for g in gold]
        if kind == 2:  # some classes never predicted
            return [rng.choice(labels[:3]) for _ in gold]
        return ["neutral"] * len(gold)  # majority class

    def test_all_metrics_match(self):
        options = dict(labels=self.labels, zero_division=0)
        for seed in range(self.RUNS):
            predicted = self.predictions(seed)
            ours = score.compute_metrics(self.gold, predicted, self.labels)
            precision, recall, f1, _ = precision_recall_fscore_support(self.gold, predicted, **options)
            expected = {
                "weighted_f1": f1_score(self.gold, predicted, average="weighted", **options),
                "macro_f1": f1_score(self.gold, predicted, average="macro", **options),
                "accuracy": accuracy_score(self.gold, predicted),
            }
            with self.subTest(seed=seed):
                for name, value in expected.items():
                    self.assertAlmostEqual(ours[name], value, delta=self.TOLERANCE, msg=name)
                for k, label in enumerate(self.labels):
                    c = ours["per_class"][label]
                    self.assertAlmostEqual(c["precision"], precision[k], delta=self.TOLERANCE)
                    self.assertAlmostEqual(c["recall"], recall[k], delta=self.TOLERANCE)
                    self.assertAlmostEqual(c["f1"], f1[k], delta=self.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
