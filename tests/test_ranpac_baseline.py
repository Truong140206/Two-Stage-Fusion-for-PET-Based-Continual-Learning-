"""CPU-only tests; these do not certify a GPU RanPAC run."""
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import run_ranpac_baseline as runner


class RanPACTests(unittest.TestCase):
    def test_unshuffled_order(self):
        self.assertEqual(runner.class_order(42, False), list(range(200)))

    def test_paper_python_shuffle(self):
        expected = list(range(200))
        random.Random(42).shuffle(expected)
        self.assertEqual(runner.class_order(42, True), expected)

    def test_order_preserves_global_rng(self):
        state = random.getstate()
        runner.class_order(42, True)
        self.assertEqual(state, random.getstate())

    def test_ambiguous_shuffle_rejected(self):
        with self.assertRaises(ValueError):
            runner.class_order(42, "False")

    def test_task_balanced_not_pooled(self):
        result = runner.exact_metrics([0, 0, 0, 0], [0, 0, 0, 20], 2)
        self.assertEqual(result["Acc@1"], 50)
        self.assertEqual(result["pooled_acc"], 75)
        self.assertEqual(result["task_counts"], [3, 1])

    def test_missing_task_rejected(self):
        with self.assertRaises(ValueError):
            runner.exact_metrics([0], [0], 2)

    def test_empty_predictions_rejected(self):
        with self.assertRaises(ValueError):
            runner.exact_metrics([], [], 1)

    def test_unseen_prediction_rejected(self):
        with self.assertRaises(ValueError):
            runner.exact_metrics([20], [0], 1)

    def test_retention_includes_final_excludes_last_task(self):
        rows = [{"per_task": [80]}, {"per_task": [85, 90]},
                {"per_task": [82, 91, 60]}]
        self.assertEqual(runner.retention(rows), {"Forgetting": 1.5, "Backward": 1.5})

    def test_bad_triangle_rejected(self):
        with self.assertRaises(ValueError):
            runner.retention([{"per_task": [80]}, {"per_task": [80]}])

    def test_single_task_retention_absent(self):
        self.assertEqual(runner.retention([{"per_task": [80]}]), {})

    def test_no_dataset_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                runner.resolve_data(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_complete_split_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "imagenet-r"
            for split in ("train", "test"):
                for i in range(200):
                    (data / split / str(i)).mkdir(parents=True)
            self.assertEqual(runner.resolve_data(root), data)
            for split in ("train", "test"):
                (root / split).mkdir()
            with self.assertRaises(ValueError):
                runner.resolve_data(root)
            self.assertEqual(runner.resolve_data(root, data), data)

    def test_changed_upstream_rejected(self):
        with patch.object(runner.subprocess, "check_output",
                          side_effect=[runner.REVISION, " M RanPAC.py"]):
            with self.assertRaises(ValueError):
                runner.check_upstream(Path("dummy"))

    def test_wrong_revision_rejected(self):
        with patch.object(runner.subprocess, "check_output", side_effect=["wrong", ""]):
            with self.assertRaises(ValueError):
                runner.check_upstream(Path("dummy"))

    def test_busy_gpu_stops_before_run(self):
        with patch.object(runner.subprocess, "check_output", return_value="12345\n"):
            with self.assertRaisesRegex(RuntimeError, "no run log"):
                runner.idle_gpu_preflight()

    def test_low_gpu_memory_rejected(self):
        with patch.object(runner.subprocess, "check_output", side_effect=["", "1000"]):
            with self.assertRaisesRegex(RuntimeError, "16 GiB"):
                runner.idle_gpu_preflight()


if __name__ == "__main__":
    unittest.main()
