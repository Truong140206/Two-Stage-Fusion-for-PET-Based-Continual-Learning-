"""CPU-only protocol and metric tests for the 5-Datasets extension."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import run_ranpac_fivedatasets as five


class FiveDatasetRanPACTests(unittest.TestCase):
    def test_fixed_scope_and_order(self):
        self.assertEqual(five.TASKS,
                         ("SVHN", "MNIST", "CIFAR10", "NotMNIST", "FashionMNIST"))
        self.assertFalse(five.CONFIG["shuffle"])
        self.assertEqual(five.CONFIG["increment"], 10)
        self.assertEqual(five.CONFIG["model_name"], "ssf")
        self.assertTrue(five.CONFIG["use_RP"])
        self.assertEqual(five.CONFIG["M"], 10000)

    def test_transform_families_match_existing_pipeline(self):
        self.assertEqual([five.transform_family(task) for task in five.TASKS],
                         ["generic", "generic", "cifar", "cifar", "cifar"])

    def test_only_complete_tasks_are_selected(self):
        self.assertEqual(five.selected_tasks(range(20)), [0, 1])
        self.assertEqual(five.selected_tasks(range(30, 50)), [3, 4])
        with self.assertRaisesRegex(ValueError, "complete"):
            five.selected_tasks([0, 1, 2])
        with self.assertRaisesRegex(ValueError, "outside"):
            five.selected_tasks(range(50, 60))

    def test_resolve_prepared_data_root_with_proven_verifier(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            calls = []

            def resolver(path, dataset):
                calls.append((path, dataset))
                return path

            self.assertEqual(five.resolve_data(root, resolver), root.resolve())
            self.assertEqual(calls, [(root.resolve(), "fivedatasets")])

    def test_resolver_cannot_redirect_data_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaisesRegex(ValueError, "changed"):
                five.resolve_data(root, lambda _path, _dataset: root / "elsewhere")

    def test_scipy_installed_only_in_private_environment(self):
        with tempfile.TemporaryDirectory() as d:
            support = Path(d)
            python = support / "env-py39/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(five.subprocess, "run", side_effect=[
                    Mock(returncode=1), Mock(returncode=0), Mock(returncode=0),
                    Mock(returncode=0)]) as run:
                five.ensure_five_environment(support, {}, python, True)
            self.assertIn("scipy==1.10.1", run.call_args_list[1].args[0])
            self.assertEqual(json.loads(
                (support / "environment-five-ready.json").read_text()),
                {"scipy": "1.10.1"})

    def make_results(self, root):
        results = root / "results"
        pred = results / "class_preds"
        pred.mkdir(parents=True)
        fields = [f"{kind}_task_{stage}" for stage in range(5)
                  for kind in ("pred", "true")]
        with (pred / "fivedatasets_extension_class_preds_publish_7.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for target in range(50):
                row = {}
                for stage in range(5):
                    value = target if target < (stage + 1) * 10 else -1
                    row[f"pred_task_{stage}"] = value
                    row[f"true_task_{stage}"] = value
                writer.writerow(row)
        with (results / "fivedatasets_extension_publish_7.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["top1_total", "ave_acc"])
            writer.writeheader()
            writer.writerows([{"top1_total": 100, "ave_acc": 100}] * 5)
        return {"counts": {"train": [1] * 50, "test": [1] * 50}}

    def test_summary_is_explicitly_non_official(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = five.summarize(root, self.make_results(root))
            self.assertEqual(result["status"], "exploratory_non_official_extension")
            self.assertEqual(result["final"]["Acc@1"], 100)
            self.assertEqual(result["final"]["Forgetting"], 0)
            self.assertEqual(result["final"]["Backward"], 0)


if __name__ == "__main__":
    unittest.main()
