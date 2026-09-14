"""CPU-only fixed-order protocol/preservation tests. GPU results are not implied."""
import ast
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from tools import run_ranpac_pairs as pairs
from tools import run_ranpac_pair_worker as worker


def native_record(seed):
    return {"metadata": {
        "config": dict(pairs.native.EXPECTED, seed=str(seed)),
        "revision": pairs.common.REVISION, "pretrained_sha256": pairs.trial.NPZ_SHA,
        "torch_seed_from_original_trainer": 1, "class_order": list(range(200)),
        "counts": {"test": [30] * 200}, "dataset_sha256": "images",
        "pretrained_url": "https://example.test/weights.npz"},
        "stages": [{}] * 10, "final": {"stage": 10, "task_counts": [600] * 10,
        "Acc@1": 77., "Forgetting": 4., "Backward": -4.}}


class PairTests(unittest.TestCase):
    def test_fixed_orders_not_training_seed_search(self):
        self.assertEqual(pairs.ORDERS, (1994, 1995))
        for phase in worker.PHASES:
            cmd = worker.cli_arguments(phase, Path("/new"), Path("/data"))
            self.assertEqual(cmd[cmd.index("--seed") + 1], "1")
        with self.assertRaises(ValueError):
            pairs.pair_folder(Path("/out"), 1993)

    def test_existing_drivers_unchanged(self):
        # Old run metadata includes these byte hashes (git LF content).
        hashes = {
            "run_ranpac_original.py": "34b9e7730bd3e66403c329466b432c655143160dff4a73f12b8dc1dd182cd996",
            "try_ranpac_pretrained.py": "d80a5f1c179077efbbcffcaf729fe46fc65a749764a3e292a872bb5c795b6470"}
        import hashlib
        for name, expected in hashes.items():
            content = (pairs.ROOT / "tools" / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected)

    def test_worker_reuses_all_safety_and_math_helpers(self):
        for name in ("cli_arguments", "checkpoint_audit", "atomic_checkpoint",
                     "ordered_split", "backbone_digest"):
            self.assertIs(getattr(worker, name), getattr(pairs.trial, name))

    def test_native_seed_only_config_change(self):
        for seed in pairs.ORDERS:
            row = dict(pairs.native.EXPECTED, seed=str(seed))
            self.assertEqual([k for k in row if row[k] != pairs.native.EXPECTED[k]], ["seed"])

    def test_checked_native(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.json"
            obj = native_record(1994)
            path.write_text(json.dumps(obj))
            self.assertEqual(pairs.checked_native(path, 1994), obj)
            with self.assertRaises(ValueError):
                pairs.checked_native(path, 1995)

    def test_bad_order_counts_completion_and_pretrained_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.json"
            for kind in ("order", "counts", "stages", "npz", "seed"):
                obj = native_record(1994)
                if kind == "order":
                    obj["metadata"]["class_order"] = [0] * 200
                elif kind == "counts":
                    obj["final"]["task_counts"][0] = 599
                elif kind == "stages":
                    obj["stages"].pop()
                elif kind == "npz":
                    obj["metadata"]["pretrained_sha256"] = "wrong"
                else:
                    obj["metadata"]["torch_seed_from_original_trainer"] = 2
                path.write_text(json.dumps(obj))
                with self.assertRaises(ValueError, msg=kind):
                    pairs.checked_native(path, 1994)

    def test_low_disk_requires_24_gib_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.assertEqual(pairs.needed_gib(out), 24)
            with patch.object(pairs.trial.shutil, "disk_usage",
                              return_value=SimpleNamespace(free=int(7.2 * pairs.GIB))):
                with self.assertRaisesRegex(RuntimeError, "24 GiB"):
                    pairs.require_space(out)
            self.assertEqual(list(out.iterdir()), [])

    def test_disk_reservation_decreases_after_completed_pair(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            folder = pairs.pair_folder(out, 1994) / "ours"
            folder.mkdir(parents=True)
            (folder / "full_done.json").write_text("{}")
            self.assertEqual(pairs.needed_gib(out), 12)

    def test_immutable_plan_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plan.json"
            pairs.immutable_json(path, {"a": 1})
            before = path.read_bytes()
            pairs.immutable_json(path, {"a": 1})
            with self.assertRaises(ValueError):
                pairs.immutable_json(path, {"a": 2})
            self.assertEqual(path.read_bytes(), before)

    def test_done_requires_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            artifact = root / "console.log"
            artifact.write_text("complete")
            pairs.mark_done(root, "native", [artifact])
            self.assertTrue(pairs.trial.validate_done(root, "native"))
            artifact.write_text("truncated")
            with self.assertRaises(ValueError):
                pairs.trial.validate_done(root, "native")

    def test_incomplete_native_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / "console.log").write_text("partial")
            with patch.object(pairs.native, "run_bounded") as run:
                with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                    pairs.run_native(SimpleNamespace(order_seed=1994), folder, Mock())
                run.assert_not_called()
            self.assertEqual((folder / "console.log").read_text(), "partial")

    def test_native_reuses_checked_completed_result(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "summary.json"
            path.write_text(json.dumps(native_record(1994)))
            pairs.mark_done(root, "native", [path])
            with patch.object(pairs.native, "run_bounded") as run:
                pairs.run_native(SimpleNamespace(order_seed=1994), root, Mock())
                run.assert_not_called()

    def test_native_entry_uses_private_env_and_local_seed_csv(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            support = out / "_ranpac_support/original"
            py = support / "env-py39/bin/python"
            py.parent.mkdir(parents=True)
            py.touch()
            (support / "environment-ready.json").write_text("{}")
            original = pairs.original_path(out)
            original.parent.mkdir()
            original.write_text(json.dumps(native_record(1993)))
            data = out / "images"
            data.mkdir()
            folder = pairs.pair_folder(out, 1994) / "ranpac"
            args = SimpleNamespace(output_root=out, data_root=data, order_seed=1994,
                                   cpu_threads=4, native_max_minutes=60)
            def execute(cmd, cwd, env, lock, log, minutes):
                self.assertEqual(cmd[0], str(py))
                self.assertIn("--native-child", cmd)
                self.assertEqual(cwd, folder)
                self.assertEqual(minutes, 60)
                with (folder / "args/imagenetr_publish.csv").open() as f:
                    self.assertEqual(list(csv.DictReader(f)),
                                     [dict(pairs.native.EXPECTED, seed="1994")])
                log.write_text("RANPAC_ORIGINAL_EXIT_CODE=0")
                (folder / "results/class_preds").mkdir(parents=True)
                (folder / "results/imagenetr_publish_7.csv").write_text("curve")
                (folder / "results/class_preds/imagenetr_class_preds_publish_7.csv").write_text("pred")
            with patch.object(pairs.subprocess, "check_output", return_value=json.dumps(list(range(200)))), \
                 patch.object(pairs.common, "resolve_data", return_value=data), \
                 patch.object(pairs.common, "check_upstream"), \
                 patch.object(pairs.native, "link_existing") as link, \
                 patch.object(pairs.native, "run_bounded", side_effect=execute), \
                 patch.object(pairs.native, "summarize", return_value={
                     "stages": [{}] * 10, "final": native_record(1994)["final"]}):
                pairs.run_native(args, folder, Mock())
                link.assert_called_once_with(folder / "data/imagenet-r", data)
            self.assertTrue(pairs.trial.validate_done(folder, "native"))
            self.assertEqual(json.loads(original.read_text())["metadata"]["config"]["seed"], "1993")

    def test_no_training_budget_change_in_paired_worker(self):
        for phase in worker.PHASES:
            self.assertEqual(worker.cli_arguments(phase, Path("/f"), Path("/d")),
                             pairs.trial.cli_arguments(phase, Path("/f"), Path("/d")))
        source = Path(worker.__file__).read_text()
        self.assertIn('kwargs["pretrained"] = False', source)
        self.assertIn('model.load_pretrained(str(npz))', source)
        self.assertIn('utils.save_on_master = atomic_checkpoint', source)
        self.assertIn('checkpoint_audit(folder, role, meta, torch)', source)

    def test_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / pairs.TAG).mkdir()
            (out / pairs.TAG / "plan.json").write_text(json.dumps(
                {"source": {"wrong": True}, "reference_files": {}}))
            with self.assertRaisesRegex(ValueError, "Code changed"):
                pairs.validate_plan(out)

    def test_reference_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            reference = out / "old.json"
            reference.write_text("before")
            (out / pairs.TAG).mkdir()
            plan = {"source": pairs.fingerprint(),
                    "reference_files": {str(reference): pairs.common.sha(reference)}}
            (out / pairs.TAG / "plan.json").write_text(json.dumps(plan))
            pairs.validate_plan(out)
            reference.write_text("after")
            with self.assertRaisesRegex(ValueError, "Reference evidence"):
                pairs.validate_plan(out)

    def test_descriptive_stats_use_sample_sd_and_three_orders(self):
        result = pairs.describe([1., 2., 3.])
        self.assertEqual(result["mean"], 2.)
        self.assertEqual(result["sample_sd"], 1.)
        self.assertAlmostEqual(result["descriptive_t95"][1], 4.4841377117)
        for values in ([1.], [1., 2., float("nan")]):
            with self.assertRaises(ValueError):
                pairs.describe(values)

    def create_three_pairs(self, out):
        for seed in (1993,) + pairs.ORDERS:
            np = pairs.original_path(out) if seed == 1993 else pairs.pair_folder(out, seed) / "ranpac/summary.json"
            op = pairs.first_trial(out) / "summary.json" if seed == 1993 else pairs.pair_folder(out, seed) / "ours/summary.json"
            np.parent.mkdir(parents=True)
            op.parent.mkdir(parents=True)
            obj = native_record(seed)
            # Distinct but valid permutations, matched within each pair.
            obj["metadata"]["class_order"] = list(range(200))[seed % 200:] + list(range(200))[:seed % 200]
            np.write_text(json.dumps(obj))
            row = {"Acc@1": 78., "Forgetting": 3., "Backward": -3.}
            ours = {"metadata": dict(obj["metadata"], ranpac_summary_sha256=pairs.common.sha(np)),
                    "baseline": {str(i): row for i in range(1, 11)},
                    "full": {str(i): dict(row, **{"Acc@1": 79.}) for i in range(1, 11)}}
            op.write_text(json.dumps(ours))
        def read(path):
            data = json.loads((path.parent / "summary.json").read_text())
            return {int(k): v for k, v in data[path.stem].items()}
        return read

    def test_three_pair_aggregation(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            read = self.create_three_pairs(out)
            with patch.object(pairs.verify, "read_stages", side_effect=read):
                summary = pairs.aggregate(out)
            self.assertEqual([r["order_seed"] for r in summary["rows"]], [1993, 1994, 1995])
            self.assertEqual(summary["full_minus"]["RanPAC"]["Acc@1"]["mean"], 2.)
            self.assertEqual(summary["full_minus"]["HRM-PET"]["Acc@1"]["mean"], 1.)
            self.assertEqual(summary["full_minus"]["RanPAC"]["Forgetting"]["mean"], -1.)

    def test_unmatched_order_rejected_in_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            read = self.create_three_pairs(out)
            path = pairs.pair_folder(out, 1995) / "ours/summary.json"
            obj = json.loads(path.read_text())
            obj["metadata"]["class_order"].reverse()
            path.write_text(json.dumps(obj))
            with patch.object(pairs.verify, "read_stages", side_effect=read):
                with self.assertRaisesRegex(ValueError, "Unmatched"):
                    pairs.aggregate(out)

    def test_summary_log_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.create_three_pairs(out)
            with patch.object(pairs.verify, "read_stages", return_value={}):
                with self.assertRaisesRegex(ValueError, "Summary disagrees"):
                    pairs.aggregate(out)

    def test_plan_rejects_unmatched_original_reference(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.create_three_pairs(out)
            path = pairs.first_trial(out) / "summary.json"
            obj = json.loads(path.read_text())
            obj["metadata"]["seed"] = 2
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "Initial pair"):
                pairs.make_plan(out, Path("/data"), 4)


if __name__ == "__main__":
    unittest.main()
