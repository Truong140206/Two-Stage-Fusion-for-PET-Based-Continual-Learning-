#!/usr/bin/env python3
"""Two sequential ImageNet-R replications, fixed orders1994/1995, training seed1.

Read-only plan by default. --run opts into training; no cleanup or downloads.
The completed order1993 experiment and all original launchers stay unchanged.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import try_ranpac_pretrained as trial
from tools import run_ranpac_original as native
from tools import run_ranpac_baseline as common
from tools import verify_paper_results as verify

ORDERS = (1994, 1995)
TAG = "ranpac_pairs_v1"
GIB = 1024**3
METRICS = ("Acc@1", "Forgetting", "Backward")
WORKER = ROOT / "tools/run_ranpac_pair_worker.py"


def pair_folder(out, seed):
    if seed not in ORDERS:
        raise ValueError("Only preregistered orders1994/1995 are allowed")
    return out / TAG / ("order%d" % seed)


def original_path(out):
    return out / "_ranpac_support/original/ranpac_original_imr_id7_v1/summary.json"


def first_trial(out):
    return out / "ranpac_pretrained_trial_v1"


def fingerprint():
    files = [Path(__file__), WORKER, Path(trial.__file__),
             Path(native.__file__), Path(common.__file__), Path(verify.__file__)]
    return {"files": {p.name: common.sha(p) for p in files},
            "evaluator": verify.source_digest()}


def checked_native(path, seed):
    result = json.loads(path.read_text())
    meta = result["metadata"]
    expected = dict(native.EXPECTED, seed=str(seed))
    if (meta["config"] != expected or meta["revision"] != common.REVISION
            or meta["pretrained_sha256"] != trial.NPZ_SHA
            or meta["torch_seed_from_original_trainer"] != 1
            or sorted(meta["class_order"]) != list(range(200))
            or len(result["stages"]) != 10 or result["final"]["stage"] != 10):
        raise ValueError("Invalid paired RanPAC configuration/completion")
    counts = [sum(meta["counts"]["test"][c] for c in meta["class_order"][i:i + 20])
              for i in range(0, 200, 20)]
    if counts != result["final"]["task_counts"]:
        raise ValueError("Paired RanPAC prediction counts disagree with class order")
    return result


def immutable_json(path, obj):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(obj)):
            raise ValueError("Existing evidence differs; preserved: " + str(path))
    else:
        trial.save_json_exclusive(path, obj)


def validate_plan(out):
    plan = json.loads((out / TAG / "plan.json").read_text())
    if plan["source"] != fingerprint():
        raise ValueError("Code changed since pair plan; existing artifacts preserved")
    for path, digest in plan["reference_files"].items():
        if common.sha(path) != digest:
            raise ValueError("Reference evidence changed: " + path)
    return plan


def mark_done(folder, phase, paths):
    trial.save_json_exclusive(folder / (phase + "_done.json"),
                              {"files": {str(p.relative_to(folder)): common.sha(p) for p in paths}})


def needed_gib(out):
    # Conservative reservation, not a disk-usage prediction. Never lower the old
    # 12GiB fresh-run threshold just to squeeze into a nearly full filesystem.
    return 12 * sum(not (pair_folder(out, s) / "ours/full_done.json").exists() for s in ORDERS)


def require_space(out):
    free = trial.shutil.disk_usage(out).free
    need = needed_gib(out)
    print("DISK_FREE_GIB=%.2f REQUIRED_GIB=%d" % (free / GIB, need), flush=True)
    if free < need * GIB:
        raise RuntimeError("Need %d GiB free for remaining pairs; no experiment created. "
                           "Do not delete verified checkpoints. Free space or arrange another disk first." % need)


def make_plan(out, data, threads):
    original = trial.checked_summary(original_path(out))
    ours_path = first_trial(out) / "summary.json"
    ours = json.loads(ours_path.read_text())
    meta = ours["metadata"]
    if (meta["pretrained_sha256"] != trial.NPZ_SHA
            or meta["class_order"] != original["metadata"]["class_order"]
            or meta["dataset_sha256"] != original["metadata"]["dataset_sha256"]
            or meta["ranpac_summary_sha256"] != common.sha(original_path(out))
            or meta["seed"] != 1):
        raise ValueError("Initial pair is not matched or source evidence differs")
    for key in ("baseline", "full"):
        if sorted(map(int, ours[key])) != list(range(1, 11)):
            raise ValueError("Initial HRM-PET/Full trajectory incomplete")
    # Include completed-phase markers, then verify their actual artifacts before
    # GPU work. Keep the original launcher's source-dependent rerun path unused.
    refs = [original_path(out), ours_path, first_trial(out) / "preflight.json"]
    refs += [first_trial(out) / (p + "_done.json") for p in trial.PHASES]
    return {"orders": list(ORDERS), "training_seed": 1, "cpu_threads": threads,
            "output_root": str(out), "data_root": str(data),
            "source": fingerprint(), "reference_files": {str(p): common.sha(p) for p in refs},
            "design": "Two fixed class-order replications; not independent training seeds. "
                      "Same pretrained/split; existing method-specific budgets/runtimes. "
                      "No hyperparameter search, no deletion, no overwrite."}


def native_child(args):
    """Validate original runtime/config, then execute unmodified upstream main."""
    validate_plan(args.output_root)
    import numpy as np
    import torch
    import torchvision
    import timm
    import pandas
    import tqdm
    expected_versions = (3, 9, "1.13.1+cu117", "0.14.1+cu117", "0.6.12",
                         "1.24.4", "1.5.2", "4.65.0")
    actual = (*sys.version_info[:2], torch.__version__, torchvision.__version__,
              timm.__version__, np.__version__, pandas.__version__, tqdm.__version__)
    if actual != expected_versions or torch.version.cuda != "11.7":
        raise ValueError("Original runtime changed: " + str(actual))
    folder = pair_folder(args.output_root, args.order_seed) / "ranpac"
    if Path.cwd().resolve() != folder.resolve():
        raise ValueError("Unexpected original working directory")
    with (folder / "args/imagenetr_publish.csv").open() as stream:
        if list(csv.DictReader(stream)) != [dict(native.EXPECTED, seed=str(args.order_seed))]:
            raise ValueError("Only class-order seed may differ from original ID7")
    meta = json.loads((folder / "metadata.json").read_text())
    if np.random.RandomState(args.order_seed).permutation(200).tolist() != meta["class_order"]:
        raise ValueError("Actual class permutation differs from plan")
    support = args.output_root / "_ranpac_support/original"
    npz = support / "torch-cache/hub/checkpoints" / meta["pretrained_url"].split("/")[-1]
    if common.sha(npz) != trial.NPZ_SHA:
        raise ValueError("Pretrained cache missing/changed; no download permitted")
    # Fail before model creation if the current images no longer match the
    # reference; a recorded hash alone is not a check of the live dataset.
    from torchvision.datasets import ImageFolder
    data = common.resolve_data(args.data_root)
    digest, counts = hashlib.sha256(), {}
    sets = [ImageFolder(str(data / s)) for s in ("train", "test")]
    if sets[0].class_to_idx != sets[1].class_to_idx:
        raise ValueError("Train/test class indices differ")
    for split, ds in zip(("train", "test"), sets):
        count = [0] * 200
        for path, label in ds.samples:
            path = Path(path)
            count[label] += 1
            digest.update((path.relative_to(data).as_posix() + ":" + common.sha(path)).encode())
        counts[split] = count
    if counts != meta["counts"] or digest.hexdigest() != meta["dataset_sha256"]:
        raise ValueError("Live dataset differs from the initial matched experiment")
    args.child = "native"
    native.child(args)


def run_native(args, folder, lock):
    if trial.validate_done(folder, "native"):
        checked_native(folder / "summary.json", args.order_seed)
        print("REUSE_NATIVE_ORDER=%d" % args.order_seed, flush=True)
        return
    if folder.exists():
        raise RuntimeError("Incomplete original phase preserved: " + str(folder))
    out = args.output_root
    support = out / "_ranpac_support/original"
    upstream = out / "_ranpac_support/upstream"
    private_py = support / "env-py39/bin/python"
    if not private_py.is_file() or not (support / "environment-ready.json").is_file():
        raise ValueError("Existing original RanPAC environment required; no installation")
    env = native.isolated_env(support, args.cpu_threads)
    # Separate RNG process, using the same NumPy version as native DataManager.
    order = json.loads(subprocess.check_output(
        [str(private_py), "-B", "-c",
         "import json,numpy as np; print(json.dumps(np.random.RandomState(%d).permutation(200).tolist()))"
         % args.order_seed], env=env, text=True, timeout=60))
    metadata = dict(trial.checked_summary(original_path(out))["metadata"])
    metadata.update(config=dict(native.EXPECTED, seed=str(args.order_seed)),
                    class_order=order, runner_sha256=common.sha(__file__),
                    limitations=["Original ID7 except class-order seed; training torch seed remains1",
                                 "Runtime/augmentation/budgets remain method-specific",
                                 "Existing split hashed, not authenticated against author archive"],
                    data_path=str(common.resolve_data(args.data_root)))
    folder.mkdir(parents=True)
    (folder / "args").mkdir()
    with (folder / "args/imagenetr_publish.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=native.EXPECTED)
        writer.writeheader()
        writer.writerow(metadata["config"])
    (folder / "data").mkdir()
    native.link_existing(folder / "data/imagenet-r", common.resolve_data(args.data_root))
    trial.save_json_exclusive(folder / "metadata.json", metadata)
    log = folder / "console.log"
    cmd = [str(private_py), "-B", "-u", str(Path(__file__).resolve()),
           "--output-root", str(out), "--data-root", str(args.data_root),
           "--order-seed", str(args.order_seed), "--native-child"]
    start = time.monotonic()
    native.run_bounded(cmd, folder, env, lock, log, args.native_max_minutes)
    common.check_upstream(upstream)
    result = native.summarize(folder)
    result.update(metadata=metadata, elapsed_seconds=time.monotonic() - start)
    trial.save_json_exclusive(folder / "summary.json", result)
    checked_native(folder / "summary.json", args.order_seed)
    mark_done(folder, "native", [log, folder / "metadata.json", folder / "summary.json",
              folder / "args/imagenetr_publish.csv", folder / "results/imagenetr_publish_7.csv",
              folder / "results/class_preds/imagenetr_class_preds_publish_7.csv"])
    common.emit("PAIRED_RANPAC_FINAL", result["final"])


def run_ours(args, folder, lock):
    if not folder.exists():
        folder.mkdir()
    if not (folder / "preflight.json").exists() and any(folder.iterdir()):
        raise ValueError("Uninitialized nonempty trial folder preserved")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               PYTHONPYCACHEPREFIX=str(folder / "unused-bytecode"),
               OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
               OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    base = [str(WORKER), "--output-root", str(args.output_root),
            "--data-root", str(args.data_root), "--folder", str(folder),
            "--order-seed", str(args.order_seed), "--cpu-threads", str(args.cpu_threads)]
    subprocess.run([sys.executable, "-B", "-u"] + base + ["--worker", "preflight"],
                   cwd=ROOT, env=env, check=True, timeout=1800)
    start = time.monotonic()
    for phase in trial.PHASES:
        if trial.validate_done(folder, phase):
            print("REUSE_ORDER=%d PHASE=%s" % (args.order_seed, phase), flush=True)
            continue
        log = folder / (phase + ".log")
        if log.exists() or (phase in ("tii", "lora") and (folder / phase).exists()):
            raise RuntimeError("Incomplete phase preserved; no automatic overwrite: " + str(log))
        common.idle_gpu_preflight()
        remaining = args.ours_max_minutes - (time.monotonic() - start) / 60
        if remaining <= 0:
            raise RuntimeError("HRM-PET time limit reached; completed phases reusable")
        print("ORDER=%d PHASE=%s LOG=%s" % (args.order_seed, phase, log), flush=True)
        cmd = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=1"] + base + ["--worker", phase]
        native.run_bounded(cmd, ROOT, env, lock, log, remaining)
        if "PRETRAINED_TRIAL_PHASE_COMPLETE=" + phase not in log.read_text(errors="replace"):
            raise ValueError("Phase incomplete: " + phase)
        audit = folder / (phase + "_audit.json")
        files = json.loads(audit.read_text())
        files[log.name], files[audit.name] = common.sha(log), common.sha(audit)
        trial.save_json_exclusive(folder / (phase + "_done.json"), {"files": files})
    baseline, full = (verify.read_stages(folder / (p + ".log")) for p in ("baseline", "full"))
    result = {"metadata": json.loads((folder / "preflight.json").read_text()),
              "baseline": baseline, "full": full}
    immutable_json(folder / "summary.json", result)


def describe(values):
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError("Exactly three finite order-level results required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)
    return {"n": 3, "mean": mean, "sample_sd": sd,
            "descriptive_t95": [mean - half, mean + half]}


def aggregate(out):
    rows, evidence = [], {}
    for seed in (1993,) + ORDERS:
        npath = original_path(out) if seed == 1993 else pair_folder(out, seed) / "ranpac/summary.json"
        opath = first_trial(out) / "summary.json" if seed == 1993 else pair_folder(out, seed) / "ours/summary.json"
        n = trial.checked_summary(npath) if seed == 1993 else checked_native(npath, seed)
        o = json.loads(opath.read_text())
        if (o["metadata"]["class_order"] != n["metadata"]["class_order"]
                or o["metadata"]["dataset_sha256"] != n["metadata"]["dataset_sha256"]
                or o["metadata"]["pretrained_sha256"] != n["metadata"]["pretrained_sha256"]
                or o["metadata"]["ranpac_summary_sha256"] != common.sha(npath)):
            raise ValueError("Unmatched pair in aggregation")
        for key in ("baseline", "full"):
            if sorted(map(int, o[key])) != list(range(1, 11)):
                raise ValueError("Incomplete HRM-PET trajectory")
        # Re-read actual stage logs, not just hand-transcribed summary values.
        for key in ("baseline", "full"):
            actual = verify.read_stages(opath.parent / (key + ".log"))
            if json.loads(json.dumps(actual)) != o[key]:
                raise ValueError("Summary disagrees with evaluation log")
        rows.append({"order_seed": seed, "RanPAC": {k: n["final"][k] for k in METRICS},
                     "HRM-PET": {k: o["baseline"]["10"][k] for k in METRICS},
                     "Full": {k: o["full"]["10"][k] for k in METRICS}})
        evidence.update({str(p): common.sha(p) for p in (npath, opath)})
    stats = {method: {k: describe([r[method][k] for r in rows]) for k in METRICS}
             for method in ("RanPAC", "HRM-PET", "Full")}
    paired = {method: {k: describe([r["Full"][k] - r[method][k] for r in rows])
                       for k in METRICS} for method in ("RanPAC", "HRM-PET")}
    return {"rows": rows, "statistics": stats, "full_minus": paired, "evidence_sha256": evidence,
            "caveat": "Exploratory three-order comparison with training seed1 fixed; "
                      "initial order already observed. Descriptive t intervals with n=3, "
                      "not independent training-seed evidence or a universal superiority claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--native-max-minutes", type=float, default=60)
    parser.add_argument("--ours-max-minutes", type=float, default=360)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--native-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--order-seed", type=int, choices=ORDERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if not 1 <= args.cpu_threads <= 16 or any(not math.isfinite(v) or not 1 <= v <= 1440
            for v in (args.native_max_minutes, args.ours_max_minutes)):
        parser.error("Invalid resource limits")
    if args.native_child:
        if args.order_seed is None:
            parser.error("Native child requires order seed")
        native_child(args)
        return
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("Run in the paper Python environment on the Linux4090 host")
    if not args.output_root.is_dir():
        raise ValueError("Existing output root required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise ValueError("CUDA_VISIBLE_DEVICES must be unset or0")
    common.check_upstream(args.output_root / "_ranpac_support/upstream")
    native.config_row(args.output_root / "_ranpac_support/upstream")
    plan = make_plan(args.output_root, args.data_root, args.cpu_threads)
    common.emit("PAIRED_PLAN", plan)
    require_space(args.output_root)
    if not args.run:
        print("READ_ONLY_PLAN; add --run only after adequate disk and idle GPU", flush=True)
        return
    for phase in trial.PHASES:
        if not trial.validate_done(first_trial(args.output_root), phase):
            raise ValueError("Initial trial missing audited phase " + phase)
    common.idle_gpu_preflight()
    root = args.output_root / TAG
    if root.exists() and not (root / "plan.json").exists():
        raise ValueError("Uninitialized pair folder preserved: " + str(root))
    root.mkdir(exist_ok=True)
    with common.acquire_shared_lock(root / ".pairs.lock"):
        immutable_json(root / "plan.json", plan)
        validate_plan(args.output_root)
        # Keep the shared lock for the entire sequential batch. Inherited by
        # GPU children so killing the launcher cannot permit overlapping jobs.
        with common.acquire_shared_lock(args.output_root / ".paper_backbone_verifier.lock") as lock:
            for seed in ORDERS:
                args.order_seed = seed
                require_space(args.output_root)
                common.idle_gpu_preflight()
                pair = pair_folder(args.output_root, seed)
                run_native(args, pair / "ranpac", lock)
                run_ours(args, pair / "ours", lock)
            result = aggregate(args.output_root)
            immutable_json(root / "summary.json", result)
            common.emit("PAIRED_THREE_ORDER_RESULTS", result)
            print("RANPAC_PAIRS_COMPLETE=3/3 INCLUDING_EXISTING_ORDER1993", flush=True)
            print("SUMMARY=" + str(root / "summary.json"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("RANPAC_PAIRS_STOP: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
