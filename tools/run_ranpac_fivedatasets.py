#!/usr/bin/env python3
"""Run a transparent, non-official RanPAC extension on 5-Datasets.

The upstream RanPAC repository has no 5-Datasets entry.  This runner keeps the
unmodified RanPAC Learner/SSF/RP implementation and supplies only a compatible
five-task data manager in the paper's fixed task order.  ``--run`` is required
for GPU work; all outputs are isolated and existing outputs are preserved.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_ranpac_baseline as common
from tools import run_ranpac_original as native
from tools import run_ranpac_original_extra as extra

TASKS = ("SVHN", "MNIST", "CIFAR10", "NotMNIST", "FashionMNIST")
SCIPY_VERSION = "1.10.1"
CONFIG = {
    "ID": 7, "dataset": "fivedatasets_extension", "shuffle": False,
    "init_cls": 10, "increment": 10, "model_name": "ssf",
    "convnet_type": "pretrained_vit_b16_224_ssf", "device": 0,
    "seed": 1993, "batch_size": 48, "tuned_epoch": 20,
    "body_lr": 0.01, "head_lr": 0.01, "weight_decay": 0.0005,
    "min_lr": 0.0, "use_RP": True, "M": 10000,
    "use_input_norm": False, "do_not_save": False,
}
CHECKPOINT_FILE = extra.SPECS["ima"]["checkpoint_file"]


def ensure_five_environment(support, env, python, install):
    marker = support / "environment-five-ready.json"
    expected = {"scipy": SCIPY_VERSION}
    if marker.exists() and json.loads(marker.read_text()) != expected:
        raise ValueError("Existing five-dataset environment marker differs; preserved")
    probe = [str(python), "-I", "-c",
             "import importlib.metadata as m, scipy.io; "
             "assert m.version('scipy') == '" + SCIPY_VERSION + "'"]
    checked = subprocess.run(probe, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, timeout=120)
    if checked.returncode:
        if not install:
            raise ValueError("Run --prepare to install scipy==" + SCIPY_VERSION)
        pip = [str(python), "-I", "-m", "pip", "--isolated", "install",
               "--no-cache-dir", "--index-url", "https://pypi.org/simple"]
        subprocess.run(pip + ["scipy==" + SCIPY_VERSION], env=env,
                       check=True, timeout=1200)
        subprocess.run(probe, env=env, check=True, timeout=120)
        subprocess.run([str(python), "-I", "-m", "pip", "check"],
                       env=env, check=True, timeout=120)
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2)


def resolve_data(root, resolver=None):
    """Reuse the paper verifier's constructor-based, no-download audit."""
    root = root.resolve()
    if resolver is None:
        from tools.verify_paper_backbone_grid import resolve_data as resolver
    resolved = Path(resolver(root, "fivedatasets")).resolve()
    if resolved != root:
        raise ValueError("5-Datasets resolver unexpectedly changed the requested root")
    return resolved


def layout(folder, upstream, data):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data").mkdir(exist_ok=True)
    native.link_existing(folder / "args", upstream / "args")
    native.link_existing(folder / "data/five-root", data)


def transform_family(task):
    if task not in TASKS:
        raise ValueError("Unknown 5-Datasets task: " + str(task))
    return "generic" if TASKS.index(task) < 2 else "cifar"


def selected_tasks(indices):
    values = sorted(set(map(int, indices)))
    if not values:
        raise ValueError("Empty class selection")
    tasks = sorted(set(value // 10 for value in values))
    if any(task < 0 or task >= len(TASKS) for task in tasks):
        raise ValueError("Class outside 5-Datasets")
    expected = [value for task in tasks for value in range(task * 10, task * 10 + 10)]
    if values != expected:
        raise ValueError("RanPAC extension requires complete ten-class tasks")
    return tasks


def _rgb(image):
    return image.convert("RGB")



def build_transform(task, train):
    from torchvision import transforms
    rgb = transforms.Lambda(_rgb)
    if transform_family(task) == "generic":
        if train:
            return transforms.Compose([
                rgb, transforms.RandomResizedCrop(224, scale=(0.05, 1.0),
                                                  ratio=(3 / 4, 4 / 3)),
                transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
        return transforms.Compose([
            rgb, transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor()])
    norm = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
    if train:
        return transforms.Compose([
            rgb, transforms.RandomResizedCrop(
                224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255), transforms.ToTensor(), norm])
    return transforms.Compose([
        rgb, transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), norm])


def base_dataset(root, task, source, train_transform):
    from torchvision import datasets
    from continual_datasets.continual_datasets import (
        FashionMNIST, MNIST_RGB, NotMNIST, SVHN)
    train = source == "train"
    transform = build_transform(task, train_transform)
    if task == "SVHN":
        return SVHN(root, split="train" if train else "test",
                    download=False, transform=transform)
    if task == "MNIST":
        return MNIST_RGB(root, train=train, download=False, transform=transform)
    if task == "CIFAR10":
        return datasets.CIFAR10(root, train=train, download=False, transform=transform)
    if task == "FashionMNIST":
        return FashionMNIST(root, train=train, download=False, transform=transform)
    return NotMNIST(root, train=train, download=False, transform=transform)


def targets_of(dataset):
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    return [int(value) for value in targets]


def configure_worker_sharing(torch):
    """Avoid exhausting file descriptors across repeated DataLoader stages."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    strategy = torch.multiprocessing.get_sharing_strategy()
    if strategy != "file_system":
        raise RuntimeError("PyTorch sharing strategy did not change: " + strategy)
    print("RANPAC_FIVE_SHARING_STRATEGY=" + strategy, flush=True)


class FiveDataManager:
    def __init__(self, root):
        self.root = Path(root)
        self.nb_tasks = len(TASKS)

    def get_task_size(self, task):
        if not 0 <= task < self.nb_tasks:
            raise IndexError(task)
        return 10

    def get_total_classnum(self):
        return 50

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        if appendent not in (None, ()) or ret_data:
            raise ValueError("Unsupported extension dataset option")
        from torch.utils.data import ConcatDataset, Dataset

        class OffsetDataset(Dataset):
            def __init__(self, base, offset):
                self.base, self.offset = base, offset
                self.labels = [value + offset for value in targets_of(base)]

            def __len__(self):
                return len(self.base)

            def __getitem__(self, index):
                image, target = self.base[index]
                return index, image, int(target) + self.offset

        parts = []
        for task_index in selected_tasks(indices):
            task = TASKS[task_index]
            base = base_dataset(self.root, task, source, mode == "train")
            labels = targets_of(base)
            if set(labels) != set(range(10)):
                raise ValueError(task + " does not contain exactly labels 0--9")
            parts.append(OffsetDataset(base, task_index * 10))
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def dataset_record(root):
    counts = {"train": [], "test": []}
    digest = hashlib.sha256()
    for source in ("train", "test"):
        for task in TASKS:
            ds = base_dataset(root, task, source, False)
            labels = targets_of(ds)
            count = [labels.count(index) for index in range(10)]
            if min(count) < 1 or sum(count) != len(ds):
                raise ValueError("Invalid class counts for %s/%s" % (task, source))
            counts[source].extend(count)
            digest.update((source + "/" + task + ":" + json.dumps(count)).encode())
    roots = [Path(root) / "train_32x32.mat", Path(root) / "test_32x32.mat",
             Path(root) / "cifar-10-batches-py", Path(root) / "MNIST_RGB",
             Path(root) / "FashionMNIST", Path(root) / "notMNIST"]
    files = []
    for item in roots:
        files.extend([item] if item.is_file() else [p for p in item.rglob("*") if p.is_file()])
    for path in sorted(set(files)):
        digest.update((path.relative_to(root).as_posix() + ":" + common.sha(path)).encode())
    return counts, digest.hexdigest()


def save_extension(folder, root):
    import logging
    import numpy as np
    import pandas as pd
    import torch
    from RanPAC import Learner

    configure_worker_sharing(torch)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = dict(CONFIG)
    args["device"] = [torch.device("cuda:0")]
    model = Learner(args)
    model.dil_init = False
    model.is_dil = False
    manager = FiveDataManager(root)
    total = sum(len(base_dataset(root, task, "test", False)) for task in TASKS)
    frame = pd.DataFrame({"init": -np.ones(total, dtype=int)})
    curve = {"top1_total": [], "ave_acc": []}
    for stage in range(len(TASKS)):
        model.incremental_train(manager)
        pooled, grouped, predicted, truth = model.eval_task()
        frame["pred_task_%d" % stage] = np.pad(
            predicted, (0, total - len(predicted)), constant_values=-1)
        frame["true_task_%d" % stage] = np.pad(
            truth, (0, total - len(truth)), constant_values=-1)
        curve["top1_total"].append(pooled)
        curve["ave_acc"].append(float(np.round(np.mean(list(grouped.values())), 2)))
        model.after_task()
        print("RANPAC_FIVE_STAGE=%d/%d" % (stage + 1, len(TASKS)), flush=True)
    results = folder / "results"
    (results / "class_preds").mkdir(parents=True)
    pd.DataFrame(curve).to_csv(results / "fivedatasets_extension_publish_7.csv")
    frame.to_csv(results / "class_preds/fivedatasets_extension_class_preds_publish_7.csv")
    logging.info("RanPAC 5-Datasets extension finished")


def summarize(folder, metadata):
    pred_path = folder / "results/class_preds/fivedatasets_extension_class_preds_publish_7.csv"
    with pred_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"%s_task_%d" % (kind, stage) for stage in range(5)
                    for kind in ("pred", "true")}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing stage columns; run incomplete")
        records = list(reader)
    total = sum(metadata["counts"]["test"])
    if len(records) != total:
        raise ValueError("Expected %d prediction rows" % total)
    rows = []
    for stage in range(5):
        pairs = [(int(row["pred_task_%d" % stage]),
                  int(row["true_task_%d" % stage])) for row in records]
        if any((pred == -1) != (truth == -1) for pred, truth in pairs):
            raise ValueError("Inconsistent prediction padding")
        pairs = [(pred, truth) for pred, truth in pairs if truth != -1]
        row = extra.exact_metrics([pred for pred, _ in pairs],
                                  [truth for _, truth in pairs], stage + 1, 10)
        row["stage"] = stage + 1
        rows.append(row)
    with (folder / "results/fivedatasets_extension_publish_7.csv").open() as stream:
        official = list(csv.DictReader(stream))
    if len(official) != 5:
        raise ValueError("Extension accuracy curve is incomplete")
    for row, saved in zip(rows, official):
        if (abs(row["Acc@1"] - float(saved["ave_acc"])) > .011 or
                abs(row["pooled_acc"] - float(saved["top1_total"])) > .011):
            raise ValueError("Exact predictions disagree with extension CSV")
    expected = [sum(metadata["counts"]["test"][i:i + 10])
                for i in range(0, 50, 10)]
    if rows[-1]["task_counts"] != expected:
        raise ValueError("Predictions disagree with dataset metadata")
    return {"status": "exploratory_non_official_extension", "stages": rows,
            "final": dict(rows[-1], **common.retention(rows)),
            "curve": official, "metadata": metadata}


def child(args):
    support = args.output_root / "_ranpac_support/original"
    upstream = args.output_root / "_ranpac_support/upstream"
    common.check_upstream(upstream)
    sys.path[:] = [str(upstream), str(ROOT)] + [p for p in sys.path if p and
                    Path(p).resolve() not in (common.ROOT, common.ROOT / "tools")]
    root = Path("data/five-root")
    if args.child == "extension":
        save_extension(Path.cwd(), root)
        return
    import importlib.metadata as package_metadata
    import numpy as np
    import pandas
    import scipy
    import timm
    import torch
    import torchvision
    import tqdm
    import inc_net
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "timm": timm.__version__,
                "pandas": pandas.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "tqdm": tqdm.__version__,
                "cuda": torch.version.cuda}
    if (sys.version_info[:2] != (3, 9) or torch.__version__ != "1.13.1+cu117"
            or torchvision.__version__ != "0.14.1+cu117" or timm.__version__ != "0.6.12"
            or pandas.__version__ != "1.5.2" or np.__version__ != "1.24.4"
            or package_metadata.version("scipy") != SCIPY_VERSION
            or tqdm.__version__ != "4.65.0" or torch.version.cuda != "11.7"):
        raise ValueError("Private environment version mismatch: " + str(versions))
    counts, dataset_sha = dataset_record(root)
    print("RANPAC_FIVE_PRETRAINED_DOWNLOAD_OR_CACHE_CHECK", flush=True)
    model = inc_net.get_convnet(dict(CONFIG))
    url = extra.pretrained_source("ima", model, timm)
    cached = support / "torch-cache/hub/checkpoints" / CHECKPOINT_FILE
    if not cached.is_file():
        raise ValueError("Original pretrained file not found in isolated cache")
    record = {"status": "exploratory_non_official_extension",
              "task_order": list(TASKS), "class_order": list(range(50)),
              "runtime": versions, "config": CONFIG, "counts": counts,
              "dataset_sha256": dataset_sha, "pretrained_url": url,
              "pretrained_sha256": common.sha(cached), "revision": common.REVISION,
              "runner_sha256": common.sha(__file__), "torch_seed": 1,
              "limitations": ["RanPAC has no official 5-Datasets configuration",
                              "Fixed benchmark task order; seed 1993 does not permute domains",
                              "Unmodified RanPAC Learner with a local data-manager extension"]}
    path = support / "preflight-five/metadata.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Existing five-dataset preflight metadata differs; preserved")
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
    common.emit("RANPAC_FIVE_PREFLIGHT", {"tasks": list(TASKS),
                "train": sum(counts["train"]), "test": sum(counts["test"]),
                "pretrained_sha256": record["pretrained_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tag", default="ranpac_fivedatasets_extension_v1")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=480)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", choices=("inspect", "extension"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.tag):
        parser.error("Unsafe tag")
    if not 1 <= args.cpu_threads <= 16 or not math.isfinite(args.max_minutes) or not 1 <= args.max_minutes <= 1440:
        parser.error("Invalid resource limits")
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if args.child:
        child(args)
        return
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("Run on the Linux x86_64 4090 host")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise ValueError("CUDA_VISIBLE_DEVICES must be unset or 0")
    if not args.output_root.is_dir():
        raise ValueError("Existing output-root required")
    if shutil.disk_usage(args.output_root).free < 12 * 1024**3:
        raise RuntimeError("Need at least 12 GiB free disk")
    upstream = args.output_root / "_ranpac_support/upstream"
    support = args.output_root / "_ranpac_support/original"
    support.mkdir(parents=True, exist_ok=True)
    env = native.isolated_env(support, args.cpu_threads)
    with common.acquire_shared_lock(support / ".prepare.lock"):
        if args.prepare and not upstream.exists():
            subprocess.run(["git", "clone", "--no-checkout", common.UPSTREAM,
                            str(upstream)], check=True)
            subprocess.run(["git", "-C", str(upstream), "checkout", "--detach",
                            common.REVISION], check=True)
        common.check_upstream(upstream)
        python = native.prepare(support, env) if args.prepare else support / "env-py39/bin/python"
        if not python.is_file():
            raise ValueError("Run --prepare first")
        ensure_five_environment(support, env, python, args.prepare)
        data = resolve_data(args.data_root)
        preflight = support / "preflight-five"
        layout(preflight, upstream, data)
        base = [str(python), "-B", "-u", str(Path(__file__).resolve()),
                "--output-root", str(args.output_root), "--data-root", str(args.data_root),
                "--tag", args.tag, "--cpu-threads", str(args.cpu_threads),
                "--max-minutes", str(args.max_minutes)]
        subprocess.run(base + ["--child", "inspect"], cwd=preflight, env=env,
                       check=True, timeout=3600)
        folder = support / args.tag
        summary_path = folder / "summary.json"
        if summary_path.exists():
            result = json.loads(summary_path.read_text())
            if (result.get("metadata", {}).get("runner_sha256") != common.sha(__file__)
                    or result.get("final", {}).get("stage") != 5):
                raise ValueError("Existing completed extension differs; preserved")
            print("REUSE_COMPLETED_RANPAC_FIVE=5/5", flush=True)
            return
        if not args.run:
            print("PLAN_RANPAC_FIVE_EXTENSION=5 tasks", flush=True)
            return
        common.idle_gpu_preflight()
        with common.acquire_shared_lock(args.output_root / ".paper_backbone_verifier.lock") as lock:
            common.idle_gpu_preflight()
            if folder.exists():
                raise RuntimeError("Incomplete extension folder preserved: " + str(folder))
            layout(folder, upstream, data)
            metadata = json.loads((preflight / "metadata.json").read_text())
            with (folder / "metadata.json").open("x") as stream:
                json.dump(metadata, stream, indent=2)
            log = folder / "console.log"
            print("RANPAC_FIVE_RUN_DIR=" + str(folder), flush=True)
            start = time.monotonic()
            native.run_bounded(base + ["--child", "extension"], folder, env, lock,
                               log, args.max_minutes)
            result = summarize(folder, metadata)
            result["elapsed_seconds"] = time.monotonic() - start
            with summary_path.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
            common.emit("RANPAC_FIVE_EXTENSION_FINAL", {
                "Acc@1": result["final"]["Acc@1"],
                "Forgetting": result["final"]["Forgetting"],
                "Backward": result["final"]["Backward"],
                "elapsed_seconds": result["elapsed_seconds"]})
            print("RANPAC_FIVE_EXTENSION_COMPLETE=5/5", flush=True)
            print("SUMMARY=" + str(summary_path), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("RANPAC_FIVE_EXTENSION_STOP: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
