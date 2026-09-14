#!/usr/bin/env python3
"""Isolated worker for the two fixed-order replications; old trial is unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import try_ranpac_pretrained as trial
from tools import run_ranpac_pairs as pairs
from tools.try_ranpac_pretrained import (
    common, verify, PHASES, NPZ_SHA, MODEL, cli_arguments, backbone_digest,
    save_json_exclusive, checkpoint_audit, ordered_split, atomic_checkpoint)

def worker(args):
    import torch
    import numpy as np
    import torchvision
    import timm
    import datasets
    from torchvision.datasets import ImageFolder
    from vits import hide_prompt_vision_transformer as tii_vit
    from vits import hrm_lora_vision_transformer as lora_vit
    folder = args.folder
    if folder.exists() and not (folder / "preflight.json").exists():
        if any(p.name != ".trial.lock" for p in folder.iterdir()):
            raise ValueError("Existing folder is not an initialized trial; preserved")
    summary_path = folder.parent / "ranpac/summary.json"
    ranpac = pairs.checked_native(summary_path, args.order_seed)
    source = ranpac["metadata"]
    npz = args.output_root / "_ranpac_support/original/torch-cache/hub/checkpoints" / source["pretrained_url"].split("/")[-1]
    if common.sha(npz) != NPZ_SHA:
        raise ValueError("RanPAC NPZ changed/missing")
    data_parent = verify.resolve_imr_data_path(args.data_root)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if args.worker == "preflight":
        data_root = data_parent / "imagenet-r"
        digest = hashlib.sha256()
        counts = {}
        for split in ("train", "test"):
            ds = ImageFolder(str(data_root / split))
            count = [0] * 200
            for path, label in ds.samples:
                path = Path(path)
                count[label] += 1
                digest.update((path.relative_to(data_root).as_posix() + ":" + common.sha(path)).encode())
            counts[split] = count
        if counts != source["counts"] or digest.hexdigest() != source["dataset_sha256"]:
            raise ValueError("Dataset differs from completed RanPAC run")
        hashes = []
        for module, extra in ((tii_vit, {"mlp_structure": [2]}),
                              (lora_vit, {"lora": True, "lora_type": "hide", "rank": 8, "lora_pool_size": 10})):
            model = module.vit_base_patch16_224(pretrained=False, num_classes=200, **extra)
            model.load_pretrained(str(npz))
            hashes.append(backbone_digest(model.state_dict()))
            del model
        if hashes[0] != hashes[1]:
            raise ValueError("TII and LoRA NPZ loaders yield different backbone tensors")
        meta = {"pretrained_sha256": NPZ_SHA, "pretrained_path": str(npz),
                "backbone_sha256": hashes[0], "class_order": source["class_order"],
                "dataset_sha256": source["dataset_sha256"], "counts": counts,
                "data_parent": str(data_parent), "ranpac_summary_sha256": common.sha(summary_path),
                "source_sha256": verify.source_digest(), "driver_sha256": common.sha(__file__),
                "seed": 1, "order_seed": args.order_seed,
                "helper_sha256": common.sha(trial.__file__), "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                                     "numpy": np.__version__, "timm": timm.__version__},
                "commands": {p: cli_arguments(p, folder, data_parent) for p in PHASES},
                "limitations": ["Fixed-order replication; training seed1; no test-selected hyperparameters",
                                "Same pretrained/images/task order as RanPAC, not identical training or runtime",
                                "HRM-PET trains adapters across tasks; RanPAC trains SSF only on task 1"]}
        path = folder / "preflight.json"
        if path.exists():
            if json.loads(path.read_text()) != meta:
                raise ValueError("Experiment inputs/source changed; existing run preserved")
        else:
            save_json_exclusive(path, meta)
        print("PRETRAINED_AND_DATA_AND_LOADER_TENSORS=PASS", flush=True)
        return

    meta = json.loads((folder / "preflight.json").read_text())
    if (verify.source_digest() != meta["source_sha256"] or common.sha(__file__) != meta["driver_sha256"]
            or common.sha(trial.__file__) != meta["helper_sha256"]):
        raise ValueError("Source changed since preflight")
    phase = args.worker
    if phase != "tii":
        checkpoint_audit(folder, "tii", meta, torch)
    if phase in ("baseline", "full"):
        checkpoint_audit(folder, "lora", meta, torch)
    datasets.split_single_dataset = ordered_split(datasets, datasets.split_single_dataset, meta["class_order"])
    import main as entry
    sys.argv = ["main.py"] + cli_arguments(phase, folder, data_parent)
    training = entry.get_args()
    training.shuffle = True
    training.experiment_class_order = meta["class_order"]
    training.experiment_pretrained_sha256 = NPZ_SHA
    training.experiment_tag = "ranpac_pairs_v1_order%d" % args.order_seed
    module = __import__("trainers.tii_trainer" if phase == "tii" else "trainers.lora_trainer", fromlist=["train"])
    def create(model_name, **kwargs):
        if model_name != MODEL:
            raise ValueError("Unexpected model name")
        # Resolve the intended module explicitly: importing both custom ViTs
        # registers the same timm name, so global registry order is unsafe.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs["pretrained"] = False
        constructor = tii_vit if phase == "tii" else lora_vit
        model = constructor.vit_base_patch16_224(**kwargs)
        model.load_pretrained(str(npz))
        if backbone_digest(model.state_dict()) != meta["backbone_sha256"]:
            raise ValueError("Factory did not load the pinned backbone exactly")
        print("EXACT_RANPAC_PRETRAINED_TENSORS=PASS", flush=True)
        return model
    module.create_model = create
    import utils
    utils.save_on_master = atomic_checkpoint(utils.save_on_master, folder)
    try:
        entry.main(training)
    finally:
        utils.cleanup_distributed()
    checkpoint_records = {}
    for role in (("tii",) if phase == "tii" else ("tii", "lora")):
        checkpoint_records.update(checkpoint_audit(folder, role, meta, torch))
    save_json_exclusive(folder / (phase + "_audit.json"), checkpoint_records)
    print("PRETRAINED_TRIAL_PHASE_COMPLETE=" + phase, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, choices=pairs.ORDERS, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--worker", choices=("preflight",) + PHASES, required=True)
    args = parser.parse_args()
    args.output_root, args.data_root, args.folder = (
        p.resolve() for p in (args.output_root, args.data_root, args.folder))
    if args.folder != pairs.pair_folder(args.output_root, args.order_seed) / "ours":
        raise ValueError("Worker output outside its fixed new pair folder")
    pairs.validate_plan(args.output_root)
    worker(args)


if __name__ == "__main__":
    main()
