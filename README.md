# Two-Stage Fusion for Routing and Classification in Parameter-Efficient Continual Learning

Code and evaluation scripts for the paper of the same name.

A continual learner built on parameter-efficient tuning keeps one adapter per
task and, at inference, must choose which adapter to attach before it can
classify. That choice comes from a single module, and everything downstream
inherits its errors. We show that a second, independent source of evidence is
already present in the trained model and is simply discarded: the LoRA features
the pipeline computes anyway, pushed through a frozen random projection and read
by a closed-form ridge classifier built from second-order statistics.

The second source is fused at the two points where the pipeline commits to a
decision — once at the routing level and once at the classification level. It
trains no parameters, stores no images, and costs one forward pass the pipeline
already performs.

Measured over three datasets and four seeds against the HRM-PET baseline:

| axis | change |
| --- | --- |
| routing accuracy (Acc@task) | `+1.10` to `+2.26` |
| classification accuracy (Acc@1) | `+0.69` to `+1.50` |
| retention (Forgetting, Backward) | no change we can resolve; we claim none |

Each accuracy gain is several times the seed-to-seed standard deviation. Across
the full grid of four datasets by five pre-training configurations, all twenty
cells improve on both accuracy axes, over baselines spanning `31.8` to `93.8`.

## Layout

```
main.py                  entry point; dispatches to a trainer by config name
configs/                 one TII config and one LoRA config per dataset
trainers/                TII training, LoRA-pool training
engines/                 evaluation engines; the two-stage fusion lives in
                         hrm_lora_wtp_and_tap_engine.py, the projection head in
                         random_projection_head.py
vits/, peft/             backbone and adapter implementations
continual_datasets/      dataset construction for the four benchmarks
training_scripts/        training and evaluation entry scripts
tools/                   dataset preparation and result collection
tests/                   unit tests for the fusion and the exemplar-free protocol
reports/                 the paper, English and Vietnamese
```

`engines/` carries more modules than the two stages need. They are reachable
from `main.py` through the trainer's imports, so they are kept rather than
removed piecemeal; the fusion described in the paper is confined to
`hrm_lora_wtp_and_tap_engine.py` and `random_projection_head.py`.

## Requirements

Python 3.12 and the packages in `requirements.txt`. A single GPU is enough; the
numbers below were produced on an RTX 4090.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Data

```bash
python tools/prepare_datasets.py --root /path/to/datasets
```

CIFAR-100 and the five constituents of 5-Datasets download automatically.
ImageNet-R and ImageNet-A are rebuilt from their public parquet mirrors by
`tools/imagenet_a_from_parquet.py`.

## Training

Two stages per dataset, in order: the task-identity module, then the LoRA pool.

```bash
DATASETS_ROOT=/path/to/datasets OUTPUT_ROOT=/path/to/output \
  bash training_scripts/train_any_4090.sh imr 42
```

The first argument selects the dataset (`imr`, `cifar100`, `ima`,
`fivedatasets`), the second the seed. Checkpoints land under `OUTPUT_ROOT`.

## Evaluation

The proposed method, at the published setting `w = 0.7`, `beta = 0.5`:

```bash
DATASETS_ROOT=/path/to/datasets OUTPUT_ROOT=/path/to/output \
DATASET=Split-Imagenet-R CONFIG=imr_lora SEED=42 NUM_TASKS=10 \
TII_DIR=/path/to/output/imr_tii_original_10tasks_seed42 \
RP_FUSE=1 RP_FUSE_DRM=1 RP_CLS_MIN=1 CALIBRATE=0 \
RP_FUSE_W=0.7 RP_CLS_GATE=margin RP_CLS_W=0.5 \
  bash training_scripts/eval_rp_head_any_4090.sh \
    /path/to/output/imr_lora_rank8_baseline_10tasks_seed42
```

On ImageNet-R with the Sup-21K backbone at seed 42 this prints

```
Acc@task: 79.9699  Acc@1: 75.5116  Acc@5: 87.9822  Loss: 1.1911
Forgetting: 3.0500  Backward: -2.8184
```

Setting `RP_FUSE_W=1.0` and `RP_CLS_W=0.0` disables both stages and reproduces
the HRM-PET baseline to the fourth decimal, so `w` doubles as a continuous
ablation axis between baseline and method.

`training_scripts/eval_imagenet_r_conventional_4090.sh` and its CIFAR-100 and
CUB-200 counterparts produce the baseline numbers directly.

## Reading results

```bash
python tools/collect_results.py                 # baseline against proposed
python tools/beta_sweep_table.py                # the beta sweep, gated and not
python tools/audit_exemplar_free_checkpoint.py  # no stored images or features
```

Every table in the paper is built from the evaluation logs by these scripts
rather than transcribed by hand.

## Tests

```bash
python -m pytest tests/ -q
```

## Citation

The paper source is in `reports/`. Bibliographic details will be added here once
the venue is settled.
