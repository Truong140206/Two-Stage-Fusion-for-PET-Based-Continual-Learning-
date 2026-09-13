# Exploratory HRM-PET/fusion trial with RanPAC pretrained

## Fixed design

New isolated trial, not a replacement for the paper's checkpoints or results.
Script: tools/try_ranpac_pretrained.py. Default tag: ranpac_pretrained_trial_v1.
Use the existing paper virtualenv, not the Python3.9 RanPAC environment.
No changes to main.py, datasets.py, configs, trainers, vits, engines, or paper.

Inputs are taken from the completed native RanPAC ImageNet-R ID7 summary at
OUT/_ranpac_support/original/ranpac_original_imr_id7_v1/summary.json:

- Exact augreg ImageNet-21K -> ImageNet-1K pretrained NPZ already cached.
  SHA256: ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124.
- Exact 200-class permutation recorded by native RanPAC (numpy seed1993).
- Same hashed 24000/6000 train/test images and per-class counts.
- Our model-training seed1, matching original trainer's torch seed1; our usual
  numpy/Python RNG seeding is retained. This does not make all RNG streams,
  augmentation, runtime or training budgets identical across methods.

Original RanPAC finished 10 tasks: task-balanced Acc@1=77.4436991431,
pooled=77.85, F=4.7484336653, B=-4.7484336653; 606.692709 seconds excluding
preparation. Independently checked all stage arithmetic, counts, official CSV
rounding and F/B from the user-supplied summary. Runner SHA matched commit508554d.
Dataset SHA matches earlier matched run65.987495%; the two runs differ in
pretraining, class order/seeding and environment. Do not select the weaker run
as sole evidence for RanPAC or infer a causal pretrained gain from these two.

## Four sequential phases

1. Fresh TII:20 epochs/task, batch128, lr0.0005, covariance, ca_lr0.005,
   correction30 epochs,10 tasks.
2. Fresh LoRA HRM-PET:50 epochs/task,batch24,lr0.03,rank8,K5,con0.2,
   En=gen,tau=-10,cosine,lora_momentum0.4,lora_type=hide,correction30.
   These match training_scripts/train_any_4090.sh except seed/order/pretrained.
3. Conventional HRM-PET evaluation across all10 saved stage checkpoints.
4. Full fusion on the same checkpoints: RP10000/ReLU/lambda10000, fixed
   adapter1 features, w0.7,beta0.5,margin gate. No weight search or selection.

Fresh learned TII/LoRA parameters and cumulative RP statistics are required.
Never replace just the frozen backbone underneath previously learned adapters.
Paper weights/configuration and legacy checkpoints remain unchanged.

## Implementation and checks

The wrapper calls existing main/trainers. It overrides only the trainer's model
factory (pretrained=False followed by the model's existing load_pretrained with
the pinned NPZ), the local permutation used by existing split_single_dataset,
and the checkpoint writer for exclusive atomic saves. It does not rewrite the
training equations, classifier correction, RP/fusion, or evaluation algorithms.

TII/LoRA constructors are resolved explicitly by module, not a global timm
registry name shared by both. None-valued factory kwargs are filtered as in
timm.create_model. CPU preflight compares all frozen backbone tensors from the
two existing NPZ loaders. Each actual model initialization is checked against
that digest. The post-phase audit checks all ten frozen checkpoint backbones,
saved pretrained/class-order provenance, and file SHA256s. Future skip/reuse
requires matching completed artifacts and source/config/data fingerprints.
Class masks are checked against recorded order; original labels are retained,
as required by our evaluator. Strict exemplar-free guard enabled.

## Storage, budget, resume

All new artifacts under OUT/ranpac_pretrained_trial_v1. No dataset copy, new
environment, or new pretrained download. Fresh training requires12GiB free disk.
Atomic checkpoint writer checks estimated tensor size plus512MiB reserve,
minimum1GiB, and never overwrites a complete or partial checkpoint.
No automatic deletion. Paper and old RanPAC outputs are never write targets.
Default360-minute budget covers new training/evaluation phases, not CPU preflight.
This is a bound, not a measured training-time prediction.

Persistent per-trial lock and shared paper GPU lock; check GPU busy/free RAM
before each new phase. Stop if another compute job is present. Timeout terminates
only the launched process group. Completed phases are reusable on rerun.
Incomplete phase is preserved and refused: automatic mid-phase training resume
is deliberately not implemented because the original trainers do not support
restoring the complete continuation state through this wrapper.

Final marker requires all four audited phases and complete10-stage evaluation
logs: RANPAC_PRETRAINED_TRIAL_COMPLETE=10/10. Summary records paired Full-minus-
HRM-PET differences. It is one exploratory run; no claim of multi-seed
significance, reduced compute, or universal superiority to RanPAC.

## Local verification

tests.test_ranpac_pretrained:13 CPU tests covering fixed arguments and parsing
by the real configs, the real dataset split function with a permutation,
restoration on error, atomic saves/low disk, and hash-verified phase reuse.
Together with both RanPAC launcher suites:50 tests,49pass/1POSIX skip on Windows.
Actual model construction/NPZ loads and all GPU training still require the lab
runtime; preflight fails closed before launching training if those checks fail.

Extended local regression:138 tests across new trial and eight related
verification suites;137pass/1POSIX skip. Canonical model/config/dataset/engine/
trainer source and paper TeX/PDF diff against508554d is empty.
