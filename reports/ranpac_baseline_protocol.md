# Bounded RanPAC baseline — ImageNet-R

New single-seed experiment, not a rerun of completed paper checks.
The current paper/checkpoints remain unchanged (paper commit 807fd6f).

## Definition

- Official RanPAC commit cf4b301d18b0c27db030f4371b72b768005ae58a, ImageNet-R
  ID 7: SSF, task-1-only training, 20 epochs, batch 48, SGD lr .01, weight
  decay .0005, cosine decay to zero.
- Official RP/ReLU M=10000, cumulative G/C, and 17-value current-task 80/20
  ridge selection unchanged. No LoRA RP-only substitution.
- Protocol adaptations: exact frozen backbone tensors from trusted paper TII
  checkpoints (task 1 and 10 must match), paper's saved shuffle setting/Python
  class permutation, seed all RNGs with 42. Upstream trainer hardcodes torch
  seed 1; we deliberately do not use that trainer.
- Existing ImageNet-R train/test split, no extraction/resplitting. Upstream
  feature/evaluation transform: resize 256 bicubic, crop 224, ToTensor, no
  normalization, matching paper ImageNet-R evaluation. SSF train augmentation
  remains upstream's.
- Private timm 0.6.13 overlay for Python 3.11 compatibility; official README
  uses 0.6.12. Existing torch/torchvision reused, runtime versions logged.
  No package in the user's current virtualenv is replaced.
- Exact task-balanced accuracy from predictions, not mean of rounded upstream
  group values; also pooled accuracy, triangle, paper Forgetting/Backward.
- Protocol-matched RanPAC with a compatibility layer, NOT an exact reproduction
  of the published RanPAC headline table.

## Safety and runtime

One reusable support directory under output-root: _ranpac_support, containing
upstream and private runtime. No datasets or backbone downloads, model saves,
image copies or paper edits. One exclusive log and small summary per tag.
Existing paper-evaluation lock respected. Busy GPU, <16 GiB free VRAM or
<12 GiB available RAM abort before opening a run log. Worker rechecks GPU.
Default child budget: 60 minutes including CPU preflight; preparation is outside
that budget. Timeout stops only this child group including its data loaders.
Partial trajectories are not final evidence. The 17 dense ridge solves per
task may be CPU-heavy: no promised completion time before a real run.
Success requires RANPAC_COMPLETE=imr:10/10, RANPAC_EXIT_CODE=0 and summary.
Never overwrites a completed/failed log; inspect failures before using a new
tag. No resume mode. --prepare without --run performs CPU-only preflight.
Metadata records checkpoint/image hashes, class order, upstream revision,
runner hash, original config and explicit deviations.

## Verification status

CPU guard/metric tests: tests/test_ranpac_baseline.py. Windows lacks torch/CUDA;
no GPU execution or measured time claimed. Linux preflight validates imports,
CPU model construction, backbone equality, data mapping and official config.
Local result: 17 new tests plus 88 existing verification tests = 105 PASS.
