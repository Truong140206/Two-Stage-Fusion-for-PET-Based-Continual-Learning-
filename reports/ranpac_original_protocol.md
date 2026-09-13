# Original-entry-point RanPAC — ImageNet-R ID 7

## Purpose and preserved evidence

This is a new original-configuration check, not a replacement for the earlier
matched-backbone experiment. The user supplied a completed ranpac_imr42_v1 log:
task-balanced final accuracy 65.987495%, pooled accuracy 66.416667%,
Forgetting 3.695434, Backward -3.695434, ten stage times sum 584.159580 seconds.
Those values were independently recomputed from its task trajectory. They do
not establish superiority to published RanPAC. Do not overwrite that log or
insert a new baseline claim in the paper before auditing the new run.

## What runs

Reuse the full clean official repository at:
OUT/_ranpac_support/upstream
Pinned revision: cf4b301d18b0c27db030f4371b72b768005ae58a.
Source: https://github.com/McDonnell-Research-Lab/RanPAC

New launcher tools/run_ranpac_original.py calls original main.py with
-i 7 -d imagenetr in a separate process through runpy. It does NOT replace the
model factory, DataManager, trainer, Learner, random seeds, transforms, PETL
training, RP generation, lambda selection or original metric calculation.
All 18 CSV fields are checked exactly. Original numpy class-order seed is 1993;
original trainer hardcodes torch seed 1. Eight data workers remain unchanged.
Original factory downloads the original ImageNet-1K-finetuned pretrained ViT,
not any checkpoint from our system. A separate CPU preflight records its source
URL and SHA256; this process cannot consume the training process's RNG stream.

## Isolated environment and storage

OUT/_ranpac_support/original contains one reusable private Python environment,
pretrained cache, a reusable CPU preflight folder, and one folder per run tag.
No shell activation/init, sudo, global installation or changes to the existing
Hybrid_ReMatching virtualenv. No full dataset copy.

Core versions from official README: Python 3.9, torch 1.13.1, torchvision 0.14.1,
torchaudio 0.13.1, CUDA 11.7, pandas 1.5.2, tqdm 4.65.0, timm 0.6.12.
Uses official pip CUDA 11.7 wheels rather than README's conda torch packaging.
Micromamba 2.3.2 bootstrap archive is SHA256-pinned from conda-forge; only its
bin/micromamba regular-file member is extracted. It provisions private Python.
Auxiliary compatibility pins: numpy 1.24.4, Pillow 9.5.0, PyYAML 6.0.1,
huggingface-hub 0.16.4. Complete pip freeze is saved.
Initial preparation downloads several GB; at least 8 GiB free disk required.
No automatic cache deletion. Existing partial environments are repaired by
re-running preparation; completed run folders and console logs are never reused.

The original relative data/imagenet-r path is a symlink to the existing split.
args is a symlink to original CSVs. Original logs/results/class_preds go into
the run folder, not the upstream checkout or paper folders.

## Evidence, safety, and limitations

Preparation imports exact versions, constructs the original CPU backbone,
checks 200 class mappings, 24000/6000 sample counts and hashes all image contents.
The image manifest is recorded, but equality to the author's archive has NOT
been established. This is an original-code/config check on our existing split,
not an unconditional promise to reproduce a published number.

CPU threads are capped at four; this and pip packaging are recorded deviations.
No source compatibility edits. If an import/download fails, stop and diagnose;
do not silently substitute a newer runtime or a paper backbone.

Persistent flock for preparation and the same shared GPU-evaluation flock as
other paper drivers. Never delete either lock. GPU-busy checks before creating
the run folder. Training timeout default 60 minutes; preparation excluded.
Timeout stops only the launcher's own process group. Native child inherits
the shared lock FD. No claim of measured runtime for this new environment.

Success requires original process exit zero, 6000 saved prediction rows with
all ten stages, agreement with the original ten-row accuracy CSV (within its
two-decimal rounding), a summary.json, and:
RANPAC_ORIGINAL_COMPLETE=imr:10/10

Summary recomputes task-balanced accuracy, pooled accuracy, Forgetting and
Backward from original saved predictions after training; nothing is injected
into original model execution. Standard original logs include selected lambdas.
Paper and the earlier hash-pinned runners remain unchanged.

## Local verification

Dependency-free tests: tests.test_ranpac_original and tests.test_ranpac_baseline.
Native entry-point dispatch tested with a fixture; source at the pinned commit
inspected locally. Windows host cannot certify Linux CUDA execution or download
compatibility: those checks execute during --prepare on the lab host.

Local verification on 2026-09-14: 61 tests across original launcher,
matched launcher, finish_soict_checks and paper_diagnostics; 60 pass and one
real POSIX flock test skipped on Windows. Direct config validation against the
full pinned upstream checkout passes. CLI help works without torch installed.
