# Checkpoint review — 2026-09-10

## Closure audit and last required diagnostic batch — 2026-09-12

Current paper/code status:

- Local dependency-free verification suite: 88 tests PASS (9 new diagnostic
  tests). This is not a claim of local torch/GPU evaluation.

- Main results, 48 controls, 9 weight cells, 20 grid cells and 15 ungated
  ablations are complete. Do NOT rerun these batches.
- Algorithm/evaluator/config files remain identical to the pushed
  system-baseline-2026-09-11 tag (4989ab8). Existing verification drivers are
  unchanged, so their completed log provenance remains reusable.
- Fixed the historical oracle paragraph: all quoted rates use equal task
  weights. Recovery is a ratio of task-balanced rates, NOT pooled image counts
  or an average of per-task conditional ratios. Removed unsupported wording
  that all analytic methods "discard routing"; softened unmeasured frequency
  claims. Historical numbers remain visible, explicitly pending verification.
- Pipeline now floats at the top of a text page instead of a half-empty
  figure-only page. No image/font scaling or margin manipulation.
- XeLaTeX twice; 12 content pages plus 2 reference-only pages. Fonts embedded,
  mallard is genuine JPEG, all citation/ref keys resolve, no missing glyphs or
  overfull/underfull warnings. Existing amsmath vec warning is nonfatal.
- Rechecked official SOICT rules on 2026-09-12:
  https://soict.org/submission/paper-submission/
  12 pages excluding references, Springer CCIS/LNCS, single-blind with author
  identities, no page numbers, full-paper deadline 16 September 2026.
- Quoted Sup-21K baseline rows match the primary HRM-PET Table 1:
  https://papers.nips.cc/paper_files/paper/2025/file/a978bdfeb195e4a574c0def98806346a-Paper-Conference.pdf
  Photo attribution/public-domain record also rechecked:
  https://commons.wikimedia.org/wiki/File:Mallard_duck_.jpg

Required before declaring every current numerical diagnostic verified:

`tools/verify_paper_diagnostics.py`: 3 datasets x 2 diagnostic arms, seed42.
The RP-only branch emits RouteTII/RP/Union/Both/Agree/RPOnly; routing-only
emits ClsRouted/RP/Union/RPOnly. These counters exist in different evaluator
branches: one run with both flags would NOT measure both. The completed
aggregate runs did not request these counters, hence six instrumented runs
are justified; this is not a new ablation or hyperparameter search.

Preflight pins restored source, all 60 checkpoint files, fixed features,
existing main baseline provenance, 3 RP-only control logs and the verified IMR
route-only arm. It compares all stages with those four non-instrumented
controls, validates overlap identities, compares the RP classifier across both
branches, and prints measured-versus-quoted values. UPDATE_PAPER means update
historical diagnostic numbers, not discard new evidence or tune parameters.
No --run means no GPU evaluation. Reuses its own completed logs; exclusive
new logs, shared verifier lock and >=16 GiB free GPU required. No training,
checkpoint writes, dataset changes, or modification of previous drivers.

Full lab command (run when the shared GPU is available):

```bash
cd /home/s24gbn1/Documents/truongnguyen/clean-check || exit
git pull --ff-only origin main || exit
PY=/home/s24gbn1/Documents/truongnguyen/Hybrid_ReMatching/.venv/bin/python
OUT=/home/s24gbn1/Documents/truongnguyen/hrm-pet-output
"$PY" -m pytest tests/ -q || exit
LOG=$(mktemp "$OUT/paper_diagnostics_finish_XXXXXX.log") || exit
nohup "$PY" -u tools/verify_paper_diagnostics.py \
  --output-root "$OUT" \
  --data-root /home/s24gbn1/Documents/truongnguyen/datasets \
  --tag paper_diagnostics_verify_v1 \
  --run >"$LOG" 2>&1 &
JOB=$!
printf 'PID=%s\nLOG=%s\n' "$JOB" "$LOG"
tail -F --pid="$JOB" "$LOG"
```

Completion: PAPER_DIAGNOSTICS_COMPLETE=6/6. Summary:
paper_diagnostics_verify_v1__summary.json. Not yet run on the lab GPU.

Scope boundary: a same-protocol RanPAC reproduction, controlled phase/peak-cost
benchmark and actual full-fusion sample repair/harm counts are further research,
not missing runs for the existing accuracy/retention tables. The paper does
not claim those measurements; its 283 s value remains explicitly historical,
not a baseline-relative or phase-cost measurement. Do not relabel oracle overlap
as actual Stage-1/Stage-2 repair. No automatic test-set tuning, new seed search,
SSH job launch, dataset repairs or submission.
Author approval of the final manuscript, author list, any competing-interest
statement and actual EasyChair upload remain author actions. Live Overleaf
compilation has NOT been tested; local XeLaTeX has.

## Completed ungated ablation — latest evidence,2026-09-12

User supplied paper_ablation_finish_M6pkdC.log, attachment
0fd9c5f5-20ec-4187-b947-d13478bd7d11, transcript SHA256
a5d145793a9195ae6fe011ccd7b14774bba8eefd729afe24e4b5928505ea1025.
Parsed15 distinct final records, each with10 stages;14 RUN lines plus1 reused
Sup class_plain. Five checkpoint/fixed-feature audits PASS. Source SHA remains
6747fed54a6f9679632bfdbbed1d3ebf2832dc801cf950b6164460f5a8858da9.
PAPER_ABLATION_COMPLETE=15/15 and shell Done. No repeat GPU run required.

Independent final/history comparison: all15 Acc@task/Acc@1/Acc@5/Loss match.
Nine arms differ in retention: four class_plain (all except MoCo) and all five
full_plain. All five route_only arms match all six historical final metrics.
These completed post-fix results supersede the "ungated ranges unverified"
notes below. The log's HISTORICAL_ONLY describes the old source logs, not the
newly verified evaluations. Do not mark new results unverified on that basis.

Verified ImageNet-R seed42 ablation, gate=none in all three arms:

| Backbone | Arm | Acc@1 | Forgetting | Backward |
|---|---|---:|---:|---:|
| Sup | route_only | 74.3112 | 3.2155 | -2.8676 |
| Sup | class_plain | 74.5300 | 3.7863 | -3.7464 |
| Sup | full_plain | 74.5499 | 3.7568 | -3.6970 |
| MoCo-v3 | route_only | 68.8050 | 3.0714 | -2.6673 |
| MoCo-v3 | class_plain | 70.6984 | 2.9888 | -2.9389 |
| MoCo-v3 | full_plain | 70.6288 | 3.0844 | -3.0190 |
| iBOT-1K | route_only | 72.4635 | 3.9249 | -3.9249 |
| iBOT-1K | class_plain | 74.9609 | 3.5125 | -3.1602 |
| iBOT-1K | full_plain | 74.9527 | 3.5055 | -3.1184 |
| iBOT-21K | route_only | 73.9070 | 3.8844 | -3.8104 |
| iBOT-21K | class_plain | 75.7244 | 3.1358 | -2.6837 |
| iBOT-21K | full_plain | 75.7164 | 3.1572 | -2.6162 |
| DINO | route_only | 70.9654 | 2.8248 | -2.5387 |
| DINO | class_plain | 72.9082 | 3.3055 | -2.5007 |
| DINO | full_plain | 72.8688 | 3.4052 | -2.5346 |

Acc@1 contrasts in Sup/MoCo/iBOT1K/iBOT21K/DINO order:
route_only minus identity: .2921/.3331/.5754/.6161/.7307;
class_plain minus identity: .5109/2.2265/3.0728/2.4335/2.6735;
full_plain minus class_plain: .0199/-.0696/-.0082/-.0080/-.0394.
Published rounded ranges remain correct: +.29..+.73, +.51..+3.07,
and -.07..+.02. No equivalence or across-backbone significance inferred.

Combining these with already-verified full-gated logs, full-gated minus
full_plain Acc@1 = +.9617/-.2635/-.5401/+.1240/+.0452. In particular,
the margin gate is not uniformly helpful on SSL backbones. The paper now
states the MoCo/iBOT1K declines explicitly; do not pick per-backbone gate
settings using these test results. The method remains w=.7,beta=.5,margin.

NotMNIST diagnosis is now specific: both reported files are in Train and
have0 bytes on disk AND in the corresponding members of the local notMNIST.zip.
All four byte hashes equal the empty-content SHA256:
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.
Thus the local archive already contains empty entries; no evidence of loss
specific to extraction or evaluator cleanup. This does not authenticate the
upstream archive or audit every sample. Loader skips them; no dataset edits.
Paper caveat updated from unknown unreadable images to empty training entries.

Main n=4,48 priority controls,9 weight cells,20 backbone-grid cells and this
15-arm ablation now have completed aggregate verification. Still separate:
historical per-sample/oracle diagnostics, phase-level cost, sample repair/harm,
same-protocol RanPAC reproduction and final submission checks. Do not claim
these optional/additional items are covered by aggregate verification.
No new local GPU run or algorithm/driver change in this synchronization.

Canonical PDF/LaTeX synchronized in place. XeLaTeX twice PASS;14 PDF pages,
content/acknowledgment end on12 and references start on13. Changed pages11–12
rendered and visually checked; no overlap, undefined references, missing
glyphs or over/underfull boxes. Existing amsmath vec warning remains.

## Remaining ungated ablation runner — 2026-09-12

Added tools/verify_paper_ablation.py, fixed ImageNet-R seed42 on the five
paper backbones. Three arms, gate=none: route_only(w=.7,beta=0),
class_plain(w=1,beta=.5), full_plain(w=.7,beta=.5). Fifteen arm results,
but Sup class_plain is read from its verified soict_final_v1 log using the
original74ff5c9 driver hash, so14 new evaluations remain.
All five identity baselines are read-only: Sup maskfix_verify_v2, four SSL
paper_backbone_verify_v1. Old drivers untouched to preserve their provenance.
No training, tuning or repeat of the completed20-cell grid/48controls.

Checks all five checkpoint sets and existing log metadata before GPU work,
uses the shared grid lock,16GiB GPU guard and exclusive logs; same-tag complete
logs are reusable. Historical ungated filenames are exact, optional comparison
only; missing old logs are explicitly NOT_FOUND, never substituted by gated
ones. The corrected fixed-arm results still allow direct paper contrasts.
Final marker PAPER_ABLATION_COMPLETE=15/15; default summary
paper_ablation_verify_v1__summary.json under the existing output root.

--audit-notmnist reads only the two reported images (Train/Test if present)
and matching ZIP members. Reports readability, SHA256 and split; never repairs,
extracts, deletes or replaces data. This is not a full dataset-integrity audit.
Local79 dependency-free tests PASS (10 new), no local torch/GPU run. The new
14 evaluations are pending; do not mark these ablation ranges verified yet.
PDF/model source unchanged in this runner-only update.

## Completed backbone grid — 2026-09-12, latest

User supplied paper_backbone_finish_HbWdK6.log in attachment
d6559399-1b00-4423-9148-8b6cafde7d87. Independently parsed40 distinct
records from the transcript with SHA256
1de722a0bf7ba3cd24291bb6b83e0e0a24aa3d882614e3ce604fa06031f63887.
The transcript contains40 distinct
BACKBONE_GRID_FINAL records,32 RUN lines and complete5/10-stage sequences.
Preflight reused8 verified arms; all20 checkpoint protocol/fixed-feature
checks PASS. Evaluator SHA remains
6747fed54a6f9679632bfdbbed1d3ebf2832dc801cf950b6164460f5a8858da9.
PAPER_BACKBONE_GRID_COMPLETE=20/20; HISTORICAL_FINAL_CORE=PASS;
HISTORICAL_RETENTION=CHANGED; shell reports Done. This batch is finished:
do not rerun it or ask the user for the same summary merely to confirm it.
The transcript includes the needed per-arm JSON and final summaries.

All40 final Acc@task/Acc@1/Acc@5/Loss values match history. Ten full arms
have retention differences; all20 identity arms retain all six final values.
This does NOT mean all intermediate metrics or retention were unchanged.
Nor does it prove that every historical difference came only from cleanup.

Corrected full-method Forgetting/Backward in changed arms:

| Dataset / backbone | Forgetting | Backward |
|---|---:|---:|
| IMR / iBOT-1K | 3.5577 | -3.5577 |
| IMR / iBOT-21K | 3.8695 | -3.8385 |
| IMR / DINO | 2.8626 | -2.7592 |
| CIFAR100 / Sup | 3.6778 | -3.6778 |
| IMA / Sup | 5.5603 | -5.1497 |
| IMA / iBOT-21K | 5.4401 | -5.1042 |
| IMA / DINO | 5.2178 | -4.9455 |
| 5-Datasets / MoCo-v3 | 0.2950 | -0.2525 |
| 5-Datasets / iBOT-21K | 0.5634 | -0.3359 |
| 5-Datasets / DINO | 0.5099 | -0.3124 |

Post-fix seed42 grid, deltas full minus identity, percentage points.
Lower delta Forgetting and higher delta Backward are better:

| Dataset / backbone | Identity Acc@1 | Full Acc@1 | Delta Acc@1 | Delta F | Delta B |
|---|---:|---:|---:|---:|---:|
| imr / sup | 74.0191 | 75.5116 | 1.4925 | -0.2301 | 0.0935 |
| imr / mocov3 | 68.4719 | 70.3653 | 1.8934 | 0.5871 | -0.7760 |
| imr / ibot1k | 71.8881 | 74.4126 | 2.5245 | -0.8791 | 0.8791 |
| imr / ibot21k | 73.2909 | 75.8404 | 2.5495 | -0.3724 | 0.4034 |
| imr / dino | 70.2347 | 72.9140 | 2.6793 | -0.5769 | 0.5211 |
| cifar100 / sup | 89.7700 | 90.5300 | 0.7600 | -0.2778 | 0.2555 |
| cifar100 / mocov3 | 85.5000 | 87.6200 | 2.1200 | -0.2444 | 0.2112 |
| cifar100 / ibot1k | 86.4600 | 88.5200 | 2.0600 | -0.8000 | 0.7556 |
| cifar100 / ibot21k | 88.5200 | 90.3200 | 1.8000 | -0.6000 | 0.6000 |
| cifar100 / dino | 85.3600 | 87.3900 | 2.0300 | -0.8222 | 0.7889 |
| ima / sup | 45.8519 | 47.4201 | 1.5682 | -1.0518 | 0.2827 |
| ima / mocov3 | 31.7988 | 34.2100 | 2.4112 | 0.0462 | 0.1165 |
| ima / ibot1k | 37.0770 | 40.6572 | 3.5802 | 0.3877 | -0.3175 |
| ima / ibot21k | 41.4776 | 44.1051 | 2.6275 | -1.1901 | 1.1515 |
| ima / dino | 36.4376 | 37.4075 | 0.9699 | 1.8988 | -1.8964 |
| fivedatasets / sup | 91.3703 | 93.7699 | 2.3996 | -1.2229 | 1.2604 |
| fivedatasets / mocov3 | 92.6277 | 94.0418 | 1.4141 | -0.6683 | 0.7108 |
| fivedatasets / ibot1k | 93.7977 | 94.6477 | 0.8500 | -0.0532 | -0.1761 |
| fivedatasets / ibot21k | 93.8320 | 94.4203 | 0.5883 | 0.1661 | -0.0836 |
| fivedatasets / dino | 92.5238 | 93.2712 | 0.7474 | -0.0769 | 0.1199 |

Retention reconstruction PASS_ROUNDED_3DP for40/40 arms,20/20 pairs.
Full-minus-identity final accuracy on old tasks improves in20/20 cells,
mean1.8278972222; learning-time old-task mean improves1.5828958333.
Their difference is about+.2450 Backward. Both paper rounded means+1.83/+1.58
remain supported after the fix. Backward worsens in5/20 cells; Forgetting
also worsens in5/20, not the identical set. These are descriptive single-seed
results, not20 independent replications or evidence of universal retention gain.

Preflight warnings: NumPy alignment deprecation; non-writable NumPy tensor
warning; two unreadable NotMNIST images:
F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png and
A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png.
The NotMNIST loader catches image-loading failures and skips these samples.
No data was repaired/replaced/deleted. Final-core agreement supports
reproducibility of the current pipeline, NOT completeness of the source
dataset or proof that these omissions have zero impact. The transcript does
not identify Train/Test for these warnings. Preserve the prepared split.

Paper updated in place to close grid pending statements, retain corrected
retention interpretation and disclose the NotMNIST caveat. Model/driver code
unchanged, so source/driver metadata for completed runs remain reusable.
Still open separately: historical ungated/routing-only five-backbone ranges,
unmeasured phase costs/sample repair-harm and same-protocol RanPAC reproduction.
None is solved merely by the20-cell identity/full verification.

PDF QA: final XeLaTeX build twice PASS; content and acknowledgment end on
page12, References start on13 (14 PDF pages total). Changed pages2,9–12
rendered and visually checked, no overlap/clipping. No undefined references,
missing glyphs or over/underfull boxes; existing amsmath vec warning only.
No model-code changes or new local GPU run in this synchronization.

## Current checkpoint — 2026-09-12

Supersedes the pending-weight-grid/CUB/SSL notes below. The user's latest
audit supplied all 40 historical logs (20 baseline/full cells), all with
complete stage sequences but none with verification metadata. Historical
completeness is not proof of post-fix provenance. Do not rerun the completed
main n=4 runs, 48 controls or nine weight cells.

The 3x3 weight grid is complete after the fix. Final Acc@1 is unchanged in
all nine cells. At beta=.7, corrected Forgetting/Backward are:
w=.6: 3.4364/-3.3817; w=.7: 3.4069/-3.3027; w=.8: 3.2889/-3.1995.
Intermediate stages differ, including Loss from task2 onward. The pattern is
consistent with masking changes, but old logs lack metadata; do not claim
that cleaning alone is proven to cause every difference.

Added tools/verify_paper_backbone_grid.py and 22 dependency-free tests:

- Fixed four datasets x five backbones, seed42, identity(w=1,beta=0) and
  full(w=.7,beta=.5). Historical baseline is NOT gated-class-only.
- Checks all selected inputs/checkpoints before GPU execution. Supports
  five-task/five-adapter 5-Datasets, and refuses missing/ambiguous ImageNet-A
  splits rather than invoking the loader's destructive automatic splitting.
- Reuses up to eight exact verified arms: Sup IMR/CIFAR identity/full, and
  four IMR SSL full logs. Original driver commits are pinned, not trusted
  from the log itself. Up to32 new evaluations if all eight are reusable.
- Checks saved configuration, strict exemplar-free protocol, source hash,
  actual frozen feature tensors, checkpoint hashes and mutation stamps.
- No training/checkpoint writes, no weight search, no log overwrites.
  Single-runner lock and16GiB free-GPU guard; stops without killing others.
- Same-tag completed logs are reusable. Incomplete or mismatched logs are
  preserved and block reuse; a failed evaluation requires investigating its
  traceback, not modifying metadata or deleting the evidence.
- Exports six final metrics, all stage rows, historical deltas, paired
  full-minus-identity differences and summary JSON. Reconstructs old-task
  final/learning-time means from three-decimal per-task summaries when
  present and consistent; otherwise explicitly UNAVAILABLE, never inferred
  from overall averages. No additional GPU run just for this decomposition.
- Completion and historical agreement have separate markers: CHANGED
  retention is not a failed evaluation and does not abort remaining cells.

Local69 dependency-free tests PASS; no torch/GPU evaluation on Windows.
The actual20-cell post-fix results still require the lab run. This runner
does not verify the historical five-backbone ungated/routing-only ablation
ranges; those remain explicitly historical in the paper. Same-feature
RP-only is not relabelled as a full RanPAC reproduction.

Canonical reports/lncs_method_en.tex and .pdf updated in place:
main/CUB and fixed-feature audits complete, five-backbone gated controls now
reported, RP-only means included, nine-cell verification described without
claiming unchanged retention. Historical grid conclusions remain qualified.
Code reference now names paper-system revision4989ab8, not pre-fix ee32564.
No algorithm changes; backup tag system-baseline-2026-09-11 remains intact.
XeLaTeX compiled twice;12 content pages +2 reference pages, embedded fonts,
no undefined references/missing glyphs/overfull or underfull boxes; existing
amsmath vec warning remains. Changed content pages9–12 visually checked.
Live Overleaf and the full final-submission audit are not certified by this.

Lab (current local branch clean-code): git pull --ff-only origin main,
then pytest and tools/verify_paper_backbone_grid.py --run, with the existing
output/data roots. Default tag paper_backbone_verify_v1. The JSON is written
under the output root as:
paper_backbone_verify_v1__imr-cifar100-ima-fivedatasets__summary.json.

## Paper system restored after exploratory trials - 2026-09-11

User stopped algorithm improvements and requested the paper system back.
Configs/engines/trainers/vits/peft/dataset source now match the immutable
system-baseline-2026-09-11 tag (4989ab8) in Git. The three experimental source,
runner and test files are removed from the active tree; recover them from
b9d8def if needed. No lab logs, checkpoints, datasets or untracked files deleted.
Canonical LaTeX/PDF hashes remain identical to the tag, not overwritten.

Exploratory route trials completed all3 datasets/four seeds; all12 reference
runs matched all10 stages of the original paper. Lab127tests passed before
restoration. These results are archived in fusion_experiments.md, not adopted
as paper-method results. Current local47 dependency-free tests PASS after
restoration, including6 new weight-grid runner guards. No local torch/GPU run.

The paper's main n=4 results, identity checks, fixed-feature audits and48
priority control evaluations are already covered. Do not rerun those GPU jobs.
Remaining verification: seven of nine joint-weight cells at ImageNet-R seed42;
the historical20-cell cross-dataset/pretraining grid and historical ungated/
routing-only ablations need provenance mapping beyond the completed controls.
Phase-level cost and sample repair/harm are optional additional measurements,
not supplied by these runners. Stale CUB/SSL/fixed-feature pending prose in the
unchanged paper must eventually be synchronized with existing evidence.

Added tools/verify_paper_weight_grid.py: fixed existing9-cell table only,
reuses the two verified cells with exact metadata (original74ff5c9 driver for
w=.6), evaluates7 with --run, preserves old logs, and checks historical values.
No model code changes. Restored Linux evaluator must have source digest
6747fed54a6f9679632bfdbbed1d3ebf2832dc801cf950b6164460f5a8858da9;
raw Windows line endings can differ, so local restoration was checked in Git.
Use a new paper_grid_verify_v1 log tag. It is not a parameter-selection sweep.

## MoCo completion and separate system experiments - 2026-09-11

User supplied MoCo seed42 class_gate/full with all10 stages, EVIDENCE_FINAL,
FIXED_FEATURE_TENSORS=PASS, CHECKPOINT_PROTOCOL=PASS and
SOICT_EVIDENCE_COMPLETE=ssl:mocov3. Lab111tests PASS. Source SHA is unchanged
from the46-run batch. This closes48/48 priority evaluations, not the historical
20-cell grid or phase/sample-level audits. The46/48 notes below are historical.

MoCo class_gate/full: Acc@1 70.0789/70.3653, Acc@task73.8206/76.7719,
Acc@5 83.8112/83.7096, Loss1.7932/1.7808, Forgetting3.4351/3.6900,
Backward-3.3851/-3.6090. Routing adds+.2864 Acc@1 but worsens Acc@5/retention.
All five ImageNet-R backbone conditional Acc@1 contrasts are positive at
seed42; this is not multi-seed statistical evidence for the four SSL backbones.

User authorized system experiments only after a recovery point. The pushed tag
system-baseline-2026-09-11 preserves4989ab8 code and paper. New opt-in routing,
gate-floor and M5000 probes are documented in fusion_experiments.md; they are
not paper results. Default formulas remain unchanged; no new GPU runs locally.
Local52 tests PASS and5 torch tests skipped. Paper/source PDF unchanged.

## Results received and MoCo audit correction — 2026-09-11 (latest)

Attachment c0e6208c-3f0b-4597-bc22-f018d4290da9 contains the complete run
transcript from soict_finish_GBaXoc.log. Lab107tests PASS. Source digest remains
6747fed54a6f9679632bfdbbed1d3ebf2832dc801cf950b6164460f5a8858da9.
Independently counted46 unique EVIDENCE_FINAL rows, each following tasks1–10:
core36, weights4, SSL6. Markers confirm audit/core/weights complete; SSL stopped
before MoCo evaluation. This is NOT completion of the whole48-run batch.

Main fixed-feature tensor checks now PASS on all12 dataset/seed combinations
across ten checkpoints each. Protocol, identity all stages and historical final
core comparisons PASS. Main means match the paper, including CUB n=4.
The small previously documented CIFAR retention changes remain; no claim that
all intermediate metrics are identical. DINO/iBOT1K/iBOT21K fixed-feature
checks also PASS at seed42. No MoCo invariant conclusion yet.

Four-seed final Acc@1, mean +/- sample SD (computed again from final rows):

| Method | ImageNet-R | CIFAR-100 | CUB-200 |
|---|---|---|---|
| HRM-PET | 73.943875 +/- .476855 | 89.782500 +/- .027538 | 86.383625 +/- .248578 |
| Same-feature RP-only | 69.885925 +/- .165950 | 88.800000 +/- .197990 | 87.259350 +/- .271426 |
| Ungated class-only | 74.493800 +/- .141171 | 90.147500 +/- .157348 | 88.006125 +/- .207748 |
| Gated class-only | 75.096975 +/- .213999 | 90.502500 +/- .078049 | 87.721900 +/- .259690 |
| Full | 75.179650 +/- .240664 | 90.472500 +/- .050580 | 87.878350 +/- .274068 |

Full minus RP-only paired Acc@1: IMR+5.293725 CI[5.048580,5.538870],
CIFAR+1.672500 CI[1.369366,1.975634], CUB+.619000 CI[.148907,1.089093].
This is a matched-head control, not a full official RanPAC reproduction.
RP-only Loss uses uncalibrated ridge scores; do not interpret its large Loss
as a controlled calibration comparison. RP-only Acc@task is the task of its
predicted class, not a DRM-proposal metric.

Verified conditional routing Acc@1 contributions: IMR+.082675 CI[.025628,.139722],
CIFAR-.030000 CI[-.085122,.025122], CUB+.156450 CI[.055407,.257493].
Gate without routing: IMR+.603175 CI[.327365,.878985], CIFAR+.355000
CI[.056884,.653116], CUB-.284225 CI[-.570428,.001978]. These reproduce the
previous historical estimates. Full is not uniformly best among the variants;
ungated class-only has the largest CUB mean. Do not select a per-dataset winner
from these evaluations and then present it as a prespecified method.

Full w=.6 minus .7 is verified: Acc@1+.108000 CI[-.063724,.279724];
Acc@task+.332350 CI[.143000,.521700]. No reason established to replace w=.7.
SSL seed42 gated class-only/full: DINO72.1447/72.9140 (+.7693),
iBOT1K73.6373/74.4126 (+.7753), iBOT21K75.4100/75.8404 (+.4304).
No multi-seed confidence interval can be inferred for these three comparisons.

The stop is an audit-tool assumption, not evidence of model tensor drift or
GPU failure: vit_base_patch16_224_mocov3 sets fc_norm=True; VisionTransformer
therefore uses norm=Identity. Its state has no norm tensors. fc_norm runs AFTER
pre_logits, so it is not part of the RP feature path. The checker now accepts
only that specific known MoCo structure, requires fc_norm weight/bias, and still
hashes all embeddings/blocks and task0 LoRA. Other backbones still require norm.
Added --ssl-backbones mocov3 to run only the remaining two evaluations.
Completion marker is ssl:mocov3, not a claim that this invocation ran all SSL.

Local15 helper tests and26 existing verification tests PASS. No engine/config/
model code or PDF changed. New helper driver hash changes, so do NOT rerun the
old all-mode loop against completed logs; use the MoCo filter and keep all46
old logs with their original metadata. The MoCo failure preceded RUN/log creation,
so the same soict_final_v1 tag can be used for those two still-missing files.
Historical20-cell grid, phase costs and sample repair/harm remain open.

## Deadline review — 2026-09-11 (current)

Read-only review of the same English source/PDF; no paper numbers, figures,
model code, checkpoints, or hyperparameters changed in this review.

### New evidence closes CUB main-result verification

User attachments 458ec443-3e22-426e-9a51-ce7d5017a0b8 and
04a3783f-8aa9-441a-b495-4ad61940ea0f show CUB seeds43–45 completing all
three arms under paperfix_verify_v3. Identity all stages, historical final
baseline/full, consistency and historical coverage all PASS. The second output
reuses completed logs, not an independent second GPU evaluation. Lab pytest:
96 passed. CUB seed42 was already completed under maskfix_verify_v2.

Acc@1 baseline/full at seeds43/44/45 respectively:
86.6136/87.7534, 86.0530/87.6587, 86.3388/88.2766. Intermediate Acc@1
and final retention match historical logs; intermediate Loss changes.
The new n=3 summary must not replace the paper's n=4 results.

### Priority findings before September16

1. Update the stale CUB-pending statements in Setup and Table1 caption.
   Main verification is now covered across all three datasets/four seeds;
   historical grid/ablation verification is a separate open item.
2. The fixed-feature assumption needs a real checkpoint test: the LoRA RP
   path calls the currently loaded model with adapter index0. pin_rp_extractor
   snapshots original_model, not this LoRA path. Compare patch embeddings,
   tokens, transformer blocks, final norm and the four task0 K/V LoRA slices
   across all ten checkpoints before asserting fixed feature space. Classifier
   MLP/fc_norm/head occur after pre_logits and must not be compared as features.
3. The nearest controlled comparator is the exact same RP head used alone.
   Add RP-only, ungated class-only, gated class-only and full in one table.
   RP-only on HRM-PET LoRA is NOT a faithful reproduction of official RanPAC.
   Published baselines are contextual, not controlled evidence of superiority.
4. Sections4.4/4.5 give twenty-cell/five-backbone ranges without the underlying
   table. Supply a compact table or accessible manifest, and keep single-seed
   and historical-provenance limitations explicit. Do not invent missing cells.
5. Full w=.6 vs .7: verify four pairs; keep .7 fixed for the deadline rather
   than choose a winner from the evaluation set. An unresolved difference is
   not equivalence. Gate benefit on CUB and retention remain unresolved.
6. Novelty is the reuse of RP evidence at two decision points, not ridge/RP.
   Existing conditional routing effects are small. A measured inference/fit
   cost comparison would strengthen the value argument, but the historical
   total283s is not such a measurement. Phase profiling and sample repair/harm
   counts are not supplied by the new checker.
7. Shorten repeated caveats in abstract/conclusion, move own diagnostics out of
   Related Work into Experiments, and recover space from the dedicated figure
   page for the ablation table. Preserve limitations once, clearly; no unsupported
   stronger claims. No new architecture or test-selected hyperparameter changes
   are recommended this close to submission.

### Visual and reference checks

Rendered and inspected all14 pages at this revision. Content/ack ends on12;
references are13–14. No visible overlapping text/figure labels or missing glyphs;
figure5 has substantial unused page space and 7pt labels remain relatively small.
All fonts embedded,30 citation keys and30 bibliography items, no missing/duplicate
keys. Existing compilation log: no undefined/overfull/underfull; vec warning only.
No recompilation or live Overleaf test was performed in this read-only review.

Official https://soict.org/submission/paper-submission/ checked September11:
full paper September16, max12 pages excluding references, CCIS/LNCS template,
single-blind with authors, PDF without page numbers. Cutoff timezone not stated
on that page; do not assume AoE. Core HRM-PET/RanPAC/DLEPEM sources rechecked
against NeurIPS/publisher/official code. Bibliography metadata could be made more
uniform (pages/DOIs), but no new false core reference was identified.

### Runnable evidence checker (not yet executed on the lab)

tools/finish_soict_checks.py has modes audit/core/weights/ssl. audit uses CPU and
existing logs only, combines CUB v2 seed42 with v3 seeds43–45, checks checkpoint
provenance/fixed-feature tensors, verifies identity and historical final core,
and prints n=4 main summaries. New evaluation requires --run. core adds36 runs
(3 controls x3 datasets x4 seeds); weights adds4 IMR full-w=.6 runs; ssl adds8
gated-control/full runs on four SSL backbones at seed42. They are separate modes;
do not start concurrently or duplicate the earlier18-run batch if already active.

Successful main baseline/identity/full logs are never rerun or rewritten. New
logs have source/checkpoint/command/driver hashes and exit markers, and are
reused only on exact provenance match. Incomplete/conflicting logs stop the run.
Before each new evaluation require16GiB free by default; this is a preflight
check, not a GPU reservation or an OOM guarantee. No automatic waiting/killing,
dataset splitting, downloads by this checker, or model checkpoint writes.
SSL constructors may still require existing external pretrained assets.

New helper tested locally:11 CPU-only unit tests PASS; existing26 verification
unit tests PASS. These do not validate GPU numerical results. Evaluator source
files unchanged. Script must be transferred explicitly to the lab before use;
it has not been pushed to the public remote.

## Thu hẹp kết luận theo audit/log trọng số — 2026-09-10, mới nhất

Nguồn: hai đầu ra audit giống nhau, attachments
e5f1c6c6-52b8-42ad-a73b-2bc954bee673 và 0a8f028d-0fd1-40d7-9566-64da1a32adfd,
và tám final rows full-method cùng kết quả kiểm tra bốn đường dẫn control
do tác giả gửi tiếp trong hội thoại. Đây không phải một lượt CUB mới.

- Audit liệt kê769 log,439 log có final row thuộc các cấu hình được nhận diện.
  Không được gọi toàn bộ phần còn lại là run lỗi. Các dòng THIEU áp dụng
  từng cấu hình riêng, không phải yêu cầu chạy tất cả.
- Đúng nhóm IMR Sup-21K, seed42, M=lambda=10000, p1, không chuẩn hóa,
  w=.6/.7/.8, beta=.3/.5/.7, margin gate: đủ9 ô, khớp Table3.
- So sánh full w=.6 trừ .7, beta=.5, margin, seeds42–45: Acc@1 deltas
  [0.1020, -0.0336, 0.2263, 0.1373], mean0.108000, sampleSD0.107920,
  CI95[-0.063724,0.279724]. Giữ số này và Acc@task+0.332 trong paper.
- Tám log trên có task10, nhưng không có VERIFICATION_META/EXIT_MARKER.
  Thiếu marker không tự chứng minh chạy lỗi; cũng không chứng nhận đúng
  source/checkpoint hoặc đã re-evaluate sau maskfix. Gắn nhãn historical.
- Không có ClsRouted trong các final rows vừa gửi. Audit không ghép được
  cặp routing-only w=.6/.7. Đã bỏ claim +0.170 CI[+0.136,+0.204]:
  chưa tìm được nguồn đủ để giữ claim, không kết luận hiệu ứng thực bằng0.
- Bốn đường dẫn class-only+margin trên DINO/iBOT1K/iBOT21K/MoCo-v3 đều
  MISSING. Đã bỏ claim routing góp +0.13..+0.77 trên cả năm backbone khi
  gate bật; không lấy full-gated minus ungated-class làm routing effect.
- Giữ ungated contrast -0.07..+0.02 và conditional contrasts Sup-21K trên
  ba dataset/bốn seed theo log lịch sử. Sửa rounding routing-only gain cao
  nhất từ0.74 thành0.73 (DINO70.9654-70.2347=0.7307).
- Abstract, tiêu đề/đoạn ablation và Conclusion đã đồng bộ phạm vi.
  Chưa thêm bảng RP-only/RanPAC: audit có RP-only nhưng phải tách feature
  source/preprocessing và xác minh provenance, không gọi tương đương RanPAC.

Không thay các ô trong main table hoặc grid; không đổi code, chạy GPU hay push.
XeLaTeX hai lượt PASS, không undefined/overfull/underfull; warning vec có sẵn.
Đã render xem đủ14 trang, kết luận/ack kết thúc trang12, references trang13–14.
Chưa thể gọi bản này sẵn sàng nộp: CUB43–45, post-fix historical experiments
và checkpoint/config provenance còn mở. Các đoạn review cũ phía dưới chỉ là
lịch sử, đặc biệt những câu đã từng coi routing-only+0.170 là đủ bằng chứng.

## Bổ sung log CIFAR tuyệt đối — 2026-09-10, mới nhất

Tác giả gửi đầu ra trích log maskfix_verify_v2, attachment
fa2495b7-393e-4bda-8887-c1d3bc7a9d0a. Đã tính độc lập lại mean và sample SD
từ bốn final rows seeds42–45. Mỗi nhánh baseline/identity/full CIFAR đều được
script báo COMPLETE (đủ 10 stage, có exit code0); các source hashes được in
đều là 6747fed54a6f9679632bfdbbed1d3ebf2832dc801cf950b6164460f5a8858da9.
Đầu ra này không chứa toàn bộ checkpoint manifest hoặc phép so từng stage;
không mở rộng kết luận provenance ngoài các kiểm tra đã có ở lượt trước.

| Metric | Baseline mean ± sample SD | Full mean ± sample SD |
|---|---|---|
| Forgetting | 3.894475 ± 0.083898 | 3.744450 ± 0.124065 |
| Backward | -3.872225 ± 0.066955 | -3.736125 ± 0.111635 |

Đã thay hai ô Pending của Table1 bằng 3.74 ± 0.12 và -3.74 ± 0.11,
bỏ câu giải thích Pending không còn dùng. Paired changes/CI giữ nguyên.
R1 nay đã xử lý đủ cả absolute và paired statistics ở precision log cung cấp.
Không chạy mô hình, không đổi code hoặc các con số của CUB.

CUB trong tag này: seed42 đủ cả ba nhánh; seed43 baseline COMPLETE nhưng
identity INCOMPLETE_OR_FAILED và full MISSING; seeds44–45 cả ba nhánh MISSING.
Chưa nhận đầu ra audit_weight_evidence; grid/ablation/weight và các mục xác
minh khác vẫn còn mở. Không coi bảng CUB bốn seed đã được xác minh đầy đủ.

Các đoạn Pending CIFAR và hashes ở mục sau mô tả bản aad3e01 TRƯỚC bổ sung
này; không còn là trạng thái hiện tại. Tag checkpoint cũ vẫn giữ nguyên.

XeLaTeX hai lượt PASS; đã render kiểm tra các trang9–14 sau cập nhật, không
thấy lỗi bố cục. PDF vẫn14 trang, References bắt đầu trang13; không có
undefined/overfull/underfull, chỉ warning vec có sẵn.

SHA-256 workspace sau bổ sung:

| Tệp | SHA-256 |
|---|---|
| lncs_method_en.tex | C413CFC78549C23A78CAE4640EEE073933978C3B4DA0213E1086F53BA447BEA1 |
| lncs_method_en.pdf | 375E93C01EC32C1445CE87B0ED30E9EAAE895E860C45038A2CD6F908A9049D91 |

## Cập nhật sau khắc phục — 2026-09-10

Đã sửa trực tiếp nguồn LaTeX và biên dịch lại PDF cùng tên. Không tạo bản sao
paper/folder mới; không đổi code, pipeline, checkpoint mô hình hoặc chạy lab/GPU.
Tag paper-checkpoint-2026-09-10 vẫn trỏ tới 64f60a3, giữ nguyên bản trước sửa.
Phần review gốc bên dưới được giữ để truy nguyên; các số dòng của nó là lịch sử.

### Đã xử lý

- R1: cập nhật paired CIFAR Forgetting thành -0.150 ± 0.095, Backward thành
  +0.136 ± 0.095, CI95 tương ứng [-0.302, +0.002] và [-0.015, +0.287].
  Acc@5 delta làm tròn +0.218 ± 0.057; Loss delta -0.0003 ± 0.0012.
  Hai ô Proposed retention tuyệt đối chuyển thành Pending có giải thích:
  chưa đủ bốn final rows sau sửa để tính mean/sample SD, không giữ số cũ
  hoặc suy ngược SD từ số làm tròn.
- R2: bỏ kết luận gate gây hại rõ ràng trên CUB. Ước lượng -0.284 có CI
  [-0.570, +0.002], chứa 0; không đồng nghĩa full thua HRM-PET.
- R3: phân biệt routing-only với full minus gated class-only; ghi rõ gate
  được đo bằng gated minus ungated class-only. Đây là hiệu ứng có điều kiện,
  không phải đóng góp độc lập của hai stage.
- R4: bỏ luận điểm chưa có căn cứ rằng w=.7 tối ưu/ổn định trên mọi metric.
  Phân biệt grid một seed, đối chiếu full bốn seed và routing-only bốn seed.
  Giữ w=.7 vì nhất quán lựa chọn trước, không khẳng định tối ưu full method.
- R5: sửa phạm vi sanity check thành IMR/CIFAR42–45 và CUB42; đánh dấu CUB
  trong main table là kết quả lịch sử chờ xác minh đầy đủ. Identity PASS
  không được dùng để chứng nhận toàn bộ full/grid/ablation.
- R6: bổ sung LoRA rank/block/KV, split số task/lớp, RP fit dùng train split
  với eval transform, cấu hình preprocessing, seed, chuẩn hóa, CRM và revision
  code. Bổ sung nguồn dataset/ViT/MoCo-v3/iBOT/DINO; hoàn thiện MoTE và DS-AL.
  Table 2 tách rõ số trích từ HRM-PET (3 seed) và chạy nội bộ (seed42);
  bold chỉ xếp hạng trong từng nhóm, không ngụ ý đối chứng cùng protocol.
- R7: giới hạn phát biểu PET vào họ task-specific; phân biệt closed-form fit
  với không huấn luyện; thêm điều kiện fixed map/exact arithmetic cho joint fit;
  phân biệt ước tính RAM với peak đo thực; chi phí forward theo ảnh, ridge theo
  task; gắn số 74.31 với diagnostic routing-only, không với full.
  Ghi rõ nguy cơ CUB/ImageNet pretraining overlap chưa được audit.
- Rút diễn giải lặp ở intro/related work/kết quả để giữ 12 trang nội dung,
  không đổi cỡ chữ, lề, class LLNCS hoặc dùng khoảng cách âm.

### Còn thiếu dữ liệu hoặc xác minh, không được gọi là đã hoàn tất

1. Bốn final rows CIFAR baseline/full seeds42–45 sau maskfix để điền hai ô
   Proposed Forgetting/Backward; cần log đủ precision, không số bảng làm tròn.
2. Xác minh hoàn chỉnh CUB43–45 và các run lịch sử grid/ablation/weight,
   kèm config/log/checkpoint manifest. Lượt này không chạy vì lab đang bận.
3. Đối chiếu tensor backbone/p1 qua checkpoint, protocol/class order và
   pretraining provenance. Mô tả code không thay thế kiểm tra checkpoint thực.
4. RanPAC/RP-only cùng protocol, chi phí theo phase và sample-level repair/harm
   là thiếu sót thực nghiệm đã công khai, chưa có số để bổ sung.

Đây là bản checkpoint đã sửa lỗi có bằng chứng, CHƯA phải bản sẵn sàng nộp.

### Kiểm tra bản sau sửa

- XeLaTeX hai lượt thành công. PDF 14 trang: nội dung/kết luận/ack kết thúc
  trang12; References trang13–14. Đã render và xem đủ từng trang; không thấy
  chồng chữ, cắt bảng, mất ảnh hoặc lỗi dấu tiếng Việt.
- Không undefined citation/reference, không overfull/underfull box; 30 citation
  keys tương ứng 30 bibliography entries, không thiếu hoặc thừa key.
- Font đều embedded. Còn warning amsmath về math accent vec có sẵn;
  không phải lỗi compile/render. Không chứng nhận compile trực tiếp trên Overleaf.
- 26 unittest công cụ verification PASS trên CPU; không phải full pytest/GPU
  và không xác minh lại kết quả mô hình.
- Ảnh mallard giữ nguyên hash; chỉ source/PDF và tài liệu review thay đổi trong
  Git. Không push remote.

SHA-256 tệp sau sửa tại workspace:

| Tệp | SHA-256 |
|---|---|
| lncs_method_en.tex | 96C931F1A2145E122E27C4C6A7CCDCCACF2DB1722E09739511504CC25DFDD578 |
| lncs_method_en.pdf | 9B8482DFD507EC6A7873D0025A451D0BA296CAA8E0D72137814D53623DFA6898 |
| figure_assets/mallard.jpg | 04B95BDE9B65ECD95931B2FA8310714EDFB12DFC70B3E4F19905969D40363B8A |

## Review gốc trước sửa — kết luận lịch sử

Có thể lưu làm mốc phát triển; CHƯA nên gọi là bản sẵn sàng nộp.
Không phát hiện sai công thức cốt lõi trong nhánh inference dùng cho paper.
Còn lỗi diễn giải ablation, số retention CIFAR chưa cập nhật, và thiếu thông tin
để kiểm chứng độc lập một số kết quả lịch sử. Không thay nội dung paper hoặc số
liệu trong lượt review này; không chạy thêm thí nghiệm trên lab.

Nguồn chuẩn: reports/lncs_method_en.tex và PDF cùng tên. Các số dòng dưới đây
tham chiếu nguồn tại commit 9db0dbae4e581aac6af7cf742603c106ccdd9b9a.
Checkpoint Git: paper-checkpoint-2026-09-10. Tag lưu cả nguồn, ảnh gốc,
PDF đang được kiểm tra và tài liệu này; không tạo folder/copy paper mới.

SHA-256 trước review:

| Tệp | SHA-256 |
|---|---|
| lncs_method_en.tex | 53588586DFFE8A3942F5B52C49217F05C1F7ED24B052F9F34FEF2C5414D47BE5 |
| lncs_method_en.pdf | 82D54D0B94211295557A6751F50F614BFB33534FC737D46F44CDFEF5F832F218 |
| figure_assets/mallard.jpg | 04B95BDE9B65ECD95931B2FA8310714EDFB12DFC70B3E4F19905969D40363B8A |

## Việc cần sửa trước khi gọi là bản hoàn chỉnh

### R1 — Số retention CIFAR cũ (ưu tiên cao)

Vị trí: Table 1, dòng 653–654; phần retention, dòng 681–691.
Log xác minh do tác giả cung cấp, tag maskfix_verify_v2, đủ bốn seed cho CIFAR:

| Paired full minus baseline | Đang viết | Tính lại từ log sau sửa |
|---|---|---|
| Forgetting, mean ± sample SD | -0.156 ± 0.098 | -0.150025 ± 0.095366 |
| Backward, mean ± sample SD | +0.142 ± 0.097 | +0.136100 ± 0.094869 |
| Forgetting, CI95 | [-0.312, +0.001] | [-0.301773, +0.001723] |
| Backward, CI95 | Không ghi ở đoạn này | [-0.014858, +0.287058] |

Full seed42/45 có Forgetting mới-cũ +0.0111, Backward mới-cũ -0.0111.
Phải cập nhật cả mean/SD tuyệt đối ở cột Proposed từ bốn final rows đầy đủ;
không suy ngược SD mới từ số đã làm tròn trong bảng.

Quan trọng: bản paper hiện tại ĐÃ nói cả sáu CI retention chứa 0. Vì vậy đây
là sửa số liệu, không phải đảo ngược một kết luận significant đang có trong paper.
Ghi chép bàn giao lịch sử nói 5/6 CI chứa 0 không còn là kết luận chuẩn.
Acc@1/Acc@task cuối của IMR và CIFAR không đổi qua lần sửa mask đã kiểm chứng.

### R2 — Kết luận về gate trên CUB mạnh hơn bằng chứng (ưu tiên cao)

Vị trí: abstract dòng 70–74; ablation dòng 771–778; conclusion dòng 841–846.
Gate trên CUB có ước lượng -0.284 pp nhưng CI95 [-0.570, +0.002] chứa 0.
Câu gate làm giảm accuracy và câu phương pháp thua trên một dataset cần:

- Ghi đó là xu hướng/ước lượng âm, chưa phân biệt được với 0 ở CI đang dùng.
- Nêu đúng đối chứng: full so với ungated class-only, KHÔNG phải HRM-PET.
- Không đánh đồng với kết quả chính: Table 1 vẫn báo full tốt hơn HRM-PET
  trên cả ba dataset.

Cách diễn đạt đề xuất: The gate has a negative estimated contribution on CUB-200,
but its confidence interval includes zero; the full method still improves over
HRM-PET.

### R3 — Hai nghĩa khác nhau của routing alone (ưu tiên cao)

Vị trí: abstract; Sect. 4.5 dòng 759–778; Sect. 4.7 dòng 828–835.
Phép so sánh bốn seed thực tế được mô tả là:

| Contrast | Điều kiện giữ nguyên | Ý nghĩa đúng |
|---|---|---|
| full - class+gate | beta=.5, gate=margin | Đóng góp routing khi đã có class fusion và gate |
| class+gate - class-only | w=1, beta=.5 | Đóng góp gate khi chưa có routing fusion |
| routing-only - baseline | beta=0 | Tác dụng routing riêng; không phải contrast thứ nhất |

Không gọi contrast thứ nhất là routing alone hoặc hai stage chạy riêng.
Đây là ablation có điều kiện theo đường thêm module, không phải phân rã đóng góp
duy nhất khi có tương tác giữa routing và gate.
Limitations nói conditional routing ablation là single-seed nhưng đoạn ngay
trước có contrast bốn seed; cần chỉ rõ single-seed là phần năm backbone.
Câu four seeds resolve its sign cũng không đúng cho CIFAR:
-0.030, CI [-0.085, +0.025].

### R4 — Căn cứ chọn w và phạm vi kiểm chứng chưa rõ (ưu tiên cao)

Vị trí: Sect. 4.6 dòng 807–823; sanity check dòng 618–624.
Câu chọn w=.7 vì cả sáu metric cải thiện ở mọi seed cần nêu rõ điều kiện
ROUTING-ONLY và đối chứng tại thời điểm chọn w, kèm bảng/log gốc.
Không được hiểu là full method: IMR seed43 có Forgetting +0.3796 và
Backward -0.4550 so với baseline sau sửa.
Chưa có raw logs cho tiêu chí lịch sử này trong lượt kiểm tra; không kết luận
tiêu chí sai, nhưng hiện chưa đủ truy vết.

Grid seed42 có max 75.61 tại w=.6 so với 75.51 tại w=.7. Câu cost nothing
measurable nên đổi thành chưa phân biệt được ở phép so sánh bốn seed
(Acc@1 +0.108, CI [-0.064,+0.280]); single-seed grid tự nó không chứng minh tương đương.
Các số Acc@task đã làm tròn 80.39/79.97/79.37 cho khoảng 1.02, trong khi văn
ghi 1.01; cần kiểm lại số chưa làm tròn trước khi sửa.

Sanity check nên ghi rõ seed, dataset và tên ba metric của ví dụ
77.7914/74.0191/1.2305 (IMR42: Acc@task/Acc@1/Loss), độ chính xác log
và phạm vi stage. Identity PASS không tự chứng minh mọi kết quả lịch sử,
mọi cấu hình ablation hoặc tính đúng đắn toàn bộ evaluator.

### R5 — Bằng chứng sau sửa chưa bao phủ toàn bộ paper

| Phạm vi | Bằng chứng đã xem | Giới hạn |
|---|---|---|
| IMR seeds42–45 | Metadata, identity all stages, historical final baseline/full PASS | Intermediate Loss và một số Acc@1 thay đổi; không nói toàn bộ log không đổi |
| CIFAR seeds42–45 | Các kiểm tra trên PASS | Retention full thay đổi nhỏ như R1 |
| CUB seed42 | PASS; Acc@1 full-baseline +1.2956 pp | Không phải mean bốn seed |
| CUB seeds43–45 | Seed43 identity log bị ngắt | Chưa có đủ bằng chứng hoàn tất sau sửa |
| 20-cell grid, ablation, weight sweep, diagnostics | Văn bản lịch sử và công cụ phân tích còn lưu | Chưa tái đối chiếu toàn bộ raw logs/provenance sau sửa |

Không được coi log CUB43 dở là kết quả hoàn chỉnh. Không có bằng chứng đủ để
gán lần ngắt đó cho OOM chỉ từ các dòng kernel mà không đối chiếu thời điểm/process.
Các kết quả lịch sử chưa tái xác minh không đồng nghĩa chúng đã sai.

### R6 — Thiếu chi tiết tái lập và baseline gần phương pháp nhất

Vị trí: Setup dòng 588–615; Sect. 4.3–4.5.

- Nêu explicit Acc@1/Acc@5/Acc@task là trung bình không trọng số qua các task
  đã thấy ở final stage: evaluator lấy sum(stat_matrix)/(task_id+1).
  Không phải pooled accuracy trên toàn bộ ảnh; khác biệt đáng lưu ý với 5-Datasets.
- Bổ sung số lớp/task, nguồn split và thứ tự lớp/domain theo seed, checkpoint
  pretrained cụ thể, preprocessing, LoRA insertion/depth/scaling, lora_type,
  thông số train/CTIRD hoặc trỏ tới cấu hình tái lập cố định.
- RP fit dùng ảnh TRAIN nhưng transform evaluation; không dùng nhãn test.
  Cần mô tả rõ thay vì chỉ nói features/fit chung chung.
- Câu retrained from scratch chỉ nên chỉ adapter/TII mới khởi tạo theo seed;
  backbone vẫn pretrained và frozen.
- Bảng 20 ô và bảng ablation hiện bị thay bằng khoảng số trong văn xuôi:
  cần bổ sung bảng/supplement hay file kết quả được dẫn rõ, cùng commands,
  checkpoint/config hashes và provenance. Không chỉ ghi released code mà
  thiếu đường dẫn/commit tái lập.
- RanPAC và APER được thảo luận nhưng không có đối chứng định lượng cùng
  protocol trong Table 2. Ít nhất đưa RP-only trên chính feature/checkpoint đang
  dùng; nếu thêm RanPAC chuẩn phải phân biệt với RP head mượn LoRA của HRM-PET.
  Đây là thí nghiệm tăng sức thuyết phục, không phải kết quả được phép tự điền.
- Grid đã được khảo sát trên dữ liệu phát triển/test; Limitations hiện có
  thừa nhận selection bias. Không đổi tên thành validation độc lập nếu không có.

### R7 — Vài câu khái quát quá rộng hoặc chưa định lượng

Vị trí: Intro dòng 87–110; Method dòng 495–510; Cost dòng 568–580.

- Không phải mọi PET method đều có task-specific pool/route; CODA/LAE/InfLoRA
  trong chính bảng đối chiếu là ví dụ cần phân biệt. Thu hẹp phát biểu về họ
  phương pháp đang xây trên HRM-PET, không coi đó là định nghĩa mọi PET.
- changes no training procedure nên là không đổi huấn luyện LoRA/TII,
  vì RP vẫn cần fit trên current-task train data.
- Ridge joint-fit/order invariance là tính chất toán học khi feature map cố định,
  không phải cam kết bitwise equality của phép cộng float64 theo mọi thứ tự.
- Fixed p1 trong code là cố định chỉ số adapter khi nạp checkpoint từng task;
  lượt này chưa so sánh tensor p1/backbone giữa toàn bộ checkpoint thực trên lab.
  Nên có kiểm tra invariant này bằng CPU/hash trước khi chứng nhận rộng hơn.
- 800 MB Gram đúng theo 10^8 float64 entries. 2.4 GB là ước lượng ba ma trận
  lớn, không phải measured peak của toàn pipeline (còn model/workspace/tensor).
  283 s cần nguồn log, điều kiện đo và không được coi là overhead so baseline.
- Diagnostic 74.31 ở Sect. 2.1 khớp số routing-only lịch sử, khác baseline
  74.02; cần gắn cấu hình của diagnostic để người đọc không lẫn với main table.

## Đối chiếu công thức và code: các phần đã khớp

Phạm vi: nhánh paper mặc định, không phải mọi thử nghiệm có trong repository.

| Nội dung | Code kiểm tra | Kết luận |
|---|---|---|
| TII chọn t0; initial adapted logits r | engines/hrm_lora_wtp_and_tap_engine.py:1254 | Khớp phân biệt u và r |
| Stage1 max theo seen task, population std, w=.7 | cùng file:730,1312 | Đối số tên tii_logits nhưng call site truyền r, không phải u |
| DRM khi proposal khác t0; CRM lấy confidence/candidate từ r | cùng file:1320–1376 | Khớp; CRM không chạy task1 |
| CRM so sánh log-sum-exp, mask trước so sánh | cùng file:1340–1365 | Khớp bản đã sửa |
| Gate top-two từ ell, beta=.5, z-score và affine map-back | cùng file:783–941 | Khớp Eqs.4–8 với sharpen=1, no ramp |
| RP features fixed adapter index0, extra pass, no gradient | cùng file:1300,2549; trainers/lora_trainer.py:311 | Khớp cấu hình; train=True chọn cách index LoRA, không bật tối ưu |
| ReLU, Gaussian W seed1993, G/C float64, ridge solve | engines/random_projection_head.py:108–175,430 | Khớp Eqs.1–3 |
| Forgetting/Backward chia T-1 | engine:2109–2111 | Khớp Eqs.9–10 |
| Acc@task là DRM proposal, không final CRM adapter | engine:1372,1423 | Paper đã giải thích đúng |

Code còn comment cũ: fuse_routers gọi log-probabilities/TII; fuse_class_scores
nói mixture unit variance. Hành vi code không theo các comment đó; paper hiện
mô tả đúng hơn comment. Nên dọn COMMENT riêng, không đổi công thức thực thi.
Không lấy lượt review này làm chứng nhận toàn bộ train/inference/options:
chưa chạy torch/GPU tests hay retrain trên máy hiện tại.

## References

Kiểm tra cấu trúc: 21 bibitems, 21 cited keys; không thiếu citation key,
không bibitem bỏ quên, không ref/label chưa định nghĩa hoặc label trùng.
Đã tra sự tồn tại và thông tin định danh của 21 tài liệu từ nguồn gốc dưới đây.
Không thấy reference bịa hoặc lỗi tên/venue rõ ràng trong các mục đã tra.
Không đồng nghĩa đã đọc và xác minh mọi khẳng định trong toàn văn cả 21 bài.

| Ref | Nguồn đối chiếu | Nhận xét |
|---|---|---|
| 1 HRM-PET | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a978bdfeb195e4a574c0def98806346a-Abstract-Conference.html) | Tên, tác giả, năm khớp |
| 2 RanPAC | [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2023/file/2793dc35e14003dd367684d93d236847-Paper-Conference.pdf) | Khớp; projection/second-order head phải tiếp tục được ghi công |
| 3 HiDe-Prompt | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d9f8b5abc8e0926539ecbb492af7b2f1-Abstract-Conference.html) | Năm 2023 trong bản ta đúng; không chép năm 2024 bị ghi ở bib HRM-PET |
| 4 L2P | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html) | Khớp; có thể bổ sung pp.139–149 |
| 5 DualPrompt | [ECCV paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860617.pdf) | Khớp nhận diện; chuẩn hóa đầy đủ metadata |
| 6 LoRA | [Author preprint](https://arxiv.org/abs/2106.09685) | Nhận diện khớp; OpenReview trực tiếp bị browser challenge trong lượt này |
| 7 CODA-Prompt | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) | Khớp |
| 8 APER | [Springer IJCV](https://link.springer.com/article/10.1007/s11263-024-02218-0) | 133,1012–1032(2025) đúng; online 2024 khác issue 2025; không nhầm APER với APER† dùng exemplar |
| 9 Survey | [Author preprint](https://arxiv.org/abs/2010.15277) | Tên/tác giả khớp; pagination TPAMI chưa xác minh lại trực tiếp từ publisher |
| 10 iCaRL | [CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html) | Khớp |
| 11 BiC | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.html) | Khớp |
| 12 WA | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_Maintaining_Discrimination_and_Fairness_in_Class_Incremental_Learning_CVPR_2020_paper.html) | Khớp |
| 13 LUCIR | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html) | Khớp |
| 14 MoTE | [Elsevier](https://www.sciencedirect.com/science/article/pii/S095070512500841X) | Còn thiếu volume324/article113795; DOI10.1016/j.knosys.2025.113795 |
| 15 DLEPEM | [MDPI](https://www.mdpi.com/2076-3417/16/12/6153) | Khớp tác giả,16(12),6153(2026); có thật, xuất bản17/06/2026 |
| 16 S-Prompts | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/25886d7a7cf4e33fd44072a0cd81bf30-Abstract-Conference.html) | Khớp; chú thích ++ là variant trong benchmark HRM-PET |
| 17 LAE | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_A_Unified_Continual_Learning_Framework_with_General_Parameter-Efficient_Tuning_ICCV_2023_paper.html) | Khớp |
| 18 CPrompt | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Consistent_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2024_paper.html) | Khớp |
| 19 InfLoRA | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html) | Khớp |
| 20 ACIL | [NeurIPS 2022](https://proceedings.neurips.cc/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract.html) | Khớp |
| 21 DS-AL | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29670) | Khớp; bổ sung38(15),17237–17244, DOI10.1609/aaai.v38i15.29670 |

Đã đối chiếu đủ 32 giá trị trích dẫn (8 phương pháp × 4 datasets) ở Table 2
với Sup-21K/Table 1 trong [HRM-PET gốc](https://proceedings.neurips.cc/paper_files/paper/2025/file/a978bdfeb195e4a574c0def98806346a-Paper-Conference.pdf):
không thấy sai chép số. Hai hàng tự chạy không được coi là đã chạy lại cùng
protocol với tám hàng trích dẫn; caption hiện đã nêu khác seed/implementation.
Nên chỉ bold best trong từng nhóm, và ghi rõ HiDe-LoRA là bản dùng LoRA của
HiDe-Prompt trong benchmark đó, tránh ngầm tuyên bố thắng có kiểm soát.

Chưa có citation trực tiếp cho datasets, ViT và các pretrained settings
MoCo-v3/iBOT/DINO trong Setup. Bổ sung nguồn đúng checkpoint/dataset đã dùng,
không chỉ citation gián tiếp qua HRM-PET. Hoàn thiện DOI/pages/venue thống nhất;
References không tính vào giới hạn nội dung.

## PDF, trình bày và yêu cầu hội nghị

Đã render và xem lần lượt toàn bộ14 trang, không chỉ trang hình.
Đọc log build đang có và kiểm tra font/cross-reference; không recompile hay
thay PDF trong lượt review read-only này.

- 12 trang nội dung, gồm acknowledgment ở cuối trang12; References ở13–14.
  Đúng giới hạn12 trang KHÔNG tính references và yêu cầu hiện tên tác giả của
  [SOICT2026](https://soict.org/submission/paper-submission/) tại ngày kiểm tra.
- Hội nghị yêu cầu Springer CCIS; bản này dùng macro llncs thuộc bộ template
  Springer Computer Science. Không nhầm tên file lncs với một venue khác;
  đối chiếu sample CCIS cuối cùng trước submission/camera-ready, không sửa lề
  hoặc font size để lách giới hạn.
- Đủ bố cục: Intro, Related Work, Method (problem statement ở đầu),
  Experiments (main/prior/grid/ablation/weights/limits), Conclusion.
  Baseline HRM-PET nằm ở Related Work; Method chỉ giải thích phần ghép thêm.
- Không thấy ảnh mất, công thức bị cắt, glyph lỗi, chữ/nhãn chồng rõ rệt.
  Các font trong PDF đều được embed; không có missing glyph/undefined
  reference/overfull/underfull trong log hiện tại. Còn warning amsmath vec,
  không phải lỗi làm hỏng PDF.
- Hình ở trang5 hiện không còn lỗi neo mũi tên và nhãn đè mà người dùng đã chỉ.
  Tuy nhiên nhãn chính khoảng7pt, vài subscript khoảng5pt; nên tăng cỡ khi
  tái thiết kế. Caption còn gánh nhiều logic DRM/CRM; độc giả mới cần đọc cả
  caption/Method, không nên gọi hình hoàn toàn tự giải thích.
- Table1 khá dày; Loss CIFAR hiển thị -0.000 và SD0.00 che mất mức thay đổi.
  Dùng đủ chữ số cho Loss/delta, đơn vị pp rõ, không suy diễn negative zero.
- Table3 ở trang12, cách lời dẫn ở11; chưa phải lỗi nhưng có thể đặt gần hơn.
  Ref21 đứng riêng ở trang14: điểm dàn trang nên chỉnh, không vượt limit.
- Chưa thấy lỗi chính tả lớn qua đọc toàn văn. Nên giảm lối viết hội thoại
  (bought, costs nothing measurable, loses to the gate), ưu tiên đối chứng rõ.
- Disclosure of Interests mới là comment trong source, chưa hiện trong PDF.
  Hoàn thiện theo bộ hướng dẫn camera-ready; không tự điền tuyên bố thay tác giả.
- Credit ảnh có trong caption; lượt này chưa kiểm độc lập chain nguồn/license
  của chính file ảnh cục bộ. Cần giữ source URL/license record khi đóng bản nộp.

## Các kiểm tra thực hiện và phần chưa thực hiện

- Đọc nguồn paper, code RP/routing/gate/metric/dataloader và tài liệu hệ thống;
  ưu tiên code thực thi hơn comment hoặc bàn giao lịch sử.
- Đọc lại output verifier người dùng cung cấp, tính độc lập sample SD và CI
  retention CIFAR từ bốn paired deltas.
- 26 tests của test_verification_tools.py chạy bằng unittest trên CPU: PASS.
  Đây là tests công cụ kiểm chứng, KHÔNG phải full torch/GPU test suite.
- Không chạy lab/GPU, không tái huấn luyện, không so tensor checkpoint thực,
  không xác nhận mọi con số lịch sử từ raw logs, không test live Overleaf.
- Source/PDF/ảnh được giữ nguyên; review nêu lỗi để sửa ở lượt kế tiếp.

## Dữ liệu cần tác giả/Claude cung cấp, ưu tiên tận dụng log đã có

1. Bốn final rows baseline/full CIFAR sau maskfix, đủ sáu metric để cập nhật
   cả absolute mean/SD, không chỉ paired delta.
2. Khi có thể: trạng thái/log hoàn tất CUB43–45. Chỉ chạy phần thiếu khi lab rảnh
   và được đồng ý, không mặc định chạy lại cả hệ thống.
3. CSV/log cho20 cells và các arms routing-only, class-only, class+gate,
   full; đủ seeds/config/commit/provenance để phân biệt lịch sử với sau sửa.
4. Bằng chứng tiêu chí chọn w=.7 trước Stage2: từng seed, sáu metric,
   đối chứng cụ thể và split đã dùng chọn hyperparameter.
5. Run manifest/checkpoint URLs/preprocessing/class order cùng invariant
   backbone+p1 qua các task; việc so tensor/hash có thể làm trên CPU.
6. Nguồn log283s, measured memory nếu đã có, RP-only cùng protocol.
   Chưa biết thì ghi chưa đo, không tạo số.

Checkpoint này được lưu LOCAL; không push remote trong lượt này.
