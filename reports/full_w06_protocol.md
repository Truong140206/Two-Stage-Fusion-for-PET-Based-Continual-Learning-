# Fixed full-w=0.6 comparison (2026-09-14)

## Completed evidence

The batch completed with marker FULL_W06_MAIN_COMPLETE=12/12; PAPER_UNCHANGED:
four verified ImageNet-R runs were reused and eight CIFAR-100/CUB-200
evaluations were newly completed. The eight new evaluations took 2441.137
seconds in total (about 40.7 minutes). All twelve checkpoint protocol and
fixed-feature checks passed. Summary: OUT/full_w06_main_v1__summary.json.

Full w=.6 versus Full w=.7, paired over seeds 42--45:

| Dataset | w=.6 Acc@1 | w=.7 Acc@1 | paired difference [95% t interval] |
|---|---:|---:|---:|
| ImageNet-R | 75.288 | 75.180 | +0.108 [-0.064, +0.280] |
| CIFAR-100 | 90.455 | 90.473 | -0.017 [-0.053, +0.018] |
| CUB-200 | 87.900 | 87.878 | +0.022 [-0.024, +0.067] |

No primary Acc@1 interval excludes zero and the direction is not consistent.
Acc@task rises by +.332 on ImageNet-R and +.137 on CUB-200, but this does
not translate into a reliable class-accuracy improvement. Forgetting and
Backward differences are near zero and every interval includes zero. Loss is
slightly worse for w=.6 on CIFAR-100 (+.00205) and CUB-200 (+.00075), with
their paired intervals above zero.

Decision: retain w=.7 as the Full reporting configuration. Do not migrate the
paper, backbone grid, ablations, diagnostics or AugReg comparison to w=.6.
The existing sensitivity paragraph already reports w=.6 as the observed
ImageNet-R grid maximum and explicitly says it is not a selected optimum.
This completed multi-dataset check strengthens the decision not to select w=.6
from a small test-set gain. No further w=.6 evaluation is required before
submission.

Paper checkpoint: 7c360a8, unchanged until new evidence is received and audited.
No automatic winner selection or replacement of manuscript cells.

Phase 1: Sup-21K, IMR/CIFAR100/CUB200, seeds 42-45, beta=.5, margin gate.
Read all 12 verified baseline/w=.7 pairs, reuse four verified IMR w=.6 runs
with the original driver's pinned hash. Run only eight missing CIFAR/CUB
w=.6 evaluations. Full method, all ten stages, not training; the analytic RP
head is refit by the established evaluation path. Old driver files stay intact.

Tool: tools/check_full_w06.py. Without --run: preflight/reuse only.
With --run: sequential evaluation, shared flock, 16GiB free GPU check,
1GiB free disk before each new small log. No new checkpoints, environments or
datasets. Requires existing trusted checkpoints and the paper's source digest.
Incomplete/provenance-mismatched logs cause a stop; never edit their metadata.
Summary full_w06_main_v1__summary.json contains per-seed final Acc@1,
Acc@task, Acc@5, loss, forgetting and backward, mean/sample SD and paired
95% t intervals. Runtime per new evaluation is printed in the launcher log.

IMR's existing w=.6 minus .7 mean Acc@1 is +.108 percentage points with a
95% paired interval including zero. Selection using these test results is
exploratory/test-informed, not independent validation. Report decreases too.
Do not choose a different w for each dataset after reading its test score.

Phase 1 is NOT the complete migration of every manuscript experiment.
If the common setting changes, reconcile the backbone grid (20 full cells,
overlapping Sup IMR/CIFAR cells reused), routing-dependent ungated ablations,
classifier diagnostics, the AugReg comparison, and derived figures/text.
Baseline, RP-only and class-only controls do not depend on w=.6 versus .7 and
need not be retrained/rerun when their provenance still matches.
Keep the old setting's evidence and disclose the configuration change.

Runtime: the historical IMR evaluation record is ~283 seconds, not a promise
for CIFAR/CUB or a controlled timing benchmark. Budget hours rather than
retraining days; measure the first new run before extrapolating.
The official SOICT website checked September14 displays September20 for
full-paper submission after the original September16 date:
https://soict.org/submission/
