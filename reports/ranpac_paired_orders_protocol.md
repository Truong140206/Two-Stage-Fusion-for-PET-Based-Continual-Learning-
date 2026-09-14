# Fixed additional class-order replications

Prepared 2026-09-14. Not yet run on the4090. No paper results are replaced.
Entry: tools/run_ranpac_pairs.py (read-only plan unless --run).
Worker: tools/run_ranpac_pair_worker.py.

## Question and fixed protocol

Does the observed Full-vs-RanPAC / Full-vs-HRM-PET gain persist under two
additional class orders? Orders are fixed before seeing their results:
NumPy RandomState1994 and1995 permutations of200 classes,20 classes/task,
10 tasks. Retain the original1993 comparison without rerunning or selecting it.
Training torch seed remains1 in every run, as in upstream RanPAC. This is
a three-class-order comparison, NOT three independent training-seed replicates.
The first order was already observed, so the expanded comparison is exploratory.
No weight grid, best-seed selection, or early stopping on test results.

Each new order runs sequentially:

1. RanPAC official main.py -i7 -d imagenetr in its existing private Python3.9
   environment. Only CSV class-order seed differs from pinned official ID7.
   Write a run-local one-row CSV; never edit upstream's CSV/source/trainer.
   Native trainer keeps torch seed1;20 SSF epochs on task1 only; all10 tasks.
2. Our TII20 epochs/task, LoRA50 epochs/task with the already verified arguments.
3. Conventional HRM-PET and Full evaluations of the same fresh checkpoints.
   Full keeps w0.7,beta0.5,RP10000/ReLU/lambda10000,margin gate.

Pretrained NPZ SHA256:
ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124.
Use the existing cache, environment, and hashed24000/6000 ImageNet-R split.
Check live images before native model creation and again before our training.
Our loaders and every checkpoint are checked against frozen-backbone hashes.
Retain method-specific augmentations, runtime and training budgets; no claim
that matching pretrained/order makes every aspect of the protocols identical.
Dataset split still is not authenticated against the authors' original archive.

## Preservation and resource rules

All new outputs: OUT/ranpac_pairs_v1/order1994/{ranpac,ours} and order1995.
Old drivers run_ranpac_original.py and try_ranpac_pretrained.py are byte-unchanged.
No dataset copies, new environments, installations, checkpoint deletion or
changes to paper/core model code. Use shared GPU lock for the sequential batch,
check for other compute jobs before every phase; never stop another job.
Children inherit the lock; timeout stops only their own process group.

Initial guard:24GiB free for two pending pairs,12GiB if one pair is complete.
This is a conservative reservation, not a measurement of final disk use.
Last lab report:7.2GB free, insufficient. Launcher fails before creating new
experiment directories in that state. Do not lower the threshold to force it.
Atomic per-checkpoint saves retain the old reserve and no-overwrite checks.
No automatic cleanup or incomplete-phase restart; incomplete outputs are
preserved for diagnosis. Completed phases resume only after verifying hashes.
GPU-busy failures before new phase creation can be retried with the same command.

Time caps:60 minutes per native run;360 minutes per new HRM-PET/eval sequence.
CPU preflight is separate. Caps do not reduce epochs, task count or sample count.
Completed phase artifacts and immutable plan must still match on every retry.
Native elapsed time here includes runtime/data verification in its child;
do not report it as a directly comparable pure-training efficiency benchmark.

## Completion and reporting

At completion, OUT/ranpac_pairs_v1/summary.json includes all three orders:
RanPAC, HRM-PET, Full Acc@1(task-balanced), forgetting and backward transfer;
Full-minus-RanPAC and Full-minus-HRM-PET paired differences; sample mean/SD
and descriptive t intervals(df2). These are not robust significance evidence
from three independently trained seeds, especially with only three orders.
Lower F is better; higher B is better. Keep all results, including regressions.
No automatic paper rewrite or claims of universal superiority.

Native metrics are recomputed from official saved predictions and checked
against official CSVs. Our aggregate is checked against full evaluation logs.
Initial order1993: RanPAC77.443699, HRM-PET77.4192, Full79.1355;
F4.748434/4.5422/4.0408; B-4.748434/-4.5422/-3.9806.
These are prior observed values, not results of the two new orders.
Completion marker: RANPAC_PAIRS_COMPLETE=3/3 INCLUDING_EXISTING_ORDER1993.

## Lab command

Use the existing paper Python, not the private RanPAC Python. Read-only plan:

    "$PY" -B tools/run_ranpac_pairs.py --output-root "$OUT" --data-root "$DATA"

After sufficient disk is available and GPU is free, add --run and run under
nohup. Default order list is fixed; no --seeds fragments need to be pasted.
Both old experiments remain available using their original launchers.
CPU tests: tests.test_ranpac_pairs plus the three existing RanPAC test modules.
CUDA execution remains a lab-side verification, not covered by local CPU tests.

Local regression:118 tests across the paired runner, the three old RanPAC
suites and four paper-verification suites;117pass/1POSIX-only skip on Windows.
Includes native private-entry/local-CSV orchestration, source preservation,
low-disk no-write, immutable plans, stale-artifact rejection, unchanged training
arguments, paired aggregation and rejection of mismatched orders/logs.
