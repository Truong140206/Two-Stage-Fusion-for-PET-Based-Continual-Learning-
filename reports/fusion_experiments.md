# Opt-in fusion probes - 2026-09-11

## Archived - restored paper system

The user stopped this direction and requested restoration. Experimental source,
flags, runner and tests have been removed from the active tree; they remain
recoverable in commit b9d8def8aec715c6fbae8920c39420a1e912306e. Do not run the
historical commands below on the restored main branch. No lab logs were deleted.
Code/configs are restored byte-for-byte in Git to system-baseline-2026-09-11.
Paper LaTeX/PDF were never modified during the probes.

Final exploratory route_class_zmax minus paper-reference Acc@1 (four seeds):
ImageNet-R +.161675 CI[.104251,.219099]; CIFAR -.012500
CI[-.060015,.035015]; CUB +.030550 CI[-.028169,.089269]. All six retention
intervals contain zero. Lab127tests passed before rollback. All12 references
matched their original ten-stage logs, and both dataset batches completed.
Only the original paper method remains active; no per-dataset winner selection.

The rest of this document records the historical experimental implementation.

## Recovery point

`system-baseline-2026-09-11` is an annotated Git tag at
`4989ab866b83bafe04ed7aa2d23ec4f7235e22c4`, pushed to origin before edits.
It preserves tracked code and the canonical LaTeX/PDF, not ignored lab datasets,
checkpoints, or untracked scratch files. Those files are left untouched.
No duplicate project directory or paper was created.

To inspect a saved file without changing the working tree:

```bash
git show system-baseline-2026-09-11:engines/hrm_lora_wtp_and_tap_engine.py
```

To run the saved system, first finish all running jobs and preserve any local
tracked changes, then `git switch --detach system-baseline-2026-09-11` from a
clean tracked working tree. Do not reset, clean, or overwrite checkpoints.
Record the current branch before switching so it can be selected again later.

## Fixed exploratory probes

All reuse the same LoRA/TII checkpoints and refit only RP statistics from
current-task training images, like the reference. No backbone/adapter training.
No parameter is fitted or chosen using test labels. These are development-set
experiments, not an untouched validation study. Do not promote a test-set winner
to a prespecified method or claim significance from seed42 alone.

| Variant | Only change from the full reference |
|---|---|
| reference | Original command and default math |
| route_class_zmax | Standardize seen CLASS scores before task max, instead of task max then task z-score |
| gate_floor025 | Margin gate becomes 0.25 + 0.75 * old_gate; at beta=0.5 the RP share is 0.125 to 0.5 |
| rp5000 | M=5000 instead of 10000; lambda=10000 and everything else unchanged |

`class_zmax` avoids forced +/-1 task scores with two multi-class tasks, but is
not calibrated probability fusion and does not preserve absolute cross-sample
confidence. It is a parameter-free first probe, not the learned adaptive router
discussed as a longer-term option. Max aggregation remains unchanged. A success
on the constructed two-task example is not evidence of dataset improvement.

The floor probe does not yet use RP confidence. It isolates whether the current
one-sided gate suppresses useful RP contributions. Floor0 is the paper;
floor1 is ungated class fusion. Nonzero floor requires the margin gate.

M=5000 quarters Gram storage mathematically, not measured total VRAM. There is
no wall-time or accuracy guarantee. Changing M also changes features and the
regularization tradeoff; this is a fixed-lambda dimensionality ablation, not a
claim of an optimally tuned low-dimensional head.

## Runner guarantees

`tools/try_fusion_improvements.py` defaults to ImageNet-R seed42: four
sequential evaluations (reference + three probes), not a new 48-run grid.
`--variants` can restrict probes. `--run` is required for new GPU execution.
Free-memory preflight is 16GiB, not a GPU reservation; never kill another job.

Before probes, verify original checkpoint hashes and exact legacy log metadata,
then compare ALL ten reference stages against the saved pre-change full log.
Any mismatch stops that seed before its probes. Historical source SHA remains
6747fed54a6f9679632bfdbbed1d3ebf2832dc801cf950b6164460f5a8858da9.
New evaluator/driver hashes are saved in exclusive new logs. No old metadata is
edited to bypass source mismatch. The old finish_soict_checks command cannot
certify old logs under changed source; use the saved tag for that workflow.

The new runner uses tags such as fusion_probe_v1, not soict_final_v1.
Existing incomplete or provenance-mismatched logs cause a stop; retain them
and choose a new tag. Do not update code during a running batch.
Without --run the runner only reads/checks completed logs (CPU/disk).

## Verification before GPU

Local Windows: 52 dependency-free tests passed; five real-tensor tests skipped
because PyTorch is absent. Real-tensor tests load the actual old/new fusion
functions and check bit-exact default outputs over ten task counts, mask/scale
properties, the two-task counterexample, and gate endpoints. Run the full lab
pytest suite before evaluation. No new dataset accuracy has been measured yet.

The paper/PDF and all previously reported scores remain unchanged.
