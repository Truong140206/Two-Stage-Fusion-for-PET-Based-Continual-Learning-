# Official RanPAC: CIFAR-100 and ImageNet-A

This auxiliary launcher runs only the two official RanPAC ID7 configurations
needed for the sparse exploratory comparison:

- `cifar100` maps to upstream `cifar224` and uses the official Adapter with
  ImageNet-21K pretraining, 10 classes per task, and class-order seed 1993.
- `ima` maps to upstream `imageneta` and uses the official SSF configuration,
  20 classes per task, and class-order seed 1993.

RanPAC has no matching implementation of the paper's heterogeneous
5-Datasets stream, so the launcher deliberately does not offer that dataset.
Its table cell must remain unreported rather than being approximated.

## Reproducibility and safety

- Upstream source is pinned to commit
  `cf4b301d18b0c27db030f4371b72b768005ae58a`.
- The exact ID7 CSV row is checked before every run.
- The original upstream `main.py` is invoked without modifying RanPAC code.
- A private Python 3.9 environment and private model cache are reused.
- Dataset contents, class order, runtime versions, pretrained source, and
  pretrained checksum are recorded before GPU execution.
- The shared paper GPU lock is honored. Existing incomplete and complete run
  directories are preserved, never overwritten.
- Saved predictions are used to recompute final task-mean Top-1, pooled Top-1,
  forgetting, and backward transfer. The recomputed values must agree with the
  original RanPAC CSV.

## Important interpretation rule

The two official ID7 rows are dataset-specific and do not use one common
backbone/PETL protocol: CIFAR-100 uses Adapter/In21K whereas ImageNet-A uses
SSF. Therefore these results must be labelled as official dataset-specific
RanPAC configurations. They must not be described as one matched AugReg run or
as a controlled backbone-matched comparison with HRM-PET/Full.

## Host command

```bash
python -B -u tools/run_ranpac_original_extra.py \
  --output-root /path/to/output \
  --data-root /path/to/datasets \
  --datasets cifar100 ima \
  --tag ranpac_original_extra_v1 \
  --cpu-threads 4 \
  --max-minutes 90 \
  --prepare \
  --run
```

The time limit applies separately to each dataset. Completion is indicated by
`RANPAC_EXTRA_COMPLETE=cifar100:10/10`,
`RANPAC_EXTRA_COMPLETE=ima:10/10`, and finally
`RANPAC_EXTRA_ALL_COMPLETE=2/2`.
