# RanPAC extension for 5-Datasets

RanPAC does not publish a 5-Datasets configuration. This run is therefore an
exploratory protocol extension, not an official reproduction. It may populate
the table only with an explicit extension marker or footnote.

The extension keeps the pinned upstream RanPAC Learner unchanged and adds a
data-manager adapter with the paper's fixed task order:

1. SVHN
2. MNIST
3. CIFAR-10
4. NotMNIST
5. Fashion-MNIST

Each dataset is a disjoint ten-class task, giving 50 class-incremental labels.
The task order is fixed; seed 1993 does not permute domains. Input transforms
match the existing 5-Datasets loader: generic transforms for SVHN/MNIST and
the CIFAR transform for CIFAR-10/NotMNIST/Fashion-MNIST.

The model-side settings extend RanPAC's official ImageNet-R ID7 recipe: the
AugReg ImageNet-21K-to-1K ViT-B/16 SSF backbone, 20 PETL epochs on the first
task only, random-projection dimension 10,000, batch size 48, body learning
rate 0.01, and closed-form updates thereafter. Torch seed is 1. SciPy 1.10.1
is installed only in the private RanPAC environment because torchvision's
SVHN loader requires it.

The runner records hashes, per-class counts, runtime versions, checkpoint
source/checksum, task order, and the pinned upstream revision. It recomputes
task-mean Top-1, pooled Top-1, forgetting, and backward transfer from saved
predictions. Existing outputs are never overwritten and the shared paper GPU
lock is respected.

Expected runtime is substantially longer than ImageNet-R because the first
task contains the full SVHN training set. The default timeout is eight hours.

```bash
python -B -u tools/run_ranpac_fivedatasets.py \
  --output-root /path/to/output \
  --data-root /path/to/datasets \
  --tag ranpac_fivedatasets_extension_v1 \
  --cpu-threads 4 \
  --max-minutes 480 \
  --prepare \
  --run
```
