"""Per-task layer-statistics router.

Adapted from the layer-wise distribution constraint of PMI-CFS (Tong et al.,
"Model Inversion with Layer-Specific Modeling and Alignment for Data-Free
Continual Learning", Appendix C.3, Eq. 21/26). Because ViTs carry no
BatchNorm, that work stores the mean and standard deviation of each
transformer block's activations over real data and uses

    KL( N(mu_hat, sigma_hat) || N(mu, sigma) )

as a prior-distribution constraint while synthesising images. The same
quantity measures something else that is useful here: how well a sample's
activations match the distribution a given task was trained on.

So we store those statistics **per task** and use the divergence as a routing
score. It is a third source of evidence, independent of both routers already in
use -- TII reads output-level prompt energy, the RP head reads second-order
class prototypes at the final feature, and this reads the distribution of
activations across all blocks. Only per-task mean/std vectors are kept
(12 x 768 x 2 floats per task), so the exemplar-free protocol is untouched, and
scoring needs no extra adapter call: one forward yields every layer at once.
"""

import torch


_state = {'hooks': [], 'captured': [], 'task_stats': {}}


def reset_layer_stats():
    remove_hooks()
    _state['task_stats'] = {}


def _blocks(model):
    module = model.module if hasattr(model, 'module') else model
    return module.blocks


def install_hooks(model):
    """Capture every transformer block's output during the next forward."""
    remove_hooks()

    def make_hook(index):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            _state['captured'].append((index, tensor.detach()))
        return hook

    for index, block in enumerate(_blocks(model)):
        _state['hooks'].append(block.register_forward_hook(make_hook(index)))


def remove_hooks():
    for handle in _state['hooks']:
        handle.remove()
    _state['hooks'] = []
    _state['captured'] = []


def _per_sample_stats():
    """Turn captured block outputs into per-sample mean/std over tokens.

    Returns tensors shaped [batch, layers, dim]. Statistics are taken across
    the token axis, so each sample gets its own summary of every block.
    """
    if not _state['captured']:
        raise RuntimeError('No block activations captured; install hooks first')
    ordered = sorted(_state['captured'], key=lambda item: item[0])
    means, stds = [], []
    for _, tensor in ordered:
        tensor = tensor.float()
        means.append(tensor.mean(dim=1))
        stds.append(tensor.std(dim=1).clamp_min(1e-5))
    _state['captured'] = []
    return torch.stack(means, dim=1), torch.stack(stds, dim=1)


@torch.no_grad()
def accumulate_task_stats(task_id):
    """Fold the current batch into the running statistics for one task."""
    means, stds = _per_sample_stats()
    key = int(task_id)
    total_mean, total_std, count = _state['task_stats'].get(
        key, (None, None, 0))
    batch_mean = means.sum(dim=0)
    batch_std = stds.sum(dim=0)
    if total_mean is None:
        total_mean, total_std = batch_mean, batch_std
    else:
        total_mean = total_mean + batch_mean
        total_std = total_std + batch_std
    _state['task_stats'][key] = (
        total_mean, total_std, count + means.shape[0])


def task_stats_ready(seen_task_count):
    return all(int(t) in _state['task_stats'] for t in range(seen_task_count))


@torch.no_grad()
def layer_stat_scores(seen_task_count, device):
    """Score each seen task by how well this batch matches its statistics.

    Uses the closed-form KL between diagonal Gaussians, summed over layers and
    dimensions, negated so that higher is better and it can be mixed with the
    other routers' scores.
    """
    means, stds = _per_sample_stats()
    batch = means.shape[0]
    scores = torch.full(
        (batch, seen_task_count), float('-inf'), device=device,
        dtype=torch.float32)
    variance = stds.pow(2)
    for task in range(seen_task_count):
        total_mean, total_std, count = _state['task_stats'][int(task)]
        ref_mean = (total_mean / max(1, count)).to(device).unsqueeze(0)
        ref_std = (total_std / max(1, count)).to(device).clamp_min(
            1e-5).unsqueeze(0)
        ref_var = ref_std.pow(2)
        kl = (torch.log(ref_std / stds)
              + (variance + (means - ref_mean).pow(2)) / (2.0 * ref_var)
              - 0.5)
        scores[:, task] = -kl.sum(dim=(1, 2))
    return scores
