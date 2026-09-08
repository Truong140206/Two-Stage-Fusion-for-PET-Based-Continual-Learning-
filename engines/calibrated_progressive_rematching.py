import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from engines.distilled_task_router import _class_stratified_split
from engines.progressive_rematching import _tii_task_prior

_HALTING_GATES = {}


def set_progressive_halting_gates(gates):
    global _HALTING_GATES
    _HALTING_GATES = gates or {}


def get_progressive_halting_gates():
    return _HALTING_GATES


def _finalize_partial_logits(logits, excluded_margin=20.0):
    finite = torch.isfinite(logits)
    row_min = logits.masked_fill(~finite, float('inf')).min(
        dim=1, keepdim=True).values
    return torch.where(finite, logits, row_min - float(excluded_margin))


def _run_lora_rank_batch(model, inputs, task_ids):
    """Evaluate several candidate LoRAs in one batched model call."""
    if task_ids.ndim != 2:
        raise ValueError('task_ids must have shape [batch, rank_count]')
    rank_count = task_ids.shape[1]
    if rank_count == 1:
        return model(inputs, task_id=task_ids[:, 0])['logits'].unsqueeze(1)
    expanded_inputs = inputs.repeat_interleave(rank_count, dim=0)
    flat_task_ids = task_ids.reshape(-1)
    logits = model(expanded_inputs, task_id=flat_task_ids)['logits']
    return logits.reshape(inputs.shape[0], rank_count, -1)


class HaltingGate(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(1)


def _stage_features(tii_prior, sorted_tii_scores, task_evidence,
                    class_margins, boundary):
    evaluated = task_evidence[:, :boundary]
    top = torch.topk(evaluated, k=min(2, boundary), dim=1)
    best_rank = top.indices[:, 0]
    best_evidence = top.values[:, 0]
    adapter_margin = (
        best_evidence - top.values[:, 1]
        if boundary > 1 else torch.zeros_like(best_evidence))
    best_class_margin = class_margins[:, :boundary].gather(
        1, best_rank.unsqueeze(1)).squeeze(1)
    selected_tii = sorted_tii_scores.gather(
        1, best_rank.unsqueeze(1)).squeeze(1)
    unseen_tii_margin = (
        selected_tii - sorted_tii_scores[:, boundary]
        if boundary < sorted_tii_scores.shape[1]
        else torch.full_like(selected_tii, 10.0))
    tii_probability = F.softmax(tii_prior, dim=1)
    tii_entropy = -(tii_probability * tii_probability.clamp_min(1e-8).log()).sum(1)
    adapter_probability = F.softmax(evaluated, dim=1)
    adapter_entropy = -(
        adapter_probability * adapter_probability.clamp_min(1e-8).log()).sum(1)
    return torch.stack([
        best_evidence,
        adapter_margin,
        best_class_margin,
        selected_tii,
        unseen_tii_margin,
        tii_entropy,
        adapter_entropy,
        best_rank.float() / max(1, boundary - 1),
    ], dim=1)


def _choose_precision_threshold(probabilities, labels, target_precision,
                                minimum_coverage):
    best = None
    for threshold in torch.unique(probabilities).sort(descending=True).values.tolist():
        selected = probabilities >= float(threshold)
        coverage = float(selected.float().mean().item())
        if coverage < minimum_coverage or not bool(selected.any()):
            continue
        precision = float(labels[selected].float().mean().item())
        if precision >= target_precision - 1e-6 and (
                best is None or coverage > best[2]):
            best = (float(threshold), precision, coverage)
    return best if best is not None else (None, 0.0, 0.0)


def _fit_gate(features, labels, class_targets, args, device, seed_offset,
              target_precision=None):
    train_index, validation_index = _class_stratified_split(
        class_targets,
        float(getattr(args, 'progressive_gate_validation_ratio', 0.25)),
        int(getattr(args, 'seed', 42)) + seed_offset)
    generator = torch.Generator().manual_seed(
        int(getattr(args, 'seed', 42)) + seed_offset + 1000)
    validation_index = validation_index[torch.randperm(
        validation_index.numel(), generator=generator)]
    calibration_count = max(1, validation_index.numel() // 2)
    calibration_count = min(calibration_count, validation_index.numel() - 1)
    calibration_index = validation_index[:calibration_count]
    report_index = validation_index[calibration_count:]

    mean = features.index_select(0, train_index).mean(0, keepdim=True)
    std = features.index_select(0, train_index).std(
        0, keepdim=True, unbiased=False).clamp_min(1e-6)
    features = ((features - mean) / std).to(device)
    labels = labels.float().to(device)
    train_index = train_index.to(device)
    calibration_index = calibration_index.to(device)
    report_index = report_index.to(device)

    gate = HaltingGate(
        features.shape[1],
        max(8, int(getattr(args, 'progressive_gate_hidden_dim', 64))),
        min(0.9, max(0.0, float(getattr(args, 'progressive_gate_dropout', 0.1)))),
    ).to(device)
    train_labels = labels.index_select(0, train_index)
    positives = train_labels.sum().clamp_min(1.0)
    pos_weight = ((train_labels.numel() - positives) / positives).clamp(0.25, 4.0)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(getattr(args, 'progressive_gate_lr', 0.001)),
        weight_decay=float(getattr(args, 'progressive_gate_weight_decay', 0.01)))
    batch_size = max(16, int(getattr(args, 'progressive_gate_batch_size', 256)))
    patience = max(1, int(getattr(args, 'progressive_gate_patience', 10)))
    best_loss = float('inf')
    best_state = copy.deepcopy(gate.state_dict())
    stale = 0
    for _ in range(max(1, int(getattr(args, 'progressive_gate_epochs', 60)))):
        gate.train()
        order = train_index[torch.randperm(train_index.numel(), device=device)]
        for start in range(0, order.numel(), batch_size):
            index = order[start:start + batch_size]
            loss = F.binary_cross_entropy_with_logits(
                gate(features.index_select(0, index)),
                labels.index_select(0, index), pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        gate.eval()
        with torch.no_grad():
            validation_loss = F.binary_cross_entropy_with_logits(
                gate(features.index_select(0, calibration_index)),
                labels.index_select(0, calibration_index)).item()
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(gate.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    gate.load_state_dict(best_state)
    gate.eval()
    with torch.no_grad():
        calibration_probability = torch.sigmoid(gate(
            features.index_select(0, calibration_index)))
    if target_precision is None:
        target_precision = float(getattr(
            args, 'progressive_gate_target_precision', 0.995))
    target_precision = min(1.0, max(0.5, float(target_precision)))
    threshold, calibration_precision, calibration_coverage = _choose_precision_threshold(
        calibration_probability,
        labels.index_select(0, calibration_index),
        target_precision,
        min(1.0, max(0.0, float(getattr(
            args, 'progressive_gate_min_coverage', 0.02)))))
    report_precision = 0.0
    report_coverage = 0.0
    if threshold is not None:
        with torch.no_grad():
            report_probability = torch.sigmoid(gate(
                features.index_select(0, report_index)))
        selected = report_probability >= threshold
        report_coverage = float(selected.float().mean().item())
        if bool(selected.any()):
            report_precision = float(labels.index_select(
                0, report_index)[selected].mean().item())
    minimum_coverage = min(1.0, max(0.0, float(getattr(
        args, 'progressive_gate_min_coverage', 0.02))))
    accepted = (threshold is not None
                and report_precision >= target_precision - 1e-6
                and report_coverage >= minimum_coverage)

    gate.register_buffer('feature_mean', mean.to(device))
    gate.register_buffer('feature_std', std.to(device))
    gate.threshold = threshold if accepted else None
    return gate, {
        'accepted': accepted,
        'samples': int(features.shape[0]),
        'safe_rate': float(labels.mean().item()),
        'calibration_precision': calibration_precision,
        'calibration_coverage': calibration_coverage,
        'report_precision': report_precision,
        'report_coverage': report_coverage,
        'threshold': threshold,
    }


@torch.no_grad()
def _collect_gate_dataset(model, original_model, data_loader_per_cls,
                          class_mask, seen_task_count, args, device):
    sample_limit = max(2, int(getattr(
        args, 'progressive_gate_samples_per_class', 12)))
    stage_features = {2: [], 4: []}
    stage_logits = {2: [], 4: []}
    exhaustive_logits = []
    output_targets = []
    labels = {2: [], 4: []}
    class_targets = []
    model.eval()
    original_model.eval()
    rank_batch_size = max(1, int(getattr(
        args, 'progressive_lora_batch_ranks', 1)))
    for task_index in range(seen_task_count):
        for class_id in class_mask[task_index]:
            collected = 0
            for inputs, targets in data_loader_per_cls[class_id]['train']:
                remaining = sample_limit - collected
                if remaining <= 0:
                    break
                inputs = inputs[:remaining].to(device, non_blocking=True)
                targets = targets[:remaining].to(device, non_blocking=True)
                tii_logits = original_model(inputs)['logits']
                tii_prior = _tii_task_prior(
                    tii_logits, class_mask, seen_task_count)
                sorted_tii_scores, candidate_tasks = torch.sort(
                    tii_prior, dim=1, descending=True)
                rank_logits = torch.full(
                    (inputs.shape[0], seen_task_count, tii_logits.shape[1]),
                    float('-inf'), dtype=tii_logits.dtype, device=device)
                task_evidence = torch.full(
                    (inputs.shape[0], seen_task_count), float('-inf'),
                    dtype=tii_logits.dtype, device=device)
                class_margins = torch.zeros_like(task_evidence)
                temperature = max(1e-6, float(getattr(
                    args, 'progressive_logit_temperature', 1.0)))
                prior_weight = max(0.0, float(getattr(
                    args, 'progressive_tii_prior_weight', 0.3)))
                for rank_start in range(
                        0, seen_task_count, rank_batch_size):
                    rank_end = min(
                        seen_task_count, rank_start + rank_batch_size)
                    rank_tasks = candidate_tasks[:, rank_start:rank_end]
                    rank_adapter_logits = _run_lora_rank_batch(
                        model, inputs, rank_tasks)
                    for rank_offset, rank in enumerate(
                            range(rank_start, rank_end)):
                        active_tasks = rank_tasks[:, rank_offset]
                        adapter_logits = rank_adapter_logits[:, rank_offset]
                        for selected_task in active_tasks.unique().tolist():
                            rows = torch.nonzero(
                                active_tasks == selected_task).flatten()
                            class_index = torch.as_tensor(
                                class_mask[selected_task], dtype=torch.long,
                                device=device)
                            local_logits = adapter_logits.index_select(
                                0, rows).index_select(1, class_index) / temperature
                            calibrated = local_logits + prior_weight * tii_prior[
                                rows, selected_task].unsqueeze(1)
                            rank_logits[
                                rows.unsqueeze(1), rank, class_index.unsqueeze(0)
                            ] = calibrated
                            task_evidence[rows, rank] = calibrated.max(1).values
                            local_top = torch.topk(
                                calibrated, k=min(2, calibrated.shape[1]),
                                dim=1).values
                            class_margins[rows, rank] = (
                                local_top[:, 0] - local_top[:, 1]
                                if local_top.shape[1] > 1 else float('inf'))
                winner_rank = task_evidence.argmax(1)
                full_logits = rank_logits.max(dim=1).values
                full_predictions = full_logits.argmax(dim=1)
                full_top5_correct = torch.topk(
                    full_logits, k=min(5, full_logits.shape[1]), dim=1
                ).indices.eq(targets.unsqueeze(1)).any(dim=1)
                full_loss = F.cross_entropy(
                    full_logits, targets, reduction='none')
                default_loss_tolerance = max(0.0, float(getattr(
                    args, 'progressive_gate_loss_tolerance', 0.05)))
                excluded_margin = max(1.0, float(getattr(
                    args, 'progressive_excluded_logit_margin', 20.0)))
                for boundary in (2, 4):
                    effective = min(boundary, seen_task_count)
                    stage_features[boundary].append(_stage_features(
                        tii_prior, sorted_tii_scores, task_evidence,
                        class_margins, effective).cpu())
                    partial_logits = _finalize_partial_logits(
                        rank_logits[:, :effective].max(dim=1).values,
                        excluded_margin)
                    stage_logits[boundary].append(partial_logits.cpu())
                    partial_top5_correct = torch.topk(
                        partial_logits, k=min(5, partial_logits.shape[1]), dim=1
                    ).indices.eq(targets.unsqueeze(1)).any(dim=1)
                    partial_loss = F.cross_entropy(
                        partial_logits, targets, reduction='none')
                    loss_tolerance = default_loss_tolerance
                    if boundary == 4:
                        loss_tolerance = max(0.0, float(getattr(
                            args, 'progressive_gate_stage2_loss_tolerance', 0.0)))
                    safe = (
                        winner_rank.lt(effective)
                        & partial_logits.argmax(dim=1).eq(full_predictions)
                        & partial_top5_correct.eq(full_top5_correct)
                        & partial_loss.le(full_loss + loss_tolerance)
                    )
                    labels[boundary].append(safe.cpu())
                exhaustive_logits.append(_finalize_partial_logits(
                    full_logits, excluded_margin).cpu())
                output_targets.append(targets.cpu())
                class_targets.append(torch.full(
                    (inputs.shape[0],), int(class_id), dtype=torch.long))
                collected += inputs.shape[0]
    class_targets = torch.cat(class_targets)
    datasets = {
        boundary: (
            torch.cat(stage_features[boundary]),
            torch.cat(labels[boundary]), class_targets)
        for boundary in (2, 4)
    }
    datasets['stage_logits'] = {
        boundary: torch.cat(stage_logits[boundary])
        for boundary in (2, 4)}
    datasets['exhaustive_logits'] = torch.cat(exhaustive_logits)
    datasets['output_targets'] = torch.cat(output_targets)
    return datasets


def _cascade_stage2_mask(stage1_gate, stage1_features, class_targets,
                         minimum_per_class=4, context_ratio=0.25):
    if stage1_gate is None or stage1_gate.threshold is None:
        return torch.ones(
            stage1_features.shape[0], dtype=torch.bool,
            device=stage1_features.device)

    gate_device = stage1_gate.feature_mean.device
    with torch.no_grad():
        features = stage1_features.to(gate_device)
        normalized = (
            features - stage1_gate.feature_mean) / stage1_gate.feature_std
        halt_probability = torch.sigmoid(
            stage1_gate(normalized)).detach().cpu()
    class_targets_cpu = class_targets.detach().cpu()
    selected = halt_probability.lt(float(stage1_gate.threshold))
    minimum_per_class = max(2, int(minimum_per_class))
    context_ratio = min(1.0, max(0.0, float(context_ratio)))

    # Stage 2 sees the hard residual from Stage 1. Keep a small hard subset
    # from every class so the class-stratified train/report split stays valid.
    for class_id in torch.unique(class_targets_cpu).tolist():
        class_index = torch.nonzero(
            class_targets_cpu == int(class_id), as_tuple=False).flatten()
        required = min(minimum_per_class, class_index.numel())
        if int(selected.index_select(0, class_index).sum().item()) < required:
            hard_order = torch.argsort(
                halt_probability.index_select(0, class_index))
            selected[class_index.index_select(0, hard_order[:required])] = True

        accepted_index = class_index[~selected.index_select(0, class_index)]
        context_count = int(round(accepted_index.numel() * context_ratio))
        if context_ratio > 0.0 and accepted_index.numel() > 0:
            context_count = max(1, context_count)
        if context_count > 0:
            # The least-confident Stage-1 accepts are the most useful
            # regularization context for the downstream gate.
            context_order = torch.argsort(
                halt_probability.index_select(0, accepted_index))
            selected[accepted_index.index_select(
                0, context_order[:context_count])] = True
    return selected.to(stage1_features.device)


def _choose_output_temperature(logits, targets, minimum=0.5, maximum=4.0,
                               steps=129):
    minimum = max(1e-3, float(minimum))
    maximum = max(minimum, float(maximum))
    steps = max(3, int(steps))
    before = float(F.cross_entropy(logits, targets).item())
    candidates = torch.exp(torch.linspace(
        math.log(minimum), math.log(maximum), steps,
        device=logits.device, dtype=logits.dtype))
    losses = torch.empty(steps, device=logits.device, dtype=logits.dtype)
    for index, temperature in enumerate(candidates):
        losses[index] = F.cross_entropy(logits / temperature, targets)
    best_index = int(losses.argmin().item())
    lower = candidates[max(0, best_index - 1)]
    upper = candidates[min(steps - 1, best_index + 1)]
    refined = torch.exp(torch.linspace(
        lower.log(), upper.log(), 65,
        device=logits.device, dtype=logits.dtype))
    refined_losses = torch.empty_like(refined)
    for index, temperature in enumerate(refined):
        refined_losses[index] = F.cross_entropy(
            logits / temperature, targets)
    refined_index = int(refined_losses.argmin().item())
    temperature = float(refined[refined_index].item())
    after = float(refined_losses[refined_index].item())
    if after > before:
        return 1.0, before, before
    return temperature, before, after


def _fit_progressive_output_temperature(datasets, gates, args, device):
    if not bool(getattr(
            args, 'progressive_output_temperature_scaling', False)):
        return 1.0, {'samples': 0, 'loss_before': 0.0, 'loss_after': 0.0}

    stage1_features, _, class_targets = datasets[2]
    stage2_features = datasets[4][0]
    stage1_stop = _gate_decision(
        gates.get(2), stage1_features.to(device)).cpu()
    remaining = ~stage1_stop
    stage2_stop = torch.zeros_like(stage1_stop)
    if bool(remaining.any()):
        remaining_index = torch.nonzero(
            remaining, as_tuple=False).flatten()
        stage2_decision = _gate_decision(
            gates.get(4), stage2_features.index_select(
                0, remaining_index).to(device)).cpu()
        stage2_stop[remaining_index] = stage2_decision

    emitted_logits = datasets['exhaustive_logits'].clone()
    emitted_logits[stage1_stop] = datasets['stage_logits'][2][stage1_stop]
    emitted_logits[stage2_stop] = datasets['stage_logits'][4][stage2_stop]
    _, calibration_index = _class_stratified_split(
        class_targets,
        float(getattr(args, 'progressive_output_temperature_ratio', 0.25)),
        int(getattr(args, 'seed', 42)) + 7919)
    calibration_logits = emitted_logits.index_select(
        0, calibration_index).to(device)
    calibration_targets = datasets['output_targets'].index_select(
        0, calibration_index).to(device)
    temperature, before, after = _choose_output_temperature(
        calibration_logits,
        calibration_targets,
        getattr(args, 'progressive_output_temperature_min', 0.5),
        getattr(args, 'progressive_output_temperature_max', 4.0),
        getattr(args, 'progressive_output_temperature_steps', 129))
    return temperature, {
        'samples': int(calibration_index.numel()),
        'loss_before': before,
        'loss_after': after,
    }


def train_progressive_halting_gates(model, original_model,
                                    data_loader_per_cls, class_mask,
                                    seen_task_count, args, device):
    if seen_task_count <= 2:
        return {}, {}
    datasets = _collect_gate_dataset(
        model, original_model, data_loader_per_cls, class_mask,
        seen_task_count, args, device)
    gates = {}
    stats = {}
    stage1_features, stage1_labels, class_targets = datasets[2]
    gates[2], stats[2] = _fit_gate(
        stage1_features, stage1_labels, class_targets, args, device,
        seen_task_count * 10 + 2)

    if seen_task_count > 4:
        stage2_features, stage2_labels, stage2_targets = datasets[4]
        stage2_mask = _cascade_stage2_mask(
            gates[2], stage1_features, class_targets,
            getattr(args, 'progressive_gate_stage2_min_samples_per_class', 4),
            getattr(args, 'progressive_gate_stage2_context_ratio', 0.25))
        stage2_features = stage2_features[stage2_mask]
        stage2_labels = stage2_labels[stage2_mask]
        stage2_targets = stage2_targets[stage2_mask]
        gates[4], stats[4] = _fit_gate(
            stage2_features, stage2_labels, stage2_targets, args, device,
            seen_task_count * 10 + 4,
            target_precision=getattr(
                args, 'progressive_gate_stage2_target_precision', 1.0))
        stats[4]['cascade_samples'] = int(stage2_mask.sum().item())
        stats[4]['cascade_candidate_rate'] = float(
            stage2_mask.float().mean().item())
    output_temperature, output_stats = _fit_progressive_output_temperature(
        datasets, gates, args, device)
    gates['_output_temperature'] = output_temperature
    gates['_output_calibration'] = output_stats
    output_stats['temperature'] = output_temperature
    return gates, stats


def _gate_probability(gate, features):
    if gate is None or gate.threshold is None:
        return torch.zeros(
            features.shape[0], dtype=features.dtype, device=features.device)
    normalized = (features - gate.feature_mean) / gate.feature_std
    return torch.sigmoid(gate(normalized))


def _gate_decision(gate, features):
    if gate is None or gate.threshold is None:
        return torch.zeros(features.shape[0], dtype=torch.bool,
                           device=features.device)
    return _gate_probability(gate, features) >= float(gate.threshold)


def _apply_rank_preserving_smoothing(logits, smoothing):
    smoothing = smoothing.to(device=logits.device, dtype=logits.dtype)
    smoothing = smoothing.clamp(0.0, 1.0 - 1e-6).unsqueeze(1)
    log_probabilities = F.log_softmax(logits, dim=1)
    uniform_log_probability = -math.log(logits.shape[1])
    smoothed = torch.logaddexp(
        log_probabilities + torch.log1p(-smoothing),
        smoothing.clamp_min(1e-30).log() + uniform_log_probability)
    return torch.where(smoothing > 0.0, smoothed, log_probabilities)


@torch.no_grad()
def calibrated_progressive_rematching(model, inputs, tii_logits, class_mask,
                                      seen_task_count, args, gates):
    device = inputs.device
    temperature = max(1e-6, float(getattr(
        args, 'progressive_logit_temperature', 1.0)))
    prior_weight = max(0.0, float(getattr(
        args, 'progressive_tii_prior_weight', 0.3)))
    excluded_margin = max(1.0, float(getattr(
        args, 'progressive_excluded_logit_margin', 20.0)))
    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    sorted_tii_scores, candidate_tasks = torch.sort(
        tii_prior, dim=1, descending=True)
    batch_size = inputs.shape[0]
    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = torch.full(
        (batch_size, seen_task_count), float('-inf'),
        dtype=tii_logits.dtype, device=device)
    class_margins = torch.zeros_like(task_evidence)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    counts = torch.zeros(batch_size, dtype=torch.float32, device=device)
    forward_calls = torch.zeros_like(counts)
    stages = torch.zeros(batch_size, dtype=torch.long, device=device)
    halt_confidence = torch.ones(
        batch_size, dtype=tii_logits.dtype, device=device)

    rank_batch_size = max(1, int(getattr(
        args, 'progressive_lora_batch_ranks', 1)))
    rank = 0
    while rank < seen_task_count:
        active_index = torch.nonzero(active).flatten()
        if active_index.numel() == 0:
            break
        next_boundary = min(
            boundary for boundary in (2, 4, seen_task_count)
            if boundary > rank)
        rank_end = min(rank + rank_batch_size, next_boundary)
        active_rank_tasks = candidate_tasks.index_select(
            0, active_index)[:, rank:rank_end]
        rank_adapter_logits = _run_lora_rank_batch(
            model, inputs.index_select(0, active_index), active_rank_tasks)
        counts[active_index] += float(rank_end - rank)
        forward_calls[active_index] += 1.0
        for rank_offset, current_rank in enumerate(range(rank, rank_end)):
            active_tasks = active_rank_tasks[:, rank_offset]
            adapter_logits = rank_adapter_logits[:, rank_offset]
            for task_index in active_tasks.unique().tolist():
                local_rows = torch.nonzero(active_tasks == task_index).flatten()
                batch_rows = active_index.index_select(0, local_rows)
                class_index = torch.as_tensor(
                    class_mask[task_index], dtype=torch.long, device=device)
                local_logits = adapter_logits.index_select(
                    0, local_rows).index_select(1, class_index) / temperature
                calibrated = local_logits + prior_weight * tii_prior[
                    batch_rows, task_index].unsqueeze(1)
                merged_logits[
                    batch_rows.unsqueeze(1), class_index.unsqueeze(0)] = calibrated
                task_evidence[batch_rows, current_rank] = calibrated.max(
                    dim=1).values
                local_top = torch.topk(
                    calibrated, k=min(2, calibrated.shape[1]), dim=1).values
                class_margins[batch_rows, current_rank] = (
                    local_top[:, 0] - local_top[:, 1]
                    if local_top.shape[1] > 1 else float('inf'))

        rank = rank_end
        boundary = rank
        if boundary >= seen_task_count:
            stages[active] = 3
            active.zero_()
            break
        if boundary not in (2, 4):
            continue
        active_index = torch.nonzero(active).flatten()
        features = _stage_features(
            tii_prior.index_select(0, active_index),
            sorted_tii_scores.index_select(0, active_index),
            task_evidence.index_select(0, active_index),
            class_margins.index_select(0, active_index), boundary)
        gate = gates.get(boundary)
        gate_probability = _gate_probability(gate, features)
        if gate is not None and gate.threshold is not None:
            stop_mask = gate_probability >= float(gate.threshold)
        else:
            stop_mask = torch.zeros_like(gate_probability, dtype=torch.bool)
        stopped = active_index[stop_mask]
        stages[stopped] = 1 if boundary == 2 else 2
        halt_confidence[stopped] = gate_probability[stop_mask]
        active[stopped] = False

    selected_rank = task_evidence.argmax(dim=1)
    routed_tasks = candidate_tasks.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    finite = torch.isfinite(merged_logits)
    row_min = merged_logits.masked_fill(~finite, float('inf')).min(
        dim=1, keepdim=True).values
    merged_logits = torch.where(finite, merged_logits, row_min - excluded_margin)
    output_temperature = max(
        1e-3, float(gates.get('_output_temperature', 1.0)))
    merged_logits = merged_logits / output_temperature
    if bool(getattr(args, 'progressive_uncertainty_smoothing', False)):
        strength = max(0.0, float(getattr(
            args, 'progressive_smoothing_strength', 0.5)))
        maximum = min(0.5, max(0.0, float(getattr(
            args, 'progressive_smoothing_max', 0.05))))
        smoothing = ((1.0 - halt_confidence) * strength).clamp_max(maximum)
        smoothing = torch.where(
            stages.lt(3), smoothing, torch.zeros_like(smoothing))
        merged_logits = _apply_rank_preserving_smoothing(
            merged_logits, smoothing)
    return merged_logits, routed_tasks, counts, forward_calls, stages
