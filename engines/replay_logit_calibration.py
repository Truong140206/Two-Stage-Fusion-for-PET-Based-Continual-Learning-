import math

import torch
import torch.nn.functional as F


def _bounded_parameters(raw_log_scale, raw_bias, max_scale, max_bias):
    log_limit = math.log(max(1.0, float(max_scale)))
    log_scale = log_limit * torch.tanh(raw_log_scale)
    scale = torch.exp(log_scale)
    bias = float(max_bias) * torch.tanh(raw_bias)
    bias = bias - bias.mean()
    return scale, bias, log_scale


def _accuracy_by_age(logits, targets, class_tasks, newest_task):
    prediction = logits.argmax(dim=1)
    correct = prediction.eq(targets)
    sample_tasks = class_tasks.index_select(0, targets)
    old_mask = sample_tasks < newest_task
    new_mask = sample_tasks == newest_task

    def masked_accuracy(mask):
        if not bool(mask.any()):
            return 0.0
        return float(correct[mask].float().mean().mul(100.0).item())

    return {
        'all': float(correct.float().mean().mul(100.0).item()),
        'old': masked_accuracy(old_mask),
        'new': masked_accuracy(new_mask),
    }


@torch.no_grad()
def _collect_memory_logits(model, feature_memory, seen_classes, device,
                           max_samples_per_class, batch_size):
    base_model = model.module if hasattr(model, 'module') else model
    features = []
    labels = []
    for local_label, class_id in enumerate(seen_classes):
        memory = feature_memory.get(int(class_id))
        if memory is None or memory.numel() == 0:
            continue
        memory = memory[:max_samples_per_class].float()
        features.append(memory)
        labels.append(torch.full(
            (memory.shape[0],), local_label, dtype=torch.long))

    if len(features) != len(seen_classes):
        raise RuntimeError(
            'Replay logit calibration requires memory for every seen class: '
            '{} of {} classes found'.format(len(features), len(seen_classes)))

    features = torch.cat(features, dim=0)
    targets = torch.cat(labels, dim=0).to(device)
    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=device)
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        feature_batch = features[start:start + batch_size].to(
            device=device, non_blocking=True)
        output = base_model(feature_batch, fc_only=True)['logits']
        chunks.append(output.index_select(1, seen_index))
    return torch.cat(chunks, dim=0).detach(), targets


def calibrate_task_logits(
        model, feature_memory, class_mask, seen_task_count, args, device,
        report_feature_memory=None, apply_to_model=True,
        max_samples_per_class=None):
    """Fit task-wise affine corrections and gate them on separate features."""
    if seen_task_count <= 1:
        return None

    base_model = model.module if hasattr(model, 'module') else model
    if not hasattr(base_model, 'head') or not hasattr(base_model, 'fc_norm'):
        raise AttributeError('Replay logit calibration requires model.head and fc_norm')

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    class_tasks = torch.as_tensor(
        [
            task_index
            for task_index in range(seen_task_count)
            for _ in class_mask[task_index]
        ],
        dtype=torch.long,
        device=device,
    )
    if max_samples_per_class is None:
        max_samples = max(
            1, int(getattr(
                args, 'replay_calibration_samples_per_class', 48)))
    else:
        max_samples = max(1, int(max_samples_per_class))
    batch_size = max(
        1, int(getattr(args, 'replay_calibration_batch_size', 1024)))
    base_model.eval()
    base_logits, targets = _collect_memory_logits(
        base_model, feature_memory, seen_classes, device, max_samples, batch_size)
    report_logits = base_logits
    report_targets = targets
    if report_feature_memory is not None:
        report_logits, report_targets = _collect_memory_logits(
            base_model, report_feature_memory, seen_classes, device,
            max_samples, batch_size)

    steps = max(1, int(getattr(args, 'replay_calibration_steps', 200)))
    learning_rate = max(
        1e-6, float(getattr(args, 'replay_calibration_lr', 0.05)))
    max_scale = max(
        1.0, float(getattr(args, 'replay_calibration_max_scale', 1.5)))
    max_bias = max(
        0.0, float(getattr(args, 'replay_calibration_max_bias', 1.0)))
    regularization = max(
        0.0, float(getattr(args, 'replay_calibration_regularization', 0.01)))
    old_margin_weight = max(
        0.0, float(getattr(args, 'replay_calibration_old_margin_weight', 0.25)))
    old_tolerance = max(
        0.0, float(getattr(args, 'replay_calibration_old_tolerance', 0.0)))
    minimum_gain = max(
        0.0, float(getattr(args, 'replay_calibration_min_gain', 0.0)))
    max_old_class_drop = max(
        0.0, float(getattr(
            args, 'replay_calibration_max_old_class_drop', float('inf'))))

    raw_log_scale = torch.zeros(seen_task_count, device=device, requires_grad=True)
    raw_bias = torch.zeros(seen_task_count, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [raw_log_scale, raw_bias], lr=learning_rate)
    column_tasks = class_tasks
    sample_tasks = column_tasks.index_select(0, targets)
    old_sample_mask = sample_tasks < seen_task_count - 1

    with torch.no_grad():
        target_base = base_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        competitor_base = base_logits.clone()
        competitor_base.scatter_(1, targets.unsqueeze(1), float('-inf'))
        base_margin = target_base - competitor_base.max(dim=1).values
        before = _accuracy_by_age(
            report_logits, report_targets, class_tasks, seen_task_count - 1)
        before_loss = float(F.cross_entropy(
            report_logits, report_targets).item())
        before_predictions = report_logits.argmax(dim=1)
        before_class_accuracy = torch.stack([
            before_predictions[report_targets == class_index].eq(
                report_targets[report_targets == class_index]
            ).float().mean()
            for class_index in range(len(seen_classes))
        ])

    best_loss = float('inf')
    best_parameters = None
    for _ in range(steps):
        scale, bias, log_scale = _bounded_parameters(
            raw_log_scale, raw_bias, max_scale, max_bias)
        calibrated = (
            base_logits * scale.index_select(0, column_tasks).unsqueeze(0)
            + bias.index_select(0, column_tasks).unsqueeze(0)
        )
        loss_ce = F.cross_entropy(calibrated, targets)
        target_logits = calibrated.gather(1, targets.unsqueeze(1)).squeeze(1)
        competitors = calibrated.clone()
        competitors.scatter_(1, targets.unsqueeze(1), float('-inf'))
        calibrated_margin = target_logits - competitors.max(dim=1).values
        if bool(old_sample_mask.any()):
            loss_old_margin = F.relu(
                base_margin[old_sample_mask]
                - calibrated_margin[old_sample_mask]
            ).mean()
        else:
            loss_old_margin = calibrated.new_zeros(())
        loss_regularization = log_scale.pow(2).mean() + bias.pow(2).mean()
        loss = (
            loss_ce
            + old_margin_weight * loss_old_margin
            + regularization * loss_regularization
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_parameters = (
                raw_log_scale.detach().clone(), raw_bias.detach().clone())

    with torch.no_grad():
        raw_log_scale.copy_(best_parameters[0])
        raw_bias.copy_(best_parameters[1])
        scale, bias, _ = _bounded_parameters(
            raw_log_scale, raw_bias, max_scale, max_bias)
        calibrated_logits = (
            report_logits * scale.index_select(0, column_tasks).unsqueeze(0)
            + bias.index_select(0, column_tasks).unsqueeze(0)
        )
        after = _accuracy_by_age(
            calibrated_logits, report_targets, class_tasks,
            seen_task_count - 1)
        after_loss = float(F.cross_entropy(
            calibrated_logits, report_targets).item())
        after_predictions = calibrated_logits.argmax(dim=1)
        after_class_accuracy = torch.stack([
            after_predictions[report_targets == class_index].eq(
                report_targets[report_targets == class_index]
            ).float().mean()
            for class_index in range(len(seen_classes))
        ])
        old_class_mask = class_tasks < seen_task_count - 1
        worst_old_class_drop = 0.0
        if bool(old_class_mask.any()):
            worst_old_class_drop = float((
                before_class_accuracy[old_class_mask]
                - after_class_accuracy[old_class_mask]
            ).max().mul(100.0).item())
        accepted = (
            after['all'] >= before['all'] + minimum_gain
            and after['old'] + old_tolerance >= before['old']
            and after_loss <= before_loss
            and worst_old_class_drop <= max_old_class_drop
        )

        if accepted and apply_to_model:
            for task_index in range(seen_task_count):
                class_index = torch.as_tensor(
                    class_mask[task_index], dtype=torch.long,
                    device=base_model.head.weight.device)
                task_scale = scale[task_index].to(base_model.head.weight.dtype)
                task_bias = bias[task_index].to(base_model.head.weight.dtype)
                scaled_weight = (
                    base_model.head.weight.index_select(0, class_index)
                    * task_scale
                )
                base_model.head.weight.index_copy_(
                    0, class_index, scaled_weight)
                if base_model.head.bias is not None:
                    scaled_bias = (
                        base_model.head.bias.index_select(0, class_index)
                        * task_scale + task_bias
                    )
                    base_model.head.bias.index_copy_(0, class_index, scaled_bias)

    return {
        'accepted': accepted,
        'before': before,
        'after': after,
        'before_loss': before_loss,
        'after_loss': after_loss,
        'scale': scale.detach().cpu().tolist(),
        'bias': bias.detach().cpu().tolist(),
        'samples': int(targets.numel()),
        'report_samples': int(report_targets.numel()),
        'worst_old_class_drop': worst_old_class_drop,
        'applied_to_model': bool(accepted and apply_to_model),
    }
