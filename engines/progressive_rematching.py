import torch


def _standardize_task_scores(scores):
    if scores.shape[1] <= 1:
        return torch.zeros_like(scores)
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


def _tii_task_prior(tii_logits, class_mask, seen_task_count):
    scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=tii_logits.device)
        scores.append(tii_logits.index_select(1, class_index).max(dim=1).values)
    return _standardize_task_scores(torch.stack(scores, dim=1))


def _stage_is_confident(task_evidence, class_margins, sorted_tii_scores,
                        boundary, tii_margin_threshold,
                        adapter_margin_threshold, class_margin_threshold):
    evaluated_evidence = task_evidence[:, :boundary]
    top_count = min(2, boundary)
    top = torch.topk(evaluated_evidence, k=top_count, dim=1)
    best_rank = top.indices[:, 0]
    if top_count == 1:
        adapter_margin = torch.full_like(top.values[:, 0], float('inf'))
    else:
        adapter_margin = top.values[:, 0] - top.values[:, 1]

    best_class_margin = class_margins.gather(
        1, best_rank.unsqueeze(1)).squeeze(1)
    selected_tii_score = sorted_tii_scores.gather(
        1, best_rank.unsqueeze(1)).squeeze(1)
    if boundary >= sorted_tii_scores.shape[1]:
        unseen_margin = torch.full_like(selected_tii_score, float('inf'))
    else:
        unseen_margin = selected_tii_score - sorted_tii_scores[:, boundary]

    return (
        (unseen_margin >= tii_margin_threshold)
        & (adapter_margin >= adapter_margin_threshold)
        & (best_class_margin >= class_margin_threshold)
    )


@torch.no_grad()
def progressive_adapter_rematching(model, inputs, tii_logits, class_mask,
                                   seen_task_count, args):
    """Evaluate LoRAs in TII order and expand uncertain samples to exhaustive."""
    device = inputs.device
    batch_size = inputs.shape[0]
    temperature = max(
        1e-6, float(getattr(args, 'progressive_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'progressive_tii_prior_weight', 0.3)))
    initial_candidates = min(
        seen_task_count,
        max(1, int(getattr(args, 'progressive_initial_candidates', 2))))
    intermediate_candidates = min(
        seen_task_count,
        max(initial_candidates, int(getattr(
            args, 'progressive_intermediate_candidates', 4))))

    stage1_thresholds = (
        max(0.0, float(getattr(args, 'progressive_stage1_tii_margin', 1.0))),
        max(0.0, float(getattr(args, 'progressive_stage1_adapter_margin', 0.75))),
        max(0.0, float(getattr(args, 'progressive_stage1_class_margin', 0.5))),
    )
    stage2_thresholds = (
        max(0.0, float(getattr(args, 'progressive_stage2_tii_margin', 0.35))),
        max(0.0, float(getattr(args, 'progressive_stage2_adapter_margin', 0.5))),
        max(0.0, float(getattr(args, 'progressive_stage2_class_margin', 0.25))),
    )

    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    sorted_tii_scores, candidate_tasks = torch.sort(
        tii_prior, dim=1, descending=True)

    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = torch.full(
        (batch_size, seen_task_count), float('-inf'),
        dtype=tii_logits.dtype, device=device)
    class_margins = torch.zeros_like(task_evidence)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    lora_counts = torch.zeros(batch_size, dtype=torch.float32, device=device)
    stop_stage = torch.zeros(batch_size, dtype=torch.long, device=device)

    boundaries = {initial_candidates: (1, stage1_thresholds)}
    if intermediate_candidates > initial_candidates:
        boundaries[intermediate_candidates] = (2, stage2_thresholds)

    for rank in range(seen_task_count):
        active_index = torch.nonzero(active).flatten()
        if active_index.numel() == 0:
            break

        active_tasks = candidate_tasks.index_select(0, active_index)[:, rank]
        adapter_logits = model(
            inputs.index_select(0, active_index), task_id=active_tasks)['logits']
        lora_counts[active_index] += 1.0

        for task_index in active_tasks.unique().tolist():
            local_rows = torch.nonzero(active_tasks == task_index).flatten()
            batch_rows = active_index.index_select(0, local_rows)
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            local_logits = adapter_logits.index_select(
                0, local_rows).index_select(1, class_index) / temperature
            calibrated_logits = local_logits + prior_weight * tii_prior[
                batch_rows, task_index].unsqueeze(1)
            merged_logits[
                batch_rows.unsqueeze(1), class_index.unsqueeze(0)
            ] = calibrated_logits
            task_evidence[batch_rows, rank] = calibrated_logits.max(dim=1).values
            top_count = min(2, calibrated_logits.shape[1])
            local_top = torch.topk(calibrated_logits, k=top_count, dim=1).values
            if top_count == 1:
                local_margin = torch.full_like(local_top[:, 0], float('inf'))
            else:
                local_margin = local_top[:, 0] - local_top[:, 1]
            class_margins[batch_rows, rank] = local_margin

        boundary = rank + 1
        if boundary >= seen_task_count:
            stop_stage[active] = 3
            active.zero_()
            break
        if boundary not in boundaries:
            continue

        stage_id, thresholds = boundaries[boundary]
        active_index = torch.nonzero(active).flatten()
        confident = _stage_is_confident(
            task_evidence.index_select(0, active_index),
            class_margins.index_select(0, active_index),
            sorted_tii_scores.index_select(0, active_index),
            boundary,
            thresholds[0], thresholds[1], thresholds[2],
        )
        stopped_index = active_index[confident]
        stop_stage[stopped_index] = stage_id
        active[stopped_index] = False

    selected_rank = task_evidence.argmax(dim=1)
    routed_tasks = candidate_tasks.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)

    finite_mask = torch.isfinite(merged_logits)
    row_min = merged_logits.masked_fill(~finite_mask, float('inf')).min(
        dim=1, keepdim=True).values
    excluded_margin = max(
        1.0, float(getattr(args, 'progressive_excluded_logit_margin', 20.0)))
    merged_logits = torch.where(
        finite_mask, merged_logits, row_min - excluded_margin)

    return merged_logits, routed_tasks, lora_counts, stop_stage
