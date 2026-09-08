import torch


def _standardize_task_scores(scores):
    if scores.shape[1] <= 1:
        return torch.zeros_like(scores)
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


def _candidate_counts(sorted_scores, max_candidates, adaptive, high_margin, low_margin):
    batch_size = sorted_scores.shape[0]
    counts = torch.full(
        (batch_size,), max_candidates, dtype=torch.long, device=sorted_scores.device)
    if not adaptive or max_candidates <= 1:
        return counts

    counts.fill_(1)
    top_gap = sorted_scores[:, 0] - sorted_scores[:, 1]
    counts = counts + (top_gap < high_margin).long()
    if max_candidates > 2:
        counts = counts + (top_gap < low_margin).long() * (max_candidates - 2)
    return counts.clamp(max=max_candidates)


@torch.no_grad()
def selective_adapter_rematching(model, inputs, tii_logits, class_mask,
                                 seen_task_count, args, candidate_scores=None):
    """Evaluate only the most plausible LoRAs selected by TII task evidence."""
    device = inputs.device
    temperature = max(
        1e-6, float(getattr(args, 'selective_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'selective_tii_prior_weight', 0.3)))
    max_candidates = min(
        seen_task_count,
        max(1, int(getattr(args, 'selective_candidate_tasks', 3))))

    tii_task_scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        task_logits = tii_logits.index_select(1, class_index)
        tii_task_scores.append(task_logits.max(dim=1).values)
    tii_task_prior = _standardize_task_scores(
        torch.stack(tii_task_scores, dim=1))

    ranking_scores = tii_task_prior
    if candidate_scores is not None:
        ranking_scores = _standardize_task_scores(candidate_scores)
    sorted_scores, candidate_tasks = torch.topk(
        ranking_scores, k=max_candidates, dim=1, largest=True)
    candidate_source = str(
        getattr(args, 'selective_candidate_source', 'tii')).lower()
    candidate_counts = _candidate_counts(
        sorted_scores,
        max_candidates,
        bool(getattr(args, 'selective_adaptive', False)),
        max(0.0, float(getattr(args, 'selective_confident_margin', 1.0))),
        max(0.0, float(getattr(args, 'selective_ambiguous_margin', 0.35))),
    )

    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = torch.full(
        (inputs.shape[0], max_candidates),
        float('-inf'), dtype=tii_logits.dtype, device=device)

    for rank in range(max_candidates):
        active_index = torch.nonzero(candidate_counts > rank).flatten()
        if active_index.numel() == 0:
            continue
        active_tasks = candidate_tasks.index_select(0, active_index)[:, rank]
        adapter_logits = model(
            inputs.index_select(0, active_index), task_id=active_tasks)['logits']

        if rank == 0 and candidate_source == 'cascade' and max_candidates > 1:
            adapter_task_scores = []
            for task_index in range(seen_task_count):
                class_index = torch.as_tensor(
                    class_mask[task_index], dtype=torch.long, device=device)
                task_logits = adapter_logits.index_select(1, class_index)
                adapter_task_scores.append(task_logits.max(dim=1).values)
            adapter_task_prior = _standardize_task_scores(
                torch.stack(adapter_task_scores, dim=1))
            cascade_weight = max(
                0.0, float(getattr(args, 'selective_cascade_weight', 0.5)))
            cascade_scores = tii_task_prior + cascade_weight * adapter_task_prior
            cascade_scores.scatter_(
                1, candidate_tasks[:, :1], float('-inf'))
            candidate_tasks[:, 1:] = torch.topk(
                cascade_scores, k=max_candidates - 1, dim=1, largest=True).indices

        for task_index in active_tasks.unique().tolist():
            task_rows = torch.nonzero(active_tasks == task_index).flatten()
            batch_rows = active_index.index_select(0, task_rows)
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            local_logits = adapter_logits.index_select(0, task_rows).index_select(
                1, class_index) / temperature
            local_logits = local_logits + prior_weight * tii_task_prior[
                batch_rows, task_index].unsqueeze(1)
            merged_logits[batch_rows.unsqueeze(1), class_index.unsqueeze(0)] = local_logits
            task_evidence[batch_rows, rank] = local_logits.max(dim=1).values

    selected_rank = task_evidence.argmax(dim=1)
    routed_tasks = candidate_tasks.gather(1, selected_rank.unsqueeze(1)).squeeze(1)

    finite_mask = torch.isfinite(merged_logits)
    row_min = merged_logits.masked_fill(~finite_mask, float('inf')).min(
        dim=1, keepdim=True).values
    excluded_margin = max(
        1.0, float(getattr(args, 'selective_excluded_logit_margin', 20.0)))
    merged_logits = torch.where(
        finite_mask, merged_logits, row_min - excluded_margin)
    return merged_logits, routed_tasks, candidate_tasks, candidate_counts
