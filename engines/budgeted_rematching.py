import torch


def _standardize_task_scores(scores):
    if scores.shape[1] <= 1:
        return torch.zeros_like(scores)
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


def _standardize_vector(values):
    if values.numel() <= 1:
        return torch.zeros_like(values)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _task_scores(tii_logits, class_mask, seen_task_count):
    scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long,
            device=tii_logits.device)
        scores.append(
            tii_logits.index_select(1, class_index).max(dim=1).values)
    return _standardize_task_scores(torch.stack(scores, dim=1))


def _select_fallback_samples(tii_task_scores, base_logits, base_tasks,
                             fallback_fraction, disagreement_weight,
                             classifier_weight):
    batch_size, task_count = tii_task_scores.shape
    fallback_mask = torch.zeros(
        batch_size, dtype=torch.bool, device=tii_task_scores.device)
    if task_count <= 1 or fallback_fraction <= 0.0:
        return fallback_mask

    tii_top = torch.topk(tii_task_scores, k=2, dim=1, largest=True)
    tii_tasks = tii_top.indices[:, 0]
    tii_margin = tii_top.values[:, 0] - tii_top.values[:, 1]

    classifier_top = torch.topk(base_logits, k=2, dim=1, largest=True).values
    classifier_margin = classifier_top[:, 0] - classifier_top[:, 1]
    disagreement = base_tasks.ne(tii_tasks).float()

    risk = (
        float(disagreement_weight) * disagreement
        - _standardize_vector(tii_margin)
        - float(classifier_weight) * _standardize_vector(classifier_margin)
    )
    fallback_count = int(round(batch_size * fallback_fraction))
    fallback_count = min(batch_size, max(1, fallback_count))
    fallback_index = torch.topk(risk, k=fallback_count, largest=True).indices
    fallback_mask[fallback_index] = True
    return fallback_mask


@torch.no_grad()
def budgeted_exhaustive_fallback(model, inputs, tii_logits, base_logits,
                                 base_tasks, base_lora_counts, class_mask,
                                 seen_task_count, args):
    """Preserve original routing except for a fixed budget of risky samples."""
    device = inputs.device
    temperature = max(
        1e-6, float(getattr(args, 'budgeted_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'budgeted_tii_prior_weight', 0.3)))
    fallback_fraction = min(
        1.0, max(0.0, float(
            getattr(args, 'budgeted_fallback_fraction', 0.2))))
    disagreement_weight = max(
        0.0, float(getattr(args, 'budgeted_disagreement_weight', 2.0)))
    classifier_weight = max(
        0.0, float(getattr(args, 'budgeted_classifier_weight', 0.5)))

    tii_task_prior = _task_scores(
        tii_logits, class_mask, seen_task_count)
    fallback_mask = _select_fallback_samples(
        tii_task_prior, base_logits, base_tasks, fallback_fraction,
        disagreement_weight, classifier_weight)
    logits = base_logits.clone()
    routed_tasks = base_tasks.clone()
    lora_counts = base_lora_counts.float().clone()

    fallback_index = torch.nonzero(fallback_mask).flatten()
    if fallback_index.numel() == 0:
        return logits, routed_tasks, fallback_mask, lora_counts

    fallback_inputs = inputs.index_select(0, fallback_index)
    merged_logits = torch.full(
        (fallback_index.numel(), tii_logits.shape[1]),
        float('-inf'), dtype=tii_logits.dtype, device=device)
    task_evidence = []

    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        task_ids = torch.full(
            (fallback_index.numel(),), task_index,
            dtype=torch.long, device=device)
        adapter_logits = model(
            fallback_inputs, task_id=task_ids)['logits']
        local_logits = adapter_logits.index_select(1, class_index)

        local_logits = local_logits / temperature
        local_logits = local_logits + prior_weight * tii_task_prior[
            fallback_index, task_index].unsqueeze(1)
        merged_logits[:, class_index] = local_logits
        task_evidence.append(local_logits.max(dim=1).values)

    logits[fallback_index] = merged_logits
    routed_tasks[fallback_index] = torch.stack(
        task_evidence, dim=1).argmax(dim=1)
    lora_counts[fallback_index] += float(seen_task_count)
    return logits, routed_tasks, fallback_mask, lora_counts
