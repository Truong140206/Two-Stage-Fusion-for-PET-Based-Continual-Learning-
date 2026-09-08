import torch

import torch.nn.functional as F


def _standardize_across_tasks(scores):
    if scores.shape[1] <= 1:
        return torch.zeros_like(scores)
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


@torch.no_grad()
def hierarchical_adapter_rematching(model, inputs, tii_logits, class_mask,
                                    seen_task_count, args):
    """Fuse task posterior and task-conditional class probabilities."""
    device = inputs.device
    class_temperature = max(
        1e-6, float(getattr(args, 'hierarchical_class_temperature', 1.0)))
    task_temperature = max(
        1e-6, float(getattr(args, 'hierarchical_task_temperature', 1.0)))
    tii_weight = max(
        0.0, float(getattr(args, 'hierarchical_tii_weight', 0.3)))
    max_weight = max(
        0.0, float(getattr(args, 'hierarchical_max_weight', 1.0)))
    margin_weight = max(
        0.0, float(getattr(args, 'hierarchical_margin_weight', 0.5)))
    entropy_weight = max(
        0.0, float(getattr(args, 'hierarchical_entropy_weight', 0.5)))

    tii_evidence = []
    local_log_probabilities = []
    adapter_maxima = []
    adapter_margins = []
    adapter_concentrations = []

    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        tii_evidence.append(
            tii_logits.index_select(1, class_index).max(dim=1).values)

        task_ids = torch.full(
            (inputs.shape[0],), task_index, dtype=torch.long, device=device)
        adapter_logits = model(inputs, task_id=task_ids)['logits']
        local_logits = adapter_logits.index_select(1, class_index)
        scaled_logits = local_logits / class_temperature
        local_log_probability = F.log_softmax(scaled_logits, dim=1)
        local_probability = local_log_probability.exp()

        top_values = torch.topk(
            scaled_logits, k=min(2, scaled_logits.shape[1]), dim=1).values
        if top_values.shape[1] == 1:
            margin = torch.zeros_like(top_values[:, 0])
        else:
            margin = top_values[:, 0] - top_values[:, 1]

        local_log_probabilities.append(local_log_probability)
        adapter_maxima.append(scaled_logits.max(dim=1).values)
        adapter_margins.append(margin)
        adapter_concentrations.append(
            (local_probability * local_log_probability).sum(dim=1))

    tii_scores = _standardize_across_tasks(torch.stack(tii_evidence, dim=1))
    maximum_scores = _standardize_across_tasks(
        torch.stack(adapter_maxima, dim=1))
    margin_scores = _standardize_across_tasks(
        torch.stack(adapter_margins, dim=1))
    concentration_scores = _standardize_across_tasks(
        torch.stack(adapter_concentrations, dim=1))

    task_scores = (
        tii_weight * tii_scores
        + max_weight * maximum_scores
        + margin_weight * margin_scores
        + entropy_weight * concentration_scores
    )
    task_log_probability = F.log_softmax(
        task_scores / task_temperature, dim=1)

    merged_log_probability = torch.full_like(tii_logits, float('-inf'))
    for task_index, local_log_probability in enumerate(
            local_log_probabilities):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        joint_log_probability = (
            local_log_probability
            + task_log_probability[:, task_index].unsqueeze(1)
        )
        merged_log_probability[:, class_index] = joint_log_probability

    routed_tasks = task_scores.argmax(dim=1)
    return merged_log_probability, routed_tasks