import torch


def complete_with_tii_probability_mass(
        candidate_logits, tii_logits, class_mask, included_task_mask):
    """Fill excluded seen classes with TII mass while preserving top-1."""
    batch_size, class_count = candidate_logits.shape
    seen_task_count = included_task_mask.shape[1]
    device = candidate_logits.device

    included_class_mask = torch.zeros(
        (batch_size, class_count), dtype=torch.bool, device=device)
    seen_class_mask = torch.zeros(
        class_count, dtype=torch.bool, device=device)
    for task_index, task_classes in enumerate(
            class_mask[:seen_task_count]):
        class_index = torch.as_tensor(
            task_classes, dtype=torch.long, device=device)
        seen_class_mask[class_index] = True
        rows = torch.nonzero(
            included_task_mask[:, task_index]).flatten()
        if rows.numel() > 0:
            included_class_mask[
                rows.unsqueeze(1), class_index.unsqueeze(0)
            ] = True
    outside_mask = seen_class_mask.unsqueeze(0) & ~included_class_mask

    candidate_probabilities = torch.softmax(candidate_logits, dim=1)
    masked_tii_logits = tii_logits.masked_fill(
        ~seen_class_mask.unsqueeze(0), float('-inf'))
    tii_probabilities = torch.softmax(masked_tii_logits, dim=1)
    outside_probabilities = tii_probabilities.masked_fill(~outside_mask, 0.0)
    outside_mass = outside_probabilities.sum(dim=1, keepdim=True)
    outside_conditional = outside_probabilities / outside_mass.clamp_min(1e-12)

    candidate_peak = candidate_probabilities.max(dim=1, keepdim=True).values
    outside_peak = outside_conditional.max(dim=1, keepdim=True).values
    top1_safe_mass = 0.99 * candidate_peak / (
        candidate_peak + outside_peak).clamp_min(1e-12)
    completion_mass = torch.minimum(outside_mass, top1_safe_mass)
    completed_probabilities = (
        (1.0 - completion_mass) * candidate_probabilities
        + completion_mass * outside_conditional
    )
    return completed_probabilities.clamp_min(1e-12).log()
