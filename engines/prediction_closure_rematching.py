import torch

from engines.progressive_rematching import _tii_task_prior
from engines.tii_tail_completion import complete_with_tii_probability_mass


@torch.no_grad()
def prediction_closure_tii_tail_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Run adaptive prediction closure and complete excluded tails with TII."""
    batch_size = inputs.shape[0]
    class_count = tii_logits.shape[1]
    device = inputs.device
    temperature = max(
        1e-6, float(getattr(args, 'progressive_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'progressive_tii_prior_weight', 0.3)))
    initial_count = min(
        max(1, int(getattr(args, 'prediction_proposal_initial_count', 2))),
        seen_task_count)
    top_classes = max(
        1, int(getattr(args, 'prediction_proposal_top_classes', 5)))

    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    tii_ranking = torch.argsort(tii_prior, dim=1, descending=True)
    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    class_to_task = torch.full(
        (class_count,), -1, dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = task_index
    top_classes = min(top_classes, len(seen_classes))

    candidate_mask = torch.zeros(
        (batch_size, seen_task_count), dtype=torch.bool, device=device)
    candidate_mask.scatter_(1, tii_ranking[:, :initial_count], True)
    processed_mask = torch.zeros_like(candidate_mask)
    task_logits = torch.empty(
        (batch_size, seen_task_count, class_count),
        dtype=tii_logits.dtype, device=device)
    forward_calls = torch.zeros(
        batch_size, dtype=torch.float32, device=device)

    for _ in range(seen_task_count):
        new_task_mask = candidate_mask & ~processed_mask
        active_rows, active_tasks = torch.nonzero(
            new_task_mask, as_tuple=True)
        if active_rows.numel() == 0:
            break

        expanded_inputs = inputs.index_select(0, active_rows)
        adapter_logits = model(
            expanded_inputs, task_id=active_tasks)['logits']
        task_logits[active_rows, active_tasks] = adapter_logits
        processed_mask |= new_task_mask
        forward_calls[new_task_mask.any(dim=1)] += 1.0

        seen_logits = adapter_logits.index_select(1, seen_class_index)
        top_positions = torch.topk(
            seen_logits, k=top_classes, dim=1).indices
        proposed_tasks = class_to_task[
            seen_class_index[top_positions]]
        proposed_mask = torch.zeros_like(candidate_mask)
        proposed_mask[
            active_rows.unsqueeze(1), proposed_tasks
        ] = True
        additions = proposed_mask & ~candidate_mask
        if not additions.any():
            break
        candidate_mask |= additions

    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = torch.full(
        (batch_size, seen_task_count), float('-inf'),
        dtype=tii_logits.dtype, device=device)
    for task_index in range(seen_task_count):
        rows = torch.nonzero(candidate_mask[:, task_index]).flatten()
        if rows.numel() == 0:
            continue
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        local_logits = task_logits[
            rows, task_index].index_select(1, class_index)
        calibrated_logits = (
            local_logits / temperature
            + prior_weight * tii_prior[
                rows, task_index].unsqueeze(1)
        )
        merged_logits[
            rows.unsqueeze(1), class_index.unsqueeze(0)
        ] = calibrated_logits
        task_evidence[rows, task_index] = calibrated_logits.max(
            dim=1).values

    routed_tasks = task_evidence.argmax(dim=1)
    output_logits = complete_with_tii_probability_mass(
        merged_logits, tii_logits, class_mask, candidate_mask)
    diagnostics = {
        'lora_counts': candidate_mask.sum(dim=1).float(),
        'forward_calls': forward_calls,
        'candidate_mask': candidate_mask,
    }
    return output_logits, routed_tasks, diagnostics
