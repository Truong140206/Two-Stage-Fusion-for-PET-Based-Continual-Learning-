import torch
from torch.nn import functional as F

from engines.progressive_rematching import _tii_task_prior


@torch.no_grad()
def soft_mixture_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Mix the TII top-k LoRA residuals in one model forward pass."""
    device = inputs.device
    batch_size = inputs.shape[0]
    top_k = min(
        int(seen_task_count),
        max(1, int(getattr(args, 'soft_mixture_top_k', 4))),
    )
    task_temperature = max(
        1e-6, float(getattr(args, 'soft_mixture_task_temperature', 1.0)))
    logit_temperature = max(
        1e-6, float(getattr(args, 'soft_mixture_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'soft_mixture_tii_prior_weight', 0.3)))

    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    top_scores, candidate_tasks = torch.topk(
        tii_prior, k=top_k, dim=1, largest=True, sorted=True)
    mixture_weights = F.softmax(top_scores / task_temperature, dim=1)

    output = model(
        inputs,
        task_id=candidate_tasks[:, 0],
        ensemble_id=candidate_tasks,
        ensemble_weights=mixture_weights,
    )
    mixed_logits = output['logits'] / logit_temperature

    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        local_logits = mixed_logits.index_select(1, class_index)
        calibrated_logits = (
            local_logits
            + prior_weight * tii_prior[:, task_index].unsqueeze(1)
        )
        merged_logits[:, class_index] = calibrated_logits
        task_evidence.append(calibrated_logits.max(dim=1).values)

    routed_tasks = torch.stack(task_evidence, dim=1).argmax(dim=1)
    diagnostics = {
        'candidate_tasks': candidate_tasks,
        'mixture_weights': mixture_weights,
        'lora_counts': torch.full(
            (batch_size,), float(top_k), dtype=torch.float32, device=device),
        'forward_calls': torch.ones(
            batch_size, dtype=torch.float32, device=device),
    }
    return merged_logits, routed_tasks, diagnostics


@torch.no_grad()
def soft_mixture_hard_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Route with a soft LoRA mixture, then classify with one hard LoRA."""
    _, routed_tasks, diagnostics = soft_mixture_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args)

    hard_output = model(inputs, task_id=routed_tasks)
    hard_logits = hard_output['logits'] / max(
        1e-6, float(getattr(args, 'soft_mixture_logit_temperature', 1.0)))
    merged_logits = torch.full_like(tii_logits, float('-inf'))
    seen_classes = []
    for task_index in range(seen_task_count):
        seen_classes.extend(class_mask[task_index])
    seen_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=inputs.device)
    merged_logits[:, seen_index] = hard_logits.index_select(1, seen_index)

    diagnostics = dict(diagnostics)
    diagnostics['lora_counts'] = diagnostics['lora_counts'] + 1.0
    diagnostics['forward_calls'] = diagnostics['forward_calls'] + 1.0
    return merged_logits, routed_tasks, diagnostics


def _scale_invariant_top1_margin(logits, class_index):
    local_logits = logits.index_select(1, class_index)
    centered = local_logits - local_logits.mean(dim=1, keepdim=True)
    normalized = centered / local_logits.std(
        dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    top_count = min(2, normalized.shape[1])
    top_values = torch.topk(normalized, k=top_count, dim=1).values
    if top_count == 1:
        return torch.full_like(top_values[:, 0], float('inf'))
    return top_values[:, 0] - top_values[:, 1]


@torch.no_grad()
def soft_hard_confidence_selector_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Select soft or hard logits using a label-free normalized margin."""
    soft_logits, routed_tasks, diagnostics = soft_mixture_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args)

    hard_output = model(inputs, task_id=routed_tasks)
    hard_output_logits = hard_output['logits'] / max(
        1e-6, float(getattr(args, 'soft_mixture_logit_temperature', 1.0)))
    hard_logits = torch.full_like(tii_logits, float('-inf'))
    seen_classes = []
    for task_index in range(seen_task_count):
        seen_classes.extend(class_mask[task_index])
    seen_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=inputs.device)
    hard_logits[:, seen_index] = hard_output_logits.index_select(
        1, seen_index)

    soft_margin = _scale_invariant_top1_margin(soft_logits, seen_index)
    hard_margin = _scale_invariant_top1_margin(hard_logits, seen_index)
    select_hard = hard_margin > soft_margin
    selected_logits = torch.where(
        select_hard.unsqueeze(1), hard_logits, soft_logits)

    diagnostics = dict(diagnostics)
    diagnostics['lora_counts'] = diagnostics['lora_counts'] + 1.0
    diagnostics['forward_calls'] = diagnostics['forward_calls'] + 1.0
    diagnostics['soft_logits'] = soft_logits
    diagnostics['hard_logits'] = hard_logits
    diagnostics['select_hard'] = select_hard
    diagnostics['soft_margin'] = soft_margin
    diagnostics['hard_margin'] = hard_margin
    return selected_logits, routed_tasks, diagnostics


@torch.no_grad()
def soft_mixture_local_hard_refinement(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Use hard LoRA rankings locally without changing soft task evidence."""
    soft_logits, routed_tasks, diagnostics = soft_mixture_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args)

    hard_output = model(inputs, task_id=routed_tasks)
    hard_logits = hard_output['logits'] / max(
        1e-6, float(getattr(args, 'soft_mixture_logit_temperature', 1.0)))
    refined_logits = soft_logits.clone()

    for task_index in range(seen_task_count):
        sample_index = routed_tasks.eq(task_index).nonzero(
            as_tuple=False).flatten()
        if sample_index.numel() == 0:
            continue
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=inputs.device)
        soft_local = soft_logits.index_select(
            0, sample_index).index_select(1, class_index)
        hard_local = hard_logits.index_select(
            0, sample_index).index_select(1, class_index)

        soft_max = soft_local.max(dim=1, keepdim=True).values
        soft_std = soft_local.std(
            dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        hard_max = hard_local.max(dim=1, keepdim=True).values
        hard_std = hard_local.std(
            dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        aligned_hard = (hard_local - hard_max) / hard_std * soft_std + soft_max
        refined_logits[
            sample_index.unsqueeze(1), class_index.unsqueeze(0)] = aligned_hard

    diagnostics = dict(diagnostics)
    diagnostics['lora_counts'] = diagnostics['lora_counts'] + 1.0
    diagnostics['forward_calls'] = diagnostics['forward_calls'] + 1.0
    diagnostics['soft_logits'] = soft_logits
    diagnostics['refined_logits'] = refined_logits
    return refined_logits, routed_tasks, diagnostics
