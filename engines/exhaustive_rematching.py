import math

import torch
import torch.nn.functional as F


def _standardize_task_scores(scores):
    if scores.shape[1] <= 1:
        return torch.zeros_like(scores)
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


def _local_prototype_scores(base_model, pre_logits, task_classes,
                            prototype_bank, temperature):
    query = F.normalize(base_model.fc_norm(pre_logits), dim=1)
    scores = []
    for class_id in task_classes:
        prototypes = prototype_bank.get(int(class_id))
        if prototypes is None:
            raise RuntimeError(
                'Missing local prototypes for class {}'.format(class_id))
        similarities = query.matmul(prototypes.transpose(0, 1))
        scores.append(
            torch.logsumexp(similarities / temperature, dim=1)
            - math.log(prototypes.shape[0])
        )
    scores = torch.stack(scores, dim=1)
    return (
        scores - scores.mean(dim=1, keepdim=True)
    ) / scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)


@torch.no_grad()
def exhaustive_adapter_rematching(model, inputs, tii_logits, class_mask,
                                  seen_task_count, args, prototype_bank=None):
    """Score each task's classes with its own LoRA, then merge all class logits."""
    device = inputs.device
    base_model = model.module if hasattr(model, 'module') else model
    temperature = max(
        1e-6, float(getattr(args, 'exhaustive_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'exhaustive_tii_prior_weight', 0.1)))
    max_calibration_weight = min(
        1.0, max(0.0, float(getattr(
            args, 'exhaustive_max_calibration_weight', 0.0))))
    tii_class_weight = max(
        0.0, float(getattr(args, 'exhaustive_tii_class_weight', 0.0)))
    tii_class_temperature = max(
        1e-6, float(getattr(
            args, 'exhaustive_tii_class_temperature', 1.0)))
    local_prototype_weight = max(
        0.0, float(getattr(
            args, 'exhaustive_local_prototype_weight', 0.0)))
    local_prototype_temperature = max(
        1e-6, float(getattr(
            args, 'exhaustive_local_prototype_temperature', 0.07)))
    if local_prototype_weight > 0.0 and not prototype_bank:
        raise RuntimeError(
            'Local prototype fusion requires restored real-feature memory')

    tii_task_scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        task_logits = tii_logits.index_select(1, class_index)
        tii_task_scores.append(task_logits.max(dim=1).values)
    tii_task_prior = _standardize_task_scores(
        torch.stack(tii_task_scores, dim=1))

    local_logits_by_task = []
    tii_class_evidence_by_task = []
    prototype_evidence_by_task = []
    adapter_maxima = []
    for task_index in range(seen_task_count):
        task_ids = torch.full(
            (inputs.shape[0],), task_index, dtype=torch.long, device=device)
        adapter_output = model(inputs, task_id=task_ids)
        adapter_logits = adapter_output['logits']
        task_classes = class_mask[task_index]
        class_index = torch.as_tensor(
            task_classes, dtype=torch.long, device=device)
        local_logits = adapter_logits.index_select(
            1, class_index) / temperature
        tii_local_logits = tii_logits.index_select(
            1, class_index) / tii_class_temperature
        tii_class_evidence = (
            tii_local_logits - tii_local_logits.max(
                dim=1, keepdim=True).values
        )

        prototype_evidence = None
        if local_prototype_weight > 0.0:
            prototype_evidence = _local_prototype_scores(
                base_model,
                adapter_output['pre_logits'],
                task_classes,
                prototype_bank,
                local_prototype_temperature,
            )

        local_logits_by_task.append(local_logits)
        tii_class_evidence_by_task.append(tii_class_evidence)
        prototype_evidence_by_task.append(prototype_evidence)
        adapter_maxima.append(local_logits.max(dim=1).values)

    adapter_maxima = torch.stack(adapter_maxima, dim=1)
    maximum_mean = adapter_maxima.mean(dim=1, keepdim=True)
    maximum_task_prior = _standardize_task_scores(adapter_maxima)
    maximum_scale_correction = (
        maximum_task_prior - (adapter_maxima - maximum_mean)
    )

    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = []
    for task_index, local_logits in enumerate(local_logits_by_task):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        fused_local_logits = (
            local_logits
            + tii_class_weight * tii_class_evidence_by_task[task_index]
        )
        if local_prototype_weight > 0.0:
            fused_local_logits = (
                fused_local_logits
                + local_prototype_weight
                * prototype_evidence_by_task[task_index]
            )

        # Class fusion may reorder classes, but cannot change task evidence.
        fused_local_logits = (
            fused_local_logits
            + local_logits.max(dim=1, keepdim=True).values
            - fused_local_logits.max(dim=1, keepdim=True).values
        )
        task_bias = (
            prior_weight * tii_task_prior[:, task_index]
            + max_calibration_weight
            * maximum_scale_correction[:, task_index]
        )
        calibrated_logits = fused_local_logits + task_bias.unsqueeze(1)
        merged_logits[:, class_index] = calibrated_logits
        task_evidence.append(calibrated_logits.max(dim=1).values)

    routed_tasks = torch.stack(task_evidence, dim=1).argmax(dim=1)
    return merged_logits, routed_tasks