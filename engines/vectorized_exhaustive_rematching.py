import math

import torch

from engines.exhaustive_rematching import (
    _local_prototype_scores,
    _standardize_task_scores,
)


@torch.no_grad()
def vectorized_exhaustive_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args,
        prototype_bank=None):
    """Preserve exhaustive scoring while batching several task LoRAs together."""
    device = inputs.device
    base_model = model.module if hasattr(model, 'module') else model
    chunk_size = max(
        1, int(getattr(args, 'vectorized_exhaustive_task_chunk_size', 4)))
    chunk_size = min(chunk_size, int(seen_task_count))
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
        tii_task_scores.append(
            tii_logits.index_select(1, class_index).max(dim=1).values)
    tii_task_prior = _standardize_task_scores(
        torch.stack(tii_task_scores, dim=1))

    local_logits_by_task = [None] * seen_task_count
    tii_class_evidence_by_task = [None] * seen_task_count
    prototype_evidence_by_task = [None] * seen_task_count
    adapter_maxima = [None] * seen_task_count
    forward_calls = 0
    batch_size = inputs.shape[0]

    for chunk_start in range(0, seen_task_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seen_task_count)
        task_indices = torch.arange(
            chunk_start, chunk_end, dtype=torch.long, device=device)
        tasks_in_chunk = task_indices.numel()
        expanded_shape = (
            batch_size * tasks_in_chunk,
            *inputs.shape[1:],
        )
        expanded_inputs = inputs.unsqueeze(1).expand(
            batch_size, tasks_in_chunk, *inputs.shape[1:]
        ).reshape(expanded_shape)
        expanded_task_ids = task_indices.unsqueeze(0).expand(
            batch_size, tasks_in_chunk).reshape(-1)
        adapter_output = model(
            expanded_inputs, task_id=expanded_task_ids)
        chunk_logits = adapter_output['logits'].reshape(
            batch_size, tasks_in_chunk, -1)
        chunk_features = None
        if local_prototype_weight > 0.0:
            chunk_features = adapter_output['pre_logits'].reshape(
                batch_size, tasks_in_chunk, -1)
        forward_calls += 1

        for chunk_offset, task_index in enumerate(
                range(chunk_start, chunk_end)):
            task_classes = class_mask[task_index]
            class_index = torch.as_tensor(
                task_classes, dtype=torch.long, device=device)
            local_logits = chunk_logits[:, chunk_offset].index_select(
                1, class_index) / temperature
            tii_local_logits = tii_logits.index_select(
                1, class_index) / tii_class_temperature
            tii_class_evidence = (
                tii_local_logits
                - tii_local_logits.max(dim=1, keepdim=True).values
            )
            prototype_evidence = None
            if local_prototype_weight > 0.0:
                prototype_evidence = _local_prototype_scores(
                    base_model,
                    chunk_features[:, chunk_offset],
                    task_classes,
                    prototype_bank,
                    local_prototype_temperature,
                )

            local_logits_by_task[task_index] = local_logits
            tii_class_evidence_by_task[task_index] = tii_class_evidence
            prototype_evidence_by_task[task_index] = prototype_evidence
            adapter_maxima[task_index] = local_logits.max(dim=1).values

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
    diagnostics = {
        'lora_counts': torch.full(
            (batch_size,), float(seen_task_count),
            dtype=torch.float32, device=device),
        'forward_calls': torch.full(
            (batch_size,), float(forward_calls),
            dtype=torch.float32, device=device),
    }
    return merged_logits, routed_tasks, diagnostics
