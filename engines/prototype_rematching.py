import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_prototype_bank(model, feature_memory, seen_classes, device):
    """Normalize stored real features in the classifier feature space."""
    base_model = model.module if hasattr(model, 'module') else model
    bank = {}
    for class_id in seen_classes:
        memory = feature_memory.get(int(class_id))
        if memory is None or memory.numel() == 0:
            continue
        features = memory.to(device=device, dtype=torch.float32, non_blocking=True)
        features = base_model.fc_norm(features)
        bank[int(class_id)] = F.normalize(features, dim=1)
    return bank


def _task_scores_from_tii(tii_logits, class_mask, seen_task_count):
    task_scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=tii_logits.device)
        task_scores.append(tii_logits.index_select(1, class_index).max(dim=1).values)
    return torch.stack(task_scores, dim=1)


@torch.no_grad()
def prototype_assisted_rematching(model, inputs, tii_logits, class_mask,
                                  seen_task_count, prototype_bank, args):
    """Rerank candidate task adapters with real-feature prototype likelihoods."""
    if not prototype_bank:
        raise RuntimeError('Prototype rematching requires a non-empty feature memory')

    device = inputs.device
    base_model = model.module if hasattr(model, 'module') else model
    candidate_count = min(
        seen_task_count,
        max(1, int(getattr(args, 'prototype_candidate_tasks', 2))),
    )
    prototype_temperature = max(
        1e-6, float(getattr(args, 'prototype_temperature', 0.07)))
    classifier_weight = max(
        0.0, float(getattr(args, 'prototype_classifier_weight', 0.5)))
    task_prior_weight = max(
        0.0, float(getattr(args, 'prototype_task_prior_weight', 0.25)))
    task_prior_temperature = max(
        1e-6, float(getattr(args, 'prototype_task_prior_temperature', 1.0)))

    task_scores = _task_scores_from_tii(tii_logits, class_mask, seen_task_count)
    task_log_prior = F.log_softmax(task_scores / task_prior_temperature, dim=1)
    candidate_tasks = torch.topk(
        task_scores, k=candidate_count, dim=1, largest=True).indices

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    fused_logits = torch.full_like(tii_logits, float('-inf'))
    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=device)
    fused_logits[:, seen_index] = -20.0

    for candidate_rank in range(candidate_count):
        routed_tasks = candidate_tasks[:, candidate_rank]
        output = model(inputs, task_id=routed_tasks)
        features = F.normalize(base_model.fc_norm(output['pre_logits']), dim=1)
        classifier_logits = output['logits']

        for task_index in torch.unique(routed_tasks).tolist():
            row_index = torch.nonzero(
                routed_tasks == int(task_index), as_tuple=False).flatten()
            if row_index.numel() == 0:
                continue

            task_classes = [int(class_id) for class_id in class_mask[int(task_index)]]
            class_index = torch.as_tensor(
                task_classes, dtype=torch.long, device=device)
            task_classifier_log_prob = F.log_softmax(
                classifier_logits.index_select(0, row_index).index_select(1, class_index),
                dim=1,
            )
            task_features = features.index_select(0, row_index)
            prototype_scores = []
            for class_id in task_classes:
                prototypes = prototype_bank.get(class_id)
                if prototypes is None:
                    prototype_scores.append(
                        torch.full(
                            (row_index.numel(),), -20.0,
                            dtype=task_features.dtype, device=device))
                    continue
                similarities = task_features.matmul(prototypes.transpose(0, 1))
                class_score = (
                    torch.logsumexp(similarities / prototype_temperature, dim=1)
                    - math.log(prototypes.shape[0])
                )
                prototype_scores.append(class_score)
            prototype_scores = torch.stack(prototype_scores, dim=1)

            prior = task_log_prior.index_select(0, row_index)[:, int(task_index)]
            combined_scores = (
                prototype_scores
                + classifier_weight * task_classifier_log_prob
                + task_prior_weight * prior.unsqueeze(1)
            )
            row_grid = row_index.unsqueeze(1).expand(-1, class_index.numel())
            class_grid = class_index.unsqueeze(0).expand(row_index.numel(), -1)
            previous_scores = fused_logits[row_grid, class_grid]
            fused_logits[row_grid, class_grid] = torch.maximum(
                previous_scores, combined_scores)

    predicted_class = fused_logits.argmax(dim=1)
    class_to_task = torch.full(
        (fused_logits.shape[1],), -1, dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = int(task_index)
    predicted_task = class_to_task.index_select(0, predicted_class)
    return fused_logits, predicted_task
