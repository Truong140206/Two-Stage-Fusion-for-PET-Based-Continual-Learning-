import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_shared_prototype_bank(feature_memory, seen_classes, device):
    bank = {}
    for class_id in seen_classes:
        memory = feature_memory.get(int(class_id))
        if memory is None or memory.numel() == 0:
            continue
        features = memory.to(device=device, dtype=torch.float32, non_blocking=True)
        bank[int(class_id)] = F.normalize(features, dim=1)
    return bank


def _task_scores_from_tii(tii_logits, class_mask, seen_task_count):
    scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=tii_logits.device)
        scores.append(tii_logits.index_select(1, class_index).max(dim=1).values)
    return torch.stack(scores, dim=1)


@torch.no_grad()
def shared_space_prototype_routing(model, inputs, shared_features, tii_logits,
                                   class_mask, seen_task_count, prototype_bank,
                                   args):
    """Route in one frozen backbone space, then classify with the selected LoRA."""
    if not prototype_bank:
        raise RuntimeError('Shared prototype routing requires a non-empty memory')

    device = inputs.device
    temperature = max(
        1e-6, float(getattr(args, 'shared_prototype_temperature', 0.07)))
    tii_weight = max(
        0.0, float(getattr(args, 'shared_prototype_tii_weight', 0.25)))
    classifier_weight = max(
        0.0, float(getattr(args, 'shared_prototype_classifier_weight', 0.5)))
    tii_temperature = max(
        1e-6, float(getattr(args, 'shared_prototype_tii_temperature', 1.0)))

    query = F.normalize(shared_features.float(), dim=1)
    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    class_scores = torch.full_like(tii_logits, float('-inf'))
    for class_id in seen_classes:
        prototypes = prototype_bank.get(class_id)
        if prototypes is None:
            continue
        similarities = query.matmul(prototypes.transpose(0, 1))
        class_scores[:, class_id] = (
            torch.logsumexp(similarities / temperature, dim=1)
            - math.log(prototypes.shape[0])
        )

    prototype_task_scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        prototype_task_scores.append(
            class_scores.index_select(1, class_index).max(dim=1).values)
    prototype_task_scores = torch.stack(prototype_task_scores, dim=1)

    tii_task_scores = _task_scores_from_tii(
        tii_logits, class_mask, seen_task_count)
    tii_log_prior = F.log_softmax(tii_task_scores / tii_temperature, dim=1)
    routed_tasks = torch.argmax(
        prototype_task_scores + tii_weight * tii_log_prior, dim=1)

    lora_output = model(inputs, task_id=routed_tasks)
    lora_logits = lora_output['logits']
    fused_logits = torch.full_like(tii_logits, float('-inf'))
    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=device)
    fused_logits[:, seen_index] = -20.0

    for task_index in torch.unique(routed_tasks).tolist():
        row_index = torch.nonzero(
            routed_tasks == int(task_index), as_tuple=False).flatten()
        class_index = torch.as_tensor(
            class_mask[int(task_index)], dtype=torch.long, device=device)
        classifier_log_prob = F.log_softmax(
            lora_logits.index_select(0, row_index).index_select(1, class_index),
            dim=1,
        )
        routed_class_scores = class_scores.index_select(
            0, row_index).index_select(1, class_index)
        combined = routed_class_scores + classifier_weight * classifier_log_prob
        row_grid = row_index.unsqueeze(1).expand(-1, class_index.numel())
        class_grid = class_index.unsqueeze(0).expand(row_index.numel(), -1)
        fused_logits[row_grid, class_grid] = combined

    return fused_logits, routed_tasks
