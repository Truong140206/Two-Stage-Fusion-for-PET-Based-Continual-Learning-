import copy

import torch
import torch.nn.functional as F

from engines.replay_task_router import ReplayTaskRouter, _baseline_task_prediction


def _standardize(scores):
    if scores.shape[1] <= 1:
        return torch.zeros_like(scores)
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


@torch.no_grad()
def _teacher_task_scores(model, inputs, tii_logits, class_mask,
                         seen_task_count, args):
    tii_scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=inputs.device)
        tii_scores.append(
            tii_logits.index_select(1, class_index).max(dim=1).values)
    tii_prior = _standardize(torch.stack(tii_scores, dim=1))

    prior_weight = max(
        0.0, float(getattr(args, 'distilled_router_teacher_prior', 0.3)))
    temperature = max(
        1e-6, float(getattr(args, 'distilled_router_teacher_temperature', 1.0)))
    evidence = []
    for task_index in range(seen_task_count):
        task_ids = torch.full(
            (inputs.shape[0],), task_index, dtype=torch.long,
            device=inputs.device)
        logits = model(inputs, task_id=task_ids)['logits']
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=inputs.device)
        local_logits = logits.index_select(1, class_index) / temperature
        evidence.append(
            local_logits.max(dim=1).values
            + prior_weight * tii_prior[:, task_index])
    return torch.stack(evidence, dim=1)


@torch.no_grad()
def _collect_teacher_dataset(model, original_model, data_loader_per_cls,
                             class_mask, seen_task_count, args, device):
    max_per_class = max(
        2, int(getattr(args, 'distilled_router_samples_per_class', 8)))
    features = []
    tii_logits = []
    teacher_scores = []
    task_targets = []
    class_targets = []

    model.eval()
    original_model.eval()
    for task_index in range(seen_task_count):
        for class_id in class_mask[task_index]:
            collected = 0
            for inputs, _ in data_loader_per_cls[class_id]['train']:
                remaining = max_per_class - collected
                if remaining <= 0:
                    break
                inputs = inputs[:remaining].to(device, non_blocking=True)
                original_output = original_model(inputs)
                current_tii_logits = original_output['logits']
                features.append(original_output['pre_logits'].detach().cpu())
                tii_logits.append(current_tii_logits.detach().cpu())
                teacher_scores.append(_teacher_task_scores(
                    model, inputs, current_tii_logits, class_mask,
                    seen_task_count, args).detach().cpu())
                task_targets.append(torch.full(
                    (inputs.shape[0],), task_index, dtype=torch.long))
                class_targets.append(torch.full(
                    (inputs.shape[0],), int(class_id), dtype=torch.long))
                collected += inputs.shape[0]

    return (
        torch.cat(features, dim=0),
        torch.cat(tii_logits, dim=0),
        torch.cat(teacher_scores, dim=0),
        torch.cat(task_targets, dim=0),
        torch.cat(class_targets, dim=0),
    )


def _class_stratified_split(class_targets, validation_ratio, seed):
    generator = torch.Generator().manual_seed(int(seed))
    train_index = []
    validation_index = []
    for class_id in torch.unique(class_targets).tolist():
        index = torch.nonzero(
            class_targets == int(class_id), as_tuple=False).flatten()
        order = index[torch.randperm(index.numel(), generator=generator)]
        validation_count = max(
            1, int(round(index.numel() * float(validation_ratio))))
        validation_count = min(validation_count, index.numel() - 1)
        validation_index.append(order[:validation_count])
        train_index.append(order[validation_count:])
    return torch.cat(train_index), torch.cat(validation_index)


def _topk_recall(logits, targets, k):
    k = min(max(1, int(k)), logits.shape[1])
    candidates = torch.topk(logits, k=k, dim=1).indices
    return float(
        candidates.eq(targets.unsqueeze(1)).any(dim=1).float().mean()
        .mul(100.0).item())


def train_distilled_task_router(model, original_model, data_loader_per_cls,
                                class_mask, seen_task_count, args, device):
    if seen_task_count <= 1:
        return None, None

    features, tii_logits, teacher_scores, targets, class_targets = (
        _collect_teacher_dataset(
            model, original_model, data_loader_per_cls, class_mask,
            seen_task_count, args, device))
    train_index, validation_index = _class_stratified_split(
        class_targets,
        validation_ratio=float(
            getattr(args, 'distilled_router_validation_ratio', 0.25)),
        seed=int(getattr(args, 'seed', 42)) + seen_task_count,
    )

    router = ReplayTaskRouter(
        feature_dim=features.shape[1],
        seen_task_count=seen_task_count,
        hidden_dim=int(getattr(args, 'distilled_router_hidden_dim', 256)),
        dropout=float(getattr(args, 'distilled_router_dropout', 0.1)),
        class_mask=class_mask[:seen_task_count],
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(getattr(args, 'distilled_router_lr', 0.001)),
        weight_decay=float(
            getattr(args, 'distilled_router_weight_decay', 0.01)),
    )
    batch_size = max(
        16, int(getattr(args, 'distilled_router_batch_size', 256)))
    epochs = max(1, int(getattr(args, 'distilled_router_epochs', 50)))
    patience = max(1, int(getattr(args, 'distilled_router_patience', 8)))
    top_k = min(
        seen_task_count,
        max(1, int(getattr(args, 'selective_candidate_tasks', 3))))
    teacher_weight = min(
        1.0, max(0.0, float(
            getattr(args, 'distilled_router_teacher_weight', 0.3))))
    distill_temperature = max(
        1e-6, float(getattr(args, 'distilled_router_temperature', 1.0)))

    features = features.to(device)
    tii_logits = tii_logits.to(device)
    teacher_scores = teacher_scores.to(device)
    targets = targets.to(device)
    train_index = train_index.to(device)
    validation_index = validation_index.to(device)

    best_score = -1.0
    best_state = None
    best_top1 = 0.0
    best_topk = 0.0
    stale_epochs = 0
    for _ in range(epochs):
        router.train()
        order = train_index[torch.randperm(train_index.numel(), device=device)]
        for start in range(0, order.numel(), batch_size):
            index = order[start:start + batch_size]
            output = router(
                features.index_select(0, index),
                tii_logits.index_select(0, index))
            hard_loss = F.cross_entropy(output, targets.index_select(0, index))
            teacher_probability = F.softmax(
                teacher_scores.index_select(0, index) / distill_temperature,
                dim=1)
            soft_loss = F.kl_div(
                F.log_softmax(output / distill_temperature, dim=1),
                teacher_probability,
                reduction='batchmean') * (distill_temperature ** 2)
            loss = (1.0 - teacher_weight) * hard_loss + teacher_weight * soft_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        router.eval()
        with torch.no_grad():
            output = router(
                features.index_select(0, validation_index),
                tii_logits.index_select(0, validation_index))
            validation_targets = targets.index_select(0, validation_index)
            top1 = _topk_recall(output, validation_targets, 1)
            topk = _topk_recall(output, validation_targets, top_k)
            score = topk + 0.01 * top1
        if score > best_score + 1e-8:
            best_score = score
            best_top1 = top1
            best_topk = topk
            best_state = copy.deepcopy(router.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    router.load_state_dict(best_state)
    router.eval()
    with torch.no_grad():
        validation_tii = tii_logits.index_select(0, validation_index)
        validation_targets = targets.index_select(0, validation_index)
        baseline_prediction = _baseline_task_prediction(
            validation_tii, class_mask, seen_task_count)
        baseline_top1 = float(
            baseline_prediction.eq(validation_targets).float().mean()
            .mul(100.0).item())
        tii_task_scores = []
        for task_index in range(seen_task_count):
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            tii_task_scores.append(
                validation_tii.index_select(1, class_index).max(dim=1).values)
        baseline_topk = _topk_recall(
            torch.stack(tii_task_scores, dim=1), validation_targets, top_k)

    minimum_gain = float(
        getattr(args, 'distilled_router_min_validation_gain', 0.0))
    accepted = best_topk >= baseline_topk + minimum_gain
    stats = {
        'accepted': accepted,
        'samples': int(features.shape[0]),
        'validation_samples': int(validation_index.numel()),
        'baseline_top1': baseline_top1,
        'baseline_topk': baseline_topk,
        'router_top1': best_top1,
        'router_topk': best_topk,
    }
    return (router if accepted else None), stats
