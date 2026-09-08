import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _task_statistics(logits, class_mask, seen_task_count):
    maximum = []
    energy = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=logits.device)
        task_logits = logits.index_select(1, class_index)
        maximum.append(task_logits.max(dim=1).values)
        energy.append(0.1 * torch.logsumexp(task_logits / 0.1, dim=1))
    statistics = torch.cat(
        [torch.stack(maximum, dim=1), torch.stack(energy, dim=1)], dim=1)
    return F.layer_norm(statistics, (statistics.shape[1],))


class ReplayTaskRouter(nn.Module):
    def __init__(self, feature_dim, seen_task_count, hidden_dim, dropout,
                 class_mask):
        super().__init__()
        self.seen_task_count = int(seen_task_count)
        self.class_mask = [list(classes) for classes in class_mask]
        input_dim = int(feature_dim) + 2 * self.seen_task_count
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.seen_task_count),
        )

    def forward(self, features, tii_logits):
        normalized = F.normalize(features.float(), dim=1)
        statistics = _task_statistics(
            tii_logits, self.class_mask, self.seen_task_count)
        return self.network(torch.cat([normalized, statistics], dim=1))

    @torch.no_grad()
    def predict(self, features, tii_logits):
        self.eval()
        return self(features, tii_logits).argmax(dim=1)


def _collect_router_dataset(original_model, feature_memory, class_mask,
                            seen_task_count, max_samples_per_class, device,
                            batch_size):
    features = []
    task_targets = []
    for task_index in range(seen_task_count):
        for class_id in class_mask[task_index]:
            memory = feature_memory.get(int(class_id))
            if memory is None or memory.numel() == 0:
                raise RuntimeError(
                    'Replay task router is missing shared memory for class {}'.format(
                        class_id))
            memory = memory[:max_samples_per_class].float()
            features.append(memory)
            task_targets.append(torch.full(
                (memory.shape[0],), task_index, dtype=torch.long))
    features = torch.cat(features, dim=0)
    targets = torch.cat(task_targets, dim=0)

    original_model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start:start + batch_size].to(
                device=device, non_blocking=True)
            logits.append(original_model(batch, fc_only=True)['logits'].cpu())
    return features, torch.cat(logits, dim=0), targets


def _stratified_split(targets, validation_ratio, seed):
    generator = torch.Generator().manual_seed(int(seed))
    train_index = []
    validation_index = []
    for task_id in torch.unique(targets).tolist():
        task_index = torch.nonzero(
            targets == int(task_id), as_tuple=False).flatten()
        order = task_index[torch.randperm(task_index.numel(), generator=generator)]
        validation_count = max(
            1, int(round(task_index.numel() * float(validation_ratio))))
        validation_count = min(validation_count, task_index.numel() - 1)
        validation_index.append(order[:validation_count])
        train_index.append(order[validation_count:])
    return torch.cat(train_index), torch.cat(validation_index)


def _baseline_task_prediction(tii_logits, class_mask, seen_task_count):
    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=tii_logits.device)
    masked = torch.full_like(tii_logits, float('-inf'))
    masked[:, seen_index] = tii_logits.index_select(1, seen_index)
    predicted_class = masked.argmax(dim=1)
    class_to_task = torch.full(
        (tii_logits.shape[1],), -1, dtype=torch.long, device=tii_logits.device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=tii_logits.device)
        class_to_task[class_index] = task_index
    return class_to_task.index_select(0, predicted_class)


def train_replay_task_router(original_model, feature_memory, class_mask,
                             seen_task_count, args, device):
    if seen_task_count <= 1:
        return None, None

    max_samples = max(
        2, int(getattr(args, 'replay_router_samples_per_class', 48)))
    batch_size = max(16, int(getattr(args, 'replay_router_batch_size', 256)))
    features, tii_logits, targets = _collect_router_dataset(
        original_model, feature_memory, class_mask, seen_task_count,
        max_samples, device, batch_size)
    train_index, validation_index = _stratified_split(
        targets,
        validation_ratio=float(getattr(args, 'replay_router_validation_ratio', 0.25)),
        seed=int(getattr(args, 'seed', 42)) + seen_task_count,
    )

    router = ReplayTaskRouter(
        feature_dim=features.shape[1],
        seen_task_count=seen_task_count,
        hidden_dim=int(getattr(args, 'replay_router_hidden_dim', 256)),
        dropout=float(getattr(args, 'replay_router_dropout', 0.1)),
        class_mask=class_mask[:seen_task_count],
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(getattr(args, 'replay_router_lr', 0.001)),
        weight_decay=float(getattr(args, 'replay_router_weight_decay', 0.01)),
    )
    epochs = max(1, int(getattr(args, 'replay_router_epochs', 50)))
    patience = max(1, int(getattr(args, 'replay_router_patience', 8)))

    features = features.to(device)
    tii_logits = tii_logits.to(device)
    targets = targets.to(device)
    train_index = train_index.to(device)
    validation_index = validation_index.to(device)
    best_accuracy = -1.0
    best_state = None
    stale_epochs = 0

    for _ in range(epochs):
        router.train()
        order = train_index[torch.randperm(train_index.numel(), device=device)]
        for start in range(0, order.numel(), batch_size):
            index = order[start:start + batch_size]
            output = router(
                features.index_select(0, index),
                tii_logits.index_select(0, index),
            )
            loss = F.cross_entropy(output, targets.index_select(0, index))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        router.eval()
        with torch.no_grad():
            validation_prediction = router.predict(
                features.index_select(0, validation_index),
                tii_logits.index_select(0, validation_index),
            )
            validation_accuracy = float(
                validation_prediction.eq(
                    targets.index_select(0, validation_index)
                ).float().mean().mul(100.0).item())
        if validation_accuracy > best_accuracy + 1e-8:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(router.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    router.load_state_dict(best_state)
    router.eval()
    with torch.no_grad():
        validation_tii_logits = tii_logits.index_select(0, validation_index)
        baseline_prediction = _baseline_task_prediction(
            validation_tii_logits, class_mask, seen_task_count)
        baseline_accuracy = float(
            baseline_prediction.eq(
                targets.index_select(0, validation_index)
            ).float().mean().mul(100.0).item())
    minimum_gain = float(getattr(args, 'replay_router_min_validation_gain', 0.25))
    accepted = best_accuracy >= baseline_accuracy + minimum_gain
    stats = {
        'accepted': accepted,
        'samples': int(features.shape[0]),
        'validation_samples': int(validation_index.numel()),
        'baseline_validation_accuracy': baseline_accuracy,
        'router_validation_accuracy': best_accuracy,
    }
    return (router if accepted else None), stats
