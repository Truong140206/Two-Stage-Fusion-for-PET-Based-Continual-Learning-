import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import utils


REPLAY_CACHE_VERSION = 2


def _base_model(model):
    return model.module if hasattr(model, 'module') else model


def _cache_path(cache_dir, class_id):
    return os.path.join(cache_dir, 'class_{:04d}.pth'.format(int(class_id)))


def _load_cache_file(path):
    return utils.load_checkpoint(path, map_location='cpu')


class ReplayAnchorMemory:
    def __init__(self, cache_dir, class_ids):
        self.cache_dir = cache_dir
        self.class_images = {}
        self.class_tasks = {}
        for class_id in class_ids:
            path = _cache_path(cache_dir, class_id)
            if not os.path.exists(path):
                continue
            payload = _load_cache_file(path)
            images = payload.get('images')
            if images is None or images.ndim != 4 or images.shape[0] == 0:
                continue
            self.class_images[int(class_id)] = images.to(dtype=torch.uint8, device='cpu')
            self.class_tasks[int(class_id)] = int(payload['task_id'])

        self.class_ids = sorted(self.class_images)

    def __len__(self):
        return sum(images.shape[0] for images in self.class_images.values())

    @property
    def empty(self):
        return not self.class_ids

    def sample(self, batch_size, device):
        if self.empty or batch_size <= 0:
            return None

        class_positions = torch.randint(len(self.class_ids), (batch_size,))
        images = []
        labels = []
        task_ids = []
        for class_position in class_positions.tolist():
            class_id = self.class_ids[class_position]
            class_images = self.class_images[class_id]
            image_id = int(torch.randint(class_images.shape[0], (1,)).item())
            images.append(class_images[image_id])
            labels.append(class_id)
            task_ids.append(self.class_tasks[class_id])

        images = torch.stack(images, dim=0).to(device=device, dtype=torch.float32) / 255.0
        if bool(torch.rand(()) < 0.5):
            images = torch.flip(images, dims=(3,))
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        task_ids = torch.tensor(task_ids, dtype=torch.long, device=device)
        return images, labels, task_ids


def build_replay_anchor_memory(args, old_classes):
    cache_dir = getattr(args, 'replay_anchor_cache_dir', '')
    if not cache_dir:
        cache_dir = os.path.join(args.output_dir, 'replay_anchor_cache')
    memory = ReplayAnchorMemory(cache_dir, old_classes)
    if utils.is_main_process():
        print(
            'Replay-anchor memory:',
            'classes=', len(memory.class_ids),
            'images=', len(memory),
            'cache=', cache_dir,
        )
    return memory


def _forward_tokens_until(model, images, task_id, stop_block):
    x = model.patch_embed(images)
    if model.cls_token is not None:
        x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    task_mask = torch.full(
        (x.shape[0],), int(task_id), dtype=torch.long, device=x.device)
    for block_id in range(stop_block):
        if model.lora_layer is not None and block_id < model.lora_depth:
            x = model.blocks[block_id](
                x,
                lora=model.lora_layer,
                task_id=task_mask,
                depth_id=block_id,
                train=False,
                old=False,
            )
        else:
            x = model.blocks[block_id](x)
    return x


def _forward_tokens_from(model, tokens, task_id, start_block):
    x = tokens
    task_mask = torch.full(
        (x.shape[0],), int(task_id), dtype=torch.long, device=x.device)
    for block_id in range(start_block, model.depth):
        if model.lora_layer is not None and block_id < model.lora_depth:
            x = model.blocks[block_id](
                x,
                lora=model.lora_layer,
                task_id=task_mask,
                depth_id=block_id,
                train=False,
                old=False,
            )
        else:
            x = model.blocks[block_id](x)
    x = model.norm(x)
    if model.class_token and model.global_pool == 'token':
        return x[:, 0]
    return x.mean(dim=1)


def _feature_alignment_loss(features, target_features):
    features = F.normalize(features.float(), dim=1)
    targets = F.normalize(target_features.float(), dim=1)
    cosine = 1.0 - (features * targets).sum(dim=1)
    return cosine.mean() + 0.1 * F.mse_loss(features, targets)


def _total_variation(images):
    horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
    vertical = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


def _initial_image_logits(batch_size, image_size, device):
    low_resolution = max(16, image_size // 4)
    images = torch.rand(batch_size, 3, low_resolution, low_resolution, device=device)
    images = F.interpolate(
        images, size=(image_size, image_size), mode='bilinear', align_corners=False)
    images = images.clamp(1e-4, 1.0 - 1e-4)
    return torch.logit(images).detach().requires_grad_(True)


def invert_cfs_features(model, target_features, class_id, task_id, args, device):
    model = _base_model(model)
    image_size = int(getattr(args, 'input_size', 224))
    split_block = int(getattr(args, 'replay_inversion_split_block', 5))
    split_block = max(1, min(split_block, model.depth - 1))
    layer_steps = max(0, int(getattr(args, 'replay_inversion_layer_steps', 200)))
    full_steps = max(1, int(getattr(args, 'replay_inversion_full_steps', 600)))
    latent_steps = layer_steps // 4
    input_steps = layer_steps - latent_steps
    layer_lr = float(getattr(args, 'replay_inversion_layer_lr', 0.1))
    full_lr = float(getattr(args, 'replay_inversion_full_lr', 0.01))
    class_weight = float(getattr(args, 'replay_inversion_class_weight', 0.1))
    tv_weight = float(getattr(args, 'replay_inversion_tv_weight', 0.0005))

    parameter_states = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    was_training = model.training
    model.eval()
    for parameter, _ in parameter_states:
        parameter.requires_grad_(False)

    raw_images = _initial_image_logits(target_features.shape[0], image_size, device)
    try:
        with torch.no_grad():
            initial_tokens = _forward_tokens_until(
                model, raw_images.sigmoid(), task_id, split_block)

        latent_tokens = initial_tokens.detach().clone().requires_grad_(True)
        if latent_steps > 0:
            latent_optimizer = torch.optim.Adam([latent_tokens], lr=layer_lr)
            for _ in range(latent_steps):
                features = _forward_tokens_from(
                    model, latent_tokens, task_id, split_block)
                loss = _feature_alignment_loss(features, target_features)
                loss = loss + 1e-4 * F.mse_loss(latent_tokens, initial_tokens)
                latent_optimizer.zero_grad()
                loss.backward()
                latent_optimizer.step()

        if input_steps > 0:
            input_optimizer = torch.optim.Adam([raw_images], lr=layer_lr)
            for _ in range(input_steps):
                images = raw_images.sigmoid()
                tokens = _forward_tokens_until(model, images, task_id, split_block)
                token_loss = F.mse_loss(tokens, latent_tokens.detach())
                loss = token_loss + tv_weight * _total_variation(images)
                input_optimizer.zero_grad()
                loss.backward()
                input_optimizer.step()

        full_optimizer = torch.optim.Adam([raw_images], lr=full_lr)
        target_labels = torch.full(
            (target_features.shape[0],), int(class_id), dtype=torch.long, device=device)
        for _ in range(full_steps):
            images = raw_images.sigmoid()
            output = model(images, task_id=int(task_id), train=False)
            feature_loss = _feature_alignment_loss(output['pre_logits'], target_features)
            class_loss = F.cross_entropy(output['logits'], target_labels)
            loss = feature_loss + class_weight * class_loss + tv_weight * _total_variation(images)
            full_optimizer.zero_grad()
            loss.backward()
            full_optimizer.step()

        images = raw_images.sigmoid().detach().clamp(0.0, 1.0)
        return (images * 255.0).round().to(dtype=torch.uint8, device='cpu')
    finally:
        for parameter, requires_grad in parameter_states:
            parameter.requires_grad_(requires_grad)
        model.train(was_training)


@torch.no_grad()
def _select_diverse_candidates(candidates, cfs_model, count, tau):
    if candidates.shape[0] <= count or cfs_model is None:
        return candidates[:count]
    embeddings = cfs_model(candidates.float())
    first = int(torch.randint(candidates.shape[0], (1,), device=candidates.device).item())
    selected = [first]
    available = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
    available[first] = False
    score_sum = torch.exp(embeddings.matmul(embeddings[first]) / tau)
    while len(selected) < count:
        scores = (score_sum / len(selected)).masked_fill(~available, float('inf'))
        next_id = int(torch.argmin(scores).item())
        selected.append(next_id)
        available[next_id] = False
        score_sum = score_sum + torch.exp(embeddings.matmul(embeddings[next_id]) / tau)
    return candidates.index_select(
        0, torch.tensor(selected, dtype=torch.long, device=candidates.device))


@torch.no_grad()
def _sample_target_features(class_id, cls_mean, cls_cov, cls_cfs_model, count, args, device):
    mean = cls_mean[class_id]
    cov = cls_cov[class_id]
    cfs_model = cls_cfs_model.get(class_id)
    if not isinstance(mean, (list, tuple)):
        covariance = cov
        if covariance.ndim == 1:
            covariance = torch.diag(covariance)
        return utils.sample_cfs_features(
            torch.as_tensor(mean, device=device).float(),
            torch.as_tensor(covariance, device=device).float(),
            count,
            args,
            device,
            cfs_model=cfs_model,
        )

    multiplier = max(2, int(getattr(args, 'cfs_candidate_multiplier', 3)))
    candidate_count = max(count, count * multiplier)
    candidates = []
    valid_clusters = [
        cluster_id for cluster_id, variance in enumerate(cov)
        if float(torch.as_tensor(variance).float().mean()) > 0.0
    ]
    if not valid_clusters:
        raise RuntimeError('No valid feature cluster for class {}'.format(class_id))
    for candidate_id in range(candidate_count):
        cluster_id = valid_clusters[candidate_id % len(valid_clusters)]
        cluster_mean = torch.as_tensor(mean[cluster_id], device=device).float()
        cluster_var = torch.as_tensor(cov[cluster_id], device=device).float().clamp_min(1e-5)
        candidates.append(cluster_mean + torch.randn_like(cluster_mean) * cluster_var.sqrt())
    candidates = torch.stack(candidates, dim=0)
    if cfs_model is not None:
        cfs_model = cfs_model.to(device)
    return _select_diverse_candidates(
        candidates,
        cfs_model,
        count,
        max(float(getattr(args, 'cfs_tau', 1.0)), 1e-6),
    )


def generate_task_replay_cache(model, task_id, class_ids, cls_mean, cls_cov,
                               cls_cfs_model, args, device):
    if not bool(getattr(args, 'replay_anchor_ctird', False)):
        return
    cache_dir = getattr(args, 'replay_anchor_cache_dir', '')
    if not cache_dir:
        cache_dir = os.path.join(args.output_dir, 'replay_anchor_cache')
    images_per_class = max(1, int(getattr(args, 'replay_anchor_images_per_class', 5)))
    candidate_multiplier = max(
        1, int(getattr(args, 'replay_inversion_candidate_multiplier', 2)))
    inversion_count = images_per_class * candidate_multiplier
    class_ids = [int(class_id) for class_id in class_ids]
    seen_class_ids = sorted(int(class_id) for class_id in cls_mean)

    if utils.is_main_process():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        for class_id in class_ids:
            path = _cache_path(cache_dir, class_id)
            if os.path.exists(path):
                payload = _load_cache_file(path)
                cached_images = payload.get('images')
                if (payload.get('version') == REPLAY_CACHE_VERSION
                        and cached_images is not None
                        and cached_images.shape[0] >= images_per_class):
                    print('Replay inversion cache hit:', path)
                    continue

            targets = _sample_target_features(
                class_id,
                cls_mean,
                cls_cov,
                cls_cfs_model,
                inversion_count,
                args,
                device,
            )
            images = invert_cfs_features(
                model, targets, class_id, task_id, args, device)
            with torch.no_grad():
                validation_output = _base_model(model)(
                    images.to(device=device, dtype=torch.float32) / 255.0,
                    task_id=int(task_id),
                    train=False,
                )
                task_index = torch.as_tensor(
                    seen_class_ids, dtype=torch.long, device=device)
                task_probabilities = F.softmax(
                    validation_output['logits'].index_select(1, task_index), dim=1)
                target_position = int(seen_class_ids.index(class_id))
                target_confidence = task_probabilities[:, target_position]
                teacher_accuracy = float(task_probabilities.argmax(dim=1).eq(
                    target_position).float().mean().item())
                correct = task_probabilities.argmax(dim=1).eq(target_position)
                ranking_score = target_confidence + correct.float()
                selected = torch.topk(
                    ranking_score,
                    k=min(images_per_class, images.shape[0]),
                    largest=True,
                ).indices.cpu()
                images = images.index_select(0, selected)
                target_confidence = target_confidence.index_select(
                    0, selected.to(device))
                retained_accuracy = float(correct.index_select(
                    0, selected.to(device)).float().mean().item())
            payload = {
                'version': REPLAY_CACHE_VERSION,
                'images': images,
                'class_id': int(class_id),
                'task_id': int(task_id),
                'teacher_accuracy': teacher_accuracy,
                'teacher_confidence': target_confidence.cpu(),
                'retained_accuracy': retained_accuracy,
            }
            temporary_path = path + '.tmp'
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
            print(
                'Replay inversion saved:',
                path,
                'images=', images.shape[0],
                'teacher_acc=', round(teacher_accuracy * 100.0, 2),
                'retained_acc=', round(retained_accuracy * 100.0, 2),
                'retained_conf=', round(
                    float(target_confidence.mean().item()), 4),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    utils.distributed_barrier()


def _relation_distribution(features, temperature):
    features = F.normalize(features.float(), dim=1)
    logits = features.matmul(features.t()) / max(float(temperature), 1e-6)
    logits.fill_diagonal_(float('-inf'))
    return F.softmax(logits, dim=1)


def replay_anchor_relation_loss(model, memory, current_task_id, seen_classes, args, device):
    base_model = _base_model(model)
    zero = next(base_model.parameters()).new_zeros(())
    if memory is None or memory.empty:
        return zero, 0, 0.0

    batch_size = max(2, int(getattr(args, 'replay_anchor_batch_size', 20)))
    sampled = memory.sample(batch_size, device)
    if sampled is None:
        return zero, 0, 0.0
    images, labels, teacher_task_ids = sampled

    with torch.no_grad():
        teacher_output = model(images, task_id=teacher_task_ids, train=False)
        seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=device)
        teacher_logits = teacher_output['logits'].index_select(1, seen_index)
        teacher_probabilities = F.softmax(teacher_logits, dim=1)
        class_positions = torch.full(
            (teacher_output['logits'].shape[1],), -1, dtype=torch.long, device=device)
        class_positions[seen_index] = torch.arange(seen_index.numel(), device=device)
        target_positions = class_positions[labels]
        target_confidence = teacher_probabilities.gather(
            1, target_positions.unsqueeze(1)).squeeze(1)
        predicted_positions = teacher_probabilities.argmax(dim=1)
        confidence_threshold = float(
            getattr(args, 'replay_anchor_teacher_confidence', 0.2))
        keep = (
            predicted_positions.eq(target_positions)
            & target_confidence.ge(confidence_threshold)
        )

    kept = int(keep.sum().item())
    if kept < 3:
        return zero, kept, float(target_confidence.mean().item())

    images = images[keep]
    teacher_features = teacher_output['features'][keep].detach()
    student_output = model(images, task_id=int(current_task_id), train=True)
    temperature = float(getattr(args, 'replay_anchor_temperature', 1.0))
    teacher_relation = _relation_distribution(teacher_features, temperature)
    student_relation = _relation_distribution(student_output['features'], temperature)
    loss = F.kl_div(
        student_relation.clamp_min(1e-12).log(),
        teacher_relation,
        reduction='batchmean',
    )
    return loss, kept, float(target_confidence[keep].mean().item())
