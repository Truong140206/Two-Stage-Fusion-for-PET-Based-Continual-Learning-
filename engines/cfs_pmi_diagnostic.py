import copy
import json
from contextlib import contextmanager

import torch
import torch.nn.functional as F

import utils


def _base_model(model):
    return model.module if hasattr(model, 'module') else model


@contextmanager
def _frozen_model(model):
    states = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    was_training = model.training
    model.eval()
    for parameter, _ in states:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in states:
            parameter.requires_grad_(requires_grad)
        model.train(was_training)


def images_to_patch_tokens(model, images):
    model = _base_model(model)
    tokens = model.patch_embed(images)
    if model.cls_token is not None:
        tokens = torch.cat(
            (model.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
    return model.pos_drop(tokens + model.pos_embed)


def _run_blocks(model, tokens, task_id, start_block=0, stop_block=None):
    model = _base_model(model)
    stop_block = model.depth if stop_block is None else int(stop_block)
    task_mask = torch.full(
        (tokens.shape[0],), int(task_id), dtype=torch.long, device=tokens.device)
    x = tokens
    for block_id in range(int(start_block), stop_block):
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


def patch_tokens_to_output(model, tokens, task_id, start_block=0):
    model = _base_model(model)
    x = _run_blocks(model, tokens, task_id, start_block=start_block)
    x = model.norm(x)
    if model.class_token and model.global_pool == 'token':
        pre_logits = x[:, 0]
    elif model.global_pool == 'avg':
        pre_logits = x.mean(dim=1)
    else:
        raise ValueError('Unsupported global pool: {}'.format(model.global_pool))
    classifier_features = model.fc_norm(model.mlp(pre_logits))
    return {
        'pre_logits': pre_logits,
        'features': pre_logits,
        'logits': model.head(classifier_features),
    }


def _feature_target_loss(output, target_features, labels, seen_classes,
                         class_weight):
    normalized_output = F.normalize(output['pre_logits'].float(), dim=1)
    normalized_target = F.normalize(target_features.float(), dim=1)
    alignment = (1.0 - (normalized_output * normalized_target).sum(dim=1)).mean()
    alignment = alignment + 0.1 * F.mse_loss(
        normalized_output, normalized_target)

    seen_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=labels.device)
    seen_logits = output['logits'].index_select(1, seen_index)
    class_positions = torch.full(
        (output['logits'].shape[1],), -1, dtype=torch.long, device=labels.device)
    class_positions[seen_index] = torch.arange(
        seen_index.numel(), dtype=torch.long, device=labels.device)
    local_labels = class_positions[labels]
    if bool((local_labels < 0).any()):
        raise ValueError('Every inversion label must belong to seen_classes')
    classification = F.cross_entropy(seen_logits, local_labels)
    return alignment + float(class_weight) * classification


def _moment_loss(tokens, reference_mean, reference_var):
    sample_mean = tokens.mean(dim=0)
    sample_var = tokens.var(dim=0, unbiased=False)
    mean_scale = reference_var.mean().sqrt().clamp_min(1e-4)
    mean_loss = F.mse_loss(sample_mean / mean_scale, reference_mean / mean_scale)
    var_loss = F.mse_loss(
        torch.log1p(sample_var.clamp_min(0.0)),
        torch.log1p(reference_var.clamp_min(0.0)),
    )
    return mean_loss + var_loss


def partial_invert_feature_targets(model, target_features, labels, task_id,
                                   seen_classes, token_mean, token_var,
                                   split_block=5, layer_steps=20,
                                   full_steps=40, layer_lr=0.05,
                                   full_lr=0.01, class_weight=0.1,
                                   moment_weight=0.01):
    """Invert final features to the frozen ViT patch-token boundary.

    This mirrors the paper's start-after-patch-embedding setting. It never
    reconstructs or stores historical images or per-example real features.
    """
    model = _base_model(model)
    split_block = max(1, min(int(split_block), model.depth - 1))
    layer_steps = max(0, int(layer_steps))
    full_steps = max(1, int(full_steps))
    token_mean = token_mean.to(target_features.device).float()
    token_var = token_var.to(target_features.device).float().clamp_min(1e-6)
    labels = labels.to(target_features.device).long()

    initial_tokens = (
        token_mean.unsqueeze(0)
        + torch.randn(
            target_features.shape[0], *token_mean.shape,
            device=target_features.device,
        ) * token_var.sqrt().unsqueeze(0)
    )

    latent_steps = layer_steps // 2
    input_steps = layer_steps - latent_steps
    with _frozen_model(model):
        with torch.no_grad():
            initial_latent = _run_blocks(
                model, initial_tokens, task_id, stop_block=split_block)

        latent = initial_latent.detach().clone().requires_grad_(True)
        if latent_steps > 0:
            optimizer = torch.optim.Adam([latent], lr=float(layer_lr))
            for _ in range(latent_steps):
                output = patch_tokens_to_output(
                    model, latent, task_id, start_block=split_block)
                loss = _feature_target_loss(
                    output, target_features, labels, seen_classes, class_weight)
                loss = loss + 1e-4 * F.mse_loss(latent, initial_latent)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        tokens = initial_tokens.detach().clone().requires_grad_(True)
        if input_steps > 0:
            optimizer = torch.optim.Adam([tokens], lr=float(layer_lr))
            for _ in range(input_steps):
                current_latent = _run_blocks(
                    model, tokens, task_id, stop_block=split_block)
                loss = F.mse_loss(current_latent, latent.detach())
                loss = loss + float(moment_weight) * _moment_loss(
                    tokens, token_mean, token_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        optimizer = torch.optim.Adam([tokens], lr=float(full_lr))
        best_tokens = tokens.detach().clone()
        best_loss = float('inf')
        for _ in range(full_steps):
            output = patch_tokens_to_output(model, tokens, task_id)
            loss = _feature_target_loss(
                output, target_features, labels, seen_classes, class_weight)
            loss = loss + float(moment_weight) * _moment_loss(
                tokens, token_mean, token_var)
            loss_value = float(loss.detach().item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_tokens = tokens.detach().clone()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_output = patch_tokens_to_output(model, tokens, task_id)
        final_loss = _feature_target_loss(
            final_output, target_features, labels, seen_classes, class_weight)
        final_loss = final_loss + float(moment_weight) * _moment_loss(
            tokens, token_mean, token_var)
        if float(final_loss.detach().item()) < best_loss:
            best_tokens = tokens.detach().clone()

        output = patch_tokens_to_output(model, best_tokens, task_id)
    return best_tokens, {key: value.detach() for key, value in output.items()}


def _off_diagonal_cosine(features):
    if features.shape[0] < 2:
        return 1.0
    normalized = F.normalize(features.float(), dim=1)
    similarity = normalized.matmul(normalized.t())
    mask = ~torch.eye(
        similarity.shape[0], dtype=torch.bool, device=similarity.device)
    return float(similarity[mask].mean().item())


@torch.no_grad()
def _diagnostic_metrics(output, targets, labels, seen_classes, real_features):
    output_features = output['pre_logits'].float()
    target_cosine = F.cosine_similarity(
        output_features, targets.float(), dim=1).mean()
    seen_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=labels.device)
    seen_logits = output['logits'].index_select(1, seen_index)
    probabilities = F.softmax(seen_logits, dim=1)
    class_positions = torch.full(
        (output['logits'].shape[1],), -1, dtype=torch.long, device=labels.device)
    class_positions[seen_index] = torch.arange(
        seen_index.numel(), dtype=torch.long, device=labels.device)
    local_labels = class_positions[labels]
    confidence = probabilities.gather(1, local_labels.unsqueeze(1)).mean()
    accuracy = probabilities.argmax(dim=1).eq(local_labels).float().mean()

    normalized_output = F.normalize(output_features, dim=1)
    normalized_real = F.normalize(real_features.float(), dim=1)
    normalized_targets = F.normalize(targets.float(), dim=1)
    target_nearest_real_distance = (
        1.0 - normalized_targets.matmul(normalized_real.t()).max(dim=1).values
    ).mean()
    nearest_real_distance = (
        1.0 - normalized_output.matmul(normalized_real.t()).max(dim=1).values
    ).mean()
    return {
        'target_cosine': float(target_cosine.item()),
        'class_accuracy': float(accuracy.item()),
        'class_confidence': float(confidence.item()),
        'nearest_real_cosine_distance': float(nearest_real_distance.item()),
        'target_nearest_real_cosine_distance': float(
            target_nearest_real_distance.item()),
        'target_pairwise_cosine': _off_diagonal_cosine(targets),
        'output_pairwise_cosine': _off_diagonal_cosine(output_features),
    }

def _evaluate_diagnostic(aggregate, reachability_threshold=0.90,
                         tolerance=0.02, diversity_margin=0.005):
    control = aggregate['real_control']
    gaussian = aggregate['gaussian']
    cfs = aggregate['cfs']

    inversion_valid = (
        control['target_cosine'] >= float(reachability_threshold)
        and control['nearest_real_cosine_distance'] <= 0.10
        and control['class_accuracy'] >= 0.98
    )
    checks = {
        'inversion_valid': inversion_valid,
        'reachable_vs_gaussian': (
            cfs['target_cosine'] >= gaussian['target_cosine'] - tolerance
        ),
        'class_consistency': (
            cfs['class_accuracy'] >= gaussian['class_accuracy'] - tolerance
            and cfs['class_confidence'] >= gaussian['class_confidence'] - tolerance
        ),
        'target_manifold': (
            cfs['target_nearest_real_cosine_distance']
            <= gaussian['target_nearest_real_cosine_distance'] + tolerance
        ),
        'output_manifold': (
            cfs['nearest_real_cosine_distance']
            <= gaussian['nearest_real_cosine_distance'] + tolerance
        ),
        'diversity_gain': (
            cfs['output_pairwise_cosine']
            <= gaussian['output_pairwise_cosine'] - diversity_margin
        ),
    }
    if not inversion_valid:
        status = 'INCONCLUSIVE'
    elif all(value for key, value in checks.items()
             if key != 'inversion_valid'):
        status = 'PASS'
    else:
        status = 'FAIL'
    return checks, status



@torch.no_grad()
def _collect_class_observations(model, loader, task_id, device, max_samples):
    model = _base_model(model)
    feature_chunks = []
    token_chunks = []
    remaining = int(max_samples)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        take = min(remaining, images.shape[0])
        images = images[:take]
        feature_chunks.append(
            model(images, task_id=int(task_id), train=False)['pre_logits'].detach())
        token_chunks.append(images_to_patch_tokens(model, images).detach())
        remaining -= take
        if remaining <= 0:
            break
    if not feature_chunks:
        raise RuntimeError('Diagnostic loader did not yield any samples')
    return torch.cat(feature_chunks), torch.cat(token_chunks)


def run_cfs_pmi_diagnostic(model, data_loader_per_cls, class_ids, task_id,
                           args, device):
    model = _base_model(model)
    class_limit = max(1, int(getattr(args, 'cfs_pmi_diag_classes', 4)))
    target_count = max(2, int(getattr(args, 'cfs_pmi_diag_targets_per_class', 5)))
    max_samples = max(
        target_count, int(getattr(args, 'cfs_pmi_diag_real_samples_per_class', 64)))
    selected_classes = [int(class_id) for class_id in class_ids[:class_limit]]
    seen_classes = [int(class_id) for class_id in class_ids]
    diagnostic_args = copy.copy(args)
    diagnostic_args.cfs_sampling = True
    diagnostic_args.cfs_epochs = max(
        1, int(getattr(args, 'cfs_pmi_diag_cfs_epochs', 200)))
    diagnostic_args.cfs_train_max_samples = max_samples
    diagnostic_args.cfs_paper_style = True
    diagnostic_args.cfs_moment_match = False

    observations = {}
    all_tokens = []
    for class_id in selected_classes:
        features, tokens = _collect_class_observations(
            model,
            data_loader_per_cls[class_id]['train'],
            task_id,
            device,
            max_samples,
        )
        observations[class_id] = features
        all_tokens.append(tokens)

    all_tokens = torch.cat(all_tokens, dim=0).float()
    token_mean = all_tokens.mean(dim=0)
    token_var = all_tokens.var(dim=0, unbiased=False).clamp_min(1e-6)
    del all_tokens

    results = {'task_id': int(task_id), 'classes': {}}
    aggregate = {'real_control': [], 'gaussian': [], 'cfs': []}
    for class_id in selected_classes:
        real_features = observations[class_id].float()
        mean = real_features.mean(dim=0)
        covariance = torch.diag(
            real_features.var(dim=0, unbiased=False).clamp_min(1e-5))
        cfs_model = utils.train_cfs_model(
            real_features, diagnostic_args, device)
        distribution = torch.distributions.MultivariateNormal(mean, covariance)
        gaussian_targets = distribution.sample((target_count,))
        real_targets = real_features[:target_count].detach().clone()
        cfs_targets = utils.sample_cfs_features(
            mean,
            covariance,
            target_count,
            diagnostic_args,
            device,
            cfs_model=cfs_model,
        )
        labels = torch.full(
            (target_count,), class_id, dtype=torch.long, device=device)

        class_result = {}
        for name, targets in (
                ('real_control', real_targets),
                ('gaussian', gaussian_targets),
                ('cfs', cfs_targets),
        ):
            _, output = partial_invert_feature_targets(
                model=model,
                target_features=targets,
                labels=labels,
                task_id=task_id,
                seen_classes=seen_classes,
                token_mean=token_mean,
                token_var=token_var,
                split_block=int(getattr(args, 'cfs_pmi_diag_split_block', 1)),
                layer_steps=int(getattr(args, 'cfs_pmi_diag_layer_steps', 100)),
                full_steps=int(getattr(args, 'cfs_pmi_diag_full_steps', 300)),
                layer_lr=float(getattr(args, 'cfs_pmi_diag_layer_lr', 0.05)),
                full_lr=float(getattr(args, 'cfs_pmi_diag_full_lr', 0.01)),
                class_weight=float(getattr(args, 'cfs_pmi_diag_class_weight', 0.1)),
                moment_weight=float(getattr(args, 'cfs_pmi_diag_moment_weight', 0.01)),
            )
            metrics = _diagnostic_metrics(
                output, targets, labels, seen_classes, real_features)
            class_result[name] = metrics
            aggregate[name].append(metrics)
        results['classes'][str(class_id)] = class_result

    results['aggregate'] = {}
    for name, rows in aggregate.items():
        results['aggregate'][name] = {
            key: sum(row[key] for row in rows) / len(rows)
            for key in rows[0]
        }

    checks, status = _evaluate_diagnostic(results['aggregate'])
    results['checks'] = checks
    results['status'] = status
    results['conclusive'] = status != 'INCONCLUSIVE'
    results['pass'] = status == 'PASS'
    print('CFS_PMI_DIAGNOSTIC=' + status)
    print(json.dumps(results, indent=2, sort_keys=True))
    return results
