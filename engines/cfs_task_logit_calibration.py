import copy
import math

import torch
import torch.distributed as dist

import utils
from engines.replay_logit_calibration import calibrate_task_logits


def _component_statistics(class_id, cls_mean, cls_cov, device):
    means = cls_mean.get(int(class_id))
    covariances = cls_cov.get(int(class_id))
    if means is None or covariances is None:
        raise RuntimeError(
            'Missing aggregate statistics for class {}'.format(class_id))
    if not isinstance(means, list):
        means = [means]
        covariances = [covariances]

    components = []
    for mean, covariance in zip(means, covariances):
        mean = torch.as_tensor(mean, device=device).float()
        covariance = torch.as_tensor(covariance, device=device).float()
        if covariance.dim() == 1:
            covariance = torch.diag(covariance)
        covariance = covariance + torch.eye(
            mean.numel(), dtype=mean.dtype, device=device) * 1e-4
        components.append((mean, covariance))
    if not components:
        raise RuntimeError(
            'No aggregate components for class {}'.format(class_id))
    return components


@torch.no_grad()
def build_cfs_synthetic_feature_memory(
        cls_mean, cls_cov, cls_cfs_model, class_mask, seen_task_count,
        samples_per_class, args, device):
    """Create transient CFS features from aggregate class statistics only."""
    samples_per_class = max(2, int(samples_per_class))
    memory = {}
    for task_index in range(seen_task_count):
        for class_id in class_mask[task_index]:
            class_id = int(class_id)
            cfs_model = cls_cfs_model.get(class_id)
            if cfs_model is None:
                raise RuntimeError(
                    'Missing CFS model for class {}'.format(class_id))
            components = _component_statistics(
                class_id, cls_mean, cls_cov, device)
            base_count, remainder = divmod(
                samples_per_class, len(components))
            chunks = []
            for component_index, (mean, covariance) in enumerate(components):
                count = base_count + int(component_index < remainder)
                if count <= 0:
                    continue
                chunks.append(utils.sample_cfs_features(
                    mean,
                    covariance,
                    count,
                    args,
                    device,
                    cfs_model=cfs_model,
                    force_cfs=True,
                ))
            features = torch.cat(chunks, dim=0)
            if features.shape[0] != samples_per_class:
                raise RuntimeError(
                    'CFS calibration generated {} instead of {} features '
                    'for class {}'.format(
                        features.shape[0], samples_per_class, class_id))
            memory[class_id] = features.detach().cpu()
    return memory


def _calibration_args(args):
    calibration_args = copy.copy(args)
    mappings = {
        'replay_calibration_steps':
            ('cfs_logit_calibration_steps', 200),
        'replay_calibration_lr':
            ('cfs_logit_calibration_lr', 0.05),
        'replay_calibration_max_scale':
            ('cfs_logit_calibration_max_scale', 1.25),
        'replay_calibration_max_bias':
            ('cfs_logit_calibration_max_bias', 0.5),
        'replay_calibration_regularization':
            ('cfs_logit_calibration_regularization', 0.05),
        'replay_calibration_old_margin_weight':
            ('cfs_logit_calibration_old_margin_weight', 0.5),
        'replay_calibration_old_tolerance':
            ('cfs_logit_calibration_old_tolerance', 0.0),
        'replay_calibration_min_gain':
            ('cfs_logit_calibration_min_gain', 0.0),
        'replay_calibration_max_old_class_drop':
            ('cfs_logit_calibration_max_old_class_drop', 0.0),
        'replay_calibration_batch_size':
            ('cfs_logit_calibration_batch_size', 1024),
    }
    for target_name, (source_name, default) in mappings.items():
        setattr(
            calibration_args,
            target_name,
            getattr(args, source_name, default),
        )
    return calibration_args


def _identity_state(seen_task_count, reason):
    return {
        'version': 1,
        'source': 'cfs_aggregate_statistics',
        'accepted': False,
        'reason': str(reason),
        'scale': [1.0] * int(seen_task_count),
        'bias': [0.0] * int(seen_task_count),
        'fit_samples': 0,
        'report_samples': 0,
        'before': None,
        'after': None,
        'before_loss': None,
        'after_loss': None,
        'worst_old_class_drop': 0.0,
    }


def fit_cfs_task_logit_calibration(
        model, cls_mean, cls_cov, cls_cfs_model, class_mask,
        seen_task_count, args, device):
    """Fit a tiny task affine map without retaining examples or features."""
    if seen_task_count <= 1:
        return _identity_state(seen_task_count, 'single_task_identity')

    state = None
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    if is_main:
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None)
        try:
            samples_per_class = max(
                2, int(getattr(
                    args,
                    'cfs_logit_calibration_samples_per_class',
                    48,
                )))
            fit_memory = build_cfs_synthetic_feature_memory(
                cls_mean,
                cls_cov,
                cls_cfs_model,
                class_mask,
                seen_task_count,
                samples_per_class,
                args,
                device,
            )
            report_memory = build_cfs_synthetic_feature_memory(
                cls_mean,
                cls_cov,
                cls_cfs_model,
                class_mask,
                seen_task_count,
                samples_per_class,
                args,
                device,
            )
            stats = calibrate_task_logits(
                model=model,
                feature_memory=fit_memory,
                class_mask=class_mask,
                seen_task_count=seen_task_count,
                args=_calibration_args(args),
                device=device,
                report_feature_memory=report_memory,
                apply_to_model=False,
                max_samples_per_class=samples_per_class,
            )
            if stats is None:
                state = _identity_state(
                    seen_task_count, 'calibrator_returned_none')
            else:
                accepted = bool(stats['accepted'])
                state = {
                    'version': 1,
                    'source': 'cfs_aggregate_statistics',
                    'accepted': accepted,
                    'reason': (
                        'synthetic_report_gate_pass'
                        if accepted else 'synthetic_report_gate_reject'),
                    'scale': (
                        stats['scale']
                        if accepted else [1.0] * seen_task_count),
                    'bias': (
                        stats['bias']
                        if accepted else [0.0] * seen_task_count),
                    'fit_samples': int(stats['samples']),
                    'report_samples': int(stats['report_samples']),
                    'before': stats['before'],
                    'after': stats['after'],
                    'before_loss': float(stats['before_loss']),
                    'after_loss': float(stats['after_loss']),
                    'worst_old_class_drop': float(
                        stats['worst_old_class_drop']),
                }
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    if dist.is_initialized():
        payload = [state]
        dist.broadcast_object_list(payload, src=0)
        state = payload[0]
    validate_cfs_task_logit_calibration_state(state, seen_task_count)
    return state


def validate_cfs_task_logit_calibration_state(state, seen_task_count):
    """Fail closed on malformed or non-aggregate calibration payloads."""
    if not isinstance(state, dict):
        raise ValueError('CFS task calibration state must be a dictionary')
    allowed_keys = {
        'version', 'source', 'accepted', 'reason', 'scale', 'bias',
        'fit_samples', 'report_samples', 'before', 'after',
        'before_loss', 'after_loss', 'worst_old_class_drop',
    }
    missing_keys = sorted(allowed_keys - set(state))
    if missing_keys:
        raise ValueError(
            'CFS task calibration is missing fields: {}'.format(
                missing_keys))
    unexpected_keys = sorted(set(state) - allowed_keys)
    if unexpected_keys:
        raise ValueError(
            'CFS task calibration contains unexpected payloads: {}'.format(
                unexpected_keys))
    if state.get('version') != 1:
        raise ValueError('CFS task calibration has an invalid version')
    if state.get('source') != 'cfs_aggregate_statistics':
        raise ValueError('CFS task calibration has an invalid source')
    if not isinstance(state.get('accepted'), bool):
        raise ValueError('CFS task calibration accepted flag must be boolean')
    if not isinstance(state.get('reason'), str):
        raise ValueError('CFS task calibration reason must be text')

    scale = state.get('scale')
    bias = state.get('bias')
    if not isinstance(scale, list) or not isinstance(bias, list):
        raise ValueError('CFS task calibration scale/bias must be lists')
    if len(scale) != seen_task_count or len(bias) != seen_task_count:
        raise ValueError(
            'CFS task calibration length does not match seen tasks')
    if not all(isinstance(value, (int, float)) for value in scale + bias):
        raise ValueError(
            'CFS task calibration contains non-scalar parameters')
    if any(not math.isfinite(float(value)) for value in scale + bias):
        raise ValueError('CFS task calibration contains non-finite values')
    if any(float(value) <= 0.0 for value in scale):
        raise ValueError('CFS task calibration scales must be positive')

    for name in ('fit_samples', 'report_samples'):
        value = state.get(name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                'CFS task calibration {} must be a non-negative integer'.format(
                    name))
    for name in ('before', 'after'):
        metrics = state.get(name)
        if metrics is None:
            continue
        if not isinstance(metrics, dict) or set(metrics) != {
                'all', 'old', 'new'}:
            raise ValueError(
                'CFS task calibration {} metrics are malformed'.format(name))
        if not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in metrics.values()):
            raise ValueError(
                'CFS task calibration {} metrics are non-finite'.format(name))
    for name in ('before_loss', 'after_loss'):
        value = state.get(name)
        if value is not None and (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise ValueError(
                'CFS task calibration {} is invalid'.format(name))
    worst_drop = state.get('worst_old_class_drop')
    if (not isinstance(worst_drop, (int, float))
            or not math.isfinite(float(worst_drop))):
        raise ValueError(
            'CFS task calibration worst_old_class_drop is invalid')
    if not state['accepted']:
        if any(abs(float(value) - 1.0) > 1e-8 for value in scale):
            raise ValueError('Rejected CFS calibration must use identity scales')
        if any(abs(float(value)) > 1e-8 for value in bias):
            raise ValueError('Rejected CFS calibration must use zero biases')
    return True