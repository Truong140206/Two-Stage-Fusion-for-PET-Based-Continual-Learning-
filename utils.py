"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import math
import hashlib
import json
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F

_SEMANTIC_SIMILARITY_CACHE = {}
_SEMANTIC_EMBEDDING_CACHE = {}
_SEMANTIC_BACKEND_WARNING_EMITTED = set()
_SEMANTIC_FEATURE_ADAPTER_CACHE = {}

_CIFAR100_COARSE_GROUPS = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man_made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non_insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

_CIFAR100_CLASS_TO_COARSE = {
    class_name: group_name
    for group_name, class_names in _CIFAR100_COARSE_GROUPS.items()
    for class_name in class_names
}

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        distributed_barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def distributed_barrier():
    if not is_dist_avail_and_initialized():
        return
    if dist.get_backend() == 'nccl' and torch.cuda.is_available():
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return
        except TypeError:
            pass
    dist.barrier()


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def load_checkpoint(*args, **kwargs):
    try:
        return torch.load(*args, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(*args, **kwargs)

class CFSContrastiveMLP(torch.nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(hidden_features, hidden_features),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


def use_cfs_sampling(args):
    return bool(getattr(args, 'cfs_sampling', False))




def use_cfs_boundary_replay(args):
    return use_cfs_sampling(args) and bool(getattr(args, 'cfs_boundary_replay', False))

def use_semantic_distillation(args):
    return bool(getattr(args, 'semantic_distill', False))



def use_semantic_projection(args):
    return bool(getattr(args, 'semantic_projection', False))

def use_semantic_feature_adapter(args):
    return bool(getattr(args, 'semantic_feature_adapter', False))
def _semantic_tokens(class_name):
    text = str(class_name).lower()
    for char in ['_', '-', '/', '.', ',', ';', ':', '(', ')', '[', ']']:
        text = text.replace(char, ' ')
    tokens = [token for token in text.split() if token]
    if not tokens:
        tokens = [text] if text else ['unknown']
    return tokens


def _semantic_key(class_name):
    return '_'.join(_semantic_tokens(class_name))


def _hashed_text_vector(tokens, dim):
    vector = torch.zeros(dim, dtype=torch.float32)
    for token in tokens:
        pieces = [token]
        if len(token) > 3:
            pieces.extend(token[i:i + 3] for i in range(len(token) - 2))
        for piece in pieces:
            digest = hashlib.md5(piece.encode('utf-8')).digest()
            index = int.from_bytes(digest[:4], byteorder='little') % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
    return vector




def _load_semantic_class_name_file(path):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError('semantic class name file not found: {}'.format(path))

    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.json'):
            data = json.load(f)
        else:
            data = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(maxsplit=1)
                data.append(parts[1] if len(parts) == 2 else parts[0])

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        return [str(v) for v in data]
    raise ValueError('semantic class name file must be a JSON dict/list or a text file')


def resolve_semantic_class_names(class_names, args):
    mapping = _load_semantic_class_name_file(getattr(args, 'semantic_class_name_file', ''))
    if mapping is None:
        return class_names

    resolved = []
    for idx, name in enumerate(class_names):
        key = str(name)
        if isinstance(mapping, dict):
            resolved.append(mapping.get(key, key))
        elif idx < len(mapping):
            resolved.append(mapping[idx])
        else:
            resolved.append(key)
    return resolved
def _semantic_backend(args):
    return str(getattr(args, 'semantic_backend', 'hash')).lower()


def _semantic_prompt_templates(args):
    raw_templates = str(getattr(args, 'semantic_clip_templates', 'a photo of a {}.'))
    templates = [item.strip() for item in raw_templates.split('|') if item.strip()]
    return templates or ['a photo of a {}.']


def _format_semantic_prompt(template, class_name):
    class_name = str(class_name).replace('_', ' ')
    if '{}' in template:
        return template.format(class_name)
    return '{} {}'.format(template.rstrip(), class_name)


def _resize_semantic_embeddings(embeddings, dim):
    if embeddings.shape[1] == dim:
        return embeddings
    if embeddings.shape[1] > dim:
        return F.normalize(embeddings[:, :dim], p=2, dim=1)
    pad = torch.zeros(
        embeddings.shape[0],
        dim - embeddings.shape[1],
        dtype=embeddings.dtype,
        device=embeddings.device)
    return F.normalize(torch.cat([embeddings, pad], dim=1), p=2, dim=1)


def _build_hash_semantic_embeddings(class_names, device, dim):
    embeddings = []
    for class_name in class_names:
        tokens = _semantic_tokens(class_name)
        name_vector = F.normalize(_hashed_text_vector(tokens, dim).unsqueeze(0), p=2, dim=1).squeeze(0)
        coarse_group = _CIFAR100_CLASS_TO_COARSE.get(_semantic_key(class_name))
        if coarse_group is not None:
            group_vector = F.normalize(
                _hashed_text_vector(['cifar100', coarse_group], dim).unsqueeze(0),
                p=2,
                dim=1).squeeze(0)
            embeddings.append((0.45 * name_vector) + (0.89 * group_vector))
        else:
            embeddings.append(name_vector)
    embeddings = torch.stack(embeddings, dim=0).to(device)
    return F.normalize(embeddings, p=2, dim=1)


@torch.no_grad()
def _build_clip_semantic_embeddings(class_names, args, device):
    model_name = str(getattr(args, 'semantic_clip_model', 'ViT-B-16'))
    pretrained = str(getattr(args, 'semantic_clip_pretrained', 'openai'))
    templates = tuple(_semantic_prompt_templates(args))
    class_names = tuple(str(name).replace('_', ' ') for name in class_names)
    cache_key = ('clip-native', str(device), class_names, model_name, pretrained, templates)
    if cache_key in _SEMANTIC_EMBEDDING_CACHE:
        return _SEMANTIC_EMBEDDING_CACHE[cache_key]

    prompts_per_class = [
        [_format_semantic_prompt(template, class_name) for template in templates]
        for class_name in class_names
    ]

    open_clip_error = None
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        embeddings = []
        for prompts in prompts_per_class:
            tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(tokens).float()
            text_features = F.normalize(text_features, p=2, dim=1)
            embeddings.append(F.normalize(text_features.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))
        embeddings = torch.stack(embeddings, dim=0).to(device)
        _SEMANTIC_EMBEDDING_CACHE[cache_key] = embeddings
        return embeddings
    except Exception as exc:
        open_clip_error = exc

    try:
        import clip
        clip_model_name = model_name.replace('-', '/') if model_name.startswith('ViT-') else model_name
        model, _ = clip.load(clip_model_name, device=device, jit=False)
        model.eval()
        embeddings = []
        for prompts in prompts_per_class:
            tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(tokens).float()
            text_features = F.normalize(text_features, p=2, dim=1)
            embeddings.append(F.normalize(text_features.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))
        embeddings = torch.stack(embeddings, dim=0).to(device)
        _SEMANTIC_EMBEDDING_CACHE[cache_key] = embeddings
        return embeddings
    except Exception as exc:
        message = (
            'semantic_backend=clip requires open_clip_torch or clip. '
            'Install with: pip install open_clip_torch. '
            'open_clip error: {}; clip error: {}'.format(open_clip_error, exc)
        )
        if _semantic_backend(args) == 'clip':
            raise RuntimeError(message)
        if message not in _SEMANTIC_BACKEND_WARNING_EMITTED:
            print('Warning:', message)
            _SEMANTIC_BACKEND_WARNING_EMITTED.add(message)
        return None


def build_semantic_class_embeddings(class_names, device, dim=512, args=None):
    if not class_names:
        return None

    backend = _semantic_backend(args) if args is not None else 'hash'
    if backend in ('clip', 'auto'):
        clip_embeddings = _build_clip_semantic_embeddings(class_names, args, device)
        if clip_embeddings is not None:
            return _resize_semantic_embeddings(clip_embeddings, dim)

    return _build_hash_semantic_embeddings(class_names, device, dim)


def get_semantic_similarity_matrix(args, device):
    if not use_semantic_distillation(args):
        return None

    class_names = getattr(args, 'class_names', None)
    if not class_names:
        return None

    class_names = resolve_semantic_class_names(class_names, args)
    dim = max(1, int(getattr(args, 'semantic_dim', 512)))
    sharpness = max(1e-6, float(getattr(args, 'semantic_sharpness', 1.0)))
    cache_key = (id(args), str(device), tuple(str(name) for name in class_names), dim, sharpness, _semantic_backend(args), str(getattr(args, 'semantic_clip_model', '')), str(getattr(args, 'semantic_clip_pretrained', '')), str(getattr(args, 'semantic_clip_templates', '')))
    if cache_key in _SEMANTIC_SIMILARITY_CACHE:
        return _SEMANTIC_SIMILARITY_CACHE[cache_key]

    embeddings = build_semantic_class_embeddings(class_names, device, dim=dim, args=args)
    if embeddings is None:
        return None

    sim = torch.mm(embeddings, embeddings.t()).clamp(min=0.0)
    sim.fill_diagonal_(1.0)
    if sharpness != 1.0:
        sim = sim.pow(sharpness)

    _SEMANTIC_SIMILARITY_CACHE[cache_key] = sim
    return sim


def apply_semantic_relation_distillation(relation_target, labels, args, device):
    semantic_sim = get_semantic_similarity_matrix(args, device)
    if semantic_sim is None:
        return relation_target

    labels = labels.long().to(device)
    if labels.numel() == 0 or labels.max().item() >= semantic_sim.shape[0]:
        return relation_target

    weights = semantic_sim.index_select(0, labels).index_select(1, labels)
    mode = getattr(args, 'semantic_mode', 'adaptive_gate')

    if mode == 'adaptive_gate':
        class_top_k = int(getattr(args, 'semantic_top_k', 5))
        if class_top_k > 0:
            keep_count = min(class_top_k + 1, semantic_sim.shape[1])
            _, global_top_ids = torch.topk(semantic_sim, k=keep_count, dim=1)
            class_keep = torch.zeros_like(semantic_sim, dtype=torch.bool)
            class_keep.scatter_(1, global_top_ids, True)
            batch_keep = class_keep.index_select(0, labels).index_select(1, labels)
            weights = weights * batch_keep.float()

        alpha = float(getattr(args, 'semantic_alpha', 0.05))
        alpha = max(0.0, min(1.0, alpha))
        gated_target = relation_target * (1.0 + alpha * weights)
        return gated_target / gated_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if mode == 'topk_mix':
        class_top_k = int(getattr(args, 'semantic_top_k', 5))
        if class_top_k > 0:
            keep_count = min(class_top_k + 1, semantic_sim.shape[1])
            _, global_top_ids = torch.topk(semantic_sim, k=keep_count, dim=1)
            class_keep = torch.zeros_like(semantic_sim, dtype=torch.bool)
            class_keep.scatter_(1, global_top_ids, True)
            batch_keep = class_keep.index_select(0, labels).index_select(1, labels)
            weights = weights * batch_keep.float()

        semantic_target = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        empty_rows = semantic_target.sum(dim=1, keepdim=True) <= 0
        if empty_rows.any():
            semantic_target = torch.where(empty_rows, relation_target, semantic_target)

        alpha = float(getattr(args, 'semantic_alpha', 0.1))
        alpha = max(0.0, min(1.0, alpha))
        mixed_target = (1.0 - alpha) * relation_target + alpha * semantic_target
        return mixed_target / mixed_target.sum(dim=1, keepdim=True).clamp_min(1e-12)

    floor = float(getattr(args, 'semantic_floor', 0.2))
    floor = max(0.0, min(1.0, floor))
    weights = floor + (1.0 - floor) * weights

    weighted_target = relation_target * weights
    weighted_target = weighted_target / weighted_target.sum(dim=1, keepdim=True).clamp_min(1e-12)

    alpha = float(getattr(args, 'semantic_alpha', 0.1))
    alpha = max(0.0, min(1.0, alpha))
    if alpha < 1.0:
        weighted_target = (1.0 - alpha) * relation_target + alpha * weighted_target
        weighted_target = weighted_target / weighted_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weighted_target



def _feature_mean_for_class(cls_id, cls_mean, device):
    mean = cls_mean.get(cls_id)
    if mean is None:
        return None
    if isinstance(mean, list):
        valid = [torch.as_tensor(item).float().to(device) for item in mean]
        valid = [item for item in valid if torch.isfinite(item).all()]
        if not valid:
            return None
        return torch.stack(valid, dim=0).mean(dim=0)
    mean = torch.as_tensor(mean).float().to(device)
    return mean if torch.isfinite(mean).all() else None


@torch.no_grad()
def update_semantic_feature_adapter(args, cls_mean, device, available_classes=None):
    if not use_semantic_feature_adapter(args):
        return None

    class_names = getattr(args, 'class_names', None)
    if not class_names:
        return None
    class_names = resolve_semantic_class_names(class_names, args)

    if available_classes is None:
        available_classes = sorted(int(c) for c in cls_mean.keys())
    else:
        available_classes = sorted(int(c) for c in available_classes if int(c) in cls_mean)

    feature_means = []
    train_ids = []
    for cls_id in available_classes:
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        mean = _feature_mean_for_class(cls_id, cls_mean, device)
        if mean is None:
            continue
        feature_means.append(mean)
        train_ids.append(cls_id)

    min_classes = max(2, int(getattr(args, 'semantic_adapter_min_classes', 5)))
    if len(train_ids) < min_classes:
        return None

    feature_dim = feature_means[0].numel()
    semantic_dim = max(1, int(getattr(args, 'semantic_adapter_dim', getattr(args, 'semantic_dim', 512))))
    semantic_embeddings = build_semantic_class_embeddings(class_names, device, dim=semantic_dim, args=args)
    if semantic_embeddings is None:
        return None

    source_ids = torch.tensor(train_ids, dtype=torch.long, device=device)
    source = semantic_embeddings.index_select(0, source_ids).float()
    target = torch.stack(feature_means, dim=0).float().to(device)
    target = F.normalize(target, p=2, dim=1)
    ones = torch.ones(source.shape[0], 1, dtype=source.dtype, device=device)
    source_aug = torch.cat([source, ones], dim=1)

    ridge = float(getattr(args, 'semantic_adapter_ridge', 1e-2))
    identity = torch.eye(source_aug.shape[1], dtype=source_aug.dtype, device=device)
    identity[-1, -1] = 0.0
    lhs = source_aug.t().matmul(source_aug) + ridge * identity
    rhs = source_aug.t().matmul(target)
    try:
        weight = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        weight = torch.linalg.pinv(lhs).matmul(rhs)

    all_source = torch.cat([
        semantic_embeddings.float(),
        torch.ones(semantic_embeddings.shape[0], 1, dtype=semantic_embeddings.dtype, device=device)
    ], dim=1)
    adapted = F.normalize(all_source.matmul(weight), p=2, dim=1)

    blend = float(getattr(args, 'semantic_adapter_blend', 1.0))
    blend = max(0.0, min(1.0, blend))
    if semantic_embeddings.shape[1] == feature_dim:
        base = F.normalize(semantic_embeddings.float(), p=2, dim=1)
        adapted = F.normalize((1.0 - blend) * base + blend * adapted, p=2, dim=1)

    cache_key = (id(args), str(device), feature_dim)
    _SEMANTIC_FEATURE_ADAPTER_CACHE[cache_key] = adapted.detach()
    return adapted


def _semantic_projection_embeddings(args, device, feature_dim):
    if use_semantic_feature_adapter(args):
        adapted = _SEMANTIC_FEATURE_ADAPTER_CACHE.get((id(args), str(device), feature_dim))
        if adapted is not None:
            return adapted.to(device)

    class_names = getattr(args, 'class_names', None)
    if not class_names:
        return None
    class_names = resolve_semantic_class_names(class_names, args)
    return build_semantic_class_embeddings(class_names, device, dim=feature_dim, args=args)

def _rotate_features_between_semantics(features, source_sem, target_sem):
    source_sem = F.normalize(source_sem.float(), p=2, dim=0)
    target_sem = F.normalize(target_sem.float(), p=2, dim=0)
    cosine = torch.dot(source_sem, target_sem).clamp(-1.0, 1.0)

    if (1.0 - cosine).abs() < 1e-6:
        return features

    if (1.0 + cosine).abs() < 1e-6:
        reflected = features - 2.0 * torch.matmul(features, source_sem).unsqueeze(1) * source_sem.unsqueeze(0)
        return reflected

    sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(1e-12))
    basis_1 = source_sem
    basis_2 = F.normalize(target_sem - cosine * basis_1, p=2, dim=0)

    coord_1 = torch.matmul(features, basis_1).unsqueeze(1)
    coord_2 = torch.matmul(features, basis_2).unsqueeze(1)
    in_plane = coord_1 * basis_1.unsqueeze(0) + coord_2 * basis_2.unsqueeze(0)
    out_plane = features - in_plane
    rotated_plane = (
        (coord_1 * cosine - coord_2 * sine) * basis_1.unsqueeze(0)
        + (coord_1 * sine + coord_2 * cosine) * basis_2.unsqueeze(0)
    )
    return out_plane + rotated_plane


def _diag_variance_from_cov(cov, device):
    if cov is None:
        return None
    cov = torch.as_tensor(cov).float().to(device)
    if cov.dim() == 2:
        return torch.diag(cov).clamp_min(1e-5)
    return cov.clamp_min(1e-5)


@torch.no_grad()
def semantic_project_features(features, source_mean, target_mean, source_cls, target_cls, args, device,
                              source_cov=None, target_cov=None):
    feature_dim = features.shape[-1]
    embeddings = _semantic_projection_embeddings(args, device, feature_dim)
    if embeddings is None or source_cls >= embeddings.shape[0] or target_cls >= embeddings.shape[0]:
        return features

    source_sem = embeddings[source_cls].float()
    target_sem = embeddings[target_cls].float()
    features = features.float().to(device)
    source_mean = source_mean.float().to(device)
    target_mean = target_mean.float().to(device)
    centered = features - source_mean.unsqueeze(0)
    projection_mode = str(getattr(args, 'semantic_projection_mode', 'mean_shift')).lower()

    if projection_mode == 'covariance_transfer':
        source_var = _diag_variance_from_cov(source_cov, device)
        target_var = _diag_variance_from_cov(target_cov, device)
        if (
            source_var is not None and target_var is not None
            and source_var.shape[0] == centered.shape[1]
            and target_var.shape[0] == centered.shape[1]
        ):
            max_scale = float(getattr(args, 'semantic_cov_transfer_max_scale', 2.0))
            min_scale = float(getattr(args, 'semantic_cov_transfer_min_scale', 0.5))
            scale = torch.sqrt(target_var / source_var).clamp(min=min_scale, max=max_scale)
            transferred = centered * scale.unsqueeze(0)
        else:
            transferred = centered

        strength = float(getattr(args, 'semantic_projection_strength', 1.0))
        strength = max(0.0, min(1.0, strength))
        residual = (1.0 - strength) * centered + strength * transferred
        return target_mean.unsqueeze(0) + residual

    if projection_mode == 'paper':
        feature_norm = features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        feature_dir = F.normalize(features, p=2, dim=1)
        rotated_dir = _rotate_features_between_semantics(feature_dir, source_sem, target_sem)
        alpha = float(getattr(args, 'semantic_projection_alpha', getattr(args, 'semantic_alpha', 0.1)))
        alpha = max(0.0, min(1.0, alpha))
        adjusted_dir = F.normalize(
            (1.0 - alpha) * rotated_dir + alpha * target_sem.unsqueeze(0),
            p=2,
            dim=1)
        if bool(getattr(args, 'semantic_projection_preserve_norm', False)):
            return adjusted_dir * feature_norm
        return adjusted_dir

    bridge = F.normalize(source_sem + target_sem, p=2, dim=0)
    if not torch.isfinite(bridge).all() or bridge.norm() < 1e-6:
        return features

    rotated = 2.0 * torch.matmul(centered, bridge).unsqueeze(1) * bridge.unsqueeze(0) - centered

    strength = float(getattr(args, 'semantic_projection_strength', 1.0))
    strength = max(0.0, min(1.0, strength))
    projected = target_mean.unsqueeze(0) + strength * rotated + (1.0 - strength) * centered
    return projected

def _class_stat_for_target_cluster(target_cls, target_mean, cls_mean, cls_cov, device):
    target_cov = cls_cov.get(target_cls)
    if target_cov is None:
        return None
    if isinstance(target_cov, list):
        target_means = cls_mean.get(target_cls)
        if not isinstance(target_means, list):
            valid = [cov for cov in target_cov if torch.as_tensor(cov).float().mean().item() != 0]
            return valid[0].float().to(device) if valid else None
        target_mean = target_mean.float().to(device)
        best_idx = None
        best_dist = None
        for idx, mean in enumerate(target_means):
            if idx >= len(target_cov):
                continue
            cov = torch.as_tensor(target_cov[idx]).float()
            if cov.mean().item() == 0:
                continue
            dist = torch.norm(torch.as_tensor(mean).float().to(device) - target_mean).item()
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            return None
        target_cov = target_cov[best_idx]
    return torch.as_tensor(target_cov).float().to(device)


@torch.no_grad()
def _filter_projected_features_for_target(projected, target_mean, target_cov, num_samples, args, device):
    if projected.shape[0] <= num_samples:
        return projected[:num_samples]

    target_mean = target_mean.float().to(device)
    target_cov = torch.as_tensor(target_cov).float().to(device)
    if target_cov.dim() == 2:
        diag_var = torch.diag(target_cov)
    else:
        diag_var = target_cov
    diag_var = diag_var.clamp_min(1e-5)

    centered = projected.float().to(device) - target_mean.unsqueeze(0)
    maha_score = (centered.pow(2) / diag_var.unsqueeze(0)).mean(dim=1)

    cosine_weight = float(getattr(args, 'semantic_projection_filter_cosine_weight', 0.1))
    if cosine_weight > 0 and target_mean.norm().item() > 1e-8:
        cosine_distance = 1.0 - F.cosine_similarity(projected.float(), target_mean.unsqueeze(0), dim=1)
        score = maha_score + cosine_weight * cosine_distance
    else:
        score = maha_score

    keep_count = min(num_samples, projected.shape[0])
    keep_ids = torch.topk(score, k=keep_count, largest=False).indices
    return projected.index_select(0, keep_ids)


@torch.no_grad()
def sample_semantic_projected_features(target_cls, target_mean, num_samples, args, device,
                                       cls_mean, cls_cov, cls_cfs_model=None, available_classes=None):
    if not use_semantic_projection(args) or num_samples <= 0:
        return None

    feature_dim = target_mean.numel()
    embeddings = _semantic_projection_embeddings(args, device, feature_dim)
    if embeddings is None or target_cls >= embeddings.shape[0]:
        return None

    if available_classes is None:
        available_classes = list(cls_mean.keys())
    available_classes = [int(c) for c in available_classes if int(c) != int(target_cls) and int(c) in cls_mean and int(c) in cls_cov]
    if not available_classes:
        return None

    source_ids = torch.tensor(available_classes, dtype=torch.long, device=device)
    sim = embeddings[target_cls].matmul(embeddings.index_select(0, source_ids).t()).clamp(min=0.0)
    top_k = int(getattr(args, 'semantic_projection_top_k', getattr(args, 'semantic_top_k', 5)))
    top_k = max(1, min(top_k, source_ids.numel()))
    _, order = torch.topk(sim, k=top_k)
    selected_sources = source_ids.index_select(0, order).tolist()

    candidate_multiplier = 1
    if bool(getattr(args, 'semantic_projection_filter', False)):
        candidate_multiplier = max(1, int(getattr(args, 'semantic_projection_filter_multiplier', 3)))
    requested_samples = max(num_samples, num_samples * candidate_multiplier)

    chunks = []
    remaining = requested_samples
    for idx, source_cls in enumerate(selected_sources):
        take = remaining // (len(selected_sources) - idx)
        if take <= 0:
            continue
        source_mean = cls_mean[source_cls]
        source_cov = cls_cov[source_cls]
        if isinstance(source_mean, list) or isinstance(source_cov, list):
            valid_clusters = [idx for idx, var in enumerate(source_cov) if torch.as_tensor(var).float().mean().item() != 0]
            if not valid_clusters:
                continue
            cluster_idx = valid_clusters[torch.randint(len(valid_clusters), (1,), device=device).item()]
            source_mean = source_mean[cluster_idx]
            source_cov = source_cov[cluster_idx]
        source_mean = source_mean.float().to(device)
        source_cov = source_cov.float().to(device)
        if source_cov.dim() == 1:
            source_cov = torch.diag(source_cov) + 1e-4 * torch.eye(source_mean.shape[0], device=device)
        source_samples = sample_cfs_features(
            source_mean, source_cov, take, args, device,
            cfs_model=cls_cfs_model.get(source_cls) if cls_cfs_model is not None else None)
        target_cov = _class_stat_for_target_cluster(target_cls, target_mean, cls_mean, cls_cov, device)
        chunks.append(semantic_project_features(
            source_samples, source_mean, target_mean, source_cls, target_cls, args, device,
            source_cov=source_cov, target_cov=target_cov))
        remaining -= take

    if not chunks:
        return None
    projected = torch.cat(chunks, dim=0)

    if bool(getattr(args, 'semantic_projection_filter', False)):
        target_cov = _class_stat_for_target_cluster(target_cls, target_mean, cls_mean, cls_cov, device)
        if target_cov is not None:
            projected = _filter_projected_features_for_target(projected, target_mean, target_cov, num_samples, args, device)

    if projected.shape[0] < num_samples:
        pad = projected[torch.randint(projected.shape[0], (num_samples - projected.shape[0],), device=device)]
        projected = torch.cat([projected, pad], dim=0)
    return projected[:num_samples]


def train_cfs_model(features, args, device, force_cfs=False):
    if (not force_cfs and not use_cfs_sampling(args)) or features.shape[0] < 2:
        return None

    with torch.enable_grad():
        features = features.detach().float().to(device)
        max_samples = int(getattr(args, 'cfs_train_max_samples', 1024))
        if features.shape[0] > max_samples:
            sample_ids = torch.randperm(features.shape[0], device=device)[:max_samples]
            features = features[sample_ids]

        in_features = features.shape[1]
        hidden_features = int(getattr(args, 'cfs_hidden_dim', 512))
        hidden_features = max(1, min(hidden_features, in_features))
        model = CFSContrastiveMLP(in_features, hidden_features).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(getattr(args, 'cfs_lr', 0.01)),
            momentum=float(getattr(args, 'cfs_momentum', 0.9)),
        )
        epochs = int(getattr(args, 'cfs_epochs', 50))
        batch_size = min(int(getattr(args, 'cfs_batch_size', 256)), features.shape[0])
        tau = float(getattr(args, 'cfs_tau', 1.0))

        model.train()
        for _ in range(epochs):
            perm = torch.randperm(features.shape[0], device=device)
            for start in range(0, features.shape[0], batch_size):
                batch = features[perm[start:start + batch_size]]
                if batch.shape[0] < 2:
                    continue
                out = model(batch)
                sim = torch.mm(out, out.t()) / tau
                sim.fill_diagonal_(float('-inf'))
                loss = torch.logsumexp(sim, dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model


@torch.no_grad()
def _filter_cfs_candidate_features(candidates, mean, cov, keep_count, args, device):
    if candidates.shape[0] <= keep_count:
        return candidates[:keep_count]

    mean = mean.float().to(device)
    cov = torch.as_tensor(cov).float().to(device)
    if cov.dim() == 2:
        diag_var = torch.diag(cov)
    else:
        diag_var = cov
    diag_var = diag_var.clamp_min(1e-5)

    candidates = candidates.float().to(device)
    centered = candidates - mean.unsqueeze(0)
    maha_score = (centered.pow(2) / diag_var.unsqueeze(0)).mean(dim=1)

    cosine_weight = float(getattr(args, 'cfs_filter_cosine_weight', 0.0))
    if cosine_weight > 0 and mean.norm().item() > 1e-8:
        cosine_distance = 1.0 - F.cosine_similarity(candidates, mean.unsqueeze(0), dim=1)
        score = maha_score + cosine_weight * cosine_distance
    else:
        score = maha_score

    keep_count = min(keep_count, candidates.shape[0])
    keep_ids = torch.topk(score, k=keep_count, largest=False).indices
    return candidates.index_select(0, keep_ids)


@torch.no_grad()
def _sample_filtered_gaussian(distribution, mean, cov, keep_count, args, device):
    if keep_count <= 0:
        return distribution.sample(sample_shape=(0,)).to(device)
    if not bool(getattr(args, 'cfs_distribution_filter', False)):
        return distribution.sample(sample_shape=(keep_count,)).to(device)

    filter_multiplier = max(1, int(getattr(args, 'cfs_filter_multiplier', 3)))
    raw_count = max(keep_count, keep_count * filter_multiplier)
    candidates = distribution.sample(sample_shape=(raw_count,)).to(device)
    return _filter_cfs_candidate_features(candidates, mean, cov, keep_count, args, device)


@torch.no_grad()
def match_cfs_feature_moments(features, mean, cov):
    """Restore class marginal moments after contrastive feature selection."""
    if features.shape[0] <= 1:
        return features

    features = features.float()
    target_mean = torch.as_tensor(mean, device=features.device).float()
    covariance = torch.as_tensor(cov, device=features.device).float()
    target_variance = (
        torch.diag(covariance) if covariance.dim() == 2 else covariance)
    target_std = target_variance.clamp_min(0.0).sqrt()

    selected_mean = features.mean(dim=0)
    selected_std = features.std(dim=0, unbiased=True)
    stable_std = selected_std.clamp_min(1e-8)
    matched = (
        (features - selected_mean.unsqueeze(0))
        * (target_std / stable_std).unsqueeze(0)
        + target_mean.unsqueeze(0)
    )
    constant_dimensions = selected_std <= 1e-8
    if bool(constant_dimensions.any()):
        matched[:, constant_dimensions] = target_mean[constant_dimensions]
    return matched


@torch.no_grad()
def _sample_cfs_features_paper_style(distribution, mean, cov, num_samples, args, device, cfs_model):
    if num_samples <= 1:
        return _sample_filtered_gaussian(distribution, mean, cov, num_samples, args, device)

    ratio = float(getattr(args, 'cfs_selection_ratio', 0.5))
    ratio = max(0.0, min(1.0, ratio))
    steps = max(1, int(getattr(args, 'cfs_selection_steps', 5)))
    multiplier = max(1, int(getattr(args, 'cfs_candidate_multiplier', 3)))
    step_candidates = int(getattr(args, 'cfs_step_candidates', 0))
    tau = float(getattr(args, 'cfs_tau', 1.0))

    selected_target = max(1, int(round(num_samples * ratio)))
    init_count = max(1, num_samples - selected_target)
    selected_features = [_sample_filtered_gaussian(distribution, mean, cov, init_count, args, device)]
    selected_embeddings = cfs_model(selected_features[0].float())
    remaining = num_samples - init_count

    for step in range(steps):
        if remaining <= 0:
            break
        take = int(math.ceil(remaining / float(steps - step)))
        candidate_count = step_candidates if step_candidates > 0 else max(take * multiplier, take)
        candidates = _sample_filtered_gaussian(distribution, mean, cov, candidate_count, args, device)
        candidate_embeddings = cfs_model(candidates.float())
        scores = torch.exp(torch.mm(candidate_embeddings, selected_embeddings.t()) / tau).mean(dim=1)
        take = min(take, candidate_count)
        selected_ids = torch.topk(scores, k=take, largest=False).indices
        chosen_features = candidates.index_select(0, selected_ids)
        chosen_embeddings = candidate_embeddings.index_select(0, selected_ids)
        selected_features.append(chosen_features)
        selected_embeddings = torch.cat([selected_embeddings, chosen_embeddings], dim=0)
        remaining -= take

    features = torch.cat(selected_features, dim=0)
    if features.shape[0] < num_samples:
        pad = _sample_filtered_gaussian(distribution, mean, cov, num_samples - features.shape[0], args, device)
        features = torch.cat([features, pad], dim=0)
    features = features[:num_samples]
    if bool(getattr(args, 'cfs_moment_match', False)):
        features = match_cfs_feature_moments(features, mean, cov)
    return features


_JITTERED = set()


@torch.no_grad()
def stable_multivariate_normal(mean, cov, label=''):
    """MultivariateNormal that tolerates a singular class covariance.

    Returns the distribution unchanged when `cov` is already positive definite,
    so behaviour on data where this never fired is bit-identical. Otherwise the
    smallest jitter from 1e-4 upward that makes the Cholesky succeed is added
    to the diagonal, and the fact is printed once per label.

    1e-4 is not arbitrary: it is the constant the original code already uses in
    the multi-centroid branch of hide_tii_engine.py, which is the one place its
    authors guarded against exactly this.
    """
    mean = mean.float()
    cov = cov.float()
    try:
        return torch.distributions.MultivariateNormal(mean, cov)
    except (ValueError, RuntimeError):
        pass

    eye = torch.eye(mean.numel(), device=cov.device, dtype=cov.dtype)
    for exponent in range(-4, 3):
        jitter = float(10.0 ** exponent)
        try:
            distribution = torch.distributions.MultivariateNormal(
                mean, cov + jitter * eye)
        except (ValueError, RuntimeError):
            continue
        if label not in _JITTERED:
            _JITTERED.add(label)
            print('hiep phuong sai suy bien tai %s: cong %g vao duong cheo'
                  % (label or 'khong ten', jitter))
        return distribution

    raise ValueError(
        'khong the lam cho hiep phuong sai xac dinh duong tai %s '
        '(da thu jitter den 100)' % (label or 'khong ten'))


@torch.no_grad()
def sample_cfs_features(
        mean, cov, num_samples, args, device, cfs_model=None,
        force_cfs=False):
    distribution = stable_multivariate_normal(
        mean, cov, 'sample_cfs_features')
    if ((not force_cfs and not use_cfs_sampling(args))
            or cfs_model is None or num_samples <= 1):
        return distribution.sample(sample_shape=(num_samples,))

    cfs_model = cfs_model.to(device)
    if bool(getattr(args, 'cfs_paper_style', False)):
        return _sample_cfs_features_paper_style(distribution, mean, cov, num_samples, args, device, cfs_model)

    multiplier = max(1, int(getattr(args, 'cfs_candidate_multiplier', 3)))
    candidate_count = max(num_samples, num_samples * multiplier)
    candidates = _sample_filtered_gaussian(distribution, mean, cov, candidate_count, args, device)

    embeddings = cfs_model(candidates.float())
    tau = float(getattr(args, 'cfs_tau', 1.0))
    sim = torch.exp(torch.mm(embeddings, embeddings.t()) / tau)
    sim.fill_diagonal_(0)

    init_strategy = str(getattr(args, 'cfs_init_strategy', 'random')).lower()
    if init_strategy == 'mean':
        mean_for_init = mean.float().to(device)
        cov_for_init = torch.as_tensor(cov).float().to(device)
        if cov_for_init.dim() == 2:
            diag_var = torch.diag(cov_for_init)
        else:
            diag_var = cov_for_init
        diag_var = diag_var.clamp_min(1e-5)
        centered = candidates.float() - mean_for_init.unsqueeze(0)
        init_scores = (centered.pow(2) / diag_var.unsqueeze(0)).mean(dim=1)
        selected = [torch.argmin(init_scores).item()]
    else:
        selected = [torch.randint(candidate_count, (1,), device=device).item()]
    score_sum = sim[:, selected[0]].clone()
    available = torch.ones(candidate_count, dtype=torch.bool, device=device)
    available[selected[0]] = False

    for _ in range(1, num_samples):
        scores = score_sum / len(selected)
        scores = scores.masked_fill(~available, float('inf'))
        next_id = torch.argmin(scores).item()
        selected.append(next_id)
        available[next_id] = False
        score_sum += sim[:, next_id]

    return candidates[torch.tensor(selected, device=device)]


@torch.no_grad()
def class_balanced_replay_order(labels):
    """Interleave shuffled per-class queues to keep replay batches balanced."""
    labels = labels.flatten()
    if labels.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=labels.device)

    class_ids = torch.unique(labels, sorted=True)
    per_class = []
    samples_per_class = None
    for class_id in class_ids:
        indexes = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if samples_per_class is None:
            samples_per_class = indexes.numel()
        elif indexes.numel() != samples_per_class:
            return torch.randperm(labels.numel(), device=labels.device)
        permutation = torch.randperm(indexes.numel(), device=labels.device)
        per_class.append(indexes.index_select(0, permutation))

    per_class = torch.stack(per_class, dim=0).transpose(0, 1)
    class_orders = torch.stack([
        torch.randperm(class_ids.numel(), device=labels.device)
        for _ in range(samples_per_class)
    ], dim=0)
    return per_class.gather(1, class_orders).reshape(-1)


@torch.no_grad()
def _sample_gaussian_core_features(mean, cov, num_samples, args, device):
    """Select high-density Gaussian samples to anchor each class replay distribution."""
    if num_samples <= 0:
        return mean.new_empty((0, mean.numel())).to(device)

    multiplier = max(1, int(getattr(args, 'cfs_core_multiplier', 4)))
    candidate_count = max(num_samples, num_samples * multiplier)
    distribution = stable_multivariate_normal(
        mean, cov, 'sample_cfs_core_features')
    candidates = distribution.sample(sample_shape=(candidate_count,)).float().to(device)
    return _filter_cfs_candidate_features(
        candidates, mean, cov, num_samples, args, device)


@torch.no_grad()
def sample_boundary_aware_cfs_features(mean, cov, num_samples, args, device, model,
                                       target_cls, seen_classes, cfs_model=None):
    """Mix diverse CFS replay with in-distribution samples near the decision boundary."""
    if not use_cfs_boundary_replay(args) or num_samples <= 1:
        return sample_cfs_features(mean, cov, num_samples, args, device, cfs_model=cfs_model)

    boundary_ratio = float(getattr(args, 'cfs_boundary_ratio', 0.5))
    boundary_ratio = max(0.0, min(1.0, boundary_ratio))
    core_ratio = float(getattr(args, 'cfs_core_replay_ratio', 0.0))
    core_ratio = max(0.0, min(1.0, core_ratio))
    core_count = min(num_samples, int(round(num_samples * core_ratio)))
    hard_count = min(
        num_samples - core_count,
        int(round(num_samples * boundary_ratio)),
    )
    diverse_count = num_samples - core_count - hard_count
    core = _sample_gaussian_core_features(
        mean, cov, core_count, args, device)
    diverse = sample_cfs_features(
        mean, cov, diverse_count, args, device, cfs_model=cfs_model)
    if hard_count <= 0:
        return torch.cat([core, diverse], dim=0)
    seen_classes = sorted({int(cls_id) for cls_id in seen_classes})
    if int(target_cls) not in seen_classes or len(seen_classes) < 2:
        fallback = sample_cfs_features(
            mean, cov, hard_count, args, device, cfs_model=cfs_model)
        return torch.cat([core, diverse, fallback], dim=0)

    multiplier = max(1, int(getattr(args, 'cfs_boundary_multiplier', 3)))
    candidate_count = max(hard_count, num_samples * multiplier)
    distribution = stable_multivariate_normal(
        mean, cov, 'sample_boundary_aware_cfs_features')
    candidates = distribution.sample(sample_shape=(candidate_count,)).float().to(device)

    outputs = model(candidates, fc_only=True)
    logits = outputs['logits'].float()
    seen_ids = torch.tensor(seen_classes, dtype=torch.long, device=device)
    seen_logits = logits.index_select(1, seen_ids)
    target_pos = seen_classes.index(int(target_cls))
    target_logits = seen_logits[:, target_pos]
    competitor_logits = seen_logits.clone()
    competitor_logits[:, target_pos] = float('-inf')
    margins = target_logits - competitor_logits.max(dim=1).values

    # Exclude only the extreme Gaussian tail before selecting boundary samples.
    cov_tensor = torch.as_tensor(cov).float().to(device)
    diag_var = torch.diag(cov_tensor) if cov_tensor.dim() == 2 else cov_tensor
    diag_var = diag_var.clamp_min(1e-5)
    centered = candidates - mean.float().to(device).unsqueeze(0)
    mahalanobis = (centered.pow(2) / diag_var.unsqueeze(0)).mean(dim=1)
    density_quantile = float(getattr(args, 'cfs_boundary_density_quantile', 0.9))
    density_quantile = max(0.5, min(1.0, density_quantile))
    density_limit = torch.quantile(mahalanobis, density_quantile)
    in_distribution = mahalanobis <= density_limit
    if bool(getattr(args, 'cfs_boundary_target_side', False)):
        target_side = in_distribution & margins.ge(0)
        target_count = min(hard_count, int(target_side.sum().item()))
        selected_parts = []
        if target_count > 0:
            target_scores = margins.masked_fill(~target_side, float('inf'))
            selected_parts.append(torch.topk(
                target_scores, k=target_count, largest=False).indices)

        remaining = hard_count - target_count
        if remaining > 0:
            fallback_scores = margins.abs().masked_fill(~in_distribution, float('inf'))
            if selected_parts:
                fallback_scores[selected_parts[0]] = float('inf')
            selected_parts.append(torch.topk(
                fallback_scores, k=remaining, largest=False).indices)
        hard_ids = torch.cat(selected_parts, dim=0)
    else:
        boundary_score = margins.abs().masked_fill(~in_distribution, float('inf'))
        hard_ids = torch.topk(boundary_score, k=hard_count, largest=False).indices
    boundary = candidates.index_select(0, hard_ids)
    return torch.cat([core, diverse.float().to(device), boundary], dim=0)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    distributed_barrier()
    setup_for_distributed(args.rank == 0)


def task_inference_accuracy(prompt_idx, target, target_task_map,filtered_index_tensor=None,corr_id=None):
    target_2_task = torch.tensor([target_task_map[v.item()] for v in target]).to(prompt_idx.device)
    # if filtered_index_tensor!=None and corr_id!=None:
    #     target_2_task[filtered_index_tensor] = corr_id
    batch_size = target.size(0)
    prompt_idx = prompt_idx.t()
    correct = prompt_idx.eq(target_2_task.reshape(1, -1).expand_as(prompt_idx))
    return correct.reshape(-1).float().sum(0) * 100. / batch_size
