import torch
import torch.nn.functional as F


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _leading_input_direction(k_a, k_b, v_a, v_b):
    """Return the dominant input direction of the joint K/V LoRA update."""
    dtype = torch.float32
    k_a = k_a.detach().to(dtype=dtype)
    k_b = k_b.detach().to(dtype=dtype)
    v_a = v_a.detach().to(dtype=dtype)
    v_b = v_b.detach().to(dtype=dtype)

    dim, rank = k_a.shape
    if k_b.square().sum().add(v_b.square().sum()).item() <= 1e-20:
        return torch.zeros(dim, dtype=dtype, device=k_a.device)

    input_factors = torch.cat([k_a, v_a], dim=1)
    output_factors = torch.zeros(
        (2 * rank, 2 * dim), dtype=dtype, device=k_a.device)
    output_factors[:rank, :dim] = k_b
    output_factors[rank:, dim:] = v_b

    basis, small_input = torch.linalg.qr(input_factors, mode='reduced')
    small_update = small_input @ output_factors
    if small_update.square().sum().item() <= 1e-20:
        return torch.zeros(dim, dtype=dtype, device=k_a.device)
    small_left = torch.linalg.svd(
        small_update, full_matrices=False).U[:, 0]
    direction = basis @ small_left
    return F.normalize(direction, dim=0, eps=1e-12)


def build_arrow_task_prototypes(model, seen_task_count):
    """Build one dominant LoRA input direction per task and LoRA depth."""
    base_model = _unwrap_model(model)
    lora = getattr(base_model, 'lora_layer', None)
    required = ('k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B')
    if lora is None or any(not hasattr(lora, name) for name in required):
        raise ValueError('Arrow audit requires a HideLoraPool K/V adapter bank')

    seen_task_count = min(int(seen_task_count), int(lora.pool_size))
    cache_key = (
        seen_task_count,
        int(lora.k_lora_A._version),
        int(lora.k_lora_B._version),
        int(lora.v_lora_A._version),
        int(lora.v_lora_B._version),
    )
    cache = lora.__dict__.get('_arrow_task_prototype_cache')
    if cache is not None and cache['key'] == cache_key:
        return cache['value']

    prototypes = []
    for task_index in range(seen_task_count):
        task_prototypes = []
        for depth_index in range(int(lora.depth)):
            task_prototypes.append(_leading_input_direction(
                lora.k_lora_A[task_index, depth_index],
                lora.k_lora_B[task_index, depth_index],
                lora.v_lora_A[task_index, depth_index],
                lora.v_lora_B[task_index, depth_index],
            ))
        prototypes.append(torch.stack(task_prototypes, dim=0))
    prototypes = torch.stack(prototypes, dim=0)
    lora.__dict__['_arrow_task_prototype_cache'] = {
        'key': cache_key,
        'value': prototypes,
    }
    return prototypes


@torch.no_grad()
def arrow_task_scores(model, inputs, seen_task_count):
    """Score task adapters from their parameter signatures at matching layers."""
    base_model = _unwrap_model(model)
    prototypes = build_arrow_task_prototypes(
        base_model, seen_task_count).to(device=inputs.device)
    lora_depth = prototypes.shape[1]

    x = base_model.patch_embed(inputs)
    if base_model.cls_token is not None:
        x = torch.cat(
            (base_model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = base_model.pos_drop(x + base_model.pos_embed)

    scores = torch.zeros(
        (inputs.shape[0], prototypes.shape[0]),
        dtype=torch.float32, device=inputs.device)
    for depth_index in range(lora_depth):
        attention_input = F.normalize(
            base_model.blocks[depth_index].norm1(x).float(),
            dim=-1, eps=1e-12)
        depth_prototypes = prototypes[:, depth_index].float()
        token_scores = torch.einsum(
            'bnd,td->bnt', attention_input, depth_prototypes).abs()
        scores.add_(token_scores.mean(dim=1))
        if depth_index + 1 < lora_depth:
            x = base_model.blocks[depth_index](x)
    return scores.div(float(lora_depth))


def arrow_candidate_diagnostics(tii_ranking, arrow_ranking, winner_tasks):
    """Measure Arrow ranking and the fixed TII-top2 + Arrow-top2 union."""
    seen_task_count = tii_ranking.shape[1]
    top2 = min(2, seen_task_count)
    top4 = min(4, seen_task_count)

    arrow_top2 = arrow_ranking[:, :top2]
    arrow_top4 = arrow_ranking[:, :top4]
    arrow_recall_2 = arrow_top2.eq(winner_tasks.unsqueeze(1)).any(dim=1)
    arrow_recall_4 = arrow_top4.eq(winner_tasks.unsqueeze(1)).any(dim=1)

    union_mask = torch.zeros(
        tii_ranking.shape[0], seen_task_count,
        dtype=torch.bool, device=tii_ranking.device)
    union_mask.scatter_(1, tii_ranking[:, :top2], True)
    union_mask.scatter_(1, arrow_top2, True)
    union_recall = union_mask.gather(
        1, winner_tasks.unsqueeze(1)).squeeze(1)
    union_counts = union_mask.sum(dim=1).float()

    return {
        'arrow_winner_recall_2': arrow_recall_2,
        'arrow_winner_recall_4': arrow_recall_4,
        'arrow_union_recall_2x2': union_recall,
        'arrow_union_lora_counts': union_counts,
        'tii_arrow_top1_agreement': tii_ranking[:, 0].eq(
            arrow_ranking[:, 0]),
    }
