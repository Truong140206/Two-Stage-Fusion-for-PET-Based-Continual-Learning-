import torch
import torch.nn.functional as F


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _factor_response_energy(normalized_inputs, factor_a, factor_b):
    """Compute ||xAB||^2 / ||AB||_F^2 without materializing AB."""
    factor_a = factor_a.detach().float()
    factor_b = factor_b.detach().float()
    low_rank_inputs = torch.einsum(
        'bnd,tdr->bntr', normalized_inputs, factor_a)
    output_gram = torch.einsum('trd,tsd->trs', factor_b, factor_b)
    response_energy = torch.einsum(
        'bntr,trs,bnts->bnt',
        low_rank_inputs, output_gram, low_rank_inputs)
    update_frobenius_sq = torch.einsum(
        'tdr,tds,trs->t', factor_a, factor_a, output_gram)
    return response_energy, update_frobenius_sq


@torch.no_grad()
def lora_response_task_scores(model, inputs, seen_task_count):
    """Score every task by its normalized full-rank K/V LoRA response."""
    base_model = _unwrap_model(model)
    lora = getattr(base_model, 'lora_layer', None)
    required = ('k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B')
    if lora is None or any(not hasattr(lora, name) for name in required):
        raise ValueError('LoRA response audit requires a HideLoraPool bank')

    seen_task_count = min(int(seen_task_count), int(lora.pool_size))
    lora_depth = int(lora.depth)
    x = base_model.patch_embed(inputs)
    if base_model.cls_token is not None:
        x = torch.cat(
            (base_model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = base_model.pos_drop(x + base_model.pos_embed)

    scores = torch.zeros(
        (inputs.shape[0], seen_task_count),
        dtype=torch.float32, device=inputs.device)
    for depth_index in range(lora_depth):
        attention_input = F.normalize(
            base_model.blocks[depth_index].norm1(x).float(),
            dim=-1, eps=1e-12)
        k_energy, k_norm = _factor_response_energy(
            attention_input,
            lora.k_lora_A[:seen_task_count, depth_index],
            lora.k_lora_B[:seen_task_count, depth_index],
        )
        v_energy, v_norm = _factor_response_energy(
            attention_input,
            lora.v_lora_A[:seen_task_count, depth_index],
            lora.v_lora_B[:seen_task_count, depth_index],
        )
        total_energy = (k_energy + v_energy).clamp_min(0.0)
        total_norm = (k_norm + v_norm).clamp_min(1e-20)
        depth_scores = torch.sqrt(
            total_energy.mean(dim=1) / total_norm.unsqueeze(0))
        scores.add_(depth_scores)
        if depth_index + 1 < lora_depth:
            x = base_model.blocks[depth_index](x)
    return scores.div(float(lora_depth))


def lora_response_candidate_diagnostics(
        tii_ranking, response_ranking, winner_tasks):
    """Measure response ranking and the fixed TII-top2 + response-top2 union."""
    seen_task_count = tii_ranking.shape[1]
    top2 = min(2, seen_task_count)
    top4 = min(4, seen_task_count)
    response_top2 = response_ranking[:, :top2]
    response_top4 = response_ranking[:, :top4]

    union_mask = torch.zeros(
        tii_ranking.shape[0], seen_task_count,
        dtype=torch.bool, device=tii_ranking.device)
    union_mask.scatter_(1, tii_ranking[:, :top2], True)
    union_mask.scatter_(1, response_top2, True)

    return {
        'response_winner_recall_2': response_top2.eq(
            winner_tasks.unsqueeze(1)).any(dim=1),
        'response_winner_recall_4': response_top4.eq(
            winner_tasks.unsqueeze(1)).any(dim=1),
        'response_union_recall_2x2': union_mask.gather(
            1, winner_tasks.unsqueeze(1)).squeeze(1),
        'response_union_lora_counts': union_mask.sum(dim=1).float(),
        'tii_response_top1_agreement': tii_ranking[:, 0].eq(
            response_ranking[:, 0]),
    }
