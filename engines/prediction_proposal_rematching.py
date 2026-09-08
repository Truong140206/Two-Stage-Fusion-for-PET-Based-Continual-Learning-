import torch
import torch.nn.functional as F

from engines.progressive_oracle_audit import (
    _finalize_partial_logits,
    prediction_proposal_candidates,
)
from engines.progressive_rematching import _tii_task_prior
from engines.tii_tail_completion import (
    complete_with_tii_probability_mass,
)


def _evaluate_candidate_tasks(model, inputs, candidate_tasks):
    batch_size, candidate_count = candidate_tasks.shape
    expanded_inputs = inputs.unsqueeze(1).expand(
        batch_size, candidate_count, *inputs.shape[1:]
    ).reshape(batch_size * candidate_count, *inputs.shape[1:])
    expanded_task_ids = candidate_tasks.reshape(-1)
    return model(
        expanded_inputs, task_id=expanded_task_ids
    )['logits'].reshape(batch_size, candidate_count, -1)


def _apply_cfs_task_logit_calibration(
        local_logits, task_index, args):
    if not bool(getattr(args, 'cfs_task_logit_calibration', False)):
        return local_logits
    state = getattr(args, 'cfs_task_logit_calibration_state', None)
    if state is None:
        raise RuntimeError(
            'CFS task-logit calibration is enabled but no checkpoint state '
            'was loaded')
    scale = state.get('scale')
    bias = state.get('bias')
    if scale is None or bias is None or task_index >= len(scale):
        raise RuntimeError('Malformed CFS task-logit calibration state')
    task_scale = local_logits.new_tensor(float(scale[task_index]))
    task_bias = local_logits.new_tensor(float(bias[task_index]))
    return local_logits * task_scale + task_bias


def _propose_unseen_tasks(
        tii_ranking, evaluated_tasks, evaluated_logits, class_mask,
        proposal_count, top_classes):
    """Propose unseen tasks from every adapter evaluated so far."""
    batch_size, seen_task_count = tii_ranking.shape
    device = tii_ranking.device
    proposal_count = min(
        max(0, int(proposal_count)),
        seen_task_count - evaluated_tasks.shape[1])
    if proposal_count == 0:
        return evaluated_tasks.new_empty((batch_size, 0))

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    class_to_task = torch.full(
        (evaluated_logits.shape[-1],), -1,
        dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = task_index

    proposal_scores = torch.full(
        (batch_size, seen_task_count), float('-inf'),
        dtype=evaluated_logits.dtype, device=device)
    top_classes = min(max(1, int(top_classes)), len(seen_classes))
    for adapter_slot in range(evaluated_tasks.shape[1]):
        seen_logits = evaluated_logits[:, adapter_slot].index_select(
            1, seen_class_index)
        top_values, top_positions = torch.topk(
            seen_logits, k=top_classes, dim=1)
        proposed_tasks = class_to_task[seen_class_index[top_positions]]
        proposal_scores.scatter_reduce_(
            1, proposed_tasks, top_values,
            reduce='amax', include_self=True)

    candidate_mask = torch.zeros(
        (batch_size, seen_task_count), dtype=torch.bool, device=device)
    candidate_mask.scatter_(1, evaluated_tasks, True)
    proposal_scores.masked_fill_(candidate_mask, float('-inf'))
    selected_parts = []
    for _ in range(proposal_count):
        proposed_task = proposal_scores.argmax(dim=1)
        proposed_score = proposal_scores.gather(
            1, proposed_task.unsqueeze(1)).squeeze(1)
        has_proposal = torch.isfinite(proposed_score)

        fallback_mask = candidate_mask.gather(1, tii_ranking)
        fallback_rank = (~fallback_mask).float().argmax(dim=1)
        fallback_task = tii_ranking.gather(
            1, fallback_rank.unsqueeze(1)).squeeze(1)
        selected_task = torch.where(
            has_proposal, proposed_task, fallback_task)
        selected_parts.append(selected_task.unsqueeze(1))
        candidate_mask.scatter_(1, selected_task.unsqueeze(1), True)
        proposal_scores.scatter_(
            1, selected_task.unsqueeze(1), float('-inf'))
    return torch.cat(selected_parts, dim=1)


def initial_branch_confidence_dominance(initial_logits, proposal_logits):
    """Select initial output only when it Pareto-dominates proposal confidence."""
    initial_probabilities = torch.softmax(initial_logits, dim=1)
    proposal_probabilities = torch.softmax(proposal_logits, dim=1)
    initial_top2 = torch.topk(initial_probabilities, k=2, dim=1).values
    proposal_top2 = torch.topk(proposal_probabilities, k=2, dim=1).values
    initial_prediction = initial_logits.argmax(dim=1)
    proposal_prediction = proposal_logits.argmax(dim=1)
    return (
        initial_prediction.ne(proposal_prediction)
        & initial_top2[:, 0].gt(proposal_top2[:, 0])
        & (initial_top2[:, 0] - initial_top2[:, 1]).gt(
            proposal_top2[:, 0] - proposal_top2[:, 1])
    )


def cross_adapter_global_consensus(candidate_logits):
    """Plurality vote over full adapter predictions with probability tie-break."""
    batch_size, candidate_count, class_count = candidate_logits.shape
    probabilities = torch.softmax(candidate_logits, dim=2)
    predictions = candidate_logits.argmax(dim=2)
    vote_counts = torch.zeros(
        (batch_size, class_count), dtype=torch.long,
        device=candidate_logits.device)
    vote_counts.scatter_add_(
        1, predictions,
        torch.ones_like(predictions, dtype=torch.long))
    max_votes = vote_counts.max(dim=1).values
    tied_classes = vote_counts.eq(max_votes.unsqueeze(1))
    probability_support = probabilities.sum(dim=1)
    consensus_prediction = probability_support.masked_fill(
        ~tied_classes, float('-inf')).argmax(dim=1)
    strict_majority = max_votes.mul(2).gt(candidate_count)
    vote_strength = max_votes.float().div(float(candidate_count))
    return {
        'adapter_predictions': predictions,
        'consensus_prediction': consensus_prediction,
        'strict_majority': strict_majority,
        'vote_strength': vote_strength,
    }


def cross_adapter_borda_consensus(candidate_logits, top_k=5):
    """Calibration-free Borda aggregation over full class rankings."""
    _, candidate_count, class_count = candidate_logits.shape
    ordering = torch.argsort(candidate_logits, dim=2, descending=True)
    ranks = torch.argsort(ordering, dim=2)
    rank_sum = ranks.sum(dim=1)
    best_rank_sum = rank_sum.min(dim=1).values
    tied_classes = rank_sum.eq(best_rank_sum.unsqueeze(1))
    probability_support = torch.softmax(candidate_logits, dim=2).sum(dim=1)
    prediction = probability_support.masked_fill(
        ~tied_classes, float('-inf')).argmax(dim=1)
    selected_ranks = ranks.gather(
        2,
        prediction.view(-1, 1, 1).expand(-1, candidate_count, 1),
    ).squeeze(2)
    top_k = min(max(1, int(top_k)), class_count)
    support_count = selected_ranks.lt(top_k).sum(dim=1)
    strict_support = support_count.mul(2).gt(candidate_count)
    return {
        'prediction': prediction,
        'topk_support': support_count.float().div(float(candidate_count)),
        'strict_support': strict_support,
    }


def _complete_with_tii_probability_mass(
        candidate_logits, tii_logits, class_mask, candidate_tasks,
        seen_task_count):
    """Give excluded seen classes TII mass without changing candidate top-1."""
    included_task_mask = torch.zeros(
        (candidate_logits.shape[0], seen_task_count),
        dtype=torch.bool, device=candidate_logits.device)
    for candidate_slot in range(candidate_tasks.shape[1]):
        included_task_mask.scatter_(
            1, candidate_tasks[:, candidate_slot: candidate_slot + 1], True)
    return complete_with_tii_probability_mass(
        candidate_logits, tii_logits, class_mask, included_task_mask
    )


def task_mass_preserving_candidate_fusion(
        candidate_logits, tii_logits, class_mask, candidate_tasks,
        seen_task_count, temperature=1.0):
    """Fuse adapters as P(task|x) P(class|task,x), preserving TII task mass."""
    batch_size, candidate_count, class_count = candidate_logits.shape
    device = candidate_logits.device
    temperature = max(float(temperature), 1e-6)

    seen_class_mask = torch.zeros(
        class_count, dtype=torch.bool, device=device)
    for task_classes in class_mask[:seen_task_count]:
        seen_class_mask[torch.as_tensor(
            task_classes, dtype=torch.long, device=device)] = True
    tii_probabilities = torch.softmax(
        tii_logits.masked_fill(~seen_class_mask.unsqueeze(0), float('-inf')),
        dim=1)

    task_mass = torch.stack([
        tii_probabilities.index_select(
            1, torch.as_tensor(task_classes, dtype=torch.long, device=device)
        ).sum(dim=1)
        for task_classes in class_mask[:seen_task_count]
    ], dim=1)
    candidate_mass = task_mass.gather(1, candidate_tasks)
    candidate_mass = candidate_mass / candidate_mass.sum(
        dim=1, keepdim=True).clamp_min(1e-12)

    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = torch.full(
        (batch_size, candidate_count), float('-inf'),
        dtype=tii_logits.dtype, device=device)
    for candidate_slot in range(candidate_count):
        active_tasks = candidate_tasks[:, candidate_slot]
        adapter_logits = candidate_logits[:, candidate_slot]
        for task_index in active_tasks.unique().tolist():
            rows = torch.nonzero(active_tasks == task_index).flatten()
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            conditional_log_probability = F.log_softmax(
                adapter_logits.index_select(0, rows).index_select(
                    1, class_index) / temperature,
                dim=1)
            joint_log_probability = (
                conditional_log_probability
                + candidate_mass[rows, candidate_slot].clamp_min(
                    1e-12).log().unsqueeze(1))
            merged_logits[
                rows.unsqueeze(1), class_index.unsqueeze(0)
            ] = joint_log_probability
            task_evidence[rows, candidate_slot] = (
                joint_log_probability.max(dim=1).values)
    return merged_logits, task_evidence


def conditional_candidate_fusion(
        candidate_logits, tii_prior, class_mask, candidate_tasks,
        temperature=1.0, prior_weight=0.3):
    """Compare task-conditional LoRA probabilities with the fixed TII prior."""
    batch_size, candidate_count, _ = candidate_logits.shape
    device = candidate_logits.device
    temperature = max(float(temperature), 1e-6)
    prior_weight = max(float(prior_weight), 0.0)
    merged_logits = candidate_logits.new_full(
        (batch_size, candidate_logits.shape[-1]), float('-inf'))
    task_evidence = torch.full(
        (batch_size, candidate_count), float('-inf'),
        dtype=candidate_logits.dtype, device=device)
    for candidate_slot in range(candidate_count):
        active_tasks = candidate_tasks[:, candidate_slot]
        adapter_logits = candidate_logits[:, candidate_slot]
        for task_index in active_tasks.unique().tolist():
            rows = torch.nonzero(active_tasks == task_index).flatten()
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            conditional_logits = F.log_softmax(
                adapter_logits.index_select(0, rows).index_select(
                    1, class_index) / temperature,
                dim=1)
            calibrated_logits = conditional_logits + prior_weight * tii_prior[
                rows, task_index].unsqueeze(1)
            merged_logits[
                rows.unsqueeze(1), class_index.unsqueeze(0)
            ] = calibrated_logits
            task_evidence[rows, candidate_slot] = calibrated_logits.max(
                dim=1).values
    return merged_logits, task_evidence


def crm_confidence_candidate_fusion(
        candidate_logits, tii_prior, class_mask, candidate_tasks,
        class_temperature=1.0, confidence_temperature=0.1,
        prior_weight=0.3, gamma=0.1, top_m=20):
    """Fuse task-conditional probabilities using HRM's GEN confidence."""
    batch_size, candidate_count, _ = candidate_logits.shape
    device = candidate_logits.device
    class_temperature = max(float(class_temperature), 1e-6)
    confidence_temperature = max(float(confidence_temperature), 1e-6)
    prior_weight = max(float(prior_weight), 0.0)
    gamma = max(float(gamma), 1e-6)

    conditional_log_probs = torch.full_like(
        candidate_logits, float('-inf'))
    confidence_scores = candidate_logits.new_full(
        (batch_size, candidate_count), float('-inf'))
    task_priors = candidate_logits.new_zeros(
        (batch_size, candidate_count))
    for candidate_slot in range(candidate_count):
        active_tasks = candidate_tasks[:, candidate_slot]
        adapter_logits = candidate_logits[:, candidate_slot]
        task_priors[:, candidate_slot] = tii_prior.gather(
            1, active_tasks.unsqueeze(1)).squeeze(1)
        for task_index in active_tasks.unique().tolist():
            rows = torch.nonzero(active_tasks == task_index).flatten()
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            local_log_probs = F.log_softmax(
                adapter_logits.index_select(0, rows).index_select(
                    1, class_index) / class_temperature,
                dim=1)
            local_probs = local_log_probs.exp()
            selected_m = min(max(1, int(top_m)), local_probs.shape[1])
            top_probs = torch.topk(
                local_probs, k=selected_m, dim=1).values.clamp(
                    min=1e-12, max=1.0 - 1e-12)
            gen_confidence = -(
                top_probs.pow(gamma)
                * (1.0 - top_probs).pow(gamma)
            ).sum(dim=1)
            conditional_log_probs[
                rows.unsqueeze(1), candidate_slot,
                class_index.unsqueeze(0)
            ] = local_log_probs
            confidence_scores[rows, candidate_slot] = gen_confidence

    task_scores = (
        confidence_scores / confidence_temperature
        + prior_weight * task_priors)
    task_log_probs = F.log_softmax(task_scores, dim=1)
    joint_log_probs = conditional_log_probs + task_log_probs.unsqueeze(2)
    merged_logits = torch.logsumexp(joint_log_probs, dim=1)
    task_evidence = joint_log_probs.max(dim=2).values
    return merged_logits, task_evidence


@torch.no_grad()
def prediction_proposal_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Route through TII top tasks plus post-LoRA prediction proposals."""
    device = inputs.device
    batch_size = inputs.shape[0]
    temperature = max(
        1e-6, float(getattr(args, 'progressive_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'progressive_tii_prior_weight', 0.3)))
    excluded_margin = max(
        1.0, float(getattr(args, 'progressive_excluded_logit_margin', 20.0)))
    initial_count = min(
        max(1, int(getattr(args, 'prediction_proposal_initial_count', 2))),
        seen_task_count)
    proposal_count = min(
        max(0, int(getattr(args, 'prediction_proposal_count', 2))),
        seen_task_count - initial_count)
    top_classes = max(
        1, int(getattr(args, 'prediction_proposal_top_classes', 5)))

    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    tii_ranking = torch.argsort(tii_prior, dim=1, descending=True)
    initial_tasks = tii_ranking[:, :initial_count]
    initial_logits = _evaluate_candidate_tasks(
        model, inputs, initial_tasks)
    iterative_proposal = bool(getattr(
        args, 'prediction_proposal_iterative', False))
    candidate_logits = initial_logits
    candidate_tasks = initial_tasks
    forward_calls = 1
    if proposal_count > 0:
        if iterative_proposal:
            first_wave_count = min(
                max(1, int(getattr(
                    args, 'prediction_proposal_first_wave_count', 2))),
                proposal_count)
            first_wave_tasks = _propose_unseen_tasks(
                tii_ranking, candidate_tasks, candidate_logits, class_mask,
                first_wave_count, top_classes)
            first_wave_logits = _evaluate_candidate_tasks(
                model, inputs, first_wave_tasks)
            candidate_tasks = torch.cat(
                [candidate_tasks, first_wave_tasks], dim=1)
            candidate_logits = torch.cat(
                [candidate_logits, first_wave_logits], dim=1)
            forward_calls += 1

            remaining_count = proposal_count - first_wave_count
            if remaining_count > 0:
                later_tasks = _propose_unseen_tasks(
                    tii_ranking, candidate_tasks, candidate_logits,
                    class_mask, remaining_count, top_classes)
                later_logits = _evaluate_candidate_tasks(
                    model, inputs, later_tasks)
                candidate_tasks = torch.cat(
                    [candidate_tasks, later_tasks], dim=1)
                candidate_logits = torch.cat(
                    [candidate_logits, later_logits], dim=1)
                forward_calls += 1
        else:
            candidate_tasks, _ = prediction_proposal_candidates(
                tii_ranking,
                initial_logits,
                class_mask,
                initial_count=initial_count,
                proposal_count=proposal_count,
                top_classes=top_classes,
            )
            proposal_tasks = candidate_tasks[:, initial_count:]
            proposal_logits = _evaluate_candidate_tasks(
                model, inputs, proposal_tasks)
            candidate_logits = torch.cat(
                [candidate_logits, proposal_logits], dim=1)
            forward_calls += 1

    candidate_count = candidate_tasks.shape[1]
    task_mass_fusion = bool(getattr(
        args, 'prediction_proposal_task_mass_fusion', False))
    conditional_fusion = bool(getattr(
        args, 'prediction_proposal_conditional_fusion', False))
    crm_confidence_fusion = bool(getattr(
        args, 'prediction_proposal_crm_confidence_fusion', False))
    if sum([
            task_mass_fusion,
            conditional_fusion,
            crm_confidence_fusion,
    ]) > 1:
        raise ValueError(
            'Prediction-proposal fusion modes are mutually exclusive')
    if (bool(getattr(args, 'cfs_task_logit_calibration', False))
            and (
                task_mass_fusion
                or conditional_fusion
                or crm_confidence_fusion)):
        raise ValueError(
            'CFS task-logit calibration is defined only for raw-logit '
            'prediction-proposal fusion')
    if task_mass_fusion:
        merged_logits, task_evidence = task_mass_preserving_candidate_fusion(
            candidate_logits=candidate_logits,
            tii_logits=tii_logits,
            class_mask=class_mask,
            candidate_tasks=candidate_tasks,
            seen_task_count=seen_task_count,
            temperature=temperature,
        )
    elif conditional_fusion:
        merged_logits, task_evidence = conditional_candidate_fusion(
            candidate_logits=candidate_logits,
            tii_prior=tii_prior,
            class_mask=class_mask,
            candidate_tasks=candidate_tasks,
            temperature=temperature,
            prior_weight=prior_weight,
        )
    elif crm_confidence_fusion:
        merged_logits, task_evidence = crm_confidence_candidate_fusion(
            candidate_logits=candidate_logits,
            tii_prior=tii_prior,
            class_mask=class_mask,
            candidate_tasks=candidate_tasks,
            class_temperature=temperature,
            confidence_temperature=float(getattr(
                args,
                'prediction_proposal_crm_confidence_temperature',
                0.1)),
            prior_weight=prior_weight,
        )
    else:
        merged_logits = torch.full_like(tii_logits, float('-inf'))
        task_evidence = torch.full(
            (batch_size, candidate_count), float('-inf'),
            dtype=tii_logits.dtype, device=device)
        for candidate_slot in range(candidate_count):
            active_tasks = candidate_tasks[:, candidate_slot]
            adapter_logits = candidate_logits[:, candidate_slot]
            for task_index in active_tasks.unique().tolist():
                rows = torch.nonzero(active_tasks == task_index).flatten()
                class_index = torch.as_tensor(
                    class_mask[task_index], dtype=torch.long, device=device)
                local_logits = adapter_logits.index_select(
                    0, rows).index_select(1, class_index)
                local_logits = _apply_cfs_task_logit_calibration(
                    local_logits, task_index, args)
                calibrated_logits = (
                    local_logits / temperature
                    + prior_weight * tii_prior[
                    rows, task_index].unsqueeze(1)
                )
                merged_logits[
                    rows.unsqueeze(1), class_index.unsqueeze(0)
                ] = calibrated_logits
                task_evidence[rows, candidate_slot] = calibrated_logits.max(
                    dim=1).values

    selected_slot = task_evidence.argmax(dim=1)
    routed_tasks = candidate_tasks.gather(
        1, selected_slot.unsqueeze(1)).squeeze(1)
    if bool(getattr(args, 'prediction_proposal_tii_completion', False)):
        merged_logits = _complete_with_tii_probability_mass(
            merged_logits, tii_logits, class_mask, candidate_tasks,
            seen_task_count)
    else:
        merged_logits = _finalize_partial_logits(
            merged_logits, excluded_margin)
    diagnostics = {
        'lora_counts': torch.full(
            (batch_size,), float(candidate_count),
            dtype=torch.float32, device=device),
        'forward_calls': torch.full(
            (batch_size,), float(forward_calls),
            dtype=torch.float32, device=device),
        # The first evaluated task is exactly the initial TII top-1
        # route, so exposing it for audit adds neither a LoRA nor a forward.
        'initial_branch_logits': initial_logits[:, 0],
        'initial_branch_tasks': initial_tasks[:, 0],
        # Full logits already exist for every evaluated adapter. Keeping them
        # transiently for audit adds no model execution or persistent memory.
        'candidate_logits': candidate_logits,
        'candidate_tasks': candidate_tasks,
    }
    return merged_logits, routed_tasks, diagnostics
