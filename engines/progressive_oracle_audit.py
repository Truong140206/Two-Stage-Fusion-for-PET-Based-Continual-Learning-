import torch

from engines.arrow_lora_audit import (
    arrow_candidate_diagnostics,
    arrow_task_scores,
)
from engines.lora_response_audit import (
    lora_response_candidate_diagnostics,
    lora_response_task_scores,
)
from engines.progressive_rematching import _tii_task_prior
from engines.tii_tail_completion import (
    complete_with_tii_probability_mass,
)


def _finalize_partial_logits(partial_logits, excluded_margin):
    finite_mask = torch.isfinite(partial_logits)
    row_min = partial_logits.masked_fill(~finite_mask, float('inf')).min(
        dim=1, keepdim=True).values
    return torch.where(
        finite_mask, partial_logits, row_min - excluded_margin)


def stage_drift_diagnostics(
        candidate_tasks, full_adapter_logits, targets, true_task,
        class_mask, seen_task_count):
    """Separate within-task drift from cross-task score competition.

    The exhaustive audit has already evaluated every seen adapter. This helper
    selects the ground-truth task adapter without another model forward, then
    evaluates it twice: once with only its local classes and once against every
    seen class. The local-to-seen gap measures competition introduced by other
    task heads independently of routing coverage.
    """
    if targets.ndim != 1:
        raise ValueError('Stage-drift targets must be a one-dimensional tensor')
    if not 0 <= int(true_task) < int(seen_task_count):
        raise ValueError('Stage-drift true task is outside the seen task range')
    if candidate_tasks.shape[:1] != targets.shape:
        raise ValueError('Stage-drift candidate rows must match target rows')
    if full_adapter_logits.shape[:2] != candidate_tasks.shape:
        raise ValueError('Stage-drift logits must follow candidate rank order')

    task_matches = candidate_tasks.eq(int(true_task))
    if not task_matches.sum(dim=1).eq(1).all():
        raise ValueError(
            'Stage-drift audit requires every seen task exactly once per row')
    true_rank = task_matches.to(torch.int64).argmax(dim=1)
    rows = torch.arange(targets.shape[0], device=targets.device)
    own_logits = full_adapter_logits[rows, true_rank]

    local_classes = torch.as_tensor(
        class_mask[int(true_task)], dtype=torch.long, device=targets.device)
    seen_classes = torch.as_tensor(
        [
            int(class_id)
            for task_index in range(int(seen_task_count))
            for class_id in class_mask[task_index]
        ],
        dtype=torch.long,
        device=targets.device,
    )
    target_is_local = targets.unsqueeze(1).eq(
        local_classes.unsqueeze(0)).any(dim=1)
    if not target_is_local.all():
        raise ValueError(
            'Stage-drift targets do not belong to the requested true task')

    local_masked = torch.full_like(own_logits, float('-inf'))
    local_masked[:, local_classes] = own_logits[:, local_classes]
    seen_masked = torch.full_like(own_logits, float('-inf'))
    seen_masked[:, seen_classes] = own_logits[:, seen_classes]

    local_prediction = local_masked.argmax(dim=1)
    seen_prediction = seen_masked.argmax(dim=1)
    local_correct = local_prediction.eq(targets)
    seen_correct = seen_prediction.eq(targets)

    class_to_task = torch.full(
        (own_logits.shape[1],), -1, dtype=torch.long, device=targets.device)
    for task_index in range(int(seen_task_count)):
        task_classes = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=targets.device)
        class_to_task[task_classes] = task_index
    seen_task_correct = class_to_task[seen_prediction].eq(int(true_task))

    return {
        'stage_drift_own_local_correct': local_correct,
        'stage_drift_own_seen_correct': seen_correct,
        'stage_drift_own_seen_task_correct': seen_task_correct,
        'stage_drift_local_to_seen_failure': local_correct & ~seen_correct,
        'stage_drift_own_local_loss': torch.nn.functional.cross_entropy(
            local_masked, targets, reduction='none'),
        'stage_drift_own_seen_loss': torch.nn.functional.cross_entropy(
            seen_masked, targets, reduction='none'),
    }


def prediction_proposal_candidates(
        tii_ranking, initial_adapter_logits, class_mask, initial_count=2,
        proposal_count=2, top_classes=5):
    """Build a fixed-budget hard task set from post-LoRA class predictions."""
    batch_size, seen_task_count = tii_ranking.shape
    device = tii_ranking.device
    initial_count = min(max(1, int(initial_count)), seen_task_count)
    proposal_count = min(
        max(0, int(proposal_count)), seen_task_count - initial_count)

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    class_to_task = torch.full(
        (initial_adapter_logits.shape[-1],), -1,
        dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = task_index

    proposal_scores = torch.full(
        (batch_size, seen_task_count), float('-inf'),
        dtype=initial_adapter_logits.dtype, device=device)
    top_classes = min(max(1, int(top_classes)), len(seen_classes))
    for initial_rank in range(initial_count):
        seen_logits = initial_adapter_logits[:, initial_rank].index_select(
            1, seen_class_index)
        top_values, top_positions = torch.topk(
            seen_logits, k=top_classes, dim=1)
        proposed_tasks = class_to_task[seen_class_index[top_positions]]
        proposal_scores.scatter_reduce_(
            1, proposed_tasks, top_values, reduce='amax', include_self=True)

    initial_tasks = tii_ranking[:, :initial_count]
    candidate_parts = [initial_tasks]
    candidate_mask = torch.zeros(
        (batch_size, seen_task_count), dtype=torch.bool, device=device)
    candidate_mask.scatter_(1, initial_tasks, True)
    proposal_scores = proposal_scores.masked_fill(candidate_mask, float('-inf'))

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
        candidate_parts.append(selected_task.unsqueeze(1))
        candidate_mask.scatter_(1, selected_task.unsqueeze(1), True)
        proposal_scores.scatter_(
            1, selected_task.unsqueeze(1), float('-inf'))

    return torch.cat(candidate_parts, dim=1), candidate_mask


def prediction_proposal_diagnostics(
        tii_ranking, initial_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, class_mask, initial_count=2,
        proposal_count=2, top_classes=5, excluded_margin=20.0):
    """Audit task proposals induced by predictions from initial hard LoRAs."""
    seen_task_count = tii_ranking.shape[1]
    _, candidate_mask = prediction_proposal_candidates(
        tii_ranking,
        initial_adapter_logits,
        class_mask,
        initial_count=initial_count,
        proposal_count=proposal_count,
        top_classes=top_classes,
    )
    selected_rank_mask = candidate_mask.gather(1, tii_ranking)
    candidate_evidence = task_evidence.masked_fill(
        ~selected_rank_mask, float('-inf'))
    selected_rank = candidate_evidence.argmax(dim=1)
    selected_tasks = tii_ranking.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    selected_logits = rank_logits.masked_fill(
        ~selected_rank_mask.unsqueeze(2), float('-inf')).max(dim=1).values
    selected_logits = _finalize_partial_logits(
        selected_logits, excluded_margin)

    winner_recall = candidate_mask.gather(
        1, winner_tasks.unsqueeze(1)).squeeze(1)
    exact_agreement = (
        selected_tasks.eq(winner_tasks)
        & selected_logits.argmax(dim=1).eq(full_predictions)
    )
    tii_top4_count = min(4, seen_task_count)
    tii_top4_recall = tii_ranking[:, :tii_top4_count].eq(
        winner_tasks.unsqueeze(1)).any(dim=1)
    return {
        'prediction_proposal_winner_recall': winner_recall,
        'prediction_proposal_exact_agreement': exact_agreement,
        'prediction_proposal_lora_counts': candidate_mask.sum(dim=1).float(),
        'prediction_proposal_new_winner': winner_recall & ~tii_top4_recall,
    }


def prediction_closure_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, class_mask, initial_count=2,
        top_classes=5, excluded_margin=20.0, tii_logits=None,
        tii_tail_completion=False):
    """Expand prediction-induced tasks until the evaluated set is closed."""
    batch_size, seen_task_count, _ = full_adapter_logits.shape
    device = full_adapter_logits.device
    initial_count = min(max(1, int(initial_count)), seen_task_count)

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    class_to_task = torch.full(
        (full_adapter_logits.shape[-1],), -1,
        dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = task_index
    top_classes = min(max(1, int(top_classes)), len(seen_classes))

    candidate_mask = torch.zeros(
        (batch_size, seen_task_count), dtype=torch.bool, device=device)
    candidate_mask.scatter_(1, tii_ranking[:, :initial_count], True)
    processed_rank_mask = torch.zeros_like(candidate_mask)
    forward_calls = torch.ones(
        batch_size, dtype=torch.float32, device=device)

    for _ in range(seen_task_count):
        selected_rank_mask = candidate_mask.gather(1, tii_ranking)
        new_rank_mask = selected_rank_mask & ~processed_rank_mask
        if not new_rank_mask.any():
            break
        processed_rank_mask |= new_rank_mask

        proposed_mask = torch.zeros_like(candidate_mask)
        for rank in range(seen_task_count):
            rows = torch.nonzero(new_rank_mask[:, rank]).flatten()
            if rows.numel() == 0:
                continue
            seen_logits = full_adapter_logits[
                rows, rank].index_select(1, seen_class_index)
            top_positions = torch.topk(
                seen_logits, k=top_classes, dim=1).indices
            proposed_tasks = class_to_task[
                seen_class_index[top_positions]]
            proposed_mask[
                rows.unsqueeze(1), proposed_tasks
            ] = True

        additions = proposed_mask & ~candidate_mask
        expanding = additions.any(dim=1)
        if not expanding.any():
            break
        candidate_mask |= additions
        forward_calls[expanding] += 1.0

    selected_rank_mask = candidate_mask.gather(1, tii_ranking)
    candidate_evidence = task_evidence.masked_fill(
        ~selected_rank_mask, float('-inf'))
    selected_rank = candidate_evidence.argmax(dim=1)
    selected_tasks = tii_ranking.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    selected_partial_logits = rank_logits.masked_fill(
        ~selected_rank_mask.unsqueeze(2), float('-inf')).max(dim=1).values
    selected_logits = _finalize_partial_logits(
        selected_partial_logits, excluded_margin)

    winner_recall = candidate_mask.gather(
        1, winner_tasks.unsqueeze(1)).squeeze(1)
    exact_agreement = (
        selected_tasks.eq(winner_tasks)
        & selected_logits.argmax(dim=1).eq(full_predictions)
    )

    full_logits = rank_logits.max(dim=1).values
    top5_count = min(5, len(seen_classes))
    full_top5_positions = torch.topk(
        full_logits.index_select(1, seen_class_index),
        k=top5_count, dim=1).indices
    full_top5_tasks = class_to_task[
        seen_class_index[full_top5_positions]]
    top5_coverage = candidate_mask.gather(
        1, full_top5_tasks).all(dim=1)
    lora_counts = candidate_mask.sum(dim=1).float()

    output_logits = selected_logits
    if tii_tail_completion:
        if tii_logits is None:
            raise ValueError(
                'TII logits are required for closure tail completion')
        output_logits = complete_with_tii_probability_mass(
            selected_partial_logits, tii_logits, class_mask, candidate_mask)

    return {
        'prediction_closure_winner_recall': winner_recall,
        'prediction_closure_exact_agreement': exact_agreement,
        'prediction_closure_top5_coverage': top5_coverage,
        'prediction_closure_lora_counts': lora_counts,
        'prediction_closure_forward_calls': forward_calls,
        'prediction_closure_full_scan': lora_counts.eq(
            float(seen_task_count)),
        'prediction_closure_output_logits': output_logits,
        'prediction_closure_output_tasks': selected_tasks,
    }


def prediction_beam_closure_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, tii_logits, class_mask,
        initial_count=2, top_classes=5, beam_width=2,
        excluded_margin=20.0):
    """Expand only from the strongest evaluated adapters until they close."""
    batch_size, seen_task_count, _ = full_adapter_logits.shape
    device = full_adapter_logits.device
    initial_count = min(max(1, int(initial_count)), seen_task_count)
    beam_width = min(max(1, int(beam_width)), seen_task_count)

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    class_to_task = torch.full(
        (full_adapter_logits.shape[-1],), -1,
        dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = task_index
    top_classes = min(max(1, int(top_classes)), len(seen_classes))

    candidate_mask = torch.zeros(
        (batch_size, seen_task_count), dtype=torch.bool, device=device)
    candidate_mask.scatter_(1, tii_ranking[:, :initial_count], True)
    forward_calls = torch.ones(
        batch_size, dtype=torch.float32, device=device)
    batch_index = torch.arange(batch_size, device=device)

    for _ in range(seen_task_count + 1):
        selected_rank_mask = candidate_mask.gather(1, tii_ranking)
        leader_evidence = task_evidence.masked_fill(
            ~selected_rank_mask, float('-inf'))
        active_beam = min(beam_width, initial_count)
        leader_ranks = torch.topk(
            leader_evidence, k=active_beam, dim=1).indices

        proposed_mask = torch.zeros_like(candidate_mask)
        for beam_index in range(active_beam):
            leader_rank = leader_ranks[:, beam_index]
            leader_logits = full_adapter_logits[batch_index, leader_rank]
            top_positions = torch.topk(
                leader_logits.index_select(1, seen_class_index),
                k=top_classes, dim=1).indices
            proposed_tasks = class_to_task[seen_class_index[top_positions]]
            unseen = ~candidate_mask.gather(1, proposed_tasks)
            has_successor = unseen.any(dim=1)
            first_unseen = unseen.float().argmax(dim=1)
            successor = proposed_tasks.gather(
                1, first_unseen.unsqueeze(1)).squeeze(1)
            rows = torch.nonzero(has_successor).flatten()
            if rows.numel() > 0:
                proposed_mask[rows, successor[rows]] = True

        additions = proposed_mask & ~candidate_mask
        expanding = additions.any(dim=1)
        if not expanding.any():
            break
        candidate_mask |= additions
        forward_calls[expanding] += 1.0

    selected_rank_mask = candidate_mask.gather(1, tii_ranking)
    candidate_evidence = task_evidence.masked_fill(
        ~selected_rank_mask, float('-inf'))
    selected_rank = candidate_evidence.argmax(dim=1)
    selected_tasks = tii_ranking.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    selected_partial_logits = rank_logits.masked_fill(
        ~selected_rank_mask.unsqueeze(2), float('-inf')).max(dim=1).values
    selected_logits = _finalize_partial_logits(
        selected_partial_logits, excluded_margin)

    winner_recall = candidate_mask.gather(
        1, winner_tasks.unsqueeze(1)).squeeze(1)
    exact_agreement = (
        selected_tasks.eq(winner_tasks)
        & selected_logits.argmax(dim=1).eq(full_predictions)
    )
    lora_counts = candidate_mask.sum(dim=1).float()
    output_logits = complete_with_tii_probability_mass(
        selected_partial_logits, tii_logits, class_mask, candidate_mask)

    return {
        'prediction_beam_closure_winner_recall': winner_recall,
        'prediction_beam_closure_exact_agreement': exact_agreement,
        'prediction_beam_closure_lora_counts': lora_counts,
        'prediction_beam_closure_forward_calls': forward_calls,
        'prediction_beam_closure_full_scan': lora_counts.eq(
            float(seen_task_count)),
        'prediction_beam_closure_output_logits': output_logits,
        'prediction_beam_closure_output_tasks': selected_tasks,
    }


def prediction_budget_closure_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, tii_logits, class_mask,
        initial_count=2, top_classes=5, max_candidates=5,
        excluded_margin=20.0):
    """Follow one successor per new frontier under a fixed task cap."""
    batch_size, seen_task_count, _ = full_adapter_logits.shape
    device = full_adapter_logits.device
    initial_count = min(max(1, int(initial_count)), seen_task_count)
    max_candidates = min(
        max(initial_count, int(max_candidates)), seen_task_count)

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    class_to_task = torch.full(
        (full_adapter_logits.shape[-1],), -1,
        dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = task_index
    top_classes = min(max(1, int(top_classes)), len(seen_classes))

    candidate_mask = torch.zeros(
        (batch_size, seen_task_count), dtype=torch.bool, device=device)
    candidate_mask.scatter_(1, tii_ranking[:, :initial_count], True)
    processed_rank_mask = torch.zeros_like(candidate_mask)
    forward_calls = torch.ones(
        batch_size, dtype=torch.float32, device=device)
    max_additions = max_candidates - initial_count

    for _ in range(seen_task_count + 1):
        selected_rank_mask = candidate_mask.gather(1, tii_ranking)
        new_rank_mask = selected_rank_mask & ~processed_rank_mask
        if not new_rank_mask.any() or max_additions == 0:
            break
        processed_rank_mask |= new_rank_mask

        proposal_scores = torch.full(
            (batch_size, seen_task_count), float('-inf'),
            dtype=full_adapter_logits.dtype, device=device)
        for rank in range(seen_task_count):
            rows = torch.nonzero(new_rank_mask[:, rank]).flatten()
            if rows.numel() == 0:
                continue
            seen_logits = full_adapter_logits[
                rows, rank].index_select(1, seen_class_index)
            top_values, top_positions = torch.topk(
                seen_logits, k=top_classes, dim=1)
            proposed_tasks = class_to_task[seen_class_index[top_positions]]
            row_candidate_mask = candidate_mask.index_select(0, rows)
            unseen = ~row_candidate_mask.gather(1, proposed_tasks)
            has_successor = unseen.any(dim=1)
            first_unseen = unseen.float().argmax(dim=1)
            successor_tasks = proposed_tasks.gather(
                1, first_unseen.unsqueeze(1)).squeeze(1)
            successor_values = top_values.gather(
                1, first_unseen.unsqueeze(1)).squeeze(1)
            active = torch.nonzero(has_successor).flatten()
            if active.numel() > 0:
                active_rows = rows[active]
                row_scores = proposal_scores.index_select(0, active_rows)
                row_scores.scatter_reduce_(
                    1, successor_tasks[active].unsqueeze(1),
                    successor_values[active].unsqueeze(1),
                    reduce='amax', include_self=True)
                proposal_scores[active_rows] = row_scores

        proposal_scores = proposal_scores.masked_fill(
            candidate_mask, float('-inf'))
        proposal_values, proposal_tasks = torch.topk(
            proposal_scores, k=max_additions, dim=1)
        remaining = max_candidates - candidate_mask.sum(dim=1)
        additions = torch.zeros_like(candidate_mask)
        for slot in range(max_additions):
            valid = remaining.gt(slot) & torch.isfinite(
                proposal_values[:, slot])
            rows = torch.nonzero(valid).flatten()
            if rows.numel() > 0:
                additions[rows, proposal_tasks[rows, slot]] = True

        expanding = additions.any(dim=1)
        if not expanding.any():
            break
        candidate_mask |= additions
        forward_calls[expanding] += 1.0
        if candidate_mask.sum(dim=1).ge(max_candidates).all():
            break

    selected_rank_mask = candidate_mask.gather(1, tii_ranking)
    candidate_evidence = task_evidence.masked_fill(
        ~selected_rank_mask, float('-inf'))
    selected_rank = candidate_evidence.argmax(dim=1)
    selected_tasks = tii_ranking.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    selected_partial_logits = rank_logits.masked_fill(
        ~selected_rank_mask.unsqueeze(2), float('-inf')).max(dim=1).values
    selected_logits = _finalize_partial_logits(
        selected_partial_logits, excluded_margin)

    winner_recall = candidate_mask.gather(
        1, winner_tasks.unsqueeze(1)).squeeze(1)
    exact_agreement = (
        selected_tasks.eq(winner_tasks)
        & selected_logits.argmax(dim=1).eq(full_predictions)
    )
    lora_counts = candidate_mask.sum(dim=1).float()
    output_logits = complete_with_tii_probability_mass(
        selected_partial_logits, tii_logits, class_mask, candidate_mask)

    return {
        'prediction_budget_closure_winner_recall': winner_recall,
        'prediction_budget_closure_exact_agreement': exact_agreement,
        'prediction_budget_closure_lora_counts': lora_counts,
        'prediction_budget_closure_forward_calls': forward_calls,
        'prediction_budget_closure_budget_hit': lora_counts.eq(
            float(max_candidates)),
        'prediction_budget_closure_candidate_mask': candidate_mask,
        'prediction_budget_closure_output_logits': output_logits,
        'prediction_budget_closure_output_tasks': selected_tasks,
    }


def prediction_majority_budget_closure_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, tii_logits, class_mask,
        initial_count=2, top_classes=5, max_candidates=5,
        excluded_margin=20.0):
    """Keep a strict TII-majority anchor; use cap closure when ambiguous."""
    budget = prediction_budget_closure_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=task_evidence,
        winner_tasks=winner_tasks,
        full_predictions=full_predictions,
        tii_logits=tii_logits,
        class_mask=class_mask,
        initial_count=initial_count,
        top_classes=top_classes,
        max_candidates=max_candidates,
        excluded_margin=excluded_margin,
    )
    batch_size, seen_task_count = tii_ranking.shape
    device = tii_ranking.device

    seen_class_mask = torch.zeros(
        tii_logits.shape[1], dtype=torch.bool, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        seen_class_mask[class_index] = True
    seen_tii_logits = tii_logits.masked_fill(
        ~seen_class_mask.unsqueeze(0), float('-inf'))
    class_probabilities = torch.softmax(seen_tii_logits, dim=1)
    task_mass = torch.zeros(
        (batch_size, seen_task_count), dtype=tii_logits.dtype, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        task_mass[:, task_index] = class_probabilities.index_select(
            1, class_index).sum(dim=1)

    top1_tasks = tii_ranking[:, 0]
    top1_mass = task_mass.gather(1, top1_tasks.unsqueeze(1)).squeeze(1)
    majority_certified = top1_mass.gt(0.5)
    top1_candidate_mask = torch.zeros(
        (batch_size, seen_task_count), dtype=torch.bool, device=device)
    top1_candidate_mask.scatter_(1, top1_tasks.unsqueeze(1), True)
    top1_partial_logits = rank_logits[:, 0]
    top1_logits = complete_with_tii_probability_mass(
        top1_partial_logits, tii_logits, class_mask, top1_candidate_mask)

    output_logits = torch.where(
        majority_certified.unsqueeze(1), top1_logits,
        budget['prediction_budget_closure_output_logits'])
    output_tasks = torch.where(
        majority_certified, top1_tasks,
        budget['prediction_budget_closure_output_tasks'])
    output_predictions = output_logits.argmax(dim=1)
    exact_agreement = (
        output_tasks.eq(winner_tasks)
        & output_predictions.eq(full_predictions)
    )
    top1_winner = top1_tasks.eq(winner_tasks)
    winner_recall = torch.where(
        majority_certified, top1_winner,
        budget['prediction_budget_closure_winner_recall'])
    lora_counts = torch.where(
        majority_certified,
        torch.ones_like(budget['prediction_budget_closure_lora_counts']),
        budget['prediction_budget_closure_lora_counts'])
    forward_calls = torch.where(
        majority_certified,
        torch.ones_like(
            budget['prediction_budget_closure_forward_calls']),
        budget['prediction_budget_closure_forward_calls'])

    return {
        'prediction_majority_closure_winner_recall': winner_recall,
        'prediction_majority_closure_exact_agreement': exact_agreement,
        'prediction_majority_closure_majority_rate': majority_certified,
        'prediction_majority_closure_lora_counts': lora_counts,
        'prediction_majority_closure_forward_calls': forward_calls,
        'prediction_majority_closure_output_logits': output_logits,
        'prediction_majority_closure_output_tasks': output_tasks,
    }

def prediction_consensus_budget_closure_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, tii_logits, class_mask,
        initial_count=2, top_classes=5, max_candidates=5,
        excluded_margin=20.0):
    """Anchor only when both initial adapters endorse the TII top task."""
    budget = prediction_budget_closure_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=task_evidence,
        winner_tasks=winner_tasks,
        full_predictions=full_predictions,
        tii_logits=tii_logits,
        class_mask=class_mask,
        initial_count=initial_count,
        top_classes=top_classes,
        max_candidates=max_candidates,
        excluded_margin=excluded_margin,
    )
    batch_size, seen_task_count = tii_ranking.shape
    device = tii_ranking.device
    initial_count = min(max(1, int(initial_count)), seen_task_count)

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    class_to_task = torch.full(
        (full_adapter_logits.shape[-1],), -1,
        dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = task_index

    top1_tasks = tii_ranking[:, 0]
    raw_consensus = torch.ones(
        batch_size, dtype=torch.bool, device=device)
    for rank in range(initial_count):
        raw_seen_logits = full_adapter_logits[
            :, rank].index_select(1, seen_class_index)
        raw_class = seen_class_index[raw_seen_logits.argmax(dim=1)]
        raw_task = class_to_task[raw_class]
        raw_consensus &= raw_task.eq(top1_tasks)
    if initial_count > 1:
        top1_local_wins = task_evidence[:, 0].ge(
            task_evidence[:, 1:initial_count].max(dim=1).values)
    else:
        top1_local_wins = torch.ones_like(raw_consensus)
    consensus_certified = raw_consensus & top1_local_wins

    top1_candidate_mask = torch.zeros(
        (batch_size, seen_task_count), dtype=torch.bool, device=device)
    top1_candidate_mask.scatter_(1, top1_tasks.unsqueeze(1), True)
    top1_logits = complete_with_tii_probability_mass(
        rank_logits[:, 0], tii_logits, class_mask, top1_candidate_mask)
    output_logits = torch.where(
        consensus_certified.unsqueeze(1), top1_logits,
        budget['prediction_budget_closure_output_logits'])
    output_tasks = torch.where(
        consensus_certified, top1_tasks,
        budget['prediction_budget_closure_output_tasks'])
    exact_agreement = (
        output_tasks.eq(winner_tasks)
        & output_logits.argmax(dim=1).eq(full_predictions)
    )
    winner_recall = torch.where(
        consensus_certified, top1_tasks.eq(winner_tasks),
        budget['prediction_budget_closure_winner_recall'])
    certified_cost = torch.full_like(
        budget['prediction_budget_closure_lora_counts'],
        float(initial_count))
    lora_counts = torch.where(
        consensus_certified, certified_cost,
        budget['prediction_budget_closure_lora_counts'])
    forward_calls = torch.where(
        consensus_certified,
        torch.ones_like(
            budget['prediction_budget_closure_forward_calls']),
        budget['prediction_budget_closure_forward_calls'])
    initial_candidate_mask = torch.zeros_like(
        budget['prediction_budget_closure_candidate_mask'])
    initial_candidate_mask.scatter_(
        1, tii_ranking[:, :initial_count], True)
    operational_candidate_mask = torch.where(
        consensus_certified.unsqueeze(1), initial_candidate_mask,
        budget['prediction_budget_closure_candidate_mask'])

    return {
        'prediction_consensus_closure_winner_recall': winner_recall,
        'prediction_consensus_closure_exact_agreement': exact_agreement,
        'prediction_consensus_closure_certified_rate': consensus_certified,
        'prediction_consensus_closure_lora_counts': lora_counts,
        'prediction_consensus_closure_forward_calls': forward_calls,
        'prediction_consensus_closure_candidate_mask':
            operational_candidate_mask,
        'prediction_consensus_closure_output_logits': output_logits,
        'prediction_consensus_closure_output_tasks': output_tasks,
    }


def prediction_self_owner_consensus_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, tii_logits, class_mask,
        initial_count=2, top_classes=5, max_candidates=5,
        excluded_margin=20.0):
    """Route to a self-endorsing evaluated adapter with task-vote support.

    An adapter self-endorses when its raw top-1 seen-class prediction belongs
    to its own task. Among self-endorsing candidates, task vote count is the
    primary label-free key and TII rank breaks ties. Samples without a
    self-endorsing candidate retain the locked consensus-cap5 output.
    """
    consensus = prediction_consensus_budget_closure_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=task_evidence,
        winner_tasks=winner_tasks,
        full_predictions=full_predictions,
        tii_logits=tii_logits,
        class_mask=class_mask,
        initial_count=initial_count,
        top_classes=top_classes,
        max_candidates=max_candidates,
        excluded_margin=excluded_margin,
    )
    batch_size, seen_task_count = tii_ranking.shape
    device = tii_ranking.device
    candidate_mask = consensus[
        'prediction_consensus_closure_candidate_mask']
    selected_rank_mask = candidate_mask.gather(1, tii_ranking)

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    class_to_task = torch.full(
        (full_adapter_logits.shape[-1],), -1,
        dtype=torch.long, device=device)
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        class_to_task[class_index] = task_index

    raw_seen_logits = full_adapter_logits.index_select(
        2, seen_class_index)
    raw_class = seen_class_index[raw_seen_logits.argmax(dim=2)]
    raw_task = class_to_task[raw_class]
    self_endorsed_rank = (
        raw_task.eq(tii_ranking) & selected_rank_mask)

    support = torch.zeros(
        (batch_size, seen_task_count),
        dtype=torch.long, device=device)
    support.scatter_add_(
        1,
        raw_task.clamp_min(0),
        selected_rank_mask.to(torch.long))
    rank_support = support.gather(1, tii_ranking).masked_fill(
        ~self_endorsed_rank, -1)
    selected_rank = rank_support.argmax(dim=1)
    has_self_owner = self_endorsed_rank.any(dim=1)
    selected_tasks = tii_ranking.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    selected_support = rank_support.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1).clamp_min(0)
    rows = torch.arange(batch_size, device=device)
    selected_raw_logits = full_adapter_logits[rows, selected_rank]
    selected_seen_logits = torch.full_like(
        selected_raw_logits, float('-inf'))
    selected_seen_logits[:, seen_class_index] = selected_raw_logits[
        :, seen_class_index]

    output_logits = torch.where(
        has_self_owner.unsqueeze(1), selected_seen_logits,
        consensus['prediction_consensus_closure_output_logits'])
    output_tasks = torch.where(
        has_self_owner, selected_tasks,
        consensus['prediction_consensus_closure_output_tasks'])
    output_predictions = output_logits.argmax(dim=1)
    exact_agreement = (
        output_tasks.eq(winner_tasks)
        & output_predictions.eq(full_predictions)
    )
    winner_recall = candidate_mask.gather(
        1, winner_tasks.unsqueeze(1)).squeeze(1)

    return {
        'prediction_self_owner_winner_recall': winner_recall,
        'prediction_self_owner_exact_agreement': exact_agreement,
        'prediction_self_owner_route_rate': has_self_owner,
        'prediction_self_owner_multiple_rate':
            self_endorsed_rank.sum(dim=1).gt(1),
        'prediction_self_owner_support': selected_support.to(
            full_adapter_logits.dtype),
        'prediction_self_owner_lora_counts': consensus[
            'prediction_consensus_closure_lora_counts'],
        'prediction_self_owner_forward_calls': consensus[
            'prediction_consensus_closure_forward_calls'],
        'prediction_self_owner_candidate_mask': candidate_mask,
        'prediction_self_owner_selected_tasks': selected_tasks,
        'prediction_self_owner_consensus_logits': consensus[
            'prediction_consensus_closure_output_logits'],
        'prediction_self_owner_consensus_tasks': consensus[
            'prediction_consensus_closure_output_tasks'],
        'prediction_self_owner_output_logits': output_logits,
        'prediction_self_owner_output_tasks': output_tasks,
    }


def prediction_corroborated_self_owner_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, tii_logits, class_mask,
        initial_count=2, top_classes=5, max_candidates=5,
        excluded_margin=20.0):
    """Rescue consensus only with independently corroborated self-ownership.

    The selected self-owner must disagree with the locked consensus route and
    receive at least two task votes from already-evaluated adapters. Rescued
    samples use only the selected adapter's task-local calibrated logits plus
    the same top-1-safe TII tail; all other samples preserve consensus exactly.
    """
    self_owner = prediction_self_owner_consensus_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=task_evidence,
        winner_tasks=winner_tasks,
        full_predictions=full_predictions,
        tii_logits=tii_logits,
        class_mask=class_mask,
        initial_count=initial_count,
        top_classes=top_classes,
        max_candidates=max_candidates,
        excluded_margin=excluded_margin,
    )
    selected_tasks = self_owner['prediction_self_owner_selected_tasks']
    consensus_tasks = self_owner['prediction_self_owner_consensus_tasks']
    selected_support = self_owner['prediction_self_owner_support']
    rescue = (
        self_owner['prediction_self_owner_route_rate']
        & selected_support.ge(2.0)
        & selected_tasks.ne(consensus_tasks)
    )

    batch_size = tii_ranking.shape[0]
    device = tii_ranking.device
    selected_rank = tii_ranking.eq(
        selected_tasks.unsqueeze(1)).to(torch.int64).argmax(dim=1)
    rows = torch.arange(batch_size, device=device)
    selected_local_logits = rank_logits[rows, selected_rank]
    selected_candidate_mask = torch.zeros_like(
        self_owner['prediction_self_owner_candidate_mask'])
    selected_candidate_mask.scatter_(
        1, selected_tasks.unsqueeze(1), True)
    rescued_logits = complete_with_tii_probability_mass(
        selected_local_logits, tii_logits, class_mask,
        selected_candidate_mask)

    output_logits = torch.where(
        rescue.unsqueeze(1), rescued_logits,
        self_owner['prediction_self_owner_consensus_logits'])
    output_tasks = torch.where(
        rescue, selected_tasks,
        consensus_tasks)
    output_predictions = output_logits.argmax(dim=1)
    exact_agreement = (
        output_tasks.eq(winner_tasks)
        & output_predictions.eq(full_predictions)
    )
    candidate_mask = self_owner[
        'prediction_self_owner_candidate_mask']
    winner_recall = candidate_mask.gather(
        1, winner_tasks.unsqueeze(1)).squeeze(1)

    return {
        'prediction_corroborated_owner_winner_recall': winner_recall,
        'prediction_corroborated_owner_exact_agreement': exact_agreement,
        'prediction_corroborated_owner_rescue_rate': rescue,
        'prediction_corroborated_owner_support': torch.where(
            rescue, selected_support, torch.zeros_like(selected_support)),
        'prediction_corroborated_owner_lora_counts': self_owner[
            'prediction_self_owner_lora_counts'],
        'prediction_corroborated_owner_forward_calls': self_owner[
            'prediction_self_owner_forward_calls'],
        'prediction_corroborated_owner_output_logits': output_logits,
        'prediction_corroborated_owner_output_tasks': output_tasks,
    }


def prediction_owner_mass_consensus_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, tii_logits, class_mask,
        initial_count=2, top_classes=5, max_candidates=5,
        excluded_margin=20.0):
    """Rescue uncertified consensus using adapter owner-task mass.

    For each already-evaluated adapter, the routing score is
    log P(y belongs to the adapter's own task | x), computed over all seen
    classes. This comparison is offset-invariant and keeps within-task class
    ranking untouched. Certified consensus rows and non-rerouted ambiguous
    rows remain bitwise equal to the locked consensus-cap5 output.
    """
    consensus = prediction_consensus_budget_closure_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=task_evidence,
        winner_tasks=winner_tasks,
        full_predictions=full_predictions,
        tii_logits=tii_logits,
        class_mask=class_mask,
        initial_count=initial_count,
        top_classes=top_classes,
        max_candidates=max_candidates,
        excluded_margin=excluded_margin,
    )
    batch_size, seen_task_count = tii_ranking.shape
    device = tii_ranking.device
    candidate_mask = consensus[
        'prediction_consensus_closure_candidate_mask']
    selected_rank_mask = candidate_mask.gather(1, tii_ranking)

    seen_classes = [
        int(class_id)
        for task_index in range(seen_task_count)
        for class_id in class_mask[task_index]
    ]
    seen_class_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    owner_log_mass = torch.full(
        (batch_size, seen_task_count), float('-inf'),
        dtype=full_adapter_logits.dtype, device=device)
    for rank in range(seen_task_count):
        active_tasks = tii_ranking[:, rank]
        adapter_seen_logits = full_adapter_logits[
            :, rank].index_select(1, seen_class_index)
        seen_log_mass = torch.logsumexp(adapter_seen_logits, dim=1)
        for task_index in active_tasks.unique().tolist():
            rows = torch.nonzero(active_tasks == task_index).flatten()
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            local_log_mass = torch.logsumexp(
                full_adapter_logits[rows, rank].index_select(
                    1, class_index), dim=1)
            owner_log_mass[rows, rank] = (
                local_log_mass - seen_log_mass.index_select(0, rows))
    owner_log_mass = owner_log_mass.masked_fill(
        ~selected_rank_mask, float('-inf'))
    selected_rank = owner_log_mass.argmax(dim=1)
    selected_tasks = tii_ranking.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    consensus_tasks = consensus[
        'prediction_consensus_closure_output_tasks']
    ambiguous = ~consensus[
        'prediction_consensus_closure_certified_rate']
    rescue = ambiguous & selected_tasks.ne(consensus_tasks)

    rows = torch.arange(batch_size, device=device)
    selected_local_logits = rank_logits[rows, selected_rank]
    selected_candidate_mask = torch.zeros_like(candidate_mask)
    selected_candidate_mask.scatter_(
        1, selected_tasks.unsqueeze(1), True)
    rescued_logits = complete_with_tii_probability_mass(
        selected_local_logits, tii_logits, class_mask,
        selected_candidate_mask)
    output_logits = torch.where(
        rescue.unsqueeze(1), rescued_logits,
        consensus['prediction_consensus_closure_output_logits'])
    output_tasks = torch.where(
        rescue, selected_tasks, consensus_tasks)
    output_predictions = output_logits.argmax(dim=1)
    exact_agreement = (
        output_tasks.eq(winner_tasks)
        & output_predictions.eq(full_predictions)
    )
    winner_recall = candidate_mask.gather(
        1, winner_tasks.unsqueeze(1)).squeeze(1)

    return {
        'prediction_owner_mass_winner_recall': winner_recall,
        'prediction_owner_mass_exact_agreement': exact_agreement,
        'prediction_owner_mass_rescue_rate': rescue,
        'prediction_owner_mass_ambiguous_rate': ambiguous,
        'prediction_owner_mass_lora_counts': consensus[
            'prediction_consensus_closure_lora_counts'],
        'prediction_owner_mass_forward_calls': consensus[
            'prediction_consensus_closure_forward_calls'],
        'prediction_owner_mass_candidate_mask': candidate_mask,
        'prediction_owner_mass_consensus_logits': consensus[
            'prediction_consensus_closure_output_logits'],
        'prediction_owner_mass_consensus_tasks': consensus_tasks,
        'prediction_owner_mass_output_logits': output_logits,
        'prediction_owner_mass_output_tasks': output_tasks,
    }


def prediction_owner_consensus_hedge_diagnostics(
        tii_ranking, full_adapter_logits, rank_logits, task_evidence,
        winner_tasks, full_predictions, tii_logits, class_mask,
        initial_count=2, top_classes=5, max_candidates=5,
        excluded_margin=20.0):
    """Hedge owner reroutes with an equal-prior consensus probability mix."""
    owner = prediction_owner_mass_consensus_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=task_evidence,
        winner_tasks=winner_tasks,
        full_predictions=full_predictions,
        tii_logits=tii_logits,
        class_mask=class_mask,
        initial_count=initial_count,
        top_classes=top_classes,
        max_candidates=max_candidates,
        excluded_margin=excluded_margin,
    )
    owner_logits = owner['prediction_owner_mass_output_logits']
    consensus_logits = owner['prediction_owner_mass_consensus_logits']
    hedge = owner['prediction_owner_mass_rescue_rate']
    equal_mix_logits = torch.logaddexp(
        owner_logits, consensus_logits) - owner_logits.new_tensor(2.0).log()
    output_logits = torch.where(
        hedge.unsqueeze(1), equal_mix_logits, owner_logits)
    output_predictions = output_logits.argmax(dim=1)

    class_to_task = torch.full(
        (output_logits.shape[1],), -1, dtype=torch.long,
        device=output_logits.device)
    for task_index in range(tii_ranking.shape[1]):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long,
            device=output_logits.device)
        class_to_task[class_index] = task_index
    mixed_tasks = class_to_task[output_predictions]
    owner_tasks = owner['prediction_owner_mass_output_tasks']
    output_tasks = torch.where(hedge, mixed_tasks, owner_tasks)
    exact_agreement = (
        output_tasks.eq(winner_tasks)
        & output_predictions.eq(full_predictions)
    )
    owner_predictions = owner_logits.argmax(dim=1)
    consensus_predictions = consensus_logits.argmax(dim=1)

    return {
        'prediction_owner_hedge_winner_recall': owner[
            'prediction_owner_mass_winner_recall'],
        'prediction_owner_hedge_exact_agreement': exact_agreement,
        'prediction_owner_hedge_rate': hedge,
        'prediction_owner_hedge_owner_agreement':
            output_predictions.eq(owner_predictions),
        'prediction_owner_hedge_consensus_agreement':
            output_predictions.eq(consensus_predictions),
        'prediction_owner_hedge_lora_counts': owner[
            'prediction_owner_mass_lora_counts'],
        'prediction_owner_hedge_forward_calls': owner[
            'prediction_owner_mass_forward_calls'],
        'prediction_owner_hedge_output_logits': output_logits,
        'prediction_owner_hedge_output_tasks': output_tasks,
    }

@torch.no_grad()
def router_recall_diagnostics(tii_logits, class_mask, seen_task_count,
                              winner_tasks):
    """Rank the exhaustive winner task under alternative TII aggregations.

    Every score below is computed only from ``tii_logits`` (no LoRA forward),
    so any aggregation that ranks the winner higher than the current ``max``
    router is a free, strict cost reduction: fewer LoRAs need to be evaluated
    before the winner is covered. ``winner_tasks`` is the task the full
    exhaustive evaluation routes to.
    """
    device = tii_logits.device
    max_scores = []
    energy_scores = []
    margin_scores = []
    mean_scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        task_logits = tii_logits.index_select(1, class_index)
        max_scores.append(task_logits.max(dim=1).values)
        energy_scores.append(torch.logsumexp(task_logits, dim=1))
        mean_scores.append(task_logits.mean(dim=1))
        top2 = task_logits.topk(
            k=min(2, task_logits.shape[1]), dim=1).values
        if top2.shape[1] >= 2:
            margin_scores.append(top2[:, 0] - top2[:, 1])
        else:
            margin_scores.append(top2[:, 0])
    routers = {
        'max': torch.stack(max_scores, dim=1),
        'energy': torch.stack(energy_scores, dim=1),
        'margin': torch.stack(margin_scores, dim=1),
        'mean': torch.stack(mean_scores, dim=1),
    }
    diagnostics = {}
    winner_index = winner_tasks.unsqueeze(1)
    for name, scores in routers.items():
        winner_score = scores.gather(1, winner_index)
        # 1-indexed rank of the winner: number of tasks scoring strictly higher.
        rank = (scores > winner_score).sum(dim=1) + 1
        diagnostics['router_{}_mean_rank'.format(name)] = rank.float()
        for k in (1, 2, 3, 4):
            diagnostics['router_{}_recall_{}'.format(name, k)] = rank.le(
                min(k, seen_task_count))
    return diagnostics


def progressive_oracle_audit(model, inputs, tii_logits, class_mask,
                             seen_task_count, args, targets=None,
                             true_task=None):
    """Run exhaustive once and measure the best possible 2->4->all cascade."""
    device = inputs.device
    batch_size = inputs.shape[0]
    temperature = max(
        1e-6, float(getattr(args, 'progressive_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'progressive_tii_prior_weight', 0.3)))
    excluded_margin = max(
        1.0, float(getattr(args, 'progressive_excluded_logit_margin', 20.0)))

    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    _, candidate_tasks = torch.sort(tii_prior, dim=1, descending=True)

    rank_logits = torch.full(
        (batch_size, seen_task_count, tii_logits.shape[1]),
        float('-inf'), dtype=tii_logits.dtype, device=device)
    task_evidence = torch.full(
        (batch_size, seen_task_count), float('-inf'),
        dtype=tii_logits.dtype, device=device)
    closure_tail_audit = bool(getattr(
        args, 'progressive_prediction_closure_tii_tail_audit', False))
    closure_audit = (
        bool(getattr(args, 'progressive_prediction_closure_audit', False))
        or closure_tail_audit)
    beam_closure_audit = bool(getattr(
        args, 'progressive_prediction_beam_closure_audit', False))
    budget_closure_audit = bool(getattr(
        args, 'progressive_prediction_budget_closure_audit', False))
    majority_closure_audit = bool(getattr(
        args, 'progressive_prediction_majority_closure_audit', False))
    consensus_closure_audit = bool(getattr(
        args, 'progressive_prediction_consensus_closure_audit', False))
    self_owner_audit = bool(getattr(
        args, 'progressive_prediction_self_owner_audit', False))
    corroborated_owner_audit = bool(getattr(
        args, 'progressive_prediction_corroborated_owner_audit', False))
    owner_mass_audit = bool(getattr(
        args, 'progressive_prediction_owner_mass_audit', False))
    owner_hedge_audit = bool(getattr(
        args, 'progressive_prediction_owner_consensus_hedge_audit', False))
    stage_drift_audit = bool(getattr(args, 'stage_drift_audit', False))
    router_recall_audit = bool(getattr(args, 'router_recall_audit', False))
    full_adapter_logits = (
        torch.empty_like(rank_logits)
        if (closure_audit or beam_closure_audit or budget_closure_audit
            or majority_closure_audit or consensus_closure_audit
            or self_owner_audit or corroborated_owner_audit
            or owner_mass_audit or owner_hedge_audit
            or stage_drift_audit)
        else None)
    initial_count = min(
        max(1, int(getattr(args, 'prediction_proposal_initial_count', 2))),
        seen_task_count)
    initial_adapter_logits = torch.empty(
        (batch_size, initial_count, tii_logits.shape[1]),
        dtype=tii_logits.dtype, device=device)

    for rank in range(seen_task_count):
        active_tasks = candidate_tasks[:, rank]
        adapter_logits = model(inputs, task_id=active_tasks)['logits']
        if full_adapter_logits is not None:
            full_adapter_logits[:, rank] = adapter_logits
        if rank < initial_count:
            initial_adapter_logits[:, rank] = adapter_logits
        for task_index in active_tasks.unique().tolist():
            rows = torch.nonzero(active_tasks == task_index).flatten()
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            local_logits = adapter_logits.index_select(
                0, rows).index_select(1, class_index) / temperature
            calibrated_logits = local_logits + prior_weight * tii_prior[
                rows, task_index].unsqueeze(1)
            rank_logits[
                rows.unsqueeze(1), rank, class_index.unsqueeze(0)
            ] = calibrated_logits
            task_evidence[rows, rank] = calibrated_logits.max(dim=1).values

    full_logits = rank_logits.max(dim=1).values
    selected_rank = task_evidence.argmax(dim=1)
    routed_tasks = candidate_tasks.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    full_predictions = full_logits.argmax(dim=1)

    boundaries = sorted(set([
        min(2, seen_task_count),
        min(4, seen_task_count),
        seen_task_count,
    ]))
    partial_results = {}
    oracle_counts = torch.full(
        (batch_size,), float(seen_task_count),
        dtype=torch.float32, device=device)
    unresolved = torch.ones(batch_size, dtype=torch.bool, device=device)

    for boundary in boundaries:
        partial_logits = rank_logits[:, :boundary].max(dim=1).values
        partial_logits = _finalize_partial_logits(
            partial_logits, excluded_margin)
        partial_rank = task_evidence[:, :boundary].argmax(dim=1)
        partial_tasks = candidate_tasks[:, :boundary].gather(
            1, partial_rank.unsqueeze(1)).squeeze(1)
        prediction_agreement = partial_logits.argmax(dim=1).eq(full_predictions)
        route_agreement = partial_tasks.eq(routed_tasks)
        exact_agreement = prediction_agreement & route_agreement
        partial_results[boundary] = {
            'prediction_agreement': prediction_agreement,
            'route_agreement': route_agreement,
            'exact_agreement': exact_agreement,
        }
        newly_resolved = unresolved & exact_agreement
        oracle_counts[newly_resolved] = float(boundary)
        unresolved[newly_resolved] = False

    winner_rank = selected_rank + 1
    boundary2 = min(2, seen_task_count)
    boundary4 = min(4, seen_task_count)
    diagnostics = {
        'winner_recall_2': winner_rank.le(boundary2),
        'winner_recall_4': winner_rank.le(boundary4),
        'exact_agreement_2': partial_results[boundary2]['exact_agreement'],
        'exact_agreement_4': partial_results[boundary4]['exact_agreement'],
        'prediction_agreement_2': partial_results[boundary2][
            'prediction_agreement'],
        'prediction_agreement_4': partial_results[boundary4][
            'prediction_agreement'],
        'oracle_lora_counts': oracle_counts,
        'actual_lora_counts': torch.full_like(
            oracle_counts, float(seen_task_count)),
    }
    if router_recall_audit:
        diagnostics.update(router_recall_diagnostics(
            tii_logits, class_mask, seen_task_count, routed_tasks))
    if stage_drift_audit:
        if targets is None or true_task is None:
            raise ValueError(
                'Stage-drift audit requires targets and the evaluated task id')
        diagnostics.update(stage_drift_diagnostics(
            candidate_tasks=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            targets=targets,
            true_task=true_task,
            class_mask=class_mask,
            seen_task_count=seen_task_count,
        ))
    if bool(getattr(args, 'progressive_prediction_proposal_audit', False)):
        diagnostics.update(prediction_proposal_diagnostics(
            candidate_tasks,
            initial_adapter_logits,
            rank_logits,
            task_evidence,
            routed_tasks,
            full_predictions,
            class_mask,
            initial_count=initial_count,
            proposal_count=getattr(args, 'prediction_proposal_count', 2),
            top_classes=getattr(args, 'prediction_proposal_top_classes', 5),
            excluded_margin=excluded_margin,
        ))
    if closure_audit:
        closure_diagnostics = prediction_closure_diagnostics(
            tii_ranking=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            rank_logits=rank_logits,
            task_evidence=task_evidence,
            winner_tasks=routed_tasks,
            full_predictions=full_predictions,
            class_mask=class_mask,
            initial_count=initial_count,
            top_classes=getattr(
                args, 'prediction_proposal_top_classes', 5),
            excluded_margin=excluded_margin,
            tii_logits=tii_logits,
            tii_tail_completion=closure_tail_audit,
        )
        diagnostics.update(closure_diagnostics)
    if beam_closure_audit:
        diagnostics.update(prediction_beam_closure_diagnostics(
            tii_ranking=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            rank_logits=rank_logits,
            task_evidence=task_evidence,
            winner_tasks=routed_tasks,
            full_predictions=full_predictions,
            tii_logits=tii_logits,
            class_mask=class_mask,
            initial_count=initial_count,
            top_classes=getattr(
                args, 'prediction_proposal_top_classes', 5),
            beam_width=2,
            excluded_margin=excluded_margin,
        ))
    if budget_closure_audit:
        diagnostics.update(prediction_budget_closure_diagnostics(
            tii_ranking=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            rank_logits=rank_logits,
            task_evidence=task_evidence,
            winner_tasks=routed_tasks,
            full_predictions=full_predictions,
            tii_logits=tii_logits,
            class_mask=class_mask,
            initial_count=initial_count,
            top_classes=getattr(
                args, 'prediction_proposal_top_classes', 5),
            max_candidates=5,
            excluded_margin=excluded_margin,
        ))
    if majority_closure_audit:
        diagnostics.update(prediction_majority_budget_closure_diagnostics(
            tii_ranking=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            rank_logits=rank_logits,
            task_evidence=task_evidence,
            winner_tasks=routed_tasks,
            full_predictions=full_predictions,
            tii_logits=tii_logits,
            class_mask=class_mask,
            initial_count=initial_count,
            top_classes=getattr(
                args, 'prediction_proposal_top_classes', 5),
            max_candidates=5,
            excluded_margin=excluded_margin,
        ))
    if consensus_closure_audit:
        diagnostics.update(prediction_consensus_budget_closure_diagnostics(
            tii_ranking=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            rank_logits=rank_logits,
            task_evidence=task_evidence,
            winner_tasks=routed_tasks,
            full_predictions=full_predictions,
            tii_logits=tii_logits,
            class_mask=class_mask,
            initial_count=initial_count,
            top_classes=getattr(
                args, 'prediction_proposal_top_classes', 5),
            max_candidates=5,
            excluded_margin=excluded_margin,
        ))
    if self_owner_audit:
        diagnostics.update(prediction_self_owner_consensus_diagnostics(
            tii_ranking=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            rank_logits=rank_logits,
            task_evidence=task_evidence,
            winner_tasks=routed_tasks,
            full_predictions=full_predictions,
            tii_logits=tii_logits,
            class_mask=class_mask,
            initial_count=initial_count,
            top_classes=getattr(
                args, 'prediction_proposal_top_classes', 5),
            max_candidates=5,
            excluded_margin=excluded_margin,
        ))
    if corroborated_owner_audit:
        diagnostics.update(
            prediction_corroborated_self_owner_diagnostics(
                tii_ranking=candidate_tasks,
                full_adapter_logits=full_adapter_logits,
                rank_logits=rank_logits,
                task_evidence=task_evidence,
                winner_tasks=routed_tasks,
                full_predictions=full_predictions,
                tii_logits=tii_logits,
                class_mask=class_mask,
                initial_count=initial_count,
                top_classes=getattr(
                    args, 'prediction_proposal_top_classes', 5),
                max_candidates=5,
                excluded_margin=excluded_margin,
            ))
    if owner_mass_audit:
        diagnostics.update(prediction_owner_mass_consensus_diagnostics(
            tii_ranking=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            rank_logits=rank_logits,
            task_evidence=task_evidence,
            winner_tasks=routed_tasks,
            full_predictions=full_predictions,
            tii_logits=tii_logits,
            class_mask=class_mask,
            initial_count=initial_count,
            top_classes=getattr(
                args, 'prediction_proposal_top_classes', 5),
            max_candidates=5,
            excluded_margin=excluded_margin,
        ))
    if owner_hedge_audit:
        diagnostics.update(prediction_owner_consensus_hedge_diagnostics(
            tii_ranking=candidate_tasks,
            full_adapter_logits=full_adapter_logits,
            rank_logits=rank_logits,
            task_evidence=task_evidence,
            winner_tasks=routed_tasks,
            full_predictions=full_predictions,
            tii_logits=tii_logits,
            class_mask=class_mask,
            initial_count=initial_count,
            top_classes=getattr(
                args, 'prediction_proposal_top_classes', 5),
            max_candidates=5,
            excluded_margin=excluded_margin,
        ))
    if bool(getattr(args, 'progressive_arrow_audit', False)):
        arrow_scores = arrow_task_scores(model, inputs, seen_task_count)
        arrow_ranking = torch.argsort(arrow_scores, dim=1, descending=True)
        diagnostics.update(arrow_candidate_diagnostics(
            candidate_tasks, arrow_ranking, routed_tasks))
    if bool(getattr(args, 'progressive_lora_response_audit', False)):
        response_scores = lora_response_task_scores(
            model, inputs, seen_task_count)
        response_ranking = torch.argsort(response_scores, dim=1, descending=True)
        diagnostics.update(lora_response_candidate_diagnostics(
            candidate_tasks, response_ranking, routed_tasks))
    if closure_tail_audit:
        return (
            diagnostics['prediction_closure_output_logits'],
            diagnostics['prediction_closure_output_tasks'],
            diagnostics,
        )
    if beam_closure_audit:
        return (
            diagnostics['prediction_beam_closure_output_logits'],
            diagnostics['prediction_beam_closure_output_tasks'],
            diagnostics,
        )
    if budget_closure_audit:
        return (
            diagnostics['prediction_budget_closure_output_logits'],
            diagnostics['prediction_budget_closure_output_tasks'],
            diagnostics,
        )
    if majority_closure_audit:
        return (
            diagnostics['prediction_majority_closure_output_logits'],
            diagnostics['prediction_majority_closure_output_tasks'],
            diagnostics,
        )
    if owner_hedge_audit:
        return (
            diagnostics['prediction_owner_hedge_output_logits'],
            diagnostics['prediction_owner_hedge_output_tasks'],
            diagnostics,
        )
    if owner_mass_audit:
        return (
            diagnostics['prediction_owner_mass_output_logits'],
            diagnostics['prediction_owner_mass_output_tasks'],
            diagnostics,
        )
    if corroborated_owner_audit:
        return (
            diagnostics['prediction_corroborated_owner_output_logits'],
            diagnostics['prediction_corroborated_owner_output_tasks'],
            diagnostics,
        )
    if self_owner_audit:
        return (
            diagnostics['prediction_self_owner_output_logits'],
            diagnostics['prediction_self_owner_output_tasks'],
            diagnostics,
        )
    if consensus_closure_audit:
        return (
            diagnostics['prediction_consensus_closure_output_logits'],
            diagnostics['prediction_consensus_closure_output_tasks'],
            diagnostics,
        )
    return full_logits, routed_tasks, diagnostics
