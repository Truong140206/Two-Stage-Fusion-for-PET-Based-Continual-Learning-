import copy
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import torchvision.transforms as transforms
import torch
import torch.distributed as dist
import numpy as np
from torch.nn import functional as F
from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from engines.exhaustive_rematching import exhaustive_adapter_rematching
from engines.vectorized_exhaustive_rematching import vectorized_exhaustive_adapter_rematching
from engines.soft_mixture_rematching import (
    soft_mixture_adapter_rematching,
    soft_hard_confidence_selector_rematching,
    soft_mixture_hard_adapter_rematching,
    soft_mixture_local_hard_refinement,
)
from engines.hierarchical_rematching import hierarchical_adapter_rematching
from engines.budgeted_rematching import budgeted_exhaustive_fallback
from engines.progressive_rematching import progressive_adapter_rematching
from engines.progressive_oracle_audit import progressive_oracle_audit
from engines.random_projection_head import (
    accumulate_rp_statistics, fit_rp_temperature, reset_rp_head,
    rp_head_predict, solve_rp_head)
from engines.layer_stat_router import (
    accumulate_task_stats, install_hooks, layer_stat_scores, remove_hooks,
    reset_layer_stats, task_stats_ready)
from engines.prediction_proposal_rematching import (
    cross_adapter_borda_consensus,
    cross_adapter_global_consensus,
    initial_branch_confidence_dominance,
    prediction_proposal_adapter_rematching,
)
from engines.prediction_closure_rematching import (
    prediction_closure_tii_tail_rematching,
)
from engines.calibrated_progressive_rematching import (
    calibrated_progressive_rematching,
    get_progressive_halting_gates,
)
from engines.selective_rematching import selective_adapter_rematching
from engines.prototype_rematching import (
    build_prototype_bank, prototype_assisted_rematching)
from engines.shared_prototype_router import (
    build_shared_prototype_bank, shared_space_prototype_routing)
from engines.replay_anchored_ctird import (
    build_replay_anchor_memory,
    generate_task_replay_cache,
    replay_anchor_relation_loss,
)
from engines.cfs_task_logit_calibration import (
    fit_cfs_task_logit_calibration,
)
from torch.distributions.multivariate_normal import MultivariateNormal



def generalized_entropy(softmax_id_val, gamma, M):
        probs =  softmax_id_val 
        probs_sorted = np.sort(probs, axis=1)[:,-M:]
        scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
           
        return -scores 

def compute_task_energy_scores(logits, class_mask, num_tasks, temperature=0.1):
    temperature = max(float(temperature), 1e-6)
    task_scores = []
    for task_idx in range(num_tasks):
        class_ids = torch.tensor(class_mask[task_idx], dtype=torch.long, device=logits.device)
        task_logits = logits.index_select(1, class_ids)
        task_scores.append(temperature * torch.logsumexp(task_logits / temperature, dim=1))
    return torch.stack(task_scores, dim=1)


def compute_relation_matrix(features):
    normalized = F.normalize(features, p=2, dim=1)
    similarities = torch.mm(normalized, normalized.t())
    return F.softmax(similarities, dim=1)


def compute_ctird_semantic_task_scores(targets, class_mask, num_old_tasks,
                                       args, device):
    """Score old tasks by text similarity to each current class label."""
    if targets is None:
        return None
    class_names = getattr(args, 'class_names', None)
    if not class_names:
        return None

    class_names = utils.resolve_semantic_class_names(class_names, args)
    semantic_dim = max(1, int(getattr(args, 'semantic_dim', 512)))
    embeddings = utils.build_semantic_class_embeddings(
        class_names, device, dim=semantic_dim, args=args)
    if embeddings is None:
        return None

    targets = targets.long().to(device)
    if targets.numel() == 0 or targets.min().item() < 0:
        return None
    if targets.max().item() >= embeddings.shape[0]:
        return None

    target_embeddings = embeddings.index_select(0, targets)
    task_scores = []
    for task_idx in range(num_old_tasks):
        class_ids = torch.as_tensor(
            class_mask[task_idx], dtype=torch.long, device=device)
        class_embeddings = embeddings.index_select(0, class_ids)
        similarities = torch.mm(target_embeddings, class_embeddings.t())
        keep = min(3, similarities.shape[1])
        task_scores.append(
            torch.topk(similarities, k=keep, dim=1).values.mean(dim=1))
    return torch.stack(task_scores, dim=1)


def fuse_ctird_task_scores(task_scores, semantic_scores,
                           max_semantic_weight=0.1,
                           confidence_margin=0.15,
                           semantic_temperature=0.1,
                           return_weight=False):
    """Use semantics only when the TII old-task decision is ambiguous."""
    task_probabilities = F.softmax(task_scores.float(), dim=1)
    if task_probabilities.shape[1] < 2:
        semantic_weight = task_probabilities.new_zeros(
            task_probabilities.shape[0])
        if return_weight:
            return task_probabilities, semantic_weight
        return task_probabilities

    semantic_temperature = max(float(semantic_temperature), 1e-6)
    semantic_probabilities = F.softmax(
        semantic_scores.float() / semantic_temperature, dim=1)
    top_two = torch.topk(
        task_probabilities, k=2, dim=1, largest=True).values
    observed_margin = top_two[:, 0] - top_two[:, 1]
    confidence_margin = max(float(confidence_margin), 1e-6)
    uncertainty = (
        (confidence_margin - observed_margin) / confidence_margin
    ).clamp(0.0, 1.0)
    max_semantic_weight = min(
        1.0, max(0.0, float(max_semantic_weight)))
    semantic_weight = max_semantic_weight * uncertainty
    fused = (
        (1.0 - semantic_weight.unsqueeze(1)) * task_probabilities
        + semantic_weight.unsqueeze(1) * semantic_probabilities
    )
    if return_weight:
        return fused, semantic_weight
    return fused


def select_ctird_source_tasks(logits, class_mask, num_old_tasks, top_k,
                              temperature=1.0, targets=None, args=None,
                              return_diagnostics=False):
    """Select unique old tasks using TII and gated semantic evidence."""
    count = min(int(num_old_tasks), int(top_k))
    if count <= 0:
        return None
    task_scores = compute_task_energy_scores(
        logits, class_mask, num_old_tasks, temperature=temperature)
    base_selection = torch.topk(
        task_scores, k=count, dim=1, largest=True).indices
    semantic_weight = task_scores.new_zeros(task_scores.shape[0])
    if bool(getattr(args, 'ctird_semantic_selection', False)):
        semantic_scores = compute_ctird_semantic_task_scores(
            targets, class_mask, num_old_tasks, args, logits.device)
        if semantic_scores is not None:
            task_scores, semantic_weight = fuse_ctird_task_scores(
                task_scores,
                semantic_scores,
                max_semantic_weight=getattr(args, 'ctird_semantic_weight', 0.1),
                confidence_margin=getattr(args, 'ctird_semantic_margin', 0.15),
                semantic_temperature=getattr(
                    args, 'ctird_semantic_temperature', 0.1),
                return_weight=True,
            )
    selection = torch.topk(
        task_scores, k=count, dim=1, largest=True).indices
    if return_diagnostics:
        changed = (selection != base_selection).any(dim=1).float()
        diagnostics = {
            'semantic_weight': float(semantic_weight.mean().item()),
            'changed_rate': float(changed.mean().item()),
        }
        return selection, diagnostics
    return selection


def online_ctird_rank_weight(num_selected_tasks, evaluated_ranks,
                             reduction='sum'):
    evaluated_ranks = max(1, int(evaluated_ranks))
    reduction = str(reduction).lower()
    if reduction == 'mean':
        return 1.0 / float(evaluated_ranks)
    if reduction == 'sum':
        return float(num_selected_tasks) / float(evaluated_ranks)
    raise ValueError('Unknown online CTIRD reduction: {}'.format(reduction))

def compute_ctird_rank_weights(top_values, args):
    temperature = max(float(getattr(args, 'ctird_weight_temperature', 1.0)), 1e-6)
    weights = F.softmax(top_values.float() / temperature, dim=1)
    floor = float(getattr(args, 'ctird_weight_floor', 0.2))
    floor = max(0.0, min(1.0, floor))
    uniform = torch.full_like(weights, 1.0 / weights.shape[1])
    weights = (1.0 - floor) * weights + floor * uniform
    return weights.mean(dim=0) * weights.shape[1]

def get_old_features(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.eval()
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    all_res = []
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        temp = old_logits.index_fill(dim=1, index=torch.tensor(class_mask[task_id], dtype=torch.int64).to(device), value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

        # here is the trick to mask out classes of non-current tasks
        top_indices=None
        ctird_rank_weights = None
        
        if task_id>0:

            probabilities = temp[:,:task_id*len(class_mask[0])]
            m = min(task_id, args.K)
            ctird_selection = str(getattr(args, 'ctird_task_selection', 'legacy')).lower()
            if ctird_selection == 'task_energy':
                task_scores = compute_task_energy_scores(
                    temp, class_mask, task_id,
                    temperature=getattr(args, 'ctird_task_temperature', 0.1))
                top_values, top5_id = torch.topk(task_scores, k=m, dim=1, largest=True)
                if str(getattr(args, 'ctird_task_weighting', 'uniform')).lower() == 'energy':
                    ctird_rank_weights = compute_ctird_rank_weights(top_values, args)
            else:
                _, top_indices = torch.topk(probabilities, k=m, dim=1, largest=False)
                top5_id = []
                for i in range(top_indices.shape[0]):
                    top5_id.append(torch.tensor([target_task_map[v.item()] for v in top_indices[i]]))
                top5_id = torch.stack(top5_id, dim=0).to(device, non_blocking=True)
        
        if task_id>0:
            # robust_logits = robust_loss(model, input, output['features'], target,device,task_id,class_mask,top5_id)
            all_old_logits = []
            for k in range(top5_id.shape[1]):
                prompt_id = top5_id[:,k]
                with torch.no_grad():
                    output = model(input, task_id=prompt_id)
                    old_logits = output['features']
                    old_norm_features = F.normalize(output['features'], p=2, dim=1)
                    old_similarity_matrix = torch.mm(old_norm_features, old_norm_features.t())
                    old_similarity_matrix = torch.exp(old_similarity_matrix)
                    old_similarity_matrix = old_similarity_matrix / old_similarity_matrix.sum(1, keepdim=True)
                    all_old_logits.append(old_similarity_matrix)

            if ctird_rank_weights is None:
                all_res.append(all_old_logits)
            else:
                all_res.append((all_old_logits, ctird_rank_weights.detach()))
                
    return all_res



def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None,
                    args=None, old_features=None, replay_anchor_memory=None):
    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    online_aligned_ctird = (
        task_id > 0 and bool(getattr(args, 'ctird_online_aligned', False)))
    replay_anchor_enabled = (
        task_id > 0
        and bool(getattr(args, 'replay_anchor_ctird', False))
        and replay_anchor_memory is not None
        and not replay_anchor_memory.empty
    )
    if replay_anchor_enabled:
        metric_logger.add_meter('ReplayCT', utils.SmoothedValue(window_size=20, fmt='{avg:.4f}'))
        metric_logger.add_meter('ReplayKeep', utils.SmoothedValue(window_size=20, fmt='{avg:.1f}'))
        metric_logger.add_meter('ReplayConf', utils.SmoothedValue(window_size=20, fmt='{avg:.3f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    global_index = 0
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        temp = old_logits.index_fill(dim=1, index=torch.tensor(class_mask[task_id], dtype=torch.int64).to(device), value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

            
        output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']
        # here is the trick to mask out classes of non-current tasks
        top_indices=None
        ctird_rank_weights = None
        
        if task_id>0:
            m = min(task_id, args.K)
            if online_aligned_ctird:
                semantic_diagnostics = bool(
                    getattr(args, 'ctird_semantic_selection', False))
                selection_result = select_ctird_source_tasks(
                    temp, class_mask, task_id, m,
                    temperature=getattr(args, 'ctird_online_temperature', 1.0),
                    targets=target, args=args,
                    return_diagnostics=semantic_diagnostics)
                if semantic_diagnostics:
                    top5_id, semantic_stats = selection_result
                    metric_logger.update(
                        SemWeight=semantic_stats['semantic_weight'],
                        SemChange=semantic_stats['changed_rate'])
                else:
                    top5_id = selection_result
            else:
                probabilities = temp[:,:task_id*len(class_mask[0])]
                ctird_selection = str(getattr(args, 'ctird_task_selection', 'legacy')).lower()
                if ctird_selection == 'task_energy':
                    task_scores = compute_task_energy_scores(
                        temp, class_mask, task_id,
                        temperature=getattr(args, 'ctird_task_temperature', 0.1))
                    top_values, top5_id = torch.topk(task_scores, k=m, dim=1, largest=True)
                    if str(getattr(args, 'ctird_task_weighting', 'uniform')).lower() == 'energy':
                        ctird_rank_weights = compute_ctird_rank_weights(top_values, args)
                else:
                    _, top_indices = torch.topk(probabilities, k=m, dim=1, largest=False)
                    top5_id = []
                    for i in range(top_indices.shape[0]):
                        top5_id.append(torch.tensor([target_task_map[v.item()] for v in top_indices[i]]))
                    top5_id = torch.stack(top5_id, dim=0).to(device, non_blocking=True)
        
        if task_id>0:
            if online_aligned_ctird:
                rank_count = max(
                    1, min(m, int(getattr(args, 'ctird_online_ranks_per_batch', 1))))
                start_rank = (epoch + global_index) % m
                selected_ranks = [
                    (start_rank + offset) % m for offset in range(rank_count)]
                robust_logits = []
                with torch.no_grad():
                    for rank in selected_ranks:
                        prompt_id = top5_id[:, rank]
                        old_output = model(input, task_id=prompt_id)
                        robust_logits.append(
                            compute_relation_matrix(old_output['features']))
                reduction = getattr(args, 'ctird_online_reduction', 'sum')
                rank_weight = online_ctird_rank_weight(
                    m, rank_count, reduction=reduction)
                ctird_rank_weights = output['features'].new_full(
                    (rank_count,), rank_weight)
            else:
                #robust_logits = robust_loss(model, input, output['features'], target,device,task_id,class_mask,top5_id)
                robust_bundle = old_features[global_index]
                ctird_rank_weights = None
                if isinstance(robust_bundle, tuple):
                    robust_logits, ctird_rank_weights = robust_bundle
                else:
                    robust_logits = robust_bundle
        
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
 
        
        loss_ctird = 0
        if task_id>0:

            loss = criterion(logits, target)
            for k in range(len(robust_logits)):
                norm_features = F.normalize(output['features'], p=2, dim=1)
                
                # 计算相似度矩阵
                similarity_matrix = torch.mm(norm_features, norm_features.t())
                similarity_matrix = torch.exp(similarity_matrix)
                similarity_matrix = similarity_matrix / similarity_matrix.sum(1, keepdim=True)
                pos_mask = (similarity_matrix > 0.038).float()
                relation_target = utils.apply_semantic_relation_distillation(robust_logits[k], target, args, device)
                loss_ctird = F.kl_div(torch.log(similarity_matrix.clamp_min(1e-12)), relation_target, reduction='batchmean')
                
                ctird_weight = 1.0 if ctird_rank_weights is None else ctird_rank_weights[k]
                loss = loss + args.con * ctird_weight * loss_ctird
        else:
            loss = criterion(logits, target)+args.con*loss_ctird
            
        replay_ctird_loss = logits.new_zeros(())
        replay_kept = 0
        replay_confidence = 0.0
        if replay_anchor_enabled:
            replay_ctird_loss, replay_kept, replay_confidence = replay_anchor_relation_loss(
                model=model,
                memory=replay_anchor_memory,
                current_task_id=task_id,
                seen_classes=replay_anchor_memory.class_ids,
                args=args,
                device=device,
            )
            replay_weight = max(
                0.0, float(getattr(args, 'replay_anchor_weight', 0.05)))
            loss = loss + replay_weight * replay_ctird_loss

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        if replay_anchor_enabled:
            metric_logger.update(ReplayCT=replay_ctird_loss.item())
            metric_logger.update(ReplayKeep=float(replay_kept))
            metric_logger.update(ReplayConf=float(replay_confidence))
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        global_index = global_index+1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

con_num=0
incon_num=0
con_all=0
incon_all=0
max_p = []
cls_mean = {}
cls_cov = {}
cls_cfs_model = {}
cls_real_features = {}
cls_shared_features = {}
replay_task_router = None
_gaussian_stat_cache = {}
_rp_frozen_extractor = {'model': None}


def reset_replay_statistics():
    global cls_mean, cls_cov, cls_cfs_model, cls_real_features, cls_shared_features, replay_task_router
    cls_mean = {}
    cls_cov = {}
    cls_cfs_model = {}
    cls_real_features = {}
    cls_shared_features = {}
    replay_task_router = None
    _gaussian_stat_cache.clear()


def _gaussian_class_stats(class_id, device, args):
    """Return (mean, precision, logdet, is_diag) for a class Gaussian.

    Precision is 1/variance vector when is_diag, else the inverse covariance
    matrix. Covariance is shrunk toward its mean-diagonal by gaussian_shrinkage
    for robustness with few samples per class. Cached per (class, mode, shrink).
    """
    mode = getattr(args, 'gaussian_cov_mode', 'diagonal')
    shrink = float(getattr(args, 'gaussian_shrinkage', 0.1))
    key = (int(class_id), mode, round(shrink, 6))
    cached = _gaussian_stat_cache.get(key)
    if cached is not None:
        return cached
    mean = torch.as_tensor(
        cls_mean[int(class_id)], device=device, dtype=torch.float32)
    cov = torch.as_tensor(
        cls_cov[int(class_id)], device=device, dtype=torch.float32)
    dim = mean.shape[0]
    if cov.dim() == 1:
        cov = torch.diag(cov)
    diag = torch.diagonal(cov)
    mean_diag = diag.mean().clamp_min(1e-6)
    if mode == 'diagonal':
        var = (1.0 - shrink) * diag + shrink * mean_diag
        var = var.clamp_min(1e-6)
        stats = (mean, 1.0 / var, torch.log(var).sum(), True)
    else:
        cov_s = (1.0 - shrink) * cov + shrink * torch.diag(
            torch.full_like(diag, float(mean_diag)))
        cov_s = cov_s + torch.eye(dim, device=device) * 1e-5
        chol = torch.linalg.cholesky(cov_s)
        precision = torch.cholesky_inverse(chol)
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        stats = (mean, precision, logdet, False)
    _gaussian_stat_cache[key] = stats
    return stats


def _tied_precision(seen_class_ids, device, args):
    """Shared inverse covariance (mean over seen classes) + logdet for tied mode."""
    shrink = float(getattr(args, 'gaussian_shrinkage', 0.1))
    key = ('__tied__', len(seen_class_ids), round(shrink, 6))
    cached = _gaussian_stat_cache.get(key)
    if cached is not None:
        return cached
    accum = None
    count = 0
    for class_id in seen_class_ids:
        if int(class_id) not in cls_cov:
            continue
        cov = torch.as_tensor(
            cls_cov[int(class_id)], device=device, dtype=torch.float32)
        if cov.dim() == 1:
            cov = torch.diag(cov)
        accum = cov if accum is None else accum + cov
        count += 1
    cov_mean = accum / max(1, count)
    diag = torch.diagonal(cov_mean)
    mean_diag = diag.mean().clamp_min(1e-6)
    cov_s = (1.0 - shrink) * cov_mean + shrink * torch.diag(
        torch.full_like(diag, float(mean_diag)))
    cov_s = cov_s + torch.eye(cov_s.shape[0], device=device) * 1e-5
    chol = torch.linalg.cholesky(cov_s)
    precision = torch.cholesky_inverse(chol)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    stats = (precision, logdet)
    _gaussian_stat_cache[key] = stats
    return stats


@torch.no_grad()
def gaussian_rescoring_predict(model, inputs, class_mask, seen_task_count,
                               args, device):
    """Class-incremental prediction by per-class Gaussian fit.

    For each seen task the query is passed through that task LoRA (the same
    feature space the class Gaussians were estimated in), then every class of
    that task is scored by -0.5 * Mahalanobis^2 (optionally + log-likelihood
    normalizer and a blended LoRA classifier logit). A wrong-task feature is
    out-of-distribution for that task Gaussians, so its Mahalanobis distance is
    large and the cross-task hijack that breaks energy re-matching is
    suppressed. Returns (scores[B, nb_classes], prompt_id[B]).
    """
    batch = inputs.shape[0]
    nb_classes = args.nb_classes
    include_logdet = bool(getattr(args, 'gaussian_include_logdet', False))
    g_weight = float(getattr(args, 'gaussian_score_weight', 1.0))
    logit_weight = float(getattr(args, 'gaussian_logit_weight', 0.0))
    logit_temp = max(1e-6, float(getattr(args, 'gaussian_logit_temperature', 1.0)))
    mode = getattr(args, 'gaussian_cov_mode', 'diagonal')
    scores = torch.full(
        (batch, nb_classes), float('-inf'), device=device, dtype=torch.float32)
    class_task = torch.full(
        (nb_classes,), -1, dtype=torch.long, device=device)
    tied_precision = None
    tied_logdet = None
    if mode == 'tied':
        seen_class_ids = [
            int(c) for t in range(seen_task_count) for c in class_mask[t]]
        tied_precision, tied_logdet = _tied_precision(
            seen_class_ids, device, args)
    for task in range(seen_task_count):
        output = model(inputs, task_id=task, train=True)
        feature = output['pre_logits'].float()
        logits_task = output['logits'].float() if logit_weight != 0.0 else None
        for class_id in class_mask[task]:
            class_id = int(class_id)
            if class_id not in cls_mean:
                continue
            if mode == 'tied':
                mean = torch.as_tensor(
                    cls_mean[class_id], device=device, dtype=torch.float32)
                precision, logdet, is_diag = tied_precision, tied_logdet, False
            else:
                mean, precision, logdet, is_diag = _gaussian_class_stats(
                    class_id, device, args)
            diff = feature - mean
            if is_diag:
                maha = (diff * diff * precision).sum(dim=1)
            else:
                maha = (diff @ precision * diff).sum(dim=1)
            class_score = -0.5 * maha
            if include_logdet:
                class_score = class_score - 0.5 * logdet
            class_score = g_weight * class_score
            if logits_task is not None:
                class_score = class_score + logit_weight * (
                    logits_task[:, class_id] / logit_temp)
            scores[:, class_id] = class_score
            class_task[class_id] = task
    prompt_id = class_task[scores.argmax(dim=1)]
    return scores, prompt_id


def restore_real_feature_memory(feature_memory):
    global cls_real_features
    cls_real_features = {
        int(class_id): features.detach().cpu().half()
        for class_id, features in feature_memory.items()
    }


def get_real_feature_memory():
    return cls_real_features


def get_shared_feature_memory():
    return cls_shared_features


def set_replay_task_router(router):
    global replay_task_router
    replay_task_router = router


def _rp_inputs(inputs, args):
    """Match the pretrained ViT's expected input range for the frozen head.

    build_transform() emits ToTensor() output in [0, 1] with no Normalize, but
    the ImageNet-21K ViT-B/16 checkpoint was trained on [-1, 1] (mean=std=0.5).
    HRM-PET is unaffected because it trains its LoRA and classifier on whatever
    features it gets, but a training-free head inherits the mismatch directly.
    Only the RP head's extraction path is rescaled; the LoRA path keeps the
    range its adapters were trained on.
    """
    mode = str(getattr(args, 'rp_input_norm', 'none'))
    if mode == 'half':
        return (inputs - 0.5) / 0.5
    if mode == 'cifar_half':
        # build_cifar_transform() already normalizes with CIFAR statistics, so
        # undo that first; applying the [-1, 1] rescale on top of it would be a
        # double normalization.
        mean = torch.tensor([0.5071, 0.4867, 0.4408], device=inputs.device)
        std = torch.tensor([0.2675, 0.2565, 0.2761], device=inputs.device)
        raw = inputs * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        return (raw - 0.5) / 0.5
    return inputs


def pin_rp_extractor(original_model, args):
    """Snapshot a fixed feature extractor for the RP head.

    Default: the task-1 TII model. Evaluation reloads the TII checkpoint for
    every task, so accumulating the Gram matrix straight from `original_model`
    would mix ten different extractors; pinning the first-task snapshot keeps
    the feature space constant.

    With --rp_bare_extractor the snapshot is a freshly created *pretrained*
    backbone that has never seen the task sequence. This is the only way to
    measure RanPAC's "no first-session adaptation" condition here, because
    `original_model` is NOT the raw pretrained ViT: its weights are overwritten
    wholesale from the TII checkpoint (trainers/lora_trainer.py:171). So a run
    with rp_feature_source=original still reads features off an *adapted*
    backbone, which is why our 56.36 on ImageNet-R was never comparable to
    RanPAC's 71.8 for that ablation.
    """
    if not bool(getattr(args, 'rp_pin_extractor', False)):
        return None
    if _rp_frozen_extractor['model'] is None:
        if bool(getattr(args, 'rp_bare_extractor', False)):
            from timm.models import create_model
            # --rp_bare_model overrides the checkpoint for THIS snapshot
            # only. Everything else keeps args.original_model, which is
            # what the LoRA and TII checkpoints were trained against.
            bare_name = (str(getattr(args, 'rp_bare_model', '') or '')
                         .strip() or args.original_model)
            if bare_name != args.original_model:
                print('RP head: bare snapshot uses', bare_name,
                      'instead of', args.original_model)
            snapshot = create_model(
                bare_name,
                pretrained=True,
                num_classes=args.nb_classes,
                mlp_structure=args.original_model_mlp_structure,
            )
            snapshot.to(next(original_model.parameters()).device)
            print('RP head: BARE pretrained extractor -- the TII checkpoint is '
                  'deliberately NOT loaded into this snapshot.')
        else:
            snapshot = copy.deepcopy(original_model)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad = False
        _rp_frozen_extractor['model'] = snapshot
    return _rp_frozen_extractor['model']


def rp_extractor(original_model, args):
    """Pinned extractor when enabled, otherwise the current TII model."""
    pinned = _rp_frozen_extractor['model']
    if bool(getattr(args, 'rp_pin_extractor', False)) and pinned is not None:
        return pinned
    return original_model


def _task_scores_from_class_scores(scores, class_mask, seen_task_count, device):
    """Reduce per-class scores to one score per task (max over its classes)."""
    out = torch.full((scores.shape[0], seen_task_count), float('-inf'),
                     device=device, dtype=torch.float32)
    for task in range(seen_task_count):
        index = torch.as_tensor(
            [int(c) for c in class_mask[task]], dtype=torch.long, device=device)
        out[:, task] = scores.index_select(1, index).max(dim=1).values
    return out


def _stage_ramp(seen_tasks, total_tasks, gamma):
    """How much of the second opinion this stage has earned, in (0, 1].

    Both fusions currently lend the same weight at every stage, and that is why
    Forgetting and Backward barely move: they compare a task's final accuracy
    against its own earlier peak, so a correction applied equally at every stage
    cancels out of both. Constant weight is also wrong on its own terms. The RP
    head's Gram estimate needs classes before it is trustworthy -- while TII has
    more tasks to tell apart as they accumulate. Both say the second opinion
    should be trusted more as the sequence grows.

    NOTE 2026-08-28: the figure this docstring used to quote (-1.38 and -0.95 at
    stages 1-2 on ImageNet-R) was a single seed and does NOT generalise. Twelve
    (dataset, seed) cells now say the harm is confined to stage 1; stage 2 is
    already net positive, and withholding the blend there costs up to -1.5 on
    CUB-200. See rp_class_fusion_min_tasks below.

    gamma = 0 reproduces the constant weight exactly, and the ramp reaches 1.0
    at the final stage, so the reported final row is untouched either way.

    Which fusion to ramp is not a free choice, and ramping both was wrong.
    Per-stage accuracy on seed 42 shows the RP head hurting the *classifier* at
    stage 1 (ImageNet-R 86.0 against the baseline's 87.5) exactly as the Gram
    argument predicts, so ramping the class blend recovers real accuracy. But
    the same head is the *better router* from the first stage (77.1 against
    TII's 63.8), so ramping the route weight only threw routing away: average
    incremental Acc@task fell on all three datasets (86.30 -> 85.66 on
    ImageNet-R at gamma=2). Hence the scope flag, and hence 'class' rather than
    'both'.
    """
    if gamma <= 0.0:
        return 1.0
    total = max(1, int(total_tasks))
    seen = min(max(1, int(seen_tasks)), total)
    return (float(seen) / float(total)) ** float(gamma)


def fuse_routers(rp_scores, tii_logits, class_mask, seen_task_count, args,
                 device, layer_scores=None):
    """Route by combining the RP head and TII, which fail on different samples.

    Measured on seed 42: the RP head routes better than raw TII everywhere
    (ImageNet-R 77.1 vs 63.8, CIFAR 89.5 vs 81.2, CUB 94.0 vs 93.0), and the
    union of the two exceeds HRM-PET's own post-DRM/CRM routing on all three
    (83.0 vs 77.8, 92.9 vs 89.8, 96.3 vs 93.2). Both are mapped to
    log-probabilities over tasks before mixing, since their raw scales differ.
    """
    weight = float(getattr(args, 'rp_route_fusion_weight', 0.5))
    # Weight sits on TII, so the RP share is 1 - weight; ramp that share.
    if getattr(args, 'rp_fusion_ramp_scope', 'both') in ('both', 'route'):
        weight = 1.0 - (1.0 - weight) * _stage_ramp(
            seen_task_count, getattr(args, 'num_tasks', seen_task_count),
            getattr(args, 'rp_fusion_ramp', 0.0))
    rp_task = _task_scores_from_class_scores(
        torch.nan_to_num(rp_scores.float(), neginf=-1e4),
        class_mask, seen_task_count, device)
    tii_task = _task_scores_from_class_scores(
        torch.nan_to_num(tii_logits.float(), neginf=-1e4),
        class_mask, seen_task_count, device)

    def standardize(scores):
        # log_softmax preserves the spread of its input, and TII logits are on a
        # far wider scale than ridge scores, so mixing them directly lets TII
        # dominate at any weight. Standardizing per sample first makes the
        # weight mean what it says.
        # unbiased=False on purpose. At stage 1 there is exactly one task, so
        # the Bessel-corrected std divides by n-1 = 0 and yields NaN, which
        # clamp_min cannot repair -- NaN compares false against everything. The
        # NaN was harmless only because argmax over a single column returns 0
        # whatever it holds, i.e. the answer was right by luck. The population
        # std gives 0 there, which clamp_min does repair. For every stage past
        # the first the two differ by sqrt(n/(n-1)), a constant shared by both
        # routers, so it cancels in the argmax and no measured number moves.
        centered = scores - scores.mean(dim=1, keepdim=True)
        return centered / centered.std(
            dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)

    fused = (weight * standardize(tii_task)
             + (1.0 - weight) * standardize(rp_task))
    layer_weight = float(getattr(args, 'rp_route_fusion_ls_weight', 0.0))
    if layer_scores is not None and layer_weight != 0.0:
        # Weak on its own (48.5 on ImageNet-R) but its errors barely overlap the
        # other two -- it alone rescues 2.8% of the samples both miss, lifting
        # the three-router union to 85.8 from 83.0. Keep the weight small.
        fused = fused + layer_weight * standardize(layer_scores.float())
    return fused.argmax(dim=1)


args_ref = [None]


def _valid_moments(scores, valid):
    """Per-sample mean and std over the valid classes only."""
    masked = torch.where(valid, scores, torch.zeros_like(scores))
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).float()
    mean = masked.sum(dim=1, keepdim=True) / count
    centered = torch.where(valid, scores - mean, torch.zeros_like(scores))
    std = (centered.pow(2).sum(dim=1, keepdim=True) / count).sqrt()
    return mean, std.clamp_min(1e-6)


def _standardize_valid(scores, valid):
    """Zero-mean unit-variance over a sample's valid classes only."""
    mean, std = _valid_moments(scores, valid)
    return torch.where(valid, scores - mean, torch.zeros_like(scores)) / std


def _top2_margin(scores, valid):
    """Normalised gap between a sample's top two classes, in [0, 1].

    Dividing by the sum of the top two rather than by 1 makes the quantity a
    function of the ratio p2/p1 alone: (1 - r) / (1 + r). It therefore measures
    how threatened the winner is by the runner-up, independently of how much
    probability mass the pair holds -- which is what we want, since fusion can
    only change a prediction by unseating the winner.
    """
    probs = torch.softmax(
        torch.where(valid, scores.float(),
                    torch.full_like(scores, -1e4, dtype=torch.float32)),
        dim=1)
    top2 = probs.topk(2, dim=1).values
    return ((top2[:, :1] - top2[:, 1:2])
            / top2.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp(0.0, 1.0)


gate_stats = {}


def _fusion_gate(routed_logits, valid, mode, rp_scores=None):
    """How much this sample should listen to the second classifier, in [0, 1].

    A fixed blend spends the same 30% of the decision on the RP head for a
    sample the routed head calls at p=0.97 as for one it is torn on. The first
    kind is the overwhelming majority on CIFAR-100, and paying the blend there
    is what made Loss regress (+0.019 +- 0.003 over three seeds) while accuracy
    still improved: flattening an already-correct peak costs cross-entropy and
    buys no argmax. Fusion can only change a prediction by unseating the top
    class, and its most likely challenger is the runner-up, so gate on how
    decided the routed head is between its own top two.

    'margin' looks at one side only. It hands the RP head up to 46% of the
    decision whenever the routed head is torn -- even when the RP head is torn
    too and has nothing to contribute. 'margin_both' closes that gap by scaling
    the same quantity by the RP head's own margin:

        margin:       beta_i = beta * (1 - conf(routed))
        margin_both:  beta_i = beta * (1 - conf(routed)) * conf(rp)

    so the blend opens only when the routed head is undecided AND the RP head
    is decided. Setting conf(rp) = 1 recovers 'margin' exactly, which gives the
    mode an identity check of the same kind as w = 1.0 and beta = 0.
    """
    # The mean gate is recorded so a run can be read without guessing. The
    # first margin_both attempt was inconclusive precisely because the gate's
    # magnitude was invisible: its numbers rose monotonically with beta, which
    # is the signature of under-fusion, but nothing in the log said by how much
    # the gate had shrunk. GateShare and ConfRP make that direct.
    gate_stats.clear()
    if mode in (None, 'none'):
        return None
    if mode == 'margin':
        gate = (1.0 - _top2_margin(routed_logits, valid)).clamp(0.0, 1.0)
        gate_stats['gate'] = gate.mean().item()
        return gate
    if mode == 'margin_both':
        if rp_scores is None:
            raise ValueError(
                "rp_class_fusion_gate='margin_both' needs the RP scores; "
                'the caller passed none')
        # Put the RP scores on the routed logits' scale before measuring their
        # margin. Ridge scores span about 1 unit while the routed logits span
        # about 35, so a softmax on the raw scores leaves every RP margin near
        # 0.05 -- the gate then comes out ~20x too small and the mode silently
        # under-fuses. Measured 2026-08-28 before this fix: 74.59 / 74.63 /
        # 74.67 at beta 0.5 / 0.8 / 1.0 on ImageNet-R seed 42, against 75.51
        # for one-sided 'margin', rising monotonically with beta -- the
        # signature of under-fusion, not of a wrong hypothesis.
        #
        # This is the same affine map the mixing below uses, so conf(rp) = 1
        # still recovers 'margin' exactly.
        mean, std = _valid_moments(routed_logits.float(), valid)
        rp_on_scale = _standardize_valid(rp_scores.float(), valid) * std + mean
        undecided = 1.0 - _top2_margin(routed_logits, valid)
        conf_rp = _top2_margin(rp_on_scale, valid)
        gate = (undecided * conf_rp).clamp(0.0, 1.0)
        gate_stats['gate'] = gate.mean().item()
        gate_stats['conf_rp'] = conf_rp.mean().item()
        gate_stats['undecided'] = undecided.mean().item()
        return gate
    if mode == 'entropy':
        probs = torch.softmax(
            torch.where(valid, routed_logits.float(),
                        torch.full_like(routed_logits, -1e4, dtype=torch.float32)),
            dim=1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
        classes = valid.sum(dim=1, keepdim=True).clamp_min(2).float()
        gate = (entropy / classes.log()).clamp(0.0, 1.0)
        gate_stats['gate'] = gate.mean().item()
        return gate
    raise ValueError('unknown rp_class_fusion_gate: %s' % mode)


def fuse_class_scores(routed_logits, rp_scores, weight, seen_tasks=None):
    """Mix the routed HRM-PET classifier with the RP head as a second classifier.

    Routing gains convert into Acc@1 poorly -- the shared head often already
    predicts the right class despite a wrong route, so fixing that route buys
    nothing and can even flip a correct prediction. The RP head is a full
    classifier that routing discards, and it is the stronger of the two on
    CUB-200 (87.96 against the baseline's 86.53) while weaker on ImageNet-R, so
    a blend can beat either alone.

    Ridge scores and softmax logits live on different scales, so each is
    standardized over the sample's valid classes before mixing; masked classes
    stay masked.
    """
    valid = torch.isfinite(routed_logits)
    routed = routed_logits.float()
    if (seen_tasks is not None
            and getattr(args_ref[0], 'rp_fusion_ramp_scope', 'both')
            in ('both', 'class')):
        weight = weight * _stage_ramp(
            seen_tasks, getattr(args_ref[0], 'num_tasks', seen_tasks),
            getattr(args_ref[0], 'rp_fusion_ramp', 0.0))
    # neginf=-1e4 here, not 0.0 as in the mixing below: the gate feeds these
    # straight into a softmax, so an unseen class must stay far away rather
    # than land in the middle of the distribution.
    gate = _fusion_gate(
        routed, valid,
        getattr(args_ref[0], 'rp_class_fusion_gate', 'none'),
        rp_scores=torch.nan_to_num(rp_scores.float(), neginf=-1e4))
    share = weight if gate is None else weight * gate
    mixed = ((1.0 - share) * _standardize_valid(routed, valid)
             + share * _standardize_valid(
                 torch.nan_to_num(rp_scores.float(), neginf=0.0), valid))
    # Standardizing puts the mixture at unit variance, which is the wrong scale
    # for cross-entropy and made Loss regress even while accuracy improved. Map
    # it back onto the routed logits' own scale: an affine map per sample, so
    # the ranking -- and therefore accuracy -- is untouched, while the loss
    # becomes comparable to the baseline's again. (This also explains why
    # --rp_calibrate was a no-op here: standardization cancels any scalar
    # temperature applied to the RP scores.)
    mean, std = _valid_moments(routed, valid)
    # Sharpening factor: mixing two classifiers shrinks the winner's margin
    # relative to the rest, which cross-entropy penalises even when the ranking
    # improves. Scaling is monotone per sample, so accuracy is bit-identical and
    # only the loss moves.
    sharpen = max(1e-3, float(getattr(args_ref[0], 'rp_class_fusion_sharpen', 1.0)))
    mixed = mixed * sharpen * std + mean
    return torch.where(
        valid, mixed, torch.full_like(mixed, float('-inf')))


def _blend_scores(head_scores, classifier_logits, weight):
    """Combine head scores with the HRM-PET classifier on a common scale.

    The two live on different scales (ridge regression values vs softmax
    logits), so each is converted to log-probabilities before mixing; masked
    classes stay at -inf.
    """
    finite = torch.isfinite(head_scores)
    head_log = torch.full_like(head_scores, float('-inf'))
    head_log[finite] = head_scores[finite]
    head_log = torch.log_softmax(head_log, dim=1)
    classifier_log = torch.log_softmax(
        classifier_logits.masked_fill(~finite, float('-inf')), dim=1)
    return head_log + weight * classifier_log


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, i=-1, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    global con_num, incon_num ,max_p, con_all, incon_all
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()
    prototype_bank = None
    local_prototype_enabled = (
        float(getattr(args, 'exhaustive_local_prototype_weight', 0.0)) > 0.0
    )
    if (bool(getattr(args, 'prototype_rematching', False))
            or local_prototype_enabled):
        seen_classes = [
            int(class_id)
            for seen_task in range(task_id + 1)
            for class_id in class_mask[seen_task]
        ]
        prototype_bank = build_prototype_bank(
            model, cls_real_features, seen_classes, device)
        if utils.is_main_process():
            if local_prototype_enabled:
                print(
                    'Task-local prototype fusion:',
                    'classes=', len(prototype_bank),
                    'weight=', float(getattr(
                        args, 'exhaustive_local_prototype_weight', 0.0)),
                    'temperature=', float(getattr(
                        args, 'exhaustive_local_prototype_temperature', 0.07)),
                )
            else:
                print(
                    'Prototype rematching:',
                    'classes=', len(prototype_bank),
                    'candidate_tasks=', int(getattr(
                        args, 'prototype_candidate_tasks', 2)),
                    'temperature=', float(getattr(
                        args, 'prototype_temperature', 0.07)),
                )
    shared_prototype_bank = None
    if bool(getattr(args, 'shared_prototype_router', False)):
        seen_classes = [
            int(class_id)
            for seen_task in range(task_id + 1)
            for class_id in class_mask[seen_task]
        ]
        shared_prototype_bank = build_shared_prototype_bank(
            cls_shared_features, seen_classes, device)
        if utils.is_main_process():
            print(
                'Shared prototype router:',
                'classes=', len(shared_prototype_bank),
                'temperature=', float(getattr(args, 'shared_prototype_temperature', 0.07)),
            )

    # Explicit sentinel rather than a `'fusion_rp_scores' in dir()` probe. dir()
    # reads the function's locals, so once any batch assigns the name it stays
    # visible to every later batch -- a batch that failed to compute its own
    # scores would silently be judged against the previous batch's. That cannot
    # happen today because the branch is gated on an args flag that is constant
    # across batches, but the guard should not depend on that staying true.
    fusion_rp_scores = None
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            fusion_rp_scores = None
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    shared_features = output.get('pre_logits')
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

            if (bool(getattr(args, 'rp_head', False))
                    and not bool(getattr(args, 'rp_route_fusion_drm', False))):
                blend_weight = float(getattr(args, 'rp_logit_blend', 0.0))
                adapter_logits = None
                layer_router = (
                    bool(getattr(args, 'layer_stat_router', False))
                    and task_stats_ready(task_id + 1))
                if layer_router:
                    install_hooks(
                        rp_extractor(original_model, args)
                        if str(getattr(args, 'rp_feature_source', 'original'))
                        == 'original' else model)
                if str(getattr(args, 'rp_feature_source', 'original')) == 'original':
                    if (bool(getattr(args, 'rp_pin_extractor', False))
                            or str(getattr(args, 'rp_input_norm', 'none')) != 'none'):
                        features = rp_extractor(original_model, args)(
                            _rp_inputs(input, args))['pre_logits']
                    else:
                        features = shared_features
                    if blend_weight != 0.0:
                        adapter_logits = old_logits
                else:
                    adapter_output = model(
                        input, task_id=int(getattr(args, 'rp_lora_task', 0)),
                        train=True)
                    features = adapter_output['pre_logits']
                    # Same forward, so blending the HRM-PET classifier costs
                    # no extra adapter call.
                    adapter_logits = adapter_output['logits']
                layer_route = None
                if layer_router:
                    layer_route = layer_stat_scores(
                        task_id + 1, device).argmax(dim=1)
                    remove_hooks()
                logits = rp_head_predict(features, args, device)
                if bool(getattr(args, 'rp_route_fusion', False)):
                    routed = fuse_routers(
                        logits, old_logits, class_mask, task_id + 1, args,
                        device)
                    routed_logits = model(input, task_id=routed)['logits']
                    if class_mask is not None:
                        seen = []
                        for seen_task in range(task_id + 1):
                            seen.extend(class_mask[seen_task])
                        unseen = np.setdiff1d(np.arange(args.nb_classes), seen)
                        routed_logits = routed_logits.index_fill(
                            dim=1,
                            index=torch.tensor(
                                unseen, dtype=torch.int64, device=device),
                            value=float('-inf'))
                    loss = criterion(routed_logits, target)
                    acc1, acc5 = accuracy(routed_logits, target, topk=(1, 5))
                    task_inference_acc = utils.task_inference_accuracy(
                        routed.unsqueeze(-1), target, target_task_map,
                        torch.empty(0, dtype=torch.long, device=device), None)
                    metric_logger.meters['Loss'].update(loss.item())
                    metric_logger.meters['Acc@1'].update(
                        acc1.item(), n=input.shape[0])
                    metric_logger.meters['Acc@5'].update(
                        acc5.item(), n=input.shape[0])
                    metric_logger.meters['Acc@task'].update(
                        task_inference_acc.item(), n=input.shape[0])
                    metric_logger.meters['LoRA/sample'].update(
                        2.0, n=input.shape[0])
                    metric_logger.meters['ForwardCalls/sample'].update(
                        2.0, n=input.shape[0])
                    continue
                if blend_weight != 0.0 and adapter_logits is not None:
                    logits = _blend_scores(
                        logits, adapter_logits.float(), blend_weight)
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                predicted_class = logits.argmax(dim=1)
                prompt_id = torch.tensor(
                    [target_task_map[int(c)] for c in predicted_class.cpu()],
                    device=device)
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    torch.empty(0, dtype=torch.long, device=device), None)
                if bool(getattr(args, 'rp_route_audit', False)):
                    # Are the RP head and TII wrong on the SAME samples? If
                    # their routing errors are decorrelated, fusing them can
                    # recover accuracy that neither reaches alone.
                    true_task = torch.tensor(
                        [target_task_map[int(t)] for t in target.cpu()],
                        device=device)
                    tii_class = old_logits.argmax(dim=1)
                    tii_task = torch.tensor(
                        [target_task_map[int(c)] for c in tii_class.cpu()],
                        device=device)
                    tii_ok = tii_task.eq(true_task)
                    rp_ok = prompt_id.eq(true_task)
                    metric_logger.meters['RouteTII'].update(
                        tii_ok.float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteRP'].update(
                        rp_ok.float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteUnion'].update(
                        (tii_ok | rp_ok).float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteBoth'].update(
                        (tii_ok & rp_ok).float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteAgree'].update(
                        tii_task.eq(prompt_id).float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteRPOnly'].update(
                        (rp_ok & ~tii_ok).float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    if layer_route is not None:
                        ls_ok = layer_route.eq(true_task)
                        metric_logger.meters['RouteLS'].update(
                            ls_ok.float().mean().mul(100.0).item(),
                            n=input.shape[0])
                        metric_logger.meters['RouteUnion3'].update(
                            (tii_ok | rp_ok | ls_ok).float().mean().mul(
                                100.0).item(), n=input.shape[0])
                        metric_logger.meters['RouteLSOnly'].update(
                            (ls_ok & ~tii_ok & ~rp_ok).float().mean().mul(
                                100.0).item(), n=input.shape[0])
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(
                    acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(
                    acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                metric_logger.meters['LoRA/sample'].update(
                    0.0 if str(getattr(
                        args, 'rp_feature_source', 'original')) == 'original'
                    else 1.0, n=input.shape[0])
                metric_logger.meters['ForwardCalls/sample'].update(
                    1.0, n=input.shape[0])
                continue

            if bool(getattr(args, 'gaussian_rescoring', False)):
                logits, prompt_id = gaussian_rescoring_predict(
                    model=model, inputs=input, class_mask=class_mask,
                    seen_task_count=task_id + 1, args=args, device=device)
                filtered_index_tensor = torch.empty(
                    0, dtype=torch.long, device=device)
                re_id = None
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    filtered_index_tensor, re_id)
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(
                    acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(
                    acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                metric_logger.meters['LoRA/sample'].update(
                    float(task_id + 1), n=input.shape[0])
                metric_logger.meters['ForwardCalls/sample'].update(
                    float(task_id + 1), n=input.shape[0])
                continue

            if bool(getattr(args, 'calibrated_progressive_rematching', False)):
                logits, prompt_id, lora_counts, forward_calls, stop_stage = calibrated_progressive_rematching(
                    model=model,
                    inputs=input,
                    tii_logits=old_logits,
                    class_mask=class_mask,
                    seen_task_count=task_id + 1,
                    args=args,
                    gates=get_progressive_halting_gates(),
                )
                filtered_index_tensor = torch.empty(
                    0, dtype=torch.long, device=device)
                re_id = None
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    filtered_index_tensor, re_id)
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                metric_logger.meters['LoRA/sample'].update(
                    lora_counts.mean().item(), n=input.shape[0])
                metric_logger.meters['ForwardCalls/sample'].update(
                    forward_calls.mean().item(), n=input.shape[0])
                metric_logger.meters['Stage1StopRate'].update(
                    stop_stage.eq(1).float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['Stage2StopRate'].update(
                    stop_stage.eq(2).float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['FullFallbackRate'].update(
                    stop_stage.eq(3).float().mean().mul(100.0).item(),
                    n=input.shape[0])
                continue

            if bool(getattr(args, 'progressive_oracle_audit', False)):
                logits, prompt_id, audit = progressive_oracle_audit(
                    model=model,
                    inputs=input,
                    tii_logits=old_logits,
                    class_mask=class_mask,
                    seen_task_count=task_id + 1,
                    args=args,
                    targets=target,
                    true_task=i,
                )
                filtered_index_tensor = torch.empty(
                    0, dtype=torch.long, device=device)
                re_id = None
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    filtered_index_tensor, re_id)
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                audit_percent_metrics = {
                    'WinnerRecall@2': 'winner_recall_2',
                    'WinnerRecall@4': 'winner_recall_4',
                    'ExactAgreement@2': 'exact_agreement_2',
                    'ExactAgreement@4': 'exact_agreement_4',
                }
                for metric_name, audit_name in audit_percent_metrics.items():
                    metric_logger.meters[metric_name].update(
                        audit[audit_name].float().mean().mul(100.0).item(),
                        n=input.shape[0])
                metric_logger.meters['OracleLoRA/sample'].update(
                    audit['oracle_lora_counts'].mean().item(), n=input.shape[0])
                metric_logger.meters['ActualLoRA/sample'].update(
                    audit['actual_lora_counts'].mean().item(), n=input.shape[0])
                if bool(getattr(args, 'router_recall_audit', False)):
                    for router_name in ('max', 'energy', 'margin', 'mean'):
                        metric_logger.meters[
                            'Router_{}_MeanRank'.format(router_name)].update(
                            audit['router_{}_mean_rank'.format(
                                router_name)].mean().item(),
                            n=input.shape[0])
                        for recall_k in (1, 2, 3, 4):
                            metric_logger.meters[
                                'Router_{}_Recall@{}'.format(
                                    router_name, recall_k)].update(
                                audit['router_{}_recall_{}'.format(
                                    router_name, recall_k)].float().mean()
                                .mul(100.0).item(),
                                n=input.shape[0])
                if bool(getattr(args, 'stage_drift_audit', False)):
                    stage_percent_metrics = {
                        'OwnLocalAcc@1':
                            'stage_drift_own_local_correct',
                        'OwnSeenAcc@1':
                            'stage_drift_own_seen_correct',
                        'OwnSeenTaskAcc':
                            'stage_drift_own_seen_task_correct',
                        'LocalToSeenFailure':
                            'stage_drift_local_to_seen_failure',
                    }
                    for metric_name, audit_name in (
                            stage_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['OwnLocalLoss'].update(
                        audit['stage_drift_own_local_loss'].mean().item(),
                        n=input.shape[0])
                    metric_logger.meters['OwnSeenLoss'].update(
                        audit['stage_drift_own_seen_loss'].mean().item(),
                        n=input.shape[0])
                if bool(getattr(args, 'progressive_arrow_audit', False)):
                    arrow_percent_metrics = {
                        'ArrowRecall@2': 'arrow_winner_recall_2',
                        'ArrowRecall@4': 'arrow_winner_recall_4',
                        'UnionRecall@2x2': 'arrow_union_recall_2x2',
                        'TIIArrowTop1Agree': 'tii_arrow_top1_agreement',
                    }
                    for metric_name, audit_name in arrow_percent_metrics.items():
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['UnionLoRA/sample'].update(
                        audit['arrow_union_lora_counts'].mean().item(),
                        n=input.shape[0])
                if bool(getattr(args, 'progressive_lora_response_audit', False)):
                    response_percent_metrics = {
                        'ResponseRecall@2': 'response_winner_recall_2',
                        'ResponseRecall@4': 'response_winner_recall_4',
                        'ResponseUnionRecall@2x2': 'response_union_recall_2x2',
                        'TIIResponseTop1Agree': 'tii_response_top1_agreement',
                    }
                    for metric_name, audit_name in response_percent_metrics.items():
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['ResponseUnionLoRA/sample'].update(
                        audit['response_union_lora_counts'].mean().item(),
                        n=input.shape[0])
                if bool(getattr(
                        args, 'progressive_prediction_proposal_audit', False)):
                    proposal_percent_metrics = {
                        'ProposalWinnerRecall':
                            'prediction_proposal_winner_recall',
                        'ProposalExactAgreement':
                            'prediction_proposal_exact_agreement',
                        'ProposalNewWinner':
                            'prediction_proposal_new_winner',
                    }
                    for metric_name, audit_name in proposal_percent_metrics.items():
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['ProposalLoRA/sample'].update(
                        audit['prediction_proposal_lora_counts'].mean().item(),
                        n=input.shape[0])
                if bool(getattr(
                        args, 'progressive_prediction_closure_audit', False)):
                    closure_percent_metrics = {
                        'ClosureWinnerRecall':
                            'prediction_closure_winner_recall',
                        'ClosureExactAgreement':
                            'prediction_closure_exact_agreement',
                        'ClosureTop5Coverage':
                            'prediction_closure_top5_coverage',
                        'ClosureFullScanRate':
                            'prediction_closure_full_scan',
                    }
                    for metric_name, audit_name in (
                            closure_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['ClosureLoRA/sample'].update(
                        audit['prediction_closure_lora_counts'].mean().item(),
                        n=input.shape[0])
                    metric_logger.meters['ClosureCalls/sample'].update(
                        audit['prediction_closure_forward_calls'].mean().item(),
                        n=input.shape[0])
                if bool(getattr(
                        args, 'progressive_prediction_beam_closure_audit',
                        False)):
                    beam_percent_metrics = {
                        'BeamClosureWinnerRecall':
                            'prediction_beam_closure_winner_recall',
                        'BeamClosureExactAgreement':
                            'prediction_beam_closure_exact_agreement',
                        'BeamClosureFullScanRate':
                            'prediction_beam_closure_full_scan',
                    }
                    for metric_name, audit_name in (
                            beam_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['BeamClosureLoRA/sample'].update(
                        audit[
                            'prediction_beam_closure_lora_counts'
                        ].mean().item(), n=input.shape[0])
                    metric_logger.meters['BeamClosureCalls/sample'].update(
                        audit[
                            'prediction_beam_closure_forward_calls'
                        ].mean().item(), n=input.shape[0])
                if bool(getattr(
                        args, 'progressive_prediction_budget_closure_audit',
                        False)):
                    budget_percent_metrics = {
                        'BudgetClosureWinnerRecall':
                            'prediction_budget_closure_winner_recall',
                        'BudgetClosureExactAgreement':
                            'prediction_budget_closure_exact_agreement',
                        'BudgetClosureBudgetHitRate':
                            'prediction_budget_closure_budget_hit',
                    }
                    for metric_name, audit_name in (
                            budget_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['BudgetClosureLoRA/sample'].update(
                        audit[
                            'prediction_budget_closure_lora_counts'
                        ].mean().item(), n=input.shape[0])
                    metric_logger.meters['BudgetClosureCalls/sample'].update(
                        audit[
                            'prediction_budget_closure_forward_calls'
                        ].mean().item(), n=input.shape[0])
                if bool(getattr(
                        args, 'progressive_prediction_majority_closure_audit',
                        False)):
                    majority_percent_metrics = {
                        'MajorityClosureWinnerRecall':
                            'prediction_majority_closure_winner_recall',
                        'MajorityClosureExactAgreement':
                            'prediction_majority_closure_exact_agreement',
                        'MajorityClosureCertifiedRate':
                            'prediction_majority_closure_majority_rate',
                    }
                    for metric_name, audit_name in (
                            majority_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['MajorityClosureLoRA/sample'].update(
                        audit[
                            'prediction_majority_closure_lora_counts'
                        ].mean().item(), n=input.shape[0])
                    metric_logger.meters['MajorityClosureCalls/sample'].update(
                        audit[
                            'prediction_majority_closure_forward_calls'
                        ].mean().item(), n=input.shape[0])
                if bool(getattr(
                        args, 'progressive_prediction_consensus_closure_audit',
                        False)):
                    consensus_percent_metrics = {
                        'ConsensusClosureWinnerRecall':
                            'prediction_consensus_closure_winner_recall',
                        'ConsensusClosureExactAgreement':
                            'prediction_consensus_closure_exact_agreement',
                        'ConsensusClosureCertifiedRate':
                            'prediction_consensus_closure_certified_rate',
                    }
                    for metric_name, audit_name in (
                            consensus_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['ConsensusClosureLoRA/sample'].update(
                        audit[
                            'prediction_consensus_closure_lora_counts'
                        ].mean().item(), n=input.shape[0])
                    metric_logger.meters['ConsensusClosureCalls/sample'].update(
                        audit[
                            'prediction_consensus_closure_forward_calls'
                        ].mean().item(), n=input.shape[0])
                if bool(getattr(
                        args, 'progressive_prediction_self_owner_audit',
                        False)):
                    self_owner_percent_metrics = {
                        'SelfOwnerWinnerRecall':
                            'prediction_self_owner_winner_recall',
                        'SelfOwnerExactAgreement':
                            'prediction_self_owner_exact_agreement',
                        'SelfOwnerRouteRate':
                            'prediction_self_owner_route_rate',
                        'SelfOwnerMultipleRate':
                            'prediction_self_owner_multiple_rate',
                    }
                    for metric_name, audit_name in (
                            self_owner_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['SelfOwnerSupport'].update(
                        audit['prediction_self_owner_support'].mean().item(),
                        n=input.shape[0])
                    metric_logger.meters['SelfOwnerLoRA/sample'].update(
                        audit['prediction_self_owner_lora_counts'].mean().item(),
                        n=input.shape[0])
                    metric_logger.meters['SelfOwnerCalls/sample'].update(
                        audit[
                            'prediction_self_owner_forward_calls'
                        ].mean().item(), n=input.shape[0])
                if bool(getattr(
                        args,
                        'progressive_prediction_corroborated_owner_audit',
                        False)):
                    corroborated_percent_metrics = {
                        'CorroboratedOwnerWinnerRecall':
                            'prediction_corroborated_owner_winner_recall',
                        'CorroboratedOwnerExactAgreement':
                            'prediction_corroborated_owner_exact_agreement',
                        'CorroboratedOwnerRescueRate':
                            'prediction_corroborated_owner_rescue_rate',
                    }
                    for metric_name, audit_name in (
                            corroborated_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    rescued = audit[
                        'prediction_corroborated_owner_rescue_rate']
                    rescued_support = audit[
                        'prediction_corroborated_owner_support']
                    support_total = rescued_support.sum()
                    support_count = rescued.sum().clamp_min(1)
                    metric_logger.meters[
                        'CorroboratedOwnerSupport'].update(
                            (support_total / support_count).item(),
                            n=max(1, int(rescued.sum().item())))
                    metric_logger.meters[
                        'CorroboratedOwnerLoRA/sample'].update(
                            audit[
                                'prediction_corroborated_owner_lora_counts'
                            ].mean().item(), n=input.shape[0])
                    metric_logger.meters[
                        'CorroboratedOwnerCalls/sample'].update(
                            audit[
                                'prediction_corroborated_owner_forward_calls'
                            ].mean().item(), n=input.shape[0])
                if bool(getattr(
                        args, 'progressive_prediction_owner_mass_audit',
                        False)):
                    owner_mass_percent_metrics = {
                        'OwnerMassWinnerRecall':
                            'prediction_owner_mass_winner_recall',
                        'OwnerMassExactAgreement':
                            'prediction_owner_mass_exact_agreement',
                        'OwnerMassRescueRate':
                            'prediction_owner_mass_rescue_rate',
                        'OwnerMassAmbiguousRate':
                            'prediction_owner_mass_ambiguous_rate',
                    }
                    for metric_name, audit_name in (
                            owner_mass_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['OwnerMassLoRA/sample'].update(
                        audit['prediction_owner_mass_lora_counts'].mean().item(),
                        n=input.shape[0])
                    metric_logger.meters['OwnerMassCalls/sample'].update(
                        audit[
                            'prediction_owner_mass_forward_calls'
                        ].mean().item(), n=input.shape[0])
                if bool(getattr(
                        args,
                        'progressive_prediction_owner_consensus_hedge_audit',
                        False)):
                    hedge_percent_metrics = {
                        'OwnerHedgeWinnerRecall':
                            'prediction_owner_hedge_winner_recall',
                        'OwnerHedgeExactAgreement':
                            'prediction_owner_hedge_exact_agreement',
                        'OwnerHedgeRate':
                            'prediction_owner_hedge_rate',
                        'OwnerHedgeOwnerAgreement':
                            'prediction_owner_hedge_owner_agreement',
                        'OwnerHedgeConsensusAgreement':
                            'prediction_owner_hedge_consensus_agreement',
                    }
                    for metric_name, audit_name in (
                            hedge_percent_metrics.items()):
                        metric_logger.meters[metric_name].update(
                            audit[audit_name].float().mean().mul(100.0).item(),
                            n=input.shape[0])
                    metric_logger.meters['OwnerHedgeLoRA/sample'].update(
                        audit['prediction_owner_hedge_lora_counts'].mean().item(),
                        n=input.shape[0])
                    metric_logger.meters['OwnerHedgeCalls/sample'].update(
                        audit[
                            'prediction_owner_hedge_forward_calls'
                        ].mean().item(), n=input.shape[0])
                continue

            if bool(getattr(args, 'progressive_rematching', False)):
                logits, prompt_id, lora_counts, stop_stage = progressive_adapter_rematching(
                    model=model,
                    inputs=input,
                    tii_logits=old_logits,
                    class_mask=class_mask,
                    seen_task_count=task_id + 1,
                    args=args,
                )
                filtered_index_tensor = torch.empty(
                    0, dtype=torch.long, device=device)
                re_id = None
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    filtered_index_tensor, re_id)
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                metric_logger.meters['LoRA/sample'].update(
                    lora_counts.mean().item(), n=input.shape[0])

                metric_logger.meters['Stage1StopRate'].update(
                    stop_stage.eq(1).float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['Stage2StopRate'].update(
                    stop_stage.eq(2).float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['FullFallbackRate'].update(
                    stop_stage.eq(3).float().mean().mul(100.0).item(),
                    n=input.shape[0])
                continue

            if (bool(getattr(args, 'soft_mixture_rematching', False))
                    or bool(getattr(args, 'soft_mixture_hard_rematching', False))
                    or bool(getattr(args, 'soft_hard_selector_rematching', False))
                    or bool(getattr(args, 'soft_local_hard_refinement', False))):
                mixture_function = soft_mixture_adapter_rematching
                if bool(getattr(args, 'soft_local_hard_refinement', False)):
                    mixture_function = soft_mixture_local_hard_refinement
                elif bool(getattr(args, 'soft_hard_selector_rematching', False)):
                    mixture_function = soft_hard_confidence_selector_rematching
                elif bool(getattr(args, 'soft_mixture_hard_rematching', False)):
                    mixture_function = soft_mixture_hard_adapter_rematching
                logits, prompt_id, mixture_diagnostics = (
                    mixture_function(
                        model=model,
                        inputs=input,
                        tii_logits=old_logits,
                        class_mask=class_mask,
                        seen_task_count=task_id + 1,
                        args=args,
                    )
                )
                filtered_index_tensor = torch.empty(
                    0, dtype=torch.long, device=device)
                re_id = None
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    filtered_index_tensor, re_id)
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(
                    acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(
                    acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                metric_logger.meters['LoRA/sample'].update(
                    mixture_diagnostics['lora_counts'].mean().item(),
                    n=input.shape[0])
                metric_logger.meters['ForwardCalls/sample'].update(
                    mixture_diagnostics['forward_calls'].mean().item(),
                    n=input.shape[0])
                if bool(getattr(args, 'soft_hard_selector_rematching', False)):
                    soft_prediction = mixture_diagnostics[
                        'soft_logits'].argmax(dim=1)
                    hard_prediction = mixture_diagnostics[
                        'hard_logits'].argmax(dim=1)
                    soft_correct = soft_prediction.eq(target)
                    hard_correct = hard_prediction.eq(target)
                    selector_metrics = {
                        'SoftAcc@1': soft_correct.float().mean() * 100.0,
                        'HardAcc@1': hard_correct.float().mean() * 100.0,
                        'SoftHardAgree': soft_prediction.eq(
                            hard_prediction).float().mean() * 100.0,
                        'SoftOnlyCorrect': (soft_correct & ~hard_correct).float(
                            ).mean() * 100.0,
                        'HardOnlyCorrect': (hard_correct & ~soft_correct).float(
                            ).mean() * 100.0,
                        'OracleAcc@1': (soft_correct | hard_correct).float(
                            ).mean() * 100.0,
                        'HardSelectRate': mixture_diagnostics[
                            'select_hard'].float().mean() * 100.0,
                    }
                    for metric_name, metric_value in selector_metrics.items():
                        metric_logger.meters[metric_name].update(
                            metric_value.item(), n=input.shape[0])
                if bool(getattr(args, 'soft_local_hard_refinement', False)):
                    soft_prediction = mixture_diagnostics[
                        'soft_logits'].argmax(dim=1)
                    refined_prediction = logits.argmax(dim=1)
                    soft_correct = soft_prediction.eq(target)
                    refined_correct = refined_prediction.eq(target)
                    refinement_metrics = {
                        'RefineSoftAcc@1': soft_correct.float().mean() * 100.0,
                        'RefinedAcc@1': refined_correct.float().mean() * 100.0,
                        'SoftRefineAgree': soft_prediction.eq(
                            refined_prediction).float().mean() * 100.0,
                        'RefineSoftOnlyCorrect': (
                            soft_correct & ~refined_correct).float().mean() * 100.0,
                        'RefineOnlyCorrect': (
                            refined_correct & ~soft_correct).float().mean() * 100.0,
                        'RefineOracleAcc@1': (
                            soft_correct | refined_correct).float().mean() * 100.0,
                    }
                    for metric_name, metric_value in refinement_metrics.items():
                        metric_logger.meters[metric_name].update(
                            metric_value.item(), n=input.shape[0])
                continue

            if (bool(getattr(args, 'hierarchical_rematching', False))
                    or bool(getattr(args, 'exhaustive_rematching', False))
                    or bool(getattr(args, 'vectorized_exhaustive_rematching', False))
                    or bool(getattr(args, 'prediction_proposal_rematching', False))
                    or bool(getattr(args, 'prediction_closure_rematching', False))):
                cost_diagnostics = None
                if bool(getattr(args, 'hierarchical_rematching', False)):
                    logits, prompt_id = hierarchical_adapter_rematching(
                        model=model,
                        inputs=input,
                        tii_logits=old_logits,
                        class_mask=class_mask,
                        seen_task_count=task_id + 1,
                        args=args,
                    )
                elif bool(getattr(args, 'prediction_closure_rematching', False)):
                    logits, prompt_id, cost_diagnostics = (
                        prediction_closure_tii_tail_rematching(
                            model=model,
                            inputs=input,
                            tii_logits=old_logits,
                            class_mask=class_mask,
                            seen_task_count=task_id + 1,
                            args=args,
                        )
                    )
                elif bool(getattr(args, 'prediction_proposal_rematching', False)):
                    logits, prompt_id, cost_diagnostics = (
                        prediction_proposal_adapter_rematching(
                            model=model,
                            inputs=input,
                            tii_logits=old_logits,
                            class_mask=class_mask,
                            seen_task_count=task_id + 1,
                            args=args,
                        )
                    )
                elif bool(getattr(args, 'vectorized_exhaustive_rematching', False)):
                    logits, prompt_id, cost_diagnostics = (
                        vectorized_exhaustive_adapter_rematching(
                            model=model,
                            inputs=input,
                            tii_logits=old_logits,
                            class_mask=class_mask,
                            seen_task_count=task_id + 1,
                            args=args,
                            prototype_bank=prototype_bank,
                        )
                    )
                else:
                    logits, prompt_id = exhaustive_adapter_rematching(
                        model=model,
                        inputs=input,
                        tii_logits=old_logits,
                        class_mask=class_mask,
                        seen_task_count=task_id + 1,
                        args=args,
                        prototype_bank=prototype_bank,
                    )
                filtered_index_tensor = torch.empty(
                    0, dtype=torch.long, device=device)
                re_id = None
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    filtered_index_tensor, re_id)
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(
                    acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(
                    acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                if cost_diagnostics is not None:
                    metric_logger.meters['LoRA/sample'].update(
                        cost_diagnostics['lora_counts'].mean().item(),
                        n=input.shape[0])
                    metric_logger.meters['ForwardCalls/sample'].update(
                        cost_diagnostics['forward_calls'].mean().item(),
                        n=input.shape[0])
                    if bool(getattr(
                            args, 'prediction_proposal_initial_branch_audit', False)):
                        initial_branch_logits = cost_diagnostics[
                            'initial_branch_logits']
                        if args.train_mask and class_mask is not None:
                            seen_classes = [
                                class_id
                                for seen_task in range(task_id + 1)
                                for class_id in class_mask[seen_task]
                            ]
                            unseen_classes = np.setdiff1d(
                                np.arange(args.nb_classes), seen_classes)
                            unseen_classes = torch.as_tensor(
                                unseen_classes, dtype=torch.long, device=device)
                            initial_branch_logits = initial_branch_logits.index_fill(
                                1, unseen_classes, float('-inf'))
                        initial_prediction = initial_branch_logits.argmax(dim=1)
                        proposal_prediction = logits.argmax(dim=1)
                        initial_correct = initial_prediction.eq(target)
                        proposal_correct = proposal_prediction.eq(target)
                        select_initial = initial_branch_confidence_dominance(
                            initial_branch_logits, logits)
                        dominance_prediction = torch.where(
                            select_initial, initial_prediction,
                            proposal_prediction)
                        dominance_correct = dominance_prediction.eq(target)
                        comparison_metrics = {
                            'InitialBranchAcc@1': initial_correct.float(
                                ).mean() * 100.0,
                            'ProposalAuditAcc@1': proposal_correct.float(
                                ).mean() * 100.0,
                            'InitialProposalAgree': initial_prediction.eq(
                                proposal_prediction).float().mean() * 100.0,
                            'InitialOnlyCorrect': (
                                initial_correct & ~proposal_correct
                                ).float().mean() * 100.0,
                            'ProposalOnlyCorrect': (
                                proposal_correct & ~initial_correct
                                ).float().mean() * 100.0,
                            'InitialProposalOracleAcc@1': (
                                initial_correct | proposal_correct
                                ).float().mean() * 100.0,
                            'DominanceAcc@1': dominance_correct.float(
                                ).mean() * 100.0,
                            'InitialSelectRate': select_initial.float(
                                ).mean() * 100.0,
                        }
                        for metric_name, metric_value in comparison_metrics.items():
                            metric_logger.meters[metric_name].update(
                                metric_value.item(), n=input.shape[0])
                    if bool(getattr(
                            args, 'prediction_proposal_cross_adapter_audit', False)):
                        candidate_logits = cost_diagnostics['candidate_logits']
                        if args.train_mask and class_mask is not None:
                            seen_classes = [
                                class_id
                                for seen_task in range(task_id + 1)
                                for class_id in class_mask[seen_task]
                            ]
                            unseen_classes = np.setdiff1d(
                                np.arange(args.nb_classes), seen_classes)
                            unseen_classes = torch.as_tensor(
                                unseen_classes, dtype=torch.long, device=device)
                            candidate_logits = candidate_logits.index_fill(
                                2, unseen_classes, float('-inf'))
                        consensus = cross_adapter_global_consensus(
                            candidate_logits)
                        borda = cross_adapter_borda_consensus(
                            candidate_logits, top_k=5)
                        adapter_predictions = consensus[
                            'adapter_predictions']
                        vote_prediction = consensus['consensus_prediction']
                        proposal_prediction = logits.argmax(dim=1)
                        vote_correct = vote_prediction.eq(target)
                        proposal_correct = proposal_prediction.eq(target)
                        adapter_any_correct = adapter_predictions.eq(
                            target.unsqueeze(1)).any(dim=1)
                        select_vote = (
                            consensus['strict_majority']
                            & vote_prediction.ne(proposal_prediction)
                        )
                        rescue_prediction = torch.where(
                            select_vote, vote_prediction, proposal_prediction)
                        rescue_correct = rescue_prediction.eq(target)
                        borda_prediction = borda['prediction']
                        borda_correct = borda_prediction.eq(target)
                        select_borda = (
                            borda['strict_support']
                            & borda_prediction.ne(proposal_prediction)
                        )
                        borda_rescue_prediction = torch.where(
                            select_borda, borda_prediction,
                            proposal_prediction)
                        borda_rescue_correct = borda_rescue_prediction.eq(target)
                        cross_metrics = {
                            'CrossVoteAcc@1': vote_correct.float().mean() * 100.0,
                            'CrossAdapterOracleAcc@1': adapter_any_correct.float(
                                ).mean() * 100.0,
                            'CrossVoteStrength': consensus['vote_strength'].mean(
                                ) * 100.0,
                            'CrossVoteOnlyCorrect': (
                                vote_correct & ~proposal_correct
                                ).float().mean() * 100.0,
                            'ProposalOnlyVsCrossVote': (
                                proposal_correct & ~vote_correct
                                ).float().mean() * 100.0,
                            'CrossProposalOracleAcc@1': (
                                vote_correct | proposal_correct
                                ).float().mean() * 100.0,
                            'CrossRescueAcc@1': rescue_correct.float(
                                ).mean() * 100.0,
                            'CrossRescueRate': select_vote.float().mean() * 100.0,
                            'CrossBordaAcc@1': borda_correct.float(
                                ).mean() * 100.0,
                            'CrossBordaOnlyCorrect': (
                                borda_correct & ~proposal_correct
                                ).float().mean() * 100.0,
                            'ProposalOnlyVsCrossBorda': (
                                proposal_correct & ~borda_correct
                                ).float().mean() * 100.0,
                            'CrossBordaProposalOracleAcc@1': (
                                borda_correct | proposal_correct
                                ).float().mean() * 100.0,
                            'CrossBordaRescueAcc@1': borda_rescue_correct.float(
                                ).mean() * 100.0,
                            'CrossBordaRescueRate': select_borda.float(
                                ).mean() * 100.0,
                            'CrossBordaTop5Support': borda['topk_support'].mean(
                                ) * 100.0,
                        }
                        for metric_name, metric_value in cross_metrics.items():
                            metric_logger.meters[metric_name].update(
                                metric_value.item(), n=input.shape[0])
                continue

            if bool(getattr(args, 'selective_rematching', False)):
                candidate_scores = None
                if (str(getattr(args, 'selective_candidate_source', 'tii')).lower()
                        == 'router' and replay_task_router is not None):
                    replay_task_router.eval()
                    candidate_scores = replay_task_router(shared_features, old_logits)
                logits, prompt_id, candidate_tasks, candidate_counts = selective_adapter_rematching(
                    model=model,
                    inputs=input,
                    tii_logits=old_logits,
                    class_mask=class_mask,
                    seen_task_count=task_id + 1,
                    args=args,
                    candidate_scores=candidate_scores,
                )
                filtered_index_tensor = torch.empty(
                    0, dtype=torch.long, device=device)
                re_id = None
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    filtered_index_tensor, re_id)
                target_tasks = torch.as_tensor(
                    [target_task_map[value.item()] for value in target],
                    dtype=torch.long, device=device)
                active_candidates = (
                    torch.arange(candidate_tasks.shape[1], device=device).unsqueeze(0)
                    < candidate_counts.unsqueeze(1))
                candidate_recall = (
                    (candidate_tasks == target_tasks.unsqueeze(1)) & active_candidates
                ).any(dim=1).float().mean() * 100.0

                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                metric_logger.meters['CandidateRecall'].update(
                    candidate_recall.item(), n=input.shape[0])
                metric_logger.meters['LoRA/sample'].update(
                    candidate_counts.float().mean().item(), n=input.shape[0])
                continue

            lora_id = torch.max(old_logits, dim=1)[1]
            lora_id = torch.tensor([target_task_map[v.item()] for v in lora_id], device=device)
            
            output = model(input, task_id=lora_id)
            logits = output['logits']
            

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[i]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                #print(logits[0])
            id_logits = logits
            routing_mode = str(getattr(args, 'task_routing_mode', 'class')).lower()
            if routing_mode == 'task_energy':
                task_scores = compute_task_energy_scores(
                    id_logits, class_mask, task_id + 1,
                    temperature=getattr(args, 'task_routing_temperature', 0.1))
                candidate_count = min(2, task_scores.shape[1])
                top5_id = torch.topk(task_scores, k=candidate_count, dim=1, largest=True).indices
                prompt_id = top5_id[:, 0]
            else:
                prompt_class = torch.max(id_logits, dim=1)[1]
                _, top_5_indices = torch.topk(id_logits, 2, dim=1)
                task_map_keys = list(target_task_map.keys())
                task_map_values = torch.tensor([target_task_map[k] for k in task_map_keys], device=device)
                task_map_tensor = torch.zeros((max(task_map_keys) + 1,), dtype=torch.long, device=device)
                task_map_tensor[task_map_keys] = task_map_values
                top5_id = torch.index_select(task_map_tensor, 0, top_5_indices.view(-1)).view_as(top_5_indices)
                prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_class], device=device)
            if bool(getattr(args, 'rp_route_fusion_drm', False)):
                # Replace only HRM-PET's first stage. Its DRM/CRM refinement is
                # what carries the baseline on ImageNet-R, so it is kept; the
                # fused router just hands it a better starting task.
                use_layer = (
                    bool(getattr(args, 'layer_stat_router', False))
                    and task_stats_ready(task_id + 1))
                if use_layer:
                    install_hooks(
                        rp_extractor(original_model, args)
                        if str(getattr(args, 'rp_feature_source', 'original'))
                        == 'original' else model)
                if str(getattr(args, 'rp_feature_source', 'original')) == 'original':
                    rp_features = rp_extractor(original_model, args)(
                        _rp_inputs(input, args))['pre_logits']
                else:
                    rp_features = model(
                        input, task_id=int(getattr(args, 'rp_lora_task', 0)),
                        train=True)['pre_logits']
                layer_scores = None
                if use_layer:
                    layer_scores = layer_stat_scores(task_id + 1, device)
                    remove_hooks()
                fusion_rp_scores = rp_head_predict(rp_features, args, device)
                prompt_id = fuse_routers(
                    fusion_rp_scores, id_logits,
                    class_mask, task_id + 1, args, device,
                    layer_scores=layer_scores)
            ##############
            # target_id = torch.tensor([target_task_map[v.item()] for v in target], device=device)

            
            ###########
            id_logits = F.softmax(id_logits,dim=1)
            values, _ = torch.topk(id_logits, 1, dim=1)
            if args.En == 'msp':
                low_confidence_indices = torch.where(values <args.tau)[0]
            else:
                softmax_ood = id_logits
                softmax_ood = softmax_ood.cpu().numpy()
                #energy = (0.1*torch.logsumexp(inp / 0.1, dim=1))
                en = generalized_entropy(softmax_ood, 0.1, 20)
                
                en = torch.tensor(en)
                en = en.to(device, non_blocking=True)
                low_confidence_indices = torch.where(en <args.tau)[0]


            filtered_index_tensor = low_confidence_indices    

            
            equal_drm = torch.nonzero(prompt_id != lora_id).flatten()
            output_drm = model(input[equal_drm], task_id=prompt_id[equal_drm])
            output_drm_logits = output_drm['logits']
            if routing_mode == 'task_energy' and args.train_mask and class_mask is not None:
                output_drm_logits = output_drm_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            logits[equal_drm] = output_drm_logits
            #promtp_idx = output['prompt_idx']  # tensor B x topk
            
            corr_id=None
            re_id = None
            if task_id>0:
                    error_input = input[filtered_index_tensor]
                    # error_target = target[filtered_index_tensor]
                    ensemble_id = top5_id[filtered_index_tensor,:2]
                    #ensemble_id = second_largest_classes
                    error_output = []
                    error_output.append(logits[filtered_index_tensor,:])
                    # for i in range(ensemble_id.shape[1]):
                    out = model(error_input, task_id=ensemble_id[:,1])
                    alternative_logits = out['logits']
                    if routing_mode == 'task_energy' and args.train_mask and class_mask is not None:
                        alternative_logits = alternative_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    error_output.append(alternative_logits)
                    error_output = torch.stack(error_output,dim=1)
                   
                    entropy = (0.1*torch.logsumexp( error_output/ 0.1, dim=2))
                    corr_id = torch.max(entropy, dim=1)[1]
                    # print(corr_id)
                    corr = corr_id.unsqueeze(1).unsqueeze(2)
                    result = torch.gather(error_output, 1, corr.expand(-1, 1, len(class_mask[0])*len(class_mask)))

                    result = result.squeeze(1)
                    logits[filtered_index_tensor]=result
                    if routing_mode == 'task_energy':
                        selected_tasks = torch.gather(ensemble_id, 1, corr_id.unsqueeze(1)).squeeze(1)
                        prompt_id[filtered_index_tensor] = selected_tasks
                        re_id = selected_tasks
                    # re_id = []
                    # for k in range(len(corr_id)):
                    #     #print(corr_id[k].item())
                    #     re_id.append(ensemble_id[k,corr_id[k].item()].item())
                    # re_id = torch.tensor(re_id)
                    # re_id = re_id.to(device, non_blocking=True)
                    # try:
                    #     prompt_id[filtered_index_tensor] = re_id
                    # except:
                    #     pass
                    
            
            # HRM-PET (DRM+CRM) cost: 1 base LoRA per sample, +1 for DRM-rerouted
            # samples, +1 for CRM-rematched low-confidence samples (task_id > 0).
            default_lora_counts = torch.ones(
                input.shape[0], dtype=torch.float32, device=device)
            default_lora_counts[equal_drm] += 1.0
            if task_id > 0:
                default_lora_counts[filtered_index_tensor] += 1.0
            if bool(getattr(args, 'rp_route_fusion_drm', False)):
                # Router fusion runs one extra adapter forward per sample to
                # extract the RP head's features; count it so the reported cost
                # is the method's real cost, not the baseline's.
                default_lora_counts += 1.0

            if bool(getattr(args, 'budgeted_rematching', False)):
                base_lora_counts = torch.ones(
                    input.shape[0], dtype=torch.float32, device=device)
                base_lora_counts[equal_drm] += 1.0
                base_lora_counts[filtered_index_tensor] += 1.0
                logits, prompt_id, fallback_mask, lora_counts = budgeted_exhaustive_fallback(
                    model=model,
                    inputs=input,
                    tii_logits=old_logits,
                    base_logits=logits,
                    base_tasks=prompt_id,
                    base_lora_counts=base_lora_counts,
                    class_mask=class_mask,
                    seen_task_count=task_id + 1,
                    args=args,
                )
                filtered_index_tensor = torch.empty(
                    0, dtype=torch.long, device=device)
                re_id = None
                metric_logger.meters['FallbackRate'].update(
                    fallback_mask.float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['LoRA/sample'].update(
                    lora_counts.mean().item(), n=input.shape[0])


            elif replay_task_router is not None:
                prompt_id = replay_task_router.predict(shared_features, old_logits)
                logits = model(input, task_id=prompt_id)['logits']
                if args.train_mask and class_mask is not None:
                    seen_mask = []
                    for seen_task in range(task_id + 1):
                        seen_mask.extend(class_mask[seen_task])
                    unseen = np.setdiff1d(np.arange(args.nb_classes), seen_mask)
                    unseen = torch.as_tensor(unseen, dtype=torch.long, device=device)
                    logits = logits.index_fill(1, unseen, float('-inf'))
                filtered_index_tensor = torch.empty(0, dtype=torch.long, device=device)
                re_id = None
            elif shared_prototype_bank is not None:
                logits, prompt_id = shared_space_prototype_routing(
                    model,
                    input,
                    shared_features,
                    old_logits,
                    class_mask,
                    task_id + 1,
                    shared_prototype_bank,
                    args,
                )
                filtered_index_tensor = torch.empty(0, dtype=torch.long, device=device)
                re_id = None
            elif prototype_bank is not None:
                logits, prompt_id = prototype_assisted_rematching(
                    model,
                    input,
                    old_logits,
                    class_mask,
                    task_id + 1,
                    prototype_bank,
                    args,
                )
                filtered_index_tensor = torch.empty(0, dtype=torch.long, device=device)
                re_id = None
            else:
                metric_logger.meters['LoRA/sample'].update(
                    default_lora_counts.mean().item(), n=input.shape[0])
                metric_logger.meters['ForwardCalls/sample'].update(
                    default_lora_counts.mean().item(), n=input.shape[0])

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            task_inference_acc = utils.task_inference_accuracy(prompt_id.unsqueeze(-1), target, target_task_map, filtered_index_tensor,re_id)

            class_weight = float(getattr(args, 'rp_class_fusion_weight', 0.0))
            # The RP head needs enough classes before its Gram estimate is
            # reliable, so the blend can be held back for the first tasks.
            #
            # Measured over 12 (dataset, seed) cells, 2026-08-28: the harm is
            # confined to stage 1, where withholding the blend gains +0.1 to
            # +1.5 and never loses. By stage 2 the blend is already net
            # positive -- withholding it there costs -0.3 to -0.4 on CIFAR-100
            # and -1.1 to -1.5 on CUB-200. Stages 3+ are bit-identical either
            # way, and so is the final row.
            #
            # min_tasks=3 is therefore WORSE than the default (AIA@1 down in 9
            # of 12 cells). min_tasks=2 is a Pareto gain but only +0.04 AIA@1 on
            # average, below the noise floor of everything we report, so the
            # shipped default stays 1.
            if task_id + 1 < int(getattr(args, 'rp_class_fusion_min_tasks', 1)):
                class_weight = 0.0
            if class_weight != 0.0 and fusion_rp_scores is None:
                # Fail loudly. fusion_rp_scores only exists on the
                # --rp_route_fusion_drm path, so asking for class fusion
                # without it used to skip stage 2 in silence -- and a run that
                # silently measured the baseline would look like a measurement
                # of the blend.
                raise RuntimeError(
                    '--rp_class_fusion_weight is %.3f but the RP scores were '
                    'never computed; class fusion requires '
                    '--rp_route_fusion_drm. Refusing to skip stage 2 silently.'
                    % class_weight)
            if class_weight != 0.0 and fusion_rp_scores is not None:
                args_ref[0] = args
                logits = fuse_class_scores(
                    logits, fusion_rp_scores, class_weight, task_id + 1)
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                # GateShare is the effective per-sample weight the RP head
                # actually receives, beta times the gate -- not the nominal
                # beta. Reading it next to Acc@1 tells a mode that lost because
                # it mixed too little apart from one that lost because mixing
                # more was the wrong thing to do.
                if 'gate' in gate_stats:
                    metric_logger.meters['GateShare'].update(
                        class_weight * gate_stats['gate'], n=input.shape[0])
                if 'conf_rp' in gate_stats:
                    metric_logger.meters['ConfRP'].update(
                        gate_stats['conf_rp'], n=input.shape[0])
                    metric_logger.meters['Undecided'].update(
                        gate_stats['undecided'], n=input.shape[0])
            if (bool(getattr(args, 'classifier_union_audit', False))
                    and fusion_rp_scores is not None):
                # Routing gains convert poorly into Acc@1 because the shared
                # classifier often gets the class right despite a wrong route.
                # This measures whether the RP head, used as a classifier in its
                # own right, is correct where the routed head is wrong.
                routed_ok = logits.argmax(dim=1).eq(target)
                rp_ok = fusion_rp_scores.argmax(dim=1).eq(target)
                metric_logger.meters['ClsRouted'].update(
                    routed_ok.float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['ClsRP'].update(
                    rp_ok.float().mean().mul(100.0).item(), n=input.shape[0])
                metric_logger.meters['ClsUnion'].update(
                    (routed_ok | rp_ok).float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['ClsRPOnly'].update(
                    (rp_ok & ~routed_ok).float().mean().mul(100.0).item(),
                    n=input.shape[0])
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))
    if 'GateShare' in metric_logger.meters:
        line = '* GateShare {g.global_avg:.4f}'.format(
            g=metric_logger.meters['GateShare'])
        if 'ConfRP' in metric_logger.meters:
            line += (' Undecided {u.global_avg:.4f} ConfRP {c.global_avg:.4f}'
                     .format(u=metric_logger.meters['Undecided'],
                             c=metric_logger.meters['ConfRP']))
        print(line)
    if bool(getattr(args, 'budgeted_rematching', False)):
        print(
            '* Budgeted FallbackRate {rate.global_avg:.3f} LoRA/sample {cost.global_avg:.3f}'
            .format(rate=metric_logger.meters['FallbackRate'],
                    cost=metric_logger.meters['LoRA/sample']))
    if bool(getattr(args, 'selective_rematching', False)):
        print(
            '* Selective CandidateRecall {recall.global_avg:.3f} LoRA/sample {cost.global_avg:.3f}'
            .format(recall=metric_logger.meters['CandidateRecall'],
                    cost=metric_logger.meters['LoRA/sample']))
    if bool(getattr(args, 'progressive_rematching', False)):
        print(
            '* Progressive Stage1Stop {stage1.global_avg:.3f} Stage2Stop {stage2.global_avg:.3f} '
            'FullFallback {fallback.global_avg:.3f} LoRA/sample {cost.global_avg:.3f}'
            .format(
                stage1=metric_logger.meters['Stage1StopRate'],
                stage2=metric_logger.meters['Stage2StopRate'],
                fallback=metric_logger.meters['FullFallbackRate'],
                cost=metric_logger.meters['LoRA/sample']))
    if bool(getattr(args, 'progressive_oracle_audit', False)):
        print(
            '* OracleAudit WinnerRecall@2 {winner2.global_avg:.3f} '
            'WinnerRecall@4 {winner4.global_avg:.3f} '
            'ExactAgreement@2 {agree2.global_avg:.3f} '
            'ExactAgreement@4 {agree4.global_avg:.3f} '
            'OracleLoRA/sample {oracle_cost.global_avg:.3f} '
            'ActualLoRA/sample {actual_cost.global_avg:.3f}'
            .format(
                winner2=metric_logger.meters['WinnerRecall@2'],
                winner4=metric_logger.meters['WinnerRecall@4'],
                agree2=metric_logger.meters['ExactAgreement@2'],
                agree4=metric_logger.meters['ExactAgreement@4'],
                oracle_cost=metric_logger.meters['OracleLoRA/sample'],
                actual_cost=metric_logger.meters['ActualLoRA/sample']))
    if bool(getattr(args, 'stage_drift_audit', False)):
        print(
            '* StageDriftAudit OwnLocalAcc@1 {local_acc.global_avg:.3f} '
            'OwnSeenAcc@1 {seen_acc.global_avg:.3f} '
            'OwnLocalLoss {local_loss.global_avg:.4f} '
            'OwnSeenLoss {seen_loss.global_avg:.4f} '
            'OwnSeenTaskAcc {task_acc.global_avg:.3f} '
            'LocalToSeenFailure {failure.global_avg:.3f}'
            .format(
                local_acc=metric_logger.meters['OwnLocalAcc@1'],
                seen_acc=metric_logger.meters['OwnSeenAcc@1'],
                local_loss=metric_logger.meters['OwnLocalLoss'],
                seen_loss=metric_logger.meters['OwnSeenLoss'],
                task_acc=metric_logger.meters['OwnSeenTaskAcc'],
                failure=metric_logger.meters['LocalToSeenFailure']))
    if bool(getattr(
            args, 'progressive_prediction_self_owner_audit', False)):
        print(
            '* PredictionSelfOwnerAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} RouteRate '
            '{route.global_avg:.3f} MultipleRate '
            '{multiple.global_avg:.3f} Support '
            '{support.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['SelfOwnerWinnerRecall'],
                agreement=metric_logger.meters['SelfOwnerExactAgreement'],
                route=metric_logger.meters['SelfOwnerRouteRate'],
                multiple=metric_logger.meters['SelfOwnerMultipleRate'],
                support=metric_logger.meters['SelfOwnerSupport'],
                cost=metric_logger.meters['SelfOwnerLoRA/sample'],
                calls=metric_logger.meters['SelfOwnerCalls/sample']))
    if bool(getattr(
            args, 'progressive_prediction_corroborated_owner_audit',
            False)):
        print(
            '* PredictionCorroboratedOwnerAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} RescueRate '
            '{rescue.global_avg:.3f} Support '
            '{support.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters[
                    'CorroboratedOwnerWinnerRecall'],
                agreement=metric_logger.meters[
                    'CorroboratedOwnerExactAgreement'],
                rescue=metric_logger.meters[
                    'CorroboratedOwnerRescueRate'],
                support=metric_logger.meters[
                    'CorroboratedOwnerSupport'],
                cost=metric_logger.meters[
                    'CorroboratedOwnerLoRA/sample'],
                calls=metric_logger.meters[
                    'CorroboratedOwnerCalls/sample']))
    if bool(getattr(
            args, 'progressive_prediction_owner_mass_audit', False)):
        print(
            '* PredictionOwnerMassAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} RescueRate '
            '{rescue.global_avg:.3f} AmbiguousRate '
            '{ambiguous.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['OwnerMassWinnerRecall'],
                agreement=metric_logger.meters['OwnerMassExactAgreement'],
                rescue=metric_logger.meters['OwnerMassRescueRate'],
                ambiguous=metric_logger.meters['OwnerMassAmbiguousRate'],
                cost=metric_logger.meters['OwnerMassLoRA/sample'],
                calls=metric_logger.meters['OwnerMassCalls/sample']))
    if bool(getattr(
            args, 'progressive_prediction_owner_consensus_hedge_audit',
            False)):
        print(
            '* PredictionOwnerHedgeAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} HedgeRate '
            '{rate.global_avg:.3f} OwnerAgreement '
            '{owner.global_avg:.3f} ConsensusAgreement '
            '{consensus.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['OwnerHedgeWinnerRecall'],
                agreement=metric_logger.meters['OwnerHedgeExactAgreement'],
                rate=metric_logger.meters['OwnerHedgeRate'],
                owner=metric_logger.meters['OwnerHedgeOwnerAgreement'],
                consensus=metric_logger.meters[
                    'OwnerHedgeConsensusAgreement'],
                cost=metric_logger.meters['OwnerHedgeLoRA/sample'],
                calls=metric_logger.meters['OwnerHedgeCalls/sample']))
    if bool(getattr(args, 'progressive_arrow_audit', False)):
        print(
            '* ArrowAudit ArrowRecall@2 {arrow2.global_avg:.3f} '
            'ArrowRecall@4 {arrow4.global_avg:.3f} '
            'UnionRecall@2x2 {union.global_avg:.3f} '
            'UnionLoRA/sample {cost.global_avg:.3f} '
            'TIIArrowTop1Agree {agreement.global_avg:.3f}'
            .format(
                arrow2=metric_logger.meters['ArrowRecall@2'],
                arrow4=metric_logger.meters['ArrowRecall@4'],
                union=metric_logger.meters['UnionRecall@2x2'],
                cost=metric_logger.meters['UnionLoRA/sample'],
                agreement=metric_logger.meters['TIIArrowTop1Agree']))
    if bool(getattr(args, 'progressive_lora_response_audit', False)):
        print(
            '* LoRAResponseAudit ResponseRecall@2 {response2.global_avg:.3f} '
            'ResponseRecall@4 {response4.global_avg:.3f} '
            'ResponseUnionRecall@2x2 {union.global_avg:.3f} '
            'ResponseUnionLoRA/sample {cost.global_avg:.3f} '
            'TIIResponseTop1Agree {agreement.global_avg:.3f}'
            .format(
                response2=metric_logger.meters['ResponseRecall@2'],
                response4=metric_logger.meters['ResponseRecall@4'],
                union=metric_logger.meters['ResponseUnionRecall@2x2'],
                cost=metric_logger.meters['ResponseUnionLoRA/sample'],
                agreement=metric_logger.meters['TIIResponseTop1Agree']))
    if bool(getattr(args, 'progressive_prediction_proposal_audit', False)):
        print(
            '* PredictionProposalAudit WinnerRecall {recall.global_avg:.3f} '
            'ExactAgreement {agreement.global_avg:.3f} '
            'NewWinner {new_winner.global_avg:.3f} '
            'LoRA/sample {cost.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['ProposalWinnerRecall'],
                agreement=metric_logger.meters['ProposalExactAgreement'],
                new_winner=metric_logger.meters['ProposalNewWinner'],
                cost=metric_logger.meters['ProposalLoRA/sample']))
    if bool(getattr(args, 'progressive_prediction_closure_audit', False)):
        print(
            '* PredictionClosureAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} Top5Coverage '
            '{top5.global_avg:.3f} FullScanRate '
            '{full_scan.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['ClosureWinnerRecall'],
                agreement=metric_logger.meters['ClosureExactAgreement'],
                top5=metric_logger.meters['ClosureTop5Coverage'],
                full_scan=metric_logger.meters['ClosureFullScanRate'],
                cost=metric_logger.meters['ClosureLoRA/sample'],
                calls=metric_logger.meters['ClosureCalls/sample']))
    if bool(getattr(
            args, 'progressive_prediction_beam_closure_audit', False)):
        print(
            '* PredictionBeamClosureAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} FullScanRate '
            '{full_scan.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['BeamClosureWinnerRecall'],
                agreement=metric_logger.meters['BeamClosureExactAgreement'],
                full_scan=metric_logger.meters['BeamClosureFullScanRate'],
                cost=metric_logger.meters['BeamClosureLoRA/sample'],
                calls=metric_logger.meters['BeamClosureCalls/sample']))
    if bool(getattr(
            args, 'progressive_prediction_budget_closure_audit', False)):
        print(
            '* PredictionBudgetClosureAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} BudgetHitRate '
            '{budget_hit.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['BudgetClosureWinnerRecall'],
                agreement=metric_logger.meters['BudgetClosureExactAgreement'],
                budget_hit=metric_logger.meters[
                    'BudgetClosureBudgetHitRate'],
                cost=metric_logger.meters['BudgetClosureLoRA/sample'],
                calls=metric_logger.meters['BudgetClosureCalls/sample']))
    if bool(getattr(
            args, 'progressive_prediction_majority_closure_audit', False)):
        print(
            '* PredictionMajorityClosureAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} CertifiedRate '
            '{certified.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['MajorityClosureWinnerRecall'],
                agreement=metric_logger.meters[
                    'MajorityClosureExactAgreement'],
                certified=metric_logger.meters[
                    'MajorityClosureCertifiedRate'],
                cost=metric_logger.meters['MajorityClosureLoRA/sample'],
                calls=metric_logger.meters['MajorityClosureCalls/sample']))
    if bool(getattr(
            args, 'progressive_prediction_consensus_closure_audit', False)):
        print(
            '* PredictionConsensusClosureAudit WinnerRecall '
            '{recall.global_avg:.3f} ExactAgreement '
            '{agreement.global_avg:.3f} CertifiedRate '
            '{certified.global_avg:.3f} LoRA/sample '
            '{cost.global_avg:.3f} ForwardCalls/sample '
            '{calls.global_avg:.3f}'
            .format(
                recall=metric_logger.meters['ConsensusClosureWinnerRecall'],
                agreement=metric_logger.meters[
                    'ConsensusClosureExactAgreement'],
                certified=metric_logger.meters[
                    'ConsensusClosureCertifiedRate'],
                cost=metric_logger.meters['ConsensusClosureLoRA/sample'],
                calls=metric_logger.meters['ConsensusClosureCalls/sample']))
    if bool(getattr(args, 'prediction_closure_rematching', False)):
        print(
            '* PredictionClosureTail LoRA/sample {cost.global_avg:.3f} '
            'ForwardCalls/sample {calls.global_avg:.3f}'
            .format(cost=metric_logger.meters['LoRA/sample'],
                    calls=metric_logger.meters['ForwardCalls/sample']))
    if bool(getattr(args, 'prediction_proposal_rematching', False)):
        print(
            '* PredictionProposal LoRA/sample {cost.global_avg:.3f} '
            'ForwardCalls/sample {calls.global_avg:.3f}'
            .format(cost=metric_logger.meters['LoRA/sample'],
                    calls=metric_logger.meters['ForwardCalls/sample']))
        if bool(getattr(
                args, 'prediction_proposal_initial_branch_audit', False)):
            print(
                '* InitialProposalAudit InitialBranchAcc@1 '
                '{initial.global_avg:.3f} ProposalAcc@1 '
                '{proposal.global_avg:.3f} Agree {agree.global_avg:.3f} '
                'InitialOnly {initial_only.global_avg:.3f} '
                'ProposalOnly {proposal_only.global_avg:.3f} '
                'OracleAcc@1 {oracle.global_avg:.3f} '
                'DominanceAcc@1 {dominance.global_avg:.3f} '
                'InitialSelectRate {select_rate.global_avg:.3f}'
                .format(
                    initial=metric_logger.meters['InitialBranchAcc@1'],
                    proposal=metric_logger.meters['ProposalAuditAcc@1'],
                    agree=metric_logger.meters['InitialProposalAgree'],
                    initial_only=metric_logger.meters[
                        'InitialOnlyCorrect'],
                    proposal_only=metric_logger.meters['ProposalOnlyCorrect'],
                    oracle=metric_logger.meters[
                        'InitialProposalOracleAcc@1'],
                    dominance=metric_logger.meters['DominanceAcc@1'],
                    select_rate=metric_logger.meters['InitialSelectRate']))
    if bool(getattr(args, 'prediction_proposal_cross_adapter_audit', False)):
        print(
            '* CrossAdapterAudit VoteAcc@1 {vote.global_avg:.3f} '
            'AdapterOracleAcc@1 {adapter_oracle.global_avg:.3f} '
            'VoteStrength {strength.global_avg:.3f} '
            'VoteOnly {vote_only.global_avg:.3f} '
            'ProposalOnly {proposal_only.global_avg:.3f} '
            'ProposalVoteOracleAcc@1 {proposal_oracle.global_avg:.3f} '
            'RescueAcc@1 {rescue.global_avg:.3f} '
            'RescueRate {rescue_rate.global_avg:.3f}'
            .format(
                vote=metric_logger.meters['CrossVoteAcc@1'],
                adapter_oracle=metric_logger.meters[
                    'CrossAdapterOracleAcc@1'],
                strength=metric_logger.meters['CrossVoteStrength'],
                vote_only=metric_logger.meters['CrossVoteOnlyCorrect'],
                proposal_only=metric_logger.meters[
                    'ProposalOnlyVsCrossVote'],
                proposal_oracle=metric_logger.meters[
                    'CrossProposalOracleAcc@1'],
                rescue=metric_logger.meters['CrossRescueAcc@1'],
                rescue_rate=metric_logger.meters['CrossRescueRate']))
        print(
            '* CrossBordaAudit BordaAcc@1 {borda.global_avg:.3f} '
            'BordaOnly {borda_only.global_avg:.3f} '
            'ProposalOnly {proposal_only.global_avg:.3f} '
            'ProposalBordaOracleAcc@1 {oracle.global_avg:.3f} '
            'BordaRescueAcc@1 {rescue.global_avg:.3f} '
            'BordaRescueRate {rate.global_avg:.3f} '
            'BordaTop5Support {support.global_avg:.3f}'
            .format(
                borda=metric_logger.meters['CrossBordaAcc@1'],
                borda_only=metric_logger.meters[
                    'CrossBordaOnlyCorrect'],
                proposal_only=metric_logger.meters[
                    'ProposalOnlyVsCrossBorda'],
                oracle=metric_logger.meters[
                    'CrossBordaProposalOracleAcc@1'],
                rescue=metric_logger.meters['CrossBordaRescueAcc@1'],
                rate=metric_logger.meters['CrossBordaRescueRate'],
                support=metric_logger.meters['CrossBordaTop5Support']))
    if bool(getattr(args, 'vectorized_exhaustive_rematching', False)):
        print(
            '* VectorizedExhaustive LoRA/sample {cost.global_avg:.3f} '
            'ForwardCalls/sample {calls.global_avg:.3f}'
            .format(cost=metric_logger.meters['LoRA/sample'],
                    calls=metric_logger.meters['ForwardCalls/sample']))
    if (bool(getattr(args, 'soft_mixture_rematching', False))
            or bool(getattr(args, 'soft_mixture_hard_rematching', False))
            or bool(getattr(args, 'soft_hard_selector_rematching', False))
            or bool(getattr(args, 'soft_local_hard_refinement', False))):
        if bool(getattr(args, 'soft_local_hard_refinement', False)):
            mixture_name = 'SoftLocalHardRefinement'
        elif bool(getattr(args, 'soft_hard_selector_rematching', False)):
            mixture_name = 'SoftHardSelector'
        elif bool(getattr(args, 'soft_mixture_hard_rematching', False)):
            mixture_name = 'SoftHard'
        else:
            mixture_name = 'SoftMixture'
        print(
            '* {name} LoRA/sample {cost.global_avg:.3f} '
            'ForwardCalls/sample {calls.global_avg:.3f}'
            .format(name=mixture_name, cost=metric_logger.meters['LoRA/sample'],
                    calls=metric_logger.meters['ForwardCalls/sample']))
        if bool(getattr(args, 'soft_hard_selector_rematching', False)):
            print(
                '* SelectorAudit SoftAcc@1 {soft.global_avg:.3f} '
                'HardAcc@1 {hard.global_avg:.3f} '
                'Agree {agree.global_avg:.3f} '
                'SoftOnly {soft_only.global_avg:.3f} '
                'HardOnly {hard_only.global_avg:.3f} '
                'OracleAcc@1 {oracle.global_avg:.3f} '
                'HardSelectRate {select.global_avg:.3f}'
                .format(
                    soft=metric_logger.meters['SoftAcc@1'],
                    hard=metric_logger.meters['HardAcc@1'],
                    agree=metric_logger.meters['SoftHardAgree'],
                    soft_only=metric_logger.meters['SoftOnlyCorrect'],
                    hard_only=metric_logger.meters['HardOnlyCorrect'],
                    oracle=metric_logger.meters['OracleAcc@1'],
                    select=metric_logger.meters['HardSelectRate']))
        if bool(getattr(args, 'soft_local_hard_refinement', False)):
            print(
                '* RefinementAudit SoftAcc@1 {soft.global_avg:.3f} '
                'RefinedAcc@1 {refined.global_avg:.3f} '
                'Agree {agree.global_avg:.3f} '
                'SoftOnly {soft_only.global_avg:.3f} '
                'RefineOnly {refine_only.global_avg:.3f} '
                'OracleAcc@1 {oracle.global_avg:.3f}'
                .format(
                    soft=metric_logger.meters['RefineSoftAcc@1'],
                    refined=metric_logger.meters['RefinedAcc@1'],
                    agree=metric_logger.meters['SoftRefineAgree'],
                    soft_only=metric_logger.meters['RefineSoftOnlyCorrect'],
                    refine_only=metric_logger.meters['RefineOnlyCorrect'],
                    oracle=metric_logger.meters['RefineOracleAcc@1']))
    if bool(getattr(args, 'calibrated_progressive_rematching', False)):
        print(
            '* CalibratedProgressive Stage1Stop {stage1.global_avg:.3f} '
            'Stage2Stop {stage2.global_avg:.3f} FullFallback {fallback.global_avg:.3f} '
            'LoRA/sample {cost.global_avg:.3f} ForwardCalls/sample {calls.global_avg:.3f}'
            .format(
                stage1=metric_logger.meters['Stage1StopRate'],
                stage2=metric_logger.meters['Stage2StopRate'],
                fallback=metric_logger.meters['FullFallbackRate'],
                cost=metric_logger.meters['LoRA/sample'],
                calls=metric_logger.meters['ForwardCalls/sample']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    global con_num, incon_num ,con_all, incon_all
    
    stat_matrix = np.zeros((160, args.num_tasks))

    for i in range(task_id + 1):
        con_num=0
        incon_num=0
        con_all = 0
        incon_all=0
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']
        stat_matrix[4, i] = test_stats.get('CandidateRecall', 0.0)
        stat_matrix[5, i] = test_stats.get('LoRA/sample', 0.0)
        stat_matrix[6, i] = test_stats.get('FallbackRate', 0.0)
        if bool(getattr(args, 'budgeted_rematching', False)):
            stat_matrix[7, i] = test_stats.get('LoRA/sample', 0.0)
        if bool(getattr(args, 'progressive_rematching', False)):
            stat_matrix[7, i] = test_stats.get('Stage1StopRate', 0.0)
            stat_matrix[8, i] = test_stats.get('Stage2StopRate', 0.0)
            stat_matrix[9, i] = test_stats.get('FullFallbackRate', 0.0)
            stat_matrix[10, i] = test_stats.get('LoRA/sample', 0.0)
        if bool(getattr(args, 'progressive_oracle_audit', False)):
            stat_matrix[11, i] = test_stats.get('WinnerRecall@2', 0.0)
            stat_matrix[12, i] = test_stats.get('WinnerRecall@4', 0.0)
            stat_matrix[13, i] = test_stats.get('ExactAgreement@2', 0.0)
            stat_matrix[14, i] = test_stats.get('ExactAgreement@4', 0.0)
            stat_matrix[15, i] = test_stats.get('OracleLoRA/sample', 0.0)
            stat_matrix[16, i] = test_stats.get('ActualLoRA/sample', 0.0)
        if bool(getattr(args, 'rp_route_audit', False)):
            for offset, name in enumerate((
                    'RouteTII', 'RouteRP', 'RouteUnion', 'RouteBoth',
                    'RouteAgree', 'RouteRPOnly')):
                stat_matrix[124 + offset, i] = test_stats.get(name, 0.0)
        if bool(getattr(args, 'layer_stat_router', False)):
            # 130-149 belong to router_recall_audit, so keep clear of them.
            for offset, name in enumerate(
                    ('RouteLS', 'RouteUnion3', 'RouteLSOnly')):
                stat_matrix[152 + offset, i] = test_stats.get(name, 0.0)
        if bool(getattr(args, 'classifier_union_audit', False)):
            for offset, name in enumerate(
                    ('ClsRouted', 'ClsRP', 'ClsUnion', 'ClsRPOnly')):
                stat_matrix[155 + offset, i] = test_stats.get(name, 0.0)
        if bool(getattr(args, 'report_conventional_cost', False)):
            stat_matrix[150, i] = test_stats.get('LoRA/sample', 0.0)
            stat_matrix[151, i] = test_stats.get('ForwardCalls/sample', 0.0)
        if bool(getattr(args, 'router_recall_audit', False)):
            router_base = {'max': 130, 'energy': 135, 'margin': 140,
                           'mean': 145}
            for router_name, base in router_base.items():
                stat_matrix[base, i] = test_stats.get(
                    'Router_{}_MeanRank'.format(router_name), 0.0)
                for offset, recall_k in enumerate((1, 2, 3, 4), start=1):
                    stat_matrix[base + offset, i] = test_stats.get(
                        'Router_{}_Recall@{}'.format(router_name, recall_k),
                        0.0)
        if bool(getattr(args, 'stage_drift_audit', False)):
            stat_matrix[98, i] = test_stats.get('OwnLocalAcc@1', 0.0)
            stat_matrix[99, i] = test_stats.get('OwnSeenAcc@1', 0.0)
            stat_matrix[100, i] = test_stats.get('OwnLocalLoss', 0.0)
            stat_matrix[101, i] = test_stats.get('OwnSeenLoss', 0.0)
            stat_matrix[102, i] = test_stats.get('OwnSeenTaskAcc', 0.0)
            stat_matrix[103, i] = test_stats.get(
                'LocalToSeenFailure', 0.0)
        if bool(getattr(args, 'progressive_arrow_audit', False)):
            stat_matrix[18, i] = test_stats.get('ArrowRecall@2', 0.0)
            stat_matrix[19, i] = test_stats.get('ArrowRecall@4', 0.0)
            stat_matrix[20, i] = test_stats.get('UnionRecall@2x2', 0.0)
            stat_matrix[21, i] = test_stats.get('UnionLoRA/sample', 0.0)
            stat_matrix[22, i] = test_stats.get('TIIArrowTop1Agree', 0.0)
        if bool(getattr(args, 'progressive_lora_response_audit', False)):
            stat_matrix[23, i] = test_stats.get('ResponseRecall@2', 0.0)
            stat_matrix[24, i] = test_stats.get('ResponseRecall@4', 0.0)
            stat_matrix[25, i] = test_stats.get('ResponseUnionRecall@2x2', 0.0)
            stat_matrix[26, i] = test_stats.get('ResponseUnionLoRA/sample', 0.0)
            stat_matrix[27, i] = test_stats.get('TIIResponseTop1Agree', 0.0)
        if bool(getattr(args, 'progressive_prediction_proposal_audit', False)):
            stat_matrix[45, i] = test_stats.get('ProposalWinnerRecall', 0.0)
            stat_matrix[46, i] = test_stats.get('ProposalExactAgreement', 0.0)
            stat_matrix[47, i] = test_stats.get('ProposalLoRA/sample', 0.0)
            stat_matrix[48, i] = test_stats.get('ProposalNewWinner', 0.0)
        if bool(getattr(args, 'progressive_prediction_closure_audit', False)):
            stat_matrix[72, i] = test_stats.get('ClosureWinnerRecall', 0.0)
            stat_matrix[73, i] = test_stats.get('ClosureExactAgreement', 0.0)
            stat_matrix[74, i] = test_stats.get('ClosureTop5Coverage', 0.0)
            stat_matrix[75, i] = test_stats.get('ClosureFullScanRate', 0.0)
            stat_matrix[76, i] = test_stats.get('ClosureLoRA/sample', 0.0)
            stat_matrix[77, i] = test_stats.get(
                'ClosureCalls/sample', 0.0)
        if bool(getattr(
                args, 'progressive_prediction_beam_closure_audit', False)):
            stat_matrix[78, i] = test_stats.get(
                'BeamClosureWinnerRecall', 0.0)
            stat_matrix[79, i] = test_stats.get(
                'BeamClosureExactAgreement', 0.0)
            stat_matrix[80, i] = test_stats.get(
                'BeamClosureFullScanRate', 0.0)
            stat_matrix[81, i] = test_stats.get(
                'BeamClosureLoRA/sample', 0.0)
            stat_matrix[82, i] = test_stats.get(
                'BeamClosureCalls/sample', 0.0)
        if bool(getattr(
                args, 'progressive_prediction_budget_closure_audit', False)):
            stat_matrix[83, i] = test_stats.get(
                'BudgetClosureWinnerRecall', 0.0)
            stat_matrix[84, i] = test_stats.get(
                'BudgetClosureExactAgreement', 0.0)
            stat_matrix[85, i] = test_stats.get(
                'BudgetClosureBudgetHitRate', 0.0)
            stat_matrix[86, i] = test_stats.get(
                'BudgetClosureLoRA/sample', 0.0)
            stat_matrix[87, i] = test_stats.get(
                'BudgetClosureCalls/sample', 0.0)
        if bool(getattr(
                args, 'progressive_prediction_majority_closure_audit', False)):
            stat_matrix[88, i] = test_stats.get(
                'MajorityClosureWinnerRecall', 0.0)
            stat_matrix[89, i] = test_stats.get(
                'MajorityClosureExactAgreement', 0.0)
            stat_matrix[90, i] = test_stats.get(
                'MajorityClosureCertifiedRate', 0.0)
            stat_matrix[91, i] = test_stats.get(
                'MajorityClosureLoRA/sample', 0.0)
            stat_matrix[92, i] = test_stats.get(
                'MajorityClosureCalls/sample', 0.0)
        if bool(getattr(
                args, 'progressive_prediction_consensus_closure_audit', False)):
            stat_matrix[93, i] = test_stats.get(
                'ConsensusClosureWinnerRecall', 0.0)
            stat_matrix[94, i] = test_stats.get(
                'ConsensusClosureExactAgreement', 0.0)
            stat_matrix[95, i] = test_stats.get(
                'ConsensusClosureCertifiedRate', 0.0)
            stat_matrix[96, i] = test_stats.get(
                'ConsensusClosureLoRA/sample', 0.0)
            stat_matrix[97, i] = test_stats.get(
                'ConsensusClosureCalls/sample', 0.0)
        if bool(getattr(
                args, 'progressive_prediction_self_owner_audit', False)):
            stat_matrix[104, i] = test_stats.get(
                'SelfOwnerWinnerRecall', 0.0)
            stat_matrix[105, i] = test_stats.get(
                'SelfOwnerExactAgreement', 0.0)
            stat_matrix[106, i] = test_stats.get(
                'SelfOwnerRouteRate', 0.0)
            stat_matrix[107, i] = test_stats.get(
                'SelfOwnerMultipleRate', 0.0)
            stat_matrix[108, i] = test_stats.get(
                'SelfOwnerSupport', 0.0)
            stat_matrix[109, i] = test_stats.get(
                'SelfOwnerLoRA/sample', 0.0)
            stat_matrix[110, i] = test_stats.get(
                'SelfOwnerCalls/sample', 0.0)
        if bool(getattr(
                args, 'progressive_prediction_corroborated_owner_audit',
                False)):
            stat_matrix[111, i] = test_stats.get(
                'CorroboratedOwnerWinnerRecall', 0.0)
            stat_matrix[112, i] = test_stats.get(
                'CorroboratedOwnerExactAgreement', 0.0)
            stat_matrix[113, i] = test_stats.get(
                'CorroboratedOwnerRescueRate', 0.0)
            stat_matrix[114, i] = test_stats.get(
                'CorroboratedOwnerSupport', 0.0)
            stat_matrix[115, i] = test_stats.get(
                'CorroboratedOwnerLoRA/sample', 0.0)
            stat_matrix[116, i] = test_stats.get(
                'CorroboratedOwnerCalls/sample', 0.0)
        if bool(getattr(
                args, 'progressive_prediction_owner_mass_audit', False)):
            stat_matrix[117, i] = test_stats.get(
                'OwnerMassWinnerRecall', 0.0)
            stat_matrix[118, i] = test_stats.get(
                'OwnerMassExactAgreement', 0.0)
            stat_matrix[119, i] = test_stats.get(
                'OwnerMassRescueRate', 0.0)
            stat_matrix[120, i] = test_stats.get(
                'OwnerMassAmbiguousRate', 0.0)
            stat_matrix[121, i] = test_stats.get(
                'OwnerMassLoRA/sample', 0.0)
            stat_matrix[122, i] = test_stats.get(
                'OwnerMassCalls/sample', 0.0)
        if bool(getattr(
                args, 'progressive_prediction_owner_consensus_hedge_audit',
                False)):
            stat_matrix[123, i] = test_stats.get(
                'OwnerHedgeWinnerRecall', 0.0)
            stat_matrix[124, i] = test_stats.get(
                'OwnerHedgeExactAgreement', 0.0)
            stat_matrix[125, i] = test_stats.get(
                'OwnerHedgeRate', 0.0)
            stat_matrix[126, i] = test_stats.get(
                'OwnerHedgeOwnerAgreement', 0.0)
            stat_matrix[127, i] = test_stats.get(
                'OwnerHedgeConsensusAgreement', 0.0)
            stat_matrix[128, i] = test_stats.get(
                'OwnerHedgeLoRA/sample', 0.0)
            stat_matrix[129, i] = test_stats.get(
                'OwnerHedgeCalls/sample', 0.0)
        if (bool(getattr(args, 'vectorized_exhaustive_rematching', False))
                or bool(getattr(args, 'prediction_proposal_rematching', False))
                or bool(getattr(args, 'prediction_closure_rematching', False))):
            stat_matrix[28, i] = test_stats.get('LoRA/sample', 0.0)
            stat_matrix[29, i] = test_stats.get('ForwardCalls/sample', 0.0)
        if bool(getattr(
                args, 'prediction_proposal_initial_branch_audit', False)):
            stat_matrix[49, i] = test_stats.get('InitialBranchAcc@1', 0.0)
            stat_matrix[50, i] = test_stats.get('ProposalAuditAcc@1', 0.0)
            stat_matrix[51, i] = test_stats.get(
                'InitialProposalAgree', 0.0)
            stat_matrix[52, i] = test_stats.get(
                'InitialOnlyCorrect', 0.0)
            stat_matrix[53, i] = test_stats.get('ProposalOnlyCorrect', 0.0)
            stat_matrix[54, i] = test_stats.get(
                'InitialProposalOracleAcc@1', 0.0)
            stat_matrix[55, i] = test_stats.get('DominanceAcc@1', 0.0)
            stat_matrix[56, i] = test_stats.get('InitialSelectRate', 0.0)
        if bool(getattr(
                args, 'prediction_proposal_cross_adapter_audit', False)):
            stat_matrix[57, i] = test_stats.get('CrossVoteAcc@1', 0.0)
            stat_matrix[58, i] = test_stats.get(
                'CrossAdapterOracleAcc@1', 0.0)
            stat_matrix[59, i] = test_stats.get('CrossVoteStrength', 0.0)
            stat_matrix[60, i] = test_stats.get(
                'CrossVoteOnlyCorrect', 0.0)
            stat_matrix[61, i] = test_stats.get(
                'ProposalOnlyVsCrossVote', 0.0)
            stat_matrix[62, i] = test_stats.get(
                'CrossProposalOracleAcc@1', 0.0)
            stat_matrix[63, i] = test_stats.get('CrossRescueAcc@1', 0.0)
            stat_matrix[64, i] = test_stats.get('CrossRescueRate', 0.0)
            stat_matrix[65, i] = test_stats.get('CrossBordaAcc@1', 0.0)
            stat_matrix[66, i] = test_stats.get(
                'CrossBordaOnlyCorrect', 0.0)
            stat_matrix[67, i] = test_stats.get(
                'ProposalOnlyVsCrossBorda', 0.0)
            stat_matrix[68, i] = test_stats.get(
                'CrossBordaProposalOracleAcc@1', 0.0)
            stat_matrix[69, i] = test_stats.get(
                'CrossBordaRescueAcc@1', 0.0)
            stat_matrix[70, i] = test_stats.get(
                'CrossBordaRescueRate', 0.0)
            stat_matrix[71, i] = test_stats.get(
                'CrossBordaTop5Support', 0.0)
        if (bool(getattr(args, 'soft_mixture_rematching', False))
                or bool(getattr(args, 'soft_mixture_hard_rematching', False))
                or bool(getattr(args, 'soft_hard_selector_rematching', False))
                or bool(getattr(args, 'soft_local_hard_refinement', False))):
            stat_matrix[30, i] = test_stats.get('LoRA/sample', 0.0)
            stat_matrix[31, i] = test_stats.get('ForwardCalls/sample', 0.0)
        if bool(getattr(args, 'soft_hard_selector_rematching', False)):
            stat_matrix[32, i] = test_stats.get('SoftAcc@1', 0.0)
            stat_matrix[33, i] = test_stats.get('HardAcc@1', 0.0)
            stat_matrix[34, i] = test_stats.get('SoftHardAgree', 0.0)
            stat_matrix[35, i] = test_stats.get('SoftOnlyCorrect', 0.0)
            stat_matrix[36, i] = test_stats.get('HardOnlyCorrect', 0.0)
            stat_matrix[37, i] = test_stats.get('OracleAcc@1', 0.0)
            stat_matrix[38, i] = test_stats.get('HardSelectRate', 0.0)
        if bool(getattr(args, 'soft_local_hard_refinement', False)):
            stat_matrix[39, i] = test_stats.get('RefineSoftAcc@1', 0.0)
            stat_matrix[40, i] = test_stats.get('RefinedAcc@1', 0.0)
            stat_matrix[41, i] = test_stats.get('SoftRefineAgree', 0.0)
            stat_matrix[42, i] = test_stats.get('RefineSoftOnlyCorrect', 0.0)
            stat_matrix[43, i] = test_stats.get('RefineOnlyCorrect', 0.0)
            stat_matrix[44, i] = test_stats.get('RefineOracleAcc@1', 0.0)
        if bool(getattr(args, 'calibrated_progressive_rematching', False)):
            stat_matrix[7, i] = test_stats.get('Stage1StopRate', 0.0)
            stat_matrix[8, i] = test_stats.get('Stage2StopRate', 0.0)
            stat_matrix[9, i] = test_stats.get('FullFallbackRate', 0.0)
            stat_matrix[10, i] = test_stats.get('LoRA/sample', 0.0)
            stat_matrix[17, i] = test_stats.get('ForwardCalls/sample', 0.0)

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if bool(getattr(args, 'budgeted_rematching', False)):
        result_str += "\tFallbackRate: {:.4f}\tLoRA/sample: {:.4f}".format(
            avg_stat[6], avg_stat[7])
    if bool(getattr(args, 'selective_rematching', False)):
        result_str += "\tCandidateRecall: {:.4f}\tLoRA/sample: {:.4f}".format(
            avg_stat[4], avg_stat[5])
    if bool(getattr(args, 'progressive_rematching', False)):
        result_str += (
            "\tStage1Stop: {:.4f}\tStage2Stop: {:.4f}"
            "\tFullFallback: {:.4f}\tLoRA/sample: {:.4f}"
        ).format(avg_stat[7], avg_stat[8], avg_stat[9], avg_stat[10])
    if bool(getattr(args, 'progressive_oracle_audit', False)):
        result_str += (
            "\tWinnerRecall@2: {:.4f}\tWinnerRecall@4: {:.4f}"
            "\tExactAgreement@2: {:.4f}\tExactAgreement@4: {:.4f}"
            "\tOracleLoRA/sample: {:.4f}\tActualLoRA/sample: {:.4f}"
        ).format(
            avg_stat[11], avg_stat[12], avg_stat[13], avg_stat[14],
            avg_stat[15], avg_stat[16])
    if bool(getattr(args, 'rp_route_audit', False)):
        result_str += "	RouteTII: {:.4f}	RouteRP: {:.4f}	RouteUnion: {:.4f}	RouteBoth: {:.4f}	RouteAgree: {:.4f}	RouteRPOnly: {:.4f}".format(
            *[np.mean(stat_matrix[124 + k, :task_id + 1]) for k in range(6)])
        if bool(getattr(args, 'layer_stat_router', False)):
            result_str += "	RouteLS: {:.4f}	RouteUnion3: {:.4f}	RouteLSOnly: {:.4f}".format(
                *[np.mean(stat_matrix[152 + k, :task_id + 1]) for k in range(3)])
    if bool(getattr(args, 'classifier_union_audit', False)):
        result_str += "	ClsRouted: {:.4f}	ClsRP: {:.4f}	ClsUnion: {:.4f}	ClsRPOnly: {:.4f}".format(
            *[np.mean(stat_matrix[155 + k, :task_id + 1]) for k in range(4)])
    if bool(getattr(args, 'report_conventional_cost', False)):
        result_str += "\tLoRA/sample: {:.4f}\tForwardCalls/sample: {:.4f}".format(
            avg_stat[150], avg_stat[151])
    if bool(getattr(args, 'router_recall_audit', False)):
        router_base = {'max': 130, 'energy': 135, 'margin': 140, 'mean': 145}
        for router_name, base in router_base.items():
            result_str += (
                "\tRouter_{0}_MeanRank: {1:.4f}"
                "\tRouter_{0}_Recall@1: {2:.4f}\tRouter_{0}_Recall@2: {3:.4f}"
                "\tRouter_{0}_Recall@3: {4:.4f}\tRouter_{0}_Recall@4: {5:.4f}"
            ).format(
                router_name, avg_stat[base], avg_stat[base + 1],
                avg_stat[base + 2], avg_stat[base + 3], avg_stat[base + 4])
    if bool(getattr(args, 'stage_drift_audit', False)):
        result_str += (
            "\tOwnLocalAcc@1: {:.4f}\tOwnSeenAcc@1: {:.4f}"
            "\tOwnLocalLoss: {:.4f}\tOwnSeenLoss: {:.4f}"
            "\tOwnSeenTaskAcc: {:.4f}\tLocalToSeenFailure: {:.4f}"
        ).format(
            avg_stat[98], avg_stat[99], avg_stat[100], avg_stat[101],
            avg_stat[102], avg_stat[103])
    if bool(getattr(args, 'progressive_arrow_audit', False)):
        result_str += (
            "\tArrowRecall@2: {:.4f}\tArrowRecall@4: {:.4f}"
            "\tUnionRecall@2x2: {:.4f}\tUnionLoRA/sample: {:.4f}"
            "\tTIIArrowTop1Agree: {:.4f}"
        ).format(
            avg_stat[18], avg_stat[19], avg_stat[20], avg_stat[21],
            avg_stat[22])
    if bool(getattr(args, 'progressive_lora_response_audit', False)):
        result_str += (
            "\tResponseRecall@2: {:.4f}\tResponseRecall@4: {:.4f}"
            "\tResponseUnionRecall@2x2: {:.4f}"
            "\tResponseUnionLoRA/sample: {:.4f}"
            "\tTIIResponseTop1Agree: {:.4f}"
        ).format(
            avg_stat[23], avg_stat[24], avg_stat[25], avg_stat[26],
            avg_stat[27])
    if bool(getattr(args, 'progressive_prediction_proposal_audit', False)):
        result_str += (
            "\tProposalWinnerRecall: {:.4f}"
            "\tProposalExactAgreement: {:.4f}"
            "\tProposalLoRA/sample: {:.4f}"
            "\tProposalNewWinner: {:.4f}"
        ).format(avg_stat[45], avg_stat[46], avg_stat[47], avg_stat[48])
    if bool(getattr(args, 'progressive_prediction_closure_audit', False)):
        result_str += (
            "\tClosureWinnerRecall: {:.4f}"
            "\tClosureExactAgreement: {:.4f}"
            "\tClosureTop5Coverage: {:.4f}"
            "\tClosureFullScanRate: {:.4f}"
            "\tClosureLoRA/sample: {:.4f}"
            "\tClosureCalls/sample: {:.4f}"
        ).format(
            avg_stat[72], avg_stat[73], avg_stat[74], avg_stat[75],
            avg_stat[76], avg_stat[77])
    if bool(getattr(
            args, 'progressive_prediction_beam_closure_audit', False)):
        result_str += (
            "\tBeamClosureWinnerRecall: {:.4f}"
            "\tBeamClosureExactAgreement: {:.4f}"
            "\tBeamClosureFullScanRate: {:.4f}"
            "\tBeamClosureLoRA/sample: {:.4f}"
            "\tBeamClosureCalls/sample: {:.4f}"
        ).format(
            avg_stat[78], avg_stat[79], avg_stat[80], avg_stat[81],
            avg_stat[82])
    if bool(getattr(
            args, 'progressive_prediction_budget_closure_audit', False)):
        result_str += (
            "\tBudgetClosureWinnerRecall: {:.4f}"
            "\tBudgetClosureExactAgreement: {:.4f}"
            "\tBudgetClosureBudgetHitRate: {:.4f}"
            "\tBudgetClosureLoRA/sample: {:.4f}"
            "\tBudgetClosureCalls/sample: {:.4f}"
        ).format(
            avg_stat[83], avg_stat[84], avg_stat[85], avg_stat[86],
            avg_stat[87])
    if bool(getattr(
            args, 'progressive_prediction_majority_closure_audit', False)):
        result_str += (
            "\tMajorityClosureWinnerRecall: {:.4f}"
            "\tMajorityClosureExactAgreement: {:.4f}"
            "\tMajorityClosureCertifiedRate: {:.4f}"
            "\tMajorityClosureLoRA/sample: {:.4f}"
            "\tMajorityClosureCalls/sample: {:.4f}"
        ).format(
            avg_stat[88], avg_stat[89], avg_stat[90], avg_stat[91],
            avg_stat[92])
    if bool(getattr(
            args, 'progressive_prediction_consensus_closure_audit', False)):
        result_str += (
            "\tConsensusClosureWinnerRecall: {:.4f}"
            "\tConsensusClosureExactAgreement: {:.4f}"
            "\tConsensusClosureCertifiedRate: {:.4f}"
            "\tConsensusClosureLoRA/sample: {:.4f}"
            "\tConsensusClosureCalls/sample: {:.4f}"
        ).format(
            avg_stat[93], avg_stat[94], avg_stat[95], avg_stat[96],
            avg_stat[97])
    if bool(getattr(
            args, 'progressive_prediction_self_owner_audit', False)):
        result_str += (
            "	SelfOwnerWinnerRecall: {:.4f}"
            "	SelfOwnerExactAgreement: {:.4f}"
            "	SelfOwnerRouteRate: {:.4f}"
            "	SelfOwnerMultipleRate: {:.4f}"
            "	SelfOwnerSupport: {:.4f}"
            "	SelfOwnerLoRA/sample: {:.4f}"
            "	SelfOwnerCalls/sample: {:.4f}"
        ).format(
            avg_stat[104], avg_stat[105], avg_stat[106], avg_stat[107],
            avg_stat[108], avg_stat[109], avg_stat[110])
    if bool(getattr(
            args, 'progressive_prediction_corroborated_owner_audit',
            False)):
        result_str += (
            "	CorroboratedOwnerWinnerRecall: {:.4f}"
            "	CorroboratedOwnerExactAgreement: {:.4f}"
            "	CorroboratedOwnerRescueRate: {:.4f}"
            "	CorroboratedOwnerSupport: {:.4f}"
            "	CorroboratedOwnerLoRA/sample: {:.4f}"
            "	CorroboratedOwnerCalls/sample: {:.4f}"
        ).format(
            avg_stat[111], avg_stat[112], avg_stat[113],
            avg_stat[114], avg_stat[115], avg_stat[116])
    if bool(getattr(
            args, 'progressive_prediction_owner_mass_audit', False)):
        result_str += (
            "\tOwnerMassWinnerRecall: {:.4f}"
            "\tOwnerMassExactAgreement: {:.4f}"
            "\tOwnerMassRescueRate: {:.4f}"
            "\tOwnerMassAmbiguousRate: {:.4f}"
            "\tOwnerMassLoRA/sample: {:.4f}"
            "\tOwnerMassCalls/sample: {:.4f}"
        ).format(
            avg_stat[117], avg_stat[118], avg_stat[119],
            avg_stat[120], avg_stat[121], avg_stat[122])
    if bool(getattr(
            args, 'progressive_prediction_owner_consensus_hedge_audit',
            False)):
        result_str += (
            "\tOwnerHedgeWinnerRecall: {:.4f}"
            "\tOwnerHedgeExactAgreement: {:.4f}"
            "\tOwnerHedgeRate: {:.4f}"
            "\tOwnerHedgeOwnerAgreement: {:.4f}"
            "\tOwnerHedgeConsensusAgreement: {:.4f}"
            "\tOwnerHedgeLoRA/sample: {:.4f}"
            "\tOwnerHedgeCalls/sample: {:.4f}"
        ).format(
            avg_stat[123], avg_stat[124], avg_stat[125],
            avg_stat[126], avg_stat[127], avg_stat[128], avg_stat[129])
    if (bool(getattr(args, 'vectorized_exhaustive_rematching', False))
            or bool(getattr(args, 'prediction_proposal_rematching', False))
            or bool(getattr(args, 'prediction_closure_rematching', False))):
        result_str += (
            "\tLoRA/sample: {:.4f}\tForwardCalls/sample: {:.4f}"
        ).format(avg_stat[28], avg_stat[29])
    if bool(getattr(args, 'prediction_proposal_initial_branch_audit', False)):
        result_str += (
            "\tInitialBranchAcc@1: {:.4f}\tProposalAuditAcc@1: {:.4f}"
            "\tInitialProposalAgree: {:.4f}"
            "\tInitialOnlyCorrect: {:.4f}"
            "\tProposalOnlyCorrect: {:.4f}"
            "\tInitialProposalOracleAcc@1: {:.4f}"
            "\tDominanceAcc@1: {:.4f}"
            "\tInitialSelectRate: {:.4f}"
        ).format(
            avg_stat[49], avg_stat[50], avg_stat[51], avg_stat[52],
            avg_stat[53], avg_stat[54], avg_stat[55], avg_stat[56])
    if bool(getattr(args, 'prediction_proposal_cross_adapter_audit', False)):
        result_str += (
            "\tCrossVoteAcc@1: {:.4f}"
            "\tCrossAdapterOracleAcc@1: {:.4f}"
            "\tCrossVoteStrength: {:.4f}"
            "\tCrossVoteOnlyCorrect: {:.4f}"
            "\tProposalOnlyVsCrossVote: {:.4f}"
            "\tCrossProposalOracleAcc@1: {:.4f}"
            "\tCrossRescueAcc@1: {:.4f}"
            "\tCrossRescueRate: {:.4f}"
            "\tCrossBordaAcc@1: {:.4f}"
            "\tCrossBordaOnlyCorrect: {:.4f}"
            "\tProposalOnlyVsCrossBorda: {:.4f}"
            "\tCrossBordaProposalOracleAcc@1: {:.4f}"
            "\tCrossBordaRescueAcc@1: {:.4f}"
            "\tCrossBordaRescueRate: {:.4f}"
            "\tCrossBordaTop5Support: {:.4f}"
        ).format(
            avg_stat[57], avg_stat[58], avg_stat[59], avg_stat[60],
            avg_stat[61], avg_stat[62], avg_stat[63], avg_stat[64],
            avg_stat[65], avg_stat[66], avg_stat[67], avg_stat[68],
            avg_stat[69], avg_stat[70], avg_stat[71])
    if (bool(getattr(args, 'soft_mixture_rematching', False))
            or bool(getattr(args, 'soft_mixture_hard_rematching', False))
            or bool(getattr(args, 'soft_hard_selector_rematching', False))
            or bool(getattr(args, 'soft_local_hard_refinement', False))):
        result_str += (
            "\tLoRA/sample: {:.4f}\tForwardCalls/sample: {:.4f}"
        ).format(avg_stat[30], avg_stat[31])
    if bool(getattr(args, 'soft_hard_selector_rematching', False)):
        result_str += (
            "\tSoftAcc@1: {:.4f}\tHardAcc@1: {:.4f}"
            "\tSoftHardAgree: {:.4f}\tSoftOnlyCorrect: {:.4f}"
            "\tHardOnlyCorrect: {:.4f}\tOracleAcc@1: {:.4f}"
            "\tHardSelectRate: {:.4f}"
        ).format(
            avg_stat[32], avg_stat[33], avg_stat[34], avg_stat[35],
            avg_stat[36], avg_stat[37], avg_stat[38])
    if bool(getattr(args, 'soft_local_hard_refinement', False)):
        result_str += (
            "\tRefineSoftAcc@1: {:.4f}\tRefinedAcc@1: {:.4f}"
            "\tSoftRefineAgree: {:.4f}\tRefineSoftOnlyCorrect: {:.4f}"
            "\tRefineOnlyCorrect: {:.4f}\tRefineOracleAcc@1: {:.4f}"
        ).format(
            avg_stat[39], avg_stat[40], avg_stat[41], avg_stat[42],
            avg_stat[43], avg_stat[44])
    if bool(getattr(args, 'calibrated_progressive_rematching', False)):
        result_str += (
            "\tStage1Stop: {:.4f}\tStage2Stop: {:.4f}"
            "\tFullFallback: {:.4f}\tLoRA/sample: {:.4f}"
            "\tForwardCalls/sample: {:.4f}"
        ).format(
            avg_stat[7], avg_stat[8], avg_stat[9], avg_stat[10], avg_stat[17])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    global cls_cfs_model
    global cls_real_features
    cls_mean = dict()
    cls_cov = dict()
    cls_cfs_model = dict()
    cls_real_features = dict()
    norm_blend_enabled = bool(getattr(args, 'continual_norm_blend', False))
    norm_update_ratio = min(
        1.0, max(0.0, float(getattr(args, 'continual_norm_update_ratio', 0.25))))
    task_logit_calibration_state = None

    task_count = args.num_tasks
    max_train_tasks = int(getattr(args, 'max_train_tasks', 0))
    if max_train_tasks > 0:
        task_count = min(task_count, max_train_tasks)
        if utils.is_main_process():
            print('Limiting run to', task_count, 'of', args.num_tasks, 'tasks')
    for task_id in range(task_count):
        replay_anchor_memory = None
        if task_id > 0 and bool(getattr(args, 'replay_anchor_ctird', False)):
            old_classes = [
                int(class_id)
                for previous_task in range(task_id)
                for class_id in class_mask[previous_task]
            ]
            replay_anchor_memory = build_replay_anchor_memory(args, old_classes)
            if replay_anchor_memory.empty:
                raise RuntimeError(
                    'Replay-Anchored CTIRD is enabled but no cached pseudo-images '
                    'were found for tasks before task {}.'.format(task_id + 1))
        previous_fc_norm = None
        if norm_blend_enabled and task_id > 0:
            previous_fc_norm = {
                name: parameter.detach().clone()
                for name, parameter in model_without_ddp.fc_norm.named_parameters()
            }
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        # load original model checkpoint
        if args.trained_original_model:
            original_checkpoint_path = os.path.join(args.trained_original_model,
                                                    'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = utils.load_checkpoint(original_checkpoint_path, map_location=device)
                original_model.load_state_dict(original_checkpoint['model'], strict=False)
            else:
                print('No checkpoint found at:', original_checkpoint_path)
                return
       
        if task_id > 0:
            with torch.no_grad():
                if args.distributed:
                    model.module.lora_layer.k_lora_A.grad.zero_()
                    model.module.lora_layer.k_lora_A[task_id] = model.module.lora_layer.k_lora_A[task_id-1]
                    model.module.lora_layer.k_lora_B.grad.zero_()
                    model.module.lora_layer.k_lora_B[task_id] = model.module.lora_layer.k_lora_B[task_id-1]
                    model.module.lora_layer.v_lora_A.grad.zero_()
                    model.module.lora_layer.v_lora_A[task_id] = model.module.lora_layer.v_lora_A[task_id-1]
                    model.module.lora_layer.v_lora_B.grad.zero_()
                    model.module.lora_layer.v_lora_B[task_id] = model.module.lora_layer.v_lora_B[task_id-1]
                else:
                    model.lora_layer.k_lora_A.grad.zero_()
                    model.lora_layer.k_lora_A[task_id] = model.lora_layer.k_lora_A[task_id-1]
                    model.lora_layer.k_lora_B.grad.zero_()
                    model.lora_layer.k_lora_B[task_id] = model.lora_layer.k_lora_B[task_id-1]
                    model.lora_layer.v_lora_A.grad.zero_()
                    model.lora_layer.v_lora_A[task_id] = model.module.lora_layer.v_lora_A[task_id-1]
                    model.lora_layer.v_lora_B.grad.zero_()
                    model.lora_layer.v_lora_B[task_id] = model.module.lora_layer.v_lora_B[task_id-1]

        if (task_id > 0
                and not bool(getattr(args, 'ctird_online_aligned', False))):
            old_features = get_old_features(model=model, original_model=original_model, criterion=criterion,
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                            device=device, epoch=0, max_norm=args.clip_grad,
                                            set_training_mode=False, task_id=task_id, class_mask=class_mask,
                                            target_task_map=target_task_map, args=args, )
        else:
            old_features = None

        for epoch in range(args.epochs):
            # model.module.init_weights_proj()
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                            device=device, epoch=epoch, max_norm=args.clip_grad,
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                            target_task_map=target_task_map, args=args, old_features=old_features,
                                            replay_anchor_memory=replay_anchor_memory)

            if lr_scheduler:
                lr_scheduler.step(epoch)
        model_without_ddp.after_task(task_id=task_id, device=device)

        if previous_fc_norm is not None:
            squared_update_before = 0.0
            squared_update_after = 0.0
            with torch.no_grad():
                for name, parameter in model_without_ddp.fc_norm.named_parameters():
                    previous = previous_fc_norm[name].to(parameter.device)
                    update = parameter - previous
                    squared_update_before += update.float().pow(2).sum().item()
                    parameter.mul_(norm_update_ratio).add_(
                        previous, alpha=1.0 - norm_update_ratio)
                    retained_update = parameter - previous
                    squared_update_after += retained_update.float().pow(2).sum().item()
            if utils.is_main_process():
                print(
                    'Continual norm blend:',
                    'update_ratio=', norm_update_ratio,
                    'delta_before=', math.sqrt(squared_update_before),
                    'delta_after=', math.sqrt(squared_update_after),
                )

        if args.lora_momentum > 0 and task_id > 0:
            with torch.no_grad():
                model.module.lora_layer.k_lora_A[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.k_lora_A[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.k_lora_A[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.k_lora_B[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.k_lora_B[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.k_lora_B[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.v_lora_A[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.v_lora_A[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.v_lora_A[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.v_lora_B[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.v_lora_B[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.v_lora_B[0:task_id].detach().clone().mean(dim=0))


        # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)
        if bool(getattr(args, 'crct_real_feature_replay', False)) and utils.is_main_process():
            memory_samples = sum(memory.shape[0] for memory in cls_real_features.values())
            memory_bytes = sum(memory.numel() * memory.element_size() for memory in cls_real_features.values())
            print(
                'Real feature memory:',
                'classes=', len(cls_real_features),
                'samples=', memory_samples,
                'MiB=', round(memory_bytes / (1024.0 * 1024.0), 3),
            )
        if utils.use_semantic_feature_adapter(args):
            seen_classes = []
            for seen_task in range(task_id + 1):
                seen_classes.extend(class_mask[seen_task])
            utils.update_semantic_feature_adapter(args, cls_mean, device, available_classes=seen_classes)


        if task_id > 0 and not args.not_train_ca:
            # The previous checkpoint has no affine entry for the new task yet.
            cfs_calibration_enabled = bool(getattr(
                args, 'cfs_task_logit_calibration', False))
            if cfs_calibration_enabled:
                args.cfs_task_logit_calibration = False
            try:
                pre_ca_test_stats = evaluate_till_now(
                    model=model, original_model=original_model,
                    data_loader=data_loader, device=device,
                    task_id=task_id, class_mask=class_mask,
                    target_task_map=target_task_map,
                    acc_matrix=pre_ca_acc_matrix, args=args)
            finally:
                if cfs_calibration_enabled:
                    args.cfs_task_logit_calibration = True
            #train_dis(model, args, device,data_loader[task_id]['train'], class_mask,target_task_map, task_id)
            train_task_adaptive_prediction(
                model, args, device, class_mask, task_id,
                data_loader_per_cls=data_loader_per_cls)

        if bool(getattr(args, 'cfs_task_logit_calibration', False)):
            task_logit_calibration_state = fit_cfs_task_logit_calibration(
                model=model,
                cls_mean=cls_mean,
                cls_cov=cls_cov,
                cls_cfs_model=cls_cfs_model,
                class_mask=class_mask,
                seen_task_count=task_id + 1,
                args=args,
                device=device,
            )
            args.cfs_task_logit_calibration_state = (
                task_logit_calibration_state)
            if utils.is_main_process():
                print(
                    'CFS task-logit calibration:',
                    'accepted=', task_logit_calibration_state['accepted'],
                    'reason=', task_logit_calibration_state['reason'],
                    'fit/report=',
                    task_logit_calibration_state['fit_samples'],
                    task_logit_calibration_state['report_samples'],
                    'before=', task_logit_calibration_state['before'],
                    'after=', task_logit_calibration_state['after'],
                    'loss_before/after=',
                    task_logit_calibration_state['before_loss'],
                    task_logit_calibration_state['after_loss'],
                    'worst_old_class_drop=',
                    task_logit_calibration_state[
                        'worst_old_class_drop'],
                    'scale=', task_logit_calibration_state['scale'],
                    'bias=', task_logit_calibration_state['bias'],
                )

        if (bool(getattr(args, 'replay_anchor_ctird', False))
                and task_id + 1 < task_count):
            generate_task_replay_cache(
                model=model_without_ddp,
                task_id=task_id,
                class_ids=class_mask[task_id],
                cls_mean=cls_mean,
                cls_cov=cls_cov,
                cls_cfs_model=cls_cfs_model,
                args=args,
                device=device,
            )
        
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            checkpoint_args = args
            if bool(getattr(args, 'cfs_task_logit_calibration', False)):
                checkpoint_args = copy.copy(args)
                if hasattr(
                        checkpoint_args,
                        'cfs_task_logit_calibration_state'):
                    delattr(
                        checkpoint_args,
                        'cfs_task_logit_calibration_state')
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': checkpoint_args,
            }
            if bool(getattr(args, 'crct_real_feature_replay', False)):
                state_dict['real_feature_memory'] = {
                    int(class_id): memory.cpu()
                    for class_id, memory in cls_real_features.items()}
            if bool(getattr(args, 'cfs_task_logit_calibration', False)):
                if task_logit_calibration_state is None:
                    raise RuntimeError(
                        'CFS task-logit calibration state was not fitted')
                state_dict['cfs_task_logit_calibration'] = copy.deepcopy(
                    task_logit_calibration_state)
            if bool(getattr(args, 'replay_anchor_ctird', False)):
                state_dict['replay_anchor_cache'] = {
                    'version': 1,
                    'images_per_class': int(getattr(
                        args, 'replay_anchor_images_per_class', 5)),
                    'classes': [
                        int(class_id)
                        for seen_task in range(task_id + 1)
                        for class_id in class_mask[seen_task]
                    ],
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                    'a') as f:
                f.write(json.dumps(log_stats) + '\n')


@torch.no_grad()
def _select_real_feature_memory(features, args):
    """Keep central but diverse real features without retaining input images."""
    features = features.detach().float()
    budget = max(1, int(getattr(args, 'crct_real_memory_per_class', 48)))
    if features.shape[0] <= budget:
        return features.cpu().half()

    normalized = F.normalize(features, dim=1)
    normalized_center = F.normalize(features.mean(dim=0), dim=0)
    center_distance = 1.0 - normalized.matmul(normalized_center)
    outlier_quantile = min(
        1.0, max(0.0, float(getattr(args, 'crct_real_outlier_quantile', 0.9))))
    distance_limit = torch.quantile(center_distance, outlier_quantile)
    eligible = torch.nonzero(center_distance <= distance_limit, as_tuple=False).flatten()
    if eligible.numel() < budget:
        eligible = torch.argsort(center_distance)[:budget]

    diversity_weight = min(
        1.0, max(0.0, float(getattr(args, 'crct_real_diversity_weight', 0.7))))
    first_index = eligible[torch.argmin(center_distance.index_select(0, eligible))]
    selected = [int(first_index.item())]
    available = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    available[eligible] = True
    available[first_index] = False
    min_distance = 1.0 - normalized.matmul(normalized[first_index])

    while len(selected) < budget and bool(available.any()):
        score = (
            diversity_weight * min_distance
            - (1.0 - diversity_weight) * center_distance
        )
        score = score.masked_fill(~available, float('-inf'))
        next_index = torch.argmax(score)
        selected.append(int(next_index.item()))
        available[next_index] = False
        distance_to_next = 1.0 - normalized.matmul(normalized[next_index])
        min_distance = torch.minimum(min_distance, distance_to_next)

    selected_index = torch.as_tensor(selected, dtype=torch.long, device=features.device)
    return features.index_select(0, selected_index).cpu().half()


@torch.no_grad()
def _compute_shared_feature_memory(original_model, data_loader, device,
                                   class_ids, args):
    original_model.eval()
    for class_id in class_ids:
        features_per_class = []
        for inputs, _ in data_loader[class_id]['train']:
            inputs = inputs.to(device, non_blocking=True)
            output = original_model(inputs)
            features_per_class.append(output['pre_logits'])
        features_per_class = torch.cat(features_per_class, dim=0)
        gathered = [
            torch.zeros_like(features_per_class, device=device)
            for _ in range(args.world_size)
        ]
        utils.distributed_barrier()
        dist.all_gather(gathered, features_per_class)
        gathered = torch.cat(gathered, dim=0)
        cls_shared_features[int(class_id)] = _select_real_feature_memory(
            gathered, args)


@torch.no_grad()
def _sample_real_feature_memory(class_id, sample_count, model, seen_classes, args, device):
    """Mix hard real features with random representatives from one class memory."""
    memory = cls_real_features.get(int(class_id))
    if memory is None or memory.numel() == 0 or sample_count <= 0:
        return None

    feature_bank = memory.to(device=device, dtype=torch.float32, non_blocking=True)
    bank_size = feature_bank.shape[0]
    hard_ratio = min(
        1.0, max(0.0, float(getattr(args, 'crct_real_hard_ratio', 0.5))))
    hard_count = min(bank_size, sample_count, int(round(sample_count * hard_ratio)))
    selected = []

    if hard_count > 0:
        logits = model(feature_bank, fc_only=True)['logits']
        seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=device)
        seen_logits = logits.index_select(1, seen_index)
        target_position = torch.nonzero(
            seen_index == int(class_id), as_tuple=False).flatten()
        if target_position.numel() == 1 and seen_index.numel() > 1:
            target_position = int(target_position.item())
            target_logit = seen_logits[:, target_position]
            competitor_logits = seen_logits.clone()
            competitor_logits[:, target_position] = float('-inf')
            margin = target_logit - competitor_logits.max(dim=1).values
            selected.extend(torch.argsort(margin)[:hard_count].tolist())

    selected_mask = torch.zeros(bank_size, dtype=torch.bool, device=device)
    if selected:
        selected_mask[torch.as_tensor(selected, dtype=torch.long, device=device)] = True
    remaining_count = sample_count - len(selected)
    available = torch.nonzero(~selected_mask, as_tuple=False).flatten()
    if remaining_count > 0 and available.numel() > 0:
        take_count = min(remaining_count, available.numel())
        permutation = torch.randperm(available.numel(), device=device)[:take_count]
        selected.extend(available.index_select(0, permutation).tolist())
        remaining_count -= take_count
    if remaining_count > 0:
        selected.extend(torch.randint(bank_size, (remaining_count,), device=device).tolist())

    selected_index = torch.as_tensor(selected, dtype=torch.long, device=device)
    return feature_bank.index_select(0, selected_index)


@torch.no_grad()
def _sample_hybrid_class_replay(class_id, total_count, model, seen_classes,
                                old_classes, args, device):
    """Use one fixed per-class budget shared by real memory and CFS replay."""
    default_ratio = float(getattr(args, 'crct_real_replay_ratio', 0.25))
    if int(class_id) in old_classes:
        real_ratio = float(getattr(args, 'crct_real_old_replay_ratio', default_ratio))
    else:
        real_ratio = float(getattr(args, 'crct_real_new_replay_ratio', default_ratio))
    real_ratio = min(1.0, max(0.0, real_ratio))
    has_memory = int(class_id) in cls_real_features
    real_count = int(round(total_count * real_ratio)) if has_memory else 0
    real_count = min(total_count, max(0, real_count))
    synthetic_count = total_count - real_count
    synthetic_chunks = []

    means = cls_mean[class_id]
    covariances = cls_cov[class_id]
    if not isinstance(means, list):
        means = [means]
        covariances = [covariances]
    valid_components = []
    for mean, covariance in zip(means, covariances):
        covariance = torch.as_tensor(covariance, device=device)
        if covariance.numel() > 0 and float(covariance.float().abs().mean()) > 0.0:
            valid_components.append((mean, covariance))

    if synthetic_count > 0 and valid_components:
        base_count, remainder = divmod(synthetic_count, len(valid_components))
        for component_id, (mean, covariance) in enumerate(valid_components):
            component_count = base_count + int(component_id < remainder)
            if component_count <= 0:
                continue
            mean = torch.as_tensor(mean, device=device).float()
            covariance = covariance.float()
            if covariance.dim() == 1:
                covariance = torch.diag(covariance)
            covariance = covariance + torch.eye(
                mean.numel(), device=device, dtype=mean.dtype) * 1e-4
            synthetic_chunks.append(utils.sample_boundary_aware_cfs_features(
                mean, covariance, component_count, args, device, model,
                class_id, seen_classes, cfs_model=cls_cfs_model.get(class_id)))

    real_features = _sample_real_feature_memory(
        class_id, real_count, model, seen_classes, args, device)
    chunks = synthetic_chunks
    if real_features is not None:
        chunks.append(real_features)
    if not chunks:
        raise RuntimeError('No replay features available for class {}'.format(class_id))
    replay = torch.cat(chunks, dim=0)
    if replay.shape[0] != total_count:
        raise RuntimeError(
            'Hybrid replay generated {} instead of {} samples for class {}'.format(
                replay.shape[0], total_count, class_id))
    return replay, 0 if real_features is None else real_features.shape[0]


@torch.no_grad()
def compute_rp_statistics(model, original_model, data_loader, device, task_id,
                          class_mask=None, args=None):
    """Accumulate Gram matrix and class prototypes for the routing-free head.

    Only aggregate second-order statistics are kept (no per-example features),
    so the strict exemplar-free protocol holds. With rp_feature_source=original
    the features come from the frozen backbone, which means the head needs no
    task inference at all -- the bottleneck that caps HRM-PET on CUB/CIFAR.
    """
    model.eval()
    original_model.eval()
    source = str(getattr(args, 'rp_feature_source', 'original'))
    for cls_id in class_mask:
        for inputs, targets in data_loader[cls_id]['train']:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            layer_router = bool(getattr(args, 'layer_stat_router', False))
            if layer_router:
                # Hooks must sit on whichever backbone actually runs the
                # forward, otherwise nothing is captured.
                install_hooks(
                    rp_extractor(original_model, args)
                    if source == 'original' else model)
            if source == 'original':
                features = rp_extractor(original_model, args)(
                    _rp_inputs(inputs, args))['pre_logits']
            else:
                # Fixed adapter for every task: this is RanPAC's first-session
                # adaptation, except the adaptation is HRM-PET's own trained
                # LoRA. The index must not vary or the Gram matrix would mix
                # incompatible feature spaces.
                features = model(
                    inputs, task_id=int(getattr(args, 'rp_lora_task', 0)),
                    train=True)['pre_logits']
            if layer_router:
                accumulate_task_stats(task_id)
                remove_hooks()
            accumulate_rp_statistics(
                features, targets, args, device, args.nb_classes)


@torch.no_grad()
def _collect_rp_calibration_scores(model, original_model, data_loader, device,
                                   task_id, class_mask, args):
    """Transient scores over current-task training data, for temperature fit."""
    source = str(getattr(args, 'rp_feature_source', 'original'))
    score_chunks = []
    target_chunks = []
    for cls_id in class_mask:
        for inputs, targets in data_loader[cls_id]['train']:
            inputs = inputs.to(device, non_blocking=True)
            if source == 'original':
                features = rp_extractor(original_model, args)(
                    _rp_inputs(inputs, args))['pre_logits']
            else:
                features = model(
                    inputs, task_id=int(getattr(args, 'rp_lora_task', 0)),
                    train=True)['pre_logits']
            score_chunks.append(rp_head_predict(features, args, device))
            target_chunks.append(targets.to(device, non_blocking=True))
    return torch.cat(score_chunks, dim=0), torch.cat(target_chunks, dim=0)


def calibrate_rp_head(model, original_model, data_loader, device, task_id,
                      class_mask, args):
    """Fit the scalar temperature that makes the head's Loss comparable."""
    scores, targets = _collect_rp_calibration_scores(
        model, original_model, data_loader, device, task_id, class_mask, args)
    return fit_rp_temperature(scores.float(), targets.long(), args, device)


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=task_id, train=True)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        utils.distributed_barrier()
        dist.all_gather(features_per_cls_list, features_per_cls)
        gathered_features_per_cls = torch.cat(features_per_cls_list, dim=0)
        if bool(getattr(args, 'crct_real_feature_replay', False)):
            cls_real_features[int(cls_id)] = _select_real_feature_memory(
                gathered_features_per_cls, args)
        cfs_calibration_enabled = bool(getattr(
            args, 'cfs_task_logit_calibration', False))
        if utils.use_cfs_sampling(args) or cfs_calibration_enabled:
            preserve_rng = (
                cfs_calibration_enabled and not utils.use_cfs_sampling(args))
            cpu_rng_state = None
            cuda_rng_state = None
            if preserve_rng:
                cpu_rng_state = torch.random.get_rng_state()
                if torch.cuda.is_available():
                    cuda_rng_state = torch.cuda.get_rng_state_all()
            try:
                cls_cfs_model[cls_id] = utils.train_cfs_model(
                    gathered_features_per_cls,
                    args,
                    device,
                    force_cfs=cfs_calibration_enabled,
                )
            finally:
                if preserve_rng:
                    torch.random.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state_all(cuda_rng_state)

        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = gathered_features_per_cls
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = gathered_features_per_cls
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = gathered_features_per_cls.cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars


def _crct_old_class_distillation_loss(student_logits, teacher_logits, targets,
                                      seen_classes, old_classes, temperature=2.0,
                                      confidence_threshold=0.6):
    """Distill confident old-class replay samples while leaving boundary samples plastic."""
    zero = student_logits.new_zeros(())
    if student_logits.numel() == 0 or not old_classes or not seen_classes:
        return zero, 0

    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=student_logits.device)
    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=student_logits.device)
    old_sample_mask = (targets.unsqueeze(1) == old_index.unsqueeze(0)).any(dim=1)
    if not bool(old_sample_mask.any()):
        return zero, 0

    student_seen = student_logits[old_sample_mask].index_select(1, seen_index)
    teacher_seen = teacher_logits[old_sample_mask].index_select(1, seen_index)
    temperature = max(float(temperature), 1e-6)

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_seen / temperature, dim=1)
        teacher_confidence_probs = F.softmax(teacher_seen, dim=1)
        class_positions = torch.full(
            (student_logits.shape[1],), -1, dtype=torch.long, device=student_logits.device)
        class_positions[seen_index] = torch.arange(
            seen_index.numel(), dtype=torch.long, device=student_logits.device)
        target_positions = class_positions[targets[old_sample_mask]]
        target_confidence = teacher_confidence_probs.gather(
            1, target_positions.unsqueeze(1)).squeeze(1)
        keep_mask = target_confidence >= float(confidence_threshold)

    kept_samples = int(keep_mask.sum().item())
    if kept_samples == 0:
        return zero, 0

    per_sample_kl = F.kl_div(
        F.log_softmax(student_seen[keep_mask] / temperature, dim=1),
        teacher_probs[keep_mask],
        reduction='none',
    ).sum(dim=1)
    return per_sample_kl.mean() * (temperature ** 2), kept_samples


def _crct_replay_reliability_weights(teacher_logits, targets, seen_classes,
                                     old_classes, floor=0.25, power=1.0,
                                     preserve_mass=False,
                                     preserve_class_mass=False):
    """Weight uncertain old-class synthetic samples without weakening new-class learning."""
    weights = teacher_logits.new_ones(targets.shape, dtype=torch.float32)
    if teacher_logits.numel() == 0 or not old_classes or not seen_classes:
        return weights, weights.new_zeros(()), 0

    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=targets.device)
    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=targets.device)
    old_sample_mask = (targets.unsqueeze(1) == old_index.unsqueeze(0)).any(dim=1)
    old_sample_count = int(old_sample_mask.sum().item())
    if old_sample_count == 0:
        return weights, weights.new_zeros(()), 0

    with torch.no_grad():
        teacher_seen = teacher_logits[old_sample_mask].index_select(1, seen_index)
        teacher_probs = F.softmax(teacher_seen, dim=1)
        class_positions = torch.full(
            (teacher_logits.shape[1],), -1, dtype=torch.long, device=targets.device)
        class_positions[seen_index] = torch.arange(
            seen_index.numel(), dtype=torch.long, device=targets.device)
        target_positions = class_positions[targets[old_sample_mask]]
        target_confidence = teacher_probs.gather(
            1, target_positions.unsqueeze(1)).squeeze(1)

        floor = min(1.0, max(0.0, float(floor)))
        power = max(0.0, float(power))
        old_weights = floor + (1.0 - floor) * target_confidence.pow(power)
        if bool(preserve_class_mass):
            old_targets = targets[old_sample_mask]
            normalized_weights = old_weights.clone()
            for class_id in torch.unique(old_targets):
                class_sample_mask = old_targets == class_id
                class_weights = old_weights[class_sample_mask]
                normalized_weights[class_sample_mask] = (
                    class_weights / class_weights.mean().clamp_min(1e-12)
                )
            old_weights = normalized_weights
        elif bool(preserve_mass):
            old_weights = old_weights / old_weights.mean().clamp_min(1e-12)
        weights[old_sample_mask] = old_weights

    return weights, target_confidence.mean(), old_sample_count


def _scale_old_classifier_row_gradients(head, old_classes, scale):
    """Apply a smaller effective learning rate to consolidated old classifier rows."""
    if not old_classes:
        return
    scale = min(1.0, max(0.0, float(scale)))
    if scale >= 1.0:
        return

    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=head.weight.device)
    if head.weight.grad is not None:
        old_weight_grad = head.weight.grad.index_select(0, old_index) * scale
        head.weight.grad.index_copy_(0, old_index, old_weight_grad)
    if head.bias is not None and head.bias.grad is not None:
        old_bias_grad = head.bias.grad.index_select(0, old_index) * scale
        head.bias.grad.index_copy_(0, old_index, old_bias_grad)


@torch.no_grad()
def _build_crct_trust_anchors(class_means, class_covariances, old_classes,
                              args, device):
    """Draw independent high-density anchors from stored real-feature statistics."""
    samples_per_component = max(
        1, int(getattr(args, 'crct_trust_samples_per_component', 4)))
    covariance_scale = max(
        0.0, float(getattr(args, 'crct_trust_cov_scale', 0.25)))
    anchor_chunks = []
    anchor_labels = []

    for class_id in old_classes:
        stored_means = class_means.get(class_id)
        stored_covariances = class_covariances.get(class_id)
        if stored_means is None or stored_covariances is None:
            continue

        means = stored_means if isinstance(stored_means, list) else [stored_means]
        covariances = (
            stored_covariances
            if isinstance(stored_covariances, list)
            else [stored_covariances]
        )
        for component_mean, component_covariance in zip(means, covariances):
            mean = torch.as_tensor(component_mean, device=device).float()
            covariance = torch.as_tensor(component_covariance, device=device).float()
            component_anchors = [mean.unsqueeze(0)]
            random_count = samples_per_component - 1
            if random_count > 0 and covariance_scale > 0.0:
                if covariance.dim() == 1:
                    std = (covariance.clamp_min(1e-6) * covariance_scale).sqrt()
                    random_anchors = (
                        mean.unsqueeze(0)
                        + torch.randn(
                            random_count, mean.numel(), device=device,
                            dtype=mean.dtype) * std.unsqueeze(0)
                    )
                else:
                    scaled_covariance = covariance * covariance_scale
                    scaled_covariance = scaled_covariance + torch.eye(
                        mean.numel(), device=device, dtype=mean.dtype) * 1e-5
                    distribution = torch.distributions.MultivariateNormal(
                        mean, scaled_covariance)
                    random_anchors = distribution.sample((random_count,))
                component_anchors.append(random_anchors)

            component_anchors = torch.cat(component_anchors, dim=0)
            anchor_chunks.append(component_anchors)
            anchor_labels.extend([int(class_id)] * component_anchors.shape[0])

    if not anchor_chunks:
        return None, None
    return (
        torch.cat(anchor_chunks, dim=0),
        torch.as_tensor(anchor_labels, dtype=torch.long, device=device),
    )


@torch.no_grad()
def _select_crct_adaptive_trust_alpha(base_model, teacher_fc_norm, teacher_head,
                                      anchors, targets, seen_classes,
                                      old_classes, args):
    """Interpolate the whole classifier to satisfy worst-class logit-drift limits."""
    if anchors is None or targets is None or anchors.numel() == 0:
        return 1.0, None

    device = anchors.device
    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=device)
    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=device)
    old_sample_mask = (targets.unsqueeze(1) == old_index.unsqueeze(0)).any(dim=1)
    if not bool(old_sample_mask.any()):
        return 1.0, None

    anchors = anchors[old_sample_mask]
    targets = targets[old_sample_mask]
    teacher_logits = teacher_head(teacher_fc_norm(anchors)).index_select(1, seen_index)
    student_logits = base_model.head(base_model.fc_norm(anchors)).index_select(1, seen_index)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=1)
    teacher_probs = teacher_log_probs.exp()

    class_positions = torch.full(
        (base_model.head.out_features,), -1, dtype=torch.long, device=device)
    class_positions[seen_index] = torch.arange(
        seen_index.numel(), dtype=torch.long, device=device)
    target_positions = class_positions[targets]
    teacher_target_confidence = teacher_probs.gather(
        1, target_positions.unsqueeze(1)).squeeze(1)
    teacher_competitors = teacher_logits.clone()
    teacher_competitors.scatter_(
        1, target_positions.unsqueeze(1), float('-inf'))
    teacher_margin = (
        teacher_logits.gather(1, target_positions.unsqueeze(1)).squeeze(1)
        - teacher_competitors.max(dim=1).values
    )

    steps = max(1, int(getattr(args, 'crct_trust_steps', 10)))
    quantile = min(
        1.0, max(0.0, float(getattr(args, 'crct_trust_quantile', 0.9))))
    max_kl = max(0.0, float(getattr(args, 'crct_trust_max_kl', 0.02)))
    max_conf_drop = max(
        0.0, float(getattr(args, 'crct_trust_max_conf_drop', 0.02)))
    max_margin_drop = max(
        0.0, float(getattr(args, 'crct_trust_max_margin_drop', 0.10)))

    best_alpha = 0.0
    best_metrics = {'kl': 0.0, 'conf_drop': 0.0, 'margin_drop': 0.0}
    class_ids = torch.unique(targets, sorted=True)
    for step in range(steps + 1):
        alpha = step / float(steps)
        candidate_logits = teacher_logits.lerp(student_logits, alpha)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=1)
        candidate_probs = candidate_log_probs.exp()
        sample_kl = (
            teacher_probs * (teacher_log_probs - candidate_log_probs)
        ).sum(dim=1)
        candidate_target_confidence = candidate_probs.gather(
            1, target_positions.unsqueeze(1)).squeeze(1)
        confidence_drop = (
            teacher_target_confidence - candidate_target_confidence
        ).clamp_min(0.0)
        candidate_competitors = candidate_logits.clone()
        candidate_competitors.scatter_(
            1, target_positions.unsqueeze(1), float('-inf'))
        candidate_margin = (
            candidate_logits.gather(1, target_positions.unsqueeze(1)).squeeze(1)
            - candidate_competitors.max(dim=1).values
        )
        margin_drop = (teacher_margin - candidate_margin).clamp_min(0.0)

        class_kl = []
        class_conf_drop = []
        class_margin_drop = []
        for class_id in class_ids:
            class_mask = targets == class_id
            class_kl.append(sample_kl[class_mask].mean())
            class_conf_drop.append(confidence_drop[class_mask].mean())
            class_margin_drop.append(margin_drop[class_mask].mean())
        kl_value = torch.quantile(torch.stack(class_kl), quantile).item()
        conf_drop_value = torch.quantile(
            torch.stack(class_conf_drop), quantile).item()
        margin_drop_value = torch.quantile(
            torch.stack(class_margin_drop), quantile).item()
        if (
                kl_value <= max_kl
                and conf_drop_value <= max_conf_drop
                and margin_drop_value <= max_margin_drop):
            best_alpha = alpha
            best_metrics = {
                'kl': kl_value,
                'conf_drop': conf_drop_value,
                'margin_drop': margin_drop_value,
            }

    return best_alpha, best_metrics


@torch.no_grad()
def _blend_crct_classifier(base_model, teacher_head, alpha):
    """Apply one shared interpolation factor to every classifier parameter."""
    for student_parameter, teacher_parameter in zip(
            base_model.head.parameters(), teacher_head.parameters()):
        student_parameter.copy_(teacher_parameter.lerp(student_parameter, alpha))


@torch.no_grad()
def _macro_crct_metrics(logits, targets, class_ids, class_to_task=None,
                        logit_class_ids=None):
    if targets.numel() == 0 or not class_ids:
        return {'accuracy': 0.0, 'top5': 0.0, 'task_accuracy': 0.0,
                'ce': float('inf'),
                'per_class_accuracy': {}}
    sample_class_index = torch.as_tensor(
        class_ids, dtype=torch.long, device=logits.device)
    sample_mask = targets.unsqueeze(1).eq(
        sample_class_index.unsqueeze(0)).any(dim=1)
    if not bool(sample_mask.any()):
        return {'accuracy': 0.0, 'top5': 0.0, 'task_accuracy': 0.0,
                'ce': float('inf'),
                'per_class_accuracy': {}}
    logits = logits[sample_mask]
    targets = targets[sample_mask]
    if logit_class_ids is None:
        logit_class_ids = class_ids
    class_index = torch.as_tensor(
        logit_class_ids, dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(1, class_index)
    class_positions = torch.full(
        (logits.shape[1],), -1, dtype=torch.long, device=logits.device)
    class_positions[class_index] = torch.arange(class_index.numel(), device=logits.device)
    target_positions = class_positions[targets]
    if bool(target_positions.lt(0).any()):
        raise ValueError('CRCT metric received targets outside the requested classes')
    predictions = selected_logits.argmax(dim=1)
    top_k = min(5, selected_logits.shape[1])
    top5_predictions = selected_logits.topk(top_k, dim=1).indices
    top5_correct = top5_predictions.eq(target_positions.unsqueeze(1)).any(dim=1)
    task_correct = predictions.new_zeros(predictions.shape, dtype=torch.bool)
    if class_to_task:
        task_lookup = torch.full(
            (logits.shape[1],), -1, dtype=torch.long, device=logits.device)
        mapped_classes = torch.as_tensor(
            list(class_to_task), dtype=torch.long, device=logits.device)
        mapped_tasks = torch.as_tensor(
            [class_to_task[class_id] for class_id in class_to_task],
            dtype=torch.long, device=logits.device)
        task_lookup[mapped_classes] = mapped_tasks
        predicted_classes = class_index.index_select(0, predictions)
        task_correct = task_lookup[predicted_classes].eq(task_lookup[targets])
    sample_ce = F.cross_entropy(selected_logits, target_positions, reduction='none')
    class_accuracy = []
    class_top5 = []
    class_task_accuracy = []
    class_ce = []
    per_class_accuracy = {}
    for class_id in class_ids:
        class_mask = targets.eq(int(class_id))
        if not bool(class_mask.any()):
            continue
        accuracy_value = (
            predictions[class_mask].eq(target_positions[class_mask]).float().mean())
        class_accuracy.append(accuracy_value)
        per_class_accuracy[int(class_id)] = float(accuracy_value.mul(100.0).item())
        class_top5.append(top5_correct[class_mask].float().mean())
        if class_to_task:
            class_task_accuracy.append(task_correct[class_mask].float().mean())
        class_ce.append(sample_ce[class_mask].mean())
    if not class_accuracy:
        return {'accuracy': 0.0, 'top5': 0.0, 'task_accuracy': 0.0,
                'ce': float('inf'),
                'per_class_accuracy': {}}
    return {
        'accuracy': float(torch.stack(class_accuracy).mean().mul(100.0).item()),
        'top5': float(torch.stack(class_top5).mean().mul(100.0).item()),
        'task_accuracy': float(
            torch.stack(class_task_accuracy).mean().mul(100.0).item()
            if class_task_accuracy else 0.0),
        'ce': float(torch.stack(class_ce).mean().item()),
        'per_class_accuracy': per_class_accuracy,
    }


@torch.no_grad()
def _select_crct_validation_alpha(base_model, teacher_fc_norm, teacher_head,
                                  anchors, targets, seen_classes, old_classes,
                                  class_to_task, args, current_classes=None):
    """Select a classifier interpolation on independent feature-statistic anchors."""
    if anchors is None or targets is None or anchors.numel() == 0:
        return 0.0, None

    teacher_logits = teacher_head(teacher_fc_norm(anchors))
    student_logits = base_model.head(base_model.fc_norm(anchors))
    teacher_all = _macro_crct_metrics(
        teacher_logits, targets, seen_classes, class_to_task)
    teacher_old = _macro_crct_metrics(
        teacher_logits, targets, old_classes, class_to_task,
        logit_class_ids=seen_classes)
    teacher_current = (
        _macro_crct_metrics(
            teacher_logits, targets, current_classes, class_to_task,
            logit_class_ids=seen_classes)
        if current_classes else None
    )
    steps = max(1, int(getattr(args, 'crct_validation_steps', 10)))
    max_alpha = min(
        1.0, max(0.0, float(getattr(args, 'crct_validation_max_alpha', 1.0))))
    max_old_drop = max(
        0.0, float(getattr(args, 'crct_validation_max_old_acc_drop', 0.0)))
    min_acc_gain = float(getattr(args, 'crct_validation_min_acc_gain', 0.0))
    min_top5_gain = float(getattr(args, 'crct_validation_min_top5_gain', 0.0))
    min_task_gain = float(getattr(args, 'crct_validation_min_task_gain', 0.0))
    min_ce_gain = float(getattr(args, 'crct_validation_min_ce_gain', 0.0))
    max_current_drop = max(
        0.0, float(getattr(args, 'crct_validation_max_current_acc_drop', 0.0)))
    current_ce_tolerance = max(
        0.0, float(getattr(args, 'crct_validation_current_ce_tolerance', 0.0)))

    best_alpha = 0.0
    best_all = teacher_all
    best_old = teacher_old
    best_current = teacher_current
    best_key = (
        teacher_all['accuracy'], teacher_all['task_accuracy'],
        teacher_all['top5'], -teacher_all['ce'], 0.0)
    tolerance = 1e-7
    for step in range(1, steps + 1):
        alpha = max_alpha * step / float(steps)
        candidate_logits = teacher_logits.lerp(student_logits, alpha)
        candidate_all = _macro_crct_metrics(
            candidate_logits, targets, seen_classes, class_to_task)
        candidate_old = _macro_crct_metrics(
            candidate_logits, targets, old_classes, class_to_task,
            logit_class_ids=seen_classes)
        candidate_current = (
            _macro_crct_metrics(
                candidate_logits, targets, current_classes, class_to_task,
                logit_class_ids=seen_classes)
            if current_classes else None
        )
        old_ok = (
            not old_classes or all(
                candidate_old['per_class_accuracy'].get(class_id, 0.0)
                + max_old_drop + tolerance
                >= teacher_old['per_class_accuracy'].get(class_id, 0.0)
                for class_id in old_classes)
        )
        accuracy_ok = (
            candidate_all['accuracy'] + tolerance
            >= teacher_all['accuracy'] + min_acc_gain
        )
        top5_ok = (
            candidate_all['top5'] + tolerance
            >= teacher_all['top5'] + min_top5_gain
        )
        task_ok = (
            candidate_all['task_accuracy'] + tolerance
            >= teacher_all['task_accuracy'] + min_task_gain
        )
        ce_ok = (
            candidate_all['ce']
            <= teacher_all['ce'] - min_ce_gain + tolerance
        )
        current_ok = (
            not current_classes
            or (
                all(
                    candidate_current['per_class_accuracy'].get(class_id, 0.0)
                    + max_current_drop + tolerance
                    >= teacher_current['per_class_accuracy'].get(class_id, 0.0)
                    for class_id in current_classes
                )
                and candidate_current['accuracy'] + tolerance
                >= teacher_current['accuracy']
                and candidate_current['top5'] + tolerance
                >= teacher_current['top5']
                and candidate_current['task_accuracy'] + tolerance
                >= teacher_current['task_accuracy']
                and candidate_current['ce']
                <= teacher_current['ce'] + current_ce_tolerance + tolerance
            )
        )
        if not (
                old_ok and accuracy_ok and top5_ok and task_ok and ce_ok
                and current_ok):
            continue
        candidate_key = (
            candidate_all['accuracy'], candidate_all['task_accuracy'],
            candidate_all['top5'], -candidate_all['ce'], alpha)
        if candidate_key > best_key:
            best_alpha = alpha
            best_all = candidate_all
            best_old = candidate_old
            best_current = candidate_current
            best_key = candidate_key

    old_class_drops = [
        teacher_old['per_class_accuracy'].get(class_id, 0.0)
        - best_old['per_class_accuracy'].get(class_id, 0.0)
        for class_id in old_classes
    ]
    compact = lambda values: {
        'accuracy': values['accuracy'], 'top5': values['top5'],
        'task_accuracy': values['task_accuracy'], 'ce': values['ce']}
    metrics = {
        'teacher_all': compact(teacher_all),
        'teacher_old': compact(teacher_old),
        'selected_all': compact(best_all),
        'selected_old': compact(best_old),
        'worst_old_class_drop': (
            max(old_class_drops) if old_class_drops else 0.0),
    }
    if teacher_current is not None:
        metrics['teacher_current'] = compact(teacher_current)
        metrics['selected_current'] = compact(best_current)
    return best_alpha, metrics


@torch.no_grad()
def _collect_current_task_validation_features(
        model, data_loader_per_cls, current_classes, task_id,
        samples_per_class, device):
    """Collect transient current-task features; callers must not persist them."""
    if data_loader_per_cls is None or samples_per_class <= 0:
        return None, None

    was_training = model.training
    model.eval()
    feature_chunks = []
    target_chunks = []
    try:
        for class_id in current_classes:
            remaining = samples_per_class
            for inputs, _ in data_loader_per_cls[int(class_id)]['train']:
                inputs = inputs.to(device, non_blocking=True)
                features = model(
                    inputs, task_id=task_id, train=True)['pre_logits'].detach()
                take = min(remaining, features.shape[0])
                feature_chunks.append(features[:take])
                target_chunks.append(torch.full(
                    (take,), int(class_id), dtype=torch.long, device=device))
                remaining -= take
                if remaining <= 0:
                    break
    finally:
        model.train(was_training)

    if not feature_chunks:
        return None, None
    return torch.cat(feature_chunks, dim=0), torch.cat(target_chunks, dim=0)


def _use_cfs_for_crct_class(class_id, old_classes, args):
    return (
        not bool(getattr(args, 'cfs_old_classes_only', False))
        or int(class_id) in {int(value) for value in old_classes}
    )


@torch.no_grad()
def _sample_crct_class_features(mean, cov, count, args, device, model,
                                class_id, seen_classes, old_classes,
                                cfs_model=None):
    if not _use_cfs_for_crct_class(class_id, old_classes, args):
        distribution = utils.stable_multivariate_normal(
            mean, cov, '_sample_crct_class_features')
        return distribution.sample(sample_shape=(count,))
    return utils.sample_boundary_aware_cfs_features(
        mean.float(), cov.float(), count, args, device, model,
        class_id, seen_classes, cfs_model=cfs_model)


def train_task_adaptive_prediction(model: torch.nn.Module, args, device,
                                   class_mask=None, task_id=-1,
                                   data_loader_per_cls=None):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    base_model = model.module if hasattr(model, 'module') else model
    head_only = bool(getattr(args, 'crct_head_only', False))
    fc_norm_grad_states = []
    if head_only:
        if not hasattr(base_model, 'head') or not hasattr(base_model, 'fc_norm'):
            raise AttributeError('Head-only CRCT requires model.head and model.fc_norm')
        fc_norm_grad_states = [
            (parameter, parameter.requires_grad)
            for parameter in base_model.fc_norm.parameters()
        ]
        for parameter, _ in fc_norm_grad_states:
            parameter.requires_grad_(False)
        param_list = [
            parameter for parameter in base_model.head.parameters()
            if parameter.requires_grad
        ]
    else:
        param_list = [
            p for n, p in model.named_parameters()
            if p.requires_grad and 'lora' not in n
        ]
    if not param_list:
        raise ValueError('CRCT has no trainable parameters')
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    old_classes = [int(cls_id) for seen_task in range(task_id) for cls_id in class_mask[seen_task]]
    seen_classes = [
        int(cls_id) for seen_task in range(task_id + 1) for cls_id in class_mask[seen_task]]
    crct_num = len(old_classes)

    distill_weight = max(0.0, float(getattr(args, 'crct_distill_weight', 0.25)))
    distill_temperature = max(
        1e-6, float(getattr(args, 'crct_distill_temperature', 2.0)))
    distill_confidence = min(
        1.0, max(0.0, float(getattr(args, 'crct_distill_confidence', 0.6))))
    distill_enabled = (
        bool(getattr(args, 'crct_old_class_distill', False))
        and distill_weight > 0.0
        and len(old_classes) > 0
    )
    anchor_weight = max(0.0, float(getattr(args, 'crct_anchor_weight', 0.0)))
    reliability_enabled = (
        bool(getattr(args, 'crct_reliability_weighting', False))
        and len(old_classes) > 0
    )
    reliability_floor = min(
        1.0, max(0.0, float(getattr(args, 'crct_reliability_floor', 0.25))))
    reliability_power = max(
        0.0, float(getattr(args, 'crct_reliability_power', 1.0)))
    reliability_preserve_mass = bool(
        getattr(args, 'crct_reliability_preserve_mass', False))
    reliability_preserve_class_mass = bool(
        getattr(args, 'crct_reliability_preserve_class_mass', False))
    hybrid_real_replay = bool(getattr(args, 'crct_real_feature_replay', False))
    if hybrid_real_replay and utils.use_semantic_projection(args):
        raise ValueError(
            'Hybrid real-feature replay and semantic projection must be evaluated separately')
    if hybrid_real_replay and not cls_real_features:
        raise RuntimeError('Real-feature replay is enabled but the feature memory is empty')
    hybrid_samples_per_class = int(
        getattr(args, 'crct_hybrid_samples_per_class', 0))
    trust_region_enabled = (
        bool(getattr(args, 'crct_adaptive_trust_region', False))
        and len(old_classes) > 0
    )
    if trust_region_enabled and not head_only:
        raise ValueError(
            'Adaptive CRCT trust region requires --crct_head_only')
    validation_gate_enabled = bool(getattr(args, 'crct_validation_gate', False))
    if validation_gate_enabled and not head_only:
        raise ValueError('CRCT validation gate requires --crct_head_only')
    if validation_gate_enabled and trust_region_enabled:
        raise ValueError(
            'Use either CRCT validation gate or adaptive trust region, not both')
    current_real_gate_enabled = (
        validation_gate_enabled
        and bool(getattr(args, 'crct_validation_current_real', False))
    )
    if current_real_gate_enabled and data_loader_per_cls is None:
        raise ValueError('Current-real CRCT validation requires per-class train loaders')
    old_row_lr_scale = min(
        1.0, max(0.0, float(getattr(args, 'crct_old_row_lr_scale', 1.0))))

    teacher_fc_norm = None
    teacher_head = None
    if (distill_enabled or anchor_weight > 0.0 or reliability_enabled
            or trust_region_enabled or validation_gate_enabled):
        if not hasattr(base_model, 'head') or not hasattr(base_model, 'fc_norm'):
            raise AttributeError('Stability-aware CRCT requires model.head and model.fc_norm')
        teacher_fc_norm = copy.deepcopy(base_model.fc_norm).to(device).eval()
        teacher_head = copy.deepcopy(base_model.head).to(device).eval()
        for teacher_parameter in list(teacher_fc_norm.parameters()) + list(teacher_head.parameters()):
            teacher_parameter.requires_grad_(False)

    current_validation_anchors = None
    current_validation_targets = None
    current_classes = [int(class_id) for class_id in class_mask[task_id]]
    if current_real_gate_enabled and (
            not dist.is_initialized() or utils.is_main_process()):
        current_validation_anchors, current_validation_targets = (
            _collect_current_task_validation_features(
                base_model, data_loader_per_cls, current_classes, task_id,
                max(1, int(getattr(
                    args, 'crct_validation_current_samples_per_class', 16))),
                device)
        )
        if current_validation_anchors is None:
            raise RuntimeError(
                'Current-real CRCT validation could not collect any features')
        print(
            'CRCT current-real validation:',
            'classes=', len(current_classes),
            'anchors=', current_validation_targets.numel())

    if utils.is_main_process() and (
            head_only or distill_enabled or anchor_weight > 0.0
            or reliability_enabled or trust_region_enabled or hybrid_real_replay
            or validation_gate_enabled or old_row_lr_scale < 1.0):
        print(
            'CRCT stability:',
            'head_only=', head_only,
            'old_classes=', len(old_classes),
            'distill_weight=', distill_weight if distill_enabled else 0.0,
            'temperature=', distill_temperature,
            'confidence=', distill_confidence,
            'anchor_weight=', anchor_weight,
            'reliability=', reliability_enabled,
            'reliability_floor=', reliability_floor,
            'reliability_power=', reliability_power,
            'reliability_preserve_mass=', reliability_preserve_mass,
            'reliability_preserve_class_mass=', reliability_preserve_class_mass,
            'adaptive_trust_region=', trust_region_enabled,
            'validation_gate=', validation_gate_enabled,
            'real_feature_replay=', hybrid_real_replay,
            'real_replay_ratio=', float(getattr(args, 'crct_real_replay_ratio', 0.25)),
            'real_old_new_ratio=', (float(getattr(args, 'crct_real_old_replay_ratio', 0.35)), float(getattr(args, 'crct_real_new_replay_ratio', 0.10))),
            'real_hard_ratio=', float(getattr(args, 'crct_real_hard_ratio', 0.5)),
            'hybrid_samples_per_class=', hybrid_samples_per_class,
            'old_row_lr_scale=', old_row_lr_scale,
        )


    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5
        if hybrid_real_replay and hybrid_samples_per_class > 0:
            num_sampled_pcls = hybrid_samples_per_class
        real_replay_total = 0

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if hybrid_real_replay:
            for seen_task in range(task_id + 1):
                for c_id in class_mask[seen_task]:
                    sampled_data_single, real_count = _sample_hybrid_class_replay(
                        c_id, num_sampled_pcls, model, seen_classes, old_classes, args, device)
                    sampled_data.append(sampled_data_single)
                    sampled_label.extend([c_id] * sampled_data_single.shape[0])
                    real_replay_total += real_count
        elif args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    projected_count = 0
                    if utils.use_semantic_projection(args):
                        projected_count = int(num_sampled_pcls * float(getattr(args, 'semantic_projection_ratio', 0.25)))
                        projected_count = max(0, min(num_sampled_pcls - 1, projected_count))
                    base_count = num_sampled_pcls - projected_count
                    sampled_data_single = _sample_crct_class_features(
                        mean, cov, base_count, args, device, model,
                        c_id, seen_classes, old_classes,
                        cfs_model=cls_cfs_model.get(c_id))
                    if projected_count > 0:
                        available_classes = []
                        for seen_task in range(task_id + 1):
                            available_classes.extend(class_mask[seen_task])
                        projected_data = utils.sample_semantic_projected_features(
                            c_id, mean.float(), projected_count, args, device,
                            cls_mean, cls_cov, cls_cfs_model=cls_cfs_model,
                            available_classes=available_classes)
                        if projected_data is not None:
                            sampled_data_single = torch.cat([sampled_data_single, projected_data], dim=0)
                        elif base_count < num_sampled_pcls:
                            fallback_data = utils.sample_cfs_features(
                                mean.float(), cov.float(), num_sampled_pcls - base_count, args, device,
                                cfs_model=cls_cfs_model.get(c_id))
                            sampled_data_single = torch.cat([sampled_data_single, fallback_data], dim=0)
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * sampled_data_single.shape[0])

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       cov = (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float()
                       projected_count = 0
                       if utils.use_semantic_projection(args):
                           projected_count = int(num_sampled_pcls * float(getattr(args, 'semantic_projection_ratio', 0.25)))
                           projected_count = max(0, min(num_sampled_pcls - 1, projected_count))
                       base_count = num_sampled_pcls - projected_count
                       sampled_data_single = _sample_crct_class_features(
                           mean, cov, base_count, args, device, model,
                           c_id, seen_classes, old_classes,
                           cfs_model=cls_cfs_model.get(c_id))
                       if projected_count > 0:
                           available_classes = []
                           for seen_task in range(task_id + 1):
                               available_classes.extend(class_mask[seen_task])
                           projected_data = utils.sample_semantic_projected_features(
                               c_id, mean.float(), projected_count, args, device,
                               cls_mean, cls_cov, cls_cfs_model=cls_cfs_model,
                               available_classes=available_classes)
                           if projected_data is not None:
                               sampled_data_single = torch.cat([sampled_data_single, projected_data], dim=0)
                           elif base_count < num_sampled_pcls:
                               fallback_data = utils.sample_cfs_features(
                                   mean.float(), cov, num_sampled_pcls - base_count, args, device,
                                   cfs_model=cls_cfs_model.get(c_id))
                               sampled_data_single = torch.cat([sampled_data_single, fallback_data], dim=0)
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * sampled_data_single.shape[0])
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        print(sampled_data.shape)
        if hybrid_real_replay and utils.is_main_process():
            print(
                'CRCT hybrid replay:',
                'epoch=', epoch + 1,
                'samples_per_class=', num_sampled_pcls,
                'real=', real_replay_total,
                'synthetic=', sampled_data.shape[0] - real_replay_total,
                'classes=', len(seen_classes),
            )

        inputs = sampled_data
        targets = sampled_label

        if bool(getattr(args, 'crct_balanced_batches', False)):
            sf_indexes = utils.class_balanced_replay_order(targets)
        else:
            sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # print(targets)

        replay_iterations = crct_num
        if bool(getattr(args, 'crct_use_all_samples', False)):
            replay_iterations = int(math.ceil(inputs.size(0) / float(num_sampled_pcls)))
        print('CRCT replay iterations:', replay_iterations, 'of', int(math.ceil(inputs.size(0) / float(num_sampled_pcls))))

        for _iter in range(replay_iterations):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            teacher_logits = None
            if distill_enabled or reliability_enabled:
                with torch.no_grad():
                    teacher_logits = teacher_head(teacher_fc_norm(inp))

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            replay_weight = logits.new_ones(tgt.shape, dtype=torch.float32)
            old_teacher_confidence = logits.new_zeros(())
            old_replay_count = 0
            if reliability_enabled:
                replay_weight, old_teacher_confidence, old_replay_count = (
                    _crct_replay_reliability_weights(
                        teacher_logits,
                        tgt,
                        seen_classes,
                        old_classes,
                        floor=reliability_floor,
                        power=reliability_power,
                        preserve_mass=reliability_preserve_mass,
                        preserve_class_mass=reliability_preserve_class_mass,
                    )
                )
                per_sample_ce = F.cross_entropy(logits, tgt, reduction='none')
                loss_ce = (
                    per_sample_ce * replay_weight
                ).sum() / replay_weight.sum().clamp_min(1e-12)
            else:
                loss_ce = criterion(logits, tgt)
            loss_kd = logits.new_zeros(())
            kd_kept = 0
            if distill_enabled:
                loss_kd, kd_kept = _crct_old_class_distillation_loss(
                    logits,
                    teacher_logits,
                    tgt,
                    seen_classes,
                    old_classes,
                    temperature=distill_temperature,
                    confidence_threshold=distill_confidence,
                )

            loss_anchor = logits.new_zeros(())
            if anchor_weight > 0.0:
                old_class_index = torch.as_tensor(
                    old_classes, dtype=torch.long, device=base_model.head.weight.device)
                weight_delta = (
                    base_model.head.weight.index_select(0, old_class_index)
                    - teacher_head.weight.index_select(0, old_class_index)
                )
                loss_anchor = weight_delta.pow(2).sum(dim=1).mean()
                if base_model.head.bias is not None and teacher_head.bias is not None:
                    bias_delta = (
                        base_model.head.bias.index_select(0, old_class_index)
                        - teacher_head.bias.index_select(0, old_class_index)
                    )
                    loss_anchor = loss_anchor + bias_delta.pow(2).mean()

            loss = (
                loss_ce
                + distill_weight * loss_kd
                + anchor_weight * loss_anchor
            )
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            _scale_old_classifier_row_gradients(
                base_model.head, old_classes, old_row_lr_scale)
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(
                Loss=loss.item(),
                CE=loss_ce.item(),
                KD=loss_kd.item(),
                Anchor=loss_anchor.item(),
                KDKeep=kd_kept,
                ReplayW=replay_weight.mean().item(),
                OldConf=old_teacher_confidence.item(),
                OldReplay=old_replay_count,
            )
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()

    if validation_gate_enabled:
        selected_alpha = 0.0
        validation_metrics = None
        validation_anchor_count = 0
        if not dist.is_initialized() or utils.is_main_process():
            validation_args = copy.copy(args)
            validation_args.crct_trust_samples_per_component = int(getattr(
                args, 'crct_validation_samples_per_component', 8))
            validation_args.crct_trust_cov_scale = float(getattr(
                args, 'crct_validation_cov_scale', 0.25))
            anchor_chunks = []
            target_chunks = []
            validation_repeats = max(
                1, int(getattr(args, 'crct_validation_repeats', 1)))
            validation_classes = (
                old_classes if current_real_gate_enabled else seen_classes)
            for _ in range(validation_repeats):
                repeat_anchors, repeat_targets = _build_crct_trust_anchors(
                    cls_mean, cls_cov, validation_classes, validation_args, device)
                if repeat_anchors is not None:
                    anchor_chunks.append(repeat_anchors)
                    target_chunks.append(repeat_targets)
            validation_anchors = (
                torch.cat(anchor_chunks, dim=0) if anchor_chunks else None)
            validation_targets = (
                torch.cat(target_chunks, dim=0) if target_chunks else None)
            if current_real_gate_enabled:
                validation_anchors = torch.cat(
                    [validation_anchors, current_validation_anchors], dim=0)
                validation_targets = torch.cat(
                    [validation_targets, current_validation_targets], dim=0)
            validation_anchor_count = (
                0 if validation_targets is None else validation_targets.numel())
            class_to_task = {
                int(class_id): int(seen_task)
                for seen_task in range(task_id + 1)
                for class_id in class_mask[seen_task]
            }
            selected_alpha, validation_metrics = _select_crct_validation_alpha(
                base_model, teacher_fc_norm, teacher_head,
                validation_anchors, validation_targets,
                seen_classes, old_classes, class_to_task, args,
                current_classes=(current_classes if current_real_gate_enabled else None))
        if dist.is_initialized():
            alpha_tensor = torch.tensor(
                selected_alpha, dtype=torch.float32, device=device)
            dist.broadcast(alpha_tensor, src=0)
            selected_alpha = float(alpha_tensor.item())
        _blend_crct_classifier(base_model, teacher_head, selected_alpha)
        if utils.is_main_process():
            print(
                'CRCT validation gate:', 'alpha=', selected_alpha,
                'anchors=', validation_anchor_count, 'metrics=', validation_metrics)
    elif trust_region_enabled:
        selected_alpha = 1.0
        trust_metrics = None
        trust_anchor_count = 0
        if not dist.is_initialized() or utils.is_main_process():
            trust_anchors, trust_targets = _build_crct_trust_anchors(
                cls_mean, cls_cov, old_classes, args, device)
            trust_anchor_count = 0 if trust_targets is None else trust_targets.numel()
            selected_alpha, trust_metrics = _select_crct_adaptive_trust_alpha(
                base_model, teacher_fc_norm, teacher_head,
                trust_anchors, trust_targets, seen_classes, old_classes, args)
        if dist.is_initialized():
            alpha_tensor = torch.tensor(selected_alpha, dtype=torch.float32, device=device)
            dist.broadcast(alpha_tensor, src=0)
            selected_alpha = float(alpha_tensor.item())
        _blend_crct_classifier(base_model, teacher_head, selected_alpha)
        if utils.is_main_process():
            print(
                'CRCT adaptive trust region:',
                'alpha=', selected_alpha,
                'anchors=', trust_anchor_count,
                'kl=', None if trust_metrics is None else trust_metrics['kl'],
                'conf_drop=', None if trust_metrics is None else trust_metrics['conf_drop'],
                'margin_drop=', None if trust_metrics is None else trust_metrics['margin_drop'],
            )

    for parameter, requires_grad in fc_norm_grad_states:
        parameter.requires_grad_(requires_grad)


def orth_loss(features, targets, device, args):
    if cls_mean:
        # orth loss of this batch
        sample_mean = []
        for k, v in cls_mean.items():
            if isinstance(v, list):
                sample_mean.extend(v)
            else:
                sample_mean.append(v)
        sample_mean = torch.stack(sample_mean, dim=0).to(device, non_blocking=True)
        M = torch.cat([sample_mean, features], dim=0)
        sim = torch.matmul(M, M.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        # print(loss)
        return args.reg * loss
    else:
        sim = torch.matmul(features, features.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        return args.reg * loss
        # return 0.

def add_gaussian_noise(tensor, mean=0., std=1.):
    noise = torch.randn_like(tensor) * std + mean
    tensor_noisy = tensor + 0.01*noise
    return tensor_noisy


def robust_loss(model, inputs, features, targets, devices, task_id, class_mask,index):
    all_old_logits = []
    bs = inputs.shape[0]
    mask = []
    for i in range(task_id+1):
        mask.extend(class_mask[i])

    
    for k in range(index.shape[1]):
        prompt_id = index[:,k]
        with torch.no_grad():

            output = model(inputs, task_id=prompt_id)
            
            old_logits = output['features']
        old_logits = 1*old_logits+0.0*features


        old_norm_features = F.normalize(output['features'], p=2, dim=1)
        old_similarity_matrix = torch.mm(old_norm_features, old_norm_features.t())
        # old_similarity_matrix.fill_diagonal_(1)
        old_similarity_matrix = torch.exp(old_similarity_matrix)
        old_similarity_matrix = old_similarity_matrix / old_similarity_matrix.sum(1, keepdim=True)
        #old_logits = F.softmax(old_logits,dim=1)
        #all_old_logits.append(old_logits)
        all_old_logits.append(old_similarity_matrix)

    
    output = all_old_logits
    return output
