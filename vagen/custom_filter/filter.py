from verl import DataProto
from typing import Any
from enum import Enum
FILTER_REGISTRY: dict[str, Any] = {}
from verl import DataProto

def register_filter(name_or_enum: str) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in FILTER_REGISTRY and FILTER_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {FILTER_REGISTRY[name]} vs {fn}"
            )
        FILTER_REGISTRY[name] = fn
        return fn

    return decorator

@register_filter("reward_variance")
def reward_variance_filter(data_proto: DataProto, metrics, **kwargs) -> tuple[DataProto, dict]:
    """A filter that removes samples with low reward variance.

    This filter computes the reward variance for each group, then keeps only
    the samples from groups with the highest variance (top-k groups based on topk ratio).

    Args:
        data_proto: `(DataProto)`
            The input data proto.
        metrics: update metrics if needed for logging purpose
        **kwargs: additional keyword arguments
            - topk: float, the ratio of groups to keep (default: 0.2)
            - ddof: int, degrees of freedom for variance calculation (default: 0)

    Returns:
        `(tuple[DataProto, dict])`
            The filtered data proto and updated metrics.
    """
    import numpy as np
    import torch
    from collections import defaultdict

    # Get parameters
    topk_ratio = kwargs.get("topk", 0.2)
    ddof = kwargs.get("ddof", 0)

    # Get data
    token_level_scores = data_proto.batch["token_level_scores"]
    group_idx = (
        data_proto.non_tensor_batch["group_idx"]
        if "group_idx" in data_proto.non_tensor_batch
        else data_proto.non_tensor_batch["uid"]
    )

    # 1) Calculate total reward per sample
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).detach().cpu().numpy()
    else:
        total_rewards = np.asarray(token_level_scores).sum(axis=-1)

    # 2) Group rewards by group_idx and track sample indices
    group_rewards = defaultdict(list)
    group_sample_indices = defaultdict(list)

    for sample_idx, (gid, reward) in enumerate(zip(group_idx, total_rewards)):
        gid_str = str(gid)
        group_rewards[gid_str].append(float(reward))
        group_sample_indices[gid_str].append(sample_idx)

    # 3) Calculate variance for each group
    group_variances = {}
    for gid, rewards in group_rewards.items():
        if len(rewards) <= 1:
            group_variances[gid] = 0.0
        else:
            group_variances[gid] = float(np.var(rewards, ddof=ddof))

    # 4) Select top-k groups by variance
    num_groups = len(group_variances)
    num_keep_groups = max(1, int(np.ceil(num_groups * topk_ratio)))

    # Sort groups by variance in descending order
    sorted_groups = sorted(group_variances.items(), key=lambda x: x[1], reverse=True)
    top_groups = set(gid for gid, _ in sorted_groups[:num_keep_groups])

    # 5) Collect indices of samples in top groups
    keep_indices = []
    for gid in top_groups:
        keep_indices.extend(group_sample_indices[gid])

    # Sort indices to maintain order
    keep_indices = sorted(keep_indices)

    # 6) Filter the data_proto
    filtered_data = data_proto.select_idxs(keep_indices)

    # 7) Update metrics
    if metrics is not None:
        original_size = len(data_proto)
        filtered_size = len(filtered_data)

        # User requested metrics
        kept_variances = [group_variances[gid] for gid in top_groups]
        metrics["filter/train/topk/mean_variance"] = float(np.mean(kept_variances))
        metrics["filter/train/topk/min_variance"] = float(np.min(kept_variances))
        metrics["filter/train/topk/max_variance"] = float(np.max(kept_variances))
        
        metrics["filter/train/original/mean_variance"] = float(np.mean(list(group_variances.values())))
        metrics["filter/train/original/min_variance"] = float(np.min(list(group_variances.values())))
        metrics["filter/train/original/max_variance"] = float(np.max(list(group_variances.values())))
        metrics["filter/train/original/batch_size"] = original_size


    return filtered_data, metrics


@register_filter("mixed_outcome")
def mixed_outcome_filter(data_proto: DataProto, metrics, **kwargs) -> tuple[DataProto, dict]:
    """InteriorGS v10, round 11 (the optimisation package): train only on the groups (the episodes of one task)
    whose episodes both succeed and fail (``traj_success``); a group that all succeeds or all fails gives a group
    comparison nothing to learn from.

    The update draws more tasks than it trains on (data.train_batch_size candidates).  Taken in draw order
    (meta_info["uid_draw_order"], uid -> position; else the order of first appearance), the first
    ``target_groups`` candidates are the nominal batch; each of them that all succeeds or all fails is dropped and
    replaced by the next mixed group among the extra candidates (the refill; at most the number of extra
    candidates).  If no group is mixed, the nominal batch is kept with its advantages set to zero (the update
    then only applies the entropy and KL terms).

    kwargs: target_groups (default: all candidates), success_key (default "traj_success").
    Metrics (filter/mixed/): candidates, nominal, kept, dropped_all_success and dropped_all_fail (among the
    nominal batch), refilled, shortfall (target - kept), mixed_share (of all candidates), empty_fallback.
    """
    import numpy as np

    key = kwargs.get("success_key", "traj_success")
    uids = [str(u) for u in data_proto.non_tensor_batch["uid"]]
    success = np.asarray(data_proto.non_tensor_batch[key], dtype=float) > 0.5
    draw = data_proto.meta_info.get("uid_draw_order") or {}
    groups: dict = {}
    for i, u in enumerate(uids):
        groups.setdefault(u, []).append(i)
    first = {u: idx[0] for u, idx in groups.items()}
    ordered = sorted(groups, key=lambda u: (draw.get(u, len(draw) + first[u]), first[u]))
    target = int(kwargs.get("target_groups") or 0) or len(ordered)

    def kind(u: str) -> str:
        s = success[groups[u]]
        return "mixed" if s.any() and not s.all() else ("all_success" if s.all() else "all_fail")

    kinds = {u: kind(u) for u in ordered}
    nominal, extra = ordered[:target], ordered[target:]
    kept = [u for u in nominal if kinds[u] == "mixed"]
    refill = [u for u in extra if kinds[u] == "mixed"][: max(0, target - len(kept))]
    keep = kept + refill
    fallback = not keep
    if fallback:
        keep = list(nominal)
    filtered = data_proto.select_idxs(sorted(i for u in keep for i in groups[u]))
    if fallback and "advantages" in filtered.batch.keys():
        filtered.batch["advantages"] = filtered.batch["advantages"] * 0.0
    if metrics is not None:
        n_mixed = sum(1 for k in kinds.values() if k == "mixed")
        metrics.update({
            "filter/mixed/candidates": len(ordered), "filter/mixed/nominal": len(nominal),
            "filter/mixed/kept": 0 if fallback else len(keep),
            "filter/mixed/dropped_all_success": sum(1 for u in nominal if kinds[u] == "all_success"),
            "filter/mixed/dropped_all_fail": sum(1 for u in nominal if kinds[u] == "all_fail"),
            "filter/mixed/refilled": len(refill),
            "filter/mixed/shortfall": target - (0 if fallback else len(keep)),
            "filter/mixed/mixed_share": n_mixed / max(1, len(ordered)),
            "filter/mixed/empty_fallback": float(fallback),
        })
    return filtered, metrics


@register_filter("reward_variance_top_p")
def reward_variance_top_p_filter(data_proto: DataProto, metrics, **kwargs) -> tuple[DataProto, dict]:
    """A filter that selects samples using top-p (nucleus) sampling based on reward variance.

    This filter computes the reward variance for each group, then selects groups
    in descending order of variance until the cumulative variance reaches top_p * total_variance.
    Similar to LLM top-p sampling, but applied to reward variance selection.

    The loss is scaled by sqrt(selected_count / total_count) to account for the reduced batch size.

    Args:
        data_proto: `(DataProto)`
            The input data proto.
        metrics: update metrics if needed for logging purpose
        **kwargs: additional keyword arguments
            - top_p: float, the cumulative variance threshold (default: 0.9)
            - ddof: int, degrees of freedom for variance calculation (default: 0)

    Returns:
        `(tuple[DataProto, dict])`
            The filtered data proto and updated metrics.
    """
    import numpy as np
    import torch
    from collections import defaultdict

    # Get parameters
    top_p = kwargs.get("top_p", 0.9)
    ddof = kwargs.get("ddof", 0)

    # Get data
    token_level_scores = data_proto.batch["token_level_scores"]
    group_idx = (
        data_proto.non_tensor_batch["group_idx"]
        if "group_idx" in data_proto.non_tensor_batch
        else data_proto.non_tensor_batch["uid"]
    )

    # 1) Calculate total reward per sample
    if isinstance(token_level_scores, torch.Tensor):
        total_rewards = token_level_scores.sum(dim=-1).detach().cpu().numpy()
    else:
        total_rewards = np.asarray(token_level_scores).sum(axis=-1)

    # 2) Group rewards by group_idx and track sample indices
    group_rewards = defaultdict(list)
    group_sample_indices = defaultdict(list)

    for sample_idx, (gid, reward) in enumerate(zip(group_idx, total_rewards)):
        gid_str = str(gid)
        group_rewards[gid_str].append(float(reward))
        group_sample_indices[gid_str].append(sample_idx)

    # 3) Calculate variance for each group
    group_variances = {}
    for gid, rewards in group_rewards.items():
        if len(rewards) <= 1:
            group_variances[gid] = 0.0
        else:
            group_variances[gid] = float(np.var(rewards, ddof=ddof))

    # 4) Sort groups by variance in descending order (like top-p sampling)
    sorted_groups = sorted(group_variances.items(), key=lambda x: x[1], reverse=True)

    # 5) Select groups until cumulative variance reaches top_p * total_variance
    total_variance = sum(group_variances.values())

    if total_variance <= 0:
        # If no variance at all, keep all samples
        return data_proto, metrics

    target_variance = top_p * total_variance
    cumulative_variance = 0.0
    top_groups = set()

    for gid, var in sorted_groups:
        if cumulative_variance >= target_variance:
            break
        top_groups.add(gid)
        cumulative_variance += var

    # Ensure at least one group is selected
    if len(top_groups) == 0 and len(sorted_groups) > 0:
        top_groups.add(sorted_groups[0][0])

    # 6) Collect indices of samples in top groups
    keep_indices = []
    for gid in top_groups:
        keep_indices.extend(group_sample_indices[gid])

    # Sort indices to maintain order
    keep_indices = sorted(keep_indices)

    # 7) Calculate loss scale: sqrt(selected_groups / total_groups)
    num_total_groups = len(group_variances)
    num_selected_groups = len(top_groups)
    loss_scale = np.sqrt(num_selected_groups / num_total_groups) if num_total_groups > 0 else 1.0

    # 8) Apply loss scale to advantages before filtering
    # 9) Filter the data_proto
    filtered_data = data_proto.select_idxs(keep_indices)

    # Apply loss scale to advantages after filtering (only to kept samples)
    if "advantages" in filtered_data.batch:
        filtered_data.batch["advantages"] = filtered_data.batch["advantages"] * loss_scale

    # 10) Update metrics
    if metrics is not None:
        original_size = len(data_proto)
        filtered_size = len(filtered_data)

        # User requested metrics
        kept_variances = [group_variances[gid] for gid in top_groups]
        metrics["filter/train/top_p/mean_variance"] = float(np.mean(kept_variances)) if kept_variances else 0.0
        metrics["filter/train/top_p/min_variance"] = float(np.min(kept_variances)) if kept_variances else 0.0
        metrics["filter/train/top_p/max_variance"] = float(np.max(kept_variances)) if kept_variances else 0.0
        metrics["filter/train/top_p/cumulative_variance_ratio"] = cumulative_variance / total_variance if total_variance > 0 else 0.0
        metrics["filter/train/top_p/num_selected_groups"] = num_selected_groups
        metrics["filter/train/top_p/num_total_groups"] = num_total_groups
        metrics["filter/train/top_p/loss_scale"] = loss_scale

        metrics["filter/train/original/mean_variance"] = float(np.mean(list(group_variances.values())))
        metrics["filter/train/original/min_variance"] = float(np.min(list(group_variances.values())))
        metrics["filter/train/original/max_variance"] = float(np.max(list(group_variances.values())))
        metrics["filter/train/original/batch_size"] = original_size
        metrics["filter/train/filtered/batch_size"] = filtered_size

    return filtered_data, metrics

