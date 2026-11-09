"""Turn-level GAE for the concat history mode (the main line's per-turn PPO, the user's decision 1A of 2026-11-09).

In the concat mode an episode is one sequence and the agent loop puts the sum of the turns' rewards on its last
response token (gym_agent_loop: reward_score = sum(env_rewards); verl's agent loop writes it at the last valid
position).  verl's GAE then discounts every response token (gamma, lam per token), so with gamma = lam = 0.95 and
about 1,400 response tokens the first turn sees almost nothing of the episode's reward, and a dense reward reaches the
update only through the episode total, exactly as under GRPO.

This estimator works on turns instead (algorithm.adv_estimator "turn_gae"):
- the agent loop records every generated turn's response span [start, end) and the env's reward for it
  (extra field "turn_rewards" = {"spans": [[start, end], ...], "rewards": [r_1, ...]}, only with this estimator);
- a turn's value V_t is the value network's output at the turn's first response token (the critic's value at a
  response position comes from the hidden state before that token: the state after the turn's observation);
- delta_t = r_t + gamma * V_{t+1} - V_t with V_{T+1} = 0 (the episode's last turn ends it, also at the turn or
  action limit, as the episode total does under GRPO), A_t = sum_k (gamma * lam)^k delta_{t+k}; gamma and lam count
  turns, not tokens;
- every response token of turn t gets A_t; the return A_t + V_t supervises the value network only at the turn's
  first token, every other position holds the sentinel -100 (dp_critic and compute_value_mask skip it);
- the estimator returns the raw advantages; the trainer whitens them after the group filter, over the response
  tokens of the episodes the update keeps (whiten), so the tasks the filter drops do not shift them (the two reviews
  of 2026-11-09; verl's GAE, and so p1gae, whitens before the filter), and keep_by_return_std does not apply its
  std-floor scale (a correction for GRPO's division by the group's spread, which this estimator does not do).
A turn with no response position left (cut off by the response length, or an empty answer) gives its reward to the
turn before it (to the next kept turn when it is the first), so a sample's turn rewards always sum to its episode
reward; the estimator checks that sum against the reward tensor and refuses a mismatch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ESTIMATOR = "turn_gae"
RECORD_KEY = "turn_rewards"
IGNORE_VALUE = -100.0


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else getattr(config, key, default)


def enabled(config: Any) -> bool:
    """Whether an algorithm config (DictConfig, dict or dataclass; missing: no) names this estimator."""
    est = _get(config, "adv_estimator", None)
    return str(getattr(est, "value", est)) == ESTIMATOR


def check_config(config: Any) -> None:
    """Refuse algorithm settings this estimator cannot honour: a per-token KL reward (dropped: the turn rewards come
    from the env), the rollout correction's rejection / token veto (they rewrite the response mask, so a turn's
    anchor would move), the invalid-turn clamp (it would act on the advantages before they are whitened)."""
    if bool(_get(config, "use_kl_in_reward", False)):
        raise ValueError("turn_gae puts the env's turn rewards on turns; a per-token KL reward would be dropped")
    rc = _get(config, "rollout_correction", None) or {}
    for key in ("rollout_rs", "rollout_token_veto_threshold"):
        if _get(rc, key, None) is not None:
            raise ValueError(f"turn_gae cannot run with algorithm.rollout_correction.{key} (it rewrites the response mask)")
    if bool(_get(config, "invalid_turn_nonpositive_adv", False)):
        raise ValueError("turn_gae cannot run with algorithm.invalid_turn_nonpositive_adv")


def whiten(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """verl's masked_whiten over the response tokens (mean 0, unbiased std 1 there), 0 elsewhere.  The trainer calls
    it after the group filter, on the episodes the update keeps (vagen/ray_trainer.py)."""
    mask = response_mask.to(dtype=advantages.dtype, device=advantages.device)
    n = mask.sum()
    if n < 2:
        return advantages * mask
    mean = (advantages * mask).sum() / n
    var = (((advantages - mean) ** 2) * mask).sum() / (n - 1)
    return (advantages - mean) * torch.rsqrt(var + 1e-8) * mask


def turn_record(spans: Sequence[Sequence[int]], rewards: Sequence[float]) -> Dict[str, List]:
    """The agent loop's per-sample record: one [start, end) response span and one env reward per generated turn."""
    if len(spans) != len(rewards):
        raise ValueError(f"turn_gae: {len(spans)} generated turns but {len(rewards)} env rewards")
    return {"spans": [[int(s[0]), int(s[1])] for s in spans], "rewards": [float(r) for r in rewards]}


def turn_anchors(record: Optional[Dict[str, Any]], response_mask_row: torch.Tensor
                 ) -> Tuple[List[Tuple[int, int, float]], int]:
    """(start, end, reward) of the turns that keep at least one response position, and how many turns were folded.

    Spans are clipped to the response length; a turn's anchor is its first position with response mask 1 (the
    first generated token).  A turn without such a position gives its reward to the kept turn before it (to the first
    kept turn after it when none comes before)."""
    if record is None or "spans" not in record or "rewards" not in record:
        raise KeyError("turn_gae: a sample has no turn record (the agent loop records it only with this estimator)")
    mask = [bool(x) for x in response_mask_row.tolist()]
    length = len(mask)
    covered = [False] * length  # the clipped spans must cover exactly the generated (mask 1) positions
    for start, end in record["spans"]:
        for p in range(max(0, int(start)), min(length, int(end))):
            covered[p] = True
    if covered != mask:
        raise ValueError("turn_gae: the turn spans do not cover exactly the sample's generated tokens "
                         "(a record from another sample, or a changed response mask)")
    kept: List[List[Any]] = []
    pending = 0.0  # rewards of dropped turns before the first kept one
    folded = 0
    for (start, end), reward in zip(record["spans"], record["rewards"]):
        s, e = max(0, int(start)), min(length, int(end))
        first = next((p for p in range(s, e) if mask[p]), None) if e > s else None
        if first is not None:
            kept.append([first, e, float(reward) + pending])
            pending = 0.0
        else:
            folded += 1
            if kept:
                kept[-1][2] += float(reward)
            else:
                pending += float(reward)
    if pending and not kept:
        raise ValueError("turn_gae: a sample has turn rewards but no response position to put them on")
    return [(int(a), int(e), float(r)) for a, e, r in kept], folded


def turn_gae(rewards: Sequence[float], values: Sequence[float], gamma: float, lam: float
             ) -> Tuple[List[float], List[float]]:
    """GAE over one episode's turns: (advantages, returns), V after the last turn 0."""
    adv = [0.0] * len(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, [a + v for a, v in zip(adv, values)]


def compute(token_level_rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor,
            records: Sequence[Optional[Dict[str, Any]]], gamma: float, lam: float,
            ignore_value: float = IGNORE_VALUE, atol: float = 1e-3
            ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Raw (not whitened) token advantages, returns with the sentinel off the anchors, and metrics."""
    bs, length = token_level_rewards.shape
    if len(records) != bs:
        raise ValueError(f"turn_gae: {len(records)} turn records for {bs} samples")
    device = token_level_rewards.device
    adv = torch.zeros((bs, length), dtype=torch.float32, device=device)
    ret = torch.full((bs, length), float(ignore_value), dtype=torch.float32, device=device)
    totals = token_level_rewards.detach().float().sum(dim=-1).cpu()
    vals = values.detach().float().cpu()
    mask_cpu = response_mask.detach().cpu()
    n_turns, n_folded, max_diff, first_values = 0, 0, 0.0, []
    for i in range(bs):
        anchors, folded = turn_anchors(records[i], mask_cpu[i])
        n_folded += folded
        n_turns += len(anchors)
        rewards = [r for _, _, r in anchors]
        diff = abs(sum(rewards) - float(totals[i]))
        max_diff = max(max_diff, diff)
        if diff > atol + 1e-5 * abs(float(totals[i])):
            raise ValueError(f"turn_gae: sample {i}: turn rewards sum to {sum(rewards):.6f}, "
                             f"the reward tensor to {float(totals[i]):.6f}")
        if not anchors:
            continue
        v = [float(vals[i, a]) for a, _, _ in anchors]
        first_values.append(v[0])
        a_t, r_t = turn_gae(rewards, v, float(gamma), float(lam))
        for (start, end, _), a_val, r_val in zip(anchors, a_t, r_t):
            seg = mask_cpu[i, start:end].to(device=device, dtype=torch.float32)
            adv[i, start:end] = seg * a_val  # response tokens of the turn only (an observation inside stays 0)
            ret[i, start] = r_val
    metrics = {"critic/turn_gae/turns_per_sample": n_turns / max(1, bs),
               "critic/turn_gae/folded_turns": float(n_folded),
               "critic/turn_gae/reward_sum_max_diff": max_diff,
               "critic/turn_gae/first_turn_value_mean": (sum(first_values) / len(first_values)) if first_values else 0.0}
    return adv, ret, metrics
