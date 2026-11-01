"""Reward line (the user's decisions of 2026-11-01, docs/reward/reports/PLAN_TURN_CREDIT_20261101.md): per-turn credit.

GRPO gives every token of an episode the episode's advantage (the episode's summed reward compared within the task's
group).  With this estimator (PPO config ``algorithm.adv_estimator: turn_credit``, settings ``algorithm.turn_credit``)
each turn gets its own credit

    c_t = r_t + scale * (gamma * phi_{t+1} - phi_t)

r_t the turn's env reward (the S0 reward with the env's progress reward off: success, unreadable-answer and collision
terms), phi the env's progress score (the depth-12 table's optimal progress with ``progress_version: optimal``), phi_0
the score at reset and phi_{t+1} = 1 on the turn that succeeds.  The turn's advantage

    A_t = clip((c_t - b_t) / s, -neg_clip) , negatives scaled so the positive token mass is at least
          min_positive_mass x the negative one, plus episode_weight x the episode's GRPO advantage

b_t the baseline ("turn_index": the mean credit of the task group's episodes at the same turn position; "none": 0),
s the standard deviation of (c - b) over the batch's turns.  Every token of the turn's answer gets A_t (observation
tokens keep 0); with episode_weight > 0 every answer token also gets episode_weight x the episode's GRPO advantage
(the episode's summed reward, std n - 1, as verl's grpo).  The agent loop records each turn ({span, reward,
phi_after, success, valid}, ``extra_fields["turn_records"]``) when the estimator is selected.

Pure functions (torch / numpy); vagen/custom_advantage/turn_credit.py registers the estimator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ESTIMATOR = "turn_credit"
DEFAULTS = {"scale": 10.0 / 3.0, "gamma": 0.99, "baseline": "turn_index", "episode_weight": 1.0,
            "min_positive_mass": 0.4, "neg_clip": 3.0, "eps": 1e-6}
BASELINES = ("none", "turn_index")


def settings(config: Any) -> Dict[str, Any]:
    """The estimator's settings from an algorithm config (``turn_credit`` sub-dict), defaults filled in."""
    getter = getattr(config, "get", None)
    raw = (getter("turn_credit", None) if callable(getter) else getattr(config, "turn_credit", None)) or {}
    out = dict(DEFAULTS)
    for k in DEFAULTS:
        v = raw.get(k) if hasattr(raw, "get") else getattr(raw, k, None)
        if v is not None:
            out[k] = v
    if out["baseline"] not in BASELINES:
        raise ValueError(f"turn_credit.baseline must be one of {BASELINES}, not {out['baseline']!r}")
    for k in ("scale", "gamma", "episode_weight", "min_positive_mass", "neg_clip", "eps"):
        out[k] = float(out[k])
    return out


def selected(config: Any) -> bool:
    """Whether a full config (or its algorithm part) selects this estimator."""
    if config is None:
        return False
    algo = config.get("algorithm", config) if hasattr(config, "get") else getattr(config, "algorithm", config)
    getter = getattr(algo, "get", None)
    est = getter("adv_estimator", None) if callable(getter) else getattr(algo, "adv_estimator", None)
    return str(getattr(est, "value", est)) == ESTIMATOR


def turn_record(span: Optional[Sequence[int]], reward: float, info: Optional[Dict[str, Any]], success: bool,
                valid: bool) -> Dict[str, Any]:
    """What the agent loop keeps of one turn: its answer's response positions [start, end), the env reward, the env's
    progress score after the turn (info current_potential_score), whether the task is met and whether the answer was
    readable."""
    phi = (info or {}).get("current_potential_score") if isinstance(info, dict) else None
    return {"span": [int(span[0]), int(span[1])] if span else [0, 0], "reward": float(reward),
            "phi_after": float(phi) if phi is not None else 0.0, "success": bool(success), "valid": bool(valid)}


def episode_record(phi0: float, turns: Sequence[Dict[str, Any]], response_length: int) -> Dict[str, Any]:
    """The per-sample extra field (a dict: the trainer packs extra fields into an object array): the score at reset
    and the turns, spans clipped to the response length (a turn cut off entirely keeps its credit, no tokens)."""
    out = []
    for t in turns:
        s, e = t["span"]
        out.append(dict(t, span=[min(int(s), response_length), min(int(e), response_length)]))
    return {"phi0": float(phi0), "turns": out}


def turn_credits(record: Dict[str, Any], scale: float, gamma: float) -> List[float]:
    """The credits of one episode's turns from its record {"phi0": phi at reset, "turns": [{reward, phi_after,
    success, ...}]}: c_t = r_t + scale * (gamma * phi_{t+1} - phi_t), phi_{t+1} = 1 on success."""
    phi = float(record.get("phi0") or 0.0)
    out = []
    for t in record.get("turns") or []:
        nxt = 1.0 if t.get("success") else float(t.get("phi_after") or 0.0)
        out.append(float(t.get("reward") or 0.0) + scale * (gamma * nxt - phi))
        phi = nxt
    return out


def baselines(credits: Sequence[Sequence[float]], uids: Sequence[Any], kind: str) -> List[List[float]]:
    """Per episode and turn the baseline: "turn_index" the mean credit of the episodes of the same uid at the same turn
    position (only the episodes that reached it), "none" 0."""
    if kind == "none":
        return [[0.0] * len(c) for c in credits]
    pool: Dict[Tuple[Any, int], List[float]] = defaultdict(list)
    for u, cs in zip(uids, credits):
        for t, c in enumerate(cs):
            pool[(u, t)].append(c)
    means = {k: float(np.mean(v)) for k, v in pool.items()}
    return [[means[(u, t)] for t in range(len(cs))] for u, cs in zip(uids, credits)]


def episode_advantages(returns: Sequence[float], uids: Sequence[Any], eps: float = 1e-6) -> List[float]:
    """verl's grpo on episode returns: (R - group mean) / (group std n - 1 + eps); a lone episode gets 0."""
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, u in enumerate(uids):
        groups[u].append(i)
    out = [0.0] * len(returns)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        r = np.asarray([returns[i] for i in idx], dtype=np.float64)
        sd = float(np.std(r, ddof=1))
        for i, v in zip(idx, r):
            out[i] = float((v - r.mean()) / (sd + eps))
    return out


def turn_advantages(records: Sequence[Dict[str, Any]], uids: Sequence[Any], cfg: Dict[str, Any],
                    lengths: Optional[Sequence[Sequence[int]]] = None) -> Tuple[List[List[float]], Dict[str, float]]:
    """Per episode and turn the advantage A_t (before the episode term), and metrics.

    ``lengths``: per episode and turn the number of answer tokens (the positive / negative mass is counted in tokens,
    as the token-mean loss weighs them); default 1 a turn."""
    credits = [turn_credits(r, cfg["scale"], cfg["gamma"]) for r in records]
    base = baselines(credits, uids, cfg["baseline"])
    cen = [[c - b for c, b in zip(cs, bs)] for cs, bs in zip(credits, base)]
    flat = np.asarray([x for xs in cen for x in xs], dtype=np.float64)
    sd = float(flat.std()) if flat.size > 1 else 1.0
    raw = [[x / (sd + cfg["eps"]) for x in xs] for xs in cen]
    clipped = sum(x < -cfg["neg_clip"] for xs in raw for x in xs)
    z = [[max(x, -cfg["neg_clip"]) for x in xs] for xs in raw]
    lens = lengths or [[1] * len(xs) for xs in z]
    pos = sum(max(a, 0.0) * n for xs, ns in zip(z, lens) for a, n in zip(xs, ns))
    neg = sum(max(-a, 0.0) * n for xs, ns in zip(z, lens) for a, n in zip(xs, ns))
    neg_scale = 1.0
    if neg > 0 and pos < cfg["min_positive_mass"] * neg:
        neg_scale = pos / (cfg["min_positive_mass"] * neg)
        z = [[a * neg_scale if a < 0 else a for a in xs] for xs in z]
    turns = [t for xs in z for t in xs]
    n = max(1, len(turns))
    metrics = {"turn_credit/turns": float(len(turns)), "turn_credit/credit_mean": float(np.mean([c for cs in credits for c in cs])) if turns else 0.0,
               "turn_credit/centered_std": sd, "turn_credit/pos_share": sum(a > 1e-9 for a in turns) / n,
               "turn_credit/neg_share": sum(a < -1e-9 for a in turns) / n,
               "turn_credit/clipped_share": clipped / n,
               "turn_credit/pos_mass_over_neg": (pos / neg) if neg > 0 else 0.0, "turn_credit/neg_scale": neg_scale}
    late = [a for xs in z for t, a in enumerate(xs) if t >= 3]
    metrics["turn_credit/late_pos_share"] = (sum(a > 1e-9 for a in late) / len(late)) if late else 0.0
    return z, metrics


def token_advantages(records: Sequence[Dict[str, Any]], uids: Sequence[Any], response_mask: torch.Tensor,
                     episode_returns: Sequence[float], cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The (batch, response length) advantages: each turn's span gets its A_t, every answer token (response_mask) also
    episode_weight x the episode's GRPO advantage; tokens outside every span get only the episode term."""
    bs, length = response_mask.shape
    spans = [[[max(0, int(t["span"][0])), min(length, int(t["span"][1]))] for t in (r.get("turns") or [])]
             for r in records]
    lengths = [[max(0, e - s) for s, e in sp] for sp in spans]
    z, metrics = turn_advantages(records, uids, cfg, lengths)
    adv = torch.zeros((bs, length), dtype=torch.float32)
    for i, (sp, zs) in enumerate(zip(spans, z)):
        for (s, e), a in zip(sp, zs):
            if e > s:
                adv[i, s:e] = float(a)
    if cfg["episode_weight"]:
        ep = torch.tensor(episode_advantages(episode_returns, uids, cfg["eps"]), dtype=torch.float32)
        adv = adv + cfg["episode_weight"] * ep.unsqueeze(-1)
    mask = response_mask.to(torch.float32)
    adv = adv.to(response_mask.device) * mask.to(response_mask.device)
    covered = sum(sum(n for n in ns) for ns in lengths)
    metrics["turn_credit/span_token_share"] = covered / max(1.0, float(mask.sum().item()))
    return adv, metrics
