"""Reward line: per-turn credit (phase 2, direction 1, the user's brief of 2026-11-01; the plan of the same day,
docs/reward/reports/PLAN_TURN_CREDIT_20261101.md, is its first form).

GRPO gives every token of an episode the episode's advantage (the episode's summed reward compared within the task's
group).  With this estimator (PPO config ``algorithm.adv_estimator: turn_credit``, settings ``algorithm.turn_credit``)
each turn also gets its own term.  The turn's value

    v_t = scale * (gamma * phi_{t+1} - phi_t)  [+ r_t with turn_reward]  [- stall_penalty when stalled]

phi the env's progress score (the depth-12 table's optimal progress with ``progress_version: optimal``), phi_0 the
score at reset and phi_{t+1} = 1 on the turn that succeeds; r_t the turn's env reward (S0: success, unreadable answer,
collision); "stalled": the table distance unchanged over this turn and the one before.  The advantage of a token of
turn t of episode i

    A = episode_weight * E_i + turn_weight * T_t

E_i the episode's GRPO advantage on its summed env reward (the S0 return; std n - 1, a group of equal returns gives 0
and stays in the batch), divided by its standard deviation over the batch's episodes with episode_norm "batch";
T_t = (v_t - b_t) / s, b_t the baseline ("batch_mean": the mean over the batch's turns, so T has mean 0 and variance 1
over the batch; "turn_index": the mean of the task group's episodes at the same turn position; "none": 0), s the
standard deviation of v - b over the batch's turns; optionally T clipped at -neg_clip and its negatives scaled so the
positive token mass is at least min_positive_mass x the negative one (0 = off).  Observation tokens keep 0.

Defaults are the brief's first round (turn term the shaping alone, batch_mean, turn_weight 0.5, episode term batch
normalised, no clip, no protection, no stall penalty); the 11-01 plan's TC-mix is turn_reward true, baseline
turn_index, turn_weight 1, episode_norm none, neg_clip 3, min_positive_mass 0.4.

The agent loop records each turn ({span, reward, phi_after, d_after, success, valid}, ``extra_fields["turn_records"]``)
when the estimator is selected.  Pure functions (torch / numpy); vagen/custom_advantage/turn_credit.py registers the
estimator.

Phase 2 direction 3 (PLAN_DIR3_20261103.md section 4): with ``turn_term: "statement"`` the turn's value is its
statement's accuracy (0..1, vagen/statement_truth.py: the env's statement_record) and T acts on the statement's own
tokens (the record's statement_span; the whole answer when the answer has no readable statement), so the statement
score never reaches the action tokens; with episode_norm "none" every token keeps verl's GRPO advantage of the S0
return, the S0 run's.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ESTIMATOR = "turn_credit"
DEFAULTS = {"scale": 10.0 / 3.0, "gamma": 0.99, "turn_reward": False, "baseline": "batch_mean", "turn_weight": 0.5,
            "episode_weight": 1.0, "episode_norm": "batch", "neg_clip": 0.0, "min_positive_mass": 0.0,
            "stall_penalty": 0.0, "eps": 1e-6, "dump_dir": None, "dump_updates": 0, "turn_term": "shaping"}
BASELINES = ("batch_mean", "turn_index", "none")
EPISODE_NORMS = ("batch", "none")
TURN_TERMS = ("shaping", "statement")
LATE_TURN = 3  # turns 4+ (0-based index 3 on)


def settings(config: Any) -> Dict[str, Any]:
    """The estimator's settings from an algorithm config (``turn_credit`` sub-dict), defaults filled in."""
    getter = getattr(config, "get", None)
    raw = (getter("turn_credit", None) if callable(getter) else getattr(config, "turn_credit", None)) or {}
    out = dict(DEFAULTS)
    for k in DEFAULTS:
        v = raw.get(k) if hasattr(raw, "get") else getattr(raw, k, None)
        if v is not None:
            out[k] = v
    unknown = set(raw.keys() if hasattr(raw, "keys") else []) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown turn_credit settings {sorted(unknown)}")
    if out["baseline"] not in BASELINES:
        raise ValueError(f"turn_credit.baseline must be one of {BASELINES}, not {out['baseline']!r}")
    if out["episode_norm"] not in EPISODE_NORMS:
        raise ValueError(f"turn_credit.episode_norm must be one of {EPISODE_NORMS}, not {out['episode_norm']!r}")
    if out["turn_term"] not in TURN_TERMS:
        raise ValueError(f"turn_credit.turn_term must be one of {TURN_TERMS}, not {out['turn_term']!r}")
    if out["turn_term"] == "statement" and (out["turn_reward"] or out["stall_penalty"]):
        raise ValueError("turn_credit.turn_term statement takes neither turn_reward nor stall_penalty")
    for k in ("scale", "gamma", "turn_weight", "episode_weight", "neg_clip", "min_positive_mass", "stall_penalty", "eps"):
        out[k] = float(out[k])
    out["turn_reward"] = bool(out["turn_reward"])
    out["dump_dir"] = str(out["dump_dir"]) if out["dump_dir"] else None  # the smoke's span check (custom_advantage)
    out["dump_updates"] = int(out["dump_updates"])
    return out


def selected(config: Any) -> bool:
    """Whether a full config (or its algorithm part) selects this estimator."""
    if config is None:
        return False
    algo = config.get("algorithm", config) if hasattr(config, "get") else getattr(config, "algorithm", config)
    getter = getattr(algo, "get", None)
    est = getter("adv_estimator", None) if callable(getter) else getattr(algo, "adv_estimator", None)
    return str(getattr(est, "value", est)) == ESTIMATOR


def _distance(metrics: Any) -> Optional[int]:
    d = metrics.get("optimal_distance") if isinstance(metrics, dict) else None
    return int(d) if d is not None else None


def reset_distance(info: Optional[Dict[str, Any]]) -> Optional[int]:
    """The table distance of the start pose (reset info initial_region_metrics), None when unknown."""
    return _distance((info or {}).get("initial_region_metrics")) if isinstance(info, dict) else None


def turn_record(span: Optional[Sequence[int]], reward: float, info: Optional[Dict[str, Any]], success: bool,
                valid: bool, statement_span: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """What the agent loop keeps of one turn: its answer's response positions [start, end), the env reward, the env's
    progress score and table distance after the turn, whether the task is met and whether the answer was readable.

    The score is the turn-end pose's (info current_region_metrics optimal_progress, the table's phi); only without it
    the env's current_potential_score, which the env leaves at the previous turn's value on the turn that uses up the
    step budget (replay of rw_s0: 4.4% of the failed turns, every one of them such a last turn).

    Direction 3: with the env's statement_record (info["statement"]) also the statement's accuracy and its token
    positions (``statement_span``, the agent loop's; None: no readable statement, the whole answer)."""
    info = info if isinstance(info, dict) else {}
    rm = info.get("current_region_metrics")
    phi = rm.get("optimal_progress") if isinstance(rm, dict) else None
    if phi is None:
        phi = info.get("current_potential_score")
    out = {"span": [int(span[0]), int(span[1])] if span else [0, 0], "reward": float(reward),
           "phi_after": float(phi) if phi is not None else 0.0, "d_after": _distance(rm),
           "success": bool(success), "valid": bool(valid)}
    statement = info.get("statement")
    if isinstance(statement, dict):
        out["statement_accuracy"] = float(statement.get("accuracy") or 0.0)
        out["statement_span"] = [int(statement_span[0]), int(statement_span[1])] if statement_span else list(out["span"])
    return out


def episode_record(phi0: float, turns: Sequence[Dict[str, Any]], response_length: int,
                   d0: Optional[int] = None) -> Dict[str, Any]:
    """The per-sample extra field (a dict: the trainer packs extra fields into an object array): the score and table
    distance at reset and the turns, spans clipped to the response length (a turn cut off entirely keeps its value,
    no tokens)."""
    out = []
    for t in turns:
        s, e = t["span"]
        clipped = dict(t, span=[min(int(s), response_length), min(int(e), response_length)])
        if "statement_span" in t:
            s, e = t["statement_span"]
            clipped["statement_span"] = [min(int(s), response_length), min(int(e), response_length)]
        out.append(clipped)
    return {"phi0": float(phi0), "d0": d0, "turns": out}


def stalled(record: Dict[str, Any]) -> List[bool]:
    """Per turn: the table distance unchanged over this turn and the turn before (all three distances known)."""
    ds = [record.get("d0")] + [t.get("d_after") for t in record.get("turns") or []]
    out = []
    for t in range(len(ds) - 1):
        out.append(t >= 1 and None not in (ds[t - 1], ds[t], ds[t + 1]) and ds[t - 1] == ds[t] == ds[t + 1])
    return out


def turn_values(record: Dict[str, Any], scale: float, gamma: float, turn_reward: bool = False,
                stall_penalty: float = 0.0, term: str = "shaping") -> List[float]:
    """The values of one episode's turns: scale * (gamma * phi_{t+1} - phi_t) (phi_{t+1} = 1 on success), plus the
    env reward with turn_reward, minus stall_penalty on a stalled turn; with ``term`` "statement" the statement's
    accuracy (a turn without a statement record: 0)."""
    if term == "statement":
        return [float(t.get("statement_accuracy") or 0.0) for t in record.get("turns") or []]
    phi = float(record.get("phi0") or 0.0)
    stall = stalled(record) if stall_penalty else []
    out = []
    for i, t in enumerate(record.get("turns") or []):
        nxt = 1.0 if t.get("success") else float(t.get("phi_after") or 0.0)
        v = scale * (gamma * nxt - phi)
        if turn_reward:
            v += float(t.get("reward") or 0.0)
        if stall_penalty and stall[i]:
            v -= stall_penalty
        out.append(v)
        phi = nxt
    return out


def turn_credits(record: Dict[str, Any], scale: float, gamma: float) -> List[float]:
    """The 11-01 plan's credit c_t = r_t + scale * (gamma * phi_{t+1} - phi_t)."""
    return turn_values(record, scale, gamma, turn_reward=True)


def baselines(values: Sequence[Sequence[float]], uids: Sequence[Any], kind: str) -> List[List[float]]:
    """Per episode and turn the baseline: "batch_mean" the mean over all turns, "turn_index" the mean of the episodes
    of the same uid at the same turn position (only the episodes that reached it), "none" 0."""
    if kind == "none":
        return [[0.0] * len(v) for v in values]
    if kind == "batch_mean":
        flat = [x for v in values for x in v]
        m = float(np.mean(flat)) if flat else 0.0
        return [[m] * len(v) for v in values]
    pool: Dict[Tuple[Any, int], List[float]] = defaultdict(list)
    for u, vs in zip(uids, values):
        for t, c in enumerate(vs):
            pool[(u, t)].append(c)
    means = {k: float(np.mean(v)) for k, v in pool.items()}
    return [[means[(u, t)] for t in range(len(vs))] for u, vs in zip(uids, values)]


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


def episode_terms(returns: Sequence[float], uids: Sequence[Any], cfg: Dict[str, Any]) -> Tuple[List[float], Dict[str, float]]:
    """E_i: the episode GRPO advantage, divided by its batch standard deviation (mean removed) with episode_norm
    "batch"; and metrics (groups, groups with equal returns, the raw advantage's batch std)."""
    raw = episode_advantages(returns, uids, cfg["eps"])
    groups: Dict[Any, List[float]] = defaultdict(list)
    for u, r in zip(uids, returns):
        groups[u].append(float(r))
    arr = np.asarray(raw, dtype=np.float64)
    sd = float(arr.std()) if arr.size > 1 else 0.0
    if cfg["episode_norm"] == "batch" and arr.size:
        out = [float(x) for x in (arr - arr.mean()) / (sd + cfg["eps"])] if sd > 0 else [0.0] * len(raw)
    else:
        out = [float(x) for x in raw]
    metrics = {"turn_credit/groups": float(len(groups)), "turn_credit/episodes": float(len(raw)),
               "turn_credit/tied_groups": float(sum(1 for v in groups.values() if max(v) - min(v) <= 1e-9)),
               "turn_credit/episode_raw_std": sd,
               "turn_credit/episode_zero_share": float(np.mean(np.abs(arr) <= 1e-9)) if arr.size else 0.0}
    return out, metrics


def turn_advantages(records: Sequence[Dict[str, Any]], uids: Sequence[Any], cfg: Dict[str, Any],
                    lengths: Optional[Sequence[Sequence[int]]] = None) -> Tuple[List[List[float]], Dict[str, float]]:
    """Per episode and turn the turn term T_t (before the weights), and metrics.

    ``lengths``: per episode and turn the number of answer tokens (the positive / negative mass is counted in tokens,
    as the token-mean loss weighs them); default 1 a turn."""
    values = [turn_values(r, cfg["scale"], cfg["gamma"], cfg["turn_reward"], cfg["stall_penalty"],
                          cfg.get("turn_term", "shaping")) for r in records]
    base = baselines(values, uids, cfg["baseline"])
    cen = [[c - b for c, b in zip(cs, bs)] for cs, bs in zip(values, base)]
    flat = np.asarray([x for xs in cen for x in xs], dtype=np.float64)
    sd = float(flat.std()) if flat.size > 1 else 1.0
    z = [[x / (sd + cfg["eps"]) for x in xs] for xs in cen]
    n = max(1, int(flat.size))
    clipped = 0
    if cfg["neg_clip"] > 0:
        clipped = sum(x < -cfg["neg_clip"] for xs in z for x in xs)
        z = [[max(x, -cfg["neg_clip"]) for x in xs] for xs in z]
    lens = lengths or [[1] * len(xs) for xs in z]
    pos = sum(max(a, 0.0) * k for xs, ks in zip(z, lens) for a, k in zip(xs, ks))
    neg = sum(max(-a, 0.0) * k for xs, ks in zip(z, lens) for a, k in zip(xs, ks))
    neg_scale = 1.0
    if cfg["min_positive_mass"] > 0 and neg > 0 and pos < cfg["min_positive_mass"] * neg:
        neg_scale = pos / (cfg["min_positive_mass"] * neg)
        z = [[a * neg_scale if a < 0 else a for a in xs] for xs in z]
    vals = np.asarray([x for xs in values for x in xs], dtype=np.float64)
    turns = np.asarray([x for xs in z for x in xs], dtype=np.float64)
    miss = [t.get("d_after") is None and not t.get("success") for r in records for t in (r.get("turns") or [])]
    stall = [s for r in records for s in stalled(r)]
    metrics = {"turn_credit/turns": float(flat.size),
               "turn_credit/value_mean": float(vals.mean()) if vals.size else 0.0,
               "turn_credit/value_std": float(vals.std()) if vals.size else 0.0,
               "turn_credit/value_pos_share": float(np.mean(vals > 1e-9)) if vals.size else 0.0,
               "turn_credit/value_neg_share": float(np.mean(vals < -1e-9)) if vals.size else 0.0,
               "turn_credit/centered_std": sd,
               "turn_credit/turn_term_p10": float(np.quantile(turns, 0.1)) if turns.size else 0.0,
               "turn_credit/turn_term_p90": float(np.quantile(turns, 0.9)) if turns.size else 0.0,
               "turn_credit/turn_term_min": float(turns.min()) if turns.size else 0.0,
               "turn_credit/turn_term_max": float(turns.max()) if turns.size else 0.0,
               "turn_credit/clipped_share": clipped / n,
               "turn_credit/pos_mass_over_neg": (pos / neg) if neg > 0 else 0.0, "turn_credit/neg_scale": neg_scale,
               "turn_credit/table_miss_share": float(np.mean(miss)) if miss else 0.0,
               "turn_credit/stalled_share": float(np.mean(stall)) if stall else 0.0}
    return z, metrics


def combine_metrics(z: Sequence[Sequence[float]], ep: Sequence[float], cfg: Dict[str, Any]) -> Dict[str, float]:
    """How the two terms add up per turn: the turn term's share of the magnitude (mean |turn_weight T| over mean
    |episode_weight E| + mean |turn_weight T|), the share of turns whose total sign differs from the episode term's
    (among turns with E != 0), the share of turns with E = 0 (the turn term alone), the total's positive share, overall
    and at turns 4+, and the episode term's 10th / 90th percentile over turns."""
    we, wt = cfg["episode_weight"], cfg["turn_weight"]
    e_t, t_t, late = [], [], []
    for zs, e in zip(z, ep):
        for t, a in enumerate(zs):
            e_t.append(we * e)
            t_t.append(wt * a)
            late.append(t >= LATE_TURN)
    if not e_t:
        return {}
    e_t, t_t, late = np.asarray(e_t), np.asarray(t_t), np.asarray(late)
    tot = e_t + t_t
    nz = np.abs(e_t) > 1e-9
    me, mt = float(np.abs(e_t).mean()), float(np.abs(t_t).mean())
    return {"turn_credit/turn_magnitude_share": mt / (me + mt) if me + mt > 0 else 0.0,
            "turn_credit/sign_flip_share": float(np.mean(np.sign(tot[nz]) != np.sign(e_t[nz]))) if nz.any() else 0.0,
            "turn_credit/turn_only_share": float(np.mean(~nz)),
            "turn_credit/pos_share": float(np.mean(tot > 1e-9)),
            "turn_credit/late_pos_share": float(np.mean(tot[late] > 1e-9)) if late.any() else 0.0,
            "turn_credit/episode_term_p10": float(np.quantile(e_t, 0.1)),
            "turn_credit/episode_term_p90": float(np.quantile(e_t, 0.9))}


def token_advantages(records: Sequence[Dict[str, Any]], uids: Sequence[Any], response_mask: torch.Tensor,
                     episode_returns: Sequence[float], cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The (batch, response length) advantages: every answer token (response_mask) gets episode_weight x E of its
    episode, each turn's span also turn_weight x its T; tokens outside every span get only the episode term.  With
    turn_term "statement" the span is the turn's statement_span."""
    bs, length = response_mask.shape
    key = "statement_span" if cfg.get("turn_term", "shaping") == "statement" else "span"
    spans = [[[max(0, int(t.get(key, t["span"])[0])), min(length, int(t.get(key, t["span"])[1]))]
              for t in (r.get("turns") or [])] for r in records]
    lengths = [[max(0, e - s) for s, e in sp] for sp in spans]
    z, metrics = turn_advantages(records, uids, cfg, lengths)
    ep, ep_metrics = episode_terms(episode_returns, uids, cfg)
    metrics.update(ep_metrics)
    metrics.update(combine_metrics(z, ep, cfg))
    adv = torch.zeros((bs, length), dtype=torch.float32)
    for i, (sp, zs) in enumerate(zip(spans, z)):
        for (s, e), a in zip(sp, zs):
            if e > s:
                adv[i, s:e] = cfg["turn_weight"] * float(a)
    if cfg["episode_weight"]:
        adv = adv + cfg["episode_weight"] * torch.tensor(ep, dtype=torch.float32).unsqueeze(-1)
    mask = response_mask.to(torch.float32)
    adv = adv.to(response_mask.device) * mask.to(response_mask.device)
    covered = sum(sum(k for k in ks) for ks in lengths)
    metrics["turn_credit/span_token_share"] = covered / max(1.0, float(mask.sum().item()))
    return adv, metrics
