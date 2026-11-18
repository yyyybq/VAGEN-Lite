"""q38 (the Qwen3.8-27B rerun line): the 8B trainer's per-update rollout metrics over every candidate episode.

The 8B trainer (vagen/ray_trainer.py, the concat branch of fit, RT:2937-3047 at qwen38 551c241e) logs, over the 192
candidate episodes of an update (before the group filter):

- episode/success_rate, episode/success_count, episode/num_episodes and their aliases train/traj_success/{mean,sum,
  count} (RT:2979-3018; in concat mode there are no group_idx / traj_idx, so the episodes are the rows);
- transition/<name>/{mean,std,sum,count} for each per-episode value of the agent loop's reward_extra_info that is in
  the list of RT:3021-3046 (only those the episodes carry: under the V4 recipe invalid_action, final_score and
  score_improvement);
- transition/reward_{mean,std,min,max,count}: the episode returns (RT:2961-2977).

RT cannot be imported in the q38 environment (it imports the 0.6.1 trainer), so this module writes the same
arithmetic again: the values are cast to float32 first (RT:2943, RT:2964) and reduced with numpy in float32, so with
the rows in the same order the results are bit-identical to the 8B log (tests/test_candidate_metrics.py, T2).

The q38 sampler (vagen/q38/sampler.py) calls ``candidate_metrics`` on every candidate of the draw batch it trains on;
``stats_metrics`` aggregates the agent loop's per-episode ``vagen_stats`` (DESIGN_ASYNC 2.5.3 item 7) into
vagen/env/* and vagen/gen/*.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# RT:3021-3046, in the same order: (reward_extra_info name, metric prefix)
TRANSITION_METRICS: tuple[tuple[str, str], ...] = (
    ("invalid_action", "transition/invalid_action_rate"),
    ("empty_action", "transition/empty_action_rate"),
    ("contradictory_action", "transition/contradictory_action_rate"),
    ("missing_action_tag", "transition/missing_action_tag_rate"),
    ("empty_action_body", "transition/empty_action_body_rate"),
    ("unknown_action_name", "transition/unknown_action_name_rate"),
    ("truncated_before_action", "transition/truncated_before_action_rate"),
    ("multiple_action_tag", "transition/multiple_action_tag_rate"),
    ("env_exception", "transition/env_exception_rate"),
    ("strict_action_tag_rate", "transition/strict_action_tag_rate"),
    ("tool_call_rate", "transition/tool_call_rate"),
    ("fallback_parse_rate", "transition/fallback_parse_rate"),
    ("strict_parse_success_rate", "transition/strict_parse_success_rate"),
    ("format_penalty_rate", "transition/format_penalty_rate"),
    ("collision_termination", "transition/collision_termination_rate"),
    ("low_info_termination", "transition/low_info_termination_rate"),
    ("renderer_failure", "transition/renderer_failure_rate"),
    ("episode_length", "transition/episode_length"),
    ("final_score", "transition/final_score"),
    ("score_improvement", "transition/score_improvement"),
) + tuple(
    (f"action_{a}", f"transition/action_distribution/{a}")
    for a in ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right")
)


def columns(extra_infos: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    """The per-episode reward_extra_info dicts as columns {name: [value per episode]}, the shape of the 8B
    trainer's reward_extra_infos_dict.  A name some episode lacks is left out (the 8B reward manager only ever saw
    one key set per batch; the q38 agent loop writes the same keys for every episode, DESIGN_ASYNC 2.5.3 item 9)."""
    if not extra_infos:
        return {}
    names = list(extra_infos[0].keys())
    shared = [k for k in names if all(k in e for e in extra_infos)]
    return {k: [e[k] for e in extra_infos] for k in shared}


def _float32(values: Sequence[Any]) -> np.ndarray | None:
    """RT:2939-2950 without the no-concat mask (concat mode: every row is a unique episode)."""
    if values is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return arr if arr.size > 0 else None


def candidate_metrics(extra: Mapping[str, Sequence[Any]], returns: Sequence[float]) -> dict[str, float | int]:
    """The 8B trainer's episode/*, train/traj_success/* and transition/* metrics.

    Args:
        extra: per-episode values by name (reward_extra_info as columns, see ``columns``), one entry per candidate
            episode, in the order the metrics are to be reduced in.
        returns: each episode's return (the sum of its token-level rewards, float32 in the trainer), same order.

    Returns:
        {metric name: python int or float}; an empty dict without episodes (as RT, which logs nothing then).
    """
    metrics: dict[str, float | int] = {}
    if not extra and len(returns) == 0:
        return metrics

    # RT:2979-3018 (concat branch: no group_idx / traj_idx, so one row is one episode)
    success = _float32(extra.get("traj_success"))
    if success is not None:
        success_count = float(success.sum())
        num_episodes = int(success.size)
        success_rate = float(success_count / max(num_episodes, 1))
        metrics["episode/success_rate"] = success_rate
        metrics["episode/success_count"] = success_count
        metrics["episode/num_episodes"] = num_episodes
        metrics["train/traj_success/mean"] = success_rate
        metrics["train/traj_success/sum"] = success_count
        metrics["train/traj_success/count"] = num_episodes

    # RT:2952-2959 and RT:3021-3046
    for name, prefix in TRANSITION_METRICS:
        arr = _float32(extra.get(name))
        if arr is None:
            continue
        metrics[f"{prefix}/mean"] = float(arr.mean())
        metrics[f"{prefix}/std"] = float(arr.std())
        metrics[f"{prefix}/sum"] = float(arr.sum())
        metrics[f"{prefix}/count"] = int(arr.size)

    # RT:2961-2977
    rewards = _float32(returns)
    if rewards is not None:
        metrics["transition/reward_mean"] = float(rewards.mean())
        metrics["transition/reward_std"] = float(rewards.std())
        metrics["transition/reward_min"] = float(rewards.min())
        metrics["transition/reward_max"] = float(rewards.max())
        metrics["transition/reward_count"] = int(rewards.size)
    return metrics


# the agent loop's per-episode counters (extra_fields.vagen_stats, DESIGN_ASYNC 2.5.3 item 7; the q38 loop's key set is
# vagen/q38/gym_agent_loop_v1.py VAGEN_STATS_KEYS), with the names a counter may carry; the first name is the q38
# loop's
_COUNTERS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("step_exceptions", ("step_exceptions",), "vagen/env/step_exceptions"),
    ("endoftext_stops", ("endoftext_stops", "endoftext_turns"), "vagen/gen/endoftext_stops"),
    ("length_stops", ("length_stops",), "vagen/gen/length_stops"),
    ("turns_resumed", ("turns_resumed", "resumed_turns"), "vagen/gen/turns_resumed"),
)
_DURATIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("render_s",), "vagen/env/render_s"),
    (("turn_gen_s", "gen_s"), "vagen/gen/turn_gen_s"),
)
# vLLM preemptions of the episode's requests; -1 = not reported by the server (summed over the episodes reporting)
_PREEMPTED = ("num_preempted", "vagen/gen/num_preempted")
# the episode's termination reason (GL:480-490 order in the q38 loop); each known reason is always logged, so the
# metric names do not change from step to step; an unknown reason is logged under its own name as well
_TERMINATION = ("termination", "vagen/env/termination")
TERMINATION_REASONS = ("success", "env_done", "env_error", "max_turns", "response_budget", "obs_overflow")
# per-episode values the sampler does not aggregate (per-turn detail for offline analysis, or already in
# reward_extra_info: env_turns is n_turns)
IGNORED_STATS = ("turn_max_tokens", "turn_spans", "gen_turns", "env_turns")


def _first(stats: Mapping[str, Any], names: Sequence[str]) -> Any:
    for n in names:
        if n in stats:
            return stats[n]
    return None


def stats_metrics(stats: Sequence[Mapping[str, Any] | None]) -> dict[str, float | int]:
    """vagen/env/* and vagen/gen/* over the candidates' ``vagen_stats`` (one dict per episode, or None).

    Counters are summed over the episodes (step_exceptions, endoftext_stops, length_stops, turns_resumed;
    num_preempted over the episodes that report it, i.e. not -1); the per-turn durations (render_s, turn_gen_s: a
    list of seconds per episode) give p50 / p95 / max over all turns of all episodes (vagen/env/render_s_{p50,p95,max},
    vagen/gen/turn_gen_s_{p50,p95,max}); the termination reasons are counted (vagen/env/termination/<reason>, every
    reason of TERMINATION_REASONS always).  A counter no episode carries is not logged;
    ``vagen/env/episodes_with_stats`` counts the episodes that carried a stats dict."""
    present = [s for s in stats if isinstance(s, Mapping)]
    metrics: dict[str, float | int] = {"vagen/env/episodes_with_stats": len(present)}
    if not present:
        return metrics
    for _canonical, names, key in _COUNTERS:
        values = [_first(s, names) for s in present]
        values = [v for v in values if v is not None]
        if values:
            metrics[key] = int(sum(int(v) for v in values))
    preempted = [int(s[_PREEMPTED[0]]) for s in present if s.get(_PREEMPTED[0]) is not None]
    reported = [v for v in preempted if v >= 0]
    if reported:
        metrics[_PREEMPTED[1]] = int(sum(reported))
    # an episode that ended without a reason (the loop's default "") is counted as "unset"
    reasons = [str(s[_TERMINATION[0]]) or "unset" for s in present if s.get(_TERMINATION[0]) is not None]
    if reasons:
        for reason in dict.fromkeys(TERMINATION_REASONS + tuple(reasons)):
            metrics[f"{_TERMINATION[1]}/{reason}"] = sum(1 for r in reasons if r == reason)
    for names, key in _DURATIONS:
        turns: list[float] = []
        for s in present:
            v = _first(s, names)
            if v is None:
                continue
            turns.extend(float(x) for x in (v if isinstance(v, (list, tuple)) else [v]))
        if turns:
            arr = np.asarray(turns, dtype=np.float64)
            metrics[f"{key}_p50"] = float(np.percentile(arr, 50))
            metrics[f"{key}_p95"] = float(np.percentile(arr, 95))
            metrics[f"{key}_max"] = float(arr.max())
    return metrics
