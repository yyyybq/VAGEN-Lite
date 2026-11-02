"""Reward line, phase 2, direction 2 (the user's brief of 2026-11-01, section 2; decision 1 of 2026-11-02): guided
sampling (jump-start RL).  Off unless the PPO config has ``algorithm.jsrl_guide``.

On a training episode of a hard task (the minimum number of actions d* >= min_d_star, or a family in ``families``,
the brief: equidistance) a guide first walks k of the d* actions, then the policy takes over:

* the guide is greedy on the task's distance table: from the current pose each allowed action is applied with the
  contract's motion model (data_gen/interiorgs_v10/difficulty.apply_action_batch, the one the tables were built with),
  a move is legal when the env's own collision detector lets it through, and the action whose pose has the smallest
  table distance (strictly below the current one) is taken, ties in the contract's action order.  It is deterministic,
  so the 8 episodes of a task share the prefix;
* k by stage of the update number (``stages``: [first, last, fraction]; the brief: updates 1-50 ceil(d*/2), 51-100
  ceil(d*/4), later 0), always below d*, so the guide never ends a task;
* the guide's actions are grouped into turns the way the SFT data is (``chunking: action_runs``: a run of one action,
  at most ``max_per_turn`` = 5 a turn) and written as the answers the policy gives (``<action>a|a|</action>``); each
  guide turn is stepped in the env, its observation rendered, and both enter the conversation before the policy's
  first turn: they are prompt (history), so they get no gradient, take no response length and the rollout
  logprobs stay aligned;
* the guide's rewards are not in the episode return (the agent loop never records them), so the group comparison
  starts at the takeover state; the guide's primitive actions use the env's step budget (12), so the slack (budget
  minus remaining distance) is the same as unguided;
* the guide stops (and the reason is counted) when the table has no pose or no closer pose, when the env blocks a
  guide move or ends the pose elsewhere than planned, or when the env ends the episode; it walks at most the table's
  own start distance - 1 actions ("capped", where the table puts the start closer than the generator's d*).

Validation and evaluation are never guided: the trainer passes the update number (``global_step``) only with the
training batches (vagen/ray_trainer.py), and the agent loop guides only when it is there.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

KEY = "jsrl_guide"
DEFAULTS = {"tables": None, "stages": [[1, 50, 0.5], [51, 100, 0.25]], "min_d_star": 6, "families": ["equidistance"],
            "max_per_turn": 5}
_CACHES: Dict[str, Any] = {}


def settings(raw: Any) -> Dict[str, Any]:
    """The guide's settings from its config dict, defaults filled in; unknown keys refused."""
    raw = dict(raw or {})
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown {KEY} settings {sorted(unknown)}")
    out = dict(DEFAULTS, **{k: v for k, v in raw.items() if v is not None})
    if not out["tables"]:
        raise ValueError(f"{KEY}.tables (the distance tables' directory) is required")
    out["tables"] = str(out["tables"])
    stages = []
    for st in out["stages"]:
        first, last, frac = st
        first, last, frac = int(first), (int(last) if last is not None else None), float(frac)
        if not 0.0 <= frac < 1.0 or (last is not None and last < first):
            raise ValueError(f"{KEY}.stages: bad stage {st}")
        stages.append([first, last, frac])
    out["stages"] = stages
    out["min_d_star"] = int(out["min_d_star"])
    out["families"] = [str(f) for f in out["families"]]
    out["max_per_turn"] = int(out["max_per_turn"])
    if out["max_per_turn"] < 1:
        raise ValueError(f"{KEY}.max_per_turn must be at least 1")
    return out


def from_config(config: Any) -> Optional[Dict[str, Any]]:
    """The settings from a full trainer config (``algorithm.jsrl_guide``), or None when the guide is off."""
    if config is None:
        return None
    algo = config.get("algorithm", None) if hasattr(config, "get") else getattr(config, "algorithm", None)
    if algo is None:
        return None
    raw = algo.get(KEY, None) if hasattr(algo, "get") else getattr(algo, KEY, None)
    if not raw:
        return None
    if hasattr(raw, "items") and not isinstance(raw, dict):  # an OmegaConf node
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(raw, resolve=True)
    return settings(raw)


def guide_count(d_star: int, step: int, stages: Sequence[Sequence[Any]]) -> int:
    """k: the number of guide actions at update ``step`` for a task of minimum d_star (0 outside every stage),
    always below d_star."""
    for first, last, frac in stages:
        if step >= first and (last is None or step <= last):
            return min(int(math.ceil(d_star * frac)), max(0, d_star - 1))
    return 0


def eligible(item: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    d_star = int(((item or {}).get("difficulty") or {}).get("d_star") or 0)
    return d_star >= cfg["min_d_star"] or str((item or {}).get("task_type")) in cfg["families"]


def chunk(actions: Sequence[str], max_per_turn: int) -> List[List[str]]:
    """The SFT data's turns (chunking action_runs): each run of one action, at most max_per_turn a turn."""
    turns: List[List[str]] = []
    for a in actions:
        if turns and turns[-1][-1] == a and len(turns[-1]) < max_per_turn:
            turns[-1].append(a)
        else:
            turns.append([a])
    return turns


def action_text(turn: Sequence[str]) -> str:
    return "<action>" + "".join(f"{a}|" for a in turn) + "</action>"


def _table(tables: str, task_id: str):
    from data_gen.interiorgs_v10.distance_table import TableCache

    cache = _CACHES.get(tables)
    if cache is None:
        cache = _CACHES[tables] = TableCache(tables)
    return cache.get(task_id)


def _distance(table: Dict[str, Any], key: int) -> Optional[int]:
    keys = table["keys"]
    i = int(np.searchsorted(keys, key))
    if i < len(keys) and int(keys[i]) == key:
        d = int(table["dist"][i])
        return d if d >= 0 else None
    return None


def pose_key(env: Any) -> int:
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys

    return int(contract_keys(*_split(env.view_engine.get_pose()))[0])


def plan(env: Any, table: Dict[str, Any], k: int) -> Tuple[List[str], List[int], str]:
    """The guide's k actions from the env's current pose (ActiveSpatialEnv), the contract key after each, and why it
    stopped early ("" when it did not).  At most the table's own start distance - 1 actions ("capped": the table puts
    the start closer than d*), so the guide never ends the task."""
    from data_gen.interiorgs_v10.contract import ACTIONS
    from data_gen.interiorgs_v10.difficulty import IS_TRANSLATION, _split, apply_action_batch, contract_keys

    P, R = _split(env.view_engine.get_pose())
    d = _distance(table, int(contract_keys(P, R)[0]))
    if d is None:
        return [], [], "start_not_in_table"
    capped = d - 1 < k
    k = min(k, d - 1)
    if k <= 0:
        return [], [], "capped" if capped else ""
    allowed = getattr(env, "_allowed_actions", None)
    detector = env.collision_detector if getattr(env.config, "enable_collision_detection", False) else None
    actions: List[str] = []
    keys: List[int] = []
    for _ in range(k):
        best = None
        for ai, a in enumerate(ACTIONS):
            if allowed is not None and a not in allowed:
                continue
            p, r = apply_action_batch(P, R, a)
            r = np.broadcast_to(r, (len(P), 3, 3))
            if IS_TRANSLATION[ai] and detector is not None:
                if detector.check_collision(position=p[0], previous_position=P[0]).has_collision:
                    continue
            key = int(contract_keys(p, r)[0])
            dn = _distance(table, key)
            if dn is not None and dn < d and (best is None or dn < best[0]):
                best = (dn, a, p, np.array(r), key)
        if best is None:
            return actions, keys, "no_closer_pose"
        d, a, P, R, key = best
        actions.append(a)
        keys.append(key)
    return actions, keys, "capped" if capped else ""


async def run_guide(cfg: Dict[str, Any], agent_data: Any, step: int, **kwargs) -> Dict[str, Any]:
    """Walk the guide's turns on a reset episode (GymAgentLoop's AgentData, before its first prompt is built): each
    turn's answer and observation are appended to the conversation as history.  Returns the episode's guide record
    (also kept for reward_extra_info)."""
    from vagen.agent_loop.gym_agent_loop import _normalize_images, convert_obs_to_content

    inner = getattr(agent_data.env, "_inner", agent_data.env)
    item = getattr(inner, "current_item", None) or {}
    d_star = int((item.get("difficulty") or {}).get("d_star") or 0)
    rec = {"guide_eligible": 0, "guide_k": 0, "guide_actions": 0, "guide_turns": 0, "guide_stop": ""}
    if not eligible(item, cfg):
        return rec
    rec["guide_eligible"] = 1
    k = guide_count(d_star, step, cfg["stages"])
    rec["guide_k"] = k
    if k <= 0:
        return rec
    table = _table(cfg["tables"], str(item.get("task_id")))
    if table is None:
        rec["guide_stop"] = "no_table"
        return rec
    actions, keys, stop = plan(inner, table, k)
    turns = chunk(actions, cfg["max_per_turn"])
    done_actions = 0
    for turn in turns:
        text = action_text(turn)
        obs, _reward, done, info = await agent_data.env.step(text)  # the guide's reward is not the policy's
        agent_data.messages.append({"role": "assistant", "content": text})
        rec["guide_turns"] += 1
        tm = ((info or {}).get("metrics") or {}).get("turn_metrics") or {}
        blocked = int(tm.get("collision_count", 0) or 0)
        done_actions += len(turn)
        if done:
            rec["guide_stop"] = "env_done"
            break
        agent_data.messages.append({"role": "user", "content": convert_obs_to_content(obs, **kwargs)})
        agent_data.image_data.extend(_normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or []))
        if blocked:
            rec["guide_stop"] = "blocked"
            break
        if pose_key(inner) != keys[done_actions - 1]:
            rec["guide_stop"] = "pose_mismatch"
            break
    else:
        rec["guide_stop"] = stop
    rec["guide_actions"] = done_actions
    return rec
