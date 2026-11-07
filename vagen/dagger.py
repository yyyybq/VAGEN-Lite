"""The distance table as action labels in the RL update (DAgger style; the reward line, phase 3, experiment B: the
user's brief of 2026-11-06, decision 2 甲 of 2026-11-07 "按完整版开工，默认参数照计划报告").  Off unless the PPO config
has ``actor_rollout_ref.actor.dagger.enable`` (training_prep rl-config --dagger, train_chain plan rl "dagger").

Each update, after the 8 episodes of every candidate task are sampled:

* **the agent loop judges every turn** of a training episode (never a validation one: the trainer marks the
  training batches with ``dagger_step``) on the task's distance table (``tables``, one ``<task_id>.npz`` a task, the
  ones the reward line built for its 3,000 tasks): before the env steps the turn, the pose and the step count are
  read; after it, the actions the env executed (the parsed ones it allows, up to "done", as many as its step count
  moved) are replayed from that pose with the tables' own motion model (difficulty.apply_action_batch) and the env's
  own collision detector, and an action is optimal when it takes the pose from table distance d to d - 1 (a blocked
  move never is).  A turn is **optimal** when it executed at least one action, every executed action was optimal,
  no "done" was reached, and the answer is in the labels' form (``<action>a|b|</action>``, nothing else) and wrote
  exactly the actions the env executed (so an own label never trains an action the env skipped: stopped at the
  goal mid-turn, out of the step budget, past five, or not allowed); a turn the env could not read executed nothing
  and is not optimal.  Distances fall back to the 26 neighbour cells when a pose's own cell is missing or has no
  success reachable (distance_table_near), as the line's reward did.  The replay's end pose must be the env's (else
  the turn is left out, counted as ``pose_mismatch``);
* **its label:** an optimal turn keeps its own answer; any other turn gets the guide's (vagen/jsrl_guide.py on
  reward-jsrl, copied here): the greedy plan on the table from the turn's start pose (each allowed action applied,
  a move legal when the env's detector lets it through, the smallest distance strictly below the current one, ties
  in the contract's action order), at most the step budget left, cut into the SFT data's turns (a run of one
  action, at most ``max_per_turn``), its first turn written as the policy writes an answer,
  ``<action>a|a|</action><|im_end|>``;
* **the trainer** takes up to ``cap`` of those turns over all the update's candidate episodes (every one of the 24
  tasks x 8, before the group filter drops any: the filter's all-fail groups are where the policy gets no signal),
  a seeded random pick (``seed`` and the update number), the same number for every GPU; ``turns`` "all" takes own and
  replaced turns alike, "replaced" only the replaced ones.  Each becomes a supervised sample cut from its episode's
  row without new rendering: the row's tokens before the turn (the same images, the same history), the label's
  tokens after them, the row's four position rows on the prefix and each continued by one a token on the label,
  the pixels of the prefix's images only;
* **the actor** adds ``lam`` x the next-token loss of those samples (the token mean over all their loss tokens) to the
  policy gradient of the update's one optimizer step, through the und-mix channel's own measurement
  (vagen/und_mix.mix_gradients: both terms' norms before clipping, their shares, the cosine); the loss tokens are a
  label's action names and separators through ``</action>`` (not ``<action>``, not the end of the turn).  With
  ``audit_only`` the supervised term is measured the same way and then dropped, so the update is the policy's own:
  the check run before the smoke.

**The check the user asked for before the smoke** (11-07: "自己答的回合，前缀算出的概率等于训练时记的"): for an own
answer, the sample's log-probabilities of its tokens must equal the row's ``old_log_probs`` at the same tokens (the
trainer's own recompute before the update, the same weights).  The actor logs the differences over every such token
(RUN/dagger/check.jsonl, one line per update): separately for samples whose prefix + answer is the whole row (the
very input the recompute had, so about 0: the comparison itself is sound) and for the others (a wrong prefix would
show there); and, with ``audit_only``, a floor for the latter: every sample's log-probabilities again with it beside
another sample in a two-row batch (the same tokens in another batch shape, what bf16 rounding alone moves).  The
trainer's builder checks the exact parts on every own sample (the continued positions equal the row's, the tokens
the agent loop's answer, the prefix's image tokens its pixels), counts any difference and leaves such a sample out.

With the switch on, an update that brings no judged turn (no records from the agent loop, unreadable tables, every
turn left out) or no sample stops the run instead of training plain RL unnoticed.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

KEY = "dagger"
DEFAULTS = {"tables": None, "lam": 0.5, "cap": 96, "turns": "all", "seed": 20261019, "max_per_turn": 5,
            "audit_only": False, "table_cache": 8}
REQUEST_KEYS = set(DEFAULTS)
TURN_SCOPES = ("all", "replaced")
ACTION_OPEN, ACTION_CLOSE = "<action>", "</action>"
END_OF_TURN = "<|im_end|>"
MAX_STEPS = 12  # the env's primitive-step budget (contract.MAX_PRIMITIVE_STEPS), when the env does not say


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------- settings

def settings(raw: Any) -> Dict[str, Any]:
    """The labels' settings (a dict, a JSON text or a JSON file's path), defaults filled in; ValueError on a bad one."""
    if isinstance(raw, str):
        text = raw.strip()
        raw = json.loads(Path(text).read_text() if text.endswith(".json") and Path(text).is_file() else text)
    if not isinstance(raw, dict):
        raise ValueError(f"a JSON object, not {type(raw).__name__}")
    unknown = sorted(set(raw) - REQUEST_KEYS)
    if unknown:
        raise ValueError(f"unknown keys {unknown} (known: {sorted(REQUEST_KEYS)})")
    out = dict(DEFAULTS, **{k: v for k, v in raw.items() if v is not None})
    if not out["tables"]:
        raise ValueError("tables (the distance tables' directory, <task_id>.npz) is required")
    out["tables"] = str(out["tables"])
    for key in ("cap", "seed", "max_per_turn", "table_cache"):
        value = out[key]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{key} must be a whole number, not {value!r}")
        out[key] = int(value)
    if isinstance(out["lam"], bool):
        raise ValueError("lam must be a number")
    out["lam"] = float(out["lam"])
    if not (out["lam"] > 0 and math.isfinite(out["lam"])):
        raise ValueError(f"lam must be above 0, not {out['lam']}")
    if out["cap"] < 1 or out["seed"] < 0 or not 1 <= out["max_per_turn"] <= 5 or out["table_cache"] < 1:
        raise ValueError(f"cap >= 1, seed >= 0, 1 <= max_per_turn <= 5, table_cache >= 1: {out}")
    if out["turns"] not in TURN_SCOPES:
        raise ValueError(f"turns must be one of {TURN_SCOPES}, not {out['turns']!r}")
    if not isinstance(out["audit_only"], bool):
        raise ValueError("audit_only must be true or false")
    return out


BLOCK_KEYS = ("enable", "log_dir", "tokenizer")  # what ppo_block adds to the request


def ppo_block(request: Any, model: str, run_root: Any) -> Dict[str, Any]:
    """actor.dagger of a PPO config: the settings, ``enable``, the RL start model (its tokenizer decodes the first
    batch's labels) and the mixer's log folder (RUN/dagger); the tables' folder must hold tables."""
    block = settings(request)
    tables = Path(block["tables"]).resolve()
    if not tables.is_dir() or not any(tables.glob("*.npz")):
        raise ValueError(f"{tables} holds no <task_id>.npz tables")
    block["tables"] = str(tables)
    return {"enable": True, **block, "tokenizer": str(model), "log_dir": str(Path(run_root).resolve() / "dagger")}


def _plain(node: Any) -> Any:
    if node is None or isinstance(node, dict):
        return node
    if hasattr(node, "items"):  # an OmegaConf node
        from omegaconf import OmegaConf

        return OmegaConf.to_container(node, resolve=True)
    return node


def from_config(config: Any) -> Optional[Dict[str, Any]]:
    """The block from a full trainer config (``actor_rollout_ref.actor.dagger``), or None when the labels are off."""
    if config is None:
        return None
    try:
        actor = config.actor_rollout_ref.actor
    except AttributeError:
        try:
            actor = config["actor_rollout_ref"]["actor"]
        except (KeyError, TypeError):
            return None
    raw = actor.get("dagger", None) if hasattr(actor, "get") else getattr(actor, "dagger", None)
    raw = _plain(raw)
    if not raw or not raw.get("enable"):
        return None
    return {**settings({k: v for k, v in raw.items() if k not in BLOCK_KEYS}),
            **{k: raw.get(k) for k in BLOCK_KEYS}, "enable": True}


# ---------------------------------------------------------------------------------------------------- tables

_LOCK = threading.Lock()
_TABLES: "OrderedDict[Tuple[str, str], Optional[Dict[str, Any]]]" = OrderedDict()


def table(root: str, task_id: str, keep: int = 8) -> Optional[Dict[str, Any]]:
    """A task's table (None when the folder has none for it), the last ``keep`` kept in memory (about 14 MB each)."""
    from data_gen.interiorgs_v10.distance_table import load, table_path

    key = (str(root), str(task_id))
    with _LOCK:
        if key in _TABLES:
            _TABLES.move_to_end(key)
            return _TABLES[key]
    path = table_path(Path(root), str(task_id))
    got = load(path) if path.is_file() else None
    with _LOCK:
        _TABLES[key] = got
        _TABLES.move_to_end(key)
        while len(_TABLES) > max(1, int(keep)):
            _TABLES.popitem(last=False)
    return got


def distance(tab: Dict[str, Any], P: np.ndarray, R: np.ndarray) -> Tuple[Optional[int], bool]:
    """(the remaining minimum number of actions, whether it came from a neighbour cell); None when unknown."""
    from data_gen.interiorgs_v10.difficulty import contract_keys
    from data_gen.interiorgs_v10.distance_table_near import neighbour_distance

    key = int(contract_keys(P, R)[0])
    keys = tab["keys"]
    i = int(np.searchsorted(keys, key))
    if i < len(keys) and int(keys[i]) == key and int(tab["dist"][i]) >= 0:
        return int(tab["dist"][i]), False
    # a missing cell, or one with no success reachable: the neighbours (as distance_table_near.lookup_near)
    nd = neighbour_distance(tab, key)
    return nd, nd is not None


def _move(P: np.ndarray, R: np.ndarray, action: str, detector: Any) -> Tuple[np.ndarray, np.ndarray, bool]:
    """One action of the contract's motion model from (P, R): the new pose and whether the env's detector blocks it
    (a blocked move leaves the pose where it was, as the env undoes it)."""
    from data_gen.interiorgs_v10.difficulty import ACTION_INDEX, IS_TRANSLATION, apply_action_batch

    p, r = apply_action_batch(P, R, action)
    r = np.array(np.broadcast_to(r, (len(P), 3, 3)))
    if IS_TRANSLATION[ACTION_INDEX[action]] and detector is not None:
        if detector.check_collision(position=p[0], previous_position=P[0]).has_collision:
            return P, R, True
    return p, r, False


def executed_actions(parsed: Sequence[str], allowed: Optional[Sequence[str]], n_steps: int) -> Tuple[List[str], bool]:
    """The actions the env executed from its parsed list (ActiveSpatialEnv.step): the allowed ones in order until its
    step count moved ``n_steps`` times, stopping at "done"; and whether a "done" was reached."""
    out: List[str] = []
    allowed_set = None if allowed is None else {str(a).lower() for a in allowed}
    for a in parsed or []:
        name = str(a).lower()
        if name == "done":
            return out, len(out) == n_steps
        if allowed_set is not None and name not in allowed_set:
            continue
        if len(out) == n_steps:
            break
        out.append(name)
    return out, False


def written_actions(text: str) -> Optional[List[str]]:
    """The action names an answer wrote: between its first ``<action>`` and the first ``</action>`` after it, split
    at '|', stripped, lower-cased, empty entries dropped (the loss tokens' region).  None without such a block."""
    a = text.find(ACTION_OPEN)
    if a < 0:
        return None
    b = text.find(ACTION_CLOSE, a + len(ACTION_OPEN))
    if b < 0:
        return None
    return [x.strip().lower() for x in text[a + len(ACTION_OPEN):b].split("|") if x.strip()]


def canonical_answer(text: str, written: Optional[List[str]]) -> bool:
    """Whether an answer (its decoded tokens) is exactly the form a label is written in, ``<action>a|b|</action>``
    and at most the end of the turn after it: nothing before the block or after it, lower-case names, each followed
    by '|'.  Only such an answer is kept as its own label, so that its loss tokens (the names and separators through
    ``</action>``) never hold a token that straddles a tag (a second block's ``><``, ``>|``, ``>\\n``) or train a
    spelling, case, think text or tool call the labels never use."""
    if written is None:
        return False
    body = text[:-len(END_OF_TURN)] if text.endswith(END_OF_TURN) else text
    return body == action_text(written)


def judge(tab: Dict[str, Any], P: np.ndarray, R: np.ndarray, executed: Sequence[str], detector: Any
          ) -> Dict[str, Any]:
    """Every executed action of a turn from its start pose (P, R): optimal when it takes table distance d to d - 1."""
    from data_gen.interiorgs_v10.difficulty import contract_keys

    d0, near = distance(tab, P, R)
    rec: Dict[str, Any] = {"d_before": d0, "near": near, "hits": [], "blocked": 0}
    if d0 is None:
        rec["status"] = "unknown_start"
        return rec
    if d0 == 0:
        rec["status"] = "at_goal"
        return rec
    d = d0
    for a in executed:
        P, R, blocked = _move(P, R, a, detector)
        if blocked:
            rec["blocked"] += 1
            rec["hits"].append(False)
            continue
        dn, n2 = distance(tab, P, R)
        near = near or n2
        if dn is None:
            rec["status"] = "unknown_mid"
            return rec
        rec["hits"].append(dn == d - 1)
        d = dn
    rec.update(status="ok", d_after=d, near=near, end_key=int(contract_keys(P, R)[0]))
    return rec


def plan(tab: Dict[str, Any], P: np.ndarray, R: np.ndarray, k: int, detector: Any,
         allowed: Optional[Sequence[str]]) -> Tuple[List[str], str]:
    """The guide's greedy plan of at most k actions from (P, R) (jsrl_guide.plan without its cap below the start's
    distance: a label may end the task) and why it stopped early ("" when it did not)."""
    from data_gen.interiorgs_v10.contract import ACTIONS

    d, _ = distance(tab, P, R)
    if d is None:
        return [], "start_not_in_table"
    allowed_set = None if allowed is None else {str(a).lower() for a in allowed}
    actions: List[str] = []
    for _ in range(int(k)):
        if d == 0:  # the goal (a step may drop the table distance by more than one: cells are merged at 1 cm / 1 deg)
            break
        best = None
        for a in ACTIONS:
            if allowed_set is not None and a not in allowed_set:
                continue
            p, r, blocked = _move(P, R, a, detector)
            if blocked:
                continue
            dn, _ = distance(tab, p, r)
            if dn is not None and dn < d and (best is None or dn < best[0]):
                best = (dn, a, p, r)
        if best is None:
            return actions, "no_closer_pose"
        d, a, P, R = best
        actions.append(a)
    return actions, ""


def chunk(actions: Sequence[str], max_per_turn: int) -> List[List[str]]:
    """The SFT data's turns (jsrl_guide.chunk, chunking action_runs): each run of one action, at most max_per_turn."""
    turns: List[List[str]] = []
    for a in actions:
        if turns and turns[-1][-1] == a and len(turns[-1]) < max_per_turn:
            turns[-1].append(a)
        else:
            turns.append([a])
    return turns


def action_text(turn: Sequence[str]) -> str:
    return ACTION_OPEN + "".join(f"{a}|" for a in turn) + ACTION_CLOSE


# ---------------------------------------------------------------------------------------------------- tokens

def loss_tokens(tokenizer: Any, ids: Sequence[int]) -> List[int]:
    """The positions of ``ids`` (an answer's tokens) that carry the loss: every token holding a character from just
    after the first ``<action>`` through the end of the first ``</action>`` after it (the action names and the
    separators; not ``<action>``, not what follows ``</action>``).  [] when the answer has no such pair."""
    ids = [int(i) for i in ids]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    a = text.find(ACTION_OPEN)
    if a < 0:
        return []
    s = a + len(ACTION_OPEN)
    b = text.find(ACTION_CLOSE, s)
    if b < 0:
        return []
    e = b + len(ACTION_CLOSE)
    out, prev = [], 0
    for j in range(len(ids)):
        end = len(tokenizer.decode(ids[: j + 1], skip_special_tokens=False))
        if end > s and prev < e:
            out.append(j)
        prev = end
    return out


def end_id(tokenizer: Any) -> int:
    i = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if i is None or (isinstance(i, int) and i < 0):
        raise ValueError(f"the tokenizer has no {END_OF_TURN}")
    return int(i)


# ---------------------------------------------------------------------------------------------------- agent loop

def pre_turn(env: Any) -> Dict[str, Any]:
    """What the turn's judgment needs from the env before it steps: the pose and the step count."""
    inner = getattr(env, "_inner", env)
    return {"pose": np.array(inner.view_engine.get_pose(), dtype=np.float64, copy=True),
            "step": int(getattr(inner, "_current_step", 0))}


def _judge_turn(cfg: Dict[str, Any], env: Any, pre: Dict[str, Any], info: Dict[str, Any],
                written: Optional[List[str]] = None, canonical: bool = False) -> Dict[str, Any]:
    """The table side of a turn (runs in a worker thread): executed actions, their judgment, the guide's label.
    ``written``: the answer's action names (written_actions), ``canonical``: whether the answer is in the labels'
    form (canonical_answer).  An own answer is kept only when it is canonical and the env executed exactly the names
    it wrote, so its loss tokens never train an action the env skipped (stopped at the goal mid-turn, out of the
    step budget, past five actions, or not one the env allows).  A label is at most the step budget left."""
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys

    inner = getattr(env, "_inner", env)
    item = getattr(inner, "current_item", None) or {}
    task_id = str(item.get("task_id") or "")
    tab = table(cfg["tables"], task_id, cfg["table_cache"]) if task_id else None
    if tab is None:
        return {"status": "no_table"}
    allowed = getattr(inner, "_allowed_actions", None)
    detector = inner.collision_detector if getattr(inner.config, "enable_collision_detection", False) else None
    n_steps = int(getattr(inner, "_current_step", 0)) - int(pre["step"])
    parsed = list((info or {}).get("actions") or [])
    valid = bool((((info or {}).get("metrics") or {}).get("turn_metrics") or {}).get("action_is_valid", bool(parsed)))
    executed, done_hit = executed_actions(parsed if valid else [], allowed, n_steps)
    P, R = _split(pre["pose"])
    rec = judge(tab, P, R, executed, detector)
    rec.update(valid=valid, n_steps=n_steps, executed=executed, done_action=done_hit)
    if len(executed) != n_steps and not done_hit:
        rec["status"] = "step_count"  # the env moved its count otherwise than the replay
    if rec["status"] == "ok":
        P1, R1 = _split(np.asarray(inner.view_engine.get_pose(), dtype=np.float64))
        if int(contract_keys(P1, R1)[0]) != rec["end_key"]:
            rec["status"] = "pose_mismatch"
    if rec["status"] != "ok":
        return rec
    rec["written"] = written
    rec["canonical"] = bool(canonical)
    rec["written_executed"] = bool(canonical) and written is not None and list(written) == list(executed)
    rec["optimal"] = bool(executed) and all(rec["hits"]) and not done_hit and rec["written_executed"]
    if not rec["optimal"]:
        budget = int(getattr(inner, "_max_episode_steps", MAX_STEPS) or MAX_STEPS) - int(pre["step"])
        steps, stop = plan(tab, P, R, min(cfg["max_per_turn"], max(budget, 0)), detector, allowed)
        turns = chunk(steps, cfg["max_per_turn"])
        rec["plan"] = steps
        rec["plan_stop"] = stop
        rec["label_actions"] = turns[0] if turns else []
        if not turns:
            rec["status"] = "no_label"
    return rec


async def after_turn(cfg: Dict[str, Any], env: Any, pre: Dict[str, Any], info: Dict[str, Any], span: Any,
                     own_ids: Sequence[int], tokenizer: Any, turn_index: int) -> Dict[str, Any]:
    """A turn's record for the trainer (GymAgentLoop, after env.step).  A failure here is recorded, never raised."""
    import asyncio

    rec: Dict[str, Any] = {"t": int(turn_index), "span": list(span) if span is not None else None,
                           "pose": [float(x) for x in np.asarray(pre["pose"]).reshape(-1)]}
    try:
        # tokens in the event loop's thread (as the agent loop's own decode of the answer), the table work in a worker
        own_text = tokenizer.decode([int(i) for i in own_ids], skip_special_tokens=False)
        written = written_actions(own_text)
        rec.update(await asyncio.to_thread(_judge_turn, cfg, env, pre, info, written,
                                           canonical_answer(own_text, written)))
        if rec.get("status") == "ok":
            if rec["optimal"]:
                ids = [int(i) for i in own_ids]
                rec["own_ids"] = ids  # the trainer checks them against the row's tokens at the span
                rec["mask"] = loss_tokens(tokenizer, ids)
                body = ids[:-1] if ids and ids[-1] == end_id(tokenizer) else ids
                rec["retok_same"] = tokenizer.encode(tokenizer.decode(body, skip_special_tokens=False),
                                                     add_special_tokens=False) == body
                if not rec["mask"]:
                    rec["status"] = "no_loss_tokens"
            else:
                text = action_text(rec["label_actions"])
                ids = tokenizer.encode(text, add_special_tokens=False) + [end_id(tokenizer)]
                rec["label_text"] = text
                rec["label_ids"] = ids
                rec["mask"] = loss_tokens(tokenizer, ids)
                if not rec["mask"]:
                    rec["status"] = "no_loss_tokens"
    except Exception as exc:  # noqa: BLE001 - a labelling failure leaves the turn out, the episode goes on
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def episode_field(records: Sequence[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    """The episode's extra field (a dict, so that the trainer's object array keeps one per episode)."""
    return {"task_id": str(task_id), "turns": list(records)}


# ---------------------------------------------------------------------------------------------------- the trainer

def _np(x: Any) -> np.ndarray:
    import torch

    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def candidates(field: Sequence[Any], turns: str) -> Tuple[List[Tuple[int, Dict[str, Any]]], Counter]:
    """(row, turn record) of every turn that can be a sample, and counts of all turns by what they were."""
    out: List[Tuple[int, Dict[str, Any]]] = []
    c: Counter = Counter()
    for i, ep in enumerate(field):
        for t in ((ep or {}).get("turns") or []):
            st = t.get("status")
            c["turns"] += 1
            c[f"status_{st}"] += 1
            if st != "ok":
                continue
            c["judged"] += 1
            c["optimal" if t["optimal"] else "replaced"] += 1
            c["actions"] += len(t.get("executed") or [])
            c["action_hits"] += sum(bool(h) for h in t.get("hits") or [])
            c["invalid"] += int(not t.get("valid", True))
            c["retok_differs"] += int(t["optimal"] and not t.get("retok_same", True))
            # every executed action optimal, but the answer wrote more (or other) actions than the env executed
            c["own_unexecuted"] += int(not t["optimal"] and bool(t.get("executed")) and all(t.get("hits") or [False])
                                       and not t.get("done_action") and not t.get("written_executed", True))
            if turns == "replaced" and t["optimal"]:
                continue
            out.append((i, t))
    return out, c


def pick(cands: Sequence[Tuple[int, Dict[str, Any]]], keys: Sequence[Any], cap: int, world: int, seed: int,
         step: int) -> List[Tuple[int, Dict[str, Any]]]:
    """Up to ``cap`` of the candidates, a multiple of ``world``, a seeded random pick in a fixed order (by the rows'
    ``keys``, the task ids: the trainer's uids are random, so ordering by them would not repeat)."""
    order = sorted(range(len(cands)), key=lambda j: (str(keys[cands[j][0]]), cands[j][0], cands[j][1]["t"]))
    perm = np.random.default_rng([int(seed), int(step)]).permutation(len(order))
    n = min(int(cap), len(order))
    n -= n % int(world)
    return [cands[order[int(j)]] for j in perm[:n]]


def build_sample(row: Dict[str, Any], t: Dict[str, Any], tokens: Dict[str, int], merge: int) -> Dict[str, Any]:
    """One supervised sample from an episode row (input_ids, attention_mask, position_ids (4, L), responses,
    old_log_probs, multi_modal_inputs) and a turn record: the row's tokens before the turn, the label's tokens.
    Raises ValueError when the row cannot give it."""
    import torch

    ids_full, att = row["input_ids"], row["attention_mask"]
    pos_full, resp = row["position_ids"], row["responses"]
    total, n_resp = int(ids_full.shape[-1]), int(resp.shape[-1])
    n_prompt = total - n_resp
    s, e = int(t["span"][0]), int(t["span"][1])
    if not 0 <= s < e <= n_resp or not bool(att[n_prompt:n_prompt + e].all()):
        raise ValueError(f"turn span {t['span']} is not inside the row's response")
    prompt_idx = torch.nonzero(att[:n_prompt], as_tuple=False).flatten()
    if len(prompt_idx) and int(prompt_idx[-1]) != n_prompt - 1:
        raise ValueError("the prompt is not left-padded")
    idx = torch.cat([prompt_idx, n_prompt + torch.arange(s)])
    prefix = ids_full[idx]
    pos_prefix = pos_full[:, idx]
    own = bool(t["optimal"])
    label = resp[s:e].clone() if own else torch.tensor(t["label_ids"], dtype=ids_full.dtype)
    n_label = int(label.shape[0])
    pos_label = pos_prefix[:, -1:] + 1 + torch.arange(n_label, dtype=pos_full.dtype)[None, :]
    checks = {}
    if own:  # the exact parts of the check: the continued positions are the row's own, the answer the agent loop's
        checks["positions_same"] = bool(torch.equal(pos_label, pos_full[:, n_prompt + s:n_prompt + e]))
        checks["tokens_same"] = t.get("own_ids") is not None and label.tolist() == [int(i) for i in t["own_ids"]]
    mmi = row.get("multi_modal_inputs") or {}
    n_vs = int((prefix == tokens["vision_start"]).sum())
    n_pad = int((prefix == tokens["image_pad"]).sum())
    media: Dict[str, Any] = {}
    if n_vs:
        grid = mmi["image_grid_thw"]
        if n_vs > int(grid.shape[0]):
            raise ValueError(f"the prefix has {n_vs} images, the row's pixels {int(grid.shape[0])}")
        patches = int(grid[:n_vs].prod(dim=-1).sum())
        media = {"pixel_values": mmi["pixel_values"][:patches].clone(), "image_grid_thw": grid[:n_vs].clone()}
        checks["image_tokens_same"] = n_pad * merge * merge == patches
    else:
        checks["image_tokens_same"] = n_pad == 0
    ids = torch.cat([prefix, label])[None, :]
    mask = torch.zeros(n_label, dtype=torch.float32)
    mask[[int(j) for j in t["mask"]]] = 1.0
    sample = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
              "position_ids": torch.cat([pos_prefix, pos_label], dim=1)[None, :, :], "responses": label[None, :],
              "multi_modal_inputs": [media], "mask": mask[None, :], "kind": "own" if own else "label",
              "checks": checks, "meta": {"t": int(t["t"]), "n_prefix": int(prefix.shape[0]), "n_label": n_label,
                                         "images": n_vs, "whole_row": own and e == int(att[n_prompt:].sum()),
                                         "label_text": t.get("label_text"), "executed": t.get("executed"),
                                         "d_before": t.get("d_before")}}
    if own:
        sample["old_lp"] = row["old_log_probs"][s:e].detach().float().clone()[None, :]
    return sample


TURN_LOG_KEYS = ("t", "status", "optimal", "valid", "written", "canonical", "written_executed", "executed", "hits",
                 "blocked", "d_before", "d_after", "near", "done_action", "n_steps", "label_actions", "plan",
                 "plan_stop", "pose")


def _log_turns(cfg: Dict[str, Any], step: int, chosen: Sequence[Tuple[int, Dict[str, Any]]], field: Sequence[Any],
               nt: Dict[str, Any], samples: Sequence[Dict[str, Any]]) -> None:
    """RUN/dagger/turns.jsonl: the update's chosen turns (task, pose, judgment, label), for the offline check against
    the table (reward/dagger_check.py).  A failed write warns and never stops training."""
    if not cfg.get("log_dir"):
        return
    used = {(s["meta"]["row"], s["meta"]["t"]) for s in samples}  # uid is a task's (8 episodes share it): the row
    try:
        path = Path(cfg["log_dir"])
        path.mkdir(parents=True, exist_ok=True)
        with (path / "turns.jsonl").open("a") as f:
            for i, t in chosen:
                rec = {"step": int(step), "row": int(i), "uid": str(nt["uid"][i]),
                       "task_id": (field[i] or {}).get("task_id"), "sample": (int(i), t["t"]) in used,
                       **{k: t.get(k) for k in TURN_LOG_KEYS}}
                f.write(json.dumps(rec) + "\n")
    except Exception as exc:  # noqa: BLE001 - a log, never a reason to stop training
        print(f"dagger: could not write the turn log: {type(exc).__name__}: {exc}", flush=True)


def build_samples(batch: Any, cfg: Dict[str, Any], step: int, world: int, tokens: Dict[str, int], merge: int
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """The update's samples from the trainer's batch (before the group filter: every candidate episode), and the
    update's label metrics (dagger/*)."""
    t0 = time.time()
    nt = batch.non_tensor_batch
    field = nt.get("dagger_turns")
    # the switch is on, so every training update must bring judged turns: anything else would run as plain RL
    # unnoticed (the agent loop not labelling: no records, e.g. a sync rollout; tables unreadable; every turn out)
    if field is None or not any(field):
        raise RuntimeError(f"dagger: update {step} brought no turn records from the agent loop (is it the async "
                           f"GymAgentLoop with dagger_step in the batch?)")
    cands, c = candidates(list(field), cfg["turns"])
    if not c["judged"]:
        errors = [t.get("error") for ep in field for t in ((ep or {}).get("turns") or []) if t.get("error")][:3]
        raise RuntimeError(f"dagger: no turn of update {step} could be judged ({dict(c)}; first errors {errors}); "
                           f"are the tables ({cfg['tables']}) readable by the agent-loop workers?")
    chosen = pick(cands, [(ep or {}).get("task_id") for ep in field], cfg["cap"], world, cfg["seed"], step)
    samples: List[Dict[str, Any]] = []
    bad = Counter()
    b = batch.batch
    mmi_all = nt.get("multi_modal_inputs")
    for i, t in chosen:
        row = {k: b[k][i] for k in ("input_ids", "attention_mask", "position_ids", "responses", "old_log_probs")}
        row["multi_modal_inputs"] = mmi_all[i] if mmi_all is not None else None
        try:
            s = build_sample(row, t, tokens, merge)
        except Exception as exc:  # noqa: BLE001 - a row that cannot give its sample is counted and left out
            bad[type(exc).__name__] += 1
            continue
        failed = [k for k, ok in s["checks"].items() if not ok]
        for k in failed:
            bad[f"check_{k}"] += 1
        if failed:  # counted, and left out: a sample whose image tokens and pixels disagree would stop one rank
            bad["dropped_by_check"] += 1
            continue
        s["meta"]["uid"] = str(nt["uid"][i])
        s["meta"]["row"] = int(i)
        samples.append(s)
    if len(samples) % world:  # a row that could not give its sample: still the same number for every GPU
        samples = samples[: len(samples) - len(samples) % world]
    if chosen and not samples:  # every chosen turn failed to give its sample: not a plain-RL update unnoticed
        raise RuntimeError(f"dagger: none of the {len(chosen)} chosen turns of update {step} gave a sample "
                           f"({dict(bad)})")
    _log_turns(cfg, step, chosen, field, nt, samples)
    judged = max(c["judged"], 1)
    m = {"dagger/turns": float(c["turns"]), "dagger/turns_judged": float(c["judged"]),
         "dagger/judged_share": c["judged"] / max(c["turns"], 1),
         "dagger/optimal_share": c["optimal"] / judged, "dagger/replaced_share": c["replaced"] / judged,
         "dagger/action_hit_rate": c["action_hits"] / max(c["actions"], 1),
         "dagger/invalid_turns": float(c["invalid"]), "dagger/retok_differs": float(c["retok_differs"]),
         "dagger/candidates": float(len(cands)), "dagger/samples": float(len(samples)),
         "dagger/samples_own": float(sum(s["kind"] == "own" for s in samples)),
         "dagger/samples_label": float(sum(s["kind"] == "label" for s in samples)),
         "dagger/build_failed": float(sum(v for k, v in bad.items()
                                          if not k.startswith("check_") and k != "dropped_by_check")),
         "dagger/dropped_by_check": float(bad.get("dropped_by_check", 0)),
         "dagger/own_unexecuted": float(c["own_unexecuted"]),
         "dagger/build_seconds": time.time() - t0}
    for k, v in c.items():
        if k.startswith("status_") and k != "status_ok":
            m[f"dagger/{k}"] = float(v)
    for k in ("check_positions_same", "check_tokens_same", "check_image_tokens_same"):
        m[f"dagger/{k}_failed"] = float(bad.get(k, 0))
    return samples, m


def attach(batch: Any, samples: Sequence[Dict[str, Any]], world: int, step: Optional[int] = None) -> None:
    """Put each GPU's share of the samples on the first row of its chunk of the batch (update_actor gives rank r the
    r-th of ``world`` equal chunks), the total and the update number in meta_info."""
    n_rows = len(batch)
    if n_rows % world or len(samples) % world:
        raise ValueError(f"{n_rows} rows / {len(samples)} samples do not split over {world} GPUs")
    per_row, per_rank = n_rows // world, len(samples) // world
    field = np.empty(n_rows, dtype=object)
    field[:] = [None] * n_rows
    for r in range(world):
        field[r * per_row] = list(samples[r * per_rank:(r + 1) * per_rank])
    batch.non_tensor_batch["dagger_samples"] = field
    batch.meta_info["dagger_samples"] = len(samples)
    batch.meta_info["dagger_step"] = None if step is None else int(step)


def detach(batch: Any) -> None:
    batch.non_tensor_batch.pop("dagger_samples", None)
    batch.meta_info.pop("dagger_samples", None)
    batch.meta_info.pop("dagger_step", None)


# ---------------------------------------------------------------------------------------------------- the actor

CHECK_GROUPS = ("whole_row", "prefix")


def pack_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Two samples as one two-row micro-batch, left-padded to the same length (the actor's forward unpads them into
    one packed sequence of both, so ``a``'s tokens meet another batch shape but no other token); and the number of
    ``a``'s label tokens: its log-probabilities are the last that many of row 0."""
    import torch

    la, lb = int(a["input_ids"].shape[1]), int(b["input_ids"].shape[1])
    na, nb = int(a["responses"].shape[1]), int(b["responses"].shape[1])
    length, n = max(la, lb), max(na, nb)
    ids = torch.zeros((2, length), dtype=a["input_ids"].dtype)
    att = torch.zeros((2, length), dtype=a["attention_mask"].dtype)
    pos = torch.ones((2, a["position_ids"].shape[1], length), dtype=a["position_ids"].dtype)
    for r, (s, ls) in enumerate(((a, la), (b, lb))):
        ids[r, length - ls:] = s["input_ids"][0]
        att[r, length - ls:] = 1
        pos[r, :, length - ls:] = s["position_ids"][0]
    media = [{**a["multi_modal_inputs"][0]}, {**b["multi_modal_inputs"][0]}]
    media = [m for m in media if m] or [{}]
    if len(media) == 2:  # one dict, its images in token order: row 0's, then row 1's
        merged: Dict[str, Any] = {}
        for k in media[0]:
            merged[k] = torch.cat([media[0][k], media[1][k]], dim=0) if k in media[1] else media[0][k]
        media = [merged]
    return {"input_ids": ids, "attention_mask": att, "position_ids": pos, "responses": ids[:, length - n:],
            "multi_modal_inputs": media}, na


class DaggerMixer:
    """verl's DataParallelPPOActor hands the update's batch to ``take_batch`` before it selects its keys, calls
    ``before_optimizer_step`` right before each optimizer step and ``end_of_update`` at the end of update_policy.
    The samples go into the update's first optimizer step only."""

    def __init__(self, cfg: Dict[str, Any], actor: Any):
        import torch.distributed as dist

        self.cfg = {**DEFAULTS, **{k: v for k, v in dict(cfg).items()}}
        self.actor = actor
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        if getattr(actor, "scaler", None) is not None:
            raise ValueError("dagger: fp16 loss scaling is not supported")
        if getattr(actor, "use_ulysses_sp", False):
            raise ValueError("dagger: Ulysses sequence parallelism is not supported")
        self.group = None
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(actor.actor_module, FSDP):
                self.group = actor.actor_module.process_group
        except ImportError:
            pass
        self.log_dir = Path(self.cfg["log_dir"]) if self.cfg.get("log_dir") else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None
        if self.cfg.get("tokenizer"):  # loaded at the actor's start, so a bad path stops the job before a rollout
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.cfg["tokenizer"]))
        self.pending: List[Dict[str, Any]] = []
        self.expected = 0
        self.temperature = 1.0
        self.step = None
        self.done_this_update = False
        self.first_written = False
        self.counts: Counter = Counter()

    # the same small helpers as und_mix.UndMixer
    def _reduce(self, values: List[Any], op: str = "sum") -> List[float]:
        import torch
        import torch.distributed as dist

        t = torch.tensor([float(v) for v in values], dtype=torch.float64)
        if dist.is_initialized() and self.world > 1:
            from verl.utils.device import get_device_id

            t = t.to(get_device_id())
            dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM, group=self.group)
        return [float(x) for x in t.cpu().tolist()]

    def _device(self) -> Any:
        import torch

        for p in self.actor.actor_module.parameters():
            return p.device
        return torch.device("cpu")

    def _repair(self) -> None:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from verl.utils.fsdp_utils import repair_unreduced_fsdp1_grad_shards_
        except ImportError:
            return
        if isinstance(self.actor.actor_module, FSDP):
            repair_unreduced_fsdp1_grad_shards_(self.actor.actor_module)

    def _write(self, name: str, text: str, mode: str = "w") -> None:
        if self.log_dir is None:
            return
        try:
            with (self.log_dir / name).open(mode) as f:
                f.write(text)
        except Exception as exc:  # noqa: BLE001 - after the collectives; a log never stops one rank
            print(f"dagger: could not write {self.log_dir / name}: {type(exc).__name__}: {exc}", flush=True)

    def _dump(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"unserialisable": f"{type(exc).__name__}: {exc}"})

    def take_batch(self, data: Any) -> None:
        field = data.non_tensor_batch.get("dagger_samples")
        mine: List[Dict[str, Any]] = []
        if field is not None:
            for x in field:
                if x:
                    mine.extend(x)
        self.pending = mine
        self.expected = int(data.meta_info.get("dagger_samples", 0) or 0)
        # the policy's own temperature (the trainer's old_log_probs and the PPO loss use it; 1.0 in the line's runs)
        self.temperature = float(data.meta_info.get("temperature", 1.0))
        self.step = data.meta_info.get("dagger_step")
        self.done_this_update = False

    def _skip(self, t0: float, reason: str, params: Sequence[Any]) -> Dict[str, float]:
        from vagen.und_mix import _dot

        norm = math.sqrt(max(self._reduce([sum(_dot(p.grad, p.grad) for p in params if p.grad is not None)])[0], 0.0))
        if self.rank == 0:
            self._write("steps.jsonl", self._dump({"utc": _utc(), "step": self.step, "inserted": False,
                                                   "reason": reason, "grad_norm_policy": norm,
                                                   "seconds": time.time() - t0}) + "\n", "a")
        return {"actor/dagger_inserted": 0.0, "actor/dagger_dropped": float(reason != "none"),
                "actor/grad_norm_policy": norm, "actor/dagger_seconds": time.time() - t0}

    def before_optimizer_step(self) -> Dict[str, float]:
        import torch

        from vagen.und_mix import MEASURES, mix_gradients

        t0 = time.time()
        params = [p for p in self.actor.actor_module.parameters() if p.requires_grad]
        self._repair()
        self.counts["opt_steps"] += 1
        if self.done_this_update:  # the same on every rank: no collective, no metrics (they would be averaged in)
            return {}
        self.done_this_update = True
        n_mine = len(self.pending)
        # every rank must run the same number of forward / backward passes: the trainer gives each the same share
        wrong, n_all = self._reduce([float(n_mine * self.world != self.expected), float(n_mine)])
        if wrong or n_all != self.expected:
            return self._skip(t0, "uneven", params)
        if self.expected == 0:
            return self._skip(t0, "none", params)
        n_tok = self._reduce([sum(float(s["mask"].sum()) for s in self.pending)])[0]
        scale = float(self.cfg["lam"]) * self.world / max(n_tok, 1.0)
        device = self._device()
        per: List[Dict[str, Any]] = []
        solo: List[Any] = []  # each sample's log-probabilities from its own forward (the floor's reference)
        chk = {g: [0.0, 0.0, 0.0, 0.0, 0.0] for g in CHECK_GROUPS}  # tokens, sum |d|, max |d|, > 1e-2, > 1e-1
        chk_mask = [0.0, 0.0, 0.0]  # the loss tokens of own samples: tokens, sum |d|, max |d|

        def _moved(s: Dict[str, Any]) -> Dict[str, Any]:
            moved = {k: s[k].to(device) for k in ("input_ids", "attention_mask", "position_ids", "responses")}
            moved["multi_modal_inputs"] = [{k: v.to(device) for k, v in m.items()} for m in s["multi_modal_inputs"]]
            return moved

        def _add(g: List[float], diff: Any) -> None:
            g[0] += diff.numel()
            g[1] += float(diff.sum())
            g[2] = max(g[2], float(diff.max()) if diff.numel() else 0.0)
            g[3] += float((diff > 1e-2).sum())
            g[4] += float((diff > 1e-1).sum())

        def sup_backward() -> float:
            total = 0.0
            for s in self.pending:
                _, log_prob = self.actor._forward_micro_batch(_moved(s), temperature=self.temperature,
                                                              calculate_entropy=False)
                mask = s["mask"].to(log_prob.device)
                nll = -(log_prob * mask).sum()
                (nll * scale).backward()
                lp = log_prob.detach().float().cpu()
                solo.append(lp)
                rec = {"kind": s["kind"], "nll_sum": float(nll.detach()), "loss_tokens": int(s["mask"].sum())}
                if s["kind"] == "own":
                    diff = (lp - s["old_lp"]).abs()[0]
                    _add(chk["whole_row" if s["meta"]["whole_row"] else "prefix"], diff)
                    md = diff[s["mask"][0].bool()]
                    chk_mask[0] += md.numel()
                    chk_mask[1] += float(md.sum())
                    chk_mask[2] = max(chk_mask[2], float(md.max()) if md.numel() else 0.0)
                    rec["max_abs_diff"] = float(diff.max())
                per.append(rec)
                total += float(nll.detach())
            self._repair()
            return total

        orig = {p: p.grad for p in params}
        stats = mix_gradients(params, sup_backward, self._reduce)
        audit = bool(self.cfg.get("audit_only"))
        floor = [0.0, 0.0, 0.0, 0.0, 0.0]
        if audit:  # measured, then dropped: the policy's own gradient goes on to the optimizer step
            for p in params:
                p.grad = orig[p]
            # the floor of the prefix check: the same sample's log-probabilities when it shares a batch with another
            # one (the same tokens, another batch shape: what a correct prefix can differ by in bf16); one extra
            # forward per sample on every rank (each holds the same number), no gradient
            n = len(self.pending)
            with torch.no_grad():
                for j, s in enumerate(self.pending):
                    batch2, n_a = pack_pair(s, self.pending[(j + 1) % n])
                    _, lp2 = self.actor._forward_micro_batch(_moved(batch2), temperature=self.temperature,
                                                             calculate_entropy=False)
                    lp2 = lp2.detach().float().cpu()[0, lp2.shape[1] - n_a:]
                    _add(floor, (lp2 - solo[j][0]).abs())
        groups = [chk[g] for g in CHECK_GROUPS] + [floor]
        sums = self._reduce([g[k] for g in groups for k in (0, 1, 3, 4)] + [chk_mask[0], chk_mask[1]])
        maxes = self._reduce([g[2] for g in groups] + [chk_mask[2]], op="max")
        check: Dict[str, Any] = {"step": self.step}
        for gi, g in enumerate(list(CHECK_GROUPS) + ["shape_floor"]):
            n_t, sabs, over2, over1 = sums[gi * 4:gi * 4 + 4]
            check[g] = {"tokens": int(n_t), "mean_abs": sabs / n_t if n_t else None,
                        "max_abs": maxes[gi] if n_t else None, "over_1e-2": int(over2), "over_1e-1": int(over1)}
        check["loss_tokens"] = {"tokens": int(sums[-2]), "mean_abs": sums[-1] / sums[-2] if sums[-2] else None,
                                "max_abs": maxes[-1] if sums[-2] else None}
        n_own = self._reduce([float(sum(s["kind"] == "own" for s in self.pending))])[0]
        seconds = time.time() - t0
        # logs after every collective of the step: a failed write warns and never holds one rank back
        if not self.first_written:
            rows = []
            tok = self.tokenizer
            for s, rec in zip(self.pending, per):
                ids = s["responses"][0].tolist()
                keep = [i for i, m in zip(ids, s["mask"][0].tolist()) if m]
                try:
                    dec = (tok.decode(ids, skip_special_tokens=False), tok.decode(keep, skip_special_tokens=False)) \
                        if tok else (None, None)
                except Exception as exc:  # noqa: BLE001
                    dec = (f"decode failed: {exc}", None)
                rows.append({**rec, **{k: v for k, v in s["meta"].items()}, "checks": s["checks"],
                             "label_ids": ids, "loss_ids": keep, "label_decoded": dec[0], "loss_decoded": dec[1]})
            tag = f"_step{int(self.step):04d}" if self.step is not None else ""  # a resumed run writes its own file
            self._write(f"first_batch_rank{self.rank}{tag}.json",
                        self._dump({"utc": _utc(), "step": self.step, "samples": rows}))
            self.first_written = True
        self.counts["inserted"] += int(stats["finite"] and not audit)
        out = {"actor/dagger_inserted": float(stats["finite"] and not audit), "actor/dagger_audit": float(audit),
               "actor/dagger_dropped": float(not stats["finite"]), "actor/dagger_samples": float(self.expected),
               "actor/dagger_samples_own": n_own, "actor/dagger_tokens": n_tok,
               "actor/grad_norm_policy": stats["grad_norm_policy"], "actor/dagger_seconds": seconds}
        if stats["finite"]:
            out["actor/dagger_loss"] = stats["loss_sum"] / max(n_tok, 1.0)
            out.update({f"actor/dagger_{k}": stats[k] for k in MEASURES})
            out["actor/dagger_grad_norm_total_preclip"] = stats["grad_norm_total"]
        for g in list(CHECK_GROUPS) + ["shape_floor"]:
            if check[g]["tokens"]:
                out[f"actor/dagger_check_{g}_max_abs"] = check[g]["max_abs"]
                out[f"actor/dagger_check_{g}_mean_abs"] = check[g]["mean_abs"]
        if self.rank == 0:
            self._write("steps.jsonl", self._dump({"utc": _utc(), "step": self.step,
                                                   "inserted": bool(stats["finite"] and not audit),
                                                   "audit_only": audit, "samples": self.expected, "own": int(n_own),
                                                   "loss_tokens": n_tok, "lam": float(self.cfg["lam"]),
                                                   **{k: v for k, v in stats.items() if k != "finite"},
                                                   "finite": bool(stats["finite"]), "seconds": seconds}) + "\n", "a")
            self._write("check.jsonl", self._dump({"utc": _utc(), **check}) + "\n", "a")
        self.pending = []
        return out

    def end_of_update(self) -> Dict[str, float]:
        out = {"actor/dagger_batches": float(self.counts["inserted"]),
               "actor/dagger_update_opt_steps": float(self.counts["opt_steps"])}
        self.counts = Counter()
        self.pending = []
        return out
