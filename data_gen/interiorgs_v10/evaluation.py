"""InteriorGS v10 evaluation (plan of 2026-10-14, round 6; run after training).

A fixed subset of the test split's formal tasks, the same env as RL (default v10 success rules,
``r1_legacy_success`` off), one greedy episode per task and model, on one card (render service and vLLM
share it):

    python -m data_gen.interiorgs_v10.evaluation select --run RUN/output --split-manifest RUN/split_manifest.json \\
        --out EVAL                                      # EVAL/tasks.jsonl + EVAL/selection.json
    # round 10: the fixed validation set (about 200 val tasks: --split val --per-family 34), tasks flagged by the
    # depth check excluded up front (--exclude depth_check_summary.json)
    # per model, a one-card sco job (sco_interiorgs_v10_train.sh eval RUN EVAL NAME MODEL [PROMPT_FORMAT]):
    python -m data_gen.interiorgs_v10.evaluation run --eval EVAL --name NAME --model HF_DIR
    python -m data_gen.interiorgs_v10.evaluation table --eval EVAL     # EVAL/table.md, EVAL/table.json
    python -m data_gen.interiorgs_v10.evaluation terminal --eval EVAL  # round 12 batch: EVAL/terminal.md / .json:
                                                                        # end progress, conditions met, collisions
    # round 7: drop tasks flagged by the rendered-depth check, refill from the same family and d* band
    python -m data_gen.interiorgs_v10.evaluation replace --eval EVAL --flagged depth_check_summary.json \
        --run RUN/output --split-manifest RUN/split_manifest.json --out EVAL2

Subset: ``PER_FAMILY`` tasks per family (all of a family's test tasks when it has fewer), spread evenly over
the d* bands 1-3 / 4-6 / 7-8 (a band short of tasks gives its share to the others), drawn with ``SEED``.
Each results row also holds the trajectory (frames, answers, actions; round 8) for the browsing page.
Metrics per model: success rate (overall, per family, per d* band), actions taken / d* on successes,
collision rate (blocked moves per action) and invalid-answer rate (turns whose answer did not parse).

Round 11: ``run --samples N --temperature T --top-p P`` runs N sampled episodes per task, as in RL training
(temperature 0.8, top-p 0.92), instead of one greedy episode; each row carries its ``sample`` number and the
metrics add the share of tasks with at least one success (``pass_at_n``) and the mean over the N episodes.

Round 12: ``top-up --eval V1 --family F --target N --split S ... --out V2`` keeps every task of V1 and adds
reserve tasks of family F from the same split (not flagged by the depth check, replayed in the env: success exactly
at d*, no collision; d* bands evened out) until V2 has N tasks of F (or the reserve runs out); ``--family-out``
also writes F's tasks of V2 alone, to evaluate existing models on them only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PER_FAMILY = 50
SEED = 20261014
BANDS = ((1, 3), (4, 6), (7, 8))
VERSION = "interiorgs_v10_eval_v2"  # v2 (round 8): results rows carry the trajectory and frames


def band_name(d: int) -> str:
    for lo, hi in BANDS:
        if lo <= d <= hi:
            return f"{lo}-{hi}"
    return "other"


def _jsonl(path: Path, partial_ok: bool = False) -> List[Dict[str, Any]]:
    """The rows of a JSON-lines file.  With ``partial_ok`` a last line that does not parse is left out: a results file
    another chain's evaluation job is still writing (round 12 batch: a table over a shared evaluation folder failed on
    such a line)."""
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if partial_ok and i == len(lines) - 1:
                break
            raise
    return rows


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
def select_tasks(rows: Sequence[Dict[str, Any]], per_family: int = PER_FAMILY, seed: int = SEED) -> List[Dict[str, Any]]:
    """Stratified by family, then by d* band; deterministic for a seed and a row set (any order)."""
    rng = random.Random(seed)
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: r["task_id"]):
        by_family[r["task_type"]][band_name(int(r["difficulty"]["d_star"]))].append(r)
    picked: List[Dict[str, Any]] = []
    for fam in sorted(by_family):
        bands = {b: list(v) for b, v in by_family[fam].items()}
        for v in bands.values():
            rng.shuffle(v)
        want = min(per_family, sum(len(v) for v in bands.values()))
        take = {b: 0 for b in bands}
        while sum(take.values()) < want:  # one at a time to the band with the fewest picks that still has tasks
            open_bands = [b for b in sorted(bands) if take[b] < len(bands[b])]
            b = min(open_bands, key=lambda b: (take[b], b))
            take[b] += 1
        for b in sorted(bands):
            picked += bands[b][: take[b]]
    return picked


def split_rows(run: Path, split_manifest: Path, split: str) -> List[Dict[str, Any]]:
    """The formal tasks of one split as published rows."""
    from .release import finalize_task_row, split_lookup

    lookup = split_lookup(json.loads(Path(split_manifest).read_text()))
    return [finalize_task_row(r, split) for r in _jsonl(Path(run) / "tasks.jsonl")
            if lookup.get(str(r["scene_id"])) == split]


def flagged_ids(path: Optional[str]) -> set:
    """Task ids of a depth_check_summary.json (round 10: excluded from the fixed sets up front)."""
    return set(json.loads(Path(path).read_text())["flagged_task_ids"]) if path else set()


def cmd_select(args: argparse.Namespace) -> None:
    split = getattr(args, "split", None) or "test"
    rows = split_rows(Path(args.run), Path(args.split_manifest), split)
    excluded = flagged_ids(getattr(args, "exclude", None))
    pool = [r for r in rows if r["task_id"] not in excluded]
    picked = select_tasks(pool, args.per_family, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in picked))
    cells = Counter((r["task_type"], band_name(int(r["difficulty"]["d_star"]))) for r in picked)
    avail = Counter((r["task_type"], band_name(int(r["difficulty"]["d_star"]))) for r in pool)
    selection = {"version": VERSION, "seed": args.seed, "per_family": args.per_family, "split": split,
                 f"{split}_tasks": len(rows), "excluded_flagged": len(rows) - len(pool),
                 "picked": len(picked), "scenes": len({r["scene_id"] for r in picked}),
                 "cells": {f"{f}|{b}": {"picked": n, "available": avail[(f, b)]} for (f, b), n in sorted(cells.items())},
                 "task_ids": [r["task_id"] for r in picked]}
    (out / "selection.json").write_text(json.dumps(selection, indent=1))
    print(json.dumps({k: v for k, v in selection.items() if k != "task_ids"}, indent=1))


def replace_flagged(picked: Sequence[Dict[str, Any]], flagged: Sequence[str], pool: Sequence[Dict[str, Any]],
                    seed: int = SEED) -> Dict[str, Any]:
    """Each flagged task of ``picked`` is replaced by an unflagged task of ``pool`` (not yet picked) from the same
    family and d* band, else from the same family; a task with no substitute is dropped.  The others stay."""
    flagged = set(flagged)
    cell = lambda r: (r["task_type"], band_name(int(r["difficulty"]["d_star"])))  # noqa: E731
    used = {r["task_id"] for r in picked}
    cand: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["task_id"]):
        if r["task_id"] not in flagged and r["task_id"] not in used:
            cand[cell(r)].append(r)
    rng = random.Random(seed)
    for v in cand.values():
        rng.shuffle(v)
    out, swaps = [], []
    for r in picked:
        if r["task_id"] not in flagged:
            out.append(r)
            continue
        c = cell(r)
        src = cand[c] or next((cand[k] for k in sorted(cand) if k[0] == c[0] and cand[k]), [])
        new = src.pop() if src else None
        if new is not None:
            out.append(new)
        swaps.append({"out": r["task_id"], "in": new["task_id"] if new else None, "cell": list(c)})
    return {"tasks": out, "swaps": swaps}


def cmd_replace(args: argparse.Namespace) -> None:
    picked = _jsonl(Path(args.eval) / "tasks.jsonl")
    flagged = sorted(flagged_ids(args.flagged))
    pool = split_rows(Path(args.run), Path(args.split_manifest), getattr(args, "split", None) or "test")
    res = replace_flagged(picked, flagged, pool, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res["tasks"]))
    summary = {"version": VERSION, "seed": args.seed, "source": str(args.eval), "tasks": len(res["tasks"]),
               "flagged_in_source": len(res["swaps"]), "replaced": sum(s["in"] is not None for s in res["swaps"]),
               "dropped": sum(s["in"] is None for s in res["swaps"]), "swaps": res["swaps"],
               "task_ids": [r["task_id"] for r in res["tasks"]]}
    (out / "selection.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("swaps", "task_ids")}, indent=1))


def reserve_rows(run: Path, split_manifest: Path, split: str) -> List[Dict[str, Any]]:
    """Round 12: the reserve tasks (tasks_reserve.jsonl) of one split as published rows."""
    from .release import finalize_task_row, split_lookup

    lookup = split_lookup(json.loads(Path(split_manifest).read_text()))
    path = Path(run) / "tasks_reserve.jsonl"
    return [finalize_task_row(r, split) for r in (_jsonl(path) if path.is_file() else [])
            if lookup.get(str(r["scene_id"])) == split]


def balanced_order(rows: Sequence[Dict[str, Any]], existing: Counter, seed: int = SEED) -> List[Dict[str, Any]]:
    """``rows`` in the order a top-up takes them: each next one from the d* band with the fewest tasks so far
    (``existing`` counts the set's tasks per band), random within a band (``seed``)."""
    rng = random.Random(seed)
    bands: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["task_id"]):
        bands[band_name(int(r["difficulty"]["d_star"]))].append(r)
    for v in bands.values():
        rng.shuffle(v)
    counts = Counter(existing)
    order = []
    while any(bands.values()):
        b = min((b for b in sorted(bands) if bands[b]), key=lambda b: (counts[b], b))
        order.append(bands[b].pop(0))
        counts[b] += 1
    return order


def top_up(tasks: Sequence[Dict[str, Any]], pool: Sequence[Dict[str, Any]], family: str, target: int,
           replay_fn: Callable[[Sequence[Dict[str, Any]]], Sequence[Dict[str, Any]]],
           flagged: Sequence[str] = (), seed: int = SEED) -> Dict[str, Any]:
    """Round 12 (the user's decision for the occlusion family): add tasks of ``family`` from ``pool`` (reserve
    tasks of the same split) until the set has ``target`` of them: flagged tasks and tasks already in the set are
    never taken; candidates in balanced_order (d* bands evened out), each replayed in the env (``replay_fn``, rows
    -> records with "ok") and taken only if the replay succeeds exactly at d* without a collision.  The set's other
    tasks stay as they are."""
    have = {r["task_id"] for r in tasks}
    flagged = set(flagged)
    existing = Counter(band_name(int(r["difficulty"]["d_star"])) for r in tasks if r["task_type"] == family)
    want = max(0, int(target) - sum(existing.values()))
    cands = [r for r in pool if r["task_type"] == family and r["task_id"] not in have and r["task_id"] not in flagged]
    order = balanced_order(cands, existing, seed)
    added: List[Dict[str, Any]] = []
    replayed: List[Dict[str, Any]] = []
    i = 0
    while len(added) < want and i < len(order):
        batch = order[i:i + want - len(added)]
        i += len(batch)
        results = {r["task_id"]: r for r in replay_fn(batch)}
        for row in batch:
            rec = results.get(row["task_id"]) or {"task_id": row["task_id"], "ok": False, "error": "not replayed"}
            replayed.append(rec)
            if rec.get("ok"):
                added.append(row)
    return {"tasks": list(tasks) + added, "added": added, "replayed": replayed, "wanted": want,
            "candidates": len(cands), "family_before": dict(existing),
            "family_after": dict(existing + Counter(band_name(int(r["difficulty"]["d_star"])) for r in added))}


def cmd_top_up(args: argparse.Namespace) -> None:
    from . import replay_env

    split = args.split
    tasks = _jsonl(Path(args.eval) / "tasks.jsonl")
    pool = reserve_rows(Path(args.run), Path(args.split_manifest), split)
    rules = replay_env.template_rules(Path(args.env_template))
    res = top_up(tasks, pool, args.family, args.target,
                 lambda rows: replay_env.replay(rows, Path(args.scenes_root), rules),
                 sorted(flagged_ids(getattr(args, "exclude", None))), args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res["tasks"]))
    summary = {"version": VERSION, "seed": args.seed, "source": str(args.eval), "split": split, "family": args.family,
               "target": args.target, "tasks": len(res["tasks"]), "source_tasks": len(tasks),
               "family_before": res["family_before"], "family_after": res["family_after"],
               "reserve_candidates": res["candidates"], "wanted": res["wanted"], "added": len(res["added"]),
               "replayed": len(res["replayed"]), "replay_failed": [r for r in res["replayed"] if not r.get("ok")],
               "added_task_ids": [r["task_id"] for r in res["added"]], "task_ids": [r["task_id"] for r in res["tasks"]]}
    (out / "selection.json").write_text(json.dumps(summary, indent=1))
    if getattr(args, "family_out", None):  # the family's tasks alone (to evaluate existing models on them only)
        fam = Path(args.family_out)
        fam.mkdir(parents=True, exist_ok=True)
        rows = [r for r in res["tasks"] if r["task_type"] == args.family]
        (fam / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (fam / "selection.json").write_text(json.dumps({**summary, "tasks": len(rows), "subset_of": str(out),
                                                        "task_ids": [r["task_id"] for r in rows]}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("task_ids", "added_task_ids")}, indent=1))


# ---------------------------------------------------------------------------
# run (one card: the render service and vLLM)
# ---------------------------------------------------------------------------
def env_config(env_template: Path, tasks: Path, gs_root: str, prompt_format: Optional[str],
               overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import os

    import yaml

    cfg = dict(yaml.safe_load(env_template.read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(tasks.resolve()), gs_root=gs_root, client_url=os.environ["R1_RENDER_URL"],
               r1_legacy_success=False)
    if prompt_format:
        cfg["prompt_format"] = prompt_format
    cfg.update(overrides or {})
    return cfg


def switch_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """The env settings changed by the round-9 switches given on the command line (none: the template's rules)."""
    out: Dict[str, Any] = {}
    if getattr(args, "check_each_action", False):
        out["success_check_each_action"] = True
    if getattr(args, "max_actions", None):
        out["max_episode_steps"] = int(args.max_actions)
    if getattr(args, "no_collision_end", False):
        out["max_consecutive_collisions"] = 0  # blocked moves still cost -0.15 each
    if getattr(args, "collision_feedback", False):
        out["collision_feedback_detail"] = True
    return out


def gate_summary(metric: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The success conditions not met at a pose (``failed_gates``) and the 0..1 progress toward success."""
    if not metric:
        return {"failed_gates": None, "progress": None}
    progress = metric.get("continuous_gate_progress", metric.get("shaping_score"))
    return {"failed_gates": sorted(k for k, v in (metric.get("gates") or {}).items() if not v),
            "progress": None if progress is None else round(float(progress), 4)}


def _images(obs: Dict[str, Any]) -> List[Any]:
    return [im for v in (obs.get("multi_modal_data") or {}).values() for im in (v or [])]


def sft_text(text: str) -> str:
    """A message as the SFT data has it: the LLaMA-Factory export (prepare_active_spatial_sft.py) passes every
    message through this function, which collapses 3+ newlines to 2 and strips leading and trailing whitespace;
    the env's text keeps them (its system prompt and later user turns end with a newline)."""
    from data_gen.active_spatial_sft.convert_to_qwen25vl_sft import _strip_think_from_content

    return _strip_think_from_content(text)


def chat_messages(system: str, earlier: Sequence[Tuple[str, Any, str]], user_text: str, image: Any,
                  image_part: Callable[[Any], Dict[str, Any]], as_sft: bool = False) -> List[Dict[str, Any]]:
    """The chat for one model call, in the layout of the SFT conversations: the system prompt, then every earlier
    turn (its user text with the frame where "<image>" stands, and the model's answer), then the current user
    turn.  ``earlier`` is empty without history (the RL input since round 6).  ``as_sft``: every message as the
    SFT data has it (``sft_text``)."""
    text = sft_text if as_sft else (lambda t: t)

    def user(raw: str, im: Any) -> Dict[str, Any]:
        before, _, after = text(raw).partition("<image>")
        content = ([{"type": "text", "text": before}] if before else []) + [image_part(im)]
        content += [{"type": "text", "text": after}] if after else []
        return {"role": "user", "content": content}

    messages: List[Dict[str, Any]] = [{"role": "system", "content": text(system)}]
    for raw, im, answer in earlier:
        messages += [user(raw, im), {"role": "assistant", "content": text(answer)}]
    return messages + [user(user_text, image)]


def call_seed(base: int, index: int, sample: int, turn: int) -> int:
    """The sampling seed of one model call: fixed per (task, sample, turn), so a sampled run is reproducible and the
    N samples of a task differ."""
    import hashlib

    digest = hashlib.sha256(f"{base}:{index}:{sample}:{turn}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def run_episode(env: Any, policy: Any, index: int, max_turns: int, frames: Optional[Path] = None,
                history: bool = False, as_sft: bool = False, sample: Optional[int] = None,
                seed: Optional[int] = None) -> Dict[str, Any]:
    """One greedy episode; per-turn bookkeeping as in scripts/r1_run_canonical_dev_eval32.py.

    With ``frames`` (round 8) the row also carries the trajectory: per turn the frame the model saw (a JPEG in
    ``frames``, named <index>_<turn>.jpg), the user text, the answer, the parsed actions, blocked moves, reward
    and whether the task was met after the turn; then the frame after the last turn and the env's end reason.
    With ``history`` (round 9 switch) every model call also gets all earlier turns of the episode, as in SFT;
    with ``as_sft`` every message is passed through ``sft_text`` first.  ``sample`` / ``seed`` (round 11): the
    sample number of a sampled run and the base seed of its model calls (``call_seed``); frames are then named
    <index>_s<sample>_<turn>.jpg."""
    obs, info = env.reset(seed=index)
    system = env.system_prompt()
    item = env.current_item or {}
    turns, invalid, attempted, collisions, success, actions_to_success = 0, 0, 0, 0, False, None
    done, trajectory, end_reason, earlier = False, [], None, []
    start = gate_summary(env._calculate_canonical_metric()) if frames is not None else None

    def keep(image: Any, turn: int) -> Optional[str]:
        if frames is None or image is None:
            return None
        frames.mkdir(parents=True, exist_ok=True)
        name = f"{index:03d}_{turn:02d}.jpg" if sample is None else f"{index:03d}_s{sample}_{turn:02d}.jpg"
        image.convert("RGB").save(frames / name, format="JPEG", quality=85)
        return f"{frames.name}/{name}"

    while not done and turns < max_turns:
        images = _images(obs)
        if len(images) != 1:
            raise RuntimeError(f"expected one observation image, got {len(images)}")
        kw: Dict[str, Any] = {}
        if history:
            kw["earlier"] = list(earlier)
        if as_sft:
            kw["as_sft"] = True
        if seed is not None:
            kw["seed"] = call_seed(seed, index, sample or 0, turns)
        text = policy.generate(system, obs["obs_str"], images[0], **kw)
        if history:
            earlier.append((obs["obs_str"], images[0], text))
        seen = {"frame": keep(images[0], turns), "user": obs["obs_str"]}
        if getattr(policy, "last_prompt_tokens", None) is not None:
            seen["prompt_tokens"] = policy.last_prompt_tokens
        obs, reward, done, step_info = env.step(text)
        turns += 1
        parsed = list(step_info.get("actions") or [])
        tm = (step_info.get("metrics") or {}).get("turn_metrics") or {}
        attempted += len(parsed)
        blocked = int(tm.get("collision_count", 0) or 0)
        collisions += blocked
        valid = bool(tm.get("action_is_valid", bool(parsed)))
        invalid += int(not valid)
        metric = env._calculate_canonical_metric()
        met = bool(metric and metric.get("success"))
        if met and not success:
            success, actions_to_success = True, int(env._current_step)
        end_reason = step_info.get("termination_reason", end_reason)
        if frames is not None:
            trajectory.append(dict(seen, answer=text, actions=[str(a) for a in parsed], valid=valid,
                                   collisions=blocked, reward=round(float(reward), 4), met=met,
                                   **gate_summary(metric)))
    row = {"index": index, "task_id": item.get("task_id"), "scene_id": item.get("scene_id"),
           "task_type": item.get("task_type"), "d_star": int((item.get("difficulty") or {}).get("d_star", 0)),
           "success": success, "actions_to_success": actions_to_success, "turns": turns,
           "actions": attempted, "collisions": collisions, "invalid_turns": invalid}
    if sample is not None:
        row["sample"] = int(sample)
    if frames is not None:
        last = _images(obs)
        row.update(task_description=item.get("task_description"), start=start, trajectory=trajectory,
                   final_frame=keep(last[0] if last else None, turns),
                   end_reason=end_reason if done else "max_turns")
    return row


def _image_url(image: Any) -> Dict[str, Any]:
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


class VllmPolicy:  # pragma: no cover - needs a GPU
    def __init__(self, model: str, gpu_memory_utilization: float = 0.6, max_model_len: int = 8192,
                 max_images: int = 1, temperature: float = 0.0, top_p: float = 1.0) -> None:
        from vllm import LLM

        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enforce_eager=True, enable_prefix_caching=True, limit_mm_per_prompt={"image": max_images},
                       trust_remote_code=False)
        self.last_prompt_tokens: Optional[int] = None
        self.temperature, self.top_p = float(temperature), float(top_p)

    def generate(self, system: str, user_text: str, image: Any, earlier: Sequence[Tuple[str, Any, str]] = (),
                 as_sft: bool = False, seed: Optional[int] = None) -> str:
        from vllm import SamplingParams

        params = SamplingParams(temperature=self.temperature, top_p=self.top_p, seed=seed, max_tokens=384,
                                stop=["</action>"], include_stop_str_in_output=True)
        out = self.llm.chat(chat_messages(system, earlier, user_text, image, _image_url, as_sft), sampling_params=params,
                            use_tqdm=False)
        self.last_prompt_tokens = len(out[0].prompt_token_ids)
        return out[0].outputs[0].text


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - needs a GPU and the render service
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    ev = Path(args.eval)
    out = ev / args.name
    out.mkdir(parents=True, exist_ok=True)
    tasks = ev / "tasks.jsonl"
    n = len(_jsonl(tasks))
    switches = switch_overrides(args)
    cfg = env_config(Path(args.env_template), tasks, args.gs_root, args.prompt_format, switches)
    keys = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in keys}))
    t0 = time.time()
    # with history a call carries every earlier frame; SFT trained with cutoff 16,384 tokens
    sampled = sampling_mode(args)
    policy = VllmPolicy(args.model, max_model_len=16384 if args.history else 8192,
                        max_images=int(args.max_turns) if args.history else 1,
                        temperature=float(args.temperature) if sampled else 0.0,
                        top_p=float(args.top_p) if sampled else 1.0)
    (out / "model.json").write_text(json.dumps({"name": args.name, "model": args.model, "version": VERSION,
                                                "prompt_format": cfg["prompt_format"], "greedy": not sampled,
                                                "samples": int(args.samples) if sampled else 1,
                                                "temperature": float(args.temperature) if sampled else 0.0,
                                                "top_p": float(args.top_p) if sampled else 1.0,
                                                "sample_seed": int(args.sample_seed) if sampled else None,
                                                "env_switches": switches, "max_turns": int(args.max_turns),
                                                "history": bool(args.history),
                                                "sft_text": bool(args.sft_text),
                                                "load_seconds": round(time.time() - t0, 1)}, indent=1))
    done = ({(r["index"], r.get("sample")) for r in _jsonl(out / "results.jsonl")}
            if (out / "results.jsonl").exists() else set())
    samples = list(range(int(args.samples))) if sampled else [None]
    with open(out / "results.jsonl", "a") as f:
        for i in range(n):
            for k in samples:
                if (i, k) in done:
                    continue
                try:
                    row = run_episode(env, policy, i, int(args.max_turns), frames=out / "frames",
                                      history=args.history, as_sft=args.sft_text, sample=k,
                                      seed=int(args.sample_seed) if sampled else None)
                except Exception as exc:  # an infrastructure error is recorded, not counted as a failure
                    row = {"index": i, "error": f"{type(exc).__name__}: {exc}"}
                    if k is not None:
                        row["sample"] = k
                f.write(json.dumps(row) + "\n")
                f.flush()
    (out / "done.json").write_text(json.dumps({"episodes": n * len(samples), "seconds": round(time.time() - t0, 1)}))


def sampling_mode(args: argparse.Namespace) -> bool:
    """A sampled run (round 11): several episodes per task, or any temperature above 0."""
    return int(getattr(args, "samples", 1) or 1) > 1 or float(getattr(args, "temperature", 0.0) or 0.0) > 0.0


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
def metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    succ = [r for r in ok if r["success"]]
    rate = lambda rs: round(sum(r["success"] for r in rs) / len(rs), 4) if rs else None  # noqa: E731
    by_family = defaultdict(list)
    by_band = defaultdict(list)
    for r in ok:
        by_family[r["task_type"]].append(r)
        by_band[band_name(r["d_star"])].append(r)
    actions = sum(r["actions"] for r in ok)
    turns = sum(r["turns"] for r in ok)
    ended = Counter(r["end_reason"] for r in ok if r.get("end_reason"))  # recorded since eval v2
    extra: Dict[str, Any] = {}
    if any("sample" in r for r in ok):  # a sampled run (round 11): per task, any success and the mean
        per_task: Dict[int, List[bool]] = defaultdict(list)
        for r in ok:
            per_task[r["index"]].append(bool(r["success"]))
        extra = {"samples": max(len(v) for v in per_task.values()), "tasks": len(per_task),
                 "pass_at_n": round(sum(any(v) for v in per_task.values()) / len(per_task), 4),
                 "success_mean": round(sum(sum(v) / len(v) for v in per_task.values()) / len(per_task), 4)}
    return {**extra,
        "episodes": len(ok), "errors": len(rows) - len(ok), "success": rate(ok),
        "success_by_family": {k: rate(v) for k, v in sorted(by_family.items())},
        "success_by_band": {k: rate(v) for k, v in sorted(by_band.items())},
        "actions_over_dstar": (round(sum(r["actions_to_success"] / r["d_star"] for r in succ) / len(succ), 3)
                               if succ else None),
        "collision_rate": round(sum(r["collisions"] for r in ok) / actions, 4) if actions else None,
        "invalid_rate": round(sum(r["invalid_turns"] for r in ok) / turns, 4) if turns else None,
        "ended_by": dict(sorted(ended.items())) if ended else None,
        # used up the action or turn limit (with or without meeting the task at the last action)
        "used_limit": sum(ended[k] for k in LIMIT_ENDS) if ended else None,
        "collision_ended": ended["consecutive_collision"] if ended else None,
    }


LIMIT_ENDS = ("max_primitive_steps", "max_primitive_steps_success", "max_turns")

# Round 12 batch (the user's decision of 2026-10-21): the success conditions of each family as the evaluation rows name
# them (failed_gates, v2 sets); "target_a_clearly_visible" and "reference_a_clearly_visible" are one condition (the
# prefix is the object's role in the task) and count once.
FAMILY_CONDITIONS: Dict[str, Tuple[str, ...]] = {
    "absolute_positioning": ("distance", "in_front", "inside_frame", "min_area", "visible", "a_clearly_visible"),
    "equidistance": ("equidistant", "in_front", "inside_frame", "min_area", "visible", "a_clearly_visible",
                     "b_clearly_visible"),
    "fov_inclusion": ("fully_in_frame", "in_front", "inside_frame", "min_area", "visible", "a_clearly_visible",
                      "b_clearly_visible"),
    "occlusion_alignment": ("occluder_in_front", "occluder_inside_frame", "occluder_min_area", "occluder_visible",
                            "target_hidden", "target_in_front", "target_in_view_direction", "b_clearly_visible"),
    "projective_relations": ("in_front", "inside_frame", "lateral_within_45deg", "margin", "min_area", "relation",
                             "visible", "a_clearly_visible", "b_clearly_visible"),
    "screen_occupancy": ("in_front", "inside_frame", "min_area", "occupancy", "vertical_contained", "visible",
                         "a_clearly_visible"),
}


def condition_name(gate: str) -> str:
    """One name per success condition: the role prefix of a clearly-visible condition dropped."""
    for role in ("target_", "reference_"):
        if gate.startswith(role) and gate.endswith("_clearly_visible") and len(gate) == len(role) + len("a_clearly_visible"):
            return gate[len(role):]
    return gate


def terminal_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Round 12 batch (the user's decision of 2026-10-21): how close the episodes end to success, from the recorded
    trajectories: the progress (0..1, the env's distance from meeting every success condition) of the last pose,
    averaged over the successful and over the failed episodes, and per family the share of its success conditions
    met at the end.  Rows without a recorded trajectory are left out; None when none has one."""
    ok = [r for r in rows if "error" not in r and "trajectory" in r]
    if not ok:
        return {"episodes": 0, "progress_end_success": None, "progress_end_fail": None, "conditions_met_share": None,
                "conditions_met_by_family": {}}

    def end(r: Dict[str, Any]) -> Tuple[float, set]:
        traj = r.get("trajectory") or []
        last = traj[-1] if traj else (r.get("start") or {})
        return float(last.get("progress") or 0.0), {condition_name(g) for g in (last.get("failed_gates") or [])}

    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    prog = {True: [], False: []}
    by_family: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"all": [], "failed": []})
    for r in ok:
        p, failed = end(r)
        success = bool(r.get("success"))
        prog[success].append(p)
        conds = FAMILY_CONDITIONS.get(r.get("task_type"))
        if conds:
            met = 1.0 if success else 1.0 - len(failed & set(conds)) / len(conds)
            by_family[r["task_type"]]["all"].append(met)
            if not success:
                by_family[r["task_type"]]["failed"].append(met)
    every = [m for v in by_family.values() for m in v["all"]]
    return {"episodes": len(ok), "progress_end_success": mean(prog[True]), "progress_end_fail": mean(prog[False]),
            "conditions_met_share": mean(every),
            "conditions_met_by_family": {k: {"conditions": len(FAMILY_CONDITIONS[k]), "met_share": mean(v["all"]),
                                             "met_share_failed": mean(v["failed"]), "episodes": len(v["all"])}
                                         for k, v in sorted(by_family.items())}}


def sign_test(b: int, c: int) -> float:
    """Exact two-sided sign (McNemar) test on the tasks only one of two models solved: how likely a split at least
    this uneven is if both models were equally good."""
    import math

    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(base: Sequence[Dict[str, Any]], other: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Task-by-task comparison of two result sets on the same tasks (infrastructure errors left out).  Greedy runs
    only: a sampled run has several rows per task."""
    if any("sample" in r for r in list(base) + list(other)):
        raise ValueError("paired compares greedy runs (one episode per task); a sampled run was given")
    a = {r["index"]: bool(r["success"]) for r in base if "error" not in r}
    b = {r["index"]: bool(r["success"]) for r in other if "error" not in r}
    idx = sorted(set(a) & set(b))
    only_base = sum(a[i] and not b[i] for i in idx)
    only_other = sum(b[i] and not a[i] for i in idx)
    both = sum(a[i] and b[i] for i in idx)
    return {"tasks": len(idx), "both": both, "only_base": only_base, "only_other": only_other,
            "neither": len(idx) - both - only_base - only_other, "p": round(sign_test(only_base, only_other), 4)}


def cmd_compare(args: argparse.Namespace) -> None:
    base = _jsonl(Path(args.base) / "results.jsonl")
    out = {Path(d).name: paired(base, _jsonl(Path(d) / "results.jsonl")) for d in args.other}
    lines = [f"基准：{args.base}", "", "| 对象 | 题数 | 都做对 | 只有基准做对 | 只有对象做对 | 都没做对 | p |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {k} | {v['tasks']} | {v['both']} | {v['only_base']} | {v['only_other']} | {v['neither']} | {v['p']:.3f} |"
              for k, v in out.items()]
    Path(args.out).write_text(json.dumps({"base": str(args.base), "compare": out}, indent=1))
    Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_table(args: argparse.Namespace) -> None:
    ev = Path(args.eval)
    models = {}
    for d in sorted(p for p in ev.iterdir() if (p / "results.jsonl").exists()):
        rows = _jsonl(d / "results.jsonl", partial_ok=True)  # other chains' jobs may still be writing theirs
        if rows:
            models[d.name] = metrics(rows)
    (ev / "table.json").write_text(json.dumps(models, indent=1))
    families = sorted({f for m in models.values() for f in m["success_by_family"]})
    bands = [f"{lo}-{hi}" for lo, hi in BANDS]
    head = ["模型", "题数", "成功率"] + families + [f"d* {b}" for b in bands] + ["步数/d*", "碰撞率", "不合法率",
                                                                                  "用满上限", "连续碰撞结束",
                                                                                  "至少一条成功（采样）"]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    for name, m in models.items():
        cells = [name, str(m["episodes"]), fmt(m["success"])] + [fmt(m["success_by_family"].get(f)) for f in families]
        cells += [fmt(m["success_by_band"].get(b)) for b in bands]
        cells += [fmt(m["actions_over_dstar"]), fmt(m["collision_rate"]), fmt(m["invalid_rate"])]
        cells += ["—" if m["used_limit"] is None else str(m["used_limit"]),
                  "—" if m["collision_ended"] is None else str(m["collision_ended"]),
                  f"{m['pass_at_n']:.3f}（{m['samples']} 条）" if "pass_at_n" in m else "—"]
        if "pass_at_n" in m:  # the success column of a sampled run is the mean over its episodes
            cells[1] = f"{m['tasks']}×{m['samples']}"
        lines.append("| " + " | ".join(cells) + " |")
    (ev / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_terminal(args: argparse.Namespace) -> None:
    """Round 12 batch: terminal_metrics and the collision rate of every result folder of an evaluation set, into
    terminal.json and terminal.md (table.json / table.md stay as they are)."""
    ev = Path(args.eval)
    models = {}
    for d in sorted(p for p in ev.iterdir() if (p / "results.jsonl").exists()):
        rows = _jsonl(d / "results.jsonl", partial_ok=True)
        if rows:
            models[d.name] = {**terminal_metrics(rows), "success": metrics(rows)["success"],
                              "collision_rate": metrics(rows)["collision_rate"]}
    (ev / "terminal.json").write_text(json.dumps(models, indent=1))
    families = list(FAMILY_CONDITIONS)
    head = ["模型", "题数", "成功率", "终点进度分（成功的题）", "终点进度分（失败的题）", "终点满足的条件比例", "碰撞率"] + \
        [f"{f} 满足比例（失败的题）" for f in families]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for name, m in models.items():
        fam = m["conditions_met_by_family"]
        cells = [name, str(m["episodes"]), fmt(m["success"]), fmt(m["progress_end_success"]),
                 fmt(m["progress_end_fail"]), fmt(m["conditions_met_share"]), fmt(m["collision_rate"])]
        cells += [fmt((fam.get(f) or {}).get("met_share_failed")) for f in families]
        lines.append("| " + " | ".join(cells) + " |")
    (ev / "terminal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--run", required=True)
    s.add_argument("--split-manifest", required=True)
    s.add_argument("--per-family", type=int, default=PER_FAMILY)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--split", default="test", help="which split to draw from (round 10: val for the fixed validation set)")
    s.add_argument("--exclude", default=None, help="depth_check_summary.json: its flagged tasks are never drawn")
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--eval", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--model", required=True, help="a Hugging Face model folder (an RL checkpoint's actor/huggingface)")
    r.add_argument("--gs-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_ply")
    r.add_argument("--env-template", default=str(Path(__file__).with_name("configs") / "rl_env_v10_S0.yaml"))
    r.add_argument("--prompt-format", default=None, help="the model's own prompt format (R1: its own)")
    r.add_argument("--max-turns", type=int, default=12)
    # round 9 switches (default: the template's rules)
    r.add_argument("--check-each-action", action="store_true", help="check success after every action, stop at the first")
    r.add_argument("--max-actions", type=int, default=None, help="action limit per task (template: 12); --max-turns sets the turn limit")
    r.add_argument("--no-collision-end", action="store_true", help="3 blocked moves in a row no longer end the task")
    r.add_argument("--collision-feedback", action="store_true", help="feedback says which moves of the turn were blocked")
    r.add_argument("--history", action=argparse.BooleanOptionalAction, default=True,
                   help="every model call gets all earlier turns, exactly as in SFT (round-10 default; --no-history "
                        "for a single-turn input)")
    r.add_argument("--samples", type=int, default=1,
                   help="round 11: episodes per task; with more than 1 (or a temperature above 0) the answers are "
                        "sampled as in RL training instead of greedy")
    r.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (RL training: 0.8)")
    r.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling (RL training: 0.92)")
    r.add_argument("--sample-seed", type=int, default=20261016, help="base seed of the sampled model calls")
    r.add_argument("--sft-text", action="store_true",
                   help="messages as the SFT data of rounds 6-9 had them (leading/trailing whitespace stripped); the "
                        "round-10 export keeps the env's text, so this is only for models of those rounds")
    t = sub.add_parser("table")
    t.add_argument("--eval", required=True)
    m = sub.add_parser("terminal", help="round 12 batch: end progress, conditions met at the end, collision rate")
    m.add_argument("--eval", required=True)
    c = sub.add_parser("compare", help="task-by-task comparison against a base result folder")
    c.add_argument("--base", required=True, help="a result folder (with results.jsonl)")
    c.add_argument("--other", required=True, nargs="+")
    c.add_argument("--out", required=True, help="JSON file; a .md table is written next to it")
    x = sub.add_parser("replace")
    x.add_argument("--eval", required=True, help="the evaluation folder to start from (tasks.jsonl)")
    x.add_argument("--flagged", required=True, help="depth_check_summary.json (flagged_task_ids)")
    x.add_argument("--run", required=True)
    x.add_argument("--split-manifest", required=True)
    x.add_argument("--split", default="test")
    x.add_argument("--seed", type=int, default=SEED)
    x.add_argument("--out", required=True)
    u = sub.add_parser("top-up", help="round 12: add one family's reserve tasks of the split, replayed, up to a target")
    u.add_argument("--eval", required=True, help="the evaluation folder to start from (tasks.jsonl; it stays as it is)")
    u.add_argument("--family", required=True)
    u.add_argument("--target", type=int, required=True, help="tasks of the family wanted in the new set")
    u.add_argument("--run", required=True, help="run_all output directory (tasks_reserve.jsonl)")
    u.add_argument("--split-manifest", required=True)
    u.add_argument("--split", required=True)
    u.add_argument("--exclude", default=None, help="depth_check_summary.json: its flagged tasks are never taken")
    u.add_argument("--scenes-root", required=True, help="InteriorGS scene folders for the replay")
    u.add_argument("--env-template", default=str(Path(__file__).with_name("configs") / "rl_env_v10_S0.yaml"),
                   help="the env rules of the replay")
    u.add_argument("--seed", type=int, default=SEED)
    u.add_argument("--out", required=True)
    u.add_argument("--family-out", default=None, help="also write the family's tasks of the new set alone here")
    args = ap.parse_args(argv)
    {"select": cmd_select, "run": cmd_run, "table": cmd_table, "replace": cmd_replace,
     "compare": cmd_compare, "top-up": cmd_top_up, "terminal": cmd_terminal}[args.cmd](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
