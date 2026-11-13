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

Experiment 3A (2026-11-06): ``run --turn-footprint-radius R`` evaluates under the turn rule (model.json env_switches).
One rule per result folder (a run resuming a folder with results must give its rule) and per set (``table`` refuses
folders of different rules); under the rule each recorded turn also carries ``footprint_blocks``, its turns blocked by
the robot-base footprint.

Grounding probe (the user's supplement of 2026-11-12, item 5): ``run --grounding-probe`` appends the probe instruction
to every user turn (state the bearing and distance of the task's first object before the action), the env reads the
answer without its ``<state>`` line, and each row carries ``grounding_turns`` (every turn's statement scored against the
truth at the pose its frame was rendered from) and ``grounding`` (the episode's shares); grounding_probe.py has the
definitions and the ``summary``.  One probe version per result folder (model.json ``grounding_probe``); off,
nothing changes.  ``run --grounding-side`` (v2, the user's choice of 2026-11-12: the trained models wrote no readable
statement under v1): the action call is the plain run's, unchanged; after it, on the same frame, a separate greedy side
call with the assistant reply prefilled (``<state>bearing``) asks for the statement (grounding_probe.py, v2); the
side calls never enter the history, and a failed side call costs only that turn's statement (``side_error``).  Qwen models only (the side call needs an assistant prefix in the chat prompt).

Critic probe (gate 0 of docs/PREREG_PPO2.md, the representation probe; a recording mode, not an evaluation rule):
``run --critic-probe TABLES_DIR`` runs the env with critic_features on (the tables of TABLES_DIR) and V4's reward (the
D_cond preset with potential_field_reward_scale 3.3333333333333335), all recorded in model.json env_switches, and
records what repr_probe.py needs to rebuild every model call: the frames as lossless PNG (<index>[_s<sample>]_<turn>.png),
the row's system prompt (``system``), per turn the env's truth at the pose of the observation the turn answered
(``critic_features``, reset's for turn 0 and step t-1's after; ``critic_features_error`` when the env could not compute
it), the unrounded reward (``reward_exact``), the sha256 of the prompt's token ids vLLM ran (``prompt_ids_sha256``) and
how the answer ended (``finish_reason``, ``stop_reason``), and the truth after the last turn (``final_critic_features``).
Needs --history; refused with --grounding-probe / --grounding-side / --sft-text and with an empty TABLES_DIR; every
task must have its table (model.json ``critic_probe_tables``), every switch must be an env-config field and reach the
built env; the run stops without done.json after CRITIC_PROBE_MAX_ERRORS_IN_A_ROW error rows in a row; a folder
resumes only with the same model.json (but name and load_seconds).  One probe version per result folder (model.json
``critic_probe``); off, nothing changes.  Under the switch the rows' progress fields (``progress``, ``failed_gates``)
are the grouped_gated progress of V4's env, not comparable with plain folders.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PER_FAMILY = 50
SEED = 20261014
BANDS = ((1, 3), (4, 6), (7, 8))
VERSION = "interiorgs_v10_eval_v2"  # v2 (round 8): results rows carry the trajectory and frames
CRITIC_PROBE_VERSION = "critic_probe_v1"  # gate 0 of docs/PREREG_PPO2.md (run --critic-probe)
V4_PROGRESS_SCALE = 3.3333333333333335  # V4: the D_cond preset with the progress (potential) scale / 3 (matrix_r12)
CRITIC_PROBE_MAX_ERRORS_IN_A_ROW = 10  # critic probe: a run stops (no done.json) after this many error rows in a row


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
    if getattr(args, "format_example", None):  # round 12 freeze: the format text the model trained with
        out["prompt_format_example"] = args.format_example
    if getattr(args, "lenient_parse", False):  # round 12 freeze: the lenient reading of the answers
        out["lenient_parse"] = True
    if getattr(args, "turn_footprint_radius", None) is not None:  # experiment 3A: the robot-base turn rule
        out["turn_footprint_radius"] = float(args.turn_footprint_radius)
    if getattr(args, "critic_probe", None) is not None:  # gate 0 (PREREG_PPO2): the env's truth per turn, V4's reward
        if not str(args.critic_probe).strip():
            raise SystemExit("--critic-probe needs the distance tables' folder (an empty value is not 'off')")
        out.update(critic_probe_overrides(args.critic_probe))
    return out


def critic_probe_overrides(tables_dir: str) -> Dict[str, Any]:
    """Gate 0 of docs/PREREG_PPO2.md (run --critic-probe): the env settings of the probe's episodes, V4's reward (the
    D_cond preset, training_prep.reward_overrides, with the progress scale 3.3333333333333335: each recorded turn reward
    is V4's) and the privileged critic's truth (env critic_features from the distance tables in ``tables_dir``)."""
    import os

    from .training_prep import reward_overrides

    return {**reward_overrides("D_cond"), "potential_field_reward_scale": V4_PROGRESS_SCALE,
            "critic_features": True, "critic_feature_tables_dir": os.path.abspath(str(tables_dir))}


def folder_rule(result: Path) -> Optional[float]:
    """Experiment 3A: the turn rule a result folder was evaluated under, its model.json env_switches
    turn_footprint_radius (missing key or no model.json: None, today's rule)."""
    model = Path(result) / "model.json"
    if not model.is_file():
        return None
    return (json.loads(model.read_text()).get("env_switches") or {}).get("turn_footprint_radius")


def check_resume_rule(result: Path, switches: Dict[str, Any]) -> None:
    """Experiment 3A: one turn rule per result folder.  A run resuming a folder whose results.jsonl already has rows
    must give the turn_footprint_radius of the folder's model.json (missing = None), else it stops before anything is
    written; only this switch is compared, so every older folder resumes as before."""
    results = Path(result) / "results.jsonl"
    if not results.is_file() or not _jsonl(results, partial_ok=True):
        return
    before, now = folder_rule(result), switches.get("turn_footprint_radius")
    if before != now:
        raise SystemExit(f"{result}: its results were evaluated with turn_footprint_radius {before!r}, this run gives "
                         f"{now!r}; one turn rule per result folder (give another --name)")


def check_resume_probe(result: Path, version: Optional[str], key: str = "grounding_probe") -> None:
    """Grounding probe (2026-11-12, item 5): one probe version per result folder.  A run resuming a folder whose
    results.jsonl already has rows must give the probe version of the folder's model.json (``version`` None and a
    missing key: not a probe run), else it stops before anything is written.  ``key``: the probe's model.json key
    (critic_probe for gate 0's recording mode)."""
    results = Path(result) / "results.jsonl"
    if not results.is_file() or not _jsonl(results, partial_ok=True):
        return
    model = Path(result) / "model.json"
    before = json.loads(model.read_text()).get(key) if model.is_file() else None
    if before != version:
        raise SystemExit(f"{result}: its results were evaluated with {key} {before!r}, this run gives "
                         f"{version!r}; one probe version per result folder (give another --name)")


def check_resume_tables(result: Path, switches: Dict[str, Any]) -> None:
    """Critic probe: a run resuming a folder with results gives the distance tables its model.json env_switches name
    (critic_feature_tables_dir), else it stops before anything is written."""
    results, model = Path(result) / "results.jsonl", Path(result) / "model.json"
    if not results.is_file() or not _jsonl(results, partial_ok=True) or not model.is_file():
        return
    before = (json.loads(model.read_text()).get("env_switches") or {}).get("critic_feature_tables_dir")
    now = switches.get("critic_feature_tables_dir")
    if before != now:
        raise SystemExit(f"{result}: its results were recorded with critic_feature_tables_dir {before!r}, this run "
                         f"gives {now!r}; one tables folder per result folder (give another --name)")


def ids_digest(ids: Sequence[int]) -> str:
    """Critic probe: the sha256 of a prompt's token ids (decimal, comma-separated), so repr_probe can check a rebuilt
    prompt against the one vLLM ran."""
    import hashlib

    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def critic_probe_fields(info: Any) -> Dict[str, Any]:
    """Critic probe: the env's truth at the pose of an observation (its reset / step info), as a turn records it."""
    if not isinstance(info, dict) or "critic_features" not in info:
        raise RuntimeError("critic probe: the env's info has no critic_features (env critic_features off?)")
    feats = info["critic_features"]
    out: Dict[str, Any] = {"critic_features": dict(feats) if feats is not None else None}
    if info.get("critic_features_error") is not None:
        out["critic_features_error"] = str(info["critic_features_error"])
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
                seed: Optional[int] = None, grounding: bool = False, grounding_side: bool = False,
                critic_probe: bool = False) -> Dict[str, Any]:
    """One greedy episode; per-turn bookkeeping as in scripts/r1_run_canonical_dev_eval32.py.

    With ``frames`` (round 8) the row also carries the trajectory: per turn the frame the model saw (a JPEG in
    ``frames``, named <index>_<turn>.jpg), the user text, the answer, the parsed actions, blocked moves, reward
    and whether the task was met after the turn; then the frame after the last turn and the env's end reason.
    With ``history`` (round 9 switch) every model call also gets all earlier turns of the episode, as in SFT;
    with ``as_sft`` every message is passed through ``sft_text`` first.  ``sample`` / ``seed`` (round 11): the
    sample number of a sampled run and the base seed of its model calls (``call_seed``); frames are then named
    <index>_s<sample>_<turn>.jpg.  ``grounding`` (2026-11-12, item 5): the grounding probe (grounding_probe.py); the
    user text the model saw (the probe instruction appended) is what the history and the trajectory keep.
    ``grounding_side`` (v2): the action call is unchanged; after it, before the env moves the camera, the policy's
    side_generate asks for the statement on the same frame with the same earlier turns (never added to them).
    ``critic_probe`` (gate 0 of PREREG_PPO2): frames as lossless PNG (.png), each turn also records the env's truth at
    the pose of the observation it answered (critic_probe_fields of reset's info for turn 0, of step t-1's after), the
    unrounded reward and the digest of the prompt's token ids (policy.last_prompt_ids); the row its system prompt and
    the truth after the last turn."""
    if critic_probe and (grounding or grounding_side):
        raise ValueError("the critic probe runs without the grounding probe")
    if critic_probe and frames is None:
        raise ValueError("the critic probe records the trajectory: give frames")
    obs, info = env.reset(seed=index)
    obs_info = info  # critic probe: the info of the observation the next answer responds to (reset's, then each step's)
    system = env.system_prompt()
    item = env.current_item or {}
    turns, invalid, attempted, collisions, success, actions_to_success = 0, 0, 0, 0, False, None
    done, trajectory, end_reason, earlier = False, [], None, []
    probe_turns: List[Dict[str, Any]] = []
    if grounding and grounding_side:
        raise ValueError("one grounding probe per run: grounding (v1) or grounding_side (v2)")
    if grounding or grounding_side:
        from data_gen.interiorgs_v10 import grounding_probe
    start = gate_summary(env._calculate_canonical_metric()) if frames is not None else None

    def keep(image: Any, turn: int) -> Optional[str]:
        if frames is None or image is None:
            return None
        frames.mkdir(parents=True, exist_ok=True)
        ext = "png" if critic_probe else "jpg"
        name = f"{index:03d}_{turn:02d}.{ext}" if sample is None else f"{index:03d}_s{sample}_{turn:02d}.{ext}"
        if critic_probe:  # lossless: the pixels the model got (its frame went to vLLM as a PNG, _image_url)
            image.convert("RGB").save(frames / name, format="PNG")
        else:
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
        user_text = obs["obs_str"]
        if grounding:  # the truth at the pose this frame was rendered from, before the action moves the camera
            truth = grounding_probe.truth(item, env.view_engine.get_pose())
            user_text += grounding_probe.instruction(item)
        if grounding_side:  # v2: the truth at this frame's pose (the side call comes after the action call)
            truth = grounding_probe.truth(item, env.view_engine.get_pose())
        probe = critic_probe_fields(obs_info) if critic_probe else {}  # the truth at this frame's pose
        text = policy.generate(system, user_text, images[0], **kw)
        if history:
            earlier.append((user_text, images[0], text))
        seen = {"frame": keep(images[0], turns), "user": user_text}
        if getattr(policy, "last_prompt_tokens", None) is not None:
            seen["prompt_tokens"] = policy.last_prompt_tokens
        if critic_probe and getattr(policy, "last_prompt_ids", None) is not None:
            seen["prompt_ids_sha256"] = ids_digest(policy.last_prompt_ids)
        if critic_probe and getattr(policy, "last_finish", None) is not None:  # how the answer ended (repr_probe:
            seen.update(finish_reason=str(policy.last_finish),  # an answer cut by the token limit has no end token)
                        stop_reason=getattr(policy, "last_stop_reason", None))
        if getattr(policy, "last_entropy", None) is not None:  # round 12 freeze: run --entropy
            seen.update(entropy=round(float(policy.last_entropy), 5), tokens=int(policy.last_tokens or 0))
        if grounding_side:  # v2: the same frame, the action call's earlier turns (kw), the plain user text + question
            try:
                side = policy.side_generate(system, user_text + grounding_probe.side_question(item), images[0],
                                            earlier=kw.get("earlier", []), as_sft=bool(kw.get("as_sft", False)),
                                            prefix=grounding_probe.SIDE_PREFIX, stop=grounding_probe.SIDE_STOP,
                                            max_tokens=grounding_probe.SIDE_MAX_TOKENS)
            except Exception as exc:  # the turn's statement is lost (scored wrong); the action stream goes on
                probe_turns.append(grounding_probe.side_error_record(f"{type(exc).__name__}: {exc}", truth, turns))
            else:
                probe_turns.append(grounding_probe.side_record(side, truth, turns))
        obs, reward, done, step_info = env.step(grounding_probe.answer_part(text) if grounding else text)
        obs_info = step_info
        if grounding:
            probe_turns.append(grounding_probe.score(text, truth, turns))
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
            # experiment 3A: under the turn rule (the env reports turn_footprint_blocks only then) the turn's turns
            # blocked by the robot-base footprint; a row under today's rule gets nothing new
            footprint = ({"footprint_blocks": int(tm["turn_footprint_blocks"])} if "turn_footprint_blocks" in tm else {})
            if critic_probe:  # gate 0: the unrounded reward (the probe's returns-to-go are summed from it)
                probe = {"reward_exact": float(reward), **probe}
            trajectory.append(dict(seen, answer=text, actions=[str(a) for a in parsed], valid=valid,
                                   collisions=blocked, **footprint, reward=round(float(reward), 4), met=met,
                                   **gate_summary(metric), **probe))
    row = {"index": index, "task_id": item.get("task_id"), "scene_id": item.get("scene_id"),
           "task_type": item.get("task_type"), "d_star": int((item.get("difficulty") or {}).get("d_star", 0)),
           "success": success, "actions_to_success": actions_to_success, "turns": turns,
           "actions": attempted, "collisions": collisions, "invalid_turns": invalid}
    if sample is not None:
        row["sample"] = int(sample)
    if grounding or grounding_side:
        version = grounding_probe.VERSION_SIDE if grounding_side else grounding_probe.VERSION
        row.update(grounding_turns=probe_turns, grounding=grounding_probe.episode_summary(probe_turns, version))
    if frames is not None:
        last = _images(obs)
        row.update(task_description=item.get("task_description"), start=start, trajectory=trajectory,
                   final_frame=keep(last[0] if last else None, turns),
                   end_reason=end_reason if done else "max_turns")
    if critic_probe:  # gate 0: what repr_probe needs to rebuild the chat, and the truth after the last turn
        final = critic_probe_fields(obs_info)
        row.update(system=system, final_critic_features=final["critic_features"])
        if "critic_features_error" in final:
            row["final_critic_features_error"] = final["critic_features_error"]
    return row


def _image_url(image: Any) -> Dict[str, Any]:
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


# Round 12 freeze (the user's 3.1.4 of 2026-10-22, the format SFT's acceptance): the token entropy of the answers, as
# the RL logs report it (the entropy of the temperature-scaled next-token distribution, averaged over the answer's
# tokens), estimated from the ENTROPY_TOP_K most likely tokens vLLM returns (raw log-probabilities, renormalised
# after the temperature; the tail beyond them is left out, a slight underestimate).
ENTROPY_TOP_K = 20


def top_logprob_entropy(positions: Sequence[Sequence[float]], temperature: float) -> Optional[float]:
    """The mean entropy (nats) over the answer's tokens of the temperature-scaled distribution renormalised over each
    position's given raw log-probabilities (temperature 0 or less: the model's own, 1)."""
    t = float(temperature) if temperature and temperature > 0 else 1.0
    out = []
    for lps in positions:
        if not lps:
            continue
        z = [float(lp) / t for lp in lps]
        m = max(z)
        w = [math.exp(v - m) for v in z]
        s = sum(w)
        out.append(-sum((x / s) * math.log(x / s) for x in w if x > 0))
    return sum(out) / len(out) if out else None


def cambrian_request(wrapper: Any, system: str, earlier: Sequence[Tuple[str, Any, str]], user_text: str, image: Any,
                     as_sft: bool = False) -> Dict[str, Any]:
    """The model line (2026-10-24): one Cambrian-S call as vLLM takes it -- the same chat as chat_messages, through
    the model's chat template with one compact <image> token per frame (CambrianProcessorWrapper, as the RL agent
    loop builds it; vLLM's Cambrian processor expands each to 756 positions) and the frames in their order."""
    messages = chat_messages(system, earlier, user_text, image, lambda im: {"type": "image"}, as_sft)
    text = wrapper.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = wrapper(text=[text], return_tensors=None)["input_ids"][0]
    images = [im for _, im, _ in earlier] + [image]
    if sum(1 for t in ids if t == wrapper.image_token_id) != len(images):
        raise ValueError("the <image> tokens and the frames differ")
    return {"prompt_token_ids": ids, "multi_modal_data": {"image": images}}


class VllmPolicy:  # pragma: no cover - needs a GPU
    def __init__(self, model: str, gpu_memory_utilization: float = 0.6, max_model_len: int = 8192,
                 max_images: int = 1, temperature: float = 0.0, top_p: float = 1.0, entropy: bool = False) -> None:
        from vllm import LLM

        from vagen.models.model_types import of_checkpoint

        self.cambrian = None
        if of_checkpoint(model) == "cambrian_qwen":  # the model line: compact <image> tokens (cambrian_request)
            from transformers import AutoTokenizer

            from vagen.models.cambrian_plugin import register
            from vagen.models.cambrian_processor import CambrianProcessorWrapper

            register()  # the vLLM model and its config in this process (the plugin entry point does the workers)
            self.cambrian = CambrianProcessorWrapper(AutoTokenizer.from_pretrained(model))
        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enforce_eager=True, enable_prefix_caching=True, limit_mm_per_prompt={"image": max_images},
                       trust_remote_code=self.cambrian is not None)
        self.last_prompt_tokens: Optional[int] = None
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.entropy = bool(entropy)  # round 12 freeze: the answers' token entropy (last_entropy, last_tokens)
        self.last_entropy: Optional[float] = None
        self.last_tokens: Optional[int] = None
        self.last_prompt_ids: Optional[List[int]] = None  # grounding probe v2: the layout check of the side call
        self.last_finish: Optional[str] = None  # critic probe: how the last answer ended (stop / length) and on what
        self.last_stop_reason: Any = None

    def generate(self, system: str, user_text: str, image: Any, earlier: Sequence[Tuple[str, Any, str]] = (),
                 as_sft: bool = False, seed: Optional[int] = None) -> str:
        from vllm import SamplingParams

        params = SamplingParams(temperature=self.temperature, top_p=self.top_p, seed=seed, max_tokens=384,
                                stop=["</action>"], include_stop_str_in_output=True,
                                **({"logprobs": ENTROPY_TOP_K} if self.entropy else {}))
        if self.cambrian is not None:
            out = self.llm.generate([cambrian_request(self.cambrian, system, earlier, user_text, image, as_sft)],
                                    sampling_params=params, use_tqdm=False)
        else:
            out = self.llm.chat(chat_messages(system, earlier, user_text, image, _image_url, as_sft),
                                sampling_params=params, use_tqdm=False)
        self.last_prompt_tokens = len(out[0].prompt_token_ids)
        self.last_prompt_ids = list(out[0].prompt_token_ids)
        self.last_finish = out[0].outputs[0].finish_reason
        self.last_stop_reason = getattr(out[0].outputs[0], "stop_reason", None)
        if self.entropy:
            steps = out[0].outputs[0].logprobs or []
            self.last_entropy = top_logprob_entropy([[lp.logprob for lp in pos.values()] for pos in steps],
                                                    self.temperature)
            self.last_tokens = len(steps)
        return out[0].outputs[0].text

    def side_generate(self, system: str, user_text: str, image: Any, earlier: Sequence[Tuple[str, Any, str]] = (),
                      as_sft: bool = False, prefix: str = "", stop: str = "</state>",
                      max_tokens: int = 24) -> Dict[str, Any]:
        """Grounding probe v2: one greedy call whose assistant reply starts with ``prefix``.  The chat is built as
        generate builds it (chat_messages, then LLM.preprocess_chat: the code LLM.chat runs, the chat template with
        its generation prompt) and the prefix's tokens are appended after the generation prompt
        (grounding_probe.prefixed_prompt), so no end-of-turn token follows the prefix.  Greedy without a seed: the
        action calls' sampling seeds are per request and their own.  Leaves last_prompt_tokens / last_entropy (the
        action call's) as they are.  Returns the completion, how it ended, the prompt's tokens and the layout check
        against the last action call's prompt."""
        if self.cambrian is not None:
            raise RuntimeError("the grounding side call needs an assistant prefix in the chat prompt; the Cambrian "
                               "path (cambrian_request) has none")
        from vllm import SamplingParams

        from data_gen.interiorgs_v10 import grounding_probe

        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_tokens), stop=[stop],
                                include_stop_str_in_output=True)
        tokenizer = self.llm.get_tokenizer()
        prompt = self.llm.preprocess_chat(chat_messages(system, earlier, user_text, image, _image_url, as_sft))[0]
        prompt, prefix_ids = grounding_probe.prefixed_prompt(prompt, tokenizer, prefix)
        out = self.llm.generate([prompt], sampling_params=params, use_tqdm=False)
        side_ids = list(out[0].prompt_token_ids)
        result = {"text": out[0].outputs[0].text, "finish": out[0].outputs[0].finish_reason,
                  # the stop string or token that ended it (None: the token limit or the end-of-sequence token)
                  "stop_reason": getattr(out[0].outputs[0], "stop_reason", None),
                  "prompt_tokens": len(side_ids)}
        if self.last_prompt_ids is not None:
            end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")  # the Qwen chat templates' end of a turn
            result.update(grounding_probe.layout_check(self.last_prompt_ids, side_ids, prefix_ids, end_id))
        return result


def check_side_support(model: str) -> None:
    """Grounding probe v2: the side call needs an assistant prefix in the chat prompt, which only the chat path (the
    Qwen models) has; a Cambrian model is refused before anything starts."""
    from vagen.models.model_types import of_checkpoint

    if of_checkpoint(model) == "cambrian_qwen":
        raise SystemExit(f"{model}: --grounding-side is not supported for cambrian_qwen (its prompt is built by "
                         "cambrian_request, without an assistant prefix)")


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - needs a GPU and the render service
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    ev = Path(args.eval)
    out = ev / args.name
    out.mkdir(parents=True, exist_ok=True)
    tasks = ev / "tasks.jsonl"
    n = len(_jsonl(tasks))
    switches = switch_overrides(args)
    check_resume_rule(out, switches)  # experiment 3A: before the env, the model and model.json
    probe = bool(getattr(args, "grounding_probe", False))
    side = bool(getattr(args, "grounding_side", False))  # grounding probe v2
    critic = getattr(args, "critic_probe", None) is not None  # gate 0 of PREREG_PPO2: the recording mode
    if probe and side:
        raise SystemExit("--grounding-probe and --grounding-side: one grounding probe per run")
    if critic and (probe or side):
        raise SystemExit("--critic-probe records the plain run's calls: not with --grounding-probe / --grounding-side")
    if critic and not args.history:
        raise SystemExit("--critic-probe needs --history (repr_probe teacher-forces each episode as one sequence)")
    if critic and args.sft_text:
        raise SystemExit("--critic-probe: not with --sft-text (RL training shows the env's text as it is)")
    if critic and not Path(args.critic_probe).is_dir():
        raise SystemExit(f"--critic-probe {args.critic_probe}: not a directory (the tasks' distance tables)")
    probe_key: Dict[str, Any] = {}
    if probe or side:
        from data_gen.interiorgs_v10 import grounding_probe

        probe_key = {"grounding_probe": grounding_probe.VERSION_SIDE if side else grounding_probe.VERSION}
    if critic:  # every task has its table, before anything starts (counted in model.json)
        probe_key = {"critic_probe": CRITIC_PROBE_VERSION,
                     "critic_probe_tables": check_task_tables(tasks, args.critic_probe)}
    if side:
        check_side_support(args.model)
    check_resume_probe(out, probe_key.get("grounding_probe"))  # 2026-11-12, item 5
    check_resume_probe(out, probe_key.get("critic_probe"), key="critic_probe")  # gate 0: one probe version
    if critic:
        check_resume_tables(out, switches)
    cfg = env_config(Path(args.env_template), tasks, args.gs_root, args.prompt_format, switches)
    keys = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    sampled = sampling_mode(args)
    if critic:  # no switch silently dropped; a resumed folder has exactly this run's model.json
        dropped = sorted(k for k in switches if k not in keys)
        if dropped:
            raise SystemExit(f"--critic-probe: env switches {dropped} are not ActiveSpatialEnvConfig fields (the env "
                             "would run without them)")
        check_resume_record(out, model_record(args, cfg, switches, sampled, probe_key, None))
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in keys}))
    if critic:
        check_env_switches(env, switches)
    t0 = time.time()
    # with history a call carries every earlier frame; SFT trained with cutoff 16,384 tokens
    policy = VllmPolicy(args.model, max_model_len=16384 if args.history else 8192,
                        max_images=int(args.max_turns) if args.history else 1,
                        temperature=float(args.temperature) if sampled else 0.0,
                        top_p=float(args.top_p) if sampled else 1.0, entropy=bool(getattr(args, "entropy", False)))
    if side and hasattr(policy, "llm") and not hasattr(policy.llm, "preprocess_chat"):
        raise SystemExit("--grounding-side needs vLLM's LLM.preprocess_chat (vLLM 0.11)")
    (out / "model.json").write_text(json.dumps(model_record(args, cfg, switches, sampled, probe_key,
                                                            round(time.time() - t0, 1)), indent=1))
    done = ({(r["index"], r.get("sample")) for r in _jsonl(out / "results.jsonl")}
            if (out / "results.jsonl").exists() else set())
    samples = list(range(int(args.samples))) if sampled else [None]
    errors_in_a_row = 0
    with open(out / "results.jsonl", "a") as f:
        for i in range(n):
            for k in samples:
                if (i, k) in done:
                    continue
                try:
                    row = run_episode(env, policy, i, int(args.max_turns), frames=out / "frames",
                                      history=args.history, as_sft=args.sft_text, sample=k,
                                      seed=int(args.sample_seed) if sampled else None, grounding=probe,
                                      grounding_side=side, **({"critic_probe": True} if critic else {}))
                except Exception as exc:  # an infrastructure error is recorded, not counted as a failure
                    row = {"index": i, "error": f"{type(exc).__name__}: {exc}"}
                    if k is not None:
                        row["sample"] = k
                f.write(json.dumps(row) + "\n")
                f.flush()
                errors_in_a_row = errors_in_a_row + 1 if "error" in row else 0
                if critic and errors_in_a_row >= CRITIC_PROBE_MAX_ERRORS_IN_A_ROW:  # a systematic fault: no done.json
                    raise SystemExit(f"--critic-probe: {errors_in_a_row} episodes in a row ended in an error (last: "
                                     f"{row['error']}); stopped without done.json")
    (out / "done.json").write_text(json.dumps({"episodes": n * len(samples), "seconds": round(time.time() - t0, 1)}))


def model_record(args: argparse.Namespace, cfg: Dict[str, Any], switches: Dict[str, Any], sampled: bool,
                 probe_key: Dict[str, Any], load_seconds: Optional[float]) -> Dict[str, Any]:
    """model.json of a run (the keys in this order; the probes' keys last)."""
    return {"name": args.name, "model": args.model, "version": VERSION,
            "prompt_format": cfg["prompt_format"], "greedy": not sampled,
            "samples": int(args.samples) if sampled else 1,
            "temperature": float(args.temperature) if sampled else 0.0,
            "top_p": float(args.top_p) if sampled else 1.0,
            "sample_seed": int(args.sample_seed) if sampled else None,
            "env_switches": switches, "max_turns": int(args.max_turns),
            "history": bool(args.history),
            "sft_text": bool(args.sft_text),
            "entropy": bool(getattr(args, "entropy", False)),
            "load_seconds": load_seconds, **probe_key}


RESUME_FREE_KEYS = ("name", "load_seconds")  # critic probe: what a resumed folder's model.json may differ in


def check_resume_record(result: Path, record: Dict[str, Any]) -> None:
    """Critic probe: a run resuming a folder with results has exactly its model.json, but for RESUME_FREE_KEYS; else
    it stops before the env and the model start."""
    results, model = Path(result) / "results.jsonl", Path(result) / "model.json"
    if not results.is_file() or not _jsonl(results, partial_ok=True):
        return
    if not model.is_file():
        raise SystemExit(f"{result}: results without model.json; give another --name")
    before = {k: v for k, v in json.loads(model.read_text()).items() if k not in RESUME_FREE_KEYS}
    now = {k: v for k, v in json.loads(json.dumps(record)).items() if k not in RESUME_FREE_KEYS}
    differ = sorted(k for k in set(before) | set(now) if before.get(k) != now.get(k))
    if differ:
        raise SystemExit(f"{result}: its model.json differs from this run's in {differ}; one setting per result "
                         "folder (give another --name)")


def check_env_switches(env: Any, switches: Dict[str, Any]) -> None:
    """Critic probe: the built env has every switch's value (nothing dropped or changed on the way)."""
    cfg = getattr(env, "config", None)
    missing = object()
    wrong = {k: getattr(cfg, k, missing) for k, v in switches.items() if getattr(cfg, k, missing) != v}
    if wrong:
        raise SystemExit(f"--critic-probe: the env's config differs from the switches in {sorted(wrong)}")


def check_task_tables(tasks: Path, tables_dir: str) -> Dict[str, int]:
    """Critic probe: every task of tasks.jsonl has its distance table (<tables_dir>/<task_id>.npz, as
    distance_table.table_path names it), else the run stops before anything starts."""
    ids = [str(r["task_id"]) for r in _jsonl(Path(tasks))]
    missing = [t for t in ids if not (Path(tables_dir) / f"{t}.npz").is_file()]
    if missing:
        raise SystemExit(f"--critic-probe: {len(missing)} of {len(ids)} tasks have no distance table in {tables_dir} "
                         f"(e.g. {missing[:3]})")
    return {"tasks": len(ids), "tables_found": len(ids)}


def sampling_mode(args: argparse.Namespace) -> bool:
    """A sampled run (round 11): several episodes per task, or any temperature above 0."""
    return int(getattr(args, "samples", 1) or 1) > 1 or float(getattr(args, "temperature", 0.0) or 0.0) > 0.0


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
def token_entropy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """{"entropy": token-weighted mean of the recorded answers' entropy, "entropy_tokens"} when run --entropy recorded
    it (round 12 freeze), else {}."""
    pairs = [(float(t["entropy"]), int(t.get("tokens") or 0)) for r in rows for t in (r.get("trajectory") or [])
             if t.get("entropy") is not None]
    tokens = sum(n for _, n in pairs)
    if not pairs or not tokens:
        return {}
    return {"entropy": round(sum(e * n for e, n in pairs) / tokens, 5), "entropy_tokens": tokens}


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
        # round 12 freeze: the stop line's measure (per episode the share of unreadable turns, averaged) and, with
        # run --entropy, the answers' token entropy weighted by their tokens (the RL logs' actor/entropy)
        "invalid_episode_mean": (round(sum(r["invalid_turns"] / r["turns"] for r in ok if r["turns"]) /
                                       sum(1 for r in ok if r["turns"]), 4) if any(r["turns"] for r in ok) else None),
        **token_entropy(ok),
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


VALID_ACTIONS = ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right")


def unreadable_form(answer: str) -> str:
    """Round 12 batch (the user's decision 22): the form of an answer the env could not read: no <action> tag, an
    empty action, the format example copied (action1, action, ...), spaces instead of underscores (turn left), or
    another made-up name; "other" when every name is a real action (e.g. contradictory moves)."""
    m = re.search(r"<action>(.*?)</action>", answer or "", re.S)
    if not m:
        return "no_action_tag"
    acts = [a.strip() for a in m.group(1).split("|") if a.strip()]
    if not acts:
        return "empty_action"
    bad = [a for a in acts if a not in VALID_ACTIONS]
    if any(re.fullmatch(r"action\d*|\.\.\.", a) for a in bad):
        return "format_example_copied"
    if any(a.replace(" ", "_") in VALID_ACTIONS for a in bad):
        return "space_instead_of_underscore"
    return "made_up_name" if bad else "other"


def unreadable_forms(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """{form: count} of the unreadable answers (turns not valid) in the recorded trajectories."""
    forms = Counter(unreadable_form(str(t.get("answer") or "")) for r in rows if "error" not in r
                    for t in (r.get("trajectory") or []) if not t.get("valid", True))
    return dict(forms.most_common())


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
    rules: Dict[Any, List[str]] = {}  # experiment 3A: turn_footprint_radius -> result folders
    for d in sorted(p for p in ev.iterdir() if (p / "results.jsonl").exists()):
        rows = _jsonl(d / "results.jsonl", partial_ok=True)  # other chains' jobs may still be writing theirs
        if rows:
            models[d.name] = metrics(rows)
            rules.setdefault(folder_rule(d), []).append(d.name)
    if len(rules) > 1:  # experiment 3A: a table compares results under one turn rule
        raise SystemExit(f"{ev}: its result folders were evaluated under different turn rules (turn_footprint_radius "
                         + "; ".join(f"{k!r}: {', '.join(v)}" for k, v in rules.items()) + "); one rule per set")
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
                              "collision_rate": metrics(rows)["collision_rate"],
                              "unreadable_share": metrics(rows)["invalid_rate"], "unreadable_forms": unreadable_forms(rows)}
    (ev / "terminal.json").write_text(json.dumps(models, indent=1))
    families = list(FAMILY_CONDITIONS)
    head = ["模型", "题数", "成功率", "终点进度分（成功的题）", "终点进度分（失败的题）", "终点满足的条件比例", "碰撞率",
            "读不出动作的回合比例", "读不出的形式"] + [f"{f} 满足比例（失败的题）" for f in families]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for name, m in models.items():
        fam = m["conditions_met_by_family"]
        cells = [name, str(m["episodes"]), fmt(m["success"]), fmt(m["progress_end_success"]),
                 fmt(m["progress_end_fail"]), fmt(m["conditions_met_share"]), fmt(m["collision_rate"]),
                 fmt(m["unreadable_share"]), "，".join(f"{k} {v}" for k, v in m["unreadable_forms"].items()) or "—"]
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
    r.add_argument("--format-example", default=None, choices=["placeholder", "action_names"],
                   help="round 12 freeze: the format line the model trained with (default the template's: placeholder)")
    r.add_argument("--entropy", action="store_true",
                   help="round 12 freeze: record each answer's token entropy (top-20 log-probabilities, after the "
                        "temperature), as the RL logs report it")
    r.add_argument("--lenient-parse", action="store_true",
                   help="round 12 freeze: read answers leniently when the strict reading finds no action")
    r.add_argument("--turn-footprint-radius", type=float, default=None,
                   help="experiment 3A (the user's decision of 2026-11-06): block in-place turns whose robot-base disc "
                        "of this radius (m, e.g. 0.25) meets a wall or an object box reaching into 0.1-1.0 m (env "
                        "turn_footprint_radius; default: today's rule, turns are never blocked)")
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
    r.add_argument("--grounding-probe", action="store_true",
                   help="the user's supplement of 2026-11-12, item 5: every turn the model first states the bearing and "
                        "distance of the task's first object (<state>bearing=B; distance=D</state>), scored against "
                        "the env's truth (grounding_probe.py); the action is read as before")
    r.add_argument("--grounding-side", action="store_true",
                   help="grounding probe v2 (2026-11-12): the action call is the plain run's; after it a separate "
                        "greedy call on the same frame, the reply prefilled with <state>bearing, asks for the "
                        "statement (grounding_probe.py); the side calls never enter the history")
    r.add_argument("--critic-probe", default=None, metavar="TABLES_DIR",
                   help="gate 0 of docs/PREREG_PPO2.md (the representation probe's episodes): the env with "
                        "critic_features on (the distance tables of TABLES_DIR) and V4's reward (D_cond, progress "
                        "scale 3.3333333333333335; model.json env_switches); frames as PNG, each turn also records the "
                        "truth at its observation's pose, the unrounded reward and the prompt's token digest "
                        "(repr_probe.py); needs --history; default off")
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
