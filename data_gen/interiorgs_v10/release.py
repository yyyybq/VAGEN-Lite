"""Release assembly for InteriorGS v10 (workstream A).

Turns admitted candidates into the files consumed by the existing release
tooling:

  * ``split_manifest.json``  - version-2 manifest (physical scene groups,
    reserved scenes, aliases) accepted by ``scripts/active_spatial_release_contract.scene_partitions``;
  * ``task_registry.jsonl``  - one row per parent task with
    ``source_task_sha256 = dataset_contract.row_digest(raw_task_row)``;
  * ``raw_tasks.jsonl`` / ``rl_tasks_<split>.jsonl`` - the raw rows (with
    ``split``) that the registry hashes; RL files hold RL-admitted rows only;
  * ``sft_trajectories.jsonl`` - internal multi-turn SFT records (actions only,
    no oracle values) accepted by ``scripts/prepare_active_spatial_sft.py``;
  * ``qa_bank.jsonl`` (Yes/No) and ``qa_bank_action_set.jsonl`` (multi-answer
    Act items; need ``configs/prepare_active_spatial_sft.patch``);
  * ``admission.json``, ``regeneration_plan.json``, ``release_summary.json``.

RL, SFT and QA share task identity (task_id + raw-row hash), versions and the
scene-level split; each is admitted by its own rule (admission.py).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .admission import (
    ADMIT,
    chunk_certificate,
    load_policy,
    merge_reviews,
    normalize_renders,
    normalize_reviews,
    turn_boundaries,
    admit,
)
from .contract import ACTIONS, CANONICAL_TASK_METRIC_V1, Funnel, _json_default
from .qa_states import balance_yes_no, certificate, certificate_poses, lineage_fields, score_task

RELEASE_VERSION = "interiorgs_v10_release_v2"
SPLITS = ("train", "val", "test")
VERSION_KEYS = ("camera_model_version", "canonical_task_metric_version", "action_protocol_version")
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Physical scene groups and split assignment
# --------------------------------------------------------------------------
def groups_to_aliases(groups: Any) -> Dict[str, str]:
    """Normalise same-source groups to ``{scene_id: physical_group_id}``.

    Accepts ``{scene: group}``, ``{group: [scenes]}``, ``[[scenes], ...]`` or
    ``{"aliases": {...}}`` / ``{"groups": [...]}``.
    """
    if not groups:
        return {}
    if isinstance(groups, Mapping) and ("aliases" in groups or "groups" in groups):
        out = groups_to_aliases(groups.get("groups") or [])
        out.update({str(k): str(v) for k, v in (groups.get("aliases") or {}).items()})
        return out
    out: Dict[str, str] = {}
    if isinstance(groups, Mapping):
        for k, v in groups.items():
            if isinstance(v, (list, tuple, set)):
                for s in v:
                    out[str(s)] = str(k)
            else:
                out[str(k)] = str(v)
        return out
    for members in groups:
        members = sorted(str(s) for s in members)
        if members:
            key = "group:" + hashlib.sha1("|".join(members).encode()).hexdigest()[:12]
            for s in members:
                out[s] = key
    return out


def same_source_groups(scene_ids: Iterable[str]) -> Dict[str, str]:
    """Same-source groups cannot be derived from scene ids alone: they need the scenes'
    geometry (``scene_identity.group_same_source`` over fingerprints, e.g. the file
    ``cluster_runs/same_source_groups_20261012.json``).  This used to feature-detect a
    function that does not exist and silently return no groups, so that same-source
    scenes could land in different splits; it now refuses."""
    raise ValueError("same-source groups must be passed explicitly (groups=<list of scene-id lists> or a "
                     "{'groups': [...]} file such as cluster_runs/same_source_groups_20261012.json); "
                     "pass groups=[] only after checking that there are none")


def load_groups(path: Union[str, Path]) -> List[List[str]]:
    """Read a same-source groups file ({"groups": [[scene, ...], ...]} or a bare list)."""
    data = json.loads(Path(path).read_text())
    groups = data.get("groups") if isinstance(data, Mapping) else data
    if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
        raise ValueError(f"{path}: expected a list of scene-id lists under 'groups'")
    return [[str(s) for s in g] for g in groups]


def _release_contract():
    path = REPO_ROOT / "scripts" / "active_spatial_release_contract.py"
    spec = importlib.util.spec_from_file_location("active_spatial_release_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assign_splits(scene_ids: Iterable[str], *, groups: Any = None, ratios: Optional[Mapping[str, float]] = None,
                  seed: str = "interiorgs_v10", reserved: Optional[Mapping[str, str]] = None,
                  frozen: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Assign physical scene groups to train/val/test (version-2 manifest).

    * same-source scenes (``groups``, required: ``[]`` means checked and none) never cross splits;
    * reserved scenes and every scene of a ``frozen`` previous manifest keep
      their split (stable across regeneration rounds), and pin their group;
    * remaining groups are visited in a seeded hash order and given to the
      split with the largest deficit against ``ratios`` (scene counts).
    """
    ratios = dict(ratios or {"train": 0.8, "val": 0.1, "test": 0.1})
    total_ratio = sum(float(ratios.get(s, 0.0)) for s in SPLITS)
    reserved = {str(k): str(v) for k, v in (reserved or {}).items()}
    scenes = {str(s) for s in scene_ids} | set(reserved)
    pins: Dict[str, str] = dict(reserved)
    frozen_aliases: Dict[str, str] = {}
    if frozen:
        for split in SPLITS:
            for s in frozen.get(f"{split}_scenes", []) or []:
                if str(s) in pins and pins[str(s)] != split:
                    raise ValueError(f"reserved scene {s} conflicts with frozen manifest")
                pins[str(s)] = split
        scenes |= set(pins)
        frozen_aliases = {str(k): str(v) for k, v in (frozen.get("aliases") or {}).items()}
        for k, v in (frozen.get("reserved_scenes") or {}).items():
            if reserved.get(str(k), v) != v:
                raise ValueError(f"reserved scene {k} reassigned against frozen manifest")
            reserved[str(k)] = str(v)
    aliases = dict(frozen_aliases)
    aliases.update(same_source_groups(scenes) if groups is None else groups_to_aliases(groups))
    members: Dict[str, set] = defaultdict(set)
    for s in scenes:
        members[aliases.get(s, s)].add(s)
    for s, g in aliases.items():  # aliased scenes outside this release still pin the group
        if s in pins:
            members[g].add(s)
    group_split: Dict[str, str] = {}
    for g, ss in members.items():
        wanted = {pins[s] for s in ss if s in pins}
        if len(wanted) > 1:
            raise ValueError(f"physical scene group {g} pinned to several splits: {sorted(wanted)}")
        if wanted:
            group_split[g] = wanted.pop()
    counts = Counter()
    for g, sp in group_split.items():
        counts[sp] += len(members[g])
    n_total = sum(len(v) for v in members.values())
    free = sorted((g for g in members if g not in group_split),
                  key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    for g in free:
        def deficit(sp: str) -> float:
            return float(ratios.get(sp, 0.0)) / total_ratio * n_total - counts[sp]
        best = max(SPLITS, key=lambda sp: (deficit(sp), -SPLITS.index(sp)))
        group_split[g] = best
        counts[best] += len(members[g])
    out_sets = {sp: sorted(s for g, ss in members.items() if group_split[g] == sp for s in ss) for sp in SPLITS}
    manifest = {
        "version": 2,
        "release_version": RELEASE_VERSION,
        "split_unit": "physical_scene_group",
        "seed": seed,
        "ratios": ratios,
        "train_scenes": out_sets["train"],
        "val_scenes": out_sets["val"],
        "test_scenes": out_sets["test"],
        "aliases": {s: g for s, g in sorted(aliases.items()) if s in scenes and (g != s or len(members[g]) > 1)},
        "reserved_scenes": dict(sorted(reserved.items())),
        "physical_groups": len(members),
    }
    _release_contract().scene_partitions(manifest)  # fail closed on leaks / reserved reassignment
    return manifest


def split_lookup(manifest: Mapping[str, Any]) -> Dict[str, str]:
    return {str(s): sp for sp in SPLITS for s in manifest.get(f"{sp}_scenes", [])}


# --------------------------------------------------------------------------
# Raw rows and registry
# --------------------------------------------------------------------------
def finalize_task_row(task_row: Mapping[str, Any], split: str) -> Dict[str, Any]:
    """Raw task row as published (JSON-normalised, private ``_*`` keys dropped, ``split`` set)."""
    from vagen.envs.active_spatial.dataset_contract import row_digest

    row = json.loads(json.dumps({k: v for k, v in task_row.items() if not str(k).startswith("_")},
                                default=_json_default, allow_nan=False))
    row["split"] = split
    row_digest(row)  # raises on NaN / unserialisable content
    return row


def registry_row(raw_row: Mapping[str, Any], modalities: Mapping[str, Any]) -> Dict[str, Any]:
    from vagen.envs.active_spatial.dataset_contract import row_digest

    return {
        "task_id": raw_row["task_id"],
        "source_task_sha256": row_digest(dict(raw_row)),
        "scene_id": str(raw_row["scene_id"]),
        "split": raw_row["split"],
        "versions": {k: raw_row.get(k) for k in VERSION_KEYS},
        "task_type": raw_row.get("task_type"),
        "generator_version": raw_row.get("generator_version"),
        "d_star": (raw_row.get("difficulty") or {}).get("d_star"),
        "modalities": dict(modalities),
    }


# --------------------------------------------------------------------------
# SFT export
# --------------------------------------------------------------------------
_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:°|%|(?:m|cm|mm|meters?|metres?|deg|degrees?|px|pixels?|rad|radians?)\b)",
                      re.IGNORECASE)
_DECIMAL_RE = re.compile(r"\d+\.\d+")
_ORACLE_RE = re.compile(r"\b(?:scores?|rewards?|distances?|d\*|d_star|potential|margins?|thresholds?|optimal|"
                        r"shortest|certificate|oracle|inside[_ ]fraction)\b", re.IGNORECASE)
# The RL env's observation line (env.py: "Current camera pose: " + format_pose6_deg(...)); SFT turns carry the
# same line (observation_pose = "runtime"), so it is not leakage.
_POSE_LINE_RE = re.compile(r"Current camera pose: \[tx=-?\d+\.\d{4}, ty=-?\d+\.\d{4}, tz=-?\d+\.\d{4}, "
                           r"rx=-?\d+\.\d{2}°, ry=-?\d+\.\d{2}°, rz=-?\d+\.\d{2}°\]")
_ACTION_ONLY_RE = re.compile(r"<action>(?:(?:%s)\|)+</action>" % "|".join(ACTIONS))


def _prompt_module():
    from vagen.envs.active_spatial import prompt as P

    return P


def format_example(policy: Mapping[str, Any]) -> str:
    """The export's format example (round 12 freeze, the user's 3.1.1): "placeholder" when the policy names none,
    the text of every SFT set up to round 12; "action_names" shows real action names (prompt.FORMAT_EXAMPLES)."""
    return str(policy["sft"]["export"].get("format_example", "placeholder"))


def protocol_system_prompt(task_type: str, policy: Mapping[str, Any]) -> str:
    """The runtime system prompt for this protocol (same text the RL env shows)."""
    exp = policy["sft"]["export"]
    kwargs = dict(exp["system_prompt"])
    if "format_example" in exp:
        kwargs["format_example"] = format_example(policy)
    return _prompt_module().system_prompt(format=exp["prompt_format"], task_type=task_type, **kwargs)


def _format_block(policy: Mapping[str, Any], first: bool) -> str:
    exp = policy["sft"]["export"]
    P = _prompt_module()
    fn = P.format_prompt.get(exp["prompt_format"], P.format_prompt["no_think"])
    return fn(max_actions_per_step=exp["max_actions_per_turn"], action_sep="|", add_example=first,
              format_example=format_example(policy))


def runtime_poses(raw_row: Mapping[str, Any]) -> List[np.ndarray]:
    """The camera pose after each action of the certificate, as the RL env computes it (its ViewManipulator).

    The generator's own replica (``certificate_poses``) agrees up to float rounding; a value such as -1e-15
    then prints as "-0.00" in one and "0.00" in the other (round 7: 6 of 70 later turns).  The pose line of
    the SFT conversations is built from these poses so it matches the env character for character."""
    from vagen.envs.active_spatial.utils import ViewManipulator

    from .contract import STEP_ROTATION_DEG, STEP_TRANSLATION_M

    vm = ViewManipulator(step_translation=STEP_TRANSLATION_M, step_rotation_deg=STEP_ROTATION_DEG, world_up_axis="Z")
    vm.reset(np.asarray(raw_row["init_camera"]["extrinsics"], dtype=np.float64))
    poses = [vm.get_pose()]
    for a in certificate(raw_row):
        vm.step(a)
        poses.append(vm.get_pose())
    return poses


def pose_line(c2w: Any, normalize_zero: bool = False) -> str:
    """The RL env's "Current camera pose" observation line for a camera-to-world pose (``normalize_zero``: the env
    setting pose_normalize_negative_zero)."""
    from vagen.envs.active_spatial.utils import c2w_extrinsic_to_se3, format_pose6_deg

    return "Current camera pose: " + format_pose6_deg(c2w_extrinsic_to_se3(np.asarray(c2w, dtype=float)),
                                                      normalize_zero=normalize_zero)


def _user_turn(desc: str, first: bool, policy: Mapping[str, Any], pose: Any = None,
               env_feedback: Optional[str] = None) -> str:
    """One user turn; with observation_pose = "runtime" it shows the pose line exactly as the RL env does.
    ``env_feedback``: the feedback line of a later turn (default the policy's, "Action executed.")."""
    P = _prompt_module()
    exp = policy["sft"]["export"]
    mode = exp.get("observation_pose", "none")
    if mode == "runtime":
        if pose is None:
            raise ValueError("observation_pose='runtime' needs the turn's camera pose")
        observation = "<image>\n" + pose_line(pose, bool(exp.get("pose_normalize_negative_zero", False)))
    elif mode == "none":
        observation = "<image>"
    else:
        raise ValueError(f"unknown observation_pose {mode!r} (use 'runtime' or 'none')")
    if first:
        text = P.init_observation_template(observation=observation, task_prompt=desc)
    else:
        text = P.action_template(observation=observation,
                                 env_feedback=env_feedback if env_feedback is not None
                                 else exp.get("env_feedback", "Action executed."), task_prompt=desc)
    return text + "\n" + _format_block(policy, first)


# Round 10 (decision 6 of 2026-10-15): "what to do after a blocked move".  In a share of the conversations one
# extra turn is inserted before a turn of the shortest path: the assistant tries a move the collision detector
# blocks from that pose, the next user turn shows the same frame and pose with the env's detailed feedback
# ("0 of 1 actions executed; blocked: move_forward (blocked by wall)."), and the shortest path goes on unchanged
# (a blocked move does not move the camera).  The move and the turn are drawn per task from a fixed seed.
MOVES = ("move_forward", "move_backward", "move_left", "move_right")
INVERSE = {"move_forward": "move_backward", "move_backward": "move_forward", "move_left": "move_right",
           "move_right": "move_left", "turn_left": "turn_right", "turn_right": "turn_left"}
_BLOCKED_FEEDBACK_RE = re.compile(r"\d+ of \d+ actions executed; blocked: [^\n]*\.")


def step_pose(c2w: Any, action: str) -> np.ndarray:
    """The camera pose after one action, as the env's ViewManipulator computes it."""
    from vagen.envs.active_spatial.utils import ViewManipulator

    from .contract import STEP_ROTATION_DEG, STEP_TRANSLATION_M

    vm = ViewManipulator(step_translation=STEP_TRANSLATION_M, step_rotation_deg=STEP_ROTATION_DEG, world_up_axis="Z")
    vm.reset(np.asarray(c2w, dtype=np.float64))
    vm.step(action)
    return vm.get_pose()


def blocked_move_results(detector: Any, c2w: Any) -> List[Tuple[str, Any]]:
    """(move, CollisionResult) for every move the env's collision detector blocks from this pose (``detector``:
    the scene's, with the env's wall model)."""
    c2w = np.asarray(c2w, dtype=np.float64)
    out = []
    for move in MOVES:
        res = detector.check_collision(step_pose(c2w, move)[:3, 3], previous_position=c2w[:3, 3])
        if res.has_collision:
            out.append((move, res))
    return out


def blocked_moves(detector: Any, c2w: Any) -> List[Tuple[str, str]]:
    """(move, collision text) for every move the env's collision detector blocks from this pose, computed with
    the env's own ViewManipulator and CollisionDetector (``detector``: the scene's, with the env's wall model)."""
    from vagen.envs.active_spatial.env import collision_text

    return [(move, collision_text(res)) for move, res in blocked_move_results(detector, c2w)]


def obstacle_distance(detector: Any, result: Any, position: Any) -> Optional[float]:
    """Distance from ``position`` to the obstacle that blocked a move: the (camera-radius expanded) box of the object
    or the wall segment named in the CollisionResult; None for other collisions (floor, ceiling)."""
    pos = np.asarray(position, dtype=np.float64)
    details = getattr(result, "details", None) or {}
    if result.collision_type == "object":
        oid = str(details.get("object_id", ""))
        boxes = [b for b in getattr(detector, "object_boxes", []) if str(getattr(b, "ins_id", "")) == oid]
        return min(float(b.distance_to_point(pos)) for b in boxes) if boxes else None
    if result.collision_type == "wall" and "wall_start" in details and "wall_end" in details:
        a, b = np.asarray(details["wall_start"], float)[:2], np.asarray(details["wall_end"], float)[:2]
        q = pos[:2]
        ab = b - a
        t = 0.0 if float(ab @ ab) == 0.0 else float(np.clip((q - a) @ ab / float(ab @ ab), 0.0, 1.0))
        return float(np.linalg.norm(q - (a + t * ab)))
    return None


def escape_moves(detector: Any, c2w: Any, result: Any) -> List[Tuple[str, float]]:
    """Round 11: the moves that leave a blocking obstacle -- not blocked themselves and ending farther from the
    obstacle than the start -- with the distance gained, largest gain first (ties in MOVES order)."""
    c2w = np.asarray(c2w, dtype=np.float64)
    d0 = obstacle_distance(detector, result, c2w[:3, 3])
    if d0 is None:
        return []
    out = []
    for move in MOVES:
        nxt = step_pose(c2w, move)
        if detector.check_collision(nxt[:3, 3], previous_position=c2w[:3, 3]).has_collision:
            continue
        d1 = obstacle_distance(detector, result, nxt[:3, 3])
        if d1 is not None and d1 > d0 + 1e-6:
            out.append((move, round(d1 - d0, 6)))
    return sorted(out, key=lambda x: (-x[1], MOVES.index(x[0])))


def env_success_oracle(raw_row: Mapping[str, Any], scene: Any) -> Any:
    """The env's success rule for a task row (the canonical metric and the visibility rules, occluders from the
    scene's detector with the env's wall model), as a difficulty.SuccessOracle for the shortest-path search."""
    from vagen.envs.active_spatial import target_visibility as TV

    from .difficulty import SuccessOracle

    occ = TV.OccluderSet.from_detector(scene.detector)
    item = dict(raw_row)

    def exact(c2w: Any) -> bool:
        return bool(TV.score_with_visibility(item, np.asarray(c2w, float), occ)["success"])

    return SuccessOracle(exact, exact_fn=exact)


def plan_blocked_escape(raw_row: Mapping[str, Any], policy: Mapping[str, Any], scene: Any,
                        max_actions: int = 12) -> Optional[Dict[str, Any]]:
    """Round 11 (decision 5): the blocked turn to insert for this task, or None.

    A task is chosen with probability ``sft.export.blocked_turn_share`` (seed ``blocked_turn_seed`` and the task
    id).  Among the turns of its shortest path whose start pose has a blocked move with at least one escape move
    (``escape_moves``), one turn and one blocked move are drawn; the conversation then shows the blocked attempt,
    the escape move (the one that gains the most distance from the obstacle) and the shortest path from the
    escaped pose (``difficulty.certify_d_star`` with the env's success rule; going back and on along the old path
    bounds it).  The whole episode must fit the env's action limit and the escaped pose must not already succeed.
    Returns {"turn", "move", "blocked_by", "escape", "gain_m", "path", "poses"} (``poses``: the runtime poses from
    the escaped pose on, one per path action plus the start)."""
    from vagen.envs.active_spatial.env import collision_text

    from .difficulty import certify_d_star

    exp = policy["sft"]["export"]
    share = float(exp.get("blocked_turn_share", 0.0) or 0.0)
    if share <= 0.0 or scene is None:
        return None
    rng = random.Random(f"{exp.get('blocked_turn_seed', 20261016)}:{raw_row['task_id']}")
    if rng.random() >= share:
        return None
    cert = certificate(raw_row)
    chunks = chunk_certificate(cert, exp["chunking"], exp["max_actions_per_turn"])
    bounds = turn_boundaries(raw_row, policy)
    shown = runtime_poses(raw_row)
    det = scene.detector
    candidates = []
    for t in range(len(chunks)):
        for move, res in blocked_move_results(det, shown[bounds[t]]):
            esc = escape_moves(det, shown[bounds[t]], res)
            if esc:
                candidates.append((t, move, res, esc))
    if not candidates:
        return None
    oracle = env_success_oracle(raw_row, scene)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    for i in order:
        t, move, res, esc = candidates[i]
        budget = int(max_actions) - bounds[t] - 2  # actions before, the blocked one, the escape
        if budget < 1:
            continue
        for escape, gain in esc:
            start = step_pose(shown[bounds[t]], escape)
            if oracle(start):
                continue
            hint = [INVERSE[escape]] + list(cert[bounds[t]:])
            found = certify_d_star(scene, oracle, start, d_max=min(budget, len(hint)), hint_actions=hint)
            if found.get("d_star") is None or not found.get("certificate") or len(found["certificate"]) > budget:
                continue
            path = list(found["certificate"])
            poses = [start]
            for a in path:
                poses.append(step_pose(poses[-1], a))
            if not oracle(poses[-1]):
                continue
            return {"turn": t, "move": move, "blocked_by": collision_text(res), "escape": escape, "gain_m": gain,
                    "path": path, "poses": [np.asarray(p).tolist() for p in poses],
                    "path_source": found.get("certificate_source")}
    return None


def blocked_selected(raw_row: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    """Whether the seeded draw of ``plan_blocked_escape`` selects this task (before looking for a blocked move)."""
    exp = policy["sft"]["export"]
    share = float(exp.get("blocked_turn_share", 0.0) or 0.0)
    return share > 0.0 and random.Random(f"{exp.get('blocked_turn_seed', 20261016)}:{raw_row['task_id']}").random() < share


def conversation_actions(raw_row: Mapping[str, Any], policy: Mapping[str, Any]) -> int:
    """How many actions the assistant turns of a row's SFT conversation hold (with a ``blocked_plan``: the path up to
    its turn, the blocked move, the escape and the new path)."""
    plan = raw_row.get("blocked_plan")
    cert = certificate(raw_row)
    if not plan:
        return len(cert)
    return turn_boundaries(raw_row, policy)[int(plan["turn"])] + 2 + len(plan["path"])


def plan_frame_states(plan: Mapping[str, Any], policy: Mapping[str, Any]) -> List[int]:
    """Indices into ``plan["poses"]`` whose frames the conversation shows: the start of every turn of the path
    (the escaped pose first) and its end."""
    exp = policy["sft"]["export"]
    ks = [0]
    for c in chunk_certificate(list(plan["path"]), exp["chunking"], exp["max_actions_per_turn"]):
        ks.append(ks[-1] + len(c))
    return sorted(set(ks))


def blocked_insert(raw_row: Mapping[str, Any], policy: Mapping[str, Any], detector: Any, n_turns: int,
                   poses: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """The blocked turn to insert for this task, or None: {"turn": t, "move", "blocked_by"}.  A task is chosen with
    probability ``sft.export.blocked_turn_share`` (seed ``blocked_turn_seed`` and the task id); the turn is drawn
    among the turns whose pose has a blocked move (generated poses keep 0.5 m from walls and objects, so most
    turns have none), the move among the moves blocked there; a task with no such turn gets no insertion."""
    exp = policy["sft"]["export"]
    share = float(exp.get("blocked_turn_share", 0.0) or 0.0)
    if share <= 0.0 or detector is None or n_turns <= 0:
        return None
    rng = random.Random(f"{exp.get('blocked_turn_seed', 20261015)}:{raw_row['task_id']}")
    if rng.random() >= share:
        return None
    candidates = [(t, blocked_moves(detector, poses[t])) for t in range(n_turns)]
    candidates = [(t, b) for t, b in candidates if b]
    if not candidates:
        return None
    t, blocked = candidates[rng.randrange(len(candidates))]
    move, text = blocked[rng.randrange(len(blocked))]
    return {"turn": t, "move": move, "blocked_by": text}


def scan_leakage(record: Mapping[str, Any], policy: Mapping[str, Any]) -> List[str]:
    """Findings of oracle/number leakage in an SFT record (empty list = clean).

    * system turn must be exactly the protocol prompt (no per-episode values);
    * assistant turns must be action-only ``<action>a|b|</action>``;
    * user turns (minus the public task description and the fixed format
      block and the env's pose line) must contain no digits-with-units, decimals or oracle words.
    """
    findings: List[str] = []
    desc = str(record.get("task_description") or "")
    fixed = {_format_block(policy, True), _format_block(policy, False)}
    for i, turn in enumerate(record.get("conversations") or []):
        role, content = turn.get("role"), str(turn.get("content", ""))
        if role == "system":
            if content != protocol_system_prompt(str(record.get("task_type")), policy):
                findings.append(f"turn{i}:system_prompt_not_protocol")
            continue
        if role == "assistant":
            if not _ACTION_ONLY_RE.fullmatch(content.strip()):
                findings.append(f"turn{i}:assistant_not_action_only")
            continue
        text = content.replace(desc, " ") if desc else content
        text = _POSE_LINE_RE.sub(" ", text)
        for block in fixed:
            text = text.replace(block, " ")
        text = _BLOCKED_FEEDBACK_RE.sub(" ", text)  # the env's own feedback line of an inserted blocked turn
        for name, rx in (("unit", _UNIT_RE), ("decimal", _DECIMAL_RE), ("oracle_word", _ORACLE_RE)):
            m = rx.search(text)
            if m:
                findings.append(f"turn{i}:{name}:{m.group(0)!r}")
    return findings


def build_sft_record(raw_row: Mapping[str, Any], images: Mapping[str, str], policy: Mapping[str, Any], *,
                     semantic_review: Optional[Mapping[str, Any]] = None,
                     detector: Any = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """One multi-turn no-think SFT conversation from the certificate (actions only).

    ``detector`` (round 10, ``blocked_turn_style`` "continue"): the scene's collision detector; with it, a share of
    the conversations get one inserted blocked turn and then go on along the shortest path (``blocked_insert``).
    Round 11 (style "escape"): a row carrying ``blocked_plan`` (``plan_blocked_escape``, made by sft-frames) shows
    the blocked attempt, the escape move and the shortest path from the escaped pose; its frames are the plan's
    states ``x<k>``."""
    from vagen.envs.active_spatial.dataset_contract import row_digest
    from vagen.envs.active_spatial.env import collision_feedback

    exp = policy["sft"]["export"]
    tid = str(raw_row["task_id"])
    cert = certificate(raw_row)
    chunks = chunk_certificate(cert, exp["chunking"], exp["max_actions_per_turn"])
    bounds = turn_boundaries(raw_row, policy)
    d = len(cert)
    if not chunks:
        return None, "sft_empty_certificate"
    plan = raw_row.get("blocked_plan")
    if plan:
        return _escape_record(raw_row, images, policy, plan, semantic_review)
    frame = {k: images.get(f"{tid}:s{k}") for k in range(d + 1)}
    if any(frame[b] is None for b in bounds):
        return None, "sft_frame_missing_image"
    poses = certificate_poses(raw_row)
    terminal = score_task(raw_row, poses[-1])
    if terminal is None or not bool(terminal.get("success")):
        return None, "sft_terminal_not_successful"
    desc = str(raw_row.get("task_description") or "").strip()
    shown = runtime_poses(raw_row)  # the pose lines exactly as the env prints them
    insert = (blocked_insert(raw_row, policy, detector, len(chunks), [shown[b] for b in bounds[:-1]])
              if exp.get("blocked_turn_style", "continue") == "continue" else None)
    conv: List[Dict[str, Any]] = [{"role": "system", "content": protocol_system_prompt(str(raw_row["task_type"]), policy)}]
    for t, chunk in enumerate(chunks):
        feedback = None
        if insert is not None and insert["turn"] == t:
            # the blocked attempt: same frame and pose before and after, the env's detailed feedback in between
            conv.append({"role": "user", "content": _user_turn(desc, t == 0, policy, shown[bounds[t]]),
                         "image_path": frame[bounds[t]]})
            conv.append({"role": "assistant", "content": "<action>" + insert["move"] + "|</action>"})
            feedback = collision_feedback([insert["move"]], [(insert["move"], insert["blocked_by"])])
        conv.append({"role": "user", "content": _user_turn(desc, t == 0 and feedback is None, policy, shown[bounds[t]],
                                                           env_feedback=feedback),
                     "image_path": frame[bounds[t]]})
        conv.append({"role": "assistant", "content": "<action>" + "".join(a + "|" for a in chunk) + "</action>"})
    if exp.get("explicit_done"):
        conv.append({"role": "user", "content": _user_turn(desc, False, policy, shown[d]), "image_path": frame[d]})
        conv.append({"role": "assistant", "content": "<action>done|</action>"})
    turn_images = [str(t["image_path"]) for t in conv if t["role"] == "user"]
    image_paths = list(turn_images)
    if frame[d] not in image_paths:
        image_paths.append(frame[d])
    record: Dict[str, Any] = {
        "id": f"sft_{tid}",
        "source_task_id": tid,
        "parent_task_id": tid,
        "source_task_sha256": row_digest(dict(raw_row)),
        "source_versions": {k: raw_row.get(k) for k in VERSION_KEYS},
        "scene_id": str(raw_row["scene_id"]),
        "split": raw_row["split"],
        "task_type": raw_row.get("task_type"),
        "task_description": desc,
        "success": True,
        "score_contract": {"success_source": "canonical_gates",
                           "metric_version": raw_row.get("canonical_task_metric_version"),
                           "terminal_checked_by": "release.build_sft_record"},
        "prompt_format": exp["prompt_format"],
        "chunking": exp["chunking"],
        "trajectory_steps": len(chunks) + (1 if insert else 0),
        "total_actions": d,
        "blocked_insert": insert,
        "d_star": (raw_row.get("difficulty") or {}).get("d_star"),
        "conversations": conv,
        "conversation_image_paths": turn_images,
        "image_paths": image_paths,
        "primitive_trajectory_trace": [{"step": k, "action": cert[k - 1] if k else None,
                                        "pose_c2w": np.asarray(p).tolist()} for k, p in enumerate(poses)],
        "semantic_review": dict(semantic_review or {}),
    }
    if all(frame[k] is not None for k in range(d + 1)):
        record["primitive_image_paths"] = [frame[k] for k in range(d + 1)]
    findings = scan_leakage(record, policy)
    if findings:
        return None, "sft_text_leakage"
    return record, None


def _escape_record(raw_row: Mapping[str, Any], images: Mapping[str, str], policy: Mapping[str, Any],
                   plan: Mapping[str, Any], semantic_review: Optional[Mapping[str, Any]]
                   ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The round-11 conversation of a row with a ``blocked_plan``: the shortest path up to the plan's turn, the
    blocked attempt (same frame and pose after it, the env's detailed feedback), the escape move, then the
    shortest path from the escaped pose in runs of identical actions."""
    from vagen.envs.active_spatial.dataset_contract import row_digest
    from vagen.envs.active_spatial.env import collision_feedback

    exp = policy["sft"]["export"]
    tid = str(raw_row["task_id"])
    cert = certificate(raw_row)
    chunks = chunk_certificate(cert, exp["chunking"], exp["max_actions_per_turn"])
    bounds = turn_boundaries(raw_row, policy)
    t = int(plan["turn"])
    path = list(plan["path"])
    ppos = [np.asarray(p, dtype=float) for p in plan["poses"]]
    frame = {k: images.get(f"{tid}:s{k}") for k in bounds[: t + 1]}
    pframe = {k: images.get(f"{tid}:x{k}") for k in plan_frame_states(plan, policy)}
    if any(v is None for v in list(frame.values()) + list(pframe.values())):
        return None, "sft_frame_missing_image"
    terminal = score_task(raw_row, ppos[-1])
    if terminal is None or not bool(terminal.get("success")):
        return None, "sft_terminal_not_successful"
    desc = str(raw_row.get("task_description") or "").strip()
    shown = runtime_poses(raw_row)
    conv: List[Dict[str, Any]] = [{"role": "system", "content": protocol_system_prompt(str(raw_row["task_type"]), policy)}]

    def turn(first: bool, pose: Any, image: str, answer: List[str], feedback: Optional[str] = None) -> None:
        conv.append({"role": "user", "content": _user_turn(desc, first, policy, pose, env_feedback=feedback),
                     "image_path": image})
        conv.append({"role": "assistant", "content": "<action>" + "".join(a + "|" for a in answer) + "</action>"})

    for k in range(t):
        turn(k == 0, shown[bounds[k]], frame[bounds[k]], chunks[k])
    turn(t == 0, shown[bounds[t]], frame[bounds[t]], [plan["move"]])
    blocked = collision_feedback([plan["move"]], [(plan["move"], plan["blocked_by"])])
    turn(False, shown[bounds[t]], frame[bounds[t]], [plan["escape"]], feedback=blocked)
    k = 0
    for c in chunk_certificate(path, exp["chunking"], exp["max_actions_per_turn"]):
        turn(False, ppos[k], pframe[k], c)
        k += len(c)
    turn_images = [str(u["image_path"]) for u in conv if u["role"] == "user"]
    image_paths = list(turn_images)
    if pframe[len(path)] not in image_paths:
        image_paths.append(pframe[len(path)])
    actions = list(cert[: bounds[t]]) + [plan["move"], plan["escape"]] + path
    record: Dict[str, Any] = {
        "id": f"sft_{tid}", "source_task_id": tid, "parent_task_id": tid,
        # the row as published in rows.jsonl, plan included: the task registry is written from those rows
        "source_task_sha256": row_digest(dict(raw_row)),
        "source_versions": {k2: raw_row.get(k2) for k2 in VERSION_KEYS},
        "scene_id": str(raw_row["scene_id"]), "split": raw_row["split"], "task_type": raw_row.get("task_type"),
        "task_description": desc, "success": True,
        "score_contract": {"success_source": "canonical_gates",
                           "metric_version": raw_row.get("canonical_task_metric_version"),
                           "terminal_checked_by": "release.build_sft_record"},
        "prompt_format": exp["prompt_format"], "chunking": exp["chunking"],
        "trajectory_steps": sum(1 for u in conv if u["role"] == "user"),
        "total_actions": len(actions),
        "blocked_insert": {"style": "escape", "turn": t, "move": plan["move"], "blocked_by": plan["blocked_by"],
                           "escape": plan["escape"], "gain_m": plan.get("gain_m"), "path_actions": len(path),
                           "path_extra_actions": len(path) - (len(cert) - bounds[t])},
        "d_star": (raw_row.get("difficulty") or {}).get("d_star"),
        "conversations": conv, "conversation_image_paths": turn_images, "image_paths": image_paths,
        "semantic_review": dict(semantic_review or {}),
    }
    findings = scan_leakage(record, policy)
    if findings:
        return None, "sft_text_leakage"
    return record, None


# --------------------------------------------------------------------------
# QA stamping
# --------------------------------------------------------------------------
def stamp_qa(q: Mapping[str, Any], raw_row: Mapping[str, Any], image_path: str,
             semantic_review: Mapping[str, Any]) -> Dict[str, Any]:
    rec = copy.deepcopy(dict(q))
    rec.update(lineage_fields(dict(raw_row)))
    rec["public_observation"] = dict(rec.get("public_observation") or {}, image_path=str(image_path),
                                     observation_config="single_image", history=[])
    rec["observability_validity"] = "valid"
    rec["render_status"] = "success"
    rec["qa_sft_allowed"] = True
    rec["semantic_review"] = dict(semantic_review)
    return rec


# --------------------------------------------------------------------------
# Regeneration planning
# --------------------------------------------------------------------------
def _reason_action(reason: str, policy: Mapping[str, Any]) -> str:
    spec = policy.get("reasons", {})
    if reason in spec:
        return spec[reason].get("regeneration", "none")
    if reason.startswith("precheck:"):
        return spec.get("precheck:*", {}).get("regeneration", "resample_episode")
    return "none"


def scene_stats(candidates: Iterable[Mapping[str, Any]], admission: Mapping[str, Any], policy: Mapping[str, Any],
                generator_status: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """Per-scene admitted counts, failure reasons, blacklists and generator state."""
    stats: Dict[str, Dict[str, Any]] = {}

    def entry(scene: str) -> Dict[str, Any]:
        return stats.setdefault(scene, {"admitted_rl": 0, "attempted": 0, "hold": 0, "reject": 0, "reasons": Counter(),
                                        "pending_rereview": 0, "blacklist_objects": set(), "blacklist_pairs": set(),
                                        "rereview_task_ids": [], "exhausted": False, "rounds": 0,
                                        "zero_admit_rounds": 0, "families": Counter()})

    rl = admission["rl"]["decisions"]
    for row in candidates:
        tid, scene = str(row["task_id"]), str(row["scene_id"])
        e = entry(scene)
        e["attempted"] += 1
        dec = rl.get(tid, {"status": "HOLD", "reasons": ["not_admitted"]})
        if dec["status"] == ADMIT:
            e["admitted_rl"] += 1
            e["families"][str(row.get("task_type"))] += 1
            continue
        e["hold" if dec["status"] == "HOLD" else "reject"] += 1
        e["reasons"].update(dec["reasons"][:1])
        if dec["status"] == "HOLD" and _reason_action(dec["reasons"][0], policy) == "rereview":
            e["pending_rereview"] += 1
            e["rereview_task_ids"].append(tid)
        ids = [str(o.get("id")) for o in ((row.get("target_object") or {}).get("objects") or [])]
        for inv in admission.get("task_invalid", {}).get(tid, []):
            act = _reason_action(inv["reason"], policy)
            if act in ("blacklist_object", "blacklist_object_ref") and inv.get("object_id") is not None:
                e["blacklist_objects"].add(str(inv["object_id"]))
            elif act == "blacklist_pair":
                e["blacklist_pairs"].add("|".join(sorted(ids)))
    for scene, gs in (generator_status or {}).items():
        e = entry(str(scene))
        for key in ("exhausted", "rounds", "zero_admit_rounds"):
            if key in gs:
                e[key] = gs[key]
        if "candidates_available" in gs:
            e["candidates_available"] = gs["candidates_available"]
    return stats


def plan_regeneration(stats: Mapping[str, Mapping[str, Any]], policy: Mapping[str, Any],
                      round_index: int = 0) -> Dict[str, Any]:
    """Per-scene regeneration requests until the targets are met or scenes are exhausted.

    Targets: >= ``min_tasks_per_scene`` RL tasks per scene, >= ``target_total_tasks``
    in total and >= ``target_scenes`` scenes with at least one task.  Request
    sizes divide the deficit by a Beta-smoothed per-scene admission rate.
    """
    cfg = policy["regeneration"]
    min_t, cap_t = int(cfg["min_tasks_per_scene"]), int(cfg["max_tasks_per_scene"])
    total = sum(int(s["admitted_rl"]) for s in stats.values())
    covered = sum(1 for s in stats.values() if int(s["admitted_rl"]) > 0)
    meeting = sum(1 for s in stats.values() if int(s["admitted_rl"]) >= min_t)

    def exhausted(s: Mapping[str, Any]) -> bool:
        return bool(s.get("exhausted")) or int(s.get("rounds", 0)) >= int(cfg["max_rounds"]) \
            or int(s.get("zero_admit_rounds", 0)) >= int(cfg["exhaust_after_zero_admit_rounds"])

    def rate(s: Mapping[str, Any]) -> float:
        a, n = float(s["admitted_rl"]), float(s["attempted"])
        k = float(cfg["prior_strength"])
        return max(float(cfg["min_admit_rate"]), (a + k * float(cfg["prior_admit_rate"])) / (n + k))

    eligible = {sc: s for sc, s in stats.items() if not exhausted(s)}
    target = {sc: max(min_t, int(s["admitted_rl"])) for sc, s in eligible.items()}
    projected = total + sum(target[sc] - int(s["admitted_rl"]) for sc, s in eligible.items())
    extra = max(0, int(cfg["target_total_tasks"]) - projected)
    order = sorted(eligible, key=lambda sc: (-rate(eligible[sc]), sc))
    while extra > 0 and order:
        progressed = False
        for sc in order:
            if extra <= 0:
                break
            if target[sc] < cap_t:
                target[sc] += 1
                extra -= 1
                progressed = True
        if not progressed:
            break
    requests = []
    for sc in sorted(eligible):
        s = eligible[sc]
        deficit = target[sc] - int(s["admitted_rl"])
        if deficit <= 0:
            continue
        r = rate(s)
        expected_from_rereview = r * int(s.get("pending_rereview", 0))
        n_new = int(math.ceil(max(0.0, deficit - expected_from_rereview) / r * float(cfg["oversample"])))
        n_new = min(n_new, int(cfg["max_request_per_scene_round"]))
        hints = Counter()
        for reason, n in Counter(s.get("reasons", {})).items():
            hints[_reason_action(reason, policy)] += int(n)
        requests.append({
            "scene_id": sc,
            "round": int(round_index) + 1,
            "priority": 0 if int(s["admitted_rl"]) == 0 else 1 if int(s["admitted_rl"]) < min_t else 2,
            "admitted_rl": int(s["admitted_rl"]),
            "target_rl": target[sc],
            "deficit": deficit,
            "estimated_admit_rate": round(r, 4),
            "request_candidates": n_new,
            "rereview_task_ids": sorted(s.get("rereview_task_ids", [])),
            "blacklist_objects": sorted(s.get("blacklist_objects", [])),
            "blacklist_pairs": sorted(s.get("blacklist_pairs", [])),
            "hints": dict(sorted(hints.items())),
        })
    requests.sort(key=lambda r: (r["priority"], -r["deficit"], r["scene_id"]))
    exhausted_scenes = sorted(sc for sc, s in stats.items() if exhausted(s))
    met = total >= int(cfg["target_total_tasks"]) and covered >= int(cfg["target_scenes"]) and \
        all(int(s["admitted_rl"]) >= min_t for s in eligible.values())
    return {
        "round": int(round_index),
        "targets": {"min_tasks_per_scene": min_t, "target_total_tasks": int(cfg["target_total_tasks"]),
                    "target_scenes": int(cfg["target_scenes"])},
        "status": {"targets_met": met, "total_admitted_rl": total, "scenes_covered": covered,
                   "scenes_meeting_min": meeting, "scenes_known": len(stats), "eligible_scenes": len(eligible),
                   "exhausted_scenes": len(exhausted_scenes),
                   "projected_total_if_all_requests_admit": total + sum(r["deficit"] for r in requests),
                   "max_reachable_scenes": covered + sum(1 for r in requests if r["admitted_rl"] == 0)},
        "requests": [] if met else requests,
        "exhausted_scene_ids": exhausted_scenes,
    }


def take_from_reserve(plan: Mapping[str, Any], stats: Mapping[str, Mapping[str, Any]],
                      reserve_rows: Iterable[Mapping[str, Any]], *, reviewed: Iterable[str] = (),
                      quotas: Optional[Mapping[str, Sequence[Optional[int]]]] = None) -> Dict[str, Any]:
    """Serve the plan's per-scene requests from the reserve pool (``tasks_reserve.jsonl``).

    Only scenes with a request get reserve tasks rendered (on-demand reserve,
    user decision 2026-10-12).  Per scene, up to ``request_candidates`` reserve
    tasks are taken in stored order, skipping tasks already reviewed, tasks on
    a blacklisted object and tasks whose object set is a blacklisted pair.
    With ``quotas`` ({family: (min, max)}), families below their minimum
    (counting admitted tasks plus the ones taken) go first and families at
    their maximum last, so the maximum is soft as in generation.
    """
    reviewed = {str(t) for t in reviewed}
    by_scene: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in reserve_rows:
        if str(r["task_id"]) not in reviewed:
            by_scene[str(r["scene_id"])].append(r)
    render: List[Mapping[str, Any]] = []
    per_scene: Dict[str, Dict[str, Any]] = {}
    for req in plan.get("requests", []):
        sc = str(req["scene_id"])
        bad_obj = set(req.get("blacklist_objects", []))
        bad_pair = set(req.get("blacklist_pairs", []))
        pool = []
        for r in by_scene.get(sc, []):
            ids = [str(o.get("id")) for o in ((r.get("target_object") or {}).get("objects") or [])]
            if bad_obj & set(ids) or "|".join(sorted(ids)) in bad_pair:
                continue
            pool.append(r)
        fam = Counter((stats.get(sc) or {}).get("families") or {})
        want = min(int(req["request_candidates"]), len(pool))
        taken: List[Mapping[str, Any]] = []
        while len(taken) < want:
            def tier(r: Mapping[str, Any]) -> int:
                lo, hi = (quotas or {}).get(str(r["task_type"]), (None, None))
                n = fam[str(r["task_type"])]
                return 0 if lo is not None and n < int(lo) else 2 if hi is not None and n >= int(hi) else 1
            pick = min(pool, key=tier)  # min() keeps the first of equal tiers, i.e. stored order
            pool.remove(pick)
            taken.append(pick)
            fam[str(pick["task_type"])] += 1
        render.extend(taken)
        per_scene[sc] = {"requested": int(req["request_candidates"]), "taken": len(taken),
                         "left_in_reserve": len(pool), "task_ids": [str(r["task_id"]) for r in taken]}
    return {"render": render, "per_scene": per_scene,
            "short_scenes": sorted(sc for sc, p in per_scene.items() if p["taken"] < p["requested"])}


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def d_star_band(d: Optional[int], bands: Sequence[Sequence[int]]) -> str:
    if d is None:
        return "unknown"
    for lo, hi in bands:
        if int(lo) <= int(d) <= int(hi):
            return str(lo) if lo == hi else f"{lo}-{hi}"
    return f">{bands[-1][1]}" if bands else str(d)


def summarize(raw_rows: Mapping[str, Mapping[str, Any]], rl_ids: Sequence[str], sft_records: Sequence[Mapping[str, Any]],
              qa_records: Sequence[Mapping[str, Any]], admission: Mapping[str, Any], manifest: Mapping[str, Any],
              policy: Mapping[str, Any], plan: Mapping[str, Any], funnel: Funnel) -> Dict[str, Any]:
    bands = policy["release"]["d_star_bands"]
    cfg = policy["regeneration"]

    def tally(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        rows = list(rows)
        out = {"total": len(rows), "by_family": Counter(), "by_split": Counter(), "by_d_star_band": Counter(),
               "by_family_split": Counter()}
        for r in rows:
            fam, sp = str(r.get("task_type")), str(r.get("split"))
            d = (r.get("difficulty") or {}).get("d_star") if "difficulty" in r else r.get("d_star")
            out["by_family"][fam] += 1
            out["by_split"][sp] += 1
            out["by_d_star_band"][d_star_band(d, bands)] += 1
            out["by_family_split"][f"{fam}/{sp}"] += 1
        return {k: (dict(sorted(v.items())) if isinstance(v, Counter) else v) for k, v in out.items()}

    rl_rows = [raw_rows[t] for t in rl_ids]
    per_scene = Counter(str(r["scene_id"]) for r in rl_rows)
    qa_kinds = Counter(f"{r.get('qa_kind')}/{r.get('private_answer') if r.get('answer_format') == 'yes_no' else 'set'}"
                       for r in qa_records)
    qa_split = Counter(str(r.get("split")) for r in qa_records)
    scene_cov = {
        "scenes_in_manifest": sum(len(manifest.get(f"{s}_scenes", [])) for s in SPLITS),
        "scenes_with_rl": len(per_scene),
        "scenes_with_min_tasks": sum(1 for n in per_scene.values() if n >= int(cfg["min_tasks_per_scene"])),
        "scenes_with_any_modality": len({str(r["scene_id"]) for r in raw_rows.values()}),
        "rl_tasks_per_scene_hist": dict(sorted(Counter(min(n, 30) for n in per_scene.values()).items())),
    }
    return {
        "release_version": RELEASE_VERSION,
        "policy_version": policy["policy_version"],
        "targets": {
            "rl_tasks": {"target": int(cfg["target_total_tasks"]), "actual": len(rl_rows),
                         "met": len(rl_rows) >= int(cfg["target_total_tasks"])},
            "scenes": {"target": int(cfg["target_scenes"]), "actual": len(per_scene),
                       "met": len(per_scene) >= int(cfg["target_scenes"])},
        },
        "rl": tally(rl_rows),
        "sft": tally(sft_records),
        "qa": {"total": len(qa_records), "by_kind_label": dict(sorted(qa_kinds.items())),
               "by_split": dict(sorted(qa_split.items()))},
        "registered_tasks": tally(raw_rows.values()),
        "scene_coverage": scene_cov,
        "split_scene_counts": {s: len(manifest.get(f"{s}_scenes", [])) for s in SPLITS},
        "admission_status": {m: admission[m]["status_counts"] for m in ("rl", "sft", "qa")},
        "admission_primary_reasons": {m: admission[m]["primary_reason_counts"] for m in ("rl", "sft", "qa")},
        "task_invalid_tasks": len(admission.get("task_invalid", {})),
        "release_funnel": funnel.to_dict(),
        "regeneration_status": plan.get("status"),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, default=_json_default, allow_nan=False) + "\n")
            n += 1
    return n


def build_release(out_dir: Union[str, Path], candidates: Sequence[Mapping[str, Any]], reviews: Any,
                  policy: Optional[Mapping[str, Any]] = None, *, qa_items: Sequence[Mapping[str, Any]] = (),
                  renders: Any = None, replay: Optional[Mapping[str, Any]] = None,
                  prechecks: Optional[Mapping[str, Any]] = None, groups: Any = None,
                  reserved: Optional[Mapping[str, str]] = None, frozen_manifest: Optional[Mapping[str, Any]] = None,
                  generator_status: Optional[Mapping[str, Mapping[str, Any]]] = None,
                  round_index: int = 0) -> Dict[str, Any]:
    """Admit, split, export and summarise one release round into a new directory."""
    policy = dict(policy or load_policy())
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"release directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    funnel = Funnel()
    candidates = [dict(c) for c in candidates]
    rows = {str(r["task_id"]): r for r in candidates}
    if len(rows) != len(candidates):
        raise ValueError("duplicate task_id among candidates")
    render_map = normalize_renders(renders) or {}
    admission = admit(candidates, reviews, policy, qa_items=qa_items, renders=render_map, replay=replay,
                      prechecks=prechecks)
    merged_reviews = {tid: merge_reviews(lst, policy) for tid, lst in normalize_reviews(reviews).items()}

    scenes = {str(r["scene_id"]) for r in candidates} | {str(s) for s in (generator_status or {})}
    rel = policy["release"]
    manifest = assign_splits(scenes, groups=groups, ratios=rel["ratios"], seed=str(rel["seed"]),
                             reserved=reserved, frozen=frozen_manifest)
    lookup = split_lookup(manifest)

    def review_meta(tid: str, modality: str) -> Dict[str, Any]:
        rv = merged_reviews.get(tid) or {}
        return {"status": "PASS", "modality": modality, "policy_version": policy["policy_version"],
                "review_schema_version": rv.get("review_schema_version"), "reviewer": rv.get("reviewer")}

    # Raw rows: every task not task_invalid is finalised (hash binds split); only
    # tasks with an admitted modality are registered below.
    raw: Dict[str, Dict[str, Any]] = {}
    for tid, row in rows.items():
        if tid not in admission["task_invalid"]:
            raw[tid] = finalize_task_row(row, lookup[str(row["scene_id"])])

    # SFT
    sft_records: List[Dict[str, Any]] = []
    for tid in admission["sft"]["admitted"]:
        rec, why = build_sft_record(raw[tid], render_map, policy, semantic_review=review_meta(tid, "sft"))
        if rec is None:
            funnel.add("sft_export", why or "unknown")
            continue
        funnel.add("sft_export", "exported")
        sft_records.append(rec)

    # QA (per state, independent of the parent RL decision)
    qa_stamped: List[Dict[str, Any]] = []
    qa_by_id = {str(q["sample_id"]): q for q in qa_items}
    for sid in admission["qa"]["admitted"]:
        q = qa_by_id[sid]
        tid = str(q["parent_task_id"])
        image = render_map.get(f"{tid}:{q['state_id']}")
        if tid not in raw or image is None:
            funnel.add("qa_export", "missing_parent_or_image")
            continue
        qa_stamped.append(stamp_qa(q, raw[tid], image, review_meta(tid, "qa")))
    if policy["qa"].get("balance_yes_no", True):
        qa_final = balance_yes_no(qa_stamped, seed=str(rel["seed"]), funnel=funnel,
                                  max_majority_ratio=float(policy["qa"].get("max_majority_ratio", 1.0)))
    else:
        qa_final = qa_stamped
    qa_yesno = [q for q in qa_final if q.get("answer_format", "yes_no") == "yes_no"]
    qa_set = [q for q in qa_final if q.get("answer_format") == "action_set"]

    # Registry: parents of every exported record
    rl_ids = [t for t in admission["rl"]["admitted"] if t in raw]
    modalities: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"rl": False, "sft": False, "qa": 0})
    for t in rl_ids:
        modalities[t]["rl"] = True
    for r in sft_records:
        modalities[r["source_task_id"]]["sft"] = True
    for q in qa_final:
        modalities[str(q["parent_task_id"])]["qa"] += 1
    registered = {t: raw[t] for t in sorted(modalities)}
    registry = [registry_row(registered[t], modalities[t]) for t in sorted(registered)]

    plan = plan_regeneration(scene_stats(candidates, admission, policy, generator_status), policy, round_index)
    summary = summarize(registered, rl_ids, sft_records, qa_final, admission, manifest, policy, plan, funnel)

    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_jsonl(out / "task_registry.jsonl", registry)
    _write_jsonl(out / "raw_tasks.jsonl", (registered[t] for t in sorted(registered)))
    files = {}
    for sp in SPLITS:
        files[f"rl_tasks_{sp}"] = _write_jsonl(out / f"rl_tasks_{sp}.jsonl",
                                               (raw[t] for t in rl_ids if raw[t]["split"] == sp))
    files["sft_trajectories"] = _write_jsonl(out / "sft_trajectories.jsonl", sft_records)
    files["qa_bank"] = _write_jsonl(out / "qa_bank.jsonl", qa_yesno)
    files["qa_bank_action_set"] = _write_jsonl(out / "qa_bank_action_set.jsonl", qa_set)
    files["task_registry"] = len(registry)
    (out / "admission.json").write_text(json.dumps(admission, indent=1, sort_keys=True, default=_json_default) + "\n")
    (out / "regeneration_plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    summary["files"] = files
    (out / "release_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
