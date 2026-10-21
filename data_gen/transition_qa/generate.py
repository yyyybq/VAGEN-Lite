"""Transition-question bank from the SFT trajectories (docs/transition_qa/PROGRESS.md, step 2).

    python -m data_gen.transition_qa.generate raw   --sft-dir SFT --scenes-root META --out OUT [--workers 12] [--tag T]
    python -m data_gen.transition_qa.generate probe --out OUT --split-manifest M [--scenes-from TASKS] [--per-qtype 40]
    python -m data_gen.transition_qa.generate build --out OUT --split-manifest M [--seed 20261018]

``raw``: per scene (one process each), every anchor state of every trajectory -> candidate questions with the
reasons for the ones not asked (OUT/raw/<scene>[__T].jsonl, .json).  Finished scenes are skipped.
``probe``: the fixed probe set from the val candidates (OUT/p1_probe.jsonl, probe_stats.json); build leaves its
questions out of the test set.  ``build``: balance answers, forward-question caps, the class rules, scene cap ->
OUT/p1_train.jsonl, p1_test.jsonl (LLaMA-Factory sharegpt rows plus metadata) and stats.json.
Field meanings and the rules: docs/transition_qa/DATA_CONTRACT.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data_gen.interiorgs_v10.contract import ACTIONS

from . import geometry as G
from . import questions as Q

ROLE_OF_POSITION = ("start", "middle", "terminal")
_INPUTS: Dict[str, Any] = {}


# =============================================================================
# io
# =============================================================================
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def frame_id_of(path: str) -> str:
    name = str(path).rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".png") else name


def message_role(m: Dict[str, Any]) -> str:
    return str(m.get("role") or m.get("from") or "")


def message_text(m: Dict[str, Any]) -> str:
    return str(m.get("content") if m.get("content") is not None else m.get("value") or "")


def turn_actions(text: str) -> List[str]:
    import re

    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    return [a.strip() for a in (match.group(1) if match else "").split("|") if a.strip()]


def trajectory_steps(traj: Dict[str, Any], poses: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(images, turns): images[i] = {state_id, image, c2w}; turns[i] = the actions between image i and i + 1
    and whether the env reported a blocked move for that turn."""
    images = []
    for p in traj["image_paths"]:
        fid = frame_id_of(p)
        images.append({"frame_id": fid, "state_id": fid.rsplit("__", 1)[-1], "image": p,
                       "c2w": np.asarray(poses[fid], float)})
    conv = [m for m in traj["conversations"] if message_role(m) != "system"]
    users = [m for m in conv if message_role(m) in ("user", "human")]
    assistants = [m for m in conv if message_role(m) in ("assistant", "gpt")]
    turns = []
    for i, msg in enumerate(assistants):
        feedback = message_text(users[i + 1]) if i + 1 < len(users) else ""
        turns.append({"actions": turn_actions(message_text(msg)), "blocked": "blocked" in feedback.lower()})
    if len(images) != len(turns) + 1:
        raise ValueError(f"{traj.get('id')}: {len(images)} images for {len(turns)} turns")
    return images, turns


# =============================================================================
# raw candidates of one scene
# =============================================================================
def pick_anchor_positions(images: Sequence[Dict[str, Any]], task_id: str, n: int) -> List[Tuple[int, str]]:
    """Positions in ``images`` of at most ``n`` distinct frames (``n`` <= 0: all of them): the terminal frame, the
    start frame and random others.  A frame shown twice (a blocked turn) is taken at its last position, so its turn
    is the one that moved."""
    last_pos: Dict[str, int] = {}
    for i, im in enumerate(images):
        last_pos[im["frame_id"]] = i
    order = sorted(last_pos.values())
    if n <= 0:
        n = len(order)
    terminal, start = order[-1], order[0]
    picks = [(terminal, "terminal")]
    if start != terminal and len(picks) < n:
        picks.append((start, "start"))
    middle = [p for p in order if p not in (start, terminal)]
    rng = random.Random(Q.stable_int(task_id, "anchors"))
    rng.shuffle(middle)
    for p in middle[: max(0, n - len(picks))]:
        picks.append((p, "middle"))
    return sorted(picks)


def pre_visible(kind: str, m: Dict[str, Any]) -> bool:
    """The asked objects are clearly visible in the image the model sees (forward questions)."""
    o = m["objects"]
    if kind in ("left_right", "closer"):
        return o["a"]["clearly_visible"] and o["b"]["clearly_visible"]
    if kind in ("distance_bin", "height_bin"):
        return o["a"]["clearly_visible"]
    if kind == "occluded":
        return o["b"]["clearly_visible"]
    return True  # both_in_view asks exactly whether they are in view


def _record(base: Dict[str, Any], qtype: str, kind: str, text: Dict[str, Any], images: List[str],
            audit: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    rec = dict(base)
    rec.update({"qtype": qtype, "qclass": Q.QCLASS[qtype.rsplit("_", 1)[-1]], "kind": kind, "images": images,
                "question": text["question"], "options": list(text["options"]), "answer": text["answer"],
                "answer_value": str(text["answer_value"]), "action": "", "k": 0, "changed": None, "flip": None,
                "state_id_2": "", "audit": json.dumps(G.jsonable(audit), sort_keys=True)})
    rec.update(extra)
    rec["sample_id"] = "p1_" + format(Q.stable_int(rec["task_id"], rec["state_id"], qtype, rec["action"], rec["k"]), "012x")
    return rec


def scene_candidates(scene_id: str, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from data_gen.interiorgs_v10.occlusion import face_occluders_for
    from data_gen.interiorgs_v10.scene import load_scene

    trajs = _INPUTS["by_scene"][scene_id]
    rows, poses = _INPUTS["rows"], _INPUTS["poses"]
    scene = load_scene(Path(cfg["scenes_root"]) / scene_id, scene_id, wall_model=cfg["wall_model"])
    occ = face_occluders_for(scene)
    out: List[Dict[str, Any]] = []
    funnel: collections.Counter = collections.Counter()
    checks: collections.Counter = collections.Counter()
    for traj in trajs:
        item = rows[traj["source_task_id"]]
        geo = G.TaskGeometry(item, scene, occ)
        images, turns = trajectory_steps(traj, poses)
        kind = Q.FAMILY_QTYPE[geo.family]
        # consistency: the terminal frame succeeds, no earlier frame does (the env would have ended there)
        cache: Dict[int, Dict[str, Any]] = {}
        for i, im in enumerate(images):
            if i and im["frame_id"] == images[i - 1]["frame_id"]:
                continue
            cache[i] = geo.measure(im["c2w"])
            terminal = i == len(images) - 1
            if cache[i]["success"] != terminal:
                checks["terminal_not_success" if terminal else "success_before_terminal"] += 1
        checks["trajectories"] += 1
        base = {"pool": "P1", "family": geo.family, "task_id": item["task_id"], "scene": scene_id,
                "split": traj["split"]}
        for pos, role in pick_anchor_positions(images, item["task_id"], int(cfg["states_per_trajectory"])):
            im = images[pos]
            m = cache.get(pos) or geo.measure(im["c2w"])
            sbase = dict(base, state_id=im["state_id"], state_pos=pos, state_role=role)
            funnel["anchor_states"] += 1
            # ---------------- (a) state questions
            ok, robust = geo.success_robust(im["c2w"], cfg["goal_perturb_m"], cfg["goal_perturb_deg"])
            if robust:
                text = Q.build_text("goal", item, geo.roles, ok, None, f"{item['task_id']}:{pos}:goal", cfg)
                out.append(_record(sbase, "goal_now", "goal", text, [im["image"]],
                                   {"potential": m["potential"], "failed_gates": m["failed_gates"]}))
                funnel["goal_now:emitted"] += 1
            else:
                funnel[f"goal_now:near_boundary_{'yes' if ok else 'no'}"] += 1
            r = Q.READERS[kind](m, cfg, True)
            if r.ok:
                text = Q.build_text(kind, item, geo.roles, r.value, None, f"{item['task_id']}:{pos}:{kind}", cfg)
                out.append(_record(sbase, f"{kind}_now", kind, text, [im["image"]], {"quantity": r.quantity}))
                funnel[f"{kind}_now:emitted"] += 1
            else:
                funnel[f"{kind}_now:{r.reason}"] += 1
            # ---------------- (b) forward questions
            out.extend(forward_candidates(geo, item, kind, im, m, sbase, cfg, funnel))
            # ---------------- (c) inverse question: the turn that starts at this anchor
            if pos < len(turns):
                rec = inverse_candidate(images, turns, pos, sbase, item, cfg, funnel)
                if rec is not None:
                    out.append(rec)
    return out, {"funnel": dict(funnel), "checks": dict(checks)}


def forward_candidates(geo: G.TaskGeometry, item: Dict[str, Any], kind: str, im: Dict[str, Any],
                       m: Dict[str, Any], sbase: Dict[str, Any], cfg: Dict[str, Any],
                       funnel: collections.Counter) -> List[Dict[str, Any]]:
    rng = random.Random(Q.stable_int(item["task_id"], im["state_id"], "forward"))
    pre_kind = Q.READERS[kind](m, cfg, True)
    goal_cands, kind_cands = [], []
    for action in ACTIONS:
        for k in cfg["forward_k"]:
            poses = geo.action_poses(im["c2w"], action, k)
            if poses is None:
                funnel["forward:blocked_move"] += 1
                continue
            mp = geo.measure(poses[-1])
            goal_cands.append((action, k, poses[-1], mp))
            if not pre_visible(kind, m):
                funnel[f"{kind}_after:not_visible_now"] += 1
                continue
            r = Q.READERS[kind](mp, cfg, False)
            if not r.ok:
                funnel[f"{kind}_after:{r.reason}"] += 1
                continue
            ch = Q.changed(kind, pre_kind, r, m, mp, cfg)
            flip = (pre_kind.value != r.value) if pre_kind.ok else None
            kind_cands.append((action, k, r, ch, flip))
    out = []
    # goal_after: every robust "yes", plus a few robust "no"
    yes = [c for c in goal_cands if c[3]["success"]]
    no = [c for c in goal_cands if not c[3]["success"]]
    rng.shuffle(no)
    checked_no = 0
    pre_goal = Q.Reading(bool(m["success"]), None)
    for action, k, pose, mp in yes + no:
        if not mp["success"] and checked_no >= int(cfg["goal_after_no_checked"]):
            break
        if not mp["success"]:
            checked_no += 1
        ok, robust = geo.success_robust(pose, cfg["goal_perturb_m"], cfg["goal_perturb_deg"])
        if not robust:
            funnel[f"goal_after:near_boundary_{'yes' if ok else 'no'}"] += 1
            continue
        post = Q.Reading(ok, None)
        text = Q.build_text("goal", item, geo.roles, ok, (action, k), f"{item['task_id']}:{im['state_id']}:goal:{action}:{k}", cfg)
        out.append(_record(sbase, "goal_after", "goal", text, [im["image"]],
                           {"potential_before": m["potential"], "potential_after": mp["potential"],
                            "failed_gates_after": mp["failed_gates"]},
                           action=action, k=int(k), changed=Q.changed("goal", pre_goal, post, m, mp, cfg),
                           flip=bool(ok) != bool(m["success"])))
        funnel["goal_after:emitted"] += 1
    # family question after the action: up to N, spread over (answer, changed) groups
    groups: Dict[Tuple[Any, bool], List[Any]] = collections.defaultdict(list)
    for c in kind_cands:
        groups[(c[2].value, c[3])].append(c)
    keys = list(groups)
    rng.shuffle(keys)
    for key in keys[: int(cfg["forward_per_qtype"])]:
        action, k, r, ch, flip = rng.choice(groups[key])
        text = Q.build_text(kind, item, geo.roles, r.value, (action, k),
                            f"{item['task_id']}:{im['state_id']}:{kind}:{action}:{k}", cfg)
        out.append(_record(sbase, f"{kind}_after", kind, text, [im["image"]],
                           {"quantity_before": pre_kind.quantity, "quantity_after": r.quantity},
                           action=action, k=int(k), changed=bool(ch), flip=flip))
        funnel[f"{kind}_after:emitted"] += 1
    return out


def inverse_candidate(images: Sequence[Dict[str, Any]], turns: Sequence[Dict[str, Any]], pos: int,
                      sbase: Dict[str, Any], item: Dict[str, Any], cfg: Dict[str, Any],
                      funnel: collections.Counter) -> Optional[Dict[str, Any]]:
    turn = turns[pos]
    acts = turn["actions"]
    if turn["blocked"] or images[pos]["frame_id"] == images[pos + 1]["frame_id"]:
        funnel["action_between:blocked_turn"] += 1
        return None
    if not acts or len(set(acts)) != 1:
        funnel["action_between:mixed_actions"] += 1
        return None
    action, k = acts[0], len(acts)
    expect = G.repeated_action_pose(images[pos]["c2w"], action, k)
    if not G.poses_match(expect, images[pos + 1]["c2w"], tol=1e-3):
        funnel["action_between:pose_mismatch"] += 1
        return None
    text = Q.build_inverse_text(action, k, f"{item['task_id']}:{images[pos]['state_id']}:between", cfg)
    funnel["action_between:emitted"] += 1
    return _record(sbase, "action_between", "action", text, [images[pos]["image"], images[pos + 1]["image"]],
                   {}, action=action, k=int(k), state_id_2=images[pos + 1]["state_id"])


def _scene_job(args: Tuple[str, Dict[str, Any]]) -> Tuple[str, str]:
    scene_id, cfg = args
    raw = Path(cfg["out"]) / "raw"
    stem = f"{scene_id}__{cfg['tag']}" if cfg.get("tag") else scene_id  # a tagged source next to the main one
    done = raw / f"{stem}.json"
    if done.exists():
        return scene_id, "skipped"
    t0 = time.time()
    try:
        recs, info = scene_candidates(scene_id, cfg)
    except G.ConsistencyError as exc:
        (raw / f"{stem}.consistency_error.txt").write_text(str(exc))
        return scene_id, f"consistency_error: {exc}"
    except Exception:
        (raw / f"{stem}.error.txt").write_text(traceback.format_exc())
        return scene_id, "error"
    write_jsonl(raw / f"{stem}.jsonl", recs)
    info.update(scene=scene_id, records=len(recs), seconds=round(time.time() - t0, 1))
    done.write_text(json.dumps(info, indent=1))
    return scene_id, f"{len(recs)} records, {info['seconds']} s"


def cmd_raw(args: argparse.Namespace) -> None:
    sft = Path(args.sft_dir)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfg = dict(Q.DEFAULT_CONFIG)
    cfg.update(scenes_root=str(args.scenes_root), wall_model=args.wall_model, out=str(out), tag=args.tag)
    if args.states_per_trajectory is not None:
        cfg["states_per_trajectory"] = int(args.states_per_trajectory)
    manifest = json.loads((sft / "split_manifest.json").read_text())
    split_of = {s: k for k in ("train", "val", "test") for s in manifest[f"{k}_scenes"]}
    _INPUTS["rows"] = {r["task_id"]: r for r in read_jsonl(sft / "rows.jsonl")}
    _INPUTS["poses"] = {f["frame_id"]: f["c2w"] for f in read_jsonl(sft / "frames.jsonl")}
    by_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for t in read_jsonl(sft / "sft_trajectories.jsonl"):
        if t.get("reserve"):
            continue
        if split_of.get(t["scene_id"]) != t["split"] or t["split"] not in ("train", "val"):
            raise SystemExit(f"{t['id']}: split {t['split']} but scene {t['scene_id']} is "
                             f"{split_of.get(t['scene_id'])} in the manifest")
        by_scene[t["scene_id"]].append(t)
    _INPUTS["by_scene"] = by_scene
    scenes = sorted(by_scene)
    if args.scenes:
        scenes = [s for s in args.scenes.split(",") if s]
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    name = f"config__{args.tag}.json" if args.tag else "config.json"
    (out / name).write_text(json.dumps(dict(cfg, sft_dir=str(sft), scenes=len(scenes)), indent=1, default=list))
    jobs = [(s, cfg) for s in scenes]
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, (sid, msg) in enumerate(pool.imap_unordered(_scene_job, jobs, chunksize=1), 1):
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {sid}: {msg}", flush=True)
                if msg.startswith("consistency_error"):
                    pool.terminate()
                    raise SystemExit(f"stopped: {sid} {msg}")
    else:
        for i, job in enumerate(jobs, 1):
            sid, msg = _scene_job(job)
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {sid}: {msg}", flush=True)
            if msg.startswith("consistency_error"):
                raise SystemExit(f"stopped: {sid} {msg}")


# =============================================================================
# build: balance, ratio, caps, split, export
# =============================================================================
BUILD_CONFIG: Dict[str, Any] = {
    # class rules of 2026-10-19 (user): every state question is kept (after the answer balance); forward questions
    # at most 1.2 x the state questions; inverse questions at most 15% of the bank; the scene cap stays 8%.
    # (2026-10-18: shares 50 / 35 / 15 and goal questions at most half of the state and of the forward class.)
    "forward_max_vs_state": 1.2,
    "inverse_max_share": 0.15,
    "goal_max_share_in_class": {"forward": 0.5},  # goal_after <= half the forward class; goal_now is a state question
    "balance_slack": 1.15,          # each answer group at most 1.15 x the smallest one (<= +3.5 points for 2 options)
    "rare_value_share": 0.05,       # categorical values below this share are kept whole and not balanced against
    # the answer balance within each stem cell (family, action, count; 2026-10-20): both_in_view has four answers
    # over 18 action cells, and 1.15 there left 94 of the 253 test questions of both_in_view_after (1.5: 112); 2.0
    # keeps 181, with the family-and-action rule at 26.5% against 25% (user's rule: no type below half)
    "cell_slack": {"both_in_view": 2.0},
    # occlusion labels checked against the rendered depth (user, 2026-10-21; occ_depth.py, PROGRESS 12.11): a "fully" /
    # "partly hidden" candidate stays only if the target is hidden in the render by at least this share of its
    # silhouette (any surface in front); OUT/<results> from occ_depth run.  None: no check.
    "occlusion_depth": {"results": "occ_depth_all/results.jsonl", "full_min": 0.70, "partial_min": 0.10},
    "unchanged_max_share": 0.10,    # forward questions whose asked quantity did not change
    "flip_band": (0.35, 0.65),      # share of forward questions whose answer differs from the current one
    "scene_cap": 0.08,
    "final_letter_slack": 1.10,      # after sampling, per file: <= +2.4 points for 2 options, +1.8 for 4
}
PROBE_CONFIG: Dict[str, Any] = {
    "per_qtype": 40,                # user, 2026-10-19: at least 40 per question type
    # the four types of the pre-registration (PREREG_D_vs_C.md), enlarged (user, 2026-10-19): "all" = every val
    # candidate that passed the filters (the answers then keep their natural shares)
    "per_qtype_overrides": {"goal_now": 200, "goal_after": 200, "occluded_now": "all", "occluded_after": "all"},
    "unchanged_max_share": 0.10,
}
ORDINAL = {"distance_bin", "height_bin"}         # options in ascending order: balance the letter
SHUFFLED = {"occluded", "both_in_view"}          # categorical: option order drawn per question


def _cap_groups(records: List[Dict[str, Any]], key, slack: float, rng: random.Random,
                rare_share: float = 0.0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    groups: Dict[Any, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    if len(groups) <= 1:
        return records, {}
    total = len(records)
    regular = {g: rs for g, rs in groups.items() if len(rs) >= rare_share * total}
    if not regular:
        return records, {}
    cap = max(1, int(math.floor(slack * min(len(rs) for rs in regular.values()))))
    kept, dropped = [], {}
    for g, rs in groups.items():
        if g in regular and len(rs) > cap:
            rs = list(rs)
            rng.shuffle(rs)
            dropped[str(g)] = len(rs) - cap
            rs = rs[:cap]
        kept.extend(rs)
    return kept, dropped


def stem_cell(r: Dict[str, Any]) -> Tuple[Any, ...]:
    """What the stem says besides the objects: the task family (its task sentence) and the action with its count."""
    return (r["family"], r.get("action") or None, r.get("k") or None)


def _balance_cells(rs: List[Dict[str, Any]], key, slack: float, rng: random.Random,
                   rare_share: float = 0.0) -> Tuple[List[Dict[str, Any]], int]:
    """The answer balance within each stem cell (2026-10-20; before, only over the whole type, and the family and
    the action alone answered goal_after 72%, goal_now 65% and both_in_view_after 39% of the test questions):
    every answer at most ``slack`` x the rarest regular answer of the cell; a cell lacking a regular answer is left
    out.  Regular: at least ``rare_share`` of the type (both_in_view "neither" is rare)."""
    total = collections.Counter(key(r) for r in rs)
    regular = sorted(v for v, c in total.items() if c >= rare_share * len(rs))
    if len(regular) <= 1:
        return rs, 0
    cells: Dict[Any, Dict[Any, List[Dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs:
        cells[stem_cell(r)][key(r)].append(r)
    kept, dropped = [], 0
    for _, groups in sorted(cells.items(), key=lambda kv: str(kv[0])):
        if any(not groups.get(v) for v in regular):
            dropped += sum(len(v) for v in groups.values())
            continue
        cap = int(math.floor(slack * min(len(groups[v]) for v in regular)))
        for v, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
            g = sorted(g, key=lambda r: r["sample_id"])
            rng.shuffle(g)
            kept.extend(g[:cap])
            dropped += max(0, len(g) - cap)
    return kept, dropped


def shuffle_options(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Categorical questions: draw the option order per question (the answer stays the same option)."""
    opts = rec["options"]
    correct = opts[Q.LETTERS.index(rec["answer"])]
    order = list(range(len(opts)))
    random.Random(Q.stable_int(rec["sample_id"], "shuffle")).shuffle(order)
    new = [opts[i] for i in order]
    head = rec["question"].rsplit("\n(A) ", 1)[0]
    rec = dict(rec, options=new, answer=Q.LETTERS[new.index(correct)],
               question=f"{head}\n{Q.options_block(new)}\n{Q.ANSWER_LETTER}")
    return rec


def _cap_unchanged(rs: List[Dict[str, Any]], share: float, rng: random.Random) -> Tuple[List[Dict[str, Any]], int]:
    unchanged = [r for r in rs if not r["changed"]]
    changed_ = [r for r in rs if r["changed"]]
    max_un = int(math.floor(share / (1 - share) * len(changed_)))
    if len(unchanged) <= max_un:
        return rs, 0
    rng.shuffle(unchanged)
    keep = {id(r) for r in unchanged[:max_un]}
    return [r for r in rs if r["changed"] or id(r) in keep], len(unchanged) - max_un


def _cap_flips(rs: List[Dict[str, Any]], band: Tuple[float, float], rng: random.Random,
               min_side: float = 0.10) -> Tuple[List[Dict[str, Any]], int]:
    """Keep the share of answers that differ from the current answer inside ``band`` (no 'flip it' shortcut).
    Only where both outcomes really occur (each at least ``min_side`` of the candidates): a turn never swaps
    the left / right order of two objects in front of the camera, so left_right_after almost never flips and
    is left as it is (its share is reported)."""
    known = [r for r in rs if r["flip"] is not None]
    if not known:
        return rs, 0
    flips = [r for r in known if r["flip"]]
    keeps = [r for r in known if not r["flip"]]
    if min(len(flips), len(keeps)) < min_side * len(known):
        return rs, 0
    lo, hi = band
    share = len(flips) / len(known)
    if share > hi:
        excess, n = flips, int(math.floor(hi / (1 - hi) * len(keeps)))
    elif share < lo:
        excess, n = keeps, int(math.floor((1 - lo) / lo * len(flips)))
    else:
        return rs, 0
    rng.shuffle(excess)
    drop = {id(r) for r in excess[n:]}
    return [r for r in rs if id(r) not in drop], len(drop)


def _balance_inverse(rs: List[Dict[str, Any]], slack: float, rng: random.Random) -> Tuple[List[Dict[str, Any]], int]:
    """Inverse questions (2026-10-19): within each cell (the pair of actions shown, the pair of counts shown) the four
    possible answers (either action x the smaller or the larger count) at most ``slack`` x the rarest; a cell where one
    of them never occurs is left out.  Then neither the option texts nor their priors tell the answer
    (shortcuts.py: the learned rules fall to chance)."""
    cells: Dict[Any, Dict[Any, List[Dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs:
        acts, action, counts, low = Q.inverse_balance_keys(r)
        cells[(acts, counts)][(action, low)].append(r)
    kept, dropped = [], 0
    for (acts, _), groups in sorted(cells.items()):
        want = [(a, low) for a in acts for low in (True, False)]
        if len(acts) != 2 or any(not groups.get(w) for w in want):
            dropped += sum(len(v) for v in groups.values())
            continue
        cap = int(math.floor(slack * min(len(groups[w]) for w in want)))
        for w in want:
            v = sorted(groups[w], key=lambda r: r["sample_id"])
            rng.shuffle(v)
            kept.extend(v[:cap])
            dropped += max(0, len(v) - cap)
    return kept, dropped


def window_quota(n_by_bin: Dict[int, int], lo: int, hi: int) -> Dict[int, int]:
    """Questions per letter for each option window (its first grid index s, options s..s+3): every letter of a window
    equally often and as many questions as possible, i.e. max sum t_s with, for every grid value b, the windows
    holding b taking together at most its n_b questions.  The constraint matrix is an interval matrix, so the
    optimum is whole."""
    from scipy.optimize import linprog

    wins = list(range(lo, hi - 2))
    bins = list(range(lo, hi + 1))  # every grid value of the windows, also the ones without questions (they cap at 0)
    if not wins or not n_by_bin:
        return {}
    a_ub = [[1.0 if s <= b <= s + 3 else 0.0 for s in wins] for b in bins]
    res = linprog([-1.0] * len(wins), A_ub=a_ub, b_ub=[n_by_bin.get(b, 0) for b in bins], bounds=[(0, None)] * len(wins),
                  method="highs")
    if not res.success:
        raise RuntimeError(f"window quota: {res.message}")
    return {s: int(math.floor(x + 1e-6)) for s, x in zip(wins, res.x) if x >= 1 - 1e-6}


def _assign_windows(rs: List[Dict[str, Any]], slack: float) -> List[Dict[str, Any]]:
    """Distance / screen-height questions (2026-10-20): choose each question's option window again so that every
    window has its four letters equally often (``window_quota``); the rest go where the slack leaves room (else to
    their first window, and _balance_windows drops them).  The raw files draw the window per question among the ones
    that fit, which near the smallest option leaves only A (and trimming those windows dropped 60% of the val
    distance_bin_now questions)."""
    if not rs:
        return rs
    kind = rs[0]["kind"]
    step, lo_v, hi_v = Q.grid_spec(kind, Q.DEFAULT_CONFIG)
    by_bin: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rs:
        b = int(round(float(r["answer_value"]) / step))
        if abs(b * step - float(r["answer_value"])) > 1e-6:
            raise ValueError(f"{r['sample_id']}: answer {r['answer_value']} is not on the {step} grid")
        by_bin[b].append(r)
    lo = int(round(lo_v / step))
    hi = int(round(hi_v / step)) if hi_v != math.inf else max(by_bin) + 3
    quota = window_quota({b: len(v) for b, v in by_bin.items()}, lo, hi)
    room = {s: int(math.floor(slack * t)) - t for s, t in quota.items()}
    used: Dict[Tuple[int, int], int] = collections.defaultdict(int)
    out = []

    def place(r: Dict[str, Any], s: int, b: int) -> Dict[str, Any]:
        opts = [Q.grid_text(kind, round((s + i) * step, 2)) for i in range(4)]
        head = r["question"].rsplit("\n(A) ", 1)[0]
        return dict(r, options=opts, answer=Q.LETTERS[b - s], question=f"{head}\n{Q.options_block(opts)}\n{Q.ANSWER_LETTER}")

    for b, v in sorted(by_bin.items()):
        v = sorted(v, key=lambda r: Q.stable_int(r["sample_id"], "window"))
        fits = [s for s in range(b - 3, b + 1) if s >= lo and s + 3 <= hi]
        i = 0
        for s in fits:
            take = quota.get(s, 0)
            out.extend(place(r, s, b) for r in v[i:i + take])
            i += take
        for r in v[i:]:
            spare = [s for s in fits if used[(s, b - s)] < room.get(s, 0)]
            s = max(spare, key=lambda s: room[s] - used[(s, b - s)]) if spare else \
                max(fits, key=lambda s: quota.get(s, 0)) if fits else None
            if s is None:  # no window of four fits (never on the current grids)
                out.append(r)
                continue
            used[(s, b - s)] += 1
            out.append(place(r, s, b))
    return out


def _balance_windows(rs: List[Dict[str, Any]], slack: float, rng: random.Random) -> Tuple[List[Dict[str, Any]], int]:
    """Distance / screen-height questions (2026-10-19): within each option window (its smallest option) every letter
    at most ``slack`` x the rarest; a window where a letter never occurs is left out, so the window does not tell
    the letter (before: the smallest option alone guessed 34%-44% against 25%).  _assign_windows comes first, so
    this only trims what the window quota could not place."""
    windows: Dict[str, Dict[str, List[Dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs:
        windows[r["options"][0]][r["answer"]].append(r)
    kept, dropped = [], 0
    for w, groups in sorted(windows.items()):
        letters = Q.LETTERS[: len(next(iter(groups.values()))[0]["options"])]
        if any(not groups.get(x) for x in letters):
            dropped += sum(len(v) for v in groups.values())
            continue
        cap = int(math.floor(slack * min(len(groups[x]) for x in letters)))
        for x in letters:
            v = sorted(groups[x], key=lambda r: r["sample_id"])
            rng.shuffle(v)
            kept.extend(v[:cap])
            dropped += max(0, len(v) - cap)
    return kept, dropped


def balance_split(records: List[Dict[str, Any]], cfg: Dict[str, Any], rng: random.Random,
                  report: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_qtype: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_qtype[r["qtype"]].append(r)
    kept: List[Dict[str, Any]] = []
    for qtype, rs in sorted(by_qtype.items()):
        kind = rs[0]["kind"]
        forward = rs[0]["qclass"] == "forward"
        rep = report.setdefault(qtype, {"before": len(rs)})
        if kind in SHUFFLED:
            rs = [shuffle_options(r) for r in rs]
        dropped = collections.Counter()
        # the answer balance and the forward caps interact (e.g. goal_after: "yes" is mostly a flip from "no"),
        # so apply them in turn until none drops anything; the unchanged cap is the hard one and comes last
        for _ in range(8):
            n0 = len(rs)
            if forward:
                rs, n = _cap_flips(rs, cfg["flip_band"], rng)
                dropped["flip_band"] += n
                rs, n = _cap_unchanged(rs, cfg["unchanged_max_share"], rng)
                dropped["unchanged"] += n
            if kind in SHUFFLED:  # the answer within each stem cell (family, action, count)
                rs, n = _balance_cells(rs, lambda r: r["answer_value"], (cfg.get("cell_slack") or {}).get(kind, cfg["balance_slack"]),
                                       rng, cfg["rare_value_share"])
                dropped["answer_balance"] += n
            elif kind in ORDINAL:  # the letter within each option window (windows chosen again after each cap)
                rs = _assign_windows(rs, cfg["balance_slack"])
                rs, n = _balance_windows(rs, cfg["balance_slack"], rng)
                dropped["answer_balance"] += n
            elif kind in ("goal", "left_right"):  # the answer within each stem cell
                rs, n = _balance_cells(rs, lambda r: r["answer"], cfg["balance_slack"], rng)
                dropped["answer_balance"] += n
            elif kind == "action":  # the answer within each (action pair, count pair) cell
                rs, n = _balance_inverse(rs, cfg["balance_slack"], rng)
                dropped["answer_balance"] += n
            if len(rs) == n0:
                break
        if forward:
            rs, n = _cap_unchanged(rs, cfg["unchanged_max_share"], rng)
            dropped["unchanged"] += n
        rep["dropped"] = dict(dropped)
        rep["after_balance"] = len(rs)
        kept.extend(rs)
    return kept


def _stratified(rs: List[Dict[str, Any]], want: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Exactly ``want`` records of ``rs``, the same fraction from every (qtype, answer, changed) stratum (largest
    remainders get the leftover places, so a cap is never exceeded)."""
    if want >= len(rs):
        return list(rs)
    strata: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rs:
        strata[(r["qtype"], r["answer"], str(r["changed"]))].append(r)
    frac = want / len(rs)
    keys = sorted(strata)
    take = {k: int(math.floor(frac * len(strata[k]))) for k in keys}
    rest = sorted(keys, key=lambda k: (-(frac * len(strata[k]) - take[k]), k))
    for k in rest[: want - sum(take.values())]:
        take[k] += 1
    out = []
    for key in keys:
        group = strata[key]
        rng.shuffle(group)
        out.extend(group[: take[key]])
    return out


def class_caps(records: List[Dict[str, Any]], cfg: Dict[str, Any], rng: random.Random,
               report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The class rules: the state questions are all kept; the goal questions of a class named in
    ``goal_max_share_in_class`` are capped at that share of it; the forward questions at
    ``forward_max_vs_state`` x the state questions; the inverse questions at ``inverse_max_share`` of the result.
    Every cut is stratified by (qtype, answer, changed), so the balance survives."""
    by_class: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_class[r["qclass"]].append(r)
    if not records:
        return []
    avail = {c: len(rs) for c, rs in by_class.items()}
    for c, share in (cfg.get("goal_max_share_in_class") or {}).items():
        goal = [r for r in by_class[c] if r["kind"] == "goal"]
        other = [r for r in by_class[c] if r["kind"] != "goal"]
        cap = int(share / (1 - share) * len(other)) if share < 1 else len(goal)
        if len(goal) > cap:
            report.setdefault("goal_cap_dropped", {})[c] = len(goal) - cap
            by_class[c] = other + _stratified(goal, cap, rng)
    n_state = len(by_class["state"])
    fwd_cap = int(math.floor(float(cfg["forward_max_vs_state"]) * n_state))
    if len(by_class["forward"]) > fwd_cap:
        by_class["forward"] = _stratified(by_class["forward"], fwd_cap, rng)
    s = float(cfg["inverse_max_share"])
    inv_cap = int(math.floor(s / (1 - s) * (n_state + len(by_class["forward"]))))
    if len(by_class["inverse"]) > inv_cap:
        by_class["inverse"] = _stratified(by_class["inverse"], inv_cap, rng)
    out = []
    for c in ("state", "forward", "inverse"):
        report[c] = {"available": avail.get(c, 0), "kept": len(by_class[c])}
        out.extend(by_class[c])
    report["caps"] = {"forward": fwd_cap, "inverse": inv_cap}
    return out


def scene_cap(records: List[Dict[str, Any]], cap: float, rng: random.Random) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    limit = int(math.floor(cap * len(records)))
    by_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_scene[r["scene"]].append(r)
    out, dropped = [], {}
    for s, rs in by_scene.items():
        if len(rs) > limit:
            rng.shuffle(rs)
            dropped[s] = len(rs) - limit
            rs = rs[:limit]
        out.extend(rs)
    return out, dropped


REORDERABLE = {"closer", "action", "occluded", "both_in_view"}   # option order carries no meaning


def place_correct(rec: Dict[str, Any], pos: int) -> Dict[str, Any]:
    """The same question with its correct option moved to position ``pos`` (the others keep their order)."""
    opts = list(rec["options"])
    correct = opts.pop(Q.LETTERS.index(rec["answer"]))
    opts.insert(pos, correct)
    head = rec["question"].rsplit("\n(A) ", 1)[0]
    return dict(rec, options=opts, answer=Q.LETTERS[pos],
                question=f"{head}\n{Q.options_block(opts)}\n{Q.ANSWER_LETTER}")


def even_letters(records: List[Dict[str, Any]], cfg: Dict[str, Any], rng: random.Random, cap: bool = True,
                 report: Optional[Dict[str, Any]] = None, place: bool = True) -> List[Dict[str, Any]]:
    """Letter balance of one file: the correct option of a reorderable question goes to the positions in turn
    (by sample id; ``place``); the other questions' letters are capped at ``final_letter_slack`` x the rarest
    (``cap``)."""
    by_q: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_q[r["qtype"]].append(r)
    out = []
    for q, rs in sorted(by_q.items()):
        rs = sorted(rs, key=lambda r: r["sample_id"])
        if rs[0]["kind"] in REORDERABLE:
            if not place:
                out.extend(rs)
                continue
            start = Q.stable_int(q, "letters") % len(rs[0]["options"])
            out.extend(place_correct(r, (start + i) % len(r["options"])) for i, r in enumerate(rs))
        elif cap:
            kept, dropped = _cap_groups(rs, lambda r: r["answer"], cfg["final_letter_slack"], rng)
            if report is not None and dropped:
                report[q] = dropped
            out.extend(kept)
        else:
            out.extend(rs)
    return sorted(out, key=lambda r: r["sample_id"])


def to_lf(rec: Dict[str, Any]) -> Dict[str, Any]:
    if len(rec["images"]) == 2:
        head = "First image: <image>\nSecond image: <image>\n"
    else:
        head = "<image>\n"
    row = {"messages": [{"role": "user", "content": head + rec["question"]},
                        {"role": "assistant", "content": rec["answer"]}],
           "images": list(rec["images"])}
    for key in ("sample_id", "pool", "qclass", "qtype", "family", "task_id", "scene", "split", "state_id",
                "state_role", "state_id_2", "answer", "answer_value", "action", "k", "changed", "flip", "audit"):
        row[key] = rec.get(key)
    row["state_index"] = int(rec["state_pos"])
    row["options"] = list(rec["options"])
    return row


def _spread(rs: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    """``rs`` reordered so that families, and tasks within a family, take turns (a probe draws from the front)."""
    by_fam: Dict[str, Dict[str, List[Dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs:
        by_fam[r["family"]][r["task_id"]].append(r)
    fam_queues = []
    for fam in sorted(by_fam):
        tasks = [list(v) for _, v in sorted(by_fam[fam].items())]
        rng.shuffle(tasks)
        for t in tasks:
            rng.shuffle(t)
        order = []
        while any(tasks):
            for t in tasks:
                if t:
                    order.append(t.pop())
        fam_queues.append(order)
    out = []
    while any(fam_queues):
        for q in fam_queues:
            if q:
                out.append(q.pop(0))
    return out


def probe_select(records: List[Dict[str, Any]], cfg: Dict[str, Any], rng: random.Random,
                 report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The fixed probe set: ``per_qtype`` questions of every question type (``per_qtype_overrides``: another number,
    or "all" candidates of that type).  Within a type the answers take turns
    (the answer value for the reorderable types, whose letters ``even_letters`` then sets; the letter for the
    others), for forward questions crossed with "the answer differs from the current one"; families and tasks take
    turns within each answer; forward questions whose asked quantity did not change at most
    ``unchanged_max_share``."""
    by_q: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(records, key=lambda r: r["sample_id"]):
        by_q[r["qtype"]].append(r)
    picks: List[Dict[str, Any]] = []
    for q, rs in sorted(by_q.items()):
        kind, forward = rs[0]["kind"], rs[0]["qclass"] == "forward"
        size = (cfg.get("per_qtype_overrides") or {}).get(q, cfg["per_qtype"])
        per_q = len(rs) if size == "all" else int(size)

        def key(r: Dict[str, Any]) -> Tuple[str, ...]:
            k = str(r["answer_value"] if kind in REORDERABLE else r["answer"])
            return (k, str(bool(r["flip"]))) if forward and r["flip"] is not None else (k,)

        buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rs:
            buckets[key(r)].append(r)
        queues = {k: _spread(v, rng) for k, v in sorted(buckets.items())}
        share = float(cfg["unchanged_max_share"])
        if not forward:
            max_unchanged = per_q
        elif size == "all":  # every changed one, the unchanged ones up to the share of the result
            max_unchanged = int(math.floor(share / (1 - share) * sum(1 for r in rs if r["changed"])))
        else:
            max_unchanged = int(math.floor(share * per_q))
        chosen: List[Dict[str, Any]] = []
        used_tasks: set = set()
        n_unchanged = 0
        turn = sorted(queues, key=lambda k: (len(buckets[k]), k))  # the scarcest answer picks its tasks first
        while len(chosen) < per_q and any(queues.values()):
            for k in turn:
                if len(chosen) >= per_q:
                    break
                queue = queues[k]
                if forward and n_unchanged >= max_unchanged:
                    queue[:] = [r for r in queue if r["changed"]]
                if not queue:
                    continue
                # a task not asked yet in this type first (answers of one task in different buckets)
                r = queue.pop(next((i for i, r in enumerate(queue) if r["task_id"] not in used_tasks), 0))
                n_unchanged += int(forward and not r["changed"])
                chosen.append(r)
                used_tasks.add(r["task_id"])
        report[q] = {"available": len(rs), "picked": len(chosen), "short": max(0, per_q - len(chosen)),
                     "by_key": {"/".join(k): len(v) for k, v in sorted(buckets.items())},
                     "picked_by_key": dict(collections.Counter("/".join(key(r)) for r in chosen)),
                     "families": dict(collections.Counter(r["family"] for r in chosen)),
                     "tasks": len({r["task_id"] for r in chosen})}
        picks.extend(chosen)
    return picks


def _read_raw(out: Path) -> Tuple[List[Dict[str, Any]], collections.Counter, collections.Counter]:
    records: List[Dict[str, Any]] = []
    funnel: collections.Counter = collections.Counter()
    checks: collections.Counter = collections.Counter()
    for info_path in sorted((out / "raw").glob("*.json")):
        info = json.loads(info_path.read_text())
        funnel.update(info["funnel"])
        checks.update(info["checks"])
        records.extend(read_jsonl(info_path.with_suffix(".jsonl")))
    return records, funnel, checks


def depth_check_occlusion(records: List[Dict[str, Any]], out: Path, cfg: Dict[str, Any],
                          report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Drop the occlusion candidates whose "fully" / "partly hidden" label the rendered depth does not bear out (user,
    2026-10-21): the target must be hidden in the render (occ_depth.py, depth_any) by at least full_min / partial_min
    of its silhouette.  A checked label without a depth result is dropped too; "not hidden" labels stay as they are
    (the box can only claim too much: PROGRESS 12.11)."""
    spec = cfg.get("occlusion_depth")
    if not spec:
        return records
    need = {"full": float(spec["full_min"]), "partial": float(spec["partial_min"])}
    if not any(r["kind"] == "occluded" and r["answer_value"] in need for r in records):
        return records
    path = out / spec["results"]
    if not path.exists():
        raise SystemExit(f"{path} missing: run occ_depth (prepare, job) first, or set occlusion_depth to None")
    res = {r["sample_id"]: r for r in read_jsonl(path) if "error" not in r}
    counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    kept = []
    for r in records:
        if r["kind"] == "occluded" and r["answer_value"] in need:
            key = f"{r['qtype']}/{r['answer_value']}"
            counts[key]["checked"] += 1
            x = res.get(r["sample_id"])
            if x is None:
                counts[key]["no_depth"] += 1
                continue
            if float(x["depth_any"]) < need[r["answer_value"]]:
                counts[key]["dropped"] += 1
                continue
        kept.append(r)
    report.update({"results": str(path), "full_min": need["full"], "partial_min": need["partial"],
                   "by_label": {k: dict(v) for k, v in sorted(counts.items())}})
    return kept


def _check_scenes(records: List[Dict[str, Any]], manifest_path: str) -> Dict[str, List[Dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text())
    train_scenes, val_scenes = set(manifest["train_scenes"]), set(manifest["val_scenes"])
    splits = {"train": [r for r in records if r["split"] == "train"],
              "val": [r for r in records if r["split"] == "val"]}
    for name, rs in splits.items():
        allowed = train_scenes if name == "train" else val_scenes
        bad = [r for r in rs if r["scene"] not in allowed]
        if bad:
            raise SystemExit(f"{len(bad)} {name} records from scenes outside the {name} scenes, e.g. {bad[0]['scene']}")
    return splits


def cmd_probe(args: argparse.Namespace) -> None:
    out = Path(args.out)
    cfg = dict(PROBE_CONFIG, per_qtype=args.per_qtype)
    rng = random.Random(args.seed)
    records, _, _ = _read_raw(out)
    depth_report: Dict[str, Any] = {}
    records = depth_check_occlusion(records, out, BUILD_CONFIG, depth_report)
    val = _check_scenes(records, args.split_manifest)["val"]
    scenes = None
    if args.scenes_from:
        scenes = {json.loads(l)["scene_id"] for l in Path(args.scenes_from).read_text().splitlines() if l.strip()}
        val = [r for r in val if r["scene"] in scenes]
    # the same answer balance as the bank (option windows, inverse cells, yes / no, option order drawn per question),
    # except for the types taken whole; their categorical options are shuffled as build does (2026-10-19: the probe
    # step had left the template order, and the moved answer stood out)
    whole = {q for q, size in (cfg.get("per_qtype_overrides") or {}).items() if size == "all"}
    balanced = balance_split([r for r in val if r["qtype"] not in whole], BUILD_CONFIG, rng, {})
    taken = [shuffle_options(r) if r["kind"] in SHUFFLED else r for r in val if r["qtype"] in whole]
    report: Dict[str, Any] = {}
    picks = even_letters(probe_select(balanced + taken, cfg, rng, report), BUILD_CONFIG, rng, cap=False)
    n = write_jsonl(out / "p1_probe.jsonl", (to_lf(r) for r in picks))
    stats = {"questions": n, "config": cfg, "seed": args.seed, "candidates": len(val),
             "scenes": len({r["scene"] for r in picks}), "scenes_allowed": len(scenes) if scenes else None,
             "scenes_from": args.scenes_from, "by_qtype": report, "occlusion_depth": depth_report}
    (out / "probe_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({"questions": n, "short": {q: v["short"] for q, v in report.items() if v["short"]}}, indent=1))


def cmd_build(args: argparse.Namespace) -> None:
    out = Path(args.out)
    cfg = dict(BUILD_CONFIG)
    rng = random.Random(args.seed)
    records, funnel, checks = _read_raw(out)
    stats: Dict[str, Any] = {"raw_records": len(records), "funnel": dict(funnel), "checks": dict(checks),
                             "config": cfg, "seed": args.seed}
    records = depth_check_occlusion(records, out, cfg, stats.setdefault("occlusion_depth", {}))
    splits = _check_scenes(records, args.split_manifest)
    probe_path = out / "p1_probe.jsonl"
    if not probe_path.exists() and not args.no_probe:
        raise SystemExit(f"{probe_path} missing: build the probe set first (generate probe), or pass --no-probe")
    probe_ids = {r["sample_id"] for r in read_jsonl(probe_path)} if probe_path.exists() else set()
    splits["val"] = [r for r in splits["val"] if r["sample_id"] not in probe_ids]
    stats["probe_excluded"] = len(probe_ids)
    built: Dict[str, List[Dict[str, Any]]] = {}
    final = stats.setdefault("final_letter_cap_dropped", {})
    for name, rs in splits.items():
        rep: Dict[str, Any] = {}
        rs = balance_split(rs, cfg, rng, rep.setdefault("balance", {}))
        # the letter cap before the class caps, so the caps hold on the written file (the stratified cuts keep
        # the letter shares); the scene cap last, the class caps again if it cut anything
        rs = even_letters(rs, cfg, rng, place=False, report=final.setdefault("test" if name == "val" else name, {}))
        rep["scene_cap_dropped"] = {}
        for _ in range(3):
            rs = class_caps(rs, cfg, rng, rep.setdefault("classes", {}))
            rs, dropped = scene_cap(rs, cfg["scene_cap"], rng)
            for s, n in dropped.items():
                rep["scene_cap_dropped"][s] = rep["scene_cap_dropped"].get(s, 0) + n
            if not dropped:
                break
        stats[name] = rep
        built[name] = sorted(rs, key=lambda r: r["sample_id"])
    files = {"p1_train.jsonl": even_letters(built["train"], cfg, rng, cap=False),
             "p1_test.jsonl": even_letters(built["val"], cfg, rng, cap=False)}
    probe_scenes = {r["scene"] for r in read_jsonl(probe_path)} if probe_path.exists() else set()
    if {r["scene"] for r in built["train"]} & ({r["scene"] for r in built["val"]} | probe_scenes):
        raise SystemExit("a scene is in both the training and the test questions")
    for name, rs in files.items():
        stats.setdefault("files", {})[name] = write_jsonl(out / name, (to_lf(r) for r in rs))
    if probe_path.exists():
        stats["files"]["p1_probe.jsonl"] = len(probe_ids)
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: stats[k] for k in ("raw_records", "files", "checks")}, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("raw")
    r.add_argument("--sft-dir", required=True, help="the SFT folder: rows.jsonl, frames.jsonl, sft_trajectories.jsonl")
    r.add_argument("--scenes-root", required=True, help="InteriorGS metadata (labels.json, structure.json)")
    r.add_argument("--out", required=True)
    r.add_argument("--wall-model", default="thick")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--scenes", default="", help="comma-separated scene ids (default: all)")
    r.add_argument("--limit-scenes", type=int, default=0)
    r.add_argument("--tag", default="", help="name suffix of the raw files (a second source, e.g. extra val tasks)")
    r.add_argument("--states-per-trajectory", type=int, default=None, help="anchor states per trajectory (0: all)")
    r.set_defaults(func=cmd_raw)
    pr = sub.add_parser("probe")
    pr.add_argument("--out", required=True)
    pr.add_argument("--split-manifest", required=True)
    pr.add_argument("--scenes-from", default="", help="a tasks.jsonl: only its scenes (e.g. the round-12 eval_val_v2)")
    pr.add_argument("--per-qtype", type=int, default=PROBE_CONFIG["per_qtype"])
    pr.add_argument("--seed", type=int, default=20261019)
    pr.set_defaults(func=cmd_probe)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--split-manifest", required=True)
    b.add_argument("--seed", type=int, default=20261018)
    b.add_argument("--no-probe", action="store_true", help="build without a probe set (the test set keeps every val question)")
    b.set_defaults(func=cmd_build)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
