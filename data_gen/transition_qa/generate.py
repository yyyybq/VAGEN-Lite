"""Transition-question bank from the SFT trajectories (docs/transition_qa/PROGRESS.md, step 2).

    python -m data_gen.transition_qa.generate raw   --sft-dir SFT --scenes-root META --out OUT [--workers 12]
    python -m data_gen.transition_qa.generate build --out OUT [--seed 20261018]

``raw``: per scene (one process each), every anchor state of every trajectory -> candidate questions with the
reasons for the ones not asked (OUT/raw/<scene>.jsonl, OUT/raw/<scene>.json).  Finished scenes are skipped.
``build``: balance answers, forward-question caps, class ratio, scene cap, train / test / probe split ->
OUT/p1_train.jsonl, p1_test.jsonl, p1_probe.jsonl (LLaMA-Factory sharegpt rows plus metadata) and stats.json.
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
    """Positions in ``images`` of at most ``n`` distinct frames: the terminal frame, the start frame and random
    others.  A frame shown twice (a blocked turn) is taken at its last position, so its turn is the one that moved."""
    last_pos: Dict[str, int] = {}
    for i, im in enumerate(images):
        last_pos[im["frame_id"]] = i
    order = sorted(last_pos.values())
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
    done = raw / f"{scene_id}.json"
    if done.exists():
        return scene_id, "skipped"
    t0 = time.time()
    try:
        recs, info = scene_candidates(scene_id, cfg)
    except G.ConsistencyError as exc:
        (raw / f"{scene_id}.consistency_error.txt").write_text(str(exc))
        return scene_id, f"consistency_error: {exc}"
    except Exception:
        (raw / f"{scene_id}.error.txt").write_text(traceback.format_exc())
        return scene_id, "error"
    write_jsonl(raw / f"{scene_id}.jsonl", recs)
    info.update(scene=scene_id, records=len(recs), seconds=round(time.time() - t0, 1))
    done.write_text(json.dumps(info, indent=1))
    return scene_id, f"{len(recs)} records, {info['seconds']} s"


def cmd_raw(args: argparse.Namespace) -> None:
    sft = Path(args.sft_dir)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfg = dict(Q.DEFAULT_CONFIG)
    cfg.update(scenes_root=str(args.scenes_root), wall_model=args.wall_model, out=str(out))
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
    (out / "config.json").write_text(json.dumps(dict(cfg, sft_dir=str(sft), scenes=len(scenes)), indent=1, default=list))
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
    "ratio": {"state": 0.50, "forward": 0.35, "inverse": 0.15},
    "balance_slack": 1.15,          # each answer group at most 1.15 x the smallest one (<= +3.5 points for 2 options)
    "rare_value_share": 0.05,       # categorical values below this share are kept whole and not balanced against
    "unchanged_max_share": 0.10,    # forward questions whose asked quantity did not change
    "flip_band": (0.35, 0.65),      # share of forward questions whose answer differs from the current one
    "scene_cap": 0.08,
    "probe_size": 500,
    "goal_max_share_in_class": 0.5,  # "is the task met (after the action)?" at most half of the state / forward class
    "final_letter_slack": 1.10,      # after sampling, per file: <= +2.4 points for 2 options, +1.8 for 4
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
            if kind in SHUFFLED:
                rs, by = _cap_groups(rs, lambda r: r["answer_value"], cfg["balance_slack"], rng, cfg["rare_value_share"])
            elif kind in ORDINAL or kind in ("goal", "left_right"):
                rs, by = _cap_groups(rs, lambda r: r["answer"], cfg["balance_slack"], rng)
            else:
                by = {}
            dropped["answer_balance"] += sum(by.values())
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
    """``want`` records of ``rs``, the same fraction from every (qtype, answer, changed) stratum."""
    if want >= len(rs):
        return list(rs)
    strata: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rs:
        strata[(r["qtype"], r["answer"], str(r["changed"]))].append(r)
    frac = want / len(rs)
    out = []
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        out.extend(group[: int(round(frac * len(group)))])
    return out


def class_ratio(records: List[Dict[str, Any]], cfg: Dict[str, Any], rng: random.Random,
                report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Downsample the classes to the target shares, stratified by (qtype, answer) so the balance survives.
    Before that the goal questions are capped at ``goal_max_share_in_class`` of their class."""
    by_class: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_class[r["qclass"]].append(r)
    if not records:
        return []
    share = float(cfg.get("goal_max_share_in_class", 1.0))
    for c in ("state", "forward"):
        goal = [r for r in by_class[c] if r["kind"] == "goal"]
        other = [r for r in by_class[c] if r["kind"] != "goal"]
        cap = int(share / (1 - share) * len(other)) if share < 1 else len(goal)
        if len(goal) > cap:
            report.setdefault("goal_cap_dropped", {})[c] = len(goal) - cap
            by_class[c] = other + _stratified(goal, cap, rng)
    ratio = cfg["ratio"]
    present = [c for c in ratio if ratio[c] > 0 and by_class[c]]  # a missing class is reported, not a zero total
    total = min(len(by_class[c]) / ratio[c] for c in present)
    out = []
    for c, rs in by_class.items():
        want = int(round(total * ratio[c]))
        report[c] = {"available": len(rs), "kept": min(want, len(rs))}
        out.extend(_stratified(rs, want, rng))
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
                 report: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Final letter balance of one file: the correct option of a reorderable question goes to the positions in
    turn (by sample id); the other questions' letters are capped at ``final_letter_slack`` x the rarest."""
    by_q: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_q[r["qtype"]].append(r)
    out = []
    for q, rs in sorted(by_q.items()):
        rs = sorted(rs, key=lambda r: r["sample_id"])
        if rs[0]["kind"] in REORDERABLE:
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


def probe_subset(records: List[Dict[str, Any]], size: int, rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fixed probe set: equal share per qtype (as far as each has questions) and, within a qtype, the answer
    letters in turn; the rest is the test set."""
    by_q: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(records, key=lambda r: r["sample_id"]):
        by_q[r["qtype"]].append(r)
    queues: Dict[str, List[Dict[str, Any]]] = {}
    for q, rs in by_q.items():
        by_letter: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rs:
            by_letter[r["answer"]].append(r)
        for group in by_letter.values():
            rng.shuffle(group)
        order: List[Dict[str, Any]] = []
        while any(by_letter.values()):
            for letter in sorted(by_letter):
                if by_letter[letter]:
                    order.append(by_letter[letter].pop())
        queues[q] = order[::-1]  # pop() from the end takes them in letter turn
    probe: List[Dict[str, Any]] = []
    while len(probe) < size and any(queues.values()):
        for q in sorted(queues):
            if queues[q] and len(probe) < size:
                probe.append(queues[q].pop())
    ids = {r["sample_id"] for r in probe}
    return probe, [r for r in records if r["sample_id"] not in ids]


def cmd_build(args: argparse.Namespace) -> None:
    out = Path(args.out)
    cfg = dict(BUILD_CONFIG)
    rng = random.Random(args.seed)
    records: List[Dict[str, Any]] = []
    funnel: collections.Counter = collections.Counter()
    checks: collections.Counter = collections.Counter()
    for info_path in sorted((out / "raw").glob("*.json")):
        info = json.loads(info_path.read_text())
        funnel.update(info["funnel"])
        checks.update(info["checks"])
        records.extend(read_jsonl(info_path.with_suffix(".jsonl")))
    manifest = json.loads(Path(args.split_manifest).read_text())
    train_scenes, val_scenes = set(manifest["train_scenes"]), set(manifest["val_scenes"])
    stats: Dict[str, Any] = {"raw_records": len(records), "funnel": dict(funnel), "checks": dict(checks),
                             "config": cfg, "seed": args.seed}
    splits = {"train": [r for r in records if r["split"] == "train"],
              "val": [r for r in records if r["split"] == "val"]}
    for name, rs in splits.items():
        allowed = train_scenes if name == "train" else val_scenes
        bad = [r for r in rs if r["scene"] not in allowed]
        if bad:
            raise SystemExit(f"{len(bad)} {name} records from scenes outside the {name} scenes, e.g. {bad[0]['scene']}")
    built: Dict[str, List[Dict[str, Any]]] = {}
    for name, rs in splits.items():
        rep: Dict[str, Any] = {}
        rs = balance_split(rs, cfg, rng, rep.setdefault("balance", {}))
        rs = class_ratio(rs, cfg, rng, rep.setdefault("classes", {}))
        rs, rep["scene_cap_dropped"] = scene_cap(rs, cfg["scene_cap"], rng)
        stats[name] = rep
        built[name] = sorted(rs, key=lambda r: r["sample_id"])
    probe, test = probe_subset(built["val"], int(cfg["probe_size"]), rng)
    final = stats.setdefault("final_letter_cap_dropped", {})
    files = {"p1_train.jsonl": even_letters(built["train"], cfg, rng, report=final.setdefault("train", {})),
             "p1_test.jsonl": even_letters(test, cfg, rng, report=final.setdefault("test", {})),
             "p1_probe.jsonl": even_letters(probe, cfg, rng, cap=False)}
    if {r["scene"] for r in built["train"]} & ({r["scene"] for r in test} | {r["scene"] for r in probe}):
        raise SystemExit("a scene is in both the training and the test questions")
    for name, rs in files.items():
        stats.setdefault("files", {})[name] = write_jsonl(out / name, (to_lf(r) for r in rs))
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
    r.set_defaults(func=cmd_raw)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--split-manifest", required=True)
    b.add_argument("--seed", type=int, default=20261018)
    b.set_defaults(func=cmd_build)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
