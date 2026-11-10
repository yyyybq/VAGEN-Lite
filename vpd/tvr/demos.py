"""TVR demos from the THOR replay of the 1,600 public action sequences (G10(a), CONTRACT_wave2 5.1.7, 6.5).

Input: the run_tvr replay run dir (protocol.json, manifest.json, results.json, tasks/<id>/trajectory.json; the
replay used the frozen per-task reach sets) and the public sft_train.json.  For every public record:

keep iff every THOR step's pose equals the FREE-SPACE implied pose of the public sequence (this covers "every Move
succeeds" and "every Rotate/Look gives the implied pose"), the last action is Stop (and the only Stop), and the
pose before Stop is exactly the target.  Drop reason = the first failing check in step order:
  move_failed@k       action k is a Move and THOR reported action_success != True
  pose_mismatch@k     the THOR pose after action k differs from the implied pose (pose_mismatch@start: the reset
                      pose differs from task.start)
  stop_not_last       a Stop that is not the last action, or more than one Stop
  no_stop             the replay has no Stop (it ran into the cap)
  final_not_target    the pose before Stop is not exactly the target
Infrastructure failures are never drop reasons: build refuses (exit 2, nothing written) unless the run dir holds a
valid results.json (no results_invalid.json / results_partial.json) with a trajectory for every public task whose
actions are the public ones, whose per-turn lists are complete, whose PNGs exist (640x360, sha256 = the one the
trajectory records, which it must record) and whose user texts are build_concat_user_text(valid), and whose
vpd.reach_digest is the frozen one; the pinned public file must have its registered sha256, and every public task
must be in the reach index.

kin check (G0, NOT a drop reason): KinEpisode(task, R(task)) fed the same actions must give, at every step, the
THOR replay's pose, action_success, valid list and user text; kin_check.json lists every mismatch;
kin_equal_all_kept is true iff no kept task has one.

Demo record (6.5): {"id", "category", "dataset", "scene", "n_turns", "messages": [system, (user, assistant) per
turn], "images": [<abs images>/<id>/step_kk.png, <abs images>/<id>/target.png] per user turn}; user text k = the
replay's own obs_str k (never rebuilt), assistant k = "Action: <a_k>".  One json.dumps(record) + "\\n" per line.

Outputs: demos_all.jsonl (kept, sft.json order), dropped.json [{"task_id", "reason", "detail"}], kin_check.json,
images_sha256.json (every PNG of every replayed task: a FILE-INTEGRITY record only, never compared with another
render, F4), manifest.json, DONE.  An existing DONE dir with an identical demos_all.jsonl is a no-op (exit 0); a
different one is refused (exit 2).

CLI (PY_Q3, 1 core, stdlib):
  python -B -m vpd.tvr.demos build --replay OUT/tvr/replay/sft1600_lvp --images OUT/tvr/demos/sft1600_lvp/images
         --out OUT/tvr/demos/sft1600_lvp [--reach OUT/tvr/reach/v1] [--public SFT_TRAIN] [--tasks-dir DIR]
  python -B -m vpd.tvr.demos check --out OUT/tvr/demos/sft1600_lvp      (exit 0 iff every sha matches)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from vpd import paths
from vpd.tvr import kin_env, pilot_cfg, tasks as T, tvr_text

SCHEMA = "vpd.tvr.demos/1"
REASONS = ("move_failed", "pose_mismatch", "stop_not_last", "no_stop", "final_not_target")
MAX_MISMATCH_PER_TASK = 20
PNG_SIG = b"\x89PNG\r\n\x1a\n"


class Refused(Exception):
    """An infrastructure problem: nothing is written, exit 2."""


def png_size(path) -> Tuple[int, int]:
    """(width, height) from the PNG IHDR (stdlib; Refused if not a PNG)."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != PNG_SIG or head[12:16] != b"IHDR":
        raise Refused(f"{path}: not a PNG")
    return struct.unpack(">II", head[16:24])


def drop_reason(task: Mapping, actions: Sequence[str], states: Sequence[Mapping],
                action_success: Sequence[Optional[bool]]) -> Optional[Tuple[str, str]]:
    """(reason, detail) of the first failing G10(a) check, or None (keep)."""
    implied = kin_env.pose_of(task["start"])
    if kin_env.pose_of(states[0]) != implied:
        return "pose_mismatch@start", f"reset pose {tuple(kin_env.pose_of(states[0]))} != start {tuple(implied)}"
    k_stop = None
    for k, a in enumerate(actions):
        if a == "Stop":
            if k != len(actions) - 1:
                return "stop_not_last", f"Stop at action {k} of {len(actions)}"
            k_stop = k
            break
        nxt = kin_env.step(implied, a, None)[0]
        thor = kin_env.pose_of(states[k + 1])
        if a in kin_env.MOVES and action_success[k] is not True:
            return f"move_failed@{k}", f"{a} at {tuple(implied)}: action_success {action_success[k]}"
        if thor != nxt:
            return f"pose_mismatch@{k}", f"{a}: THOR {tuple(thor)} != implied {tuple(nxt)}"
        implied = nxt
    if k_stop is None:
        return "no_stop", f"{len(actions)} actions, no Stop"
    before = kin_env.pose_of(states[k_stop])
    target = kin_env.pose_of(task["target"])
    if not kin_env.is_exact(before, target):
        return "final_not_target", f"pose before Stop {tuple(before)} != target {tuple(target)}"
    return None


def kin_mismatches(task: Mapping, reach, actions: Sequence[str], states: Sequence[Mapping],
                   action_success: Sequence[Optional[bool]], valid: Sequence[Sequence[str]],
                   user_texts: Sequence[str]) -> List[Dict]:
    """Every step where KinEpisode(task, R(task)) differs from the THOR replay (pose, action_success, valid list,
    user text); at most MAX_MISMATCH_PER_TASK."""
    ep = kin_env.KinEpisode(task, reach)
    o = ep.reset()
    out: List[Dict] = []

    def add(k, field, thor, kin):
        if len(out) < MAX_MISMATCH_PER_TASK:
            out.append({"task_id": task["task_id"], "k": k, "field": field, "thor": thor, "kin": kin})

    for k, a in enumerate(actions):
        tp = kin_env.pose_of(states[k])
        if tp != o["pose"]:
            add(k, "pose", list(tp), list(o["pose"]))
        if list(valid[k]) != o["valid"]:
            add(k, "valid", list(valid[k]), o["valid"])
        if user_texts[k] != o["obs_str"]:
            add(k, "user_text", user_texts[k], o["obs_str"])
        o = ep.step(a)
        if action_success[k] != o["info"]["action_success"]:
            add(k, "action_success", action_success[k], o["info"]["action_success"])
        if a == "Stop":
            break
    n = len(actions)
    if len(states) > n:
        tp = kin_env.pose_of(states[n])
        if tp != o["pose"]:
            add(n, "pose", list(tp), list(o["pose"]))
    if len(valid) > n and "Stop" not in actions:  # the unanswered observation after the cap
        if list(valid[n]) != o["valid"]:
            add(n, "valid", list(valid[n]), o["valid"])
        if len(user_texts) > n and user_texts[n] != o["obs_str"]:
            add(n, "user_text", user_texts[n], o["obs_str"])
    return out


def _ready_summary(man: Mapping) -> List[Dict]:
    """The THOR worker ready replies of the replay manifest, reduced to the render-relevant fields."""
    rows = None
    for key in ("pool_ready", "ready", "workers", "pool"):
        v = man.get(key)
        if isinstance(v, dict):
            v = v.get("ready", v.get("workers"))
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            rows = v
            break
    out = []
    for r in rows or []:
        font = r.get("font") if isinstance(r.get("font"), dict) else {}
        out.append({"wid": r.get("wid"), "lp": r.get("lp"), "profile": r.get("profile"),
                    "font_sha256": font.get("sha256"), "font_mode": font.get("mode"),
                    "settings": r.get("settings")})
    return out


def _abs(p) -> Path:
    return Path(os.path.abspath(str(p)))


def build(replay, images, out, *, reach_root=None, public=None, tasks_dir=None) -> Dict[str, Any]:
    """Everything demos build writes, in memory: {"files": {name: bytes}, "summary": ...}.  Raises Refused."""
    from vpd.tvr import reach as R

    replay, images, out = _abs(replay), _abs(images), _abs(out)
    res_p = replay / "results.json"
    for bad in ("results_invalid.json", "results_partial.json"):
        if (replay / bad).exists():
            raise Refused(f"{replay} holds {bad}: the replay is not a valid complete run")
    if not res_p.is_file():
        raise Refused(f"{replay}: no results.json")
    results = T.read_json(res_p)
    if results.get("valid") is not True:
        raise Refused(f"{res_p}: valid is {results.get('valid')!r}")
    man_p = replay / "manifest.json"
    if not man_p.is_file():
        raise Refused(f"{replay}: no manifest.json")
    replay_man = T.read_json(man_p)
    sft = T.load_split("sft", tasks_dir)
    sft_by = T.by_id(sft)
    pub_path = T.PUBLIC_JSON if public is None else Path(public)
    try:
        pub = T.public_sequences(pub_path, list(sft_by))
    except ValueError as e:
        raise Refused(f"public file: {e}") from None
    try:
        idx = R.ReachIndex(reach_root)
    except (OSError, ValueError, KeyError) as e:
        raise Refused(f"reach index {reach_root}: {e!r}") from None
    missing = [tid for tid in pub if not (replay / "tasks" / tid / "trajectory.json").is_file()]
    if missing:
        raise Refused(f"{len(missing)} public tasks have no replay trajectory (e.g. {missing[:5]})")
    no_reach = [tid for tid in pub if tid not in idx]
    if no_reach:
        raise Refused(f"{len(no_reach)} public tasks are not in the reach index (e.g. {no_reach[:5]})")
    order = [t["task_id"] for t in sft if t["task_id"] in pub]
    kept_lines: List[bytes] = []
    dropped: List[Dict] = []
    mism: List[Dict] = []
    kin_bad_kept: List[str] = []
    img_sha: Dict[str, str] = {}
    reasons: Dict[str, int] = {r: 0 for r in REASONS}
    for tid in order:
        task = sft_by[tid]
        try:
            tj = T.read_json(replay / "tasks" / tid / "trajectory.json")
            reach_set = idx.reach(tid)
        except (OSError, ValueError, KeyError) as e:
            raise Refused(f"{tid}: {e!r}") from None
        if not isinstance(tj, dict):
            raise Refused(f"{tid}: trajectory.json is not an object")
        v = tj.get("vpd") or {}
        acts = list(tj.get("actions") or [])
        pacts = pub[tid]["actions"]
        cap = pilot_cfg.cap_of(task)
        if tj.get("task_id") != tid:
            raise Refused(f"{tid}: trajectory task_id {tj.get('task_id')!r}")
        if acts != pacts and not (len(acts) == cap and acts == pacts[:cap]):
            raise Refused(f"{tid}: replayed actions differ from the public sequence")
        states, succ = v.get("states") or [], v.get("action_success") or []
        valid, texts, imgs = v.get("valid") or [], v.get("user_texts") or [], v.get("images")
        n = len(acts)
        m = len(texts)  # user turns: n, or n + 1 when the episode ran into the cap without Stop (eval_va appends
        #                 the next observation unless the env itself ended the episode)
        m_ok = m == n or (m == n + 1 and "Stop" not in acts)
        if len(states) != n + 1 or len(succ) != n or len(valid) != m or not m_ok \
                or not isinstance(imgs, list) or len(imgs) != m:
            raise Refused(f"{tid}: per-turn lists incomplete (states {len(states)}, action_success {len(succ)}, "
                          f"valid {len(valid)}, user_texts {m}, images "
                          f"{len(imgs) if isinstance(imgs, list) else imgs}) for {n} actions")
        if v.get("reach_digest") != idx.digest(tid):
            raise Refused(f"{tid}: replay used reach {v.get('reach_digest')}, frozen is {idx.digest(tid)}")
        for k in range(m):
            if texts[k] != tvr_text.build_concat_user_text(list(valid[k])):
                raise Refused(f"{tid}: user text {k} is not build_concat_user_text(valid)")
        # images: every turn's CURRENT and the target, sha256 = what the trajectory recorded (file integrity)
        files = [(f"{tid}/step_{k:02d}.png", (imgs[k] or {}).get("cur_sha256")) for k in range(m)]
        files.append((f"{tid}/target.png", (v.get("target") or {}).get("sha256")))
        for rel, want in files:
            if not want:
                raise Refused(f"{tid}: the trajectory records no sha256 for {rel}")
            p = images / rel
            if not p.is_file():
                raise Refused(f"missing image {p}")
            if png_size(p) != (pilot_cfg.FROZEN["render"]["width"], pilot_cfg.FROZEN["render"]["height"]):
                raise Refused(f"{p}: size {png_size(p)}")
            sha = T.sha256_file(p)
            if sha != want:
                raise Refused(f"{p}: sha256 {sha} != trajectory's {want}")
            img_sha[rel] = sha
        why = drop_reason(task, acts, states, succ)
        mm = kin_mismatches(task, reach_set, acts, states, succ, valid, texts)
        mism.extend(mm)
        if why is not None:
            dropped.append({"task_id": tid, "reason": why[0], "detail": why[1]})
            reasons[why[0].split("@")[0]] += 1
            continue
        if not ((tj.get("result") or {}).get("success") is True and (tj.get("result") or {}).get("stopped")):
            raise Refused(f"{tid}: kept by the drop rule but THOR result is {tj.get('result')}")
        if mm:
            kin_bad_kept.append(tid)
        msgs = [{"role": "system", "content": tvr_text.SYSTEM_PROMPT}]
        paths_ = []
        for k, a in enumerate(acts):
            msgs.append({"role": "user", "content": texts[k]})
            msgs.append({"role": "assistant", "content": f"Action: {a}"})
            paths_ += [str(images / tid / f"step_{k:02d}.png"), str(images / tid / "target.png")]
        rec = {"id": tid, "category": pilot_cfg.category_of(task), "dataset": task["dataset"],
               "scene": task["scene"], "n_turns": n, "messages": msgs, "images": paths_}
        kept_lines.append((json.dumps(rec) + "\n").encode("utf-8"))
    demos_all = b"".join(kept_lines)
    images_doc = {"schema": "vpd.tvr.images_sha256/1", "root": str(images), "n": len(img_sha),
                  "use": "file integrity of these files only; never compared with another render (F4)",
                  "files": dict(sorted(img_sha.items()))}
    images_b = T.json_bytes(images_doc)
    kin_doc = {"n_checked": len(order), "n_mismatches": len(mism),
               "tasks_with_mismatch": sorted({m["task_id"] for m in mism}), "kept_with_mismatch": kin_bad_kept,
               "kin_equal_all_kept": not kin_bad_kept, "mismatches": mism}
    kin_b = T.json_bytes(kin_doc)
    dropped_b = T.json_bytes(dropped)
    files = {"demos_all.jsonl": demos_all, "dropped.json": dropped_b, "kin_check.json": kin_b,
             "images_sha256.json": images_b}
    manifest = {"schema": SCHEMA, "kept": len(kept_lines), "dropped": len(dropped),
                "drop_reasons": {r: c for r, c in reasons.items() if c},
                "demos_all_sha256": T.sha256_bytes(demos_all),
                "images_sha256_file_sha256": T.sha256_bytes(images_b),
                "files": {n: T.sha256_bytes(b) for n, b in files.items()},
                "kin_equal_all_kept": not kin_bad_kept, "min_kept": pilot_cfg.FROZEN["gates"]["min_kept"],
                "replay_run": str(replay), "replay_manifest_sha256": T.sha256_file(man_p),
                "replay_results_sha256": T.sha256_file(res_p), "reach_index_sha256": idx.sha256,
                "public": str(pub_path), "public_sha256": T.sha256_file(pub_path), "images": str(images),
                "render": {"pilot_cfg": pilot_cfg.FROZEN["render"], "workers": _ready_summary(replay_man)},
                "code_sha256": T.code_sha256()}
    return {"files": files, "manifest": manifest}


def cmd_build(a) -> int:
    out = _abs(a.out)
    try:
        b = build(a.replay, a.images, out, reach_root=a.reach, public=a.public, tasks_dir=a.tasks_dir)
    except Refused as e:
        print(f"[demos] refusing: {e}")
        return 2
    if (out / "DONE").is_file():
        old = out / "demos_all.jsonl"
        if old.is_file() and old.read_bytes() == b["files"]["demos_all.jsonl"]:
            print(f"[demos] {out}: DONE with an identical demos_all.jsonl: no-op")
            return 0
        print(f"[demos] refusing: {out} is DONE with a different demos_all.jsonl")
        return 2
    for n, data in b["files"].items():
        T.write_bytes_atomic(out / n, data)
    T.write_json(out / "manifest.json", dict(b["manifest"], created_utc=T.utc()))
    T.write_bytes_atomic(out / "DONE", (T.utc() + "\n").encode())
    m = b["manifest"]
    print(f"[demos] {out}: kept {m['kept']} dropped {m['dropped']} {m['drop_reasons']} kin_equal_all_kept "
          f"{m['kin_equal_all_kept']} (min_kept {m['min_kept']})")
    return 0


def check(out) -> Dict[str, Any]:
    out = _abs(out)
    res: Dict[str, Any] = {"ok": False, "bad": []}
    try:
        man = T.read_json(out / "manifest.json")
        for n, want in (man.get("files") or {}).items():
            if T.sha256_file(out / n) != want:
                res["bad"].append(n)
        if T.sha256_file(out / "demos_all.jsonl") != man["demos_all_sha256"]:
            res["bad"].append("demos_all.jsonl")
        if T.sha256_file(out / "images_sha256.json") != man["images_sha256_file_sha256"]:
            res["bad"].append("images_sha256.json")
        doc = T.read_json(out / "images_sha256.json")
        root = Path(doc["root"])
        for rel, want in doc["files"].items():
            p = root / rel
            if not p.is_file() or T.sha256_file(p) != want:
                res["bad"].append(rel)
        res["n_images"] = len(doc["files"])
        res["done"] = (out / "DONE").is_file()
    except (OSError, ValueError, KeyError) as e:
        res["bad"].append(repr(e))
    res["ok"] = not res["bad"] and bool(res.get("done"))
    res["bad"] = sorted(set(res["bad"]))[:50]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.tvr.demos", description="TVR demos from the THOR replay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--replay", required=True)
    b.add_argument("--images", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--reach", default=str(paths.TVR_OUT / "reach" / "v1"))
    b.add_argument("--public", default=None)
    b.add_argument("--tasks-dir", default=None)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        try:
            return cmd_build(a)
        except (OSError, ValueError, KeyError) as e:
            print(f"[demos] error: {e!r}")
            return 1
    r = check(a.out)
    print(json.dumps(r))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
