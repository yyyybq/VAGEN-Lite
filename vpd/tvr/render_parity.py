"""Render consistency by TOLERANCE between two TVR replays (CONTRACT_wave2 5.2.6; F4: lavapipe frames are not
bit-reproducible across processes / L / paths / reloads, so no frame sha256 is ever compared).

    TVR_PY -B -m vpd.tvr.render_parity --replay D --ref OUT/tvr/replay/sft1600_lvp \
        --ref-images OUT/tvr/demos/sft1600_lvp/images --images I --task-set sft2 --out D/render_parity.json
exit 0 pass, 1 fail, 2 refused (missing files).

For every task of the set (sft2 = the probe's 2 public sample tasks, probe_lvp.SAMPLE_TASKS) and every frame
(the CURRENT image of every user turn, step_{k:02d}.png, plus target.png): probe_lvp.pair_metrics(ref, new,
mask=<all-False>) (no badge mask: both sides burn the badge with the same pinned font) and the frozen rule
render.consistency_rule (PSNR >= 40 dB AND SSIM >= 0.99, SSIM = probe_lvp.SSIM_DEFINITION, via
probe_lvp.passes_rule); plus EXACT equality of the two trajectories' vpd.states (as poses, 4.2), vpd.valid,
vpd.user_texts and vpd.action_success.  pass iff every frame passes and every exact item holds.
Python 3.10 (TVR env) and 3.12; numpy / skimage / PIL are imported inside functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from vpd.tvr import pilot_cfg, probe_lvp

SCHEMA = "vpd.tvr.render_parity/1"
EXACT_KEYS = ("states", "valid", "user_texts", "action_success")


class Refused(Exception):
    pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rule() -> Dict[str, Any]:
    return dict(pilot_cfg.FROZEN["render"]["consistency_rule"])


def _r(v) -> float:
    return round(float(v), 2) + 0.0


def pose_key(s: Dict[str, Any]) -> List[Any]:
    """A trajectory state as its 4.2 pose (x, z, rot, hor); y ignored."""
    p = s["position"]
    return [_r(p["x"]), _r(p["z"]), int(round(float(s["rotation_y"]))) % 360, int(round(float(s["horizon"])))]


def task_ids_of(name: str, replay: Path) -> List[str]:
    if name == "sft2":
        return list(probe_lvp.SAMPLE_TASKS)
    if name == "all":
        return sorted(p.parent.name for p in (replay / "tasks").glob("*/trajectory.json"))
    if name.startswith("ids:"):
        return [x for x in name[4:].split(",") if x]
    raise Refused(f"unknown task set {name!r} (sft2 | all | ids:a,b)")


def load_traj(run: Path, tid: str) -> Dict[str, Any]:
    p = run / "tasks" / tid / "trajectory.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"missing / unreadable {p}: {e!r}")


def frames_of(traj: Dict[str, Any]) -> List[str]:
    n_turns = len((traj.get("vpd") or {}).get("user_texts") or [])
    return [f"step_{k:02d}.png" for k in range(n_turns)] + ["target.png"]


def frame_pass(m: Dict[str, Any], r: Dict[str, Any]) -> bool:
    if "error" in m:
        return False
    return bool(probe_lvp.passes_rule(m) and probe_lvp.fnum(m["psnr_db"]) >= float(r["psnr_min_db"])
                and probe_lvp.fnum(m["ssim"]) >= float(r["ssim_min"]))


def _read(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def renderer_of(replay: Path) -> Dict[str, Any]:
    """The new replay's renderer: the MEASURED block of its result file (the workers' ready replies: profile, lp,
    font sha, workers_ok) when there is one, else the requested render block of its protocol.json."""
    res = {}
    for name in ("results.json", "results_invalid.json", "results_partial.json"):
        res = _read(replay / name)
        if res:
            break
    measured = res.get("renderer") if isinstance(res.get("renderer"), dict) else None
    rd = measured or (_read(replay / "protocol.json").get("render") or {})
    return {"profile": rd.get("profile"), "lp": rd.get("lp"), "envs": rd.get("envs"),
            "font_sha256": rd.get("font_sha256"), "build": rd.get("build") or pilot_cfg.FROZEN["render"]["build"],
            "workers_ok": (measured or {}).get("workers_ok"),
            "source": "results renderer (measured)" if measured else "protocol.json render (requested)",
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "root": (os.getuid() == 0) if hasattr(os, "getuid") else None, "host": socket.gethostname()}


def compare(replay, ref, ref_images, images, task_set: str = "sft2", ssim_fn: Optional[Callable] = None
            ) -> Dict[str, Any]:
    """The render_parity document (Refused when a trajectory or a frame file is missing)."""
    import numpy as np

    replay, ref, ref_images, images = Path(replay), Path(ref), Path(ref_images), Path(images)
    r = rule()
    mask = np.zeros((360, 640), dtype=bool)  # no badge mask (same pinned font on both sides)
    tids = task_ids_of(task_set, replay)
    if not tids:
        raise Refused(f"task set {task_set!r} is empty")
    frames: List[Dict[str, Any]] = []
    exact = {k: True for k in EXACT_KEYS}
    raw_states_equal = True  # report only: the raw ThorEnv.get_state() dicts (y, float noise included)
    per_task = []
    for tid in tids:
        new_t, ref_t = load_traj(replay, tid), load_traj(ref, tid)
        nv, rv = new_t.get("vpd") or {}, ref_t.get("vpd") or {}
        eq = {"states": [pose_key(s) for s in nv.get("states") or []] == [pose_key(s) for s in rv.get("states") or []],
              "valid": nv.get("valid") == rv.get("valid"),
              "user_texts": nv.get("user_texts") == rv.get("user_texts"),
              "action_success": nv.get("action_success") == rv.get("action_success")}
        eq["states_raw"] = nv.get("states") == rv.get("states")
        raw_states_equal = raw_states_equal and bool(eq["states_raw"])
        for k in EXACT_KEYS:
            exact[k] = exact[k] and bool(eq[k])
        names = frames_of(ref_t)
        for name in names:
            a, b = ref_images / tid / name, images / tid / name
            for p in (a, b):
                if not p.is_file():
                    raise Refused(f"missing frame {p}")
            m = probe_lvp.pair_metrics(a, b, mask=mask, ssim_fn=ssim_fn)
            frames.append({"task_id": tid, "frame": name, "psnr_db": m.get("psnr_db"), "ssim": m.get("ssim"),
                           "mae": m.get("mae"), "max_abs": m.get("max_abs"), "identical": m.get("identical"),
                           "error": m.get("error"), "pass": frame_pass(m, r)})
        per_task.append({"task_id": tid, "n_frames": len(names), "exact": eq})
    ok_frames = [f for f in frames if f["pass"]]
    psnrs = [probe_lvp.fnum(f["psnr_db"]) for f in frames if f.get("psnr_db") is not None]
    ssims = [probe_lvp.fnum(f["ssim"]) for f in frames if f.get("ssim") is not None]
    doc = {"schema": SCHEMA, "utc": utc(), "rule": r,
           "ssim_definition_sha256": hashlib.sha256(probe_lvp.SSIM_DEFINITION.encode()).hexdigest(),
           "task_set": task_set, "task_ids": tids, "replay": str(replay), "ref": str(ref),
           "ref_images": str(ref_images), "images": str(images), "frames": frames, "n_frames": len(frames),
           "n_pass": len(ok_frames), "min_psnr_db": (min(psnrs) if psnrs else None),
           "min_ssim": (min(ssims) if ssims else None), "exact": exact,
           "states_raw_equal": raw_states_equal, "states_rule": "4.2 poses (x, z 2 dp; rot, hor int); y ignored",
           "per_task": per_task,
           "renderer": renderer_of(replay), "note": "no frame sha256 is compared (F4); tolerance rule only"}
    doc["pass"] = bool(frames) and len(ok_frames) == len(frames) and all(exact.values())
    return doc


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(probe_lvp.jsafe(obj), f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", required=True, help="the new replay run dir (run_tvr replay --run-dir)")
    ap.add_argument("--ref", required=True, help="the reference replay run dir (the data job's)")
    ap.add_argument("--ref-images", required=True, help="the reference images dir (<task_id>/step_KK.png)")
    ap.add_argument("--images", required=True, help="the new replay's images dir")
    ap.add_argument("--task-set", default="sft2", help="sft2 | all | ids:a,b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        doc = compare(a.replay, a.ref, a.ref_images, a.images, a.task_set)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        write_json_atomic(a.out, {"schema": SCHEMA, "utc": utc(), "pass": False, "refused": str(e)})
        return 2
    write_json_atomic(a.out, doc)
    print(json.dumps({"pass": doc["pass"], "n_frames": doc["n_frames"], "n_pass": doc["n_pass"],
                      "min_psnr_db": probe_lvp.jsafe(doc["min_psnr_db"]), "min_ssim": doc["min_ssim"],
                      "exact": doc["exact"]}))
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
