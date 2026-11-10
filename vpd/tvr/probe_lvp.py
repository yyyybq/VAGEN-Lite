"""TVR lavapipe probe (DECISIONS G3; EXPLORATORY measurement that feeds the TVR pilot design; never a gate).

Runs inside the sco job vpd-tvr-probe (vpd/jobs/tvr_probe.sh) under the TVR env python (3.10, ai2thor 5.0.0), on
one 14-vCPU node, with NO GPU rendering and NO vLLM: every AI2-THOR worker is a separate process with the lavapipe
environment of vpd.tvr.lvp (one HOME, one Unity player and LP_NUM_THREADS=L each).  Never run `run` or `worker`
on the login node (they start Unity); `expect` and the pure functions are login-node safe.

    python -s -u -m vpd.tvr.probe_lvp run --out RUN/out --deadline-s S     (the parent; cwd = the worktree)
    python -s -m vpd.tvr.probe_lvp expect --out RUN/out                    (checks the outputs -> expect.json)
    python -s -m vpd.tvr.probe_lvp recommend --out RUN/out [--max-el N] [--max-cores X] [--write F]
                                            (recomputes both recommendations from summary.json; stdlib only)

Phases of `run` (each guarded by the soft deadline; summary.json is rewritten after every phase and grid cell):
  P0 preflight  lavapipe ICD (lvp.check_icd), the pinned badge font (lvp.font_status), the public sample's
                SHA256SUMS, task files, ProcTHOR houses (one pass over train.jsonl.gz -> out/houses/<scene>.json).
  P1 compare + determinism, one worker at L = P1_L (12).  The first worker's start-up fixes the LD_LIBRARY_PATH
     profile: lvp profile "envlib" first; only if that worker cannot start, one retry with "envlib+orig" (the
     measured profile); every later worker uses the profile that worked (summary.lvp_profile,
     summary.workers[].env_profile).  Then, in this order:
     (f) FIRST, right after the start-up: the 2 public sample trajectories replayed through TVRBench's
         ThorViewEnv (_sync_reset/_sync_step, i.e. the eval code path): our CURRENT/TARGET PNGs, the valid-actions
         text of every turn and every action's success; then compared with the public PNGs (MAE, max |d|, PSNR,
         SSIM, badge masked) plus controls (pose-shifted pairs, a repeat render), and the frozen rule of G3(d)
         applied: same rendering iff PSNR >= 40 dB AND SSIM >= 0.99 on every compared frame, with SSIM =
         skimage's returned mssim (frame_metrics);
     (e) the badge geometry burn_label draws with the font rule in force;
     (b) frame sha256 of fixed probe poses (render by Teleport) on FloorPlan1 and the first ProcTHOR eval house;
         two different action paths to one pose (A = RotateRight, RotateLeft, Move; B = Move, LookDown, LookUp;
         C = direct Teleport; A repeated); a reload of each scene (frames and GetReachablePositions again).
         Each of (e), the path test of a scene and the reload test of a scene is guarded: a worker error is
         recorded (summary.errors; the test counts as unmeasured) and P1 goes on with a live worker;
     and the grid cell (E=1, L=12) on both grid scenes.
  P2 grid (a), part 1: the L = REACH_L (1) group of the grid.  E parallel envs x L lavapipe threads, STEPS steps
                per env on each grid scene: aggregate steps/s, p50/p95 step latency, cgroup cpu.stat deltas, bad
                frames.  A step = one Controller step (an action drawn from the valid ones) + burn_label + PNG
                encode + the next valid list, as one eval turn costs an env worker (the target image is assumed
                cached by the harness).  Controller start-ups are serialised (eval_va's init_lock).  New workers
                render the probe poses too, all at once (determinism across L values and across processes).
  P3 reach (d)  on the pool P2 left (6 workers at L=1, so no extra start-ups): GetReachablePositions of all SFT
                and eval scenes (SFT first), each scene loaded twice in two different workers ->
                out/reach/<scene>.json {loads (with call_pose, the agent pose at the call), equal_raw, equal_xz,
                positions}.
  P4 grid (a), part 2: the other L groups with the time left.
  Cell order (cell_order, P2 and P4 alike): likely-best first, so a short deadline cuts the least useful cells:
     tier 1  eligible cells with HI_EL (6) <= E*L <= MAX_EL_RECOMMEND (11): more of the CPU budget in use;
     tier 2  the other eligible cells (E*L < 6);
     tier 3  the cells with E*L > 11 (never eligible), last.
     Within a tier: L groups in grid order, the L the previous tier ended on first (no extra pool switch), and E
     descending in each group (the pool is started once at its largest E, then reused).  Default P4 order
     (E, L): (5,2) (4,2) (3,2) (3,3) (2,3) (2,4) | (1,4) (2,2) (1,2) (1,3) | (4,3) (6,2) (3,4); P2: (6,1) then
     (4,1) (3,1) (2,1) (1,1).
  (c) load times come from every load of P1-P4.
Output: out/summary.json (schema SCHEMA, label EXPLORATORY), out/reach/, out/compare/<task>/, out/houses/,
out/homes/w<id>_L<L>/ (per-worker HOME with Unity's Player.log; summary.homes = its size), out/logs/worker_*.log.
Two recommendations (recommendations(summary), a pure function of summary.json; `recommend` recomputes them):
  recommended           the complete cell with E*L <= MAX_EL_RECOMMEND (11: >= 3 vCPU stay for vLLM + harness)
                        and the highest harmonic mean over the two grid scenes of the aggregate steps/s (ties: fewer
                        threads);
  recommended_measured  the same over the complete cells whose MEASURED cgroup cores (max over the two grid scenes
                        of the cell's cpu.stat delta) <= MAX_CORES_RECOMMEND (11); each env also costs its Unity
                        main/render thread and the Python worker, which E*L ignores (ties: fewer cores).
  Both report cores per env (cgroup cores / E, and per process: Unity + worker CPU seconds / wall).

Worker protocol: the parent writes one JSON command per line to the worker's --cmd-fd and reads one JSON reply
per line from --reply-fd (stdin is /dev/null, stdout/stderr go to the worker log, so nothing ai2thor or Unity
prints can corrupt the channel).  Reply 0 is the start-up ('ready').  Ops: load, reach, render, seq, steps,
replay, badge, quit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import queue
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from vpd import paths
from vpd.tvr import lvp

SCHEMA = "vpd.tvr.probe_lvp/1"
LABEL = "EXPLORATORY"
PURPOSE = ("TVR lavapipe probe (DECISIONS G3): CPU-render throughput grid, per-pose determinism, scene load "
           "times, reachable-set dumps, badge font, public-image comparison. Measurement only; no gate.")

# --- plan -----------------------------------------------------------------------------------------------------
# L -> E values.  The L = REACH_L group runs first (P2), the others after the reach dumps (P4), each in cell_order.
# E*L <= 12 (2 of 14 vCPU kept for Unity main threads / the parent); (5, 2) is added because it is eligible
# (10 <= 11).  (1, 12) runs in P1.
GRID_DEFAULT: Dict[int, Tuple[int, ...]] = {1: (1, 2, 3, 4, 6), 2: (1, 2, 3, 4, 5, 6), 3: (1, 2, 3, 4), 4: (1, 2, 3)}
P1_L = 12
STEPS = 100
PHASES = ("p0_preflight", "p1_compare_det", "p2_grid", "p3_reach", "p4_grid")
MAX_EL_RECOMMEND = 11
MAX_CORES_RECOMMEND = 11.0
HI_EL = 6  # cell_order tier 1 starts here
MAX_EL = 12
REACH_L, REACH_E = 1, 6
REACH_MIN_WORKERS = 3  # reuse the pool left by the grid when it has at least this many live workers
FIRST_GRID_SCENE = "FloorPlan1"
SAMPLE_DIR = paths.TVR_OUT / "hf" / "tvr-sft-va-sample"
SAMPLE_TASKS = ("ithor_FloorPlan11_easy_t00_s00", "procthor_procthor_09990_hard_t19_s01")
SFT_TRAIN = paths.TVR_OUT / "hf_meta" / "tvr-sft-va" / "sft_train.json"
# frozen rule of DECISIONS G3(d)
PSNR_MIN_DB = 40.0
SSIM_MIN = 0.99
NOISE_FLOOR_NOTE = ("the public VA and CoT repos hold different PNGs of one pose: max |d| 1, PSNR 60.73 dB "
                    "(tvr_data_paper, 2026-11-10)")
# probe poses picked from the sorted reachable set
PROBE_K = 4
PROBE_ROTS = (0.0, 90.0, 225.0, 315.0)
PROBE_HORS = (0.0, 30.0, -30.0, 0.0)
# time (s)
STARTUP_TIMEOUT_S = 150.0
LOAD_TIMEOUT_S = 180.0
CMD_TIMEOUT_S = 120.0
REPLAY_TIMEOUT_S = 420.0
REACH_JOB_TIMEOUT_S = 240.0
QUIT_TIMEOUT_S = 20.0
STARTUP_EST_S = 12.0
CELL_EST_DEFAULT_S = 45.0
REACH_RESERVE_S = 300.0
REACH_JOB_EST_S = 20.0
SUMMARY_RESERVE_S = 45.0
GO_DELAY_S = 0.5


class Terminated(BaseException):
    """SIGTERM / SIGINT in the parent: stop now, kill the workers, write the summary."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(*args) -> None:
    print("[probe_lvp]", time.strftime("%H:%M:%S", time.gmtime()), *args, flush=True)


# --- JSON helpers -----------------------------------------------------------------------------------------------
def jsafe(x: Any) -> Any:
    """JSON-safe copy: inf -> 'inf', nan -> None, numpy scalars -> python, Path -> str, tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): jsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsafe(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "item"):  # numpy scalar
        return jsafe(x.item())
    return str(x)


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as f:
        json.dump(jsafe(obj), f, indent=1, sort_keys=False)
    os.replace(tmp, path)


def dir_size(d) -> Dict[str, Any]:
    """{dir, bytes, files} of a directory tree (symlinks are neither followed nor counted beyond themselves)."""
    d = Path(d)
    tot = n = 0
    for root, _dirs, files in os.walk(d):
        for name in files:
            try:
                tot += os.lstat(os.path.join(root, name)).st_size
                n += 1
            except OSError:
                continue
    return {"dir": str(d), "bytes": tot, "files": n}


def fnum(x: Any) -> float:
    """A metric from a JSON file back to float ('inf' -> inf)."""
    if isinstance(x, str):
        return float(x)
    return float("nan") if x is None else float(x)


# --- statistics -------------------------------------------------------------------------------------------------
def percentile(xs: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy's default); None for no data."""
    s = sorted(float(x) for x in xs)
    if not s:
        return None
    k = (len(s) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def dist(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [float(x) for x in xs]
    if not xs:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}
    return {"n": len(xs), "p50": round(percentile(xs, 50), 3), "p95": round(percentile(xs, 95), 3),
            "max": round(max(xs), 3), "mean": round(sum(xs) / len(xs), 3)}


# --- images -----------------------------------------------------------------------------------------------------
def frame_sha(frame) -> str:
    import numpy as np

    a = np.ascontiguousarray(frame)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def badge_mask(h: int = lvp.H, w: int = lvp.W, x0: int = lvp.BADGE_X0, y1: int = lvp.BADGE_Y1):
    """True on the masked badge region: x >= x0 and y <= y1."""
    import numpy as np

    m = np.zeros((h, w), dtype=bool)
    m[: y1 + 1, x0:] = True
    return m


SSIM_DEFINITION = ("ssim = the mssim RETURNED by skimage.metrics.structural_similarity(public, ours', "
                   "channel_axis=2, data_range=255, full=True) with its default 7x7 window, where ours' = our frame "
                   "with the badge pixels (x >= %d, y <= %d) set to the public frame's; skimage averages its SSIM "
                   "map over every pixel and channel after cropping (win_size-1)//2 = 3 px at each border (the "
                   "badge region, equal by construction, is inside that average).  The G3(d) rule uses this ssim.  "
                   "ssim_map_mean_unmasked = the mean of the full map (no crop) over the unmasked pixels, reported "
                   "only." % (lvp.BADGE_X0, lvp.BADGE_Y1))


def frame_metrics(a, b, mask=None, ssim_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """MAE, max |d|, share of pixels with any channel |d| > 8, PSNR (dB, inf if equal) of two RGB uint8 frames over
    the pixels outside `mask`, and SSIM two ways (SSIM_DEFINITION).

    The masked pixels of b are first set to a's, so the SSIM windows that overlap the mask see no badge
    difference.  "ssim" (the rule metric) is skimage's returned mssim; "ssim_map_mean_unmasked" is the mean of
    skimage's full SSIM map over the unmasked pixels and the 3 channels.
    """
    import numpy as np

    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        return {"error": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    if mask is None:
        mask = np.zeros(a.shape[:2], dtype=bool)
    keep = ~mask
    b = b.copy()
    b[mask] = a[mask]
    d = a.astype(np.int16) - b.astype(np.int16)
    dk = np.abs(d[keep])  # (N, 3)
    mse = float((dk.astype(np.float64) ** 2).mean()) if dk.size else 0.0
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(255.0 ** 2 / mse)
    if ssim_fn is None:
        from skimage.metrics import structural_similarity as ssim_fn
    mssim, smap = ssim_fn(a, b, channel_axis=2, data_range=255, full=True)
    smap = np.asarray(smap, dtype=np.float64)
    ssim_unmasked = float(smap[keep].mean()) if dk.size else 1.0
    return {"mae": round(float(dk.mean()) if dk.size else 0.0, 5), "max_abs": int(dk.max()) if dk.size else 0,
            "frac_gt8": round(float((dk.max(axis=1) > 8).mean()) if dk.size else 0.0, 6),
            "psnr_db": psnr if math.isinf(psnr) else round(psnr, 3), "ssim": round(float(mssim), 6),
            "ssim_map_mean_unmasked": round(ssim_unmasked, 6), "identical": bool(mse == 0.0)}


def passes_rule(m: Mapping[str, Any]) -> bool:
    """The frozen G3(d) per-frame rule (ssim = skimage's returned mssim, SSIM_DEFINITION)."""
    if "error" in m:
        return False
    return fnum(m["psnr_db"]) >= PSNR_MIN_DB and fnum(m["ssim"]) >= SSIM_MIN


def badge_bbox(arr, color=(60, 60, 60), x_from: int = 480, y_to: int = 40) -> Optional[List[int]]:
    """[x0, y0, x1, y1] of the pixels exactly `color` in the top-right corner (the badge background)."""
    import numpy as np

    a = np.asarray(arr)[: y_to + 1, x_from:, :3]
    ys, xs = np.nonzero(np.all(a == np.array(color, dtype=a.dtype), axis=2))
    if ys.size == 0:
        return None
    return [int(xs.min()) + x_from, int(ys.min()), int(xs.max()) + x_from, int(ys.max())]


def load_rgb(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def pair_metrics(pub, ours, mask=None, ssim_fn=None) -> Dict[str, Any]:
    for p in (pub, ours):
        if not Path(p).is_file():
            return {"error": f"missing {p}"}
    return frame_metrics(load_rgb(pub), load_rgb(ours), mask if mask is not None else badge_mask(), ssim_fn)


# --- the public sample --------------------------------------------------------------------------------------------
def verify_sample(sample_dir=SAMPLE_DIR) -> Dict[str, Any]:
    """Check every line of sample_dir/SHA256SUMS ('<sha>  <relative path>')."""
    d = Path(sample_dir)
    out: Dict[str, Any] = {"dir": str(d), "ok": False, "n": 0, "bad": []}
    try:
        lines = (d / "SHA256SUMS").read_text().splitlines()
    except OSError as e:
        out["why"] = f"no SHA256SUMS: {e!r}"
        return out
    for line in lines:
        if not line.strip():
            continue
        sha, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        p = d / rel
        try:
            ok = hashlib.sha256(p.read_bytes()).hexdigest() == sha
        except OSError:
            ok = False
        out["n"] += 1
        if not ok:
            out["bad"].append(rel)
    out["ok"] = out["n"] > 0 and not out["bad"]
    return out


def parse_action(text: str) -> str:
    """'Action: X' (the whole reply) -> X; anything else raises (the public replies all match)."""
    t = text.strip()
    if not t.startswith("Action: ") or not t[8:].isalpha():
        raise ValueError(f"not an 'Action: X' reply: {text!r}")
    return t[8:]


def public_trajectory(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """One sft_train.json sample -> {task_id, turns: [{user_text, images}], actions}."""
    conv = sample["conversations"]
    imgs = sample["images"]
    humans = [m["value"] for m in conv if m["from"] == "human"]
    gpts = [m["value"] for m in conv if m["from"] == "gpt"]
    if len(humans) != len(gpts) or len(imgs) != 2 * len(humans):
        raise ValueError("unexpected sample layout")
    task_id = imgs[0].split("/")[1]
    turns = [{"user_text": humans[k], "images": [imgs[2 * k], imgs[2 * k + 1]]} for k in range(len(humans))]
    return {"task_id": task_id, "system": sample.get("system"), "turns": turns,
            "actions": [parse_action(g) for g in gpts]}


def load_public(sft_train, task_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    want = set(task_ids)
    with open(sft_train) as f:
        data = json.load(f)
    out = {}
    for s in data:
        imgs = s.get("images") or []
        if imgs and imgs[0].split("/")[1] in want:
            t = public_trajectory(s)
            out[t["task_id"]] = t
    missing = sorted(want - set(out))
    if missing:
        raise ValueError(f"sample tasks not in {sft_train}: {missing}")
    return out


def compare_task(pub_dir, ours_dir, traj: Mapping[str, Any], replay: Mapping[str, Any], ssim_fn=None,
                 system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Public vs our frames of one replayed trajectory, the controls and the per-turn checks."""
    pub_dir, ours_dir = Path(pub_dir), Path(ours_dir)
    mask = badge_mask()
    T = len(traj["turns"])
    rturns = list(replay.get("turns") or [])
    acts = list(traj["actions"])
    # a frame is comparable only while our replay is on the public path: every earlier action succeeded
    diverged_from = None
    for k, a in enumerate(acts[:-1]):
        succ = rturns[k].get("action_success") if k < len(rturns) else None
        if succ is not True:
            diverged_from = k + 1
            break
    frames = []
    for k in range(T):
        m = pair_metrics(pub_dir / f"step_{k:02d}.png", ours_dir / f"step_{k:02d}.png", mask, ssim_fn)
        frames.append(dict(m, frame=f"step_{k:02d}",
                           comparable=(diverged_from is None or k < diverged_from) and "error" not in m))
    m = pair_metrics(pub_dir / "target.png", ours_dir / "target.png", mask, ssim_fn)
    frames.append(dict(m, frame="target", comparable="error" not in m))
    shifted = []
    for k in range(T - 1):
        if k + 1 >= len(rturns) or rturns[k].get("state") == rturns[k + 1].get("state"):
            continue  # same pose: not a pose-shifted pair
        sm = pair_metrics(pub_dir / f"step_{k:02d}.png", ours_dir / f"step_{k + 1:02d}.png", mask, ssim_fn)
        if "error" not in sm:
            shifted.append(dict(sm, pair=f"pub step_{k:02d} / ours step_{k + 1:02d}"))
    repeat = []
    for a, b in (("step_00.png", "rep_step_00.png"), ("target.png", "rep_target.png")):
        rm = pair_metrics(ours_dir / a, ours_dir / b, mask, ssim_fn)
        repeat.append(dict(rm, pair=f"ours {a} / ours {b}"))
    text_eq = [k < len(rturns) and rturns[k].get("user_text") == traj["turns"][k]["user_text"] for k in range(T)]
    moves = []
    for k, a in enumerate(acts[:-1]):
        if a.startswith("Move"):
            moves.append({"k": k, "action": a,
                          "ok": bool(k < len(rturns) and rturns[k].get("action_success") is True)})
    pub_badge = our_badge = None
    if (pub_dir / "step_00.png").is_file():
        pub_badge = badge_bbox(load_rgb(pub_dir / "step_00.png"))
    if (ours_dir / "step_00.png").is_file():
        our_badge = badge_bbox(load_rgb(ours_dir / "step_00.png"))
    out = {"task_id": traj["task_id"], "n_turns": T, "frames": frames, "pose_shifted": shifted, "repeat": repeat,
           "diverged_from_step": diverged_from, "valid_text_equal": text_eq,
           "valid_text_equal_all": all(text_eq), "moves": moves, "moves_ok_all": all(m["ok"] for m in moves),
           "stop_success": (replay.get("stop") or {}).get("success"), "final_exact": replay.get("final_exact"),
           "badge_bbox": {"public": pub_badge, "ours": our_badge}, "replay_error": replay.get("error")}
    if system_prompt is not None:
        out["system_prompt_equal"] = traj.get("system") == system_prompt
    return out


def compare_verdict(tasks: Mapping[str, Mapping[str, Any]], n_expected: int) -> Dict[str, Any]:
    """Apply the frozen G3(d) rule over all compared frames of all tasks."""
    comp = [f for t in tasks.values() for f in t["frames"] if f.get("comparable")]
    fails = [f for f in comp if not passes_rule(f)]
    if fails:
        verdict = "different_rendering"
    elif len(comp) < n_expected:
        verdict = "incomplete"
    else:
        verdict = "same_rendering"
    shifted = [f for t in tasks.values() for f in t["pose_shifted"]]
    repeat = [f for t in tasks.values() for f in t["repeat"]]
    same_psnr = [fnum(f["psnr_db"]) for f in comp]
    sh_psnr = [fnum(f["psnr_db"]) for f in shifted]
    sep = None
    if same_psnr and sh_psnr:
        sep = min(same_psnr) > max(sh_psnr)
    return {
        "rule": (f"same rendering iff PSNR >= {PSNR_MIN_DB:g} dB AND SSIM >= {SSIM_MIN:g} on every compared frame "
                 f"(badge x >= {lvp.BADGE_X0}, y <= {lvp.BADGE_Y1} masked; SSIM = skimage's returned mssim, see "
                 f"ssim_definition; DECISIONS G3(d), frozen before looking)"),
        "ssim_definition": SSIM_DEFINITION,
        "verdict": verdict, "n_expected": n_expected, "n_compared": len(comp), "n_pass": len(comp) - len(fails),
        "min_psnr_db": min(same_psnr) if same_psnr else None,
        "min_ssim": min(fnum(f["ssim"]) for f in comp) if comp else None,
        "min_ssim_map_mean_unmasked": (min(fnum(f["ssim_map_mean_unmasked"]) for f in comp)
                                       if comp and all("ssim_map_mean_unmasked" in f for f in comp) else None),
        "max_abs_max": max(int(f["max_abs"]) for f in comp) if comp else None,
        "mae_mean": round(sum(float(f["mae"]) for f in comp) / len(comp), 5) if comp else None,
        "controls": {
            "pose_shifted": {"n": len(shifted), "max_psnr_db": max(sh_psnr) if sh_psnr else None,
                             "median_psnr_db": percentile(sh_psnr, 50),
                             "max_ssim": max(fnum(f["ssim"]) for f in shifted) if shifted else None},
            "metric_separates_poses": sep,
            "repeat": {"n": len(repeat), "identical": bool(repeat) and all(f.get("identical") for f in repeat),
                       "min_psnr_db": min(fnum(f["psnr_db"]) for f in repeat if "error" not in f)
                       if any("error" not in f for f in repeat) else None},
            "noise_floor_reference": NOISE_FLOOR_NOTE},
        "valid_text_equal_all": all(t["valid_text_equal_all"] for t in tasks.values()) if tasks else None,
        "moves_ok_all": all(t["moves_ok_all"] for t in tasks.values()) if tasks else None,
        "final_exact_all": all(t.get("final_exact") is True for t in tasks.values()) if tasks else None,
        "demos_use": "our lavapipe renders in every case (DECISIONS G3(d): ONE renderer for demos and eval)",
    }


# --- probe poses, grid analysis ----------------------------------------------------------------------------------
def pick_probe_poses(reach: Sequence[Mapping[str, float]], k: int = PROBE_K,
                     extra: Sequence[Mapping[str, Any]] = ()) -> List[Dict[str, Any]]:
    """`extra` poses, then k poses spread over the reachable set sorted by (x, z), with fixed rotations and
    horizons (PROBE_ROTS / PROBE_HORS)."""
    poses = [dict(p) for p in extra]
    pts = sorted(reach, key=lambda p: (round(p["x"], 2), round(p["z"], 2)))
    n = len(pts)
    for i in range(k if n else 0):
        p = pts[(n * (i + 1)) // (k + 1)]
        poses.append({"position": {"x": p["x"], "y": p["y"], "z": p["z"]}, "rotation_y": PROBE_ROTS[i % 4],
                      "horizon": PROBE_HORS[i % 4]})
    return poses


def pose_of(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {"position": dict(state["position"]), "rotation_y": state["rotation_y"], "horizon": state["horizon"]}


def reach_digest(pos: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    raw = hashlib.sha256(json.dumps(list(pos), sort_keys=True).encode()).hexdigest()
    xz = sorted({(round(p["x"], 2), round(p["z"], 2)) for p in pos})
    return {"n": len(pos), "sha256_raw": raw,
            "sha256_xz": hashlib.sha256(json.dumps(xz).encode()).hexdigest(), "n_xz": len(xz)}


def parse_grid(spec: str) -> Dict[int, Tuple[int, ...]]:
    """'2:1,2,3;4:1,2' -> {2: (1, 2, 3), 4: (1, 2)} (order kept); E*L must be <= MAX_EL."""
    out: Dict[int, Tuple[int, ...]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        lp, es = part.split(":")
        L = int(lp)
        E = tuple(int(e) for e in es.split(",") if e.strip())
        if L < 1 or not E or min(E) < 1 or max(E) * L > MAX_EL:
            raise ValueError(f"bad grid part {part!r} (L >= 1, E >= 1, E*L <= {MAX_EL})")
        out[L] = E
    return out


def cell_order(groups: Mapping[int, Sequence[int]], *, hi_el: int = HI_EL,
               max_el: int = MAX_EL_RECOMMEND) -> List[Tuple[int, int]]:
    """The (L, E) cells of `groups` (L -> E values) in run order, likely-best first (module doc): tier 1 eligible
    with E*L >= hi_el, tier 2 the other eligible cells, tier 3 E*L > max_el; within a tier the L groups in
    `groups` order with the L the previous tier ended on moved first, E descending (duplicates dropped)."""
    tiers = (lambda el: hi_el <= el <= max_el, lambda el: el < hi_el, lambda el: el > max_el)
    order: List[Tuple[int, int]] = []
    cur: Optional[int] = None
    for in_tier in tiers:
        by_l = {L: sorted({int(E) for E in Es if in_tier(int(E) * int(L))}, reverse=True)
                for L, Es in groups.items()}
        ls = [L for L in groups if by_l[L]]
        if cur in ls:
            ls.remove(cur)
            ls.insert(0, cur)
        for L in ls:
            order += [(int(L), E) for E in by_l[L]]
            cur = L
    return order


def cell_ok(c: Mapping[str, Any]) -> bool:
    return bool(c.get("ok")) and not c.get("skipped")


def _env_cores(e: Mapping[str, Any]) -> Optional[float]:
    """Cores one env used during its steps: (Unity + worker CPU seconds) / its wall (None when not measured)."""
    u, w, t = e.get("unity_cpu_s"), e.get("worker_cpu_s"), e.get("wall_s")
    if u is None or w is None or not t:
        return None
    return (float(u) + float(w)) / float(t)


def grid_table(cells: Sequence[Mapping[str, Any]], scenes: Sequence[str]) -> List[Dict[str, Any]]:
    """One row per (E, L): the per-scene aggregate steps/s and their harmonic mean (None unless every scene's
    cell is complete), the per-scene cgroup cores, cores_max (max over the scenes; None unless every scene's
    cell has it), cores_per_env = cores_max / E and proc_cores_per_env (mean over the envs of every scene of
    (Unity + worker CPU s) / wall)."""
    by: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = {}
    for c in cells:
        if c.get("skipped"):
            continue
        by.setdefault((c["E"], c["L"]), {})[c["scene"]] = c
    rows = []
    for (E, L), cs in by.items():
        per = {s: (cs[s]["agg_steps_per_s"] if s in cs and cell_ok(cs[s]) else None) for s in scenes}
        ok = bool(per) and all(v for v in per.values())
        hm = round(len(per) / sum(1.0 / v for v in per.values()), 3) if ok else None
        cores = {s: (cs[s].get("cpu") or {}).get("cores") for s in scenes if s in cs}
        cmax = (max(float(cores[s]) for s in scenes)
                if scenes and all(cores.get(s) is not None for s in scenes) else None)
        env_c = [x for s in scenes if s in cs for x in (_env_cores(e) for e in cs[s].get("per_env") or [])
                 if x is not None]
        rows.append({"E": E, "L": L, "EL": E * L, "agg_steps_per_s": per, "hmean_steps_per_s": hm, "ok": ok,
                     "p50_ms": {s: cs[s].get("p50_ms") for s in scenes if s in cs},
                     "p95_ms": {s: cs[s].get("p95_ms") for s in scenes if s in cs},
                     "cores": cores, "cores_max": None if cmax is None else round(cmax, 3),
                     "cores_per_env": None if cmax is None else round(cmax / E, 3),
                     "proc_cores_per_env": round(sum(env_c) / len(env_c), 3) if env_c else None,
                     "idle_workers": {s: cs[s].get("idle_workers") for s in scenes if s in cs}})
    rows.sort(key=lambda r: (r["L"], r["E"]))
    return rows


def _pick(r: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: r.get(k) for k in ("E", "L", "EL", "hmean_steps_per_s", "agg_steps_per_s", "cores", "cores_max",
                                  "cores_per_env", "proc_cores_per_env")}


ENV_ONLY_NOTE = "env-only throughput; with vLLM in the loop envs wait on the model, which favours more envs"


def recommend(rows: Sequence[Mapping[str, Any]], max_el: int = MAX_EL_RECOMMEND) -> Dict[str, Any]:
    """Max harmonic-mean aggregate steps/s among complete (E, L) with E*L <= max_el; ties -> fewer threads."""
    elig = [r for r in rows if r["ok"] and r["EL"] <= max_el]
    rule = (f"max harmonic mean (over the grid scenes) of aggregate steps/s among complete cells with "
            f"E*L <= {max_el} (>= 3 vCPU left for vLLM + harness); ties -> smaller E*L")
    if not elig:
        return {"E": None, "L": None, "rule": rule, "why": "no complete eligible cell", "n_eligible": 0}
    best = max(elig, key=lambda r: (r["hmean_steps_per_s"], -r["EL"]))
    return dict(_pick(best), rule=rule, n_eligible=len(elig), note=ENV_ONLY_NOTE)


def recommend_measured(rows: Sequence[Mapping[str, Any]],
                       max_cores: float = MAX_CORES_RECOMMEND) -> Dict[str, Any]:
    """Max harmonic-mean aggregate steps/s among complete (E, L) whose measured cgroup cores (cores_max: max over
    the grid scenes) <= max_cores; ties -> fewer cores, then smaller E*L."""
    measured = [r for r in rows if r["ok"] and r.get("cores_max") is not None]
    elig = [r for r in measured if float(r["cores_max"]) <= float(max_cores)]
    rule = (f"max harmonic mean (over the grid scenes) of aggregate steps/s among complete cells whose measured "
            f"cgroup cores (max over the grid scenes of the cell's cpu.stat usage delta / wall) <= {max_cores:g} "
            f"(>= 3 of 14 vCPU left for vLLM + harness, counting each env's Unity main/render thread and Python "
            f"worker too); ties -> fewer cores, then smaller E*L")
    extra = {"rule": rule, "n_eligible": len(elig), "n_complete_measured": len(measured),
             "n_complete_unmeasured": sum(1 for r in rows if r["ok"] and r.get("cores_max") is None),
             "note": ENV_ONLY_NOTE + "; cores include idle pool workers (grid_table idle_workers), so smaller-E "
                                     "cells of a larger pool are counted conservatively"}
    if not elig:
        return dict({"E": None, "L": None, "why": "no complete cell within the measured-core budget"}, **extra)
    best = max(elig, key=lambda r: (r["hmean_steps_per_s"], -float(r["cores_max"]), -r["EL"]))
    return dict(_pick(best), **extra)


def recommendations(summary: Mapping[str, Any], max_el: Optional[int] = None,
                    max_cores: Optional[float] = None) -> Dict[str, Any]:
    """Both recommendations and the grid table from a summary.json dict (pure; stdlib only).  The limits default
    to the summary's plan (else MAX_EL_RECOMMEND / MAX_CORES_RECOMMEND)."""
    plan = summary.get("plan") or {}
    max_el = int(plan.get("max_el_recommend", MAX_EL_RECOMMEND) if max_el is None else max_el)
    max_cores = float(plan.get("max_cores_recommend", MAX_CORES_RECOMMEND) if max_cores is None else max_cores)
    cells = list(summary.get("grid_cells") or [])
    scenes = list((summary.get("scenes") or {}).get("grid") or [])
    if not scenes:
        scenes = list(dict.fromkeys(c["scene"] for c in cells if c.get("scene")))
    rows = grid_table(cells, scenes) if scenes else []
    return {"grid_table": rows, "recommended": recommend(rows, max_el),
            "recommended_measured": recommend_measured(rows, max_cores),
            "cores_per_env": {f"E{r['E']}xL{r['L']}": {"cgroup": r["cores_per_env"], "proc": r["proc_cores_per_env"]}
                              for r in rows if r["ok"]},
            "limits": {"max_el": max_el, "max_cores": max_cores}}


def summarize_det(rows: Sequence[Mapping[str, Any]], path_tests: Mapping[str, Any],
                  reload_tests: Mapping[str, Any], reach_reload: Mapping[str, Any],
                  repeat_identical: Optional[bool]) -> Dict[str, Any]:
    """Per-pose determinism: one sha256 per (scene, pose) over every L value, worker and reload."""
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for r in rows:
        if r.get("sha256"):
            groups.setdefault((r["scene"], r["pose"]), []).append(r)
    per_pose = []
    for (scene, i), rs in sorted(groups.items()):
        per_pose.append({"scene": scene, "pose": i, "n": len(rs), "Ls": sorted({r["L"] for r in rs}),
                         "workers": sorted({r["wid"] for r in rs}), "distinct_sha256": len({r["sha256"] for r in rs}),
                         "equal": len({r["sha256"] for r in rs}) == 1})

    def flag(pred_rows, key):
        rel = [p for p in pred_rows if len(p[key]) > 1]
        return None if not rel else all(p["equal"] for p in rel)

    def measured_all(tests):  # a test whose equal is None (no valid move, a worker error) is unmeasured
        vals = [t.get("equal") for t in tests.values() if t.get("equal") is not None]
        return None if not vals else all(v is True for v in vals)

    across_L = flag(per_pose, "Ls")
    across_proc = flag(per_pose, "workers")
    paths_eq = measured_all(path_tests)
    reload_eq = measured_all(reload_tests)
    reach_eq = None if not reach_reload else all(v is True for v in reach_reload.values())
    flags = {"equal_across_L": across_L, "equal_across_processes": across_proc, "equal_across_paths": paths_eq,
             "equal_after_reload": reload_eq, "reach_equal_after_reload": reach_eq,
             "repeat_render_identical": repeat_identical}
    measured = [v for v in flags.values() if v is not None]
    return {"per_pose": per_pose, "paths": dict(path_tests), "reload": dict(reload_tests),
            "reach_reload_p1": dict(reach_reload), **flags,
            "deterministic": (all(measured) if measured else None),
            "n_flags_measured": len(measured),
            "note": "a pose -> frame cache (and a demo-replay E x L that differs from eval's) is safe only if "
                    "every flag is true"}


# --- the worker (runs in its own process, inside the job) ---------------------------------------------------------
def unity_pid(thor_env) -> Optional[int]:
    try:
        return int(thor_env.controller.server.unity_proc.pid)
    except (AttributeError, TypeError, ValueError):
        return None


def proc_info(pid: Optional[int]) -> Dict[str, Any]:
    """Threads and CPU seconds (utime + stime) of a process from /proc; {} when unavailable."""
    if not pid:
        return {}
    out: Dict[str, Any] = {"pid": pid}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    out["threads"] = int(line.split()[1])
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        tck = os.sysconf("SC_CLK_TCK")
        out["cpu_s"] = round((int(fields[11]) + int(fields[12])) / tck, 3)
    except (OSError, ValueError, IndexError):
        pass
    return out


def self_cpu_s() -> float:
    t = os.times()
    return t.user + t.system


class Worker:
    """One THOR worker: TVRBench's ThorViewEnv built by lvp.make_view_env (official eval settings).

    `tv` is the thor_view_env module (tests pass a fake).  handle(cmd) runs one op and returns its reply dict.
    """

    def __init__(self, tv, *, wid: int = 0, houses_dir=None, procthor_dir=lvp.PROCTHOR_DIR, font=None,
                 font_mode: Optional[str] = None, max_steps: int = lvp.ENV_MAX_STEPS,
                 environ: Mapping[str, str] = os.environ):
        self.tv, self.wid, self.houses_dir = tv, wid, houses_dir
        self.procthor_dir, self.font, self.font_mode = procthor_dir, font, font_mode
        self.max_steps, self.environ = max_steps, environ
        self.env = None

    # -- life cycle
    def start(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.env = lvp.make_view_env(self.tv, procthor_dir=self.procthor_dir, max_steps=self.max_steps,
                                     environ=self.environ)
        dt = time.perf_counter() - t0
        te = self.env._env
        return {"event": "ready", "startup_s": round(dt, 3), "pid": os.getpid(), "wid": self.wid,
                "lp": self.environ.get("LP_NUM_THREADS"), "env": lvp.env_summary(self.environ),
                "font": self.font, "font_mode": self.font_mode, "settings": lvp.controller_settings(te),
                "unity": proc_info(unity_pid(te)), "scene": self.env._scene}

    def close(self) -> None:
        if self.env is not None:
            self.env._sync_close()
            self.env = None

    def handle(self, cmd: Mapping[str, Any]) -> Dict[str, Any]:
        fn = getattr(self, "op_" + str(cmd.get("op")), None)
        if fn is None:
            raise ValueError(f"unknown op {cmd.get('op')!r}")
        return fn(cmd)

    # -- helpers
    def _teleport(self, pose: Mapping[str, Any]):
        return self.env._env.reset(position=pose["position"], rotation_y=pose["rotation_y"],
                                   horizon=pose["horizon"])

    # -- ops
    def op_load(self, cmd):
        s = lvp.load_scene(self.env, cmd["scene"], self.houses_dir)
        return {"scene": self.env._scene, "load_s": round(s, 3)}

    def op_reach(self, cmd):
        out: Dict[str, Any] = {}
        if cmd.get("load"):
            out["load_s"] = round(lvp.load_scene(self.env, cmd["scene"], self.houses_dir), 3)
        elif cmd.get("scene") not in (None, self.env._scene):
            raise RuntimeError(f"scene {cmd.get('scene')} is not loaded (have {self.env._scene})")
        te = self.env._env
        te._reachable_positions = None  # ThorEnv caches: force a fresh GetReachablePositions
        te._reachable_set = None
        out["call_pose"] = te.get_state()  # GetReachablePositions floods from the agent's position (inference)
        t0 = time.perf_counter()
        pos = te.get_reachable_positions()
        out["reach_s"] = round(time.perf_counter() - t0, 3)
        out.update(scene=self.env._scene, positions=list(pos), **reach_digest(pos))
        return out

    def op_render(self, cmd):
        rows = []
        for p in cmd["poses"]:
            try:
                f = self._teleport(p)
                rows.append({"sha256": frame_sha(f), "std": round(float(f.std()), 3), "shape": list(f.shape),
                             "state": self.env._env.get_state()})
            except Exception as e:  # noqa: BLE001
                rows.append({"error": repr(e)})
        return {"scene": self.env._scene, "frames": rows}

    def op_seq(self, cmd):
        te = self.env._env
        self._teleport(cmd["start"])
        v0 = te.get_valid_actions(include_stop=False)
        succ = []
        for a in cmd["actions"]:
            _, ok = te.step(a)
            succ.append(bool(ok))
        f = te.get_observation()
        return {"valid_start": v0, "succ": succ, "state": te.get_state(), "sha256": frame_sha(f),
                "valid_end": te.get_valid_actions(include_stop=False)}

    def op_steps(self, cmd):
        import numpy as np

        te = self.env._env
        self._teleport(cmd["start"])
        rng = random.Random(int(cmd.get("seed", 0)))
        valid = te.get_valid_actions(include_stop=False)
        n = int(cmd["n"])
        u0, s0 = proc_info(unity_pid(te)).get("cpu_s"), self_cpu_s()
        go = cmd.get("go_at")
        if go is not None:
            while True:
                left = float(go) - time.time()
                if left <= 0:
                    break
                time.sleep(min(left, 0.05))
        t_go = time.time()
        lat, ctrl, post = [], [], []
        n_ok = bad = 0
        counts: Dict[str, int] = {}
        for _ in range(n):
            a = rng.choice(valid)
            t0 = time.perf_counter()
            frame, ok = te.step(a)
            t1 = time.perf_counter()
            img = self.tv.burn_label(frame, "CURRENT")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            valid = te.get_valid_actions(include_stop=False)
            t2 = time.perf_counter()
            lat.append(round((t2 - t0) * 1e3, 2))
            ctrl.append(round((t1 - t0) * 1e3, 2))
            post.append(round((t2 - t1) * 1e3, 2))
            n_ok += bool(ok)
            counts[a] = counts.get(a, 0) + 1
            if frame is None or tuple(frame.shape) != (lvp.H, lvp.W, 3) or float(np.asarray(frame).std()) < 2.0:
                bad += 1
        t_end = time.time()
        u1 = proc_info(unity_pid(te)).get("cpu_s")
        return {"n": n, "t_go": t_go, "t_end": t_end, "wall_s": round(t_end - t_go, 4), "lat_ms": lat,
                "ctrl_ms": ctrl, "post_ms": post, "n_ok": n_ok, "bad_frames": bad, "actions": counts,
                "unity_cpu_s": (round(u1 - u0, 3) if (u0 is not None and u1 is not None) else None),
                "worker_cpu_s": round(self_cpu_s() - s0, 3), "unity": proc_info(unity_pid(te))}

    def op_badge(self, cmd):
        import numpy as np

        z = np.zeros((lvp.H, lvp.W, 3), dtype=np.uint8)
        out = {"font_mode": self.font_mode}
        for label in ("CURRENT", "TARGET"):
            img = np.asarray(self.tv.burn_label(z, label).convert("RGB"))
            ys, xs = np.nonzero(img.any(axis=2))
            out[label] = {"bbox_nonzero": ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                                           if ys.size else None),
                          "bbox_bg": badge_bbox(img)}
        return out

    def op_replay(self, cmd):
        tv, env = self.tv, self.env
        task, acts = cmd["task"], list(cmd["actions"])
        d = Path(cmd["save_dir"])
        d.mkdir(parents=True, exist_ok=True)
        lvp.resolve_scene(env, task["scene"], self.houses_dir)  # primes TVRBench's ProcTHOR cache
        t0 = time.perf_counter()
        obs, _ = env._sync_reset(task)
        reset_s = time.perf_counter() - t0
        cur, tgt = obs["multi_modal_input"][tv.IMAGE_PH]
        cur.save(d / "step_00.png", format="PNG")
        tgt.save(d / "target.png", format="PNG")
        turns = [{"k": 0, "user_text": obs["obs_str"], "state": env._env.get_state()}]
        stop: Dict[str, Any] = {}
        t1 = time.perf_counter()
        for k, a in enumerate(acts):
            obs, _, done, info = env._sync_step(f"Action: {a}")
            turns[k].update(action=a, executed=info.get("action_executed"), action_success=info.get("action_success"))
            if a == "Stop":
                stop = {"k": k, "success": info.get("success"), "stopped": info.get("stopped")}
                break
            cur = obs["multi_modal_input"][tv.IMAGE_PH][0]
            cur.save(d / f"step_{k + 1:02d}.png", format="PNG")
            turns.append({"k": k + 1, "user_text": obs["obs_str"], "state": env._env.get_state()})
            if done:
                stop = {"k": k, "truncated": True}
                break
        steps_s = time.perf_counter() - t1
        final = env._env.get_state()
        exact = bool(tv.is_exact_match(final, task["target"]))
        # repeat control: the same task reset again (scene already loaded) renders the same two poses
        t2 = time.perf_counter()
        obs2, _ = env._sync_reset(task)
        cur2, tgt2 = obs2["multi_modal_input"][tv.IMAGE_PH]
        cur2.save(d / "rep_step_00.png", format="PNG")
        tgt2.save(d / "rep_target.png", format="PNG")
        return {"task_id": task.get("task_id"), "scene": task["scene"], "reset_s": round(reset_s, 3),
                "steps_s": round(steps_s, 3), "repeat_reset_s": round(time.perf_counter() - t2, 3),
                "turns": turns, "stop": stop, "final_state": final, "final_exact": exact,
                "system_prompt": tv.SYSTEM_PROMPT}


def dispatch(worker: Worker, cmd: Mapping[str, Any]) -> Dict[str, Any]:
    """worker.handle with the reply envelope {id, ok, error, tb}."""
    try:
        r = dict(worker.handle(cmd))
        r.setdefault("ok", True)
    except Exception as e:  # noqa: BLE001
        r = {"ok": False, "error": repr(e), "tb": traceback.format_exc()[-3000:]}
    r["id"] = cmd.get("id")
    return r


def worker_main(a) -> int:
    reply = os.fdopen(a.reply_fd, "w", buffering=1)
    cmds = os.fdopen(a.cmd_fd, "r")

    def send(obj):
        reply.write(json.dumps(jsafe(obj)) + "\n")
        reply.flush()

    try:
        _, tv = lvp.import_tvrbench(a.repo)
        # the probe masks badges, so a missing / other font is reported (summary.font.ok false), not fatal
        font = lvp.install_font_rule(tv, allow_missing=True)
        w = Worker(tv, wid=a.wid, houses_dir=a.houses, procthor_dir=a.procthor_dir, font=font,
                   font_mode=font["mode"])
        send(dict(w.start(), id=0, ok=True))
    except BaseException as e:  # noqa: BLE001
        send({"id": 0, "ok": False, "error": repr(e), "tb": traceback.format_exc()[-3000:]})
        return 1
    rc = 0
    for line in cmds:
        try:
            cmd = json.loads(line)
        except ValueError:
            send({"id": None, "ok": False, "error": "bad command line"})
            continue
        if cmd.get("op") == "quit":
            try:
                w.close()
                send({"id": cmd.get("id"), "ok": True, "event": "bye"})
            except Exception as e:  # noqa: BLE001
                send({"id": cmd.get("id"), "ok": False, "error": repr(e)})
                rc = 1
            break
        send(dispatch(w, cmd))
    else:
        w.close()
    return rc


# --- worker handles (parent side) ------------------------------------------------------------------------------
class SubprocessHandle:
    """A worker process: commands on one pipe, replies on another, its own session (killpg reaps Unity too)."""

    def __init__(self, wid: int, L: int, argv: List[str], env: Mapping[str, str], cwd, log_path):
        self.wid, self.L = wid, L
        self.scene: Optional[str] = None
        self.probed: set = set()
        self.alive = True
        self.unity_pid: Optional[int] = None
        cr, cw = os.pipe()
        rr, rw = os.pipe()
        self._log = open(log_path, "ab")
        try:
            self.proc = subprocess.Popen(list(argv) + ["--cmd-fd", str(cr), "--reply-fd", str(rw)], env=dict(env),
                                         cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=self._log,
                                         stderr=subprocess.STDOUT, pass_fds=(cr, rw), start_new_session=True)
        except BaseException:
            for fd in (cw, rr):
                os.close(fd)
            self._log.close()
            raise
        finally:
            os.close(cr)
            os.close(rw)
        self._w = os.fdopen(cw, "w", buffering=1)
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._eof = False
        self._next = 0
        self._t = threading.Thread(target=self._read, args=(os.fdopen(rr, "r"),), daemon=True)
        self._t.start()

    @property
    def pid(self) -> int:
        return self.proc.pid

    def _read(self, f) -> None:
        try:
            for line in f:
                try:
                    self._q.put(json.loads(line))
                except ValueError:
                    self._q.put({"id": None, "ok": False, "error": "bad reply line", "line": line[:200]})
        finally:
            self._q.put({"id": None, "eof": True})

    def send(self, cmd: Mapping[str, Any]) -> int:
        self._next += 1
        cid = self._next
        try:
            self._w.write(json.dumps(jsafe(dict(cmd, id=cid))) + "\n")
            self._w.flush()
        except (OSError, ValueError) as e:
            self.alive = False
            self._q.put({"id": cid, "ok": False, "error": f"send failed: {e!r}", "died": True})
        return cid

    def wait(self, cid: int, timeout: float) -> Dict[str, Any]:
        end = time.monotonic() + max(0.0, timeout)
        while True:
            if self._eof:
                return {"id": cid, "ok": False, "error": f"worker exited rc={self.proc.poll()}", "died": True}
            left = end - time.monotonic()
            try:
                r = self._q.get(timeout=max(0.0, min(left, 0.5))) if left > 0 else self._q.get_nowait()
            except queue.Empty:
                if left <= 0:
                    return {"id": cid, "ok": False, "error": "timeout", "timeout": True}
                continue
            if r.get("eof"):
                self._eof = True
                self.alive = False
                continue
            if r.get("id") == cid:
                return r
            # a stale reply (of a command that timed out before): dropped

    def kill(self) -> None:
        self.alive = False
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for f in (self._w, self._log):
            try:
                f.close()
            except OSError:
                pass

    def quit(self, timeout: float = QUIT_TIMEOUT_S) -> Dict[str, Any]:
        r = self.wait(self.send({"op": "quit"}), timeout) if self.alive else {"ok": False, "error": "not alive"}
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.kill()  # the session may still hold a Unity player
        return r


class LocalHandle:
    """In-process stand-in for SubprocessHandle (tests): send() runs the op at once, wait() returns its reply."""

    def __init__(self, wid: int, L: int, worker: Worker):
        self.wid, self.L, self.worker = wid, L, worker
        self.scene: Optional[str] = None
        self.probed: set = set()
        self.alive = True
        self.unity_pid: Optional[int] = None
        self.pid = os.getpid()
        self._next = 0
        self._replies: Dict[int, Dict[str, Any]] = {}

    def send(self, cmd: Mapping[str, Any]) -> int:
        self._next += 1
        cid = self._next
        if cmd.get("op") == "quit":
            self.worker.close()
            self.alive = False
            self._replies[cid] = {"id": cid, "ok": True, "event": "bye"}
        else:
            self._replies[cid] = jsafe(dispatch(self.worker, dict(cmd, id=cid)))
        return cid

    def wait(self, cid: int, timeout: float) -> Dict[str, Any]:
        return self._replies.pop(cid, {"id": cid, "ok": False, "error": "timeout", "timeout": True})

    def kill(self) -> None:
        self.alive = False

    def quit(self, timeout: float = QUIT_TIMEOUT_S) -> Dict[str, Any]:
        return self.wait(self.send({"op": "quit"}), timeout) if self.alive else {"ok": True}


class SubprocessSpawner:
    """spawn(wid, L, profile) -> (handle, ready reply): a worker process with lvp.worker_env(profile=...) and its
    own HOME.  The reply also carries env_profile and spawn_env (the render-relevant environment it was given),
    also when the start-up failed."""

    def __init__(self, out, *, py: str = sys.executable, repo=lvp.TVR_REPO, houses_dir=None,
                 procthor_dir=lvp.PROCTHOR_DIR, code_root=None, base_env: Optional[Mapping[str, str]] = None,
                 startup_timeout: float = STARTUP_TIMEOUT_S, tvr_env=lvp.TVR_ENV, releases=lvp.RELEASES):
        self.out = Path(out)
        self.py, self.repo, self.houses_dir, self.procthor_dir = py, repo, houses_dir, procthor_dir
        self.code_root = Path(code_root) if code_root is not None else paths.WORKTREE
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.startup_timeout = startup_timeout
        self.tvr_env, self.releases = Path(tvr_env), releases

    def __call__(self, wid: int, L: int, profile: str = lvp.LVP_PROFILES[0]):
        home = self.out / "homes" / f"w{wid:02d}_L{L}"
        lvp.prepare_home(home, releases=self.releases)
        env = lvp.worker_env(self.base_env, lp_threads=L, home=home, tvr_env=self.tvr_env, profile=profile)
        pp = [p for p in self.base_env.get("PYTHONPATH", "").split(os.pathsep) if p and p != str(self.code_root)]
        env["PYTHONPATH"] = os.pathsep.join([str(self.code_root)] + pp)
        argv = [self.py, "-s", "-u", "-m", "vpd.tvr.probe_lvp", "worker", "--wid", str(wid), "--repo",
                str(self.repo), "--procthor-dir", str(self.procthor_dir)]
        if self.houses_dir is not None:
            argv += ["--houses", str(self.houses_dir)]
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        h = SubprocessHandle(wid, L, argv, env, self.code_root, self.out / "logs" / f"worker_w{wid:02d}_L{L}.log")
        r = dict(h.wait(0, self.startup_timeout), env_profile=profile, spawn_env=lvp.env_summary(env))
        if not r.get("ok"):
            h.kill()
        return h, r


# --- the parent --------------------------------------------------------------------------------------------------
class PoolError(RuntimeError):
    pass


class Probe:
    """Phases P0-P4 (module doc).  `spawner(wid, L, profile) -> (handle, ready)`; clocks injectable for tests.
    `font` / `font_sha256` are the pinned badge font the workers install (lvp.install_font_rule)."""

    def __init__(self, out, *, deadline_s: float, spawner, grid: Optional[Mapping[int, Sequence[int]]] = None,
                 steps: int = STEPS, p1_l: int = P1_L, repo=lvp.TVR_REPO, tasks_dir=None, sample_dir=SAMPLE_DIR,
                 sft_train=SFT_TRAIN, procthor_src=lvp.PROCTHOR_TRAIN, icd=lvp.LVP_ICD,
                 sample_tasks: Sequence[str] = SAMPLE_TASKS, reach_scenes: Optional[Sequence[str]] = None,
                 cpu_stat=lvp.CPU_STAT, cpu_max=lvp.CPU_MAX, clock=time.monotonic, wall=time.time,
                 sleep=time.sleep, font=lvp.BADGE_FONT, font_sha256: str = lvp.BADGE_FONT_SHA256):
        self.out = Path(out)
        self.deadline_s = float(deadline_s)
        self.spawner = spawner
        self.grid = dict(GRID_DEFAULT if grid is None else grid)
        self.font, self.font_sha256 = font, font_sha256
        # LD_LIBRARY_PATH profile: decided by the first worker's start-up (spawn)
        self.lvp_profile: Optional[str] = None
        self.profile_next = lvp.LVP_PROFILES[0]
        self.profile_retried = False
        self.profile_attempts: List[Dict[str, Any]] = []
        self.p2_cells = cell_order({L: E for L, E in self.grid.items() if L == REACH_L})
        self.p4_cells = cell_order({L: E for L, E in self.grid.items() if L != REACH_L})
        self.steps, self.p1_l = int(steps), int(p1_l)
        self.repo = Path(repo)
        self.tasks_dir = Path(tasks_dir) if tasks_dir is not None else self.repo / "data" / "tasks"
        self.sample_dir, self.sft_train, self.procthor_src = Path(sample_dir), Path(sft_train), Path(procthor_src)
        self.icd, self.reach_override = icd, reach_scenes
        self.sample_tasks = tuple(sample_tasks)
        self.cpu_stat, self.cpu_max = cpu_stat, cpu_max
        self.clock, self.wall, self.sleep = clock, wall, sleep
        self.t0 = clock()
        self.pool: List[Any] = []
        self.pool_L: Optional[int] = None
        self.next_wid = 0
        self.probe_poses: Dict[str, List[Dict[str, Any]]] = {}
        self.det_rows: List[Dict[str, Any]] = []
        self.load_rows: List[Dict[str, Any]] = []
        self.path_tests: Dict[str, Any] = {}
        self.reload_tests: Dict[str, Any] = {}
        self.reach_reload: Dict[str, Any] = {}
        self.cells: List[Dict[str, Any]] = []
        self.cell_wall: Dict[Tuple[int, str], float] = {}
        self.ready_to_render = False
        self.grid_scenes: List[str] = []
        self.first_reach: Dict[str, Dict[str, Any]] = {}
        self.eval_task_by_scene: Dict[str, Dict[str, Any]] = {}
        self.sft_by_id: Dict[str, Dict[str, Any]] = {}
        self.public: Dict[str, Dict[str, Any]] = {}
        self.sft_scenes: List[str] = []
        self.eval_scenes: List[str] = []
        self.reach_scenes: List[str] = []
        self.houses_dir = self.out / "houses"
        self.go_delay_s = GO_DELAY_S
        self.res: Dict[str, Any] = {
            "schema": SCHEMA, "label": LABEL, "purpose": PURPOSE, "utc_start": utc(), "deadline_s": self.deadline_s,
            "plan": {"grid": {str(k): list(v) for k, v in self.grid.items()}, "p1_cell": [1, self.p1_l],
                     "order_EL": {"p2": [[E, L] for L, E in self.p2_cells], "p4": [[E, L] for L, E in self.p4_cells]},
                     "steps_per_env": self.steps, "settings": lvp.EVAL_SETTINGS,
                     "max_el_recommend": MAX_EL_RECOMMEND, "max_cores_recommend": MAX_CORES_RECOMMEND,
                     "hi_el": HI_EL, "reach_pool": [REACH_E, REACH_L], "lvp_profiles": lvp.LVP_PROFILE_DOC,
                     "badge_font": {"path": str(font), "sha256": font_sha256}},
            "phases": {}, "errors": [], "workers": []}

    # -- bookkeeping
    def left(self) -> float:
        return self.deadline_s - (self.clock() - self.t0)

    def save(self) -> None:
        self.res["elapsed_s"] = round(self.clock() - self.t0, 2)
        self.res["utc_saved"] = utc()
        self.res["grid_cells"] = self.cells
        write_json_atomic(self.out / "summary.json", self.res)

    def err(self, where: str, msg: Any) -> None:
        self.res["errors"].append({"where": where, "error": str(msg)[:2000], "utc": utc()})
        log(f"ERROR {where}: {str(msg)[:300]}")

    def phase(self, name: str, fn: Callable[[], Any]) -> None:
        t = self.clock()
        log(f"phase {name} start, {self.left():.0f} s left")
        self.res["phases"][name] = {"status": "running"}
        st = "ok"
        try:
            fn()
        except Terminated:
            self.res["phases"][name] = {"status": "terminated", "secs": round(self.clock() - t, 2)}
            raise
        except Exception as e:  # noqa: BLE001
            self.err(name, f"{e!r}\n{traceback.format_exc()[-2000:]}")
            st = "error"
        self.res["phases"][name] = {"status": st, "secs": round(self.clock() - t, 2)}
        self.save()

    def guarded(self, where: str, fn: Callable[[], Any], on_error: Optional[Callable[[str], None]] = None) -> None:
        """Run one P1 sub-test; a PoolError is recorded (errors[], `on_error(repr)`) and the phase goes on."""
        try:
            fn()
        except PoolError as e:
            self.err(where, repr(e))
            if on_error is not None:
                on_error(repr(e))

    # -- workers
    def _spawn_once(self, L: int, profile: str):
        wid = self.next_wid
        self.next_wid += 1
        t = self.clock()
        h, ready = self.spawner(wid, L, profile)
        info = {"wid": wid, "L": L, "ok": bool(ready.get("ok")), "spawn_s": round(self.clock() - t, 2),
                "env_profile": profile,
                **{k: ready.get(k) for k in ("startup_s", "pid", "lp", "env", "spawn_env", "font", "font_mode",
                                             "settings", "unity", "error")}}
        self.res["workers"].append(info)
        return h, ready, info

    def spawn(self, L: int):
        """Start one worker.  The LD_LIBRARY_PATH profile: until a worker has started, lvp.LVP_PROFILES[0]; if
        that first start-up fails, ONE retry with LVP_PROFILES[1] (the measured profile), which every later
        worker then uses unless the first one worked (summary.lvp_profile)."""
        if self.left() < STARTUP_EST_S + SUMMARY_RESERVE_S:
            raise PoolError("no time to start a worker")
        prof = self.lvp_profile or self.profile_next
        h, ready, info = self._spawn_once(L, prof)
        first_sequence = self.lvp_profile is None and not self.profile_retried
        if first_sequence:
            self.profile_attempts.append({k: info.get(k) for k in ("wid", "L", "env_profile", "ok", "error")})
        if not ready.get("ok") and first_sequence and len(lvp.LVP_PROFILES) > 1:
            self.profile_retried = True
            self.profile_next = lvp.LVP_PROFILES[1]
            self.err("spawn.profile", f"worker {info['wid']} did not start with lavapipe profile {prof}: "
                                      f"{ready.get('error')}; retrying once with {self.profile_next}")
            if self.left() >= STARTUP_EST_S + SUMMARY_RESERVE_S:
                h, ready, info = self._spawn_once(L, self.profile_next)
                self.profile_attempts.append({k: info.get(k) for k in ("wid", "L", "env_profile", "ok", "error")})
        if ready.get("ok") and self.lvp_profile is None:
            self.lvp_profile = prof = info["env_profile"]
            env = ready.get("env") or ready.get("spawn_env") or {}
            self.res["lvp_profile"] = {
                "profile": prof, "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH"), "doc": lvp.LVP_PROFILE_DOC[prof],
                "first_try": lvp.LVP_PROFILES[0], "retried": self.profile_retried,
                "attempts": self.profile_attempts,
                "freeze": "the demo replay and eval build their workers with lvp.worker_env(profile=<profile>)"}
        if not ready.get("ok"):
            raise PoolError(f"worker {info['wid']} (L={L}, profile {info['env_profile']}) did not start: "
                            f"{ready.get('error')}")
        h.scene = ready.get("scene")
        h.unity_pid = (ready.get("unity") or {}).get("pid")
        return h

    def stop_pool(self, graceful: bool = True) -> None:
        """quit to every live worker at once, one shared QUIT_TIMEOUT_S, then kill every session (Unity too)."""
        hs, self.pool, self.pool_L = list(self.pool), [], None
        sent = []
        if graceful:
            for h in hs:
                if h.alive:
                    try:
                        sent.append((h, h.send({"op": "quit"})))
                    except Exception as e:  # noqa: BLE001
                        self.err("stop_pool", repr(e))
            end = self.clock() + QUIT_TIMEOUT_S
            for h, cid in sent:
                h.wait(cid, max(0.0, end - self.clock()))
        for h in hs:
            try:
                h.kill()
            except Exception as e:  # noqa: BLE001
                self.err("stop_pool", repr(e))

    def bounded(self, timeout: float) -> float:
        """A command timeout that never runs past the soft deadline minus the summary reserve (>= 1 s)."""
        return max(1.0, min(float(timeout), self.left() - SUMMARY_RESERVE_S))

    def ensure_pool(self, L: int, E: int) -> List[Any]:
        """E live workers at L lavapipe threads (start-ups serialised); other L's workers are stopped first."""
        if self.pool_L != L:
            self.stop_pool()
            self.pool_L = L
        self.pool = [h for h in self.pool if h.alive]
        while len(self.pool) < E:
            self.pool.append(self.spawn(L))
        return self.pool[:E]

    def call(self, h, cmd: Mapping[str, Any], timeout: float = CMD_TIMEOUT_S) -> Dict[str, Any]:
        r = h.wait(h.send(cmd), self.bounded(timeout))
        if r.get("timeout") or r.get("died"):
            h.kill()
        if not r.get("ok"):
            raise PoolError(f"w{h.wid} {cmd.get('op')}: {r.get('error')}")
        return r

    def call_many(self, hs: Sequence[Any], cmds: Sequence[Mapping[str, Any]], timeout: float) -> List[Dict]:
        ids = [h.send(c) for h, c in zip(hs, cmds)]
        end = self.clock() + self.bounded(timeout)
        out = []
        for h, cid in zip(hs, ids):
            r = h.wait(cid, max(0.0, end - self.clock()))
            if r.get("timeout") or r.get("died"):
                h.kill()
            out.append(r)
        return out

    @staticmethod
    def idle_cpu_s(hs: Sequence[Any]) -> Optional[float]:
        """CPU seconds so far of the workers `hs` (worker process + its Unity player, from /proc); None when there
        is none or any process is unreadable."""
        if not hs:
            return None
        tot = 0.0
        for h in hs:
            upid = getattr(h, "unity_pid", None)
            if not upid:
                return None
            for pid in (h.pid, upid):
                c = proc_info(pid).get("cpu_s")
                if c is None:
                    return None
                tot += c
        return tot

    # -- scenes, probe poses
    def load_on(self, hs: Sequence[Any], scene: str, force: bool = False, phase: str = "") -> None:
        """Load `scene` on the workers of `hs` that do not have it (all at once), then render the probe poses on
        every worker of `hs` that has not rendered them in this scene yet, also all at once (call_many)."""
        need = [h for h in hs if force or h.scene != scene]
        bad = []
        if need:
            rs = self.call_many(need, [{"op": "load", "scene": scene}] * len(need), LOAD_TIMEOUT_S)
            for h, r in zip(need, rs):
                if not r.get("ok"):
                    bad.append(f"w{h.wid} load {scene}: {r.get('error')}")
                    continue
                h.scene = scene
                self.load_rows.append({"scene": scene, "kind": lvp.scene_kind(scene), "wid": h.wid, "L": h.L,
                                       "load_s": r["load_s"], "concurrent": len(need), "phase": phase})
        if bad:
            raise PoolError("; ".join(bad))
        todo = [h for h in hs if scene not in h.probed]
        if todo:
            poses = self.ensure_probe_poses(todo[0], scene)
            rs = self.call_many(todo, [{"op": "render", "poses": poses}] * len(todo), CMD_TIMEOUT_S)
            for h, r in zip(todo, rs):
                if not r.get("ok"):
                    bad.append(f"w{h.wid} render {scene}: {r.get('error')}")
                    continue
                self.probe_rows(h, scene, r["frames"], tag="first")
                h.probed.add(scene)
        if bad:
            raise PoolError("; ".join(bad))

    def ensure_probe_poses(self, h, scene: str) -> List[Dict[str, Any]]:
        if scene not in self.probe_poses:
            r = self.call(h, {"op": "reach", "scene": scene, "load": False})
            t = self.eval_task_by_scene.get(scene)
            extra = [t["start"], t["target"]] if t is not None else []
            self.probe_poses[scene] = pick_probe_poses(r["positions"], PROBE_K, extra)
            self.first_reach.setdefault(scene, {k: r.get(k) for k in ("n", "sha256_raw", "sha256_xz", "call_pose")})
        return self.probe_poses[scene]

    def probe_rows(self, h, scene: str, frames: Sequence[Mapping[str, Any]], tag: str) -> List[Dict[str, Any]]:
        rows = []
        for i, f in enumerate(frames):
            row = {"scene": scene, "pose": i, "L": h.L, "wid": h.wid, "tag": tag, "sha256": f.get("sha256"),
                   "std": f.get("std"), "error": f.get("error")}
            rows.append(row)
            if tag == "first":
                self.det_rows.append(row)
        return rows

    def render_probes(self, h, scene: str, tag: str) -> List[Dict[str, Any]]:
        poses = self.ensure_probe_poses(h, scene)
        r = self.call(h, {"op": "render", "poses": poses})
        return self.probe_rows(h, scene, r["frames"], tag)

    # -- P0
    def p0_preflight(self) -> None:
        r = self.res
        r["host"] = {"hostname": socket.gethostname(), "uid": os.getuid(), "cpu_max": lvp.read_cpu_max(self.cpu_max),
                     "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                     "python": sys.version.split()[0]}
        snap = paths.WORKTREE / "SNAPSHOT.json"
        r["code"] = {"worktree": str(paths.WORKTREE), "code_root": str(Path(__file__).resolve().parents[2]),
                     "snapshot": json.loads(snap.read_text()) if snap.is_file() else None,
                     "VPD_WORKTREE": os.environ.get("VPD_WORKTREE")}
        r["icd"] = lvp.check_icd(self.icd)
        r["font"] = {"parent": lvp.font_status(self.font, self.font_sha256)}
        r["sample"] = verify_sample(self.sample_dir)
        sft = lvp.load_tasks("sft", self.tasks_dir)
        ev = lvp.load_tasks("eval", self.tasks_dir)
        self.sft_by_id = {t["task_id"]: t for t in sft}
        self.sft_scenes, self.eval_scenes = lvp.scenes_of(sft), lvp.scenes_of(ev)
        pt_task = next(t for t in ev if lvp.is_procthor(t["scene"]))
        self.grid_scenes = [FIRST_GRID_SCENE, pt_task["scene"]]
        self.eval_task_by_scene = {pt_task["scene"]: pt_task}
        reach = list(self.reach_override) if self.reach_override is not None else \
            self.sft_scenes + [s for s in self.eval_scenes if s not in self.sft_scenes]
        self.reach_scenes = reach
        sample_scenes = [self.sft_by_id[t]["scene"] for t in self.sample_tasks if t in self.sft_by_id]
        need = [s for s in dict.fromkeys(reach + self.grid_scenes + sample_scenes) if lvp.is_procthor(s)]
        t = self.clock()
        try:
            houses = lvp.extract_procthor_houses(self.procthor_src, need, self.houses_dir)
            r["houses"] = {"src": str(self.procthor_src), "dir": str(self.houses_dir), "n": len(houses),
                           "secs": round(self.clock() - t, 2),
                           "sha256": {k: v["sha256"] for k, v in houses.items()}}
        except (OSError, ValueError) as e:  # the workers then load through TVRBench's own gz scan
            r["houses"] = {"src": str(self.procthor_src), "error": repr(e)}
            self.err("p0.houses", repr(e))
        r["scenes"] = {"grid": self.grid_scenes, "n_sft": len(self.sft_scenes), "n_eval": len(self.eval_scenes),
                       "reach_planned": len(reach)}
        try:
            self.public = load_public(self.sft_train, self.sample_tasks)
        except (OSError, ValueError) as e:
            self.public = {}
            self.err("p0.public", repr(e))
        self.ready_to_render = bool(r["icd"]["ok"])
        if not self.ready_to_render:
            self.err("p0.icd", r["icd"]["why"])

    # -- P1
    def p1_worker(self):
        """The live P1 worker (L = p1_l); a dead one is replaced (one more start-up)."""
        return self.ensure_pool(self.p1_l, 1)[0]

    def p1_compare_det(self) -> None:
        self.p1_worker()  # the start-up (fixes the lvp profile); without a worker P1 has nothing to measure
        self.compare_public()  # G3(d) first: no determinism-test error can cost the comparison
        self.save()
        self.guarded("p1.badge", self.p1_badge)
        for scene in self.grid_scenes:
            self.guarded(f"p1.path.{scene}", lambda s=scene: self.p1_path(s),
                         lambda e, s=scene: self.path_tests.__setitem__(s, {"equal": None, "error": e}))
        for scene in self.grid_scenes:  # reload each scene (another scene was loaded in between)
            self.guarded(f"p1.reload.{scene}", lambda s=scene: self.p1_reload(s),
                         lambda e, s=scene: self.reload_tests.__setitem__(s, {"equal": None, "error": e}))
        self.save()
        for k, scene in enumerate(self.grid_scenes):
            self.grid_step(self.p1_l, 1, scene, k, reserve=SUMMARY_RESERVE_S)

    def p1_badge(self) -> None:
        b = self.call(self.p1_worker(), {"op": "badge"})
        self.res["font"]["badge_tvrbench_burn_label"] = {k: b.get(k) for k in ("font_mode", "CURRENT", "TARGET")}

    def p1_path(self, scene: str) -> None:
        h = self.p1_worker()
        self.load_on([h], scene, force=True, phase="p1")
        self.path_tests[scene] = self.path_test(h, scene)

    def p1_reload(self, scene: str) -> None:
        h = self.p1_worker()
        first = {d["pose"]: d["sha256"] for d in self.det_rows
                 if d["scene"] == scene and d["wid"] == h.wid and d["tag"] == "first"}
        if not first:  # e.g. the worker was replaced after its path test: no reload to compare with
            self.reload_tests[scene] = {"equal": None, "why": f"w{h.wid} has no first render of {scene}"}
            return
        self.load_on([h], scene, force=True, phase="p1_reload")
        rr = self.call(h, {"op": "reach", "scene": scene, "load": False})
        fr = self.first_reach.get(scene, {})
        self.reach_reload[scene] = (rr["sha256_raw"] == fr.get("sha256_raw"))
        rows = self.render_probes(h, scene, tag="reload")
        eq = [bool(r["sha256"]) and r["sha256"] == first.get(r["pose"]) for r in rows]
        self.reload_tests[scene] = {"n": len(eq), "equal": bool(eq) and all(eq), "per_pose": eq,
                                    "call_pose": rr.get("call_pose")}

    def path_test(self, h, scene: str) -> Dict[str, Any]:
        for i, S in enumerate(self.probe_poses[scene]):
            r0 = self.call(h, {"op": "seq", "start": S, "actions": []})
            moves = [a for a in r0["valid_start"] if a.startswith("Move")]
            if not moves:
                continue
            M = moves[0]
            look = ["LookDown", "LookUp"] if "LookDown" in r0["valid_start"] else ["LookUp", "LookDown"]
            A, B = ["RotateRight", "RotateLeft", M], [M] + look
            ra = self.call(h, {"op": "seq", "start": S, "actions": A})
            rb = self.call(h, {"op": "seq", "start": S, "actions": B})
            ra2 = self.call(h, {"op": "seq", "start": S, "actions": A})
            rc = self.call(h, {"op": "render", "poses": [pose_of(ra["state"])]})["frames"][0]
            shas = {"A": ra["sha256"], "B": rb["sha256"], "C_teleport": rc.get("sha256"), "A_repeat": ra2["sha256"]}
            return {"start_pose": i, "A": A, "B": B, "succ": {"A": ra["succ"], "B": rb["succ"]},
                    "states_equal": ra["state"] == rb["state"] == ra2["state"], "sha256": shas,
                    "all_succ": all(ra["succ"]) and all(rb["succ"]),
                    "equal": len(set(shas.values())) == 1 and ra["state"] == rb["state"]}
        return {"equal": None, "error": "no probe pose with a valid move"}

    def compare_public(self) -> None:
        """(f): replay the public sample tasks on the P1 worker (a live one per task) and apply G3(d)."""
        tasks = {}
        n_expected = 0
        for tid in self.sample_tasks:
            traj = self.public.get(tid)
            if traj is None:
                continue
            n_expected += len(traj["turns"]) + 1
            task = self.sft_by_id.get(tid)
            ours = self.out / "compare" / tid
            try:
                if task is None:
                    raise PoolError(f"{tid} not in sft.json")
                h = self.p1_worker()
                rep = self.call(h, {"op": "replay", "task": task, "actions": traj["actions"], "save_dir": str(ours)},
                                REPLAY_TIMEOUT_S)
                self.load_rows.append({"scene": task["scene"], "kind": lvp.scene_kind(task["scene"]), "wid": h.wid,
                                       "L": h.L, "load_s": rep["reset_s"], "concurrent": 1,
                                       "phase": "p1_replay (reset incl. load + 2 renders)"})
                h.scene = task["scene"]
            except PoolError as e:
                self.err(f"p1.replay.{tid}", repr(e))
                rep = {"error": repr(e), "turns": []}
            tasks[tid] = compare_task(self.sample_dir / "images" / tid, ours, traj, rep,
                                      system_prompt=rep.get("system_prompt"))
            tasks[tid]["replay"] = {k: rep.get(k) for k in ("reset_s", "steps_s", "repeat_reset_s", "stop",
                                                            "final_state", "final_exact", "error")}
            tasks[tid]["turn_log"] = [{k: t.get(k) for k in ("k", "action", "executed", "action_success", "state")}
                                      for t in rep.get("turns") or []]
        self.res["compare"] = dict(compare_verdict(tasks, n_expected), sample=self.res.get("sample"), tasks=tasks)
        why = []
        if not (self.res.get("sample") or {}).get("ok"):
            why.append("public sample failed its SHA256SUMS check")
        if len(tasks) < len(self.sample_tasks):
            why.append(f"only {sorted(tasks)} of {list(self.sample_tasks)} have a public trajectory")
        if why:
            self.res["compare"]["verdict"] = "incomplete"
            self.res["compare"]["why"] = why

    # -- grid
    def est_cell(self, L: int, E: int, scene: str) -> float:
        have = len([h for h in self.pool if h.alive]) if self.pool_L == L else 0
        base = self.cell_wall.get((L, scene), CELL_EST_DEFAULT_S)
        return base * 1.25 + max(0, E - have) * STARTUP_EST_S + (10.0 if lvp.is_procthor(scene) else 3.0)

    def run_cell(self, L: int, E: int, scene: str, k: int) -> Dict[str, Any]:
        cell: Dict[str, Any] = {"L": L, "E": E, "EL": E * L, "scene": scene, "kind": lvp.scene_kind(scene),
                                "n_per_env": self.steps, "ok": False, "utc": utc()}
        try:
            hs = self.ensure_pool(L, E)
            self.load_on(hs, scene, phase=f"grid_E{E}_L{L}")
            # live workers of the pool that do not step in this cell (smaller E than the pool): they stay in the
            # cgroup, so their CPU is recorded next to the cell's cgroup cores
            idle = [h for h in self.pool if h.alive and h not in hs]
            cell["idle_workers"] = len(idle)
            poses = self.probe_poses[scene]
            go_at = self.wall() + self.go_delay_s
            cmds = [{"op": "steps", "n": self.steps, "go_at": go_at, "seed": 1000 * k + h.wid,
                     "start": poses[i % len(poses)]} for i, h in enumerate(hs)]
            ids = [h.send(c) for h, c in zip(hs, cmds)]
            while self.wall() < go_at:
                self.sleep(min(0.02, max(0.0, go_at - self.wall())))
            a = lvp.read_cpu_stat(self.cpu_stat)
            idle_a = self.idle_cpu_s(idle)
            end = self.clock() + self.bounded(self.steps * 3.0 + 60.0)
            reps = []
            for h, cid in zip(hs, ids):
                r = h.wait(cid, max(0.0, end - self.clock()))
                if r.get("timeout") or r.get("died"):
                    h.kill()
                reps.append(r)
            t_b = self.wall()
            b = lvp.read_cpu_stat(self.cpu_stat)
            idle_b = self.idle_cpu_s(idle)
            cell["cpu"] = lvp.cpu_delta(a, b, t_b - go_at)
            cell["cpu"]["idle_workers_cpu_s"] = (round(idle_b - idle_a, 3)
                                                 if (idle and idle_a is not None and idle_b is not None) else None)
            bad = [r for r in reps if not r.get("ok")]
            if bad:
                cell["error"] = "; ".join(f"{r.get('error')}" for r in bad)[:1000]
            good = [r for r in reps if r.get("ok")]
            lat = [x for r in good for x in r["lat_ms"]]
            ctrl = [x for r in good for x in r["ctrl_ms"]]
            post = [x for r in good for x in r["post_ms"]]
            if good:
                t_end = max(float(r["t_end"]) for r in good)
                wall = max(t_end - min(float(r["t_go"]) for r in good), 1e-9)
                cell.update(agg_steps_per_s=round(sum(r["n"] for r in good) / wall, 3), wall_s=round(wall, 3),
                            p50_ms=round(percentile(lat, 50), 2), p95_ms=round(percentile(lat, 95), 2),
                            ctrl_p50_ms=round(percentile(ctrl, 50), 2), post_p50_ms=round(percentile(post, 50), 2),
                            bad_frames=sum(r["bad_frames"] for r in good))
                self.cell_wall[(L, scene)] = max(self.cell_wall.get((L, scene), 0.0), wall)
            cell["per_env"] = [{"wid": h.wid, "ok": bool(r.get("ok")), "error": r.get("error"),
                                **({"steps_per_s": round(r["n"] / max(r["wall_s"], 1e-9), 3), "wall_s": r["wall_s"],
                                    "cores": (None if _env_cores(r) is None else round(_env_cores(r), 3)),
                                    "p50_ms": round(percentile(r["lat_ms"], 50), 2),
                                    "p95_ms": round(percentile(r["lat_ms"], 95), 2), "n_ok": r["n_ok"],
                                    "bad_frames": r["bad_frames"], "unity_cpu_s": r.get("unity_cpu_s"),
                                    "worker_cpu_s": r.get("worker_cpu_s"),
                                    "unity_threads": (r.get("unity") or {}).get("threads"), "actions": r["actions"]}
                                   if r.get("ok") else {})} for h, r in zip(hs, reps)]
            cell["ok"] = (not bad and len(good) == E and all(r["n"] == self.steps for r in good)
                          and cell.get("bad_frames") == 0)
        except PoolError as e:
            cell["error"] = repr(e)
        log(f"cell E={E} L={L} {scene}: ok={cell['ok']} agg={cell.get('agg_steps_per_s')} "
            f"p50={cell.get('p50_ms')} cores={(cell.get('cpu') or {}).get('cores')} {cell.get('error') or ''}")
        return cell

    def grid_step(self, L: int, E: int, scene: str, k: int, reserve: float) -> None:
        """Run one cell if its estimate fits before the deadline minus `reserve`, else record it as skipped."""
        est = self.est_cell(L, E, scene)
        if self.left() - reserve < est:
            self.cells.append({"L": L, "E": E, "EL": E * L, "scene": scene, "skipped": "no_time",
                               "est_s": round(est, 1), "left_s": round(self.left(), 1), "ok": False})
        else:
            self.cells.append(self.run_cell(L, E, scene, k))
        self.save()

    def p2_grid(self, cells: Sequence[Tuple[int, int]], reserve: float, k0: int) -> None:
        """Grid cells (L, E) in the given order (cell_order).  A cell on the same L as the one before starts on
        the scene that one ended on (its pool has it loaded); a new pool starts on the first grid scene."""
        k = k0
        prev: Optional[Tuple[int, str]] = None
        for L, E in cells:
            scenes = list(self.grid_scenes)
            if prev is not None and prev[0] == L and prev[1] in scenes:
                scenes.remove(prev[1])
                scenes.insert(0, prev[1])
            for scene in scenes:
                k += 1
                self.grid_step(L, E, scene, k, reserve=reserve)
            if scenes:
                prev = (L, scenes[-1])

    # -- reach
    def p3_reach(self) -> None:
        live = [h for h in self.pool if h.alive]
        if len(live) < REACH_MIN_WORKERS:
            try:
                live = self.ensure_pool(REACH_L, REACH_E)
            except PoolError as e:
                self.err("p3.pool", repr(e))
                live = [h for h in self.pool if h.alive]
        if not live:
            raise PoolError("no worker for the reach dumps")
        rdir = self.out / "reach"
        rdir.mkdir(parents=True, exist_ok=True)
        pending = [(s, n) for s in self.reach_scenes for n in (1, 2)]
        first: Dict[str, int] = {}
        busy: Dict[int, Tuple[Tuple[str, int], int, float, Any, int]] = {}
        loads: Dict[str, Dict[int, Dict[str, Any]]] = {s: {} for s in self.reach_scenes}
        stopped = None
        while pending or busy:
            if self.left() < SUMMARY_RESERVE_S:
                stopped = "deadline"
                for (job, cid, t, h, _) in busy.values():
                    h.kill()
                    loads[job[0]][job[1]] = {"error": "killed at the deadline", "wid": h.wid}
                busy.clear()
                break
            alive = [h for h in live if h.alive]
            if not alive and not busy:
                stopped = "no live worker"
                break
            for h in alive:
                if h.wid in busy or not pending or self.left() < SUMMARY_RESERVE_S + REACH_JOB_EST_S:
                    continue
                pick = None
                for i, (s, n) in enumerate(pending):
                    if n == 1 or (s in first and (first[s] != h.wid or len(alive) == 1)):
                        pick = i
                        break
                if pick is None:
                    continue
                s, n = pending.pop(pick)
                if n == 1:
                    first[s] = h.wid
                cid = h.send({"op": "reach", "scene": s, "load": True})
                busy[h.wid] = ((s, n), cid, self.clock(), h, len(busy) + 1)
            if not busy and pending:
                stopped = "deadline" if self.left() < SUMMARY_RESERVE_S + REACH_JOB_EST_S else "stuck"
                break
            for wid, ((s, n), cid, t, h, conc) in list(busy.items()):
                r = h.wait(cid, 0.05)
                if r.get("timeout") and self.clock() - t < REACH_JOB_TIMEOUT_S:
                    continue
                del busy[wid]
                if r.get("timeout") or r.get("died"):
                    h.kill()
                if r.get("ok"):
                    h.scene = s
                    loads[s][n] = {"wid": h.wid, "L": h.L, "load_s": r["load_s"], "reach_s": r["reach_s"],
                                   "n": r["n"], "sha256_raw": r["sha256_raw"], "sha256_xz": r["sha256_xz"],
                                   "call_pose": r.get("call_pose"), "positions": r["positions"]}
                    self.load_rows.append({"scene": s, "kind": lvp.scene_kind(s), "wid": h.wid, "L": h.L,
                                           "load_s": r["load_s"], "concurrent": conc, "phase": "reach"})
                else:
                    loads[s][n] = {"error": r.get("error"), "wid": h.wid}
        done = equal = 0
        unequal, missing = [], []
        for s in self.reach_scenes:
            ls = loads[s]
            okl = [ls[n] for n in (1, 2) if n in ls and "error" not in ls[n]]
            if not okl:
                missing.append(s)
                continue
            doc = {"scene": s, "kind": lvp.scene_kind(s), "label": LABEL,
                   "split": [x for x, ss in (("sft", self.sft_scenes), ("eval", self.eval_scenes)) if s in ss],
                   "loads": [{k: v for k, v in ls[n].items() if k != "positions"} if n in ls else None
                             for n in (1, 2)],
                   "positions": okl[0]["positions"]}
            if len(okl) == 2:
                doc["equal_raw"] = okl[0]["sha256_raw"] == okl[1]["sha256_raw"]
                doc["equal_xz"] = okl[0]["sha256_xz"] == okl[1]["sha256_xz"]
                doc["same_worker"] = okl[0]["wid"] == okl[1]["wid"]
                if not doc["equal_raw"]:
                    doc["positions_load2"] = okl[1]["positions"]
                done += 1
                equal += doc["equal_raw"]
                if not doc["equal_raw"]:
                    unequal.append(s)
            else:
                doc["equal_raw"] = doc["equal_xz"] = None
                missing.append(s)
            write_json_atomic(rdir / f"{s}.json", doc)
        self.res["reach"] = {"dir": str(rdir), "planned": len(self.reach_scenes), "both_loads": done,
                             "equal_raw": equal, "unequal": unequal, "incomplete": missing, "stopped": stopped,
                             "workers": sorted({h.wid for h in live}), "L": sorted({h.L for h in live})}

    # -- the end
    def finish(self) -> None:
        r = self.res
        r["grid_cells"] = self.cells
        rec = recommendations(r)
        r["grid_table"] = rec["grid_table"]
        r["recommended"] = rec["recommended"]
        r["recommended_measured"] = rec["recommended_measured"]
        r["cores_per_env"] = rec["cores_per_env"]
        if "lvp_profile" not in r:
            r["lvp_profile"] = {"profile": None, "first_try": lvp.LVP_PROFILES[0], "retried": self.profile_retried,
                                "attempts": self.profile_attempts, "why": "no worker started"}
        rep = None
        cmp_ = r.get("compare")
        if cmp_:
            rep = cmp_["controls"]["repeat"]["identical"] if cmp_["controls"]["repeat"]["n"] else None
        r["determinism"] = summarize_det(self.det_rows, self.path_tests, self.reload_tests, self.reach_reload, rep)
        r["determinism"]["probe_poses"] = self.probe_poses
        if r.get("reach"):
            r["determinism"]["reach_equal_across_two_loads"] = (r["reach"]["equal_raw"] == r["reach"]["both_loads"]
                                                               if r["reach"]["both_loads"] else None)
        kinds: Dict[str, List[float]] = {}
        serial: Dict[str, List[float]] = {}
        for row in self.load_rows:
            if row["phase"].startswith("p1_replay"):
                continue
            kinds.setdefault(row["kind"], []).append(row["load_s"])
            if row["concurrent"] == 1:
                serial.setdefault(row["kind"], []).append(row["load_s"])
        r["load_times"] = {"by_kind": {k: dist(v) for k, v in kinds.items()},
                           "serial_by_kind": {k: dist(v) for k, v in serial.items()}, "rows": self.load_rows,
                           "startup_s": dist([w["startup_s"] for w in r["workers"] if w.get("startup_s")])}
        started = [w for w in r["workers"] if w.get("ok")]
        fm = sorted({str(w.get("font_mode")) for w in started})
        shas = sorted({str((w.get("font") or {}).get("sha256")) for w in started})
        f = r.setdefault("font", {})
        f.update(path=str(self.font), expected_sha256=self.font_sha256,
                 sha256=((f.get("parent") or {}).get("sha256")), worker_modes=fm, worker_sha256=shas,
                 rule="burn_label always uses the pinned badge font (lvp.install_font_rule; demos and eval refuse "
                      "any other bytes)")
        f["ok"] = bool(started) and fm == ["pinned"] and shas == [self.font_sha256]
        r["homes"] = dir_size(self.out / "homes")
        r["utc_end"] = utc()

    def run(self) -> Dict[str, Any]:
        try:
            self.phase("p0_preflight", self.p0_preflight)
            if self.ready_to_render:
                self.phase("p1_compare_det", self.p1_compare_det)
                self.phase("p2_grid", lambda: self.p2_grid(self.p2_cells, REACH_RESERVE_S + SUMMARY_RESERVE_S, 100))
                self.phase("p3_reach", self.p3_reach)
                self.phase("p4_grid", lambda: self.p2_grid(self.p4_cells, SUMMARY_RESERVE_S, 200))
        except Terminated:
            self.res["terminated"] = utc()
            self.stop_pool(graceful=False)
        finally:
            try:
                self.stop_pool(graceful=not self.res.get("terminated"))
            finally:
                try:
                    self.finish()
                except Exception as e:  # noqa: BLE001
                    self.err("finish", f"{e!r}\n{traceback.format_exc()[-2000:]}")
                self.save()
        return self.res


# --- expect -------------------------------------------------------------------------------------------------------
def expect(out) -> Dict[str, Any]:
    """The orchestrator's output check (stdlib only, login-node safe): writes out/expect.json."""
    out = Path(out)
    items: List[Dict[str, Any]] = []

    def item(name, ok, detail=None, required=True):
        items.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})

    s = None
    try:
        s = json.loads((out / "summary.json").read_text())
    except (OSError, ValueError) as e:
        item("summary.json", False, repr(e))
    if s is not None:
        item("summary.json", s.get("schema") == SCHEMA and s.get("label") == LABEL, s.get("schema"))
        item("no_terminated", not s.get("terminated"), s.get("terminated"))
        ph = s.get("phases") or {}
        for p in PHASES:
            item(f"phase.{p}", (ph.get(p) or {}).get("status") == "ok", (ph.get(p) or {}).get("status"))
        c = s.get("compare") or {}
        item("compare.verdict", c.get("verdict") in ("same_rendering", "different_rendering"), c.get("verdict"))
        item("compare.n_compared", c.get("n_compared") == c.get("n_expected") == 27,
             [c.get("n_compared"), c.get("n_expected")], required=False)
        rec = s.get("recommended") or {}
        item("recommended", isinstance(rec.get("E"), int) and isinstance(rec.get("L"), int),
             [rec.get("E"), rec.get("L")])
        recm = s.get("recommended_measured") or {}
        item("recommended_measured", isinstance(recm.get("E"), int) and isinstance(recm.get("L"), int),
             [recm.get("E"), recm.get("L"), recm.get("cores_max")], required=False)
        prof = (s.get("lvp_profile") or {}).get("profile")
        item("lvp_profile", prof in lvp.LVP_PROFILES, prof)
        cells = s.get("grid_cells") or []
        max_el = int((s.get("plan") or {}).get("max_el_recommend", MAX_EL_RECOMMEND))
        item("grid.cells_ok", any(x.get("ok") for x in cells),
             f"{sum(bool(x.get('ok')) for x in cells)}/{len(cells)} ok, "
             f"{sum(bool(x.get('skipped')) for x in cells)} skipped")
        item("grid.no_eligible_skipped",
             not any(x.get("skipped") and int(x["E"]) * int(x["L"]) <= max_el for x in cells),
             [f"E{x['E']}xL{x['L']} {x['scene']}" for x in cells
              if x.get("skipped") and int(x["E"]) * int(x["L"]) <= max_el])
        item("grid.no_skipped", not any(x.get("skipped") for x in cells),
             [f"E{x['E']}xL{x['L']} {x['scene']}" for x in cells if x.get("skipped")], required=False)
        item("grid.all_cells_ok", all(x.get("ok") for x in cells),
             [f"E{x['E']}xL{x['L']} {x['scene']}: {x.get('error')}" for x in cells
              if not x.get("ok") and not x.get("skipped")], required=False)
        d = s.get("determinism") or {}
        item("determinism.measured", d.get("n_flags_measured", 0) >= 4, d.get("n_flags_measured"))
        rch = s.get("reach") or {}
        planned = rch.get("planned")
        files = sorted((out / "reach").glob("*.json")) if (out / "reach").is_dir() else []
        item("reach.files", planned and len(files) == planned, f"{len(files)}/{planned}")
        item("reach.both_loads", planned and rch.get("both_loads") == planned, f"{rch.get('both_loads')}/{planned}")
        fo = s.get("font") or {}
        item("font.ok", fo.get("ok"), {"modes": fo.get("worker_modes"), "sha256": fo.get("worker_sha256"),
                                       "expected": fo.get("expected_sha256")})
        comp_dir = out / "compare"
        n_png = len(list(comp_dir.glob("*/*.png"))) if comp_dir.is_dir() else 0
        item("compare.pngs", n_png >= 31, n_png, required=False)  # 25 steps + 2 targets + 4 repeat renders
    ok = all(i["ok"] for i in items if i["required"])
    doc = {"ok": ok, "utc": utc(), "out": str(out), "items": items,
           "missing_required": [i["name"] for i in items if i["required"] and not i["ok"]]}
    write_json_atomic(out / "expect.json", doc)
    return doc


# --- CLI ----------------------------------------------------------------------------------------------------------
def _on_signal(signum, frame):  # noqa: ARG001
    raise Terminated(f"signal {signum}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--deadline-s", type=float, required=True)
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--grid", default=None, help="e.g. '2:1,2,3;4:1,2' (default GRID_DEFAULT)")
    r.add_argument("--p1-l", type=int, default=P1_L)
    r.add_argument("--repo", default=str(lvp.TVR_REPO))
    r.add_argument("--sample-dir", default=str(SAMPLE_DIR))
    r.add_argument("--sft-train", default=str(SFT_TRAIN))
    r.add_argument("--procthor-src", default=str(lvp.PROCTHOR_TRAIN))
    w = sub.add_parser("worker")
    w.add_argument("--cmd-fd", type=int, required=True)
    w.add_argument("--reply-fd", type=int, required=True)
    w.add_argument("--wid", type=int, required=True)
    w.add_argument("--repo", required=True)
    w.add_argument("--procthor-dir", default=str(lvp.PROCTHOR_DIR))
    w.add_argument("--houses", default=None)
    e = sub.add_parser("expect")
    e.add_argument("--out", required=True)
    rc = sub.add_parser("recommend", help="recompute recommended / recommended_measured from out/summary.json")
    rc.add_argument("--out", required=True)
    rc.add_argument("--max-el", type=int, default=None, help="default: the summary's plan (11)")
    rc.add_argument("--max-cores", type=float, default=None, help="default: the summary's plan (11)")
    rc.add_argument("--write", default=None, help="also write the result to this JSON file (never summary.json)")
    a = ap.parse_args(argv)
    if a.cmd == "worker":
        return worker_main(a)
    if a.cmd == "expect":
        doc = expect(a.out)
        print(json.dumps({"ok": doc["ok"], "missing_required": doc["missing_required"]}), flush=True)
        return 0 if doc["ok"] else 1
    if a.cmd == "recommend":
        src = Path(a.out) / "summary.json"
        try:
            summary = json.loads(src.read_text())
        except (OSError, ValueError) as ex:
            print(f"cannot read {src}: {ex!r}", file=sys.stderr)
            return 1
        doc = dict(recommendations(summary, a.max_el, a.max_cores), source=str(src), utc=utc())
        if a.write:
            if Path(a.write).resolve() == src.resolve():
                print("refusing to overwrite summary.json", file=sys.stderr)
                return 2
            write_json_atomic(a.write, doc)
        print(json.dumps(jsafe({k: doc[k] for k in ("recommended", "recommended_measured", "cores_per_env",
                                                     "limits")}), indent=1), flush=True)
        return 0
    out = Path(a.out)
    if (out / "summary.json").exists():
        print(f"refusing: {out}/summary.json exists; give a new --out", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    houses = out / "houses"
    spawner = SubprocessSpawner(out, repo=a.repo, houses_dir=houses, procthor_dir=Path(a.repo) / "data" / "procthor-10k")
    probe = Probe(out, deadline_s=a.deadline_s, spawner=spawner,
                  grid=parse_grid(a.grid) if a.grid else None, steps=a.steps, p1_l=a.p1_l, repo=a.repo,
                  sample_dir=a.sample_dir, sft_train=a.sft_train, procthor_src=a.procthor_src)
    res = probe.run()
    log("summary:", json.dumps(jsafe({k: res.get(k) for k in ("phases", "recommended")}))[:2000])
    log("compare verdict:", (res.get("compare") or {}).get("verdict"),
        "deterministic:", (res.get("determinism") or {}).get("deterministic"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
