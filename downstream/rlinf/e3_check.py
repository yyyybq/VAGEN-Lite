#!/usr/bin/env python3
"""E3 check (the user's request of 2026-10-30, at most 5 card-hours): RLinf's training rollouts succeeded 0-2% while
our evaluation of the same start on the same three tasks gives 32-43%.  Reading the code found two differences
between E3's RLinf setup and our evaluation (the success statistic itself matches: RLinf counts, per finished
episode, whether RoboCasa's _check_success() held at any step, the same function our evaluation reads):
  1. scenes: E3's training environments used RoboCasa365's "pretrain" split (env.train.split in cfg/train.yaml);
     the demonstrations of our downstream training and our evaluation are the "target" split;
  2. actions: our evaluation goes through RoboCasa's gym wrapper (PandaOmronKeyConverter.unmap_action), which turns
     the gripper and base-mode dimensions into -1 / +1 at 0.5 and puts each part at the controller's own index;
     RLinf's StarVLA + RoboCasa365 path passes the 12 unnormalized numbers to robosuite's step unchanged (no
     threshold; robosuite then closes the gripper on any positive value and switches to base mode on any value
     above zero).
This measures both on our own pipeline, then does the three things the user asked for.

    python downstream/rlinf/e3_check.py obs     --ckpt CKPT --out OUT      # 20 first frames side by side
    python downstream/rlinf/e3_check.py run     --ckpt CKPT --servers H:P,... --out OUT [--seeds 1000-1009]
    python downstream/rlinf/e3_check.py rejudge --out OUT                  # 20 failed episodes judged again
    python downstream/rlinf/e3_check.py report  --out OUT

obs: for each (task, seed) of our evaluation, the first observation as our evaluation sees it (gym wrapper, target
split, the contract's resize to the training size) and as RLinf builds it (robocasa.utils.env_utils.create_env with
RLinf's arguments, images flipped upright, the prompt from ep_meta "lang"), for the target split and for E3's
pretrain split: OUT/obs/page.html, OUT/obs/frames.json (sizes, instruction texts, whether the pixels are equal) and
the robot's action layout (the controller's part indexes).
run: the start E3-B began from, greedy, on the three E3 tasks x SEEDS in four cells: action path {gym (our
evaluation), rlinf (12 numbers unchanged)} x split {target, pretrain}, at most 512 control steps (E3's horizon),
episodes in parallel on one policy server per GPU (closed_loop's workers, retries and hang handling).  Per episode:
success, steps, and how often the raw gripper / mode numbers would be read differently by the two paths.  The
final simulator state of failed rlinf-path episodes is saved (OUT/states/).
rejudge: up to 20 of those failed episodes: a fresh environment of the same task, seed and split, the saved state
set, RoboCasa's success check read again (our judgment) and compared with RLinf's (it said failure).
report: OUT/report.json and report.md (success per cell and task with 95% Wilson intervals, the two factors, the
rejudge agreement).
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(WORKTREE / "downstream/eval"))
sys.path.insert(0, str(WORKTREE / "downstream/x"))
import closed_loop  # noqa: E402

TASKS = ("CoffeeSetupMug", "StackBowlsCabinet", "TurnOffStove")
CELLS = (("gym", "target"), ("gym", "pretrain"), ("rlinf", "target"), ("rlinf", "pretrain"))
MAX_STEPS = 512  # E3's max_episode_steps
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
OBS_PAIRS = [("CoffeeSetupMug", s) for s in range(1000, 1007)] + [("StackBowlsCabinet", s) for s in range(1000, 1007)] \
    + [("TurnOffStove", s) for s in range(1000, 1006)]
GRIPPER, MODE = 6, 11


def parse_seeds(text: str) -> list:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


# ------------------------------------------------------------------------------------------- action paths (pure)
def gym_flat(vector, split_indexes: dict, prefix: str = "robot0_"):
    """The flat robosuite action our evaluation sends (RoboCasa's gym wrapper: unmap_action, then each part at the
    composite controller's index, base mode last)."""
    import numpy as np

    v = np.asarray(vector, dtype=np.float64)
    parts = {"right": v[0:6], "right_gripper": np.array([-1.0 if v[GRIPPER] < 0.5 else 1.0]),
             "base": v[7:10], "torso": v[10:11]}
    size = max(end for _, end in split_indexes.values()) + 1
    out = np.zeros(size)
    for part, (start, end) in split_indexes.items():
        name = part[len(prefix):] if part.startswith(prefix) else part
        if name in parts and end > start:
            out[start:end] = parts[name]
    out[-1] = -1.0 if v[MODE] < 0.5 else 1.0
    return out


def rlinf_flat(vector):
    """The flat robosuite action RLinf sends: the 12 unnormalized numbers as they are."""
    import numpy as np

    return np.asarray(vector, dtype=np.float64)


def read_differently(vector) -> dict:
    """Whether robosuite reads the raw gripper / mode numbers differently from the thresholded ones: the gripper
    closes on a positive number (sign) where the gym path opens it below 0.5; base mode is on above zero where the
    gym path turns it on from 0.5."""
    g, m = float(vector[GRIPPER]), float(vector[MODE])
    return {"gripper": (g > 0) != (g >= 0.5), "mode": (m > 0) != (m >= 0.5)}


# ---------------------------------------------------------------------------------------------- episodes (sim)
def run_episode(spec: dict) -> dict:
    """One greedy episode on one path and split (mirrors closed_loop._run_episode; never raises)."""
    marker = Path(closed_loop._STATE["inflight"]) / f"{os.getpid()}.json" if closed_loop._STATE.get("inflight") else None
    if marker is not None:
        marker.write_text(json.dumps({"task": spec["task"], "seed": spec["seed"], "started": time.time(),
                                      "max_steps": spec["max_steps"]}))
    closed_loop._STATE["marker"], closed_loop._STATE["beat"] = marker, 0.0
    out = dict(spec, pid=os.getpid())
    env = None
    t0 = time.monotonic()
    try:
        if "init_error" in closed_loop._STATE:
            return dict(out, status="error", error="worker init: " + closed_loop._STATE["init_error"])
        import gymnasium as gym
        import numpy as np
        import robocasa.wrappers.gym_wrapper  # noqa: F401
        from vagen.envs.robocasa.starvla_contract import gym_action

        contract, policy = closed_loop._STATE["contract"], closed_loop._STATE["policy"]
        env = gym.make(f"robocasa/{spec['task']}", enable_render=True, split=spec["split"], seed=spec["seed"])
        obs, _ = env.reset(seed=spec["seed"])
        genv = env.unwrapped
        steps, success, diff = 0, False, {"gripper": 0, "mode": 0}
        raw_g, raw_m = [], []
        while steps < spec["max_steps"] and not success:
            closed_loop._heartbeat()
            response = policy.predict_action({"examples": [contract.example(obs)], "do_sample": False})
            for vector in contract.actions(response["data"]["actions"])[:min(spec["execute_steps"], spec["max_steps"] - steps)]:
                d = read_differently(vector)
                diff = {k: diff[k] + int(d[k]) for k in diff}
                raw_g.append(float(vector[GRIPPER]))
                raw_m.append(float(vector[MODE]))
                if spec["path"] == "gym":
                    obs, _r, _te, _tr, info = env.step(gym_action(vector, env.action_space))
                    success = bool(info.get("success", False))
                else:
                    raw_obs, _r, _d, _info = genv.env.step(rlinf_flat(vector))
                    success = bool(genv.env._check_success())
                    obs = genv.get_observation(raw_obs)
                steps += 1
                if success:
                    break
        out.update(status="complete", success=success, steps=steps, read_differently=diff,
                   gripper_raw={"min": min(raw_g), "max": max(raw_g), "mean": sum(raw_g) / len(raw_g)},
                   mode_raw={"min": min(raw_m), "max": max(raw_m), "mean": sum(raw_m) / len(raw_m)})
        if spec["path"] == "rlinf" and not success and spec.get("states_dir"):
            p = Path(spec["states_dir"]) / f"{spec['split']}_{spec['task']}_s{spec['seed']}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, np.asarray(genv.env.sim.get_state().flatten()))
            out["state_file"] = str(p)
            out["final_check"] = bool(genv.env._check_success())
    except Exception as exc:
        import traceback

        out.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500], traceback=traceback.format_exc()[-1500:])
    finally:
        closed_loop._STATE["marker"] = None
        if env is not None:
            env.close()
        if marker is not None:
            try:
                marker.unlink()
            except OSError:
                pass
    out["wall_seconds"] = round(time.monotonic() - t0, 1)
    return out


def episodes_for(seeds, cells=CELLS, tasks=TASKS, states_dir=None) -> list:
    return [{"task": t, "seed": s, "path": p, "split": sp, "max_steps": MAX_STEPS, "execute_steps": 16,
             "states_dir": states_dir, "cell": f"{p}/{sp}"}
            for p, sp in cells for t in tasks for s in seeds]


def run_cmd(args) -> int:
    out = Path(args.out)
    run_dir = out / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "episodes.jsonl"
    done = set()
    if rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "complete":
                done.add((r["cell"], r["task"], r["seed"]))
    eps = [e for e in episodes_for(parse_seeds(args.seeds), states_dir=str(out / "states"))
           if (e["cell"], e["task"], e["seed"]) not in done]
    inflight = run_dir / "inflight"
    inflight.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")
    # closed_loop.run_all keys episodes by (task, seed): run the cells one after another
    with ctx.Pool(args.workers, initializer=closed_loop._init_worker,
                  initargs=(args.ckpt, args.servers.split(","), args.starvla_root, True, str(inflight))) as pool, \
            open(rows_path, "a") as f:
        for cell in dict.fromkeys(e["cell"] for e in eps):
            todo = [e for e in eps if e["cell"] == cell]
            t = time.time()
            stats = closed_loop.run_all(pool, todo, lambda row: (f.write(json.dumps(row) + "\n"), f.flush()), inflight,
                                        run=run_episode)
            print(f"[e3_check] {cell}: {len(todo)} episodes, {time.time() - t:.0f} s, pool {stats}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ rejudge (sim)
def rejudge_cmd(args) -> int:
    import gymnasium as gym
    import numpy as np
    import robocasa.wrappers.gym_wrapper  # noqa: F401

    closed_loop.stable_counter_regions()
    out = Path(args.out)
    rows = [json.loads(line) for line in (out / "run/episodes.jsonl").read_text().splitlines() if line.strip()]
    failed = [r for r in rows if r.get("status") == "complete" and r["path"] == "rlinf" and not r["success"]
              and r.get("state_file")]
    failed = sorted(failed, key=lambda r: (r["split"] != "pretrain", r["task"], r["seed"]))[:args.n]
    res = []
    for r in failed:
        env = gym.make(f"robocasa/{r['task']}", enable_render=True, split=r["split"], seed=r["seed"])
        try:
            env.reset(seed=r["seed"])
            inner = env.unwrapped.env
            state = np.load(r["state_file"])
            same_size = state.size == inner.sim.get_state().flatten().size
            inner.sim.set_state_from_flattened(state)
            inner.sim.forward()
            ours = bool(inner._check_success())
            res.append({"task": r["task"], "seed": r["seed"], "split": r["split"], "rlinf_success": False,
                        "our_success": ours, "state_size_matches": bool(same_size)})
        except Exception as exc:
            res.append({"task": r["task"], "seed": r["seed"], "split": r["split"], "error": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            env.close()
        print(json.dumps(res[-1]), flush=True)
    (out / "rejudge.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------------------------- observations (sim)
def ours_first(task: str, seed: int, contract):
    import gymnasium as gym
    import numpy as np
    import robocasa.wrappers.gym_wrapper  # noqa: F401

    env = gym.make(f"robocasa/{task}", enable_render=True, split="target", seed=seed)
    try:
        obs, _ = env.reset(seed=seed)
        ex = contract.example(obs)
        raw = {c: np.asarray(obs[f"video.{c}"]) for c in CAMERAS}
        cc = env.unwrapped.env.robots[0].composite_controller
        layout = {k: list(v) for k, v in cc._action_split_indexes.items()}
        return {"raw": raw, "sent": [np.asarray(i) for i in ex["image"]], "lang": ex["lang"],
                "layout": layout, "action_dim": int(cc.action_limits[0].shape[0])}
    finally:
        env.close()


def rlinf_first(task: str, seed: int, split: str):
    """RLinf's Robocasa365Env path for one environment: create_env with its arguments (configs/env/robocasa365.yaml
    and E3's cfg), reset, images flipped upright, the prompt from ep_meta "lang"."""
    import numpy as np
    from robocasa.utils.env_utils import create_env

    env = create_env(env_name=task, robots="PandaOmron",
                     camera_names=["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"],
                     camera_widths=256, camera_heights=256, seed=seed, render_onscreen=False, split=split,
                     camera_depths=False, translucent_robot=False)
    try:
        raw = env.reset()
        imgs = {c: np.ascontiguousarray(np.asarray(raw[f"{c}_image"])[::-1]) for c in CAMERAS}
        return {"raw": imgs, "lang": str(env.get_ep_meta().get("lang", "")).strip()}
    finally:
        env.close()


def thumb(arr, size: int = 128) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).resize((size, size), Image.BILINEAR).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def obs_cmd(args) -> int:
    import numpy as np

    sys.path.insert(0, str(WORKTREE))
    from vagen.envs.robocasa.starvla_contract import PolicyContract, checkpoint_config

    import robocasa.wrappers.gym_wrapper  # noqa: F401

    closed_loop.stable_counter_regions()
    config, _ = checkpoint_config(args.ckpt)
    contract = PolicyContract.from_config(config)
    out = Path(args.out) / "obs"
    out.mkdir(parents=True, exist_ok=True)
    rows, html_rows, layout = [], [], None
    for task, seed in OBS_PAIRS:
        ours = ours_first(task, seed, contract)
        layout = layout or {"part_indexes": ours["layout"], "action_dim": ours["action_dim"]}
        r_t = rlinf_first(task, seed, "target")
        r_p = rlinf_first(task, seed, "pretrain")
        rec = {"task": task, "seed": seed, "our_lang": ours["lang"], "rlinf_target_lang": r_t["lang"],
               "rlinf_pretrain_lang": r_p["lang"],
               "raw_size": list(ours["raw"][CAMERAS[0]].shape), "sent_size": list(ours["sent"][0].shape),
               "contract_image_size": list(contract.image_size),
               "same_pixels_ours_vs_rlinf_target": {c: bool(np.array_equal(ours["raw"][c], r_t["raw"][c])) for c in CAMERAS},
               "same_pixels_target_vs_pretrain": {c: bool(np.array_equal(r_t["raw"][c], r_p["raw"][c])) for c in CAMERAS}}
        rows.append(rec)
        cols = []
        for title, imgs, lang in (("我们的评测（target）", ours["raw"], ours["lang"]),
                                  ("RLinf 的环境，target", r_t["raw"], r_t["lang"]),
                                  ("RLinf 的环境，pretrain（E3 训练时）", r_p["raw"], r_p["lang"])):
            cols.append(f'<div class="col"><h3>{html.escape(title)}</h3><div class="views">'
                        + "".join(f'<img src="{thumb(imgs[c])}" alt="{c}">' for c in CAMERAS)
                        + f'</div><p class="lang">{html.escape(lang)}</p></div>')
        html_rows.append(f'<section><h2>{html.escape(task)} · 种子 {seed}</h2><div class="cols">{"".join(cols)}</div>'
                         f'<p class="meta">像素相同（我们 vs RLinf target）：{rec["same_pixels_ours_vs_rlinf_target"]}</p></section>')
        print(f"[e3_check] obs {task} {seed}", flush=True)
    (out / "frames.json").write_text(json.dumps({"layout": layout, "frames": rows}, indent=1, ensure_ascii=False) + "\n")
    style = ("<style>:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#dedbd2}"
             "@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ecebe6;--muted:#a19f97;"
             "--line:#3a3935;color-scheme:dark}}:root[data-theme=dark]{--bg:#171716;--fg:#ecebe6;--muted:#a19f97;"
             "--line:#3a3935;color-scheme:dark}body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;"
             "max-width:1200px;margin:0 auto;padding-block:20px;padding-left:max(16px,env(safe-area-inset-left,0px));"
             "padding-right:max(16px,env(safe-area-inset-right,0px))}section{border-top:1px solid var(--line);padding:12px 0}"
             ".cols{display:flex;flex-wrap:wrap;gap:12px}.col{flex:1 1 330px;min-width:0}.views{display:flex;gap:4px}"
             ".views img{width:32%;max-width:128px;height:auto}h2{font-size:15px}h3{font-size:13px;margin:4px 0}"
             ".lang,.meta{color:var(--muted);font-size:12px}</style>")
    (out / "page.html").write_text(f"<title>E3 观测核对</title><meta name=\"claude:viewport-fit-sides\" content=\"cover\">"
                                   f"{style}<h1>E3 观测核对：同任务同种子的第一帧（左、右、手腕）</h1>{''.join(html_rows)}",
                                   encoding="utf-8")
    print(json.dumps(layout))
    return 0


# ---------------------------------------------------------------------------------------------------- report
def summarize(rows: list) -> dict:
    sys.path.insert(0, str(WORKTREE / "downstream/x"))
    import x0_qa

    out = {}
    for cell in dict.fromkeys(f"{p}/{s}" for p, s in CELLS):
        cr = [r for r in rows if r.get("cell") == cell and r.get("status") == "complete"]
        per = {}
        for task in TASKS + ("all",):
            tr = cr if task == "all" else [r for r in cr if r["task"] == task]
            k = sum(1 for r in tr if r["success"])
            per[task] = {"n": len(tr), "successes": k, "rate": k / len(tr) if tr else float("nan"),
                         "ci": x0_qa.wilson(k, len(tr))}
        steps = sum(r["steps"] for r in cr)
        per["read_differently"] = {d: (sum(r["read_differently"][d] for r in cr) / steps if steps else float("nan"))
                                   for d in ("gripper", "mode")}
        per["errors"] = sum(1 for r in rows if r.get("cell") == cell and r.get("status") != "complete")
        out[cell] = per
    return out


def report_cmd(args) -> int:
    out = Path(args.out)
    rows = [json.loads(line) for line in (out / "run/episodes.jsonl").read_text().splitlines() if line.strip()]
    # the last row of an episode counts (a retried episode can appear twice)
    last = {}
    for r in rows:
        last[(r.get("cell"), r["task"], r["seed"])] = r
    summary = summarize(list(last.values()))
    rejudge = json.loads((out / "rejudge.json").read_text()) if (out / "rejudge.json").is_file() else []
    frames = json.loads((out / "obs/frames.json").read_text()) if (out / "obs/frames.json").is_file() else None
    res = {"cells": summary, "rejudge": {"n": len(rejudge), "our_success": sum(1 for r in rejudge if r.get("our_success")),
                                         "errors": sum(1 for r in rejudge if "error" in r)}}
    if frames:
        fr = frames["frames"]
        res["obs"] = {"frames": len(fr), "layout": frames["layout"],
                      "lang_same_ours_rlinf": sum(1 for r in fr if r["our_lang"] == r["rlinf_target_lang"]),
                      "pixels_same_ours_rlinf_target": sum(1 for r in fr if all(r["same_pixels_ours_vs_rlinf_target"].values())),
                      "pixels_same_target_pretrain": sum(1 for r in fr if all(r["same_pixels_target_vs_pretrain"].values()))}
    (out / "report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    lines = ["| 条件 | " + " | ".join(TASKS) + " | 合计 | 夹爪读法不同的步 | 模式读法不同的步 |", "| --- " * 7 + "|"]
    for cell, per in summary.items():
        lines.append(f"| {cell} | " + " | ".join(f"{per[t]['successes']}/{per[t]['n']}" for t in TASKS)
                     + f" | {per['all']['successes']}/{per['all']['n']}（{100 * per['all']['rate']:.0f}%） | "
                     f"{100 * per['read_differently']['gripper']:.0f}% | {100 * per['read_differently']['mode']:.0f}% |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("obs")
    o.add_argument("--ckpt", required=True)
    o.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--servers", required=True)
    r.add_argument("--seeds", default="1000-1009")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--starvla-root", default=str(WORKTREE / "third_party/starVLA"))
    r.add_argument("--out", required=True)
    j = sub.add_parser("rejudge")
    j.add_argument("--out", required=True)
    j.add_argument("--n", type=int, default=20)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"obs": obs_cmd, "run": run_cmd, "rejudge": rejudge_cmd, "report": report_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
