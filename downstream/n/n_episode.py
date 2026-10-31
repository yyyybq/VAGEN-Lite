#!/usr/bin/env python3
"""N series: episodes of the upstream task in RoboCasa with a VLM answering over a socket (smoke part 2: B on the 20
left / right instances; N1 later).  Run with the simulator Python; the model runs in downstream/n/n_policy_server.py.

    python downstream/n/n_episode.py --instances GEN/instances.jsonl [...] --port 5800 --out OUT [--workers 4] [--limit 20]

The conversation is the main line's validation one (exps/.../interiorgs_v10_train_r11p_20261017/rl_ps0/val_env.yaml,
the RL env RS and RD were trained with), built with the upstream functions loaded by path:
  * system prompt: prompt.system_prompt(**ActiveSpatialEnvConfig(<val config>).system_prompt_arguments(task type));
  * each turn one user message: init_observation_template (first turn) or action_template (with the feedback), the
    image placeholder, "Current camera pose: " + format_pose6_deg(c2w_extrinsic_to_se3(c2w), normalize_zero), the
    task, then format_prompt["no_think"] (the example only on the first turn);
  * history_window_size 1: the model sees the system prompt and the current user message only (the main line's
    agent loop keeps one observation);
  * a reply is read like the env's strict parser (utils.parse_free_think, parse_actions, check_actions against the
    strafe actions; at most 5 actions); an unreadable reply executes nothing and 3 in a row end the episode;
  * the actions run one by one; a blocked one stays (feedback "<k> of <n> actions executed; blocked: <action>
    (blocked by <thing>)." via the upstream collision_feedback); after every executed action the success rule is
    checked (n_env.Scene.judge) and the episode ends on success; at most max_episode_steps (12) primitive actions
    and 12 turns.
OUT/episodes.jsonl gets one line per episode (success, turns, primitive steps, blocked and invalid counts, the poses,
every reply and its actions, the judge at the end); OUT/frames/<scene>.png the camera-A frame of every turn.
N1 (the user's decision 3 of 2026-10-31): any of the six task types (n_env.Scene.judge with the item's own rule), the
instances of n_gen.py collect (GEN/instances_eval.jsonl, one per line, keyed by instance_id) as well as the smoke's
part files; each episode renders camera A with or without the robot as its instance was made (hide_robot), and also
records the highest upstream shaping score reached (the box judge's, after the start and every executed action) and
the numbers behind the gates at the end.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import multiprocessing as mp
import socket
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import n_env as N  # noqa: E402

VAL_ENV = N.MAIN / "exps/vagen_active_spatial/interiorgs_v10_train_r11p_20261017/rl_ps0/val_env.yaml"


# --------------------------------------------------------------------------------------- the upstream text
def upstream_config(path: Path = VAL_ENV):
    """(ActiveSpatialEnvConfig with the validation config's fields, the env entry of the yaml)."""
    import yaml

    ec = N.active_spatial("env_config")
    entry = yaml.safe_load(Path(path).read_text())["envs"][0]
    names = {f.name for f in dataclasses.fields(ec.ActiveSpatialEnvConfig)}
    cfg = ec.ActiveSpatialEnvConfig(**{k: v for k, v in entry["config"].items() if k in names})
    return cfg, entry


def system_text(cfg, task_type: str) -> str:
    return N.active_spatial("prompt").system_prompt(**cfg.system_prompt_arguments(task_type))


def user_text(cfg, first: bool, pose, task_prompt: str, feedback: str = "") -> str:
    prompt, utils = N.active_spatial("prompt"), N.active_spatial("utils")
    pose_str = utils.format_pose6_deg(utils.c2w_extrinsic_to_se3(N.c2w_of(pose)),
                                      normalize_zero=bool(cfg.get("pose_normalize_negative_zero", False)))
    obs = f"<image>\nCurrent camera pose: {pose_str}"
    if first:
        text = prompt.init_observation_template(observation=obs, task_prompt=task_prompt, spatial_prior="")
    else:
        text = prompt.action_template(observation=obs, env_feedback=feedback or "Action executed.",
                                      task_prompt=task_prompt)
    fmt = prompt.format_prompt[cfg.prompt_format]
    return text + "\n" + fmt(max_actions_per_step=cfg.max_actions_per_step, action_sep=cfg.action_sep,
                             add_example=first, format_example=cfg.get("prompt_format_example", "placeholder"))


def allowed_actions(cfg) -> set:
    utils = N.active_spatial("utils")
    return set(utils.ACTION_SPACE_PRESETS[cfg.action_space])


def parse(text: str, cfg, allowed: set):
    """(actions, error) as the env's strict reading: <action>a|b|</action>, known names only, at most 5."""
    utils = N.active_spatial("utils")
    ft = utils.parse_free_think(text or "")
    if not ft["ok"]:
        return [], "missing_action_tag"
    ok, parsed = utils.parse_actions(ft["actions_blob"], sep=cfg.action_sep)
    if not ok:
        return [], "malformed_action_body"
    if not utils.check_actions(parsed, allowed=allowed):
        return [], "unknown_action_name"
    return [a.name for a in parsed[:cfg.max_actions_per_step]], ""


def turn_feedback(turn_moves, turn_blocked, moved: bool) -> str:
    """The env's feedback (collision_feedback_detail on): blocked moves listed, else executed / no effect."""
    if turn_blocked:
        return (f"{len(turn_moves) - len(turn_blocked)} of {len(turn_moves)} actions executed; blocked: "
                + "; ".join(f"{a} ({t})" for a, t in turn_blocked) + ".")
    return "Action executed." if moved else "Action had no effect."


def chat(system: str, user: str):
    """Chat-template messages with the image placeholder turned into an image part."""
    parts = []
    for k, piece in enumerate(user.split("<image>")):
        if k:
            parts.append({"type": "image"})
        if piece:
            parts.append({"type": "text", "text": piece})
    return [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": parts}]


# ------------------------------------------------------------------------------------------- the socket
def send_msg(sock, obj) -> None:
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(sock):
    head = b""
    while len(head) < 4:
        chunk = sock.recv(4 - len(head))
        if not chunk:
            raise ConnectionError("closed")
        head += chunk
    n = struct.unpack(">I", head)[0]
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return json.loads(buf)


def png_b64(arr) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def asker(port: int, host: str = "127.0.0.1"):
    def ask(messages, images, max_new_tokens: int):
        with socket.create_connection((host, port), timeout=600) as s:
            send_msg(s, {"messages": messages, "images": [png_b64(i) for i in images], "max_new_tokens": max_new_tokens})
            return recv_msg(s)["reply"]
    return ask


# ------------------------------------------------------------------------------------------- an episode
def shaping(j: dict) -> float:
    m = j["metric"]
    return float(m.get("shaping_score", m.get("score")) or 0.0)


def run_episode(scene, inst: dict, ask, cfg, entry: dict) -> dict:
    """One episode (see the module text); returns its record and the frames of every turn."""
    item = inst["item"]
    allowed = allowed_actions(cfg)
    system = system_text(cfg, item["task_type"])
    max_turns = int(entry.get("max_turns", 12))
    budget = int(cfg.max_episode_steps)
    per_turn = int(entry.get("response_length_per_turn", 384))
    pose = tuple(scene.start)
    fb, invalid, steps, success, blocked_n = "", 0, 0, False, 0
    turns, frames, end = [], [], "turns"
    best_shaping = shaping(N.box_judge(item, pose))
    for t in range(max_turns):
        img, _ = scene.render(pose)
        frames.append(img)
        reply = ask(chat(system, user_text(cfg, t == 0, pose, item["task_description"], fb)), [img], per_turn)
        actions, err = parse(reply, cfg, allowed)
        rec = {"turn": t + 1, "reply": reply, "actions": actions, "parse_error": err, "pose_before": list(pose)}
        if not actions:
            invalid += 1
            fb = "Action had no effect."
            turns.append(rec)
            if invalid >= cfg.max_consecutive_invalid_actions:
                end = "invalid"
                break
            continue
        invalid = 0
        moves, blocked, start_pose = [], [], pose
        for a in actions:
            new, hit = scene.try_action(pose, a)
            moves.append(a)
            steps += 1
            if hit is not None:
                blocked.append((a, f"blocked by {scene.owner(hit)}"))
                blocked_n += 1
            else:
                pose = new
                j = scene.judge(item, pose)
                best_shaping = max(best_shaping, shaping(j))
                if j["success"]:
                    success = True
                    break
            if steps >= budget:
                break
        fb = turn_feedback(moves, blocked, pose != start_pose)
        rec.update(executed=moves, blocked=blocked, pose_after=list(pose), feedback=fb)
        turns.append(rec)
        if success:
            end = "success"
            break
        if steps >= budget:
            end = "steps"
            break
    final = scene.judge(item, pose)
    meas = {k: (list(v) if isinstance(v, tuple) else v) for k, v in N.measurements(item, final).items()}
    out = {"instance_id": inst_id(inst), "scene_id": inst["scene_id"], "task_type": item["task_type"],
           "success": success, "end": end, "turns": len(turns), "steps": steps,
           "blocked": blocked_n, "invalid_turns": sum(1 for r in turns if not r["actions"]),
           "max_shaping_score": best_shaping, "final_pose": list(pose),
           "final_judge": {"success": final["success"], "visible": final["visible"],
                           "gates": final["metric"].get("gates"), "measurements": meas,
                           "shaping_score": shaping(final),
                           "margin": final["metric"].get("relation_margin_px"),
                           "lateral_deg": final["lateral"].get("angle_deg")},
           "turn_log": turns}
    return out, frames


def inst_id(inst: dict) -> str:
    """N1 instances have their own id (several task types per scene); the smoke's had one per scene."""
    return str(inst.get("instance_id") or inst["scene_id"])


def strip_image(frames, size: int = 128):
    from PIL import Image

    sheet = Image.new("RGB", (size * len(frames), size), (255, 255, 255))
    for i, f in enumerate(frames):
        sheet.paste(Image.fromarray(f).resize((size, size)), (i * size, 0))
    return sheet


_W = {}


def _init(port: int):
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    _W["ask"] = asker(port)
    _W["cfg"], _W["entry"] = upstream_config()


def _one(job):
    inst, out = job
    scene = None
    t0 = time.time()
    try:
        scene = N.Scene(inst["task"], inst["seed"], hide_robot=bool(inst.get("hide_robot", False)))
        rec, frames = run_episode(scene, inst, _W["ask"], _W["cfg"], _W["entry"])
        (Path(out) / "frames").mkdir(parents=True, exist_ok=True)
        strip_image(frames).save(Path(out) / "frames" / f"{inst_id(inst)}.png")
    except Exception as exc:
        import traceback

        rec = {"instance_id": inst_id(inst), "scene_id": inst.get("scene_id"), "task_type": inst.get("task_type"),
               "error": f"{type(exc).__name__}: {exc}"[:400], "traceback": traceback.format_exc()[-1500:]}
    finally:
        if scene is not None:
            scene.close()
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def load_instances(paths, limit: int):
    """The smoke's part files ({"task", "seed", "hide_robot", "instance": {...}} per scene) or N1's
    instances_eval.jsonl (one instance per line, with instance_id, task, seed, hide_robot)."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("instance_id") and r.get("item"):
                    out.append(r)
                elif r.get("instance"):
                    out.append(dict(r["instance"], task=r["task"], seed=r["seed"],
                                    hide_robot=bool(r.get("hide_robot", False))))
    return out[:limit] if limit else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "episodes.jsonl").is_file():
        done = {inst_id(json.loads(x)) for x in (out / "episodes.jsonl").read_text().splitlines() if x.strip()}
    todo = [i for i in load_instances(args.instances, args.limit) if inst_id(i) not in done]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.port,)) as pool, open(out / "episodes.jsonl", "a") as f:
        for rec in pool.imap_unordered(_one, [(i, str(out)) for i in todo]):
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print(f"[n_episode] {rec.get('instance_id')}: success={rec.get('success')} turns={rec.get('turns')} "
                  f"{rec.get('error', '')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
