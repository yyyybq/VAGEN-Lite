#!/usr/bin/env python3
"""Build/audit Phase-3 fixed transitions via an existing HTTP renderer only.

This script deliberately stops at the data gate. It does not import or run the
U1 model/FM forward path.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/fm_only"
DEFAULT_ENV_YAML = ROOT / "examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml"
DEFAULT_FINAL = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/reports/FINAL_DELIVERY.json"


def jdump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_stats(path: Path) -> dict[str, Any]:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    return {
        "width": int(im.width),
        "height": int(im.height),
        "bytes": int(path.stat().st_size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def diff_stats(a: Path, b: Path) -> dict[str, float]:
    arr = np.asarray(ImageChops.difference(Image.open(a).convert("RGB"), Image.open(b).convert("RGB")), dtype=np.float32)
    return {
        "pixel_diff_mean": float(arr.mean()),
        "pixel_diff_max": float(arr.max()),
        "pixel_diff_nonzero_fraction": float((arr > 0).mean()),
    }


def pose_parts(env) -> tuple[list[float], list[float], np.ndarray]:
    E = np.asarray(env.view_engine.get_pose(), dtype=np.float64)
    return [float(x) for x in E[:3, 3]], [float(x) for x in E[:3, 2]], E


def angle_between(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return float(math.degrees(math.acos(np.clip(float(np.dot(av, bv) / denom), -1.0, 1.0))))


def first_image(obs: dict[str, Any]) -> Image.Image:
    mm = obs.get("multi_modal_input") or obs.get("multi_modal_data") or {}
    imgs = mm.get("<image>") or mm.get("image") or []
    if not imgs:
        raise RuntimeError(f"observation has no image; obs_keys={list(obs.keys())} mm_keys={list(mm.keys())}")
    x = imgs[0]
    return x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x).convert("RGB")


def health(url: str, timeout: float) -> dict[str, Any]:
    hurl = url.rstrip("/render") + "/health" if url.endswith("/render") else url.rstrip("/") + "/health"
    t0 = time.time()
    try:
        with urllib.request.urlopen(hurl, timeout=timeout) as r:
            body = r.read(4096).decode("utf-8", errors="replace")
            return {"url": hurl, "ok": 200 <= r.status < 300, "status": r.status, "latency": time.time() - t0, "body": body}
    except Exception as exc:
        return {"url": hurl, "ok": False, "latency": time.time() - t0, "exception": repr(exc)}


def load_env_config(yaml_path: Path, renderer_url: str):
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    doc = yaml.safe_load(yaml_path.read_text())
    cfg_dict = dict(doc["env1"]["env_config"])
    cfg_dict["render_backend"] = "http"
    cfg_dict["client_url"] = renderer_url
    cfg_dict["gpu_device"] = None
    fields = {f.name for f in dataclasses.fields(ActiveSpatialEnvConfig)}
    return ActiveSpatialEnvConfig(**{k: v for k, v in cfg_dict.items() if k in fields})


def isolate_partial(out: Path) -> None:
    img_dir = out / "fixed_images"
    has_jsonl = (out / "fixed_transitions.jsonl").exists()
    if img_dir.exists() and not has_jsonl:
        dst = out / "partial_failed_attempt" / time.strftime("%Y%m%d_%H%M%S")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_dir), str(dst))


def timed_call(kind: str, fn, timeout_note: float) -> tuple[Any, dict[str, Any]]:
    t0 = time.time()
    try:
        val = fn()
        return val, {"kind": kind, "ok": True, "start_time": t0, "end_time": time.time(), "latency": time.time() - t0, "exception": None, "timeout_seconds": timeout_note}
    except Exception as exc:
        return None, {"kind": kind, "ok": False, "start_time": t0, "end_time": time.time(), "latency": time.time() - t0, "exception": repr(exc), "timeout_seconds": timeout_note}


def make_record(idx: int, env, seed: int, action: str, obs0, obs1, info1, reward: float, done: bool, out: Path, latency: dict[str, Any], renderer_url: str) -> dict[str, Any]:
    img_dir = out / "fixed_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cond_path = img_dir / f"condition_{idx}.png"
    target_path = img_dir / f"target_{idx}.png"
    first_image(obs0).save(cond_path)
    first_image(obs1).save(target_path)
    xyz0, fwd0, E0 = info1["_pose_before_parts"]
    xyz1, fwd1, E1 = info1["_pose_after_parts"]
    trans = float(np.linalg.norm(np.asarray(xyz1) - np.asarray(xyz0)))
    rot = angle_between(fwd0, fwd1)
    ds = diff_stats(cond_path, target_path)
    cs = image_stats(cond_path)
    ts = image_stats(target_path)
    reject = []
    if done:
        reject.append("done")
    if int(info1.get("collision_count", 0) or 0) > 0:
        reject.append("collision")
    if cs["std"] < 8.0 or ts["std"] < 8.0:
        reject.append("low_info_image")
    if ds["pixel_diff_mean"] <= 0.1:
        reject.append("condition_target_too_similar")
    if action == "move_forward" and trans < 0.05:
        reject.append("move_forward_without_translation")
    if action.startswith("turn_") and rot < 1.0:
        reject.append("turn_without_rotation")
    item = env.current_item or {}
    return {
        "idx": idx,
        "scene_id": item.get("scene_id"),
        "task_id": "/".join([str(item.get("scene_id")), str(item.get("object_label")), str(item.get("preset")), str(seed)]),
        "seed": int(seed),
        "action": action,
        "action_text": f"<think>Phase-3 fixed transition data gate.</think>\n<action>{action}|</action>",
        "pose_before": {"xyz": xyz0, "forward": fwd0},
        "pose_after": {"xyz": xyz1, "forward": fwd1},
        "translation_delta": trans,
        "rotation_delta": rot,
        "condition_image": str(cond_path),
        "target_image": str(target_path),
        "condition_image_std": cs["std"],
        "target_image_std": ts["std"],
        **ds,
        "condition_image_stats": cs,
        "target_image_stats": ts,
        "reward": float(reward),
        "done": bool(done),
        "collision": int(info1.get("collision_count", 0) or 0),
        "render_latency": latency,
        "retry_count": 0,
        "renderer_url": renderer_url,
        "valid_transition": not reject,
        "reject_reason": reject,
    }


def contact_sheet(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    tile_w, tile_h = 220, 250
    canvas = Image.new("RGB", (tile_w * 2, tile_h * len(records)), (245, 245, 245))
    for r_i, r in enumerate(records):
        for col, key in enumerate(["condition_image", "target_image"]):
            img = Image.open(r[key]).convert("RGB").resize((220, 220))
            y = r_i * tile_h
            canvas.paste(img, (col * tile_w, y))
            d = ImageDraw.Draw(canvas)
            label = f"{r['idx']} {key.split('_')[0]} {r['action']} Δt={r['translation_delta']:.2f} Δr={r['rotation_delta']:.1f}"
            d.text((col * tile_w + 4, y + 224), label[:45], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def update_final(final_path: Path, status: str, summary: dict[str, Any]) -> None:
    if not final_path.exists():
        return
    data = json.loads(final_path.read_text())
    data["phase3_fm_data_gate"] = {
        "status": status,
        "report": "../fm_only/phase3_data_gate_audit.json",
        "fixed_transitions": "../fm_only/fixed_transitions.jsonl" if status == "PASS" else None,
        **summary,
    }
    data["phase3_fm_only"] = {
        "status": "FAIL" if status == "FAIL" else "NOT_EXECUTED",
        "reason": "FM-only gate not executed in this turn; only fixed-transition data gate was in scope.",
    }
    data["overall_status"] = "NOT_READY_FOR_TRAINING"
    data["next_gate"] = "phase3_fm_forward_only" if status == "PASS" else "blocked_on_phase3_fm_data"
    jdump(data, final_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--env-yaml", type=Path, default=DEFAULT_ENV_YAML)
    ap.add_argument("--final-delivery", type=Path, default=DEFAULT_FINAL)
    ap.add_argument("--renderer-url", default="http://10.119.30.223:8767/render")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    os.environ["INTERIORGS_HTTP_TIMEOUT"] = str(args.timeout)
    os.environ["INTERIORGS_HTTP_RETRIES"] = str(args.retries)
    args.out.mkdir(parents=True, exist_ok=True)
    isolate_partial(args.out)

    h = health(args.renderer_url, min(args.timeout, 10.0))
    if not h["ok"]:
        audit = {"status": "FAIL", "failed_stage": "renderer_health", "renderer": h}
        jdump(audit, args.out / "phase3_data_gate_audit.json")
        update_final(args.final_delivery, "FAIL", {"failed_stage": "renderer_health"})
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 2

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    env = ActiveSpatialEnv(load_env_config(args.env_yaml, args.renderer_url))

    probes: list[dict[str, Any]] = []
    seed = 1520
    reset1, m1 = timed_call("probe_reset_1", lambda: env.reset(seed=seed), args.timeout)
    probes.append(m1)
    reset2, m2 = timed_call("probe_reset_2_same_seed_scene", lambda: env.reset(seed=seed), args.timeout)
    probes.append(m2)
    if not (m1["ok"] and m2["ok"]):
        audit = {"status": "FAIL", "failed_stage": "minimal_http_probe_reset", "renderer_health": h, "probes": probes}
        jdump(audit, args.out / "phase3_data_gate_audit.json")
        update_final(args.final_delivery, "FAIL", {"failed_stage": "minimal_http_probe_reset"})
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 3
    xyz0, fwd0, _ = pose_parts(env)
    action_text = "<think>Phase-3 fixed transition data gate.</think>\n<action>move_forward|</action>"
    step1, m3 = timed_call("probe_step_move_forward", lambda: env.step(action_text), args.timeout)
    probes.append(m3)
    if not m3["ok"]:
        audit = {"status": "FAIL", "failed_stage": "minimal_http_probe_step", "renderer_health": h, "probes": probes}
        jdump(audit, args.out / "phase3_data_gate_audit.json")
        update_final(args.final_delivery, "FAIL", {"failed_stage": "minimal_http_probe_step"})
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 4

    # Full transitions: keep candidates small and filter strictly.
    candidates = [(1520, "move_forward"), (440, "turn_left"), (830, "turn_right"), (1021, "move_forward")]
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for seed, action in candidates:
        obs0, rm = timed_call(f"reset_seed_{seed}", lambda seed=seed: env.reset(seed=seed), args.timeout)
        if not rm["ok"]:
            rejections.append({"seed": seed, "action": action, "reject_reason": ["reset_failed"], "latency": rm})
            continue
        xyz_b, fwd_b, _ = pose_parts(env)
        action_text = f"<think>Phase-3 fixed transition data gate.</think>\n<action>{action}|</action>"
        step, sm = timed_call(f"step_seed_{seed}_{action}", lambda action_text=action_text: env.step(action_text), args.timeout)
        if not sm["ok"]:
            rejections.append({"seed": seed, "action": action, "reject_reason": ["step_failed"], "latency": sm})
            continue
        obs1, reward, done, info1 = step
        xyz_a, fwd_a, _ = pose_parts(env)
        info1["_pose_before_parts"] = (xyz_b, fwd_b, None)
        info1["_pose_after_parts"] = (xyz_a, fwd_a, None)
        rec = make_record(len(records), env, seed, action, obs0[0], obs1, info1, reward, done, args.out, {"reset": rm, "step": sm}, args.renderer_url)
        if rec["valid_transition"]:
            records.append(rec)
        else:
            rejections.append(rec)
        if len(records) >= 3 and len({r["action"] for r in records}) >= 2:
            break

    latencies = [x["latency"] for x in probes if x.get("ok")] + [
        r["render_latency"]["reset"]["latency"] for r in records
    ] + [r["render_latency"]["step"]["latency"] for r in records]
    summary = {
        "requested_transitions": len(candidates),
        "successfully_rendered": len(records) + len([r for r in rejections if "condition_image" in r]),
        "valid_transitions": len(records),
        "rejected_transitions": len(rejections),
        "average_latency": float(statistics.mean(latencies)) if latencies else None,
        "p95_latency": float(sorted(latencies)[max(0, int(math.ceil(0.95 * len(latencies))) - 1)]) if latencies else None,
        "retry_count": 0,
        "timeout_count": len([r for r in rejections if "failed" in ",".join(r.get("reject_reason", []))]),
        "per_action_count": {a: sum(1 for r in records if r["action"] == a) for a in sorted({r["action"] for r in records})},
    }
    status = "PASS" if len(records) >= 2 and len({r["action"] for r in records}) >= 2 else "FAIL"
    if status == "PASS":
        (args.out / "fixed_transitions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )
    contact_sheet(records, args.out / "fixed_transition_contact_sheet.png")
    audit = {
        "status": status,
        "renderer_health": h,
        "probe_records": probes,
        "summary": summary,
        "valid_records": records,
        "rejected_records": rejections,
        "allow_fm_forward_only_next": status == "PASS",
    }
    jdump(audit, args.out / "phase3_data_gate_audit.json")
    update_final(args.final_delivery, status, summary)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
