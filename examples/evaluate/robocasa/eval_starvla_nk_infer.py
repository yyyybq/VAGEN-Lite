#!/usr/bin/env python3
"""Load a trained StarVLA-OFT NavigateKitchen ckpt and run offline inference.

Checks that PolicyServerWrapper can:
  1. restore the checkpoint + norm stats
  2. produce a finite (B, 16, 12) unnormalized action chunk
  3. run both the 3-camera training contract and the 1-camera official eval
     contract (left agentview only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


TRAIN_CAMS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


def _load_frame(mp4: Path) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read first frame: {mp4}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, size: tuple[int, int] = (224, 224)) -> Image.Image:
    return Image.fromarray(img).resize(size, Image.BILINEAR)


def _summarize(name: str, actions: np.ndarray) -> dict:
    a = np.asarray(actions)
    finite = bool(np.isfinite(a).all())
    slices = {
        "eef_pos": a[..., 0:3],
        "eef_rot": a[..., 3:6],
        "gripper": a[..., 6:7],
        "base_motion": a[..., 7:11],
        "control_mode": a[..., 11:12],
    }
    out = {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite": finite,
        "abs_mean": float(np.abs(a).mean()) if a.size else None,
        "abs_max": float(np.abs(a).max()) if a.size else None,
        "slices": {
            k: {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
            for k, v in slices.items()
        },
        "first_step": a[0, 0].tolist() if a.ndim == 3 else a[0].tolist(),
    }
    ok = finite and a.ndim == 3 and a.shape[-1] == 12 and a.shape[-2] == 16
    out["ok"] = bool(ok)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default=(
            "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA/playground/Checkpoints/"
            "starvla_qwenoft_NavigateKitchen_20261001_025134/final_model/pytorch_model.pt"
        ),
    )
    p.add_argument(
        "--data-root",
        default=(
            "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/playground/Datasets/robocasa365/"
            "v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot"
        ),
    )
    p.add_argument("--out", default="")
    args = p.parse_args()

    root = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
    star = root / "third_party/starVLA"
    sys.path.insert(0, str(star))
    os.chdir(star)

    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"ckpt missing: {ckpt}")

    t0 = time.time()
    wrapper = PolicyServerWrapper(ckpt_path=str(ckpt), device="cuda", use_bf16=True)
    load_s = time.time() - t0
    meta = wrapper.metadata
    print("[infer] loaded in {:.1f}s metadata={}".format(load_s, json.dumps(meta, default=str)))

    # 1) dummy 3-view, training camera count
    dummy = [Image.fromarray(np.full((224, 224, 3), c, dtype=np.uint8)) for c in (40, 120, 200)]
    dummy_ex = [{"image": dummy, "lang": "Navigate to the fridge."}]
    t1 = time.time()
    dummy_out = wrapper.predict_action(dummy_ex)
    dummy_ms = (time.time() - t1) * 1000
    dummy_sum = _summarize("dummy_3cam", dummy_out["actions"])
    dummy_sum["latency_ms"] = dummy_ms
    print("[infer] dummy_3cam", dummy_sum["shape"], "finite", dummy_sum["finite"], f"{dummy_ms:.0f}ms")

    # 2) real first frame, 3 training cameras
    data = Path(args.data_root)
    real_imgs = []
    for cam in TRAIN_CAMS:
        mp4 = data / "videos/chunk-000" / cam / "episode_000000.mp4"
        real_imgs.append(_resize(_load_frame(mp4)))
    real_ex = [{"image": real_imgs, "lang": "Navigate to the fridge."}]
    t2 = time.time()
    real_out = wrapper.predict_action(real_ex)
    real_ms = (time.time() - t2) * 1000
    real_sum = _summarize("real_3cam_ep0", real_out["actions"])
    real_sum["latency_ms"] = real_ms
    real_sum["lang"] = "Navigate to the fridge."
    print("[infer] real_3cam", real_sum["shape"], "finite", real_sum["finite"], f"{real_ms:.0f}ms")
    print("[infer] real first step", np.round(real_out["actions"][0, 0], 4).tolist())

    # 3) official eval contract: left agentview only
    left_ex = [{"image": [real_imgs[0]], "lang": "Navigate to the fridge."}]
    t3 = time.time()
    left_out = wrapper.predict_action(left_ex)
    left_ms = (time.time() - t3) * 1000
    left_sum = _summarize("real_1cam_left", left_out["actions"])
    left_sum["latency_ms"] = left_ms
    print("[infer] real_1cam_left", left_sum["shape"], "finite", left_sum["finite"], f"{left_ms:.0f}ms")

    report = {
        "ckpt": str(ckpt),
        "load_s": load_s,
        "metadata": meta,
        "cases": [dummy_sum, real_sum, left_sum],
        "all_ok": all(c["ok"] for c in [dummy_sum, real_sum, left_sum]),
    }
    out = Path(args.out) if args.out else Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/playground/Checkpoints/starvla_nk_eval/infer_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[infer] wrote {out}")
    print("[infer] ALL_OK" if report["all_ok"] else "[infer] FAIL")
    return 0 if report["all_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
