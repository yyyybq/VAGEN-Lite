#!/usr/bin/env python3
"""Convert RoboCasa LeRobot v2 demos into LLaMA-Factory sharegpt SFT data.

Scans directories that look like LeRobot v2 roots (meta/info.json, data/*.parquet,
videos/) and writes parquet + jsonl with messages + absolute image paths.

Assistant content uses Active Spatial tags so both base Qwen2.5-VL and
Active Spatial HF actor checkpoints can be fine-tuned on the same format:

    <think>...</think><action>[12-D] ...</action>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("robocasa_sft.convert")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.envs.robocasa.utils.actions import (
    ROBOCASA_LEROBOT_SLICES,
    format_action_text,
    from_demo_row,
)
from vagen.envs.robocasa.utils.prompt import init_observation_template, system_prompt
from vagen.envs.robocasa.utils.tasks import get_horizon, resolve_tasks

PRIMARY_VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "observation.images.robot0_agentview_left",
    "observation.images.agentview_left",
)
WRIST_VIDEO_KEYS = (
    "video.robot0_eye_in_hand",
    "observation.images.robot0_eye_in_hand",
    "observation.images.eye_in_hand",
)
LANG_KEYS = (
    "annotation.human.task_description",
    "language",
    "task",
    "task_description",
)

DUMMY_THINK = "I will follow the demonstration and execute the next kitchen actions."


def _load_info(root: Path) -> Dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    return json.loads(info_path.read_text())


def is_lerobot_v2(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file() and (root / "data").is_dir()


def load_action_layout(root: Path):
    """Read meta/modality.json action packing, or None if already canonical."""
    path = root / "meta" / "modality.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    action = data.get("action") or {}
    layout = {}
    for name, spec in action.items():
        if isinstance(spec, dict) and "start" in spec and "end" in spec:
            layout[str(name)] = (int(spec["start"]), int(spec["end"]))
    needed = {
        "end_effector_position",
        "end_effector_rotation",
        "gripper_close",
        "base_motion",
        "control_mode",
    }
    if not needed.issubset(layout):
        return None
    # Identity layouts do not need remapping.
    from vagen.envs.robocasa.utils.actions import CANONICAL_SLICES

    if all(layout.get(k) == CANONICAL_SLICES[k] for k in needed):
        return None
    return layout


def load_task_texts(root: Path) -> Dict[int, str]:
    """Map task_index -> language from meta/tasks.jsonl."""
    path = root / "meta" / "tasks.jsonl"
    out: Dict[int, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        idx = rec.get("task_index", rec.get("index"))
        text = rec.get("task") or rec.get("task_description") or rec.get("name")
        if idx is None or not text:
            continue
        out[int(idx)] = str(text)
    return out


def scan_lerobot_dirs(data_root: Path, task_hints: Optional[Sequence[str]] = None) -> List[Path]:
    """Find LeRobot v2 dataset roots under ``data_root``."""
    data_root = Path(data_root)
    found: List[Path] = []
    if is_lerobot_v2(data_root):
        found.append(data_root)
    if data_root.is_dir():
        for dirpath, dirnames, _ in os.walk(data_root):
            path = Path(dirpath)
            if is_lerobot_v2(path):
                found.append(path)
                # Do not recurse into a dataset root.
                dirnames[:] = []
    # Dedup
    uniq = []
    seen = set()
    for p in found:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    if task_hints:
        lowered = [t.lower() for t in task_hints]
        filtered = [p for p in uniq if any(t in str(p).lower() for t in lowered)]
        if filtered:
            return filtered
    return uniq


def _iter_parquets(root: Path) -> List[Path]:
    data = root / "data"
    return sorted(data.rglob("*.parquet"))


def _feature_keys(info: Dict[str, Any], prefixes: Sequence[str]) -> List[str]:
    features = info.get("features") or {}
    keys = []
    for name in prefixes:
        if name in features:
            keys.append(name)
    if keys:
        return keys
    for feat_name, spec in features.items():
        dtype = str((spec or {}).get("dtype", "")).lower()
        if dtype in {"video", "image"}:
            for pref in prefixes:
                if pref.split(".")[-1] in feat_name:
                    keys.append(feat_name)
                    break
    return keys


def _chunks_size(info: Dict[str, Any]) -> int:
    return int(info.get("chunks_size") or info.get("chunks_size".upper()) or 1000)


def _video_path(root: Path, info: Dict[str, Any], video_key: str, episode_index: int) -> Optional[Path]:
    chunk = int(episode_index) // _chunks_size(info)
    candidates = [
        root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{int(episode_index):06d}.mp4",
        root / "videos" / video_key / f"episode_{int(episode_index):06d}.mp4",
        root / "videos" / f"{video_key}_episode_{int(episode_index):06d}.mp4",
    ]
    template = (info.get("video_path") or "").strip()
    if template:
        try:
            candidates.insert(
                0,
                root / template.format(
                    episode_chunk=chunk,
                    video_key=video_key,
                    episode_index=int(episode_index),
                ),
            )
        except Exception:
            pass
    for path in candidates:
        if path.is_file():
            return path
    return None


def _extract_frame_cv2(video_path: Path, frame_index: int):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"opencv cannot open {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"opencv failed frame {frame_index} of {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _extract_frame_pyav(video_path: Path, frame_index: int):
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        target = int(frame_index)
        for i, frame in enumerate(container.decode(stream)):
            if i == target:
                return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"pyav exhausted before frame {frame_index} of {video_path}")
    finally:
        container.close()


def _extract_frame_imageio(video_path: Path, frame_index: int):
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    try:
        return reader.get_data(int(frame_index))
    finally:
        reader.close()


def extract_frame(video_path: Path, frame_index: int):
    """Extract an RGB frame with opencv, then pyav, then imageio."""
    errors = []
    for fn, name in (
        (_extract_frame_cv2, "opencv"),
        (_extract_frame_pyav, "pyav"),
        (_extract_frame_imageio, "imageio"),
    ):
        try:
            return fn(video_path, frame_index)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(f"Could not extract frame {frame_index} from {video_path}: {errors}")


def _save_image(arr, dest: Path) -> Path:
    from PIL import Image
    import numpy as np

    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.asarray(arr).astype("uint8"), mode="RGB")
    dest.write_bytes(b"")  # ensure parent exists; overwritten below
    img.save(dest)
    return dest.resolve()


def _row_get(row: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if hasattr(row, "to_dict"):
        return {k: row[k] for k in row.index}
    return dict(row)


def _language_from_row(
    row: Dict[str, Any],
    fallback: str = "",
    task_texts: Optional[Dict[int, str]] = None,
) -> str:
    # RoboCasa LeRobot stores annotation.human.task_description as a task_index.
    idx = _row_get(row, ("task_index", "annotation.human.task_description"), default=None)
    if task_texts and idx is not None:
        try:
            key = int(idx if not hasattr(idx, "__len__") else idx[0])
            if key in task_texts:
                return task_texts[key]
        except (TypeError, ValueError, IndexError):
            pass
    val = _row_get(row, LANG_KEYS, default=fallback)
    if val is None:
        return fallback
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="replace")
    # Skip numeric-only placeholders when we have a fallback task name.
    try:
        int(val)
        return fallback
    except (TypeError, ValueError):
        return str(val)


def convert_dataset(
    root: Path,
    output_dir: Path,
    image_dir: Path,
    task_name: str,
    stride: int = 1,
    horizon: int = 8,
    max_samples: int = 0,
    include_think: bool = True,
    include_wrist: bool = False,
    split: str = "target",
) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to read LeRobot parquet files") from exc

    info = _load_info(root)
    action_layout = load_action_layout(root)
    task_texts = load_task_texts(root)
    if action_layout:
        LOGGER.info("Using modality.json action layout for %s: %s", root, action_layout)
    video_keys = _feature_keys(info, PRIMARY_VIDEO_KEYS)
    wrist_keys = _feature_keys(info, WRIST_VIDEO_KEYS) if include_wrist else []
    if not video_keys:
        # Fall back to common RoboCasa mapped names even if info.json is sparse.
        video_keys = list(PRIMARY_VIDEO_KEYS[:1])
        LOGGER.warning("No video feature in %s; trying %s", root, video_keys[0])

    records: List[Dict[str, Any]] = []
    written = 0
    sys_text = system_prompt(
        format_name="free_think" if include_think else "no_think",
        max_actions_per_step=horizon,
        include_wrist_image=include_wrist,
    )

    for pq in _iter_parquets(root):
        df = pd.read_parquet(pq)
        rows = [_row_to_dict(raw) for _, raw in df.iterrows()]
        extracted = []
        for row in rows:
            try:
                extracted.append(from_demo_row(row, action_layout=action_layout))
            except ValueError:
                extracted.append(None)
        for i, row in enumerate(rows):
            if max_samples and written >= max_samples:
                return records
            frame_index = int(_row_get(row, ("frame_index", "index"), default=0) or 0)
            episode_index = int(_row_get(row, ("episode_index",), default=0) or 0)
            if stride > 1 and (frame_index % stride) != 0:
                continue
            if extracted[i] is None:
                continue
            actions = []
            for j in range(i, min(len(rows), i + max(1, int(horizon)))):
                if extracted[j] is None:
                    break
                if int(_row_get(rows[j], ("episode_index",), default=0) or 0) != episode_index:
                    break
                actions.append(extracted[j])
            if not actions:
                continue
            lang = _language_from_row(row, fallback=task_name, task_texts=task_texts)

            primary_key = video_keys[0]
            video_path = _video_path(root, info, primary_key, episode_index)
            if video_path is None:
                LOGGER.debug("missing video for ep=%s key=%s", episode_index, primary_key)
                continue
            try:
                frame = extract_frame(video_path, frame_index)
            except Exception as exc:
                LOGGER.warning("frame extract failed %s#%s: %s", video_path, frame_index, exc)
                continue

            rel_name = f"{task_name}_{split}_ep{episode_index:06d}_f{frame_index:06d}.jpg"
            img_path = _save_image(frame, image_dir / rel_name)
            images = [str(img_path)]

            if include_wrist and wrist_keys:
                wpath = _video_path(root, info, wrist_keys[0], episode_index)
                if wpath is not None:
                    try:
                        wframe = extract_frame(wpath, frame_index)
                        wname = rel_name.replace(".jpg", "_wrist.jpg")
                        images.append(str(_save_image(wframe, image_dir / wname)))
                    except Exception as exc:
                        LOGGER.debug("wrist extract failed: %s", exc)

            placeholders = "<image>" if len(images) == 1 else "Agent view:\n<image>\nWrist camera:\n<image>"
            user_text = init_observation_template(placeholders, lang)
            action_body = "\n".join(format_action_text(a) for a in actions[:horizon])
            if include_think:
                assistant = f"<think>{DUMMY_THINK}</think><action>\n{action_body}\n</action>"
            else:
                assistant = f"<action>\n{action_body}\n</action>"

            records.append(
                {
                    "messages": [
                        {"role": "system", "content": sys_text},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant},
                    ],
                    "images": images,
                    "task": task_name,
                    "split": split,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                }
            )
            written += 1
    return records


def _expand_horizon(records_src_df_rows, horizon: int) -> None:
    """Placeholder kept for API stability; horizon is applied per-row above."""
    return None


def write_outputs(records: List[Dict[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "robocasa_sft.jsonl"
    parquet_path = output_dir / "robocasa_sft.parquet"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        import pandas as pd

        slim = [{"messages": r["messages"], "images": r["images"]} for r in records]
        pd.DataFrame(slim).to_parquet(parquet_path, index=False)
    except Exception as exc:
        LOGGER.warning("Could not write parquet (%s); jsonl is still available.", exc)
        parquet_path = jsonl_path
    return jsonl_path, parquet_path


def infer_task_name(root: Path, requested: Optional[str] = None) -> str:
    if requested:
        return requested
    name = root.name
    for token in ("target", "pretrain", "human", "human_im"):
        name = name.replace(token, "").strip("_- ")
    return name or root.parent.name


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert RoboCasa LeRobot demos to sharegpt SFT")
    p.add_argument("--data-root", required=True, help="Root that contains LeRobot v2 dirs")
    p.add_argument("--task", default="", help="Single task or comma-separated list")
    p.add_argument("--task-set", default="", help="atomic_seen | composite_seen | seen")
    p.add_argument("--split", default="target", help="Dataset split label (pretrain/target)")
    p.add_argument("--stride", type=int, default=4, help="Keep every N-th frame")
    p.add_argument("--horizon", type=int, default=8, help="Actions per assistant turn")
    p.add_argument("--max-samples", type=int, default=0, help="0 = no cap")
    p.add_argument("--output-dir", default="", help="Where to write jsonl/parquet")
    p.add_argument("--image-dir", default="", help="Where to write extracted JPEG frames")
    p.add_argument("--no-think", action="store_true", help="Omit dummy <think> tags")
    p.add_argument("--include-wrist", action="store_true", help="Also extract wrist camera")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_argparser().parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        LOGGER.error("data-root does not exist: %s", data_root)
        return 2

    tasks: List[str] = []
    if args.task or args.task_set:
        tasks = resolve_tasks(task=args.task or None, task_set=args.task_set or None)
    hints = tasks or None

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (data_root.parent / "robocasa_sft_out")
    image_dir = Path(args.image_dir).expanduser() if args.image_dir else (output_dir / "images")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    roots = scan_lerobot_dirs(data_root, task_hints=hints)
    if not roots:
        LOGGER.error("No LeRobot v2 dirs under %s", data_root)
        return 2
    LOGGER.info("Found %d LeRobot dataset(s)", len(roots))

    all_records: List[Dict[str, Any]] = []
    remaining = args.max_samples
    for root in roots:
        task_name = infer_task_name(root, requested=tasks[0] if len(tasks) == 1 else None)
        if tasks and not any(t.lower() in str(root).lower() or t.lower() == task_name.lower() for t in tasks):
            # Still convert if the user pointed at a single explicit root.
            if len(roots) > 1:
                continue
        cap = remaining if remaining else 0
        LOGGER.info("Converting %s as task=%s", root, task_name)
        recs = convert_dataset(
            root=root,
            output_dir=output_dir,
            image_dir=image_dir,
            task_name=task_name,
            stride=max(1, int(args.stride)),
            horizon=max(1, int(args.horizon)),
            max_samples=cap,
            include_think=not args.no_think,
            include_wrist=args.include_wrist,
            split=args.split,
        )
        all_records.extend(recs)
        if remaining:
            remaining = max(0, remaining - len(recs))
            if remaining == 0:
                break

    if not all_records:
        LOGGER.error("No SFT samples written. Check parquet action columns and videos/.")
        return 3

    jsonl_path, parquet_path = write_outputs(all_records, output_dir)
    LOGGER.info("Wrote %d samples", len(all_records))
    LOGGER.info("jsonl   %s", jsonl_path)
    LOGGER.info("parquet %s", parquet_path)
    LOGGER.info("images  %s", image_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
