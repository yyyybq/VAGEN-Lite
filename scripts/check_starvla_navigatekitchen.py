#!/usr/bin/env python3
"""Verify StarVLA action_dim=12 against the local NavigateKitchen LeRobot dump.

Does not import starVLA or robocasa. Checks:

* LeRobot v2.1 path: v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot
* packed action column is 12-D, state is 16-D
* modality.json packed slices match ROBOCASA_LEROBOT_SLICES
* StarVLA named-key concat is gym/canonical 12-D
* mixture string exists in the StarVLA RoboCasa365 data_config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vagen.envs.robocasa.utils.actions import (
    ACTION_DIM,
    CANONICAL_SLICES,
    ROBOCASA_LEROBOT_SLICES,
    STARVLA_ACTION_DIM,
    STARVLA_ACTION_KEY_DIMS,
    STARVLA_ACTION_KEYS,
    STARVLA_NAVIGATE_KITCHEN_RELPATH,
    STARVLA_STATE_DIM,
    STARVLA_STATE_KEY_DIMS,
    STARVLA_STATE_KEYS,
)

STARVLA_DATA_CONFIG = (
    REPO
    / "third_party"
    / "starVLA"
    / "examples"
    / "simBenchmarks"
    / "Robocasa_365"
    / "train_files"
    / "data_registry"
    / "data_config.py"
)
MIXTURE_NAME = "robocasa365_navigate_kitchen_pretrain_human"


def _fail(msg: str) -> None:
    raise SystemExit(f"[check] {msg}")


def load_json(path: Path):
    if not path.is_file():
        _fail(f"missing {path}")
    return json.loads(path.read_text())


def check_lerobot(root: Path) -> None:
    info = load_json(root / "meta" / "info.json")
    features = info.get("features") or {}
    action = features.get("action") or {}
    state = features.get("observation.state") or {}
    if list(action.get("shape") or []) != [ACTION_DIM]:
        _fail(f"action shape {action.get('shape')} != [{ACTION_DIM}] at {root}")
    if list(state.get("shape") or []) != [STARVLA_STATE_DIM]:
        _fail(f"state shape {state.get('shape')} != [{STARVLA_STATE_DIM}] at {root}")
    if info.get("fps") != 20:
        print(f"[check] warn fps={info.get('fps')} (expected 20)")

    modality = load_json(root / "meta" / "modality.json")
    packed = {
        name: (int(spec["start"]), int(spec["end"]))
        for name, spec in (modality.get("action") or {}).items()
        if isinstance(spec, dict) and "start" in spec
    }
    for name, slc in ROBOCASA_LEROBOT_SLICES.items():
        if packed.get(name) != slc:
            _fail(f"modality.json action.{name}={packed.get(name)} != packed {slc}")

    video = modality.get("video") or {}
    for cam in (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ):
        if cam not in video:
            _fail(f"modality.json missing video.{cam}")
        orig = (video[cam] or {}).get("original_key")
        expected = f"observation.images.{cam}"
        if orig != expected:
            _fail(f"video.{cam} original_key={orig!r} != {expected!r}")

    lang = (modality.get("annotation") or {}).get("human.task_description") or {}
    if lang.get("original_key") != "annotation.human.task_description":
        _fail("language key is not annotation.human.task_description")

    if not (root / "data").is_dir():
        _fail(f"missing data/ under {root}")
    if not (root / "videos").is_dir():
        _fail(f"missing videos/ under {root}")
    if not (root / "meta" / "tasks.jsonl").is_file():
        _fail(f"missing meta/tasks.jsonl under {root}")


def check_starvla_contract() -> None:
    if STARVLA_ACTION_DIM != ACTION_DIM:
        _fail(f"STARVLA_ACTION_DIM={STARVLA_ACTION_DIM} != ACTION_DIM={ACTION_DIM}")
    if sum(STARVLA_ACTION_KEY_DIMS[k] for k in STARVLA_ACTION_KEYS) != ACTION_DIM:
        _fail("StarVLA action key dims do not sum to 12")
    if sum(STARVLA_STATE_KEY_DIMS[k] for k in STARVLA_STATE_KEYS) != STARVLA_STATE_DIM:
        _fail("StarVLA state key dims do not sum to 16")
    offset = 0
    for key in STARVLA_ACTION_KEYS:
        name = key.split(".", 1)[1]
        dim = STARVLA_ACTION_KEY_DIMS[key]
        if CANONICAL_SLICES[name] != (offset, offset + dim):
            _fail(
                f"{key} concat slice {(offset, offset + dim)} "
                f"!= canonical {CANONICAL_SLICES[name]}"
            )
        offset += dim


def check_mixture() -> None:
    if not STARVLA_DATA_CONFIG.is_file():
        _fail(f"missing StarVLA data_config: {STARVLA_DATA_CONFIG}")
    text = STARVLA_DATA_CONFIG.read_text()
    if MIXTURE_NAME not in text:
        _fail(f"{MIXTURE_NAME} not registered in {STARVLA_DATA_CONFIG}")
    rel = STARVLA_NAVIGATE_KITCHEN_RELPATH
    rel_no_leaf = rel[:-len("/lerobot")] if rel.endswith("/lerobot") else rel
    if rel not in text and rel_no_leaf not in text:
        _fail(f"{rel} not in data_config")
    if "action_dim: 12" not in (
        STARVLA_DATA_CONFIG.parent.parent / "starvla_qwenoft_navigatekitchen.yaml"
    ).read_text():
        _fail("NavigateKitchen yaml is missing action_dim: 12")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/playground/Datasets/robocasa365"),
    )
    parser.add_argument("--relpath", default=STARVLA_NAVIGATE_KITCHEN_RELPATH)
    args = parser.parse_args()
    root = args.data_root / args.relpath
    if not root.is_dir():
        _fail(f"NavigateKitchen LeRobot root missing: {root}")
    check_starvla_contract()
    check_lerobot(root)
    check_mixture()
    print(f"[check] ok action_dim={STARVLA_ACTION_DIM} state_dim={STARVLA_STATE_DIM}")
    print(f"[check] packed slices={ROBOCASA_LEROBOT_SLICES}")
    print(f"[check] starvla keys={STARVLA_ACTION_KEYS}")
    print(f"[check] mix={MIXTURE_NAME}")
    print(f"[check] path={root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
