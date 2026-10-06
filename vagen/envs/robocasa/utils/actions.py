"""12-D Panda-Omron OSC action helpers for RoboCasa VLM control.

Canonical order (matches robocasa.wrappers.gym_wrapper.PandaOmronKeyConverter):

    [eef_dx, eef_dy, eef_dz, eef_droll, eef_dpitch, eef_dyaw,
     gripper_close, base_x, base_y, base_yaw, torso, control_mode]

Gym dict mapping:

    action.end_effector_position = [0:3]
    action.end_effector_rotation = [3:6]
    action.gripper_close         = [6:7]
    action.base_motion           = [7:11]
    action.control_mode          = [11:12]

This module must stay free of robocasa / gymnasium imports so unit tests
and SFT conversion can run outside the policy venv.
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

ACTION_DIM = 12

ACTION_NAMES = [
    "eef_dx",
    "eef_dy",
    "eef_dz",
    "eef_droll",
    "eef_dpitch",
    "eef_dyaw",
    "gripper_close",
    "base_x",
    "base_y",
    "base_yaw",
    "torso",
    "control_mode",
]

GYM_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}

# StarVLA PandaOmronRoboCasa365DataConfig contract.
# Named keys are concatenated in this order into the 12-D OFT action head.
# Packed LeRobot parquet is remapped via meta/modality.json
# (see ROBOCASA_LEROBOT_SLICES) and is NOT this order.
STARVLA_ACTION_DIM = ACTION_DIM
STARVLA_STATE_DIM = 16
STARVLA_NAVIGATE_KITCHEN_RELPATH = (
    "v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot"
)
STARVLA_ACTION_KEYS = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]
STARVLA_ACTION_KEY_DIMS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
    "action.base_motion": 4,
    "action.control_mode": 1,
}
STARVLA_STATE_KEYS = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]
STARVLA_STATE_KEY_DIMS = {
    "state.base_position": 3,
    "state.base_rotation": 4,
    "state.end_effector_position_relative": 3,
    "state.end_effector_rotation_relative": 4,
    "state.gripper_qpos": 2,
}

# Aliases seen in LeRobot / robomimic rows.
_KEY_ALIASES = {
    "end_effector_position": "action.end_effector_position",
    "end_effector_rotation": "action.end_effector_rotation",
    "gripper_close": "action.gripper_close",
    "base_motion": "action.base_motion",
    "control_mode": "action.control_mode",
    "action.end_effector_position": "action.end_effector_position",
    "action.end_effector_rotation": "action.end_effector_rotation",
    "action.gripper_close": "action.gripper_close",
    "action.base_motion": "action.base_motion",
    "action.control_mode": "action.control_mode",
}

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_VECTOR_RE = re.compile(r"\[\s*" + rf"(?:{_NUMBER}\s*,\s*){{11}}{_NUMBER}" + r"\s*\]")
_NAMED_PAIR_RE = re.compile(
    rf"({'|'.join(ACTION_NAMES)})\s*[=:]\s*({_NUMBER})",
    re.IGNORECASE,
)


def clip_action(
    vec: Sequence[float],
    low: float = -1.0,
    high: float = 1.0,
) -> List[float]:
    """Clip each component of a 12-D action to ``[low, high]``."""
    out = _as_list(vec)
    if len(out) != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}-D action, got {len(out)}")
    return [min(high, max(low, float(x))) for x in out]


def format_action_text(
    vec: Sequence[float],
    precision: int = 4,
    named: bool = False,
) -> str:
    """Format one 12-D action as a bracket list or named assignment."""
    values = clip_action(vec)
    if named:
        parts = [f"{name}={v:.{precision}f}" for name, v in zip(ACTION_NAMES, values)]
        return ", ".join(parts)
    inner = ", ".join(f"{v:.{precision}f}" for v in values)
    return f"[{inner}]"


def parse_vector(text: str) -> List[float]:
    """Parse a single 12-D vector from a bracket list or comma/space list."""
    if text is None:
        raise ValueError("empty action vector")
    raw = str(text).strip()
    if not raw:
        raise ValueError("empty action vector")

    # JSON / Python literal list
    if raw[0] in "[{" and raw[-1] in "]}":
        try:
            parsed = json.loads(raw.replace("(", "[").replace(")", "]"))
        except json.JSONDecodeError:
            parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parse_named_action(raw)
        return _as_dim12(parsed)

    # Bare numbers
    nums = re.findall(_NUMBER, raw)
    if len(nums) >= ACTION_DIM:
        return [float(x) for x in nums[:ACTION_DIM]]
    raise ValueError(f"Could not parse 12-D vector from: {raw[:160]!r}")


def parse_named_action(text: str) -> List[float]:
    """Parse ``eef_dx=0.1, gripper_close=1, ...`` (missing dims default to 0)."""
    if not text or not str(text).strip():
        raise ValueError("empty named action")
    vec = [0.0] * ACTION_DIM
    found = 0
    for match in _NAMED_PAIR_RE.finditer(str(text)):
        name = match.group(1).lower()
        if name not in ACTION_NAMES:
            continue
        vec[ACTION_NAMES.index(name)] = float(match.group(2))
        found += 1
    if found == 0:
        raise ValueError(f"No named action fields in: {str(text)[:160]!r}")
    return vec


def parse_action_block(
    text: str,
    max_actions: Optional[int] = None,
) -> List[List[float]]:
    """Extract one or more 12-D vectors from an ``<action>`` body.

    Accepts:
      - one or more ``[v0, ..., v11]`` lists
      - newline / semicolon separated lists
      - named assignments (one action per line)
    """
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []

    actions: List[List[float]] = []

    for match in _VECTOR_RE.finditer(raw):
        try:
            actions.append(parse_vector(match.group(0)))
        except ValueError:
            continue

    if not actions:
        chunks = [c.strip() for c in re.split(r"[;\n]+", raw) if c.strip()]
        for chunk in chunks:
            try:
                if any(name in chunk for name in ACTION_NAMES):
                    actions.append(parse_named_action(chunk))
                else:
                    actions.append(parse_vector(chunk))
            except ValueError:
                continue

    if max_actions is not None and max_actions > 0:
        actions = actions[: int(max_actions)]
    return actions


def to_gym_action(vec: Sequence[float]) -> Dict[str, List[float]]:
    """Map a 12-D vector to the official RoboCasa gym action dict."""
    values = clip_action(vec)
    return {
        "action.end_effector_position": values[0:3],
        "action.end_effector_rotation": values[3:6],
        "action.gripper_close": values[6:7],
        "action.base_motion": values[7:11],
        "action.control_mode": values[11:12],
    }


# Canonical VLM / gym order slices.
CANONICAL_SLICES = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}

# Official RoboCasa LeRobot modality.json action packing.
ROBOCASA_LEROBOT_SLICES = {
    "base_motion": (0, 4),
    "control_mode": (4, 5),
    "end_effector_position": (5, 8),
    "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}


def remap_stored_action(
    vec: Sequence[float],
    stored_slices: Mapping[str, tuple],
) -> List[float]:
    """Map a stored 12-D vector into canonical gym / VLM order."""
    src = [float(x) for x in vec]
    if len(src) < ACTION_DIM:
        raise ValueError(f"stored action has length {len(src)}, expected {ACTION_DIM}")
    out = [0.0] * ACTION_DIM
    for name, dst in CANONICAL_SLICES.items():
        if name not in stored_slices:
            raise ValueError(f"action layout missing {name}")
        s0, s1 = stored_slices[name]
        d0, d1 = dst
        chunk = src[s0:s1]
        if len(chunk) != (d1 - d0):
            raise ValueError(
                f"{name} stored slice {s0}:{s1} has length {len(chunk)}, expected {d1 - d0}"
            )
        out[d0:d1] = chunk
    return out


def from_demo_row(
    row: Mapping[str, Any],
    action_layout: Optional[Mapping[str, tuple]] = None,
) -> List[float]:
    """Extract a 12-D action from a LeRobot / parquet / dict demo row.

    If ``action_layout`` is given (name -> (start, end) in the stored vector),
    a flat 12-D ``action`` column is remapped into canonical gym order.
    RoboCasa LeRobot v2 packs actions as base/control/eef/gripper, which is
    NOT the eval order.
    """
    if row is None:
        raise ValueError("empty demo row")

    # Direct 12-D under common keys.
    for key in ("action", "actions", "action.action"):
        if key in row:
            candidate = _maybe_array(row[key])
            if candidate is not None and len(candidate) >= ACTION_DIM:
                values = [float(x) for x in candidate[:ACTION_DIM]]
                if action_layout:
                    return remap_stored_action(values, action_layout)
                return values
            if isinstance(row[key], Mapping):
                try:
                    return _from_mapped_action(row[key])
                except ValueError:
                    pass

    # Flattened gym keys on the row itself.
    try:
        return _from_mapped_action(row)
    except ValueError:
        pass

    # Nested under "action.*" prefix already handled; try named fields.
    try:
        return parse_named_action(_join_named_from_row(row))
    except ValueError as exc:
        raise ValueError(f"Could not extract 12-D action from demo row keys={list(row)[:20]}") from exc


def _from_mapped_action(mapping: Mapping[str, Any]) -> List[float]:
    vec = [0.0] * ACTION_DIM
    found = 0
    for raw_key, slc in GYM_SLICES.items():
        value = _lookup_key(mapping, raw_key)
        if value is None:
            continue
        arr = _maybe_array(value)
        if arr is None:
            continue
        start, end = slc
        width = end - start
        if len(arr) < width:
            raise ValueError(f"{raw_key} has length {len(arr)}, expected {width}")
        for i, item in enumerate(arr[:width]):
            vec[start + i] = float(item)
        found += 1
    if found == 0:
        raise ValueError("no gym action keys found")
    return vec


def _lookup_key(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    alias = _KEY_ALIASES.get(key)
    if alias and alias in mapping:
        return mapping[alias]
    # Drop the "action." prefix.
    short = key.split(".", 1)[-1]
    if short in mapping:
        return mapping[short]
    # Nested {"action": {"end_effector_position": ...}}
    if "action" in mapping and isinstance(mapping["action"], Mapping):
        inner = mapping["action"]
        if short in inner:
            return inner[short]
        if key in inner:
            return inner[key]
    return None


def _join_named_from_row(row: Mapping[str, Any]) -> str:
    parts = []
    for name in ACTION_NAMES:
        if name in row:
            parts.append(f"{name}={row[name]}")
    return ", ".join(parts)


def _maybe_array(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        try:
            return parse_vector(value if isinstance(value, str) else value.decode("utf-8"))
        except ValueError:
            return None
    if isinstance(value, Mapping):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    try:
        return [float(x) for x in list(value)]
    except (TypeError, ValueError):
        return None


def _as_list(vec: Sequence[float]) -> List[float]:
    return [float(x) for x in vec]


def _as_dim12(parsed: Any) -> List[float]:
    if isinstance(parsed, (int, float)):
        raise ValueError("scalar is not a 12-D action")
    values = [float(x) for x in list(parsed)]
    if len(values) != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} values, got {len(values)}")
    return values
