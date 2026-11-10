"""Pure-Python TVRBench transition model on a frozen reachable set (CONTRACT_wave2 4.2-4.4, 5.1.3).

kin_env reproduces TVRBench's ThorEnv / ThorViewEnv transitions EXACTLY (thor_env.py:131-142, 220-333, 402-435;
thor_view_env.py:449-523) without ai2thor, given the reachable set the THOR worker injects (the frozen per-task
start-pose set R(task), CONTRACT 4.3):

  Pose        (x, z, rot, hor): x, z = r2(v) = round(float(v), 2) + 0.0; rot = int(round(rotation_y)) % 360;
              hor = int(round(horizon)).  y is ignored.
  moves       heading deltas _MOVE_DELTAS; MoveAhead (dx,dz), MoveBack (-dx,-dz), MoveLeft (-dz,dx), MoveRight
              (dz,-dx); target cell (r2(x + dx*0.25), r2(z + dz*0.25)); success iff the cell is in the set, else
              the pose is unchanged and success is False.  reach None = free space (every move succeeds; the
              replay drop rule's "implied" pose).
  rotations   RotateRight +45, RotateLeft -45 (mod 360); always succeed.
  looks       LookUp fails (no change) at hor <= -30, else hor -= 30; LookDown fails at hor >= 30, else +30.
  Stop        no Controller step; the episode ends; success = is_exact(pose, target).  step() returns
              (pose, None) for Stop (ThorViewEnv's info has no action_success for Stop).
  InvalidAction  (an unparseable reply) no change, step spent, action_success None (ThorViewEnv: action None,
              action_executed False).
  valid list  the moves (MoveAhead, MoveBack, MoveLeft, MoveRight order) whose cell is in the set, then
              RotateRight, RotateLeft, then LookUp if hor > -30, LookDown if hor < 30, then Stop.
  user text   tvr_text.build_concat_user_text(valid) (TVRBench verbatim).

Python 3.10 and 3.12, stdlib only (CONTRACT R3).
"""

from __future__ import annotations

import hashlib
import math
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Tuple

from vpd.tvr import pilot_cfg, tvr_text

GRID = 0.25
ROT_STEP = 45
HOR_STEP = 30
HOR_MIN = int(tvr_text.HORIZON_MIN)
HOR_MAX = int(tvr_text.HORIZON_MAX)
ACTIONS = pilot_cfg.ACTIONS
NON_STOP = pilot_cfg.NON_STOP
MOVES = ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight")
INVALID = "InvalidAction"

# thor_env.py:133-142
_MOVE_DELTAS = {
    0: (0, 1),
    45: (1, 1),
    90: (1, 0),
    135: (1, -1),
    180: (0, -1),
    225: (-1, -1),
    270: (-1, 0),
    315: (-1, 1),
}

Pose = namedtuple("Pose", "x z rot hor")


def r2(v) -> float:
    """CONTRACT 4.2 _r: round(float(v), 2) + 0.0 (the + 0.0 turns -0.0 into 0.0)."""
    return round(float(v), 2) + 0.0


def pose_of(d) -> Pose:
    """Pose of {"position": {x, y, z}, "rotation_y", "horizon"} (a task pose or a ThorEnv.get_state()), or of a
    Pose / (x, z, rot, hor) tuple."""
    if isinstance(d, Pose):
        return d
    if isinstance(d, (tuple, list)):
        x, z, rot, hor = d
    else:
        x, z = d["position"]["x"], d["position"]["z"]
        rot, hor = d["rotation_y"], d["horizon"]
    return Pose(r2(x), r2(z), int(round(float(rot))) % 360, int(round(float(hor))))


def pose_dict(p: Pose, y: float = 0.0) -> Dict:
    """The TVRBench pose dict of a Pose (y as given; rotation and horizon as floats, like ThorEnv.get_state)."""
    return {"position": {"x": p.x, "y": float(y), "z": p.z}, "rotation_y": float(p.rot), "horizon": float(p.hor)}


def digest_cells(cells: Iterable[Tuple[float, float]]) -> str:
    """CONTRACT 4.3: sha256 hex of "\\n".join(f"{x:.2f},{z:.2f}" for (x, z) in sorted(cells)) (cells normalized by
    r2, so -0.0 never prints as "-0.00")."""
    norm = sorted({(r2(x), r2(z)) for (x, z) in cells})
    text = "\n".join(f"{x:.2f},{z:.2f}" for (x, z) in norm)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReachSet:
    """An immutable reachable set: cells = frozenset of (r2(x), r2(z)); digest (CONTRACT 4.3); n = len(cells)."""

    __slots__ = ("cells", "digest", "n")

    def __init__(self, cells: Iterable[Tuple[float, float]]):
        c = frozenset((r2(x), r2(z)) for (x, z) in cells)
        object.__setattr__(self, "cells", c)
        object.__setattr__(self, "n", len(c))
        object.__setattr__(self, "digest", digest_cells(c))

    def __setattr__(self, k, v):
        raise AttributeError("ReachSet is immutable")

    @classmethod
    def from_positions(cls, positions: List[Dict]) -> "ReachSet":
        """From GetReachablePositions' actionReturn (dicts with x, y, z)."""
        return cls((p["x"], p["z"]) for p in positions)

    @classmethod
    def from_cells(cls, cells: Iterable[Tuple[float, float]]) -> "ReachSet":
        """From (x, z) pairs, e.g. ThorEnv._get_reachable_set() (its round(.., 2) values may hold -0.0)."""
        return cls(cells)

    def __contains__(self, xz) -> bool:
        return (r2(xz[0]), r2(xz[1])) in self.cells

    def __len__(self) -> int:
        return self.n

    def __eq__(self, other) -> bool:
        return isinstance(other, ReachSet) and other.cells == self.cells

    def __hash__(self) -> int:
        return hash(self.cells)

    def __repr__(self) -> str:
        return f"ReachSet(n={self.n}, digest={self.digest[:12]})"


def move_target(p: Pose, action: str) -> Optional[Tuple[float, float]]:
    """The cell a move action goes to (thor_env.py:220-250); None for a heading that is not a multiple of 45
    (ThorEnv then fails the move; cannot happen with 45-degree rotations)."""
    if p.rot not in _MOVE_DELTAS:
        return None
    dx, dz = _MOVE_DELTAS[p.rot]
    if action == "MoveBack":
        dx, dz = -dx, -dz
    elif action == "MoveLeft":
        dx, dz = -dz, dx
    elif action == "MoveRight":
        dx, dz = dz, -dx
    elif action != "MoveAhead":
        raise ValueError(f"not a move action: {action!r}")
    return r2(p.x + dx * GRID), r2(p.z + dz * GRID)


def _in(reach: Optional[ReachSet], cell: Optional[Tuple[float, float]]) -> bool:
    if cell is None:
        return False
    return True if reach is None else cell in reach.cells


def valid_actions(p: Pose, reach: Optional[ReachSet], include_stop: bool = True) -> List[str]:
    """ThorEnv.get_valid_actions at pose p (thor_env.py:402-435); reach None = free space."""
    valid = [a for a in MOVES if _in(reach, move_target(p, a))]
    valid.extend(["RotateRight", "RotateLeft"])
    if p.hor > HOR_MIN:
        valid.append("LookUp")
    if p.hor < HOR_MAX:
        valid.append("LookDown")
    if include_stop:
        valid.append("Stop")
    return valid


def step(p: Pose, action: str, reach: Optional[ReachSet]) -> Tuple[Pose, Optional[bool]]:
    """One transition (thor_env.py:278-333): (new pose, success).  Stop and InvalidAction -> (p, None)."""
    if action in MOVES:
        cell = move_target(p, action)
        if not _in(reach, cell):
            return p, False
        return Pose(cell[0], cell[1], p.rot, p.hor), True
    if action == "RotateRight":
        return Pose(p.x, p.z, (p.rot + ROT_STEP) % 360, p.hor), True
    if action == "RotateLeft":
        return Pose(p.x, p.z, (p.rot - ROT_STEP) % 360, p.hor), True
    if action == "LookUp":
        if p.hor <= HOR_MIN:
            return p, False
        return Pose(p.x, p.z, p.rot, p.hor - HOR_STEP), True
    if action == "LookDown":
        if p.hor >= HOR_MAX:
            return p, False
        return Pose(p.x, p.z, p.rot, p.hor + HOR_STEP), True
    if action in ("Stop", INVALID):
        return p, None
    raise ValueError(f"unknown action {action!r}")


def is_exact(p: Pose, target: Pose) -> bool:
    """TVRBench metrics.is_exact_match on poses: xz distance < 0.01, rotation error < 1, horizon error < 1."""
    dpos = math.sqrt((p.x - target.x) ** 2 + (p.z - target.z) ** 2)
    dr = abs(p.rot - target.rot) % 360
    dr = min(dr, 360 - dr)
    return dpos < 0.01 and dr < 1.0 and abs(p.hor - target.hor) < 1.0


def within_one(p: Pose, target: Pose, reach: Optional[ReachSet]) -> bool:
    """p is the target, or one non-Stop transition (on `reach`) maps p to the target (CONTRACT 4.7 within1,
    without the 'stopped' condition)."""
    if is_exact(p, target):
        return True
    for a in NON_STOP:
        q, ok = step(p, a, reach)
        if ok and is_exact(q, target):
            return True
    return False


class KinEpisode:
    """One TVR episode in kin_env (the blind arm, the kin shadow, floors).  The harness enforces the cap.

    reset() -> {"obs_str", "valid", "state", "pose"}
    step(a) -> {"obs_str", "valid", "state", "pose", "info": {"action", "action_executed", "action_success",
                "success", "stopped"}, "done"}
    `a` is one of ACTIONS or "InvalidAction".  Info mirrors ThorViewEnv._sync_step: InvalidAction -> action None,
    action_executed False; Stop -> action_executed True, success = is_exact, stopped True, done; a move / rotate /
    look -> action_executed True and its success.  success is None except on Stop; stopped is False except on Stop.
    """

    def __init__(self, task: Dict, positions: List[Dict]):
        self.task = task
        self.reach = positions if isinstance(positions, ReachSet) else ReachSet.from_positions(positions)
        self.start = pose_of(task["start"])
        self.target = pose_of(task["target"])
        self.y = round(float(task["start"]["position"].get("y", 0.0)), 4)
        self.pose = self.start
        self.done = False
        self.steps = 0

    def _obs(self) -> Dict:
        valid = valid_actions(self.pose, self.reach, include_stop=True)
        return {"obs_str": tvr_text.build_concat_user_text(valid), "valid": valid,
                "state": pose_dict(self.pose, self.y), "pose": self.pose}

    def reset(self) -> Dict:
        self.pose, self.done, self.steps = self.start, False, 0
        return self._obs()

    def step(self, action: str) -> Dict:
        if self.done:
            raise RuntimeError("episode is done")
        if action not in ACTIONS and action != INVALID:
            raise ValueError(f"unknown action {action!r}")
        info = {"action": None if action == INVALID else action, "action_executed": action != INVALID,
                "action_success": None, "success": None, "stopped": False}
        if action == "Stop":
            info["success"] = is_exact(self.pose, self.target)
            info["stopped"] = True
            self.done = True
        elif action != INVALID:
            self.pose, info["action_success"] = step(self.pose, action, self.reach)
        self.steps += 1
        out = self._obs()
        out["info"] = info
        out["done"] = self.done
        return out
