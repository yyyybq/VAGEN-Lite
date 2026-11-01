"""Reward line (phase 2, direction 1, the user's brief of 2026-11-01): the distance table's neighbour-cell fallback.

A pose whose contract key (1 cm / 1 deg) is missing from the task's table gets phi = 0 from ``distance_table`` (a
float drift of the runtime pose off the table's grid, reward/miss_shift.py).  With a per-turn credit one such miss is a
spike of about scale x phi (up to 3.3 down and back up), so with the env's ``optimal_progress_near`` (off by default)
a missing key takes the smallest reachable distance among its 26 neighbours (x, y +-1 cm, yaw +-1 deg); a pose with no
neighbour in the table, or only neighbours with no success reachable, stays unknown (phi 0) as before.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from .difficulty import _split, contract_keys
from .distance_table import DEFAULT_SLACK, TableCache, lookup, phi

OFF = 1 << 21
NEIGHBOURS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


def unpack(k: int):
    """A contract key's (x, y, yaw) cells (difficulty.contract_keys' packing)."""
    return (k >> 31) - OFF, ((k >> 9) & ((1 << 22) - 1)) - OFF, k & 511


def pack(kx: int, ky: int, kyaw: int, n_yaw: int = 360) -> int:
    return ((kx + OFF) << 31) | ((ky + OFF) << 9) | (kyaw % n_yaw)


def neighbour_distance(table: Mapping[str, Any], key: int) -> Optional[int]:
    """The smallest reachable distance among the 26 cells next to ``key``, None when none of them has one."""
    kx, ky, kyaw = unpack(int(key))
    cand = np.array([pack(kx + dx, ky + dy, kyaw + dz) for dx, dy, dz in NEIGHBOURS], dtype=np.int64)
    keys = table["keys"]
    if not len(keys):
        return None
    j = np.minimum(np.searchsorted(keys, cand), len(keys) - 1)
    hit = keys[j] == cand
    ds = table["dist"][j][hit]
    ds = ds[ds >= 0]
    return int(ds.min()) if len(ds) else None


def lookup_near(table: Mapping[str, Any], c2w: Any) -> Dict[str, Any]:
    """{"distance": d or None, "near": whether it came from a neighbour}."""
    d = lookup(table, c2w)
    if d is not None:
        return {"distance": d, "near": False}
    P, R = _split(c2w)
    nd = neighbour_distance(table, int(contract_keys(P, R)[0]))
    return {"distance": nd, "near": nd is not None}


class NearTableCache(TableCache):
    """TableCache whose lookups fall back to the neighbour cells (``optimal_near`` says when one did)."""

    def progress(self, item: Mapping[str, Any], c2w: Any, success: bool, slack: int = DEFAULT_SLACK) -> Dict[str, Any]:
        table = self.get(str(item.get("task_id")))
        d_star = int((item.get("difficulty") or {}).get("d_star") or 0)
        found = lookup_near(table, c2w) if table is not None else {"distance": None, "near": False}
        return {"optimal_distance": found["distance"], "optimal_table": table is not None,
                "optimal_near": found["near"], "optimal_progress": phi(found["distance"], d_star, slack, success)}
