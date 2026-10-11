"""Same-source scene detection (InteriorGS scenes re-exported under several ids).

v1 isolated 38 scenes as suspected same-source duplicates across splits.  This
module gives a geometric fingerprint that is invariant to translation, to the
8 axis-aligned symmetries of the floor plane (90-degree rotations and
mirroring, e.g. a structure/label Y-sign flip) and robust to small label
noise (a few objects added/removed/relabelled, centimetre jitter).

    fp  = fingerprint(scene)                     # JSON-serialisable dict
    s   = similarity(fp_a, fp_b)                  # float in [0, 1]
    grp = group_same_source({sid: fp, ...}, threshold=0.6)

similarity: for each of the 8 symmetries g, candidate translations are voted
by pairs of objects with compatible footprint dimensions (labels are NOT
required to agree, so relabelling does not hurt), the best few translations
are refined, and the score is the fraction of objects of both scenes that
find a dimension-compatible partner within ``MATCH_RADIUS_M`` after the
transform: (matched_a + matched_b) / (n_a + n_b).  Different scenes score
near 0 because a common rigid transform aligning many objects does not exist.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

FINGERPRINT_VERSION = "scene_identity_v1"
MATCH_RADIUS_M = 0.25
DIM_TOL_M = 0.10
DIM_TOL_REL = 0.10
VOTE_BIN_M = 0.2
TOP_BINS = 4
MAX_OBJECTS = 400
_EXCLUDE = {"floor", "ceiling", "room"}
_MAX_EXTENT_M = 8.0

# the 8 symmetries of the square acting on (x, y)
_SYMS = [np.array(m, dtype=float) for m in (
    [[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
    [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[-1, 0], [0, 1]], [[0, -1], [-1, 0]],
)]


def _objects_of(scene: Any) -> List[tuple]:
    """(label, min xyz, max xyz) from a Scene, a list of Obj or a labels.json list."""
    items = getattr(scene, "objects", scene)
    out = []
    for it in items or []:
        if isinstance(it, Mapping):
            bbox = it.get("bounding_box") or []
            if len(bbox) < 8:
                continue
            pts = np.array([[c["x"], c["y"], c["z"]] for c in bbox], dtype=float)
            out.append((str(it.get("label", "")).strip().lower(), pts.min(0), pts.max(0)))
        else:
            out.append((str(it.label), np.asarray(it.aabb_min, float), np.asarray(it.aabb_max, float)))
    return out


def fingerprint(scene: Any) -> Dict[str, Any]:
    objs = []
    for label, mn, mx in _objects_of(scene):
        ext = mx - mn
        if label in _EXCLUDE or max(ext[0], ext[1]) > _MAX_EXTENT_M:
            continue
        objs.append((label, (mn[:2] + mx[:2]) / 2.0, sorted([float(ext[0]), float(ext[1])]), float(ext[2])))
    # deterministic subsample of very large scenes: keep the largest footprints
    objs.sort(key=lambda o: -(o[2][0] * o[2][1]))
    objs = objs[:MAX_OBJECTS]
    centres = np.array([o[1] for o in objs], dtype=float).reshape(-1, 2)
    dims = np.array([[o[2][0], o[2][1], o[3]] for o in objs], dtype=float).reshape(-1, 3)
    tokens = Counter(f"{round(d[0] / 0.1)}_{round(d[1] / 0.1)}_{round(d[2] / 0.1)}" for d in dims)
    return {
        "version": FINGERPRINT_VERSION,
        "n": len(objs),
        "labels": [o[0] for o in objs],
        "centres": centres.round(4).tolist(),
        "dims": dims.round(4).tolist(),
        "label_counts": dict(Counter(o[0] for o in objs)),
        "dim_tokens": dict(tokens),
    }


def _compatible(da: np.ndarray, db: np.ndarray) -> np.ndarray:
    tol = DIM_TOL_M + DIM_TOL_REL * np.maximum(da[:, None, :], db[None, :, :])
    return np.all(np.abs(da[:, None, :] - db[None, :, :]) <= tol, axis=2)


def similarity(fp_a: Mapping[str, Any], fp_b: Mapping[str, Any]) -> float:
    ca = np.asarray(fp_a["centres"], float).reshape(-1, 2)
    cb = np.asarray(fp_b["centres"], float).reshape(-1, 2)
    na, nb = len(ca), len(cb)
    if na == 0 or nb == 0:
        return 1.0 if na == nb == 0 else 0.0
    comp = _compatible(np.asarray(fp_a["dims"], float).reshape(-1, 3), np.asarray(fp_b["dims"], float).reshape(-1, 3))
    ia, ib = np.nonzero(comp)
    if len(ia) == 0:
        return 0.0
    best = 0.0
    for g in _SYMS:
        gb = cb @ g.T
        trans = ca[ia] - gb[ib]
        keys = np.floor(trans / VOTE_BIN_M).astype(np.int64)
        uniq, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        for u in np.argsort(-cnt)[:TOP_BINS]:
            if cnt[u] < 2 and best > 0:
                break
            nbh = np.all(np.abs(keys - uniq[u]) <= 1, axis=1)
            t = np.median(trans[nbh], axis=0)
            d = np.linalg.norm(ca[:, None, :] - (gb[None, :, :] + t), axis=2)
            ok = comp & (d <= MATCH_RADIUS_M)
            score = (np.count_nonzero(ok.any(axis=1)) + np.count_nonzero(ok.any(axis=0))) / float(na + nb)
            best = max(best, score)
            if best >= 0.999:
                return 1.0
    return float(best)


def _token_cosine(fps: Sequence[Mapping[str, Any]]) -> np.ndarray:
    vocab: Dict[str, int] = {}
    rows = []
    for fp in fps:
        rows.append({vocab.setdefault(k, len(vocab)): v for k, v in fp.get("dim_tokens", {}).items()})
    M = np.zeros((len(fps), max(1, len(vocab))))
    for i, r in enumerate(rows):
        for j, v in r.items():
            M[i, j] = v
    norm = np.linalg.norm(M, axis=1)
    norm[norm == 0] = 1.0
    M /= norm[:, None]
    return M @ M.T


def group_same_source(fingerprints: Mapping[str, Mapping[str, Any]], threshold: float = 0.6,
                      prefilter: float = 0.3, return_pairs: bool = False):
    """Connected components of the graph {similarity >= threshold}.

    A cheap cosine prefilter on quantised-dimension token histograms (invariant
    to every rigid motion) skips pairs that cannot be the same scene.
    Returns groups of >= 2 scene ids (each sorted, groups sorted); with
    ``return_pairs`` also the list of (a, b, similarity) edges.
    """
    ids = sorted(fingerprints)
    fps = [fingerprints[i] for i in ids]
    cos = _token_cosine(fps) if ids else np.zeros((0, 0))
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ni, nj = fps[i]["n"], fps[j]["n"]
            if cos[i, j] < prefilter or min(ni, nj) < 0.5 * max(ni, nj):
                continue
            s = similarity(fps[i], fps[j])
            if s >= threshold:
                pairs.append((ids[i], ids[j], s))
                parent[find(i)] = find(j)
    groups: Dict[int, List[str]] = {}
    for i, sid in enumerate(ids):
        groups.setdefault(find(i), []).append(sid)
    out = sorted(sorted(g) for g in groups.values() if len(g) >= 2)
    return (out, pairs) if return_pairs else out
