"""Initial states with certified difficulty for InteriorGS v10 episodes.

Pipeline per task item (one target pair / relation):

1. Endpoints (canonical-success poses from the endpoint workstream) are
   re-checked: generator-legal and successful.
2. Reverse exploration from ALL endpoints at once (level-synchronous, beam
   limited per layer with a seeded random subsample): a predecessor
   ``q = apply(p, INVERSE_ACTION[a])`` is kept only if ``q`` is
   generator-legal, the forward transition ``q --a--> p`` is runtime-legal and
   re-applying ``a`` to ``q`` lands on ``p``'s exact key.  A state found in
   layer r therefore has an explicit, generator-safe r-step path to an
   endpoint, so d* <= r.
3. Candidates are drawn per difficulty band from layers r in [lo, hi] first,
   then from deeper layers (never shallower: d* <= r).  Cheap filters run
   first (already successful, diversity, exact generator legality, at least
   one target identifiable), then ``difficulty.certify_d_star`` with the
   reverse path as upper bound / hint, then an exact scalar replay.
4. The certified d* decides the band.  Every examined candidate is counted
   in the Funnel exactly once with a stable reason.

Turn chunking policy (``chunk_turns``): the certificate is split into the
minimum number of assistant turns with <= 3 actions each; among those splits
we avoid (a) a turn containing an action and its inverse (the env counts it
as ``contradictory_action``), (b) boundaries inside a run of identical
actions, then prefer balanced turns.  Why 3 and not the parser maximum of 5:
the runtime only evaluates success at the END of a turn, so every chunking
ends exactly on the certificate's last action (auto-termination fires on the
right step); shorter turns give the SFT trajectory an observation after each
maneuver (closer to what an RL policy should do: act, look, re-plan) and keep
headroom under ``max_actions_per_step = 5``.  With d* <= 8 this is <= 3 turns,
far below ``turn_budget = 12``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .contract import ACTIONS, INVERSE_ACTION, MAX_PRIMITIVE_STEPS, Funnel, Obj, yaw_of
from .difficulty import (
    ACTION_INDEX,
    DIFFICULTY_VERSION,
    IS_TRANSLATION,
    FastGeometry,
    SuccessOracle,
    apply_action_batch,
    as_oracle,
    certify_d_star,
    make_keys,
    make_success_fn,
    replay_certificate,
    to_c2w,
    _split,
)

Band = Tuple[int, int]
DEFAULT_BANDS: Tuple[Band, ...] = ((1, 1), (2, 3), (4, 6), (7, 8))
DEFAULT_BAND_WEIGHTS = (0.10, 0.20, 0.35, 0.35)  # scene allocation prefers medium / hard
MAX_ACTIONS_PER_ASSISTANT_TURN = 3
INITIAL_STATES_VERSION = "interiorgs_v10_initial_states_v1"

STAGE_ENDPOINTS = "init_endpoints"
STAGE_REVERSE = "init_reverse_edges"
STAGE_CANDIDATES = "init_candidates"
STAGE_SCENE = "init_scene_allocation"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def band_of(d: Optional[int], bands: Sequence[Band] = DEFAULT_BANDS) -> Optional[Band]:
    if d is None:
        return None
    for lo, hi in bands:
        if lo <= d <= hi:
            return (int(lo), int(hi))
    return None


def band_name(band: Band) -> str:
    return f"{band[0]}-{band[1]}"


_CONTRADICTORY = (("turn_left", "turn_right"), ("move_forward", "move_backward"), ("move_left", "move_right"))


def _contradictory(chunk: Sequence[str]) -> bool:
    s = set(chunk)
    return any(a in s and b in s for a, b in _CONTRADICTORY)


def chunk_turns(actions: Sequence[str], max_per_turn: int = MAX_ACTIONS_PER_ASSISTANT_TURN) -> List[List[str]]:
    """Split a certificate into assistant turns (see module docstring)."""
    acts = list(actions)
    n = len(acts)
    if n == 0:
        return []
    k = -(-n // max_per_turn)

    def cost(i: int, j: int) -> int:
        c = (j - i) ** 2
        if _contradictory(acts[i:j]):
            c += 1000
        if j < n and acts[j - 1] == acts[j]:
            c += 10
        return c

    best: Dict[Tuple[int, int], Tuple[int, int]] = {(0, 0): (0, -1)}
    for t in range(1, k + 1):
        for j in range(t, n + 1):
            cand = None
            for size in range(max_per_turn, 0, -1):
                i = j - size
                if i < 0 or (t - 1, i) not in best:
                    continue
                c = best[(t - 1, i)][0] + cost(i, j)
                if cand is None or c < cand[0]:
                    cand = (c, i)
            if cand is not None:
                best[(t, j)] = cand
    out: List[List[str]] = []
    t, j = k, n
    while t > 0:
        i = best[(t, j)][1]
        out.append(acts[i:j])
        t, j = t - 1, i
    out.reverse()
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
def projection_identifiable(scene: Any, c2w: Any, obj: Any, *, min_inside: float = 0.5,
                            min_area_ratio: float = 0.002, min_visible: float = 0.5) -> Dict[str, Any]:
    """Fallback (no occlusion model): canonical H1 projection inside-frame
    fraction and area only.  ``min_visible`` is accepted for signature
    compatibility with occlusion.identifiable and ignored."""
    from vagen.envs.active_spatial.canonical_camera import CANONICAL_CAMERA_H1_RESIZE_V1, build_canonical_camera
    from vagen.envs.active_spatial.visual_bbox_metrics import _project_object

    from .contract import NATIVE_K, NATIVE_SIZE, RENDER_SIZE

    camera = build_canonical_camera(K_native=NATIVE_K, native_size=NATIVE_SIZE, render_size=RENDER_SIZE,
                                    transform="resize", c2w=np.asarray(c2w, float),
                                    camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1)
    entry = obj.target_object_entry() if hasattr(obj, "target_object_entry") else dict(obj)
    proj = _project_object(entry, np.asarray(camera.c2w), np.asarray(camera.K_effective), *camera.render_resolution)

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) if b else 0.0
    raw = area(proj.get("bbox_raw"))
    inside = area(proj.get("bbox")) / raw if raw > 1e-8 else 0.0
    ratio = float(proj.get("area_ratio", 0.0) or 0.0)
    ok = bool(proj.get("center_in_front") and proj.get("visible") and inside >= min_inside
              and ratio >= min_area_ratio)
    return {"identifiable": ok, "inside_frame_fraction": float(inside), "area_ratio": ratio,
            "source": "fallback_projection"}


def resolve_observability_fn(observability_fn: Optional[Callable] = None) -> Tuple[Callable, str]:
    if observability_fn is not None:
        return observability_fn, getattr(observability_fn, "__name__", "custom")
    try:
        from .occlusion import identifiable  # workstream E
        return identifiable, "occlusion.identifiable"
    except Exception:
        return projection_identifiable, "fallback_projection"


def item_targets(scene: Any, item: Dict[str, Any]) -> List[Obj]:
    by_id = {str(o.id): o for o in getattr(scene, "objects", [])}
    out: List[Obj] = []
    for e in (item.get("target_object") or {}).get("objects") or []:
        o = by_id.get(str(e.get("id")))
        if o is None:
            o = Obj(id=str(e.get("id")), label=str(e.get("label", "")),
                    aabb_min=np.asarray(e["bbox_min"], float), aabb_max=np.asarray(e["bbox_max"], float),
                    ref=e.get("ref"))
        out.append(o)
    return out


# ---------------------------------------------------------------------------
# Per-item search
# ---------------------------------------------------------------------------
class _Layer:
    __slots__ = ("P", "R", "keys", "parent", "action", "success")

    def __init__(self, P, R, keys, parent, action, success):
        self.P, self.R, self.keys, self.parent, self.action, self.success = P, R, keys, parent, action, success

    def __len__(self) -> int:
        return len(self.keys)


class ItemSearch:
    """Resumable initial-state search for one item (used by both public helpers)."""

    def __init__(self, scene: Any, item: Dict[str, Any], success_fn: Any, endpoints: Sequence[Any], *,
                 bands: Sequence[Band] = DEFAULT_BANDS, seed: int = 0,
                 observability_fn: Optional[Callable] = None, beam: int = 1500, reverse_slack: int = 1,
                 max_states: int = 200000, max_certify: int = 48, diversity_m: float = 0.6,
                 diversity_deg: float = 30.0, require_terminal_identifiable: bool = True,
                 require_generator_safe_certificate: bool = False, funnel: Optional[Funnel] = None,
                 item_key: Any = None, max_examined: int = 1500, max_terminal_checks: int = 24,
                 terminal_roles: Optional[Sequence[int]] = None) -> None:
        self.scene, self.item = scene, item
        self.bands = [tuple(int(v) for v in b) for b in bands]
        self.d_max = max(hi for _, hi in self.bands)
        if self.d_max > MAX_PRIMITIVE_STEPS:
            raise ValueError("bands exceed the runtime primitive step budget")
        self.oracle: SuccessOracle = as_oracle(success_fn if success_fn is not None else make_success_fn(item))
        self.endpoints = list(endpoints)
        self.rng = np.random.default_rng(seed)
        self.obs_fn, self.obs_source = resolve_observability_fn(observability_fn)
        self.beam, self.reverse_depth = int(beam), self.d_max + int(reverse_slack)
        self.max_states, self.max_certify = int(max_states), int(max_certify)
        self.max_examined = int(max_examined)
        self.max_terminal_checks = int(max_terminal_checks)
        self.div_m, self.div_deg = float(diversity_m), float(diversity_deg)
        self.require_terminal = bool(require_terminal_identifiable)
        self.require_safe_cert = bool(require_generator_safe_certificate)
        self.funnel = funnel if funnel is not None else Funnel()
        self.item_key = item_key
        self.targets = item_targets(scene, item)
        # targets that must be identifiable in the terminal frame (default: all of them)
        self.terminal_targets = (self.targets if terminal_roles is None
                                 else [self.targets[i] for i in terminal_roles if i < len(self.targets)])
        self.reasons: Dict[str, int] = {}  # per-item STAGE_CANDIDATES counts (diagnostics only)
        self.geom = FastGeometry.for_scene(scene)
        self.layers: List[_Layer] = []
        self.accepted: List[Dict[str, Any]] = []
        self.surplus: List[Dict[str, Any]] = []
        self.certified = 0
        self.examined: set = set()
        self.band_stop: Dict[Band, str] = {}
        self._pools: Dict[Band, List[np.ndarray]] = {}
        self._pool_pos: Dict[Band, int] = {}
        self.timing = {"reverse_s": 0.0, "certify_s": 0.0, "filters_s": 0.0, "replay_s": 0.0}
        self.certify_stats: List[Dict[str, Any]] = []
        self._prepared = False

    # ------------------------------------------------------------ preparation
    def prepare(self) -> None:
        if self._prepared:
            return
        self._prepared = True
        t0 = time.perf_counter()
        f = self.funnel
        if not self.endpoints:
            f.add(STAGE_ENDPOINTS, "no_endpoints")
        else:
            P, R = _split(np.stack([np.asarray(e, float) for e in self.endpoints]))
            keys = make_keys(P, R)
            legal = self.geom.poses_legal(P, R)
            succ = self.oracle.batch(to_c2w(P, R), keys)
            _, first = np.unique(keys, return_index=True)
            uniq = np.zeros(len(keys), dtype=bool)
            uniq[first] = True
            keep = legal & succ & uniq
            f.add(STAGE_ENDPOINTS, "not_generator_legal", int((~legal).sum()))
            f.add(STAGE_ENDPOINTS, "not_success", int((legal & ~succ).sum()))
            f.add(STAGE_ENDPOINTS, "duplicate", int((legal & succ & ~uniq).sum()))
            f.add(STAGE_ENDPOINTS, "ok", int(keep.sum()))
            idx = np.flatnonzero(keep)
            if len(idx):
                self.layers.append(_Layer(P[idx], R[idx], keys[idx], idx, np.full(len(idx), -1),
                                          np.ones(len(idx), dtype=bool)))
        if self.layers:
            self._reverse_explore()
        self.timing["reverse_s"] = time.perf_counter() - t0
        self._build_pools()

    def _reverse_explore(self) -> None:
        f = self.funnel
        seen = set(self.layers[0].keys.tolist())
        for r in range(1, self.reverse_depth + 1):
            cur = self.layers[-1]
            n = len(cur)
            Ps, Rs = [], []
            for a in ACTIONS:
                p, q = apply_action_batch(cur.P, cur.R, INVERSE_ACTION[a])
                Ps.append(p)
                Rs.append(np.broadcast_to(q, (n, 3, 3)))
            P = np.stack(Ps, axis=1).reshape(-1, 3)
            R = np.stack(Rs, axis=1).reshape(-1, 3, 3)
            child = np.repeat(np.arange(n), len(ACTIONS))
            fwd = np.tile(np.arange(len(ACTIONS)), n)
            keys = make_keys(P, R)
            fresh = np.fromiter((k not in seen for k in keys.tolist()), dtype=bool, count=len(keys))
            f.add(STAGE_REVERSE, "pred_already_seen", int((~fresh).sum()))
            idx = np.flatnonzero(fresh)
            # forward re-application must land on the child's exact key
            Pf = P[idx].copy()
            Rf = R[idx].copy()
            for ai, a in enumerate(ACTIONS):
                m = fwd[idx] == ai
                if m.any():
                    Pf[m], Rf[m] = apply_action_batch(P[idx][m], R[idx][m], a)
            same = make_keys(Pf, Rf) == cur.keys[child[idx]]
            f.add(STAGE_REVERSE, "quantization_mismatch", int((~same).sum()))
            idx = idx[same]
            legal = self.geom.poses_legal(P[idx], R[idx])
            f.add(STAGE_REVERSE, "pred_not_generator_legal", int((~legal).sum()))
            idx = idx[legal]
            tr = IS_TRANSLATION[fwd[idx]]
            ok = np.ones(len(idx), dtype=bool)
            if tr.any():
                ok[tr] = self.geom.transitions_legal(P[idx[tr]], cur.P[child[idx[tr]]], R[idx[tr]])
            f.add(STAGE_REVERSE, "forward_transition_collision", int((~ok).sum()))
            idx = idx[ok]
            _, first = np.unique(keys[idx], return_index=True)
            f.add(STAGE_REVERSE, "pred_duplicate_in_layer", int(len(idx) - len(first)))
            idx = idx[np.sort(first)]
            if len(idx) > self.beam:
                f.add(STAGE_REVERSE, "beam_dropped", int(len(idx) - self.beam))
                idx = np.sort(self.rng.choice(idx, size=self.beam, replace=False))
            f.add(STAGE_REVERSE, "kept", int(len(idx)))
            if len(idx) == 0:
                break
            seen.update(keys[idx].tolist())
            succ = self.oracle.batch(to_c2w(P[idx], R[idx]), keys[idx])
            self.layers.append(_Layer(P[idx], R[idx], keys[idx], child[idx], fwd[idx], succ))

    def _build_pools(self) -> None:
        depth = len(self.layers) - 1
        for b in self.bands:
            lo, hi = b
            pools = []
            for rng_layers in (range(lo, min(hi, depth) + 1), range(hi + 1, depth + 1)):
                pairs = [np.stack([np.full(len(self.layers[r]), r), np.arange(len(self.layers[r]))], axis=1)
                         for r in rng_layers if len(self.layers[r])]
                arr = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
                pools.append(arr[self.rng.permutation(len(arr))])
            self._pools[b] = [np.concatenate(pools)] if pools else []
            self._pool_pos[b] = 0
            if not len(self._pools[b][0]):
                self.band_stop[b] = ("no_valid_endpoints" if not self.layers
                                     else "no_reverse_states_deep_enough")

    # ------------------------------------------------------------- search
    def exhausted(self, band: Band) -> bool:
        return band in self.band_stop

    def fill(self, needs: Dict[Band, int], max_new: Optional[int] = None) -> List[Dict[str, Any]]:
        """Certify candidates until ``max_new`` episodes landed in bands with
        ``needs[band] > 0`` (``needs`` is decremented in place) or no needed
        band can be served any more."""
        self.prepare()
        new: List[Dict[str, Any]] = []
        while max_new is None or len(new) < max_new:
            open_bands = [b for b in self.bands if needs.get(b, 0) > 0 and not self.exhausted(b)]
            if not open_bands:
                break
            if self.certified >= self.max_certify or len(self.examined) >= self.max_examined:
                why = ("certify_budget_exhausted" if self.certified >= self.max_certify
                       else "examine_budget_exhausted")
                for b in open_bands:
                    self.band_stop[b] = why
                break
            band = max(open_bands, key=lambda b: (needs[b], self.bands.index(b)))
            cand = self._next(band)
            if cand is None:
                self.band_stop[band] = "candidates_exhausted"
                continue
            ep = self._examine(int(cand[0]), int(cand[1]), needs)
            if ep is not None:
                new.append(ep)
        return new

    def _next(self, band: Band) -> Optional[np.ndarray]:
        pool = self._pools[band][0]
        pos = self._pool_pos[band]
        while pos < len(pool):
            c = pool[pos]
            pos += 1
            if (int(c[0]), int(c[1])) not in self.examined:
                self._pool_pos[band] = pos
                return c
        self._pool_pos[band] = pos
        return None

    def _too_close(self, c2w: np.ndarray) -> bool:
        x, y = float(c2w[0, 3]), float(c2w[1, 3])
        yd = math.degrees(yaw_of(c2w))
        for ep in self.accepted + self.surplus:
            q = np.asarray(ep["init_c2w"])
            if math.hypot(q[0, 3] - x, q[1, 3] - y) < self.div_m:
                dy = abs((math.degrees(yaw_of(q)) - yd + 180.0) % 360.0 - 180.0)
                if dy < self.div_deg:
                    return True
        return False

    def _observe(self, c2w: np.ndarray) -> Dict[str, Any]:
        per: Dict[str, Any] = {}
        n_ok = 0
        for obj in self.targets:
            try:
                res = dict(self.obs_fn(self.scene, c2w, obj))
            except Exception as exc:  # never let an observability bug pass a candidate
                res = {"identifiable": False, "error": repr(exc)}
            res = _jsonable(res)
            per[str(obj.id)] = res
            n_ok += bool(res.get("identifiable"))
        return {"source": self.obs_source, "targets": per, "n_identifiable": n_ok, "n_targets": len(self.targets)}

    def _cand(self, reason: str) -> None:
        self.funnel.add(STAGE_CANDIDATES, reason)
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def _examine(self, r: int, i: int, needs: Dict[Band, int]) -> Optional[Dict[str, Any]]:
        self.examined.add((r, i))
        layer = self.layers[r]
        c2w = to_c2w(layer.P[i:i + 1], layer.R[i:i + 1])[0]
        t0 = time.perf_counter()
        if bool(layer.success[i]):
            self._cand("already_success")
            return None
        if self._too_close(c2w):
            self._cand("diversity_too_close")
            return None
        if not bool(self.scene.is_pose_legal(c2w)[0]):
            self._cand("not_generator_legal")
            return None
        obs = self._observe(c2w)
        self.timing["filters_s"] += time.perf_counter() - t0
        if obs["n_identifiable"] < 1:
            self._cand("no_target_identifiable")
            return None
        # explicit reverse path -> forward hint
        hint: List[str] = []
        node = i
        for lv in range(r, 0, -1):
            hint.append(ACTIONS[int(self.layers[lv].action[node])])
            node = int(self.layers[lv].parent[node])
        endpoint_index = int(self.layers[0].parent[node])

        def terminal_ok(pose: np.ndarray) -> bool:
            try:
                return all(bool(self.obs_fn(self.scene, pose, o).get("identifiable")) for o in self.terminal_targets)
            except Exception:
                return False

        t1 = time.perf_counter()
        cert = certify_d_star(self.scene, self.oracle, c2w, d_max=self.d_max, max_states=self.max_states,
                              hint_actions=hint, terminal_ok=terminal_ok, geometry=self.geom,
                              max_terminal_checks=self.max_terminal_checks)
        dt = time.perf_counter() - t1
        self.timing["certify_s"] += dt
        self.certified += 1
        self.certify_stats.append({"d_star": cert["d_star"], "states": cert["states"], "seconds": dt,
                                   "reverse_depth": r, "source": cert["certificate_source"]})
        d = cert["d_star"]
        if cert["status"] == "state_cap":
            self._cand("certify_state_cap")
            return None
        if d is None:
            self._cand("d_star_above_max_band")
            return None
        if d == 0:
            self._cand("already_success")
            return None
        band = band_of(d, self.bands)
        if band is None:
            self._cand("d_star_outside_bands")
            return None
        if self.require_terminal and cert.get("terminal_ok") is False:
            self._cand("terminal_not_identifiable")
            return None
        if self.require_safe_cert and not cert.get("certificate_generator_safe"):
            self._cand("certificate_not_generator_safe")
            return None
        t2 = time.perf_counter()
        rep = replay_certificate(self.scene, self.oracle, c2w, cert["certificate"], check_generator_safe=True)
        self.timing["replay_s"] += time.perf_counter() - t2
        if not rep["ok"]:
            self._cand("certificate_replay_failed")
            return None
        terminal = np.asarray(rep["poses"][-1])
        ep = {
            "version": INITIAL_STATES_VERSION,
            "item_key": self.item_key,
            "init_c2w": c2w.tolist(),
            "d_star": int(d),
            "lower_bound_complete": bool(cert["lower_bound_complete"]),
            "certificate": list(cert["certificate"]),
            "turns": chunk_turns(cert["certificate"]),
            "band": [band[0], band[1]],
            "band_name": band_name(band),
            "difficulty": {
                "d_star": int(d), "lower_bound_complete": bool(cert["lower_bound_complete"]),
                "certificate": list(cert["certificate"]), "version": DIFFICULTY_VERSION,
                "key_mode": cert["key_mode"], "certificate_source": cert["certificate_source"],
                "reverse_depth_upper_bound": int(r), "bfs_states": int(cert["states"]),
                "bfs_expansions": int(cert["expansions"]),
                "certificate_generator_safe": bool(all(rep["generator_safe"])),
                "certificate_switches": cert.get("certificate_switches"),
            },
            "initial_observability": obs,
            "terminal_observability": self._observe(terminal),
            "reverse_depth": int(r),
            "endpoint_index": endpoint_index if cert["certificate_source"] == "hint" else None,
            "certify_seconds": float(dt),
        }
        if needs.get(band, 0) > 0:
            needs[band] -= 1
            self.accepted.append(ep)
            self._cand("accepted")
            return ep
        self.surplus.append(ep)
        self._cand("band_quota_full")
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "item_key": self.item_key,
            "task_type": self.item.get("task_type"),
            "endpoints": len(self.endpoints),
            "reverse_layer_sizes": [len(l) for l in self.layers],
            "certified": self.certified,
            "accepted": len(self.accepted),
            "surplus": len(self.surplus),
            "band_stop": {band_name(b): v for b, v in self.band_stop.items()},
            "candidate_reasons": dict(sorted(self.reasons.items())),
            "timing": dict(self.timing),
            "observability_source": self.obs_source,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _quota(quota_per_band: Union[int, Dict[Any, int]], band: Band) -> int:
    if isinstance(quota_per_band, dict):
        for k in (band, list(band), band_name(band)):
            try:
                if k in quota_per_band:
                    return int(quota_per_band[k])
            except TypeError:
                continue
        return 0
    return int(quota_per_band)


def propose_initial_states(scene: Any, item: Dict[str, Any], success_fn: Any, endpoints: Sequence[Any], *,
                           bands: Sequence[Band] = DEFAULT_BANDS,
                           quota_per_band: Union[int, Dict[Any, int]] = 2, max_per_item: int = 8, seed: int = 0,
                           observability_fn: Optional[Callable] = None,
                           **search_kwargs: Any) -> Tuple[List[Dict[str, Any]], Funnel]:
    """Certified, diverse initial states for one item, bucketed by exact d*.

    ``success_fn``: callable(c2w) -> bool (or dict with 'success'); may expose
    ``batch``/``exact`` (``difficulty.make_success_fn(item)`` for canonical v1
    Projective/FOV; None builds it).  ``search_kwargs`` go to ``ItemSearch``
    (beam, reverse_slack, max_states, max_certify, diversity_m, diversity_deg,
    require_terminal_identifiable, require_generator_safe_certificate).
    """
    search = ItemSearch(scene, item, success_fn, endpoints, bands=bands, seed=seed,
                        observability_fn=observability_fn, **search_kwargs)
    needs = {tuple(b): _quota(quota_per_band, tuple(b)) for b in search.bands}
    search.fill(needs, max_new=max_per_item)
    for b, n in needs.items():
        if n > 0:
            search.funnel.add(STAGE_SCENE, f"item_band_shortfall:{band_name(b)}:"
                              f"{search.band_stop.get(b, 'max_per_item_reached')}", n)
    return list(search.accepted), search.funnel


def allocate_band_targets(total: int, bands: Sequence[Band] = DEFAULT_BANDS,
                          weights: Sequence[float] = DEFAULT_BAND_WEIGHTS) -> Dict[Band, int]:
    """Largest-remainder split of ``total`` over bands; ties go to harder bands."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    raw = w * total
    base = np.floor(raw).astype(int)
    rem = total - int(base.sum())
    order = sorted(range(len(bands)), key=lambda i: (-(raw[i] - base[i]), -i))
    for i in order[:rem]:
        base[i] += 1
    return {tuple(bands[i]): int(base[i]) for i in range(len(bands))}


def _unpack_item(entry: Any) -> Tuple[Dict[str, Any], Any, List[Any], Any]:
    if isinstance(entry, dict) and "item" in entry:
        return entry["item"], entry.get("success_fn"), list(entry.get("endpoints") or []), entry.get("key")
    if isinstance(entry, (tuple, list)):
        if len(entry) == 2:
            return entry[0], None, list(entry[1] or []), None
        if len(entry) == 3:
            return entry[0], entry[1], list(entry[2] or []), None
    raise ValueError("items_with_endpoints entries must be {'item','endpoints'[,'success_fn','key']} "
                     "or (item, endpoints) or (item, success_fn, endpoints)")


FamilyQuotas = Dict[str, Tuple[int, Optional[int]]]


def parse_family_quotas(spec: Any) -> Optional[FamilyQuotas]:
    """``"fam:min:max,fam:min"`` (max empty / ``*`` / absent = unlimited) or a mapping
    ``{fam: (min, max)}`` / ``{fam: {"min": .., "max": ..}}`` -> ``{fam: (min, max|None)}``."""
    if spec is None or spec == "" or spec == {}:
        return None
    out: FamilyQuotas = {}
    if isinstance(spec, str):
        for part in spec.split(","):
            bits = [b.strip() for b in part.split(":")]
            if not bits[0]:
                continue
            lo = int(bits[1]) if len(bits) > 1 and bits[1] else 0
            hi = int(bits[2]) if len(bits) > 2 and bits[2] not in ("", "*") else None
            out[bits[0]] = (lo, hi)
    else:
        for fam, q in dict(spec).items():
            if isinstance(q, dict):
                lo, hi = int(q.get("min", 0) or 0), q.get("max")
            else:
                lo, hi = (int(q[0] or 0), q[1] if len(q) > 1 else None)
            out[str(fam)] = (lo, None if hi is None else int(hi))
    for fam, (lo, hi) in out.items():
        if lo < 0 or (hi is not None and hi < lo):
            raise ValueError(f"bad family quota for {fam}: min={lo} max={hi}")
    return out


def _scaled_quotas(quotas: FamilyQuotas, k: int, target: int) -> FamilyQuotas:
    """Family quotas for a reserve pool of ``k`` next to a main set of ``target``."""
    out: FamilyQuotas = {}
    for fam, (lo, hi) in quotas.items():
        out[fam] = (int(math.floor(lo * k / max(target, 1))),
                    None if hi is None else int(math.ceil(hi * k / max(target, 1))))
    return out


def _quota_fill(searches: Sequence[ItemSearch], needs: Dict[Band, int], counts: List[int], cap: int,
                quotas: FamilyQuotas, fam_count: Dict[str, int], out: List[Dict[str, Any]],
                deadline: Optional[float]) -> bool:
    """Fill ``needs`` round-robin under per-family [min, max] counts; returns timed_out.

    Phase 1 serves the families still below their minimum (one item per family
    per round, families in first-appearance order); phase 2 is the default
    round-robin over all items, skipping items whose family reached its max.
    """
    fams = [str(s.item.get("task_type")) for s in searches]
    active = set(range(len(searches)))

    def at_max(f: str) -> bool:
        hi = quotas.get(f, (0, None))[1]
        return hi is not None and fam_count.get(f, 0) >= hi

    def step(j: int) -> None:
        if counts[j] >= cap or at_max(fams[j]):
            active.discard(j)
            return
        got = searches[j].fill(needs, max_new=1)
        if not got:
            active.discard(j)
            return
        counts[j] += len(got)
        fam_count[fams[j]] = fam_count.get(fams[j], 0) + len(got)
        out.extend(got)

    def late() -> bool:
        return deadline is not None and time.perf_counter() > deadline

    fam_order = list(dict.fromkeys(fams))
    turn = {f: 0 for f in fam_order}
    while sum(needs.values()) > 0:
        wanting = [f for f in fam_order if fam_count.get(f, 0) < quotas.get(f, (0, None))[0]
                   and any(fams[j] == f for j in active)]
        if not wanting:
            break
        for f in wanting:
            if sum(needs.values()) <= 0:
                break
            if late():
                return True
            members = [j for j in range(len(searches)) if fams[j] == f and j in active]
            if members:
                step(members[turn[f] % len(members)])
                turn[f] += 1
    order = [j for j in range(len(searches)) if j in active]
    while order and sum(needs.values()) > 0:
        for j in list(order):
            if sum(needs.values()) <= 0:
                break
            if late():
                return True
            step(j)
            if j not in active:
                order.remove(j)
    return False


def _backfill(searches: Sequence[ItemSearch], out: List[Dict[str, Any]], counts: List[int], cap: int, total: int,
              skip, on_take) -> None:
    """Hardest-band-first backfill from surplus episodes not used yet."""
    used = {id(e) for e in out}
    pool = []
    for j, s in enumerate(searches):
        for ep in s.surplus:
            pool.append((-ep["band"][0], -ep["d_star"], j, ep))
    pool.sort(key=lambda t: t[:3])
    for _, _, j, ep in pool:
        if len(out) >= total:
            break
        if id(ep) in used or counts[j] >= cap or skip(j, ep):
            continue
        out.append(ep)
        counts[j] += 1
        on_take(j, ep)


def episodes_for_scene(scene: Any, items_with_endpoints: Sequence[Any], target_per_scene: int = 12, *,
                       bands: Sequence[Band] = DEFAULT_BANDS,
                       band_weights: Sequence[float] = DEFAULT_BAND_WEIGHTS,
                       max_per_item: Optional[int] = None, seed: int = 0,
                       observability_fn: Optional[Callable] = None, time_budget_s: Optional[float] = None,
                       family_quotas: Any = None, reserve_k: int = 0,
                       terminal_identifiable: str = "all",
                       **search_kwargs: Any) -> Tuple[List[Dict[str, Any]], Funnel, Dict[str, Any]]:
    """Allocate ``target_per_scene`` episodes across items and bands.

    Band targets follow ``band_weights`` (medium/hard preferred).  Items are
    served round-robin, one accepted episode at a time, each item capped at
    ``max_per_item`` (default ceil(target / n_items) + 2) so that several
    target pairs contribute.  Episodes certified into an already-full band are
    kept as surplus and used to backfill (hardest first) when a band falls
    short.  Returns (episodes, funnel, report) where report['shortfall'] lists
    missing episodes per band with the reasons each item stopped.

    Optional (all off by default; off reproduces the allocation above exactly):
      family_quotas  {task_type: (min, max)} (see ``parse_family_quotas``): families
                     below their min are served first, families at max stop.
      reserve_k      after the main set, certify up to ``reserve_k`` more episodes
                     (``report["reserve"]``) with the same band weights and, if
                     quotas are set, the quotas scaled to ``reserve_k``; unused
                     surplus is used first.  For backfilling after review rejections.
      terminal_identifiable  "all" (every target identifiable at the terminal pose)
                     or "family" (only the family's TERMINAL_IDENTIFIABLE_ROLES, e.g.
                     the occluder of occlusion_alignment, whose target is hidden by definition).
    """
    t0 = time.perf_counter()
    funnel = Funnel()
    bands = [tuple(int(v) for v in b) for b in bands]
    targets = allocate_band_targets(int(target_per_scene), bands, band_weights)
    quotas = parse_family_quotas(family_quotas)
    if quotas and sum(lo for lo, _ in quotas.values()) > int(target_per_scene):
        raise ValueError("family minimums exceed target_per_scene")
    if terminal_identifiable not in ("all", "family"):
        raise ValueError(f"terminal_identifiable must be 'all' or 'family', not {terminal_identifiable!r}")
    searches: List[ItemSearch] = []
    for k, entry in enumerate(items_with_endpoints):
        item, sfn, eps, key = _unpack_item(entry)
        if sfn is None:
            try:
                sfn = make_success_fn(item)
            except Exception:
                funnel.add(STAGE_SCENE, "item_without_success_fn")
                continue
        extra = {}
        if terminal_identifiable == "family":
            extra["terminal_roles"] = _family_terminal_roles(item)
        searches.append(ItemSearch(scene, item, sfn, eps, bands=bands, seed=seed + 1009 * k,
                                   observability_fn=observability_fn, funnel=funnel,
                                   item_key=key if key is not None else k, **extra, **search_kwargs))
    n = max(1, len(searches))
    cap = int(max_per_item) if max_per_item is not None else -(-int(target_per_scene) // n) + 2
    needs = dict(targets)
    counts = [0] * len(searches)
    out: List[Dict[str, Any]] = []
    timed_out = False
    fam_count: Dict[str, int] = {}
    if quotas is None:
        active = list(range(len(searches)))
        while active and sum(needs.values()) > 0:
            for j in list(active):
                if sum(needs.values()) <= 0:
                    break
                if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
                    timed_out = True
                    break
                if counts[j] >= cap:
                    active.remove(j)
                    continue
                got = searches[j].fill(needs, max_new=1)
                if not got:
                    active.remove(j)
                    continue
                counts[j] += len(got)
                out.extend(got)
            if timed_out:
                break
        # backfill from surplus (hardest band first, then larger d*)
        if len(out) < target_per_scene:
            pool = []
            for j, s in enumerate(searches):
                for ep in s.surplus:
                    pool.append((-ep["band"][0], -ep["d_star"], j, ep))
            pool.sort(key=lambda t: t[:3])
            for _, _, j, ep in pool:
                if len(out) >= target_per_scene:
                    break
                if counts[j] >= cap:
                    continue
                out.append(ep)
                counts[j] += 1
                ep["backfilled"] = True
                funnel.add(STAGE_SCENE, "backfilled_from_surplus")
    else:
        deadline = None if time_budget_s is None else t0 + time_budget_s
        timed_out = _quota_fill(searches, needs, counts, cap, quotas, fam_count, out, deadline)
        if len(out) < target_per_scene:
            fams = [str(s.item.get("task_type")) for s in searches]

            def _skip(j: int, ep: Dict[str, Any]) -> bool:
                hi = quotas.get(fams[j], (0, None))[1]
                return hi is not None and fam_count.get(fams[j], 0) >= hi

            def _take(j: int, ep: Dict[str, Any]) -> None:
                fam_count[fams[j]] = fam_count.get(fams[j], 0) + 1
                ep["backfilled"] = True
                funnel.add(STAGE_SCENE, "backfilled_from_surplus")
            _backfill(searches, out, counts, cap, int(target_per_scene), _skip, _take)
    band_counts = {band_name(b): 0 for b in bands}
    for ep in out:
        band_counts[ep["band_name"]] += 1
    shortfall: Dict[str, Any] = {}
    for b in bands:
        missing = targets[b] - band_counts[band_name(b)]
        if missing <= 0:
            continue
        reasons: Dict[str, int] = {}
        for j, s in enumerate(searches):
            why = s.band_stop.get(b) or ("max_per_item_reached" if counts[j] >= cap else
                                         ("scene_time_budget" if timed_out else "not_needed_after_backfill"))
            reasons[why] = reasons.get(why, 0) + 1
        if not searches:
            reasons["no_items"] = 1
        shortfall[band_name(b)] = {"missing": int(missing), "item_stop_reasons": reasons}
    reserve: List[Dict[str, Any]] = []
    reserve_report: Dict[str, Any] = {}
    if reserve_k > 0:
        reserve, reserve_report = _fill_reserve(searches, out, int(reserve_k), int(target_per_scene), bands,
                                                band_weights, quotas, funnel, t0, time_budget_s, timed_out)
    used = {id(e) for e in out} | {id(e) for e in reserve}
    n_unused = sum(1 for s in searches for e in s.surplus if id(e) not in used)
    if n_unused:
        funnel.add(STAGE_SCENE, "surplus_unused", n_unused)
    total_missing = max(0, int(target_per_scene) - len(out))
    if total_missing:
        funnel.add(STAGE_SCENE, "scene_shortfall", total_missing)
    report = {
        "version": INITIAL_STATES_VERSION,
        "target_per_scene": int(target_per_scene),
        "band_targets": {band_name(b): v for b, v in targets.items()},
        "band_counts": band_counts,
        "episodes": len(out),
        "missing_total": total_missing,
        "shortfall": shortfall,
        "max_per_item": cap,
        "per_item": [dict(s.summary(), accepted_in_scene=counts[j]) for j, s in enumerate(searches)],
        "timed_out": timed_out,
        "seconds": time.perf_counter() - t0,
    }
    if quotas is not None:
        report["family_quotas"] = {f: list(q) for f, q in quotas.items()}
        report["family_counts"] = dict(sorted(fam_count.items()))
    if reserve_k > 0:
        report["reserve"] = reserve
        report.update(reserve_report)
    return out, funnel, report


def _family_terminal_roles(item: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """TERMINAL_IDENTIFIABLE_ROLES of a v2 family (None = all targets, e.g. Projective / FOV)."""
    try:
        from .families.canonical_v2 import FAMILIES
    except Exception:
        return None
    fam = FAMILIES.get(str(item.get("task_type")))
    roles = getattr(fam, "TERMINAL_IDENTIFIABLE_ROLES", None) if fam is not None else None
    return tuple(int(r) for r in roles) if roles is not None else None


def _fill_reserve(searches: Sequence[ItemSearch], main: Sequence[Dict[str, Any]], k: int, target: int,
                  bands: Sequence[Band], band_weights: Sequence[float], quotas: Optional[FamilyQuotas],
                  funnel: Funnel, t0: float, time_budget_s: Optional[float],
                  timed_out: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Up to ``k`` reserve episodes with the main set's band weights (and scaled family quotas)."""
    r_targets = allocate_band_targets(k, bands, band_weights)
    r_needs = dict(r_targets)
    n = max(1, len(searches))
    r_cap = -(-k // n) + 2
    r_counts = [0] * len(searches)
    r_quotas = _scaled_quotas(quotas, k, target) if quotas else {}
    r_fam: Dict[str, int] = {}
    fams = [str(s.item.get("task_type")) for s in searches]
    reserve: List[Dict[str, Any]] = []
    main_ids = {id(e) for e in main}

    def at_max(j: int) -> bool:
        hi = r_quotas.get(fams[j], (0, None))[1]
        return hi is not None and r_fam.get(fams[j], 0) >= hi

    # 1) surplus that is already certified (cheap), hardest band first, only into open bands; inside a
    #    band the d* values take turns, so the reserve covers the band instead of only its largest d*
    pool = sorted(((j, ep) for j, s in enumerate(searches) for ep in s.surplus if id(ep) not in main_ids),
                  key=lambda t: (-t[1]["band"][0], -t[1]["d_star"], t[0]))
    turn: Dict[int, int] = {}
    keyed = []
    for j, ep in pool:
        nth = turn.get(ep["d_star"], 0)
        turn[ep["d_star"]] = nth + 1
        keyed.append(((-ep["band"][0], nth, -ep["d_star"], j), j, ep))
    keyed.sort(key=lambda t: t[0])
    for _, j, ep in keyed:
        if len(reserve) >= k:
            break
        band = tuple(ep["band"])
        if r_needs.get(band, 0) <= 0 or r_counts[j] >= r_cap or at_max(j):
            continue
        r_needs[band] -= 1
        r_counts[j] += 1
        r_fam[fams[j]] = r_fam.get(fams[j], 0) + 1
        reserve.append(ep)
        funnel.add(STAGE_SCENE, "reserve_from_surplus")
    # 2) certify more candidates for the bands still open
    late = timed_out
    if not late and sum(r_needs.values()) > 0:
        deadline = None if time_budget_s is None else t0 + time_budget_s
        before = len(reserve)
        late = _quota_fill(searches, r_needs, r_counts, r_cap, r_quotas, r_fam, reserve, deadline)
        if len(reserve) > before:
            funnel.add(STAGE_SCENE, "reserve_certified", len(reserve) - before)
    for ep in reserve:
        ep["reserve"] = True
    if len(reserve) < k:
        funnel.add(STAGE_SCENE, "reserve_shortfall", k - len(reserve))
    band_counts = {band_name(b): 0 for b in bands}
    for ep in reserve:
        band_counts[ep["band_name"]] += 1
    return reserve, {"reserve_target": k, "reserve_band_targets": {band_name(b): v for b, v in r_targets.items()},
                     "reserve_band_counts": band_counts, "reserve_family_counts": dict(sorted(r_fam.items())),
                     "reserve_family_quotas": {f: list(q) for f, q in r_quotas.items()},
                     "reserve_timed_out": bool(late)}
