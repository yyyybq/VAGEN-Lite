"""Auditable reward accounting primitives for Active Spatial.

This module deliberately knows nothing about task success or episode termination.
Those remain owned by :mod:`env`; the helpers here only compute reward terms and
record what was actually added.  Keeping this boundary explicit is what allows a
``sparse`` run to retain the same scorer and success gate while disabling shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict, Iterable, Optional, Tuple


REWARD_COMPONENTS: Tuple[str, ...] = (
    "potential",
    "near",
    "visibility",
    "success",
    "format",
    "invalid",
    "collision",
    "consecutive_collision",
    "premature_done",
    "step",
    "low_info",
    "legacy",
)


@dataclass
class RewardTerm:
    """One named term, with pre-scale, post-scale, and applied accounting."""

    raw: float = 0.0
    scale: Optional[float] = None
    scaled: float = 0.0
    applied: float = 0.0
    enabled: bool = True
    events: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        *,
        raw: float,
        scale: float,
        enabled: bool = True,
        scaled: Optional[float] = None,
        applied: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> float:
        raw_f = float(raw)
        scale_f = float(scale)
        scaled_f = raw_f * scale_f if scaled is None else float(scaled)
        applied_f = (scaled_f if enabled else 0.0) if applied is None else float(applied)
        self.raw += raw_f
        self.scaled += scaled_f
        self.applied += applied_f
        self.enabled = bool(self.enabled and enabled)
        self.events += 1
        if self.scale is None:
            self.scale = scale_f
        elif not math.isclose(self.scale, scale_f, rel_tol=0.0, abs_tol=0.0):
            self.scale = None
            self.details["mixed_scales"] = True
        if details:
            self.details.update(details)
        return applied_f

    def as_dict(self) -> Dict[str, Any]:
        return {
            "raw": float(self.raw),
            "scale": None if self.scale is None else float(self.scale),
            "scaled": float(self.scaled),
            "applied": float(self.applied),
            "enabled": bool(self.enabled),
            "events": int(self.events),
            "details": dict(self.details),
        }


@dataclass
class TurnRewardTrace:
    """Complete reward/scorer/outcome record for one LLM turn."""

    task_id: str
    episode_id: str
    turn_id: int
    task_family: str
    scene_id: str = "unknown"
    primitive_step_start: int = 0
    primitive_step_end: int = 0
    phi_prev: Optional[float] = None
    phi: Optional[float] = None
    position_score: Optional[float] = None
    orientation_score: Optional[float] = None
    dynamic_position_weight: Optional[float] = None
    dynamic_orientation_weight: Optional[float] = None
    score_backend: str = "unknown"
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = "continuing"
    actual_total_reward: Optional[float] = None
    components: Dict[str, RewardTerm] = field(default_factory=dict)
    overrides: list[Dict[str, Any]] = field(default_factory=list)
    extra_processing: list[Dict[str, Any]] = field(default_factory=list)

    SCHEMA_VERSION = "active_spatial_reward_trace_v1"

    def __post_init__(self) -> None:
        for name in REWARD_COMPONENTS:
            self.components.setdefault(name, RewardTerm())

    @property
    def primitive_action_count(self) -> int:
        return max(0, int(self.primitive_step_end) - int(self.primitive_step_start))

    def record(
        self,
        name: str,
        *,
        raw: float,
        scale: float,
        enabled: bool = True,
        scaled: Optional[float] = None,
        applied: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> float:
        term = self.components.setdefault(name, RewardTerm())
        return term.add(
            raw=raw,
            scale=scale,
            enabled=enabled,
            scaled=scaled,
            applied=applied,
            details=details,
        )

    def set_score(
        self,
        *,
        phi: float,
        position_score: float,
        orientation_score: float,
        dynamic_position_weight: Optional[float] = None,
        dynamic_orientation_weight: Optional[float] = None,
        backend: str = "unknown",
    ) -> None:
        self.phi = float(phi)
        self.position_score = float(position_score)
        self.orientation_score = float(orientation_score)
        self.dynamic_position_weight = (
            None if dynamic_position_weight is None else float(dynamic_position_weight)
        )
        self.dynamic_orientation_weight = (
            None if dynamic_orientation_weight is None else float(dynamic_orientation_weight)
        )
        self.score_backend = str(backend)

    def add_override(self, kind: str, **details: Any) -> None:
        self.overrides.append({"kind": str(kind), **details})

    def add_processing(self, kind: str, **details: Any) -> None:
        self.extra_processing.append({"kind": str(kind), **details})

    def reconstructed_total(self) -> float:
        return float(math.fsum(term.applied for term in self.components.values()))

    def finalize(
        self,
        *,
        actual_total_reward: float,
        primitive_step_end: int,
        success: bool,
        terminated: bool,
        truncated: bool,
        termination_reason: str,
    ) -> None:
        self.actual_total_reward = float(actual_total_reward)
        self.primitive_step_end = int(primitive_step_end)
        self.success = bool(success)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.termination_reason = str(termination_reason)

    @property
    def reconstruction_error(self) -> Optional[float]:
        if self.actual_total_reward is None:
            return None
        return float(self.actual_total_reward - self.reconstructed_total())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "identity": {
                "task_id": self.task_id,
                "episode_id": self.episode_id,
                "turn_id": int(self.turn_id),
                "task_family": self.task_family,
                "scene_id": self.scene_id,
            },
            "clock": {
                "turn_unit": "llm_response",
                "primitive_unit": "executed_atomic_action",
                "primitive_step_start": int(self.primitive_step_start),
                "primitive_step_end": int(self.primitive_step_end),
                "primitive_action_count": int(self.primitive_action_count),
            },
            "score": {
                "phi_prev": self.phi_prev,
                "phi": self.phi,
                "position": self.position_score,
                "orientation": self.orientation_score,
                "dynamic_position_weight": self.dynamic_position_weight,
                "dynamic_orientation_weight": self.dynamic_orientation_weight,
                "backend": self.score_backend,
            },
            "components": {name: term.as_dict() for name, term in self.components.items()},
            "processing": {
                "clipping": [],
                "overrides": list(self.overrides),
                "extra": list(self.extra_processing),
            },
            "outcome": {
                "success": bool(self.success),
                "terminated": bool(self.terminated),
                "truncated": bool(self.truncated),
                "termination_reason": self.termination_reason,
            },
            "accounting": {
                "reconstructed_total_reward": self.reconstructed_total(),
                "actual_total_reward": self.actual_total_reward,
                "reconstruction_error": self.reconstruction_error,
            },
        }

    def flat(self, prefix: str = "reward_trace") -> Dict[str, Any]:
        """Return scalar/string fields safe for the trainer's per-turn arrays."""
        out: Dict[str, Any] = {
            f"{prefix}/schema_version": self.SCHEMA_VERSION,
            f"{prefix}/task_id": self.task_id,
            f"{prefix}/episode_id": self.episode_id,
            f"{prefix}/turn_id": int(self.turn_id),
            f"{prefix}/task_family": self.task_family,
            f"{prefix}/scene_id": self.scene_id,
            f"{prefix}/phi_prev": self.phi_prev,
            f"{prefix}/phi": self.phi,
            f"{prefix}/position_score": self.position_score,
            f"{prefix}/orientation_score": self.orientation_score,
            f"{prefix}/dynamic_position_weight": self.dynamic_position_weight,
            f"{prefix}/dynamic_orientation_weight": self.dynamic_orientation_weight,
            f"{prefix}/score_backend": self.score_backend,
            f"{prefix}/primitive_action_count": int(self.primitive_action_count),
            f"{prefix}/success": float(self.success),
            f"{prefix}/terminated": float(self.terminated),
            f"{prefix}/truncated": float(self.truncated),
            f"{prefix}/termination_reason": self.termination_reason,
            f"{prefix}/total_reward": self.actual_total_reward,
            f"{prefix}/reconstructed_total_reward": self.reconstructed_total(),
            f"{prefix}/reconstruction_error": self.reconstruction_error,
            f"{prefix}/override_count": len(self.overrides),
            f"{prefix}/extra_processing_count": len(self.extra_processing),
            f"{prefix}/processing_json": json.dumps(
                {"clipping": [], "overrides": self.overrides, "extra": self.extra_processing},
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for name, term in self.components.items():
            base = f"{prefix}/{name}"
            out[f"{base}/raw"] = float(term.raw)
            out[f"{base}/scale"] = term.scale
            out[f"{base}/scaled"] = float(term.scaled)
            out[f"{base}/applied"] = float(term.applied)
            out[f"{base}/enabled"] = float(term.enabled)
            out[f"{base}/events"] = int(term.events)
        return out


def potential_reward_values(
    *,
    mode: str,
    phi_prev: float,
    phi: float,
    scale: float,
    gamma: float,
    position_prev: float = 0.0,
    position: float = 0.0,
    orientation_prev: float = 0.0,
    orientation: float = 0.0,
    position_scale: float = 0.0,
    orientation_scale: float = 0.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Return ``(raw, scaled, details)`` using the historical formulas."""
    mode = str(mode)
    details = {"delta_phi": float(phi - phi_prev), "gamma": float(gamma)}
    if mode == "delta":
        raw = phi - phi_prev
        scaled = scale * raw
    elif mode == "potential":
        raw = gamma * phi - phi_prev
        scaled = scale * raw
    elif mode == "dual":
        d_pos = position - position_prev
        d_ori = orientation - orientation_prev
        raw = position_scale * d_pos + orientation_scale * d_ori
        scaled = raw
        details.update(
            delta_position=float(d_pos),
            delta_orientation=float(d_ori),
            position_reward_scale=float(position_scale),
            orientation_reward_scale=float(orientation_scale),
        )
    else:  # historical "absolute" fallback
        raw = phi * 0.1
        scaled = scale * raw
    return float(raw), float(scaled), details


def near_reward_values(
    *,
    phi: float,
    threshold: float,
    bonus: float,
    mode: str,
    steepness: float,
) -> Tuple[float, float, bool]:
    """Return ``(raw_gate, scaled_bonus, threshold_hit)``."""
    hit = bool(threshold > 0.0 and phi >= threshold)
    if bonus <= 0.0:
        return 0.0, 0.0, hit
    if mode == "sigmoid":
        raw = 1.0 / (1.0 + math.exp(-steepness * (phi - threshold)))
    else:
        raw = 1.0 if hit else 0.0
    return float(raw), float(raw * bonus), hit


def visibility_reward_values(current: Any, previous: Any = None) -> Tuple[float, Dict[str, float]]:
    """Return the unscaled historical visibility reward and its subterms."""
    current_score = float(current.visibility_score)
    if previous is None:
        raw = 0.5 * current_score
        return raw, {
            "current_visibility_score": current_score,
            "previous_visibility_score": 0.0,
            "delta_visibility_score": current_score,
            "enter_fov_bonus": 0.0,
            "initial_absolute_factor": 0.5,
        }
    previous_score = float(previous.visibility_score)
    delta = current_score - previous_score
    fov_bonus = 0.1 if bool(current.in_fov) and not bool(previous.in_fov) else 0.0
    return float(delta + fov_bonus), {
        "current_visibility_score": current_score,
        "previous_visibility_score": previous_score,
        "delta_visibility_score": float(delta),
        "enter_fov_bonus": float(fov_bonus),
        "initial_absolute_factor": 0.0,
    }


def max_abs_reconstruction_error(traces: Iterable[TurnRewardTrace]) -> float:
    errors = [abs(float(trace.reconstruction_error or 0.0)) for trace in traces]
    return max(errors, default=0.0)
