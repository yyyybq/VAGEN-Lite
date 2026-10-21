"""M-check, strong-model implementation: judge the current subtask from legal observations.

Returns done / in_progress / failed / unknown with a confidence, VISIBLE evidence, an optional
failure type (agent.interfaces.FAILURE_TYPES), per-subgoal estimates and holding estimates.
Low-confidence verdicts are downgraded to UNKNOWN (params.min_confidence) so a shaky "done" does not
end a subtask. Provider errors -> UNKNOWN (keep executing), logged.
"""

from __future__ import annotations

import json

import numpy as np

from agent.fields import PROPRIO_SLICES
from agent.interfaces import FAILURE_TYPES, FULL_TASK_SKILL, Judgment, JudgmentStatus
from agent.prompts import render
from providers.base import ProviderRequest

_STATUS = {s.value: s for s in JudgmentStatus}


def _schema(subgoal_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": [s.value for s in JudgmentStatus]},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
            "failure_type": {"type": "string", "enum": list(FAILURE_TYPES) + ["none"]},
            "subgoals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": subgoal_ids or ["none"]},
                        "estimate": {"type": "string", "enum": ["satisfied", "unsatisfied", "unknown"]},
                    },
                    "required": ["id", "estimate"],
                    "additionalProperties": False,
                },
            },
            "holding_left": {"type": "string"},
            "holding_right": {"type": "string"},
        },
        "required": ["status", "confidence", "evidence", "failure_type", "subgoals", "holding_left", "holding_right"],
        "additionalProperties": False,
    }


def proprio_summary(p: np.ndarray) -> str:
    lw = float(np.sum(p[PROPRIO_SLICES["gripper_left_qpos"]]))
    rw = float(np.sum(p[PROPRIO_SLICES["gripper_right_qpos"]]))
    base = np.round(p[PROPRIO_SLICES["base_qvel"]], 3).tolist()
    return f"gripper widths (m): left {lw:.3f}, right {rw:.3f} (open ~0.10, closed ~0.0); base velocity (vx, vy, wz): {base}"


class LLMChecker:
    def __init__(self, deps):
        if deps.provider is None or deps.prompts is None or not deps.prompt_version:
            raise ValueError("check.llm needs a provider and prompt_version")
        self.provider = deps.provider
        self.prompt = deps.prompts.load(deps.prompt_version)
        p = dict(deps.params)
        self.cameras = tuple(p.get("cameras", ["head", "left_wrist", "right_wrist"]))
        self.min_confidence = float(p.get("min_confidence", 0.6))
        self.max_tokens = int(p.get("max_tokens", 1024))
        self.timeout_s = float(p.get("timeout_s", 60.0))

    def reset(self, ctx):
        self.ctx = ctx

    def judge(self, obs, state) -> Judgment:
        sub = state.current_subtask
        gs = self.ctx.goal_spec
        subgoal_ids = [sg.subgoal_id for sg in gs.subgoals] if gs else []
        goal_lines = "\n".join(f"{sg.subgoal_id}: {sg.literal}" for sg in gs.subgoals) if gs else "(unavailable)"
        if sub is None:
            subtask_text = "(none)"
        elif sub.skill == FULL_TASK_SKILL:
            subtask_text = f"the whole task: {sub.instruction}"
        else:
            subtask_text = f"{sub.skill} {dict(sub.args)}: '{sub.instruction}' (targets {', '.join(sub.target_subgoals) or 'n/a'})"
        user = render(
            self.prompt["USER"],
            task_prompt=self.ctx.task_instruction,
            goal=goal_lines,
            subtask=subtask_text,
            steps_in_subtask=str(state.steps_in_subtask()),
            proprio=proprio_summary(obs.proprio),
            history=state.summary_text() or "(start of episode)",
        )
        images = [(cam, obs.rgb[cam]) for cam in self.cameras if cam in obs.rgb]
        resp = self.provider.complete(ProviderRequest(
            system=self.prompt["SYSTEM"], user_text=user, images=images, json_schema=_schema(subgoal_ids),
            max_tokens=self.max_tokens, timeout_s=self.timeout_s, purpose="check.judge"))
        if resp.error or not isinstance(resp.parsed, dict):
            return Judgment(status=JudgmentStatus.UNKNOWN, evidence=f"provider: {resp.error or 'unparseable'}",
                            source="check.llm.fallback")
        r = resp.parsed
        status = _STATUS.get(str(r.get("status")), JudgmentStatus.UNKNOWN)
        try:
            conf = float(r.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(max(conf, 0.0), 1.0)
        if status in (JudgmentStatus.DONE, JudgmentStatus.FAILED) and conf < self.min_confidence:
            status = JudgmentStatus.UNKNOWN
        ft = str(r.get("failure_type") or "none")
        failure_type = ft if (status == JudgmentStatus.FAILED and ft in FAILURE_TYPES) else None
        updates = {}
        for item in r.get("subgoals") or []:
            if isinstance(item, dict) and item.get("id") in subgoal_ids and item.get("estimate") in ("satisfied", "unsatisfied", "unknown"):
                updates[item["id"]] = item["estimate"]
        holding = {}
        for arm in ("left", "right"):
            v = r.get(f"holding_{arm}")
            if isinstance(v, str):
                holding[arm] = v.strip()[:80]
        return Judgment(status=status, confidence=conf, evidence=str(r.get("evidence", ""))[:600],
                        failure_type=failure_type, subgoal_updates=updates, holding_updates=holding,
                        source=f"check.llm@{resp.provider}")


__all__ = ["LLMChecker", "proprio_summary", "json"]
