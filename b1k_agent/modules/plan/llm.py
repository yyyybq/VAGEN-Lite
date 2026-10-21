"""M-plan, strong-model implementation: choose the next subtask from the AVAILABLE skills only.

Inputs (all legal): head (and optionally wrist) RGB, the BDDL goal (objects + subgoal literals),
the natural-language task prompt, the executor's skill catalog (available skills only, with their
documented end conditions and known failures), the episode history summary, the system's subgoal
and holding estimates, retrieved experiences / spatial summary when those modules are on.

Output contract: the model picks a skill NAME and its arguments; the instruction sent to the
executor is rendered from that skill's template, so the model cannot invent executor instructions
for skills that do not exist. "__full_task__" renders to the exact trained task prompt.

Fallback (provider error, unparseable or invalid reply): the full-task subtask, logged with
source "plan.llm.fallback" -- deterministic and recorded.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agent.interfaces import FULL_TASK_SKILL, Subtask
from agent.prompts import render
from providers.base import ProviderRequest


def _schema(skill_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "task_complete": {"type": "boolean"},
            "skill": {"type": "string", "enum": skill_names},
            "args": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
            },
            "target_subgoals": {"type": "array", "items": {"type": "string"}},
            "max_steps": {"type": "integer"},
            "evidence": {"type": "string"},
        },
        "required": ["task_complete", "skill", "args", "target_subgoals", "max_steps", "evidence"],
        "additionalProperties": False,
    }


class LLMPlanner:
    def __init__(self, deps):
        if deps.provider is None or deps.prompts is None or not deps.prompt_version:
            raise ValueError("plan.llm needs a provider and prompt_version")
        self.provider = deps.provider
        self.prompt = deps.prompts.load(deps.prompt_version)
        p = dict(deps.params)
        self.cameras = tuple(p.get("cameras", ["head"]))
        self.max_tokens = int(p.get("max_tokens", 1024))
        self.timeout_s = float(p.get("timeout_s", 60.0))
        self.default_max_steps = int(p.get("default_max_steps", 900))
        self.min_max_steps = int(p.get("min_max_steps", 60))
        self.allow_task_complete = bool(p.get("allow_task_complete", True))

    def reset(self, ctx):
        self.ctx = ctx
        self.skills = {s.name: s for s in ctx.skills if s.available}
        if FULL_TASK_SKILL not in self.skills:
            raise ValueError("skill catalog must contain the full-task skill")

    # ------------------------------------------------------------------

    def _fallback(self, why: str) -> Subtask:
        return Subtask(subtask_id="", skill=FULL_TASK_SKILL, instruction=self.ctx.task_instruction,
                       max_steps=None, source="plan.llm.fallback", evidence=why)

    def _skills_text(self) -> str:
        lines = []
        for s in self.skills.values():
            status = "verified" if s.verified else "NOT verified (executor response to this instruction untested)"
            lines.append(f"- {s.name}: {s.description} | inputs: {', '.join(s.inputs) or 'none'} | "
                         f"ends when: {s.ends_when or 'n/a'} | known failures: {'; '.join(s.known_failures) or 'n/a'} | {status}")
        return "\n".join(lines)

    def _goal_text(self, state) -> str:
        gs = self.ctx.goal_spec
        if gs is None:
            return "(BDDL goal unavailable)"
        objs = ", ".join(f"{k} ({v})" for k, v in gs.objects.items())
        est = state.subgoal_estimates
        lines = [f"objects: {objs}"]
        for sg in gs.subgoals:
            e = est.get(sg.subgoal_id)
            lines.append(f"{sg.subgoal_id} [option {sg.option}] {sg.literal} -> estimate: {e.value if e else 'unknown'}")
        if state.subgoal_order:
            lines.append("suggested order: " + ", ".join(state.subgoal_order))
        return "\n".join(lines)

    def next_subtask(self, obs, state) -> Optional[Subtask]:
        holding = {k: v.value for k, v in state.holding.items()} or {"left": "unknown", "right": "unknown"}
        rejected = [f"{s.skill} {dict(s.args)}" for s in state.rejected_candidates]
        user = render(
            self.prompt["USER"],
            task_prompt=self.ctx.task_instruction,
            goal=self._goal_text(state),
            skills=self._skills_text(),
            history=state.summary_text() or "(start of episode)",
            holding=json.dumps(holding),
            experiences="\n".join(f"- [{e.outcome}] {e.skill}: {e.summary} ({e.provenance})" for e in state.retrieved) or "(none)",
            spatial=state.spatial_summary or "(none)",
            rejected="; ".join(rejected) or "(none)",
            low_risk="yes" if state.low_risk else "no",
            remaining=f"{state.remaining_budget_s():.0f}s" if state.remaining_budget_s() is not None else "unknown",
        )
        images = [(cam, obs.rgb[cam]) for cam in self.cameras if cam in obs.rgb]
        resp = self.provider.complete(ProviderRequest(
            system=self.prompt["SYSTEM"], user_text=user, images=images,
            json_schema=_schema(sorted(self.skills)), max_tokens=self.max_tokens,
            timeout_s=self.timeout_s, purpose="plan.next_subtask"))
        if resp.error or not isinstance(resp.parsed, dict):
            return self._fallback(f"provider: {resp.error or 'unparseable reply'}")
        r: dict[str, Any] = resp.parsed
        if r.get("task_complete") and self.allow_task_complete:
            return None
        skill = self.skills.get(str(r.get("skill")))
        if skill is None:
            return self._fallback(f"unknown skill {r.get('skill')!r}")
        args = {}
        for item in r.get("args") or []:
            if isinstance(item, dict) and "name" in item:
                args[str(item["name"])] = str(item.get("value", ""))
        if skill.name == FULL_TASK_SKILL:
            instruction = self.ctx.task_instruction
        else:
            missing = [a for a in skill.inputs if a not in args]
            if missing:
                return self._fallback(f"skill {skill.name} missing args {missing}")
            instruction = skill.instruction_template.format(**args)
        known = {sg.subgoal_id for sg in self.ctx.goal_spec.subgoals} if self.ctx.goal_spec else set()
        targets = tuple(t for t in r.get("target_subgoals") or [] if t in known)
        max_steps = int(r.get("max_steps") or self.default_max_steps)
        max_steps = max(self.min_max_steps, min(max_steps, self.default_max_steps * 4))
        return Subtask(subtask_id="", skill=skill.name, instruction=instruction, args=args,
                       target_subgoals=targets, max_steps=max_steps if skill.name != FULL_TASK_SKILL else None,
                       source=f"plan.llm@{resp.provider}", evidence=str(r.get("evidence", ""))[:500])
