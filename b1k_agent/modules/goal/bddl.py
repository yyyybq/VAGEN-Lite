"""M-goal, BDDL-based goal manager (no model calls).

Q-score counts goal literals that were false at the start and are true at the end, maximised over
goal options (OmniGibson/omnigibson/metrics/task_metric.py). So this module:

  order()            sorts the subgoals: easy before hard, enabling before dependent
                     (open X before put-inside X; close X after everything that goes inside X),
                     already-estimated-satisfied ones last;
  regression_risk()  for a candidate subtask, 1.0 when it would move the object whose state an
                     already-satisfied (estimated), untargeted literal describes, else 0.0;
  low_risk_mode()    True once less than ``low_risk_fraction`` of the time budget is left.

Satisfaction estimates come from EpisodeState.subgoal_estimates (written from M-check returns),
never from simulator truth.
"""

from __future__ import annotations

from agent.interfaces import FULL_TASK_SKILL

# Rough difficulty classes for R1Pro + a learned executor (lower = earlier). Unlisted -> 2.
DIFFICULTY = {
    "toggled_on": 0, "open": 0, "closed": 1, "nextto": 1, "ontop": 1, "under": 1, "touching": 1,
    "inside": 2, "overlaid": 2, "draped": 3, "filled": 3, "contains": 3, "covered": 3, "saturated": 3,
    "cooked": 3, "frozen": 3, "hot": 3, "attached": 3, "folded": 3, "unfolded": 3, "on_fire": 3,
}


class BDDLGoalManager:
    def __init__(self, deps):
        p = dict(deps.params)
        self.low_risk_fraction = float(p.get("low_risk_fraction", 0.05))
        self.full_task_risk = float(p.get("full_task_risk", 0.0))
        self.unresolved_risk = float(p.get("unresolved_risk", 0.5))

    def reset(self, ctx):
        self.ctx = ctx

    def order(self, goal_spec, state) -> list[str]:
        if goal_spec is None:
            return []
        est = state.subgoal_estimates
        subgoals = list(goal_spec.subgoals)
        opened = {sg.args[0] for sg in subgoals if sg.predicate == "open" and sg.args and not sg.negated}
        closed_later = {sg.args[0] for sg in subgoals if (sg.predicate == "open" and sg.negated) or sg.predicate == "closed"}

        def key(sg):
            satisfied = 1 if (est.get(sg.subgoal_id) and est[sg.subgoal_id].value == "satisfied") else 0
            diff = DIFFICULTY.get(sg.predicate, 2)
            phase = 1
            if sg.predicate == "open" and not sg.negated:
                phase = 0  # enables putting things inside
            if sg.args and sg.args[0] in closed_later and (sg.predicate == "closed" or (sg.predicate == "open" and sg.negated)):
                phase = 2  # close containers last
            if sg.predicate in ("inside",) and len(sg.args) > 1 and sg.args[1] in opened:
                phase = 1
            return (satisfied, phase, sg.quantified, diff, sg.option, sg.subgoal_id)

        return [sg.subgoal_id for sg in sorted(subgoals, key=key)]

    def _resolve(self, value: str) -> set:
        """Map a skill argument to BDDL instances: exact instance name, else instances whose category
        or name contains the words of the value (e.g. "radio" -> radio_receiver.n.01_1)."""
        gs = self.ctx.goal_spec
        v = str(value).strip().lstrip("?").lower()
        if v in gs.objects:
            return {v}
        words = [w for w in v.replace("_", " ").split() if len(w) > 2]
        return {inst for inst, cat in gs.objects.items()
                if words and all(w in (inst + " " + cat).lower().replace("_", " ") for w in words)}

    def regression_risk(self, plan, state) -> float:
        """1.0 if the subtask would move an object whose state an already-satisfied, untargeted goal
        literal describes (its FIRST argument; a shared receptacle or support is not "moved"), else 0.0.
        Unresolvable object arguments count as risky only when something is already satisfied."""
        gs = self.ctx.goal_spec
        if gs is None or plan is None:
            return 0.0
        if plan.skill == FULL_TASK_SKILL:
            return self.full_task_risk
        est = state.subgoal_estimates
        satisfied = [sg for sg in gs.subgoals if est.get(sg.subgoal_id) and est[sg.subgoal_id].value == "satisfied"
                     and sg.subgoal_id not in plan.target_subgoals and sg.args]
        if not satisfied:
            return 0.0
        touched: set = set()
        for value in (plan.args or {}).values():
            resolved = self._resolve(value)
            if not resolved:
                return self.unresolved_risk
            touched |= resolved
        return 1.0 if any(sg.args[0] in touched for sg in satisfied) else 0.0

    def low_risk_mode(self, state) -> bool:
        budget = self.ctx.wallclock_budget_s
        remaining = state.remaining_budget_s()
        if budget and remaining is not None and remaining < self.low_risk_fraction * budget:
            return True
        if self.ctx.max_steps and state.step >= 0:
            return (self.ctx.max_steps - state.step) < self.low_risk_fraction * self.ctx.max_steps
        return False
