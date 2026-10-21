"""Null planner: the whole task as one subtask with the executor's trained task prompt.

First call -> the full-task subtask; later calls (only reached after a checker says DONE) -> None,
i.e. "nothing left to plan". With every module off this makes the loop a pure executor passthrough.
"""

from agent.interfaces import FULL_TASK_SKILL, Subtask


class NullPlanner:
    def reset(self, ctx):
        self._ctx = ctx
        self._issued = False

    def next_subtask(self, obs, state):
        if self._issued:
            return None
        self._issued = True
        return Subtask(subtask_id="", skill=FULL_TASK_SKILL, instruction=self._ctx.task_instruction, source="plan.null")
