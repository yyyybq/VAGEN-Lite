"""M-goal (goal management against the BDDL goal). See README.md."""

from agent.registry import ImplSpec
from modules.goal.null import NullGoalManager


def _bddl(deps):
    from modules.goal.bddl import BDDLGoalManager

    return BDDLGoalManager(deps)


IMPLS = {
    "null": ImplSpec(slot="goal", factory=lambda deps: NullGoalManager(), description="no-op"),
    "bddl": ImplSpec(slot="goal", factory=_bddl, description="order / protect / low-risk from the BDDL goal"),
}
