"""M-plan (high-level planning). See README.md."""

from agent.registry import ImplSpec
from modules.plan.null import NullPlanner


def _llm(deps):
    from modules.plan.llm import LLMPlanner

    return LLMPlanner(deps)


IMPLS = {
    "null": ImplSpec(slot="plan", factory=lambda deps: NullPlanner(), description="full task prompt, once"),
    "llm": ImplSpec(slot="plan", factory=_llm, uses_provider=True, description="strong-model skill selection"),
}
