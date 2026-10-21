"""M-check (progress checking). See README.md."""

from agent.registry import ImplSpec
from modules.check.null import NullChecker


def _llm(deps):
    from modules.check.llm import LLMChecker

    return LLMChecker(deps)


def _rule(deps):
    from modules.check.rule import RuleChecker

    return RuleChecker(deps)


IMPLS = {
    "null": ImplSpec(slot="check", factory=lambda deps: NullChecker(), description="always unknown"),
    "llm": ImplSpec(slot="check", factory=_llm, uses_provider=True, description="strong-model judgment"),
    "rule": ImplSpec(slot="check", factory=_rule, description="proprioception stall / holding rules"),
}
