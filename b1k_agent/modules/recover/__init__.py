"""M-recover (bounded recovery). See README.md."""

from agent.registry import ImplSpec


class NullRecovery:
    def reset(self, ctx):
        pass

    def propose(self, obs, state, failure_type):
        return None


def _scripted(deps):
    from modules.recover.scripted import ScriptedRecovery

    return ScriptedRecovery(deps)


IMPLS = {
    "null": ImplSpec(slot="recover", factory=lambda deps: NullRecovery(), description="no recovery"),
    "scripted": ImplSpec(slot="recover", factory=_scripted, description="proprio-only primitives (+ verified instruction skills)"),
}
