"""Null Approach: propose(obs, state, subtask) -> None. The default; the loop behaves as if the module did not exist."""


class NullApproach:
    def reset(self, ctx):
        pass

    def propose(self, obs, state, subtask):
        return None
