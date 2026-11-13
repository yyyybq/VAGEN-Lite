"""Null Memory: retrieve -> [], propose_adjustment -> None. The default; the loop behaves as if the module did not exist."""


class NullMemory:
    def reset(self, ctx):
        pass

    def retrieve(self, obs, state, k):
        return []

    def propose_adjustment(self, obs, state, next_subtask):
        return None
