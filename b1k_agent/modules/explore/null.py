"""Null Explorer: propose -> None. The default; the loop behaves as if the module did not exist."""


class NullExplorer:
    def reset(self, ctx):
        pass

    def propose(self, obs, state):
        return None
