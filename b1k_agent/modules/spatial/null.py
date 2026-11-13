"""Null SpatialMemory: update -> None. The default; the loop behaves as if the module did not exist."""


class NullSpatialMemory:
    def reset(self, ctx):
        pass

    def update(self, obs, state):
        return None
