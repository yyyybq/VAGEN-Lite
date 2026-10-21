"""Null Scorer: score -> 0.0 for every candidate. The default; the loop behaves as if the module did not exist."""


class NullScorer:
    def reset(self, ctx):
        pass

    def score(self, obs, state, candidates):
        return [0.0 for _ in candidates]
