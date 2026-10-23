"""Null checker: always UNKNOWN, which the loop treats as "keep executing"."""

from agent.interfaces import Judgment, JudgmentStatus


class NullChecker:
    def reset(self, ctx):
        pass

    def judge(self, obs, state):
        return Judgment(status=JudgmentStatus.UNKNOWN, confidence=0.0, evidence="", source="check.null")
