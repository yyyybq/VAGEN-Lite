"""Null goal manager: no ordering, no veto, never low-risk."""


class NullGoalManager:
    def reset(self, ctx):
        pass

    def order(self, goal_spec, state):
        return []

    def regression_risk(self, plan, state):
        return 0.0

    def low_risk_mode(self, state):
        return False
