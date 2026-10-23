"""M-score. See README.md."""

from agent.registry import ImplSpec
from modules.score.null import NullScorer

IMPLS = {
    "null": ImplSpec(slot="score", factory=lambda deps: NullScorer(), description="no-op"),
}
