"""M-explore. See README.md."""

from agent.registry import ImplSpec
from modules.explore.null import NullExplorer

IMPLS = {
    "null": ImplSpec(slot="explore", factory=lambda deps: NullExplorer(), description="no-op"),
}
