"""M-approach. See README.md."""

from agent.registry import ImplSpec
from modules.approach.null import NullApproach

IMPLS = {
    "null": ImplSpec(slot="approach", factory=lambda deps: NullApproach(), description="no-op"),
}
