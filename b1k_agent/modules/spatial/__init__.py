"""M-spatial. See README.md."""

from agent.registry import ImplSpec
from modules.spatial.null import NullSpatialMemory

IMPLS = {
    "null": ImplSpec(slot="spatial", factory=lambda deps: NullSpatialMemory(), description="no-op"),
}
