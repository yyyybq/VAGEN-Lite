"""M-memory. See README.md."""

from agent.registry import ImplSpec
from modules.memory.null import NullMemory

IMPLS = {
    "null": ImplSpec(slot="memory", factory=lambda deps: NullMemory(), description="no-op"),
}
