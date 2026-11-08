"""The clean K-turn history window, shared by the trainer (--history-k) and the eval agent (--history-turns).

    window(messages, k)        k == 0: full history (messages unchanged)
                               k > 0 : [sys] + (u, a) * k previous turns + u_t   (plan section 1.2, DECISIONS C1)
    train_windows(messages, k) one (context, i) per assistant turn messages[i]; context = window(messages[:i], k)

WalkerBench's stock _trim_history keeps [sys] + the last 2K messages, which starts with an orphan assistant
message and holds only K-1 full previous turns; it is never relied on (MAX_HISTORY_TURNS stays at its default).
The agent calls its trim right after appending the user message of step t, so the input always ends with a user
message.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

ROLE_CHAR = {"system": "s", "user": "u", "assistant": "a"}


def window(messages: Sequence[dict], k: int) -> Sequence[dict]:
    """[messages[0]] + messages[-(2k+1):] when len(messages) > 2k+2, else messages; k == 0 is the identity."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k == 0:
        return messages
    if len(messages) > 2 * k + 2:
        return [messages[0]] + list(messages[-(2 * k + 1):])
    return messages


def train_windows(messages: Sequence[dict], k: int) -> Iterator[tuple[Sequence[dict], int]]:
    """Yield (context ending with the user turn, index of the assistant reply) for every assistant turn."""
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        ctx = window(messages[:i], k)
        if not ctx or ctx[-1].get("role") != "user":
            raise ValueError(f"assistant turn {i} does not follow a user turn")
        yield ctx, i


def role_pattern(messages: Sequence[dict]) -> str:
    """'suaua...' (system/user/assistant) for assertions and logs."""
    return "".join(ROLE_CHAR.get(m.get("role"), "?") for m in messages)


def is_clean_window(messages: Sequence[dict], k: int) -> bool:
    """True iff messages is sys, (u, a) * j, u with j <= k (any j when k == 0)."""
    p = role_pattern(messages)
    if len(p) < 2 or p[0] != "s" or p[-1] != "u":
        return False
    body = p[1:-1]
    if len(body) % 2 or body != "ua" * (len(body) // 2):
        return False
    return k == 0 or len(body) // 2 <= k
