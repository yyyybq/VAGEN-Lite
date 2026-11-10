"""EXPLORATORY walked-distance odometer of the WalkerBench variants (DECISIONS G4: V1, V1T, V2).  See
vpd/EXPLORATORY_WB_VARIANTS.md.  The single source of both the eval-time line and the demo rewrite.

walked = the sum of the SHOWN distances of the MOVEs executed so far in the episode, before the current decision.
"Shown" is what the stock user text prints for a move (examples/vln_agent.py _build_user_message:
f" ({m['distance']:.1f}m)" when the distance is truthy, nothing otherwise), so a move without a printed distance adds
0.  The sum is rounded to 0.1 m after every addition (add()), identically on both paths:
- eval:  observation["available_moves"] of the step at which a MOVE was chosen -> shown(m["distance"]), added once the
         move executed successfully (vpd.walkerbench.variants' eval agent);
- demos: the "**Available Moves:**" block of the user text the expert's MOVE answered -> moves_in_text().
It is derived from the transcript alone (no lat/lng, no pano or target id), the same quantity the blind ODO baseline
sums (baselines.blind_action).

    line(w)                 "- Walked so far: 12.3 m"
    insert_line(text, w)    the line right after the unique "- Pitch: ..." line of a stock user text
    thought(w, stated)      "walked 12.3 of 146 m" (nav with a stated total) / "walked 12.3 m" (otherwise);
                            pure ASCII, <= 32 characters, so it passes run_vpd's short-THOUGHT regex
    demo_turns(messages)    per assistant turn: walked before the decision, the moves shown, the stated total
    with_line(rec) / with_thought(rec)   a demo record rewritten for V1T / V2 (images and every other byte kept)
"""

from __future__ import annotations

import copy
import json
import re

from vpd.walkerbench import action_regex as AR

LINE_PREFIX = "- Walked so far: "
PITCH_PREFIX = "- Pitch:"
THOUGHT_MAX_CHARS = 32  # run_vpd.SHORT_THOUGHT_MAX_CHARS (B19)
MOVES_HEADER = "**Available Moves:**"
IMAGE = "<image>"
# the stock move line: f"  {id}: {direction}" + (f" ({distance:.1f}m)" if distance else "")
MOVE_LINE = re.compile(r"^  (\d+): (.*?)(?: \((\d+\.\d)m\))?$")
STATED_RE = re.compile(r"Total distance: approximately (\d+(?:\.\d+)?) m")  # learn483.STATED_RE
THOUGHT_RE = re.compile(r"walked (\d+\.\d)(?: of (\d+(?:\.\d+)?))? m")
_ASCII = re.compile(r"[ !#-\[\]-~]*")  # run_vpd.ASCII_THOUGHT_CLASS: printable ASCII without '"' and '\'


def shown(distance) -> float:
    """The distance as the stock prompt prints it (0.1 m); 0.0 when the prompt prints none."""
    return float(f"{distance:.1f}") if distance else 0.0


def add(walked: float, distance) -> float:
    """walked + the shown distance of one executed MOVE, rounded to 0.1 m."""
    return round(float(walked) + shown(distance), 1)


def fmt(walked: float) -> str:
    return "%.1f" % walked


def line(walked: float) -> str:
    return LINE_PREFIX + fmt(walked) + " m"


def pitch_index(lines: list) -> int:
    idx = [i for i, ln in enumerate(lines) if ln.startswith(PITCH_PREFIX)]
    if len(idx) != 1:
        raise ValueError(f"expected exactly one '{PITCH_PREFIX}' line in the user text, found {len(idx)}")
    return idx[0]


def insert_line(text: str, walked: float) -> str:
    """text with line(walked) inserted right after its unique '- Pitch:' line (refuses a second odometer line)."""
    lines = text.split("\n")
    i = pitch_index(lines)
    if any(ln.startswith(LINE_PREFIX) for ln in lines):
        raise ValueError("the user text already holds an odometer line")
    return "\n".join(lines[:i + 1] + [line(walked)] + lines[i + 1:])


def stated_total(text: str) -> str | None:
    """The instruction's stated total distance as printed ('146'), None when the text states none."""
    m = STATED_RE.search(text or "")
    return m.group(1) if m else None


def thought(walked: float, stated: str | None = None) -> str:
    """V2's THOUGHT.  stated: the stated total as printed (nav), None (vis, or a nav task without one)."""
    t = f"walked {fmt(walked)} of {stated} m" if stated else f"walked {fmt(walked)} m"
    if len(t) > THOUGHT_MAX_CHARS or not _ASCII.fullmatch(t):
        raise ValueError(f"THOUGHT {t!r} is not <= {THOUGHT_MAX_CHARS} printable ASCII characters")
    return t


def parse_thought(text) -> dict | None:
    """{'walked': float, 'stated': str | None} of a THOUGHT in thought()'s exact format, else None."""
    m = THOUGHT_RE.fullmatch(text or "") if isinstance(text, str) else None
    if not m:
        return None
    return {"walked": float(m.group(1)), "stated": m.group(2)}


def moves_in_text(text: str) -> dict:
    """{move id: shown distance} of the '**Available Moves:**' block of a stock user text."""
    if MOVES_HEADER not in text:
        raise ValueError("no '**Available Moves:**' block in the user text")
    after = text.split(MOVES_HEADER, 1)[1]
    if not after.startswith("\n"):
        raise ValueError("the '**Available Moves:**' header is not followed by a newline")
    block = after[1:].split("\n\n", 1)[0]  # the stock text: header, the move lines (maybe none), a blank line
    out = {}
    for ln in block.split("\n"):
        m = MOVE_LINE.match(ln)
        if m is None:
            if ln.strip():
                raise ValueError(f"unexpected line in the moves block: {ln!r}")
            continue
        out[int(m.group(1))] = float(m.group(3)) if m.group(3) else 0.0
    return out


def moves_in_obs(available_moves) -> dict:
    """{move id: shown distance} of an observation's available_moves (what the prompt prints for them)."""
    return {int(m["id"]): shown(m.get("distance")) for m in available_moves or []}


def reply_action(reply: str) -> str:
    return json.loads(reply)["ACTION"]


def move_id(action: str) -> int | None:
    m = re.fullmatch(r"MOVE\((\d+)\)", action)
    return int(m.group(1)) if m else None


class Odometer:
    """Per-episode state of the eval agent: walked before the next decision."""

    def __init__(self):
        self.walked = 0.0

    def add(self, distance) -> float:
        self.walked = add(self.walked, distance)
        return self.walked


def demo_turns(messages: list) -> list:
    """Per assistant turn of a demo: {'msg_index', 'user_index', 'walked' (before the decision), 'moves' (shown),
    'stated', 'action'}.  A MOVE adds the shown distance of its id in the user text it answered (demos: every expert
    MOVE executed)."""
    out, walked, user_i = [], 0.0, None
    for i, m in enumerate(messages):
        if m["role"] == "user":
            user_i = i
            continue
        if m["role"] != "assistant":
            continue
        if user_i is None:
            raise ValueError(f"assistant message {i} answers no user turn")
        text = messages[user_i]["content"]
        moves = moves_in_text(text)
        act = reply_action(m["content"])
        out.append({"msg_index": i, "user_index": user_i, "walked": walked, "moves": moves,
                    "stated": stated_total(text), "action": act})
        mid = move_id(act)
        if mid is not None:
            if mid not in moves:
                raise ValueError(f"assistant message {i}: {act} is not among the shown moves {sorted(moves)}")
            walked = add(walked, moves[mid])
        user_i = None
    return out


def with_line(rec: dict) -> dict:
    """V1T: every user turn carries line(walked before its decision) after its '- Pitch:' line.  The '<image>'
    placeholder stays at the end of the text; replies, images and every other field are unchanged."""
    out = copy.deepcopy(rec)
    msgs = out["messages"]
    for t in demo_turns(rec["messages"]):
        u = msgs[t["user_index"]]
        u["content"] = insert_line(u["content"], t["walked"])
    return out


def with_thought(rec: dict, kind: str | None = None) -> dict:
    """V2: every reply's THOUGHT = thought(walked before the decision, the stated total for nav); user turns,
    actions and every other field are unchanged."""
    kind = kind or rec.get("task_type")
    out = copy.deepcopy(rec)
    msgs = out["messages"]
    for t in demo_turns(rec["messages"]):
        th = thought(t["walked"], t["stated"] if kind == "nav" else None)
        msgs[t["msg_index"]]["content"] = AR.reply(t["action"], th)
    return out
