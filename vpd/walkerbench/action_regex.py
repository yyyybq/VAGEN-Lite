"""The constrained-decoding regex for one WalkerBench step (survey 2's builder, checked with xgrammar 0.1.25).

    action_regex(n_moves, tmax=0)

matches exactly json.dumps({"THOUGHT": t, "ACTION": a}) with
- a = MOVE(i) for a legal id i in n..1 (sort_moves_by_direction renumbers the moves 1..n,
  direction_calculator.py:247-248), ROTATION(heading, pitch) with heading 0..360 and pitch -85..85, each with an
  optional single decimal, or STOP(\\"ARRIVED\\") (json.dumps escapes the quotes);
- t empty when tmax == 0 (the primary THOUGHT-empty mode), else 0..tmax characters without '"', '\\' or newline.
tmax counts characters; the report's "short" THOUGHT is <= 32 tokens (DECISIONS B19), which the caller converts.
"""

from __future__ import annotations

import json

HEAD = r"0|[1-9][0-9]?|[12][0-9][0-9]|3[0-5][0-9]|360"
PITCH = r"-?([0-9]|[1-7][0-9]|8[0-5])"
STOP_ACTION = 'STOP("ARRIVED")'


def action_regex(n_moves: int, tmax: int = 0) -> str:
    if n_moves < 0 or tmax < 0:
        raise ValueError(f"n_moves and tmax must be >= 0, got {n_moves}, {tmax}")
    thought = "" if tmax == 0 else r'[^"\\\n]{0,%d}' % tmax
    alts = ([r"MOVE\((" + "|".join(str(i) for i in range(n_moves, 0, -1)) + r")\)"] if n_moves else [])
    alts += [r"ROTATION\((" + HEAD + r")(\.[0-9])?, (" + PITCH + r")(\.[0-9])?\)", r'STOP\(\\"ARRIVED\\"\)']
    return r'\{"THOUGHT": "' + thought + r'", "ACTION": "(' + "|".join(alts) + r')"\}'


def reply(action: str, thought: str = "") -> str:
    """The canonical reply text for an action (what demos train on and the regex accepts)."""
    return json.dumps({"THOUGHT": thought, "ACTION": action})
