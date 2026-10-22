"""The truth JSONL format: written by replay_truth.py, read by join.py and label_distill.py.

Category: TRAINING_ANNOTATION (agent/fields.py "sim.predicate_truth"). Offline only; nothing at run
time reads these files.

One file holds one or more episodes; each episode is a header, truth records and a footer:

  {"kind": "truth_header", "schema_version": 1, "episode_id", "run_id", "task", "instance_id", "mode",
   "literals": [canonical goal literal, ...],            # unique, in first-appearance order
   "options": [[literal index, ...], ...],               # BehaviorTask.ground_goal_state_options
   "initial_satisfied": [literal index, ...],            # after the evaluator's load_batch / reset
   "every": N, "n_actions": n, "steps_path", "steps_sha256", ...}
  {"kind": "truth", "episode_id", "step": k, "satisfied": [literal index, ...], "success": bool,
   "q_now": float, "terminated": bool, "truncated": bool}
  {"kind": "truth_footer", "episode_id", "replay_steps", "replay_success", "replay_q_final",
   "official_steps", "official_q_final", "consistent": bool | null, "overrun_actions": int}

``step`` k = the state after k executed actions = the state the agent observed at its step k
(steps.jsonl / events.jsonl "step"), so a decision made at step k is compared with record k.

Literals are canonical s-expressions with "?" stripped: "(toggled_on radio_receiver.n.01_1)",
"(not (open cabinet.n.01_1))". The BDDL goal literals of agent/bddl.py GoalSpec map onto them with
``canonical_from_text``; quantified subgoals that agent/bddl.py could not ground stay unmapped.

Status: written and unit-tested on SYNTHETIC truth files; no truth file from the simulator exists yet.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, IO, Iterable, Optional, Sequence

from agent.bddl import QUANTIFIERS, parse, tokenize

TRUTH_SCHEMA_VERSION = 1
_NOT_A_LITERAL = QUANTIFIERS | {"and", "or", "imply"}


def canonical_literal(expr: Any) -> str:
    """Nested-list s-expression (bddl HEAD.body, agent.bddl.parse output) -> canonical text."""
    if isinstance(expr, str):
        return expr.lstrip("?")
    return "(" + " ".join(canonical_literal(x) for x in expr) + ")"


def canonical_from_text(text: str) -> Optional[str]:
    """Canonical literal of a BDDL literal text, or None when it is not a single (negated) atom."""
    try:
        e = parse(tokenize(text))
    except (ValueError, IndexError):
        return None
    if not isinstance(e, list) or not e:
        return None
    head = e[0]
    if head == "not":
        if len(e) != 2 or not isinstance(e[1], list) or not e[1] or e[1][0] in _NOT_A_LITERAL or e[1][0] == "not":
            return None
    elif head in _NOT_A_LITERAL or not isinstance(head, str):
        return None
    if any(isinstance(x, list) for x in (e[1][1:] if head == "not" else e[1:])):
        return None
    return canonical_literal(e)


def compute_q_score(success: bool, now: Sequence[Sequence[bool]], initial: Sequence[Sequence[bool]]) -> float:
    """Mirror of OmniGibson/omnigibson/metrics/task_metric.py compute_q_score (v3.9.3-post2): 1 on
    success, else max over goal options of (#literals false at start and true now) / #literals."""
    if success:
        return 1.0
    scores = []
    for now_opt, init_opt in zip(now, initial):
        if len(now_opt) == 0:
            scores.append(0.0)
            continue
        scores.append(sum(int((not i) and n) for n, i in zip(now_opt, init_opt)) / len(now_opt))
    return max(scores) if scores else 0.0


# ------------------------------------------------------------------------------------ writer


class TruthWriter:
    """Writes one episode. ``option_literals`` is one list of canonical literal texts per goal option."""

    def __init__(self, f: IO[str], episode_id: str, option_literals: Sequence[Sequence[str]], meta: dict,
                 q_fn: Callable[..., float] = compute_q_score):
        self.f = f
        self.episode_id = episode_id
        self.q_fn = q_fn
        self.literals: list[str] = list(dict.fromkeys(lit for opt in option_literals for lit in opt))
        index = {lit: i for i, lit in enumerate(self.literals)}
        self.options = [[index[lit] for lit in opt] for opt in option_literals]
        self.meta = dict(meta)
        self.initial: Optional[list[list[bool]]] = None
        self.last: Optional[dict] = None
        self.n_records = 0

    def _satisfied(self, masks: Sequence[Sequence[bool]]) -> list[int]:
        if len(masks) != len(self.options) or any(len(m) != len(o) for m, o in zip(masks, self.options)):
            raise ValueError("goal-option satisfaction does not match the header's options")
        return sorted({self.options[o][i] for o, mask in enumerate(masks) for i, v in enumerate(mask) if v})

    def _write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, sort_keys=True) + "\n")

    def start(self, initial_masks: Sequence[Sequence[bool]], success: bool) -> dict:
        """Header plus the step-0 record (the state after the evaluator's reset)."""
        self.initial = [[bool(v) for v in m] for m in initial_masks]
        self._write({"kind": "truth_header", "schema_version": TRUTH_SCHEMA_VERSION, "episode_id": self.episode_id,
                     "literals": self.literals, "options": self.options,
                     "initial_satisfied": self._satisfied(self.initial), **self.meta})
        return self.record(0, self.initial, success)

    def record(self, step: int, masks: Sequence[Sequence[bool]], success: bool, terminated: bool = False,
               truncated: bool = False) -> dict:
        if self.initial is None:
            raise RuntimeError("start() must be called first")
        masks = [[bool(v) for v in m] for m in masks]
        rec = {"kind": "truth", "episode_id": self.episode_id, "step": int(step), "satisfied": self._satisfied(masks),
               "success": bool(success), "q_now": float(self.q_fn(bool(success), masks, self.initial)),
               "terminated": bool(terminated), "truncated": bool(truncated)}
        self._write(rec)
        self.last = rec
        self.n_records += 1
        return rec

    def finish(self, **fields: Any) -> dict:
        rec = {"kind": "truth_footer", "episode_id": self.episode_id, **fields}
        self._write(rec)
        self.f.flush()
        return rec


# ------------------------------------------------------------------------------------ reader


@dataclasses.dataclass
class TruthTrack:
    episode_id: str
    header: dict
    records: list  # truth records sorted by step
    footer: Optional[dict] = None
    source: str = ""  # "<path>@<sha256[:12]>"

    @property
    def literals(self) -> list[str]:
        return list(self.header.get("literals") or [])

    @property
    def options(self) -> list[list[int]]:
        return [list(o) for o in self.header.get("options") or []]

    @property
    def consistent(self) -> Optional[bool]:
        """Footer verdict: did the replay reproduce the official outcome? None when unknown."""
        return None if self.footer is None else self.footer.get("consistent")

    def literal_truth(self, rec: dict) -> dict[str, bool]:
        sat = set(rec.get("satisfied") or [])
        return {lit: i in sat for i, lit in enumerate(self.literals)}

    def at(self, step: int, max_lag: int = 0) -> Optional[dict]:
        """The latest record at or before ``step`` that is at most ``max_lag`` steps older, else None."""
        best = None
        for rec in self.records:
            if rec["step"] <= step:
                best = rec
            else:
                break
        if best is None or step - best["step"] > max_lag:
            return None
        return best

    def regressions(self) -> list[dict]:
        """Goal literals that were true at one recorded step and false at a later one (flicker between
        two samples is not seen)."""
        out = []
        prev: dict[str, bool] = {}
        since: dict[str, int] = {}
        for rec in self.records:
            truth = self.literal_truth(rec)
            for lit, v in truth.items():
                if v and not prev.get(lit, False):
                    since[lit] = rec["step"]
                if prev.get(lit, False) and not v:
                    out.append({"literal": lit, "true_step": since.get(lit), "false_step": rec["step"]})
            prev = truth
        return out

    def first_true_steps(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.records:
            for i in rec.get("satisfied") or []:
                out.setdefault(self.literals[i], rec["step"])
        return out

    def max_q_now(self) -> Optional[float]:
        vals = [r.get("q_now") for r in self.records if r.get("q_now") is not None]
        return max(vals) if vals else None


def load_truth(paths: Iterable[str]) -> dict[str, TruthTrack]:
    """episode_id -> TruthTrack for every episode in the given truth files."""
    tracks: dict[str, TruthTrack] = {}
    for path in paths:
        with open(path, "rb") as f:
            data = f.read()
        source = f"{path}@{hashlib.sha256(data).hexdigest()[:12]}"
        for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            eid = str(rec.get("episode_id") or "")
            kind = rec.get("kind")
            if kind == "truth_header":
                if eid in tracks:
                    raise ValueError(f"{path}:{n}: episode {eid} already loaded (from {tracks[eid].source})")
                if rec.get("schema_version") != TRUTH_SCHEMA_VERSION:
                    raise ValueError(f"{path}:{n}: truth schema_version {rec.get('schema_version')} != {TRUTH_SCHEMA_VERSION}")
                tracks[eid] = TruthTrack(episode_id=eid, header=rec, records=[], source=source)
            elif kind in ("truth", "truth_footer"):
                if eid not in tracks:
                    raise ValueError(f"{path}:{n}: {kind} for episode {eid!r} before its header")
                if kind == "truth":
                    tracks[eid].records.append(rec)
                else:
                    tracks[eid].footer = rec
    for t in tracks.values():
        t.records.sort(key=lambda r: r["step"])
    return tracks
