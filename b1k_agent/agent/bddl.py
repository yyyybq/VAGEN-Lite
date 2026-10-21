"""Minimal BDDL reader for the task definition (a legal policy input).

Parses :objects, :init and :goal of an activity problem file into a GoalSpec. It does not evaluate
anything against the simulator (that would need ground truth); it only turns the goal formula into
a list of subgoal literals a planner / goal manager can talk about.

Goal flattening:
  (and A B ...)           -> subgoals of A, B, ... in order
  (or A B ...)            -> each disjunct becomes its own option index (best-matched-clause scoring)
  (not (pred ...))        -> negated literal
  (forall/exists/forn/forpairs/fornpairs ...) -> kept as ONE quantified subgoal with its text
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Union

from agent.interfaces import GoalSpec, Subgoal

SExpr = Union[str, List["SExpr"]]
QUANTIFIERS = {"forall", "exists", "forn", "forpairs", "fornpairs"}


def tokenize(text: str) -> list[str]:
    text = re.sub(r";[^\n]*", "", text)  # comments
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse(tokens: list[str]) -> SExpr:
    stack: list[list] = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) < 2:
                raise ValueError("unbalanced ')' in BDDL")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise ValueError("unbalanced '(' in BDDL")
    return stack[0][0] if len(stack[0]) == 1 else stack[0]


def to_text(e: SExpr) -> str:
    if isinstance(e, str):
        return e
    return "(" + " ".join(to_text(x) for x in e) + ")"


def _section(problem: list, name: str) -> Optional[list]:
    for item in problem:
        if isinstance(item, list) and item and item[0] == name:
            return item
    return None


def _strip_var(a: str) -> str:
    return a[1:] if a.startswith("?") else a


# Stack of :objects mappings for the problem being parsed (set by parse_problem; used to ground forall).
_objects_ctx: list = []


def _expandable(decl: SExpr) -> bool:
    return isinstance(decl, list) and len(decl) == 3 and decl[1] == "-" and str(decl[0]).startswith("?")


def _substitute(e: SExpr, var: str, value: str) -> SExpr:
    if isinstance(e, str):
        return value if e == var else e
    return [_substitute(x, var, value) for x in e]


def _flatten_goal(e: SExpr, out: list, option: int, counter: list, notes: list) -> None:
    if isinstance(e, str):
        notes.append(f"bare atom in goal: {e}")
        return
    head = e[0] if e else ""
    if head == "and":
        for child in e[1:]:
            _flatten_goal(child, out, option, counter, notes)
    elif head == "or":
        for i, child in enumerate(e[1:]):
            _flatten_goal(child, out, option * 100 + i + 1 if option else i + 1, counter, notes)
    elif head == "not" and len(e) == 2 and isinstance(e[1], list) and e[1] and e[1][0] not in QUANTIFIERS | {"and", "or"}:
        lit = e[1]
        out.append(
            Subgoal(
                subgoal_id=f"g{counter[0]}",
                literal=to_text(e),
                predicate=str(lit[0]),
                args=tuple(_strip_var(str(a)) for a in lit[1:]),
                negated=True,
                option=option,
            )
        )
        counter[0] += 1
    elif head == "forall" and len(e) == 3 and _objects_ctx and _expandable(e[1]):
        # (forall (?x - cat) body): ground over the declared instances of cat
        var, cat = str(e[1][0]), str(e[1][2])
        insts = [o for o, c in _objects_ctx[-1].items() if c == cat]
        if not insts:
            notes.append(f"forall over {cat}: no declared instances; kept unexpanded")
            out.append(Subgoal(subgoal_id=f"g{counter[0]}", literal=to_text(e), predicate="forall", args=(),
                               option=option, quantified=True))
            counter[0] += 1
            return
        for inst in insts:
            _flatten_goal(_substitute(e[2], var, "?" + inst), out, option, counter, notes)
    elif head in QUANTIFIERS or head in {"not", "imply"}:
        out.append(
            Subgoal(
                subgoal_id=f"g{counter[0]}",
                literal=to_text(e),
                predicate=str(head),
                args=(),
                option=option,
                quantified=True,
            )
        )
        counter[0] += 1
    else:
        out.append(
            Subgoal(
                subgoal_id=f"g{counter[0]}",
                literal=to_text(e),
                predicate=str(head),
                args=tuple(_strip_var(str(a)) for a in e[1:] if isinstance(a, str)),
                option=option,
            )
        )
        counter[0] += 1


def parse_problem(text: str, task_name: str = "", path: str = "") -> GoalSpec:
    tree = parse(tokenize(text))
    if not (isinstance(tree, list) and tree and tree[0] == "define"):
        raise ValueError("not a BDDL problem (missing define)")
    objects_sec = _section(tree, ":objects") or [":objects"]
    objects = {}
    items = objects_sec[1:]
    pending: list[str] = []
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "-" and i + 1 < len(items):
            for inst in pending:
                objects[inst] = str(items[i + 1])
            pending = []
            i += 2
            continue
        pending.append(str(tok))
        i += 1
    init_sec = _section(tree, ":init") or [":init"]
    init_literals = tuple(to_text(x) for x in init_sec[1:])
    goal_sec = _section(tree, ":goal")
    subgoals: list = []
    notes: list = []
    goal_text = ""
    if goal_sec is not None and len(goal_sec) > 1:
        goal_text = to_text(goal_sec[1])
        _objects_ctx.append(objects)
        try:
            _flatten_goal(goal_sec[1], subgoals, 0, [0], notes)
        finally:
            _objects_ctx.pop()
    return GoalSpec(
        task_name=task_name,
        bddl_path=path,
        objects=objects,
        init_literals=init_literals,
        subgoals=tuple(subgoals),
        goal_text=goal_text,
        parse_notes=tuple(notes),
    )


def load_goal_spec(bddl_root: str, task_name: str, definition_id: int = 0) -> GoalSpec:
    path = os.path.join(bddl_root, task_name, f"problem{definition_id}.bddl")
    with open(path, "r", encoding="utf-8") as f:
        return parse_problem(f.read(), task_name=task_name, path=path)
