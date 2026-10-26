"""How the failed evaluation episodes fail, per family, from the recorded trajectories (round 8 method, in the
repo since round 10 so rounds can be compared with one definition).

    python -m data_gen.interiorgs_v10.fail_analysis --eval EVAL --models NAME [NAME ...] --out OUT.json
    python -m data_gen.interiorgs_v10.fail_analysis --eval EVAL_A --models A --eval EVAL_B --models B --out OUT.json

Each failed episode gets one primary failure mode, the first that applies in this order:

  collision      the camera ran into things: ended by 3 blocked moves in a row (rounds 6-9 rules) or, under the
                 round-10 rules where blocked moves never end the episode, at least COLLISION_MIN blocked moves
  invalid_end    ended by 3 unparseable answers in a row
  low_info_end   ended by 3 near-uniform frames in a row (e.g. facing a wall)
  near_miss      at some turn exactly one success condition was unmet (the env's gates), but never none
  went_backwards progress at the end below the start
  oscillating    the turns alternate opposite moves (turn_left/turn_right or forward/backward) 2+ times; since
                 round 11 backing off from a blocked move does not count (backs_off)
  wandered       none of the above: the budget ran out without getting close

plus the success conditions still unmet at the end (the env's gate names) and blocked moves per failure.  The
report's user-facing groups: 撞墙 = collision, 来回走 = oscillating, 丢物体 = went_backwards + wandered (the camera
lost the target or never approached it); near_miss and the two other endings are listed separately.

Round 11 adds two views (per family and "all"):

  blocked_reaction  in every episode, success or not, every turn with a blocked move that has a next turn: does
                    the next turn start with the move that was just blocked (repeat, read from the next turn's
                    feedback line), start with its opposite (back_off), or have a blocked move itself
                    (blocked_again)?
  focus             the three conditions the round-10 failures missed most (FOCUS: the target not in the frame,
                    no face of the target clearly visible, the target too small in the frame): in how many failed
                    episodes each is still unmet at the end, and in how many near misses it is the one unmet
                    condition at the closest turn.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

COLLISION_MIN = 3
MODES = ("collision", "invalid_end", "low_info_end", "near_miss", "went_backwards", "oscillating", "wandered")
GROUPS = {"撞墙": ("collision",), "来回走": ("oscillating",), "丢物体": ("went_backwards", "wandered"),
          "差一个条件": ("near_miss",), "格式/画面结束": ("invalid_end", "low_info_end")}
OPP = {("turn_left", "turn_right"), ("turn_right", "turn_left"), ("move_forward", "move_backward"),
       ("move_backward", "move_forward"), ("move_left", "move_right"), ("move_right", "move_left")}
# the env's feedback after a turn with blocked moves: "2 of 3 actions executed; blocked: move_backward (blocked by wall)."
BLOCKED_RE = re.compile(r"\b(move_forward|move_backward|move_left|move_right|turn_left|turn_right) \(blocked by")
FOCUS = ("目标不在画面里", "没有一个面看清楚", "目标太小")


def focus_of(gate: str) -> Optional[str]:
    """Which of the three FOCUS conditions an env gate name is (the target or reference object, not an occluder)."""
    if gate == "inside_frame":
        return "目标不在画面里"
    if gate.endswith("_clearly_visible") and not gate.startswith("occluder"):
        return "没有一个面看清楚"
    if gate == "min_area":
        return "目标太小"
    return None


def blocked_in_feedback(text: str) -> List[str]:
    """The moves a turn's feedback line names as blocked (those of the turn before)."""
    return BLOCKED_RE.findall(text.split("blocked:", 1)[1]) if "blocked:" in text else []


def blocked_reaction(traj: Sequence[Dict[str, Any]]) -> collections.Counter:
    """Counts over the turns with a blocked move that have a next turn (see the module docstring)."""
    c: collections.Counter = collections.Counter()
    for t, nxt in zip(traj, traj[1:]):
        if not int(t.get("collisions") or 0):
            continue
        c["turns"] += 1
        blocked = blocked_in_feedback(nxt.get("user") or "")
        first = (nxt.get("actions") or [None])[0]
        c["named"] += bool(blocked)
        c["repeat"] += first is not None and first in blocked
        c["back_off"] += first is not None and any((b, first) in OPP for b in blocked)
        c["blocked_again"] += int(nxt.get("collisions") or 0) > 0
    return c


def backs_off(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Round 11: turn b starts by backing off from a move blocked in turn a (the opposite of a blocked move named
    in b's feedback line, or of a's last action when the feedback names none) - what the round-11 blocked-turn
    samples teach, not oscillation."""
    if not int(a.get("collisions") or 0) or not b.get("actions"):
        return False
    blocked = blocked_in_feedback(b.get("user") or "") or list(a.get("actions") or [])[-1:]
    return any((m, b["actions"][0]) in OPP for m in blocked)


def flips(traj: Sequence[Dict[str, Any]]) -> int:
    """How often a turn's first action undoes the previous turn's last action, not counting backing off from a
    blocked move (round 11, ``backs_off``)."""
    n = 0
    for a, b in zip(traj, traj[1:]):
        if a.get("actions") and b.get("actions") and (a["actions"][-1], b["actions"][0]) in OPP and not backs_off(a, b):
            n += 1
    return n


def closest(traj: Sequence[Dict[str, Any]]) -> Optional[int]:
    """The fewest success conditions unmet after any turn (None without gate records)."""
    n = [len(t["failed_gates"]) for t in traj if t.get("failed_gates") is not None]
    return min(n) if n else None


def mode(row: Dict[str, Any]) -> str:
    end = row.get("end_reason")
    if end == "consecutive_collision" or int(row.get("collisions") or 0) >= COLLISION_MIN:
        return "collision"
    if end == "consecutive_invalid":
        return "invalid_end"
    if end == "low_info":
        return "low_info_end"
    traj = row.get("trajectory") or []
    prog = [t["progress"] for t in traj if t.get("progress") is not None]
    start = (row.get("start") or {}).get("progress")
    if closest(traj) == 1:
        return "near_miss"
    if prog and start is not None and prog[-1] < start - 0.02:
        return "went_backwards"
    if flips(traj) >= 2:
        return "oscillating"
    return "wandered"


def analyse(rows: Sequence[Dict[str, Any]], tasks: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Per family: failures, their modes, the unmet conditions at the end, blocked moves per failure, examples;
    plus an "all" entry and the user-facing groups (shares of all episodes)."""
    fam: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: {
        "failed": 0, "episodes": 0, "modes": collections.Counter(), "unmet_at_end": collections.Counter(),
        "examples": collections.defaultdict(list), "ends": collections.Counter(), "blocked": 0,
        "unmet_when_closest": collections.Counter(), "reaction": collections.Counter(),
        "focus_at_end": collections.Counter(), "focus_near_miss": collections.Counter()})
    for r in rows:
        if "error" in r:
            continue
        for key in (r["task_type"], "all"):
            f = fam[key]
            f["episodes"] += 1
            f["reaction"].update(blocked_reaction(r.get("trajectory") or []))
            if r["success"]:
                continue
            f["failed"] += 1
            m = mode(r)
            f["modes"][m] += 1
            f["ends"][r.get("end_reason")] += 1
            f["blocked"] += int(r.get("collisions") or 0)
            traj = r.get("trajectory") or []
            last = traj[-1] if traj else {}
            for g in last.get("failed_gates") or []:
                f["unmet_at_end"][g] += 1
            f["focus_at_end"].update({x for x in map(focus_of, last.get("failed_gates") or []) if x})
            best = min(traj, key=lambda t: len(t.get("failed_gates") or [0] * 99)) if traj else {}
            if m == "near_miss":
                for g in best.get("failed_gates") or []:
                    f["unmet_when_closest"][g] += 1
                f["focus_near_miss"].update({x for x in map(focus_of, best.get("failed_gates") or []) if x})
            if key != "all" and len(f["examples"][m]) < 3:
                prog = [t["progress"] for t in traj if t.get("progress") is not None]
                f["examples"][m].append({
                    "index": r.get("index"), "task_id": r.get("task_id"), "d_star": r.get("d_star"),
                    "desc": (tasks[r["index"]].get("task_description") if tasks and r.get("index") is not None
                             and r["index"] < len(tasks) else None),
                    "actions": " ".join("|".join(t.get("actions") or ["(none)"]) for t in traj),
                    "turns": r.get("turns"), "blocked": r.get("collisions"), "end": r.get("end_reason"),
                    "start_progress": (r.get("start") or {}).get("progress"),
                    "max_progress": max(prog) if prog else None, "end_progress": prog[-1] if prog else None,
                    "unmet_at_end": last.get("failed_gates"), "unmet_when_closest": best.get("failed_gates")})
    out = {}
    for k, v in sorted(fam.items()):
        n = max(1, v["episodes"])
        rc = v["reaction"]
        out[k] = {"failed": v["failed"], "episodes": v["episodes"], "modes": dict(v["modes"].most_common()),
                  "mode_share_of_episodes": {m: round(v["modes"][m] / n, 4) for m in MODES if v["modes"][m]},
                  "groups": {g: sum(v["modes"][m] for m in ms) for g, ms in GROUPS.items()},
                  "group_share_of_episodes": {g: round(sum(v["modes"][m] for m in ms) / n, 4) for g, ms in GROUPS.items()},
                  "ends": dict(v["ends"].most_common()), "unmet_at_end": dict(v["unmet_at_end"].most_common()),
                  "unmet_when_closest_near_misses": dict(v["unmet_when_closest"].most_common()),
                  "blocked_per_failure": round(v["blocked"] / max(1, v["failed"]), 2),
                  "blocked_reaction": {"turns": rc["turns"], "named": rc["named"],
                                       **{m: rc[m] for m in ("repeat", "back_off", "blocked_again")},
                                       **{f"{m}_share": round(rc[m] / max(1, rc["turns"]), 4)
                                          for m in ("repeat", "back_off", "blocked_again")}},
                  "focus_at_end": {x: v["focus_at_end"][x] for x in FOCUS},
                  "focus_near_miss": {x: v["focus_near_miss"][x] for x in FOCUS},
                  "near_misses": v["modes"]["near_miss"],
                  "examples": dict(v["examples"])}
    return out


def markdown(report: Dict[str, Dict[str, Any]]) -> str:
    """One row per model: episodes, failures, and the groups as counts (share of all episodes)."""
    groups = list(GROUPS)
    lines = ["| 模型 | 题数 | 失败 | " + " | ".join(groups) + " | 每道失败题的被挡动作 |",
             "| --- | --- | --- | " + " | ".join("---" for _ in groups) + " | --- |"]
    for name, fams in report.items():
        a = fams.get("all") or {"episodes": 0, "failed": 0, "groups": {}, "group_share_of_episodes": {},
                                "blocked_per_failure": 0}
        cells = [f"{a['groups'].get(g, 0)}（{100 * a['group_share_of_episodes'].get(g, 0):.1f}%）" for g in groups]
        lines.append(f"| {name} | {a['episodes']} | {a['failed']} | " + " | ".join(cells) + f" | {a['blocked_per_failure']} |")
    # round 11: the turn after a blocked move, and the three conditions (failed episodes / near misses)
    lines += ["", "| 模型 | 有动作被挡的回合（有下一回合） | 下一回合先做刚被挡的动作 | 下一回合先做相反的动作 | 下一回合又被挡 |",
              "| --- | --- | --- | --- | --- |"]
    for name, fams in report.items():
        b = (fams.get("all") or {}).get("blocked_reaction")
        if b:
            lines.append(f"| {name} | {b['turns']} | " + " | ".join(
                f"{b[m]}（{100 * b[m + '_share']:.1f}%）" for m in ("repeat", "back_off", "blocked_again")) + " |")
    lines += ["", "| 模型 | 失败 | " + " | ".join(f"最后{x}" for x in FOCUS) + " | 差一个条件 | "
              + " | ".join(f"其中缺{x}" for x in FOCUS) + " |",
              "| --- | --- | " + " | ".join("---" for _ in FOCUS) + " | --- | " + " | ".join("---" for _ in FOCUS) + " |"]
    for name, fams in report.items():
        a = fams.get("all") or {}
        if "focus_at_end" in a:
            lines.append(f"| {name} | {a['failed']} | " + " | ".join(str(a["focus_at_end"][x]) for x in FOCUS)
                         + f" | {a['near_misses']} | " + " | ".join(str(a["focus_near_miss"][x]) for x in FOCUS) + " |")
    return "\n".join(lines) + "\n"


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True, help="an evaluation folder (tasks.jsonl, NAME/results.jsonl)")
    ap.add_argument("--models", action="append", nargs="+", required=True,
                    help="result names in the preceding --eval folder; a label can be given as NAME=LABEL")
    ap.add_argument("--out", required=True, help="JSON file; a .md table is written next to it")
    args = ap.parse_args(argv)
    if len(args.eval) != len(args.models):
        ap.error("give one --models list per --eval folder")
    report: Dict[str, Dict[str, Any]] = {}
    for eval_dir, names in zip(args.eval, args.models):
        tasks = _jsonl(Path(eval_dir) / "tasks.jsonl")
        for item in names:
            name, _, label = item.partition("=")
            report[label or name] = analyse(_jsonl(Path(eval_dir) / name / "results.jsonl"), tasks)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    Path(args.out).with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
