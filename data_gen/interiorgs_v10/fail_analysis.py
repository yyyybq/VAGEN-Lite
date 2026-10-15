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
  oscillating    the turns alternate opposite moves (turn_left/turn_right or forward/backward) 2+ times
  wandered       none of the above: the budget ran out without getting close

plus the success conditions still unmet at the end (the env's gate names) and blocked moves per failure.  The
report's user-facing groups: 撞墙 = collision, 来回走 = oscillating, 丢物体 = went_backwards + wandered (the camera
lost the target or never approached it); near_miss and the two other endings are listed separately.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

COLLISION_MIN = 3
MODES = ("collision", "invalid_end", "low_info_end", "near_miss", "went_backwards", "oscillating", "wandered")
GROUPS = {"撞墙": ("collision",), "来回走": ("oscillating",), "丢物体": ("went_backwards", "wandered"),
          "差一个条件": ("near_miss",), "格式/画面结束": ("invalid_end", "low_info_end")}
OPP = {("turn_left", "turn_right"), ("turn_right", "turn_left"), ("move_forward", "move_backward"),
       ("move_backward", "move_forward"), ("move_left", "move_right"), ("move_right", "move_left")}


def flips(traj: Sequence[Dict[str, Any]]) -> int:
    """How often a turn's first action undoes the previous turn's last action."""
    n = 0
    for a, b in zip(traj, traj[1:]):
        if a.get("actions") and b.get("actions") and (a["actions"][-1], b["actions"][0]) in OPP:
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
        "unmet_when_closest": collections.Counter()})
    for r in rows:
        if "error" in r:
            continue
        for key in (r["task_type"], "all"):
            f = fam[key]
            f["episodes"] += 1
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
            best = min(traj, key=lambda t: len(t.get("failed_gates") or [0] * 99)) if traj else {}
            if m == "near_miss":
                for g in best.get("failed_gates") or []:
                    f["unmet_when_closest"][g] += 1
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
        out[k] = {"failed": v["failed"], "episodes": v["episodes"], "modes": dict(v["modes"].most_common()),
                  "mode_share_of_episodes": {m: round(v["modes"][m] / n, 4) for m in MODES if v["modes"][m]},
                  "groups": {g: sum(v["modes"][m] for m in ms) for g, ms in GROUPS.items()},
                  "group_share_of_episodes": {g: round(sum(v["modes"][m] for m in ms) / n, 4) for g, ms in GROUPS.items()},
                  "ends": dict(v["ends"].most_common()), "unmet_at_end": dict(v["unmet_at_end"].most_common()),
                  "unmet_when_closest_near_misses": dict(v["unmet_when_closest"].most_common()),
                  "blocked_per_failure": round(v["blocked"] / max(1, v["failed"]), 2),
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
