#!/usr/bin/env python3
"""Tier g's check (the user's task book of 2026-11-06; design G_DESIGN_20261106.md section 4): before the runs with
dialogue history, tiers d and c are played again WITHOUT history on small subsets (run_n3.sh tiers dr, drc) and set
against the episodes recorded then (N1's, tier c's).  Greedy decoding and the same code path should give the same
episodes; g - d is read as the effect of history only if they do.  No simulator, no model.

    python downstream/n/n_dr_check.py pick --instances I.jsonl --set1 N1/B/episodes.jsonl --per 2 --out SUB.jsonl
    python downstream/n/n_dr_check.py compare --rerun G/dr --recorded N1 N1b --models B S ... --out OUT

pick: `per` instances of every (set, task type) in file order (set 1 = the instance ids in --set1's episodes, the
others set 2), so both N1 sets and all six types are in (24 with per 2).
compare: per model, each rerun episode against the recorded one of the same instance: the executed actions of every
turn, the number of turns, success.  An episode with a frame problem in either run (the rerun's frames_scan.json
lists it among corrupted / broken / hinted instances, or the recorded run replayed it: episodes_corrupted.jsonl) is
reported apart and left out of the rule.  Rule (per model, set before running): pass when at most 1 clean episode's
success differs and at most 2 clean episodes' actions differ; otherwise stop and ask the user.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

MAX_FLIPS, MAX_DIVERGED = 1, 2


def read_jsonl(p) -> list:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def pick(insts: list, set1: set, per: int) -> list:
    taken, out = collections.Counter(), []
    for i in insts:
        key = ("1" if i["instance_id"] in set1 else "2", i["task_type"])
        if taken[key] < per:
            taken[key] += 1
            out.append(i)
    return out


def actions_of(row: dict) -> list:
    return [list(t.get("executed") or t.get("actions") or []) for t in row.get("turn_log") or []]


def frame_flagged(scan: dict) -> set:
    """Instance ids a frames_scan.json names (corrupted tiles, broken or hinted frames), any evaluation in it."""
    out = set()
    for ev in (scan.get("evals") or {}).values():
        for key in ("instances", "broken", "hints"):
            out |= set((ev.get(key) or {}).keys())
    return out


def compare_model(rerun: list, recorded: dict, flagged: set) -> dict:
    rows, n_clean, flips, diverged, missing, same_text = [], 0, 0, 0, 0, 0
    for r in rerun:
        iid = r["instance_id"]
        old = recorded.get(iid)
        if old is None or r.get("error") or old.get("error"):
            missing += 1
            continue
        a, b = actions_of(r), actions_of(old)
        first = next((k + 1 for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b)) + 1
        flip = bool(r.get("success")) != bool(old.get("success"))
        clean = iid not in flagged
        same_text += [t.get("reply") for t in r.get("turn_log") or []] == [t.get("reply") for t in old.get("turn_log") or []]
        if clean:
            n_clean += 1
            flips += flip
            diverged += first is not None
        if first is not None or flip:
            rows.append({"instance_id": iid, "first_different_turn": first, "success_rerun": bool(r.get("success")),
                         "success_recorded": bool(old.get("success")), "frame_flag": not clean})
    ok = flips <= MAX_FLIPS and diverged <= MAX_DIVERGED
    return {"episodes": len(rerun), "clean": n_clean, "frame_flagged": len(rerun) - n_clean - missing,
            "missing": missing, "success_flips": flips, "actions_diverged": diverged, "replies_identical": same_text,
            "pass": ok, "different": rows}


def compare_cmd(args) -> int:
    recorded_dirs = [Path(d) for d in args.recorded]
    rep = {"rule": f"per model: success flips <= {MAX_FLIPS} and diverged actions <= {MAX_DIVERGED} (clean episodes)",
           "models": {}}
    for m in args.models:
        rr = Path(args.rerun) / m
        if not (rr / "episodes.jsonl").exists():
            rep["models"][m] = {"pass": False, "note": "no rerun episodes"}
            continue
        rerun = read_jsonl(rr / "episodes.jsonl")
        recorded, flagged = {}, set()
        for d in recorded_dirs:
            if (d / m / "episodes.jsonl").exists():
                recorded.update({r["instance_id"]: r for r in read_jsonl(d / m / "episodes.jsonl")})
            if (d / m / "episodes_corrupted.jsonl").exists():
                flagged |= {r["instance_id"] for r in read_jsonl(d / m / "episodes_corrupted.jsonl")}
        if (rr / "frames_scan.json").exists():
            flagged |= frame_flagged(json.loads((rr / "frames_scan.json").read_text()))
        if (rr / "episodes_corrupted.jsonl").exists():
            flagged |= {r["instance_id"] for r in read_jsonl(rr / "episodes_corrupted.jsonl")}
        rep["models"][m] = compare_model(rerun, recorded, flagged)
    rep["pass"] = all(v.get("pass") for v in rep["models"].values())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dr_check.json").write_text(json.dumps(rep, indent=1) + "\n")
    lines = ["| 模型 | 回合 | 干净 | 有帧问题 | 缺记录 | 成败不同 | 动作不同 | 通过 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for m, v in rep["models"].items():
        if "episodes" in v:
            lines.append(f"| {m} | {v['episodes']} | {v['clean']} | {v['frame_flagged']} | {v['missing']} | "
                         f"{v['success_flips']} | {v['actions_diverged']} | {'是' if v['pass'] else '否'} |")
        else:
            lines.append(f"| {m} | — | — | — | — | — | — | 否（{v.get('note')}） |")
    (out / "dr_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": rep["pass"], **{m: {k: v.get(k) for k in ("success_flips", "actions_diverged", "pass")}
                                              for m, v in rep["models"].items()}}))
    return 0 if rep["pass"] else 1


def pick_cmd(args) -> int:
    set1 = {r["instance_id"] for r in read_jsonl(args.set1)}
    sub = pick(read_jsonl(args.instances), set1, args.per)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in sub))
    print(json.dumps({"picked": len(sub), "by": dict(collections.Counter(
        f"set{'1' if r['instance_id'] in set1 else '2'}/{r['task_type']}" for r in sub))}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--instances", required=True)
    p.add_argument("--set1", required=True, help="episodes.jsonl of N1's first set (its instance ids)")
    p.add_argument("--per", type=int, default=2)
    p.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--rerun", required=True, help="the check tier's folder (G/dr or G/drc) with <MODEL>/episodes.jsonl")
    c.add_argument("--recorded", nargs="+", required=True, help="the recorded evaluation folders with <MODEL>/")
    c.add_argument("--models", nargs="+", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"pick": pick_cmd, "compare": compare_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
