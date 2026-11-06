#!/usr/bin/env python3
"""Tier g's check (the user's task book of 2026-11-06; design G_DESIGN_20261106.md section 4): before the runs with
dialogue history, tiers d and c are played again WITHOUT history on small subsets (run_n3.sh tiers dr, drc) and set
against the episodes recorded then (N1's, tier c's).  Greedy decoding on the same code path should give the same
episodes; g - d is read as the effect of history only if they do.  No simulator, no model.

    python downstream/n/n_dr_check.py pick --instances I.jsonl --set1 N1/B/episodes.jsonl --per 2 [--seed S]
                                           [--avoid OTHER_SUB.jsonl] --out SUB.jsonl
    python downstream/n/n_dr_check.py compare --instances SUB.jsonl --rerun G/dr --recorded N1 N1b --models B S ...
                                              --out OUT

pick: `per` instances of every (set, task type), at most one per scene (and none from the scenes of --avoid's
subset), drawn in a fixed-seed shuffle; set 1 = the instance ids in --set1's episodes, the others set 2.
compare, per model, each subset instance's rerun episode against the recorded one: per turn the attempted actions,
the blocked ones and the pose after the turn (to 1e-6 m / rad), the first pose, the end pose, the number of turns,
success.  An instance whose rerun frames are still corrupted or broken at the end (the rerun's frames_scan.json:
"instances" / "broken") is listed apart and left out of the rule; frames the scan only hints at stay in.
Rule (set before running): a model passes when every subset instance has an episode in both runs, neither an error,
at most MAX_FLIPS clean episodes differ in success and at most MAX_DIVERGED clean episodes differ at all; the check
passes when every model passes and at most MAX_TOTAL_FLIPS success flips occur over all models.  Every difference
is listed with its first differing turn, to be looked at before the runs with history are submitted.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

MAX_FLIPS, MAX_DIVERGED, MAX_TOTAL_FLIPS = 1, 2, 2
POSE_TOL = 1e-6


def read_jsonl(p) -> list:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def pick(insts: list, set1: set, per: int, seed: int = 20261106, avoid_scenes=()) -> list:
    rng = random.Random(seed)
    order = list(insts)
    rng.shuffle(order)
    used, taken, chosen = set(avoid_scenes), collections.Counter(), set()
    for i in order:
        key = ("1" if i["instance_id"] in set1 else "2", i["task_type"])
        if taken[key] < per and i["scene_id"] not in used:
            taken[key] += 1
            used.add(i["scene_id"])
            chosen.add(i["instance_id"])
    return [i for i in insts if i["instance_id"] in chosen]  # in file order


def _pose(p) -> tuple:
    return tuple(round(float(v) / POSE_TOL) for v in (p or []))


def trace(row: dict) -> tuple:
    """What the comparison looks at: the first pose, per turn (attempted actions, blocked ones, pose after), the end
    pose."""
    turns = row.get("turn_log") or []
    first = _pose(turns[0].get("pose_before")) if turns else ()
    return first, [(tuple(t.get("executed") or t.get("actions") or []), tuple(t.get("blocked") or []),
                    _pose(t.get("pose_after"))) for t in turns], _pose(row.get("final_pose"))


def first_difference(a: dict, b: dict):
    """None when the two episodes match, else the first differing turn (0: the first pose)."""
    fa, ta, ea = trace(a)
    fb, tb, eb = trace(b)
    if fa != fb:
        return 0
    for k, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            return k + 1
    if len(ta) != len(tb):
        return min(len(ta), len(tb)) + 1
    return None if ea == eb else len(ta)


def frame_flags(scan: dict) -> tuple:
    """(instances whose frames are corrupted or broken, instances the scan only hints at) over its evaluations."""
    bad, hinted = set(), set()
    for ev in (scan.get("evals") or {}).values():
        bad |= set((ev.get("instances") or {}).keys()) | set((ev.get("broken") or {}).keys())
        hinted |= set((ev.get("hints") or {}).keys())
    return bad, hinted - bad


def compare_model(subset: list, rerun: dict, recorded: dict, bad=(), hinted=()) -> dict:
    absent, errors, rows = [], [], []
    clean = flips = diverged = same_text = 0
    for iid in subset:
        r, old = rerun.get(iid), recorded.get(iid)
        if r is None or old is None:
            absent.append(iid)
            continue
        if r.get("error") or old.get("error"):
            errors.append(iid)
            continue
        first = first_difference(r, old)
        flip = bool(r.get("success")) != bool(old.get("success"))
        same_text += [t.get("reply") for t in r.get("turn_log") or []] == [t.get("reply") for t in old.get("turn_log") or []]
        if iid not in bad:
            clean += 1
            flips += flip
            diverged += first is not None
        if first is not None or flip:
            rows.append({"instance_id": iid, "first_different_turn": first, "success_rerun": bool(r.get("success")),
                         "success_recorded": bool(old.get("success")), "frames_bad": iid in bad,
                         "frames_hinted": iid in hinted})
    ok = not absent and not errors and flips <= MAX_FLIPS and diverged <= MAX_DIVERGED
    return {"expected": len(subset), "compared": len(subset) - len(absent) - len(errors), "clean": clean,
            "frames_bad": sum(i in bad for i in subset), "frames_hinted": sum(i in hinted for i in subset),
            "absent": absent, "errors": errors, "success_flips": flips, "actions_diverged": diverged,
            "replies_identical": same_text, "pass": ok, "different": rows}


def compare_cmd(args) -> int:
    subset = [r["instance_id"] for r in read_jsonl(args.instances)]
    cap = MAX_TOTAL_FLIPS if args.max_total_flips is None else args.max_total_flips
    rep = {"rule": (f"per model: every instance in both runs, no error, success flips <= {MAX_FLIPS}, diverged <= "
                    f"{MAX_DIVERGED} (clean episodes); all models: success flips <= {cap}"),
           "subset": str(args.instances), "models": {}}
    for m in args.models:
        rr = Path(args.rerun) / m
        rerun = {r["instance_id"]: r for r in read_jsonl(rr / "episodes.jsonl")} if (rr / "episodes.jsonl").exists() else {}
        recorded = {}
        for d in map(Path, args.recorded):
            if (d / m / "episodes.jsonl").exists():
                recorded.update({r["instance_id"]: r for r in read_jsonl(d / m / "episodes.jsonl")})
        bad, hinted = (frame_flags(json.loads((rr / "frames_scan.json").read_text()))
                       if (rr / "frames_scan.json").exists() else (set(), set()))
        rep["models"][m] = compare_model(subset, rerun, recorded, bad, hinted)
    total = sum(v["success_flips"] for v in rep["models"].values())
    rep["total_success_flips"] = total
    rep["pass"] = all(v["pass"] for v in rep["models"].values()) and total <= cap
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dr_check.json").write_text(json.dumps(rep, indent=1) + "\n")
    lines = ["| 模型 | 应有 | 比上 | 干净 | 帧坏 | 帧提示 | 缺 / 出错 | 成败不同 | 回合不同 | 回复全同 | 通过 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for m, v in rep["models"].items():
        lines.append(f"| {m} | {v['expected']} | {v['compared']} | {v['clean']} | {v['frames_bad']} | {v['frames_hinted']} | "
                     f"{len(v['absent'])} / {len(v['errors'])} | {v['success_flips']} | {v['actions_diverged']} | "
                     f"{v['replies_identical']} | {'是' if v['pass'] else '否'} |")
    lines += ["", f"全部模型成败不同合计 {total}（上限 {cap}）；检查{'通过' if rep['pass'] else '不通过'}。", ""]
    for m, v in rep["models"].items():
        for d in v["different"]:
            lines.append(f"- {m} {d['instance_id']}：第 {d['first_different_turn']} 轮起不同；成败 重跑 "
                         f"{d['success_rerun']} / 当时 {d['success_recorded']}；帧坏 {d['frames_bad']}，帧提示 {d['frames_hinted']}")
    (out / "dr_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": rep["pass"], "total_success_flips": total,
                      **{m: {k: v[k] for k in ("compared", "success_flips", "actions_diverged", "pass")}
                         for m, v in rep["models"].items()}}))
    return 0 if rep["pass"] else 1


def pick_cmd(args) -> int:
    set1 = {r["instance_id"] for r in read_jsonl(args.set1)}
    avoid = {r["scene_id"] for r in read_jsonl(args.avoid)} if args.avoid else set()
    sub = pick(read_jsonl(args.instances), set1, args.per, args.seed, avoid)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in sub))
    print(json.dumps({"picked": len(sub), "scenes": len({r["scene_id"] for r in sub}), "by": dict(collections.Counter(
        f"set{'1' if r['instance_id'] in set1 else '2'}/{r['task_type']}" for r in sub))}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--instances", required=True)
    p.add_argument("--set1", required=True, help="episodes.jsonl of N1's first set (its instance ids)")
    p.add_argument("--per", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--avoid", default=None, help="another subset whose scenes are not to be used")
    p.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--instances", required=True, help="the subset (pick's file)")
    c.add_argument("--rerun", required=True, help="the check tier's folder (G/dr or G/drc) with <MODEL>/episodes.jsonl")
    c.add_argument("--recorded", nargs="+", required=True, help="the recorded evaluation folders with <MODEL>/")
    c.add_argument("--models", nargs="+", required=True)
    c.add_argument("--max-total-flips", type=int, default=None,
                   help="the all-models cap (default MAX_TOTAL_FLIPS, set for six models; the user approved 1 for three, 2026-11-06)")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"pick": pick_cmd, "compare": compare_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
