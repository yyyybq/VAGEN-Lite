#!/usr/bin/env python3
"""N series terminal metrics (the user's task book of 2026-11-05, item 6; nothing re-run): how close the episodes came
to success, from their records.

    python downstream/n/n_terminal.py rebuild --instances I.jsonl [...] --eval DIR [--models M ...] --out ROWS.jsonl
    python downstream/n/n_terminal.py report --set NAME=ROWS.jsonl [...] [--pair A=B ...] [--exclude IDS ...]
                                             --instances I.jsonl [...] --out DIR

rebuild (one evaluation folder, DIR/<model>/episodes.jsonl): an episode records each turn's start and end pose, the
  actions it tried and the blocked ones in order, and its highest shaping score (max_shaping_score); not each step's
  score.  Each step's pose is rebuilt from the turn's start with n_env.apply_action (a blocked action leaves the pose
  where it was); which of the turn's actions were blocked is taken from the assignments that follow the recorded
  blocked ones in order, end at the recorded end pose and treat an action from one pose alike every time (blocked
  or not: the collision check is deterministic), and must be the same in all of them.  Each pose is scored with
  n_env.box_judge (the box judge the recorded score comes from; no rendering), with the instances' camera
  (n3_episode.camera_of, tier e's robot camera).  Per episode: success, the recorded and the rebuilt maximum, the
  highest score before the first blocked action (the start's included) and that action's step, and whether the
  rebuild holds (every turn rebuilt unambiguously and the rebuilt maximum equal to the recorded one to 1e-6).
report: per set and model, successes and failures apart: max_shaping_score (mean, quartiles, the share >= 0.5: a pose
  that passed every box gate of the canonical metric, the upper half being reserved for those), and, over the
  episodes whose rebuild holds, the highest score before the first block (an episode never blocked: its maximum);
  --pair A=B: per model, the two sets on their common instances: the mean difference of max_shaping_score with a 95%
  interval (whole scenes resampled) and the sign test (exact binomial).  Also the one lenient success of
  EVAL_PROTOCOL section 6 (max_shaping_score >= 0.5) beside the formal one: per set n3_report.tier_table's rates and
  pairings, per pair the same-instance gaps (n3_report.paired_gap).  DIR/report.json, DIR/report.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n_env as N  # noqa: E402
import n_report as R  # noqa: E402

TOL = 1e-6


def shaping(judge: dict) -> float:
    m = judge["metric"]
    return float(m.get("shaping_score", m.get("score")) or 0.0)


def key(pose) -> tuple:
    return tuple(round(float(v), 6) for v in pose)


def close(a, b) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= TOL for x, y in zip(a, b))


def turn_options(start, actions: list, blocked: list) -> list:
    """Every (flags, poses) for a turn: flags[k] True if action k was blocked, poses[k] the pose after it; following
    the blocked names in order, one outcome per (pose, action)."""
    out = []

    def go(k, pose, bi, flags, poses, seen):
        if k == len(actions):
            if bi == len(blocked):
                out.append((flags, poses))
            return
        a, here = actions[k], (key(pose), actions[k])
        if bi < len(blocked) and blocked[bi] == a and seen.get(here) is not False:
            go(k + 1, pose, bi + 1, flags + [True], poses + [pose], {**seen, here: True})
        if seen.get(here) is not True:
            new = tuple(N.apply_action(pose, a))
            go(k + 1, new, bi, flags + [False], poses + [new], {**seen, here: False})

    go(0, tuple(start), 0, [], [], {})
    return out


def rebuild_episode(row: dict, score) -> dict:
    """score(pose) -> shaping score.  See the module text."""
    turns = [t for t in row.get("turn_log") or [] if t.get("executed")]
    log = row.get("turn_log") or []
    start = tuple(log[0]["pose_before"]) if log else tuple(row["final_pose"])
    best = before = score(start)
    first_block, step, why = None, 0, None
    pose = start
    for t in turns:
        if not close(t["pose_before"], pose):
            why = why or f"turn {t['turn']} starts elsewhere"
        pose = tuple(t["pose_before"])
        opts = [o for o in turn_options(pose, t["executed"], [b[0] for b in t.get("blocked") or []])
                if close(o[1][-1], t["pose_after"])]
        if not opts:
            why = why or f"turn {t['turn']}: no assignment ends at its recorded pose"
            break
        if any(o[0] != opts[0][0] for o in opts[1:]):
            why = why or f"turn {t['turn']}: the blocked actions are ambiguous"
        for blocked, p in zip(*opts[0]):
            step += 1
            if blocked:
                first_block = first_block or step
                continue
            s = score(p)
            best = max(best, s)
            if first_block is None:
                before = max(before, s)
        pose = tuple(t["pose_after"])
    recorded = float(row.get("max_shaping_score") or 0.0)
    if why is None and abs(best - recorded) > TOL:
        why = f"rebuilt maximum {best:.6f} against the recorded {recorded:.6f}"
    return {"instance_id": row["instance_id"], "scene_id": row.get("scene_id"), "task_type": row.get("task_type"),
            "success": bool(row.get("success")), "max_shaping": recorded, "rebuilt_max": best,
            "before_first_block": before, "first_block_step": first_block, "steps": step,
            "blocked": int(row.get("blocked") or 0), "ok": why is None, "why": why}


def rebuild_cmd(args) -> int:
    insts = {r["instance_id"]: r for p in args.instances for r in R.read_jsonl(p)}
    import n3_episode
    mode, values = n3_episode.camera_of(list(insts.values()))
    if values:
        import n3_env
        n3_env.use_camera(values["height_m"], values["fovy_deg"])
    cache = {}

    def scorer(iid):
        item = insts[iid]["item"]

        def score(pose):
            k = (iid, key(pose))
            if k not in cache:
                cache[k] = shaping(N.box_judge(item, pose))
            return cache[k]
        return score

    out, tally = [], collections.Counter()
    for m in args.models:
        for row in R.read_jsonl(args.eval / m / "episodes.jsonl"):
            if row.get("error") or row["instance_id"] not in insts:
                tally[f"{m} skipped"] += 1
                continue
            r = dict(rebuild_episode(row, scorer(row["instance_id"])), model=m)
            tally[f"{m} {'ok' if r['ok'] else 'not rebuilt'}"] += 1
            out.append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({"camera": mode, **dict(sorted(tally.items()))}))
    return 0


# ------------------------------------------------------------------------------------------------ report
def quantile(xs: list, q: float) -> float:
    if not xs:
        return math.nan
    xs = sorted(xs)
    h = (len(xs) - 1) * q
    lo = math.floor(h)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (h - lo)


def dist(xs: list) -> dict:
    return {"n": len(xs), "mean": sum(xs) / len(xs) if xs else math.nan, "q25": quantile(xs, 0.25),
            "median": quantile(xs, 0.5), "q75": quantile(xs, 0.75),
            "at_least_half": sum(x >= 0.5 for x in xs), "share_at_least_half": (sum(x >= 0.5 for x in xs) / len(xs)
                                                                                if xs else math.nan)}


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return math.nan
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def paired(a: dict, b: dict, scenes: dict, draws: int = 2000, seed: int = 20261105) -> dict:
    """a, b: {instance: value}.  Mean of a - b over the common instances, 95% interval with whole scenes resampled,
    sign test."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    d = {i: a[i] - b[i] for i in ids}
    groups = collections.defaultdict(list)
    for i in ids:
        groups[scenes.get(i, i)].append(d[i])
    g, rng, means = list(groups.values()), random.Random(seed), []
    for _ in range(draws):
        pick = [g[rng.randrange(len(g))] for _ in g]
        means.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
    means.sort()
    up, down = sum(v > TOL for v in d.values()), sum(v < -TOL for v in d.values())
    return {"n": len(ids), "mean_a": sum(a[i] for i in ids) / len(ids), "mean_b": sum(b[i] for i in ids) / len(ids),
            "diff": sum(d.values()) / len(ids), "low": means[int(0.025 * draws)], "high": means[int(0.975 * draws) - 1],
            "a_higher": up, "b_higher": down, "p": binom_two_sided(up, up + down)}


def by_model(rows: list, key) -> dict:
    """{model: {instance: key(row)}}."""
    out = collections.defaultdict(dict)
    for r in rows:
        out[r["model"]][r["instance_id"]] = key(r)
    return dict(out)


def lenient(r: dict) -> bool:
    """EVAL_PROTOCOL section 6: a visited pose passed the canonical box geometry (shaping >= 0.5)."""
    return r["max_shaping"] >= 0.5


def lenient_part(sets: dict, pairs=(), exclude=()) -> dict:
    """Formal and lenient success side by side: per set n3_report.tier_table (rates, RL against SFT, the two RL
    against each other, trained against base), per pair A=B the same-instance gaps of those differences."""
    import n3_report as N3R

    keep = {n: [r for r in rows if r["instance_id"] not in exclude] for n, rows in sets.items()}
    succ = {n: {"formal": by_model(rows, lambda r: bool(r["success"])), "lenient": by_model(rows, lenient)}
            for n, rows in keep.items()}
    out = {"sets": {n: {k: N3R.tier_table(v) for k, v in s.items()} for n, s in succ.items()}, "pairs": {}}
    combos = [(m, R.sft_of(m)) for m in N3R.RL] + list(N3R.RL_PAIRS)
    for a, b in pairs:
        if a in succ and b in succ:
            out["pairs"][f"{a}-{b}"] = {f"{m}-{ref}": {k: N3R.paired_gap(succ[a][k], succ[b][k], m, ref)
                                                       for k in ("formal", "lenient")} for m, ref in combos}
    return out


def build(sets: dict, pairs=(), exclude=(), scenes=None) -> dict:
    """sets: {name: rows (rebuild's)}."""
    scenes = scenes or {}
    rep = {"sets": {}, "pairs": {}, "excluded": sorted(exclude), "lenient": lenient_part(sets, pairs, exclude)}
    for name, rows in sets.items():
        rows = [r for r in rows if r["instance_id"] not in exclude]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["model"]].append(r)
        rep["sets"][name] = {}
        for m, rs in by.items():
            cell = {"episodes": len(rs), "rebuilt": sum(r["ok"] for r in rs),
                    "not_rebuilt": collections.Counter((r["why"] or "").split(":")[0] for r in rs if not r["ok"])}
            for side, want in (("success", True), ("failure", False)):
                sel = [r for r in rs if r["success"] is want]
                ok = [r for r in sel if r["ok"]]
                cell[side] = {"max_shaping": dist([r["max_shaping"] for r in sel]),
                              "before_first_block": dist([r["before_first_block"] for r in ok]),
                              "ever_blocked": sum(r["first_block_step"] is not None for r in ok)}
            rep["sets"][name][m] = cell
    for a, b in pairs:
        if a not in sets or b not in sets:
            continue
        rep["pairs"][f"{a}-{b}"] = {}
        for m in rep["sets"][a]:
            va = {r["instance_id"]: r["max_shaping"] for r in sets[a] if r["model"] == m and r["instance_id"] not in exclude}
            vb = {r["instance_id"]: r["max_shaping"] for r in sets[b] if r["model"] == m and r["instance_id"] not in exclude}
            if vb:
                rep["pairs"][f"{a}-{b}"][m] = paired(va, vb, scenes)
    return rep


def f3(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def dist_cell(d: dict) -> str:
    if not d["n"]:
        return "—"
    return (f"{f3(d['mean'])}（{f3(d['q25'])} / {f3(d['median'])} / {f3(d['q75'])}；≥0.5：{d['at_least_half']}/{d['n']}）")


def markdown(rep: dict) -> str:
    lines = ["# N 系列终点指标（自动生成）", "",
             "每格：平均（四分位 25% / 中位 / 75%；≥ 0.5 的回合数 / 回合数）。≥ 0.5 = 走到过通过全部盒判定门槛的位姿。", ""]
    for name, by in rep["sets"].items():
        lines += [f"## {name}", "", "| 模型 | 成功：最高塑形分 | 失败：最高塑形分 | 失败：第一次被挡之前的最高分 | 失败里被挡过 | 重建成立 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for m, c in by.items():
            f = c["failure"]
            lines.append(f"| {m} | {dist_cell(c['success']['max_shaping'])} | {dist_cell(f['max_shaping'])} | "
                         f"{dist_cell(f['before_first_block'])} | {f['ever_blocked']}/{f['before_first_block']['n']} | "
                         f"{c['rebuilt']}/{c['episodes']} |")
        lines.append("")
    for name, by in rep["pairs"].items():
        lines += [f"## 配对：{name}（最高塑形分，同一批实例）", "", "| 模型 | 实例 | 前者 | 后者 | 差（95% 区间，按场景重抽） | 前者高 : 后者高 | 符号检验 p |",
                  "| --- | ---: | --- | --- | --- | --- | --- |"]
        for m, c in by.items():
            if not c["n"]:
                continue
            lines.append(f"| {m} | {c['n']} | {f3(c['mean_a'])} | {f3(c['mean_b'])} | {c['diff']:+.3f}（{c['low']:+.3f} 到 "
                         f"{c['high']:+.3f}） | {c['a_higher']} : {c['b_higher']} | {c['p']:.3g} |")
        lines.append("")
    return "\n".join(lines + lenient_markdown(rep.get("lenient") or {})) + "\n"


def lenient_markdown(part: dict) -> list:
    if not part:
        return []
    import n3_report as N3R

    lines = ["## 宽松成功（EVAL_PROTOCOL 第 6 节），和正式成功并排", "",
             "宽松成功 = 走到过通过全部盒判定门槛的位姿（塑形分 ≥ 0.5），不要求画面可见和 45 度规则；只作参考。", ""]
    for name, s in part["sets"].items():
        lines += [f"### {name}", "", "| 模型 | 正式成功 | 宽松成功 |", "| --- | --- | --- |"]
        for m, r in s["formal"]["models"].items():
            lines.append(f"| {m} | {R.cell(r)} | {R.cell(s['lenient']['models'][m])} |")
        lines += ["", "| 配对 | 正式成功之差 | 宽松成功之差 |", "| --- | --- | --- |"]
        for k, c in s["formal"]["pairs"].items():
            lines.append(f"| {k} | {N3R.diff_cell(c)} | {N3R.diff_cell(s['lenient']['pairs'].get(k))} |")
        lines.append("")
    for name, by in part["pairs"].items():
        lines += [f"### 同一批实例上两档之差：{name}", "", "| 配对 | 正式成功 | 宽松成功 |", "| --- | --- | --- |"]
        for k, g in by.items():
            lines.append(f"| {k} | {N3R.gap_cell(g['formal'])} | {N3R.gap_cell(g['lenient'])} |")
        lines.append("")
    return lines


def report_cmd(args) -> int:
    sets = {}
    for spec in args.set:
        name, path = spec.split("=", 1)
        sets[name] = R.read_jsonl(Path(path))
    scenes = {r["instance_id"]: r.get("scene_id") for p in args.instances for r in R.read_jsonl(p)}
    pairs = [tuple(s.split("=", 1)) for s in args.pair]
    rep = build(sets, pairs, R.excluded(args.exclude), scenes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (args.out / "report.md").write_text(markdown(rep), encoding="utf-8")
    print(json.dumps({n: {m: c["rebuilt"] for m, c in by.items()} for n, by in rep["sets"].items()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("rebuild")
    b.add_argument("--instances", type=Path, nargs="+", required=True)
    b.add_argument("--eval", type=Path, required=True)
    b.add_argument("--models", nargs="+", default=list(R.MODELS))
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--set", nargs="+", required=True, help="NAME=ROWS.jsonl (rebuild's)")
    r.add_argument("--pair", nargs="*", default=[], help="A=B: two sets paired on their common instances")
    r.add_argument("--exclude", type=Path, nargs="*", default=[])
    r.add_argument("--instances", type=Path, nargs="*", default=[], help="for the scenes of the paired interval")
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"rebuild": rebuild_cmd, "report": report_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
