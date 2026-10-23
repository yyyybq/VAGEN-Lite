#!/usr/bin/env python3
"""Build the result report of the downstream line (login node, CPU; the vagen-lite Python has matplotlib / scipy).

    python -m downstream.report.build --out RUNS/report                      # real runs (downstream_runs)
    python -m downstream.report.build --runs-root /tmp/sim --out /tmp/sim/report   # simulated (report.simulate)
    python -m downstream.report.build --stage0 [--partial] --out DIR         # stage 0 (stopped D5 checkpoints);
                                                    # --partial: also evaluations still missing episodes, marked

Writes OUT/index.html (self-contained: figures and example strips embedded; publishable as an Artifact),
OUT/figures/*.png, OUT/results.json (every number on the page) and OUT/report.md (summary and figure links).

Comparison points:
  * D5: the latest checkpoint at which every start of the 100% / seed-1 runs has an evaluation (all present
    starts if none is common yet);
  * step 3: step 200,000 for each data share, over the training seeds whose evaluation exists for every start.
The main comparison (tables, per-task figures, examples) is step 3 at 100% when it exists, else D5.  All
comparisons are paired against the reference start B; see the glossary on the page.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path

from . import collect, examples, page, stats

FINAL_STEP = 200000
RUNS_ROOT = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")


def jsonable(x):
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 6)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def code_version() -> str:
    try:
        res = subprocess.run(["git", "-C", str(collect.DOWNSTREAM), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return res.stdout.strip() or None
    except Exception:
        return None


def d5_point(coll, ratio="100", seed=1):
    steps = stats.common_steps(coll, ratio, [seed])
    if not steps:
        steps = sorted({e.step for e in coll.first() if e.ratio == ratio and e.seed == seed})
    return steps[-1] if steps else None


def final_seeds(coll, ratio):
    have = {}
    for e in coll.first():
        if e.ratio == ratio and e.step == FINAL_STEP:
            have.setdefault(e.seed, set()).add(e.start)
    starts = set(coll.starts())
    full = sorted(k for k, got in have.items() if got >= starts)
    return full or sorted(have)


def noise_rows(coll):
    first = {(e.exp, e.step): e for e in coll.first()}
    out = []
    for r in coll.evals:
        if r.kind != "repeat" or (r.exp, r.step) not in first:
            continue
        a = first[(r.exp, r.step)]
        oa = {(x["task"], x["eval_seed"]): x["success"] for x in a.episodes}
        ob = {(x["task"], x["eval_seed"]): x["success"] for x in r.episodes}
        common = set(oa) & set(ob)
        sa, sb = a.summary.get("episode_wall_mean_seconds") or 0.0, r.summary.get("episode_wall_mean_seconds") or 0.0
        out.append({"exp": a.exp, "step": a.step, "workers": [a.workers, r.workers],
                    "successes": [a.successes, r.successes], "episodes": [len(a.episodes), len(r.episodes)],
                    "rate_diff": (sum(ob[k] for k in common) - sum(oa[k] for k in common)) / max(1, len(common)),
                    "changed": sum(oa[k] != ob[k] for k in common), "compared": len(common),
                    "seconds": [sa, sb], "slowdown": sb / sa if sa else math.nan})
    return out


def describe(cmp, what: str) -> list:
    """Plain statements of the numbers of one comparison (no interpretation)."""
    if not cmp or not cmp["starts"]:
        return []
    lines = []
    for g in ("all", "spatial"):
        parts = []
        for s in cmp["starts"]:
            k, n = cmp["groups"][s][g]
            r, lo, hi = stats.wilson(k, n)
            parts.append(f"{s} {k}/{n}（{page.pct(r)}，区间 {page.pct(lo)}–{page.pct(hi)}）")
        lines.append(f"{what}，{stats.GROUP_ZH[g]}的成功回合 / 回合：" + "；".join(parts) + "。")
        diffs = []
        for s, v in cmp["vs_reference"].items():
            d = v["groups"][g]
            diffs.append(f"{s} {page.pts(d['task_mean_diff'])} 个百分点（{d['tasks']} 个任务的平均，区间 "
                         f"{page.pts(d['task_low'])} 到 {page.pts(d['task_high'])}；{d['tasks_up']} 个任务高于 "
                         f"{cmp['reference']}、{d['tasks_down']} 个低于；Wilcoxon p = {page.pval(d['wilcoxon_p'])}，"
                         f"符号检验 p = {page.pval(d['sign_p'])}）")
        if diffs:
            lines.append(f"{what}，{stats.GROUP_ZH[g]}与 {cmp['reference']} 的逐任务配对差：" + "；".join(diffs) + "。")
    return lines


STAGE0_RUN = "run_stopped_20261022"
BAND = (0.10, 0.60)


def in_band(x) -> bool:
    return BAND[0] <= x <= BAND[1]


def evaluation_rows(coll, evs) -> list:
    """Every stage-0 evaluation (first and repeat): episodes done of planned, the missing ones, and the success
    counts with the range they could still take if the missing episodes ran (all fail .. all succeed)."""
    rows = []
    for e in sorted(evs, key=lambda e: (collect.START_ORDER.index(e.start) if e.start in collect.START_ORDER else 9,
                                        e.step, e.kind)):
        c = stats.group_counts([dict(x, train_seed=e.seed) for x in e.episodes], coll.tasks)
        row = {"start": e.start, "step": e.step, "kind": e.kind, "partial": e.partial, "episodes": len(e.episodes),
               "planned": len(e.episodes) + len(e.missing), "missing": [f"{t} {s}" for t, s in e.missing]}
        for g in ("all", "spatial"):
            m = sum(1 for t, _ in e.missing if t in coll.tasks and stats.in_group(coll.tasks[t], g))
            k, n = c[g]
            row[g] = {"k": k, "n": n, "missing": m, "rate": k / n if n else math.nan,
                      "low": k / (n + m) if n + m else math.nan, "high": (k + m) / (n + m) if n + m else math.nan}
        rows.append(row)
    return rows


def spatial_band(coll, starts, evs, latest) -> list:
    """Per spatial task: the four-start average at `latest` (the list of 2026-10-22: 10%-60% -> E3 candidates) and
    the average over every evaluated checkpoint of every start, with its range if the missing episodes ran, and
    the four-start average at each checkpoint (does the task stay in the band?)."""
    out = []
    steps = sorted({e.step for e in evs})
    for t in sorted(t for t, i in coll.tasks.items() if i.klass == "spatial"):
        rates, lows, highs = {}, {}, {}
        for e in evs:
            k, n = stats.task_counts([dict(x, train_seed=e.seed) for x in e.episodes]).get(t, (0, 0))
            m = sum(1 for tt, _ in e.missing if tt == t)
            if n + m == 0:
                continue
            key = f"{e.start}@{e.step}"
            rates[key], lows[key], highs[key] = (k / n if n else math.nan), k / (n + m), (k + m) / (n + m)
        at = {s: rates[f"{s}@{latest}"] for s in starts if f"{s}@{latest}" in rates} if latest else {}
        mean = sum(at.values()) / len(at) if at else math.nan
        known = [v for v in rates.values() if not math.isnan(v)]
        mean_all = sum(known) / len(known) if known else math.nan
        low_all = sum(lows.values()) / len(lows) if lows else math.nan
        high_all = sum(highs.values()) / len(highs) if highs else math.nan
        per_step = {}
        for step in steps:
            vals = [rates.get(f"{s}@{step}") for s in starts]
            if all(v is not None and not math.isnan(v) for v in vals):
                per_step[step] = sum(vals) / len(vals)
        out.append({"task": t, "rates": at, "mean": mean, "in_band": in_band(mean),
                    "points": rates, "mean_all": mean_all, "low_all": low_all, "high_all": high_all,
                    "in_band_all": in_band(mean_all), "band_certain": in_band(low_all) == in_band(high_all),
                    "per_step": per_step, "steps_in_band": sum(in_band(v) for v in per_step.values())})
    return out


def fluctuation(coll, starts, reference: str = "B") -> dict:
    """The scale of differences that are not between starts: each start's adjacent checkpoints (same scenes; mid-
    training, learning rate still high), against the differences between starts at one checkpoint."""
    within, between = [], []
    groups = {g: {t for t, i in coll.tasks.items() if stats.in_group(i, g)} for g in ("all", "spatial")}
    have = {s: sorted({e.step for e in coll.first() if (e.start, e.ratio, e.seed) == (s, "100", 1)}) for s in starts}
    for s in starts:
        for a, b in zip(have[s], have[s][1:]):
            within.append({"start": s, "from": a, "to": b, **{g: pair(coll, s, a, s, b, keep) for g, keep in groups.items()}})
    for step in sorted(set().union(*have.values())) if have else []:
        if step not in have.get(reference, []):
            continue
        for s in starts:
            if s != reference and step in have[s]:
                between.append({"start": s, "reference": reference, "step": step,
                                **{g: pair(coll, reference, step, s, step, keep) for g, keep in groups.items()}})
    summary = {}
    for name, rows in (("within", within), ("between", between)):
        for g in ("all", "spatial"):
            pooled = [abs(r[g]["diff"]) for r in rows if not math.isnan(r[g]["diff"])]
            task = [abs(r[g]["task_mean_diff"]) for r in rows if not math.isnan(r[g]["task_mean_diff"])]
            summary[f"{name}_{g}"] = {"pairs": len(rows), "pooled_abs_mean": sum(pooled) / len(pooled) if pooled else math.nan,
                                      "pooled_abs_max": max(pooled, default=math.nan),
                                      "task_abs_mean": sum(task) / len(task) if task else math.nan,
                                      "task_abs_max": max(task, default=math.nan),
                                      "significant": sum(1 for r in rows if r[g]["p"] < RULE_P)}
    return {"within": within, "between": between, "summary": summary}


def stage0(coll, reference: str = "B") -> dict:
    """Stage 0 (user, 2026-10-22): the checkpoints kept from the stopped D5 runs (100% data, seed 1).
    Success by checkpoint per start (all tasks, the 9 spatial tasks); the spatial tasks whose success averaged over
    the four starts lies in 10%-60% at the latest checkpoint every start has (and averaged over every checkpoint);
    per start, the paired difference between its last two evaluated checkpoints (same scenes), to see whether the
    curve has flattened, and between all adjacent checkpoints against the differences between starts."""
    evs = [e for e in coll.first() if e.ratio == "100" and e.seed == 1]
    starts = [s for s in coll.starts() if any(e.start == s for e in evs)]
    curve = {s: {e.step: stats.group_counts([dict(x, train_seed=1) for x in e.episodes], coll.tasks)
                 for e in evs if e.start == s} for s in starts}
    common = stats.common_steps(coll, "100", [1])
    latest = common[-1] if common else None
    spatial = spatial_band(coll, starts, evs, latest) if latest else []
    flat = {}
    for s in starts:
        steps = sorted(curve[s])
        if len(steps) < 2:
            continue
        a, b = steps[-2], steps[-1]
        ea, eb = stats.episodes_of(coll, s, "100", a, [1]), stats.episodes_of(coll, s, "100", b, [1])
        row = {"from": a, "to": b}
        for g in ("all", "spatial"):
            keep = {t for t, i in coll.tasks.items() if stats.in_group(i, g)}
            ma = {k: v for k, v in stats.outcome_map(ea).items() if k[0] in keep}
            mb = {k: v for k, v in stats.outcome_map(eb).items() if k[0] in keep}
            ta, tb = stats.task_counts([x for x in ea if x["task"] in keep]), stats.task_counts([x for x in eb if x["task"] in keep])
            diffs = [tb[t][0] / tb[t][1] - ta[t][0] / ta[t][1] for t in sorted(set(ta) & set(tb)) if ta[t][1] and tb[t][1]]
            row[g] = dict(stats.paired(mb, ma), **{"task_" + k: v for k, v in stats.task_level(diffs).items()})
        flat[s] = row
    all_evs = [e for e in coll.evals if e.ratio == "100" and e.seed == 1]
    return {"starts": starts, "steps": sorted({e.step for e in evs}), "curve": curve, "latest_common": latest,
            "spatial": spatial, "flat": flat, "band": BAND, "step_rule": step_rule(coll, starts),
            "evaluations": evaluation_rows(coll, all_evs), "fluctuation": fluctuation(coll, starts, reference)}


RULE_FROM, RULE_TO, RULE_P, RULE_POINTS = 50000, 100000, 0.05, 0.03


def pair(coll, start_a, a, start_b, b, keep=None) -> dict:
    """(start_b, checkpoint b) minus (start_a, checkpoint a), 100% data, seed 1, on the episodes both completed (same
    scenes): pooled (McNemar) and task-level (34 tasks or `keep` as the pairing unit)."""
    ea, eb = stats.episodes_of(coll, start_a, "100", a, [1]), stats.episodes_of(coll, start_b, "100", b, [1])
    if keep is not None:
        ea, eb = [x for x in ea if x["task"] in keep], [x for x in eb if x["task"] in keep]
    ma, mb = stats.outcome_map(ea), stats.outcome_map(eb)
    return dict(stats.paired(mb, ma), **stats.task_level(stats.common_task_diffs(ma, mb)))


def pair_diff(coll, start, a, b, keep=None) -> dict:
    """Checkpoint b minus checkpoint a of one start (100%, seed 1), same scenes: pooled (McNemar) and task-level."""
    return pair(coll, start, a, start, b, keep)


def step_rule(coll, starts) -> dict:
    """The user's rule of 2026-10-22 for stage 1's number of steps: if, for all four starts, the paired difference
    between their 50k and 100k checkpoints is within the noise (p > 0.05 and less than 3 points), 100k steps,
    otherwise 200k.  Applied with the primary statistic (34 tasks as the pairing unit: mean per-task difference and
    its Wilcoxon p); the pooled difference with McNemar's p is shown next to it."""
    rows = {}
    for s in starts:
        have = {e.step for e in coll.first() if (e.start, e.ratio, e.seed) == (s, "100", 1)}
        if {RULE_FROM, RULE_TO} <= have:
            d = pair_diff(coll, s, RULE_FROM, RULE_TO)
            primary = (not math.isnan(d["wilcoxon_p"]) and d["wilcoxon_p"] > RULE_P
                       and abs(d["task_mean_diff"]) < RULE_POINTS) or (d["tasks_up"] + d["tasks_down"] == 0)
            pooled = d["p"] > RULE_P and abs(d["diff"]) < RULE_POINTS
            rows[s] = dict(d, within=primary, pooled_within=pooled)
    complete = len(rows) == 4 and set(rows) == set(starts)
    out = {"from": RULE_FROM, "to": RULE_TO, "p": RULE_P, "points": RULE_POINTS, "starts": rows, "complete": complete}
    if complete:
        out["steps"] = RULE_TO if all(r["within"] for r in rows.values()) else 200000
        out["pooled_steps"] = RULE_TO if all(r["pooled_within"] for r in rows.values()) else 200000
    return out


def stage0_lines(s0) -> list:
    if not s0["curve"]:
        return ["阶段 0 还没有评完的检查点。"]
    lines = []
    for s in s0["starts"]:
        pts_ = []
        for step, c in sorted(s0["curve"][s].items()):
            pts_.append(f"{step // 1000}k {page.pct(c['all'][0] / c['all'][1])}（空间 "
                        f"{page.pct(c['spatial'][0] / c['spatial'][1]) if c['spatial'][1] else '—'}）")
        lines.append(f"{s}：" + "，".join(pts_) + "。")
    short = [r for r in s0.get("evaluations", []) if r["missing"]]
    if short:
        lines.append("缺回合的评测：" + "；".join(
            f"{r['start']} {r['step'] // 1000}k{'（第二遍）' if r['kind'] == 'repeat' else ''}"
            f"{'（未完成）' if r['partial'] else ''} {r['episodes']}/{r['planned']}，缺 {'、'.join(r['missing'])}"
            for r in short) + "。")
    if s0["latest_common"]:
        lo, hi = int(100 * s0["band"][0]), int(100 * s0["band"][1])
        band = [x["task"] for x in s0["spatial"] if x["in_band"]]
        lines.append(f"四个起点都评过的最后一个检查点是 {s0['latest_common'] // 1000}k；9 个空间任务里，四个起点平均成功率在 "
                     f"{lo}%–{hi}% 的有 {len(band)} 个：{'、'.join(band) or '无'}。")
        band_all = [x for x in s0["spatial"] if x["in_band_all"]]
        lines.append(f"改用全部检查点（每个起点每个检查点一个成功率，再平均）：在 {lo}%–{hi}% 的有 {len(band_all)} 个："
                     + ("、".join(f"{x['task']} {page.pct(x['mean_all'])}" for x in band_all) or "无")
                     + ("" if all(x["band_certain"] for x in s0["spatial"]) else
                        "；缺的回合补完后可能改变归类的："
                        + "、".join(x["task"] for x in s0["spatial"] if not x["band_certain"])) + "。")
    fl = (s0.get("fluctuation") or {}).get("summary") or {}
    if fl.get("within_all", {}).get("pairs") and fl.get("between_all", {}).get("pairs"):
        w, b = fl["within_all"], fl["between_all"]
        lines.append(f"同一起点相邻两个检查点（{w['pairs']} 对）之差：合并差绝对值平均 {page.pts(w['pooled_abs_mean']).lstrip('+')}、"
                     f"最大 {page.pts(w['pooled_abs_max']).lstrip('+')} 个百分点，{w['significant']} 对 McNemar p < 0.05；"
                     f"同一检查点起点与 {s0['fluctuation']['between'][0]['reference']} 之差（{b['pairs']} 对）：平均 "
                     f"{page.pts(b['pooled_abs_mean']).lstrip('+')}、最大 {page.pts(b['pooled_abs_max']).lstrip('+')} 个百分点，"
                     f"{b['significant']} 对 p < 0.05。")
    for s, r in s0["flat"].items():
        lines.append(f"{s} 最后两个检查点 {r['from'] // 1000}k → {r['to'] // 1000}k：全部任务 {page.pts(r['all']['diff'])} 个百分点"
                     f"（McNemar p = {page.pval(r['all']['p'])}；逐任务平均 {page.pts(r['all']['task_task_mean_diff'])}，"
                     f"Wilcoxon p = {page.pval(r['all']['task_wilcoxon_p'])}），空间组 {page.pts(r['spatial']['diff'])} 个百分点"
                     f"（McNemar p = {page.pval(r['spatial']['p'])}）。")
    rule = s0.get("step_rule") or {}
    for s, r in rule.get("starts", {}).items():
        lines.append(f"阶段 1 步数规则，{s} 的 50k → 100k（同一批场景）：逐任务平均差 {page.pts(r['task_mean_diff'])} 个百分点，"
                     f"Wilcoxon p = {page.pval(r['wilcoxon_p'])}，{'在' if r['within'] else '不在'}噪声内；"
                     f"合并差 {page.pts(r['diff'])} 个百分点，McNemar p = {page.pval(r['p'])}。")
    if rule.get("complete"):
        lines.append(f"按规则（四个起点都满足 p > 0.05 且差小于 3 个百分点才用 10 万步）：阶段 1 用 {rule['steps'] // 1000}k 步"
                     + ("" if rule["pooled_steps"] == rule["steps"] else
                        f"；改用合并差判断会得到 {rule['pooled_steps'] // 1000}k 步，两种判断不一致") + "。")
    elif rule:
        lines.append(f"阶段 1 步数规则：四个起点的 50k 和 100k 还没都评完（已有 {len(rule.get('starts', {}))} 个起点）。")
    return lines


def build(runs_root: Path, out: Path, reference: str = "B", stage0_mode: bool = False, partial: bool = False) -> dict:
    from . import figures as F

    if stage0_mode:
        coll = collect.collect(runs_root, evals_name=f"evals_{STAGE0_RUN}", run=STAGE0_RUN, partial=partial)
    else:
        coll = collect.collect(runs_root, partial=partial)
    simulated = (Path(runs_root) / "SIMULATED").exists()
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    d5_step = d5_point(coll)
    d5 = stats.compare(coll, "100", d5_step, [1], reference) if d5_step else None
    finals = {} if stage0_mode else {r: stats.compare(coll, r, FINAL_STEP, final_seeds(coll, r), reference)
                                     for r in ("010", "030", "100") if final_seeds(coll, r)}
    seed_tables = {r: stats.seed_table(coll, r, FINAL_STEP) for r in ("010", "030", "100")}
    main = finals.get("100") or d5
    s0 = stage0(coll, reference) if stage0_mode else None
    if stage0_mode:
        main_label = (f"阶段 0：10-22 停下的 D5（100% 数据、训练种子 1），四个起点都评过的第 {d5_step // 1000}k 步"
                      if d5_step else "阶段 0：还没有评完的检查点")
    elif main is not None and main is finals.get("100"):
        main_label = f"第三步，100% 数据，第 {FINAL_STEP // 1000}k 步，训练种子 {','.join(map(str, main['seeds']))}"
    elif main is not None:
        main_label = f"D5，100% 数据，训练种子 1，第 {d5_step // 1000}k 步检查点"
    else:
        main_label = "还没有评测结果"
    lines = stage0_lines(s0) if s0 else []
    if not coll.first():
        lines.append("还没有完成的闭环评测。训练损失见下文；评测完成后重新生成本页即可。")
    if d5:
        lines += describe(d5, f"D5 第 {d5_step // 1000}k 步（100% 数据、训练种子 1）")
        lines.append("这是中间检查点：学习率按 20 万步余弦衰减，此时仍高，结果只用来看成功率何时离开地板，"
                     "不能当作更短训练的结果。")
    for r, cmp in finals.items():
        lines += describe(cmp, f"第三步 {collect.RATIO_PERCENT[r]}% 数据第 {FINAL_STEP // 1000}k 步"
                               f"（训练种子 {','.join(map(str, cmp['seeds']))}）")
    noise = noise_rows(coll)
    floor = stats.noise_floor(coll)
    for x, f in zip(noise, floor):
        lines.append(f"评测噪声下限：{x['exp']} 第 {x['step'] // 1000}k 步用同一批种子评两遍，成功率差 "
                     f"{page.pts(f['pooled_diff'])} 个百分点；逐任务差的平均绝对值 {page.pts(f['task_mean_abs_diff']).lstrip('+')} "
                     f"个百分点、最大 {page.pts(f['task_max_abs_diff']).lstrip('+')}；{f['episodes_changed']}/{f['episodes_compared']} "
                     f"个回合结果不同。起点之间的逐任务平均差小于这个量级时分不出来。")

    figs = {}

    def save(name, fig):
        if fig is not None:
            path = out / "figures" / f"{name}.png"
            fig.savefig(path)
            F.plt.close(fig)
            figs[name] = path
    save("loss", F.loss_curves(coll, reference=reference))
    if stage0_mode:
        save("checkpoints", F.checkpoint_curves(coll, groups=("all", "spatial")))
    else:
        save("checkpoints", F.checkpoint_curves(coll))
        save("learning", F.learning_curves(coll, FINAL_STEP))
    if main:
        save("heatmap", F.task_heatmap(coll, main))
        save("forest", F.forest(coll, main))
        save("agreement", F.agreement(coll, main))
        save("efficiency", F.efficiency(coll, main["ratio"], main["step"], main["seeds"]))
    save("noise", F.noise(coll))
    picked = examples.pick(coll, main) if main else []

    results = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "simulated": simulated, "code": code_version(), "runs_root": str(runs_root),
        "counts": {"experiments": len(coll.experiments), "evaluations": len(coll.first()),
                   "episodes": sum(len(e.episodes) for e in coll.first())},
        "reference": reference, "d5_step": d5_step, "d5": d5, "finals": finals, "final_step": FINAL_STEP,
        "seed_tables": seed_tables, "main_cmp": main, "main_label": main_label,
        "task_order": F.task_order(coll, main) if main else [], "noise": noise, "noise_floor": floor,
        "summary_lines": lines, "stage0": s0,
        "examples_rule": ("范例按固定规则挑选，不凭观感：同一场景（同任务、同评测种子、同训练种子）里起点结果不一致的回合，"
                          "每个起点各取同样多个“它成功、B 失败”和“B 成功、它失败”，空间任务优先，再按任务名和种子排序；"
                          "另加一个全部成功、一个全部失败的场景。每条画面是该回合等间隔的 8 帧（左侧相机），"
                          "下方色条绿为成功、红为失败。"),
        "examples": [{k: v for k, v in x.items() if k != "per_start"} | {"starts": list(x["per_start"])} for x in picked],
        "sources": [f"实验目录：{runs_root}/experiments/DS-<起点>-<数据比例>-s<训练种子>",
                    "评测协议：docs/downstream/EVAL_PROTOCOL.md（950 回合、评测种子、台面区域顺序固定）",
                    "每次评测：evals/steps_<步数>/out/episodes/{summary.json, episodes.jsonl, frames/}",
                    "统计与图：downstream/report/（stats.py、figures.py、examples.py）",
                    f"本页数字的原始值：{out}/results.json"],
    }
    (out / "results.json").write_text(json.dumps(jsonable(results), ensure_ascii=False, indent=1) + "\n")
    html_text = page.render(results, coll, figs, picked)
    (out / "index.html").write_text(html_text, encoding="utf-8")
    md = [f"# 下游起点对照（{results['generated']}）", ""]
    if simulated:
        md += ["> 模拟数据：所有数字都是编出来的，只用来预览。", ""]
    md += [f"- {x}" for x in lines] + [""] + [f"![{k}](figures/{p.name})" for k, p in figs.items()]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"out": str(out), "figures": sorted(figs), "examples": len(picked), "html_bytes": len(html_text.encode()),
            "simulated": simulated, "evaluations": results["counts"]["evaluations"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="B")
    ap.add_argument("--stage0", action="store_true",
                    help="stage 0: the stopped D5 runs (run_stopped_20261022, evals_run_stopped_20261022)")
    ap.add_argument("--partial", action="store_true",
                    help="also count evaluations without a summary yet (their completed episodes; marked on the page)")
    args = ap.parse_args(argv)
    print(json.dumps(build(args.runs_root, args.out, args.reference, stage0_mode=args.stage0, partial=args.partial)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
