"""Statistics of the result report.  Every comparison between starts is paired: the same task, the same evaluation
seed (hence the same initial scene, EVAL_PROTOCOL.md) and the same training seed.

  wilson            95% Wilson interval of a success rate
  paired            two starts on the same episodes: both / only A / only B / neither, the difference in success
                    rate with a 95% interval, and McNemar's exact test (two-sided binomial on the discordant pairs)
  holm              Holm's step-down adjustment of the per-task p values (family = the tasks of one comparison)
  wilcoxon_tasks    Wilcoxon signed-rank test of the per-task differences (do the tasks lean one way?)
  seed_spread       mean and sample standard deviation over training seeds
"""
from __future__ import annotations

import math
from collections import defaultdict

GROUPS = ("all", "atomic_seen", "composite_seen", "spatial", "borderline", "other")
GROUP_LABEL = {"all": "All 34 tasks", "atomic_seen": "Atomic (18)", "composite_seen": "Composite (16)",
               "spatial": "Spatial (9)", "borderline": "Borderline (3)", "other": "Other (22)"}
GROUP_ZH = {"all": "全部 34 个任务", "atomic_seen": "单步 18 个", "composite_seen": "多步 16 个",
            "spatial": "空间组 9 个", "borderline": "边界 3 个", "other": "其余 22 个"}


def in_group(info, group: str) -> bool:
    return group == "all" or info.group == group or info.klass == group


def wilson(k: int, n: int, z: float = 1.96):
    """(rate, low, high); (nan, nan, nan) when n == 0."""
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided p of k or fewer of n fair coin flips landing one way (k <= n / 2)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    return binom_two_sided(min(only_a, only_b), only_a + only_b)


def paired(a: dict, b: dict) -> dict:
    """a, b: episode key -> success.  Only keys in both count."""
    keys = sorted(set(a) & set(b))
    both = sum(a[k] and b[k] for k in keys)
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    n = len(keys)
    out = {"n": n, "both": both, "only_a": only_a, "only_b": only_b, "neither": n - both - only_a - only_b,
           "rate_a": (both + only_a) / n if n else math.nan, "rate_b": (both + only_b) / n if n else math.nan}
    if n:
        d = (only_a - only_b) / n
        se = math.sqrt(max(0.0, only_a + only_b - (only_a - only_b) ** 2 / n)) / n
        out.update(diff=d, low=max(-1.0, d - 1.96 * se), high=min(1.0, d + 1.96 * se),
                   p=mcnemar_exact(only_a, only_b))
    else:
        out.update(diff=math.nan, low=math.nan, high=math.nan, p=math.nan)
    return out


def holm(pvalues: list) -> list:
    """Holm-adjusted p values, same order; nan stays nan."""
    idx = sorted((i for i, p in enumerate(pvalues) if not math.isnan(p)), key=lambda i: pvalues[i])
    out, running, m = [math.nan] * len(pvalues), 0.0, len(idx)
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


def wilcoxon_tasks(diffs: list):
    """(statistic, p) of the Wilcoxon signed-rank test over per-task differences; (nan, 1.0) without any nonzero."""
    nonzero = [d for d in diffs if d and not math.isnan(d)]
    if not nonzero:
        return (math.nan, 1.0)
    try:
        from scipy.stats import wilcoxon

        res = wilcoxon(nonzero)
        return (float(res.statistic), float(res.pvalue))
    except Exception:
        return (math.nan, math.nan)


def seed_spread(values: list):
    """(mean, sample std, n); std is nan for one value."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return (math.nan, math.nan, 0)
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else math.nan
    return (mean, std, len(vals))


# ------------------------------------------------------------------------------------------- on a Collection
def episodes_of(coll, start: str, ratio: str, step: int, seeds=None, kind: str = "first") -> list:
    """Episodes of the evaluations of one start at one (ratio, step), each with its training seed."""
    out = []
    for e in coll.evals:
        if e.kind == kind and (e.start, e.ratio, e.step) == (start, ratio, step) and (seeds is None or e.seed in seeds):
            out += [dict(x, train_seed=e.seed) for x in e.episodes]
    return out


def outcome_map(episodes: list) -> dict:
    return {(x["task"], x["eval_seed"], x["train_seed"]): bool(x["success"]) for x in episodes}


def group_counts(episodes: list, tasks: dict) -> dict:
    """group -> [successes, episodes]."""
    out = {g: [0, 0] for g in GROUPS}
    for x in episodes:
        info = tasks.get(x["task"])
        if info is None:
            continue
        for g in GROUPS:
            if in_group(info, g):
                out[g][0] += int(x["success"])
                out[g][1] += 1
    return out


def task_counts(episodes: list) -> dict:
    out = defaultdict(lambda: [0, 0])
    for x in episodes:
        out[x["task"]][0] += int(x["success"])
        out[x["task"]][1] += 1
    return dict(out)


def common_steps(coll, ratio: str, seeds) -> list:
    """Checkpoint steps at which every start present at this ratio has a first evaluation for every seed."""
    have = defaultdict(set)
    starts = set()
    for e in coll.first():
        if e.ratio == ratio and e.seed in seeds:
            have[e.step].add((e.start, e.seed))
            starts.add(e.start)
    return sorted(s for s, got in have.items() if starts and got >= {(a, k) for a in starts for k in seeds})


def compare(coll, ratio: str, step: int, seeds, reference: str = "B") -> dict:
    """Every start against `reference` at one (ratio, step), on the episodes both have (paired by task, evaluation
    seed and training seed): per group and per task, with Holm-adjusted per-task p values and the Wilcoxon test over
    the per-task differences."""
    starts = [s for s in coll.starts() if episodes_of(coll, s, ratio, step, seeds)]
    eps = {s: episodes_of(coll, s, ratio, step, seeds) for s in starts}
    out = {"ratio": ratio, "step": step, "seeds": sorted(seeds), "reference": reference, "starts": starts,
           "groups": {s: group_counts(eps[s], coll.tasks) for s in starts},
           "tasks": {s: task_counts(eps[s]) for s in starts}, "vs_reference": {}}
    if reference not in eps:
        return out
    ref = outcome_map(eps[reference])
    for s in starts:
        if s == reference:
            continue
        mine = outcome_map(eps[s])
        per_task = {}
        for task in sorted(coll.tasks):
            a = {k: v for k, v in mine.items() if k[0] == task}
            b = {k: v for k, v in ref.items() if k[0] == task}
            if set(a) & set(b):
                per_task[task] = paired(a, b)
        adj = holm([r["p"] for r in per_task.values()])
        for r, q in zip(per_task.values(), adj):
            r["p_holm"] = q
        groups = {}
        for g in GROUPS:
            keep = {t for t, info in coll.tasks.items() if in_group(info, g)}
            groups[g] = paired({k: v for k, v in mine.items() if k[0] in keep},
                               {k: v for k, v in ref.items() if k[0] in keep})
            groups[g].update(task_level([r["diff"] for t, r in per_task.items() if t in keep]))
        out["vs_reference"][s] = {"tasks": per_task, "groups": groups}
    return out


def compare_mean(coll, ratio: str, steps, seeds, reference: str = "B") -> dict:
    """Every start against `reference` on the mean over several checkpoints (the user's decision 2 of 2026-10-23
    evening: E1 is evaluated at 150k, 175k and 200k and its main numbers are the mean of the three).  Only starts with
    an evaluation at every step count.
      groups[start][group]   {"rates": the success rate at each step, "mean": their mean}
      tasks[start][task]     the mean over the steps of the task's success rate
      vs_reference[start][group]   per task the mean over the steps of the paired difference (start minus
                                   reference, episodes both have), then the task-level statistics over the tasks
                                   (mean, bootstrap interval, Wilcoxon); plus the pooled difference at each step
                                   (McNemar p, for reference) and their mean"""
    steps = list(steps)
    starts = [s for s in coll.starts() if all(episodes_of(coll, s, ratio, st, seeds) for st in steps)]
    eps = {s: {st: episodes_of(coll, s, ratio, st, seeds) for st in steps} for s in starts}
    out = {"ratio": ratio, "steps": steps, "seeds": sorted(seeds), "reference": reference, "starts": starts,
           "groups": {}, "tasks": {}, "vs_reference": {}}
    for s in starts:
        out["groups"][s] = {}
        for g in GROUPS:
            rates = []
            for st in steps:
                k, n = group_counts(eps[s][st], coll.tasks)[g]
                rates.append(k / n if n else math.nan)
            known = [r for r in rates if not math.isnan(r)]
            out["groups"][s][g] = {"rates": rates, "mean": sum(known) / len(known) if len(known) == len(rates) else math.nan}
        counts = [task_counts(eps[s][st]) for st in steps]
        out["tasks"][s] = {t: sum(c[t][0] / c[t][1] for c in counts) / len(counts)
                           for t in coll.tasks if all(t in c and c[t][1] for c in counts)}
    if reference not in starts:
        return out
    ref = {st: outcome_map(eps[reference][st]) for st in steps}
    for s in starts:
        if s == reference:
            continue
        mine = {st: outcome_map(eps[s][st]) for st in steps}
        groups = {}
        for g in GROUPS:
            keep = {t for t, info in coll.tasks.items() if in_group(info, g)}
            diffs = []
            for t in sorted(keep):
                per_step = []
                for st in steps:
                    d = common_task_diffs({k: v for k, v in ref[st].items() if k[0] == t},
                                          {k: v for k, v in mine[st].items() if k[0] == t})
                    if d:
                        per_step.append(d[0])
                if len(per_step) == len(steps):
                    diffs.append(sum(per_step) / len(per_step))
            pooled = [paired({k: v for k, v in mine[st].items() if k[0] in keep},
                             {k: v for k, v in ref[st].items() if k[0] in keep}) for st in steps]
            groups[g] = dict(task_level(diffs), pooled_by_step=[{"step": st, "diff": p["diff"], "p": p["p"], "n": p["n"]}
                                                                for st, p in zip(steps, pooled)],
                             pooled_mean_diff=sum(p["diff"] for p in pooled) / len(pooled))
        out["vs_reference"][s] = {"groups": groups}
    return out


def task_level(diffs: list, resamples: int = 2000, seed: int = 0) -> dict:
    """The tasks as the pairing unit (the user's choice of 2026-10-22): one success-rate difference per task.
    Mean difference over tasks with a 95% bootstrap interval (resampling tasks), the sign test (tasks up vs down)
    and the Wilcoxon signed-rank test."""
    import random

    d = [x for x in diffs if not math.isnan(x)]
    up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
    stat, p = wilcoxon_tasks(d)
    out = {"tasks": len(d), "tasks_up": up, "tasks_down": down, "task_mean_diff": sum(d) / len(d) if d else math.nan,
           "sign_p": binom_two_sided(min(up, down), up + down), "wilcoxon_stat": stat, "wilcoxon_p": p,
           "task_low": math.nan, "task_high": math.nan}
    if d:
        rng = random.Random(seed)
        means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(resamples))
        out["task_low"], out["task_high"] = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    return out


def common_task_diffs(a: dict, b: dict) -> list:
    """a, b: episode key (task, ...) -> success.  Per task, b's success rate minus a's over the episodes both have
    (an episode one side lacks, e.g. one whose process died, drops out of both), in task order."""
    per = defaultdict(lambda: [0, 0, 0])
    for k in set(a) & set(b):
        per[k[0]][0] += int(a[k])
        per[k[0]][1] += int(b[k])
        per[k[0]][2] += 1
    return [(sb - sa) / n for _, (sa, sb, n) in sorted(per.items())]


def noise_floor(coll) -> list:
    """Two evaluations of one checkpoint with the same seeds (first and repeat): the differences between them are
    what the evaluation alone produces.  Per pair, on the episodes both completed: the pooled success difference,
    the per-task differences (mean of their absolute values, largest), and the episodes whose outcome differs."""
    first = {(e.exp, e.step): e for e in coll.first()}
    out = []
    for r in (e for e in coll.evals if e.kind == "repeat"):
        a = first.get((r.exp, r.step))
        if a is None:
            continue
        oa = {(x["task"], x["eval_seed"]): x["success"] for x in a.episodes}
        ob = {(x["task"], x["eval_seed"]): x["success"] for x in r.episodes}
        common = set(oa) & set(ob)
        diffs = common_task_diffs(oa, ob)
        out.append({"exp": a.exp, "step": a.step,
                    "pooled_diff": (sum(ob[k] for k in common) - sum(oa[k] for k in common)) / max(1, len(common)),
                    "task_mean_abs_diff": sum(abs(x) for x in diffs) / len(diffs) if diffs else math.nan,
                    "task_max_abs_diff": max((abs(x) for x in diffs), default=math.nan),
                    "episodes_changed": sum(oa[k] != ob[k] for k in common), "episodes_compared": len(common),
                    "task_level": task_level(diffs)})
    return out


def seed_table(coll, ratio: str, step: int) -> dict:
    """start -> group -> (mean, std, n seeds) of the per-seed success rates."""
    out = {}
    for s in coll.starts():
        per_seed = defaultdict(dict)
        for e in coll.first():
            if (e.start, e.ratio, e.step) == (s, ratio, step):
                for g, (k, n) in group_counts([dict(x, train_seed=e.seed) for x in e.episodes], coll.tasks).items():
                    per_seed[g][e.seed] = k / n if n else math.nan
        if per_seed:
            out[s] = {g: seed_spread(list(v.values())) for g, v in per_seed.items()}
    return out
