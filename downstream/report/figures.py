"""Figures of the result report (matplotlib, PNG).  Labels are in English: the cluster has no CJK font.  One colour
per start everywhere (START_COLOR); intervals are 95% Wilson intervals (rates) or paired 95% intervals
(differences); error bars over training seeds are one sample standard deviation.  Every function returns a Figure,
or None when its data do not exist yet."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import stats  # noqa: E402
from .collect import RATIO_PERCENT  # noqa: E402

START_COLOR = {"B": "#7a8291", "S": "#3569b5", "RS": "#d9822b", "RD": "#2f9a6a", "R10S5": "#8e5bb5", "R1": "#b5475b"}
START_LABEL = {"B": "B  base 7B", "S": "S  action SFT e9", "RS": "RS  P-S0 300", "RD": "RD  P-D 150"}
KLASS_COLOR = {"spatial": "#c2410c", "borderline": "#7c6f2a", "other": "#4b5563"}
INK, MUTED, GRID = "#1f2933", "#66707c", "#e4e7eb"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False, "figure.dpi": 100,
    "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def color(start: str) -> str:
    return START_COLOR.get(start, "#555555")


def label(start: str) -> str:
    return START_LABEL.get(start, start)


def _smooth(values, window: int):
    v = np.asarray(values, dtype=float)
    if len(v) < window or window <= 1:
        return v
    kernel = np.ones(window) / window
    out = np.convolve(v, kernel, mode="same")
    edge = window // 2
    out[:edge], out[-edge:] = np.nan, np.nan
    return out


def loss_curves(coll, ratio: str = "100", seed: int = 1, reference: str = "B", window: int = 21):
    """Action loss (L1) on the training batches, and each start minus the reference on the same batches."""
    series = {s: dict(coll.losses.get(f"DS-{s}-{ratio}-s{seed}", [])) for s in coll.starts()}
    series = {s: v for s, v in series.items() if v}
    if not series:
        return None
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for s, v in series.items():
        steps = sorted(v)
        top.plot(np.array(steps) / 1000, _smooth([v[k] for k in steps], window), color=color(s), lw=1.6, label=label(s))
    top.set_yscale("log")
    top.set_ylabel("action loss, L1 (moving mean of %d logs)" % window)
    top.set_title(f"Training loss, {RATIO_PERCENT.get(ratio, ratio)}% data, seed {seed} (same batches for every start)")
    top.legend(ncol=4, loc="upper right", fontsize=8)
    if reference not in series or len(series) == 1:
        bottom.set_visible(False)
        top.set_xlabel("training step (thousands)")
    else:
        ref = series[reference]
        for s, v in series.items():
            if s == reference:
                continue
            steps = sorted(set(v) & set(ref))
            d = _smooth([v[k] - ref[k] for k in steps], window)
            bottom.plot(np.array(steps) / 1000, d, color=color(s), lw=1.4, label=f"{s} - {reference}")
        bottom.axhline(0, color=INK, lw=0.8)
        bottom.set_ylabel(f"loss minus {reference}\n(same step, same batch)")
        bottom.legend(ncol=3, loc="upper right", fontsize=8)
    bottom.set_xlabel("training step (thousands)")
    fig.tight_layout()
    return fig


def _rate_axes(groups):
    n = len(groups)
    cols = 3 if n > 2 else n
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False, sharex=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def checkpoint_curves(coll, ratio: str = "100", seed: int = 1, groups=stats.GROUPS):
    """Success rate at each evaluated checkpoint of the seed-`seed` runs (D5), per group, with Wilson intervals.
    An evaluation still missing episodes (collect(partial=True)) is drawn with a hollow marker."""
    evs = [e for e in coll.first() if e.ratio == ratio and e.seed == seed]
    if not evs:
        return None
    fig, axes = _rate_axes(groups)
    for g, ax in zip(groups, axes):
        for s in coll.starts():
            pts = sorted((e.step, stats.group_counts([dict(x, train_seed=seed) for x in e.episodes], coll.tasks)[g],
                          e.partial) for e in evs if e.start == s)
            pts = [(st, k, n, part) for st, (k, n), part in pts if n]
            if not pts:
                continue
            x = np.array([p[0] for p in pts]) / 1000
            w = np.array([stats.wilson(k, n) for _, k, n, _ in pts])
            ax.plot(x, 100 * w[:, 0], "-o", color=color(s), ms=3.5, lw=1.4, label=s)
            hollow = [i for i, p in enumerate(pts) if p[3]]
            if hollow:
                ax.plot(x[hollow], 100 * w[hollow, 0], "o", color=color(s), mfc="white", ms=5, lw=0)
            ax.fill_between(x, 100 * w[:, 1], 100 * w[:, 2], color=color(s), alpha=0.12, lw=0)
        ax.set_title(stats.GROUP_LABEL[g])
        ax.set_ylim(bottom=0)
        steps = sorted({e.step for e in evs})
        if len(steps) <= 8:
            ax.set_xticks([s / 1000 for s in steps], [f"{s / 1000:g}k" for s in steps])
    axes[0].set_ylabel("success rate (%)")
    for ax in axes[-3:]:
        ax.set_xlabel("checkpoint (thousand steps)")
    axes[0].legend(fontsize=8, ncol=len(coll.starts()), loc="lower center", handlelength=1.2, columnspacing=0.9)
    hollow_note =" Hollow: evaluation still missing episodes." if any(e.partial for e in evs) else ""
    fig.suptitle(f"Success by checkpoint, {RATIO_PERCENT.get(ratio, ratio)}% data, seed {seed} (95% Wilson bands).\n"
                 f"Learning rate still high before 200k: not a shorter run.{hollow_note}", fontsize=9, color=MUTED)
    fig.tight_layout()
    return fig


def learning_curves(coll, step: int = 200000, groups=stats.GROUPS):
    """Mean success over training seeds against the share of demonstrations, per group; bars = 1 sample std."""
    table = {r: stats.seed_table(coll, r, step) for r in ("010", "030", "100")}
    if not any(table.values()):
        return None
    fig, axes = _rate_axes(groups)
    starts = coll.starts()
    for g, ax in zip(groups, axes):
        for i, s in enumerate(starts):
            xs, ms, sds = [], [], []
            for r in ("010", "030", "100"):
                if s in table[r]:
                    mean, sd, n = table[r][s][g]
                    if n:
                        xs.append(RATIO_PERCENT[r] * (1 + 0.04 * (i - (len(starts) - 1) / 2)))
                        ms.append(100 * mean)
                        sds.append(100 * sd if not math.isnan(sd) else 0.0)
            if xs:
                ax.errorbar(xs, ms, yerr=sds, color=color(s), marker="o", ms=4, lw=1.5, capsize=3, label=s)
        ax.set_xscale("log")
        ax.set_xticks([10, 30, 100], ["10%", "30%", "100%"])
        ax.minorticks_off()
        ax.set_title(stats.GROUP_LABEL[g])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("success rate (%), mean over seeds")
    for ax in axes[-3:]:
        ax.set_xlabel("share of target demonstrations")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Sample efficiency at step {step // 1000}k (error bars: 1 std over training seeds)",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    return fig


def task_order(coll, cmp) -> list:
    """Spatial, then borderline, then the other tasks; within each, by the reference start's success rate."""
    ref = cmp["tasks"].get(cmp["reference"], {})
    key = {"spatial": 0, "borderline": 1, "other": 2}

    def rate(t):
        k, n = ref.get(t, (0, 0))
        return k / n if n else -1
    return sorted([t for t in coll.tasks if any(t in v for v in cmp["tasks"].values())],
                  key=lambda t: (key[coll.tasks[t].klass], -rate(t), t))


def task_heatmap(coll, cmp):
    """Per-task success rate of every start (rows: tasks; cell text: successes / episodes)."""
    if not cmp["starts"]:
        return None
    tasks, starts = task_order(coll, cmp), cmp["starts"]
    grid = np.full((len(tasks), len(starts)), np.nan)
    for j, s in enumerate(starts):
        for i, t in enumerate(tasks):
            k, n = cmp["tasks"][s].get(t, (0, 0))
            if n:
                grid[i, j] = k / n
    fig, ax = plt.subplots(figsize=(1.25 * len(starts) + 3.2, 0.26 * len(tasks) + 1.4))
    ax.grid(False)
    im = ax.imshow(100 * grid, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for i, t in enumerate(tasks):
        for j, s in enumerate(starts):
            k, n = cmp["tasks"][s].get(t, (0, 0))
            if n:
                ax.text(j, i, f"{k}/{n}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(starts)), starts)
    ax.set_yticks(range(len(tasks)), tasks, fontsize=7.5)
    for tick, t in zip(ax.get_yticklabels(), tasks):
        tick.set_color(KLASS_COLOR[coll.tasks[t].klass])
    bounds = [i for i in range(1, len(tasks)) if coll.tasks[tasks[i]].klass != coll.tasks[tasks[i - 1]].klass]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=3)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="success rate (%)")
    ax.set_title(f"Per task, {RATIO_PERCENT.get(cmp['ratio'], cmp['ratio'])}% data, step {cmp['step'] // 1000}k, "
                 f"seeds {','.join(map(str, cmp['seeds']))}\n(task names: orange spatial, olive borderline, grey other)",
                 fontsize=9)
    fig.tight_layout()
    return fig


def forest(coll, cmp):
    """Per-task paired difference to the reference with its 95% interval; filled marker = Holm-adjusted p < 0.05."""
    others = [s for s in cmp["starts"] if s in cmp["vs_reference"]]
    if not others:
        return None
    tasks = task_order(coll, cmp)
    fig, axes = plt.subplots(1, len(others), figsize=(2.9 * len(others) + 1.8, 0.24 * len(tasks) + 1.5),
                             sharey=True, squeeze=False)
    y = np.arange(len(tasks))
    for ax, s in zip(axes[0], others):
        rows = cmp["vs_reference"][s]["tasks"]
        for i, t in enumerate(tasks):
            r = rows.get(t)
            if not r or math.isnan(r["diff"]):
                continue
            c = KLASS_COLOR[coll.tasks[t].klass]
            ax.plot([100 * r["low"], 100 * r["high"]], [i, i], color=c, lw=1.2)
            filled = r.get("p_holm", 1.0) < 0.05
            ax.plot(100 * r["diff"], i, "o", ms=4.5, color=c, mfc=c if filled else "white")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(f"{s} minus {cmp['reference']}")
        ax.set_xlabel("success difference (points)")
        ax.set_ylim(len(tasks) - 0.5, -0.5)
    axes[0][0].set_yticks(y, tasks, fontsize=7.5)
    for tick, t in zip(axes[0][0].get_yticklabels(), tasks):
        tick.set_color(KLASS_COLOR[coll.tasks[t].klass])
    fig.suptitle("Paired per-task differences (same scenes): 95% interval; filled = Holm-adjusted McNemar p < 0.05",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    return fig


def agreement(coll, cmp, groups=("all", "spatial", "other")):
    """How the same episodes ended for a start and the reference."""
    others = [s for s in cmp["starts"] if s in cmp["vs_reference"]]
    if not others:
        return None
    fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups), 0.55 * len(others) + 1.6), sharey=True,
                             squeeze=False)
    parts = (("both", "#9fb7c9", "both succeed"), ("only_a", "#2f9a6a", "only this start"),
             ("only_b", "#c2410c", f"only {cmp['reference']}"), ("neither", "#eef0f3", "both fail"))
    for ax, g in zip(axes[0], groups):
        for i, s in enumerate(others):
            r = cmp["vs_reference"][s]["groups"][g]
            left = 0.0
            for key, c, name in parts:
                w = 100 * r[key] / r["n"] if r["n"] else 0
                ax.barh(i, w, left=left, color=c, edgecolor="white", label=name if i == 0 else None)
                if w >= 6 and key != "neither":
                    ax.text(left + w / 2, i, str(r[key]), ha="center", va="center", fontsize=7, color=INK)
                left += w
        ax.set_yticks(range(len(others)), others)
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of paired episodes (%)")
        ax.set_title(stats.GROUP_LABEL[g])
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def efficiency(coll, ratio: str, step: int, seeds, groups=("all", "spatial")):
    """Among successful episodes: control steps used as a share of the task's time limit."""
    data = {}
    for s in coll.starts():
        eps = stats.episodes_of(coll, s, ratio, step, seeds)
        for g in groups:
            vals = [x["steps"] / x["max_steps"] for x in eps if x["success"] and x["task"] in coll.tasks
                    and stats.in_group(coll.tasks[x["task"]], g)]
            if vals:
                data.setdefault(g, {})[s] = vals
    if not data:
        return None
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 2.9), squeeze=False, sharey=True)
    for ax, g in zip(axes[0], groups):
        starts = list(data.get(g, {}))
        if not starts:
            ax.set_visible(False)
            continue
        bp = ax.boxplot([data[g][s] for s in starts], tick_labels=starts, widths=0.55, patch_artist=True,
                        medianprops={"color": INK}, flierprops={"ms": 2, "alpha": 0.4})
        for patch, s in zip(bp["boxes"], starts):
            patch.set_facecolor(color(s))
            patch.set_alpha(0.55)
        ax.set_title(f"{stats.GROUP_LABEL[g]} (successes: " + ", ".join(str(len(data[g][s])) for s in starts) + ")",
                     fontsize=8.5)
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("steps used / time limit")
    fig.tight_layout()
    return fig


def noise(coll):
    """The same checkpoint evaluated twice: per-task success of the first job against the repeat at each P, and the
    mean time per episode."""
    reps = [e for e in coll.evals if e.kind == "repeat"]
    if not reps:
        return None
    first = {(e.exp, e.step): e for e in coll.first()}
    pairs = [(first[(r.exp, r.step)], r) for r in reps if (r.exp, r.step) in first]
    if not pairs:
        return None
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    marks = ["o", "s", "^", "D"]
    top = 0.0
    for i, (a, r) in enumerate(pairs):
        keys = {(x["task"], x["eval_seed"]) for x in a.episodes} & {(x["task"], x["eval_seed"]) for x in r.episodes}
        ta, tr = stats.task_counts([x for x in a.episodes if (x["task"], x["eval_seed"]) in keys]), \
            stats.task_counts([x for x in r.episodes if (x["task"], x["eval_seed"]) in keys])
        common = sorted(set(ta) & set(tr))
        xs, ys = [100 * ta[t][0] / ta[t][1] for t in common], [100 * tr[t][0] / tr[t][1] for t in common]
        top = max([top] + xs + ys)
        ax.scatter(xs, ys, s=16, marker=marks[i % 4], alpha=0.75, label=f"repeat P={r.workers}")
    lim = min(100.0, top + 5)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8)
    ax.set_xlim(-2, lim)
    ax.set_ylim(-2, lim)
    ax.set_xlabel(f"first job, P={pairs[0][0].workers} (% per task)")
    ax.set_ylabel("repeat (% per task)")
    ax.set_title(f"{pairs[0][0].exp} step {pairs[0][0].step // 1000}k, same seeds")
    ax.legend(fontsize=8)
    names = [f"P={pairs[0][0].workers}\nfirst"] + [f"P={r.workers}\nrepeat" for _, r in pairs]
    vals = [pairs[0][0].summary.get("episode_wall_mean_seconds") or 0] + \
        [r.summary.get("episode_wall_mean_seconds") or 0 for _, r in pairs]
    bx.bar(names, vals, color=["#7a8291"] + ["#3569b5"] * len(pairs))
    for i, v in enumerate(vals):
        bx.text(i, v, f"{v:.0f} s\n{v / vals[0]:.2f}x" if vals[0] else f"{v:.0f} s", ha="center", va="bottom", fontsize=8)
    bx.set_ylabel("mean wall time per episode (s)")
    bx.set_ylim(0, max(vals) * 1.35)
    bx.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
