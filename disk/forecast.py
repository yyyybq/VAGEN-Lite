#!/usr/bin/env python3
"""Seven-day forecast of the user folder's free space from a list of planned writes (no cluster access needed).

    python disk/forecast.py PLAN.json --out DIR

PLAN.json:
  {"start": "2026-10-26T07:00Z", "days": 7, "quota_bytes": ..., "used_bytes": ...,
   "used_low_bytes": ..., "used_high_bytes": ...,          # optional band around used (the quota table rounds)
   "scenarios": {"key": {"include": ["tag", ...], "label": "..."}, ...},  # optional: tasks each scenario counts
   "tasks": [{"name": ..., "line": ..., "tags": [...],
              "ramp": [["T0", "T1", bytes], ...],          # written evenly between T0 and T1 (negative = freed)
              "step": [["T", bytes], ...]}, ...]}         # written or freed at once at T
Times are ISO UTC ("2026-10-26T07:00Z"). A task without tags is in every scenario.

Writes DIR/forecast_<scenario>.tsv (hourly: time, used, remaining, per-line cumulative), DIR/daily.md (one row per
day per scenario: new writes, automatic deletions, net, remaining at the end of the day, lowest remaining),
DIR/forecast.png (remaining space against time, one line per scenario, the 3 TB and 1.5 TB lines and zero) and
DIR/forecast.json (when each scenario first crosses 3 TB, 1.5 TB and 0).
"""
import argparse
import collections
import datetime as dt
import json
import os

H = dt.timedelta(hours=1)


def t(s):
    return dt.datetime.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc)


def series(plan, include):
    start = t(plan["start"])
    hours = int(plan.get("days", 7) * 24)
    grid = [start + i * H for i in range(hours + 1)]
    written = collections.defaultdict(lambda: [0.0] * len(grid))  # per line, cumulative positive bytes
    freed = collections.defaultdict(lambda: [0.0] * len(grid))
    for task in plan["tasks"]:
        tags = task.get("tags")
        if include is not None and tags and not set(tags) & set(include):
            continue
        line = task.get("line", "?")
        for t0, t1, b in task.get("ramp", []):
            a, z = t(t0), t(t1)
            for i, g in enumerate(grid):
                frac = 0.0 if g <= a else 1.0 if g >= z else (g - a) / (z - a)
                (written if b > 0 else freed)[line][i] += abs(b) * frac
        for ts, b in task.get("step", []):
            a = t(ts)
            for i, g in enumerate(grid):
                if g >= a:
                    (written if b > 0 else freed)[line][i] += abs(b)
    return grid, written, freed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    os.makedirs(args.out, exist_ok=True)
    quota = plan["quota_bytes"]
    used0 = plan["used_bytes"]
    scen = plan.get("scenarios") or {"plan": {"include": None}}
    out_json, daily_rows, curves = {}, [], {}
    for name, sc in scen.items():
        grid, written, freed = series(plan, sc.get("include"))
        lines = sorted(set(written) | set(freed))
        used = [used0 + sum(written[l][i] for l in lines) - sum(freed[l][i] for l in lines) for i in range(len(grid))]
        rem = [quota - u for u in used]
        curves[sc.get("label", name)] = (grid, rem)
        with open(os.path.join(args.out, "forecast_%s.tsv" % name), "w") as f:
            f.write("utc\tused_TB\tremaining_TB\t" + "\t".join("net_%s_TB" % l for l in lines) + "\n")
            for i, g in enumerate(grid):
                f.write("%s\t%.3f\t%.3f\t%s\n" % (g.strftime("%Y-%m-%dT%H:%MZ"), used[i] / 1e12, rem[i] / 1e12,
                                                 "\t".join("%.3f" % ((written[l][i] - freed[l][i]) / 1e12) for l in lines)))
        cross = {}
        for lim in (3e12, 1.5e12, 0.0):
            hit = next((g for g, r in zip(grid, rem) if r < lim), None)
            cross["below_%.1f_TB" % (lim / 1e12)] = hit.strftime("%Y-%m-%dT%H:%MZ") if hit else None
        lo = min(range(len(rem)), key=lambda i: rem[i])
        out_json[name] = {"crossings": cross, "lowest_remaining_TB": round(rem[lo] / 1e12, 3),
                          "lowest_at": grid[lo].strftime("%Y-%m-%dT%H:%MZ"), "end_remaining_TB": round(rem[-1] / 1e12, 3),
                          "written_TB_by_line": {l: round(written[l][-1] / 1e12, 3) for l in lines},
                          "freed_TB_by_line": {l: round(freed[l][-1] / 1e12, 3) for l in lines}}
        for d in range(int(plan.get("days", 7))):
            i0, i1 = d * 24, min((d + 1) * 24, len(grid) - 1)
            w = sum(written[l][i1] - written[l][i0] for l in lines)
            fr = sum(freed[l][i1] - freed[l][i0] for l in lines)
            daily_rows.append((sc.get("label", name), grid[i0].strftime("%m-%d %H:%M"), grid[i1].strftime("%m-%d %H:%M"), w, fr, w - fr,
                               rem[i1], min(rem[i0:i1 + 1])))
    with open(os.path.join(args.out, "forecast.json"), "w") as f:
        json.dump(out_json, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "daily.md"), "w") as f:
        f.write("| 情形 | 从（UTC） | 到（UTC） | 新写入（TB） | 自动删除（TB） | 净增（TB） | 当天结束时剩余（TB） | 当天最低剩余（TB） |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in daily_rows:
            f.write("| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f |\n" % (r[0], r[1], r[2], r[3] / 1e12, r[4] / 1e12,
                                                                        r[5] / 1e12, r[6] / 1e12, r[7] / 1e12))
    try:
        plot(curves, plan, os.path.join(args.out, "forecast.png"))
    except ImportError as e:
        print("no plot:", e)
    print(json.dumps(out_json, indent=1, ensure_ascii=False))


def plot(curves, plan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # reference palette (dataviz skill): categorical slots in fixed order; text in ink tokens, not series colours
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ink, ink2, grid_c, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for k, (name, (grid, rem)) in enumerate(curves.items()):
        ys = [r / 1e12 for r in rem]
        ax.plot(grid, ys, color=colors[k % len(colors)], lw=2, solid_capstyle="round", label=name)
        ax.annotate("%s  %.1f TB" % (name, ys[-1]), (grid[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=ink)
    for lim, lab in ((3.0, "3 TB warning line"), (1.5, "1.5 TB stop-submitting line"), (0.0, "quota full")):
        ax.axhline(lim, color=ink2, lw=1, ls=(0, (4, 3)))
        ax.annotate(lab, (curves[next(iter(curves))][0][len(curves[next(iter(curves))][0]) // 2], lim), xytext=(0, 3), textcoords="offset points", fontsize=8, color=ink2)
    ax.set_ylabel("free space (TB) = quota - used", color=ink2, fontsize=9)
    ax.grid(axis="y", color=grid_c, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid_c)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_title(plan.get("title", "Free space of /users/yinbaiqiao, next 7 days"), fontsize=10, color=ink, loc="left")
    if len(curves) > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


if __name__ == "__main__":
    main()
