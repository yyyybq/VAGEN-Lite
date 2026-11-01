"""Reward line (the user's item 2 of 2026-11-01): the drift tables from reward/drift_s0.py's output.

1. Every run: its arm (the chain plan's rl settings), start, seed, how far it trained, the stop and its reason, the
   highest 10-update share of unreadable answers, and the onsets (unreadable 1% / 2% / 5%; entropy 1.5x / 2x its
   updates 1-50 mean; and each series' first 3-sigma rise over its updates 1-50) -- which moved first.
2. For the named runs (--detail), the windows around their onset (updates 1-50, the 25 before, the 25 from it):
   unreadable share, entropy, answer length and actions per answer, unreadable forms, the family / turn-position /
   outcome rates; and the same updates of the comparison runs (--compare).

    python -m reward.drift_mech --drift OUT.json --args ARGS --detail rw_s0,q3s0,rw_s2s0 --compare rw_o,rw_g,rw_ok,rw_s2ok
        --out FILE.json

ARGS: the drift_s0 --run arguments (NAME=RUN_DIR, the chain folder is RUN_DIR's parent).  FILE.json and FILE.md.
"""

import argparse
import json
import statistics
from pathlib import Path

from reward.drift_s0 import window


def arm(rl):
    std = (rl.get("standard") or "").replace("v10_p1_", "")
    pv = rl.get("progress_version")
    name = {"S0": "S0", "Dcond": "D_cond", "D": "D"}.get(std, std)
    if pv == "optimal":
        name = "O"
    if rl.get("group_filter") == "keep_by_return_std":
        name += f" + keep {rl.get('keep_std_threshold')}"
    scale = rl.get("progress_reward_scale")
    if scale and name != "S0" and not name.startswith("S0"):
        name += f" ×{round(float(scale), 2)}"
    return name


def first_sigma(ups, key, k=3.0):
    base = [s[key] for s in ups if s["update"] <= 50 and s.get(key) is not None]
    if len(base) < 10:
        return None
    mu, sd = statistics.mean(base), statistics.pstdev(base) or 1e-9
    roll = []
    for s in ups:
        roll.append(s.get(key))
        w = [x for x in roll[-10:] if x is not None]
        if s["update"] > 50 and w and statistics.mean(w) > mu + k * sd:
            return s["update"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", required=True)
    ap.add_argument("--args", required=True)
    ap.add_argument("--detail", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.loads(Path(args.drift).read_text())
    per, summ = d["per_update"], d["summary"]
    for name, ups in per.items():
        for s in ups:
            s["rate"] = s["unreadable"] / s["answers"] if s["answers"] else None
    dirs = dict(x.split("=", 1) for x in Path(args.args).read_text().split() if "=" in x)
    rows = []
    for name, run_dir in dirs.items():
        chain = Path(run_dir).parent
        plan = json.loads((chain / "plan.json").read_text())
        st = json.loads((chain / "progress.json").read_text())
        rl = dict(plan.get("rl") or {}, **(((plan.get("rl_runs") or {}).get(name) or {}).get("rl") or {}))
        stop = ((st.get("guard_stops") or {}).get(name) or {}).get("reason")
        start = plan.get("rl_start_model") or ""
        start = "SFT e8" if "lf_q3formal/model/checkpoint-1232" in start else ("c3 SFT" if "q3c3sft" in start else
                                                                                ("weak (fmt)" if "q3fmt" in start else start[-40:]))
        s = summ.get(name, {})
        ups = per.get(name, [])
        rows.append({"run": name, "line": "本线" if "reward_runs" in run_dir else "主线", "chain": chain.name,
                     "arm": arm(rl), "seed": rl.get("seed"), "start": start, "updates": s.get("updates"),
                     "stop": stop, "max_unreadable_10": s.get("max_unreadable_10"),
                     "unreadable_1pct_at": s.get("unreadable_1pct_at"), "unreadable_2pct_at": s.get("unreadable_2pct_at"),
                     "unreadable_5pct_at": s.get("unreadable_5pct_at"), "entropy_1.5x_at": s.get("entropy_1.5x_at"),
                     "entropy_2.0x_at": s.get("entropy_2.0x_at"), "unreadable_3sigma_at": first_sigma(ups, "rate"),
                     "entropy_3sigma_at": first_sigma(ups, "entropy"), "check": s.get("check_invalid_share")})
    detail = [x for x in args.detail.split(",") if x]
    compare = [x for x in args.compare.split(",") if x]
    windows = {}
    for name in detail:
        on = summ[name].get("unreadable_2pct_at") or summ[name].get("unreadable_1pct_at")
        if not on:
            continue
        spans = [(1, 50), (max(1, on - 25), on - 1), (on, on + 24)]
        windows[name] = {"onset": on, "spans": {f"{lo}-{hi}": {n: window(per[n], lo, hi) for n in [name] + compare if n in per}
                                                 for lo, hi in spans}}
    out = {"runs": rows, "windows": windows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    f = lambda x: "—" if x is None else str(x)  # noqa: E731
    L = ["| 运行 | 线 | 臂 | 种子 | 起点 | 训到 | 停下 | 读不出 10 次均值最高 | 读不出 1% / 2% / 5% | 熵 1.5× / 2× | 3σ：读不出 / 熵 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda r: (r["start"] != "SFT e8", r["arm"], r["run"])):
        stop = "—" if not r["stop"] else ("读不出线" if r["stop"].startswith("unparseable") else "用户停（" + r["stop"].split(":")[1].split(";")[0].strip()[:40] + "）")
        L.append(f"| {r['run']} | {r['line']} | {r['arm']} | {r['seed']} | {r['start']} | {r['updates']} | {stop} | "
                 f"{100 * (r['max_unreadable_10'] or 0):.1f}% | {f(r['unreadable_1pct_at'])} / {f(r['unreadable_2pct_at'])} / "
                 f"{f(r['unreadable_5pct_at'])} | {f(r['entropy_1.5x_at'])} / {f(r['entropy_2.0x_at'])} | "
                 f"{f(r['unreadable_3sigma_at'])} / {f(r['entropy_3sigma_at'])} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
