"""Reward line (the user's item 2 of 2026-11-01): how S0's format drift starts, from the training records only (no GPU).

For each given run (NAME=RUN_DIR, the chain's rl_run_<name> folder) and every update:

* every assistant answer of every recorded episode classified the evaluation's way (evaluation.unreadable_form:
  readable, or no_action_tag / empty_action / format_example_copied / space_instead_of_underscore / made_up_name /
  other), checked against the records' own per-episode invalid_action share;
* unreadable answers by family, by turn position in the episode (1st, 2nd-3rd, 4th+), by the episode's outcome, and
  how concentrated they are in a few tasks; the answers' length (characters) and number of actions;
* from the run's RL log the update's actor/entropy and response_length/mean.

Then per run the onset: the first update where the 10-update mean of unreadable answers reaches 1% / 2% / 5%, and
where the 10-update mean entropy reaches 1.5x / 2x its mean over updates 1-50 -- which moves first -- and, for the
25 updates before and after the unreadable onset (2%), the unreadable forms, families and turn positions, with the
same updates of the comparison runs beside them.

    python -m reward.drift_s0 --run rw_s0=DIR --run q3s0=DIR ... --out OUT.json [--workers 4]

OUT.json (per run and update) and OUT.md.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from data_gen.interiorgs_v10.evaluation import unreadable_form

LOG_STEP = re.compile(r"step:(\d+) - (.*)")
DONE = re.compile(r"<action>[^<]*\bdone\b[^<]*</action>", re.I)  # the model declaring the task done (no such action)


def answers_of(output):
    """Every assistant answer of a record's output: the first one opens the output (its tag ends the input), the
    others follow "<|im_start|>assistant" (replay_q3's split, kept unstripped of nothing but whitespace)."""
    out = []
    for t in re.split(r"<\|im_start\|>assistant\n", output or ""):
        a = t.split("<|im_end|>")[0].strip()
        if a and not a.startswith("[Observation]"):
            out.append(a)
    return out


def position(i):
    return "turn1" if i == 0 else ("turn2-3" if i <= 2 else "turn4+")


_PARSE = None


def env_parse(answer):
    """The env's own strict reading (ActiveSpatialEnv._strict_parse_func, the training's setting: no lenient read):
    (readable, parse_error).  Readable = at least one action read, the env's action_is_valid."""
    global _PARSE
    if _PARSE is None:
        from types import SimpleNamespace

        from data_gen.interiorgs_v10.evaluation import VALID_ACTIONS
        from vagen.envs.active_spatial.env import ActiveSpatialEnv

        # the six moves only: the training env has enable_explicit_done off, so "done" is not executable
        dummy = SimpleNamespace(_allowed_actions=set(VALID_ACTIONS), config=SimpleNamespace(lenient_parse=False))
        _PARSE = lambda text: ActiveSpatialEnv._strict_parse_func(dummy, text, action_sep="|", max_actions=5)  # noqa: E731
    res = _PARSE(answer)
    return bool(res.get("actions")), str(res.get("parse_error") or "")


def update_stats(path):
    u = int(Path(path).stem)
    st = {"update": u, "episodes": 0, "answers": 0, "unreadable": 0, "forms": Counter(), "errors": Counter(),
          "family": defaultdict(lambda: [0, 0]),
          "position": defaultdict(lambda: [0, 0]), "outcome": defaultdict(lambda: [0, 0]), "tasks": Counter(),
          "chars": 0, "n_actions": 0, "check_ok": 0, "check_n": 0}
    st["broken_lines"] = 0
    for line in Path(path).read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a record cut off when a trainer crashed mid-write
            st["broken_lines"] += 1
            continue
        ans = answers_of(r.get("output"))
        st["episodes"] += 1
        bad = 0
        for i, a in enumerate(ans):
            ok, err = env_parse(a)
            form = None if ok else ("done_action" if DONE.search(a) else unreadable_form(a))
            if not ok:
                st["errors"][err] += 1
            st["answers"] += 1
            st["chars"] += len(a)
            m = re.search(r"<action>(.*?)</action>", a, re.S)
            st["n_actions"] += len([x for x in (m.group(1).split("|") if m else []) if x.strip()])
            fam = st["family"][r.get("task_type")]
            pos = st["position"][position(i)]
            out = st["outcome"]["success" if float(r.get("traj_success") or 0) > 0.5 else "fail"]
            for c in (fam, pos, out):
                c[1] += 1
            if not ok:
                bad += 1
                st["unreadable"] += 1
                st["forms"][form] += 1
                st["tasks"][r.get("task_id")] += 1
                for c in (fam, pos, out):
                    c[0] += 1
        rec = float(r.get("invalid_action") or 0)
        mine = bad / len(ans) if ans else 0.0
        st["check_n"] += 1
        st["check_ok"] += int(abs(rec - mine) < 1e-6)
    st["forms"] = dict(st["forms"])
    st["errors"] = dict(st["errors"])
    st["family"] = {k: v for k, v in st["family"].items()}
    st["position"] = dict(st["position"])
    st["outcome"] = dict(st["outcome"])
    st["top_task_share"] = (sum(n for _, n in st["tasks"].most_common(3)) / st["unreadable"]) if st["unreadable"] else None
    st["tasks"] = len(st["tasks"])
    return st


def log_series(run_dir):
    """{update: {"entropy", "response_length"}} from the run's RL logs (the last log wins for a repeated update)."""
    out = {}
    for f in sorted(Path(run_dir, "logs").glob("rl-*.log")):
        for line in f.read_text(errors="ignore").splitlines():
            m = LOG_STEP.search(line)
            if not m:
                continue
            row = {}
            for key, name in (("actor/entropy", "entropy"), ("response_length/mean", "response_length")):
                mm = re.search(re.escape(key) + r":(?:np\.float64\()?([-0-9.eE+]+)", m.group(2))
                if mm:
                    row[name] = float(mm.group(1))
            if row:
                out[int(m.group(1))] = row
    return out


def run_stats(args):
    name, run_dir = args
    files = sorted(Path(run_dir, "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem))
    ups = [update_stats(f) for f in files]
    logs = log_series(run_dir)
    for s in ups:
        s.update(logs.get(s["update"], {}))
    return name, ups


def rolling(xs, k=10):
    out = []
    for i in range(len(xs)):
        w = [x for x in xs[max(0, i - k + 1): i + 1] if x is not None]
        out.append(sum(w) / len(w) if w else None)
    return out


def onset(ups):
    rate = [s["unreadable"] / s["answers"] if s["answers"] else None for s in ups]
    ent = [s.get("entropy") for s in ups]
    r10, e10 = rolling(rate), rolling(ent)
    base = [e for s, e in zip(ups, ent) if s["update"] <= 50 and e is not None]
    eb = sum(base) / len(base) if base else None
    first = lambda xs, cond: next((s["update"] for s, x in zip(ups, xs) if x is not None and cond(x)), None)  # noqa: E731
    out = {"entropy_base_1_50": round(eb, 4) if eb else None, "max_unreadable_10": round(max(x for x in r10 if x is not None), 4)}
    for thr in (0.01, 0.02, 0.05):
        out[f"unreadable_{int(thr * 100)}pct_at"] = first(r10, lambda x, t=thr: x >= t)
    if eb:
        for f in (1.5, 2.0):
            out[f"entropy_{f}x_at"] = first(e10, lambda x, f=f: x >= f * eb)
    return out


def window(ups, lo, hi):
    sel = [s for s in ups if lo <= s["update"] <= hi]
    tot = defaultdict(float)
    forms, errors, fam, pos = Counter(), Counter(), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    out_rate = defaultdict(lambda: [0, 0])
    ent = []
    top = []
    for s in sel:
        tot["answers"] += s["answers"]
        tot["unreadable"] += s["unreadable"]
        tot["chars"] += s["chars"]
        tot["n_actions"] += s["n_actions"]
        forms.update(s["forms"])
        errors.update(s["errors"])
        for k, v in s["outcome"].items():
            out_rate[k][0] += v[0]
            out_rate[k][1] += v[1]
        if s["top_task_share"] is not None:
            top.append(s["top_task_share"])
        for k, v in s["family"].items():
            fam[k][0] += v[0]
            fam[k][1] += v[1]
        for k, v in s["position"].items():
            pos[k][0] += v[0]
            pos[k][1] += v[1]
        if s.get("entropy") is not None:
            ent.append(s["entropy"])
    n = tot["answers"] or 1
    return {"updates": f"{lo}-{hi}", "n_updates": len(sel), "unreadable": round(tot["unreadable"] / n, 4),
            "entropy": round(sum(ent) / len(ent), 4) if ent else None, "chars_per_answer": round(tot["chars"] / n, 1),
            "actions_per_answer": round(tot["n_actions"] / n, 2),
            "forms": dict(forms.most_common()), "parse_errors": dict(errors.most_common()),
            "outcome_rate": {k: round(v[0] / v[1], 4) for k, v in sorted(out_rate.items()) if v[1]},
            "top3_task_share_mean": round(sum(top) / len(top), 3) if top else None,
            "family_rate": {k: round(v[0] / v[1], 4) for k, v in sorted(fam.items()) if v[1]},
            "position_rate": {k: round(v[0] / v[1], 4) for k, v in sorted(pos.items()) if v[1]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR (an S0 run first, comparison runs after)")
    ap.add_argument("--drift", default=None, help="comma-separated names of the drifting runs (default: those reaching 2%)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    runs = [tuple(r.split("=", 1)) for r in args.run]
    with Pool(min(args.workers, len(runs))) as pool:
        res = dict(pool.map(run_stats, runs))
    summary = {}
    for name, ups in res.items():
        o = onset(ups)
        o["check_invalid_share"] = [sum(s["check_ok"] for s in ups), sum(s["check_n"] for s in ups)]
        o["updates"] = len(ups)
        summary[name] = o
    windows = {}
    for name, ups in res.items():
        at = summary[name].get("unreadable_2pct_at")
        if not at:
            continue
        spans = [(1, 50), (max(1, at - 25), at - 1), (at, at + 24)]
        windows[name] = {"onset": at, "windows": {f"{lo}-{hi}": {n: window(u, lo, hi) for n, u in res.items()}
                                                   for lo, hi in spans}}
    Path(args.out).write_text(json.dumps({"summary": summary, "windows": windows,
                                          "per_update": {n: [{k: v for k, v in s.items()} for s in u] for n, u in res.items()}},
                                         indent=1, default=dict))
    L = ["| 运行 | 更新 | 读不出 10 次均值最高 | 读不出到 1% / 2% / 5% 的更新 | 熵基线（1–50） | 熵到 1.5× / 2× 的更新 | 和记录的读不出比例一致 |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, o in summary.items():
        L.append(f"| {n} | {o['updates']} | {100 * o['max_unreadable_10']:.1f}% | {o.get('unreadable_1pct_at')} / "
                 f"{o.get('unreadable_2pct_at')} / {o.get('unreadable_5pct_at')} | {o.get('entropy_base_1_50')} | "
                 f"{o.get('entropy_1.5x_at')} / {o.get('entropy_2.0x_at')} | {o['check_invalid_share'][0]} / {o['check_invalid_share'][1]} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
