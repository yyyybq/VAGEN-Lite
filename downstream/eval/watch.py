#!/usr/bin/env python3
"""Evaluate every checkpoint of the downstream runs as it appears, gate step 3, and keep a board (login node, CPU).

    python downstream/eval/watch.py --every 900 --board RUNS/BOARD.md       # started by downstream/eval/watch.sh

Experiments: the 36 step-3 runs DS-<start>-<ratio>-s<seed> (downstream/train/runs.py; the D5 runs are DS-*-100-s1);
those without a directory are listed, not watched.  Checkpoints: EXP/run/checkpoints/steps_<N>_pytorch_model.pt and
EXP/run/final_model/pytorch_model.pt (tagged steps_<max_train_steps>).  A checkpoint unchanged for --settle seconds
gets one evaluation job EXP/evals/<tag>/ (downstream/eval/run_eval_ckpt.sh, the fixed plan eval/eval_d6.json, --cards
cards, --servers policy servers, the current number of simulator workers), on the pool with fewer waiting jobs.

Rules of the user's decisions of 2026-10-21 on the step-2 report:
  * before every submission (evaluation or training) the free space under the folder quota is read
    (downstream/quota.py); below --min-free-tb, or unreadable, nothing is submitted and the board raises an alarm;
  * --repeat EXP:TAG: that checkpoint is also evaluated by a second job, EXP/evals/<TAG>_repeat/, with the same seeds,
    once per value of --repeat-workers (e.g. 48,96: the parallelism test).  When both jobs are done the board shows
    the success difference (evaluation noise) and each P's mean episode time against the first job's P; later
    evaluations then use the largest tested P whose mean episode time is at most --max-slowdown times the first's;
  * --gate EXP:TAG: when that checkpoint's first evaluation is done, the runs of --gate-ratios x --gate-seeds x all
    starts are made (if needed) and submitted if the 34 tasks had more than 0 successful episodes; with 0 nothing is
    submitted and the board says so.  The decision is kept in --state.
  * when the set of finished evaluations changed, the result report (downstream/report) is rebuilt into
    --report-out with --report-python (it needs matplotlib).
Nothing is ever deleted.  Finished jobs are remembered in --state and not queried again.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
sys.path.insert(0, str(DOWNSTREAM))
sys.path.insert(0, str(DOWNSTREAM / "train"))
import quota  # noqa: E402
import runs  # noqa: E402
import sco_job  # noqa: E402

RUNS_ROOT = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")


def checkpoints(exp: Path) -> list:
    """[(tag, path)] in step order."""
    out = []
    for p in sorted((exp / "run/checkpoints").glob("steps_*_pytorch_model.pt")):
        out.append((p.name.split("_pytorch")[0], p))
    final = exp / "run/final_model/pytorch_model.pt"
    if final.is_file():
        steps = json.loads((exp / "experiment.json").read_text())["steps"]
        out.append((f"steps_{steps}", final))
    return sorted(out, key=lambda x: int(x[0].split("_")[1]))


def settled(path: Path, settle: float, now=time.time) -> bool:
    return path.is_file() and now() - path.stat().st_mtime > settle


def groups():
    """(task -> atomic_seen / composite_seen, spatial tasks, borderline tasks)."""
    tasks = json.loads((DOWNSTREAM / "demos/target50_tasks.json").read_text())["tasks"]
    sp = json.loads((DOWNSTREAM / "demos/spatial_tasks.json").read_text())
    return {t["task"]: t["group"] for t in tasks}, set(sp["spatial"]), set(sp.get("borderline_not_in_group", {}))


GROUP_KEYS = ("all", "atomic_seen", "composite_seen", "spatial", "borderline")


def score(summary):
    """{group: [successes, episodes]} of a finished evaluation (a summary dict or a summary.json path), None before."""
    if isinstance(summary, Path):
        if not summary.is_file():
            return None
        summary = json.loads(summary.read_text())
    if not summary or "per_task" not in summary:
        return None
    group_of, spatial, borderline = groups()
    agg = {k: [0, 0] for k in GROUP_KEYS}
    for task, s in summary["per_task"].items():
        keys = ["all", group_of.get(task, "?")]
        keys += ["spatial"] if task in spatial else ["borderline"] if task in borderline else []
        for k in keys:
            a = agg.setdefault(k, [0, 0])
            a[0] += s["successes"]
            a[1] += s["n"]
    return agg


def fmt(agg) -> str:
    return " / ".join(f"{agg[k][0]}/{agg[k][1]}" if agg and agg.get(k, [0, 0])[1] else "-" for k in GROUP_KEYS)


def last_metric(exp: Path):
    path = exp / "run/metrics.jsonl"
    if not path.is_file():
        return None
    lines = path.read_text().splitlines()
    return json.loads(lines[-1]) if lines else None


def outcomes(episodes_dir: Path) -> dict:
    """(task, seed) -> success of a finished evaluation."""
    path = episodes_dir / "episodes.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []
    return {(r["task"], r["seed"]): bool(r.get("success")) for r in rows if r.get("status") == "complete"}


def merged(evaldir: Path):
    """(summary, outcomes) of one evaluation: its single job (out/episodes) or all its one-card shards
    (shards/s<I>of<K>/out/episodes) combined; None until every part has a result."""
    single = evaldir / "out/episodes"
    if (single / "summary.json").is_file():
        s = json.loads((single / "summary.json").read_text())
        return (s, outcomes(single)) if "per_task" in s else None
    parts = sorted((evaldir / "shards").glob("s*of*"))
    if not parts:
        return None
    k = int(parts[0].name.split("of")[1])
    got = []
    for i in range(1, k + 1):
        d = evaldir / "shards" / f"s{i}of{k}" / "out/episodes"
        if not (d / "summary.json").is_file():
            return None
        s = json.loads((d / "summary.json").read_text())
        if "per_task" not in s:
            return None
        got.append((s, d))
    per_task, outs, walls, means = {}, {}, [], []
    for s, d in got:
        for t, v in s["per_task"].items():
            p = per_task.setdefault(t, {"n": 0, "successes": 0})
            p["n"] += v["n"]
            p["successes"] += v["successes"]
        outs.update(outcomes(d))
        walls.append(s["wall_seconds"])
        if s.get("episode_wall_mean_seconds") is not None:
            means.append((s["episode_wall_mean_seconds"], s["complete"]))
    summary = {"per_task": per_task, "successes": sum(s["successes"] for s, _ in got),
               "complete": sum(s["complete"] for s, _ in got), "wall_seconds": max(walls),
               "episode_wall_mean_seconds": round(sum(m * n for m, n in means) / max(1, sum(n for _, n in means)), 3),
               "workers": got[0][0]["workers"], "shards": k}
    return summary, outs


def compare(first, other) -> dict:
    """Noise and speed of `other` against `first` (two evaluations of one checkpoint, same seeds); each is merged()
    output or a directory holding summary.json and episodes.jsonl."""
    first, other = (x if isinstance(x, tuple) else (json.loads((x / "summary.json").read_text()), outcomes(x))
                    for x in (first, other))
    (a, oa), (b, ob) = first, other
    common = sorted(set(oa) & set(ob))
    rate = lambda s: s["successes"] / max(1, s["complete"])  # noqa: E731
    return {"successes": [a["successes"], b["successes"]], "episodes": [a["complete"], b["complete"]],
            "rate_diff_points": round(100 * (rate(b) - rate(a)), 2),
            "episodes_compared": len(common), "episodes_with_other_outcome": sum(oa[k] != ob[k] for k in common),
            "episode_mean_seconds": [a["episode_wall_mean_seconds"], b["episode_wall_mean_seconds"]],
            "slowdown": round(b["episode_wall_mean_seconds"] / a["episode_wall_mean_seconds"], 3)
            if a.get("episode_wall_mean_seconds") else None,
            "wall_seconds": [a["wall_seconds"], b["wall_seconds"]], "workers": [a["workers"], b["workers"]]}


def parse_target(text: str):
    exp, tag = text.split(":")
    return exp, tag


class Watcher:
    def __init__(self, args, jobs=sco_job, free_space=quota.free_space, run_tools=runs, now=time.time):
        self.a, self.jobs, self.free_space, self.runs, self.now = args, jobs, free_space, run_tools, now
        self.experiments = args.runs_root / "experiments"
        self.state = json.loads(args.state.read_text()) if args.state.is_file() else {}
        self.state.setdefault("finished", {})
        self.alerts, self.notes = [], []
        self.disk = None
        self._hold, self._jobs, self._views, self._rooms = None, None, {}, {}

    # ------------------------------------------------------------------------------------------- bookkeeping
    def save(self):
        self.a.state.write_text(json.dumps(self.state, indent=1) + "\n")

    def job(self, run: Path) -> dict:
        key = str(run)
        if key in self.state["finished"]:
            return self.state["finished"][key]
        st = self.jobs.status(run)
        if st.get("finished"):
            self.state["finished"][key] = st
        return st

    def disk_ok(self) -> bool:
        if self.disk is None:
            self.disk = self.free_space() or {}
            if not self.disk:
                self.alerts.append("剩余空间读不出（`sco afs dir-quota list` 失败），本轮不提交任何任务")
            elif self.disk["free_tb_min"] < self.a.min_free_tb:
                self.alerts.append(f"剩余空间不足：配额 {self.disk['quota_tb']} TB、已用 {self.disk['used_tb']} TB，"
                                   f"剩余至少 {self.disk['free_tb_min']} TB < {self.a.min_free_tb} TB，本轮不提交任何任务")
        return bool(self.disk) and self.disk["free_tb_min"] >= self.a.min_free_tb

    def workers(self) -> int:
        return int(self.state.get("workers", {}).get("chosen", self.a.workers))

    def submit_eval(self, run: Path, ckpt: Path, workers: str, name: str) -> str:
        if self.held():
            return "未提交（等密集奖励第一批上机器）"
        if not self.disk_ok():
            return "未提交（剩余空间不足或读不出）"
        if self.a.dry:
            return f"would submit (workers {workers})"
        pool = self.jobs.pick_pool()
        self.jobs.prepare(run, self.a.cards, pool, name[:40],
                          ["bash", str(HERE / "run_eval_ckpt.sh"), str(ckpt), str(run / "out"), str(self.a.servers),
                           str(workers)])
        return f"submitted {self.jobs.submit(run)} ({pool}, workers {workers})"

    # ------------------------------------------------------------------------------- one-card shards (2026-10-22)
    def held(self) -> bool:
        """True while any job of --hold-until-running has not started: nothing is submitted before the user's
        priority jobs (the dense-reward first batch) are on machines.  Released once, for good."""
        if not self.a.hold_until_running or self.state.get("hold_released"):
            return False
        if self._hold is None:
            states = {j: self.jobs.job_state(j) for j in self.a.hold_until_running}
            started = [j for j, s in states.items() if s and s not in sco_job.WAITING_STATES]
            if len(started) == len(states):
                self.state["hold_released"] = utc()
                self._hold = False
            else:
                self._hold = True
                self.notes.append(f"等密集奖励第一批上机器：{len(started)}/{len(states)} 个已开始，之前不提交任何任务")
        return self._hold

    def job_list(self):
        if self._jobs is None:
            self._jobs = self.jobs.list_jobs() or []
        return self._jobs

    def view(self, pool: str):
        if pool not in self._views:
            self._views[pool] = self.jobs.pool_view(pool, jobs=self.job_list())
        return self._views[pool]

    def room(self, pool: str) -> int:
        """Free cards on machines of `pool` that already run another job, minus our shards still waiting there."""
        if pool not in self._rooms:
            v = self.view(pool)
            waiting = sum(1 for j in self.job_list()
                          if (j.get("display_name") or "").startswith("downstream-ev")
                          and j.get("state") in sco_job.WAITING_STATES
                          and ((j.get("resource_pool") or {}).get("name") if isinstance(j.get("resource_pool"), dict)
                               else j.get("resource_pool")) == pool)
            self._rooms[pool] = 0 if v is None else max(0, sum(sco_job.busy_room(v).values()) - waiting)
        return self._rooms[pool]

    def submit_shard(self, d: Path, ckpt: Path, i: int, k: int, name: str) -> str:
        if self.held():
            return "未提交（等密集奖励第一批上机器）"
        if not self.disk_ok():
            return "未提交（剩余空间不足或读不出）"
        pool = max(sco_job.POOLS, key=self.room)
        if self.room(pool) <= 0:
            return "未提交（两个池里都没有已有任务、还有空卡的机器）"
        if self.a.dry:
            return f"would submit ({pool})"
        self.jobs.prepare(d, 1, pool, name[:60], ["bash", str(HERE / "run_eval_ckpt.sh"), str(ckpt), str(d / "out"), "1",
                                                  str(self.a.shard_workers), f"{i}/{k}"])
        self._rooms[pool] -= 1
        return f"submitted {self.jobs.submit(d)} ({pool})"

    def shard_entry(self, exp: Path, tag: str, path: Path, suffix: str) -> dict:
        evaldir, k = exp / "evals" / f"{tag}{suffix}", self.a.shards
        counts = {"done": 0, "running": 0, "waiting": 0, "failed": 0, "to submit": 0}
        notes = []
        bounced = self.state.setdefault("bounced", {})
        for i in range(1, k + 1):
            d = evaldir / "shards" / f"s{i}of{k}"
            if (d / "job/submission.json").exists():
                st = self.job(d)
                if not st["finished"]:
                    counts["running" if st["state"] == "RUNNING" else "waiting"] += 1
                    continue
                if score(d / "out/episodes/summary.json"):
                    counts["done"] += 1
                    continue
                fails = len(list(d.glob("job_failed_*")))
                if fails >= 2 or self.a.dry:
                    counts["failed"] += 1
                    continue
                (d / "job").rename(d / f"job_failed_{fails + 1}")
                self.state["finished"].pop(str(d), None)
            last = (bounced.get(str(d)) or [None])[-1]
            if last and self.now() - last < self.a.bounce_wait_min * 60:
                counts["waiting"] += 1
                continue
            if not settled(path, self.a.settle, self.now):
                continue
            counts["to submit"] += 1
            name = f"ev-{exp.name.lower()}-{tag.replace('_', '')}{suffix.replace('_', '-')}-s{i}of{k}"
            notes.append(self.submit_shard(d, path, i, k, name))
        m = merged(evaldir)
        entry = {"tag": f"{tag}{suffix}", "finished": m is not None, "score": score(m[0]) if m else None,
                 "state": f"{k} 个 1 卡分片：完成 {counts['done']}、运行 {counts['running']}、排队 {counts['waiting']}"
                          + (f"、失败 {counts['failed']}" if counts["failed"] else "")
                          + (f"；本轮：{'；'.join(sorted(set(notes)))}" if notes else "")}
        if counts["failed"]:
            self.alerts.append(f"{exp.name} {tag}{suffix} 有 {counts['failed']} 个分片失败了 3 次，不再重提，请看 shards/*/job_failed_*")
        return entry

    def guard(self) -> list:
        """The user's rule (2026-10-22): a one-card evaluation must start on a machine that already runs something.
        Every running shard not yet checked is looked up (sco get-workers); one that has nothing but our shards on
        its machine took an idle machine: it is deleted at once and resubmitted after --bounce-wait-min."""
        verified = self.state.setdefault("verified", {})
        bounced = self.state.setdefault("bounced", {})
        out = []
        for sub in sorted(self.experiments.glob("DS-*/evals/*/shards/*/job/submission.json")):
            d, jid = sub.parent.parent, json.loads(sub.read_text())["job_id"]
            if jid in verified or str(d) in self.state["finished"]:
                continue
            if self.job(d)["state"] != "RUNNING":
                continue
            pool = json.loads((d / "job/prepared.json").read_text())["pool"]
            alone = sco_job.alone_on_machine(self.view(pool), jid)
            if alone is None:
                continue
            if alone:
                self.jobs.delete(d)
                (d / "job").rename(d / f"job_bounced_{int(self.now())}")
                bounced.setdefault(str(d), []).append(self.now())
                out.append(f"{jid} 落在没有别的任务的机器上，已删除，{self.a.bounce_wait_min:.0f} 分钟后重提")
            else:
                verified[jid] = utc()
        self.notes += out
        return out

    def eval_entry(self, exp: Path, tag: str, path: Path, suffix: str, workers: str) -> dict:
        if self.a.shards:
            return self.shard_entry(exp, tag, path, suffix)
        run = exp / "evals" / f"{tag}{suffix}"
        entry = {"tag": f"{tag}{suffix}", "state": "waiting for the file to settle"}
        if (run / "job/submission.json").exists():
            st = self.job(run)
            entry["state"] = st["state"] if not st["finished"] else f"{st['state']} exit {st['exit_code']}"
            entry["finished"] = bool(st["finished"])
            entry["score"] = score(run / "out/episodes/summary.json")
        elif settled(path, self.a.settle, self.now):
            name = f"ev-{exp.name.lower()}-{tag.replace('_', '')}{suffix.replace('_', '-')}"
            entry["state"] = self.submit_eval(run, path, workers, name)
        return entry

    # ------------------------------------------------------------------------------------------- one pass
    def experiment_row(self, exp: Path) -> dict:
        train = self.job(exp)["state"] if (exp / "job/submission.json").exists() else "未提交"
        metric = last_metric(exp)
        row = {"exp": exp.name, "train": train, "step": metric.get("step") if metric else None,
               "loss": metric.get("action_dit_loss") if metric else None, "evals": []}
        repeats = {parse_target(t) for t in self.a.repeat}
        for tag, path in checkpoints(exp):
            row["evals"].append(self.eval_entry(exp, tag, path, "", str(self.workers())))
            if (exp.name, tag) in repeats:
                row["evals"].append(self.eval_entry(exp, tag, path, "_repeat", self.a.repeat_workers))
        return row

    def repeat_results(self):
        """Noise / speed comparisons of each --repeat target whose two jobs are done; sets the worker count."""
        out = []
        for target in self.a.repeat:
            exp, tag = parse_target(target)
            first = merged(self.experiments / exp / "evals" / tag)
            if first is None:
                continue
            if self.a.shards:  # one-card shards: the repeat is a second full set, a noise measurement only
                again = merged(self.experiments / exp / "evals" / f"{tag}_repeat")
                if again:
                    out.append(dict(compare(first, again), target=target, workers_tested=again[0]["workers"]))
                continue
            rep = self.experiments / exp / "evals" / f"{tag}_repeat/out"
            for p in self.a.repeat_workers.split(","):
                d = rep / f"episodes_P{p}" if "," in self.a.repeat_workers else rep / "episodes"
                if (d / "summary.json").is_file() and "per_task" in json.loads((d / "summary.json").read_text()):
                    out.append(dict(compare(first, d), target=target, workers_tested=int(p)))
        tested = {r["workers_tested"]: r for r in out}
        wanted = [int(p) for p in self.a.repeat_workers.split(",")]
        if not self.a.shards and tested and set(wanted) <= set(tested) and "workers" not in self.state:
            ok = [p for p in wanted if tested[p]["slowdown"] is not None and tested[p]["slowdown"] <= self.a.max_slowdown]
            chosen = max(ok) if ok else self.a.workers
            self.state["workers"] = {"chosen": chosen, "utc": utc(),
                                     "why": {p: tested[p]["slowdown"] for p in wanted}, "limit": self.a.max_slowdown}
        return out

    def gate(self):
        if not self.a.gate:
            return None
        if "gate" in self.state and self.state["gate"].get("decided"):
            return self.state["gate"]
        exp, tag = parse_target(self.a.gate)
        run = self.experiments / exp / "evals" / tag
        m = merged(run)
        if m is None:
            return {"decided": False, "note": f"等 {exp} {tag} 的评测"}
        agg = score(m[0])
        successes = agg["all"][0]
        if successes == 0:
            self.state["gate"] = {"decided": True, "utc": utc(), "successes": 0, "submitted": [],
                                  "note": f"{exp} {tag} 在 34 个任务上成功 0 个回合：按规定停下，不提交第三步其余任务"}
            self.alerts.append(self.state["gate"]["note"])
            return self.state["gate"]
        if self.held():
            return {"decided": False, "note": f"{exp} {tag} 成功 {successes} 个回合 > 0，等密集奖励第一批上机器后提交"}
        if not self.disk_ok():
            return {"decided": False, "note": f"{exp} {tag} 成功 {successes} 个回合，但剩余空间不足或读不出，暂不提交"}
        if self.a.dry:
            return {"decided": False, "note": f"would submit ({successes} successes)"}
        done, errors = [], []
        for start, ratio, seed in self.runs.grid(runs.STARTS, self.a.gate_ratios, self.a.gate_seeds):
            made = self.runs.make(start, ratio, seed, self.a.runs_root)
            if "error" in made:
                errors.append(made)
                continue
            out = self.runs.submit(start, ratio, seed, self.a.runs_root)
            (errors if "error" in out else done).append(out)
        self.state["gate"] = {"decided": not errors, "utc": utc(), "successes": successes, "submitted": done,
                              "errors": errors,
                              "note": f"{exp} {tag} 成功 {successes} 个回合 > 0：提交第三步其余 {len(done)} 个任务"}
        if errors:
            self.alerts.append(f"第三步其余任务有 {len(errors)} 个没建成或没提交成功，下一轮重试：{errors[:2]}")
        return self.state["gate"]

    def tick(self):
        exp_ids = self.a.exps or [runs.exp_id(s, r, k) for s, r, k in runs.grid(runs.STARTS, runs.RATIOS, (1, 2, 3))]
        rows, missing = [], []
        self.guard()  # first: shards that started on an idle machine since the last look
        self.held()
        self.disk_ok()  # read once per pass, so the board always shows the free space (and its alarm)
        for exp_id in exp_ids:
            exp = self.experiments / exp_id
            if (exp / "experiment.json").exists():
                rows.append(self.experiment_row(exp))
            else:
                missing.append(exp_id)
        rep = self.repeat_results()
        gate = self.gate()
        self.report()
        self.save()
        return rows, missing, rep, gate

    def report(self, run=subprocess.run):
        """Rebuild the result report (downstream/report) when the set of finished evaluations changed.  It needs
        matplotlib, so it runs in --report-python; a failure is shown on the board, never raised."""
        if not self.a.report_out or self.a.dry:
            return
        done = sorted(k for k in self.state["finished"] if "/evals/" in k)
        if done == self.state.get("report", {}).get("evaluations"):
            return
        t0 = time.time()
        try:
            res = run([self.a.report_python, "-m", "downstream.report.build", "--runs-root", str(self.a.runs_root),
                       "--out", self.a.report_out], cwd=str(DOWNSTREAM.parent), capture_output=True, text=True,
                      timeout=1800, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            rc, tail = res.returncode, (res.stderr or res.stdout)[-300:]
        except Exception as exc:
            rc, tail = -1, f"{type(exc).__name__}: {exc}"[:300]
        self.state["report"] = {"utc": utc(), "rc": rc, "seconds": round(time.time() - t0),
                                "evaluations": done if rc == 0 else self.state.get("report", {}).get("evaluations")}
        if rc != 0:
            self.alerts.append(f"结果报告没有生成（退出码 {rc}）：{tail}")


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def board(w: Watcher, rows, missing, rep, gate) -> str:
    lines = [f"# 下游线看板（自动生成，{utc()}）", ""]
    for a in w.alerts:
        lines.append(f"> **报警**：{a}")
    if w.alerts:
        lines.append("")
    for n in w.notes:
        lines.append(f"> {n}")
    if w.notes:
        lines.append("")
    if w.a.shards:
        bounced = sum(len(v) for v in w.state.get("bounced", {}).values())
        lines += [f"- 评测方式：每个检查点 {w.a.shards} 个 1 卡分片（每片 {w.a.shard_workers} 个仿真进程），只在已有别的任务、"
                  f"还有空卡的机器上启动；落到空闲机器上的会被立刻删掉重提（累计 {bounced} 次）", ""]
    d = w.disk or {}
    lines += [f"- 剩余空间：{'配额 %s TB，已用 %s TB，剩余至少 %s TB' % (d['quota_tb'], d['used_tb'], d['free_tb_min']) if d else '本轮未读或读不出'}"
              f"（少于 {w.a.min_free_tb} TB 不提交）",
              (f"- 评测并行进程数：每个 1 卡分片 {w.a.shard_workers} 个" if w.a.shards else
               f"- 评测并行进程数：{w.workers()}"
               + (f"（{w.state['workers']['utc']} 按并行度测试选定：每回合时间是 P={w.a.workers} 的 "
                  f"{w.state['workers']['why']} 倍，上限 {w.state['workers']['limit']}）" if "workers" in w.state else
                  f"（并行度测试未完成前用 {w.a.workers}）")),
              f"- 第三步其余任务的门槛（{w.a.gate}）：{(gate or {}).get('note', '-')}",
              f"- 结果报告 `{w.a.report_out}/index.html`：" + (
                  f"{w.state['report']['utc']} 生成（退出码 {w.state['report']['rc']}）" if "report" in w.state
                  else "评测完成后自动生成"), ""]
    if rep:
        lines += ["## 同一检查点的两次评测（噪声与并行度）", "",
                  "| 检查点 | 第二次的 P | 成功（第一次 / 第二次） | 成功率差（百分点） | 结果不同的回合 / 比较的回合 | 平均每回合秒数（第一次 / 第二次） | 倍数 | 总墙钟秒数 |",
                  "| --- | ---: | --- | ---: | --- | --- | ---: | --- |"]
        for r in rep:
            lines.append(f"| {r['target']} | {r['workers_tested']} | {r['successes'][0]} / {r['successes'][1]} | "
                         f"{r['rate_diff_points']} | {r['episodes_with_other_outcome']} / {r['episodes_compared']} | "
                         f"{r['episode_mean_seconds'][0]} / {r['episode_mean_seconds'][1]} | {r['slowdown']} | "
                         f"{r['wall_seconds'][0]} / {r['wall_seconds'][1]} |")
        lines.append("")
    lines += ["## 实验", "",
              "| 实验 | 训练任务 | 最近记录的步数 | 动作损失 | 检查点：评测状态（成功回合/回合：全部 / 单步 / 多步 / 空间组 / 边界 3 个） |",
              "| --- | --- | ---: | ---: | --- |"]
    for r in rows:
        evals = "<br>".join(f"{e['tag']}: {e['state']}" + (f" — {fmt(e['score'])}" if e.get("score") else "")
                            for e in r["evals"]) or "—"
        loss = f"{r['loss']:.4f}" if isinstance(r["loss"], float) else "—"
        lines.append(f"| {r['exp']} | {r['train']} | {r['step'] if r['step'] is not None else '—'} | {loss} | {evals} |")
    if missing:
        lines += ["", f"未建目录（不看）：{', '.join(missing)}"]
    return "\n".join(lines) + "\n"


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="*", default=None, help="default: the 36 step-3 runs")
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--cards", type=int, default=8, help="cards per eval job (they buy CPUs: 14 vCPU each)")
    ap.add_argument("--servers", type=int, default=1, help="policy servers per eval job (one GPU each)")
    ap.add_argument("--workers", type=int, default=24, help="simulator workers until the parallelism test decides")
    ap.add_argument("--repeat", nargs="*", default=["DS-B-100-s1:steps_25000"])
    ap.add_argument("--repeat-workers", default="48,96")
    ap.add_argument("--max-slowdown", type=float, default=1.5)
    ap.add_argument("--gate", default="DS-B-100-s1:steps_25000")
    ap.add_argument("--gate-ratios", nargs="+", default=["030", "010"])
    ap.add_argument("--gate-seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--min-free-tb", type=float, default=1.0)
    ap.add_argument("--settle", type=float, default=600, help="seconds a checkpoint file must be unchanged")
    ap.add_argument("--state", type=Path, default=RUNS_ROOT / "WATCH_STATE.json")
    ap.add_argument("--every", type=float, default=0, help="loop period in seconds (0: run once)")
    ap.add_argument("--board", type=Path, default=None)
    ap.add_argument("--dry", action="store_true", help="never submit, never write --state")
    ap.add_argument("--report-out", default=str(RUNS_ROOT / "report"),
                    help="rebuild the result report here when evaluations finish (empty: never)")
    ap.add_argument("--report-python", default="/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python")
    ap.add_argument("--shards", type=int, default=8,
                    help="one-card shards per evaluation (0: one job of --cards cards, the old way)")
    ap.add_argument("--shard-workers", type=int, default=12, help="simulator workers per one-card shard (14 vCPU)")
    ap.add_argument("--bounce-wait-min", type=float, default=30,
                    help="minutes before a shard deleted for starting on an idle machine is submitted again")
    ap.add_argument("--guard-every", type=float, default=120, help="seconds between placement checks")
    ap.add_argument("--hold-until-running", nargs="*", default=[],
                    help="job ids that must all have started before anything is submitted")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    last_full = -math.inf
    while True:
        w = Watcher(args)
        if args.dry:
            w.save = lambda: None
        if time.time() - last_full >= args.every:
            last_full = time.time()
            text = board(w, *w.tick())
            if args.board:
                args.board.write_text(text)
            print(text, flush=True)
            if args.every <= 0:
                return 0
        else:  # between full passes: only the placement check of fresh one-card shards
            for line in w.guard():
                print(f"[guard {utc()}] {line}", flush=True)
            w.save()
        time.sleep(min(args.guard_every, args.every))


if __name__ == "__main__":
    sys.exit(main())
