#!/usr/bin/env python3
"""Evaluate every checkpoint of the downstream runs as it appears, gate stage 2, and keep a board (login node, CPU).

    python downstream/eval/watch.py --every 900 --board RUNS/BOARD.md       # started by downstream/eval/watch.sh

Experiments: the 36 step-3 runs DS-<start>-<ratio>-s<seed> (downstream/train/runs.py; the D5 runs are DS-*-100-s1);
those without a directory are listed, not watched.  Checkpoints: EXP/run/checkpoints/steps_<N>_pytorch_model.pt and
EXP/run/final_model/pytorch_model.pt (tagged steps_<max_train_steps>).  A checkpoint unchanged for --settle seconds
gets one evaluation job EXP/evals/<tag>/ (downstream/eval/run_eval_ckpt.sh, the fixed plan eval/eval_d6.json, --cards
cards, --servers policy servers, the current number of simulator workers), on the pool with fewer waiting jobs.

Rules of 2026-10-22 (the user's staged plan, and the main window's conditions for one-card jobs):
  * one-card evaluation shards (--shards) are queued in priority order (stage 0 first: the stopped D5 runs'
    checkpoints, 100k -> 50k -> 25k / 75k, start by start; then later checkpoints by step) and submitted only while
    no --pause-file exists, no --yield-to job (main line interiorgs-v10-*) and no --yield-to-same-size job of the same
    card count (bq_tp_mmxbf16) waits for cards, at most
    --max-live-shards of ours are alive, and only into free cards of machines already running another job;
  * the same pause / yield rules hold for every other submission (8-card evaluations, the stage-2 runs).

Rules of the user's decisions of 2026-10-21 on the step-2 report:
  * before every submission (evaluation or training) the free space under the folder quota is read
    (downstream/quota.py); below --min-free-tb, or unreadable, nothing is submitted and the board raises an alarm;
  * --repeat EXP:TAG: that checkpoint is also evaluated by a second job, EXP/evals/<TAG>_repeat/, with the same seeds,
    once per value of --repeat-workers (e.g. 48,96: the parallelism test).  When both jobs are done the board shows
    the success difference (evaluation noise) and each P's mean episode time against the first job's P; later
    evaluations then use the largest tested P whose mean episode time is at most --max-slowdown times the first's;
  * when the set of finished evaluations changed, the result report (downstream/report) is rebuilt into
    --report-out with --report-python (it needs matplotlib).

Rules of the user's decisions of 2026-10-22 on the stage-1 plan:
  * segmented runs (experiment.json "segmented": run_segmented.sh) evaluate their checkpoints inside the training
    job; the watcher only shows those, submits one-card shards for the intermediate ones only when the job's speed
    rule (EXP/injob_eval.json "final_only") says so, and always the second evaluation of the final checkpoint
    (same seeds, EXP/evals/steps_<last>_repeat/, the noise floor);
  * all our 8-card jobs (training, RL, tests) share --slots places (4); the same-size yield (bq_tp_mmxbf16) holds an
    8-card job back only in the pool where such a job waits;
  * stage-2 gate: passed once any start's step --gate-step (100k) checkpoint, of stage 0 or stage 1, had more than 0
    successful episodes on the 9 spatial tasks; then the --gate-ratios x --gate-seeds runs whose experiment directory
    is set up for the staged plan are submitted one per free place.  The passing is kept in --state.

Rules of the user's decisions of 2026-10-23 evening (after the stage-0 report):
  * GPU order: our new jobs wait for idle machines.  Nothing is submitted while a --yield-to job (the main line,
    interiorgs-v10-*) waits anywhere in the workspace; an 8-card job goes only to a pool with an idle machine, at most
    --max-whole-per-pass of them per pass (the idle estimate cannot see other workspaces' jobs);
  * E1 is evaluated at --extra-points (150k, 175k) besides its final checkpoint (200k, inside the job).  A run keeps
    only its newest full state, so as soon as the state of such a point exists its weights are copied out
    (downstream/train/extract_weights.py, EXP/evals_src/steps_<N>/, with --extract-python) before the next state
    replaces it; the copy is then evaluated by one 8-card job (EXP/evals/steps_<N>/, 8 servers x 96 processes);
  * the second evaluation of a final checkpoint ("repeat_final", B) is off unless --repeat-final is given;
  * runs stopped to free machines (EXP/job renamed job_released_<time>) of --resume-ratios (E2) resume from their
    newest full state when a pool has an idle machine (resume_released);
  * prepared jobs in --queue-dir (*.json: {"run": DIR, "cards": 8, "what": ...}; DIR made by sco_job.prepare) are
    submitted under the same rule (the RL speed test of 64 environments);
  * docs/downstream/MATRIX.md holds two tables with the same rows and columns, training and evaluation.
  * (2026-10-25) a training job that ended without evaluating its saved final checkpoint (E1 B: NCCL failed while
    shutting down, rc 1, run_segmented.sh stopped before the evaluation) leaves that checkpoint to one 8-card
    evaluation under the same idle-machine rule (injob_entry -> whole_entry, same seeds and output folder), ahead of
    the extra points (the job would have evaluated it at once).
Rule of the user's decision of 2026-10-25 ("只给同一个池让路就行，不用一直让，没有卡空着为准"), --yield-pool-only:
  * a waiting main-line job holds back only submissions to the pool it waits in, and withdraws only our jobs still
    waiting there; idle machines of the other pool are used (the idle-machine estimate is unchanged).
Order within a pass: 8-card evaluations, then queued jobs, then resumed runs.
Nothing is ever deleted.  Finished jobs are remembered in --state and not queried again.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
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


STOPPED_RUN = "run_stopped_20261022"
STAGE0_ORDER = {100000: 0, 50000: 1, 25000: 2, 75000: 2}  # the user's order for the stopped runs' checkpoints
# a job of the main line in any of these states still wants cards: we yield to it.  SUSPENDED is not one: in the
# job list it marks stopped jobs (e.g. the main line's rl-run-rd of 10-16), which never run again (checked 10-23)
HELD_STATES = tuple(sco_job.WAITING_STATES)


def job_cards(job: dict):
    """Cards of a job from the sco job list (worker spec N4lS.Iq.I80.<cards>), None if not readable."""
    try:
        return int(str(job["roles"][0]["resource_spec"][0]["name"]).rsplit(".", 1)[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def job_pool(job: dict):
    pool = job.get("resource_pool")
    return pool.get("name") if isinstance(pool, dict) else pool


def prepared_pool(run: Path):
    """The pool a job of ours was prepared for (RUN/job/prepared.json), None if unknown."""
    try:
        return json.loads((run / "job/prepared.json").read_text()).get("pool")
    except (OSError, ValueError):
        return None


def segmented(exp: Path) -> dict:
    """experiment.json of a stage-1 style run (full state, evaluation inside the job), else {}."""
    try:
        rec = json.loads((exp / "experiment.json").read_text())
    except (OSError, ValueError):
        return {}
    return rec if rec.get("segmented") else {}


def start_rank(exp_name: str) -> int:
    start = exp_name.split("-")[1] if exp_name.count("-") >= 3 else ""
    return runs.STARTS.index(start) if start in runs.STARTS else len(runs.STARTS)


def checkpoints(exp: Path, run: str = "run") -> list:
    """[(tag, path)] in step order."""
    out = []
    for p in sorted((exp / run / "checkpoints").glob("steps_*_pytorch_model.pt")):
        out.append((p.name.split("_pytorch")[0], p))
    final = exp / run / "final_model/pytorch_model.pt"
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


def excluded_keys() -> set:
    """The fixed exclusion list (downstream/eval/excluded_episodes.json, the user's decision 1 of 2026-10-23)."""
    path = HERE / "excluded_episodes.json"
    return {(t, int(s)) for t, s in json.loads(path.read_text())["episodes"]} if path.is_file() else set()


def without_excluded(summary: dict, outs: dict):
    """(summary, outcomes) with the excluded episodes taken out: each one the outcomes hold is subtracted from the
    summary's counts (per task, successes, complete)."""
    skip = excluded_keys()
    gone = {k: v for k, v in outs.items() if (k[0], int(k[1])) in skip}
    if not gone:
        return summary, outs
    per_task = {t: dict(v) for t, v in summary["per_task"].items()}
    for (task, _), ok in gone.items():
        if task in per_task:
            per_task[task]["n"] -= 1
            per_task[task]["successes"] -= int(ok)
    out = dict(summary, per_task=per_task)
    for key, value in (("successes", sum(gone.values())), ("complete", len(gone))):
        if key in out:
            out[key] -= value
    return out, {k: v for k, v in outs.items() if k not in gone}


def merged(evaldir: Path):
    """(summary, outcomes) of one evaluation: its single job (out/episodes) or all its one-card shards
    (shards/s<I>of<K>/out/episodes) combined; None until every part has a result.  Excluded episodes never count."""
    single = evaldir / "out/episodes"
    if (single / "summary.json").is_file():
        s = json.loads((single / "summary.json").read_text())
        return without_excluded(s, outcomes(single)) if "per_task" in s else None
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
    return without_excluded(summary, outs)


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
        if getattr(args, "queue_dir", None) is None:
            args.queue_dir = str(Path(args.runs_root) / "queue")
        self.state = json.loads(args.state.read_text()) if args.state.is_file() else {}
        self.state.setdefault("finished", {})
        self.alerts, self.notes = [], []
        self.disk = None
        self._hold, self._jobs, self._views, self._rooms = None, None, {}, {}
        self.pending = []  # (priority key, shard dir, checkpoint, i, k, job name) collected in a pass
        self.pending8 = []  # (priority key, evaluation dir, checkpoint, job name): 8-card evaluations for idle machines
        self.waiting_whole = []  # the same, for our 8-card evaluations submitted but still waiting for a machine
        self.sent8, self.sent8_pool = 0, {}  # 8-card jobs submitted in this pass (all kinds), per pool
        self.sent1, self.sent1_pool = 0, {}  # one-card queue jobs submitted in this pass, per pool
        self.extract_run = subprocess.run  # the weight copy (tests replace it)

    # ------------------------------------------------------------------------------------------- bookkeeping
    def save(self):
        self.a.state.write_text(json.dumps(self.state, indent=1) + "\n")

    def job(self, run: Path) -> dict:
        key = str(run)
        cached = self.state["finished"].get(key)
        if cached and cached.get("job_id") in (None, _job_id(run)):  # a later job of the same run is asked again
            return cached
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
        reason = self.blocked(self.a.cards)
        if reason:
            return f"未提交（{reason}）"
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
                          if (j.get("display_name") or "").startswith("downstream-ev") and job_cards(j) in (None, 1)
                          and j.get("state") in sco_job.WAITING_STATES
                          and ((j.get("resource_pool") or {}).get("name") if isinstance(j.get("resource_pool"), dict)
                               else j.get("resource_pool")) == pool)
            self._rooms[pool] = 0 if v is None else max(0, sum(sco_job.busy_room(v).values()) - waiting)
        return self._rooms[pool]

    def priority_waiting(self, cards: int, pool: str = None) -> list:
        """Jobs of the user's earlier priorities still waiting for cards, which hold back a submission of `cards`
        cards: any --yield-to job anywhere in the workspace (the main line's interiorgs-v10-*, the main window's
        condition of 2026-10-22; with --yield-pool-only only those waiting in `pool`, the user's decision of
        2026-10-25), and --yield-to-same-size jobs (bq_tp_mmxbf16, first in the user's GPU order) of the same card
        count (a one-card shard on a busy machine does not take a whole machine from an 8-card job) waiting in `pool`
        (a machine of the other pool cannot run them; pool None: in any pool)."""
        out = []
        for j in self.job_list():
            name = str(j.get("display_name") or "")
            if j.get("state") not in HELD_STATES:
                continue
            here = pool is None or job_pool(j) in (None, pool)
            if any(name.startswith(p) for p in self.a.yield_to):
                if here or not self.a.yield_pool_only:
                    out.append(name)
            elif any(name.startswith(p) for p in self.a.yield_to_same_size) and job_cards(j) in (None, cards) and here:
                out.append(name)
        return out

    def blocked(self, cards: int = 1, pool: str = None):
        """Why nothing of `cards` cards may be submitted (to `pool`, None: anywhere) in this pass, None if allowed."""
        if self.a.pause_file and Path(self.a.pause_file).exists():
            return f"暂停提交（{self.a.pause_file} 存在）"
        if self.held():
            return "等密集奖励第一批上机器"
        waiting = self.priority_waiting(cards, pool)
        if waiting:
            return f"给排在前面的任务让路：{len(waiting)} 个在排队（{', '.join(sorted(set(waiting))[:3])}…）"
        if not self.disk_ok():
            return "剩余空间不足或读不出"
        return None

    def open_pool(self, cards: int = 8):
        """(pool, None) of the pool an 8-card job may go to now (not blocked; fewer waiting jobs first), or (None, why)."""
        reasons = {}
        for pool in sorted(sco_job.POOLS, key=lambda p: sum(1 for j in self.job_list()
                                                             if j.get("state") in sco_job.WAITING_STATES
                                                             and job_pool(j) == p)):
            reasons[pool] = self.blocked(cards, pool)
            if not reasons[pool]:
                return pool, None
        return None, "；".join(f"{p}：{r}" for p, r in reasons.items())

    def live8(self) -> list:
        """Our 8-card training jobs alive (waiting or running): stage 1-2 training, RL, tests; they share --slots
        places.  8-card evaluation jobs (downstream-ev8-*) are not training and do not count."""
        return [str(j.get("display_name")) for j in self.job_list()
                if str(j.get("display_name") or "").startswith("downstream-") and job_cards(j) == 8
                and not str(j.get("display_name")).startswith("downstream-ev")
                and (j.get("state") in sco_job.WAITING_STATES or j.get("state") == "RUNNING")]

    def live_shards(self) -> int:
        return sum(1 for j in self.job_list() if str(j.get("display_name") or "").startswith("downstream-ev")
                   and job_cards(j) in (None, 1)
                   and (j.get("state") in sco_job.WAITING_STATES or j.get("state") == "RUNNING"))

    # ------------------------------------------------------------- whole idle machines (user, 2026-10-22 16:30)
    def idle_nodes(self, pool: str):
        """Machines of `pool` with nothing of this workspace running on them (--pool-nodes minus the machines in the
        pool view), minus our 8-card evaluation jobs still waiting there; None if the view cannot be read.  Jobs of
        other workspaces on the pool are not seen, so this can be too high; a waiting job of ours stops further ones."""
        nodes = dict(x.split(":") for x in self.a.pool_nodes).get(pool)
        v = self.view(pool)
        if nodes is None or v is None:
            return None
        return max(0, int(nodes) - len(v["hosts"]) - len(self.waiting8(pool)))

    def waiting8(self, pool: str = None) -> list:
        return [str(j.get("display_name")) for j in self.job_list()
                if str(j.get("display_name") or "").startswith("downstream-ev") and job_cards(j) == 8
                and j.get("state") in sco_job.WAITING_STATES and (pool is None or job_pool(j) == pool)]

    def stalled(self, d: Path, st: dict) -> bool:
        """A running evaluation that has finished no episode for --stall-min minutes (and started earlier than that):
        its main process lost a dead worker's episode (closed_loop before 2026-10-22 18:42 waited for it forever) or a
        worker hangs.  The user's decision 3 of 2026-10-22: such a job is deleted and submitted again; the rerun
        resumes and runs only the missing episodes."""
        if not self.a.stall_min or st.get("finished") or st.get("state") != "RUNNING":
            return False
        limit = self.a.stall_min * 60
        try:
            started = datetime.datetime.strptime(st["start_time"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=datetime.timezone.utc).timestamp()
        except (KeyError, TypeError, ValueError):
            started = (d / "job/submission.json").stat().st_mtime
        if self.now() - started < limit:
            return False
        ep = d / "out/episodes/episodes.jsonl"
        last = ep.stat().st_mtime if ep.exists() else started
        return self.now() - last > limit

    def release(self, d: Path) -> None:
        """Delete a stalled evaluation job and keep its record as job_hung_<time>; it is submitted again as usual."""
        res = "dry run" if self.a.dry else self.jobs.delete(d)
        if not self.a.dry:
            (d / "job").rename(d / f"job_hung_{int(self.now())}")
            self.state["finished"].pop(str(d), None)
        self.alerts.append(f"{d.relative_to(self.experiments)} 有 {self.a.stall_min:.0f} 分钟没有完成一个回合：已删除任务，"
                           f"重提后只补缺的回合（{res}）")

    def whole_entry(self, exp: Path, tag: str, path: Path, root: Path, key, name: str = None) -> dict:
        """One checkpoint evaluated by one 8-card job (8 policy servers x 96 simulator processes) on an idle machine of
        --whole-pools: EVALDIR/job, EVALDIR/out/episodes (EVALDIR = ROOT/TAG)."""
        evaldir = root / tag
        name = name or f"ev8-{exp.name.lower()}-stopped-{tag.replace('_', '')}"
        entry = {"tag": tag, "finished": False, "score": None}
        done = merged(evaldir)
        if done:  # finished, whatever happened to its job record
            return dict(entry, finished=True, score=score(done[0]), state="8 卡评测完成")
        if (evaldir / "job/submission.json").exists():
            st = self.job(evaldir)
            m = merged(evaldir)
            if m:
                return dict(entry, finished=True, score=score(m[0]), state="8 卡评测完成")
            if not st["finished"] and self.stalled(evaldir, st):
                self.release(evaldir)
            elif not st["finished"]:
                if st["state"] in sco_job.WAITING_STATES:
                    self.waiting_whole.append((key, evaldir, path, name))
                return dict(entry, state=f"8 卡评测（8 个策略服务 × 96 个仿真进程）：{st['state']}")
            else:
                fails = len(list(evaldir.glob("job_failed_*")))
                if fails >= 2 or self.a.dry:
                    self.alerts.append(f"{exp.name} {tag} 的 8 卡评测失败了 3 次，不再重提，请看 job_failed_*")
                    return dict(entry, state="8 卡评测失败 3 次")
                (evaldir / "job").rename(evaldir / f"job_failed_{fails + 1}")
                self.state["finished"].pop(str(evaldir), None)
        if (evaldir / "job").exists():  # a dry pass kept the stalled job
            return dict(entry, state="8 卡评测：卡住（dry run，没有删除）")
        if not settled(path, self.a.settle, self.now):
            return dict(entry, state="waiting for the file to settle")
        self.pending8.append((key, evaldir, path, name))
        return dict(entry, state="待提交（8 卡，等空闲机器）")

    def yield_waiting(self, pool: str = None) -> list:
        """--yield-to jobs (the main line) waiting for cards anywhere in the workspace; with --yield-pool-only and a
        pool, only those waiting in that pool (a job of ours waiting in a pool is withdrawn only for those)."""
        return [str(j.get("display_name")) for j in self.job_list() if j.get("state") in HELD_STATES
                and any(str(j.get("display_name") or "").startswith(p) for p in self.a.yield_to)
                and (pool is None or not self.a.yield_pool_only or job_pool(j) in (None, pool))]

    def withdraw_waiting_whole(self) -> list:
        """While a main-line job waits (with --yield-pool-only: in the pool ours waits in), our 8-card evaluations
        that have not started are withdrawn (sco job deleted, EVALDIR/job renamed job_withdrawn_<time>, nothing else
        touched) and go back to the queue: they must not take a freed machine before the main line (the agreement
        with the main window, 2026-10-22)."""
        out = []
        if not self.waiting_whole or not self.yield_waiting():
            return out
        for key, evaldir, path, name in self.waiting_whole:
            if not self.yield_waiting(prepared_pool(evaldir)):
                continue
            if self.a.dry:
                out.append(f"{name} (would withdraw)")
                continue
            res = self.jobs.delete(evaldir)
            (evaldir / "job").rename(evaldir / f"job_withdrawn_{int(self.now())}")
            self.state["finished"].pop(str(evaldir), None)
            self.pending8.append((key, evaldir, path, name))
            out.append(f"{name}（{res}）")
        if out:
            self.notes.append(f"主线有任务在排队：撤下我们还没开始的 8 卡评测 {len(out)} 个：{'；'.join(out)}")
        return out

    def flush_whole(self) -> None:
        """Submit pending 8-card evaluations to idle machines of --whole-pools, in priority order: none while blocked,
        none while one of ours waits there, at most --max-whole-per-pass and idle_nodes() per pass.  First, waiting
        ones are withdrawn while the main line waits (withdraw_waiting_whole)."""
        self.withdraw_waiting_whole()
        if not self.pending8:
            return
        todo, sent = sorted(self.pending8, key=lambda x: x[0]), []
        for pool in self.a.whole_pools:
            reason = self.blocked(8, pool)
            if reason:
                self.notes.append(f"{pool}：8 卡评测本轮不提交：{reason}")
                continue
            if self.a.queue_all:  # the user's decision 3 of 2026-10-23: straight into the queue, no idle machine needed
                n = len(todo)
            elif self.waiting8(pool):
                self.notes.append(f"{pool}：我们的 8 卡评测还有在排队的（{self.waiting8(pool)[0]}），先不提交更多")
                continue
            else:
                n = self.idle_left(pool)
            while n > 0 and todo:
                _, evaldir, ckpt, name = todo.pop(0)
                if self.a.dry:
                    sent.append(f"{name} (would, {pool})")
                else:
                    self.jobs.prepare(evaldir, 8, pool, name[:40], ["bash", str(HERE / "run_eval_ckpt.sh"), str(ckpt),
                                                                   str(evaldir / "out"), "8", "96"])
                    sent.append(f"{self.jobs.submit(evaldir)} {name} ({pool})")
                self.sent8 += 1
                self.sent8_pool[pool] = self.sent8_pool.get(pool, 0) + 1
                n -= 1
        self.notes.append(f"8 卡评测（空闲机器）本轮提交 {len(sent)} 个{'：' + '；'.join(sent) if sent else ''}；"
                          f"还有 {len(todo)} 个等空闲机器")

    # ------------------------------------------------------ idle machines only (the user's decisions, 10-23 evening)
    def ours_waiting8(self, pool: str) -> list:
        """Our 8-card jobs of any kind (evaluation, training, test) waiting for cards in `pool`."""
        return [str(j.get("display_name")) for j in self.job_list()
                if str(j.get("display_name") or "").startswith("downstream-") and job_cards(j) == 8
                and j.get("state") in sco_job.WAITING_STATES and job_pool(j) == pool]

    def idle_left(self, pool: str) -> int:
        """8-card jobs that may still go to `pool` in this pass: none while one of ours already waits there (the idle
        estimate was wrong); else its idle machines (idle_nodes, which counts our waiting 8-card evaluations) minus
        those we sent there in this pass, and at most --max-whole-per-pass for the whole pass."""
        if self.ours_waiting8(pool):
            return 0
        idle = (self.idle_nodes(pool) or 0) - self.sent8_pool.get(pool, 0)
        return max(0, min(idle, self.a.max_whole_per_pass - self.sent8))

    def can8(self, pool: str):
        """Why no 8-card job may go to `pool` now (blocked, or no idle machine left in this pass), None if one may."""
        reason = self.blocked(8, pool)
        if not reason and self.idle_left(pool) <= 0:
            reason = (f"我们的 8 卡任务在排队（{self.ours_waiting8(pool)[0]}）" if self.ours_waiting8(pool) else
                      "本轮已提交到上限" if self.sent8 >= self.a.max_whole_per_pass else "没有空闲机器")
        return reason

    def place8(self, what: str, pools=None):
        """(pool, None) for one 8-card job now, else (None, why)."""
        reasons = []
        for pool in pools or self.a.whole_pools:
            reason = self.can8(pool)
            if not reason:
                return pool, None
            reasons.append(f"{pool}：{reason}")
        return None, f"{what}本轮不提交（" + "；".join(reasons) + "）"

    def took8(self, pool: str) -> None:
        self.sent8 += 1
        self.sent8_pool[pool] = self.sent8_pool.get(pool, 0) + 1

    def sent_on_idle(self, name: str, run: Path, kind: str, job: str) -> None:
        self.state.setdefault("idle_sent", {})[name] = {"run": str(run), "kind": kind, "job": job, "utc": utc()}

    def withdraw_for_main_line(self) -> list:
        """Our 8-card jobs sent under the idle-machine rule (kind "resume": a stopped run; "queue": a prepared job)
        that still wait for cards while a main-line job waits (--yield-pool-only: in their pool) are withdrawn: the
        sco job deleted, a run's record
        renamed job_released_<time> (it resumes later the same way), a prepared job's submission.json renamed
        submission_withdrawn_<time>.json (its prepared.json stays, so it is sent again later).  They must not take a
        freed machine before the main line (the user's GPU order of 2026-10-23 evening)."""
        if not self.yield_waiting():
            return []
        out = []
        for name, rec in sorted(self.state.get("idle_sent", {}).items()):
            run = Path(rec["run"])
            if _job_id(run) != rec["job"] or not self.yield_waiting(prepared_pool(run)):
                continue
            st = self.job(run)
            if st.get("finished") or st.get("state") not in sco_job.WAITING_STATES:
                continue
            if self.a.dry:
                out.append(f"{name} (would withdraw)")
                continue
            res = self.jobs.delete(run)
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            if rec["kind"] == "resume":
                (run / "job").rename(run / f"job_released_{stamp}")
            else:
                (run / "job/submission.json").rename(run / f"job/submission_withdrawn_{stamp}.json")
            self.state["finished"].pop(str(run), None)
            out.append(f"{name}（{res}）")
        if out:
            self.notes.append(f"主线有任务在排队：撤下我们按空闲机器提交、还在排队的 {len(out)} 个：{'；'.join(out)}")
        return out

    # ------------------------------------------------ E1 at 150k / 175k (the user's decision 2, 2026-10-23 evening)
    def extra_entries(self, exp: Path) -> list:
        """The --extra-points of a segmented run of --extra-ratios: the weights copied out of the point's full state,
        then one 8-card evaluation of the copy on an idle machine (whole_entry)."""
        out = []
        for step in self.a.extra_points:
            tag = f"steps_{step}"
            ckpt = exp / "evals_src" / tag / "checkpoints" / f"{tag}_pytorch_model.pt"
            waiting = self.copy_weights(exp, step)
            if waiting:
                out.append(waiting)
                continue
            out.append(self.whole_entry(exp, tag, ckpt, exp / "evals", key=(1, step, start_rank(exp.name), 0),
                                        name=f"ev8-{exp.name.lower()}-{tag.replace('_', '')}"))
        return out

    def copy_weights(self, exp: Path, step: int):
        """None once EXP/evals_src/steps_<N>/ holds the copied weights (source.json written last); else a board entry
        saying what it waits for.  The copy runs here, synchronously (about a minute for 16.8 GB), as soon as the
        complete state RUN/full_state/steps_<N> exists; a state replaced before it was copied is reported."""
        tag = f"steps_{step}"
        dest, state = exp / "evals_src" / tag, exp / "run" / "full_state" / tag
        entry = {"tag": tag, "finished": False, "score": None}
        if (dest / "source.json").is_file():
            return None
        if not (state / "downstream_state.json").is_file():
            newest = max((int(p.name.split("_")[1]) for p in (exp / "run/full_state").glob("steps_*")
                          if p.is_dir() and p.name.split("_")[1].isdigit()), default=0)
            if newest > step:
                self.alerts.append(f"{exp.name} 第 {step // 1000}k 步的全状态已被第 {newest // 1000}k 步的替换，"
                                   f"权重没有复制出来，这个点评不了")
                return dict(entry, state="全状态已被替换，没复制到")
            metric = last_metric(exp)
            at = f"，训练到第 {metric['step']} 步" if metric and "step" in metric else ""
            return dict(entry, state=f"等第 {step // 1000}k 步的全状态{at}")
        if self.a.dry:
            return dict(entry, state="dry run：会从全状态复制权重")
        t0 = time.time()
        try:
            res = self.extract_run([self.a.extract_python, str(DOWNSTREAM / "train/extract_weights.py"), str(state),
                                    str(exp / "run"), str(dest)], capture_output=True, text=True, timeout=3600,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            rc, tail = res.returncode, (res.stderr or res.stdout)[-300:]
        except Exception as exc:
            rc, tail = -1, f"{type(exc).__name__}: {exc}"[:300]
        if rc != 0 or not (dest / "source.json").is_file():
            self.alerts.append(f"{exp.name} 第 {step // 1000}k 步的权重复制失败（退出码 {rc}），下一轮重试：{tail}")
            return dict(entry, state="权重复制失败，下一轮重试")
        self.notes.append(f"{exp.name} 第 {step // 1000}k 步：权重已从全状态复制到 evals_src/{tag}/"
                          f"（{time.time() - t0:.0f} 秒）")
        return None

    # --------------------------------------------- prepared jobs and resumed runs, idle machines only (10-23 evening)
    def group_live(self, qdir: Path, group: str) -> list:
        """Runs of the --queue-dir entries of `group` whose job waits or runs (the E3 runs: at most 2 at once, the
        user's decision of 2026-10-22)."""
        out = []
        for entry in sorted(qdir.glob("*.json")):
            spec = json.loads(entry.read_text())
            run = Path(spec["run"])
            if spec.get("group") == group and (run / "job/submission.json").exists() and not self.job(run)["finished"]:
                out.append(spec.get("what", entry.stem))
        return out

    def flush_queue(self) -> list:
        """Prepared jobs of --queue-dir, in name order, one 8-card job per idle machine (place8); each is submitted
        once (RUN/job/submission.json).  An entry goes to the pool it was prepared for, or with "pools" to the first
        of those with an idle machine (its prepared.json is rewritten for that pool); entries of one "group" have at
        most "max_live" jobs waiting or running at once."""
        sent = []
        qdir = Path(self.a.queue_dir) if self.a.queue_dir else None
        for entry in sorted(qdir.glob("*.json")) if qdir and qdir.is_dir() else []:
            spec = json.loads(entry.read_text())
            run = Path(spec["run"])
            if (run / "job/submission.json").exists():
                continue
            cards = int(spec.get("cards", 8))
            if cards not in (1, 8) or not (run / "job/prepared.json").is_file():
                self.alerts.append(f"排队任务 {entry.name}：只支持已 prepare 的 1 卡或 8 卡任务，没提交")
                continue
            missing = [p for p in spec.get("requires", []) if not Path(p).exists()]
            if missing:  # not ready yet (X4's evaluations wait for their export); later entries are not held back
                self.notes.append(f"{spec.get('what', entry.stem)}本轮不提交：还缺 {Path(missing[0]).name}")
                continue
            if spec.get("group") and len(self.group_live(qdir, spec["group"])) >= int(spec.get("max_live", 1)):
                self.notes.append(f"{spec.get('what', entry.stem)}本轮不提交：{spec['group']} 已有 "
                                  f"{len(self.group_live(qdir, spec['group']))} 个在跑或排队（同时最多 {spec.get('max_live', 1)} 个）")
                continue
            prepared = json.loads((run / "job/prepared.json").read_text())
            place = self.place8 if cards == 8 else self.place1
            pool, why = place(spec.get("what", entry.stem), pools=spec.get("pools") or [prepared.get("pool")])
            if why:
                self.notes.append(why)
                break
            if self.a.dry:
                sent.append(f"{spec.get('what', entry.stem)} (would, {pool})")
            else:
                if prepared.get("pool") != pool:
                    (run / "job/prepared.json").write_text(json.dumps(dict(prepared, pool=pool), indent=1) + "\n")
                jid = self.jobs.submit(run)
                self.sent_on_idle(entry.stem, run, "queue", jid)
                sent.append(f"{spec.get('what', entry.stem)} {jid} ({pool})")
            if cards == 8:
                self.took8(pool)
            else:
                self.sent1 += 1
                self.sent1_pool[pool] = self.sent1_pool.get(pool, 0) + 1
        if sent:
            self.notes.append(f"排队任务本轮提交：{'；'.join(sent)}")
        return sent

    def ours_waiting1(self, pool: str) -> list:
        """Our one-card queue jobs (not the evaluation shards) waiting for a card in `pool`."""
        return [str(j.get("display_name")) for j in self.job_list()
                if str(j.get("display_name") or "").startswith("downstream-")
                and not str(j.get("display_name")).startswith("downstream-ev") and job_cards(j) == 1
                and j.get("state") in sco_job.WAITING_STATES and job_pool(j) == pool]

    def place1(self, what: str, pools=None):
        """(pool, None) for one one-card queue job now, else (None, why).  Same main-line rule as the 8-card jobs;
        a card counts as free on a machine that already runs something (room) or on an idle machine (8 each); none
        while one of ours waits there; at most --max-ones-per-pass per pass (X4's evaluations, 2026-10-26)."""
        reasons = []
        for pool in pools or self.a.whole_pools:
            reason = self.blocked(1, pool)
            if not reason and self.ours_waiting1(pool):
                reason = f"我们的 1 卡任务在排队（{self.ours_waiting1(pool)[0]}）"
            if not reason and self.sent1 >= self.a.max_ones_per_pass:
                reason = "本轮 1 卡任务已提交到上限"
            if not reason:
                free = self.room(pool) + sco_job.CARDS_PER_MACHINE * (self.idle_nodes(pool) or 0) \
                    - self.sent1_pool.get(pool, 0)
                if free <= 0:
                    reason = "没有空卡"
            if not reason:
                return pool, None
            reasons.append(f"{pool}：{reason}")
        return None, f"{what}本轮不提交（" + "；".join(reasons) + "）"

    def run_finished(self, exp: Path) -> bool:
        rec = segmented(exp)
        steps = int(rec.get("steps", 0)) if rec else 0
        return any(t == f"steps_{steps}" for t, _ in checkpoints(exp)) if steps else False

    def resume_released(self) -> list:
        """Runs of --resume-ratios stopped to free machines (EXP/job renamed job_released_<time>, the run not finished)
        go back in one at a time when a pool has an idle machine (place8); run_segmented.sh resumes from the newest
        full state."""
        sent = []
        for ratio in self.a.resume_ratios:
            for s in runs.STARTS:
                exp = self.experiments / runs.exp_id(s, ratio, 1)
                if not segmented(exp) or (exp / "job/submission.json").exists() or not list(exp.glob("job_released_*")):
                    continue
                if self.run_finished(exp):
                    continue
                done = self.state.get("release", {}).get(exp.name, {}).get("released_at")
                if done and self.now() - done < self.a.release_hold_min * 60:  # the machine is the main line's now
                    continue
                pool, why = self.place8(f"{exp.name} 续训")
                if why:
                    self.notes.append(why)
                    return sent
                if self.a.dry:
                    sent.append(f"{exp.name} (would, {pool})")
                else:
                    out = self.runs.submit(s, ratio, 1, self.a.runs_root, pool=pool)
                    if "error" in out:
                        self.alerts.append(f"{exp.name} 续训没提交成功：{out['error'][:200]}")
                        continue
                    self.sent_on_idle(exp.name, exp, "resume", out.get("job"))
                    sent.append(f"{exp.name} {out.get('job')} ({pool})")
                self.took8(pool)
        if sent:
            self.notes.append(f"续训（有空闲机器）本轮提交：{'；'.join(sent)}")
        return sent

    def release_at_full_state(self) -> list:
        """Runs of --release-at-full-state are stopped as soon as they save a full state newer than the one they had
        when first seen here (the user's choice of 2026-10-23 21:3x: give machines back to the main line without
        losing progress): the sco job deleted, EXP/job renamed job_released_<time>.  They resume later like any
        released run (resume_released), not before --release-hold-min minutes, so the freed machine goes to the main
        line's queued job.  At most --every seconds of training past the state are lost."""
        out = []
        book = self.state.setdefault("release", {})
        for name in self.a.release_at_full_state:
            exp, rec = self.experiments / name, book.setdefault(name, {})
            if rec.get("released_at") or not (exp / "job/submission.json").exists():
                continue
            states = full_states(exp)
            if "after" not in rec:
                rec.update(after=states[-1] if states else 0, requested=utc())
            if not states or states[-1] <= rec["after"]:
                continue
            st = self.job(exp)
            if st.get("finished") or st.get("state") != "RUNNING":
                continue
            if self.a.dry:
                out.append(f"{name} (would release after step {states[-1]})")
                continue
            res = self.jobs.delete(exp)
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            (exp / "job").rename(exp / f"job_released_{stamp}")
            self.state["finished"].pop(str(exp), None)
            rec.update(released_at=self.now(), released_utc=utc(), full_state=states[-1], record=f"job_released_{stamp}")
            out.append(f"{name}（第 {states[-1]:,} 步的全状态已存好；{res}）")
        if out:
            self.notes.append(f"按要求在全状态检查点停下、机器让给主线：{'；'.join(out)}")
        return out

    def flush_shards(self) -> None:
        """Submit the pending shards in priority order, as far as the rules allow: nothing while blocked(); at most
        --max-live-shards of our one-card jobs alive at once; only into busy_room of the pools."""
        if not self.pending:
            return
        reason = self.blocked()
        if reason:
            self.notes.append(f"{len(self.pending)} 个分片待提交，本轮不提交：{reason}")
            return
        live, sent, left = self.live_shards(), [], None
        for n, (_, d, ckpt, i, k, name) in enumerate(sorted(self.pending, key=lambda x: x[0])):
            if live >= self.a.max_live_shards:
                left = f"同时最多 {self.a.max_live_shards} 个 1 卡任务，已到上限"
            else:
                pool = max(sco_job.POOLS, key=self.room)
                if self.room(pool) <= 0 and self.a.idle_ok_pools:
                    pool = self.a.idle_ok_pools[0]  # the user's go-ahead (10-22 16:30): idle machines there are fine
                    self._rooms[pool] = self.room(pool) + 1
                elif self.room(pool) <= 0:
                    left = "两个池里都没有已有别的任务、还有空卡的机器"
            if left:
                self.notes.append(f"本轮提交 {len(sent)} 个分片；其余 {len(self.pending) - n} 个等下一轮：{left}")
                return
            if self.a.dry:
                sent.append(f"{name} (would, {pool})")
            else:
                self.jobs.prepare(d, 1, pool, name[:60], ["bash", str(HERE / "run_eval_ckpt.sh"), str(ckpt),
                                                          str(d / "out"), "1", str(self.a.shard_workers), f"{i}/{k}"])
                sent.append(f"{self.jobs.submit(d)} {name} ({pool})")
            self._rooms[pool] -= 1
            live += 1
        self.notes.append(f"本轮提交 {len(sent)} 个分片：{'；'.join(sent[:4])}{'…' if len(sent) > 4 else ''}")

    def shard_entry(self, exp: Path, tag: str, path: Path, suffix: str, root: Path = None, key=None) -> dict:
        evaldir, k = (root or exp / "evals") / f"{tag}{suffix}", self.a.shards
        counts = {"done": 0, "running": 0, "waiting": 0, "failed": 0, "to submit": 0}
        notes = []
        bounced = self.state.setdefault("bounced", {})
        for i in range(1, k + 1):
            d = evaldir / "shards" / f"s{i}of{k}"
            if (d / "job/submission.json").exists():
                st = self.job(d)
                if not st["finished"] and self.stalled(d, st):
                    self.release(d)
                    if self.a.dry:
                        counts["running"] += 1
                        continue
                elif not st["finished"]:
                    counts["running" if st["state"] == "RUNNING" else "waiting"] += 1
                    continue
                elif score(d / "out/episodes/summary.json"):
                    counts["done"] += 1
                    continue
                else:
                    fails = len(list(d.glob("job_failed_*")))
                    if fails >= 2 or self.a.dry:
                        counts["failed"] += 1
                        continue
                    (d / "job").rename(d / f"job_failed_{fails + 1}")
                    self.state["finished"].pop(str(d), None)
            if score(d / "out/episodes/summary.json"):  # finished (its job record renamed or gone): never again
                counts["done"] += 1
                continue
            last = (bounced.get(str(d)) or [None])[-1]
            if last and self.now() - last < self.a.bounce_wait_min * 60:
                counts["waiting"] += 1
                continue
            if not settled(path, self.a.settle, self.now):
                continue
            counts["to submit"] += 1
            stopped = "-stopped" if root is not None and root.name != "evals" else ""
            name = f"ev-{exp.name.lower()}{stopped}-{tag.replace('_', '')}{suffix.replace('_', '-')}-s{i}of{k}"
            self.pending.append(((key or (1, 0, 0, 0)) + (i,), d, path, i, k, name))
        m = merged(evaldir)
        entry = {"tag": f"{tag}{suffix}", "finished": m is not None, "score": score(m[0]) if m else None,
                 "state": f"{k} 个 1 卡分片：完成 {counts['done']}、运行 {counts['running']}、排队 {counts['waiting']}"
                          + (f"、失败 {counts['failed']}" if counts["failed"] else "")
                          + (f"、待提交 {counts['to submit']}" if counts["to submit"] else "")}
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
        for sub in sorted(self.experiments.glob("DS-*/evals*/*/shards/*/job/submission.json")):
            d, jid = sub.parent.parent, json.loads(sub.read_text())["job_id"]
            if jid in verified or str(d) in self.state["finished"]:
                continue
            if self.job(d)["state"] != "RUNNING":
                continue
            pool = json.loads((d / "job/prepared.json").read_text())["pool"]
            if pool in self.a.idle_ok_pools:  # idle machines allowed there (the user, 10-22 16:30)
                verified[jid] = utc()
                continue
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
            step = int(tag.split("_")[1])
            return self.shard_entry(exp, tag, path, suffix, key=(1, step, start_rank(exp.name), 1 if suffix else 0))
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
    def repeat_targets(self) -> list:
        """--repeat targets plus the final checkpoint of every segmented run whose experiment.json says
        "repeat_final": its second evaluation with the same seeds (the noise floor).  The user's decision 1 of
        2026-10-22 (second message): only start B's, as one 8-card job on an idle machine."""
        out = list(self.a.repeat)
        for exp in sorted(self.experiments.glob("DS-*")) if self.a.repeat_final else []:  # off: the user, 10-23 evening
            rec = segmented(exp)
            if rec and rec.get("repeat_final"):
                out.append(f"{exp.name}:steps_{rec['steps']}")
        return list(dict.fromkeys(out))

    def training_over(self, exp: Path) -> bool:
        """No training job of the run is alive: none submitted now (EXP/job renamed job_released_*, ...) or ended."""
        return not (exp / "job/submission.json").exists() or bool(self.job(exp).get("finished"))

    def injob_entry(self, exp: Path, tag: str, path: Path = None) -> dict:
        """An evaluation the training job runs itself (run_segmented.sh): only shown while the job lives.  If the job
        ended without it (E1 B, 2026-10-25 03:47: the final checkpoint and record were saved, then the NCCL teardown
        ran out of GPU memory, training exited 1 and run_segmented.sh stopped before evaluating), the checkpoint goes
        to one 8-card evaluation on an idle machine (whole_entry: same seeds, same folder EXP/evals/<TAG>/out, where a
        partial in-job evaluation is resumed)."""
        m = merged(exp / "evals" / tag)
        if m:
            how = "8 卡评测完成（训练任务没评）" if (exp / "evals" / tag / "job").exists() else "任务内评测完成"
            return {"tag": tag, "finished": True, "score": score(m[0]), "state": how}
        if path is not None and self.training_over(exp):
            key = (0, int(tag.split("_")[1]), start_rank(exp.name), 0)  # ahead of the extra points
            entry = self.whole_entry(exp, tag, path, exp / "evals", key=key,
                                     name=f"ev8-{exp.name.lower()}-{tag.replace('_', '')}")
            return entry if entry.get("finished") else dict(entry, state=f"训练任务没评就结束了；{entry['state']}")
        return {"tag": tag, "finished": False, "score": None, "state": "由训练任务自己评（8 个策略服务 × 12 个仿真进程）"}

    def experiment_row(self, exp: Path) -> dict:
        train = self.job(exp)["state"] if (exp / "job/submission.json").exists() else "未提交"
        metric = last_metric(exp)
        row = {"exp": exp.name, "train": train, "step": metric.get("step") if metric else None,
               "loss": metric.get("action_dit_loss") if metric else None, "evals": []}
        repeats = {parse_target(t) for t in self.repeat_targets()}
        rec = segmented(exp)
        rule = {}
        if rec and (exp / "injob_eval.json").is_file():
            rule = json.loads((exp / "injob_eval.json").read_text())
        for tag, path in checkpoints(exp):
            step = int(tag.split("_")[1])
            if rec and (step == int(rec["steps"]) or rule.get("mode") != "final_only"):
                row["evals"].append(self.injob_entry(exp, tag, path))
            else:
                row["evals"].append(self.eval_entry(exp, tag, path, "", str(self.workers())))
            if (exp.name, tag) in repeats and rec:  # the noise floor of a stage-1 run: one 8-card job, idle machine
                row["evals"].append(self.whole_entry(exp, f"{tag}_repeat", path, exp / "evals",
                                                     key=(1, step, start_rank(exp.name), 1),
                                                     name=f"ev8-{exp.name.lower()}-{tag.replace('_', '')}-rep"))
            elif (exp.name, tag) in repeats:
                row["evals"].append(self.eval_entry(exp, tag, path, "_repeat", self.a.repeat_workers))
        if rec and exp.name.split("-")[2] in self.a.extra_ratios:  # E1 at 150k / 175k (the user, 10-23 evening)
            row["evals"] = self.extra_entries(exp) + row["evals"]
        return row

    def repeat_results(self):
        """Noise / speed comparisons of each --repeat target whose two jobs are done; sets the worker count."""
        out = []
        for target in self.repeat_targets():
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
        """Stage 2 (the user's decision 4 of 2026-10-22).  The gate is passed when any start's step --gate-step
        checkpoint (100k: of stage 0's stopped D5 runs or of stage 1) had more than 0 successful episodes on the 9
        spatial tasks.  Then the runs of the four starts x --gate-ratios x --gate-seeds are submitted, in that order,
        one per free place: all our 8-card jobs (training, RL, tests) share --slots places.  Only runs whose
        experiment directory is set up for the staged plan (experiment.json "segmented") are submitted; the others
        wait and the board says so.  The passing is kept in the state file."""
        if not self.a.gate_ratios:
            return None
        st = self.state.setdefault("gate2", {"submitted": []})
        step = self.a.gate_step
        if not st.get("passed"):
            hits, zero = [], []
            for s in runs.STARTS:
                exp = self.experiments / runs.exp_id(s, "100", 1)
                for name in ("evals", f"evals_{STOPPED_RUN}"):
                    m = merged(exp / name / f"steps_{step}")
                    agg = score(m[0]) if m else None
                    if agg and agg["spatial"][0] > 0:
                        hits.append(f"{exp.name}{'（停下的那次）' if name != 'evals' else ''} 空间组 "
                                    f"{agg['spatial'][0]}/{agg['spatial'][1]}")
                    elif agg:
                        zero.append(exp.name)
            if not hits:
                note = f"等任一起点第 {step // 1000}k 步检查点在 9 个空间任务上成功回合 > 0"
                if len(set(zero)) == len(runs.STARTS):
                    note += f"（四个起点第 {step // 1000}k 步已评的都是 0：闸门没过，等阶段 1 的或你的决定）"
                    self.alerts.append(note)
                return {"passed": False, "note": note}
            st.update(passed=utc(), by=hits)
        head = f"闸门 {st['passed']} 已过（{'；'.join(st['by'][:2])}）"
        todo = []
        for start, ratio, seed in self.runs.grid(runs.STARTS, self.a.gate_ratios, self.a.gate_seeds):
            exp = self.experiments / runs.exp_id(start, ratio, seed)
            # only runs that never had a job: a run stopped to free machines (job_released_*) resumes through
            # resume_released (idle machines only); on 2026-10-23 19:30 this gate sent E2's four back in
            if exp.name not in st["submitted"] and not (exp / "job").exists() and not list(exp.glob("job_*")):
                todo.append((start, ratio, seed, exp))
        if not todo:
            return {"passed": True, "note": head + "；阶段 2 的任务都已提交"}
        ready = [t for t in todo if segmented(t[3])]
        waiting = f"；{len(todo) - len(ready)} 个实验目录还没按阶段 1 的设置重建" if len(ready) < len(todo) else ""
        live = self.live8()
        free = self.a.slots - len(live)
        if not ready:
            return {"passed": True, "note": head + waiting}
        if free <= 0:
            return {"passed": True, "note": head + f"；{self.a.slots} 个 8 卡名额已满（{', '.join(live[:4])}）" + waiting}
        pool, reason = self.open_pool(8)
        if reason:
            return {"passed": True, "note": head + f"；暂不提交：{reason}" + waiting}
        sent = []
        for start, ratio, seed, exp in ready[:free]:
            if self.a.dry:
                sent.append(f"{exp.name} (would, {pool})")
                continue
            out = self.runs.submit(start, ratio, seed, self.a.runs_root, pool=pool)
            if "error" in out:
                self.alerts.append(f"{exp.name} 没提交成功，下一轮重试：{out['error'][:200]}")
                continue
            st["submitted"].append(exp.name)
            sent.append(f"{exp.name} {out.get('job')} ({pool})")
        return {"passed": True, "note": head + f"；本轮提交 {len(sent)} 个：{'；'.join(sent)}" + waiting}

    def stage0_row(self, exp: Path) -> dict:
        """Stage 0 (user, 2026-10-22): the checkpoints kept from the stopped D5 run (run_stopped_<date>/checkpoints),
        evaluated into evals_run_stopped_<date>/ in the order 100k -> 50k -> 25k and 75k, start by start."""
        stopped = exp / STOPPED_RUN
        row = {"exp": f"{exp.name}（10-22 停下的那次）", "train": "已停", "step": None, "loss": None, "evals": []}
        root = exp / f"evals_{STOPPED_RUN}"
        for tag, path in checkpoints(exp, run=STOPPED_RUN):
            step = int(tag.split("_")[1])
            key = (0, STAGE0_ORDER.get(step, 9), start_rank(exp.name), step)
            if self.a.stage0_cards == 8 and not (root / tag / "shards").exists():
                row["evals"].append(self.whole_entry(exp, tag, path, root, key))
            else:  # one-card shards (a checkpoint whose shards started stays with them)
                row["evals"].append(self.shard_entry(exp, tag, path, "", root=root, key=key))
            if f"{exp.name}:{tag}" in self.a.stage0_repeat:
                # the user's decision 7 of 2026-10-22: B 100k evaluated a second time as one 8-card job (same seeds);
                # its difference to the shard version is the first estimate of the evaluation noise.  After the 50k
                # evaluations, before 25k / 75k.
                row["evals"].append(self.whole_entry(exp, f"{tag}_repeat", path, root, key=(0, 1, 9, step),
                                                     name=f"ev8-{exp.name.lower()}-stopped-{tag.replace('_', '')}-rep"))
        return row

    def tick(self):
        exp_ids = self.a.exps or [runs.exp_id(s, r, k) for s, r, k in runs.grid(runs.STARTS, runs.RATIOS, (1,))]
        rows, missing = [], []
        self.guard()  # first: shards that started on an idle machine since the last look
        self.held()
        self.disk_ok()  # read once per pass, so the board always shows the free space (and its alarm)
        if self.a.stage0 and self.a.shards:
            for s in runs.STARTS:
                exp = self.experiments / runs.exp_id(s, "100", 1)
                if (exp / STOPPED_RUN / "checkpoints").is_dir():
                    rows.append(self.stage0_row(exp))
        for exp_id in exp_ids:
            exp = self.experiments / exp_id
            if (exp / "experiment.json").exists():
                rows.append(self.experiment_row(exp))
            else:
                missing.append(exp_id)
        self.release_at_full_state()
        self.flush_shards()
        self.flush_whole()
        self.withdraw_for_main_line()
        self.flush_queue()
        self.resume_released()
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
        # (state key, evaluations it reads, output folder, extra build options); with --stage0 also the report of
        # the stopped D5 runs' checkpoints, in REPORT_OUT_stage0
        targets = [("report", "/evals/", self.a.report_out, [])]
        if self.a.stage0:
            targets.append(("report_stage0", f"/evals_{STOPPED_RUN}/", self.a.report_out.rstrip("/") + "_stage0",
                            ["--stage0"]))
        for name, mark, out, extra in targets:
            done = sorted(k for k in self.state["finished"] if mark in k)
            if done == self.state.get(name, {}).get("evaluations"):
                continue
            t0 = time.time()
            try:
                res = run([self.a.report_python, "-m", "downstream.report.build", "--runs-root", str(self.a.runs_root),
                           "--out", out] + extra, cwd=str(DOWNSTREAM.parent), capture_output=True, text=True,
                          timeout=1800, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                rc, tail = res.returncode, (res.stderr or res.stdout)[-300:]
            except Exception as exc:
                rc, tail = -1, f"{type(exc).__name__}: {exc}"[:300]
            self.state[name] = {"utc": utc(), "rc": rc, "seconds": round(time.time() - t0), "out": out,
                                "evaluations": done if rc == 0 else self.state.get(name, {}).get("evaluations")}
            if rc != 0:
                self.alerts.append(f"结果报告 {out} 没有生成（退出码 {rc}）：{tail}")


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ----------------------------------------------------------------------------- experiment matrix (user, 2026-10-22)
MATRIX_PENDING = (
    ("r12k S0-keep", "密集奖励第一批的 S0-keep（`pt-ma8luzju`）训完后接入"),
    ("r12k D-keep", "密集奖励第一批的 D-keep（`pt-zhuu1j9h`、`pt-p6g2le55`）训完后接入"),
    ("P0", "训练线 P0 的结果出来后接入"),
    ("P1", "训练线 P1 的结果出来后接入"),
    ("A 组", "A 组的结果出来后接入"),
    ("复现种子", "复现种子的结果出来后接入"))


def _rate(kn) -> str:
    k, n = kn
    return f"{100 * k / n:.1f}%（{k}/{n}）" if n else "—"


def _stamp(paths) -> str:
    times = [p.stat().st_mtime for p in paths if p.exists()]
    return datetime.datetime.fromtimestamp(max(times), datetime.timezone.utc).strftime("%m-%d %H:%M") if times else ""


def _job_id(d: Path):
    sub = d / "job/submission.json"
    return json.loads(sub.read_text()).get("job_id") if sub.is_file() else None


def _cell(*parts) -> str:
    return "<br>".join(p for p in parts if p) or "—"


def matrix_e0(w: "Watcher", start: str) -> str:
    exp = w.experiments / runs.exp_id(start, "100", 1)
    root = exp / f"evals_{STOPPED_RUN}"
    if not (exp / STOPPED_RUN / "checkpoints").is_dir():
        return "—"
    lines, done, jobs, files = [], 0, [], []
    for step in (25000, 50000, 75000, 100000):
        d = root / f"steps_{step}"
        m = merged(d)
        jid = _job_id(d) or ("8 个 1 卡分片" if (d / "shards").is_dir() else None)
        if jid:
            jobs.append(f"{step // 1000}k `{jid}`" if jid.startswith("pt-") else f"{step // 1000}k {jid}")
        files += list(d.glob("out/episodes/summary.json")) + list(d.glob("shards/*/out/episodes/summary.json"))
        if m:
            done += 1
            agg = score(m[0])
            lines.append(f"{step // 1000}k：总体 {_rate(agg['all'])}，空间组 {_rate(agg['spatial'])}")
        else:
            lines.append(f"{step // 1000}k：{'评测中' if jid else '待提交'}")
    for target in w.a.stage0_repeat:
        name, tag = parse_target(target)
        if name == exp.name:
            m = merged(root / f"{tag}_repeat")
            if m:
                agg = score(m[0])
                lines.append(f"{tag.split('_')[1][:-3]}k 第二遍（整机）：总体 {_rate(agg['all'])}，空间组 {_rate(agg['spatial'])}")
            elif _job_id(root / f"{tag}_repeat"):
                lines.append(f"{tag.split('_')[1][:-3]}k 第二遍（整机）：评测中 `{_job_id(root / f'{tag}_repeat')}`")
    return _cell(f"**{done}/4 个检查点评完**", "；".join(jobs), f"`{root.relative_to(w.a.runs_root)}/`",
                 "<br>".join(lines), f"更新 {_stamp(files)}" if files else "")


def full_states(exp: Path) -> list:
    """Steps of the complete full states of a run (RUN/full_state/steps_<N>; a .partial is not one)."""
    return sorted(int(p.name.split("_")[1]) for p in (exp / "run/full_state").glob("steps_*")
                  if p.is_dir() and p.name.split("_")[1].isdigit())


def matrix_train(w: "Watcher", start: str, ratio: str, what: str) -> str:
    """Training cell of a stage-1 style run: state, job, steps, the full state it can resume from."""
    exp = w.experiments / runs.exp_id(start, ratio, 1)
    rec = segmented(exp)
    if not rec:
        return _cell(f"未开始（{what}的实验目录还没按阶段 1 的设置建）")
    steps, jid, released = int(rec["steps"]), _job_id(exp), sorted(exp.glob("job_released_*"))
    if w.run_finished(exp):
        state = "训完"
    elif jid:
        state = w.job(exp)["state"]
    elif released:
        state = "已停（机器让给主线），有空闲机器时续训"
    else:
        state = "未提交"
    metric, fs = last_metric(exp), full_states(exp)
    lines = [f"训练到第 {metric['step']:,} 步（共 {steps:,}）" if metric and "step" in metric else "",
             f"可续的全状态：第 {fs[-1]:,} 步" if fs and not w.run_finished(exp) else ""]
    rel = w.state.get("release", {}).get(exp.name, {})
    if exp.name in w.a.release_at_full_state and jid and not rel.get("released_at"):
        every = int(rec.get("full_state_interval") or 25000)
        lines.append(f"第 {(rel.get('after', fs[-1] if fs else 0) // every + 1) * every:,} 步的全状态存好后停下（机器让给主线）")
    return _cell(f"**{state}**", f"任务 `{jid}`" if jid else (f"上一个任务记录 `{released[-1].name}`" if released else ""),
                 f"`{exp.relative_to(w.a.runs_root)}/`", "<br>".join(x for x in lines if x),
                 f"更新 {_stamp([exp / 'run/metrics.jsonl'])}" if _stamp([exp / "run/metrics.jsonl"]) else "")


def matrix_eval(w: "Watcher", start: str, ratio: str, points=()) -> str:
    """Evaluation cell of a stage-1 style run: each evaluated checkpoint (the extra points, then the final one, which
    the training job evaluates itself) and, once all are in, their mean."""
    exp = w.experiments / runs.exp_id(start, ratio, 1)
    rec = segmented(exp)
    if not rec:
        return "—"
    final = int(rec["steps"])
    pts = [p for p in points if p < final] + [final]
    aggs, lines, files = [], [], []
    for p in pts:
        d = exp / "evals" / f"steps_{p}"
        m = merged(d)
        files += list(d.glob("out/episodes/summary.json"))
        if m:
            agg = score(m[0])
            aggs.append(agg)
            lines.append(f"{p // 1000}k：总体 {_rate(agg['all'])}，空间组 {_rate(agg['spatial'])}")
        elif _job_id(d):
            lines.append(f"{p // 1000}k：评测 `{_job_id(d)}` {w.job(d)['state']}")
        elif p == final and w.run_finished(exp) and w.training_over(exp):
            lines.append(f"{p // 1000}k：训练任务没评就结束了，等空闲机器评测（8 卡）")
        elif p == final:
            lines.append(f"{p // 1000}k：训完后由训练任务自己评")
        elif (exp / "evals_src" / f"steps_{p}" / "source.json").is_file():
            lines.append(f"{p // 1000}k：权重已复制，等空闲机器评测")
        else:
            lines.append(f"{p // 1000}k：等第 {p // 1000}k 步的全状态，出现后复制权重")
    if len(pts) > 1 and len(aggs) == len(pts):
        mean = {g: 100 * sum(a[g][0] / a[g][1] for a in aggs) / len(aggs) for g in ("all", "spatial")}
        lines.append(f"**{len(pts)} 点平均**：总体 {mean['all']:.1f}%，空间组 {mean['spatial']:.1f}%")
    return _cell(f"**{len(aggs)}/{len(pts)} 个检查点评完**", f"`{(exp / 'evals').relative_to(w.a.runs_root)}/`",
                 "<br>".join(lines), f"更新 {_stamp(files)}" if files else "")


START_NAMES = {"B": "B 原始 7B", "S": "S 动作 SFT 第 9 个 epoch", "RS": "RS P-S0 第 300 次", "RD": "RD P-D 第 150 次"}
MATRIX_COLUMNS = "| 起点 | E0 曲线 | E1 主对比 | E2 样本效率 | E3 RL | E4 种子 |"
E3_TASKS = ("CoffeeSetupMug", "StackBowlsCabinet", "TurnOffStove")  # 10-26 evening: 10%-60% on the E2 30% checkpoints
E3_SOURCE = ("030", "E2 30% 第 20 万步")  # E3 starts from DS-<start>-030-s1 at 200k (the user's decision of 10-26 evening)
E3_UPDATES = 100


def e3_run(w: "Watcher", start: str) -> Path:
    """The E3 run of a start (downstream/rlinf/run_e3.sh): EXPERIMENTS/E3-<start>."""
    return w.experiments / f"E3-{start}"


def rl_global_step(run: Path):
    """(update, total) of the newest RLinf training log of an E3 run (its last "Global Step: N/M" line), or None."""
    logs = sorted(run.glob("rlinf_train_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    with open(logs[-1], "rb") as f:
        f.seek(max(0, logs[-1].stat().st_size - 1_000_000))
        found = re.findall(r"Global Step:\s+(\d+)/(\d+)", f.read().decode("utf-8", "replace"))
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def tasks_rate(summary: dict, tasks) -> list:
    """[successes, episodes] of a finished evaluation over some tasks."""
    per = summary.get("per_task", {})
    return [sum(per[t]["successes"] for t in tasks if t in per), sum(per[t]["n"] for t in tasks if t in per)]


def matrix_e3_train(w: "Watcher", start: str) -> str:
    """Training cell of E3: the job, the stage (train / export / evaluate) and the last PPO update logged."""
    run = e3_run(w, start)
    if not (run / "cfg/train.yaml").is_file():
        return "未开始"
    jid = _job_id(run)
    state = w.job(run)["state"] if jid else "排队等空闲机器（同时最多 2 个）"
    final = run / f"logs/robocasa365_starvla_ppo/checkpoints/global_step_{E3_UPDATES}/actor/dcp_checkpoint/.metadata"
    if merged(run / "evals/final"):
        stage = "训完、导出、评完"
    elif (run / "export/rlinf_export.json").is_file():
        stage = "已导出，944 回合评测中"
    elif final.is_file():
        stage = f"训完 {E3_UPDATES} 次更新，导出中"
    else:
        g = rl_global_step(run)
        stage = f"训练到第 {g[0]} / {g[1]} 次更新" if g else ""
    return _cell(f"**{state}**", f"任务 `{jid}`" if jid else "", f"`{run.relative_to(w.a.runs_root)}/`", stage,
                 f"更新 {_stamp(list(run.glob('rlinf_train_*.log')))}" if list(run.glob("rlinf_train_*.log")) else "")


def matrix_e3_eval(w: "Watcher", start: str) -> str:
    """Evaluation cell of E3: the 944-episode result after RL next to the start's source checkpoint before RL
    (E3_SOURCE; all tasks, the 9 spatial tasks, the RL tasks)."""
    run = e3_run(w, start)
    before = merged(w.experiments / runs.exp_id(start, E3_SOURCE[0], 1) / "evals/steps_200000")
    after = merged(run / "evals/final")

    def line(label, m):
        a = score(m[0])
        return (f"{label}：总体 {_rate(a['all'])}，空间组 {_rate(a['spatial'])}，"
                f"RL 的 {len(E3_TASKS)} 个任务 {_rate(tasks_rate(m[0], E3_TASKS))}")
    if not (run / "cfg/train.yaml").is_file() and not after:
        return "—"
    return _cell(line("RL 后", after) if after else "RL 后：未评",
                 line(f"RL 前（{E3_SOURCE[1]}）", before) if before else "")


X_EXPERIMENTS = (  # the first wave of new experiments (the user's decision of 2026-10-26)
    ("X0", "域内空间问答：34 个任务的重置首帧（三个相机），左右 / 前后 / 远近三类，约 500 道，零样本问四个起点"),
    ("X1", "冻结主干：只训动作头（视觉塔、合并层、语言模型、lm_head 冻结），30% 示教，5 万步，评 944 回合"),
    ("X2", "极低数据：1%、3% 示教（每任务 5、15 条，嵌套），2 个 epoch，评 944 回合"),
    ("X3", "落点准确率：重放 E1 四个最终检查点的空间组 447 回合，记录实际操作的目标"),
    ("X4", "能力保留：E1 四个最终权重和四个原始起点，在主线 v2 验证集（413 道）和评测集（292 道）上各评一次"),
    ("X5", "未见任务：E1 四个最终检查点评 16 个未见原子任务（主）和 16 个未见组合任务（补），每任务 20 回合"),
    ("XE", "对照：B 起点、30% 示教、6 万步（0.36 个 epoch，学习率按 6 万步衰减），其余同 E2，评 944 回合"))
T_EXPERIMENTS = (  # new training runs (the user's decision of 2026-10-26 evening)
    ("T1", "种子 2：B、RS 各训一个 30% 示教 20 万步（DS-<起点>-030-s2），和种子 1 并排"),
    ("T2", "LoRA：语言模型注意力和前馈层 rank 16，视觉塔、合并层冻结，动作头全训；四个起点，30% 示教，10 万步"),
    ("T5", "NavigateKitchen 单任务：全部人类示教，四个起点，3 个 epoch，评 100 回合（种子 1000–1099）"),
    ("T3/T4", "主线新权重：给出路径后先做 X0 问答，再按 DS-<起点>-030-s1 的配方训练（起点表加一行）"))


def matrix_x(w: "Watcher", table=X_EXPERIMENTS, folder: str = "x") -> list:
    """One row per new experiment; its state and per-start cells come from downstream_runs/<folder>/<X>/matrix.json
    ({"state": ..., "B": ..., "S": ..., "RS": ..., "RD": ...}) once there is one, else its design waits."""
    rows = []
    for x, what in table:
        f = Path(w.a.runs_root) / folder / x.split("/")[0] / "matrix.json"
        try:
            rec = json.loads(f.read_text()) if f.is_file() else {}
        except ValueError:
            rec = {"state": f"`{folder}/{x}/matrix.json` 读不出"}
        rows.append(f"| {x} | {what} | {rec.get('state', '设计待你确认（10-26）')} | "
                    + " | ".join(str(rec.get(s, "—")) for s in runs.STARTS) + " |")
    return rows


def matrix(w: "Watcher") -> str:
    """docs/downstream/MATRIX.md: rows are the starts (the four of now and the ones to be added), columns the
    experiments E0-E4 (the user's decision 5 of 2026-10-22); two tables of the same rows and columns, training and
    evaluation (the user's decision 6 of 2026-10-23 evening).  Rewritten by the watcher every pass."""
    sep = "| --- | --- | --- | --- | --- | --- |"
    head = ["# 下游线实验矩阵（看板每轮自动重写，不要手改）", "",
            f"生成于 {utc()}。E0 = 阶段 0 的检查点曲线（D5 停下时保留的 2.5 万–10 万步）；E1 = 阶段 1 主对比（100% 数据、"
            f"种子 1，评第 {'、'.join(f'{p // 1000}k' for p in w.a.extra_points)} 和最终 200k 步，主数字用三点平均）；"
            f"E2 = 样本效率（10%、30% 数据）；E3 = RL（PPO，{len(E3_TASKS)} 个空间任务，从 {E3_SOURCE[1]}出发）；"
            f"E4 = 种子（10-26 取消）；X0–X5、XE = 第一波新增实验；T 系列 = 新增训练。成功率：总体 = 34 个任务 944 回合，"
            f"空间组 = 9 个空间任务 447 回合（剔除 6 个回合后）；括号里是成功回合 / 回合。时间是 UTC。",
            "路径相对于 `/mnt/umm/users/yinbaiqiao/downstream_runs/`。状态和决定见 [STATUS.md](STATUS.md)。", "",
            "## 训练", "", MATRIX_COLUMNS, sep]
    train, evals = [], []
    for s in runs.STARTS:
        exp0 = w.experiments / runs.exp_id(s, "100", 1)
        e0 = (_cell("不再训练：D5（10-21 开跑、10-22 停在约 11.5 万步）的检查点",
                    f"`{(exp0 / STOPPED_RUN).relative_to(w.a.runs_root)}/`")
              if (exp0 / STOPPED_RUN / "checkpoints").is_dir() else "—")
        e2t = "<br><br>".join(f"**{p}%**：{matrix_train(w, s, r, f'{p}% 数据')}" for r, p in (("010", 10), ("030", 30)))
        e2e = "<br><br>".join(f"**{p}%**：{matrix_eval(w, s, r)}" for r, p in (("010", 10), ("030", 30)))
        train.append(f"| {START_NAMES.get(s, s)} | {e0} | {matrix_train(w, s, '100', 'E1 ')} | {e2t} | "
                     f"{matrix_e3_train(w, s)} | 取消（你 10-26 的决定） |")
        evals.append(f"| {START_NAMES.get(s, s)} | {matrix_e0(w, s)} | {matrix_eval(w, s, '100', w.a.extra_points)} | "
                     f"{e2e} | {matrix_e3_eval(w, s)} | — |")
    for name, note in MATRIX_PENDING:
        train.append(f"| {name}（待接入） | {note} | — | — | — | — |")
        evals.append(f"| {name}（待接入） | {note} | — | — | — | — |")
    xhead = ["| 实验 | 内容 | 状态 | B | S | RS | RD |", "| --- | --- | --- | --- | --- | --- | --- |"]
    xs = (["", "## 新增实验（第一波，你 10-26 的决定）", ""] + xhead + matrix_x(w)
          + ["", "## 新增训练（T 系列，你 10-26 晚的决定）", ""] + xhead + matrix_x(w, T_EXPERIMENTS, "t"))
    return "\n".join(head + train + ["", "## 评测", "", MATRIX_COLUMNS, sep] + evals + xs) + "\n"


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
              f"- 我们的 8 卡任务（训练、RL、测试共用 {w.a.slots} 个名额）：{len(w.live8())} 个在跑或排队"
              + (f"（{', '.join(w.live8())}）" if w.live8() else ""),
              f"- 阶段 2 闸门（任一起点第 {w.a.gate_step // 1000}k 步在 9 个空间任务上成功 > 0）："
              f"{(gate or {}).get('note', '-')}",
              f"- 结果报告 `{w.a.report_out}/index.html`：" + (
                  f"{w.state['report']['utc']} 生成（退出码 {w.state['report']['rc']}）" if "report" in w.state
                  else "评测完成后自动生成")]
    if w.a.stage0:
        r0 = w.state.get("report_stage0")
        lines.append(f"- 阶段 0 报告 `{w.a.report_out.rstrip('/')}_stage0/index.html`：" + (
            f"{r0['utc']} 生成（退出码 {r0['rc']}）" if r0 else "阶段 0 的分片完成后自动生成"))
    lines.append("")
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
    ap.add_argument("--repeat", nargs="*", default=[],
                    help="EXP:TAG evaluated a second time; the final checkpoint of every segmented run always is")
    ap.add_argument("--repeat-workers", default="48,96")
    ap.add_argument("--max-slowdown", type=float, default=1.5)
    ap.add_argument("--gate-step", type=int, default=100000,
                    help="stage-2 gate: any start's checkpoint of this step with > 0 spatial successes (user, 10-22)")
    ap.add_argument("--slots", type=int, default=4, help="places shared by all our 8-card jobs (user, 10-22)")
    ap.add_argument("--gate-ratios", nargs="*", default=["010"], help="stage 2 (user, 10-22): 010 first; 030 later")
    ap.add_argument("--gate-seeds", nargs="+", type=int, default=[1], help="seeds 2 and 3 are not run (user, 10-22)")
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
    ap.add_argument("--stage0", action="store_true",
                    help="also evaluate the checkpoints kept from the stopped D5 runs (run_stopped_20261022)")
    ap.add_argument("--max-live-shards", type=int, default=8,
                    help="at most this many of our one-card evaluation jobs alive at once (main window, 10-22)")
    ap.add_argument("--yield-to", nargs="*", default=["interiorgs-v10-"],
                    help="submit nothing while any workspace job whose name starts so is waiting for cards")
    ap.add_argument("--yield-to-same-size", nargs="*", default=["bq_tp_mmxbf16"],
                    help="submit nothing of N cards while such a job of N cards waits (unknown size: wait too)")
    ap.add_argument("--yield-pool-only", action="store_true",
                    help="a waiting --yield-to job holds back (and withdraws) only our jobs of the pool it waits in; "
                         "idle machines of the other pool are used (the user's decision of 2026-10-25)")
    ap.add_argument("--stage0-cards", type=int, choices=(1, 8), default=1,
                    help="8: a stage-0 checkpoint whose shards have not started is evaluated by one 8-card job "
                         "(8 servers x 96 processes) on an idle machine of --whole-pools (the user, 10-22 16:30)")
    ap.add_argument("--whole-pools", nargs="*", default=["h800"], help="pools for the 8-card evaluations")
    ap.add_argument("--stage0-repeat", nargs="*", default=["DS-B-100-s1:steps_100000"],
                    help="stage-0 checkpoints evaluated a second time as one 8-card job (noise estimate)")
    ap.add_argument("--stall-min", type=float, default=90,
                    help="a running evaluation with no finished episode for this many minutes is deleted and "
                         "submitted again (it resumes); 0: never")
    ap.add_argument("--matrix", default=str(DOWNSTREAM.parent / "docs/downstream/MATRIX.md"),
                    help="the experiment matrix rewritten every pass (empty: off)")
    ap.add_argument("--queue-all", action="store_true",
                    help="submit every pending 8-card evaluation to the queue at once, idle machine or not (the user's "
                         "decision 3 of 2026-10-23; start the watcher with --yield-to --yield-to-same-size empty and "
                         "--slots / --max-live-shards high, so only the pause file holds submissions back)")
    ap.add_argument("--max-ones-per-pass", type=int, default=8,
                    help="one-card queue jobs (X4's evaluations) submitted per pass at most")
    ap.add_argument("--max-whole-per-pass", type=int, default=1,
                    help="8-card evaluations submitted per pass at most (the idle-machine estimate cannot see other "
                         "workspaces' jobs; on 10-22 two of four went to the queue)")
    ap.add_argument("--pool-nodes", nargs="*", default=["h800:29", "zoetrope:15"],
                    help="POOL:N machines usable in the pool (sco aec2 clusters list, minus exception nodes)")
    ap.add_argument("--idle-ok-pools", nargs="*", default=[],
                    help="one-card shards may also go to idle machines of these pools (no placement check there)")
    ap.add_argument("--pause-file", default=str(RUNS_ROOT / "PAUSE_SUBMISSIONS"),
                    help="while this file exists nothing is submitted (e.g. on the main window's request)")
    ap.add_argument("--extra-points", nargs="*", type=int, default=[150000, 175000],
                    help="checkpoints of the segmented --extra-ratios runs (E1) copied out of their full states and "
                         "evaluated by 8-card jobs, besides the final one (the user's decision 2, 2026-10-23 evening)")
    ap.add_argument("--extra-ratios", nargs="*", default=["100"])
    ap.add_argument("--extract-python", default="/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python",
                    help="Python with torch and deepspeed for downstream/train/extract_weights.py")
    ap.add_argument("--repeat-final", action="store_true",
                    help="also evaluate a final checkpoint a second time when experiment.json says repeat_final "
                         "(off: the user cancelled B's on 2026-10-23 evening)")
    ap.add_argument("--resume-ratios", nargs="*", default=["010"],
                    help="runs of these ratios stopped to free machines (job_released_*) resume on idle machines")
    ap.add_argument("--queue-dir", default=None,
                    help="prepared 8-card jobs (*.json) submitted when a pool has an idle machine (default: queue/ "
                         "under --runs-root; a fixed default let the tests read the real queue on 2026-10-26)")
    ap.add_argument("--release-at-full-state", nargs="*", default=[],
                    help="experiments stopped once they save their next full state (machines back to the main line)")
    ap.add_argument("--release-hold-min", type=float, default=30,
                    help="minutes a run stopped by --release-at-full-state is not resumed")
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
            if args.matrix:
                try:
                    Path(args.matrix).write_text(matrix(w), encoding="utf-8")
                except Exception as exc:  # the matrix must never stop the watcher
                    print(f"[matrix {utc()}] {type(exc).__name__}: {exc}", flush=True)
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
