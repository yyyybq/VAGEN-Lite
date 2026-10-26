"""The round-12 batch board (InteriorGS v10): one row per experiment, read from the training chains' progress files
and the evaluation results; rewritten every half hour by a loop on the login node.

    python -m data_gen.interiorgs_v10.batch_board --manifest batch.json            # once
    python -m data_gen.interiorgs_v10.batch_board --manifest batch.json --every 1800   # every half hour

The manifest names the evaluation sets and the experiments: {"title", "out", "v1": {"val", "test"}, "v2": {...},
"v2_oa": {...}, "experiments": [{"id", "name", "chain" (folder), "runs": {run: "S0" | "D"}, "controls": {run:
result name}, "sft_group", "sft_control"}]}.  Per experiment: the state (queued, running, done, failed), the step,
the updates done, how long its jobs waited for cards, the expected end; when evaluated, each run's checkpoint picked
on the validation set (ties: the earlier) with its validation and test success, the task-by-task comparison with its
control and of S0 with D (sign test, both sets together), and for an SFT group the picked epoch.

Round-11 models have v1 results plus the occlusion v2 re-evaluation; their v2 results are merged from the two.

Round 12 (the user's section 4 of 2026-10-21), optional manifest keys: an experiment's "section" puts it in its own
table under that heading (the dense reward's first batch apart from the batch); its "comparisons" ([[run, run or
result name], ...]) add task-by-task comparisons between its runs' picks (or a fixed result); "health": true adds
each chain's health (its process alive, how long since its progress file changed, its last restart and why, from
the events); "events" (a JSON-lines file of {"utc", "kind", "text", "chains"}) and "heartbeat" (a file holding the
guard's last check as epoch seconds) add the guard's records: when it last looked, and the latest restarts and
changes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

STEPS_ZH = {"check_code": "核对代码", "render": "渲染", "sft_data": "SFT 数据", "sft": "SFT", "sft_eval": "SFT 评测",
            "pick_sft": "挑 SFT", "rl_config": "RL 配置", "rl": "RL", "rl_extend": "RL 续训", "evaluate": "评测"}
EVAL_HOURS = 1.5  # the evaluation step after RL, rough (12 single-card jobs, at most 10 at a time)
DEFAULT_SECONDS = 135.0  # seconds per update before a run has logged any
EVENTS_SHOWN = 15  # the latest guard records on the board
HEARTBEAT_LATE_MIN = 15  # the guard checks every 5 minutes; later than this it is reported as possibly stopped


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def results(sets: Dict[str, Dict[str, str]], name: str, part: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """{task_id: row} of a result name on the v2 set ``part`` (val / test), None unless complete: results evaluated
    on v2 directly, else the v1 results with the occlusion family taken from the occlusion v2 re-evaluation."""
    v2 = Path(sets["v2"][part])
    ids = [r["task_id"] for r in _jsonl(v2 / "tasks.jsonl")]

    def rows(d: Path) -> Optional[List[Dict[str, Any]]]:
        return _jsonl(d / "results.jsonl") if (d / "done.json").is_file() else None

    direct = rows(v2 / name)
    if direct:
        got = {r["task_id"]: r for r in direct}
    else:
        got = {r["task_id"]: r for r in rows(Path(sets["v1"][part]) / name) or [] if r["task_type"] != "occlusion_alignment"}
        if sets.get("v2_oa"):
            got.update({r["task_id"]: r for r in rows(Path(sets["v2_oa"][part]) / name) or []})
    out = {k: got[k] for k in ids if k in got}
    return out if len(out) == len(ids) else None


def success(res: Optional[Dict[str, Dict[str, Any]]]) -> Optional[float]:
    return None if not res else sum(bool(r["success"]) for r in res.values()) / len(res)


def compare(sets: Dict[str, Dict[str, str]], base: str, other: str) -> Optional[Dict[str, Any]]:
    """``other`` against ``base`` on both v2 sets together: success rates, tasks only one solved, sign-test p."""
    from .evaluation import sign_test

    a, b = {}, {}
    for part in ("val", "test"):
        ra, rb = results(sets, base, part), results(sets, other, part)
        if ra is None or rb is None:
            return None
        a.update({f"{part}/{k}": bool(v["success"]) for k, v in ra.items()})
        b.update({f"{part}/{k}": bool(v["success"]) for k, v in rb.items()})
    ao = sum(a[k] and not b[k] for k in a)
    bo = sum(b[k] and not a[k] for k in a)
    return {"base": sum(a.values()) / len(a), "other": sum(b.values()) / len(b), "base_only": ao, "other_only": bo,
            "p": sign_test(bo, ao)}


def best_checkpoint(sets: Dict[str, Dict[str, str]], run: str, last: int, every: int = 50) -> Optional[str]:
    names = [f"{run}_{s:03d}" for s in range(every, last + 1, every)]
    scored = [(success(results(sets, n, "val")), -i, n) for i, n in enumerate(names)]
    scored = [x for x in scored if x[0] is not None]
    return max(scored)[2] if scored else None


def current_job(chain: str, what: str, job_id: str) -> bool:
    """Whether a queue record's job is still the job of its run: the receipt of its job folder names it.  A job
    deleted while it waited (cancelled, moved to another pool, resubmitted) is not; an unknown kind of record is."""
    parts = what.split("/")
    if len(parts) == 3 and parts[0] in ("rl", "rl_extend") and parts[2] in ("rl", "renderer"):
        receipt = Path(chain) / f"rl_run_{parts[1]}" / f"job_{parts[2]}" / "submission.json"
    elif len(parts) == 2 and parts[0] in ("evaluate", "sft_eval"):
        receipt = Path(chain) / "eval_runs" / parts[1] / "job" / "submission.json"
    else:
        return True
    if not receipt.is_file():
        return False
    try:
        return json.loads(receipt.read_text()).get("job_id") == job_id
    except ValueError:
        return True


def experiment_row(exp: Dict[str, Any], sets: Dict[str, Dict[str, str]], now: datetime.datetime) -> Dict[str, Any]:
    p = Path(exp["chain"]) / "progress.json"
    if not p.is_file():
        return {"state": "未启动", "step": "", "progress": "", "queue": "", "eta": "", "notes": []}
    st = json.loads(p.read_text())
    steps = st.get("steps") or {}
    current = next((s for s in STEPS_ZH if (steps.get(s) or {}).get("state") in ("running", "failed")), None)
    jobs = (steps.get(current) or {}).get("jobs") or {}
    job_states = [v for k, v in jobs.items() if k != "queued" and isinstance(v, str) and "ended" not in k
                  and v not in ("SUCCEEDED", "FAILED", "STOPPED", "CANCELLED")]  # finished jobs neither run nor wait
    trainers = [v for k, v in jobs.items() if k.endswith("/rl") and isinstance(v, str)]
    if trainers:  # an RL step waits as long as its 8-card trainers wait (the render services start at once)
        job_states = trainers
    # the step's own queue records (the SFT step lists its jobs as running while they wait for cards)
    step_queue = [q for q in (st.get("queue") or {}).values() if str(q.get("what", "")).startswith(f"{current}/")
                  and not str(q.get("what", "")).endswith("/renderer")]
    if st.get("state") == "done":
        state = "已完成"
    elif st.get("state") == "failed":
        state = "失败"
    elif job_states and all(v in ("STARTING", "PENDING", "QUEUING", "QUEUEING", "QUEUED", "WAITING", "CREATING")
                            for v in job_states):
        state = "排队中"
    elif step_queue and not any(q.get("started") for q in step_queue):
        state = "排队中"
    else:
        state = "运行中"
    runs = st.get("rl_runs") or {}
    endpoint = int((st.get("plan") or {}).get("endpoint") or exp.get("endpoint") or 300)
    progress = "，".join(f"{n} {r.get('last_update') or 0}/{endpoint}" for n, r in runs.items())
    if current == "sft":
        groups = st.get("sft_groups") or {}
        progress = "，".join(f"{g} 第 {s.get('current_steps')}/{s.get('total_steps')} 步" for g, s in groups.items() if s)
    eta = ""
    if state == "运行中" and current in ("rl", "rl_extend") and runs:
        left = max((endpoint - int(r.get("last_update") or 0)) * float(r.get("seconds_per_update_mean_last10") or DEFAULT_SECONDS)
                   for r in runs.values())
        eta = (now + datetime.timedelta(seconds=left, hours=EVAL_HOURS)).strftime("%m-%d %H:%M")
    queue = st.get("queue") or {}
    waiting = [q for jid, q in queue.items() if not q.get("started") and current_job(exp["chain"], str(q.get("what", "")), jid)]
    started = [q for q in queue.values() if q.get("started")]
    qtext = ""
    if waiting:
        qtext = f"{len(waiting)} 个在排队（最长 {max(q['queued_s'] for q in waiting) / 3600:.1f} 小时）"
    if started:
        qtext += ("；" if qtext else "") + f"已开始的最长排了 {max(q['queued_s'] for q in started) / 3600:.1f} 小时"
    notes = []
    for key, rec in (st.get("job_retries") or {}).items():
        notes.append(f"{key} 重提 {len(rec)} 次（{rec[-1].get('reason')}）")
    for key, rec in (st.get("startup_retries") or {}).items():
        notes.append(f"{key} 启动重提 {len(rec)} 次")
    for key, g in (st.get("guard_stops") or {}).items():
        notes.append(f"{key} 被自动停止：{g.get('reason')}")
    if st.get("state") == "failed":
        err = next((s.get("error") for s in steps.values() if s.get("error")), "")
        notes.append(f"失败原因：{err}")
    return {"state": state, "step": STEPS_ZH.get(current, "") if current else "", "progress": progress,
            "queue": qtext, "eta": eta, "notes": notes, "progress_state": st}


def experiment_results(exp: Dict[str, Any], sets: Dict[str, Dict[str, str]], st: Dict[str, Any]) -> List[str]:
    lines = []
    steps = st.get("steps") or {}
    if exp.get("sft_group"):
        pick = (steps.get("pick_sft") or {}).get("decision") or {}
        if pick.get("best"):
            val, test = success(results(sets, pick["best"], "val")), success(results(sets, pick["best"], "test"))
            line = f"- SFT 挑出 {pick['best']}：验证集 {fmt(val)}、评测集 {fmt(test)}{quality(sets, pick['best'])}"
            c = compare(sets, exp["sft_control"], pick["best"]) if exp.get("sft_control") else None
            if c:
                line += f"；对 {exp['sft_control']}：{cmp_text(c)}"
            lines.append(line)
    if (steps.get("evaluate") or {}).get("state") != "done":
        return lines
    decisions = (steps.get("rl_extend") or {}).get("decisions") or {}
    endpoint = int((st.get("plan") or {}).get("endpoint") or 300)
    picks = {}
    by_run = {}
    for run, reward in (exp.get("runs") or {}).items():
        last = int((decisions.get(run) or {}).get("extend_to") or endpoint)
        best = best_checkpoint(sets, run, last)
        if not best:
            continue
        picks[reward] = best
        by_run[run] = best
        line = f"- {reward}（{run}）挑出 {best}：验证集 {fmt(success(results(sets, best, 'val')))}、评测集 " \
               f"{fmt(success(results(sets, best, 'test')))}{quality(sets, best)}"
        control = (exp.get("controls") or {}).get(run)
        c = compare(sets, control, best) if control else None
        if c:
            line += f"；对 {control}：{cmp_text(c)}"
        lines.append(line)
    if "S0" in picks and "D" in picks:
        c = compare(sets, picks["S0"], picks["D"])
        if c:
            lines.append(f"- D 对 S0：{cmp_text(c)}")
    for other, base in exp.get("comparisons") or []:  # named pairs: a run's pick (or a fixed result) against another
        a, b = by_run.get(other, other), by_run.get(base, base)
        c = compare(sets, b, a) if other in by_run and (base in by_run or base not in (exp.get("runs") or {})) else None
        if c:
            lines.append(f"- {other}（{a}）对 {base}（{b}）：{cmp_text(c)}")
    return lines


def read_events(path: Optional[str]) -> List[Dict[str, Any]]:
    """The guard's records (JSON lines), oldest first; [] without the file; a cut-off last line is skipped."""
    if not path or not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def chain_health(chain: str, events: List[Dict[str, Any]], now: datetime.datetime) -> Dict[str, str]:
    """A chain's process (alive / gone / ended), minutes since its progress file changed and its last restart."""
    p = Path(chain) / "progress.json"
    if not p.is_file():
        return {"process": "未启动", "age": "", "restart": ""}
    st = json.loads(p.read_text())
    pid = st.get("chain_pid")
    if st.get("state") != "running":
        process = "已结束"
    else:
        process = "在跑" if pid and Path(f"/proc/{int(pid)}").exists() else "**进程不在**"
    age = (now.timestamp() - os.path.getmtime(p)) / 60
    restarts = [e for e in events if e.get("kind") == "chain_restart" and chain in (e.get("chains") or [])]
    last = restarts[-1] if restarts else None
    return {"process": process, "age": f"{age:.0f} 分钟前",
            "restart": f"{str(last.get('utc', ''))[5:16].replace('T', ' ')}（{last.get('text', '')}）" if last else ""}


def quality(sets: Dict[str, Dict[str, str]], name: str) -> str:
    """Round 12 batch (the user's decision of 2026-10-21): a model's end progress over its failed episodes, the share
    of the success conditions met at the end, the collision rate and the share of turns with an unreadable answer
    (validation + test); "" without recorded trajectories."""
    from .evaluation import terminal_metrics

    rows = []
    for part in ("val", "test"):
        res = results(sets, name, part)
        rows += list((res or {}).values())
    t = terminal_metrics(rows)
    if not t["episodes"]:
        return ""
    actions = sum(int(r.get("actions") or 0) for r in rows)
    collisions = sum(int(r.get("collisions") or 0) for r in rows)
    turns = sum(int(r.get("turns") or 0) for r in rows)
    invalid = sum(int(r.get("invalid_turns") or 0) for r in rows)  # the user's decision 22: unreadable answers
    prog = "—" if t["progress_end_fail"] is None else f"{t['progress_end_fail']:.3f}"
    return (f"；终点进度分（失败的题）{prog}、终点满足的条件 {fmt(t['conditions_met_share'])}、"
            f"碰撞率 {fmt(collisions / actions if actions else None)}、读不出动作的回合 {fmt(invalid / turns if turns else None)}")


def fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def cmp_text(c: Dict[str, Any]) -> str:
    return (f"两集合并 {fmt(c['other'])} 对 {fmt(c['base'])}，只有它做成 {c['other_only']} 道、只有对照做成 "
            f"{c['base_only']} 道，p = {c['p']:.3f}")


def render(manifest: Dict[str, Any], now: Optional[datetime.datetime] = None) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    sets = {k: manifest[k] for k in ("v1", "v2", "v2_oa") if k in manifest}
    head = ["| 编号 | 实验 | 状态 | 当前步骤 | 进度 | 排队 | 预计完成（UTC） | 说明 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines = [f"# {manifest.get('title', '批次看板')}", "",
             f"更新于 {now.strftime('%Y-%m-%d %H:%M')} UTC（每半小时更新一次）。状态：排队中 = 任务都在等卡；"
             f"预计完成 = 剩下的更新次数 × 最近 10 次的平均秒数，再加约 {EVAL_HOURS} 小时评测。", ""] + head
    details = []
    section = None
    for exp in manifest["experiments"]:
        if exp.get("section") and exp["section"] != section:  # an experiment group listed apart (its own table)
            section = exp["section"]
            lines += ["", f"## {section}", ""] + head
        row = experiment_row(exp, sets, now)
        lines.append(f"| {exp['id']} | {exp['name']} | {row['state']} | {row['step']} | {row['progress']} | "
                     f"{row['queue']} | {row['eta']} | {'；'.join(row['notes'])} |")
        if row.get("progress_state"):
            res = experiment_results(exp, sets, row["progress_state"])
            if res:
                details += ["", f"**{exp['id']} {exp['name']}**", ""] + res
    events = read_events(manifest.get("events"))
    if manifest.get("health"):
        lines += ["", "## 训练链健康", "", "| 编号 | 训练链 | 进程 | 状态文件更新 | 最近一次重启（原因） |",
                  "| --- | --- | --- | --- | --- |"]
        for exp in manifest["experiments"]:
            h = chain_health(exp["chain"], events, now)
            lines.append(f"| {exp['id']} | {Path(exp['chain']).name} | {h['process']} | {h['age']} | {h['restart']} |")
    if manifest.get("events") or manifest.get("heartbeat"):
        lines += ["", "## 守护进程与重启记录", ""]
        beat = Path(manifest["heartbeat"]) if manifest.get("heartbeat") else None
        if beat and beat.is_file():
            late = (now.timestamp() - float(beat.read_text().strip() or 0)) / 60
            lines.append(f"守护进程最近一次检查在 {late:.0f} 分钟前（每 5 分钟一次）"
                         + ("" if late <= HEARTBEAT_LATE_MIN else "，**超过 15 分钟没有检查，可能已停**") + "。")
        else:
            lines.append("**没有守护进程的检查记录。**")
        if events:
            lines += ["", "| 时间（UTC） | 类型 | 内容 |", "| --- | --- | --- |"]
            for e in events[-EVENTS_SHOWN:][::-1]:
                lines.append(f"| {str(e.get('utc', ''))[5:16].replace('T', ' ')} | {e.get('kind', '')} | "
                             f"{str(e.get('text', '')).replace('|', '/')} |")
    if details:
        lines += ["", "## 结果（v2：验证集 413 道、评测集 292 道；按验证集挑检查点）"] + details
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--every", type=float, default=0, help="seconds between rewrites (0: once)")
    args = ap.parse_args(argv)
    while True:
        manifest = json.loads(args.manifest.read_text())
        out = Path(manifest["out"])
        tmp = out.with_suffix(".tmp")
        tmp.write_text(render(manifest), encoding="utf-8")
        tmp.replace(out)
        if not args.every:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
