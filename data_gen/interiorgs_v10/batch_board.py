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
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

STEPS_ZH = {"check_code": "核对代码", "render": "渲染", "sft_data": "SFT 数据", "sft": "SFT", "sft_eval": "SFT 评测",
            "pick_sft": "挑 SFT", "rl_config": "RL 配置", "rl": "RL", "rl_extend": "RL 续训", "evaluate": "评测"}
EVAL_HOURS = 1.5  # the evaluation step after RL, rough (12 single-card jobs, at most 10 at a time)
DEFAULT_SECONDS = 135.0  # seconds per update before a run has logged any


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


def experiment_row(exp: Dict[str, Any], sets: Dict[str, Dict[str, str]], now: datetime.datetime) -> Dict[str, Any]:
    p = Path(exp["chain"]) / "progress.json"
    if not p.is_file():
        return {"state": "未启动", "step": "", "progress": "", "queue": "", "eta": "", "notes": []}
    st = json.loads(p.read_text())
    steps = st.get("steps") or {}
    current = next((s for s in STEPS_ZH if (steps.get(s) or {}).get("state") in ("running", "failed")), None)
    jobs = (steps.get(current) or {}).get("jobs") or {}
    job_states = [v for k, v in jobs.items() if k != "queued" and isinstance(v, str) and "ended" not in k]
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
    waiting = [q for q in queue.values() if not q.get("started")]
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
    for run, reward in (exp.get("runs") or {}).items():
        last = int((decisions.get(run) or {}).get("extend_to") or endpoint)
        best = best_checkpoint(sets, run, last)
        if not best:
            continue
        picks[reward] = best
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
    return lines


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
    lines = [f"# {manifest.get('title', '批次看板')}", "",
             f"更新于 {now.strftime('%Y-%m-%d %H:%M')} UTC（每半小时更新一次）。状态：排队中 = 任务都在等卡；"
             f"预计完成 = 剩下的更新次数 × 最近 10 次的平均秒数，再加约 {EVAL_HOURS} 小时评测。", "",
             "| 编号 | 实验 | 状态 | 当前步骤 | 进度 | 排队 | 预计完成（UTC） | 说明 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    details = []
    for exp in manifest["experiments"]:
        row = experiment_row(exp, sets, now)
        lines.append(f"| {exp['id']} | {exp['name']} | {row['state']} | {row['step']} | {row['progress']} | "
                     f"{row['queue']} | {row['eta']} | {'；'.join(row['notes'])} |")
        if row.get("progress_state"):
            res = experiment_results(exp, sets, row["progress_state"])
            if res:
                details += ["", f"**{exp['id']} {exp['name']}**", ""] + res
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
