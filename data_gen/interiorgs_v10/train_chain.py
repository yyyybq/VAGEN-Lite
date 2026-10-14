"""The InteriorGS v10 training chain, run unattended (decisions of 2026-10-14, round 6).

    nohup python -m data_gen.interiorgs_v10.train_chain run --plan TR/plan.json > TR/chain.log 2>&1 &
    python -m data_gen.interiorgs_v10.train_chain status --dir TR      # or read TR/PROGRESS.md

Steps, each started only after the previous one succeeded:

1. ``check_code``  the cluster checkout is the commit named in the plan (every tracked file) with no local
                   change besides the verl submodule;
2. ``render``      SFT frames, one sco card (pilot_sco.py), every frame of frames.jsonl present afterwards;
3. ``sft_data``    conversations (training_prep sft-records) and the LLaMA-Factory data and recipe
                   (scripts/prepare_active_spatial_sft.py, model saved every epoch), on this node;
4. ``sft``         Qwen2.5-VL-7B SFT, 8 cards (job_sco.py, sco_interiorgs_v10_train.sh sft);
5. ``rl_config``   PPO config from the SFT model (training_prep rl-config: a checkpoint every 25 updates);
6. ``rl``          PPO with the dense-score S0 recipe: render service on 1 card + trainer on 8 cards, up to the
                   plan's endpoint;
7. ``rl_extend``   optional (plan "extend_to"): unless validation success at the endpoint is more than 10 points
                   below the best earlier value (``keep_training``, round 8 rule), continue from the latest
                   checkpoint to "extend_to";
8. ``evaluate``    optional (plan "eval": {"dir", "every", "models"}): one single-card evaluation job per RL
                   checkpoint every "every" updates and per extra model (evaluation.py, set EVAL/tasks.jsonl),
                   then the table (EVAL/table.md).

A failed job or check stops the chain: the step is marked failed with the reason, the jobs are deleted (their
logs stay in the run folders) and nothing further starts.  Finished steps are marked done, so ``run`` again
continues after the last one.  Every minute the chain writes TR/progress.json and TR/PROGRESS.md (current
step, SFT loss, RL success rates).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
STEPS = ("check_code", "render", "sft_data", "sft", "rl_config", "rl", "rl_extend", "evaluate")
STEP_ZH = {"check_code": "核对代码", "render": "渲染 SFT 画面", "sft_data": "SFT 对话和数据", "sft": "SFT（7B）",
           "rl_config": "RL 配置", "rl": "RL（S0）", "rl_extend": "RL 续训（验证仍上升时）", "evaluate": "评测"}
CARDS = {"render": "1", "sft": "8", "rl": "1 + 8", "rl_extend": "1 + 8", "evaluate": "每个模型 1"}
POLL_S = 60.0


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class ChainError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Progress readers (pure; tested)
# ---------------------------------------------------------------------------
def sft_progress(model_dir: Path) -> Dict[str, Any]:
    """Latest training and eval loss from LLaMA-Factory's trainer_log.jsonl in the output folder."""
    log = Path(model_dir) / "trainer_log.jsonl"
    out: Dict[str, Any] = {}
    if not log.is_file():
        return out
    for line in log.read_text(errors="ignore").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        for k in ("current_steps", "total_steps", "epoch", "percentage", "remaining_time"):
            if k in r:
                out[k] = r[k]
        if r.get("loss") is not None:
            out["loss"] = r["loss"]
        if r.get("eval_loss") is not None:
            out["eval_loss"] = r["eval_loss"]
            out["eval_at_step"] = r.get("current_steps")
    return out


_METRIC = re.compile(r"([\w./@-]+):(?:np\.float64\()?(-?[0-9.]+(?:e-?[0-9]+)?)\)?")
RL_KEYS = {"train_success": "episode/success_rate", "val_success": "val-aux/active_spatial/traj_success/mean@4",
           "seconds_per_update": "timing_s/step"}


def rl_progress(log_text: str) -> Dict[str, Any]:
    """Per update, from verl's console lines "step:N - key:value - ...": training success (the episode success
    rate of the update's rollouts), validation success (mean over 4 rollouts per val task) and seconds."""
    per_step: Dict[int, Dict[str, float]] = {}
    for line in log_text.splitlines():
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        step = int(m.group(1))
        values = dict(_METRIC.findall(m.group(2)))
        row = per_step.setdefault(step, {})
        for name, key in RL_KEYS.items():
            if key in values:
                row[name] = float(values[key])
    updates = sorted(s for s in per_step if s > 0)
    val = [(s, per_step[s]["val_success"]) for s in sorted(per_step) if "val_success" in per_step[s]]
    return {"last_update": updates[-1] if updates else 0,
            "train_success_last": per_step[updates[-1]].get("train_success") if updates else None,
            "train_success_mean_last10": (sum(per_step[s].get("train_success", 0.0) for s in updates[-10:])
                                          / max(1, len(updates[-10:]))) if updates else None,
            "seconds_per_update_mean_last10": (sum(per_step[s].get("seconds_per_update", 0.0) for s in updates[-10:])
                                               / max(1, len(updates[-10:]))) if updates else None,
            "val_success": val}


def keep_training(val: Sequence[Sequence[float]], max_drop: float = 0.10) -> Optional[bool]:
    """Continue past the endpoint?  Yes unless the last validation success is more than ``max_drop`` below the
    best earlier value (round 8: 64 val tasks give about +-6 points at 40%, so smaller moves are noise);
    None with < 2 points."""
    if len(val) < 2:
        return None
    ys = [float(v) for _, v in val]
    return bool(ys[-1] >= max(ys[:-1]) - max_drop)


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------
class Chain:
    def __init__(self, plan: Dict[str, Any], sleep: Callable[[float], None] = time.sleep) -> None:
        self.plan = plan
        self.tr = Path(plan["tr"])
        self.sft_dir = Path(plan["sft_dir"])
        self.state_path = self.tr / "progress.json"
        self.sleep = sleep
        self.state: Dict[str, Any] = (json.loads(self.state_path.read_text()) if self.state_path.exists() else
                                      {"plan": plan, "steps": {}})
        for s in STEPS:  # progress files written before a step existed
            self.state["steps"].setdefault(s, {"state": "pending"})

    # -------------------------------------------------------------- bookkeeping
    def save(self) -> None:
        self.state["updated_utc"] = now()
        self.state["sft"] = sft_progress(self.tr / "lf" / "model")
        logs = sorted((self.tr / "rl_run" / "logs").glob("rl-*.log")) if (self.tr / "rl_run" / "logs").is_dir() else []
        if logs:
            self.state["rl"] = rl_progress("\n".join(p.read_text(errors="ignore") for p in logs))
            self.state["rl"]["val_keep_training"] = keep_training(self.state["rl"]["val_success"])
        self.tr.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)
        (self.tr / "PROGRESS.md").write_text(render_markdown(self.state))

    def mark(self, step: str, **fields: Any) -> None:
        self.state["steps"][step].update(fields)
        self.save()

    def sh(self, args: List[str], cwd: Optional[Path] = None, log: Optional[Path] = None) -> None:
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        res = subprocess.run(args, cwd=str(cwd or ROOT), env=env, capture_output=True, text=True)
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"$ {' '.join(args)}\n{res.stdout}\n{res.stderr}")
        if res.returncode != 0:
            raise ChainError(f"{' '.join(args[:4])} ... exited {res.returncode}: {res.stderr.strip()[-600:]}")

    # -------------------------------------------------------------- sco jobs
    def wait_jobs(self, step: str, jobs: Dict[str, Callable[[], Dict[str, Any]]], main: str,
                  on_finish: Optional[Callable[[], None]] = None) -> None:
        """Poll until ``main`` finishes; any job failing first stops the step."""
        while True:
            states = {name: fn() for name, fn in jobs.items()}
            self.mark(step, jobs={n: s.get("state") for n, s in states.items()})
            if states[main].get("finished"):
                if on_finish is not None:
                    on_finish()
                if states[main].get("state") != "SUCCEEDED":
                    raise ChainError(f"job {main} ended {states[main].get('state')}")
                return
            for name, s in states.items():
                if name != main and s.get("finished") and s.get("state") != "SUCCEEDED":
                    raise ChainError(f"job {name} ended {s.get('state')} while {main} was running")
                if s.get("state") in ("NOT_FOUND", "SUSPENDED"):  # deleted, or stopped by someone / the pool
                    raise ChainError(f"job {name} is {s.get('state')}")
            self.sleep(POLL_S)

    # -------------------------------------------------------------- steps
    def check_code(self) -> None:
        want = self.plan["commit"]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if not head.startswith(want) and not want.startswith(head):
            raise ChainError(f"checkout is {head}, the plan names {want}")
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.splitlines()
        changed = [c for c in changed if c.strip() != "M verl"]
        if changed:
            raise ChainError(f"tracked files differ from {head[:8]}: {changed[:5]}")
        from .job_sco import WORKTREE

        if Path(WORKTREE).resolve() != ROOT.resolve():
            raise ChainError(f"job_sco.WORKTREE {WORKTREE} is not this checkout {ROOT}")

    def render(self) -> None:
        from . import pilot_sco
        from .pilot import safe_name

        frames = [json.loads(l) for l in (self.sft_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
        plain = self.sft_dir / "frames" / "plain"

        def missing() -> int:
            return sum(1 for f in frames if not (plain / safe_name(f["frame_id"])).is_file())

        if missing() == 0:
            return
        if not (self.sft_dir / "job" / "submission.json").exists():
            pilot_sco.prepare(self.sft_dir, self.plan["gs_root"])
            pilot_sco.submit(self.sft_dir)
        job_id = json.loads((self.sft_dir / "job" / "submission.json").read_text())["job_id"]
        self.mark("render", job_ids={"render": job_id})

        def st() -> Dict[str, Any]:
            d = pilot_sco.describe(job_id)
            return {"state": d.get("state"), "finished": d.get("state") in pilot_sco.FINISHED}

        try:
            self.wait_jobs("render", {"render": st}, "render")
        finally:
            pilot_sco.delete(self.sft_dir)
        n = missing()
        if n:
            raise ChainError(f"{n} of {len(frames)} SFT frames missing after the render job")

    def sft_data(self) -> None:
        py = sys.executable
        self.sh([py, "-m", "data_gen.interiorgs_v10.training_prep", "sft-records", "--dir", str(self.sft_dir)],
                log=self.tr / "logs" / "sft_records.log")
        lf = self.tr / "lf"
        self.sh([py, "prepare_active_spatial_sft.py", "--trajectory", str(self.sft_dir / "sft_trajectories.jsonl"),
                 "--split-manifest", str(self.sft_dir / "split_manifest.json"),
                 "--task-registry", str(self.sft_dir / "task_registry.jsonl"), "--qa-probability", "0",
                 "--allow-unreviewed-trajectories", "--save-per-epoch", "--model", self.plan["model"],
                 "--out", str(lf)], cwd=ROOT / "scripts", log=self.tr / "logs" / "prepare_sft.log")
        if not (lf / "train.yaml").is_file():
            raise ChainError("prepare_active_spatial_sft.py wrote no train.yaml")

    def sft(self) -> None:
        from . import job_sco

        run = self.tr / "sft"
        run.mkdir(parents=True, exist_ok=True)
        if not (run / "job" / "submission.json").exists():
            job_sco.prepare(run, 8, ["sft", str(self.tr / "lf")])
            job_sco.submit(run)
        self.mark("sft", job_ids={"sft": json.loads((run / "job" / "submission.json").read_text())["job_id"]})
        try:
            self.wait_jobs("sft", {"sft": lambda: job_sco.status(run)}, "sft")
        finally:
            job_sco.delete(run)
        model = self.tr / "lf" / "model"
        if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
            raise ChainError(f"no final SFT model in {model}")

    def rl_config(self) -> None:
        self.sh([sys.executable, "-m", "data_gen.interiorgs_v10.training_prep", "rl-config",
                 "--rows", str(self.sft_dir / "rows.jsonl"), "--ppo-template", self.plan["ppo_template"],
                 "--gs-root", self.plan["gs_root"], "--model", str(self.tr / "lf" / "model"),
                 "--name", self.plan.get("rl_name", "v10_rl_s0"), "--run-root", str(self.tr / "rl_run"),
                 "--val-limit", str(self.plan.get("val_limit", 64)), "--out", str(self.tr / "rl")],
                log=self.tr / "logs" / "rl_config.log")

    def rl(self) -> None:
        self._ppo("rl", int(self.plan["endpoint"]))

    def _ppo(self, step: str, endpoint: int, suffix: str = "") -> None:
        """Render service (1 card) + PPO (8 cards) to ``endpoint``; resumes from the latest checkpoint."""
        from . import job_sco

        run = self.tr / "rl_run"
        run.mkdir(parents=True, exist_ok=True)
        jr, jl = "job_renderer" + suffix, "job_rl" + suffix
        if not (run / jr / "submission.json").exists():
            for f in ("STOP_RENDERER", "renderer_endpoint.txt", "renderer_endpoint.tmp"):
                (run / f).unlink(missing_ok=True)
            job_sco.prepare(run, 1, ["renderer"], job_dir=jr)
            job_sco.submit(run, job_dir=jr)
        if not (run / jl / "submission.json").exists():
            job_sco.prepare(run, 8, ["rl", str(self.tr / "rl"), str(endpoint)], job_dir=jl)
            job_sco.submit(run, job_dir=jl)
        self.mark(step, job_ids={d: json.loads((run / d / "submission.json").read_text())["job_id"] for d in (jr, jl)})

        def stop_renderer() -> None:
            (run / "STOP_RENDERER").touch()

        try:
            self.wait_jobs(step, {"renderer": lambda: job_sco.status(run, jr),
                                  "rl": lambda: job_sco.status(run, jl)}, "rl", on_finish=stop_renderer)
        finally:
            stop_renderer()
            for _ in range(20):  # the render service exits within a minute of STOP_RENDERER
                if job_sco.status(run, jr).get("finished"):
                    break
                self.sleep(15)
            job_sco.delete(run, jl)
            job_sco.delete(run, jr)
        if not (run / "checkpoints" / f"global_step_{endpoint}").is_dir():
            raise ChainError(f"no checkpoint at update {endpoint}")

    def rl_extend(self) -> None:
        target = self.plan.get("extend_to")
        if not target:
            self.mark("rl_extend", note="not planned")
            return
        decision = self.state["steps"]["rl_extend"].get("decision")
        if decision is None:  # decided once, at the endpoint; a resumed chain keeps that decision
            logs = sorted((self.tr / "rl_run" / "logs").glob("rl-*.log"))
            val = rl_progress("\n".join(p.read_text(errors="ignore") for p in logs))["val_success"]
            keep = keep_training(val)
            decision = {"val_success": val, "keep_training": keep, "extend_to": int(target) if keep else None}
            self.mark("rl_extend", decision=decision)
        if decision["extend_to"]:
            self._ppo("rl_extend", int(decision["extend_to"]), suffix="_ext")

    def final_update(self) -> int:
        decision = self.state["steps"].get("rl_extend", {}).get("decision") or {}
        return int(decision.get("extend_to") or self.plan["endpoint"])

    def evaluate(self) -> None:
        from . import job_sco

        ev = self.plan.get("eval")
        if not ev:
            self.mark("evaluate", note="not planned")
            return
        eval_dir, every = Path(ev["dir"]), int(ev.get("every", 50))
        ckpts = self.tr / "rl_run" / "checkpoints"
        models = {f"v10_rl_{s:03d}": (str(ckpts / f"global_step_{s}" / "actor" / "huggingface"), None)
                  for s in range(every, self.final_update() + 1, every)}
        models.update({name: (m["model"], m.get("prompt_format")) for name, m in (ev.get("models") or {}).items()})
        models = {n: m for n, m in models.items() if not (eval_dir / n / "done.json").exists()}  # evaluated before
        missing = [f"{n}: {m}" for n, (m, _) in models.items() if not (Path(m) / "config.json").is_file()]
        if missing:  # checked before any job is submitted
            raise ChainError("no model to evaluate for " + "; ".join(missing))
        runs: Dict[str, Path] = {}
        failed: List[str] = []
        try:
            for name, (model, prompt_format) in models.items():
                run = self.tr / "eval_runs" / name
                run.mkdir(parents=True, exist_ok=True)
                if not (run / "job" / "submission.json").exists():
                    job_sco.prepare(run, 1, ["eval", str(eval_dir), name, model] + ([prompt_format] if prompt_format else []))
                    job_sco.submit(run)
                runs[name] = run
            self.mark("evaluate", job_ids={n: json.loads((r / "job" / "submission.json").read_text())["job_id"]
                                           for n, r in runs.items()})
            pending = dict(runs)
            while pending:
                for name, run in list(pending.items()):
                    st = job_sco.status(run)
                    if st.get("finished") or st.get("state") in ("SUSPENDED", "NOT_FOUND"):
                        if not (eval_dir / name / "done.json").exists():
                            failed.append(f"{name} ({st.get('state')})")
                        job_sco.delete(run)
                        pending.pop(name)
                self.save()
                if pending:
                    self.sleep(POLL_S)
        finally:  # a failed submission or poll leaves no job behind
            for run in runs.values():
                if (run / "job" / "submission.json").exists():
                    job_sco.delete(run)
        if failed:
            raise ChainError("evaluation jobs without complete results: " + ", ".join(failed))
        self.sh([sys.executable, "-m", "data_gen.interiorgs_v10.evaluation", "table", "--eval", str(eval_dir)],
                log=self.tr / "logs" / "eval_table.log")

    # -------------------------------------------------------------- driver
    def run(self) -> int:
        self.state["chain_pid"] = os.getpid()
        self.state["state"] = "running"
        self.save()
        for step in STEPS:
            info = self.state["steps"][step]
            if info.get("state") == "done":
                continue
            self.mark(step, state="running", started_utc=now(), error=None)
            try:
                getattr(self, step)()
            except (Exception, SystemExit) as exc:  # stop the chain, keep everything for the post-mortem
                self.mark(step, state="failed", ended_utc=now(), error=f"{type(exc).__name__}: {exc}")
                self.state["state"] = "failed"
                self.save()
                return 1
            self.mark(step, state="done", ended_utc=now())
        self.state["state"] = "done"
        self.save()
        return 0


def render_markdown(state: Dict[str, Any]) -> str:
    zh = {"pending": "未开始", "running": "进行中", "done": "完成", "failed": "失败"}
    lines = [f"# v10 训练进度（更新于 {state.get('updated_utc', '')} UTC）", "",
             f"整体状态：**{zh.get(state.get('state', ''), state.get('state', ''))}**", "",
             "| 步骤 | 卡数 | 状态 | 开始（UTC） | 结束（UTC） | sco 任务 | 说明 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in STEPS:
        i = state["steps"].get(s, {})
        jobs = ", ".join(f"{k} {v}" for k, v in (i.get("job_ids") or {}).items())
        note = i.get("error") or i.get("note") or ""
        if not note and i.get("decision"):
            to = i["decision"].get("extend_to")
            note = (f"验证成功率没有比之前最好的低 10 个百分点以上，续训到 {to} 次" if to
                    else "验证成功率比之前最好的低 10 个百分点以上，不续训")
        lines.append(f"| {STEP_ZH[s]} | {CARDS.get(s, 'CPU')} | {zh.get(i.get('state'), i.get('state'))} | "
                     f"{i.get('started_utc', '')} | {i.get('ended_utc', '')} | {jobs} | {note} |")
    sft = state.get("sft") or {}
    if sft:
        lines += ["", f"**SFT：** 第 {sft.get('current_steps')} / {sft.get('total_steps')} 步（epoch {sft.get('epoch')}），"
                      f"训练损失 {sft.get('loss')}，验证损失 {sft.get('eval_loss')}（第 {sft.get('eval_at_step')} 步），"
                      f"预计剩余 {sft.get('remaining_time')}"]
    rl = state.get("rl") or {}
    if rl:
        val = "，".join(f"{s} 次：{v:.3f}" for s, v in rl.get("val_success") or [])
        lines += ["", f"**RL：** 已完成 {rl.get('last_update')} 次更新；最近一次更新的训练成功率 {rl.get('train_success_last')}，"
                      f"最近 10 次平均 {rl.get('train_success_mean_last10')}；每次更新约 {rl.get('seconds_per_update_mean_last10')} 秒",
                  "", f"验证成功率（64 道 val 题、每题 4 条轨迹的平均）：{val or '还没有'}"]
        if rl.get("val_keep_training") is not None:
            lines.append(f"按续训规则（最后一次比之前最好的低 10 个百分点以上才停）：{'接着训' if rl['val_keep_training'] else '停'}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True, type=Path)
    s = sub.add_parser("status")
    s.add_argument("--dir", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return Chain(json.loads(args.plan.read_text())).run()
    state = json.loads((args.dir / "progress.json").read_text())
    print(render_markdown(state))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
