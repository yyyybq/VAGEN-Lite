"""The InteriorGS v10 training chain, run unattended (decisions of 2026-10-14, round 6; extended 2026-10-15, round 10).

    nohup python -m data_gen.interiorgs_v10.train_chain run --plan TR/plan.json > TR/chain.log 2>&1 &
    python -m data_gen.interiorgs_v10.train_chain status --dir TR      # or read TR/PROGRESS.md

Steps, each started only after the previous one succeeded:

1. ``check_code``  the cluster checkout is the commit named in the plan (every tracked file) with no local
                   change besides the verl submodule;
2. ``render``      SFT frames, one sco card (pilot_sco.py), every frame of frames.jsonl present afterwards;
3. ``sft_data``    conversations (training_prep sft-records; with "scenes_root" a share of them get an inserted
                   blocked turn) and the LLaMA-Factory data and recipe (scripts/prepare_active_spatial_sft.py,
                   model saved every epoch; "keep_text" keeps the env's text; "sft_epochs" epochs).  With
                   "sft_groups" ({name: trajectory file in sft_dir}) one recipe per group, TR/lf_<name>;
4. ``sft``         Qwen2.5-VL-7B SFT, 8 cards per group (job_sco.py, sco_interiorgs_v10_train.sh sft), the
                   groups' jobs submitted together;
5. ``sft_eval``    round 10, with "eval_sets" ({name: folder with tasks.jsonl}): every epoch checkpoint of every
                   group on every set, one single-card evaluation job each (at most "max_eval_jobs" at a time);
6. ``pick_sft``    round 10: the checkpoint with the highest success on the set "select_by" (ties: the earlier
                   epoch, then the group named first) becomes the RL start model;
7. ``rl_config``   PPO config from that model (training_prep rl-config: a checkpoint every 25 updates; the
                   plan's "rl" options: history_mode, critic_warmup, test_freq, val_tasks, greedy_val);
8. ``rl``          PPO with the dense-score S0 recipe: render service on 1 card + trainer on 8 cards, up to the
                   plan's endpoint;
9. ``rl_extend``   optional (plan "extend_to"): unless validation success at the endpoint is more than 10 points
                   below the best earlier value (``keep_training``, round 8 rule), continue from the latest
                   checkpoint to "extend_to";
10. ``evaluate``   optional (plan "eval": {"dir" or the plan's "eval_sets", "every", "models", "options"}): one
                   single-card evaluation job per RL checkpoint every "every" updates and per extra model, on
                   every set, then the tables (SET/table.md).

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
STEPS = ("check_code", "render", "sft_data", "sft", "sft_eval", "pick_sft", "rl_config", "rl", "rl_extend", "evaluate")
STEP_ZH = {"check_code": "核对代码", "render": "渲染 SFT 画面", "sft_data": "SFT 对话和数据", "sft": "SFT（7B）",
           "sft_eval": "SFT 每个 epoch 评测", "pick_sft": "按验证集挑 SFT 检查点", "rl_config": "RL 配置",
           "rl": "RL（S0）", "rl_extend": "RL 续训（验证仍上升时）", "evaluate": "评测"}
CARDS = {"render": "1", "sft": "每组 8", "sft_eval": "每个 1", "rl": "1 + 8", "rl_extend": "1 + 8", "evaluate": "每个模型 1"}
POLL_S = 60.0
DEFAULT_MAX_EVAL_JOBS = 10


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
# validation success: the mean over the rollouts per val task (4 in rounds 6-8, 1 with greedy validation)
_VAL_KEY = re.compile(r"^val-aux/active_spatial/traj_success/mean@\d+$")
RL_KEYS = {"train_success": "episode/success_rate", "seconds_per_update": "timing_s/step"}


def rl_progress(log_text: str) -> Dict[str, Any]:
    """Per update, from verl's console lines "step:N - key:value - ...": training success (the episode success
    rate of the update's rollouts), validation success (mean over the rollouts per val task) and seconds."""
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
        for key, value in values.items():
            if _VAL_KEY.match(key):
                row["val_success"] = float(value)
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


def epoch_checkpoints(model_dir: Path) -> List[Tuple[int, Path]]:
    """(epoch number, folder) of the per-epoch checkpoints LLaMA-Factory left in the model folder
    (checkpoint-<step>, one per epoch with save_strategy epoch), in training order."""
    found = []
    for p in Path(model_dir).glob("checkpoint-*"):
        try:
            found.append((int(p.name.split("-")[-1]), p))
        except ValueError:
            continue
    return [(k + 1, p) for k, (_, p) in enumerate(sorted(found))]


def success_rate(results: Path) -> Optional[float]:
    """The success rate of an evaluation result folder (results.jsonl), None without complete results."""
    if not (results / "done.json").is_file() or not (results / "results.jsonl").is_file():
        return None
    from .evaluation import metrics

    rows = [json.loads(l) for l in (results / "results.jsonl").read_text().splitlines() if l.strip()]
    return float(metrics(rows)["success"]) if rows else None


def pick_best(scores: Dict[str, Optional[float]], order: Sequence[str]) -> Optional[str]:
    """The name with the highest score; ties go to the name earlier in ``order`` (earlier epoch, first group)."""
    ranked = [n for n in order if scores.get(n) is not None]
    if not ranked:
        return None
    return max(ranked, key=lambda n: (scores[n], -order.index(n)))


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

    # -------------------------------------------------------------- plan views
    def sft_groups(self) -> Dict[str, str]:
        """{group name: trajectory file in sft_dir}; the single legacy group is {"": "sft_trajectories.jsonl"}."""
        groups = self.plan.get("sft_groups")
        return dict(groups) if groups else {"": "sft_trajectories.jsonl"}

    def lf_dir(self, group: str) -> Path:
        return self.tr / (f"lf_{group}" if group else "lf")

    def eval_sets(self) -> Dict[str, Path]:
        sets = self.plan.get("eval_sets")
        if sets:
            return {k: Path(v) for k, v in sets.items()}
        ev = self.plan.get("eval") or {}
        return {"eval": Path(ev["dir"])} if ev.get("dir") else {}

    def sft_model(self) -> Path:
        """The RL start model: the picked SFT checkpoint, else the (single) final SFT model."""
        decision = self.state["steps"].get("pick_sft", {}).get("decision") or {}
        if decision.get("model"):
            return Path(decision["model"])
        return self.lf_dir("") / "model"

    # -------------------------------------------------------------- bookkeeping
    def save(self) -> None:
        self.state["updated_utc"] = now()
        groups = self.sft_groups()
        if list(groups) == [""]:
            self.state["sft"] = sft_progress(self.lf_dir("") / "model")
        else:
            self.state["sft_groups"] = {g: sft_progress(self.lf_dir(g) / "model") for g in groups}
        logs = sorted((self.tr / "rl_run" / "logs").glob("rl-*.log")) if (self.tr / "rl_run" / "logs").is_dir() else []
        if logs:
            self.state["rl"] = rl_progress("\n".join(p.read_text(errors="ignore") for p in logs))
            self.state["rl"]["val_keep_training"] = keep_training(self.state["rl"]["val_success"])
        self.tr.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False))
        tmp.replace(self.state_path)
        (self.tr / "PROGRESS.md").write_text(render_markdown(self.state), encoding="utf-8")

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

    def wait_all(self, step: str, runs: Dict[str, Path]) -> None:
        """Poll until every job of ``runs`` ({name: run folder}) finished; every job is deleted afterwards; a job
        that did not succeed fails the step (after the others ended too)."""
        from . import job_sco

        pending, ended = dict(runs), {}
        try:
            while pending:
                for name, run in list(pending.items()):
                    st = job_sco.status(run)
                    if st.get("finished") or st.get("state") in ("SUSPENDED", "NOT_FOUND"):
                        ended[name] = st.get("state")
                        job_sco.delete(run)
                        pending.pop(name)
                self.mark(step, jobs={**{n: "RUNNING" for n in pending}, **ended})
                if pending:
                    self.sleep(POLL_S)
        finally:  # a failed poll leaves no job behind
            for run in runs.values():
                if (run / "job" / "submission.json").exists():
                    job_sco.delete(run)
        bad = {n: s for n, s in ended.items() if s != "SUCCEEDED"}
        if bad:
            raise ChainError("jobs did not succeed: " + ", ".join(f"{n} ({s})" for n, s in sorted(bad.items())))

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
        records = [py, "-m", "data_gen.interiorgs_v10.training_prep", "sft-records", "--dir", str(self.sft_dir)]
        if self.plan.get("scenes_root"):  # round 10: inserted blocked turns need the scenes' collision detectors
            records += ["--scenes-root", str(self.plan["scenes_root"])]
        self.sh(records, log=self.tr / "logs" / "sft_records.log")
        for group, trajectory in self.sft_groups().items():
            if not (self.sft_dir / trajectory).is_file():
                raise ChainError(f"sft-records wrote no {trajectory} for group {group or 'default'}")
            lf = self.lf_dir(group)
            args = [py, "prepare_active_spatial_sft.py", "--trajectory", str(self.sft_dir / trajectory),
                    "--split-manifest", str(self.sft_dir / "split_manifest.json"),
                    "--task-registry", str(self.sft_dir / "task_registry.jsonl"), "--qa-probability", "0",
                    "--allow-unreviewed-trajectories", "--save-per-epoch", "--model", self.plan["model"],
                    "--epochs", str(int(self.plan.get("sft_epochs", 3))), "--out", str(lf)]
            if self.plan.get("keep_text"):
                args.append("--keep-text")
            self.sh(args, cwd=ROOT / "scripts", log=self.tr / "logs" / f"prepare_sft{'_' + group if group else ''}.log")
            if not (lf / "train.yaml").is_file():
                raise ChainError(f"prepare_active_spatial_sft.py wrote no train.yaml in {lf}")

    def sft(self) -> None:
        from . import job_sco

        runs = {}
        for group in self.sft_groups():
            run = self.tr / (f"sft_{group}" if group else "sft")
            run.mkdir(parents=True, exist_ok=True)
            if not (run / "job" / "submission.json").exists():
                job_sco.prepare(run, 8, ["sft", str(self.lf_dir(group))])
                job_sco.submit(run)
            runs[group or "sft"] = run
        self.mark("sft", job_ids={n: json.loads((r / "job" / "submission.json").read_text())["job_id"]
                                  for n, r in runs.items()})
        self.wait_all("sft", runs)
        for group in self.sft_groups():
            model = self.lf_dir(group) / "model"
            if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
                raise ChainError(f"no final SFT model in {model}")

    def eval_jobs(self, step: str, jobs: Dict[str, Tuple[Path, str, str, Optional[str], Sequence[str]]]) -> None:
        """Single-card evaluation jobs {run name: (eval set folder, result name, model, prompt format, options)},
        at most max_eval_jobs at a time; a result folder with done.json is not evaluated again; every job is
        deleted when it ends; incomplete results fail the step."""
        from . import job_sco

        jobs = {n: j for n, j in jobs.items() if not (j[0] / j[1] / "done.json").exists()}
        missing = [f"{n}: {j[2]}" for n, j in jobs.items() if not (Path(j[2]) / "config.json").is_file()]
        if missing:  # checked before any job is submitted
            raise ChainError("no model to evaluate for " + "; ".join(missing))
        limit = int(self.plan.get("max_eval_jobs", DEFAULT_MAX_EVAL_JOBS))
        queue = list(jobs.items())
        running: Dict[str, Path] = {}
        ended: Dict[str, str] = {}
        failed: List[str] = []
        try:
            while queue or running:
                while queue and len(running) < limit:
                    name, (eval_dir, result, model, prompt_format, options) = queue.pop(0)
                    run = self.tr / "eval_runs" / name
                    run.mkdir(parents=True, exist_ok=True)
                    if not (run / "job" / "submission.json").exists():
                        job_sco.prepare(run, 1, ["eval", str(eval_dir), result, str(model), prompt_format or "-", *options])
                        job_sco.submit(run)
                    running[name] = run
                for name, run in list(running.items()):
                    st = job_sco.status(run)
                    if st.get("finished") or st.get("state") in ("SUSPENDED", "NOT_FOUND"):
                        eval_dir, result = jobs[name][0], jobs[name][1]
                        if not (eval_dir / result / "done.json").exists():
                            failed.append(f"{name} ({st.get('state')})")
                        ended[name] = st.get("state")
                        job_sco.delete(run)
                        running.pop(name)
                self.mark(step, jobs={**{n: "RUNNING" for n in running}, **ended, "queued": len(queue)})
                if queue or running:
                    self.sleep(POLL_S)
        finally:  # a failed submission or poll leaves no job behind
            for run in running.values():
                if (run / "job" / "submission.json").exists():
                    job_sco.delete(run)
        if failed:
            raise ChainError("evaluation jobs without complete results: " + ", ".join(failed))

    def tables(self) -> None:
        for eval_dir in self.eval_sets().values():
            self.sh([sys.executable, "-m", "data_gen.interiorgs_v10.evaluation", "table", "--eval", str(eval_dir)],
                    log=self.tr / "logs" / f"eval_table_{eval_dir.name}.log")

    def sft_eval(self) -> None:
        sets = self.eval_sets()
        if not sets or not self.plan.get("sft_groups"):
            self.mark("sft_eval", note="not planned")
            return
        options = list((self.plan.get("eval") or {}).get("options") or [])
        jobs = {}
        for group in self.sft_groups():
            ckpts = epoch_checkpoints(self.lf_dir(group) / "model")
            if not ckpts:
                raise ChainError(f"no epoch checkpoints in {self.lf_dir(group) / 'model'}")
            for epoch, path in ckpts:
                for set_name, eval_dir in sets.items():
                    result = f"sft_{group}_e{epoch}"
                    jobs[f"{set_name}__{result}"] = (eval_dir, result, str(path), None, options)
        self.mark("sft_eval", planned=sorted(jobs))
        self.eval_jobs("sft_eval", jobs)
        self.tables()

    def pick_sft(self) -> None:
        sets = self.eval_sets()
        if not sets or not self.plan.get("sft_groups"):
            self.mark("pick_sft", note="not planned")
            return
        select_by = self.plan.get("select_by") or next(iter(sets))
        if select_by not in sets:
            raise ChainError(f"select_by {select_by!r} is not one of the eval sets {sorted(sets)}")
        candidates: Dict[str, str] = {}  # result name -> checkpoint, in tie-break order (epoch, then group)
        for group in self.sft_groups():
            for epoch, path in epoch_checkpoints(self.lf_dir(group) / "model"):
                candidates[f"sft_{group}_e{epoch}"] = str(path)
        order = sorted(candidates, key=lambda n: (int(n.rsplit("_e", 1)[1]), list(self.sft_groups()).index(n[4:].rsplit("_e", 1)[0])))
        scores = {n: success_rate(sets[select_by] / n) for n in order}
        if any(v is None for v in scores.values()):
            raise ChainError("missing validation results for " + ", ".join(n for n, v in scores.items() if v is None))
        best = pick_best(scores, order)
        others = {s: {n: success_rate(d / n) for n in order} for s, d in sets.items() if s != select_by}
        self.mark("pick_sft", decision={"select_by": select_by, "scores": scores, "best": best,
                                        "model": candidates[best], "other_sets": others})

    def rl_config(self) -> None:
        rl = self.plan.get("rl") or {}
        args = [sys.executable, "-m", "data_gen.interiorgs_v10.training_prep", "rl-config",
                "--rows", str(self.sft_dir / "rows.jsonl"), "--ppo-template", self.plan["ppo_template"],
                "--gs-root", self.plan["gs_root"], "--model", str(self.sft_model()),
                "--name", self.plan.get("rl_name", "v10_rl_s0"), "--run-root", str(self.tr / "rl_run"),
                "--val-limit", str(self.plan.get("val_limit", 64)), "--out", str(self.tr / "rl")]
        for key, flag in (("history_mode", "--history-mode"), ("critic_warmup", "--critic-warmup"),
                          ("test_freq", "--test-freq"), ("val_tasks", "--val-tasks")):
            if rl.get(key) is not None:
                args += [flag, str(rl[key])]
        if rl.get("greedy_val"):
            args.append("--greedy-val")
        self.sh(args, log=self.tr / "logs" / "rl_config.log")

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
        ev = self.plan.get("eval")
        sets = self.eval_sets()
        if not ev or not sets:
            self.mark("evaluate", note="not planned")
            return
        every = int(ev.get("every", 50))
        options = list(ev.get("options") or [])
        ckpts = self.tr / "rl_run" / "checkpoints"
        models: Dict[str, Tuple[str, Optional[str], Sequence[str]]] = {
            f"v10_rl_{s:03d}": (str(ckpts / f"global_step_{s}" / "actor" / "huggingface"), None, options)
            for s in range(every, self.final_update() + 1, every)}
        models.update({name: (m["model"], m.get("prompt_format"), list(m.get("options", options)))
                       for name, m in (ev.get("models") or {}).items()})
        single = len(sets) == 1 and not self.plan.get("eval_sets")  # the legacy layout: run names without the set
        jobs = {(name if single else f"{set_name}__{name}"): (eval_dir, name, model, prompt_format, opts)
                for set_name, eval_dir in sets.items() for name, (model, prompt_format, opts) in models.items()}
        self.eval_jobs("evaluate", jobs)
        self.tables()

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
        if not jobs and i.get("jobs"):
            running = [k for k, v in i["jobs"].items() if v == "RUNNING"]
            jobs = f"{len(running)} 个在跑，排队 {i['jobs'].get('queued', 0)}" if s in ("sft_eval", "evaluate") else \
                ", ".join(f"{k} {v}" for k, v in i["jobs"].items())
        note = i.get("error") or i.get("note") or ""
        if not note and i.get("decision"):
            d = i["decision"]
            if s == "rl_extend":
                to = d.get("extend_to")
                note = (f"验证成功率没有比之前最好的低 10 个百分点以上，续训到 {to} 次" if to
                        else "验证成功率比之前最好的低 10 个百分点以上，不续训")
            elif s == "pick_sft":
                note = f"按 {d.get('select_by')} 挑出 {d.get('best')}（成功率 {d['scores'].get(d.get('best'))}）"
        lines.append(f"| {STEP_ZH[s]} | {CARDS.get(s, 'CPU')} | {zh.get(i.get('state'), i.get('state'))} | "
                     f"{i.get('started_utc', '')} | {i.get('ended_utc', '')} | {jobs} | {note} |")
    sft = state.get("sft") or {}
    if sft:
        lines += ["", f"**SFT：** 第 {sft.get('current_steps')} / {sft.get('total_steps')} 步（epoch {sft.get('epoch')}），"
                      f"训练损失 {sft.get('loss')}，验证损失 {sft.get('eval_loss')}（第 {sft.get('eval_at_step')} 步），"
                      f"预计剩余 {sft.get('remaining_time')}"]
    for g, sft in (state.get("sft_groups") or {}).items():
        if sft:
            lines += ["", f"**SFT（{g}）：** 第 {sft.get('current_steps')} / {sft.get('total_steps')} 步（epoch {sft.get('epoch')}），"
                          f"训练损失 {sft.get('loss')}，验证损失 {sft.get('eval_loss')}（第 {sft.get('eval_at_step')} 步），"
                          f"预计剩余 {sft.get('remaining_time')}"]
    pick = state["steps"].get("pick_sft", {}).get("decision")
    if pick:
        lines += ["", f"**SFT 各检查点在 {pick['select_by']} 上的成功率：** "
                      + "，".join(f"{n} {v:.3f}" for n, v in pick["scores"].items() if v is not None)
                      + f"；选 **{pick['best']}** 作为 RL 起点"]
    rl = state.get("rl") or {}
    if rl:
        val = "，".join(f"{s} 次：{v:.3f}" for s, v in rl.get("val_success") or [])
        lines += ["", f"**RL：** 已完成 {rl.get('last_update')} 次更新；最近一次更新的训练成功率 {rl.get('train_success_last')}，"
                      f"最近 10 次平均 {rl.get('train_success_mean_last10')}；每次更新约 {rl.get('seconds_per_update_mean_last10')} 秒",
                  "", f"验证成功率（验证集每题轨迹的平均）：{val or '还没有'}"]
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
