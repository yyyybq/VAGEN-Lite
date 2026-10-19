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

Round 11: plan "rl_runs" ({name: {"reward_preset", "rl": {rl-config options}}}) configures and trains several RL
runs from the picked SFT checkpoint at the same time (each its own render service and 8-card trainer, folders
TR/rl_<name> and TR/rl_run_<name>); "evaluate" then evaluates every run's checkpoints (results <name>_<update>).
Plan "eval"."variants" ([{"suffix", "sets", "options", "models"}]) adds evaluation runs with other options (e.g.
sampled: "--samples 4 --temperature 0.8 --top-p 0.92") for the listed models ("sft_pick" = the picked SFT
checkpoint, "rl" = every RL checkpoint, or result names).  While RL runs, PROGRESS.md shows each run's training
success per task family in blocks of 25 updates (from the rollout dumps).  Plan "pool" picks the sco pool of
every job the chain submits (default zoetrope; round 11 moved to h800 when zoetrope was full, the user's call).

Round 11, second part (the user's decisions): the RL tasks must be ones SFT never trained on.  Plan "rl" options
"rows" (a rows.jsonl with reserve rows, sft-frames --include-reserve), "task_pool" ("reserve") and "exclude" (the
depth-check flags) reach rl-config, which also gets the SFT conversations and refuses any overlap.  Every 25
updates PROGRESS.md shows per run the training success (all, per family), the share of tasks whose 4 episodes both
succeed and fail ("mixed"), the total reward per episode, entropy and KL (block_stats).  Plan "rl"."signal_check"
({"at": 25, "success_min", "success_max", "mixed_min"}) checks every run at that update; if any run misses, the
runs are stopped and moved to TR/rl_attempts/, and RL restarts from plan "rl_fallback_model" (once); a second miss
stops the chain.  Plan "eval"."extra_sets" ({name: folder}) are evaluation sets only variants use, and variant
models may name SFT epoch checkpoints ("sft_<group>_e<k>").

Round 11, third part (the optimisation package): plan "rl" options "opt_package" ("p1"), "candidates",
"target_groups", "rollouts" reach rl-config; the 25-update blocks add the tasks dropped (all success / all
failure) and refilled per update, the clip fraction, gradient norm, seconds per update and the share of
unparseable answers.  Plan "rl"."guards" stop a run (only that run; its logs stay, its checkpoints are still
evaluated) on NaN in the trainer's main metrics, KL to the start model above kl_max, the share of unparseable
answers above invalid_max (mean of the last invalid_window updates), or training success below the start (mean
of the first start_updates) by more than success_drop in each of drop_window consecutive updates (guard_reason).

Round 12 (the user's decisions): an RL trainer that ends at Ray start-up (STARTUP_FAILURE_MARKERS in its log)
or writes nothing for plan "rl"."startup"."stall_minutes" (default 40) before its first update is deleted and
submitted again, at most "retries" (default 2) times per run (startup_problem; the render service keeps running);
PROGRESS.md lists every resubmission.  Plan "rl" options "standard" (a named configuration of
configs/rl_standards.json; the plan's "ppo_template" is then optional) and "seed" reach rl-config, and a run's
"reward_preset" only when the plan names one.  Plan "eval_only": true runs only check_code and evaluate (the
plan's "eval"."models" and variants on its sets; no RL checkpoints), e.g. to evaluate existing models on a new set.

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
EVAL_ONLY_STEPS = ("check_code", "evaluate")
# round 12: how a trainer that failed at Ray start-up ends its log (round 11, H800: "The current node timed out
# during startup", after "GCS cannot find the node ... registration may not be complete")
STARTUP_FAILURE_MARKERS = ("timed out during startup", "GCS cannot find the node", "Failed to connect to GCS",
                           "raylet failed to start")
STARTUP_DEFAULTS = {"retries": 2, "stall_minutes": 40}
_UPDATE_LINE = re.compile(r"step:\d+ - ")
_LOG_START = re.compile(r"-(\d{8}T\d{6}Z)-")


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


def family_success(rollout_dir: Path, block: int = 25) -> Dict[str, Dict[str, Any]]:
    """Training success per task family in blocks of ``block`` updates, from verl's rollout dumps
    (rollout_data/<update>.jsonl, one row per trajectory with task_type and traj_success):
    {"1-25": {family: {"success": rate, "episodes": n}}, ...}."""
    from collections import defaultdict

    cells: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for f in Path(rollout_dir).glob("*.jsonl") if Path(rollout_dir).is_dir() else []:
        try:
            step = int(f.stem)
        except ValueError:
            continue
        lo = (step - 1) // block * block + 1
        label = f"{lo}-{lo + block - 1}"
        for line in f.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "task_type" in r and "traj_success" in r:
                cells[label][str(r["task_type"])].append(float(r["traj_success"]))
    out = {}
    for label in sorted(cells, key=lambda x: int(x.split("-")[0])):
        out[label] = {fam: {"success": round(sum(v) / len(v), 4), "episodes": len(v)}
                      for fam, v in sorted(cells[label].items())}
    return out


BLOCK_LOG_KEYS = {"entropy": "actor/entropy", "kl": "actor/kl_loss", "ppo_kl": "actor/ppo_kl",
                  # round 11, the optimisation package
                  "clip": "actor/pg_clipfrac", "grad_norm": "actor/grad_norm", "seconds": "timing_s/step",
                  "invalid": "transition/invalid_action_rate/mean", "kept": "filter/mixed/kept",
                  "dropped_all_success": "filter/mixed/dropped_all_success",
                  "dropped_all_fail": "filter/mixed/dropped_all_fail", "refilled": "filter/mixed/refilled",
                  "shortfall": "filter/mixed/shortfall"}
GUARD_NAN_KEYS = ("actor/pg_loss", "actor/kl_loss", "actor/entropy", "actor/grad_norm", "actor/ppo_kl",
                  "critic/score/mean", "critic/advantages/max", "critic/advantages/min")


def guard_reason(log_text: str, guards: Dict[str, Any]) -> Optional[str]:
    """Round 11 (the user's stop conditions for the optimisation-package runs), from the trainer's console lines
    "step:N - key:value - ...": a reason to stop the run, or None."""
    per: Dict[int, Dict[str, float]] = {}
    for line in log_text.splitlines():
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        step, body = int(m.group(1)), m.group(2)
        for key in GUARD_NAN_KEYS:
            if re.search(re.escape(key) + r":(?:np\.float64\()?-?nan\b", body):
                return f"NaN in {key} at update {step}"
        per[step] = {k: float(v) for k, v in _METRIC.findall(body)}
    steps = sorted(s for s in per if s > 0)
    if not steps:
        return None
    kl = per[steps[-1]].get("actor/kl_loss")
    kl_max = float(guards.get("kl_max", 0.2))
    if kl is not None and kl > kl_max:
        return f"KL to the start model {kl:.3f} > {kl_max} at update {steps[-1]}"
    w = int(guards.get("invalid_window", 10))
    inv = [per[s]["transition/invalid_action_rate/mean"] for s in steps[-w:] if "transition/invalid_action_rate/mean" in per[s]]
    inv_max = float(guards.get("invalid_max", 0.05))
    if len(inv) >= w and sum(inv) / len(inv) > inv_max:
        return f"unparseable answers {sum(inv) / len(inv):.3f} > {inv_max} (mean of updates {steps[-w]}-{steps[-1]})"
    succ = [(s, per[s]["episode/success_rate"]) for s in steps if "episode/success_rate" in per[s]]
    n0, nd, drop = int(guards.get("start_updates", 5)), int(guards.get("drop_window", 50)), float(guards.get("success_drop", 0.10))
    if len(succ) >= n0 + nd:
        start = sum(v for _, v in succ[:n0]) / n0
        tail = succ[-nd:]
        if all(v < start - drop for _, v in tail):
            return (f"training success below the start ({start:.3f}, updates 1-{n0}) by more than {drop} in each of "
                    f"updates {tail[0][0]}-{tail[-1][0]}")
    return None


def block_stats(rollout_dir: Path, log_text: str = "", block: int = 25) -> Dict[str, Dict[str, Any]]:
    """Round 11, per block of ``block`` updates, from verl's rollout dumps (rollout_data/<update>.jsonl, one row per
    episode) and the trainer's console log: training success (all and per family), the share of tasks whose episodes
    in one update both succeed and fail ("mixed_share": the tasks a comparison within the group can learn from; a
    task is its scene and its gts within one update), the total reward per episode ("reward_mean", the dumps' score)
    and the means of the actor's entropy, its KL to the start model (actor/kl_loss) and the policy change per update
    (actor/ppo_kl).  {"1-25": {"updates", "episodes", "tasks", "success", "mixed_share", "reward_mean", "entropy",
    "kl", "ppo_kl", "families": {family: {"success", "episodes"}}}}."""
    from collections import defaultdict

    def label(step: int) -> str:
        lo = (step - 1) // block * block + 1
        return f"{lo}-{lo + block - 1}"

    eps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    steps: Dict[str, set] = defaultdict(set)
    for f in Path(rollout_dir).glob("*.jsonl") if Path(rollout_dir).is_dir() else []:
        try:
            step = int(f.stem)
        except ValueError:
            continue
        for line in f.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "traj_success" not in r:
                continue
            steps[label(step)].add(step)
            eps[label(step)].append({"step": step, "success": float(bool(r["traj_success"])),
                                     "score": float(r.get("score") or 0.0), "family": str(r.get("task_type")),
                                     "task": r.get("task_id") or (r.get("scene_id"), json.dumps(r.get("gts"), sort_keys=True))})
    logged: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for line in log_text.splitlines():
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        values = dict(_METRIC.findall(m.group(2)))
        for name, key in BLOCK_LOG_KEYS.items():
            if key in values:
                logged[label(int(m.group(1)))][name].append(float(values[key]))
    out = {}
    for lab in sorted(eps, key=lambda x: int(x.split("-")[0])):
        rows = eps[lab]
        groups: Dict[Any, List[float]] = defaultdict(list)
        fams: Dict[str, List[float]] = defaultdict(list)
        for e in rows:
            groups[(e["step"], e["task"] if isinstance(e["task"], str) else tuple(e["task"]))].append(e["success"])
            fams[e["family"]].append(e["success"])
        mixed = sum(1 for v in groups.values() if 0 < sum(v) < len(v))
        mean = lambda v: round(sum(v) / len(v), 4) if v else None  # noqa: E731
        out[lab] = {"updates": len(steps[lab]), "episodes": len(rows), "tasks": len(groups),
                    "success": mean([e["success"] for e in rows]), "mixed_share": round(mixed / len(groups), 4),
                    "reward_mean": mean([e["score"] for e in rows]),
                    **{k: mean(logged[lab].get(k, [])) for k in BLOCK_LOG_KEYS},
                    "families": {f: {"success": mean(v), "episodes": len(v)} for f, v in sorted(fams.items())}}
    return out


def startup_problem(log_text: str, state: Optional[str], finished: bool, quiet_s: Optional[float],
                    stall_s: float) -> Optional[str]:
    """Round 12 (the user's decision 5): why a trainer job needs resubmitting, or None.  Only before its first
    update: the job ended (not SUCCEEDED) with a Ray start-up failure in its log, or it is running and its log has
    not grown for ``stall_s`` seconds (``quiet_s``)."""
    if _UPDATE_LINE.search(log_text):
        return None
    if finished:
        if state == "SUCCEEDED":
            return None
        marker = next((m for m in STARTUP_FAILURE_MARKERS if m in log_text), None)
        return f"ended {state} at Ray start-up ({marker})" if marker else None
    if quiet_s is not None and quiet_s >= stall_s:
        return f"no output for {quiet_s / 60:.0f} minutes before the first update"
    return None


def job_logs(run_dir: Path, prefix: str, since_utc: Optional[str]) -> List[Path]:
    """The logs of one job (logs/<prefix>-<start UTC>-<id>-worker-0.log) started after its submission at
    ``since_utc`` (two minutes of clock slack); all of them when the submission time is unknown."""
    d = run_dir / "logs"
    if not d.is_dir():
        return []
    t0 = (datetime.datetime.fromisoformat(since_utc).timestamp() - 120) if since_utc else None
    out = []
    for p in sorted(d.glob(f"{prefix}-*.log")):
        m = _LOG_START.search(p.name)
        if t0 is not None and m and datetime.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(
                tzinfo=datetime.timezone.utc).timestamp() < t0:
            continue
        out.append(p)
    return out


class SignalCheckFailed(Exception):
    """An RL run missed the signal check (plan "rl"."signal_check")."""


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
    def __init__(self, plan: Dict[str, Any], sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time) -> None:
        self.plan = plan
        self.clock = clock
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

    def rl_runs(self) -> Dict[str, Dict[str, Any]]:
        """{name: {"rl_dir", "run_dir", "reward_preset", "rl"}}; the single legacy run is named "" (TR/rl, TR/rl_run)."""
        runs = self.plan.get("rl_runs")
        if not runs:
            return {"": {"rl_dir": self.tr / "rl", "run_dir": self.tr / "rl_run", "reward_preset": "S0",
                         "rl": dict(self.plan.get("rl") or {})}}
        return {name: {"rl_dir": self.tr / f"rl_{name}", "run_dir": self.tr / f"rl_run_{name}",
                       "reward_preset": spec.get("reward_preset"),  # None: rl-config's default or the standard's
                       "rl": {**dict(self.plan.get("rl") or {}), **dict(spec.get("rl") or {})}}
                for name, spec in runs.items()}

    def sft_model(self) -> Path:
        """The RL start model: the fallback after a missed signal check, else the picked SFT checkpoint, else the
        (single) final SFT model."""
        if self.state.get("rl_model_override"):
            return Path(self.state["rl_model_override"])
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
        if self.plan.get("rl_runs"):
            runs_state = {}
            for name, run in self.rl_runs().items():
                logs = sorted((run["run_dir"] / "logs").glob("rl-*.log")) if (run["run_dir"] / "logs").is_dir() else []
                if logs:
                    text = "\n".join(p.read_text(errors="ignore") for p in logs)
                    runs_state[name] = rl_progress(text)
                    blocks = block_stats(run["run_dir"] / "rollout_data", text)
                    runs_state[name]["blocks"] = {k: {x: y for x, y in v.items() if x != "families"}
                                                  for k, v in blocks.items()}
                    runs_state[name]["family_success"] = {k: v["families"] for k, v in blocks.items()}
            if runs_state:
                self.state["rl_runs"] = runs_state
        else:
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
                job_sco.prepare(run, 8, ["sft", str(self.lf_dir(group))], pool=self.plan.get("pool"))
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
                        job_sco.prepare(run, 1, ["eval", str(eval_dir), result, str(model), prompt_format or "-", *options],
                                        pool=self.plan.get("pool"))
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
        for name, run in self.rl_runs().items():
            rl = run["rl"]
            args = [sys.executable, "-m", "data_gen.interiorgs_v10.training_prep", "rl-config",
                    "--rows", str(rl.get("rows") or self.sft_dir / "rows.jsonl"),
                    "--gs-root", self.plan["gs_root"], "--model", str(self.sft_model()),
                    "--name", self.plan.get("rl_name", "v10_rl_s0") + (f"_{name}" if name else ""),
                    "--run-root", str(run["run_dir"]),
                    "--val-limit", str(self.plan.get("val_limit", 64)), "--out", str(run["rl_dir"])]
            if self.plan.get("ppo_template"):  # round 12: optional with a standard configuration
                args += ["--ppo-template", str(self.plan["ppo_template"])]
            for key, flag in (("standard", "--standard"), ("seed", "--seed"), ("history_mode", "--history-mode"), ("critic_warmup", "--critic-warmup"),
                              ("test_freq", "--test-freq"), ("val_tasks", "--val-tasks"),
                              ("train_batch_size", "--train-batch-size"),
                              ("ppo_mini_batch_size", "--ppo-mini-batch-size"), ("save_freq", "--save-freq")):
                if rl.get(key) is not None:
                    args += [flag, str(rl[key])]
            for key, flag in (("greedy_val", "--greedy-val"), ("no_internal_val", "--no-internal-val")):
                if rl.get(key):
                    args.append(flag)
            if self.plan.get("rl_runs") and run["reward_preset"]:
                args += ["--reward-preset", run["reward_preset"]]
            if rl.get("task_pool"):  # round 11: RL trains on tasks SFT never trained on, checked against the SFT data
                args += ["--task-pool", str(rl["task_pool"]), "--sft-trajectories",
                         *[str(self.sft_dir / t) for t in self.sft_groups().values()]]
            if rl.get("exclude"):
                args += ["--exclude", str(rl["exclude"])]
            for key, flag in (("opt_package", "--opt-package"), ("candidates", "--candidates"),
                              ("target_groups", "--target-groups"), ("rollouts", "--rollouts")):
                if rl.get(key) is not None:  # round 11: the optimisation package
                    args += [flag, str(rl[key])]
            self.sh(args, log=self.tr / "logs" / f"rl_config{'_' + name if name else ''}.log")

    def rl(self) -> None:
        if not self.plan.get("rl_runs"):
            self._ppo("rl", int(self.plan["endpoint"]))
            return
        while True:
            try:
                self._ppo_runs("rl", int(self.plan["endpoint"]))
                return
            except SignalCheckFailed as exc:
                fallback = self.plan.get("rl_fallback_model")
                if not fallback or self.state.get("rl_model_override"):
                    raise ChainError(f"signal check missed again, RL stopped: {exc}") from None
                attempt = int(self.state.get("rl_attempt", 1))
                self.archive_rl_runs(f"attempt{attempt}")
                self.state["rl_model_override"] = str(fallback)
                self.state["rl_attempt"] = attempt + 1
                self.mark("rl", note=f"signal check missed ({exc}); restarting from {fallback}")
                self.rl_config()

    def archive_rl_runs(self, tag: str) -> None:
        """Move every run's config and run folder to TR/rl_attempts/<tag>_<name>_{config,run} (a restart starts clean)."""
        dest = self.tr / "rl_attempts"
        dest.mkdir(parents=True, exist_ok=True)
        for name, run in self.rl_runs().items():
            for key, kind in (("rl_dir", "config"), ("run_dir", "run")):
                src = run[key]
                if src.exists():
                    target = dest / f"{tag}_{name or 'rl'}_{kind}"
                    k = 2
                    while target.exists():
                        target, k = dest / f"{tag}_{name or 'rl'}_{kind}_{k}", k + 1
                    src.rename(target)

    def signal_check(self, runs: Dict[str, Dict[str, Any]]) -> None:
        """Plan "rl"."signal_check": once every run has its first ``at`` updates, each run's training success must be
        within [success_min, success_max] and its mixed share at least mixed_min; the result is kept in
        state["signal_checks"]; a miss raises SignalCheckFailed.  Checked once per attempt."""
        check = (self.plan.get("rl") or {}).get("signal_check")
        if not check:
            return
        attempt = int(self.state.get("rl_attempt", 1))
        if any(c.get("attempt") == attempt for c in self.state.get("signal_checks", [])):
            return
        at = int(check.get("at", 25))
        stats = {}
        for name, run in runs.items():
            d = run["run_dir"] / "rollout_data"
            have = {int(f.stem) for f in d.glob("*.jsonl") if f.stem.isdigit()} if d.is_dir() else set()
            if not set(range(1, at + 1)) <= have:
                return
            b = block_stats(d, block=at)[f"1-{at}"]
            ok = (float(check["success_min"]) <= b["success"] <= float(check["success_max"])
                  and b["mixed_share"] >= float(check["mixed_min"]))
            stats[name] = {k: b[k] for k in ("success", "mixed_share", "episodes", "tasks", "reward_mean")}
            stats[name]["ok"] = bool(ok)
        rec = {"attempt": attempt, "model": str(self.sft_model()), "at": at, "check": dict(check), "runs": stats,
               "ok": all(s["ok"] for s in stats.values()), "utc": now()}
        self.state.setdefault("signal_checks", []).append(rec)
        self.save()
        if not rec["ok"]:
            raise SignalCheckFailed(f"update {at}, start {rec['model']}: " + "; ".join(
                f"{n} success {s['success']:.3f} mixed {s['mixed_share']:.3f}" for n, s in stats.items()))

    def _ppo_runs(self, step: str, endpoint: int) -> None:
        """Round 11: every run of plan "rl_runs" at the same time (its render service on 1 card, PPO on 8), up to
        ``endpoint``; a run whose endpoint checkpoint exists is skipped.  A failed run does not stop the others;
        the step fails after all ended if any failed."""
        from . import job_sco

        runs = {n: r for n, r in self.rl_runs().items()
                if not (r["run_dir"] / "checkpoints" / f"global_step_{endpoint}").is_dir()}
        for name, run in runs.items():
            d = run["run_dir"]
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "job_renderer" / "submission.json").exists():
                for f in ("STOP_RENDERER", "renderer_endpoint.txt", "renderer_endpoint.tmp"):
                    (d / f).unlink(missing_ok=True)
                job_sco.prepare(d, 1, ["renderer"], job_dir="job_renderer", pool=self.plan.get("pool"))
                job_sco.submit(d, job_dir="job_renderer")
            if not (d / "job_rl" / "submission.json").exists():
                job_sco.prepare(d, 8, ["rl", str(run["rl_dir"]), str(endpoint)], job_dir="job_rl", pool=self.plan.get("pool"))
                job_sco.submit(d, job_dir="job_rl")
        self.mark(step, job_ids={f"{n}/{j}": json.loads((r["run_dir"] / j / "submission.json").read_text())["job_id"]
                                 for n, r in runs.items() for j in ("job_renderer", "job_rl")})
        active, ended = dict(runs), {}
        try:
            while active:
                states = {}
                for name, run in list(active.items()):
                    d = run["run_dir"]
                    st_rl, st_r = job_sco.status(d, "job_rl"), job_sco.status(d, "job_renderer")
                    states[f"{name}/rl"], states[f"{name}/renderer"] = st_rl.get("state"), st_r.get("state")
                    problem = self.startup_check(name, run, st_rl)  # round 12: Ray start-up failure or silence
                    if problem:
                        retries = self.state.setdefault("startup_retries", {}).setdefault(name, [])
                        limit = int(self.startup_settings(run)["retries"])
                        if len(retries) < limit:
                            job_sco.delete(d, "job_rl")
                            retries.append({"utc": now(), "job_id": st_rl.get("job_id"), "reason": problem})
                            job_sco.prepare(d, 8, ["rl", str(run["rl_dir"]), str(endpoint)], job_dir="job_rl",
                                            pool=self.plan.get("pool"))
                            job_sco.submit(d, job_dir="job_rl")
                            retries[-1]["new_job_id"] = json.loads((d / "job_rl" / "submission.json").read_text())["job_id"]
                            self.save()
                            continue
                        if not st_rl.get("finished"):  # silent again after the last resubmission: give up on it
                            (d / "STOP_RENDERER").touch()
                            job_sco.delete(d, "job_rl")
                            ended[name] = f"start-up: {problem} (after {limit} resubmissions)"
                            active.pop(name)
                            continue
                    if st_rl.get("finished") or st_rl.get("state") in ("NOT_FOUND", "SUSPENDED"):
                        (d / "STOP_RENDERER").touch()
                        ended[name] = st_rl.get("state")
                        active.pop(name)
                    elif st_r.get("finished") or st_r.get("state") in ("NOT_FOUND", "SUSPENDED"):
                        (d / "STOP_RENDERER").touch()  # the render service died: the trainer cannot go on
                        job_sco.delete(d, "job_rl")
                        ended[name] = f"renderer {st_r.get('state')}"
                        active.pop(name)
                for name, run in list(active.items()):  # round 11: the stop conditions (plan rl.guards)
                    guards = run["rl"].get("guards")
                    logs = sorted((run["run_dir"] / "logs").glob("rl-*.log")) if guards else []
                    reason = guard_reason("\n".join(p.read_text(errors="ignore") for p in logs), guards) if logs else None
                    if reason:
                        (run["run_dir"] / "STOP_RENDERER").touch()
                        job_sco.delete(run["run_dir"], "job_rl")
                        self.state.setdefault("guard_stops", {})[name] = {"reason": reason, "utc": now()}
                        ended[name] = f"stopped: {reason}"
                        active.pop(name)
                self.mark(step, jobs={**states, **{f"{n}/ended": v for n, v in ended.items()}})
                self.signal_check(runs)
                if active:
                    self.sleep(POLL_S)
        finally:
            for run in runs.values():
                (run["run_dir"] / "STOP_RENDERER").touch()
            for _ in range(20):  # the render services exit within a minute of STOP_RENDERER
                if all(job_sco.status(r["run_dir"], "job_renderer").get("finished") for r in runs.values()):
                    break
                self.sleep(15)
            for run in runs.values():
                for j in ("job_rl", "job_renderer"):
                    if (run["run_dir"] / j / "submission.json").exists():
                        job_sco.delete(run["run_dir"], j)
        stopped = self.state.get("guard_stops") or {}  # stopped by a guard: not a failure of the step
        bad = [f"{n} ({ended.get(n)})" for n, r in runs.items()
               if not (r["run_dir"] / "checkpoints" / f"global_step_{endpoint}").is_dir() and n not in stopped]
        if bad:
            raise ChainError("RL runs without a checkpoint at update " + str(endpoint) + ": " + ", ".join(bad))

    def startup_settings(self, run: Dict[str, Any]) -> Dict[str, Any]:
        return {**STARTUP_DEFAULTS, **dict(run["rl"].get("startup") or {})}

    def startup_check(self, name: str, run: Dict[str, Any], st: Dict[str, Any]) -> Optional[str]:
        """Round 12: startup_problem for the run's current trainer job (its logs since its submission; silence
        counted from the later of its last log write and the first poll that saw it running)."""
        d = run["run_dir"]
        receipt = d / "job_rl" / "submission.json"
        if not receipt.exists():
            return None
        sub = json.loads(receipt.read_text())
        logs = job_logs(d, "rl", sub.get("utc"))
        text = "\n".join(p.read_text(errors="ignore") for p in logs)
        quiet = None
        if st.get("state") == "RUNNING":
            seen = self.state.setdefault("startup_seen_running", {})
            key = str(st.get("job_id") or sub.get("job_id"))
            seen.setdefault(key, self.clock())
            last = max([seen[key]] + [p.stat().st_mtime for p in logs])
            quiet = self.clock() - last
        return startup_problem(text, st.get("state"), bool(st.get("finished")), quiet,
                               float(self.startup_settings(run)["stall_minutes"]) * 60)

    def _ppo(self, step: str, endpoint: int, suffix: str = "") -> None:
        """Render service (1 card) + PPO (8 cards) to ``endpoint``; resumes from the latest checkpoint."""
        from . import job_sco

        run = self.tr / "rl_run"
        run.mkdir(parents=True, exist_ok=True)
        jr, jl = "job_renderer" + suffix, "job_rl" + suffix
        if not (run / jr / "submission.json").exists():
            for f in ("STOP_RENDERER", "renderer_endpoint.txt", "renderer_endpoint.tmp"):
                (run / f).unlink(missing_ok=True)
            job_sco.prepare(run, 1, ["renderer"], job_dir=jr, pool=self.plan.get("pool"))
            job_sco.submit(run, job_dir=jr)
        if not (run / jl / "submission.json").exists():
            job_sco.prepare(run, 8, ["rl", str(self.tr / "rl"), str(endpoint)], job_dir=jl, pool=self.plan.get("pool"))
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
        models: Dict[str, Tuple[str, Optional[str], Sequence[str]]] = {}
        rl_names: List[str] = []
        for run_name, run in ({} if self.plan.get("eval_only") else self.rl_runs()).items():
            ckpts = run["run_dir"] / "checkpoints"
            for s in range(every, self.final_update() + 1, every):
                name = f"{run_name}_{s:03d}" if run_name else f"v10_rl_{s:03d}"
                hf = ckpts / f"global_step_{s}" / "actor" / "huggingface"
                if run_name in (self.state.get("guard_stops") or {}) and not hf.is_dir():
                    continue  # a run stopped by a guard: the checkpoints it reached
                models[name] = (str(hf), None, options)
                rl_names.append(name)
        models.update({name: (m["model"], m.get("prompt_format"), list(m.get("options", options)))
                       for name, m in (ev.get("models") or {}).items()})
        single = len(sets) == 1 and not self.plan.get("eval_sets")  # the legacy layout: run names without the set
        jobs = {(name if single else f"{set_name}__{name}"): (eval_dir, name, model, prompt_format, opts)
                for set_name, eval_dir in sets.items() for name, (model, prompt_format, opts) in models.items()}
        extra = {k: Path(v) for k, v in (ev.get("extra_sets") or {}).items()}  # round 11: sets only variants use
        jobs.update(self.variant_jobs(ev, {**sets, **extra}, models, rl_names))
        self.eval_jobs("evaluate", jobs)
        self.tables()

    def variant_jobs(self, ev: Dict[str, Any], sets: Dict[str, Path],
                     models: Dict[str, Tuple[str, Optional[str], Sequence[str]]],
                     rl_names: Sequence[str]) -> Dict[str, Tuple[Path, str, str, Optional[str], Sequence[str]]]:
        """Round 11: extra runs of listed models with other options (plan "eval"."variants"), e.g. sampled."""
        pick = (self.state["steps"].get("pick_sft", {}).get("decision") or {})
        jobs = {}
        for v in ev.get("variants") or []:
            names: List[str] = []
            for m in v.get("models") or []:
                if m == "rl":
                    names += list(rl_names)
                elif m == "sft_pick":
                    if pick.get("best"):
                        names.append(pick["best"])
                else:
                    names.append(m)
            for name in names:
                if name in models:
                    model, prompt_format, _ = models[name]
                elif pick.get("best") == name:
                    model, prompt_format = pick["model"], None
                elif self.sft_epoch_model(name) is not None:  # round 11: an SFT epoch checkpoint by result name
                    model, prompt_format = str(self.sft_epoch_model(name)), None
                else:
                    raise ChainError(f"variant {v.get('suffix')}: no model named {name}")
                for set_name in v.get("sets") or list(sets):
                    result = name + v["suffix"]
                    jobs[f"{set_name}__{result}"] = (sets[set_name], result, model, prompt_format, list(v.get("options") or []))
        return jobs

    def sft_epoch_model(self, name: str) -> Optional[Path]:
        """The checkpoint of an SFT epoch by its result name ("sft_<group>_e<k>", "sft_e<k>" for the single group)."""
        for group in self.sft_groups():
            prefix = f"sft_{group}_e" if group else "sft_e"
            if name.startswith(prefix) and name[len(prefix):].isdigit():
                found = dict(epoch_checkpoints(self.lf_dir(group) / "model"))
                return found.get(int(name[len(prefix):]))
        return None

    # -------------------------------------------------------------- driver
    def run(self) -> int:
        self.state["chain_pid"] = os.getpid()
        self.state["state"] = "running"
        self.save()
        for step in STEPS:
            info = self.state["steps"][step]
            if info.get("state") == "done":
                continue
            if self.plan.get("eval_only") and step not in EVAL_ONLY_STEPS:  # round 12: evaluation of existing models
                self.mark(step, state="done", note="not planned (eval_only)")
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
                note = f"按 {d.get('select_by')} 挑出 {d.get('best')}（成功率 {(d.get('scores') or {}).get(d.get('best'))}）"
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
                      + "，".join(f"{n} {v:.3f}" for n, v in (pick.get("scores") or {}).items() if v is not None)
                      + f"；选 **{pick['best']}** 作为 RL 起点"]
    for name, rl in (state.get("rl_runs") or {}).items():
        lines += ["", f"**RL {name}：** 已完成 {rl.get('last_update')} 次更新；最近一次更新的训练成功率 {rl.get('train_success_last')}，"
                      f"最近 10 次平均 {rl.get('train_success_mean_last10')}；每次更新约 {rl.get('seconds_per_update_mean_last10')} 秒"]
        blocks = rl.get("blocks") or {}
        if blocks:
            fmt = lambda v, p=3: "—" if v is None else f"{v:.{p}f}"  # noqa: E731
            lines += ["", "| 更新 | 训练成功率 | 有成有败的题 | 每条轨迹的总奖励 | 熵 | KL（对起点模型） | 每次更新的策略变化 "
                          "| clip 比例 | 梯度大小 | 每次更新秒数 | 格式错误 | 每次更新：保留 / 丢掉（全成功、全失败）/ 补抽 / 缺 |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for label, b in blocks.items():
                filt = ("—" if b.get("kept") is None else
                        f"{fmt(b.get('kept'), 1)} / {fmt(b.get('dropped_all_success'), 1)}、{fmt(b.get('dropped_all_fail'), 1)} / "
                        f"{fmt(b.get('refilled'), 1)} / {fmt(b.get('shortfall'), 1)}")
                lines.append(f"| {label}（{b.get('updates')} 次） | {fmt(b.get('success'))} | {fmt(b.get('mixed_share'))}"
                             f"（{b.get('tasks')} 题） | {fmt(b.get('reward_mean'))} | {fmt(b.get('entropy'))} | "
                             f"{fmt(b.get('kl'), 5)} | {fmt(b.get('ppo_kl'), 5)} | {fmt(b.get('clip'), 4)} | "
                             f"{fmt(b.get('grad_norm'), 2)} | {fmt(b.get('seconds'), 0)} | {fmt(b.get('invalid'))} | {filt} |")
        fam = rl.get("family_success") or {}
        if fam:
            families = sorted({f for b in fam.values() for f in b})
            lines += ["", "| 更新 | " + " | ".join(families) + " |", "| --- | " + " | ".join("---" for _ in families) + " |"]
            for label, cells in fam.items():
                lines.append(f"| {label} | " + " | ".join(
                    f"{cells[f]['success']:.2f}（{cells[f]['episodes']}）" if f in cells else "—" for f in families) + " |")
    for name, g in (state.get("guard_stops") or {}).items():
        lines += ["", f"**RL {name} 被自动停止**（{g.get('utc')}）：{g.get('reason')}"]
    for name, retries in (state.get("startup_retries") or {}).items():
        for k, r in enumerate(retries, 1):
            lines += ["", f"**RL {name} 启动失败，第 {k} 次自动重新提交**（{r.get('utc')}）：{r.get('job_id')} → "
                          f"{r.get('new_job_id')}；原因：{r.get('reason')}"]
    for c in state.get("signal_checks") or []:
        lines += ["", f"**第 {c['attempt']} 次 RL 的信号检查（第 {c['at']} 次更新，起点 {c['model']}）：**"
                      f"{'通过' if c['ok'] else '没通过'}；" + "；".join(
                          f"{n} 训练成功率 {s['success']:.3f}、有成有败的题 {s['mixed_share']:.3f}"
                          f"{'' if s['ok'] else '（不满足）'}" for n, s in c["runs"].items())]
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
