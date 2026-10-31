"""Replay an episode's executed actions in OmniGibson and record BDDL goal-literal truth (offline).

STATUS: NOT RUN. Written against BEHAVIOR-1K v3.9.3-post2 (a8247a8c) and unit-tested only with a fake
evaluator (tests/test_data_diag_replay.py); it has never touched a simulator. Run it on server 105 in
the behavior conda env. It must never be imported by runtime code (tests/test_import_guards.py), and
every simulator import (omnigibson, torch, omegaconf) stays inside ``main()`` so the module can be
imported, AST-scanned and asked for --help without the simulator.

    # behavior env on 105, from the b1k_agent checkout; B1K_ENV_SETUP / the b1k_runs helpers may wrap it
    [ -n "$B1K_ENV_SETUP" ] && source "$B1K_ENV_SETUP"
    PYTHONPATH=. conda run -p /nas/baiqiao/envs/behavior --no-capture-output python -m diagnostics.replay_truth \
        --episode-dir ~/b1k_local_logs/<run_id>/episodes/<episode_id> \
        --official-json /nas/baiqiao/b1k_runs/logs/<run_id>/eval_output/json/<task>_<iid>_0.json \
        --out /nas/baiqiao/b1k_runs/logs/<run_id>/truth/<episode_id>.truth.jsonl [--every 30] [--gpu auto]
    ... --dry-run    # check inputs and print the plan; imports nothing from the simulator

How it mirrors the official evaluator (OmniGibson/omnigibson/eval/eval.py main + evaluator.py):
  * the same cfg as eval.py (task, mode, robot config, env wrapper, partial_scene_load, seed_everything
    with DEFAULT_EVAL_SEED; the wrapper defaults to the episode's eval.launch "wrapper" tag, and
    --robot-config must be passed again when the original run had one) and the same BatchedEvaluator:
    env creation, robot eval settings, and load_batch (TRO instance state, 25 settle steps, env.reset)
    are inherited, not re-implemented;
  * a subclass replaces only the policy (actions from <episode>/steps.jsonl "a", 23 floats, in step
    order) and hooks _step_fn / load_batch to record truth; num_envs = 1;
  * cfg.max_steps = n_actions - 1, so the official Timeout (fires when episode_steps >= max_steps,
    i.e. on env step max_steps + 1) stops the replay after exactly n_actions steps, unless the goal
    is reached first;
  * run() writes the replay's own official-format result JSON (<replay-output-dir>/json/, never the
    original's eval_output: refused when it would overwrite --official-json), which the footer
    compares with the original official JSON (q_score.final, steps, success; steps only for a
    2026/eval failure record, whose success / q are forced).

Truth (diagnostics/truth.py format): BehaviorTask.get_goal_option_satisfaction(env_idx) through the
evaluator's InstanceEnvAccessor (per goal option, per grounded literal), task success, and the Q
the official TaskMetric would report now (metrics/task_metric.py compute_q_score). Recorded after
reset (step 0), every --every steps, at every step where the agent made a decision or a model call
(from events.jsonl), and at the final step. Record k = the state after k actions = the state the
agent observed at its step k.

Limits, to keep in mind when reading the output:
  * the simulator is not deterministic (docs/challenge/evaluation.md), and agent/recorder.py rounds
    actions to 5 decimals, so a replay can diverge from the original; ``consistent`` in the footer
    says whether the replay reproduced the official q_score.final and steps (labels from a
    divergent replay are skipped by label_distill.py unless asked);
  * the original may have run batched (num_envs > 1); the replay runs alone;
  * in a batched run the agent keeps stepping an env the evaluator already finished (it sends that
    env's frozen last observation), so steps.jsonl can be longer than the official episode; with
    --official-json only its ``steps`` actions are replayed;
  * one episode per process (scene loading dominates the cost; batching episodes of one task is a
    possible extension).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from typing import Any, Callable, Optional, Sequence

import numpy as np

from diagnostics.truth import TruthWriter, canonical_literal, compute_q_score
from eval.launch import DEFAULT_APPDATA, WRAPPERS

ACTION_DIM = 23  # R1Pro (eval_utils.py ACTION_QPOS_INDICES["R1Pro"])
DEFAULT_WRAPPER = "omnigibson.eval.wrappers.DefaultWrapper"
DECISION_KINDS = ("decision", "provider_call")
_EP_RE = re.compile(r"-ep(\d+)-env(\d+)$")
_TEST_START = {"public_test": 301, "hidden_test": 321}  # eval_utils.TEST_INSTANCE_IDS = range(301, 341)


class ReplayInputError(ValueError):
    pass


# ------------------------------------------------------------------------------------ inputs (pure)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_actions(steps_path: str, limit: Optional[int] = None) -> np.ndarray:
    """(n, 23) float32 actions from steps.jsonl, ordered by step; the steps must be 0..n-1 without gaps."""
    by_step: dict[int, list] = {}
    with open(steps_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            step, a = int(rec["step"]), rec.get("a")
            if not isinstance(a, list) or len(a) != ACTION_DIM:
                raise ReplayInputError(f"{steps_path}:{n}: action must be {ACTION_DIM} floats")
            if step in by_step:
                raise ReplayInputError(f"{steps_path}:{n}: step {step} logged twice")
            by_step[step] = a
    steps = sorted(by_step)
    if not steps:
        raise ReplayInputError(f"{steps_path}: no actions")
    if steps != list(range(len(steps))):
        missing = sorted(set(range(steps[-1] + 1)) - set(steps))[:10]
        raise ReplayInputError(f"{steps_path}: steps must be 0..n-1 without gaps (missing {missing})")
    if limit is not None:
        if limit > len(steps):
            raise ReplayInputError(f"{steps_path}: the official episode has {limit} steps, the log only {len(steps)}")
        steps = steps[:limit]
    return np.asarray([by_step[s] for s in steps], dtype=np.float32)


def decision_steps(events_path: Optional[str]) -> set[int]:
    """Steps of the agent's decisions and model calls (events.jsonl)."""
    out: set[int] = set()
    if not events_path or not os.path.isfile(events_path):
        return out
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") in DECISION_KINDS and isinstance(ev.get("step"), int) and ev["step"] >= 0:
                out.add(ev["step"])
    return out


def record_steps(n_actions: int, every: int, extra: Sequence[int] = ()) -> set[int]:
    steps = set(range(0, n_actions + 1, every)) if every > 0 else {0}
    steps |= {int(s) for s in extra if 0 <= int(s) <= n_actions}
    steps.add(n_actions)
    return steps


def read_official(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    q = (rec.get("q_score") or {}).get("final")
    return {"path": os.path.abspath(path), "task": rec.get("task"), "instance_id": rec.get("instance_id"),
            "rollout_id": rec.get("rollout_id"), "steps": rec.get("steps"), "success": rec.get("success"),
            "q_final": None if q is None else float(q), "failure_reason": rec.get("failure_reason")}


def mode_for_instance(instance_id: int) -> str:
    if 301 <= instance_id <= 320:
        return "public_test"
    if 321 <= instance_id <= 340:
        return "hidden_test"
    return "train"


def episode_identity(episode_dir: str) -> dict:
    """task / instance id / mode / env wrapper / episode id from the episode's summary.json tags
    (eval/launch.py passes task, mode, instance_ids in env order, wrapper) and its id "<run>-epNNN-envK"."""
    out: dict[str, Any] = {"episode_id": os.path.basename(os.path.normpath(episode_dir))}
    path = os.path.join(episode_dir, "summary.json")
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    tags = s.get("tags") or {}
    out["episode_id"] = str(s.get("episode_id") or out["episode_id"])
    out["run_id"] = s.get("run_id")
    out["task"] = tags.get("task") or s.get("task_name")
    out["mode"] = tags.get("mode")
    out["wrapper"] = tags.get("wrapper")
    m = _EP_RE.search(out["episode_id"])
    env = int(m.group(2)) if m else 0
    raw = tags.get("instance_id") or tags.get("instance") or tags.get("instance_ids")
    ids = [int(x) for x in str(raw).split(",") if x.strip().lstrip("-").isdigit()] if raw is not None else []
    if len(ids) == 1:
        out["instance_id"] = ids[0]
    elif env < len(ids):
        out["instance_id"] = ids[env]
    return out


def plan(args: argparse.Namespace) -> dict:
    """Resolve and check every input; no simulator involved."""
    ep_dir = os.path.abspath(os.path.expanduser(args.episode_dir))
    ident = episode_identity(ep_dir)
    official = read_official(args.official_json)
    task = args.task or ident.get("task") or (official or {}).get("task")
    iid = args.instance_id if args.instance_id is not None else ident.get("instance_id")
    if iid is None and official is not None:
        iid = official.get("instance_id")
    if not task or iid is None:
        raise ReplayInputError("task and instance id unknown: pass --task and --instance-id (resolved id, e.g. 311)")
    iid = int(iid)
    mode = args.mode or ident.get("mode") or mode_for_instance(iid)
    if mode in _TEST_START and not _TEST_START[mode] <= iid < _TEST_START[mode] + 20:
        raise ReplayInputError(f"instance id {iid} is not a {mode} instance (resolved ids {_TEST_START[mode]}-"
                               f"{_TEST_START[mode] + 19}); pass the resolved id, not the index")
    if official is not None:
        if official.get("task") not in (None, task) or official.get("instance_id") not in (None, iid):
            raise ReplayInputError(f"official JSON is for {official.get('task')}/{official.get('instance_id')}, "
                                   f"not {task}/{iid}")
    steps_path = os.path.join(ep_dir, "steps.jsonl")
    limit = int(official["steps"]) if official and official.get("steps") is not None else None
    actions = load_actions(steps_path, limit)
    with open(steps_path, encoding="utf-8") as f:
        steps_in_log = sum(1 for line in f if line.strip())
    if len(actions) < 2:
        raise ReplayInputError("need at least 2 actions to replay")
    events_path = os.path.join(ep_dir, "events.jsonl")
    at = record_steps(len(actions), args.every, decision_steps(events_path) if args.at_decisions else ())
    out = os.path.abspath(os.path.expanduser(args.out))
    replay_dir = os.path.abspath(os.path.expanduser(args.replay_output_dir or os.path.join(
        os.path.dirname(out), "replay_eval", ident["episode_id"])))
    replay_json = os.path.join(replay_dir, "json", f"{task}_{iid}_0.json")
    if official is not None and os.path.realpath(os.path.dirname(replay_json)) == os.path.realpath(
            os.path.dirname(official["path"])):
        raise ReplayInputError(f"--replay-output-dir {replay_dir} would overwrite the official result "
                               f"{official['path']}; the replay's JSON must go elsewhere")
    wrapper = args.env_wrapper or ident.get("wrapper") or DEFAULT_WRAPPER
    return {"episode_id": ident["episode_id"], "run_id": ident.get("run_id"), "task": task, "instance_id": iid,
            "mode": mode, "steps_path": steps_path, "steps_sha256": _sha256(steps_path),
            "steps_in_log": steps_in_log, "actions": actions, "env_wrapper": WRAPPERS.get(wrapper, wrapper),
            "n_actions": int(len(actions)), "record_at": at, "official": official, "out": out,
            "replay_output_dir": replay_dir, "events_path": events_path if os.path.isfile(events_path) else None}


# ------------------------------------------------------------------------------------ replay pieces


class ReplayPolicy:
    """Stands in for the evaluator's policy: returns the logged actions in order (batch of one env)."""

    def __init__(self, actions: np.ndarray, to_tensor: Callable[[np.ndarray], Any] = lambda x: x):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.to_tensor = to_tensor
        self.i = 0
        self.overrun = 0

    def set_action_dim(self, action_dim: int) -> None:
        if int(action_dim) != self.actions.shape[1]:
            raise ReplayInputError(f"robot action_dim {action_dim} != logged action width {self.actions.shape[1]}")

    def reset(self) -> None:
        self.i = 0

    def forward(self, obs: Any = None, *args: Any, **kwargs: Any) -> Any:
        if self.i >= len(self.actions):
            self.overrun += 1  # cannot happen with cfg.max_steps = n - 1; repeats the last action if it does
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return self.to_tensor(a[None])


def goal_option_literals(task: Any) -> list[list[str]]:
    """Canonical literal texts of BehaviorTask.ground_goal_state_options (lists of bddl HEAD nodes)."""
    return [[canonical_literal(head.body) for head in option] for option in task.ground_goal_state_options]


def make_replay_evaluator(base_cls: type, policy: ReplayPolicy, make_writer: Callable[[Any], TruthWriter],
                          record_at: set) -> type:
    """Subclass of the official BatchedEvaluator (or a test fake) that replays and records truth."""

    class ReplayEvaluator(base_cls):
        def load_policy(self):
            if hasattr(policy, "set_action_dim"):
                policy.set_action_dim(self.instance_eval_states[0].env_accessor.robot.action_dim)
            return policy

        def load_batch(self, env_idx_to_instance, *args, **kwargs):
            super().load_batch(env_idx_to_instance, *args, **kwargs)
            self.replay_step = 0
            accessor = self.instance_eval_states[0].env_accessor
            self.truth_writer = make_writer(self.env.task)
            self.truth_writer.start(accessor.get_goal_option_satisfaction(), accessor.success)

        def _step_fn(self, active_env_indices):
            terminated, truncated = super()._step_fn(active_env_indices)
            self.replay_step += 1
            term, trunc = bool(terminated[0]), bool(truncated[0])
            if self.replay_step in record_at or term or trunc:
                accessor = self.instance_eval_states[0].env_accessor
                self.truth_writer.record(self.replay_step, accessor.get_goal_option_satisfaction(), accessor.success,
                                         terminated=term, truncated=trunc)
            return terminated, truncated

    return ReplayEvaluator


def footer_fields(p: dict, result: Optional[dict], policy: ReplayPolicy) -> dict:
    """``consistent``: the replay reproduced the official steps, success and q_score.final. A 2026/eval
    failure record (failure_reason set) forces success = false and q = 0 (evaluator.py record_fn), so
    only its steps can be compared: a step mismatch is false, a match is None (unknown)."""
    official = p.get("official") or {}
    replay_q = None if result is None else (result.get("q_score") or {}).get("final")
    consistent = None
    if result is not None and official.get("failure_reason"):
        consistent = None if result.get("steps") == official.get("steps") else False
    elif replay_q is not None and official.get("q_final") is not None:
        consistent = (abs(float(replay_q) - official["q_final"]) < 1e-6 and result.get("steps") == official.get("steps")
                      and bool(result.get("success")) == bool(official.get("success")))
    return {"replay_steps": None if result is None else result.get("steps"),
            "replay_success": None if result is None else bool(result.get("success")),
            "replay_q_final": None if replay_q is None else float(replay_q),
            "official_steps": official.get("steps"), "official_q_final": official.get("q_final"),
            "official_success": official.get("success"), "official_failure_reason": official.get("failure_reason"),
            "consistent": consistent, "overrun_actions": policy.overrun}


def replay_json_path(p: dict) -> str:
    return os.path.join(p["replay_output_dir"], "json", f"{p['task']}_{p['instance_id']}_0.json")


def run_replay(p: dict, base_cls: type, make_cfg: Callable[[dict], Any], to_tensor: Callable = lambda x: x,
               q_fn: Callable[..., float] = compute_q_score, meta: Optional[dict] = None,
               report: Callable[[dict], None] = lambda footer: None) -> dict:
    """Replay with ``base_cls`` (the official BatchedEvaluator on 105, a fake in tests); returns the footer.

    The truth file is written as <out>.partial and renamed to <out> only after the footer, so a crashed
    replay never leaves a truth file that reads as complete. The rename and ``report`` happen before the
    evaluator's __exit__, which shuts the simulator app down (og.shutdown) and may end the process."""
    policy = ReplayPolicy(p["actions"], to_tensor)
    os.makedirs(os.path.dirname(p["out"]), exist_ok=True)
    json_dir = os.path.join(p["replay_output_dir"], "json")
    video_dir = os.path.join(p["replay_output_dir"], "videos")
    os.makedirs(json_dir, exist_ok=True)
    if p.get("write_video"):
        os.makedirs(video_dir, exist_ok=True)
    header = {"run_id": p.get("run_id"), "task": p["task"], "instance_id": p["instance_id"], "mode": p["mode"],
              "every": p.get("every"), "n_actions": p["n_actions"], "steps_path": p["steps_path"],
              "steps_sha256": p["steps_sha256"], "official_json": (p.get("official") or {}).get("path"),
              "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "category": "training_annotation", **(meta or {})}
    partial = p["out"] + ".partial"
    with open(partial, "w", encoding="utf-8") as f:
        def make_writer(task):
            return TruthWriter(f, p["episode_id"], goal_option_literals(task), header, q_fn=q_fn)

        cls = make_replay_evaluator(base_cls, policy, make_writer, p["record_at"])
        with cls(make_cfg(p)) as ev:
            results = ev.run([p["instance_id"]], write_video=bool(p.get("write_video")), video_path=video_dir,
                             metrics_dir=json_dir, rollout_id=0)
            result = next(iter(results.values()), None) if results else None
            footer = ev.truth_writer.finish(**footer_fields(p, result, policy))
            f.flush()
            os.fsync(f.fileno())
            os.replace(partial, p["out"])
            report(footer)
    return footer


# ------------------------------------------------------------------------------------ CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dir", required=True, help="<local_root>/<run_id>/episodes/<episode_id> (steps.jsonl)")
    ap.add_argument("--out", required=True, help="truth JSONL to write")
    ap.add_argument("--official-json", default=None, help="the episode's official result JSON (replay length + check)")
    ap.add_argument("--task", default=None, help="default: summary.json tags / official JSON")
    ap.add_argument("--instance-id", type=int, default=None, help="RESOLVED instance id (public_test index i -> 301+i)")
    ap.add_argument("--mode", choices=("train", "public_test", "hidden_test"), default=None)
    ap.add_argument("--every", type=int, default=30, help="record truth every N steps (30 = 1 s of sim time)")
    ap.add_argument("--no-at-decisions", dest="at_decisions", action="store_false",
                    help="do not add the agent's decision / model-call steps from events.jsonl")
    ap.add_argument("--env-wrapper", default=None,
                    help="dotted class or eval.launch name (rgbd | default); default: the episode's launch tag, "
                         f"else {DEFAULT_WRAPPER}")
    ap.add_argument("--robot-config", default=None)
    ap.add_argument("--replay-output-dir", default=None, help="replay's official-format JSON (default: next to --out)")
    ap.add_argument("--write-video", action="store_true")
    ap.add_argument("--data-path", default=None, help="sets OMNIGIBSON_DATA_PATH")
    ap.add_argument("--gpu", default="auto", help="'auto' = a free GPU via eval/gpu.py, or an nvidia-smi index")
    ap.add_argument("--wait-gpu", action="store_true", help="poll until a GPU is free instead of exiting")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    ap.add_argument("--dry-run", action="store_true", help="check inputs, print the plan, import no simulator code")
    return ap.parse_args(argv)


def _select_gpu(args: argparse.Namespace) -> list[int]:
    from eval import gpu as gpumod

    if args.gpu != "auto":
        ids = [int(args.gpu)]
        gpumod.check_gpus_free(ids)
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        gpumod.check_gpus_free(ids)
    else:
        ids = gpumod.wait_for_free_gpus(1) if args.wait_gpu else gpumod.pick_free_gpus(1)
    os.environ.update(gpumod.cuda_env(ids))
    return ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        p = plan(args)
    except (ReplayInputError, OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    p["every"], p["write_video"] = args.every, args.write_video
    summary = {k: p[k] for k in ("episode_id", "task", "instance_id", "mode", "env_wrapper", "n_actions",
                                 "steps_in_log", "out", "replay_output_dir")}
    summary["truth_records_planned"] = len(p["record_at"])
    summary["official"] = p["official"]
    print(json.dumps(summary, indent=2, default=str))
    if args.dry_run:
        return 0
    for path in (p["out"], replay_json_path(p)):
        if os.path.exists(path) and not args.force:
            print(f"error: {path} exists; pass --force to overwrite", file=sys.stderr)
            return 1
    try:
        gpus = _select_gpu(args)
    except Exception as e:  # eval.gpu: no free GPU / nvidia-smi failed; never touch others' processes
        print(f"error: no usable GPU: {e}", file=sys.stderr)
        return 1
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (GPU {gpus})", file=sys.stderr)
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_APPDATA_PATH", os.path.expanduser(DEFAULT_APPDATA))  # as eval/launch.py
    if args.data_path:
        os.environ["OMNIGIBSON_DATA_PATH"] = os.path.abspath(os.path.expanduser(args.data_path))

    # Simulator imports: only here (the module must stay importable without them).
    import torch as th
    from omegaconf import OmegaConf

    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
    from omnigibson.macros import gm
    from omnigibson.metrics.task_metric import compute_q_score as official_q

    gm.HEADLESS = True
    if os.getenv("OMNIGIBSON_KEEP_VIEWER_CAMERA", "0") != "1":
        gm.RENDER_VIEWER_CAMERA = False
    seed = seed_everything(DEFAULT_EVAL_SEED)
    robot_config = OmegaConf.load(os.path.expanduser(args.robot_config)) if args.robot_config else None

    def make_cfg(plan_: dict):
        return OmegaConf.create({
            "env_wrapper": {"_target_": plan_["env_wrapper"]},
            "policy_name": "replay",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},  # replaced
            "headless": True,
            "partial_scene_load": True,
            "max_steps": plan_["n_actions"] - 1,
            "write_video": bool(args.write_video),
            "mode": plan_["mode"],
            "seed": seed,
            "num_envs": 1,
            "task": {"name": plan_["task"]},
            "robot": robot_config,
        })

    import omnigibson as og

    meta = {"seed": seed, "env_wrapper": p["env_wrapper"], "omnigibson_file": getattr(og, "__file__", None),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "status": "replay (not the original run)"}
    run_replay(p, BatchedEvaluator, make_cfg, to_tensor=lambda a: th.as_tensor(a, dtype=th.float32), q_fn=official_q,
               meta=meta, report=lambda footer: print(json.dumps(footer, indent=2, default=str), flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
