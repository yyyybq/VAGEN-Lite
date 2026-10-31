#!/usr/bin/env python3
"""RLinf's own evaluation of a StarVLA start on E3's tasks, without any update (the user's decision 2 of 2026-10-31:
with the image-size fix, RLinf must reproduce our evaluation's 23%-27% on 3 tasks x 10 episodes).

    python downstream/rlinf/rlinf_eval.py make   --train-cfg E3/cfg/train.yaml --out OUT [--envs 30]
    python downstream/rlinf/rlinf_eval.py report --out OUT

make: OUT/cfg/eval.yaml for RLinf's evaluations/eval_embodied_agent.py (task_type embodied_eval, only_eval): E3's
env.eval block as it was (RoboCasa365 target split, the three E3 tasks in order, seed 1042, 512 steps, the same
cameras, flip and action space), with total_num_envs = ENVS and auto_reset off, so every environment runs exactly one
episode (30 = 10 per task); rollout.model is E3's actor model (the start's checkpoint, QwenOFT, 12 x 16 actions,
min-max statistics new_embodiment, no state, policy_setup robocasa365) without the value head; env and rollout
share the job's cards.
report: the eval metrics of each run's log (OUT/rlinf_<name>.log, the line RLinf's EmbodiedEvalRunner logs)
into OUT/report.json and report.md.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from pathlib import Path

ACTOR_KEYS = ("model_path", "action_dim", "num_action_chunks", "unnorm_key", "action_stats_source", "starvla",
              "policy_setup", "use_proprio")


def eval_config(train: dict, out: Path, envs: int = 30) -> dict:
    """The eval config built from E3's train config (a dict as PyYAML reads it)."""
    env_eval = copy.deepcopy(train["env"]["eval"])
    env_eval.update(total_num_envs=int(envs), auto_reset=False, rollout_epoch=1)
    env_eval["video_cfg"] = {"save_video": True, "video_base_dir": str(out / "video")}
    actor = train["actor"]["model"]
    model = {k: copy.deepcopy(actor[k]) for k in ACTOR_KEYS if k in actor}
    model["add_value_head"] = False
    starvla = model.get("starvla") or {}
    for k in ("expected_action_dim", "expected_num_action_chunks"):  # their ${actor.model...} links, made literal
        if k in starvla:
            starvla[k] = actor["action_dim"] if k == "expected_action_dim" else actor["num_action_chunks"]
    sampling = copy.deepcopy(train.get("rollout", {}).get("sampling_params", {}))
    return {
        "defaults": ["env/robocasa365@env.eval", "model/starvla@rollout.model",
                     {"override hydra/job_logging": "stdout"}],
        "hydra": copy.deepcopy(train["hydra"]),
        "cluster": {"num_nodes": 1, "component_placement": {"env,rollout": "all"}},
        "runner": {"task_type": "embodied_eval", "only_eval": True, "val_check_interval": -1, "save_interval": -1,
                   "max_epochs": 1, "max_steps": -1, "resume_dir": None, "ckpt_path": None,
                   "logger": {"log_path": str(out / "logs"), "project_name": "rlinf",
                              "experiment_name": "e3_rlinf_eval", "logger_backends": ["tensorboard"]}},
        "env": {"group_name": "EnvGroup", "enable_offload": False, "eval": env_eval},
        "rollout": {"sampling_params": sampling, "group_name": "RolloutGroup", "generation_backend": "huggingface",
                    "recompute_logprobs": False, "unnorm_key": actor["unnorm_key"], "enable_offload": False,
                    "pipeline_stage_num": 1, "model": model},
    }


def make_cmd(args) -> int:
    import yaml

    out = Path(args.out)
    (out / "cfg").mkdir(parents=True, exist_ok=True)
    train = yaml.safe_load(Path(args.train_cfg).read_text())
    cfg = eval_config(train, out, args.envs)
    (out / "cfg/eval.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(json.dumps({"config": str(out / "cfg/eval.yaml"), "envs": args.envs,
                      "tasks": cfg["env"]["eval"].get("task_names"), "model": cfg["rollout"]["model"]["model_path"]}))
    return 0


METRICS_LINE = re.compile(r"(\{['\"]eval/[^{}]*\})")
# RLinf logs numpy scalars: {'eval/success_once': array(0.36666667, dtype=float32), ...} (2026-10-31)
NUMPY_SCALAR = re.compile(r"(?:np\.|numpy\.)?(?:array|float\d*|int\d*)\(\s*([-+0-9.eE]+|nan|inf)\s*(?:,\s*dtype=[\w.]+\s*)?\)")


def read_metrics(log_text: str):
    """The last {'eval/...': value, ...} dict a run logged (numpy scalars read as numbers), or None."""
    found = None
    for m in METRICS_LINE.finditer(log_text):
        try:
            found = ast.literal_eval(NUMPY_SCALAR.sub(r"\1", m.group(1)))
        except (ValueError, SyntaxError):
            continue
    return found


def report_cmd(args) -> int:
    out = Path(args.out)
    runs = {}
    for rc_file in sorted(out.glob("rc_*.txt")):  # the runs run_rlinf_eval.sh finished (not its own rlinf_eval.log)
        name = rc_file.stem[len("rc_"):]
        log = out / f"rlinf_{name}.log"
        metrics = read_metrics(log.read_text(errors="replace")) if log.is_file() else None
        runs[name] = {"metrics": metrics, "rc": rc_file.read_text().strip()}
    (out / "report.json").write_text(json.dumps(runs, indent=1) + "\n")
    lines = ["| 运行 | 退出码 | 成功率（RLinf 的 success_once） | 其他指标 |", "| --- | --- | --- | --- |"]
    for name, r in runs.items():
        m = r["metrics"] or {}
        succ = {k: v for k, v in m.items() if "success" in k}
        rest = {k: v for k, v in m.items() if "success" not in k}
        lines.append(f"| {name} | {r['rc']} | {json.dumps(succ)} | {json.dumps(rest)[:300]} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(runs)[:2000])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--train-cfg", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--envs", type=int, default=30)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"make": make_cmd, "report": report_cmd}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
