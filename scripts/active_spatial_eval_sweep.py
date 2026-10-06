#!/usr/bin/env python3
"""
Active Spatial checkpoint evaluation sweep.

This script is the orchestration layer above evaluation/run_eval.py:

1. discover experiments under exps/vagen_active_spatial;
2. discover HuggingFace actor checkpoints saved by training;
3. run each checkpoint on a named set of test distributions;
4. aggregate test metrics and align them with train-time validation metrics.

Typical usage:

    python scripts/active_spatial_eval_sweep.py \
        --suite-config examples/evaluate/active_spatial/test_suites.yaml \
        --exps v38_masked_gae_6types_fastcosine \
        --steps latest \
        --dry-run

    python scripts/active_spatial_eval_sweep.py \
        --suite-config examples/evaluate/active_spatial/test_suites.yaml \
        --exps v38_masked_gae_6types_fastcosine,v33_grpo_rewscale_klhi \
        --steps best-val,latest \
        --run
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_ROOT = PROJECT_ROOT / "exps" / "vagen_active_spatial"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "evaluation" / "sweeps" / "active_spatial"

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_VAL_N = 4


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")


def parse_step_name(path: Path) -> Optional[int]:
    m = re.search(r"global_step_(\d+)$", path.name)
    return int(m.group(1)) if m else None


def checkpoint_model_dir(global_step_dir: Path) -> Optional[Path]:
    """Return a complete HF model directory for a VERL checkpoint.

    Training creates the ``actor/huggingface`` directory before saving its
    config and weights.  Treating that empty directory as a checkpoint makes
    evaluation fail later inside vLLM or lmms-eval, and can make old results
    look resumable even when their source model has been deleted.
    """
    candidates = [
        global_step_dir / "actor" / "huggingface",
        global_step_dir / "actor" / "hf_model",
        global_step_dir / "actor",
    ]
    for cand in candidates:
        has_config = (cand / "config.json").is_file()
        has_weights = any(cand.glob("*.safetensors")) or any(
            (cand / filename).is_file()
            for filename in (
                "pytorch_model.bin",
                "tf_model.h5",
                "flax_model.msgpack",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
        )
        if has_config and has_weights:
            return cand
    return None


@dataclasses.dataclass
class Checkpoint:
    exp_name: str
    step: int
    global_step_dir: Path
    model_dir: Path


@dataclasses.dataclass
class Experiment:
    name: str
    path: Path
    train_yaml: Optional[Path]
    val_yaml: Optional[Path]
    train_log: Optional[Path]
    checkpoints: List[Checkpoint]


def discover_experiments(exp_root: Path) -> List[Experiment]:
    if not exp_root.exists():
        return []

    experiments: List[Experiment] = []
    for exp_dir in sorted(p for p in exp_root.iterdir() if p.is_dir()):
        train_yaml = exp_dir / "train.yaml"
        val_yaml = exp_dir / "val.yaml"
        train_log = exp_dir / "train.log"
        ckpts: List[Checkpoint] = []
        ckpt_root = exp_dir / "checkpoints"
        if ckpt_root.exists():
            for step_dir in sorted(ckpt_root.glob("global_step_*"), key=lambda p: parse_step_name(p) or -1):
                step = parse_step_name(step_dir)
                model_dir = checkpoint_model_dir(step_dir)
                if step is not None and model_dir is not None:
                    ckpts.append(Checkpoint(exp_dir.name, step, step_dir, model_dir))

        # "Already started" means at least a train log/yaml/checkpoint exists.
        if train_yaml.exists() or val_yaml.exists() or train_log.exists() or ckpts:
            experiments.append(
                Experiment(
                    name=exp_dir.name,
                    path=exp_dir,
                    train_yaml=train_yaml if train_yaml.exists() else None,
                    val_yaml=val_yaml if val_yaml.exists() else None,
                    train_log=train_log if train_log.exists() else None,
                    checkpoints=ckpts,
                )
            )
    return experiments


def select_experiments(experiments: List[Experiment], spec: str) -> List[Experiment]:
    if spec in ("all", "*"):
        return experiments
    if spec.startswith("re:"):
        pat = re.compile(spec[3:])
        return [e for e in experiments if pat.search(e.name)]
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    return [e for e in experiments if e.name in wanted]


def validation_success_by_step(exp: Experiment, val_n: int) -> Dict[int, Dict[str, float]]:
    """Parse training validation JSONL files into coarse split metrics.

    Returns:
        step -> {
            "val/overall_success_best": ...,
            "val/<data_source>/success_best": ...,
            "val/<data_source>/success_mean": ...,
        }
    """
    val_dir = exp.path / "validation"
    if not val_dir.exists():
        return {}

    split_specs: List[Tuple[str, int]] = []
    if exp.val_yaml and exp.val_yaml.exists():
        try:
            val_cfg = load_yaml(exp.val_yaml)
            for env in val_cfg.get("envs", []) or []:
                name = str(env.get("data_source") or env.get("name") or f"split_{len(split_specs)}")
                n_envs = int(env.get("n_envs") or 0)
                if n_envs > 0:
                    split_specs.append((safe_name(name), n_envs))
        except Exception as exc:
            print(f"[warn] Could not parse val yaml for {exp.name}: {exc}", file=sys.stderr)

    results: Dict[int, Dict[str, float]] = {}
    for fpath in sorted(val_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        if not fpath.stem.isdigit():
            continue
        step = int(fpath.stem)
        try:
            entries = [json.loads(line) for line in fpath.read_text().splitlines() if line.strip()]
        except Exception as exc:
            print(f"[warn] Could not parse validation file {fpath}: {exc}", file=sys.stderr)
            continue

        row: Dict[str, float] = {}
        offset = 0
        all_best: List[float] = []
        all_mean: List[float] = []

        if not split_specs:
            split_specs = [("active_spatial_val", len(entries) // max(val_n, 1))]

        for split_name, n_envs in split_specs:
            split_entries = entries[offset : offset + n_envs * val_n]
            offset += n_envs * val_n
            bests, means = aggregate_success_groups(split_entries, val_n)
            if bests:
                row[f"val/{split_name}/success_best"] = sum(bests) / len(bests)
                row[f"val/{split_name}/success_mean"] = sum(means) / len(means)
                all_best.extend(bests)
                all_mean.extend(means)

        if all_best:
            row["val/overall_success_best"] = sum(all_best) / len(all_best)
            row["val/overall_success_mean"] = sum(all_mean) / len(all_mean)
        results[step] = row
    return results


def aggregate_success_groups(entries: Sequence[Dict[str, Any]], val_n: int) -> Tuple[List[float], List[float]]:
    bests: List[float] = []
    means: List[float] = []
    for i in range(0, len(entries), val_n):
        chunk = entries[i : i + val_n]
        if not chunk:
            continue
        succs = [float(e.get("traj_success", 0.0)) for e in chunk]
        bests.append(max(succs))
        means.append(sum(succs) / len(succs))
    return bests, means


def select_checkpoints(exp: Experiment, step_spec: str, val_n: int) -> List[Checkpoint]:
    if not exp.checkpoints:
        return []

    ckpt_by_step = {c.step: c for c in exp.checkpoints}
    all_steps = sorted(ckpt_by_step)
    selected: List[int] = []

    for raw_part in step_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part == "all":
            selected.extend(all_steps)
        elif part == "latest":
            selected.append(all_steps[-1])
        elif part == "first":
            selected.append(all_steps[0])
        elif part.startswith("every:"):
            interval = int(part.split(":", 1)[1])
            selected.extend([s for s in all_steps if s % interval == 0])
        elif part.startswith("range:"):
            _, rest = part.split(":", 1)
            start_s, end_s, stride_s = (rest.split(":") + ["1"])[:3]
            start, end, stride = int(start_s), int(end_s), int(stride_s)
            selected.extend([s for s in all_steps if start <= s <= end and (s - start) % stride == 0])
        elif part == "best-val":
            val_metrics = validation_success_by_step(exp, val_n)
            best_step = None
            best_val = -1.0
            for step, metrics in val_metrics.items():
                if step not in ckpt_by_step:
                    continue
                value = metrics.get("val/overall_success_mean", metrics.get("val/overall_success_best", -1.0))
                if value > best_val:
                    best_step = step
                    best_val = value
            if best_step is not None:
                selected.append(best_step)
        else:
            selected.append(int(part))

    unique = []
    seen = set()
    for step in selected:
        if step in ckpt_by_step and step not in seen:
            unique.append(ckpt_by_step[step])
            seen.add(step)
    return unique


def env_defaults_from_exp(exp: Experiment) -> Dict[str, Any]:
    """Use the training env config as defaults for eval env where names overlap."""
    defaults: Dict[str, Any] = {}
    source_yaml = exp.train_yaml or exp.val_yaml
    if not source_yaml:
        return defaults
    try:
        cfg = load_yaml(source_yaml)
        envs = cfg.get("envs", []) or []
        if not envs:
            return defaults
        env_cfg = dict(envs[0].get("config", {}) or {})
    except Exception as exc:
        print(f"[warn] Could not load env defaults for {exp.name}: {exc}", file=sys.stderr)
        return defaults

    from dataclasses import asdict, fields
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
    allowed = {f.name for f in fields(ActiveSpatialEnvConfig)}
    return asdict(ActiveSpatialEnvConfig(**{k: v for k, v in env_cfg.items() if k in allowed}))


def load_suites(path: Path) -> Dict[str, Any]:
    cfg = load_yaml(path)
    suites = cfg.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError(f"{path} must define a non-empty 'suites' list")
    names = set()
    for suite in suites:
        if not isinstance(suite, dict) or "name" not in suite or "jsonl_path" not in suite:
            raise ValueError("Each suite must include name and jsonl_path")
        name = str(suite["name"])
        if name in names:
            raise ValueError(f"Duplicate suite name: {name}")
        names.add(name)
    return cfg


def filter_suites(suites: List[Dict[str, Any]], suite_spec: str) -> List[Dict[str, Any]]:
    if suite_spec in ("all", "*"):
        return suites
    wanted = {s.strip() for s in suite_spec.split(",") if s.strip()}
    return [s for s in suites if str(s.get("name")) in wanted]


def make_eval_config(
    exp: Experiment,
    ckpt: Checkpoint,
    suite: Dict[str, Any],
    suite_cfg: Dict[str, Any],
    output_dir: Path,
    agent: str,
) -> Dict[str, Any]:
    global_defaults = dict(suite_cfg.get("defaults", {}) or {})
    env_defaults = env_defaults_from_exp(exp)
    suite_env = dict(suite.get("env", {}) or {})

    # Suite defaults fill missing configuration; training defines the protocol.
    env = dict(global_defaults.get("env", {}) or {})
    env.update(env_defaults)
    infrastructure = {"render_backend", "gs_root", "gpu_device", "client_url", "client_origin", "dataset_root"}
    for key in infrastructure:
        if key in (global_defaults.get("env") or {}):
            env[key] = global_defaults["env"][key]
    overrides = dict(suite_env)
    for key in infrastructure:
        if key in suite:
            overrides[key] = suite[key]
    if "success_threshold" in suite:
        overrides["success_score_threshold"] = suite["success_threshold"]
    permitted = set(global_defaults.get("protocol_overrides", [])) | set(suite.get("protocol_overrides", []))
    differences = {key: {"train": env.get(key), "eval": value} for key, value in overrides.items()
                   if key not in infrastructure and key != "jsonl_path" and key in env and env[key] != value}
    undeclared = set(differences) - permitted
    if undeclared:
        raise ValueError(f"declare protocol_overrides for intentional eval changes: {sorted(undeclared)}")
    env.update(overrides)
    env["jsonl_path"] = suite["jsonl_path"]
    if env.get("require_verified_dataset"):
        env["dataset_contract_path"] = suite.get("dataset_contract_path", str(suite["jsonl_path"]) + ".contract.json")
    train_turns = global_defaults.get("max_steps_per_episode", 20)
    if exp.train_yaml:
        train_spec = load_yaml(exp.train_yaml)["envs"][0]
        train_turns = train_spec.get("max_turns", train_turns)

    model_defaults = dict(global_defaults.get("model", {}) or {})
    model = {
        "provider": model_defaults.get("provider", "vllm"),
        "model_name": model_defaults.get("model_name", DEFAULT_MODEL_NAME),
        "checkpoint_path": str(ckpt.model_dir),
        "temperature": model_defaults.get("temperature", 0.1),
        "top_p": model_defaults.get("top_p", 0.95),
        "max_tokens": model_defaults.get("max_tokens", 512),
        "tensor_parallel_size": model_defaults.get("tensor_parallel_size", 1),
        "gpu_memory_utilization": model_defaults.get("gpu_memory_utilization", 0.8),
    }
    model.update(dict(suite.get("model", {}) or {}))
    if agent == "frozen":
        model["checkpoint_path"] = None

    eval_name = f"{exp.name}_step{ckpt.step}_{suite['name']}_{agent}"
    config = {
        "eval_name": eval_name,
        "output_dir": str(output_dir),
        "agent_type": agent,
        "max_steps_per_episode": suite.get("max_turns", train_turns),
        "training_jsonl_path": env_defaults.get("jsonl_path"),
        "split_role": suite.get("split_role", "id" if suite["name"] in ("id_test", "smoke") else suite["name"]),
        "protocol_differences": differences,
        "num_eval_episodes": suite.get("max_episodes", global_defaults.get("num_eval_episodes")),
        "seed_offset": suite.get("seed_offset", 0),
        "use_wandb": bool(suite.get("use_wandb", global_defaults.get("use_wandb", False))),
        "save_trajectories": bool(suite.get("save_trajectories", global_defaults.get("save_trajectories", True))),
        "verbose": bool(suite.get("verbose", global_defaults.get("verbose", False))),
        "env": env,
        "model": model,
    }
    task_types = suite.get("task_types", global_defaults.get("task_types"))
    if task_types:
        config["task_types"] = task_types
    return config


def result_json_path(output_dir: Path, agent: str) -> Path:
    return output_dir / f"results_{agent}.json"


def run_eval(config_path: Path) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "evaluation" / "run_eval.py"), "--config", str(config_path)]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


def extract_result_row(
    result_path: Path,
    exp: Experiment,
    ckpt: Checkpoint,
    suite: Dict[str, Any],
    agent: str,
    val_metrics: Dict[int, Dict[str, float]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "experiment": exp.name,
        "step": ckpt.step,
        "agent": agent,
        "suite": suite["name"],
        "checkpoint": str(ckpt.model_dir),
        "result_path": str(result_path),
    }
    for key, value in val_metrics.get(ckpt.step, {}).items():
        row[key] = value
    if not result_path.exists():
        row["status"] = "missing"
        return row

    data = read_json(result_path)
    # Current evaluation outputs nest aggregate metrics under
    # ``metrics.overall``; retain compatibility with older result files that
    # wrote ``overall`` at the top level.
    overall = data.get("overall") or (data.get("metrics") or {}).get("overall") or {}
    row.update(
        {
            "status": "ok",
            "test_num_episodes": overall.get("num_episodes"),
            "test_success_rate": overall.get("success_rate"),
            "test_mean_final_score": overall.get("mean_final_score"),
            "test_mean_score_improvement": overall.get("mean_score_improvement"),
            "test_spl": overall.get("spl"),
            "test_mean_steps": overall.get("mean_steps"),
            "test_mean_turns": overall.get("mean_turns"),
            "test_mean_collisions": overall.get("mean_collisions"),
            "test_mean_action_validity": overall.get("mean_action_validity"),
            "test_monotonic_improvement_rate": overall.get("monotonic_improvement_rate"),
        }
    )
    test_success = row.get("test_success_rate")
    val_success = row.get("val/overall_success_mean")
    if test_success is not None and val_success is not None:
        try:
            row["test_minus_val_success_mean"] = float(test_success) - float(val_success)
        except (TypeError, ValueError):
            pass
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_float(value: Any, scale: float = 1.0, digits: int = 1) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value) * scale:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "experiment",
        "step",
        "suite",
        "success%",
        "val_mean%",
        "delta_pp",
        "final_score",
        "improvement",
        "spl",
        "result",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        success = row.get("test_success_rate")
        val_mean = row.get("val/overall_success_mean")
        delta = row.get("test_minus_val_success_mean")
        result_rel = row.get("result_path", "")
        try:
            result_rel = str(Path(result_rel).relative_to(PROJECT_ROOT))
        except Exception:
            pass
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("experiment", "")),
                    str(row.get("step", "")),
                    str(row.get("suite", "")),
                    fmt_float(success, 100, 1),
                    fmt_float(val_mean, 100, 1),
                    fmt_float(delta, 100, 1),
                    fmt_float(row.get("test_mean_final_score"), 1, 3),
                    fmt_float(row.get("test_mean_score_improvement"), 1, 3),
                    fmt_float(row.get("test_spl"), 1, 3),
                    result_rel,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def append_manifest(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or summarize Active Spatial checkpoint test sweeps.")
    parser.add_argument("--suite-config", required=True, help="YAML file that defines test distributions.")
    parser.add_argument("--exp-root", default=str(DEFAULT_EXP_ROOT), help=f"Experiment root. Default: {DEFAULT_EXP_ROOT}")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help=f"Sweep output root. Default: {DEFAULT_OUT_ROOT}")
    parser.add_argument("--sweep-name", default=None, help="Output subdirectory name. Default: derived from suite config name.")
    parser.add_argument("--exps", default="all", help="'all', comma list, or re:<regex>.")
    parser.add_argument("--steps", default="latest", help="Checkpoint selector: latest, best-val, all, every:N, range:start:end:stride, or comma list.")
    parser.add_argument("--suites", default="all", help="'all' or comma-separated suite names.")
    parser.add_argument("--agents", default="model", help="Usually 'model'. Also supports random/heuristic/frozen if run_eval supports them.")
    parser.add_argument("--val-n", type=int, default=DEFAULT_VAL_N, help="Number of validation samples per env during training.")
    parser.add_argument("--run", action="store_true", help="Actually run evaluation jobs. Without this, only configs/manifest are generated.")
    parser.add_argument("--rerun", action="store_true", help="Run even when results JSON already exists.")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize existing outputs; do not generate configs or run.")
    parser.add_argument("--dry-run", action="store_true", help="Alias for not passing --run; print the planned jobs.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        args.run = False
    suite_config_path = Path(args.suite_config).resolve()
    exp_root = Path(args.exp_root).resolve()
    suite_cfg = load_suites(suite_config_path)
    sweep_name = args.sweep_name or safe_name(str(suite_cfg.get("name") or suite_config_path.stem))
    out_root = Path(args.out_root).resolve() / sweep_name
    manifest_path = out_root / "manifest.jsonl"
    if not args.summarize_only and manifest_path.exists():
        manifest_path.unlink()

    experiments = discover_experiments(exp_root)
    selected_exps = select_experiments(experiments, args.exps)
    suites = filter_suites(suite_cfg["suites"], args.suites)
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    if not selected_exps:
        print(f"[warn] No experiments matched under {exp_root}")
    if not suites:
        print("[warn] No suites selected")

    rows: List[Dict[str, Any]] = []
    planned = 0
    failed = 0

    for exp in selected_exps:
        val_metrics = validation_success_by_step(exp, args.val_n)
        ckpts = select_checkpoints(exp, args.steps, args.val_n)
        if not ckpts:
            print(f"[warn] {exp.name}: no checkpoints selected")
            continue

        for ckpt in ckpts:
            for suite in suites:
                for agent in agents:
                    suite_name = safe_name(str(suite["name"]))
                    output_dir = out_root / exp.name / f"global_step_{ckpt.step}" / suite_name / agent
                    config_path = output_dir / "eval_config.yaml"
                    result_path = result_json_path(output_dir, agent)

                    if not args.summarize_only:
                        config = make_eval_config(exp, ckpt, suite, suite_cfg, output_dir, agent)
                        if result_path.exists() and not args.rerun:
                            from evaluation.eval_config import evaluation_fingerprint
                            previous = read_json(result_path).get("metrics", {}).get("protocol_fingerprint")
                            if previous != evaluation_fingerprint(config):
                                raise ValueError(f"stale or incompatible result: {result_path}; choose a new sweep name or explicitly --rerun")
                        write_yaml(config_path, config)
                        append_manifest(
                            manifest_path,
                            {
                                "experiment": exp.name,
                                "step": ckpt.step,
                                "suite": suite["name"],
                                "agent": agent,
                                "checkpoint": str(ckpt.model_dir),
                                "config": str(config_path),
                                "output_dir": str(output_dir),
                                "result": str(result_path),
                            },
                        )

                    planned += 1
                    if args.run and not args.summarize_only:
                        if result_path.exists() and not args.rerun:
                            print(f"[skip] {exp.name} step={ckpt.step} suite={suite['name']} agent={agent}: result exists")
                        else:
                            print(f"[run] {exp.name} step={ckpt.step} suite={suite['name']} agent={agent}")
                            code = run_eval(config_path)
                            if code != 0:
                                print(f"[error] run_eval failed with code {code}: {config_path}", file=sys.stderr)
                                failed += 1
                    else:
                        print(f"[plan] {exp.name} step={ckpt.step} suite={suite['name']} agent={agent}")
                        print(f"       config={config_path}")

                    rows.append(extract_result_row(result_path, exp, ckpt, suite, agent, val_metrics))

    summary_csv = out_root / "summary.csv"
    summary_md = out_root / "summary.md"
    if rows:
        write_csv(summary_csv, rows)
        write_markdown(summary_md, rows)

    print()
    print(f"Planned jobs: {planned}")
    if rows:
        print(f"Summary CSV:  {summary_csv}")
        print(f"Summary MD:   {summary_md}")
    if manifest_path.exists():
        print(f"Manifest:     {manifest_path}")
    if failed:
        print(f"Failed jobs:  {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
