#!/usr/bin/env python3
"""
Run GPT-6 Astra on the Active Spatial ID / OOD evaluation matrix.

This is the API-model counterpart of scripts/active_spatial_eval_sweep.py.
It does not look up training checkpoints. It:

1. loads examples/evaluate/active_spatial/test_suites_gpt6.yaml;
2. writes a stratified JSONL slice per suite (so env.reset seed and the
   runner iterate the same items);
3. calls evaluation/run_eval.py with provider=openai_responses;
4. aggregates summary.csv / summary.md under evaluation/sweeps/active_spatial/.

Typical usage:

    export OPENAI_API_KEY=...

    # Write configs and slices only
    python scripts/run_gpt6_id_ood_eval.py --mode smoke --dry-run

    # 3 ID episodes: verify API + renderer + action parse
    python scripts/run_gpt6_id_ood_eval.py --mode smoke --run

    # Small ID + every OOD axis
    python scripts/run_gpt6_id_ood_eval.py --mode canary --run

    # Protocol-sized subsample
    python scripts/run_gpt6_id_ood_eval.py --mode standard --run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PROJECT_ROOT / "examples/evaluate/active_spatial/test_suites_gpt6.yaml"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "evaluation" / "sweeps" / "active_spatial"

# Per-suite episode caps. None means "use the whole split".
BUDGETS: Dict[str, Dict[str, Optional[int]]] = {
    "smoke": {
        "smoke": 3,
    },
    "canary": {
        "id_test": 24,
        "ood_v2_centering": 25,
        "ood_scene": 8,
        "ood_instance": 8,
        "ood_category": 8,
        "ood_template": 8,
        "ood_geometry": 8,
    },
    "standard": {
        "id_test": 80,
        "ood_v2_centering": 25,
        "ood_scene": 40,
        "ood_instance": 40,
        "ood_category": 40,
        "ood_template": 40,
        "ood_geometry": 40,
    },
    "full": {
        "id_test": 200,
        "ood_v2_centering": None,
        "ood_scene": None,
        "ood_instance": None,
        "ood_category": None,
        "ood_template": None,
        "ood_geometry": None,
    },
}

# Rough GPT-6 Astra Standard pricing for a 12-turn 256px episode.
USD_PER_EPISODE_LO = 0.40
USD_PER_EPISODE_HI = 1.50


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    task_types: Optional[Sequence[str]],
    exclude_task_types: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    include = set(task_types or [])
    exclude = set(exclude_task_types or [])
    out: List[Dict[str, Any]] = []
    for row in rows:
        task = row.get("task_type", "unknown")
        if include and task not in include:
            continue
        if task in exclude:
            continue
        out.append(row)
    return out


def stratified_sample(
    rows: Sequence[Dict[str, Any]],
    cap: Optional[int],
    seed: int,
    key: str = "task_type",
) -> List[Dict[str, Any]]:
    if cap is None or cap >= len(rows):
        return list(rows)
    if cap <= 0:
        return []
    rng = random.Random(seed)
    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row.get(key, "unknown")].append(row)
    keys = sorted(buckets, key=lambda x: str(x))
    sizes = {k: len(buckets[k]) for k in keys}
    total = sum(sizes.values())
    alloc = {k: min(sizes[k], int(round(cap * sizes[k] / total))) for k in keys}
    while sum(alloc.values()) > cap:
        k = max(alloc, key=lambda x: alloc[x] - cap * sizes[x] / total)
        if alloc[k] <= 0:
            break
        alloc[k] -= 1
    while sum(alloc.values()) < cap:
        candidates = [k for k in keys if alloc[k] < sizes[k]]
        if not candidates:
            break
        k = max(candidates, key=lambda x: sizes[x] - alloc[x])
        alloc[k] += 1
    out: List[Dict[str, Any]] = []
    for k in keys:
        pool = list(buckets[k])
        rng.shuffle(pool)
        out.extend(pool[: alloc[k]])
    out.sort(key=lambda r: (str(r.get("task_type", "")), str(r.get("scene_id", "")), str(r.get("object_label", ""))))
    return out


def resolve_jsonl(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def filter_suites(suites: List[Dict[str, Any]], suite_spec: str) -> List[Dict[str, Any]]:
    if suite_spec in ("all", "*", ""):
        return suites
    wanted = {s.strip() for s in suite_spec.split(",") if s.strip()}
    return [s for s in suites if str(s.get("name")) in wanted]


def suites_for_mode(suites: List[Dict[str, Any]], mode: str, suite_spec: str) -> List[Dict[str, Any]]:
    budget = BUDGETS[mode]
    if suite_spec in ("all", "*", ""):
        names = set(budget)
        selected = [s for s in suites if s["name"] in names]
    else:
        selected = filter_suites(suites, suite_spec)
    if not selected:
        raise ValueError(f"No suites selected for mode={mode} spec={suite_spec}")
    return selected


def make_eval_config(
    *,
    eval_name: str,
    output_dir: Path,
    suite: Dict[str, Any],
    suite_cfg: Dict[str, Any],
    jsonl_path: Path,
    model_overrides: Dict[str, Any],
    env_overrides: Dict[str, Any],
    max_episodes: Optional[int],
) -> Dict[str, Any]:
    defaults = dict(suite_cfg.get("defaults", {}) or {})
    env = {}
    env.update(defaults.get("env", {}) or {})
    env.update(dict(suite.get("env", {}) or {}))
    env.update(env_overrides)
    env["jsonl_path"] = str(jsonl_path)

    model = {}
    model.update(defaults.get("model", {}) or {})
    model.update(dict(suite.get("model", {}) or {}))
    model.update(model_overrides)
    model.setdefault("provider", "openai_responses")
    model.setdefault("model_name", "gpt-6-astra")

    config = {
        "eval_name": eval_name,
        "output_dir": str(output_dir),
        "agent_type": "model",
        "training_jsonl_path": defaults.get("training_jsonl_path"),
        "split_role": "id" if suite["name"] in ("id_test", "smoke") else suite["name"],
        "max_steps_per_episode": suite.get("max_turns", defaults.get("max_steps_per_episode", 12)),
        "num_eval_episodes": max_episodes,
        "seed_offset": suite.get("seed_offset", 0),
        "use_wandb": bool(suite.get("use_wandb", defaults.get("use_wandb", False))),
        "save_trajectories": bool(suite.get("save_trajectories", defaults.get("save_trajectories", True))),
        "verbose": bool(suite.get("verbose", defaults.get("verbose", False))),
        "env": env,
        "model": model,
    }
    task_types = suite.get("task_types", defaults.get("task_types"))
    if task_types:
        config["task_types"] = task_types
    return config


def result_json_path(output_dir: Path) -> Path:
    return output_dir / "results_model.json"


def extract_row(result_path: Path, suite_name: str, eval_name: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model": "gpt-6-astra",
        "suite": suite_name,
        "eval_name": eval_name,
        "result_path": str(result_path),
    }
    if not result_path.exists():
        row["status"] = "missing"
        return row
    with result_path.open("r") as f:
        data = json.load(f)
    overall = data.get("overall") or (data.get("metrics") or {}).get("overall") or {}
    row.update(
        {
            "status": "ok",
            "num_episodes": overall.get("num_episodes"),
            "success_rate": overall.get("success_rate"),
            "mean_final_score": overall.get("mean_final_score"),
            "mean_score_improvement": overall.get("mean_score_improvement"),
            "spl": overall.get("spl"),
            "mean_steps": overall.get("mean_steps"),
            "mean_turns": overall.get("mean_turns"),
            "mean_collisions": overall.get("mean_collisions"),
            "mean_action_validity": overall.get("mean_action_validity"),
            "monotonic_improvement_rate": overall.get("monotonic_improvement_rate"),
        }
    )
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


def fmt_float(value: Any, scale: float = 1.0, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value) * scale:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    headers = [
        "suite",
        "n",
        "success%",
        "final_score",
        "improvement",
        "spl",
        "validity%",
        "status",
    ]
    lines = [
        f"# GPT-6 Astra ID/OOD evaluation",
        "",
        f"- mode: `{meta.get('mode')}`",
        f"- model: `{meta.get('model')}`",
        f"- reasoning_effort: `{meta.get('reasoning_effort')}`",
        f"- output: `{meta.get('sweep_dir')}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("suite", "")),
                    str(row.get("num_episodes", "")),
                    fmt_float(row.get("success_rate"), 100, 1),
                    fmt_float(row.get("mean_final_score"), 1, 3),
                    fmt_float(row.get("mean_score_improvement"), 1, 3),
                    fmt_float(row.get("spl"), 1, 3),
                    fmt_float(row.get("mean_action_validity"), 100, 1),
                    str(row.get("status", "")),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def preflight(config: Dict[str, Any], require_api_key: bool) -> List[str]:
    errors: List[str] = []
    env = config.get("env", {})
    jsonl = resolve_jsonl(str(env.get("jsonl_path", "")))
    if not jsonl.is_file():
        errors.append(f"missing jsonl: {jsonl}")
    gs_root = Path(str(env.get("gs_root", "")))
    if not gs_root.is_dir():
        errors.append(f"missing gs_root: {gs_root}")
    if require_api_key and not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is not set")
    return errors


def run_eval(config_path: Path) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "evaluation" / "run_eval.py"), "--config", str(config_path)]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-6 Astra Active Spatial ID/OOD eval")
    parser.add_argument("--suite-config", default=str(DEFAULT_SUITE))
    parser.add_argument("--mode", choices=sorted(BUDGETS), default="smoke")
    parser.add_argument("--suites", default="all", help="Comma list or 'all' within the mode budget.")
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--model-name", default="gpt-6-astra")
    parser.add_argument("--provider", default="openai_responses")
    parser.add_argument("--reasoning-effort", default=None, help="Override suite default (low|medium|high|xhigh|max).")
    parser.add_argument("--service-tier", default=None, help="Optional OpenAI service_tier, e.g. flex.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Global cap applied after the mode budget.")
    parser.add_argument("--sample-seed", type=int, default=6)
    parser.add_argument("--gpu-device", type=int, default=None)
    parser.add_argument("--gs-root", default=None)
    parser.add_argument("--render-backend", default=None)
    parser.add_argument("--run", action="store_true", help="Actually call the model.")
    parser.add_argument("--dry-run", action="store_true", help="Write slices/configs only (default if --run is absent).")
    parser.add_argument("--rerun", action="store_true", help="Re-run suites that already have results_model.json.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dry_run = args.dry_run or not args.run
    suite_cfg = load_yaml(Path(args.suite_config))
    all_suites = suite_cfg.get("suites") or []
    selected = suites_for_mode(all_suites, args.mode, args.suites)
    budget = BUDGETS[args.mode]
    stamp = datetime.now().strftime("%Y%m%d")
    sweep_name = args.sweep_name or f"gpt6_astra_id_ood_{args.mode}_{stamp}"
    sweep_dir = Path(args.out_root).resolve() / sweep_name
    slice_dir = sweep_dir / "slices"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    model_overrides: Dict[str, Any] = {
        "provider": args.provider,
        "model_name": args.model_name,
    }
    if args.reasoning_effort:
        model_overrides["reasoning_effort"] = args.reasoning_effort
    if args.service_tier:
        model_overrides["service_tier"] = args.service_tier

    env_overrides: Dict[str, Any] = {}
    if args.gpu_device is not None:
        env_overrides["gpu_device"] = args.gpu_device
    if args.gs_root:
        env_overrides["gs_root"] = args.gs_root
    if args.render_backend:
        env_overrides["render_backend"] = args.render_backend

    defaults = suite_cfg.get("defaults", {}) or {}
    exclude = ((defaults.get("env") or {}).get("exclude_task_types")) or ["delta_control"]
    task_types = defaults.get("task_types")
    reasoning = model_overrides.get("reasoning_effort") or (defaults.get("model") or {}).get("reasoning_effort")

    plan: List[Dict[str, Any]] = []
    total_eps = 0
    for suite in selected:
        name = str(suite["name"])
        src = resolve_jsonl(str(suite["jsonl_path"]))
        if not src.is_file():
            print(f"[error] missing suite jsonl: {src}", file=sys.stderr)
            return 2
        rows = filter_rows(load_jsonl(src), task_types, exclude)
        cap = budget.get(name, suite.get("max_episodes"))
        if args.max_episodes is not None and cap is not None:
            cap = min(cap, args.max_episodes)
        elif args.max_episodes is not None and cap is None:
            cap = args.max_episodes
        sliced = stratified_sample(rows, cap, seed=args.sample_seed)
        slice_path = slice_dir / f"{name}.jsonl"
        write_jsonl(slice_path, sliced)
        out_dir = sweep_dir / name / "model"
        eval_name = f"gpt6_astra_{args.mode}_{name}"
        # Slice file already has the exact episodes; do not re-cap in the runner.
        config = make_eval_config(
            eval_name=eval_name,
            output_dir=out_dir,
            suite=suite,
            suite_cfg=suite_cfg,
            jsonl_path=slice_path,
            model_overrides=model_overrides,
            env_overrides=env_overrides,
            max_episodes=None,
        )
        config_path = out_dir / "eval_config.yaml"
        write_yaml(config_path, config)
        plan.append(
            {
                "suite": name,
                "n": len(sliced),
                "src": str(src),
                "slice": str(slice_path),
                "config": str(config_path),
                "output_dir": str(out_dir),
                "eval_name": eval_name,
            }
        )
        total_eps += len(sliced)
        print(f"[slice] {name}: {len(sliced)} / {len(rows)} -> {slice_path}")

    plan_path = sweep_dir / "plan.json"
    plan_path.write_text(json.dumps({"mode": args.mode, "suites": plan}, indent=2) + "\n")
    print(f"[plan] {total_eps} episodes across {len(plan)} suites")
    print(f"[cost] rough ${total_eps * USD_PER_EPISODE_LO:.0f}–${total_eps * USD_PER_EPISODE_HI:.0f} at Standard GPT-6 Astra rates")
    print(f"[out]  {sweep_dir}")

    if dry_run:
        print("[dry-run] configs and slices written; pass --run to evaluate")
        return 0

    first_errors = preflight(load_yaml(Path(plan[0]["config"])), require_api_key=True)
    if first_errors:
        print("[preflight]", "; ".join(first_errors), file=sys.stderr)
        return 2

    for item in plan:
        result_path = result_json_path(Path(item["output_dir"]))
        if result_path.is_file() and not args.rerun:
            from evaluation.eval_config import evaluation_fingerprint
            previous = json.loads(result_path.read_text()).get("metrics", {}).get("protocol_fingerprint")
            if previous != evaluation_fingerprint(load_yaml(Path(item["config"]))):
                raise ValueError(f"stale result: {result_path}; choose a new sweep directory or explicitly --rerun")
            print(f"[skip] {item['suite']} already has {result_path}")
            continue
        print(f"[run]  {item['suite']} n={item['n']}")
        code = run_eval(Path(item["config"]))
        if code != 0:
            print(f"[error] run_eval failed for {item['suite']} with code {code}", file=sys.stderr)
            # Continue remaining suites; summary will mark missing/failed.

    rows = [extract_row(result_json_path(Path(item["output_dir"])), item["suite"], item["eval_name"]) for item in plan]
    write_csv(sweep_dir / "summary.csv", rows)
    write_markdown(
        sweep_dir / "summary.md",
        rows,
        {
            "mode": args.mode,
            "model": args.model_name,
            "reasoning_effort": reasoning,
            "sweep_dir": str(sweep_dir),
        },
    )
    print(f"[done] {sweep_dir / 'summary.md'}")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
