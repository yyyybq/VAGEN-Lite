#!/usr/bin/env python3
"""First-class RoboCasa eval CLI (same role as scripts/easi_eval.py).

Thin wrapper around ``python -m vagen.evaluate.run_eval``. Extra flags
expand the env list (task / task_set) and override backend / model.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "examples" / "evaluate" / "robocasa" / "config.yaml"

# Import path for task lists (no robocasa / gym).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.envs.robocasa.utils.tasks import get_horizon, resolve_tasks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a VLM on RoboCasa via vagen.evaluate")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Eval YAML (remote or local)")
    p.add_argument("--task", default="", help="Task name or comma-separated list")
    p.add_argument("--task_set", "--task-set", dest="task_set", default="", help="atomic_seen | composite_seen | seen")
    p.add_argument("--model", default="", help="Override backends.<backend>.model")
    p.add_argument("--backend", default="", help="Override run.backend (openai, vllm, sglang, ...)")
    p.add_argument("--split", default="", help="Override env config.split (pretrain/target)")
    p.add_argument(
        "overrides",
        nargs="*",
        help="Extra OmegaConf dotlist overrides forwarded to run_eval",
    )
    return p.parse_args()


def _slug(name: str) -> str:
    return "robocasa_" + "".join(ch if ch.isalnum() else "_" for ch in name).lower()


def _expand_envs(base_env: Dict[str, Any], tasks: List[str], split: str) -> List[Dict[str, Any]]:
    envs = []
    for task in tasks:
        item = {k: v for k, v in base_env.items() if k != "config"}
        cfg = dict(base_env.get("config") or {})
        cfg["env_id"] = task
        if split:
            cfg["split"] = split
        cfg.setdefault("max_steps", get_horizon(task))
        item["config"] = cfg
        item["tag_id"] = _slug(task)
        envs.append(item)
    return envs


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file():
        print(f"[error] config not found: {cfg_path}", file=sys.stderr)
        return 2

    overrides: List[str] = list(args.overrides)
    if args.backend:
        overrides.append(f"run.backend={args.backend}")
        if args.model:
            overrides.append(f"backends.{args.backend}.model={args.model}")
    elif args.model:
        overrides.append(f"backends.openai.model={args.model}")

    extra_env = os.environ.copy()
    extra_env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + extra_env["PYTHONPATH"] if extra_env.get("PYTHONPATH") else ""
    )

    if args.task or args.task_set:
        try:
            import yaml
        except ImportError:
            print("[error] PyYAML is required to expand --task / --task_set", file=sys.stderr)
            return 2
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        defaults = raw.get("defaults") or []
        envs = raw.get("envs") or []
        if not envs:
            print("[error] config has no envs:", cfg_path, file=sys.stderr)
            return 2
        tasks = resolve_tasks(task=args.task or None, task_set=args.task_set or None)
        expanded = _expand_envs(envs[0], tasks, args.split)
        raw["envs"] = expanded
        tmp_dir = REPO_ROOT / "outputs" / "robocasa_eval_cfgs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{cfg_path.stem}_expanded.yaml"
        # defaults: paths are relative to the ORIGINAL config. Rewrite them
        # as absolute paths so run_eval still finds eval_default.yaml.
        if defaults:
            orig_dir = cfg_path.resolve().parent
            resolved = []
            for entry in defaults:
                ref = str(entry)
                if not ref.endswith((".yaml", ".yml")):
                    ref += ".yaml"
                ref_path = Path(ref)
                if not ref_path.is_absolute():
                    ref_path = (orig_dir / ref).resolve()
                resolved.append(str(ref_path))
            raw["defaults"] = resolved
        tmp_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg_path = tmp_path
        print(f"[robocasa_eval] expanded {len(tasks)} task(s) -> {tmp_path}")
    elif args.split:
        overrides.append(f"envs.0.config.split={args.split}")

    cmd = [sys.executable, "-m", "vagen.evaluate.run_eval", "--config", str(cfg_path)]
    cmd.extend(overrides)
    print("[robocasa_eval]", " ".join(cmd))
    return subprocess.call(cmd, env=extra_env, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
