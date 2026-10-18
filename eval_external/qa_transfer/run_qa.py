"""Run one model on one QA benchmark with the benchmark's own backend, prompt, frames and greedy decoding.

    python -m eval_external.qa_transfer.run_qa --model base-7B --bench VSI-Bench --run-dir RUN_DIR

Writes RUN_DIR/manifest.json first (models with weight hash, benchmark source, backend commit, prompt, frames,
decoding, date), then the backend's raw outputs (all predictions) under RUN_DIR/vlmeval or RUN_DIR/lmms, and
RUN_DIR/outputs.json listing them.  Scores are read from those files by qa_transfer/collect.py.
Runs inside a one-card sco job (sco/job.py); every input must already be downloaded (HF offline).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval_external.adapters import models as M
from eval_external.common.manifest import git_state, write_manifest
from eval_external.qa_transfer.benchmarks import (BENCHMARKS, LMMS_INCLUDE, LMMS_QWEN25VL_ARGS,
                                                  VLMEVAL_DECODING, VLMEVAL_QWEN25VL)

WORKTREE = Path(__file__).resolve().parents[2]
REPOS = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets/repos")
VLMEVAL_DIR = REPOS / "EASI" / "VLMEvalKit"
LMMS_DIR = REPOS / "lmms-eval"


def hf_dataset_revision(repo: str) -> str | None:
    """The snapshot revision of a dataset in $HF_HOME/hub (what an unpinned download picked)."""
    ref = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub" / (
        "datasets--" + repo.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def run_vlmevalkit(code: str, bench: dict, run_dir: Path) -> list[str]:
    cfg = {"model": {code: {**VLMEVAL_QWEN25VL, "model_path": str(M.view_dir(code))}},
           "data": {bench["data"]: bench["data_cfg"]}}
    cfg_path = run_dir / "vlmeval_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
    return [sys.executable, "-m", "eval_external.qa_transfer.vlmeval_launch", "--config", str(cfg_path),
            "--work-dir", str(run_dir / "vlmeval"), "--mode", "all", "--judge", "exact_matching", "--verbose",
            "--reuse"]


def run_lmms_eval(code: str, bench: dict, run_dir: Path) -> list[str]:
    args = {"pretrained": str(M.view_dir(code)), **LMMS_QWEN25VL_ARGS}
    return [sys.executable, "-m", "lmms_eval", "--model", "qwen2_5_vl", "--model_args",
            ",".join(f"{k}={v}" for k, v in args.items()), "--tasks", bench["task"], "--batch_size", "1",
            "--include_path", str(LMMS_INCLUDE), "--log_samples", "--output_path", str(run_dir / "lmms")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()
    bench = BENCHMARKS[args.bench]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not M.view_dir(args.model).exists():
        M.build_view(args.model)
    model = M.describe(args.model)
    source = dict(bench["source"])
    if args.bench.startswith("VSI-Bench"):
        source["videos_revision"] = hf_dataset_revision("nyu-visionx/VSI-Bench")
    if bench["backend"] == "vlmevalkit":
        cmd = run_vlmevalkit(args.model, bench, run_dir)
        backend = {"name": "VLMEvalKit (EASI v0.2.2 submodule)", "repo": git_state(VLMEVAL_DIR),
                   "model_class": VLMEVAL_QWEN25VL, "judge": "exact_matching"}
        decoding = VLMEVAL_DECODING
    else:
        cmd = run_lmms_eval(args.model, bench, run_dir)
        backend = {"name": "lmms-eval", "repo": git_state(LMMS_DIR), "model": "qwen2_5_vl",
                   "model_args": LMMS_QWEN25VL_ARGS}
        decoding = {"from": f"lmms_eval/tasks/{bench['task']} yaml generation_kwargs (greedy)",
                    "view_generation_config": {"do_sample": False, "repetition_penalty": 1.0}}
    write_manifest(run_dir, models=[model],
                   benchmark={"name": args.bench, "items_expected": bench["items"], "metric": bench["metric"],
                              "source": source},
                   backend={**backend, "command": cmd}, prompt=bench["prompt"], frames=bench["frames"],
                   decoding=decoding)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(WORKTREE), env.get("PYTHONPATH")) if p)
    rc = subprocess.run(cmd, cwd=str(VLMEVAL_DIR if bench["backend"] == "vlmevalkit" else LMMS_DIR),
                        env=env).returncode
    files = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
    (run_dir / "outputs.json").write_text(json.dumps({"returncode": rc, "files": files}, indent=1) + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
