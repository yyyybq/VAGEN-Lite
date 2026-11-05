"""One-card sco job that asks models the downstream line's question sets (X0, G0) with the downstream line's own scripts,
read by path from its checkout: downstream/x/q0_ask.py in the StarVLA environment's Python (greedy, the model's chat
template, its own preprocessor), then on the login node downstream/x/x0b_qa.py analyze and g0_qa.py analyze for the
scores.  The reward line, 2026-11-05: the user's understanding-mix screen evaluates X0 and G0 "with the downstream
line's questions and scripts, no substitutes"; the downstream line agreed to read-only use (its fcb173c6, --model-path).

The training chain's diagnostics step calls prepare() / answered() / analyze() for plan "diagnostics"."downstream_qa":

    {"sets": {"x0": X0/questions.jsonl, "g0": G0/questions.jsonl},
     "reference": {"x0": answers/x0_Q3B.jsonl, "g0": answers/g0_Q3B.jsonl},   # the base model's answers, scored alongside
     "downstream": VAGEN-Lite-downstream, "python": starVLA/bin/python, "family": "Q3B", "models_per_job": 6}

Answers go to OUT/<set>_<name>.jsonl (q0_ask skips ids already answered, so a rerun asks only what is missing), the
scores to OUT/<set>_scores.json (the downstream scripts' output) and OUT/summary.json.  Models are named "Q3_<name>" in
the scoring, as the downstream line asked: its G0 script compares a model with Q3B only when the name starts with Q3.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from data_gen.interiorgs_v10 import job_sco

CHAIN_POOL_MODES = ("auto", "queue", "idle")  # train_chain plan "pool" modes, not sco pools
ANALYZERS = {"x0": ("downstream/x/x0b_qa.py", ["--reference", "Q3B"]), "g0": ("downstream/x/g0_qa.py", [])}
REQUIRED = ("sets", "reference", "downstream", "python")

LAUNCH = """#!/usr/bin/env bash
set -euo pipefail
RUN={run}
OUT={out}
OWNER=$(stat -c %u:%g "$RUN")
mkdir -p "$RUN/logs" "$OUT"
trap 'status=$?; printf "{{\\"exit_status\\":%s,\\"ended_utc\\":\\"%s\\"}}\\n" "$status" "$(date -u +%FT%TZ)" > "$RUN/logs/downstream_qa_exit.json"; chown -R "$OWNER" "$RUN" "$OUT" 2>/dev/null || true; exit $status' EXIT
exec > >(tee -a "$RUN/logs/downstream_qa-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd {downstream}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
git -C {downstream} rev-parse HEAD || true
{asks}
"""


def check(dq: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED if not dq.get(k)]
    if missing:
        raise ValueError(f"downstream_qa needs {missing}")
    unknown = sorted(set(dq["sets"]) - set(ANALYZERS))
    if unknown or set(dq["reference"]) != set(dq["sets"]):
        raise ValueError(f"downstream_qa: sets must be among {sorted(ANALYZERS)} with a reference each, not "
                         f"{sorted(dq['sets'])} / {sorted(dq['reference'])}")


def _ids(path: Path) -> set:
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()


def answered(out_dir: Path, name: str, dq: Dict[str, Any]) -> bool:
    """Every question of every set answered for ``name``."""
    return all(_ids(Path(questions)) <= _ids(Path(out_dir) / f"{key}_{name}.jsonl")
               for key, questions in dq["sets"].items())


def run_name(names: Sequence[str]) -> str:
    """A job folder name of this list of models (a restart with other unanswered models gets another folder)."""
    return "dq_" + hashlib.sha1(",".join(sorted(names)).encode()).hexdigest()[:10]


def prepare(run: Path, dq: Dict[str, Any], models: Dict[str, str], pool: Optional[str], out_dir: Path) -> Path:
    """RUN/job/launch.sh and prepared.json (job_sco's format): one q0_ask.py call per model, every set at once."""
    check(dq)
    if pool in CHAIN_POOL_MODES or not pool:
        pool = job_sco.pick_pool()
    run, out_dir = Path(run).resolve(), Path(out_dir).resolve()
    job = run / "job"
    job.mkdir(parents=True, exist_ok=True)
    asks = []
    for name, path in models.items():
        args = [dq["python"], "downstream/x/q0_ask.py", "--model", dq.get("family", "Q3B"), "--model-path", str(path),
                "--smoke", "2"]
        for key, questions in dq["sets"].items():
            args += ["--ask", f"{Path(questions).resolve()}={out_dir / f'{key}_{name}.jsonl'}"]
        asks.append(" ".join(shlex.quote(a) for a in args))
    launch = job / "launch.sh"
    launch.write_text(LAUNCH.format(run=shlex.quote(str(run)), out=shlex.quote(str(out_dir)),
                                    downstream=shlex.quote(str(dq["downstream"])), asks="\n".join(asks)))
    launch.chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "gpus": 1, "spec": job_sco.SPECS[1],
        "pool": pool or job_sco.POOL, "mode_args": ["downstream-qa", str(run)], "models": models}, indent=1))
    return launch


def _run(cmd: List[str], cwd: str, log: Path) -> None:
    """A downstream script as the downstream line runs it: from its checkout, in its Python, without this checkout's
    PYTHONPATH (the chain's own commands run with it); one thread."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])} ... exited {res.returncode}: {res.stderr.strip()[-600:]}")


def analyze(out_dir: Path, names: Sequence[str], dq: Dict[str, Any], logs: Optional[Path] = None,
            runner: Optional[Callable[[List[str], str, Path], None]] = None) -> Dict[str, Any]:
    """The downstream scripts' scores of every model (and of the reference), then OUT/summary.json:
    {name: {set: {correct, n, accuracy, unparsed}}} plus the reference and the downstream checkout's commit."""
    check(dq)
    out_dir = Path(out_dir)
    for key, questions in dq["sets"].items():
        script, extra = ANALYZERS[key]
        answers = [f"Q3B={dq['reference'][key]}"] + [f"Q3_{n}={out_dir / f'{key}_{n}.jsonl'}" for n in names]
        cmd = [dq["python"], "-B", script, "analyze", "--questions", str(questions), "--answers", *answers, *extra,
               "--out", str(out_dir / f"{key}_scores.json")]
        (runner or _run)(cmd, str(dq["downstream"]), (Path(logs) if logs else out_dir) / f"downstream_qa_{key}.log")
    summary: Dict[str, Any] = {"downstream_commit": _commit(dq["downstream"]), "sets": dict(dq["sets"]),
                               "reference": dict(dq["reference"]), "models": {}}
    for key in dq["sets"]:
        scores = json.loads((out_dir / f"{key}_scores.json").read_text())
        for label, block in scores.items():
            name = "Q3B (reference)" if label == "Q3B" else label[len("Q3_"):] if label.startswith("Q3_") else label
            a = block["all"]
            summary["models"].setdefault(name, {})[key] = {
                "correct": a["correct"], "n": a["n"], "accuracy": a["accuracy"], "unparsed": a.get("unparsed")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def _commit(checkout: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None
