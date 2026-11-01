"""One-card sco job that runs ``qa_probe run`` from this checkout (the training script's job modes have no probe).

    python -m eval.diagnostics.sco_probe --run RUN --probe p1_probe.jsonl --from-eval EVAL --names a,b,c [--submit]
    # the text-only baseline: --model base_7b_blind=/path/to/Qwen2.5-VL-7B-Instruct --no-images
    # Qwen3-VL models: the Python of their runtime (vagen/models/model_types.py on qwen3-main)
    python -m eval.diagnostics.sco_probe ... --python /mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime/bin/python

Writes RUN/job/launch.sh and RUN/job/prepared.json in the format of data_gen/interiorgs_v10/job_sco.py and, with
--submit, submits through job_sco.submit (status / delete with job_sco.py as usual).  The environment block is the
one of examples/train/active_spatial/sco_interiorgs_v10_train.sh; sco jobs run as root, so RUN is handed back to
its owner on exit.  Results go to RUN/probe/<name>.json(l).  ``--python`` runs the probe with another Python (the
model type's runtime, as the job script's VAGEN_LITE_PYTHON; default the vagen-lite env).
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
from pathlib import Path

from data_gen.interiorgs_v10 import job_sco

CHECKOUT = Path(__file__).resolve().parents[2]
ENV_ROOT = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite"

LAUNCH = """#!/usr/bin/env bash
set -euo pipefail
RUN={run}
ENV_ROOT={env_root}
OUT={out}
OWNER=$(stat -c %u:%g "$RUN")
mkdir -p "$RUN/logs" "$OUT"
trap 'status=$?; printf "{{\\"exit_status\\":%s,\\"ended_utc\\":\\"%s\\"}}\\n" "$status" "$(date -u +%FT%TZ)" > "$RUN/logs/qa_probe_exit.json"; chown -R "$OWNER" "$RUN" "$OUT" 2>/dev/null || true; exit $status' EXIT
exec > >(tee -a "$RUN/logs/qa_probe-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd {checkout}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*'
export PYTHONPATH="{checkout}:${{PYTHONPATH:-}}" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0 CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=1
export CC=$ENV_ROOT/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV_ROOT/bin/x86_64-conda-linux-gnu-g++
export TORCH_EXTENSIONS_DIR=$RUN/torch_extensions
WHEEL_ROOT=$ENV_ROOT/lib/python3.12/site-packages/nvidia/curand
export CUDA_HOME=$ENV_ROOT CURAND_INCLUDE_DIR=$WHEEL_ROOT/include CURAND_LIBRARY_DIR=$WHEEL_ROOT/lib
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export CPATH="$CURAND_INCLUDE_DIR:$ENV_ROOT/targets/x86_64-linux/include:${{CPATH:-}}"
export LIBRARY_PATH="$CURAND_LIBRARY_DIR:$ENV_ROOT/targets/x86_64-linux/lib:${{LIBRARY_PATH:-}}"
export LD_LIBRARY_PATH="$CURAND_LIBRARY_DIR:$ENV_ROOT/lib:$ENV_ROOT/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
git -C {checkout} rev-parse HEAD || true
{python} -m eval.diagnostics.qa_probe run --probe {probe} --out "$OUT" {models}
"""


def prepare(run: Path, probe: str, from_eval: str = "", names: str = "", models=(), pool: str = None,
            out_dir: str = None, blind: bool = False, python: str = None) -> Path:
    """RUN/job/launch.sh and prepared.json; results go to ``out_dir`` (default RUN/probe), handed back to the owner.
    ``python``: the probe's Python (default the vagen-lite env's)."""
    run = run.resolve()
    job = run / "job"
    job.mkdir(parents=True, exist_ok=True)
    args = []
    if from_eval:
        args += ["--from-eval", from_eval, "--names", names]
    for m in models:
        args += ["--model", m]
    if blind:  # the text-only baseline (qa_probe run --no-images)
        args.append("--no-images")
    launch = job / "launch.sh"
    out = shlex.quote(str(Path(out_dir).resolve())) if out_dir else "\"$RUN/probe\""
    launch.write_text(LAUNCH.format(run=run, out=out, env_root=ENV_ROOT, checkout=CHECKOUT, probe=shlex.quote(str(Path(probe).resolve())),
                                    models=" ".join(shlex.quote(a) for a in args),
                                    python=shlex.quote(python) if python else "$ENV_ROOT/bin/python"))
    launch.chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "gpus": 1, "spec": job_sco.SPECS[1],
        "pool": pool or job_sco.POOL, "mode_args": ["qa-probe", str(run)], **({"python": python} if python else {})},
        indent=1))
    return launch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--from-eval", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--pool", choices=job_sco.POOLS, default=None)
    ap.add_argument("--out-dir", default=None, help="where the results go (default RUN/probe)")
    ap.add_argument("--no-images", action="store_true", help="the text-only baseline")
    ap.add_argument("--python", default=None, help="the probe's Python, e.g. the Qwen3-VL runtime's (default: vagen-lite)")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    print(prepare(a.run, a.probe, a.from_eval, a.names, a.model, a.pool, a.out_dir, a.no_images, a.python))
    if a.submit:
        print(job_sco.submit(a.run.resolve()))


if __name__ == "__main__":
    main()
