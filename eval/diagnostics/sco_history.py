"""One-card sco job that runs diagnostic 2 (``history_ablation run``) from this checkout: the render service and
vLLM, one greedy episode per task, with the history as it is, shuffled, or left out.

    python -m eval.diagnostics.sco_history --run RUN --eval EVAL --name NAME --model PATH --history shuffle \
        [--pool h800] [--submit] [-- EVALUATION_OPTIONS ...]

The training script's eval mode runs evaluation.py from the main checkout, which has no history_ablation; this job
uses the same environment block (examples/train/active_spatial/sco_interiorgs_v10_train.sh) from this checkout.
The verl submodule is read from the main checkout (it is not checked out in this worktree; nothing there is
written).  Results go to EVAL/NAME (evaluation.py), handed back to the owner with RUN on exit.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
from pathlib import Path

from data_gen.interiorgs_v10 import job_sco

CHECKOUT = Path(__file__).resolve().parents[2]
MAIN = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")  # read only: its verl submodule
ENV_ROOT = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite"
GSPLAT = "/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007"
GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"

LAUNCH = """#!/usr/bin/env bash
set -euo pipefail
RUN={run}
RESULTS={results}
ENV_ROOT={env_root}
PY=$ENV_ROOT/bin/python
mkdir -p "$RUN/logs"
OWNER=$(stat -c %u:%g "$RUN")
trap 'status=$?; [[ -n ${{renderer_pid:-}} ]] && kill "$renderer_pid" 2>/dev/null; printf "{{\\"exit_status\\":%s,\\"ended_utc\\":\\"%s\\"}}\\n" "$status" "$(date -u +%FT%TZ)" > "$RUN/logs/history_exit.json"; chown -R "$OWNER" "$RUN" "$RESULTS" 2>/dev/null || true; exit $status' EXIT
exec > >(tee -a "$RUN/logs/history-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd {checkout}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export PYTHONPATH="{main}/verl:{checkout}/scripts:{checkout}:${{PYTHONPATH:-}}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0
export CC=$ENV_ROOT/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV_ROOT/bin/x86_64-conda-linux-gnu-g++
export TORCH_EXTENSIONS_DIR=$RUN/torch_extensions
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2 INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10
WHEEL_ROOT=$ENV_ROOT/lib/python3.12/site-packages/nvidia/curand
export CUDA_HOME=$ENV_ROOT CURAND_INCLUDE_DIR=$WHEEL_ROOT/include CURAND_LIBRARY_DIR=$WHEEL_ROOT/lib
export PATH="$CUDA_HOME/bin:$ENV_ROOT/bin:$PATH"
export CPATH="$CURAND_INCLUDE_DIR:$ENV_ROOT/targets/x86_64-linux/include:${{CPATH:-}}"
export LIBRARY_PATH="$CURAND_LIBRARY_DIR:$ENV_ROOT/targets/x86_64-linux/lib:${{LIBRARY_PATH:-}}"
export LD_LIBRARY_PATH="$CURAND_LIBRARY_DIR:$ENV_ROOT/lib:$ENV_ROOT/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}"
export VLLM_USE_FLASHINFER_SAMPLER=1 CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
git -C {checkout} rev-parse HEAD || true
mkdir -p "$TORCH_EXTENSIONS_DIR"
[[ -d $TORCH_EXTENSIONS_DIR/gsplat_cuda ]] || cp -a {gsplat} "$TORCH_EXTENSIONS_DIR/gsplat_cuda"
bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root {gs_root} --host 0.0.0.0 --port 8915 \\
  --gpus 0 --max-workers 12 --max-inflight 12 --admit-timeout 900 --conda-env "$ENV_ROOT" > "$RUN/logs/renderer_service.log" 2>&1 &
renderer_pid=$!
for _ in $(seq 1 180); do
  "$PY" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8915/health", timeout=2)' 2>/dev/null && break
  kill -0 "$renderer_pid"; sleep 2
done
"$PY" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8915/health", timeout=2)'
export R1_RENDER_URL=http://127.0.0.1:8915/render
"$PY" -m eval.diagnostics.history_ablation run --history {history} -- --eval {eval} --name {name} --model {model} {options}
"""


def prepare(run: Path, eval_dir: str, name: str, model: str, history: str, options=(), pool: str = None) -> Path:
    run = run.resolve()
    job = run / "job"
    job.mkdir(parents=True, exist_ok=True)
    q = shlex.quote
    launch = job / "launch.sh"
    launch.write_text(LAUNCH.format(run=run, results=q(str(Path(eval_dir).resolve() / name)), env_root=ENV_ROOT,
                                    checkout=CHECKOUT, main=MAIN, gsplat=GSPLAT, gs_root=GS_ROOT, history=q(history),
                                    eval=q(str(Path(eval_dir).resolve())), name=q(name), model=q(model),
                                    options=" ".join(q(o) for o in options)))
    launch.chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "gpus": 1, "spec": job_sco.SPECS[1],
        "pool": pool or job_sco.POOL, "mode_args": ["history", str(run)]}, indent=1))
    return launch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--history", choices=("normal", "shuffle", "none"), required=True)
    ap.add_argument("--pool", choices=job_sco.POOLS, default=None)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("options", nargs="*", help="after --: options for evaluation.py run")
    a = ap.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    print(prepare(a.run, a.eval, a.name, a.model, a.history, a.options, a.pool))
    if a.submit:
        print(job_sco.submit(a.run.resolve()))


if __name__ == "__main__":
    main()
