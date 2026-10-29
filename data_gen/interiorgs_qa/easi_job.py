"""Run the official EASI-8 suite (scripts/easi_eval.py, unchanged) on one checkpoint inside an SCO job.

    python -m data_gen.interiorgs_qa.sco prepare --job JOB --gpus 8 -- easi_job --name NAME --model MODEL_DIR --out OUT
    python -m data_gen.interiorgs_qa.sco submit --job JOB

scripts/easi_eval.py hands every benchmark to EASI v0.2.2 (third_party/EASI), which runs the lmms-eval task with the
model's HF weights; ``--nproc`` GPUs share the questions (accelerate data parallel). The SCO workers have no
internet: the benchmark data come from the shared offline cache, as in
examples/train/active_spatial/sco_active_spatial_eval_entry.sh (HF_HOME, HF_DATASETS_CACHE=datasets_easi_shared,
HF_*_OFFLINE=1). Results: OUT/NAME/easi_results.json (scores and sub-scores per benchmark) and EASI's logs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")       # the checkout: scripts/ and third_party/ are not in the job tarball
HF_HOME = "/mnt/umm/users/yinbaiqiao/.cache/huggingface"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="checkpoint name in the results")
    ap.add_argument("--model", required=True, help="HF model folder")
    ap.add_argument("--out", required=True, help="EASI output folder (one subfolder per checkpoint)")
    ap.add_argument("--benchmarks", default="easi_8")
    ap.add_argument("--nproc", type=int, default=0, help="GPUs (default: all visible)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: questions per benchmark")
    ap.add_argument("--stack-dump", type=int, default=0, metavar="SECONDS",
                    help="diagnosis: every Python process of the run prints all thread stacks to stderr every SECONDS "
                         "(a sitecustomize.py on PYTHONPATH); used to find where SITE video runs stalled (2026-10-29)")
    a = ap.parse_args()
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()]
    nproc = a.nproc or len(gpus)
    env = dict(os.environ, HF_HOME=HF_HOME, TRANSFORMERS_CACHE=f"{HF_HOME}/hub",
               HF_DATASETS_CACHE=f"{HF_HOME}/datasets_easi_shared", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               NO_PROXY="*", no_proxy="*", TOKENIZERS_PARALLELISM="false")
    cmd = [sys.executable, str(ROOT / "scripts/easi_eval.py"), "--ckpts", "", "--add", f"{a.name}:{a.model}:qwen2_5_vl",
           "--benchmarks", a.benchmarks, "--output_dir", a.out, "--gpu", ",".join(gpus[:nproc]), "--nproc", str(nproc),
           "--no-summary"]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stack_dump:
        d = Path(a.out) / f"_stack_dump_{os.getpid()}"
        d.mkdir(exist_ok=True)
        (d / "sitecustomize.py").write_text(
            "import faulthandler, sys\n"
            f"faulthandler.dump_traceback_later({a.stack_dump}, repeat=True, file=sys.stderr)\n")
        env["PYTHONPATH"] = str(d) + os.pathsep + env.get("PYTHONPATH", "")
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
