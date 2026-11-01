"""Qwen3 main line (the user's matrix batch 1 instruction of 2026-11-01, section 4; decision 2 甲): the official EASI
evaluation of one model on one benchmark, as a single-card job the training chain submits (plan "qa_eval",
train_chain.qa_jobs; job script mode qa) in the model type's Python.

    python -m data_gen.interiorgs_v10.qa_eval easi --name NAME --model HF_DIR --out OUT/BENCH --benchmark BENCH \
        --model-type qwen3_vl [--limit N]

As the P0 line's data_gen/interiorgs_qa/easi_job.py does for Qwen2.5-VL: scripts/easi_eval.py (which now also accepts
qwen3_vl) hands the benchmark to EASI v0.2.2 (third_party/EASI), which runs the lmms-eval task with the model's HF
weights -- for Qwen3-VL lmms-eval's qwen3_vl class with its defaults, EASI's official configuration.  The jobs have no
internet: the benchmark data come from the shared offline cache (HF_HOME, HF_DATASETS_CACHE=datasets_easi_shared,
HF_*_OFFLINE=1).  SITE runs with the stack-dump thread on, as easi_job.py (the P0 line's one-card SITE video runs
stalled without it, 5 of 5, and finished with it, 5 of 5; the cause is not known).  One folder a benchmark (OUT is
OUT/BENCH), so two benchmarks of one model never share lmms-eval's result files.  Results: OUT/NAME/easi_results.json
and EASI's logs; OUT/NAME/done.json once the benchmark has its score (the chain's mark that the job is done; a smoke
run's records its --limit, so a smoke's output folder is never a full run's).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]  # scripts/easi_eval.py; third_party/ is read from the registry's paths
HF_HOME = "/mnt/umm/users/yinbaiqiao/.cache/huggingface"
STACK_DUMP_S = 900
EASI_8 = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site", "blink", "3dsrbench", "embspatial")
# SITE in two jobs (its image and video parts), as the P0 line ran it after whole-SITE runs stalled (2026-10-29/30)
EASI_8_JOBS = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site_image", "site_video", "blink",
               "3dsrbench", "embspatial")


def easi(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HOME=HF_HOME, TRANSFORMERS_CACHE=f"{HF_HOME}/hub",
               HF_DATASETS_CACHE=f"{HF_HOME}/datasets_easi_shared", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               NO_PROXY="*", no_proxy="*", TOKENIZERS_PARALLELISM="false")
    if a.benchmark.startswith("site"):
        d = out / f"_stack_dump_{os.getpid()}"
        d.mkdir(exist_ok=True)
        (d / "sitecustomize.py").write_text("import faulthandler, sys\n"
                                            f"faulthandler.dump_traceback_later({STACK_DUMP_S}, repeat=True, "
                                            "file=sys.stderr)\n")
        env["PYTHONPATH"] = str(d) + os.pathsep + env.get("PYTHONPATH", "")
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()]
    cmd = [sys.executable, str(WORKTREE / "scripts/easi_eval.py"), "--ckpts", "",
           "--add", f"{a.name}:{a.model}:{a.model_type}", "--benchmarks", a.benchmark, "--output_dir", str(out),
           "--gpu", gpus[0], "--nproc", "1", "--no-summary"]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=WORKTREE, env=env).returncode
    result = out / a.name / "easi_results.json"
    scores = json.loads(result.read_text()).get("scores", {}) if result.is_file() else {}
    if rc == 0 and scores.get(a.benchmark) is not None:  # a smoke run (--limit) says so: never a full result
        (out / a.name / "done.json").write_text(json.dumps({"benchmark": a.benchmark, "model": a.model,
                                                            "model_type": a.model_type, "limit": a.limit,
                                                            "score": scores[a.benchmark]}) + "\n")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("easi")
    e.add_argument("--name", required=True, help="the model's name in the results")
    e.add_argument("--model", required=True, help="HF model folder")
    e.add_argument("--out", required=True, help="the benchmark's output folder (one subfolder a model)")
    e.add_argument("--benchmark", required=True, help="one EASI benchmark key (vsi_bench, site, ...)")
    e.add_argument("--model-type", default="qwen3_vl", choices=("qwen2_5_vl", "qwen3_vl"))
    e.add_argument("--limit", type=int, default=None, help="smoke test: questions (done.json records the limit)")
    a = ap.parse_args()
    return easi(a)


if __name__ == "__main__":
    raise SystemExit(main())
