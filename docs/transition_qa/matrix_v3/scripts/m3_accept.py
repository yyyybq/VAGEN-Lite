"""A.2 (the user, 2026-11-03): the sampled acceptance of the three SFT starts -- C epoch 4, D epoch 9, C0 epoch 8 --
on the main line's acceptance set (200 RL reserve tasks; tasks.jsonl and selection.json copied here, as the main line
asked), 4 samples per task at temperature 0.8, top-p 0.92, with the answers' entropy: one evaluation-only chain, the
main line's way (its interiorgs_v10_train_q3c3acc_20261027 plan).  Pass: entropy <= 0.06, unreadable turns <= 1%.

    python exps/matrix_v3/watch/m3_accept.py start      # copy the set, write the plan, start the chain
    python exps/matrix_v3/watch/m3_accept.py summary    # the main line's accept_summary on the three results
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

W = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
M3 = W / "exps" / "matrix_v3"
PY = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python"
SRC = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_q3_20261027/eval_accept_reserve200")
SET = M3 / "eval_accept_reserve200"
TR = M3 / "accept_20261103"
SUMMARY = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-q3dev")   # accept_summary.py, qwen3-main-dev ea38cae3
SAMPLED = ["--samples", "4", "--temperature", "0.8", "--top-p", "0.92", "--entropy", "--format-example", "action_names"]
MODELS = {
    "sft_m3c_e4_s4": M3 / "sft" / "C" / "model" / "checkpoint-1204",
    "sft_m3d_e9_s4": M3 / "sft" / "D" / "model" / "checkpoint-3987",
    "c0q3_sft_e8_s4": Path("/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232"),
}
ENV = {**os.environ, "VAGEN_LITE_WORKTREE": str(W), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def start() -> None:
    if TR.exists():
        sys.exit(f"{TR} exists")
    for name, model in MODELS.items():
        if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
            sys.exit(f"{name}: no weights in {model}")
    SET.mkdir(parents=True, exist_ok=False)
    for f in ("tasks.jsonl", "selection.json"):
        shutil.copyfile(SRC / f, SET / f)           # the main line's selection.json is a link: copy what it names
        assert sha(SET / f) == sha((SRC / f).resolve()), f
    v = yaml.safe_load((W / "configs" / "matrix_v3" / "vars.yaml").read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "-C", str(W), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if subprocess.run(["git", "-C", str(W), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                      text=True, check=True).stdout.strip():
        sys.exit("tracked changes in the matrix worktree")
    plan = {
        "tr": str(TR), "commit": commit, "eval_only": True, "model_type": "qwen3_vl",
        "sft_dir": v["format_check_sft_dir"], "job_prefix": v["chain"]["job_prefix"], "pool": v["chain"]["pool"],
        "max_eval_jobs": 3, "retries": 4, "suspended_grace_min": 10, "gpu_log": True,
        "eval_sets": {"accept": str(SET)},
        "rl": {"rows": v["rl"]["rows"], "exclude": v["rl"]["exclude"]},
        "eval": {"models": {n: {"model": str(m), "options": SAMPLED} for n, m in MODELS.items()}},
        "note": ("A.2 (the user, 2026-11-03): the sampled acceptance of C epoch 4, D epoch 9 and C0 epoch 8 on the "
                 "main line's 200 reserve tasks x 4 at T 0.8, top-p 0.92 (pass: entropy <= 0.06, unreadable <= 1%)"),
    }
    TR.mkdir(parents=True)
    (TR / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    out = open(TR / "chain.log", "a")
    proc = subprocess.Popen(["nice", "-n", "19", PY, "-m", "data_gen.interiorgs_v10.train_chain", "run", "--plan",
                             str(TR / "plan.json")], cwd=W, env=ENV, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps({"tr": str(TR), "commit": commit, "pid": proc.pid, "set": str(SET),
                      "tasks_sha256": sha(SET / "tasks.jsonl")}, indent=1))


def summary() -> None:
    out = TR / "accept_summary.json"
    env = {**ENV, "PYTHONPATH": str(SUMMARY), "PYTHONDONTWRITEBYTECODE": "1"}   # nothing written in the main line's tree
    res = subprocess.run([PY, "-m", "data_gen.interiorgs_v10.accept_summary", "--set", str(SET), "--out", str(out),
                          *MODELS], cwd=SUMMARY, env=env, capture_output=True, text=True)
    print(res.stdout[-6000:], res.stderr[-3000:])


if __name__ == "__main__":
    {"start": start, "summary": summary}[sys.argv[1]]()
