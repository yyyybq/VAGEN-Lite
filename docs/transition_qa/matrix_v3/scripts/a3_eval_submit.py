"""Analysis 3 (goal swap, the user's instruction of 2026-11-06 and choice "甲：重找共同起点"): the 100-task evaluation set
(train and validation scenes, each task with its swap from the same shared start) and one single-card evaluation job
per checkpoint (inference only, sco, the pool with the shorter queue).  Usage: a3_eval_submit.py build | submit."""
import json
import os
import sys
from pathlib import Path

W = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix"
os.environ.update(VAGEN_LITE_WORKTREE=W, VAGEN_LITE_PYTHON="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime/bin/python",
                  VAGEN_LITE_MODEL_TYPE="qwen3_vl", VAGEN_LITE_GPU_LOG="1", TRANSFORMERS_ATTN_IMPLEMENTATION="eager")
sys.path.insert(0, W)
from data_gen.interiorgs_v10 import job_sco  # noqa: E402

M = Path(W) / "exps/matrix_v3"
A = M / "analysis_20261106/a3"
EV = A / "eval_pairs"
PARTS = ("train_standard", "train_swap", "val_standard", "val_swap")
RL = lambda chain, run, step: str(M / chain / f"rl_run_{run}/checkpoints/global_step_{step}/actor/huggingface")  # noqa: E731
C0 = "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026"
MODELS = {  # seed 20261019, S0 and V4, the pre-registered checkpoints (PREREG_D_vs_C section 2)
    "c0q3_s1s0_150": f"{C0}/rl_run_q3s0/checkpoints/global_step_150/actor/huggingface",
    "c0q3_s1d_300": f"{C0}/rl_run_q3d/checkpoints/global_step_300/actor/huggingface",
    "m3cs1s0_050": RL("m3_c_s20261019", "m3cs1s0", 50),
    "sft_m3c_e4": str(M / "sft/C/model/checkpoint-1204"),  # C seed 1 V4 stopped before its first checkpoint
    "m3crs1s0_250": RL("m3_cr_s20261019", "m3crs1s0", 250),
    "m3crs1d_250": RL("m3_cr_s20261019", "m3crs1d", 250),
    "m3ds1s0_300": RL("m3_d_s20261019", "m3ds1s0", 300),
    "m3ds1d_300": RL("m3_d_s20261019", "m3ds1d", 300),
    "m3dns1s0_300": RL("m3_dn_s20261019", "m3dns1s0", 300),
    "m3dns1d_300": RL("m3_dn_s20261019", "m3dns1d", 300),
}


def build() -> None:
    rows, parts = [], {}
    for p in PARTS:
        src = A / "pairs_run/out" / p
        rs = [json.loads(l) for l in (src / "tasks.jsonl").read_text().splitlines() if l.strip()]
        rows += rs
        parts[p] = {"source": str(src), "tasks": len(rs), "task_ids": [r["task_id"] for r in rs]}
    ids = [r["task_id"] for r in rows]
    assert len(set(ids)) == len(ids), "task ids repeat"
    for split in ("train", "val"):  # every standard task has its swap, from the same start
        std = {i[: -len("_ps")] for i in parts[f"{split}_standard"]["task_ids"]}
        swp = {i[: -len("_ps_swap")] for i in parts[f"{split}_swap"]["task_ids"]}
        assert std == swp, split
    text = "".join(json.dumps(r) + "\n" for r in rows)
    sel = {"version": "a3_goal_swap_pairs_20261106", "parts": parts, "tasks": len(rows),
           "note": "analysis 3 (exploratory): the screened train / validation tasks with a start shared by the task and "
                   "its goal swap (A and B exchanged, success region recomputed); a3/pairs_run/out/summary"}
    EV.mkdir(parents=True, exist_ok=True)
    if (EV / "tasks.jsonl").exists():
        assert (EV / "tasks.jsonl").read_text() == text, "eval_pairs/tasks.jsonl differs: not overwritten"
    else:
        (EV / "tasks.jsonl").write_text(text)
        (EV / "selection.json").write_text(json.dumps(sel, indent=1))
    missing = [n for n, m in MODELS.items() if not (Path(m) / "config.json").is_file()]
    assert not missing, missing
    print(f"{len(rows)} tasks in {EV}; {len(MODELS)} models found")


def submit() -> None:
    pool = job_sco.pick_pool()
    for name, model in MODELS.items():
        run = A / "eval_runs" / name
        if (run / "job" / "submission.json").exists() or (EV / name / "done.json").exists():
            print(name, "submitted before")
            continue
        job_sco.prepare(run, 1, ["eval", str(EV), name, model, "-", "--format-example", "action_names"], pool=pool)
        print(name, pool, job_sco.submit(run))


if __name__ == "__main__":
    {"build": build, "submit": submit}[sys.argv[1]]()
