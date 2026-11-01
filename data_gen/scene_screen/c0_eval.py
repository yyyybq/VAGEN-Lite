"""C0 on the screened test set (the user, 2026-10-29): evaluate C0's step-300 pair (restored from AOSS) on the 59 tasks
the screen added to eval_test_v2_screened, plus 30 kept tasks (5 per family, seeded) whose stored 2026-10-19 results
check that a rerun today gives the same episodes.  `prepare` writes the task folder, `submit` one 1-card eval job per
checkpoint (the main line's eval mode, run from its checkout, results in this branch's folder)."""
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data")
sys.path.insert(0, str(ROOT))
from data_gen.interiorgs_v10 import job_sco  # noqa: E402

R12 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r12_20261019")
SCREENED = ROOT / "exps/matrix_inputs_20261029/eval_test_v2_screened"
EVAL = ROOT / "exps/matrix_inputs_20261029/eval_test_v2_screened_c0_topup"
RESTORE = ROOT / "exps/c0_restore_20261029"
MODELS = {"c0_p1s0_300": RESTORE / "rl_run_p1s0/global_step_300/actor/huggingface",
          "c0_p1d_300": RESTORE / "rl_run_p1d/global_step_300/actor/huggingface"}
L = lambda p: [json.loads(l) for l in open(p) if l.strip()]  # noqa: E731


def prepare():
    old = {r["task_id"] for r in L(R12 / "eval_test_v2/tasks.jsonl")}
    scr = L(SCREENED / "tasks.jsonl")
    new = [r for r in scr if r["task_id"] not in old]
    kept = [r for r in scr if r["task_id"] in old]
    rng = random.Random(20261029)
    by = collections.defaultdict(list)
    for r in sorted(kept, key=lambda r: r["task_id"]):
        by[r["task_type"]].append(r)
    control = [r for f in sorted(by) for r in rng.sample(by[f], 5)]
    EVAL.mkdir(parents=True, exist_ok=True)
    rows = new + control
    (EVAL / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (EVAL / "selection.json").write_text(json.dumps({
        "purpose": "C0 step 300 on the tasks the scene screen added to the test set, plus kept tasks as a rerun check",
        "source": str(SCREENED), "new": len(new), "control": len(control),
        "new_task_ids": [r["task_id"] for r in new], "control_task_ids": [r["task_id"] for r in control],
        "control_by_family": dict(collections.Counter(r["task_type"] for r in control))}, indent=1))
    print(json.dumps({"tasks": len(rows), "new": len(new), "control": len(control)}))


def submit():
    pool = job_sco.pick_pool()
    for name, model in MODELS.items():
        assert (model / "config.json").is_file() and len(list(model.glob("model-*.safetensors"))) == 7, model
        run = RESTORE / "eval_runs" / name
        run.mkdir(parents=True, exist_ok=True)
        if not (run / "job" / "submission.json").exists():
            job_sco.prepare(run, 1, ["eval", str(EVAL), name, str(model), "-"], pool=pool)
            print(name, pool, job_sco.submit(run))


if __name__ == "__main__":
    {"prepare": prepare, "submit": submit}[sys.argv[1]]()
