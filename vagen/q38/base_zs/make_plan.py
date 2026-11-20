"""q38 step 1 of the user's 2026-11-20 order ("27B按照实验矩阵的顺序去执行"): the base Qwen3.8-27B zero-shot, evaluated
only by the training chain (the user's rule), plan "eval_only".  Builds the chain folder, the copies of the task sets
(so no result is ever written into the 8B folders) and the plan(s); submits nothing.

    PYTHONPATH=. python -B vagen/q38/base_zs/make_plan.py --commit <q38 HEAD>            # plan A: actions, P0, probe
    PYTHONPATH=. python -B vagen/q38/base_zs/make_plan.py --commit <q38 HEAD> --easi     # plan B: EASI-8 (later)

Plan A, chain folder q38/runs/q38-base-zs (RUN):
- action tasks, 2-card jobs (plan eval_cards 2: card 0 the render service, card 1 vLLM at 0.9), each on
  RUN/sets/q38bz_eval_val_v2 (413), q38bz_eval_test_v2 (292) and q38bz_eval_test_v2_screened_new59 (the 59 tasks
  the scene screen put into the screened test set; the other 233 of its 292 and all 382 of the screened val set are
  byte-identical rows of the v2 sets, so the screened 674 is composed from these three, as the 8B's screened numbers
  were from its v2 results plus eval_test_v2_screened_topup):
  - q38base      greedy, --format-example action_names (as every 8B greedy evaluation; the 8B base has none);
  - q38base_t08  one sampled pass, T 0.8, top-p 0.92, --format-example action_names, --entropy, sample seed 20261016
                 (the evaluator's default) -- the 8B base's own options (models_runs/chains/m_qwen3vl8b_base_20261024,
                 model.json of m_qwen3vl8b_base_t08), the paper's base row;
- QA jobs, one card each (plan qa_eval): p0 (P0 test selection, 5,590 questions, 256 px; qa_eval p0, as
  plan_q3m3p0 for the 8B q3base) and probe (the P1 probe, 838 questions; qa_eval probe, as the 8B base_qwen3 of
  VAGEN-Lite-data/exps/diagnostics_20261101/probe_v3g_q3) of q38base; results RUN/qa/{p0,probe}/q38base.
Plan B (q38/runs/q38-base-zs-easi): EASI-8 of q38base (9 one-card jobs, SITE in its two parts; results
RUN/qa/<benchmark>/q38base beside plan A's), once Q38_EASI_PYTHON (lmms-eval's dependencies over the q38 env) exists.

Checks before writing anything: the model is qwen3_5; the sources' sha256 and row counts; the 59 new tasks; the P0
selection (5,590) and the probe (838); the chain folder does not exist; with plan A's probe, train_chain.py has the
qa_eval "probe" switch (the gated diff), else --no-probe.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

U = Path("/mnt/umm/users/yinbaiqiao")
WORKTREE = U / "VAGEN-Lite-q38"
MODEL = U / "q38/models/Qwen3.8-27B"
RUNS = U / "q38/runs"
RUN = RUNS / "q38-base-zs"
RUN_EASI = RUNS / "q38-base-zs-easi"
Q = U / "VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_q3_20261027"
SCREENED = U / "VAGEN-Lite-data/exps/matrix_inputs_20261029"
P0_SEL = U / "VAGEN-Lite/exps/interiorgs_qa_final_20261028/selection/test.jsonl"
PROBE = U / "VAGEN-Lite-data/exps/transition_qa_20261019b/p1_probe.jsonl"
SOURCES = {  # set name: (source tasks.jsonl, sha256, rows)
    "q38bz_eval_val_v2": (Q / "eval_val_v2/tasks.jsonl",
                          "db0f9a039369", 413),
    "q38bz_eval_test_v2": (Q / "eval_test_v2/tasks.jsonl", "de167c287136", 292),
}
SCREENED_TEST = (SCREENED / "eval_test_v2_screened/tasks.jsonl", "b31bfe77ea16", 292)
SCREENED_VAL = (SCREENED / "eval_val_v2_screened/tasks.jsonl", "f5596bf31cae", 382)
NEW59 = "q38bz_eval_test_v2_screened_new59"
SET_KEYS = {"val": "q38bz_eval_val_v2", "test": "q38bz_eval_test_v2", "new59": NEW59}
P0_ROWS, PROBE_ROWS = 5590, 838
GPU = "0.9"
GREEDY = ["--format-example", "action_names", "--gpu-memory-utilization", GPU]
# the 8B base's options, in its order (m_qwen3vl8b_base_20261024 plan.json), plus the card share
T08 = ["--temperature", "0.8", "--top-p", "0.92", "--format-example", "action_names", "--entropy",
       "--gpu-memory-utilization", GPU]
NAME, T08_SUFFIX = "q38base", "_t08"
Q38_EASI_PYTHON = Path("/mnt/umm/users/yinbaiqiao/q38/envs/easi/bin/python")  # as data_gen.interiorgs_v10.qa_eval


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def check_source(path: Path, prefix: str, n: int) -> None:
    h = sha(path)
    if not h.startswith(prefix) or len(rows(path)) != n:
        raise SystemExit(f"{path}: sha256 {h[:12]} / {len(rows(path))} rows, expected {prefix} / {n}")


def write_set(top: Path, name: str, lines, source: Dict) -> None:
    d = top / "sets" / name
    d.mkdir(parents=True)
    f = d / "tasks.jsonl"
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines) if not isinstance(lines, bytes)
                 else lines.decode())
    os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    (d / "source.json").write_text(json.dumps({**source, "sha256": sha(f), "rows": len(rows(f)),
                                               "copied_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                                              indent=1) + "\n")


def base_plan(tr: Path, commit: str) -> dict:
    return {"tr": str(tr), "sft_dir": str(tr / "sft_unused"), "commit": commit, "eval_only": True,
            "model_type": "qwen3_5", "model": str(MODEL), "job_prefix": "q38-bz", "pool": "zoetrope",
            "eval_cards": 2, "max_eval_jobs": 8, "retries": 2, "suspended_grace_min": 10}


def check_commit(commit: str) -> None:
    head = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if len(commit) < 8 or not head.startswith(commit):
        raise SystemExit(f"--commit {commit!r} is not the q38 checkout's HEAD {head[:8]} (commit the changes first)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", required=True, help="the q38 worktree's HEAD the chain checks (check_code)")
    ap.add_argument("--easi", action="store_true", help="plan B: EASI-8 (needs Q38_EASI_PYTHON)")
    ap.add_argument("--no-probe", action="store_true", help="plan A without the probe (train_chain lacks the switch)")
    a = ap.parse_args()
    check_commit(a.commit)
    cfg = json.loads((MODEL / "config.json").read_text())
    if cfg.get("model_type") != "qwen3_5":
        raise SystemExit(f"{MODEL}: model_type {cfg.get('model_type')}")
    if a.easi:
        if not Q38_EASI_PYTHON.is_file():
            raise SystemExit(f"{Q38_EASI_PYTHON} does not exist: EASI-8 cannot run in the q38 env (lmms-eval's "
                             "dependencies), build it first (the user's OK)")
        if RUN_EASI.exists():
            raise SystemExit(f"{RUN_EASI} exists")
        plan = base_plan(RUN_EASI, a.commit)
        plan.update({"max_eval_jobs": 9, "eval_sets": {},
                     "eval": {"models": {NAME: {"model": str(MODEL), "options": []}}},
                     "qa_eval": {"out": str(RUN / "qa"), "easi": "easi_8"},
                     "note": "q38 base zero-shot, plan B: EASI-8 (v0.2.2, lmms-eval qwen3_vl class with thinking off, "
                             "SITE as image + video parts) of the base Qwen3.8-27B, 9 one-card jobs; results "
                             "q38-base-zs/qa/<benchmark>/q38base."})
        RUN_EASI.mkdir(parents=True)
        (RUN_EASI / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
        print(RUN_EASI / "plan.json")
        return 0
    if RUN.exists():
        raise SystemExit(f"{RUN} exists")
    tc = (WORKTREE / "data_gen/interiorgs_v10/train_chain.py").read_text()
    probe_ok = 'qa.get("probe")' in tc
    if not probe_ok and not a.no_probe:
        raise SystemExit("train_chain.py has no qa_eval probe switch (the gated diff): apply it, or --no-probe")
    for name, (src, prefix, n) in SOURCES.items():
        check_source(src, prefix, n)
    check_source(*SCREENED_TEST)
    check_source(*SCREENED_VAL)
    test_ids = {r["task_id"] for r in rows(SOURCES["q38bz_eval_test_v2"][0])}
    val_rows = {r["task_id"]: r for r in rows(SOURCES["q38bz_eval_val_v2"][0])}
    new = [r for r in rows(SCREENED_TEST[0]) if r["task_id"] not in test_ids]
    kept_test = {r["task_id"]: r for r in rows(SOURCES["q38bz_eval_test_v2"][0])}
    if len(new) != 59 or any(kept_test[r["task_id"]] != r for r in rows(SCREENED_TEST[0]) if r["task_id"] in test_ids) \
            or any(val_rows.get(r["task_id"]) != r for r in rows(SCREENED_VAL[0])):
        raise SystemExit("the screened sets are not the v2 rows plus 59 new test tasks")
    if len(rows(P0_SEL)) != P0_ROWS or len(rows(PROBE)) != PROBE_ROWS:
        raise SystemExit("P0 selection / probe row counts changed")
    RUN.mkdir(parents=True)
    for name, (src, _, _) in SOURCES.items():
        write_set(RUN, name, src.read_bytes(), {"source": str(src)})
    write_set(RUN, NEW59, new, {"source": str(SCREENED_TEST[0]), "what": "the screened test set's tasks not in "
                                "eval_test_v2 (the scene screen's 59 replacements), in its order",
                                "task_ids": [r["task_id"] for r in new]})
    sets = {k: str(RUN / "sets" / v) for k, v in SET_KEYS.items()}
    plan = base_plan(RUN, a.commit)
    plan.update({
        "eval_sets": sets,
        "eval": {"models": {NAME: {"model": str(MODEL), "options": GREEDY}},
                 "variants": [{"suffix": T08_SUFFIX, "sets": list(sets), "options": T08, "models": [NAME]}]},
        "qa_eval": {"out": str(RUN / "qa"), "easi": [], "p0": str(P0_SEL),
                    **({} if a.no_probe else {"probe": str(PROBE)}), "models": [NAME]},
        "note": "q38 base zero-shot, plan A (the user 2026-11-20: 27B in the matrix's order, the base first): the base "
                "Qwen3.8-27B on val 413 / test 292 / the screened test's 59 new tasks, greedy (q38base) and one "
                "sampled pass as the 8B base (q38base_t08: T 0.8, top-p 0.92, entropy), 2-card jobs, vLLM at 0.9, "
                "thinking off; P0 test 5,590 and the P1 probe 838 of q38base (one-card QA jobs). NORMAL priority "
                "(VAGEN_LITE_JOB_PRIORITY in the chain's environment)."})
    (RUN / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    print(RUN / "plan.json")
    print(json.dumps({k: len(rows(Path(v) / "tasks.jsonl")) for k, v in sets.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
