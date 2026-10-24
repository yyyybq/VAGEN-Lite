"""The model line's step-2 smoke (the user's brief of 2026-10-24, sections 2.4-2.5): the evaluation-set folders of
the line and the training-chain plans, written once (an existing plan is never overwritten).

    python -m model_matrix.smoke.plans --commit HEAD

Plans (each chain runs from the worktree: VAGEN_LITE_WORKTREE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models):
- m_qwen3vl8b_base:  zero-shot evaluation of Qwen3-VL-8B-Instruct on the 705 tasks (validation 413 + test 292), one
                     episode per task sampled as the RL rollout samples (temperature 0.8, top-p 0.92), the new format
                     example, token entropy recorded;
- m_qwen3vl8b_smoke: SFT on 440 new-format conversations for one epoch (the C0 recipe otherwise; loss logged each
                     step), its evaluation on both sets, RL from it for 10 updates (the dense arm the line uses today,
                     V4: D-cond + keep, threshold 0.3318, floor 1.0, progress scale 10 / 3; checkpoint at 10), the
                     rollout-vs-trainer log-probability audit on, the update-10 checkpoint evaluated on both sets;
- m_qwen25vl7b_smoke: the same for Qwen2.5-VL-7B from the original model: the default path end to end, and the
                     log-probability audit's reference;
- m_cambrian7b_base, m_cambrian7b_smoke: the first two for Cambrian-S-7B (the user's section 3; the line's download,
                     model_type cambrian_qwen: its SFT tool, the concat-path expansion, the evaluator's Cambrian
                     branch);
- m_qwen25vl7b_base: the zero-shot row of the original Qwen2.5-VL-7B in the same setting, for the comparison (the
                     user's decision 3 of 2026-10-24 22:45; the main line has no row at temperature 0.8 on these sets).
Every job is named models-* and goes only to idle machines (plan pool "idle").  Results go to the line's own
evaluation folders (tasks.jsonl and selection.json linked from the main line's v2 sets), so no result folder of the
main line is written.  ``--recommit`` sets the commit of plans whose chain has not started (no progress.json): the
chain's check_code refuses a checkout at another commit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("/mnt/umm/users/yinbaiqiao/models_runs")
E = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial")
V2 = E / "interiorgs_v10_r12_20261019"
QWEN3VL = ("/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
           "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
QWEN25VL = str(E / "interiorgs_v10_r11_20261016" / "inputs" / "base_7b")
CAMBRIAN = ("/mnt/umm/shared_model/vagen-lite-models/hub/models--nyu-visionx--Cambrian-S-7B/snapshots/"
            "556e72fded09352f6aa4e9c100c9f39e6fe0fba7")
SFT_DIR = RUNS / "sft_v10_newformat_20261024"
EVAL_SETS = {"val": RUNS / "eval_sets" / "eval_val_v2", "test": RUNS / "eval_sets" / "eval_test_v2"}
RL_BASE = {"standard": "v10_p1_Dcond", "seed": 20261019, "task_pool": "reserve",
           "rows": str(E / "interiorgs_v10_r11_20261016" / "rl_rows" / "rows.jsonl"),
           "exclude": str(E / "interiorgs_v10_r10_20261015" / "depth_check" / "depth_check_summary.json"),
           "logprob_audit": True, "checkpoint_every": 10}
V4 = {"standard": "v10_p1_Dcond", "group_filter": "keep_by_return_std", "keep_std_threshold": 0.3318,
      "keep_std_floor": 1.0, "progress_reward_scale": 10.0 / 3.0}
SAMPLED = ["--temperature", "0.8", "--top-p", "0.92", "--format-example", "action_names", "--entropy"]


def eval_sets() -> None:
    for name, d in EVAL_SETS.items():
        src = V2 / ("eval_val_v2" if name == "val" else "eval_test_v2")
        d.mkdir(parents=True, exist_ok=True)
        for f in ("tasks.jsonl", "selection.json"):
            if not (d / f).exists():
                (d / f).symlink_to(src / f)


def smoke(tag: str, model: str, model_type, group: str, run: str, commit: str) -> dict:
    plan = {"tr": str(RUNS / "chains" / tag), "sft_dir": str(SFT_DIR), "commit": commit, "model": model,
            "job_prefix": "models", "gs_root": "/mnt/umm/users/yinbaiqiao/InteriorGS_ply",
            "scenes_root": "/mnt/umm/users/yinbaiqiao/InteriorGS_meta", "keep_text": True, "sft_records": False,
            "sft_epochs": 1, "sft_logging_steps": 1, "sft_groups": {group: "sft_trajectories_smoke.jsonl"},
            "eval_sets": {k: str(v) for k, v in EVAL_SETS.items()}, "select_by": "val", "max_eval_jobs": 2,
            "pool": "idle", "retries": 1, "endpoint": 10, "rl_name": f"m_{group}",
            "eval": {"every": 10, "options": []}, "rl": dict(RL_BASE), "rl_runs": {run: {"rl": dict(V4)}}}
    if model_type:
        plan["model_type"] = model_type
    return plan


def base_eval(tag: str, model: str, model_type, result: str, commit: str) -> dict:
    plan = {"tr": str(RUNS / "chains" / tag), "sft_dir": str(SFT_DIR), "commit": commit, "eval_only": True,
            "job_prefix": "models", "pool": "idle", "max_eval_jobs": 2,
            "eval_sets": {k: str(v) for k, v in EVAL_SETS.items()},
            "eval": {"options": [], "models": {result: {"model": model, "options": SAMPLED}}}}
    if model_type:  # none: the default Qwen2.5-VL path
        plan["model_type"] = model_type
    return plan


def plans(commit: str, date: str) -> dict:
    return {
        f"m_qwen3vl8b_base_{date}": base_eval(f"m_qwen3vl8b_base_{date}", QWEN3VL, "qwen3_vl", "m_qwen3vl8b_base_t08",
                                              commit),
        f"m_qwen3vl8b_smoke_{date}": smoke(f"m_qwen3vl8b_smoke_{date}", QWEN3VL, "qwen3_vl", "q3smoke", "q3v4smoke",
                                           commit),
        f"m_qwen25vl7b_smoke_{date}": smoke(f"m_qwen25vl7b_smoke_{date}", QWEN25VL, None, "q25smoke", "q25v4smoke",
                                            commit),
        f"m_cambrian7b_base_{date}": base_eval(f"m_cambrian7b_base_{date}", CAMBRIAN, "cambrian_qwen",
                                               "m_cambrian7b_base_t08", commit),
        f"m_cambrian7b_smoke_{date}": smoke(f"m_cambrian7b_smoke_{date}", CAMBRIAN, "cambrian_qwen", "csmoke",
                                            "cv4smoke", commit),
        f"m_qwen25vl7b_base_{date}": base_eval(f"m_qwen25vl7b_base_{date}", QWEN25VL, None, "m_qwen25vl7b_base_t08",
                                               commit),
    }


def write(all_plans: dict, recommit: bool = False) -> None:
    for tag, plan in all_plans.items():
        tr = Path(plan["tr"])
        tr.mkdir(parents=True, exist_ok=True)
        path = tr / "plan.json"
        if path.exists():
            old = json.loads(path.read_text())
            if recommit and not (tr / "progress.json").exists() and old.get("commit") != plan["commit"]:
                path.write_text(json.dumps(dict(old, commit=plan["commit"]), indent=1))
                print("recommitted", path, old.get("commit"), "->", plan["commit"])
            else:
                print("kept", path)
            continue
        path.write_text(json.dumps(plan, indent=1))
        print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--date", default="20261024")
    ap.add_argument("--recommit", action="store_true", help="give plans not yet started this commit")
    a = ap.parse_args()
    eval_sets()
    write(plans(a.commit, a.date), a.recommit)


if __name__ == "__main__":
    main()
