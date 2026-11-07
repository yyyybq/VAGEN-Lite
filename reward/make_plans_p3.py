"""Reward line, phase 3 (the user's brief of 2026-11-06): plan templates for experiments A and D, and the screened-59
evaluation set.

Experiment D (docs/reward/PREREG_P3_D.md): one chain per run, each a copy of task B's plan (ROOT/runs/rw_b_w2/plan.json:
the same start C0 e8, the same 3,000 tasks, the same guards, 300 updates, checkpoints every 50) where only these change:
"tr", "commit" (COMMIT_PLACEHOLDER, filled in by reward/start_chain_p3.sh), "rl"."seed", "rl_runs" (one run), "run_pools"
(removed: the pool is "queue", the shorter queue), "rl_name", "eval_sets" (+ the screened-59 set) and "note".  The arms'
settings are copied from B's own plans: V4 = rl_runs.rw_g of rw_b_w2, S0 = rl_runs.rw_s0 of rw_b_w1 ({}), and the
ablation = S0 with "invalid_penalty" -1.0 (the env's invalid_format_penalty is -0.1; the value must be negative).

Experiment A (docs/reward/PREREG_P3_A.md): two evaluation-only chains run from the reward-mix worktree (the only code with
the QA steps): "diag" = the P1 probe and the downstream line's X0 / G0 for 12 models (copied from rw_p2_st1_p1 and
rw_mix_qa_ctl), "easi" = EASI-8 for the six validation-best checkpoints (copied from rw_p2_st1_easi).

    python -m reward.make_plans_p3 d --root REWARD_RUNS --arm rw_s2g [--arm ...]
    python -m reward.make_plans_p3 a --root REWARD_RUNS --c0 C0_E8_CHECKPOINT
    python -m reward.make_plans_p3 a2 --root REWARD_RUNS    # A's extension (PREREG_P3_A2): D's kept checkpoints
    python -m reward.make_plans_p3 scr59 --root REWARD_RUNS --screened SCREENED_TEST_DIR

Each plan is written to ROOT/runs/<chain>/plan_template.json (refused if the file exists).  scr59 writes
ROOT/eval_sets/eval_test_v2_screened59/tasks.jsonl: the rows of SCREENED_TEST_DIR/tasks.jsonl whose task id is not in
ROOT/eval_sets/eval_test_v2/tasks.jsonl, in file order, plus source.json (counts and sha256).  No default path: --root
is required.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

D_ARMS = {  # name: (reward, seed); PREREG_P3_D section 2
    "rw_s2g": ("v4", 20261021),
    "rw_s3s0": ("s0", 20261023),
    "rw_s3g": ("v4", 20261023),
    "rw_pen_s0": ("pen", 20261019),
}
PEN_INVALID = -1.0
D_CHAIN = {"rw_s2g": "rw_p3d_s2g", "rw_s3s0": "rw_p3d_s3s0", "rw_s3g": "rw_p3d_s3g", "rw_pen_s0": "rw_p3d_pen"}

# experiment A: the 11 kept checkpoints of task B (PREREG_P3_A section 2); EASI-8 only on the validation-best six
A_CKPTS = {
    "rw_s0_150": "rw_b_w1/rl_run_rw_s0/checkpoints/global_step_150",
    "rw_s0_200": "rw_b_w1/rl_run_rw_s0/checkpoints/global_step_200",
    "rw_o_200": "rw_b_w1/rl_run_rw_o/checkpoints/global_step_200",
    "rw_o_300": "rw_b_w1/rl_run_rw_o/checkpoints/global_step_300",
    "rw_g_250": "rw_b_w2/rl_run_rw_g/checkpoints/global_step_250",
    "rw_g_300": "rw_b_w2/rl_run_rw_g/checkpoints/global_step_300",
    "rw_ok_250": "rw_b_w2/rl_run_rw_ok/checkpoints/global_step_250",
    "rw_ok_300": "rw_b_w2/rl_run_rw_ok/checkpoints/global_step_300",
    "rw_s2s0_300": "rw_b_s2/rl_run_rw_s2s0/checkpoints/global_step_300",
    "rw_s2ok_250": "rw_b_s2/rl_run_rw_s2ok/checkpoints/global_step_250",
    "rw_s2ok_300": "rw_b_s2/rl_run_rw_s2ok/checkpoints/global_step_300",
}
A_EASI = ("rw_s0_150", "rw_s2s0_300", "rw_g_250", "rw_o_200", "rw_ok_250", "rw_s2ok_250")
C0_NAME = "sft_q3formal_e8"


def _read(p: Path) -> Dict[str, Any]:
    with open(p) as f:
        return json.load(f)


def _write_new(p: Path, obj: Dict[str, Any]) -> Path:
    if p.exists():
        raise SystemExit(f"{p} exists: refusing to overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return p


def d_plan(root: Path, arm: str) -> Dict[str, Any]:
    kind, seed = D_ARMS[arm]
    w2 = _read(root / "runs" / "rw_b_w2" / "plan.json")
    w1 = _read(root / "runs" / "rw_b_w1" / "plan.json")
    spec = {"v4": copy.deepcopy(w2["rl_runs"]["rw_g"]), "s0": copy.deepcopy(w1["rl_runs"]["rw_s0"]),
            "pen": {"rl": {**copy.deepcopy((w1["rl_runs"]["rw_s0"].get("rl") or {})), "invalid_penalty": PEN_INVALID}}}[kind]
    p = copy.deepcopy(w2)
    chain = D_CHAIN[arm]
    p["tr"] = str(root / "runs" / chain)
    p["commit"] = "COMMIT_PLACEHOLDER"
    p["rl"]["seed"] = seed
    p["rl_runs"] = {arm: spec}
    p.pop("run_pools", None)
    p["rl_name"] = "rw_p3d"
    p["eval_sets"] = {**w2["eval_sets"], "scr59": str(root / "eval_sets" / "eval_test_v2_screened59")}
    p["note"] = (f"Reward line, phase 3, experiment D (docs/reward/PREREG_P3_D.md; the user's brief of 2026-11-06): "
                 f"{arm} = {'V4 (rw_g settings)' if kind == 'v4' else 'S0' if kind == 's0' else 'S0 with the unreadable-answer penalty -1.0 (was -0.1)'}, "
                 f"seed {seed}, 300 updates; a copy of task B's plan rw_b_w2 with only tr, commit, seed, rl_runs, "
                 f"run_pools (removed: pool queue), rl_name, eval_sets (+ the screened-59 set) and this note changed; "
                 f"started from the frozen worktree VAGEN-Lite-reward-p3run.")
    return p


def a_plans(root: Path, c0: str) -> Dict[str, Dict[str, Any]]:
    runs = root / "runs"
    p1, ctl, easi = (_read(runs / n / "plan.json") for n in ("rw_p2_st1_p1", "rw_mix_qa_ctl", "rw_p2_st1_easi"))
    models = {n: {"model": str(runs / rel / "actor" / "huggingface"), "options": []} for n, rel in A_CKPTS.items()}
    models[C0_NAME] = {"model": c0, "options": []}
    base = {k: ctl[k] for k in ("model_type", "sft_dir", "retries", "suspended_grace_min", "eval_only", "pool",
                                "job_prefix")}
    diag = {"tr": str(runs / "rw_p3a_diag"), "commit": "COMMIT_PLACEHOLDER", **base, "eval_sets": {},
            "eval": {"models": models},
            "diagnostics": {"failure_breakdown": False, "probes": copy.deepcopy(p1["diagnostics"]["probes"]),
                            "models_per_job": 6,
                            "downstream_qa": copy.deepcopy(ctl["diagnostics"]["downstream_qa"])},
            "note": "Reward line, phase 3, experiment A (docs/reward/PREREG_P3_A.md): the P1 probe (838) and the "
                    "downstream line's X0 (510) / G0 (300) of task B's 11 kept checkpoints and C0 e8, through the "
                    "diagnostics step of the reward-mix chain (VAGEN-Lite-reward-mix; read-only use of the P1 line's "
                    "probe and the downstream line's questions and scripts)."}
    e_models = {n: models[n] for n in A_EASI}
    e = {"tr": str(runs / "rw_p3a_easi"), "commit": "COMMIT_PLACEHOLDER", **base, "eval_sets": {},
         "qa_eval": copy.deepcopy(easi["qa_eval"]), "max_eval_jobs": 9, "eval": {"models": e_models},
         "note": "Reward line, phase 3, experiment A (docs/reward/PREREG_P3_A.md): EASI-8 (official and MindCube "
                 "rejudged, reward/easi8_score.py) of task B's six validation-best checkpoints, 54 single-card jobs, "
                 "through the reward-mix chain's qa_eval (the same EASI code as qwen3-main 22a0fc62)."}
    return {"rw_p3a_diag": diag, "rw_p3a_easi": e}


# the extension of A (docs/reward/PREREG_P3_A2.md, the user's 2026-11-07): D's kept checkpoints
A2_CKPTS = {
    "rw_s2g_150": "rw_p3d_s2g/rl_run_rw_s2g/checkpoints/global_step_150",
    "rw_s2g_300": "rw_p3d_s2g/rl_run_rw_s2g/checkpoints/global_step_300",
    "rw_s3s0_300": "rw_p3d_s3s0/rl_run_rw_s3s0/checkpoints/global_step_300",
    "rw_s3g_300": "rw_p3d_s3g/rl_run_rw_s3g/checkpoints/global_step_300",
}
A2_EASI = ("rw_s2g_150", "rw_s3s0_300", "rw_s3g_300")


def a2_plans(root: Path) -> Dict[str, Dict[str, Any]]:
    """A's two plans with D's kept checkpoints (no C0, no B checkpoints; rw_s2s0_300 is reused from A)."""
    base = a_plans(root, "/unused")
    runs = root / "runs"
    models = {n: {"model": str(runs / rel / "actor" / "huggingface"), "options": []} for n, rel in A2_CKPTS.items()}
    diag, easi = base["rw_p3a_diag"], base["rw_p3a_easi"]
    diag.update(tr=str(runs / "rw_p3a2_diag"), eval={"models": models},
                note="Reward line, phase 3, the extension of experiment A (docs/reward/PREREG_P3_A2.md): X0 / G0 / P1 of "
                     "experiment D's four kept checkpoints, the same reward-mix chain and code as rw_p3a_diag.")
    easi.update(tr=str(runs / "rw_p3a2_easi"), eval={"models": {n: models[n] for n in A2_EASI}},
                note="Reward line, phase 3, the extension of experiment A (docs/reward/PREREG_P3_A2.md): EASI-8 of D's "
                     "three validation-best checkpoints (rw_s2s0_300 reused from rw_p3a_easi), 27 single-card jobs.")
    return {"rw_p3a2_diag": diag, "rw_p3a2_easi": easi}


# experiment C (docs/reward/PREREG_P3_C.md, the user's 2026-11-07 decision 3): the smoke, then one V4 arm with und_mix
C_POOL = "und_mix/pool_c0_20261107"


def c_plan(root: Path, which: str) -> Dict[str, Any]:
    """which: "smoke" (10 updates, run rw_mx_smoke) or "arm" (300 updates, run rw_mx_g); both = rw_g's plan + und_mix."""
    w2 = _read(root / "runs" / "rw_b_w2" / "plan.json")
    p = copy.deepcopy(w2)
    spec = copy.deepcopy(w2["rl_runs"]["rw_g"])
    spec.setdefault("rl", {})["und_mix"] = {"pool": str(root / C_POOL), "batch_size": 128}
    name = {"smoke": "rw_mx_smoke", "arm": "rw_mx_g"}[which]
    p["tr"] = str(root / "runs" / {"smoke": "rw_p3c_smoke", "arm": "rw_p3c_mx_g"}[which])
    p["commit"] = "COMMIT_PLACEHOLDER"
    p["rl_runs"] = {name: spec}
    p.pop("run_pools", None)
    p["rl_name"] = "rw_p3c"
    if which == "smoke":  # no evaluation (as rw_mix_smoke): no sets, no pick, no evaluation stop rule
        p["endpoint"] = 10
        p["eval_sets"] = {}
        p.pop("select_by", None)
        p["eval"] = {k: v for k, v in p["eval"].items() if k != "stop_invalid"}
    else:
        p["eval_sets"] = {**w2["eval_sets"], "scr59": str(root / "eval_sets" / "eval_test_v2_screened59")}
    p["note"] = (f"Reward line, phase 3, experiment C (docs/reward/PREREG_P3_C.md; the user's decision 3 of 2026-11-07): "
                 f"{name} = V4 (rw_g settings) + und_mix (pool {C_POOL}, batch 128, interval 2, weight 0.1), seed "
                 f"{p['rl']['seed']}, {p['endpoint']} updates; a copy of task B's plan rw_b_w2 with tr, commit, rl_runs, "
                 f"run_pools (removed), rl_name, "
                 f"{'endpoint, eval_sets (none), select_by and eval stop_invalid (removed)' if which == 'smoke' else 'eval_sets (+ screened 59)'} and this "
                 f"note changed; started from a frozen worktree of reward-design.")
    return p


def scr59(root: Path, screened: Path) -> Dict[str, Any]:
    orig = root / "eval_sets" / "eval_test_v2" / "tasks.jsonl"
    with open(orig) as f:
        have = {json.loads(l)["task_id"] for l in f if l.strip()}
    src = screened / "tasks.jsonl"
    rows: List[str] = []
    n = 0
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if json.loads(line)["task_id"] not in have:
                rows.append(line if line.endswith("\n") else line + "\n")
    out = root / "eval_sets" / "eval_test_v2_screened59"
    if (out / "tasks.jsonl").exists():
        raise SystemExit(f"{out}/tasks.jsonl exists: refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    text = "".join(rows)
    (out / "tasks.jsonl").write_text(text)
    info = {"source": str(src), "source_rows": n, "excluded_in": str(orig), "rows": len(rows),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "note": "phase 3 (PREREG_P3_D section 5): the screened test set's tasks that are not in eval_test_v2; the "
                    "screened 674 = eval_val_v2 filtered to the screened val ids (382) + eval_test_v2 filtered (233) "
                    "+ these"}
    (out / "source.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("d")
    d.add_argument("--root", required=True)
    d.add_argument("--arm", action="append", required=True, choices=sorted(D_ARMS))
    a = sub.add_parser("a")
    a.add_argument("--root", required=True)
    a.add_argument("--c0", required=True)
    a2 = sub.add_parser("a2")
    a2.add_argument("--root", required=True)
    cc = sub.add_parser("c")
    cc.add_argument("--root", required=True)
    cc.add_argument("--which", required=True, choices=("smoke", "arm"))
    s = sub.add_parser("scr59")
    s.add_argument("--root", required=True)
    s.add_argument("--screened", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root)
    if args.cmd == "d":
        for arm in args.arm:
            p = d_plan(root, arm)
            print(_write_new(Path(p["tr"]) / "plan_template.json", p))
    elif args.cmd == "a":
        for name, p in a_plans(root, args.c0).items():
            print(_write_new(Path(p["tr"]) / "plan_template.json", p))
    elif args.cmd == "a2":
        for name, p in a2_plans(root).items():
            print(_write_new(Path(p["tr"]) / "plan_template.json", p))
    elif args.cmd == "c":
        p = c_plan(root, args.which)
        print(_write_new(Path(p["tr"]) / "plan_template.json", p))
    else:
        print(json.dumps(scr59(root, Path(args.screened)), indent=1))


if __name__ == "__main__":
    main()
