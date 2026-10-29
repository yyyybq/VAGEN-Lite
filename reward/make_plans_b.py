"""Reward line, step 3 of task B: the training chain's plan for rw_s0 / rw_g / rw_o / rw_ok and its rows file.

The plan is the Qwen3 two-arm plan (models_runs/chains/m_qwen3vl8b_formal_rl_20261026/plan.json) with only these
keys changed: tr, rl_name, job_prefix (rw), eval_sets (the line's own result folders, linked to the main line's v2
sets), rl.rows (the subset rows file), rl_runs (the four arms), run_pools (two a pool), suspended_grace_min (10, the
main line's value since 4512b319), and commit (COMMIT_PLACEHOLDER, which reward/start_chain_b.sh replaces by the
worktree's HEAD).  No "retain": checkpoints are deleted only by hand, after
the results are in a report and the user has the list (the user's rule 7).

The rows file is the RL rows file with the reserve training rows cut to the 3,000 tasks that have tables (every
other row kept: the formal validation rows feed the RL's validation set), as the main line's C1 subset did; one
file (--rows-out) for both waves.

The user's decisions of 2026-10-29: two waves of two runs, each its own chain (--arms): rw_o + rw_s0 first (rw_o on
h800, rw_s0 on zoetrope), then rw_g + rw_ok, both on zoetrope, once the first wave is evaluated and cleaned up.

    python -m reward.make_plans_b --final-ids FILE [FILE ...] --tables DIR --keep-o THRESHOLD --tr DIR
        --arms rw_o,rw_s0 [--rows-out FILE] [--seed 20261019] [--pools rw_o=h800,rw_s0=zoetrope,...]
"""

import argparse
import json
from pathlib import Path

E = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial"
TEMPLATE = "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026/plan.json"
ROWS = f"{E}/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl"
RR = Path("/mnt/umm/users/yinbaiqiao/reward_runs")
V4 = {"standard": "v10_p1_Dcond", "group_filter": "keep_by_return_std", "keep_std_threshold": 0.3318,
      "keep_std_floor": 1.0, "progress_reward_scale": 10.0 / 3.0}


def arms(tables, keep_o):
    o = {"standard": "v10_p1_Dcond", "progress_version": "optimal", "distance_tables": str(tables),
         "progress_reward_scale": 10.0 / 3.0}
    return {"rw_s0": {}, "rw_g": {"rl": dict(V4)}, "rw_o": {"rl": dict(o)},
            "rw_ok": {"rl": dict(o, group_filter="keep_by_return_std", keep_std_threshold=float(keep_o),
                                 keep_std_floor=1.0)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-ids", nargs="+", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--keep-o", type=float, required=True)
    ap.add_argument("--tr", required=True)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--pools", default="rw_o=h800,rw_s0=zoetrope,rw_g=zoetrope,rw_ok=zoetrope",
                    help="run=pool,... (default: the user's decisions of 2026-10-29: the first wave rw_o on h800 and "
                         "rw_s0 on zoetrope, as submitted; the second wave rw_g + rw_ok on zoetrope, the user's later "
                         "rule that this round's new submissions go to zoetrope)")
    ap.add_argument("--arms", required=True, help="the runs of this chain, e.g. rw_o,rw_s0 (first wave)")
    ap.add_argument("--rows-out", default=str(RR / "tasks/rows_b3000.jsonl"))
    args = ap.parse_args()
    wanted = [a for a in args.arms.split(",") if a]

    final = sorted({l.strip() for f in args.final_ids for l in Path(f).read_text().splitlines() if l.strip()})
    assert len(final) == 3000, f"{len(final)} final tasks, not 3,000"
    tables = Path(args.tables)
    missing = [t for t in final if not (tables / f"{t}.npz").is_file()]
    assert not missing, f"{len(missing)} final tasks without a table, e.g. {missing[:3]}"
    keep = set(final)
    tr = Path(args.tr)
    tr.mkdir(parents=True, exist_ok=True)
    rows_out = Path(args.rows_out)
    lines, n_res = [], 0
    for l in Path(ROWS).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("reserve") and r.get("split") == "train":
            if r["task_id"] not in keep:
                continue
            n_res += 1
        lines.append(l + "\n")
    assert n_res == 3000, n_res
    text = "".join(lines)
    if rows_out.exists():  # the second wave reuses the first wave's file: it must be the same
        assert rows_out.read_text() == text, f"{rows_out} differs from the rows these tasks give"
    else:
        rows_out.write_text(text)
    n_all = len(lines)

    plan = json.loads(Path(TEMPLATE).read_text())
    plan.pop("dense_variant", None)
    plan.update(tr=str(tr), commit="COMMIT_PLACEHOLDER", rl_name="rw_b", job_prefix="rw",
                eval_sets={"val": str(RR / "eval_sets/eval_val_v2"), "test": str(RR / "eval_sets/eval_test_v2")},
                note=("Reward line, task B (docs/reward/PREREG_B.md): S0, G (= V4), O (optimal progress from the "
                      "depth-12 tables) and O-keep on the same 3,000 reserve tasks, Qwen3 SFT epoch 8, one seed; "
                      f"this chain runs {', '.join(wanted)} (the user's two waves of 2026-10-29)."))
    plan["rl"] = dict(plan["rl"], rows=str(rows_out), seed=args.seed)
    every = arms(tables, args.keep_o)
    assert set(wanted) <= set(every), wanted
    plan["rl_runs"] = {a: every[a] for a in wanted}
    pools = dict(x.split("=") for x in args.pools.split(","))
    plan["run_pools"] = {a: pools[a] for a in wanted}
    # the main line's setting (4512b319): a job sco shows SUSPENDED for less than 10 minutes is still waiting
    plan["suspended_grace_min"] = 10
    for k in ("rows", "exclude"):
        assert Path(plan["rl"][k]).exists(), plan["rl"][k]
    for s in plan["eval_sets"].values():
        assert Path(s, "tasks.jsonl").exists(), s
    (tr / "plan_template.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps({"tr": str(tr), "rows": str(rows_out), "rows_written": n_all, "reserve_train": n_res,
                      "runs": {k: v.get("rl", {}) for k, v in plan["rl_runs"].items()},
                      "run_pools": plan.get("run_pools"), "pool": plan.get("pool")}, indent=1))


if __name__ == "__main__":
    main()
