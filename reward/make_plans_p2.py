"""Reward line, phase 2 (the user's brief of 2026-11-01): the training chain's plans for the screening runs.

The plan is task B's first-wave plan (reward_runs/runs/rw_b_w1/plan.json: the same 3,000 reserve tasks with tables,
Qwen3 SFT epoch 8, seed 20261019, evaluation sets, guards, render settings) with only these keys changed: tr,
rl_name, rl_runs (one design), run_pools, endpoint, note, and commit (COMMIT_PLACEHOLDER, which
reward/start_chain_b.sh replaces by the worktree's HEAD).  So a design's checkpoints at 50 / 100 / 150 compare with
rw_s0 and rw_g of the same seed (brief section 3.2).  No "retain": checkpoints are deleted only by hand, after the
results are in a report and the user has the list.

Designs (rl-config options on top of the base plan's rl, standard v10_p1_S0):

* ``tc``  direction 1, per-turn credit, first round: the S0 reward; the turn term from the table's optimal progress
  (vagen/turn_credit.py defaults: shaping alone, batch-standardised, weight 0.5; the episode term the S0 return's
  GRPO advantage, batch-standardised); the neighbour fallback; every one of the 24 candidates in the update
  (nominal filter, target_groups 24).

Run names are never reused (evaluation result names are shared with the main line): the smoke is rw_tcs (endpoint 10,
no evaluation), the first-round run rw_tc1.

    python -m reward.make_plans_p2 --design tc --run rw_tcs --endpoint 10 --tr DIR [--pool zoetrope] [--seed 20261019]
"""

import argparse
import json
from pathlib import Path

RR = Path("/mnt/umm/users/yinbaiqiao/reward_runs")
TEMPLATE = RR / "runs/rw_b_w1/plan.json"
TABLES = RR / "tables/b3000"


def designs(tables):
    return {"tc": {"progress_version": "optimal", "distance_tables": str(tables), "turn_credit": {},
                   "table_near": True, "group_filter": "nominal", "target_groups": 24}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=sorted(designs(TABLES)))
    ap.add_argument("--run", required=True, help="the run's name (never one used before), e.g. rw_tcs, rw_tc1")
    ap.add_argument("--endpoint", type=int, required=True, help="10 for the smoke, 150 for the screening run")
    ap.add_argument("--tr", required=True)
    ap.add_argument("--pool", default="zoetrope")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--turn-credit", default=None, help="JSON settings over the design's (default: the estimator's)")
    args = ap.parse_args()
    used = {n for p in RR.glob("runs/*/plan.json") for n in json.loads(p.read_text()).get("rl_runs", {})}
    assert args.run not in used, f"{args.run} was already used by a chain under {RR / 'runs'}"
    tr = Path(args.tr)
    assert not (tr / "plan.json").exists(), f"{tr} already has a plan"
    plan = json.loads(TEMPLATE.read_text())
    rl = dict(designs(TABLES)[args.design])
    if args.turn_credit:
        rl["turn_credit"] = dict(rl["turn_credit"], **json.loads(args.turn_credit))
    plan.update(tr=str(tr), commit="COMMIT_PLACEHOLDER", rl_name="rw_p2", endpoint=int(args.endpoint),
                rl_runs={args.run: {"rl": rl}}, run_pools={args.run: args.pool},
                note=(f"Reward line, phase 2 (the user's brief of 2026-11-01), design {args.design}: run {args.run}, "
                      f"endpoint {args.endpoint}, seed {args.seed}; task B's first-wave plan otherwise (same 3,000 "
                      "tasks, start, evaluation sets), so it compares with rw_s0 and rw_g."))
    plan["rl"] = dict(plan["rl"], seed=args.seed)
    for k in ("rows", "exclude"):
        assert Path(plan["rl"][k]).exists(), plan["rl"][k]
    for s in plan["eval_sets"].values():
        assert Path(s, "tasks.jsonl").exists(), s
    assert Path(rl["distance_tables"]).is_dir(), rl["distance_tables"]
    tr.mkdir(parents=True, exist_ok=True)
    (tr / "plan_template.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps({"tr": str(tr), "run": args.run, "endpoint": plan["endpoint"], "rl": rl,
                      "run_pools": plan["run_pools"], "seed": plan["rl"]["seed"]}, indent=1))


if __name__ == "__main__":
    main()
