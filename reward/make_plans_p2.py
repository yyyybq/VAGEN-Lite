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
* ``jsrl``  direction 2, guided sampling, first round (docs/reward/reports/DESIGN_JSRL_20261102.md): S0 as rw_s0
  (reward, GRPO, standard filter) with the guide (vagen/jsrl_guide.py defaults: the brief's stages, d* >= 6 or
  equidistance, the SFT data's turns) on the b3000 tables.  Runs rw_jss (smoke), rw_js1.

Run names are never reused (evaluation result names are shared with the main line): the smoke is rw_tcs (endpoint 10,
no evaluation), the first-round runs rw_tc1 and rw_tc2 (the user's decision of 2026-11-02: rw_tc2 the turn term
alone, episode_weight 0, otherwise as rw_tc1; both in one chain, like task B's waves).

    python -m reward.make_plans_p2 --design tc --run rw_tcs --endpoint 10 --tr DIR [--pool zoetrope] [--seed 20261019]
    python -m reward.make_plans_p2 --design tc --run rw_tc1 --run rw_tc2 --run-turn-credit rw_tc2 '{"episode_weight": 0}' \
        --endpoint 150 --tr DIR --pool zoetrope
"""

import argparse
import json
from pathlib import Path

RR = Path("/mnt/umm/users/yinbaiqiao/reward_runs")
TEMPLATE = RR / "runs/rw_b_w1/plan.json"
TABLES = RR / "tables/b3000"


def designs(tables):
    return {"tc": {"progress_version": "optimal", "distance_tables": str(tables), "turn_credit": {},
                   "table_near": True, "group_filter": "nominal", "target_groups": 24},
            "jsrl": {"jsrl_guide": {"tables": str(tables)}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=sorted(designs(TABLES)))
    ap.add_argument("--run", action="append", required=True,
                    help="a run's name (never one used before), e.g. rw_tcs, rw_tc1; repeat for runs in one chain")
    ap.add_argument("--endpoint", type=int, required=True, help="10 for the smoke, 150 for the screening run")
    ap.add_argument("--tr", required=True)
    ap.add_argument("--pool", default="zoetrope")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--turn-credit", default=None, help="JSON settings over the design's (default: the estimator's)")
    ap.add_argument("--run-turn-credit", nargs=2, action="append", default=[], metavar=("RUN", "JSON"),
                    help="JSON settings over --turn-credit for one run")
    args = ap.parse_args()
    used = {n for p in RR.glob("runs/*/plan.json") for n in json.loads(p.read_text()).get("rl_runs", {})}
    assert len(set(args.run)) == len(args.run), args.run
    for run in args.run:
        assert run not in used, f"{run} was already used by a chain under {RR / 'runs'}"
    per_run = {name: json.loads(s) for name, s in args.run_turn_credit}
    assert set(per_run) <= set(args.run), f"--run-turn-credit for runs not in --run: {sorted(set(per_run) - set(args.run))}"
    tr = Path(args.tr)
    assert not (tr / "plan.json").exists(), f"{tr} already has a plan"
    plan = json.loads(TEMPLATE.read_text())
    base = dict(designs(TABLES)[args.design])
    if args.turn_credit:
        base["turn_credit"] = dict(base["turn_credit"], **json.loads(args.turn_credit))
    rls = {}
    for run in args.run:
        rl = dict(base)
        if run in per_run:
            rl["turn_credit"] = dict(rl["turn_credit"], **per_run[run])
        rls[run] = rl
    plan.update(tr=str(tr), commit="COMMIT_PLACEHOLDER", rl_name="rw_p2", endpoint=int(args.endpoint),
                rl_runs={run: {"rl": rl} for run, rl in rls.items()}, run_pools={run: args.pool for run in args.run},
                note=(f"Reward line, phase 2 (the user's brief of 2026-11-01), design {args.design}: runs "
                      f"{', '.join(args.run)}, endpoint {args.endpoint}, seed {args.seed}; task B's first-wave plan "
                      "otherwise (same 3,000 tasks, start, evaluation sets), so it compares with rw_s0 and rw_g."))
    plan["rl"] = dict(plan["rl"], seed=args.seed)
    for k in ("rows", "exclude"):
        assert Path(plan["rl"][k]).exists(), plan["rl"][k]
    for s in plan["eval_sets"].values():
        assert Path(s, "tasks.jsonl").exists(), s
    for tables in (base.get("distance_tables"), (base.get("jsrl_guide") or {}).get("tables")):
        assert tables is None or Path(tables).is_dir(), tables
    tr.mkdir(parents=True, exist_ok=True)
    (tr / "plan_template.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps({"tr": str(tr), "runs": rls, "endpoint": plan["endpoint"],
                      "run_pools": plan["run_pools"], "seed": plan["rl"]["seed"]}, indent=1))


if __name__ == "__main__":
    main()
