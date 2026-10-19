"""configs/matrix_v2: the group configs agree with each other, with the mixture config, with the round-12 rules
(S0 + D pairs of rl_standards.json with one seed, SFT 10 epochs, the v2 evaluation sets), with the plan fields
train_chain.py reads, and fill in the RL standard variable for either package (python -m pytest data_gen/sft_mixture/tests -q)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from data_gen.sft_mixture import matrix_plan as MP

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs" / "matrix_v2"
MIXTURE = ROOT / "data_gen" / "sft_mixture" / "configs" / "mixture_matrix_v2_20261018.yaml"
CHAIN = ROOT / "data_gen" / "interiorgs_v10" / "train_chain.py"
STANDARDS = ROOT / "data_gen" / "interiorgs_v10" / "configs" / "rl_standards.json"
GROUPS = ("A", "B", "C", "D", "E")       # groups that train; C0 reuses round 12
PACKAGES = ("v10_p1", "v10_p2")
V2 = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r12_20261019/"

# the plan fields train_chain.py reads (checked against its source below, so a renamed field fails here)
PLAN_KEYS = {"tr", "sft_dir", "commit", "model", "ppo_template", "gs_root", "scenes_root", "keep_text", "sft_epochs",
             "sft_groups", "eval_sets", "select_by", "max_eval_jobs", "pool", "endpoint", "rl_name", "rl", "rl_runs",
             "eval", "rl_fallback_model", "val_limit", "extend_to", "eval_only"}
RL_KEYS = {"history_mode", "critic_warmup", "test_freq", "val_tasks", "train_batch_size", "ppo_mini_batch_size",
           "save_freq", "greedy_val", "no_internal_val", "task_pool", "rows", "exclude", "opt_package", "candidates",
           "target_groups", "rollouts", "signal_check", "guards", "standard", "seed", "startup"}
GUARD_KEYS = {"kl_max", "invalid_max", "invalid_window", "success_drop", "drop_window", "start_updates"}
STARTUP_KEYS = {"retries", "stall_minutes"}
EVAL_KEYS = {"every", "options", "models", "variants", "extra_sets", "dir"}


def pair(package):
    short = package.split("_", 1)[1]
    return {f"{short}s0": {}, f"{short}d": {"rl": {"standard": f"{package}_D"}}}


@pytest.fixture(scope="module")
def cfgs():
    return {g: MP.load_group(g) for g in GROUPS}


def test_known_keys_are_read_by_train_chain():
    src = CHAIN.read_text(encoding="utf-8")
    for key in PLAN_KEYS | RL_KEYS | GUARD_KEYS | STARTUP_KEYS | EVAL_KEYS:
        assert re.search(rf'["\']{key}["\']', src), f"train_chain.py does not read {key!r}"


def test_plans_use_only_fields_the_chain_reads(cfgs):
    for g, c in cfgs.items():
        plan = MP.plan(g)
        assert set(plan) <= PLAN_KEYS, (g, set(plan) - PLAN_KEYS)
        assert set(plan["rl"]) <= RL_KEYS, (g, set(plan["rl"]) - RL_KEYS)
        assert set(plan["rl"]["guards"]) <= GUARD_KEYS and set(plan["rl"]["startup"]) <= STARTUP_KEYS
        assert set(plan["eval"]) <= EVAL_KEYS
        for run in plan["rl_runs"].values():
            assert set(run.get("rl") or {}) <= RL_KEYS
        assert c["group"] == g


def test_rl_standard_is_a_variable():
    default = yaml.safe_load((MATRIX / "vars.yaml").read_text(encoding="utf-8"))["rl_package"]
    assert default == "v10_p1"
    for g in GROUPS:
        raw = yaml.safe_load((MATRIX / f"{g}.yaml").read_text(encoding="utf-8"))["train_chain_plan"]
        text = json.dumps(raw)
        assert "{rl_package}" in text and "v10_p1" not in text and "v10_p2" not in text, g
        for package in PACKAGES:                         # one switch moves every group to the other package
            p = MP.plan(g, package)
            assert p["rl"]["standard"] == f"{package}_S0" and p["rl_runs"] == pair(package), (g, package)
    with pytest.raises(SystemExit):
        MP.plan("D", "v10_p9")                           # no such standard
    assert MP.plan("D", commit="abc")["commit"] == "abc"


def test_round12_rules(cfgs):
    standards = json.loads(STANDARDS.read_text())
    for package in PACKAGES:
        s0, d = standards[f"{package}_S0"], standards[f"{package}_D"]
        assert {k for k in s0 if s0[k] != d.get(k)} == {"reward_preset"}   # the pair differs only in the reward
    seeds = set()
    for g, c in cfgs.items():
        p = MP.plan(g)
        assert c["rl"]["pair"] == {"p1s0": "v10_p1_S0", "p1d": "v10_p1_D"} and c["rl"]["seed"] == p["rl"]["seed"]
        seeds.add(p["rl"]["seed"])
        assert p["eval_sets"] == {"val": V2 + "eval_val_v2", "test": V2 + "eval_test_v2"} and p["select_by"] == "val"
        assert p["rl"]["startup"] == {"retries": 2, "stall_minutes": 40}
        assert "ppo_template" not in p                                       # the standard names it
    assert len(seeds) == 1                                                   # one seed for every group and run


def test_shared_settings(cfgs):
    plans = {g: MP.plan(g) for g in cfgs}
    for key in ("sft_dir", "model", "eval_sets", "select_by", "endpoint", "pool"):
        assert len({yaml.safe_dump(p[key]) for p in plans.values()}) == 1, key
    for g, p in plans.items():
        assert p["rl"]["task_pool"] == "reserve"                             # RL on the reserve tasks
        assert p["rl"]["rows"] == plans["A"]["rl"]["rows"] and p["rl"]["exclude"] == plans["A"]["rl"]["exclude"]
        assert p["sft_groups"] == {"mix": "sft_trajectories.jsonl"}         # SFT data from the formal tasks only
    assert len({p["tr"] for p in plans.values()}) == 5 and len({p["rl_name"] for p in plans.values()}) == 5


def test_data_recipes_match_the_mixture_config(cfgs):
    mix = yaml.safe_load(MIXTURE.read_text(encoding="utf-8"))
    assert mix["train_overrides"]["num_train_epochs"] == 10                  # round 12, decision 6
    for g in ("B", "C", "D"):
        c = cfgs[g]
        recipe = mix["recipes"][c["data"]["recipe"]]
        used = {p for p, n in c["data"]["train_rows"].items() if n}
        assert used == set(recipe["pools"]), g
        assert c["data"]["lf_dir"].endswith(f"sft_mixture_matrix_v2_20261018/{g}")
        assert c["needs_chain_support"]["sft_lf_dirs"] == {"mix": c["data"]["lf_dir"]}
        assert MP.plan(g)["sft_epochs"] == c["sft"]["epochs"] == 10
        assert c["output_format"] == "action_only"
    assert mix["recipes"]["D"]["pools"]["P1"] == {"times": 1.0, "of": "P2"}   # 9,802 = the P2 count
    assert mix["recipes"]["E"]["same_as"] == "D"


def test_group_a_and_e(cfgs):
    a, e = cfgs["A"], cfgs["E"]
    assert a["sft"] is None and a["needs_chain_support"]["start_model"] == MP.plan("A")["model"]
    assert e["output_format"] == "think_action" and e["sft"] == {"same_as": "D"}
    assert MP.plan("E")["rl"]["guards"]["invalid_max"] == 0.20
    assert e["needs_chain_support"]["rl_output_format"] == "think_action"
    assert e["needs_chain_support"]["eval_rl_prompt_format"] == "free_think"
    assert e["needs_chain_support"]["rl_max_response_length"] == e["rl"]["max_response_length"] == 6144
    assert "pending" not in e and set(e["decided"]) == {"think_share", "max_response_length"}
    for g in "ABCD":
        assert MP.plan(g)["rl"]["guards"]["invalid_max"] == 0.05
        assert "max_response_length" not in cfgs[g]["rl"]                  # 6,144 only for E


def test_d_nop0_is_d_without_p0():
    alt, d = MP.load_group("D_noP0"), MP.load_group("D")
    pa, pd = MP.plan("D_noP0"), MP.plan("D")
    assert {k for k in pd if pa.get(k) != pd.get(k)} == {"tr", "rl_name"}           # same SFT / RL / eval settings
    assert alt["data"]["train_rows"] == {"P0": 0, "P1": 9802, "P2": 9802}
    assert {k: v for k, v in alt.items() if k not in ("group", "description", "status", "data", "train_chain_plan",
                                                        "needs_chain_support")} == \
        {k: v for k, v in d.items() if k not in ("group", "description", "status", "data", "train_chain_plan",
                                                 "needs_chain_support")}
    mix = yaml.safe_load(MIXTURE.read_text(encoding="utf-8"))["recipes"]
    assert mix["D_noP0"]["draw_as"] == "D"                                           # the same P1 / P2 rows as D
    assert mix["D_noP0"]["pools"] == {k: v for k, v in mix["D"]["pools"].items() if k != "P0"}
    assert alt["needs_chain_support"]["sft_lf_dirs"]["mix"].endswith("/D_noP0")


def test_c0_reuses_the_round12_pair():
    c0 = MP.load_group("C0")
    assert "train_chain_plan" not in c0 and c0["sft"]["model"].endswith("lf_formal/model/checkpoint-1386")
    assert c0["sft"]["epoch"] == 9 and c0["data"]["train_rows"] == {"P0": 0, "P1": 0, "P2": 9802}
    d = MP.plan("D")
    for package in PACKAGES:
        r = MP.reuse("C0", package)
        short = package.split("_", 1)[1]
        assert r["rl_runs"] == {f"{short}s0": f"rl_run_{short}s0", f"{short}d": f"rl_run_{short}d"}
        path = Path(r["plan"])
        if not path.exists():                            # the round-12 plans live in the main checkout's exps
            continue
        r12 = json.loads(path.read_text())
        assert r12["tr"] == r["tr"]
        assert r12["rl"]["standard"] == f"{package}_S0" and r12["rl_runs"] == pair(package)
        for key in ("seed", "task_pool", "rows", "exclude", "guards", "startup"):  # the same RL as every matrix group
            assert r12["rl"][key] == d["rl"][key], (package, key)
        for key in ("endpoint", "eval_sets", "eval", "sft_dir", "model"):
            assert r12[key] == d[key], (package, key)
