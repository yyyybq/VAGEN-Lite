"""configs/matrix_v2: the five group configs agree with each other, with the mixture config, with the round-12 rules
(S0 + D pairs of rl_standards.json with one seed, SFT 10 epochs, the v2 evaluation sets) and with the plan fields
train_chain.py reads (python -m pytest data_gen/sft_mixture/tests -q)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs" / "matrix_v2"
MIXTURE = ROOT / "data_gen" / "sft_mixture" / "configs" / "mixture_matrix_v2_20261018.yaml"
CHAIN = ROOT / "data_gen" / "interiorgs_v10" / "train_chain.py"
STANDARDS = ROOT / "data_gen" / "interiorgs_v10" / "configs" / "rl_standards.json"
GROUPS = ("A", "B", "C", "D", "E")
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
PAIR = {"p1s0": {}, "p1d": {"rl": {"standard": "v10_p1_D"}}}


def load(g):
    return yaml.safe_load((MATRIX / f"{g}.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfgs():
    return {g: load(g) for g in GROUPS}


def test_known_keys_are_read_by_train_chain():
    src = CHAIN.read_text(encoding="utf-8")
    for key in PLAN_KEYS | RL_KEYS | GUARD_KEYS | STARTUP_KEYS | EVAL_KEYS:
        assert re.search(rf'["\']{key}["\']', src), f"train_chain.py does not read {key!r}"


def test_plans_use_only_fields_the_chain_reads(cfgs):
    for g, c in cfgs.items():
        plan = c["train_chain_plan"]
        assert set(plan) <= PLAN_KEYS, (g, set(plan) - PLAN_KEYS)
        assert set(plan["rl"]) <= RL_KEYS, (g, set(plan["rl"]) - RL_KEYS)
        assert set(plan["rl"]["guards"]) <= GUARD_KEYS and set(plan["rl"]["startup"]) <= STARTUP_KEYS
        assert set(plan["eval"]) <= EVAL_KEYS
        for run in plan["rl_runs"].values():
            assert set(run.get("rl") or {}) <= RL_KEYS
        assert c["group"] == g


def test_round12_rules(cfgs):
    standards = json.loads(STANDARDS.read_text())
    s0, d = standards["v10_p1_S0"], standards["v10_p1_D"]
    assert {k for k in s0 if s0[k] != d.get(k)} == {"reward_preset"}       # the pair differs only in the reward
    seeds = set()
    for g, c in cfgs.items():
        p = c["train_chain_plan"]
        assert p["rl"]["standard"] == "v10_p1_S0" and p["rl_runs"] == PAIR, g   # every RL experiment is a pair
        assert c["rl"]["pair"] == {"p1s0": "v10_p1_S0", "p1d": "v10_p1_D"} and c["rl"]["seed"] == p["rl"]["seed"]
        seeds.add(p["rl"]["seed"])
        assert p["eval_sets"] == {"val": V2 + "eval_val_v2", "test": V2 + "eval_test_v2"} and p["select_by"] == "val"
        assert p["rl"]["startup"] == {"retries": 2, "stall_minutes": 40}
        assert "ppo_template" not in p                                           # the standard names it
    assert len(seeds) == 1                                                       # one seed for every group and run


def test_shared_settings(cfgs):
    plans = {g: c["train_chain_plan"] for g, c in cfgs.items()}
    for key in ("sft_dir", "model", "eval_sets", "select_by", "endpoint", "pool"):
        assert len({yaml.safe_dump(p[key]) for p in plans.values()}) == 1, key
    for g, p in plans.items():
        assert p["rl"]["task_pool"] == "reserve"                                 # RL on the reserve tasks
        assert p["rl"]["rows"] == plans["A"]["rl"]["rows"] and p["rl"]["exclude"] == plans["A"]["rl"]["exclude"]
        assert p["sft_groups"] == {"mix": "sft_trajectories.jsonl"}             # SFT data from the formal tasks only
    assert len({p["tr"] for p in plans.values()}) == 5 and len({p["rl_name"] for p in plans.values()}) == 5


def test_data_recipes_match_the_mixture_config(cfgs):
    mix = yaml.safe_load(MIXTURE.read_text(encoding="utf-8"))
    assert mix["train_overrides"]["num_train_epochs"] == 10                      # round 12, decision 6
    for g in ("B", "C", "D"):
        c = cfgs[g]
        recipe = mix["recipes"][c["data"]["recipe"]]
        used = {p for p, n in c["data"]["train_rows"].items() if n}
        assert used == set(recipe["pools"]), g
        assert c["data"]["lf_dir"].endswith(f"sft_mixture_matrix_v2_20261018/{g}")
        assert c["needs_chain_support"]["sft_lf_dirs"] == {"mix": c["data"]["lf_dir"]}
        assert c["train_chain_plan"]["sft_epochs"] == c["sft"]["epochs"] == 10
        assert c["output_format"] == "action_only"
    assert mix["recipes"]["D"]["pools"]["P1"] == {"times": 1.0, "of": "P2"}       # 9,802 = the P2 count
    assert mix["recipes"]["E"]["same_as"] == "D"


def test_group_a_and_e(cfgs):
    a, e = cfgs["A"], cfgs["E"]
    assert a["sft"] is None and a["needs_chain_support"]["start_model"] == a["train_chain_plan"]["model"]
    assert e["output_format"] == "think_action" and e["sft"] == {"same_as": "D"}
    assert e["train_chain_plan"]["rl"]["guards"]["invalid_max"] == 0.20
    assert e["needs_chain_support"]["rl_output_format"] == "think_action"
    assert e["needs_chain_support"]["eval_rl_prompt_format"] == "free_think"
    assert set(e["pending"]) == {"think_share_metric", "max_response_length"}
    for g in "ABCD":
        assert cfgs[g]["train_chain_plan"]["rl"]["guards"]["invalid_max"] == 0.05
