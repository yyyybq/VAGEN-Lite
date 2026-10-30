"""configs/matrix_v3 (Qwen3-VL-8B; the user chose 2026-10-30 to follow the main line's registration of 2026-10-27,
configs/matrix_r12.json on qwen3-main d530b546): the group files agree with vars.yaml, with the registration, with the
mixture config and, where the cluster has them, with the built mixtures (python -m pytest data_gen/sft_mixture/tests -q)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from data_gen.sft_mixture import matrix_plan as MP

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs" / "matrix_v3"
MIXTURE = ROOT / "data_gen" / "sft_mixture" / "configs" / "mixture_matrix_v3_qwen3_20261030.yaml"
MIXTURE_V2 = ROOT / "data_gen" / "sft_mixture" / "configs" / "mixture_matrix_v2_20261018.yaml"
STANDARDS = ROOT / "data_gen" / "interiorgs_v10" / "configs" / "rl_standards.json"
GROUPS = ("A", "B", "C", "C0", "D", "D_noP0", "E")
TRAINED = ("B", "C", "D", "D_noP0")          # groups with their own SFT data
WITH_RL = ("B", "C", "D", "D_noP0", "E")
# the registration (qwen3-main d530b546, configs/matrix_r12.json): what each group is
REGISTERED = {
    "base_qwen3": "/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
                  "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "pair": {"s0": "v10_p1_S0", "d": "v10_p1_Dcond"},
    "dense": {"variant": "V4", "progress_reward_scale": 3.3333333333333335, "group_filter": "keep_by_return_std",
              "keep_std_threshold": 0.3318, "keep_std_floor": 1.0},
    "seeds": [20261019, 20261021],
    "guards": {"invalid_max": 0.10, "kl_max": 1.0},
    "attention": "eager",
    "pools": {"B": {"P0", "P2f"}, "C": {"P0", "P2q"}, "D": {"P0", "P1", "P2q"}},
    "P2f": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r12_20261019/format_sft_v2a",
}


def load(group):
    return yaml.safe_load((MATRIX / f"{group}.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfgs():
    return {g: load(g) for g in GROUPS}


@pytest.fixture(scope="module")
def vars_():
    return yaml.safe_load((MATRIX / "vars.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mix():
    return yaml.safe_load(MIXTURE.read_text(encoding="utf-8"))


def test_every_group_names_itself_and_the_new_format(cfgs, vars_):
    for g, c in cfgs.items():
        assert c["group"] == g and c["model_type"] == vars_["model_type"] == "qwen3_vl", g
        assert c["format_example"] == vars_["format_example"] == "action_names", g
    assert vars_["base_qwen3"] == REGISTERED["base_qwen3"]


def test_group_a_is_closed(cfgs):
    a = cfgs["A"]
    assert a["status"] == "closed" and "决定 74" in a["closed"]
    assert a["data"] is None and a["sft"] is None and a["rl"] is None and a["chain"] == {"plan": None}


def test_rl_follows_the_registration(cfgs, vars_):
    standards = json.loads(STANDARDS.read_text())
    s0, d = standards[REGISTERED["pair"]["s0"]], standards[REGISTERED["pair"]["d"]]
    assert {k for k in set(s0) | set(d) if s0.get(k) != d.get(k)} == {"reward_preset"}   # the pair differs only in the reward
    assert vars_["rl"]["pair"] == REGISTERED["pair"] and vars_["rl"]["dense"] == REGISTERED["dense"]
    assert vars_["rl"]["seeds"] == REGISTERED["seeds"] and vars_["rl"]["guards"] == REGISTERED["guards"]
    assert vars_["rl"]["attention"] == REGISTERED["attention"]
    for g in WITH_RL + ("C0",):
        rl = cfgs[g]["rl"]
        for key in ("pair", "dense", "seeds", "task_pool"):
            assert rl[key] == vars_["rl"][key], (g, key)
        if g != "C0":                                  # C0 is the main line's runs (its own rows and stop lines)
            for key in ("rows", "exclude", "updates", "attention"):
                assert rl[key] == vars_["rl"][key], (g, key)
            assert rl["guards"] == (vars_["rl"]["guards"] if g != "E" else {"invalid_max": 0.20, "kl_max": 1.0}), g
    e = cfgs["E"]["rl"]
    assert (e["invalid_until"], e["max_response_length"], e["output_format"]) == (50, 6144, "think_action")
    assert all("max_response_length" not in cfgs[g]["rl"] for g in ("B", "C", "D", "D_noP0"))


def test_eval_sets_are_the_screened_ones(cfgs, vars_):
    for g in WITH_RL:
        assert cfgs[g]["eval"]["sets"] == {"val": vars_["eval"]["val"], "test": vars_["eval"]["test"]}, g
        assert cfgs[g]["eval"]["every"] == vars_["eval"]["every"] == 50
    assert cfgs["E"]["eval"]["prompt_format"] == "free_think"


def test_data_recipes_match_the_mixture_config(cfgs, vars_, mix):
    assert mix["base_recipe"] == vars_["sft_recipe"] and mix["train_overrides"]["num_train_epochs"] == 10
    assert mix["seed"] == yaml.safe_load(MIXTURE_V2.read_text(encoding="utf-8"))["seed"]   # P0 / P1 draws as in v2
    assert mix["exclude_scenes"] == "configs/scene_screen/scene_exclusions_20261029.json"
    assert {p: s.get("format_example") for p, s in mix["pools"].items()} == {"P0": None, "P1": None, "P2q": "action_names",
                                                                            "P2f": "action_names"}
    assert mix["pools"]["P2f"]["train"].startswith(str(Path(REGISTERED["P2f"]).parent.parent / "interiorgs_v10_train_r12o_20261023"))
    assert mix["recipes"]["A"]["pools"] == {} and mix["recipes"]["E"]["same_as"] == "D"
    for g in TRAINED:
        c = cfgs[g]
        recipe = mix["recipes"][c["data"]["recipe"]]
        used = {p for p, n in c["data"]["train_rows"].items() if n}
        assert used == set(recipe["pools"]), g
        if g in REGISTERED["pools"]:
            assert used == REGISTERED["pools"][g], g
        assert c["data"]["lf_dir"] == f"{vars_['mixture_out']}/{g}" and c["data"]["mixture_config"] == vars_["mixture_config"]
        assert c["data"]["val_rows"] == {"P0": 17, "P1": 300, "P2q": 300, "P2f": 38}
        assert c["sft"]["base_model"] == vars_["base_qwen3"] and c["sft"]["epochs"] == 10
    assert mix["recipes"]["D"]["pools"]["P1"] == {"times": 1.0, "of": "P2q"}
    assert mix["recipes"]["D_noP0"]["draw_as"] == "D"
    assert mix["recipes"]["D_noP0"]["pools"] == {k: v for k, v in mix["recipes"]["D"]["pools"].items() if k != "P0"}


def test_built_mixtures_have_the_configured_rows(cfgs, vars_):
    out = Path(vars_["mixture_out"])
    if not out.is_dir():                              # the built data lives on the cluster
        pytest.skip(f"{out} not here")
    for g in TRAINED:
        rec = json.loads((out / g / "recipe.json").read_text(encoding="utf-8"))
        want = {p: n for p, n in cfgs[g]["data"]["train_rows"].items() if n}
        assert rec["counts"] == want, g
        assert {k[: -len("_val")]: v["rows"] for k, v in rec["summary"].items() if k.endswith("_val")} == cfgs[g]["data"]["val_rows"]
        y = yaml.safe_load((out / g / "train.yaml").read_text())
        assert y["model_name_or_path"] == vars_["base_qwen3"] and y["template"] == "qwen3_vl_nothink"
        assert y["output_dir"] == str((out / g / "model").resolve()) and y["save_strategy"] == "epoch"


def test_chain_part_waits_for_qwen3_main(cfgs, vars_):
    names = set()
    for g in WITH_RL:
        ch = cfgs[g]["chain"]
        assert ch["plan"] is None, g                   # no executable plan before qwen3-main is in main
        assert ch["sft_dir"] == vars_["format_check_sft_dir"] and ch["sft_groups"] == {"mix": "sft_trajectories.jsonl"}
        assert "_m3_" in ch["tr"] and ch["rl_name"].startswith("v10_rl_m3_"), g
        names |= {ch["tr"], ch["rl_name"]}
        for key in ch["needs_chain_support"]:
            assert key in MP.CHAIN_FIELDS, (g, key)     # the fields patches 0001 / 0002 read
        if g in TRAINED:
            assert ch["needs_chain_support"]["sft_lf_dirs"] == {"mix": cfgs[g]["data"]["lf_dir"]}, g
    assert len(names) == 2 * len(WITH_RL)              # every chain and run name is new
    assert cfgs["E"]["chain"]["needs_chain_support"]["start_model"] == {"from_chain": cfgs["D"]["chain"]["tr"]}


def test_c0_is_the_qwen3_sft_and_its_seed_pairs(cfgs):
    c0 = cfgs["C0"]
    assert c0["sft"]["model"].endswith("lf_q3formal/model/checkpoint-1232") and c0["sft"]["epoch"] == 8
    assert set(c0["runs"]) == {str(s) for s in REGISTERED["seeds"]}
    for seed, r in c0["runs"].items():
        chain = Path(r["chain"])
        if not chain.is_dir():
            continue
        for side in ("s0", "d"):
            for step in r[side]["checkpoints"]:
                hf = chain / r[side]["run"] / "checkpoints" / f"global_step_{step}" / "actor" / "huggingface"
                assert (hf / "config.json").is_file(), (seed, side, step)
