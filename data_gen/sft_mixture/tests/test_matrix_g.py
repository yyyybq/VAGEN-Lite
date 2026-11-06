"""Group G of matrix v3 ("point first, then act", the user's instruction of 2026-11-06; prep only): its group file
agrees with D's registration except for the targets format and the pointing reward, its mixture configs with the
group file, and its plans carry the format, the pointing reward (both sides) and the pointing probe to the chain."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from data_gen.sft_mixture import matrix_plan as MP

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs" / "matrix_v3"
SHA = "0123456789abcdef0123456789abcdef01234567"


def load(name):
    return yaml.safe_load((MATRIX / f"{name}.yaml").read_text(encoding="utf-8"))


def test_g_is_d_with_the_targets_format_and_the_pointing_reward():
    from vagen.envs.active_spatial.prompt import TARGET_FORMATS

    g, d, v = load("G"), load("D"), MP.v3_vars()
    assert g["group"] == "G" and g["status"] == "prep" and g["model_type"] == "qwen3_vl"
    assert g["format_example"] in TARGET_FORMATS and g["output_format"] == d["output_format"]
    assert g["rl"]["pointing_reward"] == 0.1
    same = ("pair", "dense", "seeds", "task_pool", "rows", "exclude", "updates", "output_format", "guards", "attention")
    assert {k: g["rl"][k] for k in same} == {k: d["rl"][k] for k in same}
    assert g["sft"]["base_model"] == d["sft"]["base_model"] == v["base_qwen3"] and g["sft"]["epochs"] == d["sft"]["epochs"]
    assert g["eval"]["sets"] == d["eval"]["sets"]
    assert MP.V3_CELLS["G"] == ("g", 2)
    assert g["format_example"] in g["eval"]["pg_probe"] and g["format_example"] in g["chain"]["sft_dir"]


def test_g_mixtures():
    from vagen.envs.active_spatial.prompt import TARGET_FORMATS

    g = load("G")
    for fmt in TARGET_FORMATS:
        short = TARGET_FORMATS[fmt]
        cfg = yaml.safe_load((ROOT / "data_gen/sft_mixture/configs" / f"mixture_matrix_v3_qwen3_g_{short}_20261106.yaml")
                             .read_text(encoding="utf-8"))
        rec = cfg["recipes"]["G"]
        assert rec["draw_as"] == "D" and rec["pools"] == {"P0": "all", "P1": 9075, "P2t": "all", "Pg": 9075}
        assert cfg["pools"]["P2t"]["format_example"] == fmt
        assert all(f"/{fmt}/" in cfg["pools"][p][s] for p in ("P2t", "Pg") for s in ("train", "val"))
        assert cfg["val"] == {"P0": 300, "P1": 300, "P2q": 300, "P2f": "all", "P2t": "all", "Pg": 300}
        if fmt == g["format_example"]:
            assert g["data"]["mixture_config"].endswith(f"mixture_matrix_v3_qwen3_g_{short}_20261106.yaml")
            assert g["data"]["lf_dir"].endswith(f"sft_mixture_g_{short}_20261106/G")


def test_g_plans_reach_the_chain(tmp_path):
    from data_gen.interiorgs_v10 import train_chain as TC

    g = load("G")
    sft = MP.plan_v3_cell_sft("G", SHA)
    assert sft["sft_only"] and sft["sft_lf_dirs"] == {"m3g": g["chain"]["sft_lf_dir"]}
    assert sft["rl"]["format_example"] == g["format_example"]
    start = "/m3_g_sft/lf_m3g/model/checkpoint-1"
    plans = [MP.plan_v3_cell_rl("G", s, SHA, start, dense="registered") for s in g["rl"]["seeds"]]
    assert [list(p["rl_runs"]) for p in plans] == [["m3gs1s0", "m3gs1d"], ["m3gs2s0", "m3gs2d"]]
    for p in plans:
        assert (p["rl"]["format_example"], p["rl"]["pointing_reward"]) == (g["format_example"], 0.1)
        assert p["diagnostics"]["probes"]["Pg"] == g["eval"]["pg_probe"]
        p["tr"] = str(tmp_path / p["rl_name"])
        chain = TC.Chain(p, sleep=lambda s: None)
        runs = chain.rl_runs()
        assert all(r["rl"]["pointing_reward"] == 0.1 and r["rl"]["format_example"] == g["format_example"]
                   for r in runs.values())
        calls = {}
        chain.sh = lambda args, **k: calls.__setitem__(args[args.index("--out") + 1], args)
        chain.rl_config()
        for args in calls.values():
            assert args[args.index("--pointing-reward") + 1] == "0.1"
            assert args[args.index("--format-example") + 1] == g["format_example"]
    # the other cells and groups: no pointing reward, the matrix format, no pointing probe
    d = MP.plan_v3_cell_rl("D_noP0", 20261019, SHA, start, dense="registered")
    assert "pointing_reward" not in d["rl"] and d["rl"]["format_example"] == "action_names"
    assert "Pg" not in d["diagnostics"]["probes"]
    assert json.dumps(MP.plan_v3("D", 20261019, SHA)).count("pointing") == 0
