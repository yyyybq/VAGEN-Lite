"""configs/matrix_v3/Go.yaml (G-oracle / G', the user's instruction of 2026-11-07, item 6c): its plan carries the oracle
marker in RL and the evaluation (oracle mode) and the self-mode variant; no other group's plan changes."""
import yaml

from data_gen.sft_mixture import matrix_plan as MP

COMMIT = "0" * 40


def test_go_plan_oracle_marker_and_self_mode():
    p = MP.plan_v3_cell_rl("Go", 20261019, COMMIT, "/x/checkpoint-1", dense="registered")
    assert p["rl"]["oracle_marker"] is True and p["rl"]["format_example"] == "action_names"
    assert "pointing_reward" not in p["rl"] and "entropy_key" not in p["rl"]["guards"]
    assert p["eval"]["options"] == ["--oracle-marker"]
    assert p["eval"]["variants"] == [{"suffix": "_self", "options": ["--format-example", "action_names", "--self-marker"],
                                     "models": ["sft_pick", "rl"]}]
    assert p["diagnostics"]["wrong_target"] is True and "Pg" in p["diagnostics"]["probes"]
    assert "keep_std_exclude" not in p["rl_runs"][next(k for k in p["rl_runs"] if k.endswith("d"))]["rl"]
    g = yaml.safe_load((MP.ROOT / "configs" / "matrix_v3" / "Go.yaml").read_text(encoding="utf-8"))
    assert g["status"] == "prep" and g["data"]["train_rows"] == {"P0": 10144, "P1": 9075, "P2m": 9075, "Pg": 9075}


def test_other_groups_unchanged():
    for group in ("G", "Cr", "D_noP0"):
        p = MP.plan_v3_cell_rl(group, 20261019, COMMIT, "/x/checkpoint-1", dense="registered")
        assert "oracle_marker" not in p["rl"] and p["eval"]["options"] == [] and "variants" not in p["eval"], group
