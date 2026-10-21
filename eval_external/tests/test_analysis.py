from eval_external.diagnostics.compare_ownenv import mcnemar_exact, paired
from eval_external.qa_transfer.scale_bias import first_number, ratio_stats


def test_first_number():
    assert first_number("1 meter") == 1.0 and first_number("0.15") == 0.15 and first_number(".5 m") == 0.5
    assert first_number("about two meters") is None and first_number(None) is None


def test_ratio_stats_counts_and_shares():
    s = ratio_stats([(0.5, 2.0), (1.0, 2.0), (2.0, 2.0), (8.0, 2.0), (None, 2.0), (0.0, 2.0)])
    assert s["n"] == 6 and s["n_no_number_or_zero"] == 2
    assert s["median"] == 0.75 and s["share_below_half"] == 0.25 and s["share_above_double"] == 0.25
    assert ratio_stats([(None, 1.0)]) == {"n": 1, "n_no_number_or_zero": 1}


def test_mindcube_text_rule():
    from eval_external.qa_transfer.mindcube_text_rule import rule_letter

    c = ["A. Diagonally forward and left", "B. Directly right", "C. Yes", "D. Yes"]
    assert rule_letter("<answer>Directly right</answer>", c) == "B"    # text of exactly one option
    assert rule_letter("<answer>Directly right.</answer>", c) == "B"   # one trailing period dropped
    assert rule_letter("<answer>B. Directly right</answer>", c) is None  # a letter answer: official result stands
    assert rule_letter("<answer>B</answer>", c) is None and rule_letter("(B)", c) is None
    assert rule_letter("<answer>Yes</answer>", c) is None              # matches two options
    assert rule_letter("<answer>directly right</answer>", c) is None   # case-sensitive
    assert rule_letter("Directly right", c) == "B"                     # no tags: the whole reply


def test_mcnemar_and_paired():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 6) - 2 / 64) < 1e-12        # all six discordant pairs one way
    assert abs(mcnemar_exact(3, 3) - 1.0) < 1e-12
    ref = {"a": {"success": True}, "b": {"success": False}, "c": {"success": False}, "d": {"success": True}}
    run = {"a": {"success": False}, "b": {"success": True}, "c": {"success": True}, "d": {"success": True}}
    p = paired(ref, run)
    assert (p["success"], p["ref_success"], p["only_run"], p["only_ref"], p["diff_pts"]) == (3, 2, 2, 1, 25.0)


def test_paired_headline_rules():
    import pytest

    from eval_external.qa_transfer.paired import headline

    sums = {"object_rel_direction_easy": 1.0, "object_rel_direction_medium": 0.0, "object_rel_direction_hard": 2.0,
            "route_planning": 3.0}
    ns = {"object_rel_direction_easy": 2, "object_rel_direction_medium": 2, "object_rel_direction_hard": 2,
          "route_planning": 4}
    assert headline("items", sums, ns) == pytest.approx(0.6)       # 6 / 10
    assert headline("types", sums, ns) == pytest.approx(0.5625)    # mean(0.5, 0, 1, 0.75)
    assert headline("vsi8", sums, ns) == pytest.approx(0.625)      # mean(mean(0.5, 0, 1), 0.75)
    v = headline("vsti5", {"camera_displacement": 1.0, "obj_obj_relative_pos_lr": 1.0, "obj_obj_relative_pos_nf": 0.0},
                 {"camera_displacement": 2, "obj_obj_relative_pos_lr": 1, "obj_obj_relative_pos_nf": 1})
    assert v == pytest.approx(0.5)                                  # mean(1/2, (1 + 0) / 2)


def test_vsti_direction_discrimination_and_baselines():
    from eval_external.qa_transfer.vsti_direction import baselines, discrimination

    rs = [{"truth": "Backward", "options": ["Forward", "Backward"], "pred": "Backward"},
          {"truth": "Backward", "options": ["Forward", "Backward", "Left"], "pred": "Forward"},
          {"truth": "Forward", "options": ["Forward", "Backward"], "pred": "Backward"},
          {"truth": "Forward", "options": ["Forward", "Backward"], "pred": "Forward"},
          {"truth": "Left", "options": ["Left", "Right"], "pred": "Right"}]
    d = discrimination(rs, ("Forward", "Backward"))
    assert (d["when_truth_Backward"], d["when_truth_Forward"], d["diff"]) == (50.0, 50.0, 0.0)  # answer ignores truth
    assert d["ci95"][0] < 0 < d["ci95"][1]
    b = baselines(rs)
    assert b["random"] == 46.7                                 # mean of 1/2, 1/3, 1/2, 1/2, 1/2
    assert b["best_fixed_order"]["accuracy"] == 60.0           # e.g. Forward > Left: items 3, 4, 5
    assert b["n_options"] == {2: 4, 3: 1}
