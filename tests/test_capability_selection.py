import copy
import random

import pytest

from data_gen.interiorgs_qa.capability.selection import (
    balance_report, estimate_capacity, select_questions,
)


def bank(n=40, choices=4, group="family", kind="kind", prefix="q"):
    return [{"qid": f"{prefix}{i}", "group": group, "kind": kind,
             "scene_id": f"scene{i % 4}", "parent_task_id": f"{prefix}_parent{i // 4}",
             "answer": chr(65 + i % choices), "answer_key": f"meaning{i % choices}",
             "options": [f"choice{j}" for j in range(choices)], "images": [f"{prefix}{i}.png"],
             "template_id": f"template{i % 3}"} for i in range(n)]


def test_only_one_semantic_answer_cannot_fill_quota():
    rows = bank(40)
    for row in rows:
        row["answer_key"] = "always_left"
    selected, report = select_questions(rows)
    assert selected == []
    assert report["families"]["family"]["shortfall"] == 20
    assert report["maximum_size_proven"]
    assert report["balance"]["kind"]["status"] == "empty"


def test_letter_shuffle_cannot_remove_semantic_bias():
    rows = bank(40)
    for i, row in enumerate(rows):
        row["answer_key"] = "left" if i < 36 else "right"
        row["answer"] = chr(65 + i % 4)
    # Four balanced letters but only two meanings: neither may exceed 40%.
    selected, report = select_questions(rows)
    assert selected == []
    assert report["kinds"]["kind"]["available_letter_counts"] == {"A": 10, "B": 10, "C": 10, "D": 10}


def test_balanced_semantics_do_not_excuse_biased_letters():
    rows = bank(40)
    for row in rows:
        row["answer"] = "A"
    selected, _ = select_questions(rows)
    assert selected == []


def test_balanced_true_subset_no_duplicates_or_mutation():
    rows = bank(40)
    rows.append(copy.deepcopy(rows[0]))
    before = copy.deepcopy(rows)
    selected, report = select_questions(rows)
    assert len(selected) == 20
    assert len({r["qid"] for r in selected}) == 20
    assert report["duplicate_rows_removed"] == 1
    assert rows == before
    assert all(r in rows for r in selected)
    assert all(r["valid"] for r in balance_report(selected).values())
    assert len(report["families"]["family"]["selected_by_scene"]) == 4


def test_parent_cap_is_never_relaxed_to_fill_family():
    rows = bank(40)
    for row in rows:
        row["parent_task_id"] = "only_parent"
    selected, report = select_questions(rows)
    assert len(selected) == 8
    assert report["families"]["family"]["shortfall"] == 12
    assert report["maximum_parent_count"] == 8
    assert report["quality_relaxed"] is False


def test_binary_cap_is_55_percent_in_both_spaces():
    rows = bank(30, choices=2)
    for i, row in enumerate(rows):
        row["answer_key"] = "near" if i < 20 else "far"
        row["answer"] = "A" if i < 20 else "B"
    selected, report = select_questions(rows, family_cap=30)
    assert len(selected) == 22
    check = report["balance"]["kind"]
    assert check["maximum_allowed_share"] == 0.55
    assert check["max_semantic_share"] <= 0.55
    assert check["max_letter_share"] <= 0.55


def test_expected_missing_family_is_reported():
    selected, report = select_questions(bank(12), families=["family", "absent"])
    assert len(selected) == 12
    assert report["families"]["family"]["shortfall"] == 8
    assert report["families"]["absent"] == {
        "available": 0, "selected": 0, "target": 20, "shortfall": 20,
        "reason": "no_candidates", "available_by_scene": {}, "selected_by_scene": {},
        "selected_parent_count": 0}


def test_global_total_and_family_limits_hold():
    rows = bank(40, group="f1", prefix="a") + bank(40, group="f2", prefix="b")
    selected, report = select_questions(rows, total_cap=12)
    assert len(selected) == 12
    assert report["families"]["f1"]["selected"] == 6
    assert report["families"]["f2"]["selected"] == 6
    assert report["balance_valid"]


def test_same_kind_cannot_hide_bias_by_adding_another_kind():
    rows = bank(40, kind="biased", prefix="b") + bank(40, kind="balanced", prefix="a")
    for row in rows:
        if row["kind"] == "biased":
            row["answer_key"] = "same"
    selected, report = select_questions(rows)
    assert len(selected) == 20
    assert all(r["kind"] == "balanced" for r in selected)
    assert report["balance"]["biased"]["selected"] == 0


def test_same_seed_is_independent_of_input_order():
    rows = bank(32)
    selected, _ = select_questions(rows, seed=17)
    random.Random(4).shuffle(rows)
    again, _ = select_questions(rows, seed=17)
    assert [r["qid"] for r in selected] == [r["qid"] for r in again]


def test_conflicting_question_identity_is_an_error():
    rows = bank(4)
    other = copy.deepcopy(rows[0]); other["answer_key"] = "changed"
    with pytest.raises(ValueError, match="same question ID"):
        select_questions(rows + [other])


def test_empty_bank_is_not_claimed_to_supply_questions():
    selected, report = select_questions([], families=["f1", "f2"])
    assert selected == []
    assert report["total_shortfall"] == 40
    assert report["families"]["f1"]["reason"] == "no_candidates"


def test_capacity_is_observed_scene_rate_not_verified_extrapolation():
    rows = bank(8)
    rows[0]["images"].append("another.png")
    selected, _ = select_questions(rows)
    result = estimate_capacity(rows, selected, scene_ids=["scene0", "scene1", "scene2", "scene3"],
                               families=["family", "missing"])
    assert result["act_image_inputs_per_epoch"] == 25243
    assert result["families"]["family"]["mean_per_scene"] == 2
    assert result["families"]["family"]["min_per_scene"] == 2
    assert result["families"]["missing"]["max_per_scene"] == 0
    assert result["raw_bank_optimistic_scaling"]["image_input_occurrences"] == 9
    assert not result["extrapolation_verified"]


def test_capacity_rejects_a_selection_from_another_bank():
    with pytest.raises(ValueError, match="not a subset"):
        estimate_capacity(bank(8), bank(4, prefix="other"))
