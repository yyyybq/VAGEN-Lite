from eval_external.qa_transfer.parse_check import mcq_letter, number


def test_mcq_letter_forms():
    for reply in ("A", "A.", "(A)", " a ".upper(), "Answer: A", "<think>B or A?</think><answer>A</answer>",
                  "The answer is A."):
        assert mcq_letter(reply, 4) == "A", reply
    assert mcq_letter("E", 4) is None            # not an option
    assert mcq_letter("A or B", 4) is None       # two letters
    assert mcq_letter("I think so", 4) is None   # the pronoun I is outside A-D
    assert mcq_letter("", 2) is None
    assert mcq_letter("the chair is closer", 2, ["A. the table", "B. the chair"]) == "B"


def test_vsti_five_categories_pool_their_items():
    from eval_external.qa_transfer.collect import vsti_avg5

    by_type = {"camera_obj_abs_dist": [0.5], "camera_displacement": [0.1], "camera_movement_direction": [1.0, 0.0],
               "obj_obj_relative_pos_lr": [1, 1], "obj_obj_relative_pos_nf": [0], "obj_obj_relative_pos_ud": [1],
               "camera_obj_rel_dist_v1": [0], "camera_obj_rel_dist_v2": [1, 1, 1], "camera_obj_rel_dist_v3": []}
    avg, groups = vsti_avg5(by_type)
    assert groups["obj_obj_relative_pos"] == 75.0 and groups["camera_obj_rel_dist"] == 75.0
    assert abs(avg - (50 + 10 + 50 + 75 + 75) / 5) < 1e-9


def test_number_forms():
    assert number("3.5") == 3.5
    assert number("About 1,200 cm") == 1200.0
    assert number("<answer>two</answer>") == 2.0
    assert number("-0.5 meters") == -0.5
    assert number("I cannot tell") is None
