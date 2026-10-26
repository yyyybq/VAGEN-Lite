"""CPU tests of the X0 question rules (downstream/x/x0_qa.py): thresholds, answers, option letters, picking."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x"))
import x0_qa as X  # noqa: E402


def ent(name, cx=100.0, depth=1.0, forward=0.5, key=None):
    return {"key": key or name, "name": name, "pixels": 500, "cx": cx, "cy": 120.0, "depth": depth, "forward": forward}


def test_thresholds_leave_close_calls_out():
    assert X.answer_left_right(ent("mug", cx=60), ent("sink", cx=100)) == "left"
    assert X.answer_left_right(ent("mug", cx=140), ent("sink", cx=100)) == "right"
    assert X.answer_left_right(ent("mug", cx=110), ent("sink", cx=100)) is None  # under 30 px
    assert X.answer_near_far(ent("mug", depth=0.8), ent("sink", depth=1.2)) == "A"
    assert X.answer_near_far(ent("mug", depth=1.9), ent("sink", depth=1.6)) is None  # 0.3 m but under 1.2x
    assert X.answer_near_far(ent("mug", depth=1.0), ent("sink", depth=1.15)) is None  # under 0.2 m
    assert X.answer_front_back(ent("mug", forward=0.9), ent("sink", forward=0.5)) == "front"
    assert X.answer_front_back(ent("mug", forward=0.4), ent("sink", forward=0.5)) is None


def test_questions_carry_the_right_letter_whatever_the_option_order():
    for seed in range(20):
        rng = random.Random(seed)
        q = X.question("near_far", ent("mug", depth=0.6), ent("sink", depth=1.3), "robot0_eye_in_hand", rng)
        assert q["options"][" AB".index(q["answer"]) - 1] == "mug" and q["answer_text"] == "mug"
        assert "wrist camera" in q["prompt"] and q["prompt"].endswith("Answer with the letter of the correct option only.")
        q = X.question("front_back", ent("bowl", forward=0.2), ent("stove", forward=0.9), "robot0_agentview_left", rng)
        assert q["answer_text"] == "stove" and q["options"][" AB".index(q["answer"]) - 1] == "stove"  # farther ahead
    assert X.question("left_right", ent("mug", cx=100), ent("sink", cx=110), "robot0_agentview_left", random.Random(0)) is None


def test_names_and_seeds():
    assert X.display_name("wooden_spoon") == "wooden spoon"
    assert X.display_name("knife_block_main_group") == "knife block"
    assert X.display_name("paper_towel_right_group") == "paper towel"
    assert X.parse_seeds("1000-1002,1010") == [1000, 1001, 1002, 1010]


def test_pick_spreads_over_kinds_and_balances_answers():
    rng = random.Random(1)
    cands = []
    for kind in X.KINDS:
        for cam in X.CAMERAS:
            for i in range(10):
                cands.append({"kind": kind, "camera": cam, "answer": "AB"[i % 2], "task": f"T{i % 5}", "seed": 1000})
    picked = X.pick(cands, 18, rng)
    assert len(picked) == 18 and {q["kind"] for q in picked} == set(X.KINDS)
    for kind in X.KINDS:
        answers = [q["answer"] for q in picked if q["kind"] == kind]
        assert abs(answers.count("A") - answers.count("B")) <= 1
