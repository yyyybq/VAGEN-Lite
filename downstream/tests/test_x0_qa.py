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
    assert X.display_name("paper_towel_dining_area_group") == "paper towel"  # seen in the first samples
    assert X.display_name("utensil_set_front_group") == "utensil set"
    assert X.display_name("outlet_room") == "outlet" and X.display_name("range hood") == "range hood"
    assert X.parse_seeds("1000-1002,1010") == [1000, 1001, 1002, 1010]


def cand(kind, task, cam, a_kind, b_kind, answer="A", i=0):
    a, b = ent("mug", key=f"a{i}"), ent("sink", key=f"b{i}")
    a["kind"], b["kind"] = a_kind, b_kind
    return {"kind": kind, "camera": cam, "task": task, "seed": 1000, "question": "Which?", "options": ["mug", "sink"],
            "answer": answer, "answer_text": "mug" if answer == "A" else "sink", "a": a, "b": b,
            "prompt": X.make_prompt(cam, "Which?", ["mug", "sink"])}


def test_select_puts_object_questions_first_and_alternates_letters():
    cands = []
    for i in range(6):
        cands.append(cand("near_far", f"T{i}", "robot0_agentview_left", "object", "fixture", i=i))
    for i in range(10):
        cands.append(cand("near_far", f"U{i}", "robot0_agentview_right", "fixture", "fixture", i=100 + i))
    chosen = X.select(cands, 8, random.Random(0))
    assert len(chosen) == 8 and sum(q["object_question"] for q in chosen) == 6  # every object question, then fixtures
    assert [q["answer"] for q in chosen] == list("ABABABAB")
    for q in chosen:  # the option under the letter is still the right one, and the prompt shows the new order
        assert q["options"]["AB".index(q["answer"])] == q["answer_text"]
        assert f"A. {q['options'][0]}\nB. {q['options'][1]}" in q["prompt"]
    assert len({q["id"] for q in chosen}) == 8


def test_parse_letter_and_scene_parts():
    assert X.parse_letter("A") == "A" and X.parse_letter("B.") == "B" and X.parse_letter("(B) behind it") == "B"
    assert X.parse_letter("Answer: A") == "A" and X.parse_letter("The answer is B") == "B"
    assert X.parse_letter("A or B") is None and X.parse_letter("<action>move_forward</action>") is None
    pairs = X.scene_pairs(["T1", "T2"], [1, 2, 3], "2/4")
    assert pairs == [("T1", 2), ("T2", 3)]  # indices 1 and 5 of the six pairs
    assert sorted(sum((X.scene_pairs(["T1", "T2"], [1, 2, 3], f"{i}/4") for i in range(1, 5)), [])) == \
        sorted(X.scene_pairs(["T1", "T2"], [1, 2, 3]))


def test_analyze_accuracy_interval_unparsed_and_mcnemar():
    qs = [{"id": f"q{i}", "kind": X.KINDS[i % 3], "answer": "A", "object_question": i < 6} for i in range(30)]
    answers = {"B": {f"q{i}": ("A" if i < 15 else "B") for i in range(30)},
               "RS": {f"q{i}": ("A" if i < 27 else None) for i in range(30)}}
    res = X.analyze(qs, answers)
    assert res["B"]["all"]["accuracy"] == 0.5 and res["RS"]["all"]["correct"] == 27
    assert res["RS"]["all"]["unparsed"] == 3 and abs(res["RS"]["all"]["unparsed_rate"] - 0.1) < 1e-9
    lo, hi = res["RS"]["all"]["ci"]
    assert 0.73 < lo < 0.75 and 0.96 < hi < 0.98  # Wilson 95% for 27 / 30
    vs = res["RS"]["all"]["vs_reference"]
    assert vs["only_model"] == 12 and vs["only_reference"] == 0 and vs["p"] < 0.001
    assert X.mcnemar_exact(0, 0) == 1.0 and abs(X.mcnemar_exact(1, 1) - 1.0) < 1e-12
    assert res["RS"]["object"]["n"] == 6 and res["RS"]["fixture_only"]["n"] == 24


def test_the_job_script_parses():
    import subprocess

    assert subprocess.run(["bash", "-n", str(ROOT / "x/run_x0.sh")], capture_output=True).returncode == 0


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
