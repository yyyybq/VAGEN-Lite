"""Tests of the diagnostics (python -m pytest eval/diagnostics/tests -q); no GPU needed."""

from __future__ import annotations

import pytest

from eval.diagnostics import failure_breakdown as FB
from eval.diagnostics import qa_probe as QP


def turn(met=False, gates=("distance", "visible")):
    return {"met": met, "failed_gates": [] if met else list(gates)}


def test_classes():
    ok = {"success": True, "end_reason": "success_auto", "trajectory": [turn(), turn(True)]}
    assert FB.classify(ok)["class"] == "reached"
    drift = {"success": True, "end_reason": "max_primitive_steps", "trajectory": [turn(True), turn()]}
    assert FB.classify(drift)["class"] == "drifted"
    stop = {"success": False, "end_reason": "low_info", "trajectory": [turn()]}
    assert FB.classify(stop) == {"class": "stopped", "end_reason": "low_info"}
    close = {"success": False, "end_reason": "max_primitive_steps", "start": {"failed_gates": ["a", "b"]},
             "trajectory": [turn(), turn(gates=("distance",)), turn()]}
    assert FB.classify(close) == {"class": "budget_one_gate", "gate": "distance", "start_one_gate": False}
    far = {"success": False, "end_reason": "max_turns", "trajectory": [turn(), turn()]}
    assert FB.classify(far)["class"] == "budget_other"
    with pytest.raises(ValueError):
        FB.classify({"success": False, "end_reason": "max_turns"})


def test_breakdown_counts_per_family():
    rows = [dict(r, task_type="equidistance") for r in (
        {"success": True, "end_reason": "success_auto", "trajectory": [turn(True)]},
        {"success": False, "end_reason": "max_primitive_steps", "trajectory": [turn(gates=("equidistant",))]},
    )] + [{"index": 9, "error": "RuntimeError: x"}]
    res = FB.breakdown(rows)
    assert res["by_family"]["equidistance"] == {"reached": 1, "budget_one_gate": 1}
    assert res["by_family"]["all"] == {"reached": 1, "budget_one_gate": 1} and res["errors"] == 1
    assert res["one_gate"] == {"equidistant": 1}


def test_parse_answer():
    opts = ["about 1.5 meters", "about 2.0 meters", "about 2.5 meters", "about 3.0 meters"]
    assert QP.parse_answer("B", opts, False) == "B"
    assert QP.parse_answer("(C) about 2.5 meters", opts, False) == "C"
    assert QP.parse_answer("The answer is D.", opts, False) == "D"
    assert QP.parse_answer("about 2.0 meters", opts, False) == "B"
    assert QP.parse_answer("<action>move_forward|</action>", opts, False) is None
    assert QP.parse_answer("Yes.", [], True) == "Yes"
    assert QP.parse_answer("no, it is not", [], True) == "No"
    assert QP.parse_answer("<action>turn_left|</action>", [], True) is None


def test_parse_answer_lenient_and_compliance():
    dist = ["about 1.5 meters", "about 2.0 meters", "about 2.5 meters", "about 3.0 meters"]
    assert QP.parse_answer("I estimate 2.5 m.", dist, False) == "C"                       # the option's number
    assert QP.parse_answer("<think>far</think><answer>(B)</answer>", dist, False) == "B"   # after think, in answer
    assert QP.parse_answer("b", dist, False) == "B" and QP.parse_answer("**D**", dist, False) == "D"
    assert QP.parse_answer("Option c", dist, False) == "C"
    assert QP.parse_answer("between 2.0 and 2.5", dist, False) is None                      # two options named
    lr = ["to the left", "to the right"]
    assert QP.parse_answer("It is on the left side.", lr, False) == "A"
    occ = ["fully hidden behind the bed", "partly hidden behind the bed", "not hidden behind the bed"]
    assert QP.parse_answer("The table is partially covered.", occ, False) == "B"
    assert QP.parse_answer("not fully", occ, False) is None
    assert QP.compliant("B", dist, False) and QP.compliant("B.", dist, False)
    assert not QP.compliant("(B)", dist, False) and not QP.compliant("The answer is B", dist, False)
    assert QP.compliant("yes", [], True) and not QP.compliant("Yes, it is.", [], True)
    assert not QP.compliant("E", dist, False)


def test_chat_puts_images_where_the_markers_are(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    row = {"sample_id": "s", "images": [str(a), str(b)],
           "messages": [{"role": "user", "content": "First image: <image>\nSecond image: <image>\nWhat?"}]}
    content = QP.chat(row)[0]["content"]
    assert [c["type"] for c in content] == ["text", "image_url", "text", "image_url", "text"]
    row["images"] = [str(a)]
    with pytest.raises(ValueError):
        QP.chat(row)


def test_score_and_summary():
    rows = [{"sample_id": "1", "qtype": "goal_now", "qclass": "state", "family": "f", "options": [],
             "messages": [{}, {"content": "Yes"}]},
            {"sample_id": "2", "qtype": "closer_now", "qclass": "state", "family": "f", "options": ["x", "y"],
             "messages": [{}, {"content": "B"}]}]
    ans = QP.score(rows, ["Yes", "<action>move_left|</action>"])
    s = QP.summary(ans)
    formats = {"exact": 0.5, "letter_variant": 0.0, "reasoning": 0.0, "action_tag": 0.5, "other": 0.0}
    assert s["all"] == {"n": 2, "accuracy": 0.5, "unreadable": 0.5, "noncompliant": 0.5, "strict_accuracy": 0.5,
                        "chance": 0.5, "formats": formats}
    assert s["qtype"]["closer_now"]["unreadable"] == 1.0
    ans = QP.score(rows, ["Yes, it is.", "B"])
    s = QP.summary(ans)
    assert s["all"]["accuracy"] == 1.0 and s["all"]["noncompliant"] == 0.5 and s["all"]["strict_accuracy"] == 0.5
    assert s["all"]["formats"]["letter_variant"] == 0.5


def test_yes_no_split_and_inverse_confusion():
    rows = [{"sample_id": str(i), "qtype": "goal_now", "qclass": "state", "family": "f", "options": [],
             "messages": [{}, {"content": "Yes" if i < 2 else "No"}]} for i in range(4)]
    s = QP.summary(QP.score(rows, ["No", "Yes", "No", "No"]))
    assert s["qtype"]["goal_now"]["by_gold"] == {"Yes": 0.5, "No": 1.0} and s["qtype"]["goal_now"]["said_yes"] == 0.25
    opts = ["the camera turned left by 40 degrees", "the camera turned left by 60 degrees",
            "the camera turned right by 40 degrees", "the camera moved sideways to the left by 0.6 meters"]
    probe = {"x": {"options": opts, "answer": "A"}}
    for chosen, kind in (("A", "correct"), ("B", "count"), ("C", "opposite"), ("D", "other_kind"), (None, "unreadable")):
        t = QP.inverse_confusion([{"sample_id": "x", "qtype": "action_between", "parsed": chosen}], probe)
        assert t["turn_left"][kind] == 1


def test_format_classes():
    opts = ["to the left", "to the right"]
    assert QP.format_class("B", opts, False) == "exact"
    assert QP.format_class("(B)", opts, False) == "letter_variant"
    assert QP.format_class("(A) to the left", opts, False) == "letter_variant"
    assert QP.format_class("To determine the side, look at the sofa and the lamp; the answer is A", opts, False) == "reasoning"
    assert QP.format_class("<action>turn_left|</action>", opts, False) == "action_tag"
    assert QP.format_class("No, it is not.", [], True) == "letter_variant"
    assert QP.format_class("left", opts, False) == "other"


def test_history_permutation_differs_and_keeps_turns():
    from eval.diagnostics import history_ablation as HA

    assert HA.permutation(0, "k") == [] and HA.permutation(1, "k") == [0]
    for n in range(2, 8):
        for k in range(20):
            p = HA.permutation(n, f"t{k}")
            assert sorted(p) == list(range(n)) and p != list(range(n))

    class Base:
        def generate(self, system, user_text, image, earlier=(), **kw):
            return earlier

    out = HA.shuffled_policy(Base)().generate("s", "u", None, earlier=[("o1", 1, "a1"), ("o2", 2, "a2"), ("o3", 3, "a3")])
    assert sorted(out) == [("o1", 1, "a1"), ("o2", 2, "a2"), ("o3", 3, "a3")] and out != sorted(out)


def test_transfer_table(tmp_path):
    import json

    from eval.diagnostics import transfer_table as TT

    def write(name, succ):
        d = tmp_path / name
        d.mkdir()
        rows = [{"index": i, "task_type": f, "success": s} for i, (f, s) in enumerate(succ)]
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fams = ["projective_relations", "projective_relations", "equidistance", "fov_inclusion"]
    write("base", list(zip(fams, [False, False, False, True])))
    write("proj", list(zip(fams, [True, False, False, True])))
    t = TT.table({"baseline": {"eval": str(tmp_path), "name": "base"},
                  "runs": [{"eval": str(tmp_path), "name": "proj", "qa_groups": ["投影"]}]})
    assert t["投影"]["投影"] == (50.0, 1) and t["投影"]["距离类"] == (0.0, 1) and t["距离类"]["投影"] == (None, 0)
    assert "+50.0" in TT.markdown(t)


def test_format_check_rewrites(tmp_path):
    import json

    from eval.diagnostics import format_check as FC

    assert FC.as_action_only("<think>go left</think>\n<action>move_left|</action>") == "<action>move_left|</action>"
    assert FC.as_think_action("<action>turn_right|</action>").startswith("<think>")
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.json").write_text(json.dumps({"prompt_format": "free_think"}))
    turns = [{"answer": "<think>x</think><action>turn_left|turn_left|</action>", "actions": ["turn_left", "turn_left"]},
             {"answer": "<action>move_forward|</action>", "actions": ["move_forward"]},
             {"answer": "<think>cut off before the action", "actions": []}]
    (d / "results.jsonl").write_text(json.dumps({"index": 0, "trajectory": turns}) + "\n")
    r = FC.check_folder(d, FC.parser())
    assert (r["turns"], r["agree"], r["unreadable"], r["has_think"]) == (3, 3, 1, 2)


def test_parse_chain():
    name, pts = QP.parse_chain("R11-SFT=sft_formal_e{1..3}@154")
    assert name == "R11-SFT" and pts == [("sft_formal_e1", 154), ("sft_formal_e2", 308), ("sft_formal_e3", 462)]
    assert QP.parse_chain("RS0=sft_formal_e6:0,rs0_050:50")[1] == [("sft_formal_e6", 0), ("rs0_050", 50)]
