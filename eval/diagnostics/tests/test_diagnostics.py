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
    assert s["all"] == {"n": 2, "accuracy": 0.5, "unreadable": 0.5}
    assert s["qtype"]["closer_now"]["unreadable"] == 1.0


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


def test_parse_chain():
    name, pts = QP.parse_chain("R11-SFT=sft_formal_e{1..3}@154")
    assert name == "R11-SFT" and pts == [("sft_formal_e1", 154), ("sft_formal_e2", 308), ("sft_formal_e3", 462)]
    assert QP.parse_chain("RS0=sft_formal_e6:0,rs0_050:50")[1] == [("sft_formal_e6", 0), ("rs0_050", 50)]
