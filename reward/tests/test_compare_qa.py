"""reward.compare_qa on synthetic answers (temp folders only)."""

import json

import pytest

from reward import compare_qa as cq


def test_parsers_match_the_downstream_rules():
    assert cq.parse_letter("A") == "A" and cq.parse_letter("(B).") == "B" and cq.parse_letter("A or B") is None
    assert cq.parse_letter("<answer>B</answer>") == "B" and cq.parse_letter("left") is None
    q2 = {"kind": "existence", "options": ["yes", "no"]}
    assert cq.parse_reply(q2, "no.") == "B" and cq.parse_reply(q2, "A") == "A"
    qc = {"kind": "counting", "options": []}
    assert cq.parse_reply(qc, "There are three.") == "3" and cq.parse_reply(qc, "02") == "2"


def test_sign_p():
    assert cq.sign_p(0, 0) == 1.0
    assert cq.sign_p(5, 5) == 1.0
    assert abs(cq.sign_p(10, 0) - 2 / 1024) < 1e-12


def _write(tmp_path, right):
    """right: {model: {set: [bool per question]}}; writes questions and answers like the chain does."""
    n = {s: len(next(iter(right.values()))[s]) for s in ("x0", "g0", "p1")}
    qx = [{"id": f"x{i}", "kind": "left_right", "options": ["left", "right"], "answer": "A"} for i in range(n["x0"])]
    qg = [{"id": f"g{i}", "kind": "existence", "options": ["yes", "no"], "answer": "B"} for i in range(n["g0"])]
    (tmp_path / "x0.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qx))
    (tmp_path / "g0.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qg))
    dq, pr = tmp_path / "diag/downstream_qa", tmp_path / "diag/probe_P1"
    dq.mkdir(parents=True)
    pr.mkdir(parents=True)
    for m, r in right.items():
        (dq / f"x0_{m}.jsonl").write_text("".join(json.dumps({"id": q["id"], "reply": "A" if ok else "B"}) + "\n"
                                                for q, ok in zip(qx, r["x0"])))
        (dq / f"g0_{m}.jsonl").write_text("".join(json.dumps({"id": q["id"], "reply": "no" if ok else "garbled"}) + "\n"
                                                for q, ok in zip(qg, r["g0"])))
        (pr / f"{m}.jsonl").write_text("".join(json.dumps({"sample_id": f"p{i}", "correct": ok, "compliant": i % 2 == 0})
                                               + "\n" for i, ok in enumerate(r["p1"])))


def test_pairs_and_verdicts(tmp_path):
    base = {"x0": [True] * 50 + [False] * 50, "g0": [True] * 5 + [False] * 5, "p1": [True, False] * 5}
    right = {m: {s: list(v) for s, v in base.items()} for m in cq.MODELS}
    right["rw_g_250"]["x0"] = [True] * 54 + [False] * 46  # +4 points: passes
    right["rw_o_200"]["x0"] = [True] * 52 + [False] * 48  # +2: fails
    right["rw_ok_250"]["x0"] = [True] * 55 + [False] * 45  # +5
    right["rw_s2ok_250"]["x0"] = [True] * 51 + [False] * 49  # +1: mean +3, both > 0 -> passes
    _write(tmp_path, right)
    res = cq.run(tmp_path / "x0.jsonl", tmp_path / "g0.jsonl", tmp_path / "diag")
    v = res["verdicts_x0"]
    assert v["V4"]["pass"] and abs(v["V4"]["diff"] - 4.0) < 1e-12
    assert not v["O"]["pass"]
    assert v["O-keep"]["pass"] and abs(v["O-keep"]["mean"] - 3.0) < 1e-12
    r = res["pairs"]["V4"]["x0"]
    assert (r["wins"], r["losses"], r["n"]) == (4, 0, 100)
    assert res["models"]["rw_s0_150"]["g0"]["unparsed"] == 5
    assert res["models"]["rw_s0_150"]["p1"]["strict"] == 5  # correct at even i, compliant at even i
    cq.write_md(res, tmp_path / "out.md")
    assert "X0 主判据" in (tmp_path / "out.md").read_text()


def test_chain_count_mismatch_stops(tmp_path):
    base = {"x0": [True, False], "g0": [True, False], "p1": [True, False]}
    _write(tmp_path, {m: base for m in cq.MODELS})
    (tmp_path / "diag/downstream_qa/summary.json").write_text(
        json.dumps({"models": {"rw_s0_150": {"x0": {"correct": 2}}}}))
    with pytest.raises(SystemExit):
        cq.run(tmp_path / "x0.jsonl", tmp_path / "g0.jsonl", tmp_path / "diag")
