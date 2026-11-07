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


def _easi_model(root, name, right):
    """Synthetic lmms-eval per-sample files for every EASI job; right: {job: [score per question]}."""
    for job, scores in right.items():
        d = root / job / name / "actor__huggingface"
        d.mkdir(parents=True)
        (d / "x_results.json").write_text("{}")
        rows = []
        for i, s in enumerate(scores):
            r = {"doc_id": str(i), "target": "A", "filtered_resps": "A"}
            if job == "vsi_bench":
                r["vsibench_overall"] = ({"id": str(i), "accuracy": s} if i % 2 == 0 else
                                         {"id": str(i), "MRA:.5:.95:.05": s})
            elif job == "mmsi_bench":
                r["average"] = {"question_id": i, "score": s}
            elif job == "mindcube_tiny":
                r["overall_accuracy"] = {"score": s}
                r["input"] = "[Question] Which? A. red chair B. blue sofa"
                r["target"] = "<answer>A. red chair</answer>"
                r["filtered_resps"] = "A. red chair" if s else "blue sofa"
            elif job == "viewspatial":
                r["overall_accuracy"] = str({"score": s})  # stored as text in some files
            elif job in ("site_image", "site_video"):
                r["accuracy"] = {"overall": s, "total": 1}
            elif job in ("blink", "embspatial"):
                r[cq._METRIC[job]] = {"id": f"{job}_{i}", "is_correct": bool(s)}
            elif job == "3dsrbench":
                r["vanilla_accuracy"] = {"qid": f"Q{i}", "score": s}
            rows.append(r)
        (d / f"20261107_samples_{job}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


JOBS9 = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site_image", "site_video", "blink", "3dsrbench",
         "embspatial")


def test_easi_pairing_and_verdict(tmp_path):
    base = {j: [1.0, 0.0, 1.0, 0.0] for j in JOBS9}
    base["vsi_bench"] = [1.0, 0.25, 0.0, 0.5]
    better = {j: [1.0, 1.0, 1.0, 0.0] for j in JOBS9}
    better["vsi_bench"] = [1.0, 0.5, 0.0, 0.5]  # one numeric question higher, the rest tied
    for m in cq.MODELS:
        if m == cq.C0:
            continue
        _easi_model(tmp_path, m, better if m in ("rw_ok_250", "rw_s2ok_250") else base)
    site = cq.bench_items(tmp_path, "site", "rw_s0_150")
    assert len(site) == 8 and "site_image:0" in site and "site_video:0" in site
    p = cq.score_pair(cq.bench_items(tmp_path, "vsi_bench", "rw_ok_250"), cq.bench_items(tmp_path, "vsi_bench", "rw_s0_150"))
    assert (p["wins"], p["losses"], p["ties"]) == (1, 0, 3)
    mc = cq.bench_items(tmp_path, "mindcube_tiny", "rw_ok_250")
    assert sum(mc.values()) == 3  # rejudged: the named option's text counts
    scores = {m: {"rejudged": 45.0, "official": 41.0} for m in cq.MODELS}
    scores["rw_ok_250"]["rejudged"] = 46.5
    scores["rw_s2ok_250"]["rejudged"] = 45.6  # +1.5 / +0.6: mean 1.05 >= 1, both > 0 -> holds
    sp = tmp_path / "scores.json"
    sp.write_text(json.dumps(scores))
    res = cq.run_easi(tmp_path, sp)
    v = res["verdict"]
    assert v["O-keep"]["holds"] and abs(v["O-keep"]["mean"] - 1.05) < 1e-9 and v["O-keep"]["each_ge_margin"] == [True, False]
    assert v["V4"]["holds"] is False and v["O"]["holds"] is False
    cq.write_easi_md(res, tmp_path / "e.md")
    assert "外推成立" in (tmp_path / "e.md").read_text()
    scores["rw_s2ok_250"]["rejudged"] = 44.9  # second seed below: does not hold
    sp.write_text(json.dumps(scores))
    assert cq.run_easi(tmp_path, sp)["verdict"]["O-keep"]["holds"] is False


def test_mindcube_rejudge_matches_easi8_score(tmp_path):
    from reward import easi8_score
    _easi_model(tmp_path, "m", {"mindcube_tiny": [1.0, 0.0, 1.0, 1.0, 0.0]})
    folder = tmp_path / "mindcube_tiny" / "m" / "actor__huggingface"
    items = cq.easi_items(tmp_path, "mindcube_tiny", "m")
    assert abs(100 * sum(items.values()) / len(items) - easi8_score.mindcube_lenient(folder)) < 1e-9


def test_chain_count_mismatch_stops(tmp_path):
    base = {"x0": [True, False], "g0": [True, False], "p1": [True, False]}
    _write(tmp_path, {m: base for m in cq.MODELS})
    (tmp_path / "diag/downstream_qa/summary.json").write_text(
        json.dumps({"models": {"rw_s0_150": {"x0": {"correct": 2}}}}))
    with pytest.raises(SystemExit):
        cq.run(tmp_path / "x0.jsonl", tmp_path / "g0.jsonl", tmp_path / "diag")
