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


def test_chat_blind_keeps_only_the_text(tmp_path):
    row = {"sample_id": "s", "images": [str(tmp_path / "missing.png")] * 2,  # never read
           "messages": [{"role": "user", "content": "First image: <image>\nSecond image: <image>\nWhat?"}]}
    content = QP.chat(row, blind=True)[0]["content"]
    assert [c["type"] for c in content] == ["text"]
    text = content[0]["text"]
    assert text.startswith(QP.BLIND_NOTE) and "<image>" not in text and text.count(QP.BLIND_MARK) == 2
    assert text.endswith("What?")


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


def test_free_form_rows_are_kept_as_text():
    """A pointing question (no options, an answer that is neither Yes nor No) is kept unjudged instead of breaking
    the letter reader; questions with options or Yes / No are read as before."""
    msgs = lambda gold: [{"role": "user", "content": "<image>q"}, {"role": "assistant", "content": gold}]
    rows = [{"sample_id": "p", "qtype": "point_now", "qclass": "pointing", "family": "f", "options": [],
             "messages": msgs("(190, 179)")},
            {"sample_id": "y", "qtype": "t", "qclass": "state", "family": "f", "options": [], "messages": msgs("No")},
            {"sample_id": "l", "qtype": "t", "qclass": "goal", "family": "f", "options": ["left", "right"],
             "messages": msgs("B")}]
    assert [QP.free_form(r) for r in rows] == [True, False, False]
    a = QP.score(rows, ["(120, 88)", "No.", "B"])
    assert (a[0]["output"], a[0]["readable"], a[0]["correct"], a[0]["n_options"]) == ("(120, 88)", False, False, None)
    assert a[1]["correct"] and a[1]["compliant"] and a[2]["correct"] and a[2]["compliant"]
    summ = QP.summary(a)
    assert summ["qclass"]["pointing"]["unreadable"] == 1.0 and summ["all"]["chance"] is None


def test_pointing_questions_are_judged():
    """Group G of matrix v3: a point inside the true box, a box with IoU >= 0.5, in pixels or 0-1000 coordinates."""
    box = [100.0, 80.0, 140.0, 100.0]
    row = lambda judge, coords="pixel": {"judge": judge, "box": box, "coords": coords, "render_size": [256, 256]}  # noqa: E731
    assert QP.judge_pointing(row("point"), "(120,90)")["correct"]
    assert QP.judge_pointing(row("point"), "(120,90)")["compliant"]
    assert not QP.judge_pointing(row("point"), "(10, 10)")["correct"]
    loose = QP.judge_pointing(row("point"), "The lamp is at (120, 90).")
    assert loose["correct"] and loose["readable"] and not loose["compliant"]
    assert QP.judge_pointing(row("point", "norm1000"), "(469,352)")["correct"]
    assert not QP.judge_pointing(row("point", "norm1000"), "(120,90)")["correct"]
    assert QP.judge_pointing(row("box"), "(100,80,140,100)")["correct"]
    assert QP.judge_pointing(row("box"), "(105,82,140,100)")["correct"]  # IoU 0.79
    assert not QP.judge_pointing(row("box"), "(100,80,120,90)")["correct"]  # IoU 0.25
    assert QP.judge_pointing(row("box", "norm1000"), "(391,313,547,391)")["correct"]
    assert not QP.judge_pointing(row("box"), "(120,90)")["readable"]
    rows = [{"sample_id": "p", "qtype": "point_now", "qclass": "pointing", "family": "f", "options": [],
             "messages": [{"role": "user", "content": "<image>q"}, {"role": "assistant", "content": "(120,90)"}],
             **row("point")}]
    a = QP.score(rows, ["(121, 91)"])[0]
    assert a["correct"] and a["readable"] and a["compliant"] and a["parsed"] == "121,91"


def test_pointing_answers_in_qwen_forms():
    """JSON grounding output, bracketed groups and loose numbers are read as the coordinates they give."""
    box = [100.0, 80.0, 140.0, 100.0]
    row = lambda judge: {"judge": judge, "box": box, "coords": "norm1000", "render_size": [256, 256]}  # noqa: E731
    point = row("point")
    j = QP.judge_pointing(point, '{"point_2d": [469, 352], "label": "lamp"}')
    assert (j["correct"], j["read_as"]) == (True, "point_2d")
    j = QP.judge_pointing(point, '[{"bbox_2d": [391, 313, 547, 391], "label": "lamp"}]')
    assert (j["correct"], j["read_as"]) == (True, "bbox_2d_centre")
    assert QP.judge_pointing(point, "1. (469, 352)")["read_as"] == "group" and QP.judge_pointing(point, "1. (469, 352)")["correct"]
    assert QP.judge_pointing(point, "On a 0-1000 scale: (469, 352).")["correct"]
    assert QP.judge_pointing(point, "x=469, y=352")["correct"]
    assert not QP.judge_pointing(dict(point, coords="pixel"), "(-1, 0)")["correct"]  # off the image
    assert QP.judge_pointing(point, "from 469-352")["parsed"] == "469,352"  # a range is no minus
    j = QP.judge_pointing(point, "(391,313),(547,391)")  # a box's two corners to a point question: their centre
    assert (j["correct"], j["read_as"], j["parsed"]) == (True, "corners_centre", "469,352")
    assert not QP.judge_pointing(point, "(600,700),(700,800)")["correct"]
    b = row("box")
    j = QP.judge_pointing(b, '[{"bbox_2d": [391, 313, 547, 391], "label": "lamp"}]')
    assert (j["correct"], j["read_as"]) == (True, "bbox_2d")
    assert QP.judge_pointing(b, "x1=391, y1=313, x2=547, y2=391")["correct"]
    assert QP.judge_pointing(b, "(391,313),(547,391)")["correct"]
    assert QP.judge_pointing(b, "[391, 313, 547, 391]")["correct"]
    assert not QP.judge_pointing(b, '{"point_2d": [469, 352]}')["readable"]
    pix = dict(point, coords="pixel")
    assert QP.judge_pointing(pix, "(120, 90)")["correct"] and QP.judge_pointing(pix, "(120, 90)")["compliant"]


def test_rescore_follows_the_probe_relabelled_and_dropped(tmp_path):
    import argparse
    import json

    rows = [{"sample_id": s, "qtype": "occluded_now", "qclass": "state", "family": "f", "options": ["a", "b", "c"],
             "messages": [{}, {"content": gold}]} for s, gold in (("1", "A"), ("2", "B"), ("3", "C"))]
    out = tmp_path / "res"
    out.mkdir()
    answers = QP.score(rows, ["A", "A", "C"])
    (out / "m.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
    rows[0]["messages"][1]["content"] = "C"                      # relabelled
    probe = tmp_path / "probe.jsonl"
    probe.write_text("".join(json.dumps(r) + "\n" for r in rows[:1] + rows[2:]))  # question 2 dropped
    QP.cmd_rescore(argparse.Namespace(out=str(out), probe=str(probe)))
    new = [json.loads(l) for l in (out / "m.jsonl").read_text().splitlines()]
    assert [a["sample_id"] for a in new] == ["1", "3"] and [a["correct"] for a in new] == [False, True]
    assert json.loads((out / "m.json").read_text())["all"]["n"] == 2


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


# the probe comparison of PREREG_D_vs_C.md section 3.3 (``compare``)
OCC_SMALL = [("occluded_now", "full")] * 3 + [("occluded_now", "partial")] * 2 + [("occluded_after", "full")] \
    + [("occluded_after", "partial")] + [("occluded_after", "none")] * 5  # full / partial / none = 4 / 3 / 5


def small_probe(tmp_path):
    """goal_now and goal_after 10 questions each (Yes on the first 5), 12 occlusion questions, one other type."""
    import json

    rows = [{"sample_id": f"{q}{i}", "qtype": q, "answer": "Yes" if i < 5 else "No",
             "answer_value": "yes" if i < 5 else "no"} for q in ("goal_now", "goal_after") for i in range(10)]
    rows += [{"sample_id": f"o{i}", "qtype": q, "answer": "A", "answer_value": v} for i, (q, v) in enumerate(OCC_SMALL)]
    rows.append({"sample_id": "x", "qtype": "closer_now", "answer": "B", "answer_value": None})  # not compared
    (tmp_path / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path / "probe.jsonl", rows


def write_result(tmp_path, name, rows, right):
    import json

    (tmp_path / f"{name}.jsonl").write_text("".join(
        json.dumps({"sample_id": r["sample_id"], "qtype": r["qtype"], "gold": r["answer"], "correct": bool(right(r))})
        + "\n" for r in rows))


def test_mcnemar_exact():
    assert QP.mcnemar(0, 0) == 1.0 and QP.mcnemar(1, 0) == 1.0 and QP.mcnemar(7, 7) == 1.0
    assert QP.mcnemar(5, 0) == QP.mcnemar(0, 5) == 0.0625                        # 2 / 2^5
    assert QP.mcnemar(10, 2) == QP.mcnemar(2, 10) == 2 * (1 + 12 + 66) / 2 ** 12  # 0.0386
    assert QP.mcnemar(9, 6) == 2 * 9949 / 2 ** 15                                 # 0.607: C0 P1-S0 250 - SFT e9, goal_now
    assert 0 < QP.mcnemar(150, 100) < 0.01                                        # 250 discordant: no overflow


def test_verdict_thresholds_and_sign():
    V = QP.verdict
    assert V(0.12, [0.10, 0.14], 0.10, p=0.01) == {"verdict": "difference", "failed": [], "same_sign": True}
    assert V(0.12, [0.10, 0.14], 0.10, p=0.05)["failed"] == ["p"]                # p must be below 0.05
    assert V(-0.10, [-0.12, -0.08], 0.10, p=0.01)["verdict"] == "difference"     # on the threshold, both negative
    assert V((0.13 + 0.07) / 2, [0.13, 0.07], 0.10, p=0.01)["verdict"] == "difference"  # rounding at the threshold
    assert V(0.099, [0.099, 0.099], 0.10, p=0.01)["failed"] == ["size"]
    assert V(0.12, [0.25, -0.01], 0.10, p=0.01)["failed"] == ["sign"]
    assert V(0.10, [0.20, 0.0], 0.10, p=0.01)["failed"] == ["sign"]              # a zero difference has no sign
    assert V(0.0, [0.1, -0.1], 0.10, p=1.0)["failed"] == ["p", "size", "sign"]
    assert V(0.25, [0.30, 0.20], 0.20)["verdict"] == "difference"                # occlusion: no p decides
    assert V(0.19, [0.30, 0.08], 0.20)["failed"] == ["size"]
    assert V(-0.20, [-0.21, -0.19], 0.20)["verdict"] == "difference"
    assert V(0.15, [0.15], 0.10, p=0.01)["verdict"] == "difference" and V(0.0, [0.0], 0.10)["failed"] == ["size", "sign"]


def test_bootstrap_keeps_the_answer_counts():
    import collections

    values = [v for _, v in OCC_SMALL]
    strata = [[i for i, x in enumerate(values) if x == v] for v in ("full", "partial", "none")]
    draws = QP.stratified_draws([len(s) for s in strata], 300, QP.BOOT_SEED)
    assert [d.shape for d in draws] == [(300, 4), (300, 3), (300, 5)]
    for rep in range(300):
        picked = [st[j] for st, idx in zip(strata, draws) for j in idx[rep]]
        assert collections.Counter(values[i] for i in picked) == {"full": 4, "partial": 3, "none": 5}
    assert any(len(set(idx[rep])) < idx.shape[1] for idx in draws for rep in range(300))  # drawn with replacement
    again = QP.stratified_draws([4, 3, 5], 300, QP.BOOT_SEED)
    assert all((x == y).all() for x, y in zip(draws, again))                             # the seed fixes the draw


def test_compare_verdicts_end_to_end(tmp_path):
    import json

    probe, rows = small_probe(tmp_path)

    def goal(r, q, yes=None):
        return r["qtype"] == q and (yes is None or (r["answer_value"] == "yes") == yes)

    # goal_now: A right everywhere, B only when the answer is No (+50 for both RLs); goal_after: +100 for S0,
    # -100 for D; occlusion: A right on full and partial, B only on none (balanced 2/3 against 1/3)
    write_result(tmp_path, "as0", rows, lambda r: goal(r, "goal_now") or goal(r, "goal_after")
                 or r["answer_value"] in ("full", "partial"))
    write_result(tmp_path, "ad", rows, lambda r: goal(r, "goal_now") or r["answer_value"] in ("full", "partial"))
    write_result(tmp_path, "bs0", rows, lambda r: goal(r, "goal_now", False) or r["answer_value"] == "none")
    write_result(tmp_path, "bd", rows, lambda r: goal(r, "goal_now", False) or goal(r, "goal_after")
                 or r["answer_value"] == "none")
    QP.main(["compare", "--probe", str(probe), "--out", str(tmp_path), "--a", "D=as0,ad", "--b", "C=bs0,bd"])
    res = json.loads((tmp_path / "compare_D_vs_C.json").read_text(encoding="utf-8"))
    assert [(p["pair"], p["a"], p["b"]) for p in res["pairs"]] == [("S0", "as0", "bs0"), ("D", "ad", "bd")]
    g = res["goal_now"]
    assert g["n"] == 10 and g["n_by_answer"] == {"yes": 5, "no": 5}
    assert g["per_pair"][0]["by_answer_a"] == {"yes": 1.0, "no": 1.0}
    assert g["per_pair"][0]["by_answer_b"] == {"yes": 0.0, "no": 1.0}
    assert [x["diff"] for x in g["per_pair"]] == [0.5, 0.5] and [x["discordance"] for x in g["per_pair"]] == [0.5, 0.5]
    assert (g["pooled"]["n"], g["pooled"]["only_a"], g["pooled"]["only_b"]) == (20, 10, 0)
    assert g["pooled"]["p"] == 2 / 2 ** 10 and g["pooled"]["verdict"] == "difference"
    g = res["goal_after"]
    assert [x["diff"] for x in g["per_pair"]] == [1.0, -1.0] and g["pooled"]["diff"] == 0.0
    assert g["pooled"]["p"] == 1.0 and g["pooled"]["failed"] == ["p", "size", "sign"]
    o = res["occlusion"]
    assert o["n"] == 12 and o["n_by_answer"] == {"full": 4, "partial": 3, "none": 5}
    x = o["per_pair"][0]
    assert x["by_answer_a"] == {"full": 1.0, "partial": 1.0, "none": 0.0} and x["bal_a"] == pytest.approx(2 / 3)
    assert x["diff"] == pytest.approx(1 / 3) and x["discordance"] == 1.0
    # every replicate keeps each answer's questions, so the per-answer accuracies (all 0 or all 1) never move
    assert o["pooled"]["ci"] == pytest.approx([1 / 3, 1 / 3]) and o["pooled"]["p"] == 0.0
    assert o["pooled"]["verdict"] == "difference" and o["pooled"]["diff"] == pytest.approx(1 / 3)
    md = (tmp_path / "compare_D_vs_C.md").read_text(encoding="utf-8")
    assert "# 探针比较：D − C" in md and "| `goal_now` | 合起来 | 20 |" in md and md.count("**有差异**") == 4


def test_compare_one_draw_for_every_pair(tmp_path):
    probe, rows = small_probe(tmp_path)
    write_result(tmp_path, "x", rows, lambda r: r["sample_id"] in ("o0", "o3", "o5", "o6", "o7", "goal_now1"))
    write_result(tmp_path, "y", rows, lambda r: r["sample_id"] in ("o1", "o4", "o8", "o9", "goal_now7"))
    res = QP.compare(probe, tmp_path, ("A", ["x", "y"]), ("B", ["y", "x"]), "swap", reps=500)
    o = res["occlusion"]
    s0, d = o["per_pair"]
    assert s0["diff"] == pytest.approx(-d["diff"]) and s0["diff"] != 0
    assert s0["ci"] == pytest.approx([-d["ci"][1], -d["ci"][0]]) and s0["p"] == d["p"]
    # with the same draw for both pairs the pooled replicates are all exactly 0
    assert o["pooled"]["ci"] == [0.0, 0.0] and o["pooled"]["p"] == 1.0 and o["pooled"]["diff"] == pytest.approx(0)
    assert o["pooled"]["failed"] == ["size", "sign"]


def test_compare_single_results_and_errors(tmp_path):
    import json

    probe, rows = small_probe(tmp_path)
    write_result(tmp_path, "x", rows, lambda r: r["qtype"] == "goal_now" or r["sample_id"] in ("goal_after0", "goal_after1"))
    write_result(tmp_path, "y", rows, lambda r: False)
    QP.main(["compare", "--probe", str(probe), "--out", str(tmp_path), "--a", "x", "--b", "y", "--reps", "100"])
    res = json.loads((tmp_path / "compare_x_vs_y.json").read_text(encoding="utf-8"))
    assert [p["pair"] for p in res["pairs"]] == ["single"] and res["rules"]["reps"] == 100
    assert res["goal_now"]["pooled"]["diff"] == 1.0 and res["goal_now"]["pooled"]["p"] == 2 / 2 ** 10
    assert res["goal_now"]["pooled"]["verdict"] == "difference"
    assert res["goal_after"]["pooled"]["diff"] == 0.2 and res["goal_after"]["pooled"]["failed"] == ["p"]  # p 0.5
    assert res["occlusion"]["pooled"]["failed"] == ["size", "sign"]                       # 0 has no sign
    assert QP.make_pairs(["a0", "a1"], ["b"]) == [("S0", "a0", "b"), ("D", "a1", "b")]
    assert QP.parse_group("D=a,b") == ("D", [("", ["a", "b"])]) and QP.parse_group("x") == ("x", [("", ["x"])])
    with pytest.raises(ValueError):
        QP.parse_group("a,b,c")
    (tmp_path / "sub").mkdir()                                                            # results in another folder
    write_result(tmp_path / "sub", "z", rows, lambda r: r["qtype"] == "goal_now")
    QP.main(["compare", "--probe", str(probe), "--out", str(tmp_path), "--a", "sub/z", "--b", "y", "--reps", "10"])
    res = json.loads((tmp_path / "compare_z_vs_y.json").read_text(encoding="utf-8"))
    assert res["pairs"][0]["a"] == "sub/z" and res["goal_now"]["pooled"]["diff"] == 1.0
    write_result(tmp_path, "short", rows[:-4], lambda r: True)                            # o9-o11 missing
    with pytest.raises(ValueError, match="missing"):
        QP.compare(probe, tmp_path, ("A", ["short"]), ("B", ["y"]), "t")
    write_result(tmp_path, "other", [dict(r, answer="C") if r["sample_id"] == "o2" else r for r in rows], lambda r: True)
    with pytest.raises(ValueError, match="another type or answer"):
        QP.compare(probe, tmp_path, ("A", ["other"]), ("B", ["y"]), "t")


# pre-registration v3 (the user, 2026-10-30): two seeds per group, pairs of the same seed and side, the sign over seeds
def test_parse_group_with_seeds():
    assert QP.parse_group("D=20261019:a,b;20261021:c,d") == ("D", [("20261019", ["a", "b"]), ("20261021", ["c", "d"])])
    assert QP.parse_group("a,b;c,d") == ("a-b-c-d", [("seed1", ["a", "b"]), ("seed2", ["c", "d"])])
    assert QP.parse_group("C0=s1:sub/a,/abs/b") == ("C0", [("s1", ["sub/a", "/abs/b"])])   # folders in result names
    for bad in ("D=a,b;c", "D=x:a,b;x:c,d", "D=a,b;c,d;e,f", "D=a,b,c;d,e"):
        with pytest.raises(ValueError):
            QP.parse_group(bad)


def test_seed_pairs_go_by_seed_and_side():
    two = [("s1", ["a1", "a2"]), ("s2", ["a3", "a4"])]
    other = [("s1", ["b1", "b2"]), ("s2", ["b3", "b4"])]
    pairs = QP.make_seed_pairs(two, other)
    assert pairs == [("s1/S0", "a1", "b1", "s1", "S0"), ("s1/D", "a2", "b2", "s1", "D"),
                     ("s2/S0", "a3", "b3", "s2", "S0"), ("s2/D", "a4", "b4", "s2", "D")]
    assert QP.seed_groups(pairs) == {"s1": [0, 1], "s2": [2, 3]}
    assert QP.make_seed_pairs(two, [("", ["y"])])[1] == ("s1/D", "a2", "y", "s1", "D")     # a single result serves all
    assert QP.make_seed_pairs([("", ["y"])], two)[2] == ("s2/S0", "y", "a3", "s2", "S0")
    one = QP.make_seed_pairs([("", ["a", "b"])], [("20261019", ["c", "d"])])              # one seed each: as v2
    assert one == [("S0", "a", "c", "20261019", "S0"), ("D", "b", "d", "20261019", "D")] and QP.seed_groups(one) == {}
    for b in ([("", ["b1", "b2"])],                                    # one seed against two
              [("s1", ["b1", "b2"]), ("s3", ["b3", "b4"])]):            # seeds that do not match
        with pytest.raises(ValueError):
            QP.make_seed_pairs(two, b)
    with pytest.raises(ValueError):
        QP.make_seed_pairs([("s1", ["a", "b"])], [("s2", ["c", "d"])])


def test_compare_two_seeds_sign_over_the_seeds(tmp_path):
    import json

    probe, rows = small_probe(tmp_path)
    # the goal questions each result answers right (indices).  goal_now, A - B per pair: seed 1 S0 +0.6, D -0.2;
    # seed 2 S0 +0.2, D +0.2: one pair is negative but both seeds are +0.2.  goal_after: seed 1 +0.8 / +0.8, seed 2
    # -0.1 / -0.1: the seeds differ in sign.  Occlusion: D results right on full and partial, C results on none.
    right = {"d1s0": (set(range(6)), set(range(8))), "d1d": (set(), set(range(8))), "d2s0": ({0, 1}, set()),
             "d2d": ({0, 1}, set()), "c1s0": (set(), set()), "c1d": ({0, 1}, set()), "c2s0": (set(), {9}),
             "c2d": (set(), {9})}
    for name, (now, after) in right.items():
        occ = ("full", "partial") if name.startswith("d") else ("none",)
        write_result(tmp_path, name, rows, lambda r, now=now, after=after, occ=occ:
                     (r["qtype"] == "goal_now" and int(r["sample_id"][8:]) in now)
                     or (r["qtype"] == "goal_after" and int(r["sample_id"][10:]) in after) or r["answer_value"] in occ)
    QP.main(["compare", "--probe", str(probe), "--out", str(tmp_path), "--reps", "200",
             "--a", "D=20261019:d1s0,d1d;20261021:d2s0,d2d", "--b", "C=20261019:c1s0,c1d;20261021:c2s0,c2d"])
    res = json.loads((tmp_path / "compare_D_vs_C.json").read_text(encoding="utf-8"))
    assert res["rules"]["same_sign_over"] == "seeds"
    assert [(p["pair"], p["a"], p["b"]) for p in res["pairs"]] == [
        ("20261019/S0", "d1s0", "c1s0"), ("20261019/D", "d1d", "c1d"),
        ("20261021/S0", "d2s0", "c2s0"), ("20261021/D", "d2d", "c2d")]
    g = res["goal_now"]
    assert [x["diff"] for x in g["per_pair"]] == pytest.approx([0.6, -0.2, 0.2, 0.2])
    assert [(s["seed"], s["n"]) for s in g["per_seed"]] == [("20261019", 20), ("20261021", 20)]
    assert [s["diff"] for s in g["per_seed"]] == pytest.approx([0.2, 0.2])
    assert [(s["only_a"], s["only_b"], s["p"]) for s in g["per_seed"]] == [(6, 2, QP.mcnemar(6, 2)), (4, 0, QP.mcnemar(4, 0))]
    assert (g["pooled"]["n"], g["pooled"]["only_a"], g["pooled"]["only_b"]) == (40, 10, 2)
    assert g["pooled"]["diff"] == pytest.approx(0.2) and g["pooled"]["p"] == QP.mcnemar(10, 2) < 0.05
    assert g["pooled"]["verdict"] == "difference"                                   # both seeds positive
    assert QP.verdict(0.2, [0.6, -0.2, 0.2, 0.2], 0.10, p=0.04)["failed"] == ["sign"]  # over the four pairs: no
    a = res["goal_after"]
    assert [s["diff"] for s in a["per_seed"]] == pytest.approx([0.8, -0.1])
    assert a["pooled"]["diff"] == pytest.approx(0.35) and a["pooled"]["p"] < 0.05 and a["pooled"]["failed"] == ["sign"]
    o = res["occlusion"]
    assert o["pooled"]["n"] == 48 and [s["n"] for s in o["per_seed"]] == [24, 24]
    assert [s["diff"] for s in o["per_seed"]] == pytest.approx([1 / 3, 1 / 3])
    assert o["per_seed"][0]["ci"] == pytest.approx([1 / 3, 1 / 3]) and o["pooled"]["verdict"] == "difference"
    md = (tmp_path / "compare_D_vs_C.md").read_text(encoding="utf-8")
    assert "且两个种子的差同号" in md and "种子 20261019 +20.0（S0 +60.0、D -20.0）" in md
    assert "| `goal_now` | 种子 20261019 S0 | 10 |" in md and "| `goal_now` | 种子 20261021（两对） | 20 |" in md
    assert "| 种子 20261019（两对） | 24 |" in md and "没达到：两个种子的差不同号或为 0" in md
    assert md.count("**有差异**") == 4                                              # goal_now and occlusion
    with pytest.raises(ValueError):                                                 # the seeds must match
        QP.main(["compare", "--probe", str(probe), "--out", str(tmp_path),
                 "--a", "D=20261019:d1s0,d1d;20261021:d2s0,d2d", "--b", "C=20261019:c1s0,c1d;20261022:c2s0,c2d"])


def test_sco_probe_python(tmp_path, monkeypatch):
    """sco_probe --python: the probe runs with that Python (Qwen3-VL: its runtime); without it the launch is unchanged."""
    import json

    from eval.diagnostics import sco_probe as SP

    monkeypatch.delenv("VAGEN_LITE_PYTHON", raising=False)
    probe = tmp_path / "probe.jsonl"
    probe.write_text("")
    plain = SP.prepare(tmp_path / "a", str(probe), models=["m=/x"]).read_text()
    assert "\n$ENV_ROOT/bin/python -m eval.diagnostics.qa_probe run --probe " in plain
    assert "python" not in json.loads((tmp_path / "a" / "job" / "prepared.json").read_text())
    q3 = "/venv q3/bin/python"
    text = SP.prepare(tmp_path / "b", str(probe), models=["m=/x"], python=q3).read_text()
    assert "\n'/venv q3/bin/python' -m eval.diagnostics.qa_probe run --probe " in text
    assert text.replace("'/venv q3/bin/python'", "$ENV_ROOT/bin/python").replace("/b/", "/a/").replace("/b\n", "/a\n") \
        == plain
    assert json.loads((tmp_path / "b" / "job" / "prepared.json").read_text())["python"] == q3


def test_sco_probe_from_chain(tmp_path, monkeypatch):
    """The chain's diagnostics step: VAGEN_LITE_PYTHON (the plan's model type) is the default Python, and a chain pool
    mode becomes the pool with the fewest waiting jobs; an sco pool and an explicit --python stay as given."""
    import json

    from data_gen.interiorgs_v10 import job_sco
    from eval.diagnostics import sco_probe as SP

    probe = tmp_path / "probe.jsonl"
    probe.write_text("")
    monkeypatch.setenv("VAGEN_LITE_PYTHON", "/q3/bin/python")
    monkeypatch.setattr(job_sco, "pick_pool", lambda pools=job_sco.POOLS: "h800")
    text = SP.prepare(tmp_path / "a", str(probe), models=["m=/x"], pool="queue").read_text()
    assert "\n/q3/bin/python -m eval.diagnostics.qa_probe run --probe " in text
    prepared = json.loads((tmp_path / "a" / "job" / "prepared.json").read_text())
    assert (prepared["pool"], prepared["python"]) == ("h800", "/q3/bin/python")
    for mode in SP.CHAIN_POOL_MODES:
        SP.prepare(tmp_path / mode, str(probe), models=["m=/x"], pool=mode)
        assert json.loads((tmp_path / mode / "job" / "prepared.json").read_text())["pool"] == "h800"
    SP.prepare(tmp_path / "z", str(probe), models=["m=/x"], pool="zoetrope", python="/other/python")
    prepared = json.loads((tmp_path / "z" / "job" / "prepared.json").read_text())
    assert (prepared["pool"], prepared["python"]) == ("zoetrope", "/other/python")
