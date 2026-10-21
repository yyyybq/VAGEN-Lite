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
    assert QP.parse_group("D=a,b") == ("D", ["a", "b"]) and QP.parse_group("x") == ("x", ["x"])
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
