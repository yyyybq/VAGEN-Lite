"""The question-answering batch of 2026-10-30 (CPU, no model, no simulator): q0_ask.py's bookkeeping, X0b's
question building and parsing, G0's question rules and analysis, the linear probe, the parameter distances and the
analysis stage of run_qa_batch.sh."""
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x"))
import g0_qa as G  # noqa: E402
import param_dist as PD  # noqa: E402
import probe_lp as LP  # noqa: E402
import q0_ask as Q  # noqa: E402
import x0b_qa as XB  # noqa: E402


# ------------------------------------------------------------------------------------------------- q0_ask
def test_models_and_families(tmp_path):
    starts = tmp_path / "starts.json"
    starts.write_text(json.dumps({"B": {"path": str(tmp_path / "b")}, "S": {"path": str(tmp_path / "s")}}))
    (tmp_path / "b").mkdir()
    (tmp_path / "b/preprocessor_config.json").write_text("{}")
    (tmp_path / "s").mkdir()
    assert Q.family("Q3RD") == "qwen3" and Q.family("RS") == "qwen2.5"
    with pytest.raises(KeyError):
        Q.family("X")
    assert Q.model_path("S", starts) == str(tmp_path / "s")
    assert Q.processor_dir("S", starts) == str(tmp_path / "b")  # no processor files: the family's base
    assert Q.processor_dir("B", starts) == str(tmp_path / "b")
    assert Q.QWEN3["Q3RD"].endswith("rl_run_q3d/checkpoints/global_step_300/actor/huggingface")
    assert Q.layer_indices(28) == [7, 14, 21, 28] and Q.layer_indices(36) == [9, 18, 27, 36]


def fake_questions(path, n=4):
    rows = [{"id": f"q{i}", "prompt": f"p{i}", "image_path": f"/img{i}.png"} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_main_asks_what_is_missing_and_writes_hidden_states(tmp_path, monkeypatch):
    qf = tmp_path / "q.jsonl"
    fake_questions(qf)
    out = tmp_path / "a.jsonl"
    out.write_text(json.dumps({"id": "q0", "reply": "A"}) + "\n")
    asked = []
    monkeypatch.setattr(Q, "load", lambda name: ("proc", "model", "/proc"))
    monkeypatch.setattr(Q, "model_path", lambda name, starts=None: f"/models/{name}")
    monkeypatch.setattr(Q, "ask", lambda p, m, q: asked.append(q["id"]) or f"reply {q['id']}")
    monkeypatch.setattr(Q, "n_text_layers", lambda m: 8)
    monkeypatch.setattr(Q, "hidden", lambda p, m, q, layers: np.full((len(layers), 3), int(q["id"][1:]), "float16"))
    hid = tmp_path / "h.npz"
    assert Q.main(["--model", "B", "--ask", f"{qf}={out}", "--hidden", f"{qf}={hid}", "--smoke", "1"]) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["q0", "q1", "q2", "q3"] and asked == ["q0", "q1", "q2", "q3"]  # smoke + 3
    z = np.load(hid)
    assert z["ids"].tolist() == ["q0", "q1", "q2", "q3"] and z["layers"].tolist() == [2, 4, 6, 8]
    assert z["h"].shape == (4, 4, 3) and float(z["h"][2, 0, 0]) == 2.0
    assert json.loads(Path(str(out) + ".meta.json").read_text())["processor"] == "/proc"


def test_a_failing_smoke_test_stops_with_3(tmp_path, monkeypatch):
    qf = tmp_path / "q.jsonl"
    fake_questions(qf)
    monkeypatch.setattr(Q, "load", lambda name: ("proc", "model", "/proc"))
    monkeypatch.setattr(Q, "model_path", lambda name, starts=None: f"/models/{name}")

    def boom(p, m, q):
        raise RuntimeError("no")
    monkeypatch.setattr(Q, "ask", boom)
    assert Q.main(["--model", "Q3B", "--ask", f"{qf}={tmp_path / 'a.jsonl'}", "--smoke", "2"]) == 3


# ------------------------------------------------------------------------------------------------- X0b
def bank_record(qid, sub, objs, values, split="test", view="v1", scene="0001_1"):
    return {"qid": qid, "group": "S1", "subtasks": sub, "scene_id": scene, "split": split, "media": "image",
            "views": [view], "objects": objs, "gt": {"values": values}}


def obj(i, name, role="object", center=(0.0, 0.0, 1.0), method="unique_in_image"):
    return {"id": str(i), "display": name, "name": f"the {name}", "role": role, "name_method": method,
            "obb": {"center": list(center)}}


VIEW = {"view_id": "v1", "c2w": [[1, 0, 0, 0.0], [0, 0, 1, 0.0], [0, -1, 0, 1.5], [0, 0, 0, 1]]}  # looks along +y


def test_bank_candidates_use_x0_thresholds():
    lr = bank_record("a", ["ES-01", "SITE-C02"], [obj(1, "oven", "left_object", (0, 3, 1)),
                                                  obj(2, "sink", "right_object", (1, 3.1, 1))],
                     {"bbox_left": [20, 0, 100, 50], "bbox_right": [300, 0, 380, 50]})
    near = bank_record("b", ["3D-03"], [obj(3, "plant", center=(0, 1, 1)), obj(4, "table", center=(0, 2.5, 1))],
                       {"mean_depths": [1.0, 2.4]})
    close = bank_record("c", ["ES-02"], [obj(5, "mug", "left_object"), obj(6, "cup", "right_object")],
                        {"bbox_left": [100, 0, 140, 10], "bbox_right": [150, 0, 190, 10]})  # 50 px at 512: too close
    named = bank_record("d", ["3D-03"], [obj(7, "chair", method="scene_v10_r5"), obj(8, "bed")],
                        {"mean_depths": [1.0, 3.0]})
    train = bank_record("e", ["3D-03"], [obj(9, "sofa"), obj(10, "lamp")], {"mean_depths": [1.0, 3.0]}, split="train")
    c = XB.bank_candidates([lr, near, close, named, train], {"v1": VIEW}, lambda s, v: f"/bank/{s}/{v}_256.png",
                           random.Random(0))
    kinds = sorted((q["kind"], q["bank_qid"]) for q in c)
    # lr: left_right + front_back (oven 3.0 vs sink 3.1 forward: under 0.2 m, so none); near: near_far + front_back
    assert ("left_right", "a") in kinds and ("near_far", "b") in kinds and ("front_back", "b") in kinds
    assert not any(k[1] in ("c", "d", "e") and k[0] != "front_back" for k in kinds)
    assert not any(k[1] in ("d", "e") for k in kinds)
    lrq = next(q for q in c if q["kind"] == "left_right")
    assert lrq["answer_text"] == ("left" if lrq["a"]["name"] == "oven" else "right")
    assert lrq["prompt"].startswith("This image is from the robot's camera. In this image, is the ")
    nf = next(q for q in c if q["kind"] == "near_far")
    assert nf["answer_text"] == "plant" and nf["image_path"] == "/bank/0001_1/v1_256.png"
    fb = next(q for q in c if q["kind"] == "front_back" and q["bank_qid"] == "b")
    assert fb["answer_text"] == "table"  # 2.5 m ahead vs 1.0 m


def test_bank_selection_alternates_letters():
    rng = random.Random(1)
    cands = []
    for i in range(12):
        a, b = {"key": f"a{i}", "name": "mug", "kind": "object", "cx": 10.0}, {"key": f"b{i}", "name": "sink",
                                                                                "kind": "object", "cx": 100.0}
        q = XB.x0_qa.question("left_right", a, b, "robot0_agentview_left", rng)
        cands.append(dict(q, camera="bank", task=f"s{i % 4}", seed=f"v{i}", prompt=XB.x0_prompt(q["question"], q["options"])))
    chosen = XB.select_bank(cands, 10, random.Random(0))
    assert len(chosen) == 10 and [q["answer"] for q in chosen] == list("ABABABABAB")
    for q in chosen:
        assert q["options"]["AB".index(q["answer"])] == q["answer_text"] == "left"
        assert q["prompt"].endswith(f"A. {q['options'][0]}\nB. {q['options'][1]}\nAnswer with the letter of the "
                                    "correct option only.")


class FakeQA:
    STYLES = {"mindcube": lambda q, o: "HEADER\n" + q + " " + " ".join(f"{'AB'[i]}. {x}" for i, x in enumerate(o))}


def test_tagged_layout_and_its_parsing():
    q = {"prompt": "This image is from the robot's left camera. Is the mug left or right of the sink?\nA. left\n"
                   "B. right\nAnswer with the letter of the correct option only.", "options": ["left", "right"]}
    assert XB.tagged_prompt(q, FakeQA) == ("HEADER\nThis image is from the robot's left camera. Is the mug left or "
                                           "right of the sink? A. left B. right")
    assert XB.parse_tagged("<answer>B. right</answer>") == "B"
    assert XB.parse_tagged(" <answer> A. left</answer>") == "A"
    assert XB.parse_tagged("B") == "B" and XB.parse_tagged("left") is None
    assert XB.tagged_ok("<answer>A. left</answer>") and not XB.tagged_ok("A") and not XB.tagged_ok("<answer>A</answer>")
    qs = [{"id": "1", "kind": "left_right"}, {"id": "2", "kind": "left_right"}, {"id": "3", "kind": "near_far"}]
    share = XB.a_share(qs, {"1": "A", "2": None, "3": "B"})
    assert share["left_right"] == 1.0 and share["near_far"] == 0.0 and math.isnan(share["front_back"])


def test_main_qa_module_loads_by_path(tmp_path):
    f = tmp_path / "data_gen/interiorgs_qa/qa.py"
    f.parent.mkdir(parents=True)
    f.write_text("import dataclasses\n@dataclasses.dataclass\nclass P:\n    a: int = 0\n"
                 "STYLES = {'mindcube': lambda q, o: 'X' + q}\n")  # a dataclass, as the real qa.py has
    assert XB.main_qa_module(tmp_path).STYLES["mindcube"]("q", []) == "Xq"


# ------------------------------------------------------------------------------------------------- G0
def g0_scene(task="T", seed=1, names=("mug", "sink", "stove"), views=None):
    return {"task": task, "seed": seed, "names": list(names), "views": views or {}}


def ent(key, name, pixels, kind="object", openness=None, cx=10.0):
    e = {"key": key, "name": name, "kind": kind, "pixels": pixels, "cx": cx, "cy": 10.0}
    if openness is not None:
        e["openness"] = openness
    return e


def test_g0_candidates_follow_the_rules():
    view = {"image": "/x.png", "same_as_x0": True, "entities": [
        ent("m1", "mug", 900), ent("m2", "mug", 500), ent("a1", "apple", 450), ent("a2", "apple", 120),
        ent("mw", "microwave", 3000, "fixture", openness=0.8), ent("dw", "dishwasher", 2000, "fixture", openness=0.2),
        ent("ov", "oven", 2500, "fixture", openness=0.0), ent("ov2", "oven", 60, "fixture", openness=0.0),
        ent("hid", "kettle", 0)]}
    s = g0_scene(names=("mug", "apple", "microwave", "dishwasher", "oven", "kettle", "sink"),
                 views={"robot0_agentview_left": view})
    pool = ["sink", "bowl", "cup", "banana", "kettle non electric", "pear"]
    c = G.candidates([s], pool, random.Random(0))
    kinds = {}
    for q in c:
        kinds.setdefault(q["kind"], []).append(q)
    yes = {q["question"] for q in kinds["existence"] if q["truth"] == "yes"}
    assert yes == {"Is there a mug in this image?", "Is there an apple in this image?",
                   "Is there a microwave in this image?", "Is there a dishwasher in this image?",
                   "Is there an oven in this image?"}
    no = {q["negative"] for q in kinds["existence"] if q["truth"] == "no"}
    assert no == {"bowl", "banana", "pear"}  # sink is in the scene; cup is in mug's synonym group; kettle a word
    oc = {q["target"]["name"]: q["answer_text"] for q in kinds["open_closed"]}
    assert oc == {"microwave": "open"}  # dishwasher 0.2 is in between; two ovens visible (60 px): name not unique
    counts = {q["question"].split("How many ")[1].split(" can")[0]: q["answer"] for q in kinds["counting"]}
    assert counts["mugs"] == "2" and "apples" not in counts  # an apple with 120 px makes the count ambiguous
    for q in kinds["identification"]:
        assert q["negative"] in ("bowl", "banana", "pear") and q["options"]["AB".index(q["answer"])] == q["target"]["name"]
    assert len({q["id"] for q in c}) == len(c)


def test_g0_selection_balances_answers():
    views = {}
    scenes = []
    for i in range(30):
        es = [ent(f"m{i}", "mug", 900), ent(f"w{i}", "microwave", 2000, "fixture", openness=1.0 if i % 2 else 0.0),
              ent(f"p{i}", "plate", 800), ent(f"q{i}", "plate", 700 if i % 3 == 0 else 0)]
        scenes.append(g0_scene(task=f"T{i % 5}", seed=i, names=("mug", "microwave", "plate"),
                               views={"robot0_agentview_left": {"image": "/x.png", "same_as_x0": True, "entities": es}}))
    c = G.candidates(scenes, ["banana", "pear", "apple"], random.Random(0))
    chosen = G.select(c, 60, random.Random(0), targets={"existence": 20, "identification": 20, "open_closed": 10,
                                                         "counting": 10})
    by = {}
    for q in chosen:
        by.setdefault(q["kind"], []).append(q)
    oc = by["open_closed"]
    assert len(oc) == 10 and sum(q["answer_text"] == "open" for q in oc) == 5
    ex = by["existence"]
    assert sum(q["answer_text"] == "yes" for q in ex) == len(ex) // 2
    assert abs(sum(q["answer"] == "A" for q in ex) - len(ex) / 2) <= 1
    for q in ex + by["identification"] + oc:
        assert q["options"]["AB".index(q["answer"])] == q["answer_text"]
        assert q["prompt"].endswith("Answer with the letter of the correct option only.")
    assert len(chosen) == 60 and len(by["counting"]) == 10


def test_qwen3_gate_follows_the_three_fixed_rules():
    import qa_report as QR

    def row(acc, p=None, share=0.5, unparsed=0.0):
        r = {"all": {"accuracy": acc, "unparsed_rate": unparsed}, "a_share": {k: share for k in QR.KINDS}}
        if p is not None:
            r["all"]["vs_reference"] = {"p": p, "only_model": 1, "only_reference": 0}
        return r
    q0 = {"Q3B": row(0.50), "Q3RS": row(0.62, p=0.001), "Q3RD": row(0.55, p=0.2)}
    g0 = {"Q3B": {"all": {"accuracy": 0.72}}, "Q3RS": {"all": {"accuracy": 0.70, "vs_reference": {"p": 0.3}}},
          "Q3RD": {"all": {"accuracy": 0.60, "vs_reference": {"p": 0.01}}}}
    g = QR.gate(q0, g0)
    assert g["Q3RS"]["rule1"] and g["Q3RS"]["rule2"] and g["Q3RS"]["rule3"] and g["Q3RS"]["passes"]
    assert not g["Q3RD"]["rule1"] and g["Q3RD"]["rule3"] is False and g["gate"] is True
    assert g["Q3RS"]["diff_points"] == pytest.approx(12.0)
    biased = dict(q0, Q3RS=row(0.62, p=0.001, share=0.65))
    assert QR.gate(biased, g0)["Q3RS"]["rule2"] is False and QR.gate(biased, g0)["gate"] is False
    exactly_ten = dict(q0, Q3RS=row(0.60, p=0.01))
    assert QR.gate(exactly_ten, g0)["Q3RS"]["rule1"]  # 10.0 points counts
    assert QR.gate(q0, None)["gate"] is None  # G0 not in yet: undecided


def test_g0_sample_notes_for_every_kind():
    assert G.sample_note({"kind": "existence", "negative": "pear"}) == "目标：pear（场景里没有）"
    assert G.sample_note({"kind": "existence", "target": {"name": "mug", "pixels": 900}}) == "目标：mug（900 像素）"
    assert "场景里没有：pear" in G.sample_note({"kind": "identification", "target": {"name": "mug", "pixels": 9},
                                                "negative": "pear"})
    assert G.sample_note({"kind": "counting", "targets": [{"pixels": 500}, {"pixels": 700}]}) == "每个 500 像素，700 像素"


def test_g0_scene_reading_skips_unreadable_lines(tmp_path):
    d = tmp_path / "part1"
    d.mkdir()
    (d / "scenes.jsonl").write_text(json.dumps({"task": "T", "seed": 1}) + "\n" + "\x00" * 40 + "\n{broken\n\n"
                                    + json.dumps({"task": "T", "seed": 2}) + "\n")
    scenes, bad = G.read_scenes([d])
    assert [s["seed"] for s in scenes] == [1, 2] and bad == 1


def test_g0_parsing_and_analysis():
    two = {"id": "e1", "kind": "existence", "options": ["no", "yes"], "answer": "B", "answer_text": "yes"}
    cnt = {"id": "c1", "kind": "counting", "options": None, "answer": "2", "answer_text": "2"}
    assert G.parse_reply(two, "B.") == "B" and G.parse_reply(two, "Yes") == "B" and G.parse_reply(two, "maybe") is None
    assert G.parse_reply(cnt, "2") == "2" and G.parse_reply(cnt, "There are two.") == "2"
    assert G.compliant(two, " A. ") and not G.compliant(two, "yes") and G.compliant(cnt, "3.") and not G.compliant(cnt, "3 mugs")
    qs = [two, cnt, dict(two, id="e2", options=["yes", "no"], answer="A")]
    res = G.analyze(qs, {"B": {"e1": "B", "c1": "2", "e2": "A"}, "S": {"e1": "A", "c1": "two", "e2": "A"}},
                    {"B": None, "S": "B"})
    assert res["B"]["all"]["correct"] == 3 and res["S"]["all"]["correct"] == 2
    assert res["S"]["all"]["vs_reference"]["only_reference"] == 1 and res["S"]["all"]["compliance"] == 2 / 3
    assert res["S"]["existence"]["yes_share"] == 0.5 and res["B"]["existence"]["a_share"] == 0.5
    assert G.reference_of("Q3RS") == "Q3B" and G.reference_of("RD") == "B" and G.reference_of("Q3B") is None
    assert G.plural("knife") == "knifes"  # a simple rule: no irregular plurals
    assert G.plural("glass") == "glasses" and G.plural("cherry") == "cherries" and G.plural("mug") == "mugs"


# ------------------------------------------------------------------------------------------------- probe
def test_probe_labels_and_grouped_folds():
    q = {"kind": "near_far", "a": {"name": "mug"}, "b": {"name": "sink"}, "answer_text": "mug", "answer": "B"}
    assert LP.labels(q) == {"relation": 1, "letter": 0}
    assert LP.labels(dict(q, kind="left_right", answer_text="right", answer="A")) == {"relation": 0, "letter": 1}
    qs = [{"task": f"T{i % 7}", "seed": i % 3} for i in range(60)]
    folds = LP.scene_folds(qs)
    by_scene = {}
    for q, f in zip(qs, folds):
        by_scene.setdefault((q["task"], q["seed"]), set()).add(f)
    assert all(len(v) == 1 for v in by_scene.values()) and set(folds) == set(range(5))


def test_logistic_regression_learns_a_linear_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 20))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    folds = [i % 5 for i in range(120)]
    pred = LP.cv_predict(X, y, folds)
    assert (pred == y).mean() > 0.8
    noise = LP.cv_predict(X, rng.integers(0, 2, 120).astype(float), folds)
    assert 0.2 < noise.mean() < 0.8


def test_probe_structure_and_comparison():
    rng = np.random.default_rng(1)
    qs = []
    for i in range(60):
        kind = ["left_right", "near_far", "front_back"][i % 3]
        right = i % 2 == 0
        qs.append({"id": f"q{i}", "kind": kind, "task": f"T{i % 10}", "seed": i % 4, "a": {"name": "mug"},
                   "answer_text": ("left" if right else "right") if kind == "left_right" else ("mug" if right else "sink"),
                   "answer": "A" if i % 4 < 2 else "B"})
    rel = np.array([LP.labels(q)["relation"] for q in qs], float)
    h_good = rng.normal(size=(60, 2, 50))
    h_good[:, :, 0] += 4 * (2 * rel[:, None] - 1)
    h_bad = rng.normal(size=(60, 2, 50))
    res = {"B": LP.probe(qs, h_good, [7, 28]), "S": LP.probe(qs, h_bad, [7, 28])}
    LP.compare(res)
    assert res["B"][28]["all"]["relation"]["accuracy"] > 0.85
    vs = res["S"][28]["all"]["relation"]["vs_reference"]
    assert vs["reference"] == "B" and vs["only_reference"] > vs["only_model"] and vs["p"] < 0.05
    assert set(res["B"][7]) == {"all", "left_right", "near_far", "front_back"}


# ------------------------------------------------------------------------------------------- parameter distance
def test_names_are_brought_to_one_form():
    assert PD.canon("model.layers.3.self_attn.q_proj.weight") == ("layers.3.self_attn.q_proj.weight", "llm.03")
    assert PD.canon("model.language_model.layers.3.mlp.up_proj.weight") == ("layers.3.mlp.up_proj.weight", "llm.03")
    assert PD.canon("qwen_vl_interface.model.model.visual.blocks.12.attn.qkv.weight")[1] == "vision.12"
    assert PD.canon("visual.blocks.12.attn.qkv.weight")[0] == "visual.blocks.12.attn.qkv.weight"
    assert PD.canon("model.visual.deepstack_merger_list.0.norm.weight")[1] == "merger"
    assert PD.canon("visual.patch_embed.proj.weight")[1] == "vision.embed"
    assert PD.canon("qwen_vl_interface.model.lm_head.weight") == ("lm_head.weight", "lm_head")
    assert PD.canon("model.embed_tokens.weight")[1] == "embed" and PD.canon("model.norm.weight")[1] == "llm.norm"
    assert PD.canon("action_model.net.0.weight") is None
    assert PD.module_order(["lm_head", "llm.10", "vision.02", "llm.02", "merger", "embed", "llm.norm"]) == \
        ["vision.02", "merger", "llm.02", "llm.10", "llm.norm", "embed", "lm_head"]


class DictReader:
    def __init__(self, tensors):
        import torch

        self.t = {k: torch.tensor(v, dtype=torch.float32) for k, v in tensors.items()}
        self.names = {PD.canon(k)[0]: (k, None) for k in tensors}

    def get(self, cname):
        return self.t[self.names[cname][0]]


def test_relative_distances_and_cosines():
    pytest.importorskip("torch")
    group = {"deltas": [("S-B", "S", "B"), ("R-B", "R", "B")], "cosines": [("S-B", "R-B")]}
    base = {"model.layers.0.w": [3.0, 4.0], "lm_head.weight": [1.0, 0.0], "model.layers.1.w": [1.0]}
    readers = {"B": DictReader(base),
               "S": DictReader({"model.layers.0.w": [3.0, 4.5], "lm_head.weight": [1.0, 0.0], "model.layers.1.w": [2.0, 2.0]}),
               "R": DictReader({"model.layers.0.w": [3.0, 3.5], "lm_head.weight": [1.0, 1.0], "model.layers.1.w": [1.0]})}
    raw = PD.compute(group, readers, log=lambda m: None)
    assert [s[0] for s in raw["skipped"]] == ["layers.1.w"]  # shapes differ: left out everywhere
    res = PD.summarize(group, raw)
    assert res["per_module"]["llm.00"]["relative"]["S-B"] == pytest.approx(0.5 / 5)
    assert res["per_module"]["llm.00"]["cosine"]["S-B|R-B"] == pytest.approx(-1.0)
    assert res["per_module"]["lm_head"]["relative"]["S-B"] == 0.0
    assert res["all"]["relative"]["R-B"] == pytest.approx(math.sqrt(0.25 + 1) / math.sqrt(25 + 1))
    assert res["coarse"]["llm"]["relative"]["S-B"] == pytest.approx(0.1)


# ------------------------------------------------------------------------------------------- the job script
def test_analysis_stage_runs_on_what_exists(tmp_path):
    x0 = tmp_path / "x0"
    x0.mkdir()
    qs = []
    for i in range(30):
        kind = ["left_right", "near_far", "front_back"][i % 3]
        qs.append({"id": f"q{i}", "kind": kind, "task": f"T{i % 10}", "seed": i % 3, "a": {"name": "mug"},
                   "b": {"name": "sink"}, "answer": "AB"[i % 2], "answer_text": "mug", "object_question": True,
                   "prompt": "p", "image_path": "/x.png"})
    (x0 / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    (x0 / "answers_B.jsonl").write_text("".join(json.dumps({"id": q["id"], "reply": "A"}) + "\n" for q in qs))
    out = tmp_path / "out"
    (out / "answers").mkdir(parents=True)
    (out / "hidden").mkdir()
    (out / "answers/x0_Q3B.jsonl").write_text("".join(json.dumps({"id": q["id"], "reply": q["answer"]}) + "\n" for q in qs))
    rng = np.random.default_rng(0)
    for m in ("B", "Q3B"):
        np.savez(out / f"hidden/x0_{m}.npz", ids=np.array([q["id"] for q in qs]), layers=np.array([2, 4, 6, 8]),
                 h=rng.normal(size=(30, 4, 6)).astype("float16"))
    env = dict(os.environ, QA_X0=str(x0), QA_VLM_PY=sys.executable)
    res = subprocess.run(["bash", str(ROOT / "x/run_qa_batch.sh"), str(out), "analyze"], capture_output=True,
                         text=True, env=env, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    q0 = json.loads((out / "results/q0_vs_B.json").read_text())
    assert q0["Q3B"]["all"]["accuracy"] == 1.0 and q0["B"]["all"]["accuracy"] == 0.5
    assert q0["Q3B"]["all"]["vs_reference"]["only_model"] == 15
    assert set(json.loads((out / "results/probe.json").read_text())) == {"B", "Q3B"}
    assert "analysis: x0 2 models" in (out / "qa_analyze.log").read_text()
