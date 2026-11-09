"""Tests of the EXPLORATORY learn483 package: vpd/walkerbench/learn483.py and vpd/jobs/wb_learn483.sh.

Pure parts (demo parsing, the restricted-candidate type probabilities, summaries, AUROC, PartWriter resume, s2in50
build/verify, the run-time task-set registration, the closed-loop analysis on a synthetic eval with a temp cache.db),
the payload (bash -n, VPD_DRYRUN=1 order of both arms, soft deadline, refusals), the floors on 2 real s2val tasks
(sandbox under the test dir), and the teacher-forced probe / merge parity end to end on a tiny random Qwen3-VL
(2 layers, hidden 64, Q3S's config and processor; CPU, 1 thread).

Login node, 1 core, no GPU; never reads or writes real OUT run dirs (temp dirs + VPD_X483_ROOT / VPD_X483_PILOT).
Scratch: eval_external_assets/vpd/scratch/impl-wb483/pytest_<rand>/ (env VPD_X483_TEST_DIR overrides), removed at
the end.
    cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD
    nice -n 19 $PY_Q3 -m pytest -q -p no:cacheprovider --basetemp=$SCRATCH/basetemp vpd/tests/test_learn483.py
    nice -n 19 $PY_TRAIN vpd/tests/test_learn483.py        # starVLA (the probe's interpreter): standalone runner
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_WT = Path(__file__).resolve().parents[2]
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

try:
    import pytest
except ImportError:  # starVLA: the standalone runner at the bottom
    pytest = None

from vpd import paths  # noqa: E402
from vpd.walkerbench import learn483 as L  # noqa: E402

JOBS_DIR = paths.VPD_PKG / "jobs"
PAYLOAD = JOBS_DIR / "wb_learn483.sh"


class Skip(Exception):
    pass


def skip(msg: str):
    if pytest is not None:
        pytest.skip(msg)
    raise Skip(msg)


def raises(exc, fn, *a, match: str = None, **k):
    try:
        fn(*a, **k)
    except exc as e:
        if match and match not in str(e):
            raise AssertionError(f"{e!r} does not contain {match!r}") from e
        return e
    raise AssertionError(f"{exc.__name__} not raised")


def _make_workdir() -> Path:
    base = Path(os.environ.get("VPD_X483_TEST_DIR") or paths.scratch("impl-wb483"))
    d = base / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    return d


if pytest is not None:
    @pytest.fixture(scope="module")
    def workdir():
        d = _make_workdir()
        yield d
        shutil.rmtree(d, ignore_errors=True)  # only what this module created


def _sub(workdir: Path, name: str) -> Path:
    d = workdir / f"{name}_{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True)
    return d


# ------------------------------------------------------------------------------------------ synthetic demos
def user_text(step: int, moves, heading: float, stated=120) -> str:
    mv = "\n".join(f"  {i}: {d}" + (f" ({dist:.1f}m)" if dist else "") for i, d, dist in moves)
    return (f"**Step {step}**\n\n**Task:** Navigate to Quick Mart. Head southeast for 106m. Turn right for 14m. "
            f"Total distance: approximately {stated} m.\n\n**Current View:**\n- Heading: {int(heading)}°SE\n"
            f"- Pitch: 0°\n\n**Available Moves:**\n{mv}\n\n🔄 **ROTATE** - Look in a different direction "
            f"(without moving):\n  - Look LEFT: heading={(heading - 90) % 360}\n  - Look RIGHT: heading="
            f"{(heading + 90) % 360}\n  - Look BACK: heading={(heading + 180) % 360}\n  - Look UP: pitch=30 (keep "
            f"current heading={heading})\n  - Look DOWN: pitch=-30 (keep current heading={heading})\n  - Or any "
            f"custom heading (0-360) and pitch (-85 to 85)\n\nAnalyze the image and decide your next action.<image>")


ACTS = ["MOVE(1)", "ROTATION(90.5, 0)", "MOVE(2)", 'STOP("ARRIVED")']
MOVES = [[(1, "front", 9.6)], [(1, "front", 10.4), (2, "back", 9.6)],
         [(1, "front", 8.0), (2, "right", 12.5), (3, "back", 10.4)], [(1, "front", 7.0), (2, "back", 12.5)]]


def _image(path: Path, seed: int, size=(640, 400)) -> str:
    from PIL import Image

    rnd = random.Random(seed)
    im = Image.frombytes("RGB", size, bytes(rnd.getrandbits(8) for _ in range(size[0] * size[1] * 3)))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    return str(path)


def make_demo(rid="nav_t1", img_dir: Path = None, kind="nav", split="val", acts=ACTS, seed=0) -> dict:
    msgs = [{"role": "system", "content": "You are a test walker. Reply with one JSON action."}]
    imgs = []
    heading = 154.40994
    for t, a in enumerate(acts):
        msgs.append({"role": "user", "content": user_text(t + 1, MOVES[t % len(MOVES)], heading)})
        msgs.append({"role": "assistant", "content": json.dumps({"THOUGHT": "", "ACTION": a})})
        if a.startswith("ROTATION"):
            heading = 90.5
        imgs.append(_image(img_dir / rid / f"turn_{t:02d}.jpg", seed * 100 + t) if img_dir else f"/x/{rid}/{t}.jpg")
    return {"id": rid, "task_type": kind, "split": split, "messages": msgs, "images": imgs, "n_turns": len(acts),
            "size": [640, 400]}


# ------------------------------------------------------------------------------------------ pure python
def test_parse_user_turn_and_action():
    u = L.parse_user_turn(user_text(2, MOVES[1], 154.40994, stated=146))
    assert u["moves"] == [{"id": 1, "direction": "front", "distance": 10.4},
                          {"id": 2, "direction": "back", "distance": 9.6}]
    assert u["stated_m"] == 146.0 and abs(u["heading"] - 154.40994) < 1e-9
    assert L.parse_user_turn(user_text(1, [], 3.5))["moves"] == []
    assert L.parse_action("MOVE(11)") == {"type": "MOVE", "id": 11}
    r = L.parse_action("ROTATION(244.4, 0)")
    assert r["type"] == "ROTATION" and r["heading"] == 244.4 and r["heading_text"] == "244.4"
    assert L.parse_action('STOP("ARRIVED")') == {"type": "STOP"}
    raises(ValueError, L.parse_action, "JUMP(1)")


def test_turn_table_walked_nodes_distance():
    rec = make_demo()
    task = {"spawn_point": "p0", "target_pano_ids": ["p2"],
            "visual_path": [{"pano_id": p} for p in ("p0", "p1", "p2")]}
    coords = {"p0": (0.0, 0.0), "p1": (0.0, 0.0001), "p2": (0.0, 0.0002)}
    rows = L.turn_table(rec, task, coords)
    assert [r["expert_type"] for r in rows] == ["MOVE", "ROTATION", "MOVE", "STOP"]
    assert [r["node"] for r in rows] == ["straight", "turn", "turn", "stop"]
    assert [r["n_legal"] for r in rows] == [1, 2, 3, 2]
    assert [r["walked_m"] for r in rows] == [0.0, 9.6, 9.6, 22.1]  # MOVE(1) 9.6 m, then MOVE(2) 12.5 m
    assert [r["pano"] for r in rows] == ["p0", "p1", "p1", "p2"]
    d = [r["dist_to_target_m"] for r in rows]
    assert d[0] > d[1] == d[2] > d[3] == 0.0
    assert [r["is_final"] for r in rows] == [False, False, False, True]
    assert rows[0]["stated_m"] == 120.0 and rows[1]["expert_heading_text"] == "90.5"
    assert rows[0]["reply"] == rec["messages"][2]["content"]
    assert all(r["dist_to_target_m"] is None for r in L.turn_table(rec))  # no task: no distances


def _fake_lp(table):
    return lambda j, tok: table[(j, tok)]


def test_cand_logprobs_distinct_tokens():
    pre = [1, 2, 3]
    cands = {"MOVE": pre + [10, 11], "ROTATION": pre + [20, 21], "STOP": pre + [30, 31]}
    lp = _fake_lp({(3, 10): math.log(0.6), (3, 20): math.log(0.1), (3, 30): math.log(0.1)})
    r = L.cand_logprobs(cands, cands["MOVE"], lp)
    assert r["d"] == 3 and r["resolved"] and r["pred"] == "MOVE"
    assert abs(r["p"]["MOVE"] - 0.75) < 1e-9 and abs(sum(r["p"].values()) - 1) < 1e-9
    raises(ValueError, L.cand_logprobs, cands, [1, 9, 3, 10], lp)


def test_cand_logprobs_shared_token_on_and_off_the_expert_path():
    pre = [1, 2]
    cands = {"MOVE": pre + [5, 6], "ROTATION": pre + [5, 7], "STOP": pre + [8, 9]}
    table = {(2, 5): math.log(0.5), (2, 8): math.log(0.25), (3, 6): math.log(0.8), (3, 7): math.log(0.2)}
    # the expert is MOVE: the shared token 5 is on its path, so MOVE / ROTATION extend by one token
    r = L.cand_logprobs(cands, cands["MOVE"], _fake_lp(table))
    assert r["resolved"] and r["pred"] == "MOVE"
    z = 0.5 * 0.8 + 0.5 * 0.2 + 0.25
    assert abs(r["p"]["MOVE"] - 0.4 / z) < 1e-9 and abs(r["p"]["ROTATION"] - 0.1 / z) < 1e-9
    # the expert is STOP: the pair leaves the forced path -> one group with mass p(5)
    r = L.cand_logprobs(cands, cands["STOP"], _fake_lp(table))
    assert not r["resolved"] and r["p"]["MOVE"] is None and r["p"]["ROTATION"] is None
    assert abs(r["p"]["STOP"] - 0.25 / 0.75) < 1e-9 and abs(r["groups"]["MOVE|ROTATION"] - 0.5 / 0.75) < 1e-9
    assert r["pred"] == "MOVE|ROTATION"


def test_candidate_texts_and_next_ok():
    rows = L.turn_table(make_demo())
    c = L.candidate_texts(rows[0])
    assert c["MOVE"] == rows[0]["reply"] and c["STOP"] == '{"THOUGHT": "", "ACTION": "STOP(\\"ARRIVED\\")"}'
    assert c["ROTATION"] == '{"THOUGHT": "", "ACTION": "ROTATION(154.4, 0)"}'  # current heading, 1 decimal
    assert L.candidate_texts(rows[1])["ROTATION"] == rows[1]["reply"]
    assert L.candidate_texts(rows[2])["MOVE"] == '{"THOUGHT": "", "ACTION": "MOVE(2)"}'
    base = {"expert_type": "MOVE", "expert_id": 2, "pred_type": "MOVE", "move_pred_id": 2}
    assert L.next_ok(base, False) == (True, "type+id")
    assert L.next_ok(dict(base, move_pred_id=1), False) == (False, "type+id")
    assert L.next_ok(dict(base, pred_type="STOP"), False) == (False, "type")
    rot = {"expert_type": "ROTATION", "pred_type": "ROTATION", "rot_tf_int_match": False, "rot_abs_delta": 12.0,
           "rot_greedy_text": "102.0, 0"}
    assert L.next_ok(rot, True) == (True, "greedy15") and L.next_ok(rot, False) == (False, "tf_int")
    assert L.next_ok(dict(rot, rot_abs_delta=None, rot_greedy_text="x"), True) == (False, "greedy15")
    assert L.next_ok({"pred_type": None}, True) == (None, None)


def test_auroc_bins_wilson():
    assert L.auroc([0.9, 0.8], [0.1, 0.2, 0.3]) == 1.0
    assert L.auroc([0.1], [0.9]) == 0.0
    assert L.auroc([0.5, 0.5], [0.5]) == 0.5
    assert L.auroc([], [0.1]) is None
    assert [L.stop_bin(x) for x in (0.0, 0.49, 0.5, 0.8, 0.95, 1.0, 1.05, 1.06)] == \
        ["[0,0.5)", "[0,0.5)", "[0.5,0.8)", "[0.8,0.95)", "[0.95,1.05]", "[0.95,1.05]", "[0.95,1.05]", ">1.05"]
    r = L.rate(8, 85)
    assert r["k"] == 8 and r["rate"] == 0.0941 and r["wilson95"] == [0.0485, 0.1749]  # = results.json of the pilot
    assert L.rate(0, 0)["rate"] is None


def _row(demo, turn, et, pt, p_stop, final=False, kind="nav", node="straight", walked=0.0, stated=100.0, nok=None,
         n_reply=16, ce=0.5, n_legal=2, eid=1):
    p = {"MOVE": (1 - p_stop) * 0.9, "ROTATION": (1 - p_stop) * 0.1, "STOP": p_stop}
    return {"demo": demo, "turn": turn, "kind": kind, "expert_type": et, "expert_id": eid if et == "MOVE" else None,
            "pred_type": pt, "p_type": p, "tok_ok": True, "type_resolved": True, "is_final": final, "node": node,
            "walked_m": walked, "stated_m": stated, "next_ok": (pt == et) if nok is None else nok, "next_rule": "type",
            "n_reply_tokens": n_reply, "ce_reply_sum": ce, "ce_decision_sum": 0.1, "n_decision_tokens": 1,
            "ce_type": 0.1, "n_legal": n_legal, "move_pred_id": 1 if et == "MOVE" else None, "move_id_flag": False}


def test_summarize_rows_numbers():
    rows = [_row("a", 0, "MOVE", "MOVE", 0.1, walked=0), _row("a", 1, "MOVE", "MOVE", 0.2, walked=60),
            _row("a", 2, "STOP", "MOVE", 0.4, final=True, node="stop", walked=100),
            _row("b", 0, "ROTATION", "MOVE", 0.05, node="turn", kind="vis", stated=None),
            _row("b", 1, "STOP", "STOP", 0.7, final=True, node="stop", kind="vis", stated=None)]
    s = L.summarize_rows(rows)
    assert s["n_turns"] == 5 and s["n_demos"] == 2
    assert s["type_acc"]["k"] == 3 and s["type_acc"]["n"] == 5
    assert s["recall"]["STOP"]["k"] == 1 and s["recall"]["STOP"]["n"] == 2 and s["recall"]["ROTATION"]["k"] == 0
    assert s["stop_precision"]["k"] == 1 and s["stop_precision"]["n"] == 1
    assert s["next_exact_turn"]["n"] == 1 and s["next_exact_straight"]["k"] == 2
    assert s["stop_auroc"] == 1.0  # 0.4, 0.7 above every non-STOP
    assert s["greedy_stop_recall_final"]["k"] == 1 and s["greedy_stop_recall_final"]["n"] == 2
    b = s["nav_p_stop_by_walked_over_stated"]
    assert b["[0,0.5)"]["n"] == 1 and b["[0.5,0.8)"]["n"] == 1 and b["[0.95,1.05]"]["expert_stop_rate"] == 1.0
    rule = s["nav_stop_detection"]["rule_walked_ge_stated"]
    assert rule["n"] == 3 and rule["acc"] == 1.0 and rule["stop_recall"] == 1.0
    assert s["nav_stop_detection"]["model_argmax"]["stop_recall"] == 0.0
    # trainer weighting: per demo sum CE / reply tokens, averaged over demos
    assert abs(s["reply_ce_trainer_weighting"] - 0.5 / 16) < 1e-9
    m = s["marginal_reference"]
    assert m["majority_type"] == "MOVE" and m["majority_type_acc"] == 0.4 and m["always_MOVE1_exact"] == 0.4
    # type-only cost of demo a: -log .4 -log .4 -log .4 over 48 reply tokens; demo b: -log .2 -log .4 over 32
    want = ((-3 * math.log(0.4)) / 48 + (-math.log(0.2) - math.log(0.4)) / 32) / 2
    assert abs(m["type_only_per_reply_token"] - round(want, 5)) < 1e-5
    want_ids = ((-3 * math.log(0.4) + 2 * math.log(2)) / 48 + (-math.log(0.2) - math.log(0.4)) / 32) / 2
    assert abs(m["type_uniform_id_per_reply_token"] - round(want_ids, 5)) < 1e-5  # 2 legal ids per MOVE turn
    assert m["type_emp_id_per_reply_token"] == m["type_only_per_reply_token"]  # every MOVE is MOVE(1): no id cost
    rows2 = rows + [_row("c", 0, "MOVE", "MOVE", 0.1, eid=3, n_legal=4)]
    m2 = L.summarize_rows(rows2)["marginal_reference"]
    # ids 1, 1, 3 among the MOVE turns: extra -log(2/3) twice and -log(1/3) once, over 6 turns
    want = (2 * math.log(1.5) + math.log(3)) / 6
    assert abs(m2["type_emp_id_nats_per_turn"] - m2["type_only_nats_per_turn"] - want) < 1e-4
    sub = L.summarize_set(rows, {"only_a": {"a"}})
    assert list(sub) == ["all", "nav", "vis", "only_a"] and sub["only_a"]["n_demos"] == 1 and sub["vis"]["n_turns"] == 2


def test_part_writer_resume_and_set_done(workdir):
    out = _sub(workdir, "tf")
    prov = {"adapter_sha256": "abc", "ids_sha256": "ids1", "set": "VAL56"}
    w = L.PartWriter(out, "raw", "VAL56", prov)
    w.append("d1", [{"demo": "d1", "turn": 0}, {"demo": "d1", "turn": 1}])
    w.append("d2", [{"demo": "d2", "turn": 0}])
    w._f.write(json.dumps({"demo": "d3", "turn": 0}) + "\n{\"demo\": \"d3\", \"tu")  # cut mid-demo, torn line
    w.close()
    w2 = L.PartWriter(out, "raw", "VAL56", prov)
    assert list(w2.done) == ["d1", "d2"]
    w2.append("d3", [{"demo": "d3", "turn": 0}])
    w2.close()
    assert not L.set_done(out, "raw", "VAL56", prov)
    raises(RuntimeError, L.PartWriter(out, "raw", "VAL56", prov).finish, ["d1", "d2", "d3", "d4"])
    f = L.PartWriter(out, "raw", "VAL56", prov).finish(["d2", "d1", "d3"])
    rows = L.read_jsonl(f)
    assert [(r["demo"], r["turn"]) for r in rows] == [("d2", 0), ("d1", 0), ("d1", 1), ("d3", 0)]
    assert L.set_done(out, "raw", "VAL56", prov) and not L.set_done(out, "raw", "VAL56", dict(prov, adapter_sha256="x"))
    assert not (out / "tf_raw_VAL56.jsonl.part").exists() and not (out / "tf_raw_VAL56.part.json").exists()
    w3 = L.PartWriter(out, "raw", "VAL56", dict(prov, ids_sha256="other"))  # another set: the old part is dropped
    assert not w3.done
    w3.close()


def test_parse_adapters(workdir):
    d = _sub(workdir, "ad")
    (d / "adapter.pt").write_bytes(b"x")
    a = L.parse_adapters(["483A=" + str(d), "raw="])
    assert list(a) == ["483A", "raw"] and a["raw"] is None and a["483A"] == d
    raises(SystemExit, L.parse_adapters, ["raw=" + str(d)])
    raises(SystemExit, L.parse_adapters, ["50="])
    raises(SystemExit, L.parse_adapters, ["50=" + str(d / "nope")])
    raises(SystemExit, L.parse_adapters, ["raw=", "raw="])
    raises(SystemExit, L.parse_adapters, [])


def test_refusals_under_the_pilot_tree(workdir):
    raises(SystemExit, L.refuse_pilot, L.PILOT_Q3B / "anything", match="read only")
    raises(SystemExit, L.refuse_pilot, L.PILOT_ROOT)
    L.refuse_pilot(L.X_ROOT)
    L.refuse_pilot(workdir)
    assert not L.is_under(L.X_ROOT, L.PILOT_ROOT)
    raises(SystemExit, L.cmd_eval, ["--model", "Q3B", "--run-name", "x", "--out-root", str(L.PILOT_Q3B / "evals")])
    raises(SystemExit, L.analyze, L.PILOT_Q3B / "x")
    raises(SystemExit, L.run_floors, "s2val", L.PILOT_Q3B / "f.json", workdir, workdir)
    raises(SystemExit, L.run_floors, "s2val", paths.WB_BASELINES / "f.json", workdir, workdir)


# ------------------------------------------------------------------------------------------ s2in50 + run_vpd
def _fake_split(root: Path, n=6) -> tuple:
    stems = [f"nav_{i:04d}_t" if i % 2 == 0 else f"vis_{i:04d}_t" for i in range(n)]
    tasks = root / "s2train" / "tasks"
    tasks.mkdir(parents=True)
    doc = {"splits": {"train_rest": stems[:-1], "s2train": stems}, "tasks": {}}
    for s in stems:
        body = json.dumps({"task_id": s, "visual_path": []}).encode()
        (tasks / f"{s}.json").write_bytes(body)
        doc["tasks"][s] = {"file": f"{s}.json", "sha256": L.sha256_bytes(body), "kind": s.split("_")[0]}
    (root / "split_s2.json").write_text(json.dumps(doc))
    tier = root / "demos_50.jsonl"
    tier.write_text("".join(json.dumps({"id": s}) + "\n" for s in stems[:4]))
    return stems, tier


def test_insample_build_verify_and_refuse(workdir):
    root = _sub(workdir, "split")
    stems, tier = _fake_split(root)
    dest = root / "s2in50"
    r = L.build_insample(dest, tier, root)
    assert r["state"] == "created" and r["nav"] == 2 and r["vis"] == 2
    sel = json.loads((dest / "selection.json").read_text())
    assert sel["ids"] == stems[:4] and sel["exploratory"] and set(sel["files"]) == {f"{s}.json" for s in stems[:4]}
    assert sorted(p.name for p in (dest / "tasks").glob("*.json")) == sorted(f"{s}.json" for s in stems[:4])
    assert L.build_insample(dest, tier, root)["state"] == "verified"
    assert not list(root.glob(".s2in50.tmp-*"))
    (dest / "tasks" / f"{stems[0]}.json").write_text("{}")
    raises(SystemExit, L.build_insample, dest, tier, root, match="differs")
    bad = root / "bad.jsonl"
    bad.write_text(json.dumps({"id": stems[-1]}) + "\n")  # not in train_rest
    raises(SystemExit, L.build_insample, root / "other", bad, root, match="train_rest")
    x = root / "X"
    rep = L.setup(x, root / "s2in50b", tier, root)
    assert rep["s2in50"]["state"] == "created" and all((x / d).is_dir() for d in ("adapters", "tf", "evals"))


def test_register_task_sets_runtime_only(workdir):
    try:
        from vpd.walkerbench import run_vpd as R
        from vpd.walkerbench import pilot as P
    except ImportError as e:  # starVLA has no eval_external deps
        skip(f"run_vpd not importable here: {e}")
    before = dict(R.TASK_SETS)
    d = _sub(workdir, "ev") / "e1"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"adapter": {"adapter_sha256": "zz"}}))
    ad = _sub(workdir, "ad")
    (ad / "adapter.pt").write_bytes(b"w")
    ok_before = (P.eval_check(d), P.eval_check(d, ad))
    try:
        R.TASK_SETS.pop("s2in50", None)
        e = raises(SystemExit, R.main, ["--rescore", "--run-dir", str(d / "nope"), "--task-set", "s2in50"])
        assert e.code == 2  # argparse: invalid choice without the registration
        L.register_task_sets(R)
        assert R.TASK_SETS["s2in50"] == L.S2IN50_TASKS
        assert {k: v for k, v in R.TASK_SETS.items() if k != "s2in50"} == {k: v for k, v in before.items()
                                                                            if k != "s2in50"}
        e = raises(SystemExit, R.main, ["--rescore", "--run-dir", str(d / "nope"), "--task-set", "s2in50"])
        assert "no results.json" in str(e.code)  # parsed; failed later for the missing run
        assert R.tasks_source("s2test") == json.loads((paths.WB_S2TEST_TASKS.parent / "selection.json").read_text())
        assert (P.eval_check(d), P.eval_check(d, ad)) == ok_before  # the pilot's checks never read TASK_SETS
        R.TASK_SETS["s2in50"] = Path("/elsewhere")
        raises(SystemExit, L.register_task_sets, R, match="expected")
    finally:
        R.TASK_SETS.clear()
        R.TASK_SETS.update(before)


def test_init_check(workdir):
    d = _sub(workdir, "init")
    (d / "adapter.pt").write_bytes(b"not the pilot 0W")
    (d / "meta.json").write_text(json.dumps({"code_sha256": L.CODE_SHA_PREFIX + "0" * 52}))
    probs, facts = L.init_check(d)
    assert any("does not start with 5cdbfd4366a3" in p for p in probs)
    probs, facts = L.init_check(d, expect_sha=facts["init_sha256"][:12])
    from vpd.train import n2_train_vpd as T

    cur = T.code_fingerprint()["code_sha256"]
    assert facts["trainer_code_sha256"] == cur
    assert (probs == []) == cur.startswith(L.CODE_SHA_PREFIX)  # trainer unchanged since the pilot -> no problem
    (d / "meta.json").write_text(json.dumps({"code_sha256": "ffff"}))
    assert any("meta.json code" in p for p in L.init_check(d, expect_sha=facts["init_sha256"][:12])[0])


def test_expected_outputs():
    a, b = L.expected_outputs("A"), L.expected_outputs("B")
    assert len([p for p in a if p.name == "results.json"]) == 12 and len([p for p in b if p.name == "results.json"]) == 3
    assert L.X_ROOT / "adapters" / "483A" / "adapter.pt" in a and L.X_ROOT / "tf" / "merge_parity.json" in a
    assert L.X_ROOT / "evals" / "explore__Q3B__raw__s2test__1280__off" / "results.json" in a
    assert L.X_ROOT / "evals" / "explore__Q3B__483A__s2test__640__empty__rep2" / "results.json" in a
    assert all(not L.is_under(p, L.PILOT_ROOT) for p in a + b)
    assert sum(1 for p in a if p.name.startswith("tf_") and p.suffix == ".jsonl") == 15
    assert L.X_ROOT / "tf" / "tf_483B_TR283.done.json" in b


# ------------------------------------------------------------------------------------------ payload
_KNOBS = ("VPD_THREADS", "VPD_JOB_DEADLINE_S", "VPD_MIN_STEP_S", "VPD_MERGE_DIR", "VPD_RERUN_STALE",
          "VPD_RERUN_INVALID", "VPD_DRYRUN", "RUN", "VPD_X483_ROOT", "VPD_X483_PILOT")


def _bash(args, env_extra, cwd):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update(env_extra)
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env, cwd=str(cwd), timeout=300)


def _dry(out: str):
    return [ln[len("[dry] "):] for ln in out.splitlines() if ln.startswith("[dry] ")]


def _kinds(lines):
    out = []
    for ln in lines:
        if "n2_train_vpd.py" in ln:
            out.append("train_" + ln.split("--out ")[1].split()[0].rsplit("/", 1)[1])
        elif "merge.py" in ln:
            out.append("merge_" + ln.split("--adapter ")[1].split()[0].rsplit("/", 1)[1])
        elif "rm -rf" in ln:
            out.append("rm_" + ln.rsplit("__", 1)[1])
        elif "learn483 eval" in ln:
            out.append("eval_" + ln.split("--run-name explore__Q3B__")[1].split()[0])
        elif "learn483 tf-parity" in ln:
            out.append("parity")
        elif "learn483 tf " in ln:
            out.append("tf:" + ",".join(a.split("=")[0] for a in ln.split("--adapter ")[1:]))
        elif "learn483 floors" in ln:
            out.append("floors_" + ln.split("--task-set ")[1].split()[0])
        elif "learn483 setup" in ln:
            out.append("setup")
        elif "learn483 init-check" in ln:
            out.append("init_check")
    return out


def _env(base: Path, **extra):
    env = {"VPD_DRYRUN": "1", "VPD_MERGE_DIR": "/dev/shm/vpd_merged", "VPD_X483_ROOT": str(base / "X"),
           "VPD_X483_PILOT": str(base / "P")}
    env.update(extra)
    return env


def test_payload_bash_syntax():
    res = subprocess.run(["bash", "-n", str(PAYLOAD)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_payload_dryrun_arm_a(workdir):
    base = _sub(workdir, "pa")
    run = base / "jobs" / "wb-x483a-202611100000"
    res = _bash([str(PAYLOAD), "A", str(run)], _env(base), base)
    assert res.returncode == 0, res.stderr + res.stdout
    lines = _dry(res.stdout)
    want = ["setup", "floors_s2val", "floors_s2in50", "init_check", "train_483A",
            "rm_483A", "merge_483A", "eval_483A__s2test__640__empty", "eval_483A__s2test__640__empty__blind",
            "eval_483A__s2val__640__empty", "eval_483A__s2in50__640__empty", "eval_483A__s2test__640__empty__rep2",
            "parity", "rm_483A", "tf:483A,raw,0W,50,200",
            "rm_200", "merge_200", "eval_200__s2val__640__empty", "eval_200__s2in50__640__empty", "rm_200",
            "rm_50", "merge_50", "eval_50__s2in50__640__empty", "rm_50",
            "eval_raw__s2val__640__empty", "eval_raw__s2in50__640__empty", "eval_raw__s2test__640__off",
            "eval_raw__s2test__1280__off"]
    assert _kinds(lines) == want
    X, P = base / "X", base / "P"
    tr = next(ln for ln in lines if "n2_train_vpd.py" in ln)
    assert tr.endswith(f"vpd/train/n2_train_vpd.py --start Q3B --data {paths.WB_TIERS}/seed1001/demos_all.jsonl "
                       f"--init-adapter {P}/adapters/0W --out {X}/adapters/483A --epochs auto --seed 1001 "
                       f"--phase tier483 --threads 2 --num-workers 3")
    assert str(paths.PY_TRAIN) in tr
    ev = {ln.split("--run-name ")[1].split()[0]: ln for ln in lines if "learn483 eval" in ln}
    names = [L.eval_name(m, s, r, mo, v) for (_, m, s, r, mo, v) in L.ARM_EVALS["A"]]
    assert sorted(ev) == sorted(names)
    x1 = ev["explore__Q3B__483A__s2test__640__empty"]
    for frag in ("--model Q3B --task-set s2test --render-size 640x400 --constrained empty --max-tokens 512 "
                 "--history-turns 0 --workers 8 --temperature 0", f"--out-root {X}/evals",
                 "--model-dir /dev/shm/vpd_merged/x483__Q3B__483A", f"--adapter-meta {X}/adapters/483A/meta.json"):
        assert frag in x1, (frag, x1)
    assert str(paths.PY_Q3) in x1 and "--blind" not in x1
    assert "--blind" in ev["explore__Q3B__483A__s2test__640__empty__blind"]
    assert "--workers 4" in ev["explore__Q3B__483A__s2test__640__empty__rep2"]
    x10, x11 = ev["explore__Q3B__raw__s2test__640__off"], ev["explore__Q3B__raw__s2test__1280__off"]
    assert "--constrained off --max-tokens 1024" in x10 and "--render-size 640x400" in x10 and "--model-dir" not in x10
    assert "--render-size 1280x800 --constrained off --max-tokens 2048" in x11
    assert f"--adapter-meta {P}/adapters/200/meta.json" in ev["explore__Q3B__200__s2val__640__empty"]
    tf = next(ln for ln in lines if "learn483 tf " in ln)
    assert f"tf --out {X}/tf --threads 2 --num-workers 3 --adapter 483A={X}/adapters/483A --adapter raw= " \
           f"--adapter 0W={P}/adapters/0W --adapter 50={P}/adapters/50 --adapter 200={P}/adapters/200" in tf
    par = next(ln for ln in lines if "tf-parity" in ln)
    assert f"--merged /dev/shm/vpd_merged/x483__Q3B__483A --adapter {X}/adapters/483A --out {X}/tf/merge_parity.json" in par
    fl = next(ln for ln in lines if "floors --task-set s2val" in ln)
    assert f"--out {X}/floors_s2val.json --logs {X}/floors_logs/s2val --work {run}/floors_work" in fl
    assert f"adapter-ok {P}/adapters/50 --init {P}/adapters/0W" in res.stdout
    assert f"adapter-ok {X}/adapters/483A --init {P}/adapters/0W" in res.stdout
    gpu = [ln for ln in lines if any(k in ln for k in ("n2_train_vpd", "merge.py", "learn483 eval", "learn483 tf",
                                                         "floors"))]
    assert gpu and all(ln.startswith("timeout -k 30 ") and int(ln.split()[3]) <= 15720 for ln in gpu)
    assert "deadline=15720s" in res.stdout and "EXPLORATORY" in res.stdout
    steps = [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]
    assert all(s["rc"] == 0 for s in steps) and not (run / "DONE").exists()
    assert not (base / "X").exists()  # a dry run writes only RUN/steps.jsonl and RUN/node.txt
    assert sorted(p.name for p in run.iterdir()) == ["df_end.txt", "node.txt", "steps.jsonl"]
    assert "runs/pilot" not in " ".join(ln for ln in lines if "--out " in ln and "n2_train" in ln)


def test_payload_dryrun_arm_b(workdir):
    base = _sub(workdir, "pb")
    run = base / "jobs" / "wb-x483b-202611100000"
    res = _bash([str(PAYLOAD), "B", str(run)], _env(base), base)
    assert res.returncode == 0, res.stderr + res.stdout
    lines = _dry(res.stdout)
    assert _kinds(lines) == ["setup", "init_check", "train_483B", "rm_483B", "merge_483B",
                             "eval_483B__s2test__640__empty", "eval_483B__s2val__640__empty",
                             "eval_483B__s2in50__640__empty", "rm_483B", "tf:483B"]
    tr = next(ln for ln in lines if "n2_train_vpd.py" in ln)
    assert f"--out {base}/X/adapters/483B --epochs 10 --seed 1001 --phase tier483" in tr
    assert "floors" not in res.stdout.split("[wb_learn483] VPD_MERGE_DIR", 1)[1]


def test_payload_soft_deadline_and_refusals(workdir):
    base = _sub(workdir, "pd")
    run = base / "jobs" / "wb-x483a-202611100001"
    res = _bash([str(PAYLOAD), "A", str(run)], _env(base, VPD_JOB_DEADLINE_S="30"), base)
    assert res.returncode == 1, res.stderr + res.stdout
    lines = _dry(res.stdout)
    assert not any("n2_train_vpd.py" in ln or "merge.py" in ln or "learn483 eval" in ln or "learn483 tf" in ln
                   for ln in lines)
    names = [json.loads(s)["name"] for s in (run / "steps.jsonl").read_text().splitlines()]
    assert {"floors_s2val:no_time", "train_483A:no_time", "eval_raw_s2val_640_empty:no_time",
            "eval_raw_s2test_1280_off:no_time"} <= set(names)
    assert "merge_483A:no_adapter" in names and (run / "df_end.txt").exists() and not (run / "DONE").exists()
    for bad in ("0", "abc"):
        r = _bash([str(PAYLOAD), "A", str(run)], _env(base, VPD_JOB_DEADLINE_S=bad), base)
        assert r.returncode == 2 and "VPD_JOB_DEADLINE_S" in r.stderr
    r = _bash([str(PAYLOAD), "C", str(run)], _env(base), base)
    assert r.returncode == 2 and "usage" in r.stderr
    r = _bash([str(PAYLOAD), "A"], _env(base, RUN=""), base)
    assert r.returncode == 2
    r = _bash([str(PAYLOAD), "A", str(run)], _env(base, VPD_X483_ROOT=str(paths.WB_RUNS / "pilot" / "Q3B" / "x")), base)
    assert r.returncode == 2 and "refusing" in r.stderr


# ------------------------------------------------------------------------------------------ analysis
def _log_line(event, **k):
    return json.dumps(dict(event=event, **k))


def _state(pano, coords, heading=0.0):
    lat, lng = coords[pano]
    return {"pano_id": pano, "heading": heading, "pitch": 0, "lat": lat, "lng": lng}


def _write_log(path: Path, stem: str, steps, coords):
    """steps: [(pano, action dict, available moves)]."""
    lines = [_log_line("session_start", task_id=stem, initial_state=_state(steps[0][0], coords))]
    for i, (p, act, moves) in enumerate(steps):
        lines.append(_log_line("action", step=i + 1, state=_state(p, coords), action=act, available_moves=moves,
                               reason=""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _mv(i, *ids):
    return {"type": "move", "move_id": i}, [{"id": j, "direction": "front", "distance": 11.1} for j in ids]


def test_analyze_synthetic(workdir):
    base = _sub(workdir, "an")
    coords = {f"p{i}": (0.0, 0.0001 * i) for i in range(6)}  # ~11.1 m apart on the equator
    coords["q1"] = (0.0001, 0.0)
    db = base / "cache.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE locations (pano_id TEXT PRIMARY KEY, lat REAL, lng REAL)")
    con.execute("CREATE TABLE metadata (pano_id TEXT PRIMARY KEY, lat REAL, lng REAL)")
    con.executemany("INSERT INTO locations VALUES (?, ?, ?)", [(k, a, b) for k, (a, b) in coords.items()])
    con.commit()
    con.close()
    tdir = base / "tasks"
    tdir.mkdir()
    nav = {"spawn_point": "p0", "target_pano_ids": ["p3"], "visual_path": [{"pano_id": f"p{i}"} for i in range(4)],
           "ground_truth": {"optimal_distance_meters": 33}}
    vis = {"spawn_point": "p0", "target_pano_ids": ["q1"], "visual_path": [{"pano_id": "p0"}, {"pano_id": "q1"}],
           "ground_truth": {"optimal_distance_meters": 11}}
    (tdir / "nav_0001.json").write_text(json.dumps(nav))
    (tdir / "vis_0001.json").write_text(json.dumps(vis))
    # model: nav walks p0..p5 past the target and hits the cap (6 = round(1.5 * 4)); vis stops at once (wrong)
    ev = base / "pilot" / "evals" / "pilot__Q3B__200__640__empty"
    logs = ev / "WalkerBench" / "logs"
    _write_log(logs / "a.jsonl", "nav_0001", [(f"p{i}",) + _mv(1, 1, 2) for i in range(5)] +
               [("p5",) + _mv(1, 1)], coords)
    _write_log(logs / "b.jsonl", "vis_0001", [("p0", {"type": "stop", "answer": "ARRIVED"}, [])], coords)
    (ev / "per_task.jsonl").write_text(
        json.dumps({"stem": "nav_0001", "success": 0, "oracle": 1, "steps": 6, "log": "a.jsonl"}) + "\n" +
        json.dumps({"stem": "vis_0001", "success": 0, "oracle": 0, "steps": 1, "log": "b.jsonl"}) + "\n")
    (ev / "results.json").write_text(json.dumps({"adapter": {"adapter_sha256": "s200"}, "blind": False}))
    # RM floor logs: nav rotates at p1 and moves with id 2 at p2
    blogs = base / "blogs"
    _write_log(blogs / "RM" / "r.jsonl", "nav_0001",
               [("p0",) + _mv(1, 1), ("p1", {"type": "rotation", "heading": 90}, []), ("p1",) + _mv(1, 1),
                ("p2",) + _mv(2, 1, 2), ("p3", {"type": "stop"}, [])], coords)
    (base / "baselines.json").write_text(json.dumps({"logs": str(blogs), "policies": {
        "RM": {"per_task": {"nav_0001": {"success": 1, "steps": 5}, "vis_0001": {"success": 1}}},
        "ODO": {"per_task": {"nav_0001": {"success": 1}}}, "FWD": {"per_task": {}}, "STOP0": {"per_task": {}}}}))
    old = (L.TASK_DIRS, L.BASELINES_S2TEST)
    try:
        L.TASK_DIRS = {"s2test": tdir, "s2val": base / "none", "s2in50": base / "none"}
        L.BASELINES_S2TEST = base / "baselines.json"
        doc = L.analyze(base / "X", base / "pilot", db)
    finally:
        L.TASK_DIRS, L.BASELINES_S2TEST = old
    s = doc["sets"]["s2test"]
    assert s["missing_coords"] == 0 and s["trivial_tasks_d0_le15"] == ["vis_0001"]  # q1 is 11.1 m from p0
    e = s["evals"]["200"]
    assert e["state"] == "ok" and e["adapter_sha256"] == "s200" and s["evals"]["483A"]["state"] == "not run"
    nv, vs, al = e["summary"]["nav"], e["summary"]["vis"], e["summary"]["all"]
    assert nv["success"]["k"] == 0 and nv["oracle"]["k"] == 1 and nv["cap_hits"] == 1 and nv["mean_steps"] == 6
    assert vs["ended_stop"]["k"] == 1 and al["ended_stop"]["k"] == 1
    # cf: walked >= 33 m first at p3 (3 x 11.1 m) -> on the target
    assert nv["cf_own_path_stop_at_walked_ge_stated"]["k"] == 1
    assert nv["first_arrival_target_pano_next"] == {"M": 1}
    # node decisions on the nav path: p0 straight ok, p1 rot-node ok, p2 k-node ok (moved to p3)
    assert nv["node_turn_rot"] == L.rate(1, 1) and nv["node_turn_any"] == L.rate(2, 2)
    assert nv["node_straight_rot"]["n"] == 2 and nv["node_straight_any"]["n"] == 1
    assert nv["fwd_step_agreement"]["k"] == 6 and nv["p_stop_within15"]["n"] == 3  # p2, p3 (0 m), p4 (11.1 m)
    assert nv["p_stop_within15"]["k"] == 0 and nv["p_stop_beyond15"]["n"] == 3
    assert vs["nontrivial_success"]["n"] == 0  # the vis task is trivial
    assert s["floors"]["RM"]["nav"]["k"] == 1 and s["floors"]["RM"]["behaviour"]["nav"]["ended_stop"]["k"] == 1
    assert doc["sets"]["s2val"]["missing"].startswith("no task dir")
    md = (base / "X" / "REPORT_EXPLORATORY.md").read_text(encoding="utf-8")
    assert "探索性" in md and "不做判断" in md and "| 200 |" in md and "没有闸门" in md
    assert json.loads((base / "X" / "diag.json").read_text())["exploratory"] is True


# ------------------------------------------------------------------------------------------ floors (real sandbox)
def test_floors_two_real_s2val_tasks(workdir):
    tasks = sorted(p.stem for p in paths.WB_S2VAL_TASKS.glob("*.json"))
    if not tasks or not paths.REPO_WB.is_dir():
        skip("no s2val tasks / WalkerBench checkout")
    import importlib.util

    if importlib.util.find_spec("dotenv") is None:
        skip("not the WalkerBench env (no dotenv): the payload runs floors with PY_EVAL")
    stems = [next(t for t in tasks if t.startswith("nav")), next(t for t in tasks if t.startswith("vis"))]
    base = _sub(workdir, "fl")
    cwd = os.getcwd()
    try:
        doc = L.run_floors("s2val", base / "floors_s2val.json", base / "logs", base / "work", stems)
    finally:
        os.chdir(cwd)
    assert doc["subset"] and doc["n_tasks"] == 2 and set(doc["policies"]) == {"STOP0", "FWD", "ODO", "RM"}
    assert set(doc["floors"]) == {"nav_ODO", "vis_FWD", "pooled_F"} and doc["floors"]["nav_ODO"]["n"] == 1
    for m in ("STOP0", "FWD", "ODO", "RM"):
        assert set(doc["policies"][m]["per_task"]) == set(stems)
        assert len(list((Path(doc["logs"]) / m).glob("*.jsonl"))) == 2
    assert doc["policies"]["RM"]["summary"]["all"]["k"] >= 1
    assert not any((base / "work").iterdir())  # the sandbox copy is removed
    assert json.loads((base / "floors_s2val.json").read_text())["exploratory"] is True


# ------------------------------------------------------------------------------------------ TF probe (tiny model)
def _torch():
    try:
        import torch
    except ImportError:
        skip("no torch")
    torch.set_num_threads(1)
    return torch


def _tiny_config():
    from transformers import AutoConfig

    from vpd import starts

    cfg = AutoConfig.from_pretrained(starts.resolve("Q3S")["path"])
    tc, vc = cfg.text_config, cfg.vision_config
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16).items():
        setattr(tc, k, v)
    tc.rope_scaling = {**(tc.rope_scaling or {}), "mrope_section": [2, 3, 3]}
    for k, v in dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                     deepstack_visual_indexes=[1]).items():
        setattr(vc, k, v)
    cfg.tie_word_embeddings = False
    return cfg


def _tiny_model(dtype=None, seed=0):
    torch = _torch()
    try:
        import transformers
    except ImportError:
        skip("no transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        skip("transformers without Qwen3-VL")
    torch.manual_seed(seed)
    return transformers.Qwen3VLForConditionalGeneration(_tiny_config()).to(dtype or torch.float32).eval()


def _save_tiny(model, d: Path) -> Path:
    from safetensors.torch import save_file

    d.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(str(d))
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, str(d / "model.safetensors"),
              metadata={"format": "pt"})
    return d


def _random_adapter(model, d: Path, seed=3, std=0.05) -> Path:
    """A non-zero adapter (random A and B) of the wrapped tiny model, saved as n2_train_vpd saves one."""
    torch = _torch()
    from vpd.train import n2_train_vpd as T

    g = torch.Generator().manual_seed(seed)
    bank = L.LoRABank(model)
    with torch.no_grad():
        for m in bank.mods.values():
            m.lora_A.copy_(torch.randn(m.lora_A.shape, generator=g) * std)
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * std)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(T.adapter_state(model), d / "adapter.pt")
    (d / "meta.json").write_text(json.dumps({"rank": L.RANK, "alpha": L.ALPHA, "start": "Q3S", "phase": "test"}))
    return d


def _processor():
    from vpd import starts

    try:
        return starts.load_processor("Q3S")
    except Exception as e:  # noqa: BLE001
        skip(f"Q3S processor not readable: {e}")


def test_tf_rows_tiny_model(workdir):
    torch = _torch()
    from vpd.train import n2_train_vpd as T

    proc = _processor()
    base = _sub(workdir, "tiny")
    rec = make_demo("nav_tiny", base / "img")
    model = _tiny_model()
    L.wrap_for_probe(model)
    bank = L.LoRABank(model)
    bank.add("raw", None, model)
    adir = _random_adapter(model, base / "rnd")
    prov = bank.add("rnd", adir, model)
    assert prov["sha256"] == L.sha256_file(adir / "adapter.pt") and prov["modules"] == len(bank.mods)
    runner = L.TFRunner(model, proc, "cpu")
    assert len(runner.p_move) == 12 and len(runner.p_rot) == 13 and len(runner.digit_ids) == 10
    turns = L.turn_table(rec)
    enc = runner.encode(rec)
    bank.use("raw")
    raw = runner.demo_rows(enc, rec, turns, "VAL56", greedy=True)
    bank.use("rnd")
    rnd = runner.demo_rows(enc, rec, turns, "VAL56", greedy=True)
    for rows in (raw, rnd):
        assert len(rows) == 4 and all(r["tok_ok"] and r["type_resolved"] for r in rows)
        assert all(abs(sum(r["p_type"].values()) - 1) < 1e-6 and r["pred_type"] in L.TYPES for r in rows)
        assert rows[0]["move_pred_id"] == 1 and 1 <= rows[2]["move_pred_id"] <= 3 and rows[2]["move_id_flag"] is False
        assert isinstance(rows[1]["rot_tf_int_match"], bool) and isinstance(rows[1]["rot_greedy_text"], str)
        assert rows[1]["n_decision_tokens"] == 3 and rows[2]["n_decision_tokens"] == 2 and rows[3]["n_decision_tokens"] == 1
        assert [r["n_reply_tokens"] for r in rows] == [16, 23, 16, 17]  # + <|im_end|>; "90.5" = 4 tokens
        assert all(r["next_rule"] is not None for r in rows)
    assert abs(L.demo_ce(raw) - L.demo_ce(rnd)) > 1e-4  # the adapter changes the model
    # the probe's reply CE is the trainer's response-only loss on the same inputs
    labels = enc["labels"]
    inputs = dict(enc["inputs"])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ce_rnd = float(T.response_loss(model, inputs, labels))
    assert abs(ce_rnd - L.demo_ce(rnd)) < 1e-4 * max(1.0, ce_rnd)
    # raw (A = B = 0) is exactly the plain model
    plain = _tiny_model()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ce_plain = float(T.response_loss(plain, inputs, labels))
    assert abs(ce_plain - L.demo_ce(raw)) < 1e-5 * max(1.0, ce_plain)
    s = L.summarize_set(raw)
    assert s["all"]["n_turns"] == 4 and s["all"]["rotation_greedy"]["n"] == 1 and s["nav"]["n_demos"] == 1


def _tiny_demo_files(base: Path) -> tuple:
    recs = [make_demo("nav_v1", base / "img", "nav", "val", seed=1),
            make_demo("vis_v2", base / "img", "vis", "val", acts=["MOVE(1)", 'STOP("ARRIVED")'], seed=2),
            make_demo("nav_r1", base / "img", "nav", "train_rest", seed=3)]
    da = base / "demos_all.jsonl"
    da.write_text("".join(json.dumps(r) + "\n" for r in recs))
    tiers = base / "tiers"
    tiers.mkdir()
    for n in ("demos_200.jsonl", "demos_all.jsonl"):
        (tiers / n).write_text(json.dumps(recs[2]) + "\n")
    return da, tiers


def test_tf_cli_and_merge_parity_tiny(workdir):
    torch = _torch()
    from vpd.train import n2_train_vpd as T

    _processor()
    base = _sub(workdir, "cli")
    da, tiers = _tiny_demo_files(base)
    tiny = _save_tiny(_tiny_model(torch.bfloat16), base / "tiny")
    wrapped = _tiny_model(torch.bfloat16)
    L.wrap_for_probe(wrapped)
    adir = _random_adapter(wrapped, base / "rnd")
    out = base / "tf"
    common = ["--start", "Q3S", "--model-path", str(tiny), "--device", "cpu", "--threads", "1", "--num-workers", "0",
              "--demos-all", str(da), "--tiers", str(tiers)]
    rc = L.main(["tf", "--out", str(out), "--adapter", "raw=", "--adapter", f"rnd={adir}", "--sets", "VAL56,TR200",
                 "--limit", "2"] + common)
    assert rc == 0
    for a in ("raw", "rnd"):
        rows = L.read_jsonl(out / f"tf_{a}_VAL56.jsonl")
        assert [r["demo"] for r in rows] == ["nav_v1"] * 4 + ["vis_v2"] * 2
        assert L.read_json(out / f"tf_{a}_VAL56.done.json")["n_demos"] == 2
        assert len(L.read_jsonl(out / f"tf_{a}_TR200.jsonl")) == 4
        sm = L.read_json(out / f"tf_summary_{a}.json")
        assert sm["complete_sets"] == ["VAL56", "TR200"] and sm["sets"]["VAL56"]["all"]["n_turns"] == 6
        assert sm["sets"]["TR200"]["in_sample_demos"] == 0 and "first50" in sm["sets"]["TR200"]
    assert set(L.read_json(out / "tf_summary.json")["adapters"]) == {"raw", "rnd"}
    # a second run finds every (adapter, set) done and never loads a model (a bad model path would fail)
    bad = list(common)
    bad[bad.index("--model-path") + 1] = str(base / "no_such_model")
    assert L.main(["tf", "--out", str(out), "--adapter", "raw=", "--adapter", f"rnd={adir}", "--sets", "VAL56,TR200",
                   "--limit", "2"] + bad) == 0
    # merge parity: tiny bf16 + the adapter merged in memory vs the LoRA view
    merged_model = _tiny_model(torch.bfloat16)
    T.merge_adapter(merged_model, adir / "adapter.pt")
    merged = _save_tiny(merged_model, base / "merged")
    pout = base / "tf" / "merge_parity.json"
    assert L.main(["tf-parity", "--out", str(pout), "--merged", str(merged), "--adapter", str(adir), "--n", "2"] + common) == 0
    doc = L.read_json(pout)
    assert doc["n"] == 2 and doc["adapter_sha256"] == L.sha256_file(adir / "adapter.pt")
    assert doc["max_abs_diff_ce"] < 0.05 * doc["mean_ce_lora"] and doc["pred_type_agree"]["n"] == 6
    ns = type("A", (), {"file": pout, "adapter": adir})
    assert L.cmd_parity_ok(ns) == 1  # n 2 != PARITY_N 20: the job would re-run it
    doc["n"] = L.PARITY_N
    pout.write_text(json.dumps(doc))
    assert L.cmd_parity_ok(ns) == 0
    refuse = ["tf", "--out", str(L.PILOT_Q3B / "tf"), "--adapter", "raw="] + common
    raises(SystemExit, L.main, refuse, match="read only")


# ------------------------------------------------------------------------------------------ standalone runner
def _run_standalone() -> int:
    import inspect

    d = _make_workdir()
    failed = 0
    try:
        for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
            args = [d] if "workdir" in inspect.signature(fn).parameters else []
            try:
                fn(*args)
                print(f"PASS {name}")
            except Skip as e:
                print(f"SKIP {name}: {e}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
