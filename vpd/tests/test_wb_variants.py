"""Tests of the EXPLORATORY WalkerBench variants (DECISIONS G4, G6): vpd/walkerbench/odometer.py,
vpd/walkerbench/variants.py and vpd/jobs/wb_variants.sh.

Hermetic: fakes for the agent, the trainer outputs and the evals; no GPU, no network, no rendering, no model.  Some
tests read real files read-only and fast (split_s2.json, the s2train task JSONs copied into a temp dir by setup, the
first demos of tiers/seed1001/demos_all.jsonl, the stock prompt code in the WalkerBench checkout); they skip when a
file is missing.  The payload runs with VPD_DRYRUN=1 (it then prints its commands) and, for its failure handling,
non-dry from a temp code tree whose common_env.sh shim sources the real one and then points PY_EVAL / PY_Q3 /
PY_TRAIN at a fake interpreter (a bash script that logs and answers each call) and VPD_OUT at a temp dir.
    cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd && env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 nice -n 19 \
      $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=$SCRATCH/pt vpd/tests/test_wb_variants.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_WT = Path(__file__).resolve().parents[2]
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

from vpd import paths  # noqa: E402
from vpd.walkerbench import learn483 as L  # noqa: E402
from vpd.walkerbench import odometer as O  # noqa: E402
from vpd.walkerbench import run_vpd as R  # noqa: E402
from vpd.walkerbench import variants as V  # noqa: E402
from vpd.walkerbench.action_regex import reply  # noqa: E402

PAYLOAD = Path(__file__).resolve().parents[1] / "jobs" / "wb_variants.sh"


@pytest.fixture(autouse=True)
def fresh_counts():
    R.COUNTS.clear()
    yield
    R.COUNTS.clear()


def need(p: Path) -> Path:
    if not Path(p).exists():
        pytest.skip(f"{p} not present")
    return Path(p)


# ------------------------------------------------------------------------------------------ the stock user text
def stock_text_class():
    """VLNAgent's _build_user_message / _format_heading_compass / _format_pitch, cut out of the real checkout's
    examples/vln_agent.py with ast (the module itself imports the engine, so it is not imported)."""
    src_path = need(paths.REPO_WB / "examples" / "vln_agent.py")
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLNAgent")
    want = ("_build_user_message", "_format_heading_compass", "_format_pitch")
    funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert sorted(f.name for f in funcs) == sorted(want)
    kw = {"type_params": []} if sys.version_info >= (3, 12) else {}
    node = ast.ClassDef(name="StockText", bases=[], keywords=[], body=funcs, decorator_list=[], **kw)
    mod = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    ns = {"Dict": Dict, "Any": Any, "Optional": Optional, "List": List}
    exec(compile(mod, str(src_path), "exec"), ns)  # the original nodes: the template strings stay byte-exact
    return ns["StockText"]


def obs(moves, heading=211.6, pitch=0, task="Navigate to X. Head south for 35m. Total distance: approximately 35 m."):
    return {"task_description": task, "current_image": "/temp_images/s/step_0.jpg", "panorama_url": None,
            "heading": heading, "pitch": pitch, "fov": 90,
            "available_moves": [{"id": i, "direction": d, "distance": dist, "heading": 0.0} for i, d, dist in moves]}


class _Completions:
    def __init__(self, replies):
        self.replies = list(replies)

    def create(self, **k):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)),
                                                        finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3))


def fake_vln(stock):
    """The parts of WalkerBench's VLNAgent the hooks touch, with the REAL stock user text."""

    class FakeVLN(stock):
        REPLIES: list = []
        ACTIONS: list = []
        RESULTS: list = []

        def __init__(self, config=None):
            self.config = SimpleNamespace(max_tokens=16384, max_history_turns=1000, temperature=0.0)
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(self.REPLIES)))
            self.messages = [{"role": "system", "content": "sys"}]
            self.step_count = 0
            self.actions, self.results = list(self.ACTIONS), list(self.RESULTS)

        def get_image_base64(self, image_url):
            return "AAAA"

        def _trim_history(self):
            pass

        def decide_action(self, observation):
            self.messages.append(self._build_user_message(observation))
            self._trim_history()
            r = self.client.chat.completions.create(model="m", messages=self.messages, max_tokens=1, temperature=0)
            self.messages.append({"role": "assistant", "content": r.choices[0].message.content})
            return dict(self.actions.pop(0))

        def execute_action(self, action):
            return self.results.pop(0)

    return FakeVLN


def ok_result():
    return {"success": True, "observation": {"x": 1}, "done": False, "done_reason": None, "error": None}


# an episode: (moves shown, expert action, its reply action text)
EPISODE = [
    ([(1, "front", 9.64), (2, "front-left 31°", 15.5)], {"type": "move", "move_id": 1}, "MOVE(1)"),
    ([(1, "front", 10.0), (2, "back", 9.6)], {"type": "rotation", "heading": 90.0, "pitch": 0}, "ROTATION(90, 0)"),
    ([(1, "right", None), (2, "left-back 120°", 12.25)], {"type": "move", "move_id": 1}, "MOVE(1)"),
    ([(1, "front", 7.05), (2, "back", 12.25)], {"type": "move", "move_id": 2}, "MOVE(2)"),
    ([(1, "front", 101.94)], {"type": "move", "move_id": 1}, "MOVE(1)"),
    ([(1, "back", 3.0)], {"type": "stop", "answer": "ARRIVED"}, 'STOP("ARRIVED")'),
]
EXPECTED_WALKED = [0.0, 9.6, 9.6, 9.6, 21.8, 123.7]  # shown: 9.6, -, 0 (no distance printed), 12.2 (half-even), 101.9


def run_episode(agent_cls, episode=EPISODE, results=None):
    agent_cls.REPLIES = [reply(a) for _, _, a in episode]
    agent_cls.ACTIONS = [act for _, act, _ in episode]
    agent_cls.RESULTS = results or [ok_result() for _ in episode]
    agent = agent_cls()
    texts = []
    for step, (moves, _, _) in enumerate(episode):
        agent.step_count = step
        agent.decide_action(obs(moves))
        texts.append(agent.messages[-2]["content"][0]["text"])
        agent.execute_action(dict(episode[step][1]))
    return agent, texts


def test_odometer_line_equals_demo_rewrite_on_replayed_observations(tmp_path):
    stock = stock_text_class()
    base = fake_vln(stock)
    opts = R.EvalOptions(constrained="empty")
    undo = V.install_odometer(R, "line")
    try:
        cls = R.make_eval_agent(base, opts, tmp_path)
        assert cls.__name__.startswith("Odometer")
        agent, eval_texts = run_episode(cls)
    finally:
        undo()
    assert R.make_eval_agent is not None and "Odometer" not in R.make_eval_agent(base, opts, tmp_path).__name__
    # the same episode without the hook: the stock texts a demo would hold ('<image>' appended, demos.to_messages)
    _, stock_texts = run_episode(R.make_eval_agent(base, opts, tmp_path))
    rec = {"id": "nav_t", "task_type": "nav", "messages": [{"role": "system", "content": "sys"}]}
    for text, (_, _, a) in zip(stock_texts, EPISODE):
        assert sum(ln.startswith("- Pitch:") for ln in text.split("\n")) == 1  # the stock template's unique line
        rec["messages"] += [{"role": "user", "content": text + O.IMAGE}, {"role": "assistant", "content": reply(a)}]
    v1t = O.with_line(rec)
    demo_texts = [m["content"][:-len(O.IMAGE)] for m in v1t["messages"] if m["role"] == "user"]
    assert demo_texts == eval_texts
    got = [float(re.search(r"- Walked so far: (\d+\.\d) m", t).group(1)) for t in eval_texts]
    assert got == EXPECTED_WALKED
    assert agent._odo.walked == 123.7 and R.COUNTS["odometer_lines"] == len(EPISODE)
    # the line sits right after '- Pitch:' and '<image>' stays last
    t = v1t["messages"][3]["content"].split("\n")
    i = next(j for j, ln in enumerate(t) if ln.startswith("- Pitch:"))
    assert t[i + 1] == "- Walked so far: 9.6 m" and v1t["messages"][3]["content"].endswith(O.IMAGE)
    assert [m["content"] for m in v1t["messages"] if m["role"] == "assistant"] == \
           [m["content"] for m in rec["messages"] if m["role"] == "assistant"]
    # V2: the THOUGHT carries the same walked values and passes run_vpd's short regex
    v2 = O.with_thought(rec)
    for t, m, (moves, _, a) in zip(EXPECTED_WALKED, [m for m in v2["messages"] if m["role"] == "assistant"], EPISODE):
        assert m["content"] == reply(a, f"walked {t:.1f} of 35 m")
        assert re.fullmatch(R.step_regex(len(moves), "short", 32), m["content"])
    assert [m["content"] for m in v2["messages"] if m["role"] == "user"] == \
           [m["content"] for m in rec["messages"] if m["role"] == "user"]


def test_failed_move_adds_nothing(tmp_path):
    base = fake_vln(stock_text_class())
    undo = V.install_odometer(R, "line")
    try:
        cls = R.make_eval_agent(base, R.EvalOptions(constrained="off"), tmp_path)
    finally:
        undo()
    ep = [([(1, "front", 9.6)], {"type": "move", "move_id": 1}, "MOVE(1)"),
          ([(1, "front", 5.0)], {"type": "move", "move_id": 9}, "MOVE(9)"),  # not among the moves: crash
          ([(1, "front", 4.0)], {"type": "move", "move_id": 1}, "MOVE(1)")]  # executed but reported failed
    crash = {"success": False, "observation": None, "done": False, "done_reason": None, "error": "Invalid move_id"}
    fail = {"success": False, "observation": {"x": 1}, "done": False, "done_reason": None, "error": "blocked"}
    agent, texts = run_episode(cls, ep, [ok_result(), crash, fail])
    assert agent._odo.walked == 9.6 and R.COUNTS["odometer_moves_added"] == 1 and R.COUNTS["invalid_move_crash"] == 1


def test_odometer_helpers():
    assert O.shown(9.649) == 9.6 and O.shown(None) == 0.0 and O.shown(0) == 0.0 and O.add(0.1, 0.2) == 0.3
    assert O.line(12.0) == "- Walked so far: 12.0 m"
    text = "a\n- Heading: 1°N\n- Pitch: 0°\n\nb"
    assert O.insert_line(text, 3.25).split("\n")[3] == "- Walked so far: 3.2 m"
    with pytest.raises(ValueError):
        O.insert_line("no pitch here", 0.0)
    with pytest.raises(ValueError):
        O.insert_line("- Pitch: 0°\n- Pitch: 1°", 0.0)
    with pytest.raises(ValueError):
        O.insert_line(O.insert_line(text, 1.0), 2.0)
    long = O.thought(99999.9, "99999")
    assert len(long) <= 32 and re.fullmatch(R.step_regex(9, "short", 32), reply("MOVE(9)", long))
    assert O.thought(5.0) == "walked 5.0 m" and O.thought(5.0, "146") == "walked 5.0 of 146 m"
    with pytest.raises(ValueError):
        O.thought(1e30, "1")
    assert O.parse_thought("walked 5.0 of 146 m") == {"walked": 5.0, "stated": "146"}
    assert O.parse_thought("walked 5.0 m") == {"walked": 5.0, "stated": None}
    assert O.parse_thought("walked 5 m") is None and O.parse_thought(None) is None
    t = "x\n\n**Available Moves:**\n  1: front (10.0m)\n  2: right\n  3: front-left 31° (15.5m)\n\n🔄 **ROTATE**"
    assert O.moves_in_text(t) == {1: 10.0, 2: 0.0, 3: 15.5}
    assert O.moves_in_text("x\n\n**Available Moves:**\n\n\n🔄 **ROTATE** - x\n  - Look LEFT: heading=1") == {}
    assert O.moves_in_obs([{"id": 2, "distance": 12.25}, {"id": 1, "distance": None}]) == {2: 12.2, 1: 0.0}
    assert O.stated_total("... Total distance: approximately 146 m.") == "146" and O.stated_total("vis") is None


def test_vis_thought_and_nav_without_stated():
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "T\n- Pitch: 0°\n\n**Available Moves:**\n  1: front (4.0m)\n\nz<image>"},
            {"role": "assistant", "content": reply("MOVE(1)")},
            {"role": "user", "content": "T\n- Pitch: 0°\n\n**Available Moves:**\n  1: front (4.0m)\n\nz<image>"},
            {"role": "assistant", "content": reply('STOP("ARRIVED")')}]
    v = O.with_thought({"id": "vis_x", "task_type": "vis", "messages": msgs})
    assert [json.loads(m["content"])["THOUGHT"] for m in v["messages"] if m["role"] == "assistant"] == \
           ["walked 0.0 m", "walked 4.0 m"]
    n = O.with_thought({"id": "nav_x", "task_type": "nav", "messages": msgs})  # nav without a stated total
    assert json.loads(n["messages"][4]["content"])["THOUGHT"] == "walked 4.0 m"
    bad = [dict(m) for m in msgs]
    bad[2] = {"role": "assistant", "content": reply("MOVE(3)")}
    with pytest.raises(ValueError):
        O.demo_turns(bad)


def test_odometer_on_real_demos_agrees_with_turn_table():
    """The first demos of the 483 tier file: the odometer's walked equals learn483.turn_table's walked_m; the V1T /
    V2 rewrites pass the eval regexes; '- Pitch:' is unique in every user turn."""
    src = need(V.TIER_ALL)
    recs = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
            if len(recs) == 12:
                break
    for rec in recs:
        turns = O.demo_turns(rec["messages"])
        assert [t["walked"] for t in turns] == [round(r["walked_m"], 1) for r in L.turn_table(rec)]
        assert V.check_record(O.with_line(rec), ("empty", "line")) == []
        assert V.check_record(O.with_thought(rec), ("short", "off")) == []
        assert V.check_record(rec, ("empty", "off")) == []
        assert all(sum(ln.startswith("- Pitch:") for ln in m["content"].split("\n")) == 1
                   for m in rec["messages"] if m["role"] == "user")
    assert V.check_record(O.with_line(recs[0]), ("empty", "off"))  # a line where none is expected is flagged


# ------------------------------------------------------------------------------------------ carve
def test_dev_carve_real_split():
    doc = V.split_doc(need(V.SPLIT_S2))
    c = V.dev_carve(doc)
    assert tuple(c["cities"]) == V.DEV_CITIES and len(c["dev"]) == 74 and c["counts"]["dev"] == {
        "total": 74, "nav": 36, "vis": 38}
    assert len(c["pool"]) == 409 and c["counts"]["sel"] == {"total": 131, "nav": 65, "vis": 66}
    assert c["leakage_dev_vs_pool"] == {"routes": 0, "geofences": 0, "panos": 0, "cities": 0}
    assert not set(c["dev"]) & set(doc["splits"]["s2test"]) and not set(c["dev"]) & set(doc["splits"]["s2val"])
    assert set(c["dev"]) | set(c["pool"]) == set(doc["splits"]["train_rest"])
    # independent recount: the cities right after the s2val ones in the md5 order, whole, until >= 60 tasks
    order = [cc for cc in doc["train_city_md5_order"] if cc not in doc["val_cities"]]
    n, k = 0, 0
    while n < 60:
        n += sum(1 for s in doc["splits"]["train_rest"] if doc["tasks"][s]["city"] == order[k])
        k += 1
    assert order[:k] == list(V.DEV_CITIES) and n == 74


def _syn_doc(dev_share_pano=False):
    """A tiny split: val cities vA; md5 order vA, d1, d2, r1; d1+d2 = the dev carve with DEV_MIN_TASKS 3."""
    tasks, routes, rest = {}, {}, []
    spec = [("nav_d1a", "d1", ["p1", "p2"]), ("vis_d1b", "d1", ["p3"]), ("nav_d2a", "d2", ["p4"]),
            ("nav_r1a", "r1", ["p5", "p2" if dev_share_pano else "p6"]), ("vis_r1b", "r1", ["p7"]),
            ("nav_va", "vA", ["p8"]), ("nav_ta", "T", ["p9"])]
    for i, (s, city, path) in enumerate(spec):
        rid = f"r{i}"
        routes[rid] = [path[0], [path[-1]], path]
        split = "val" if city == "vA" else ("test" if city == "T" else "rest")
        tasks[s] = {"file": s + ".json", "sha256": "x", "kind": s[:3], "city": city, "geofence": f"g{city}",
                    "route_id": rid, "split": split}
        if split == "rest":
            rest.append(s)
    return {"train_city_md5_order": ["vA", "d1", "d2", "r1"], "val_cities": ["vA"], "tasks": tasks, "routes": routes,
            "splits": {"train_rest": rest, "s2val": ["nav_va"], "s2test": ["nav_ta"]}}


def _syn_constants(monkeypatch):
    monkeypatch.setattr(V, "DEV_MIN_TASKS", 3)
    monkeypatch.setattr(V, "DEV_CITIES", ("d1", "d2"))
    monkeypatch.setattr(V, "DEV_COUNTS", {"total": 3, "nav": 2, "vis": 1})
    monkeypatch.setattr(V, "SEL_COUNTS", {"total": 4, "nav": 3, "vis": 1})
    monkeypatch.setattr(V, "POOL_N", 2)


def test_dev_carve_synthetic_and_leak_refusal(monkeypatch):
    _syn_constants(monkeypatch)
    c = V.dev_carve(_syn_doc())
    assert c["cities"] == ["d1", "d2"] and c["dev"] == ["nav_d1a", "nav_d2a", "vis_d1b"]
    assert c["pool"] == ["nav_r1a", "vis_r1b"] and c["sel"] == ["nav_d1a", "nav_d2a", "nav_va", "vis_d1b"]
    with pytest.raises(SystemExit, match="shares"):
        V.dev_carve(_syn_doc(dev_share_pano=True))
    assert V.dev_carve(_syn_doc(dev_share_pano=True), strict=False)["leakage_dev_vs_pool"]["panos"] == 1


def test_setup_real_files_into_temp_dirs(tmp_path):
    need(V.SPLIT_S2)
    need(paths.WB_S2TRAIN_TASKS)
    dev, sel, x = tmp_path / "s2dev74", tmp_path / "s2sel", tmp_path / "X"
    rep = V.setup(x, V.SPLIT_S2, dev, sel)
    assert rep["s2dev74"]["state"] == "created" and rep["s2dev74"]["total"] == 74
    assert rep["s2sel"]["total"] == 131 and rep["pool"] == 409
    assert len(list((dev / "tasks").glob("*.json"))) == 74 and len(list((sel / "tasks").glob("*.json"))) == 131
    s = json.loads((sel / "selection.json").read_text())
    assert s["exploratory"] is True and len(s["members"]["s2val"]) == 57 and len(s["members"]["s2dev74"]) == 74
    d = json.loads((dev / "selection.json").read_text())
    assert d["train_pool"]["n"] == 409 and d["leakage_vs_train_pool"]["panos"] == 0
    doc = V.split_doc()
    for f in (dev / "tasks").glob("*.json"):
        assert L.sha256_file(f) == doc["tasks"][f.stem]["sha256"]
    assert V.members(sel)["SEL"] == sorted(s["ids"]) and (x / "EXPLORATORY.md").exists()
    assert V.setup(x, V.SPLIT_S2, dev, sel)["s2sel"]["state"] == "verified"
    f = next((dev / "tasks").glob("*.json"))
    f.write_text(f.read_text() + " ")
    with pytest.raises(SystemExit, match="differs"):
        V.setup(x, V.SPLIT_S2, dev, sel)
    with pytest.raises(SystemExit, match="pilot"):
        V.setup(V.PILOT_ROOT / "x", V.SPLIT_S2, tmp_path / "a", tmp_path / "b")


def test_register_task_sets_runtime_only():
    fake = SimpleNamespace(TASK_SETS={"s2test": Path("t"), "s2val": Path("v")})
    V.register_task_sets(fake)
    assert fake.TASK_SETS["s2sel"] == V.S2SEL_TASKS and fake.TASK_SETS["s2dev74"] == V.S2DEV_TASKS
    fake.TASK_SETS["s2sel"] = Path("elsewhere")
    with pytest.raises(SystemExit, match="expected"):
        V.register_task_sets(fake)
    assert "s2sel" not in R.TASK_SETS or R.TASK_SETS["s2sel"] == V.S2SEL_TASKS


# ------------------------------------------------------------------------------------------ s2test refusal
def test_s2test_refusal(tmp_path):
    doc = V.split_doc(need(V.SPLIT_S2))
    for bad in ("s2test", "S2TEST", "my_s2test", "standard", "extra300", None, ""):
        with pytest.raises(SystemExit):
            V.refuse_s2test(bad)
    V.refuse_s2test("s2val")
    d = tmp_path / "tasks"
    d.mkdir()
    (d / (doc["splits"]["s2test"][0] + ".json")).write_text("{}")
    with pytest.raises(SystemExit, match="holds s2test"):
        V.refuse_s2test("s2sel", d, doc)
    with pytest.raises(SystemExit, match="s2test task dir"):
        V.refuse_s2test("s2sel", paths.WB_S2TEST_TASKS, doc)
    for argv in (["--model", "Q3B", "--task-set", "s2test", "--run-name", "a", "--out-root", str(tmp_path)],
                 ["--model", "Q3B", "--run-name", "a", "--out-root", str(tmp_path)],  # run_vpd's default is s2test
                 ["--odometer", "line", "--model", "Q3B", "--task-set", "standard", "--run-name", "a"]):
        with pytest.raises(SystemExit):
            V.cmd_eval(argv)
    with pytest.raises(SystemExit, match="pilot"):
        V.cmd_eval(["--model", "Q3B", "--task-set", "s2sel", "--run-name", "a", "--out-root",
                    str(V.PILOT_ROOT / "Q3B" / "evals")])
    with pytest.raises(SystemExit):
        V.floors(tmp_path / "f.json", tmp_path / "l", tmp_path / "w", tasks_dir=d)
    assert not list(tmp_path.glob("a*"))


def test_split_eval_argv():
    assert V.split_eval_argv(["--odometer", "line", "--model", "Q3B"]) == ("line", ["--model", "Q3B"])
    assert V.split_eval_argv(["--model", "Q3B", "--odometer=off"]) == ("off", ["--model", "Q3B"])
    assert V.split_eval_argv(["--model", "Q3B"]) == ("off", ["--model", "Q3B"])
    for bad in (["--odometer"], ["--odometer", "pos"]):
        with pytest.raises(SystemExit):
            V.split_eval_argv(bad)


def test_cmd_eval_fails_when_the_odometer_hook_did_not_act(tmp_path, monkeypatch):
    """--odometer line with 0 odometer lines (or off with some) fails: rc 1, results.json -> results_invalid.json,
    explore.json says why.  An earlier run's results.json is never judged, moved or re-described."""
    need(V.SPLIT_S2)

    def fake_main_with(n_lines):
        def fake_main(argv):
            d = Path(argv[argv.index("--out-root") + 1]) / argv[argv.index("--run-name") + 1]
            d.mkdir(parents=True, exist_ok=True)
            (d / "results.json").write_text("{}\n")  # run_vpd.finalize of a valid run
            R.COUNTS["odometer_lines"] += n_lines
        return fake_main

    for mode, n, ok in (("line", 0, False), ("line", 3, True), ("off", 2, False), ("off", 0, True)):
        R.COUNTS.clear()
        monkeypatch.setattr(R, "main", fake_main_with(n))
        name = f"e_{mode}_{n}"
        rc = V.cmd_eval(["--odometer", mode, "--model", "Q3B", "--task-set", "s2sel", "--run-name", name,
                         "--out-root", str(tmp_path)])
        d = tmp_path / name
        ex = json.loads((d / "explore.json").read_text())
        assert (rc == 0) is ok and ex["rc"] == rc and ex["odometer_check"]["ok"] is ok, (mode, n)
        assert ex["odometer_counts"]["odometer_lines"] == n
        assert (d / "results.json").exists() is ok and (d / "results_invalid.json").exists() is (not ok)
        if not ok:
            assert ex["odometer_check"]["results_json_moved_to"] == "results_invalid.json"
    assert V.odometer_problem("line", {}) and V.odometer_problem("off", {}) is None
    # run_vpd refuses an existing run (no --overwrite): the earlier results.json and explore.json stay as they were
    d = tmp_path / "e_line_0"
    (d / "results_invalid.json").rename(d / "results.json")
    before = {p.name: p.read_bytes() for p in d.iterdir()}

    def refuse(argv):
        raise SystemExit(f"{d / 'results.json'} exists; pass --overwrite to replace this run")

    R.COUNTS.clear()
    monkeypatch.setattr(R, "main", refuse)
    assert V.cmd_eval(["--odometer", "line", "--model", "Q3B", "--task-set", "s2sel", "--run-name", "e_line_0",
                       "--out-root", str(tmp_path)]) == 1
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_atomic_writes_use_unique_temp_names(tmp_path, monkeypatch):
    p = tmp_path / "analysis" / "ANALYZE.json"
    t1, t2 = V.tmp_sibling(p), V.tmp_sibling(p)
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())[:40]
    assert t1 != t2 and t1.parent == p.parent and t1.name.startswith(f".ANALYZE.json.tmp-{host}-{os.getpid()}-")
    V.write_json(p, {"b": [1, 2], "u": "é"})
    q = tmp_path / "L.json"
    L.write_json(q, {"b": [1, 2], "u": "é"})
    assert p.read_bytes() == q.read_bytes()  # learn483.write_json's bytes
    assert sorted(x.name for x in p.parent.iterdir()) == ["ANALYZE.json"]
    V.write_text(p.with_name("R.md"), "x\n")
    assert p.with_name("R.md").read_text(encoding="utf-8") == "x\n"

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(V.os, "replace", boom)
    with pytest.raises(OSError):
        V.write_text(p, "new")
    monkeypatch.undo()
    assert json.loads(p.read_text()) == {"b": [1, 2], "u": "é"}
    assert sorted(x.name for x in p.parent.iterdir()) == ["ANALYZE.json", "R.md"]  # no temp file left behind


# ------------------------------------------------------------------------------------------ data writer
def _syn_user(moves, step, stated=True):
    mv = "\n".join(f"  {i}: {d}" + (f" ({x:.1f}m)" if x else "") for i, d, x in moves)
    task = "Navigate to Y. Head north for 20m." + (" Total distance: approximately 20 m." if stated else "")
    return (f"**Step {step}**\n\n**Task:** {task}\n\n**Current View:**\n- Heading: 0°N\n- Pitch: 0°\n\n"
            f"**Available Moves:**\n{mv}\n\n🔄 **ROTATE** - Look in a different direction (without moving):\n"
            f"  - Look LEFT: heading=270\n\nAnalyze the image and decide your next action.<image>")


def _syn_demo(stem, n_move=2):
    kind = stem[:3]
    msgs = [{"role": "system", "content": "sys"}]
    acts = ["MOVE(1)"] * n_move + ["ROTATION(90, 0)", "MOVE(2)", 'STOP("ARRIVED")']
    for j, a in enumerate(acts):
        msgs += [{"role": "user", "content": _syn_user([(1, "front", 10.0), (2, "right", 4.5)], j + 1, kind == "nav")},
                 {"role": "assistant", "content": reply(a)}]
    return {"id": stem, "task_type": kind, "split": "train_rest", "messages": msgs,
            "images": [f"/x/{stem}/turn_{j:02d}.jpg" for j in range(len(acts))], "n_turns": len(acts),
            "size": [640, 400]}


def test_data_writer_tier_order_bytes_and_variants(tmp_path, monkeypatch):
    _syn_constants(monkeypatch)
    doc = _syn_doc()
    split = tmp_path / "split_s2.json"
    split.write_text(json.dumps(doc))
    order = ["vis_r1b", "nav_d1a", "nav_r1a", "vis_d1b", "nav_d2a"]  # tier order: pool = vis_r1b, nav_r1a
    (tmp_path / "order.json").write_text(json.dumps({"order": order}))
    lines = [json.dumps(_syn_demo(s, 2 if s == "nav_r1a" else 1), ensure_ascii=False) for s in order]
    tier = tmp_path / "demos_all.jsonl"
    tier.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    x = tmp_path / "X"
    kw = dict(order_path=tmp_path / "order.json", tier_all=tier, split_path=split)
    r0 = V.build_data("V0", x, **kw)
    out = V.data_path(x, "V0").read_text(encoding="utf-8")
    assert out == lines[0] + "\n" + lines[2] + "\n" and r0["n"] == 2 and r0["eval"] == ["empty", "off"]
    man = json.loads((x / "V0" / "data" / "manifest.json").read_text())
    assert man["exploratory"] and man["sha256"] == V.sha256_text(out) and man["epochs"] == 3
    assert man["reply_classes"] == {"STOP": 2, "STRAIGHT": 3, "TURN": 4}
    assert V.build_data("V0", x, **kw)["state"] == "same"
    # V1T / V2: the same records, only the odometer edits
    V.build_data("V1T", x, **kw)
    V.build_data("V2", x, **kw)
    v1t = [json.loads(ln) for ln in V.data_path(x, "V1T").read_text(encoding="utf-8").splitlines()]
    v2 = [json.loads(ln) for ln in V.data_path(x, "V2").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in v1t] == [r["id"] for r in v2] == ["vis_r1b", "nav_r1a"]
    assert v1t[0]["images"] == json.loads(lines[0])["images"]
    walked_v1t = [re.search(r"Walked so far: (\S+) m", m["content"]).group(1) for m in v1t[1]["messages"]
                  if m["role"] == "user"]
    assert walked_v1t == ["0.0", "10.0", "20.0", "20.0", "24.5"]
    th = [json.loads(m["content"])["THOUGHT"] for m in v2[1]["messages"] if m["role"] == "assistant"]
    assert th == ["walked 0.0 of 20 m", "walked 10.0 of 20 m", "walked 20.0 of 20 m", "walked 20.0 of 20 m",
                  "walked 24.5 of 20 m"]
    assert json.loads(v2[0]["messages"][2]["content"])["THOUGHT"] == "walked 0.0 m"  # vis
    # V5 on V1T: per assistant message loss_weight, everything else as the base
    r5 = V.build_data("V5", x, base="V1T")
    assert r5["eval"] == ["empty", "line"]
    v5 = [json.loads(ln) for ln in V.data_path(x, "V5").read_text(encoding="utf-8").splitlines()]
    for a, b in zip(v5, v1t):
        for ma, mb in zip(a["messages"], b["messages"]):
            if ma["role"] != "assistant":
                assert ma == mb and "loss_weight" not in ma
                continue
            act = json.loads(ma["content"])["ACTION"]
            want = 5.0 if act.startswith("STOP") else (1.0 if act == "MOVE(1)" else 3.0)
            assert ma["loss_weight"] == want and {k: v for k, v in ma.items() if k != "loss_weight"} == mb
    assert json.loads((x / "V5" / "data" / "manifest.json").read_text())["base"] == "V1T"
    # V4 on V5: a byte copy, eval mode of V5's base
    r4 = V.build_data("V4", x, base="V5")
    assert V.data_path(x, "V4").read_bytes() == V.data_path(x, "V5").read_bytes() and r4["eval"] == ["empty", "line"]
    assert V.eval_mode("V4", "V5", x) == ("empty", "line") and V.EPOCHS["V4"] == 10
    with pytest.raises(SystemExit):
        V.build_data("V5", x, base="V2")  # V5 exists, built on V1T
    with pytest.raises(SystemExit):
        V.build_data("V5", tmp_path / "Y", base="V5")
    # an existing file with other content is never replaced
    p = V.data_path(x, "V2")
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="different content"):
        V.build_data("V2", x, **kw)
    # the tier file must be the order
    (tmp_path / "order.json").write_text(json.dumps({"order": list(reversed(order))}))
    with pytest.raises(SystemExit, match="not the order"):
        V.pool_lines(tmp_path / "order.json", tier, split)


def test_v5_weights_per_reply_class():
    assert [V.reply_class(a) for a in ('STOP("ARRIVED")', "ROTATION(195.2, 0)", "MOVE(2)", "MOVE(10)", "MOVE(1)")] \
        == ["STOP", "TURN", "TURN", "TURN", "STRAIGHT"]
    rec = {"id": "nav_x", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                       {"role": "assistant", "content": reply("MOVE(1)")},
                                       {"role": "user", "content": "u"},
                                       {"role": "assistant", "content": reply("ROTATION(90, 0)", "walked 1.0 m")},
                                       {"role": "user", "content": "u"},
                                       {"role": "assistant", "content": reply("MOVE(3)")},
                                       {"role": "user", "content": "u"},
                                       {"role": "assistant", "content": reply('STOP("ARRIVED")')}]}
    w = V.with_weights(rec)
    assert [m.get("loss_weight") for m in w["messages"]] == [None, None, 1.0, None, 3.0, None, 3.0, None, 5.0]
    assert "loss_weight" not in rec["messages"][2]
    with pytest.raises(ValueError):
        V.reply_class("JUMP(1)")


# ------------------------------------------------------------------------------------------ modes, names, expect
def test_eval_mode_job_evals_and_expected_outputs(tmp_path):
    x = tmp_path / "X"
    assert V.eval_mode("V0") == ("empty", "off") and V.eval_mode("V1T") == ("empty", "line")
    assert V.eval_mode("V2") == ("short", "off") and V.eval_mode("V5", "V2", x) == ("short", "off")
    for bad in (("V5", None), ("V5", "V5"), ("V4", "V4"), ("V0", "V1T"), ("V9", None)):
        with pytest.raises(SystemExit):
            V.eval_mode(bad[0], bad[1], x)
    assert V.eval_name("V0", "s2sel", "empty", "line", True) == "wbv__Q3B__V0__s2sel__640__empty__odo__blind"
    assert V.EVAL_NAME_RE.match("wbv__Q3B__raw__s2sel__640__off__odo").group("odo")
    a = V.expected_outputs("V0", x=x, b1_out=tmp_path / "B1")
    res = [p for p in a if p.name == "results.json"]
    assert len(res) == 6 and sum("/V1/evals/" in str(p) for p in res) == 4
    assert x / "V1" / "evals" / "wbv__Q3B__V0__s2sel__640__empty__odo" / "results.json" in a
    assert x / "V1" / "evals" / "wbv__Q3B__raw__s2sel__640__off__odo" / "results.json" in a
    assert x / "b1" / "B1_CHECK.json" in a and tmp_path / "B1" / "demos_all.jsonl" in a
    assert x / "V0" / "adapter" / "adapter.pt" in a and x / "floors" / "floors_s2sel.json" in a
    b = V.expected_outputs("V2", x=x)
    assert [p.parent.name for p in b if p.name == "results.json"] == [
        "wbv__Q3B__V2__s2sel__640__short", "wbv__Q3B__V2__s2sel__640__short__blind"]
    c = V.expected_outputs("V5", "V1T", x=x)
    assert x / "V5" / "evals" / "wbv__Q3B__V5__s2sel__640__empty__odo__blind" / "results.json" in c
    assert all(not V.is_under(p, V.PILOT_ROOT) and "s2test" not in str(p) for p in a + b + c)


# ------------------------------------------------------------------------------------------ checks
def test_init_check_and_adapter_check(tmp_path, monkeypatch):
    from vpd.train import n2_train_vpd as T

    init = tmp_path / "0W"
    init.mkdir()
    (init / "adapter.pt").write_bytes(b"init")
    (init / "meta.json").write_text(json.dumps({"start": "Q3B", "optimizer_steps": 30, "optimizer_steps_planned": 30}))
    sha = L.sha256_file(init / "adapter.pt")
    probs, facts = V.init_check(init, sha[:12])
    assert probs == [] and facts["init_sha256"] == sha
    assert V.init_check(init)[0]  # not the pilot's 5cdbfd4366a3
    tr = tmp_path / "trainer.py"
    tr.write_text("x = 1\n")
    assert any("loss_weight" in p for p in V.init_check(init, sha[:12], True, tr)[0])
    tr.write_text('w = m.get("loss_weight", 1.0)\n')
    assert V.init_check(init, sha[:12], True, tr)[0] == []
    # adapter-ok of a variant
    monkeypatch.setattr(V, "INIT_SHA_PREFIX", sha[:12])
    x = tmp_path / "X"
    assert V.adapter_check("V1T", x, init)[0] == 1
    data = V.data_path(x, "V1T")
    data.parent.mkdir(parents=True)
    data.write_text("{}\n")
    ad = x / "V1T" / "adapter"
    ad.mkdir(parents=True)
    (ad / "adapter.pt").write_bytes(b"a")
    meta = {"start": "Q3B", "epochs": 3, "optimizer_steps": 154, "optimizer_steps_planned": 154,
            "code_sha256": T.code_fingerprint()["code_sha256"], "data_sha256": L.sha256_file(data),
            "init_adapter": {"sha256": sha}}
    (ad / "meta.json").write_text(json.dumps(meta))
    assert V.adapter_check("V1T", x, init)[0] == 0
    for k, v, rc in (("code_sha256", "other", 3), ("data_sha256", "other", 1), ("epochs", 10, 1),
                     ("optimizer_steps", 100, 1), ("init_adapter", {"sha256": "x"}, 1)):
        (ad / "meta.json").write_text(json.dumps(dict(meta, **{k: v})))
        assert V.adapter_check("V1T", x, init)[0] == rc, k
    # weighted data (V5) needs a weighted training run (the trainer's meta.loss_weights)
    (data.parent / "manifest.json").write_text(json.dumps({"loss_weight": dict(V.V5_WEIGHTS)}))
    (ad / "meta.json").write_text(json.dumps(meta))
    assert V.adapter_check("V1T", x, init)[0] == 1
    (ad / "meta.json").write_text(json.dumps(dict(meta, loss_weights={"mode": "weighted"})))
    assert V.adapter_check("V1T", x, init)[0] == 0


def test_init_check_code_reference_for_v5_v4(tmp_path, monkeypatch):
    """V5 / V4 refuse to train when the current trainer code_sha256 differs from X/V0's or X/BASE's adapter."""
    init = tmp_path / "0W"
    init.mkdir()
    (init / "adapter.pt").write_bytes(b"init")
    (init / "meta.json").write_text(json.dumps({"start": "Q3B", "optimizer_steps": 30, "optimizer_steps_planned": 30}))
    pre = L.sha256_file(init / "adapter.pt")[:12]
    x = tmp_path / "X"

    def meta(v, code):
        (x / v / "adapter").mkdir(parents=True, exist_ok=True)
        (x / v / "adapter" / "meta.json").write_text(json.dumps({"code_sha256": code}))

    probs, facts = V.init_check(init, pre, variant="V5", x=x, base="V1T", current_code="c1")
    assert probs == [] and facts["code_reference"] == {}  # no V0 (nor V1T) adapter yet: nothing to compare with
    meta("V0", "c1")
    assert V.init_check(init, pre, variant="V5", x=x, base="V1T", current_code="c1")[0] == []
    probs, facts = V.init_check(init, pre, variant="V4", x=x, base="V1T", current_code="c2")
    assert facts["code_differs"] is True and len(probs) == 1
    assert "refusing to train V4" in probs[0] and "X/V0/adapter/meta.json code_sha256 c1" in probs[0]
    meta("V1T", "c3")  # the base trained with other code than V0
    probs, facts = V.init_check(init, pre, variant="V5", x=x, base="V1T", current_code="c1")
    assert facts["code_differs"] and "X/V1T/adapter" in probs[0] and "X/V0/" not in probs[0]
    assert facts["code_reference"] == {"V0": "c1", "V1T": "c3"}
    for v in ("V0", "V1T", "V2"):  # trained together from one commit: no reference check
        assert V.init_check(init, pre, variant=v, x=x, current_code="zz")[0] == []
    # the CLI: exit 4 when the code differs (the payload records train_V:code_differs), 0 when it is equal
    monkeypatch.setattr(V, "trainer_code_sha256", lambda: "c3")
    cli = ["init-check", "--init", str(init), "--expect-sha", pre, "--x", str(x)]
    assert V.main(cli + ["--variant", "V5", "--base", "V1T"]) == V.INIT_RC_CODE_DIFFERS == 4
    meta("V0", "c3")
    assert V.main(cli + ["--variant", "V5", "--base", "V1T"]) == 0
    assert V.main(cli + ["--variant", "V4", "--base", "V5", "--expect-sha", "ffffffffffff"]) == 1  # other problem
    monkeypatch.setattr(V, "trainer_code_sha256", lambda: "c9")
    assert V.main(cli + ["--variant", "V4", "--base", "V5", "--expect-sha", "ffffffffffff"]) == 4


def test_real_pilot_0W_init_check():
    pt = need(V.INIT / "adapter.pt")
    if not os.access(pt, os.R_OK):
        pytest.skip("pilot 0W not readable")
    probs, facts = V.init_check(V.INIT)
    assert probs == [] and facts["init_sha256"].startswith("5cdbfd4366a3")


def test_b1_checks(tmp_path, monkeypatch):
    from vpd.walkerbench import baselines as B

    monkeypatch.setattr(B, "rm_expert", lambda s, mv, rot: {"type": "rotation", "heading": round(mv()[0]["heading"], 1)})
    assert V.b1_precheck()[0]  # the unfixed rule (-164.8) is refused
    monkeypatch.setattr(B, "rm_expert", lambda s, mv, rot: {"type": "rotation", "heading": round(mv()[0]["heading"] % 360, 1)})
    assert V.b1_precheck()[0] == []
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text("frozen\n")
    monkeypatch.setattr(V, "FROZEN_DEMOS_SHA256", L.sha256_file(frozen))
    out = tmp_path / "B1"
    (out / "images").mkdir(parents=True)
    imgs = []
    for j in range(3):
        p = out / "images" / f"turn_{j:02d}.jpg"
        p.write_bytes(b"j")
        imgs.append(str(p))
    rec = {"id": V.B1_STEM, "n_turns": 3, "images": imgs, "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": reply("MOVE(1)")}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": reply(V.B1_ROTATION)}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": reply('STOP("ARRIVED")')}]}
    (out / "demos_all.jsonl").write_text(json.dumps(rec) + "\n")
    (out / "manifest.json").write_text(json.dumps({"dropped": [], "code": {"baselines.py (rm_expert)": "new"}}))
    before = {"frozen_sha256": V.FROZEN_DEMOS_SHA256}
    probs, facts = V.b1_after(out, before, frozen)
    assert probs == [] and facts["records"] == [V.B1_STEM] and facts["baselines_sha256"] == "new"
    frozen.write_text("changed\n")
    assert len(V.b1_after(out, before, frozen)[0]) == 2
    frozen.write_text("frozen\n")
    (out / "manifest.json").write_text(json.dumps({"dropped": [{"id": V.B1_STEM}]}))
    assert V.b1_after(out, before, frozen)[0]


def test_stats_helpers():
    assert V.exact_mcnemar(0, 0) == 1.0 and V.exact_mcnemar(4, 3) == 1.0
    assert abs(V.exact_mcnemar(6, 0) - 0.03125) < 1e-12 and abs(V.exact_mcnemar(8, 1) - 0.0390625) < 1e-12
    assert V.exact_mcnemar(1, 8) == V.exact_mcnemar(8, 1)
    assert V.bar_k(65, 20 / 65) == 28 and V.bar_k(66, 9 / 66) == 15  # the G4(g) bars on SEL


# ------------------------------------------------------------------------------------------ analyze
def _log(path: Path, stem: str, steps):
    lines = [json.dumps({"event": "session_start", "task_id": stem, "initial_state": {"pano_id": "p0", "lat": 0.0,
                                                                                     "lng": 0.0}})]
    for i, (pano, act, moves) in enumerate(steps):
        lines.append(json.dumps({"event": "action", "step": i + 1, "state": {"pano_id": pano, "lat": 0.0,
                                                                            "lng": 0.0001 * i},
                                 "action": act, "available_moves": moves, "reason": ""}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _eval(d: Path, success: dict, results=None, explore=None):
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(results or {"adapter": {"adapter_sha256": "a"}}))
    (d / "explore.json").write_text(json.dumps(explore or {"odometer": "off"}))
    (d / "per_task.jsonl").write_text("".join(json.dumps({"stem": s, "kind": s[:3], "success": v, "oracle": v,
                                                          "steps": 1, "log": None}) + "\n"
                                              for s, v in success.items()))


def test_analyze_synthetic(tmp_path):
    x = tmp_path / "X"
    sel = tmp_path / "s2sel"
    (sel / "tasks").mkdir(parents=True)
    navs = [f"nav_{i:02d}" for i in range(10)]
    viss = [f"vis_{i:02d}" for i in range(10)]
    for s in navs + viss:
        (sel / "tasks" / f"{s}.json").write_text(json.dumps({
            "spawn_point": "p0", "target_pano_ids": ["p1"], "visual_path": [{"pano_id": "p0"}, {"pano_id": "p1"}],
            "description": "Go. Total distance: approximately 20 m." if s.startswith("nav") else "Find the shop.",
            "ground_truth": {"optimal_distance_meters": 20}}))
    (sel / "selection.json").write_text(json.dumps({"ids": navs + viss, "members": {
        "s2val": navs[:5] + viss[:5], "s2dev74": navs[5:] + viss[5:]}}))
    db = tmp_path / "cache.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE locations (pano_id TEXT PRIMARY KEY, lat REAL, lng REAL)")
    con.execute("CREATE TABLE metadata (pano_id TEXT PRIMARY KEY, lat REAL, lng REAL)")
    con.executemany("INSERT INTO locations VALUES (?, ?, ?)", [("p0", 0.0, 0.0), ("p1", 0.0, 0.001)])
    con.commit()
    con.close()
    (x / "floors").mkdir(parents=True)
    (x / "floors" / "floors_s2sel.json").write_text(json.dumps({"logs": None, "policies": {
        "ODO": {"per_task": {s: {"success": 1} for s in navs[:3]}},
        "FWD": {"per_task": {s: {"success": 1} for s in viss[:1]} | {viss[9]: {"success": 0}}}}}))
    # V0: nav 8/10 with images (7 image-only wins vs blind), vis 2/10; blind nav 1/10, vis 2/10
    _eval(x / "V0" / "evals" / V.eval_name("V0", "s2sel", "empty"),
          {**{s: int(i < 8) for i, s in enumerate(navs)}, **{s: int(i < 2) for i, s in enumerate(viss)}})
    _eval(x / "V0" / "evals" / V.eval_name("V0", "s2sel", "empty", blind=True),
          {**{s: int(i == 0) for i, s in enumerate(navs)}, **{s: int(i < 2) for i, s in enumerate(viss)}},
          results={"adapter": {"adapter_sha256": "a"}, "blind": True})
    _eval(x / "V1" / "evals" / V.eval_name("raw", "s2sel", "off", "line"), {s: 0 for s in navs + viss},
          explore={"odometer": "line", "odometer_counts": {"odometer_lines": 40, "odometer_moves_added": 12}})
    # an odometer eval whose hook never inserted the line (scored before the check existed): flagged
    _eval(x / "V1" / "evals" / V.eval_name("V0", "s2sel", "empty", "line"), {s: 0 for s in navs + viss},
          explore={"odometer": "line", "odometer_counts": {"odometer_lines": 0, "odometer_moves_added": 0}})
    # one cmd_eval failed on the hook check: results_invalid.json, with the reason in explore.json
    dinv = x / "V1" / "evals" / V.eval_name("raw", "s2sel", "empty")
    dinv.mkdir(parents=True)
    (dinv / "results_invalid.json").write_text("{}")
    (dinv / "explore.json").write_text(json.dumps({"odometer": "off", "odometer_check": {"ok": False, "problem": "p"}}))
    # trained variants: V0 and V2 from one init but with two trainer codes
    for v, code in (("V0", "c1" * 32), ("V2", "c2" * 32)):
        (x / v / "adapter").mkdir(parents=True)
        (x / v / "adapter" / "meta.json").write_text(json.dumps({
            "code_sha256": code, "init_adapter": {"sha256": "i1" * 32}, "optimizer_steps": 5,
            "optimizer_steps_planned": 5}))
    (x / "V1T" / "evals" / V.eval_name("V1T", "s2sel", "empty", "line")).mkdir(parents=True)  # no results yet
    # V2 short eval with one logged nav episode for the THOUGHT odometer stats
    d2 = x / "V2" / "evals" / V.eval_name("V2", "s2sel", "short")
    _eval(d2, {s: 0 for s in navs + viss})
    rows = [json.loads(ln) for ln in (d2 / "per_task.jsonl").read_text().splitlines()]
    rows[0]["log"] = "a.jsonl"
    (d2 / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    mv = [{"id": 1, "direction": "front", "distance": 9.64}]
    _log(d2 / "WalkerBench" / "logs" / "a.jsonl", navs[0],
         [("p0", {"type": "move", "move_id": 1}, mv), ("p1", {"type": "move", "move_id": 1}, mv),
          ("p1", {"type": "stop"}, [])])
    calls = [{"task": navs[0], "step": 1, "reply": reply("MOVE(1)", "walked 0.0 of 20 m")},
             {"task": navs[0], "step": 2, "reply": reply("MOVE(1)", "walked 9.0 of 20 m")},
             {"task": navs[0], "step": 3, "reply": reply('STOP("ARRIVED")', "close enough")}]
    (d2 / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    doc = V.analyze(x, sel, db)
    fl = doc["floors"]["subsets"]
    assert fl["SEL"]["ODO"]["nav"]["k"] == 3 and fl["s2val"]["ODO"]["nav"]["k"] == 3 and fl["s2dev74"]["ODO"]["nav"]["k"] == 0
    assert fl["SEL"]["FWD"]["vis"] == L.rate(1, 10)
    v0 = doc["variants"]["V0"]
    assert v0["selection_score_nav_k_plus_vis_k"] == 10
    nav, vis = v0["g4g"]["nav"], v0["g4g"]["vis"]
    assert nav["A3_paired"]["image_only"] == 7 and nav["A3_paired"]["blind_only"] == 0
    assert nav["A3_images_better_p_lt_0.05"] is True and abs(nav["A3_paired"]["p_two_sided_exact"] - 0.01562) < 1e-4
    assert nav["A2_lb_above_floor"] is True and nav["floor"]["k"] == 3 and nav["A4_rate_le_0.80"] is True
    assert vis["A3_paired"]["image_only"] == 0 and vis["A3_images_better_p_lt_0.05"] is False
    assert vis["all_three"] is False and nav["all_three"] is True
    t = doc["evals"]["V0/" + V.eval_name("V0", "s2sel", "empty")]["table"]
    assert t["s2val"]["nav"]["k"] == 5 and t["s2dev74"]["nav"]["k"] == 3 and t["SEL"]["vis"]["k"] == 2
    assert doc["evals"]["V1T/" + V.eval_name("V1T", "s2sel", "empty", "line")]["state"] == "no results.json"
    assert doc["evals"]["V1/" + V.eval_name("raw", "s2sel", "off", "line")]["odometer"] == "line"
    th = doc["evals"]["V2/" + V.eval_name("V2", "s2sel", "short")]["thought"]
    assert th["calls"] == 3 and th["format_ok"]["k"] == 2 and th["walked_exact_0.05"]["k"] == 1
    assert th["walked_abs_err_max"] == 0.6 and th["nav_stated_matches_task"]["k"] == 2
    assert doc["g4h_inputs"]["best_nav_k_SEL"] == 8
    # consistency: the distinct adapter code_sha256 / init_adapter sha256 across the trained variants
    cons = doc["consistency"]
    assert list(cons["variants"]) == ["V0", "V2"] and cons["variants"]["V0"]["finished"] is True
    assert cons["distinct_code_sha256"] == [{"value": "c1" * 32, "variants": ["V0"]},
                                            {"value": "c2" * 32, "variants": ["V2"]}]
    assert cons["distinct_init_adapter_sha256"] == [{"value": "i1" * 32, "variants": ["V0", "V2"]}]
    assert len(cons["problems"]) == 1 and "2 distinct adapter code_sha256" in cons["problems"][0]
    assert "c1c1c1c1c1c1 (V0)" in cons["problems"][0] and "c2c2c2c2c2c2 (V2)" in cons["problems"][0]
    # odometer: the line eval without lines is flagged, the counted one is not, the failed one shows its reason
    bad = doc["evals"]["V1/" + V.eval_name("V0", "s2sel", "empty", "line")]
    assert bad["state"] == "ok" and "odometer_lines == 0" in bad["odometer_problem"]
    assert "odometer_problem" not in doc["evals"]["V1/" + V.eval_name("raw", "s2sel", "off", "line")]
    inv = doc["evals"]["V1/" + V.eval_name("raw", "s2sel", "empty")]
    assert inv["state"] == "invalid" and inv["invalid_reason"] == "p"
    assert doc["problems"] == cons["problems"] + [
        f"eval V1/{V.eval_name('V0', 's2sel', 'empty', 'line')}: " + bad["odometer_problem"]]
    md = (x / "analysis" / "REPORT_EXPLORATORY.md").read_text(encoding="utf-8")
    assert "EXPLORATORY" in md and "Numbers only" in md and json.loads((x / "analysis" / "ANALYZE.json").read_text())[
        "exploratory"] is True
    assert "## Problems" in md and "## Consistency (trained variants)" in md and "| code_sha256 | c2c2c2c2c2c2 | V2 |" in md
    # no consistency problem with one code and one init
    (x / "V2" / "adapter" / "meta.json").write_text(json.dumps({"code_sha256": "c1" * 32,
                                                                "init_adapter": {"sha256": "i1" * 32}}))
    assert V.analyze(x, sel, db)["consistency"]["problems"] == []
    assert sorted(p.name for p in (x / "analysis").iterdir()) == ["ANALYZE.json", "REPORT_EXPLORATORY.md"]


# ------------------------------------------------------------------------------------------ payload (dry run)
_KNOBS = ("VPD_THREADS", "VPD_JOB_DEADLINE_S", "VPD_MIN_STEP_S", "VPD_MERGE_DIR", "VPD_RERUN_STALE",
          "VPD_RERUN_INVALID", "VPD_DRYRUN", "RUN", "VPD_WBV_ROOT", "VPD_WBV_PILOT", "VPD_WBV_B1_OUT")


def _bash(args, extra, cwd):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update(extra)
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env, cwd=str(cwd), timeout=300)


def _env(base: Path, **extra):
    env = {"VPD_DRYRUN": "1", "VPD_MERGE_DIR": "/dev/shm/vpd_merged", "VPD_WBV_ROOT": str(base / "X"),
           "VPD_WBV_PILOT": str(base / "P"), "VPD_WBV_B1_OUT": str(base / "B1")}
    env.update(extra)
    return env


def _dry(out: str):
    return [ln[len("[dry] "):] for ln in out.splitlines() if ln.startswith("[dry] ")]


def _kinds(lines):
    out = []
    for ln in lines:
        if "n2_train_vpd.py" in ln:
            out.append("train")
        elif "merge.py" in ln:
            out.append("merge")
        elif "rm -rf" in ln:
            out.append("rm")
        elif "variants eval " in ln:
            out.append("eval:" + ln.split("--run-name wbv__Q3B__")[1].split()[0])
        elif "demos --stems" in ln:
            out.append("b1_render")
        elif "demos --merge" in ln:
            out.append("b1_merge")
        else:
            m = re.search(r"vpd\.walkerbench\.variants (\S+)(?: --stage (\S+))?", ln)
            if m:
                out.append(m.group(1) + (":" + m.group(2) if m.group(2) else ""))
    return out


def test_payload_bash_syntax():
    res = subprocess.run(["bash", "-n", str(PAYLOAD)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_payload_dryrun_v0(tmp_path):
    run = tmp_path / "jobs" / "wb-v0v1-202611100000"
    res = _bash([str(PAYLOAD), "V0", str(run)], _env(tmp_path), tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    lines = _dry(res.stdout)
    assert _kinds(lines) == [
        "b1:precheck", "b1:before", "b1_render", "b1_merge", "b1:after", "setup", "floors", "data", "init-check",
        "adapter-ok", "train", "rm", "merge", "eval:V0__s2sel__640__empty", "eval:V0__s2sel__640__empty__blind",
        "eval:V0__s2sel__640__empty__odo", "rm", "eval:raw__s2sel__640__off", "eval:raw__s2sel__640__off__odo",
        "eval:raw__s2sel__640__empty", "analyze", "expect"]
    X, P, B1 = tmp_path / "X", tmp_path / "P", tmp_path / "B1"
    tr = next(ln for ln in lines if "n2_train_vpd.py" in ln)
    assert tr.endswith(f"vpd/train/n2_train_vpd.py --start Q3B --data {X}/V0/data/train.jsonl --init-adapter "
                       f"{P}/adapters/0W --out {X}/V0/adapter --epochs 3 --seed 1001 --phase wbv_V0 --threads 2 "
                       f"--num-workers 3") and str(paths.PY_TRAIN) in tr
    ev = {ln.split("--run-name ")[1].split()[0]: ln for ln in lines if "variants eval " in ln}
    want = {V.eval_name(m, s, c, o, b) for (_, m, s, c, o, b) in V.JOB_EVALS["V0"]}
    assert set(ev) == want
    p = ev["wbv__Q3B__V0__s2sel__640__empty"]
    for frag in ("eval --odometer off --model Q3B --task-set s2sel --render-size 640x400 --constrained empty "
                 "--max-tokens 512 --history-turns 0 --workers 8 --temperature 0", f"--out-root {X}/V0/evals",
                 "--model-dir /dev/shm/vpd_merged/wbv__Q3B__V0", f"--adapter-meta {X}/V0/adapter/meta.json"):
        assert frag in p, (frag, p)
    assert "--blind" in ev["wbv__Q3B__V0__s2sel__640__empty__blind"] and "--blind" not in p
    odo = ev["wbv__Q3B__V0__s2sel__640__empty__odo"]
    assert "--odometer line" in odo and f"--out-root {X}/V1/evals" in odo and "--model-dir" in odo
    r1, r2 = ev["wbv__Q3B__raw__s2sel__640__off"], ev["wbv__Q3B__raw__s2sel__640__off__odo"]
    assert "--odometer off" in r1 and "--constrained off --max-tokens 1024" in r1 and "--model-dir" not in r1
    assert "--odometer line" in r2 and "--constrained off --max-tokens 1024" in r2
    assert "--constrained empty --max-tokens 512" in ev["wbv__Q3B__raw__s2sel__640__empty"]
    assert all("--task-set s2sel" in ln for ln in ev.values()) and "s2test" not in res.stdout
    b1r = next(ln for ln in lines if "demos --stems" in ln)
    assert f"--stems {V.B1_STEM} --out {B1} --work {run}/b1_work --strict" in b1r
    assert f"demos --merge --allow-partial --out {B1}" in next(ln for ln in lines if "demos --merge" in ln)
    fl = next(ln for ln in lines if "variants floors" in ln)
    assert f"--out {X}/floors/floors_s2sel.json --logs {X}/floors/logs --work {run}/floors_work" in fl
    gpu = [ln for ln in lines if any(k in ln for k in ("n2_train_vpd", "merge.py", "variants eval", "demos --",
                                                         "variants floors"))]
    assert gpu and all(ln.startswith("timeout -k 30 ") and int(ln.split()[3]) <= 8520 for ln in gpu)
    assert "deadline=8520s" in res.stdout and "EXPLORATORY" in res.stdout
    assert "--need-loss-weight" not in res.stdout
    steps = [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]
    assert all(s["rc"] == 0 for s in steps) and not (run / "DONE").exists()
    assert not X.exists() and not B1.exists()
    assert sorted(q.name for q in run.iterdir()) == ["df_end.txt", "node.txt", "steps.jsonl"]
    assert "runs/pilot" not in " ".join(ln for ln in lines if "--out " in ln)


@pytest.mark.parametrize("variant,base,mode,epochs", [("V1T", "", "empty:line", 3), ("V2", "", "short:off", 3),
                                                      ("V5", "V1T", "empty:line", 3), ("V4", "V5", "short:off", 10)])
def test_payload_dryrun_trained_variants(tmp_path, variant, base, mode, epochs):
    if variant == "V4":  # V4 on V5: the eval mode comes from V5's data manifest (V5 built on V2)
        (tmp_path / "X" / "V5" / "data").mkdir(parents=True)
        (tmp_path / "X" / "V5" / "data" / "manifest.json").write_text(json.dumps(
            {"base": "V2", "eval": {"constrained_odometer": ["short", "off"]}}))
    run = tmp_path / "jobs" / f"wb-{variant.lower()}-202611100000"
    args = [str(PAYLOAD), variant, str(run)] + ([base] if base else [])
    res = _bash(args, _env(tmp_path), tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    lines = _dry(res.stdout)
    cm, odo = mode.split(":")
    suf = "__odo" if odo == "line" else ""
    k = _kinds(lines)
    assert k[0] == "setup" and "b1:precheck" not in k and "eval:raw__s2sel__640__off" not in k
    assert ("floors" in k) == (variant in ("V5", "V4"))
    assert [e for e in k if e.startswith("eval:")] == [f"eval:{variant}__s2sel__640__{cm}{suf}",
                                                       f"eval:{variant}__s2sel__640__{cm}{suf}__blind"]
    tr = next(ln for ln in lines if "n2_train_vpd.py" in ln)
    assert f"--out {tmp_path}/X/{variant}/adapter --epochs {epochs} --seed 1001 --phase wbv_{variant}" in tr
    ev = [ln for ln in lines if "variants eval " in ln]
    assert all(f"--odometer {odo}" in ln and f"--constrained {cm}" in ln for ln in ev)
    assert all(("--thought-max-chars 32" in ln) == (cm == "short") for ln in ev)
    data = next(ln for ln in lines if "variants data" in ln)
    assert f"data --variant {variant}" in data and (f"--base {base}" in data) == bool(base)
    assert ("--need-loss-weight" in res.stdout) == (variant == "V5" or (variant == "V4" and base == "V5"))
    dl = {"V1T": 6000, "V2": 6000, "V5": 6000, "V4": 15720}[variant]
    assert f"deadline={dl}s" in res.stdout


def test_payload_refusals_and_deadline(tmp_path):
    run = tmp_path / "jobs" / "wb-v0v1-202611100001"
    for args in (["V3", str(run)], ["V0", str(run), "V1T"], ["V5", str(run)], ["V5", str(run), "V5"],
                 ["V4", str(run), "V4"], ["V0"]):
        r = _bash([str(PAYLOAD), *args], _env(tmp_path, RUN=""), tmp_path)
        assert r.returncode == 2 and "usage" in r.stderr, args
    r = _bash([str(PAYLOAD), "V0", str(run)], _env(tmp_path, VPD_WBV_ROOT=str(paths.WB_RUNS / "pilot" / "Q3B" / "x")),
              tmp_path)
    assert r.returncode == 2 and "refusing" in r.stderr
    for bad in ("0", "abc"):
        r = _bash([str(PAYLOAD), "V0", str(run)], _env(tmp_path, VPD_JOB_DEADLINE_S=bad), tmp_path)
        assert r.returncode == 2 and "VPD_JOB_DEADLINE_S" in r.stderr
    r = _bash([str(PAYLOAD), "V0", str(run)], _env(tmp_path, VPD_JOB_DEADLINE_S="30"), tmp_path)
    assert r.returncode == 1
    names = [json.loads(s)["name"] for s in (run / "steps.jsonl").read_text().splitlines()]
    assert {"b1_render:no_time", "train_V0:no_time", "eval_V1_raw_s2sel_off:no_time"} <= set(names)
    assert "merge_V0:no_adapter" in names and not (run / "DONE").exists()


# ------------------------------------------------------------------------------------------ payload (non-dry, fakes)
FAKE_PY = r"""#!/usr/bin/env bash
# fake PY_EVAL / PY_Q3 / PY_TRAIN of the non-dry payload tests: log the call, answer it, touch nothing else
printf '%s\n' "$*" >> "$FAKE_LOG"
a=" $* "
case $a in
  *" vpd.walkerbench.variants init-check "*)
    echo '{"ok": false, "problems": ["fake init-check"]}'; exit "${FAKE_INIT_RC:-0}" ;;
  *" vpd.walkerbench.variants adapter-ok "*) exit "${FAKE_ADAPTER_RC:-1}" ;;
  *" vpd.walkerbench.variants b1 "*"--stage done "*) exit 1 ;;
  *" vpd.walkerbench.variants eval-mode "*) echo "${FAKE_MODE:-empty off}" ;;
  *" vpd.walkerbench.variants expect "*) echo "[variants] fake expect"; exit "${FAKE_EXPECT_RC:-1}" ;;
esac
exit 0
"""


def _fake_tree(tmp_path: Path, data_variant: str | None = None):
    """A temp code tree holding a byte copy of the payload and a common_env.sh shim (the real prelude, then fake
    interpreters and VPD_OUT = tmp_path/OUT with the payload's input files; X/<data_variant>/data/train.jsonl as the
    fake data step would have left it); returns (payload, env, WB)."""
    if data_variant:
        (tmp_path / "X" / data_variant / "data").mkdir(parents=True)
        (tmp_path / "X" / data_variant / "data" / "train.jsonl").write_text("{}\n")
    jobs = tmp_path / "code" / "vpd" / "jobs"
    jobs.mkdir(parents=True)
    shutil.copyfile(PAYLOAD, jobs / "wb_variants.sh")
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    wb = tmp_path / "OUT" / "walkerbench"
    for rel in ("split/split_s2.json", "tiers/order_seed1001.json", "tiers/seed1001/demos_all.jsonl"):
        (wb / rel).parent.mkdir(parents=True, exist_ok=True)
        (wb / rel).write_text("{}\n")
    (tmp_path / "P" / "adapters" / "0W").mkdir(parents=True)
    (tmp_path / "P" / "adapters" / "0W" / "adapter.pt").write_bytes(b"0W")
    q = shlex.quote
    (jobs / "common_env.sh").write_text(
        "# test shim: the real prelude, then fake interpreters and a temp OUT\n"
        f"source {q(str(PAYLOAD.parent / 'common_env.sh'))}\n"
        f"export VPD_OUT={q(str(tmp_path / 'OUT'))} PY_EVAL={q(str(fake))} PY_Q3={q(str(fake))} "
        f"PY_TRAIN={q(str(fake))}\n")
    env = {"VPD_WORKTREE": str(tmp_path / "code"), "VPD_WBV_ROOT": str(tmp_path / "X"),
           "VPD_WBV_PILOT": str(tmp_path / "P"), "VPD_WBV_B1_OUT": str(tmp_path / "B1"),
           "VPD_MERGE_DIR": str(tmp_path / "merged"),  # not a /tmp|/dev/shm vpd_merged dir: nothing is ever removed
           "FAKE_LOG": str(tmp_path / "fake.log")}
    return jobs / "wb_variants.sh", env, wb


def _steps(run: Path) -> list:
    return [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("variant,base,fake,rc_want,train_step", [
    ("V0", "", {"FAKE_INIT_RC": "1"}, 1, ("train_V0:no_init", "init_check", 1)),
    ("V5", "V1T", {"FAKE_INIT_RC": "4", "FAKE_MODE": "empty line"}, 1, ("train_V5:code_differs", "init_check", 4)),
    ("V0", "", {"FAKE_INIT_RC": "0", "FAKE_ADAPTER_RC": "3"}, 3, ("train_V0:code_mismatch", "init_check", 0)),
])
def test_payload_failed_step_does_not_end_the_payload(tmp_path, variant, base, fake, rc_want, train_step):
    """No adapter (init-check refused, or trained with other code): the training and merged-eval failures are
    recorded and every independent step still runs: the raw probes (job A), analyze and expect; no RUN/DONE (a stale
    one is removed); exit 1, or 3 for an adapter trained with other code."""
    payload, env, _ = _fake_tree(tmp_path, variant)
    run = tmp_path / "jobs" / f"wb-{variant.lower()}-202611100002"
    run.mkdir(parents=True)
    (run / "DONE").write_text("stale\n")
    res = _bash([str(payload), variant, str(run)] + ([base] if base else []), dict(env, **fake), tmp_path)
    assert res.returncode == rc_want, res.stderr + res.stdout
    steps = _steps(run)
    names = [s["name"] for s in steps]
    rc = {s["name"]: s["rc"] for s in steps}
    tstep, istep, irc = train_step
    assert rc[istep] == irc and rc[tstep] == 1 and rc[f"merge_{variant}:no_adapter"] == 1
    assert names.index(istep) < names.index(tstep) < names.index("analyze") < names.index("expect")
    assert names[-2:] == ["analyze", "expect"] and rc["analyze"] == 0 and rc["expect"] == 1
    assert rc["setup"] == 0 and rc[f"data_{variant}"] == 0
    raw = ["eval_V1_raw_s2sel_off", "eval_V1_raw_s2sel_off_odo", "eval_V1_raw_s2sel_empty"]
    if variant == "V0":
        assert [n for n in names if n.startswith("eval_V1_raw")] == raw and all(rc[n] == 0 for n in raw)
        assert names.index(tstep) < names.index(raw[0]) and names[0] == "b1_precheck"
        assert all(rc[n] == 0 for n in ("b1_precheck", "b1_before", "b1_render", "b1_merge", "b1_after"))
    else:
        assert not any(n.startswith(("eval_V1", "b1")) for n in names) and rc["floors_s2sel"] == 0
    assert not (run / "DONE").exists() and (run / "expect.txt").read_text() == "[variants] fake expect\n"
    log = (tmp_path / "fake.log").read_text()
    assert "n2_train_vpd.py" not in log and "merge.py" not in log  # no training, no merge
    if variant == "V5":
        assert "init-check --init" in log and "--variant V5 --base V1T" in log and "--need-loss-weight" in log
        assert "init-check refused V5" in res.stderr
    assert "failed steps: none" not in res.stdout
    assert not (tmp_path / "merged").exists()


def test_payload_done_only_when_every_step_and_expect_passed(tmp_path):
    """V1T with a finished adapter and fresh evals: every step passes and expect passes -> RUN/DONE, exit 0."""
    payload, env, _ = _fake_tree(tmp_path, "V1T")
    for blind in (False, True):
        d = tmp_path / "X" / "V1T" / "evals" / V.eval_name("V1T", "s2sel", "empty", "line", blind)
        d.mkdir(parents=True)
        (d / "results.json").write_text("{}\n")
    run = tmp_path / "jobs" / "wb-v1t-202611100003"
    fake = {"FAKE_INIT_RC": "0", "FAKE_ADAPTER_RC": "0", "FAKE_MODE": "empty line", "FAKE_EXPECT_RC": "0"}
    res = _bash([str(payload), "V1T", str(run)], dict(env, **fake), tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    names = [s["name"] for s in _steps(run)]
    assert names == ["setup", "data_V1T", "init_check", "train_V1T:skip", "eval_V1T_V1T_s2sel_empty_odo:skip",
                     "eval_V1T_V1T_s2sel_empty_odo_blind:skip", "merge_V1T:skip", "analyze", "expect"]
    assert all(s["rc"] == 0 for s in _steps(run)) and (run / "DONE").exists()
    # the same run with a failing expect list: no DONE (the earlier one is removed), exit 1
    res = _bash([str(payload), "V1T", str(run)], dict(env, **dict(fake, FAKE_EXPECT_RC="1")), tmp_path)
    assert res.returncode == 1 and not (run / "DONE").exists()


def test_payload_b1_partial_dir_aside_and_frozen_guard(tmp_path):
    payload, env, wb = _fake_tree(tmp_path)
    fake = {"FAKE_INIT_RC": "1"}
    # a cut render: B1 out dir with images but without shards/shard0of1.json -> moved aside, then re-rendered
    img = tmp_path / "B1" / "images" / V.B1_STEM / "turn_00.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"j")
    run = tmp_path / "jobs" / "wb-v0v1-202611100004"
    _bash([str(payload), "V0", str(run)], dict(env, **fake), tmp_path)
    names = [s["name"] for s in _steps(run)]
    assert names[:4] == ["b1_precheck", "b1_before", "b1_partial_aside", "b1_render"]
    assert all(s["rc"] == 0 for s in _steps(run)[:6])
    aside = sorted(tmp_path.glob("B1.partial-*"))
    assert len(aside) == 1 and re.fullmatch(r"B1\.partial-\d{8}T\d{6}Z", aside[0].name)
    assert (aside[0] / "images" / V.B1_STEM / "turn_00.jpg").read_bytes() == b"j" and not (tmp_path / "B1").exists()
    # a complete render (its shard manifest exists) is not moved
    (tmp_path / "B1" / "shards").mkdir(parents=True)
    (tmp_path / "B1" / "shards" / "shard0of1.json").write_text("{}")
    run2 = tmp_path / "jobs" / "wb-v0v1-202611100005"
    _bash([str(payload), "V0", str(run2)], dict(env, **fake), tmp_path)
    assert [s["name"] for s in _steps(run2)][:3] == ["b1_precheck", "b1_before", "b1_render:skip"]
    assert len(list(tmp_path.glob("B1.partial-*"))) == 1 and (tmp_path / "B1" / "shards").is_dir()
    # the frozen demos dir (or a dir holding it, or one inside it) is never used as the B1 out dir
    frozen = wb / "demos" / "s2train_640x400"
    (frozen / "images").mkdir(parents=True)
    (frozen / "demos_all.jsonl").write_text("frozen\n")
    for i, b1 in enumerate((frozen, wb / "demos", frozen / "images" / "x")):
        r = tmp_path / "jobs" / f"wb-v0v1-20261110001{i}"
        res = _bash([str(payload), "V0", str(r)], dict(env, VPD_WBV_B1_OUT=str(b1) + "/", **fake), tmp_path)
        steps = _steps(r)
        assert steps[0]["name"] == "b1:frozen_dir_refused" and steps[0]["rc"] == 1, res.stderr
        assert not any(s["name"].startswith("b1_") for s in steps) and "refusing B1 out dir" in res.stderr
        assert [s["name"] for s in steps][1] == "setup" and "expect" in [s["name"] for s in steps]
    assert sorted(p.name for p in frozen.iterdir()) == ["demos_all.jsonl", "images"]
    assert (frozen / "demos_all.jsonl").read_text() == "frozen\n" and not list(wb.glob("demos/*.partial-*"))
