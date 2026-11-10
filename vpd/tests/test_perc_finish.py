"""Tests of vpd/walkerbench/perc_finish.py (EXPLORATORY offline finish of a crashed perception scoring, DECISIONS G5).

Hermetic: a fake perc root under tmp_path with one run dir per start (manifest.json, explore.json, tasks.json,
calls.jsonl, lenient_repairs.jsonl, prompt_dump.jsonl, vllm.log, WalkerBench/{logs,tasks/vpd_perc294,scripts}) whose
scripts/evaluate_all.py is a small stand-in holding the stock functions that matter (b197f51: haversine,
calculate_trajectory_length, evaluate_session's perception branch), a fake job RUN with job/worker.log, and the
perc294 selection of test_perc's fake checkout.  The real run_vpd.finalize / score run on it (hard subsets and split
files off); the image-token audit is off or stubbed.  No GPU, no model, no network.  One read-only check rebuilds the
real Q3B run's meta and compares it with its results.json (skips when absent).
    cd $WT && env OMP_NUM_THREADS=1 ... nice -n 19 $PY -B -m pytest -q -p no:cacheprovider --basetemp=<dir> \
        vpd/tests/test_perc_finish.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_WT = Path(__file__).resolve().parents[2]
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

import pytest  # noqa: E402

from vpd.tests.test_perc import SMALL, built_set  # noqa: E402
from vpd.walkerbench import perc as P  # noqa: E402
from vpd.walkerbench import perc_finish as F  # noqa: E402


@pytest.fixture
def R(monkeypatch):
    try:
        from vpd.walkerbench import run_vpd
    except ImportError as e:  # an interpreter without eval_external deps
        pytest.skip(f"run_vpd not importable here: {e}")
    monkeypatch.setattr(run_vpd.score, "__defaults__", (None, None, None))  # no real hard_subsets / split files
    monkeypatch.delitem(sys.modules, F.EV_MODULE, raising=False)
    return run_vpd


# ------------------------------------------------------------------------------------------ fake evaluator
FAKE_EVALUATE_ALL = '''"""Stand-in for WalkerBench scripts/evaluate_all.py (b197f51): the stock functions that matter, copied."""
import json
import math
import re
from pathlib import Path

NAV_SUCCESS_THRESHOLD_M = 15.0
TOL = {"height": 0.15, "dis": 0.2}


def haversine(lat1, lng1, lat2, lng2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def calculate_trajectory_length(events):
    points = []
    for e in events:
        if e.get("event") == "session_start":
            state = e.get("initial_state", {})
            if "lat" in state and "lng" in state:
                points.append((state["lat"], state["lng"]))
        elif e.get("event") == "action":
            state = e.get("state", {})
            if "lat" in state and "lng" in state:
                pt = (state["lat"], state["lng"])
                if not points or points[-1] != pt:
                    points.append(pt)
    total_len = 0.0
    for i in range(len(points) - 1):
        total_len += haversine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
    return total_len


def load_task_config(task_id, custom_tasks_dir=None):
    p = Path(custom_tasks_dir) / f"{task_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def extract_number(s):
    if s is None:
        return None
    m = re.findall(r"[-+]?\\d*\\.?\\d+", str(s))
    return float(m[0]) if m else None


def reconstruct_path_from_events(events, start_pano=None):
    return [start_pano] if start_pano else []


def get_distance(a, b, local=None):
    return float("inf")


def evaluate_session(log_file, custom_tasks_dir=None):
    events = [json.loads(x) for x in open(log_file) if x.strip()]
    start = next(e for e in events if e.get("event") == "session_start")
    tid = start["task_id"]
    ttype = tid.split("_", 1)[0]
    cfg = load_task_config(tid, custom_tasks_dir)
    gt = cfg.get("ground_truth", {})
    steps = sum(1 for e in events if e.get("event") == "action")
    traj_len = calculate_trajectory_length(events)
    res = {"agent": start.get("agent_id"), "task_id": tid, "task_type": ttype, "success": 0, "spl": 0.0,
           "steps": steps, "length": traj_len, "error": None}
    ans = next((e["action"].get("answer") for e in reversed(events)
                if e.get("event") == "action" and e["action"].get("type") == "stop"), None)
    pred = extract_number(ans)
    if pred is None:
        return res
    if ttype == "angle":
        d = abs(pred - gt["bearing_a_to_b_deg"]) % 360
        res["error"] = min(d, 360 - d)
        res["success"] = int(res["error"] <= 30.0)
    else:
        truth = gt["height_meters"] if ttype == "height" else gt["distance_between_pois_m"]
        res["error"] = abs(pred - truth)
        res["success"] = int(abs(pred - truth) / truth <= TOL[ttype])
    return res
'''

GT = {"angle": {"bearing_a_to_b_deg": 45.0}, "dis": {"distance_between_pois_m": 100.0},
      "height": {"height_meters": 30.0}}
A, B, C = (52.0, -1.0), (52.0001, -1.0), (52.0002, -1.0001)
NULL = (None, None)


def _mv():
    return {"type": "move", "move_id": 1}


def _stop(ans):
    return {"type": "stop", "answer": ans}


# stem -> (success, [(action, position, reason)]) ; the spawn is at A
EPISODES = {
    "angle_0001_a": (1, [(_mv(), B, ""), (_stop("45"), B, "")]),
    "angle_0002_a": (0, [(_stop("200"), A, "")]),
    "dis_0001_d": (1, [(_stop("110 m"), A, "")]),
    "dis_0002_d": (0, [(_mv(), B, ""), (_mv(), NULL, ""), (_mv(), NULL, "[vpd: invalid_move_crash] No available moves")]),
    "height_0001_h": (1, [(_mv(), B, ""), (_mv(), NULL, ""), (_mv(), C, ""), (_stop("31 m"), C, "")]),
    "height_0002_h": (0, [(_stop("10"), A, "Invalid action format: x")]),
}
NULL_LOGS = {"dis_0002_d": 2, "height_0001_h": 1}


def _hav(p, q):
    r = 6371000
    p1, p2 = math.radians(p[0]), math.radians(q[0])
    dp, dl = math.radians(q[0] - p[0]), math.radians(q[1] - p[1])
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _state(pos):
    return {"pano_id": f"p{pos}", "heading": 0.0, "pitch": 0, "fov": 90, "lat": pos[0], "lng": pos[1]}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tree(d: Path) -> dict:
    """path -> (sha256, mtime_ns) of every file under d, and every dir's mtime."""
    out = {}
    for dp, dirs, files in os.walk(d):
        out[dp] = os.stat(dp).st_mtime_ns
        for f in files:
            p = Path(dp) / f
            out[str(p)] = (_sha(p), p.stat().st_mtime_ns)
    return out


TRACEBACK = """[c0q3_sft_e8] [Task height_0247] [Step 30] WARN: Reason='Failed to parse response'. RAW: 'xx'Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "{code}/vpd/walkerbench/perc.py", line 517, in cmd_eval
    R.main(rv)
  File "{code}/vpd/walkerbench/run_vpd.py", line 991, in main
    res, valid = finalize(run_dir, copy, tasks_dir, tasks, meta=meta, server_alive=server_alive, wall=wall,
  File "{code}/vpd/walkerbench/run_vpd.py", line 836, in finalize
    res = score(copy, tasks, tasks_dir, calls_log=run_dir / "calls.jsonl")
  File "{code}/vpd/walkerbench/run_vpd.py", line 652, in score
    r = ev.evaluate_session(log, tasks_dir)
  File "{run_dir}/WalkerBench/scripts/evaluate_all.py", line 234, in evaluate_session
    traj_len = calculate_trajectory_length(events)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "{run_dir}/WalkerBench/scripts/evaluate_all.py", line 175, in calculate_trajectory_length
    total_len += haversine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "{run_dir}/WalkerBench/scripts/evaluate_all.py", line 103, in haversine
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
                                     ^^^^^^^^^^^^^^^^^^
TypeError: must be real number, not NoneType
"""


def fake_job_run(base: Path, root: Path, start: str, R, *, episodes=EPISODES, worker_tail=None, shutdown=True,
                 explore_tamper=None, manifest_tamper=None) -> Path:
    """One crashed perc run of `start` under root (as vpd-wb-perc left Q3S / E2) and its job RUN."""
    run = base / "jobs" / f"wb-perc-{start}"
    (run / "job").mkdir(parents=True, exist_ok=True)
    d = P.run_dir_of(start, root)
    copy = d / "WalkerBench"
    tdir = copy / "tasks" / "vpd_perc294"
    for sub in ("logs", "scripts"):
        (copy / sub).mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    (copy / "scripts" / "evaluate_all.py").write_text(FAKE_EVALUATE_ALL)
    info = R.resolve_model(start)
    calls, tasks = [], sorted(episodes)
    for stem, (_, steps) in episodes.items():
        t = stem.split("_", 1)[0]
        (tdir / f"{stem}.json").write_text(json.dumps({"task_id": stem, "ground_truth": GT[t],
                                                       "max_steps": None if t == "height" else 20}))
        ev = [{"event": "session_start", "agent_id": info["served"], "task_id": stem, "timestamp": "20261110_160625",
               "initial_state": _state(A)}]
        for i, (act, pos, reason) in enumerate(steps):
            ev.append({"event": "action", "step": i + 1, "timestamp": f"2026-11-10T16:06:{30 + i:02d}.5",
                       "state": _state(pos), "action": act, "reason": reason,
                       "available_moves": [] if pos == NULL else [{"id": 1}]})
            calls.append({"task": stem, "step": i + 1, "pattern": "s u", "n_images": i + 1, "n_moves": 1,
                          "max_tokens": 1024, "constrained": False, "prompt_tokens": 1000 + 500 * i,
                          "completion_tokens": 20, "finish_reason": "length" if stem == "height_0002_h" else "stop",
                          "reply": "x"})
        (copy / "logs" / f"{info['served']}_{stem}_20261110_160625.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in ev))
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (d / "prompt_dump.jsonl").write_text(json.dumps({"task": tasks[0], "messages": []}) + "\n")
    (d / "lenient_repairs.jsonl").write_text(json.dumps({"original": "a", "repaired": "{}"}) + "\n"
                                             + json.dumps({"original": "b", "repaired": None}) + "\n")
    (d / "vllm.log").write_text("[vllm_server_vpd] [...]\nINFO: Shutting down\nINFO: Waiting for application shutdown.\n"
                                + ("INFO:     Application shutdown complete.\n" if shutdown else "INFO: still up\n"))
    (d / "tasks.json").write_text(json.dumps(tasks) + "\n")
    server = {"source": "x", "code": info["code"], "model_dir": None, "model_path": "/views/x",
              "cfg": {"max_model_len": 128000, "max_images": 80, "gpu_memory_utilization": 0.85}, "port": 1, "threads": 4}
    manifest = {
        "models": [{"start": info["start"], "code": info["code"], "served_model_name": info["served"]}],
        "backend": {"render_size": [640, 400], "server": server, "workers": 8,
                    "history": {"turns": 0, "rule": "full history"}},
        "prompt": {"images": "sent (local file)", "json_repair": "lenient local rule (diagnostic): ...",
                   "constrained": {"mode": "off", "thought_max_chars": 0, "example_regex_3_moves": None}},
        "decoding": {"temperature": 0.0, "max_tokens": 1024, "structured_outputs": False},
        "subset": {"kind": "full", "limit": None, "seed": None},
        "extra": {"vpd": {"model_dir": None, "adapter": None, "run_name": P.run_name(start), "task_set": "perc294"}}}
    if manifest_tamper:
        manifest_tamper(manifest)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    ex = {"schema": P.SCHEMA + ":eval", "start": start, "rc": 1, "run_vpd_argv": P.protocol_argv(start, root),
          "run": str(run), "utc": "2026-11-10T16:29:32Z", "code_snapshot": {"mode": "snapshot", "commit": "5614"},
          "task_sets_added": {"perc294": str(base / "perception" / "perc294" / "tasks")},
          "code": {"vpd/walkerbench/run_vpd.py": _sha(Path(R.__file__))}}
    if explore_tamper:
        explore_tamper(ex)
    (d / "explore.json").write_text(json.dumps(ex, indent=1))
    log = run / "job" / "worker.log"
    old = log.read_text() if log.exists() else "[vpd_step] eval_Q3B end 2026-11-10T16:04:25Z rc=0\n"
    tail = TRACEBACK.format(code=run / "code", run_dir=d) if worker_tail is None else worker_tail
    log.write_text(old + f"[vpd_step] eval_{start} start x\n" + tail + f"[vpd_step] eval_{start} end y rc=1\n")
    return d


@pytest.fixture
def world(tmp_path, R):
    _, _, dest = built_set(tmp_path)  # perc294 selection of the same 6 stems (test_perc's fake checkout)
    root = tmp_path / "R"
    return {"base": tmp_path, "root": root, "dest": dest}


# ------------------------------------------------------------------------------------------ the patch
def test_wrap_drops_only_null_points(tmp_path, R):
    ev = R.load_module("wb_fake_ev_test", _write(tmp_path / "ev.py", FAKE_EVALUATE_ALL))
    stock = ev.calculate_trajectory_length
    events = [{"event": "session_start", "initial_state": _state(A)},
              {"event": "action", "state": _state(B)}, {"event": "action", "state": _state(NULL)},
              {"event": "action", "state": {**_state(C), "lng": None}}, {"event": "action", "state": _state(C)},
              {"event": "other", "state": _state(NULL)}]
    with pytest.raises(TypeError, match="must be real number, not NoneType"):
        stock(events)  # the job's crash
    stats = {}
    fixed = F.wrap_trajectory_length(stock, stats)
    assert fixed(events) == pytest.approx(_hav(A, B) + _hav(B, C))
    assert events[2]["state"]["lat"] is None and "lat" in events[3]["state"]  # input unchanged
    clean = [e for i, e in enumerate(events) if i not in (2, 3)]
    assert fixed(clean) == stock(clean)  # no null state: the stock function on the same events
    assert stats == {"calls": 2, "calls_with_null": 1, "null_events": 2}
    assert fixed.__wrapped__ is stock and fixed.vpd_perc_finish
    assert [F.null_coord_state(e) is not None for e in events] == [False, False, True, True, False, False]
    only_lat = {"event": "action", "state": {"pano_id": "x", "lat": None}}  # the stock code skips it (no lng key)
    assert F.null_coord_state(only_lat) is None


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def test_patched_evaluator_wraps_only_the_evaluate_all_module(tmp_path, R):
    orig = R.load_module
    stats = {}
    with F.patched_evaluator(R, stats):
        assert R.load_module is not orig
        other = R.load_module("wb_other_test", _write(tmp_path / "o.py", FAKE_EVALUATE_ALL))
        assert not getattr(other.calculate_trajectory_length, "vpd_perc_finish", False)
        ev = R.load_module(F.EV_MODULE, _write(tmp_path / "evaluate_all.py", FAKE_EVALUATE_ALL))
        assert ev.calculate_trajectory_length.vpd_perc_finish
    assert R.load_module is orig and stats["patched"] == [str(tmp_path / "evaluate_all.py")]
    assert "vpd_perc_finish" not in (tmp_path / "evaluate_all.py").read_text()  # the file is not edited


# ------------------------------------------------------------------------------------------ end to end
def test_unpatched_scoring_reproduces_the_crash(world, R):
    d = fake_job_run(world["base"], world["root"], "Q3S", R)
    tasks = json.loads((d / "tasks.json").read_text())
    copy = d / "WalkerBench"
    with pytest.raises(TypeError, match="must be real number, not NoneType"):
        R.score(copy, tasks, copy / "tasks" / "vpd_perc294", calls_log=d / "calls.jsonl")


def test_finish_end_to_end_and_perc_accepts_it(world, R, monkeypatch):
    root, base = world["root"], world["base"]
    d = fake_job_run(base, root, "Q3S", R)
    seen = {}

    def fake_audit(copy, processor_dir, prompt_dump=None, **kw):
        seen["audit"] = (str(copy), str(processor_dir), str(prompt_dump), os.environ.get("CUDA_VISIBLE_DEVICES"))
        return {"tokens_per_image": [242], "processor_dir": str(processor_dir)}

    monkeypatch.setattr(R, "image_token_audit", fake_audit)
    R.COUNTS["leftover_from_another_test"] = 7
    explore_before, ev_before = _sha(d / "explore.json"), _sha(d / "WalkerBench" / "scripts" / "evaluate_all.py")
    orig_load = R.load_module
    out = F.finish("Q3S", root=root, expected=SMALL, R=R)
    assert out["valid"] and out["results_file"] == "results.json" and out["n_null_events"] == 3
    assert R.load_module is orig_load and dict(R.COUNTS) == {"leftover_from_another_test": 7}
    R.COUNTS.clear()
    assert _sha(d / "explore.json") == explore_before  # the job's record is not rewritten
    assert _sha(d / "WalkerBench" / "scripts" / "evaluate_all.py") == ev_before
    res = json.loads((d / "results.json").read_text())
    assert not (d / "results_invalid.json").exists()
    # the meta run_vpd.main would have passed
    for k, v in P.protocol_want("Q3S").items():
        assert res[k] == v, k
    assert set(R.RESULT_META_KEYS) <= set(res)
    assert res["max_step_cap"] == 30 and res["max_images"] == 80 and res["thought_max_chars"] == 0
    assert res["code"] == R.resolve_model("Q3S")["code"] and res["server"]["ready_seconds"] is None
    assert res["server_alive_at_end"] is True and res["wall_seconds"] is None and res["seconds_per_step"] is None
    assert res["counts"] == {"model_calls": 12, "finish_length": 1, "prompt_dumped": 1, "lenient_repairs": 1,
                             "lenient_no_action": 1, "invalid_move_crash": 1, "tasks_without_log": 0}
    fo = res["finished_offline"]
    assert fo["by"] == F.MODULE and fo["exploratory"] and fo["explore_rc"] == 1 and fo["n_null_events"] == 3
    assert fo["logs_with_null_coords"] == 2 and fo["server_alive_at_end"]["value"] is True
    assert fo["wall_estimate_seconds_from_logs"] == 8.5 and "wall_seconds" in fo["not_recorded_by_the_job"]
    assert res["image_tokens"] == {"tokens_per_image": [242], "processor_dir": seen["audit"][1]}
    assert seen["audit"][0] == str(d / "WalkerBench") and seen["audit"][2] == str(d / "prompt_dump.jsonl")
    # numbers: success is answer-based, untouched by the patch; only length of the null logs is affected
    assert [(res["per_kind"][t]["k"], res["per_kind"][t]["n"]) for t in P.TYPES] == [(1, 2), (1, 2), (1, 2)]
    pt = res["per_task"]
    assert pt["height_0001_h"]["length"] == pytest.approx(_hav(A, B) + _hav(B, C))
    assert pt["dis_0002_d"]["length"] == pytest.approx(_hav(A, B))
    assert pt["angle_0001_a"]["length"] == pytest.approx(_hav(A, B)) and pt["angle_0002_a"]["length"] == 0.0
    assert (res["invalid_move_crash"], res["unparseable_steps"], res["steps_logged"]) == (1, 1, 12)
    rows = [json.loads(x) for x in (d / "per_task.jsonl").read_text().splitlines()]
    assert [r["stem"] for r in rows] == sorted(EPISODES) and sum(r["success"] for r in rows) == 3
    # finish.json
    fj = json.loads((d / "finish.json").read_text())
    assert fj["schema"] == F.SCHEMA and fj["exploratory"] is True and fj["label"] == "EXPLORATORY"
    assert fj["utc"].endswith("Z") and fj["why"] == F.WHY and fj["patch"] == F.PATCH
    names = {stem: f"{R.resolve_model('Q3S')['served']}_{stem}_20261110_160625.jsonl" for stem in NULL_LOGS}
    assert fj["logs_with_null_coords"] == sorted(names.values()) and fj["n_null_events"] == 3
    assert fj["null_events_per_log"] == {names[s]: n for s, n in NULL_LOGS.items()}
    assert fj["patch_stats"]["calls"] == 6 and fj["patch_stats"]["calls_with_null"] == 2
    assert fj["code_sha256"] == {"vpd/walkerbench/perc_finish.py": _sha(Path(F.__file__)),
                                 "vpd/walkerbench/run_vpd.py": _sha(Path(R.__file__))}
    assert fj["job"]["run"] == str(base / "jobs" / "wb-perc-Q3S") and fj["job"]["explore_rc"] == 1
    assert fj["job"]["crash"]["error"] == F.CRASH_ERROR and fj["job"]["crash"]["step"].endswith("rc=1")
    assert fj["lengths_of_null_logs"]["height_0001_h"] == pytest.approx(_hav(A, B) + _hav(B, C))
    assert fj["sandbox_side_effects"] == {"added": [], "removed": [], "changed": []}
    assert fj["evaluator"]["sha256"] == ev_before and fj["valid"] is True
    assert fj["python"]["executable"] == sys.executable
    # perc analyze / expect accept the start (explore.json rc 1 is recorded, not required to be 0)
    doc = P.analyze(root, world["dest"], starts=("Q3S",), R=R)
    a = doc["starts"]["Q3S"]
    assert a["state"] == "ok" and a["protocol_ok"] and a["consistent"], (a["protocol_problems"],
                                                                          a["consistency_problems"])
    assert a["produced_by"]["rc"] == 1 and a["totals"]["wall_seconds"] is None
    lines, bad = P.expect(root, world["dest"], starts=("Q3S",))
    assert bad == 0, lines
    # a second finish: refused, unless --overwrite-finish (its own output)
    with pytest.raises(F.Refused, match="already has"):
        F.finish("Q3S", root=root, expected=SMALL, R=R, audit=False)
    out2 = F.finish("Q3S", root=root, expected=SMALL, R=R, audit=False, overwrite_finish=True)
    assert out2["valid"] and json.loads((d / "finish.json").read_text())["replaced_own_output"] == [
        "results.json", "per_task.jsonl", "finish.json"]
    assert json.loads((d / "results.json").read_text())["image_tokens"] is None  # audit off this time
    R.COUNTS.clear()


def test_overwrite_finish_never_replaces_a_job_result(world, R):
    d = fake_job_run(world["base"], world["root"], "E2", R)
    for name in ("results.json", "results_invalid.json"):
        (d / name).write_text(json.dumps({"schema": R.SCHEMA}))  # as run_vpd.main writes it: no finished_offline
        for flag in (False, True):
            with pytest.raises(F.Refused, match="already has" if not flag else "not this module's output"):
                F.finish("E2", root=world["root"], expected=SMALL, R=R, audit=False, overwrite_finish=flag)
        (d / name).unlink()
    (d / "per_task.jsonl").write_text("{}\n")  # a per_task.jsonl without a result file of ours
    with pytest.raises(F.Refused, match="per_task.jsonl"):
        F.finish("E2", root=world["root"], expected=SMALL, R=R, audit=False, overwrite_finish=True)


def test_dry_run_writes_nothing(world, R):
    d = fake_job_run(world["base"], world["root"], "E2", R)
    before = _tree(world["base"])
    out = F.finish("E2", root=world["root"], expected=SMALL, R=R, dry_run=True)
    assert _tree(world["base"]) == before
    assert out["dry_run"] and out["n_tasks"] == 6 and out["n_logs"] == 6 and out["n_null_events"] == 3
    assert out["meta"]["run_name"] == P.run_name("E2") and out["meta"]["counts"]["tasks_without_log"] == 0
    assert out["crash"]["error"] == F.CRASH_ERROR and out["server_alive_at_end"]["value"] is True
    assert F.EV_MODULE not in sys.modules  # the sandbox evaluator is not loaded
    assert not any((d / n).exists() for n in F.OUTPUTS)


def test_server_not_shown_alive_gives_results_invalid(world, R):
    d = fake_job_run(world["base"], world["root"], "Q3S", R, shutdown=False)
    out = F.finish("Q3S", root=world["root"], expected=SMALL, R=R, audit=False)
    assert not out["valid"] and out["results_file"] == "results_invalid.json"
    assert (d / "results_invalid.json").exists() and not (d / "results.json").exists()
    res = json.loads((d / "results_invalid.json").read_text())
    assert res["server_alive_at_end"] is False and res["finished_offline"]["server_alive_at_end"][
        "vllm_log_orderly_shutdown"] is False
    R.COUNTS.clear()


def _drop_task(d: Path):
    t = json.loads((d / "tasks.json").read_text())
    (d / "tasks.json").write_text(json.dumps(t[:-1]))


def _dup_log(d: Path):
    logs = sorted((d / "WalkerBench" / "logs").glob("*.jsonl"))
    (logs[0].parent / ("dup_" + logs[0].name)).write_bytes(logs[0].read_bytes())


def _rm_log(d: Path):
    sorted((d / "WalkerBench" / "logs").glob("*.jsonl"))[0].unlink()


def _no_nulls(d: Path):
    for p in (d / "WalkerBench" / "logs").glob("*.jsonl"):
        p.write_text(p.read_text().replace('"lat": null, "lng": null', '"lat": 1.0, "lng": 1.0'))


@pytest.mark.parametrize("kind, tamper, kw, match", [
    ("tasks_count", _drop_task, {}, "tasks.json"),
    ("duplicate_log", _dup_log, {}, "exactly one session log"),
    ("missing_log", _rm_log, {}, "exactly one session log"),
    ("no_nulls", _no_nulls, {}, "not the documented one"),
    ("other_error", None, {"worker_tail": "Traceback (most recent call last):\nValueError: boom\n"}, "worker.log"),
    ("other_run_dir", None, {"worker_tail": TRACEBACK.format(code="/c", run_dir="/elsewhere")}, "worker.log"),
    ("rc0", None, {"explore_tamper": lambda e: e.update(rc=0)}, "rc 0"),
    ("argv", None, {"explore_tamper": lambda e: e["run_vpd_argv"].__setitem__(-5, "4")}, "protocol_argv"),
    ("run_vpd_sha", None, {"explore_tamper": lambda e: e["code"].update({"vpd/walkerbench/run_vpd.py": "ab"})},
     "run_vpd.py sha256"),
    ("workers", None, {"manifest_tamper": lambda m: m["backend"].update(workers=4)}, "workers"),
    ("constrained", None, {"manifest_tamper": lambda m: m["prompt"]["constrained"].update(mode="empty")},
     "constrained"),
    ("no_manifest_field", None, {"manifest_tamper": lambda m: m.pop("decoding")}, "lacks a field"),
])
def test_refusals(world, R, kind, tamper, kw, match):
    d = fake_job_run(world["base"], world["root"], "Q3S", R, **kw)
    if tamper:
        tamper(d)
    before = _tree(d)
    with pytest.raises(F.Refused, match=match):
        F.finish("Q3S", root=world["root"], expected=SMALL, R=R, audit=False)
    assert _tree(d) == before  # a refusal writes nothing


def test_one_start_per_process(world, R, monkeypatch, tmp_path):
    d = fake_job_run(world["base"], world["root"], "Q3S", R)
    other = type(sys)("engine")
    other.__file__ = str(tmp_path / "other_copy" / "engine" / "__init__.py")
    monkeypatch.setitem(sys.modules, "engine", other)  # another copy's engine package already imported
    with pytest.raises(RuntimeError, match="one start per process"):
        F.finish("Q3S", root=world["root"], expected=SMALL, R=R, audit=False)
    assert not any((d / n).exists() for n in F.OUTPUTS)


def test_refuses_a_wrong_294_inventory_by_default(world, R):
    fake_job_run(world["base"], world["root"], "Q3S", R)
    with pytest.raises(F.Refused, match="expected 294 distinct tasks"):
        F.finish("Q3S", root=world["root"], R=R, dry_run=True)  # the real EXPECTED: 104 / 107 / 83


def test_cli_only_the_two_crashed_starts():
    for bad in (["--start", "Q3B"], ["--start", "Q3RD"], []):
        with pytest.raises(SystemExit) as e:
            F.main(bad)
        assert e.value.code == 2
    assert F.FINISH_STARTS == ("Q3S", "E2") and set(F.FINISH_STARTS) <= set(P.STARTS)


# ------------------------------------------------------------------------------------------ real data (read only)
def test_rebuilt_meta_matches_the_real_q3b_run(R):
    """Q3B's results.json was written by run_vpd.main in the same job: the meta and counts rebuilt from its records
    must equal it (server.ready_seconds and the image counters are what the job alone knew)."""
    d = P.run_dir_of("Q3B")
    if not (d / "results.json").exists():
        pytest.skip(f"{d}/results.json absent")
    res = json.loads((d / "results.json").read_text())
    probs = []
    manifest = json.loads((d / "manifest.json").read_text())
    ex = json.loads((d / "explore.json").read_text())
    tdir = d / "WalkerBench" / "tasks" / res["tasks_label"]
    meta = F.rebuild_meta(d, "Q3B", manifest, ex, tdir, R, probs)
    assert probs == []
    for k in R.RESULT_META_KEYS:
        want = dict(res[k], ready_seconds=None) if k == "server" else res[k]
        assert json.loads(json.dumps(meta[k])) == want, k
    tasks = json.loads((d / "tasks.json").read_text())
    counts = F.rebuild_counts(d, R.read_calls(d / "calls.jsonl"), F.scan_logs(d / "WalkerBench" / "logs", R), tasks)
    assert counts == {k: v for k, v in res["counts"].items() if k != "images_sent"}
