"""Tests of the EXPLORATORY perception zero-shot package (DECISIONS G5): vpd/walkerbench/perc.py, vpd/jobs/wb_perc.sh.

Hermetic: a fake WalkerBench checkout and panorama dir under tmp_path for the task-set build / verify / refusals, a
fake run_vpd module for the eval wrapper, synthetic run dirs (results.json, per_task.jsonl, logs, calls.jsonl) for
the analysis, and the payload in VPD_DRYRUN=1 mode.  No GPU, no model, no network, no rendering.  Two read-only checks
look at the real checkout (the 104 / 107 / 83 inventory) and at the built OUT perc294 dir; they skip when absent.
    cd $WT && env OMP_NUM_THREADS=1 ... nice -n 19 $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=<dir> \
        vpd/tests/test_perc.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_WT = Path(__file__).resolve().parents[2]
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

import pytest  # noqa: E402

from vpd import paths  # noqa: E402
from vpd.walkerbench import perc as P  # noqa: E402

PAYLOAD = paths.VPD_PKG / "jobs" / "wb_perc.sh"
if not PAYLOAD.exists():  # run from a code snapshot (RUN/code)
    PAYLOAD = _WT / "vpd" / "jobs" / "wb_perc.sh"


def run_vpd():
    try:
        from vpd.walkerbench import run_vpd as R
    except ImportError as e:  # an interpreter without eval_external deps
        pytest.skip(f"run_vpd not importable here: {e}")
    return R


# ------------------------------------------------------------------------------------------ fake checkout
FULL = {"angle_0001_a": "Pa1", "angle_0002_a": "Pa2", "dis_0002_d": "Pd2", "height_0001_h": "Ph1",
        "height_0002_h": "Ph2", "height_0003_h": "Ph3"}
STANDARD = {"angle_0001_a": "Pa1", "dis_0001_d": "Pd1", "height_0002_h": "Ph2"}
PANOS_PRESENT = ("Pa1", "Pa2", "Pd1", "Pd2", "Ph1", "Ph2")  # Ph3 missing -> height_0003_h dropped
SMALL = {"angle": 2, "dis": 2, "height": 2}


def _task(stem: str, spawn: str) -> bytes:
    t = stem.split("_", 1)[0]
    return json.dumps({"task_id": stem, "task_type": t, "spawn_pano_id": spawn,
                       "max_steps": None if t == "height" else 20}, indent=1).encode()


def fake_repo(base: Path, full=FULL, standard=STANDARD, panos=PANOS_PRESENT) -> tuple:
    repo, pdir = base / "WalkerBench", base / "panoramas"
    for src, files in (("full", full), ("standard", standard)):
        d = repo / "tasks" / src
        d.mkdir(parents=True)
        for stem, spawn in files.items():
            (d / f"{stem}.json").write_bytes(_task(stem, spawn))
        (d / f"nav_0009_{src}.json").write_text(json.dumps({"task_id": "nav", "spawn_point": "Pa1"}))  # not perception
    pdir.mkdir(parents=True)
    for p in panos:
        (pdir / f"{p}_z2.jpg").write_bytes(b"jpg")
    (pdir / "Ph3_z3.jpg").write_bytes(b"other zoom")  # only z2 counts
    return repo, pdir


def built_set(base: Path) -> tuple:
    repo, pdir = fake_repo(base)
    dest = base / "perception" / "perc294"
    P.build_set(dest, repo, pdir, SMALL)
    return repo, pdir, dest


# ------------------------------------------------------------------------------------------ pure helpers
def test_wilson_rate_and_number():
    lo, _ = P.wilson(28, 65)
    assert round(lo, 3) == 0.318 and round(P.wilson(27, 65)[0], 3) == 0.304  # the frozen bars (critic C5 check)
    assert P.rate(0, 0) == {"k": 0, "n": 0, "rate": None, "wilson95": None}
    r = P.rate(39, 100)
    assert r["rate"] == 0.39 and r["wilson95"][0] < 0.39 < r["wilson95"][1]
    assert P.extract_number("45.5 meters") == 45.5 and P.extract_number("-12") == -12.0
    assert P.extract_number(30) == 30.0 and P.extract_number("ARRIVED") is None and P.extract_number("") is None
    assert P.extract_number(None) is None and P.extract_number("about 1.5e2") == 1.5
    assert P.task_cap({"max_steps": None}) == 30 and P.task_cap({"max_steps": 20}) == 20
    assert P.task_cap({"visual_path": [1, 2, 3], "max_steps": 99}) == 4


def test_refuse_pilot():
    with pytest.raises(SystemExit, match="read only"):
        P.refuse_pilot(P.PILOT_ROOT / "Q3B" / "x")
    P.refuse_pilot(P.ROOT)
    assert not P.is_under(P.ROOT, P.PILOT_ROOT) and not P.is_under(P.PERC_DIR, P.PILOT_ROOT)
    assert P.run_dir_of("E2") == P.ROOT / "E2" / "evals" / "explore__E2__raw__perc294__640__off_lenient"


def test_protocol_argv_identical_for_every_start(tmp_path):
    vary = {"--model", "--run-name", "--out-root"}

    def fixed(argv):
        out, skip = [], False
        for a in argv:
            if skip:
                skip = False
                continue
            if a in vary:
                skip = True
                continue
            out.append(a)
        return out

    argvs = {s: P.protocol_argv(s, tmp_path) for s in P.STARTS}
    assert len({tuple(fixed(a)) for a in argvs.values()}) == 1
    a = argvs["Q3S"]
    assert " ".join(fixed(a)) == ("--task-set perc294 --render-size 640x400 --constrained off --lenient "
                                  "--max-tokens 1024 --history-turns 0 --temperature 0 --workers 8")
    assert a[a.index("--model") + 1] == "Q3S" and a[a.index("--out-root") + 1] == str(tmp_path / "Q3S" / "evals")
    assert a[a.index("--run-name") + 1] == "explore__Q3S__raw__perc294__640__off_lenient"
    with pytest.raises(ValueError):
        P.protocol_argv("Q3RD", tmp_path)


# ------------------------------------------------------------------------------------------ task set
def test_plan_flags_and_drops(tmp_path):
    repo, pdir = fake_repo(tmp_path)
    plan = P.plan_set(repo, pdir, SMALL)
    assert plan["n_tasks"] == 6 and plan["exploratory"] and plan["label"] == "EXPLORATORY"
    c = plan["counts"]
    assert {t: (c[t]["kept"], c[t]["paper_full"], c[t]["standard20"], c[t]["dropped_no_panorama"]) for t in P.TYPES} \
        == {"angle": (2, 2, 1, 0), "dis": (2, 1, 1, 0), "height": (2, 2, 1, 1)}
    assert c["height"]["step_cap"] == [30] and c["angle"]["step_cap"] == [20]
    e = {x["file"]: x for x in plan["entries"]}
    assert e["angle_0001_a.json"]["sources"] == ["full", "standard"] and e["angle_0001_a.json"]["standard20"]
    assert e["dis_0001_d.json"]["sources"] == ["standard"] and not e["dis_0001_d.json"]["paper_full"]
    assert [d["file"] for d in plan["dropped"]] == ["height_0003_h.json"]
    assert plan["same_bytes_other_name"] == [] and "nav_0009_full.json" not in plan["files"]
    assert list(plan["files"]) == sorted(plan["files"])
    with pytest.raises(SystemExit, match="differ from the expected"):
        P.plan_set(repo, pdir, P.EXPECTED)


def test_plan_refuses_a_name_clash_and_lists_same_bytes(tmp_path):
    repo, pdir = fake_repo(tmp_path / "a")
    (repo / "tasks" / "standard" / "angle_0001_a.json").write_text('{"spawn_pano_id": "Pa1", "x": 1}')
    with pytest.raises(SystemExit, match="different bytes"):
        P.plan_set(repo, pdir, None)
    repo, pdir = fake_repo(tmp_path / "b")
    (repo / "tasks" / "standard" / "dis_0003_d.json").write_bytes(
        (repo / "tasks" / "full" / "dis_0002_d.json").read_bytes())
    plan = P.plan_set(repo, pdir, None)
    assert plan["same_bytes_other_name"] == [["dis_0002_d.json", "dis_0003_d.json"]]
    assert plan["counts"]["dis"]["kept"] == 3


def test_build_verify_and_refuse(tmp_path):
    repo, pdir = fake_repo(tmp_path)
    dest = tmp_path / "perception" / "perc294"
    with pytest.raises(SystemExit, match="missing"):
        P.build_set(dest, repo, pdir, SMALL, check_only=True)
    r = P.build_set(dest, repo, pdir, SMALL)
    assert r["state"] == "created" and r["n_tasks"] == 6 and (r["angle"], r["dis"], r["height"]) == (2, 2, 2)
    assert not list(dest.parent.glob(".perc294.tmp-*"))
    sel = json.loads((dest / "selection.json").read_text())
    assert sel["exploratory"] and sel["counts"]["height"]["dropped_no_panorama"] == 1 and "created_utc" in sel
    for f, sha in sel["files"].items():
        assert P.sha256_file(dest / "tasks" / f) == sha
        src = next(s for s in P.SOURCES if (repo / "tasks" / s / f).exists())
        assert (dest / "tasks" / f).read_bytes() == (repo / "tasks" / src / f).read_bytes()
    assert P.build_set(dest, repo, pdir, SMALL)["state"] == "verified"
    assert P.build_set(dest, repo, pdir, SMALL, check_only=True)["state"] == "verified"
    (dest / "tasks" / "dis_0001_d.json").write_text("{}")
    with pytest.raises(SystemExit, match="sha256 differs"):
        P.build_set(dest, repo, pdir, SMALL, check_only=True)
    (dest / "tasks" / "dis_0001_d.json").write_bytes((repo / "tasks" / "standard" / "dis_0001_d.json").read_bytes())
    (pdir / "Ph1_z2.jpg").unlink()  # the plan changes: height_0001_h would now be dropped
    with pytest.raises(SystemExit):
        P.build_set(dest, repo, pdir, SMALL, check_only=True)
    with pytest.raises(SystemExit, match="read only"):
        P.build_set(P.PILOT_ROOT / "perc294", repo, pdir, SMALL)


def test_real_checkout_inventory():
    """Read-only: the real checkout gives 104 / 107 / 83 (paper_full 100 / 100 / 76), and the built set verifies."""
    if not (paths.REPO_WB / "tasks" / "full").is_dir() or not paths.PANOS.is_dir():
        pytest.skip("WalkerBench checkout or panoramas not here")
    plan = P.plan_set()
    c = plan["counts"]
    assert plan["n_tasks"] == 294 and plan["same_bytes_other_name"] == []
    assert [c[t]["paper_full"] for t in P.TYPES] == [100, 100, 76]
    assert [c[t]["standard20"] for t in P.TYPES] == [20, 20, 15]
    assert {d["type"] for d in plan["dropped"]} == {"height"} and len(plan["dropped"]) == 26
    if P.PERC_DIR.exists():
        assert P.set_problems(P.PERC_DIR, plan) == []


# ------------------------------------------------------------------------------------------ eval wrapper
def test_cmd_eval_with_a_fake_run_vpd(tmp_path):
    repo, pdir, dest = built_set(tmp_path)
    root = tmp_path / "R"
    seen = []

    def fake_main(argv):
        seen.append(list(argv))
        a = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
        assert R.TASK_SETS["perc294"] == dest / "tasks"
        d = Path(a["--out-root"]) / a["--run-name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text("{}")
        if "--overwrite" in argv:
            raise SystemExit("invalid run: fake")

    R = types.SimpleNamespace(TASK_SETS={"s2test": Path("/x")}, main=fake_main, code_sha256=lambda: {"run_vpd": "ab"})
    rc = P.cmd_eval(["--start", "Q3S", "--root", str(root)], R, dest, repo, pdir, SMALL)
    assert rc == 0 and seen[0] == P.protocol_argv("Q3S", root)
    ex = json.loads((P.run_dir_of("Q3S", root) / "explore.json").read_text())
    assert ex["label"] == "EXPLORATORY" and ex["exploratory"] and ex["rc"] == 0 and ex["start"] == "Q3S"
    assert ex["run_vpd_argv"] == P.protocol_argv("Q3S", root) and ex["code"]["run_vpd"] == "ab"
    assert ex["selection_sha256"] == P.sha256_file(dest / "selection.json")
    rc = P.cmd_eval(["--start", "E2", "--root", str(root), "--overwrite"], R, dest, repo, pdir, SMALL)
    assert rc == 1 and seen[-1][-1] == "--overwrite"
    assert json.loads((P.run_dir_of("E2", root) / "explore.json").read_text())["rc"] == 1
    with pytest.raises(SystemExit):
        P.cmd_eval(["--start", "Q3RD", "--root", str(root)], R, dest, repo, pdir, SMALL)
    with pytest.raises(SystemExit, match="read only"):
        P.cmd_eval(["--start", "Q3B", "--root", str(P.PILOT_ROOT / "Q3B")], R, dest, repo, pdir, SMALL)
    (dest / "tasks" / "angle_0002_a.json").write_text("{}")
    n = len(seen)
    with pytest.raises(SystemExit, match="perc294 check failed"):
        P.cmd_eval(["--start", "Q3B", "--root", str(root)], R, dest, repo, pdir, SMALL)
    assert len(seen) == n  # nothing ran
    R.TASK_SETS["perc294"] = Path("/elsewhere")
    with pytest.raises(SystemExit, match="expected"):
        P.register_task_sets(R, dest / "tasks")


def test_registration_in_the_real_run_vpd(tmp_path):
    R = run_vpd()
    _, _, dest = built_set(tmp_path)
    before = dict(R.TASK_SETS)
    try:
        R.TASK_SETS.pop("perc294", None)
        with pytest.raises(SystemExit) as e:
            R.main(["--rescore", "--run-dir", str(tmp_path / "nope"), "--task-set", "perc294"])
        assert e.value.code == 2  # argparse: invalid choice without the registration
        P.register_task_sets(R, dest / "tasks")
        assert {k: v for k, v in R.TASK_SETS.items() if k != "perc294"} == {k: v for k, v in before.items()
                                                                            if k != "perc294"}
        with pytest.raises(SystemExit) as e:
            R.main(["--rescore", "--run-dir", str(tmp_path / "nope"), "--task-set", "perc294"])
        assert "no results.json" in str(e.value.code)
        assert R.tasks_source("perc294") == json.loads((dest / "selection.json").read_text())  # into the manifest
        from vpd.walkerbench import sandbox

        assert sandbox.task_label(dest / "tasks") == "vpd_perc294"
    finally:
        R.TASK_SETS.clear()
        R.TASK_SETS.update(before)


# ------------------------------------------------------------------------------------------ analysis
def _mv():
    return {"type": "move", "move_id": 1}


def _stop(ans):
    return {"type": "stop", "answer": ans}


# stem -> (success, [(action, reason)]) ; None = no log
OUTCOMES = {
    "angle_0001_a": (1, [(_mv(), ""), (_stop("45"), "")]),
    "angle_0002_a": (0, [({"type": "rotation", "heading": 90.0, "pitch": 0}, "[Repaired] x"), (_stop("ARRIVED"), "")]),
    "dis_0001_d": (0, [(_mv(), "")] * 20),
    "dis_0002_d": (0, [(_mv(), ""), (_mv(), "MOVE(7) [vpd: invalid_move_crash] bad id")]),
    "height_0001_h": (1, [(_mv(), "Failed to parse response"), (_mv(), ""), (_stop("31 m"), "")]),
    "height_0002_h": (0, None),
}


def fake_run(root: Path, start: str, R, *, constrained="off", workers=8, invalid=False, tamper=None) -> Path:
    d = P.run_dir_of(start, root)
    logs = d / "WalkerBench" / "logs"
    logs.mkdir(parents=True)
    rows, calls, per_type = [], [], {t: {"n": 0, "success": 0} for t in P.TYPES}
    tot = {k: 0 for k in ("steps_logged", "parse_valid_steps", "repaired_steps", "unparseable_steps")}
    for stem, (succ, steps) in OUTCOMES.items():
        t = stem.split("_", 1)[0]
        per_type[t]["n"] += 1
        per_type[t]["success"] += succ
        log = None
        if steps is not None:
            log = f"{start}_{stem}_x.jsonl"
            ev = [{"event": "session_start", "task_id": stem}]
            ev += [{"event": "action", "step": i + 1, "action": a, "reason": r} for i, (a, r) in enumerate(steps)]
            (logs / log).write_text("".join(json.dumps(e) + "\n" for e in ev))
            st = R.step_stats(ev)
            for k in tot:
                tot[k] += st[k]
            calls += [{"task": stem, "step": i + 1, "finish_reason": "length" if t == "height" and i == 0 else "stop"}
                      for i in range(len(steps))]
        rows.append({"stem": stem, "kind": t, "success": succ, "log": log, "has_log": log is not None})
    (d / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    res = {"start": start, "model": start, "task_set": "perc294", "tasks_label": "vpd_perc294", "render_size": [640, 400],
           "constrained": constrained, "lenient": True, "max_tokens": 1024, "history_turns": 0, "temperature": 0.0,
           "blind": False, "limit": None, "model_dir": None, "adapter": None, "run_name": P.run_name(start),
           "per_type": {**per_type, "nav": {"n": 0, "success": 0}}, "n_tasks": 6, "tasks_without_log": 1,
           "regex": {"calls": len(calls), "finish_length": 1}, "counts": {"lenient_repairs": 1},
           "server": {"ready_seconds": 90.0}, "server_alive_at_end": True, "image_tokens": {"tokens_per_image": [242]},
           "seconds_per_step": 0.2, "wall_seconds": 10.0, **tot}
    if tamper:
        tamper(res)
    (d / ("results_invalid.json" if invalid else "results.json")).write_text(json.dumps(res))
    (d / "manifest.json").write_text(json.dumps({"backend": {"workers": workers}}))
    (d / "explore.json").write_text("{}")
    return d


def _walk_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _walk_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_keys(v)


def test_analyze_synthetic(tmp_path):
    R = run_vpd()
    _, _, dest = built_set(tmp_path)
    root = tmp_path / "R"
    fake_run(root, "Q3B", R)
    fake_run(root, "Q3S", R)
    doc = P.analyze(root, dest, R=R)
    a = doc["starts"]["Q3B"]
    assert a["state"] == "ok" and a["protocol_ok"] and a["consistent"], (a["protocol_problems"],
                                                                          a["consistency_problems"])
    s = a["success"]
    assert [(s["perc294"][t]["k"], s["perc294"][t]["n"]) for t in P.TYPES] == [(1, 2), (0, 2), (1, 2)]
    assert [(s["paper_full"][t]["k"], s["paper_full"][t]["n"]) for t in P.TYPES] == [(1, 2), (0, 1), (1, 2)]
    assert [(s["standard20"][t]["k"], s["standard20"][t]["n"]) for t in P.TYPES] == [(1, 1), (0, 1), (0, 1)]
    assert s["perc294"]["angle"]["wilson95"] == [round(x, 4) for x in P.wilson(1, 2)]
    e = a["ends"]
    assert dict(e["angle"]) == {"end_stop_number": 1, "end_stop_no_number": 1, "end_cap": 0, "end_crash": 0,
                                "end_other": 0, "no_log": 0}
    assert (e["dis"]["end_cap"], e["dis"]["end_crash"]) == (1, 1)
    assert (e["height"]["end_stop_number"], e["height"]["no_log"]) == (1, 1)
    st = a["steps"]
    assert (st["angle"]["steps_logged"], st["angle"]["repaired_steps"], st["angle"]["parse_valid_steps"]) == (4, 1, 3)
    assert (st["dis"]["steps_logged"], st["dis"]["invalid_move_crash_steps"]) == (22, 1)
    assert (st["height"]["unparseable_steps"], st["height"]["parse_valid_pct"], st["height"]["mean_steps"]) \
        == (1, 66.67, 3.0)
    assert a["calls"]["height"] == {"calls": 3, "errors": 0, "finish_length": 1}
    assert a["totals"]["image_tokens_per_image"] == [242] and a["totals"]["server_ready_seconds"] == 90.0
    assert doc["starts"]["E2"] == {"run_dir": str(P.run_dir_of("E2", root)), "run_name": P.run_name("E2"),
                                   "state": "missing"}
    assert doc["paper_reference"]["success_pct"] == {"angle": 8.0, "dis": 6.0, "height": 39.0}
    assert doc["label"] == "EXPLORATORY" and doc["exploratory"]
    # numbers only: no between-start difference anywhere
    assert not any(w in k.lower() for k in _walk_keys(doc) for w in ("diff", "delta", "minus", "rank", "gap"))
    on_disk = json.loads((root / "PERC.json").read_text())
    assert on_disk["starts"]["Q3S"]["success"] == json.loads(json.dumps(doc["starts"]["Q3S"]["success"]))
    md = (root / "REPORT_EXPLORATORY.md").read_text(encoding="utf-8")
    assert md.startswith("# EXPLORATORY") and "39.0%" in md and "| Q3B | ok | 1/2 (50.0%" in md
    assert "| E2 | 缺 |" in md and "没有闸门" in md
    lines, bad = P.expect(root, dest)
    assert bad == sum(1 for ln in lines if not ln.startswith("ok")) and bad >= 5  # E2's 4 files + PERC.json E2
    assert any(ln.startswith("PROBLEM PERC.json E2") for ln in lines)
    fake_run(root, "E2", R)
    P.analyze(root, dest, R=R)
    lines, bad = P.expect(root, dest)
    assert bad == 0, lines
    out = tmp_path / "RUN" / "expect.txt"
    assert P.main(["expect", "--root", str(root), "--perc-dir", str(dest), "--check", "--out", str(out)]) == 0
    assert out.read_text().splitlines() == lines + ["[perc] 0 missing or not ok"]


def test_analyze_flags_protocol_and_consistency_problems(tmp_path):
    R = run_vpd()
    _, _, dest = built_set(tmp_path)
    root = tmp_path / "R"
    fake_run(root, "Q3B", R, constrained="empty", workers=4)

    def bump(res):
        res["per_type"]["dis"]["success"] = 2
        res["repaired_steps"] = 5

    fake_run(root, "Q3S", R, tamper=bump)
    fake_run(root, "E2", R, invalid=True)
    doc = P.analyze(root, dest, R=R)
    q3b, q3s, e2 = (doc["starts"][s] for s in P.STARTS)
    assert not q3b["protocol_ok"] and {p.split(":")[0] for p in q3b["protocol_problems"]} == {"constrained", "workers"}
    assert q3b["consistent"]
    assert not q3s["consistent"] and len(q3s["consistency_problems"]) == 2 and q3s["protocol_ok"]
    assert e2["state"] == "invalid" and e2["results_file"] == "results_invalid.json"
    lines, bad = P.expect(root, dest)
    assert sum(ln.startswith("PROBLEM") for ln in lines) == 3
    assert P.main(["expect", "--root", str(root), "--perc-dir", str(dest), "--check"]) == 1
    md = (root / "REPORT_EXPLORATORY.md").read_text(encoding="utf-8")
    assert "## 问题" in md and "| E2 | invalid |" in md
    with pytest.raises(SystemExit, match="read only"):
        P.analyze(P.PILOT_ROOT / "x", dest, R=R, write=False)


# ------------------------------------------------------------------------------------------ payload
_KNOBS = ("VPD_THREADS", "VPD_JOB_DEADLINE_S", "VPD_MIN_STEP_S", "VPD_TAIL_RESERVE_S", "VPD_RERUN_INVALID",
          "VPD_DRYRUN", "RUN", "VPD_PERC_ROOT", "RESERVE")


def _bash(args, env_extra, cwd):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update(env_extra)
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env, cwd=str(cwd), timeout=120)


def _dry(out: str):
    return [ln[len("[dry] "):] for ln in out.splitlines() if ln.startswith("[dry] ")]


def _steps(run: Path):
    return [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]


def test_payload_bash_syntax():
    res = subprocess.run(["bash", "-n", str(PAYLOAD)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_payload_dryrun_order_and_flags(tmp_path):
    run, root = tmp_path / "jobs" / "wb-perc-202611100000", tmp_path / "R"
    res = _bash([str(PAYLOAD), str(run)], {"VPD_DRYRUN": "1", "VPD_PERC_ROOT": str(root)}, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    lines = _dry(res.stdout)
    perc = [ln for ln in lines if "-m vpd.walkerbench.perc " in ln]
    assert [ln.split("-m vpd.walkerbench.perc ", 1)[1] for ln in perc] == [
        "build --check", f"eval --start Q3B --root {root}", f"eval --start Q3S --root {root}",
        f"eval --start E2 --root {root}", f"analyze --root {root}",
        f"expect --root {root} --check --out {run}/expect.txt"]
    assert all(ln.startswith("timeout -k 30 ") and 0 < int(ln.split()[3]) <= 5280 for ln in perc)
    assert all(int(ln.split()[3]) <= 5280 - 120 for ln in perc if " eval " in ln)  # the tail reserve
    assert all(str(paths.PY_Q3) in ln for ln in perc if " eval " in ln)
    assert all(str(paths.PY_EVAL) in ln for ln in perc if " eval " not in ln)
    assert "deadline=5280s" in res.stdout and "EXPLORATORY" in res.stdout
    assert [s["name"] for s in _steps(run)] == ["check_set", "eval_Q3B", "eval_Q3S", "eval_E2", "analyze", "expect"]
    assert all(s["rc"] == 0 for s in _steps(run))
    assert sorted(p.name for p in run.iterdir()) == ["df_end.txt", "node.txt", "steps.jsonl"]
    assert not root.exists()  # a dry run writes only RUN/{steps.jsonl,node.txt,df_end.txt}
    assert f"[dry] date -u > {run}/DONE" in res.stdout


def test_payload_skip_invalid_and_rerun(tmp_path):
    run, root = tmp_path / "jobs" / "wb-perc-202611100001", tmp_path / "R"
    for s, f in (("Q3S", "results.json"), ("E2", "results_invalid.json")):
        d = P.run_dir_of(s, root)
        d.mkdir(parents=True)
        (d / f).write_text("{}")
    env = {"VPD_DRYRUN": "1", "VPD_PERC_ROOT": str(root)}
    res = _bash([str(PAYLOAD), str(run)], env, tmp_path)
    assert res.returncode == 1, res.stderr + res.stdout
    names = [s["name"] for s in _steps(run)]
    assert "eval_Q3S:skip" in names and "eval_E2:invalid_kept" in names and "eval_Q3B" in names
    assert not any("--start Q3S" in ln or "--start E2" in ln for ln in _dry(res.stdout))
    run2 = tmp_path / "jobs" / "wb-perc-202611100002"
    res = _bash([str(PAYLOAD), str(run2)], {**env, "VPD_RERUN_INVALID": "1"}, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    e2 = [ln for ln in _dry(res.stdout) if "--start E2" in ln]
    assert len(e2) == 1 and e2[0].endswith(f"eval --start E2 --root {root} --overwrite")


def test_payload_soft_deadline_and_refusals(tmp_path):
    run, root = tmp_path / "jobs" / "wb-perc-202611100003", tmp_path / "R"
    env = {"VPD_DRYRUN": "1", "VPD_PERC_ROOT": str(root)}
    res = _bash([str(PAYLOAD), str(run)], {**env, "VPD_JOB_DEADLINE_S": "30"}, tmp_path)
    assert res.returncode == 1 and not _dry(res.stdout)  # no step ran, no DONE
    names = [s["name"] for s in _steps(run)]
    assert names == ["check_set:no_time", "eval_Q3B:no_set", "eval_Q3S:no_set", "eval_E2:no_set", "analyze:no_time",
                     "expect:no_time"]
    run2 = tmp_path / "jobs" / "wb-perc-202611100005"  # time for check_set / analyze / expect, not for an eval
    res = _bash([str(PAYLOAD), str(run2)], {**env, "VPD_JOB_DEADLINE_S": "150"}, tmp_path)
    assert res.returncode == 1, res.stderr + res.stdout
    assert [s["name"] for s in _steps(run2)] == ["check_set", "eval_Q3B:no_time", "eval_Q3S:no_time",
                                                 "eval_E2:no_time", "analyze", "expect"]
    for bad in ("0", "abc"):
        r = _bash([str(PAYLOAD), str(run)], {**env, "VPD_JOB_DEADLINE_S": bad}, tmp_path)
        assert r.returncode == 2 and "VPD_JOB_DEADLINE_S" in r.stderr
    r = _bash([str(PAYLOAD), str(run)], {**env, "VPD_TAIL_RESERVE_S": "-1"}, tmp_path)
    assert r.returncode == 2 and "VPD_TAIL_RESERVE_S" in r.stderr
    r = _bash([str(PAYLOAD)], {**env, "RUN": ""}, tmp_path)
    assert r.returncode == 2 and "usage" in r.stderr
    r = _bash([str(PAYLOAD), str(run)], {**env, "VPD_PERC_ROOT": str(paths.WB_RUNS / "pilot" / "Q3B" / "x")}, tmp_path)
    assert r.returncode == 2 and "refusing" in r.stderr
    if os.geteuid() != 0:  # a real (non-dry) run as a normal user stops before writing anything
        run4 = tmp_path / "jobs" / "wb-perc-202611100004"
        r = _bash([str(PAYLOAD), str(run4)], {}, tmp_path)
        assert r.returncode == 2 and "must run as root" in r.stderr and not run4.exists()
