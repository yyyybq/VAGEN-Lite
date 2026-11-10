"""Tests of the EXPLORATORY perception zero-shot package (DECISIONS G5): vpd/walkerbench/perc.py, vpd/jobs/wb_perc.sh.

Hermetic: a fake WalkerBench checkout and panorama dir under tmp_path for the task-set build / verify / refusals, a
fake run_vpd module for the eval wrapper (with the job guard's test override VPD_PERC_TEST_ALLOW_LIVE=1, or a fake
code snapshot), the run-dir lock under tmp_path, synthetic run dirs (results.json, per_task.jsonl, timestamped logs,
calls.jsonl) for the analysis (incl. the inferred end_max_time class and the timing stats), and the payload in
VPD_DRYRUN=1 mode (incl. the optional start list).  No GPU, no model, no network, no rendering.  Two read-only checks
look at the real checkout (the 104 / 107 / 83 inventory) and at the built OUT perc294 dir; they skip when absent.
    cd $WT && env OMP_NUM_THREADS=1 ... nice -n 19 $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=<dir> \
        vpd/tests/test_perc.py
"""

from __future__ import annotations

import datetime
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
    task = {"task_id": stem, "task_type": t, "spawn_pano_id": spawn, "max_steps": None if t == "height" else 20}
    if t == "height":
        task["max_time_seconds"] = 180  # as every real height task (upstream)
    return json.dumps(task, indent=1).encode()


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
@pytest.fixture
def in_job(tmp_path, monkeypatch):
    """The eval guard's test override and a job RUN dir (what wb_perc.sh exports in a job)."""
    run = tmp_path / "jobs" / "wb-perc-test"
    run.mkdir(parents=True)
    monkeypatch.setenv(P.ALLOW_LIVE_ENV, "1")
    monkeypatch.setenv("RUN", str(run))
    monkeypatch.delenv(P.TAKEOVER_ENV, raising=False)
    return run


def _fake_run_vpd(seen: list, dest: Path, check=None):
    def fake_main(argv):
        seen.append(list(argv))
        a = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
        assert R.TASK_SETS["perc294"] == dest / "tasks"
        if check is not None:
            check(argv)
        d = Path(a["--out-root"]) / a["--run-name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text("{}")
        if "--overwrite" in argv:
            raise SystemExit("invalid run: fake")

    R = types.SimpleNamespace(TASK_SETS={"s2test": Path("/x")}, main=fake_main, code_sha256=lambda: {"run_vpd": "ab"})
    return R


def test_cmd_eval_with_a_fake_run_vpd(tmp_path, in_job):
    repo, pdir, dest = built_set(tmp_path)
    root = tmp_path / "R"
    seen, held = [], []

    def lock_held(argv):  # while run_vpd runs, this start's lock is held by this RUN
        start = argv[argv.index("--model") + 1]
        held.append(json.loads(P.lock_path(start, root).read_text()))

    R = _fake_run_vpd(seen, dest, lock_held)
    rc = P.cmd_eval(["--start", "Q3S", "--root", str(root)], R, dest, repo, pdir, SMALL)
    assert rc == 0 and seen[0] == P.protocol_argv("Q3S", root)
    h = held[0]
    assert (h["run"], h["start"], h["pid"], h["took_over"]) == (str(in_job), "Q3S", os.getpid(), None)
    assert h["host"] and h["utc"].endswith("Z") and h["token"]
    assert not P.lock_path("Q3S", root).exists()  # released
    ex = json.loads((P.run_dir_of("Q3S", root) / "explore.json").read_text())
    assert ex["label"] == "EXPLORATORY" and ex["exploratory"] and ex["rc"] == 0 and ex["start"] == "Q3S"
    assert ex["run_vpd_argv"] == P.protocol_argv("Q3S", root) and ex["code"]["run_vpd"] == "ab"
    assert ex["selection_sha256"] == P.sha256_file(dest / "selection.json")
    assert ex["run"] == str(in_job) and ex["code_snapshot"]["override"] == f"{P.ALLOW_LIVE_ENV}=1"
    assert ex["lock"]["path"] == str(P.lock_path("Q3S", root)) and ex["lock"]["took_over"] is None
    rc = P.cmd_eval(["--start", "E2", "--root", str(root), "--overwrite"], R, dest, repo, pdir, SMALL)
    assert rc == 1 and seen[-1][-1] == "--overwrite"
    assert json.loads((P.run_dir_of("E2", root) / "explore.json").read_text())["rc"] == 1
    assert not P.lock_path("E2", root).exists()  # released after a failed run too
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


def _snapshot_root(base: Path, record=None) -> Path:
    root = base / "RUN" / "code"
    (root / "vpd").mkdir(parents=True)
    if record is not None:
        (root / "SNAPSHOT.json").write_text(record if isinstance(record, str) else json.dumps(record))
    return root


def test_eval_refused_outside_a_job(tmp_path, monkeypatch):
    """perc eval starts vLLM: refused unless the code is a job's snapshot (or the tests' explicit override)."""
    repo, pdir, dest = built_set(tmp_path)
    root, seen = tmp_path / "R", []
    R = _fake_run_vpd(seen, dest)
    argv = ["--start", "Q3B", "--root", str(root)]
    for k in (P.ALLOW_LIVE_ENV, P.TAKEOVER_ENV, "RUN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(paths, "WORKTREE", _snapshot_root(tmp_path / "live"))  # a live tree: no SNAPSHOT.json
    with pytest.raises(SystemExit, match="not inside a job"):
        P.cmd_eval(argv, R, dest, repo, pdir, SMALL)
    monkeypatch.setenv(P.ALLOW_LIVE_ENV, "0")  # only "1" overrides
    with pytest.raises(SystemExit, match="not inside a job"):
        P.cmd_eval(argv, R, dest, repo, pdir, SMALL)
    monkeypatch.delenv(P.ALLOW_LIVE_ENV)
    monkeypatch.setattr(paths, "WORKTREE", _snapshot_root(tmp_path / "bad", "not json"))
    with pytest.raises(SystemExit, match="mode 'unreadable'"):
        P.cmd_eval(argv, R, dest, repo, pdir, SMALL)
    assert seen == [] and not root.exists()  # nothing ran, nothing written
    snap = _snapshot_root(tmp_path / "job", {"format": 1, "commit": "c0ffee" * 6 + "abcd", "utc": "x", "files": {}})
    monkeypatch.setattr(paths, "WORKTREE", snap)
    with pytest.raises(SystemExit, match="env RUN"):  # a snapshot, but no RUN
        P.cmd_eval(argv, R, dest, repo, pdir, SMALL)
    monkeypatch.setenv("RUN", "relative/run")
    with pytest.raises(SystemExit, match="env RUN"):
        P.cmd_eval(argv, R, dest, repo, pdir, SMALL)
    assert seen == []
    run = tmp_path / "RUN"
    monkeypatch.setenv("RUN", str(run))
    assert P.cmd_eval(argv, R, dest, repo, pdir, SMALL) == 0 and len(seen) == 1
    ex = json.loads((P.run_dir_of("Q3B", root) / "explore.json").read_text())
    assert ex["code_snapshot"] == {"mode": "snapshot", "root": str(snap), "commit": "c0ffee" * 6 + "abcd", "utc": "x"}
    assert ex["run"] == str(run)


def test_run_dir_lock_rules(tmp_path, monkeypatch):
    monkeypatch.delenv(P.TAKEOVER_ENV, raising=False)
    lp = P.lock_path("E2", tmp_path / "R")
    run_a, run_b = tmp_path / "jobs" / "A", tmp_path / "jobs" / "B"
    a = P.acquire_lock(lp, run_a, "E2")
    on_disk = json.loads(lp.read_text())
    assert {k: on_disk[k] for k in ("host", "pid", "utc", "run", "start")} == \
        {k: a[k] for k in ("host", "pid", "utc", "run", "start")} and on_disk["run"] == str(run_a)
    with pytest.raises(SystemExit, match="neither DONE nor job/exit_code"):  # A still runs
        P.acquire_lock(lp, run_b, "E2")
    assert json.loads(lp.read_text())["token"] == a["token"]  # untouched
    a2 = P.acquire_lock(lp, run_a, "E2")  # the same RUN takes over its own earlier lock
    assert a2["took_over"]["reason"].startswith("same RUN") and a2["took_over"]["holder"]["token"] == a["token"]
    assert P.release_lock(a) is False and lp.exists()  # not ours any more
    for marker in ("DONE", "job/exit_code"):  # the holder's RUN finished: taken over without the override
        (run_a / marker).parent.mkdir(parents=True, exist_ok=True)
        (run_a / marker).write_text("x\n")
        b = P.acquire_lock(lp, run_b, "E2")
        assert b["took_over"]["reason"].startswith("the holder's RUN has finished")
        assert json.loads(lp.read_text())["took_over"]["holder"]["run"] == str(run_a)
        assert P.release_lock(b) is True and not lp.exists()
        (run_a / marker).unlink()
        P.acquire_lock(lp, run_a, "E2")
    monkeypatch.setenv(P.TAKEOVER_ENV, "1")  # the operator saw A's job terminal (no marker)
    b = P.acquire_lock(lp, run_b, "E2")
    assert b["took_over"]["reason"] == f"{P.TAKEOVER_ENV}=1" and json.loads(lp.read_text())["run"] == str(run_b)
    assert P.release_lock(b) and not list(lp.parent.glob(P.LOCK_NAME + "*"))  # no files left beside it
    monkeypatch.delenv(P.TAKEOVER_ENV)
    lp.write_text("garbage")  # an unreadable lock refuses unless overridden
    with pytest.raises(SystemExit, match="unreadable lock: True"):
        P.acquire_lock(lp, run_b, "E2")
    monkeypatch.setenv(P.TAKEOVER_ENV, "1")
    assert P.acquire_lock(lp, run_b, "E2")["took_over"]["holder"] is None


def test_eval_refused_while_another_job_holds_the_start(tmp_path, in_job):
    repo, pdir, dest = built_set(tmp_path)
    root, seen = tmp_path / "R", []
    R = _fake_run_vpd(seen, dest)
    other = tmp_path / "jobs" / "wb-perc-first"
    P.acquire_lock(P.lock_path("E2", root), other, "E2")
    with pytest.raises(SystemExit, match="held by RUN"):
        P.cmd_eval(["--start", "E2", "--root", str(root)], R, dest, repo, pdir, SMALL)
    assert seen == [] and not P.run_dir_of("E2", root).exists()
    assert P.cmd_eval(["--start", "Q3B", "--root", str(root)], R, dest, repo, pdir, SMALL) == 0  # other starts: free
    (other / "job").mkdir(parents=True)
    (other / "job" / "exit_code").write_text("124\n")  # the first job ended (outer timeout)
    assert P.cmd_eval(["--start", "E2", "--root", str(root)], R, dest, repo, pdir, SMALL) == 0
    ex = json.loads((P.run_dir_of("E2", root) / "explore.json").read_text())
    assert ex["lock"]["took_over"]["holder"]["run"] == str(other) and not P.lock_path("E2", root).exists()


def test_protocol_keys_are_run_vpd_result_meta_keys():
    """protocol_problems reads only keys run_vpd copies into results.json (guards against run_vpd drifting)."""
    R = run_vpd()
    assert set(P.protocol_want("Q3B")) <= set(R.RESULT_META_KEYS)


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


T0 = datetime.datetime(2026, 11, 10, 12, 0, 0)
STEP_S = {"angle": 2.0, "dis": 1.0, "height": 10.0}  # seconds between logged steps, per type (or per stem)
BASE_TOKENS = {"angle": 30, "dis": 40, "height": 60}  # completion_tokens of step i: base + i; 1024 when cut


def _events(stem: str, steps: list, dt: float, start_ts: str | None = None) -> list:
    ev = [{"event": "session_start", "task_id": stem, "timestamp": start_ts or T0.strftime("%Y%m%d_%H%M%S")}]
    ev += [{"event": "action", "step": i + 1, "timestamp": (T0 + datetime.timedelta(seconds=dt * (i + 1))).isoformat(),
            "action": a, "reason": r} for i, (a, r) in enumerate(steps)]
    return ev


def fake_run(root: Path, start: str, R, *, constrained="off", workers=8, invalid=False, tamper=None,
             outcomes=OUTCOMES, step_s=None) -> Path:
    d = P.run_dir_of(start, root)
    logs = d / "WalkerBench" / "logs"
    logs.mkdir(parents=True)
    step_s = {**STEP_S, **(step_s or {})}
    rows, calls, per_type = [], [], {t: {"n": 0, "success": 0} for t in P.TYPES}
    tot = {k: 0 for k in ("steps_logged", "parse_valid_steps", "repaired_steps", "unparseable_steps")}
    for stem, (succ, steps) in outcomes.items():
        t = stem.split("_", 1)[0]
        per_type[t]["n"] += 1
        per_type[t]["success"] += succ
        log = None
        if steps is not None:
            log = f"{start}_{stem}_x.jsonl"
            ev = _events(stem, steps, step_s.get(stem, step_s[t]))
            (logs / log).write_text("".join(json.dumps(e) + "\n" for e in ev))
            st = R.step_stats(ev)
            for k in tot:
                tot[k] += st[k]
            for i in range(len(steps)):
                cut = t == "height" and i == 0
                calls.append({"task": stem, "step": i + 1, "finish_reason": "length" if cut else "stop",
                              "completion_tokens": 1024 if cut else BASE_TOKENS[t] + i})
            if stem == "dis_0002_d":
                calls.append({"task": stem, "step": len(steps) + 1, "error": "boom"})  # no answer, no tokens
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
    assert dict(e["angle"]) == {"end_stop_number": 1, "end_stop_no_number": 1, "end_cap": 0, "end_max_time": 0,
                                "end_crash": 0, "end_other": 0, "no_log": 0}
    assert (e["dis"]["end_cap"], e["dis"]["end_crash"]) == (1, 1)
    assert (e["height"]["end_stop_number"], e["height"]["no_log"], e["height"]["end_max_time"]) == (1, 1, 0)
    st = a["steps"]
    assert (st["angle"]["steps_logged"], st["angle"]["repaired_steps"], st["angle"]["parse_valid_steps"]) == (4, 1, 3)
    assert (st["dis"]["steps_logged"], st["dis"]["invalid_move_crash_steps"]) == (22, 1)
    assert (st["height"]["unparseable_steps"], st["height"]["parse_valid_pct"], st["height"]["mean_steps"]) \
        == (1, 66.67, 3.0)
    assert a["calls"]["height"] == {"calls": 3, "errors": 0, "finish_length": 1}
    assert a["calls"]["dis"] == {"calls": 23, "errors": 1, "finish_length": 0}
    tm = a["timing"]
    assert tm["angle"]["completion_tokens"] == {"n": 4, "mean": 30.5, "p95": 31.0, "max": 31.0}
    assert tm["dis"]["completion_tokens"]["n"] == 22  # the error call has no tokens
    assert tm["height"]["completion_tokens"] == {"n": 3, "mean": 382.3, "p95": 1024.0, "max": 1024.0}
    assert tm["angle"]["step_seconds"] == {"n": 4, "mean": 2.0, "p95": 2.0, "max": 2.0}
    assert tm["dis"]["episode_seconds"] == {"n": 2, "mean": 11.0, "p95": 20.0, "max": 20.0}
    assert tm["height"]["episode_seconds"]["n"] == 1 and tm["height"]["step_seconds"]["mean"] == 10.0
    assert [tm[t]["max_time_seconds"] for t in P.TYPES] == [[None], [None], [180]]
    assert all(tm[t]["episodes_without_timestamps"] == 0 for t in P.TYPES)
    assert doc["task_set"]["max_time_seconds"] == {"angle": {"None": 2}, "dis": {"None": 2}, "height": {"180": 2}}
    assert "推断" in doc["rules"]["end_max_time"] and "wall-clock" in doc["rules"]["height_time_limit"]
    assert a["produced_by"] == {"run": None, "utc": None, "rc": None, "snapshot_commit": None, "perc_py_sha256": None}
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
    assert "限时结束（推断）" in md and "| Q3B | height | 3 | 382.3 | 1024.0 | 10.0 | 10.0 | 30.0 | 30.0 | 30.0 | 180 |" in md
    assert "180 秒限时是从会话开始算的墙钟时间" in md and "取决于解码速度和同一个 vLLM 服务器上的并发" in md
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


def test_episode_end_max_time_rule_and_helpers():
    """end_max_time (推断): last action a MOVE, fewer steps than the cap, the task has a limit, last stamp - start
    stamp >= the limit; STOP / crash / cap come first; no limit, a non-MOVE last action or a missing stamp -> other."""
    R = run_vpd()
    mv, rot = (_mv(), ""), ({"type": "rotation", "heading": 90.0, "pitch": 0}, "")
    crash = (_mv(), "MOVE(7) [vpd: invalid_move_crash] bad id")

    def ev(steps, dt, **kw):
        return _events("height_x", steps, dt, **kw)

    assert P.episode_end(ev([mv, mv], 90.0), 30, R, 180) == "end_max_time"  # exactly 180 s
    assert P.episode_end(ev([mv, mv], 89.95), 30, R, 180) == "end_other"  # 179.9 s
    assert P.episode_end(ev([mv, mv], 100.0), 30, R, None) == "end_other"  # no limit (angle / dis)
    assert P.episode_end(ev([mv, mv], 100.0), 2, R, 180) == "end_cap"  # at the cap
    assert P.episode_end(ev([mv, rot], 100.0), 30, R, 180) == "end_other"  # the engine checks only after a MOVE
    assert P.episode_end(ev([mv, (_stop("12 m"), "")], 100.0), 30, R, 180) == "end_stop_number"
    assert P.episode_end(ev([mv, crash], 100.0), 30, R, 180) == "end_crash"
    assert P.episode_end(ev([mv, mv], 100.0, start_ts="2026-11-10T12:00:00"), 30, R, 180) == "end_other"  # format
    no_ts = ev([mv, mv], 100.0)
    no_ts[-1]["timestamp"] = ""
    assert P.episode_end(no_ts, 30, R, 180) == "end_other" and P.action_elapsed(no_ts) is None
    assert P.episode_end(ev([], 1.0), 30, R, 180) == "end_other"
    assert P.action_elapsed(ev([mv, mv], 1.5)) == [1.5, 3.0] and P.action_elapsed(ev([], 1.0)) == []
    aware = ev([mv], 1.0)
    aware[-1]["timestamp"] = "2026-11-10T12:03:20+00:00"
    assert P.action_elapsed(aware) is None
    assert P.action_elapsed([{"event": "action", "timestamp": "2026-11-10T12:00:01"}]) is None  # no session_start
    assert P.dist([]) == {"n": 0, "mean": None, "p95": None, "max": None}
    assert P.dist(range(1, 21)) == {"n": 20, "mean": 10.5, "p95": 19.0, "max": 20.0}  # nearest rank
    assert P.dist([5]) == {"n": 1, "mean": 5.0, "p95": 5.0, "max": 5.0}


def test_analyze_infers_end_max_time_for_height_only(tmp_path):
    R = run_vpd()
    _, _, dest = built_set(tmp_path)
    root = tmp_path / "R"
    outcomes = dict(OUTCOMES)
    outcomes["height_0002_h"] = (0, [(_mv(), ""), (_mv(), "")])  # 2 x 95 s: last MOVE at 190 s
    outcomes["dis_0002_d"] = (0, [(_mv(), ""), (_mv(), "")])  # 200 s, but dis has no time limit
    fake_run(root, "Q3B", R, outcomes=outcomes, step_s={"height_0002_h": 95.0, "dis_0002_d": 100.0})
    a = P.analyze(root, dest, R=R)["starts"]["Q3B"]
    assert a["consistent"] and a["protocol_ok"], a["consistency_problems"]
    e = a["ends"]
    assert (e["height"]["end_max_time"], e["height"]["end_stop_number"], e["height"]["no_log"]) == (1, 1, 0)
    assert (e["dis"]["end_max_time"], e["dis"]["end_other"], e["dis"]["end_cap"]) == (0, 1, 1)
    tm = a["timing"]["height"]
    assert tm["step_seconds"] == {"n": 5, "mean": 44.0, "p95": 95.0, "max": 95.0}
    assert tm["episode_seconds"] == {"n": 2, "mean": 110.0, "p95": 190.0, "max": 190.0}
    md = (root / "REPORT_EXPLORATORY.md").read_text(encoding="utf-8")
    assert "| Q3B | height | 5 | 80.0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 2 |" in md


# ------------------------------------------------------------------------------------------ payload
_KNOBS = ("VPD_THREADS", "VPD_JOB_DEADLINE_S", "VPD_MIN_STEP_S", "VPD_TAIL_RESERVE_S", "VPD_RERUN_INVALID",
          "VPD_DRYRUN", "RUN", "VPD_PERC_ROOT", "RESERVE", "VPD_PERC_TAKEOVER", "VPD_PERC_TEST_ALLOW_LIVE")


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
    assert all(ln.startswith("timeout -k 30 ") and 6720 - 60 < int(ln.split()[3]) <= 6720 for ln in perc
               if " eval " not in ln)  # the soft deadline: est 2.0 GPU-h -> outer 6900 s, soft 6720 s
    assert all(6720 - 120 - 60 < int(ln.split()[3]) <= 6720 - 120 for ln in perc if " eval " in ln)  # tail reserve
    assert all(str(paths.PY_Q3) in ln for ln in perc if " eval " in ln)
    assert all(str(paths.PY_EVAL) in ln for ln in perc if " eval " not in ln)
    assert "deadline=6720s" in res.stdout and "EXPLORATORY" in res.stdout and "starts=Q3B Q3S E2" in res.stdout
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


def test_payload_start_list(tmp_path):
    """wb_perc.sh RUN [START ...]: a targeted resubmission runs only the listed starts, in the fixed order; analyze and
    expect still cover all three starts (they read R/<START>/evals of whichever job produced them)."""
    root = tmp_path / "R"
    env = {"VPD_DRYRUN": "1", "VPD_PERC_ROOT": str(root)}
    run = tmp_path / "jobs" / "wb-perc-202611100010"
    res = _bash([str(PAYLOAD), str(run), "E2"], env, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    perc = [ln.split("-m vpd.walkerbench.perc ", 1)[1] for ln in _dry(res.stdout) if "-m vpd.walkerbench.perc " in ln]
    assert perc == ["build --check", f"eval --start E2 --root {root}", f"analyze --root {root}",
                    f"expect --root {root} --check --out {run}/expect.txt"]
    assert [(s["name"], s["rc"]) for s in _steps(run)] == [
        ("check_set", 0), ("eval_Q3B:not_listed", 0), ("eval_Q3S:not_listed", 0), ("eval_E2", 0), ("analyze", 0),
        ("expect", 0)]
    assert "starts=E2 " in res.stdout and f"[dry] date -u > {run}/DONE" in res.stdout
    run2 = tmp_path / "jobs" / "wb-perc-202611100011"  # given out of order: still Q3B before E2
    res = _bash([str(PAYLOAD), str(run2), "E2", "Q3B"], {**env, "VPD_PERC_TAKEOVER": "1"}, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    assert [ln.split("--start ", 1)[1].split()[0] for ln in _dry(res.stdout) if " eval --start " in ln] == ["Q3B", "E2"]
    assert [s["name"] for s in _steps(run2)][1:4] == ["eval_Q3B", "eval_Q3S:not_listed", "eval_E2"]
    assert "takeover=1" in res.stdout
    d = P.run_dir_of("E2", root)  # a listed start that already finished is skipped
    d.mkdir(parents=True)
    (d / "results.json").write_text("{}")
    run3 = tmp_path / "jobs" / "wb-perc-202611100012"
    res = _bash([str(PAYLOAD), str(run3), "E2"], env, tmp_path)
    assert res.returncode == 0 and not any(" eval --start " in ln for ln in _dry(res.stdout))
    assert [s["name"] for s in _steps(run3)][1:4] == ["eval_Q3B:not_listed", "eval_Q3S:not_listed", "eval_E2:skip"]
    for args, msg in ((["Q3RD"], "must be one of"), (["E2", "E2"], "given twice"), (["q3b"], "must be one of")):
        r = _bash([str(PAYLOAD), str(tmp_path / "jobs" / "bad"), *args], env, tmp_path)
        assert r.returncode == 2 and msg in r.stderr and "usage" in r.stderr
    r = _bash([str(PAYLOAD), "E2"], {**env, "RUN": str(run)}, tmp_path)  # RUN must come first
    assert r.returncode == 2 and "absolute path" in r.stderr
    assert not (tmp_path / "jobs" / "bad").exists()


def test_payload_soft_deadline_and_refusals(tmp_path):
    run, root = tmp_path / "jobs" / "wb-perc-202611100003", tmp_path / "R"
    env = {"VPD_DRYRUN": "1", "VPD_PERC_ROOT": str(root)}
    res = _bash([str(PAYLOAD), str(run)], {**env, "VPD_JOB_DEADLINE_S": "30"}, tmp_path)
    assert res.returncode == 1 and not _dry(res.stdout)  # no step ran, no DONE
    names = [s["name"] for s in _steps(run)]
    assert names == ["check_set:no_time", "eval_Q3B:no_set", "eval_Q3S:no_set", "eval_E2:no_set", "analyze:no_time",
                     "expect:no_time"]
    run1 = tmp_path / "jobs" / "wb-perc-202611100006"  # with a start list: the unlisted starts are not_listed
    res = _bash([str(PAYLOAD), str(run1), "Q3S"], {**env, "VPD_JOB_DEADLINE_S": "30"}, tmp_path)
    assert res.returncode == 1 and [s["name"] for s in _steps(run1)] == [
        "check_set:no_time", "eval_Q3B:not_listed", "eval_Q3S:no_set", "eval_E2:not_listed", "analyze:no_time",
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
