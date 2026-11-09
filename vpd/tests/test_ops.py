"""Tests of WP-OPS: vpd.sco (name rules, prepare, submit guards, delete rules), vpd.ledger (accounting, budget,
atomic append) and the shell prelude vpd/jobs/common_env.sh (+ syntax of preflight.sh).

Login node, 1 core, no network, no sco: every sco call is replaced by a fake.  Scratch lives under
eval_external_assets/vpd/scratch/ops/pytest_<rand>/ (env VPD_OPS_TEST_DIR overrides) and is removed at teardown.
    PYTHONPATH=$WT OMP_NUM_THREADS=1 PY_EVAL -m pytest -q -p no:cacheprovider vpd/tests/test_ops.py
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import pytest

from vpd import ledger, paths, sco

sco_job = sco.sco_job
JOBS_DIR = paths.VPD_PKG / "jobs"
T0 = datetime.datetime(2026, 11, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- fixtures -----------------------------------------------------------------------------------------------

@pytest.fixture
def base():
    root = Path(os.environ.get("VPD_OPS_TEST_DIR") or paths.scratch("ops"))
    d = root / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class FakeSco:
    """In-memory sco: job states, a list page, submit / delete call logs."""

    def __init__(self, base: Path):
        self.base = base
        self.jobs = {}  # pt -> {"state", "start_time", "complete_time", "display_name", "create_time"}
        self.n = 0
        self.submits = []
        self.deleted = []
        self.stopped = []
        self.list_ok = True
        self.status_fail = set()
        self.extra_list = []
        self.uncertain_next = False

    # sco_job.submit
    def submit(self, run):
        run = Path(run)
        job = run / "job"
        assert (job / "prepared.json").exists()
        if (job / "submission.json").exists():
            return json.loads((job / "submission.json").read_text())["job_id"]
        self.submits.append(str(run))
        prepared = json.loads((job / "prepared.json").read_text())
        if self.uncertain_next:
            (job / "submit.log").write_text("FAKE: dns timeout\n")
            (job / sco_job.UNCERTAIN).write_text(json.dumps({"utc": iso(T0),
                                                              "job_name": "downstream-" + prepared["name"]}))
            raise sco_job.SubmitUncertain(run, "submission uncertain")
        self.n += 1
        pt = f"pt-fake{self.n:04d}"
        self.jobs[pt] = {"state": "QUEUEING", "start_time": None, "complete_time": None,
                         "display_name": "downstream-" + prepared["name"], "create_time": iso(T0)}
        (job / "submit.log").write_text(f"created {pt}\n")
        (job / "submission.json").write_text(json.dumps({"job_id": pt}))
        return pt

    # sco_job.status
    def status(self, run):
        pt = json.loads((Path(run) / "job/submission.json").read_text())["job_id"]
        j = self.jobs.get(pt)
        state = "NOT_FOUND" if (j is None or pt in self.status_fail) else j["state"]
        return {"job_id": pt, "state": state, "finished": state in sco_job.FINISHED,
                "create_time": j and j["create_time"], "start_time": j and j["start_time"], "exit_code": None}

    def describe(self, pt):
        j = self.jobs.get(pt)
        return None if j is None else {"state": j["state"], "start_time": j["start_time"],
                                       "complete_time": j["complete_time"]}

    def list_page(self):
        if not self.list_ok:
            return None
        rows = [{"name": pt, "display_name": j["display_name"], "state": j["state"],
                 "ownership": {"user_name": "yinbaiqiao"}, "create_time": j["create_time"],
                 "start_time": j["start_time"]} for pt, j in self.jobs.items()]
        return rows + list(self.extra_list)

    def sco_delete(self, pt):
        self.deleted.append(pt)
        self.jobs.pop(pt, None)
        return 0, "deleted"

    def force_delete(self, run):
        pt = json.loads((Path(run) / "job/submission.json").read_text())["job_id"]
        self.stopped.append(pt)
        self.jobs.pop(pt, None)
        return f"{pt}: stop+delete"

    def set(self, pt, state, start=None, end=None):
        j = self.jobs[pt]
        j["state"] = state
        j["start_time"] = start and iso(start)
        j["complete_time"] = end and iso(end)


@pytest.fixture
def fake(base, monkeypatch):
    f = FakeSco(base)
    monkeypatch.setenv("VPD_LEDGER", str(base / "ledger.jsonl"))
    monkeypatch.delenv("VPD_GPUH_CAP", raising=False)
    disk = base / "disk_runs"
    (disk / "flags").mkdir(parents=True)
    usage = base / "usage.csv"
    usage.write_text("utc,quota_TB,table_used_TB,remaining_TB,table_files,rfbytes_TB,level,seconds\n"
                     f"{iso(datetime.datetime.now(datetime.timezone.utc))},22.0,20.0,2.0,1,13.0,1,800\n")
    monkeypatch.setattr(sco, "DISK_RUNS", disk)
    monkeypatch.setattr(sco, "STOP_FLAG", disk / "flags" / "STOP_SUBMIT_DISK_LOW")
    monkeypatch.setattr(sco, "WARN_FLAG", disk / "WARN_DISK_LOW")
    monkeypatch.setattr(sco, "USAGE_CSV", usage)
    monkeypatch.setattr(sco, "MAX_LIVE", 4)
    monkeypatch.setattr(sco_job, "code_state", lambda wt: {"commit": "fake", "branch": "fake",
                                                           "dirty_downstream_files": []})
    monkeypatch.setattr(sco, "vpd_code_state", lambda: {"commit": "fake", "vpd_status": [],
                                                        "vpd_tree_sha256": "0" * 64, "vpd_files": 0})
    monkeypatch.setattr(sco_job, "submit", f.submit)
    monkeypatch.setattr(sco_job, "status", f.status)
    monkeypatch.setattr(sco_job, "delete", f.force_delete)
    monkeypatch.setattr(sco_job, "pick_pool", lambda pools=sco_job.POOLS: "h800")
    monkeypatch.setattr(sco, "describe", f.describe)
    monkeypatch.setattr(sco, "list_page", f.list_page)
    monkeypatch.setattr(sco, "_sco_delete", f.sco_delete)
    return f


def prep(base, name, est=1.0, gate=None, as_root=False, pool="zoetrope"):
    run = base / "jobs" / name
    sco.prepare(run, name, ["bash", str(JOBS_DIR / "payload.sh"), "x"], pool=pool, as_root=as_root,
                est_gph=est, gate=gate)
    return run


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def do_submit(run, clock=None, **kw):
    clock = clock or Clock()
    return sco.submit(run, sleep=clock.sleep, clock=clock, **kw)


# --- names and prepare --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["vpd-wb-smoke", "vpd-e3vs-q3b", "vpd-tvr-smoke", "vpd-test", "vpd-wb-q3rs150"])
def test_valid_names(name):
    assert sco.validate_name(name) == name


@pytest.mark.parametrize("name", ["wb-smoke", "vpd_wb", "VPD-wb", "vpd-WB", "vpd-", "vpd--x", "vpd-x-", "ev-vpd",
                                  "evvpd-x", "vpd-wb smoke", "vpd-" + "a" * 37, "", None, "downstream-vpd-x"])
def test_invalid_names(name):
    with pytest.raises(ValueError):
        sco.validate_name(name)


def test_gate_of_and_caps(monkeypatch):
    assert ledger.gate_of("vpd-wb-smoke") == "wb"
    assert ledger.gate_of("vpd-tvr-smoke") == "tvr"
    assert ledger.gate_of("vpd-e3vs-prep") == "e3vs"
    assert ledger.gate_of("vpd-test") is None and ledger.gate_of("vpd-wbx") is None
    assert ledger.gate_caps() == {"wb": 12.0, "tvr": 0.5, "e3vs": 10.5}
    monkeypatch.setenv("VPD_GPUH_CAP", "20")
    assert ledger.total_cap() == 20.0 and ledger.gate_caps()["wb"] == 9.0
    with pytest.raises(ValueError):
        ledger.gate_caps(5.0)
    monkeypatch.setenv("VPD_GPUH_CAP", "-1")
    with pytest.raises(ValueError):
        ledger.total_cap()


def test_prepare_writes_launch_and_meta(base, fake):
    run = prep(base, "vpd-wb-smoke", est=0.8)
    launch = (run / "job" / "launch.sh").read_text()
    assert "cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd\n" in launch
    assert sco_job.RUN_AS_OWNER in launch  # non-root: payload runs as the file owner
    meta = json.loads((run / "job" / "vpd.json").read_text())
    assert meta["gate"] == "wb" and meta["est_gph"] == 0.8 and meta["display_name"] == "downstream-vpd-wb-smoke"
    assert meta["gpus"] == 1 and meta["as_root"] is False and meta["vpd_code"]["vpd_tree_sha256"] == "0" * 64
    prepared = json.loads((run / "job" / "prepared.json").read_text())
    assert prepared["spec"] == "N4lS.Iq.I80.1" and prepared["pool"] == "zoetrope" and "as_root" not in prepared

    root_run = prep(base, "vpd-wb-q3b", est=3.4, as_root=True, pool="auto")
    text = (root_run / "job" / "launch.sh").read_text()
    assert sco_job.RUN_AS_OWNER not in text and "cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd\n" in text
    prepared = json.loads((root_run / "job" / "prepared.json").read_text())
    assert prepared["as_root"] is True and prepared["pool"] == "h800"  # pick_pool fake

    t = prep(base, "vpd-test", est=None)
    assert json.loads((t / "job" / "vpd.json").read_text())["gate"] is None
    g = prep(base, "vpd-test2", est=0.1, gate="tvr")
    assert json.loads((g / "job" / "vpd.json").read_text())["gate"] == "tvr"


def test_prepare_refusals(base, fake):
    with pytest.raises(ValueError):
        prep(base, "vpd-wb-x", gate="e3vs")  # contradicts the name
    with pytest.raises(ValueError):
        prep(base, "dsx-wb")
    with pytest.raises(ValueError):
        prep(base, "vpd-wb-x2", est=0)
    with pytest.raises(ValueError):
        sco.prepare(base / "jobs" / "r", "vpd-wb-r", [], est_gph=1)
    run = base / "jobs" / "vpd-wb-self"
    with pytest.raises(ValueError, match="self-exec"):
        sco.prepare(run, "vpd-wb-self", ["bash", str(run / "job" / "launch.sh")], est_gph=1)
    with pytest.raises(ValueError):
        sco.prepare(base / "jobs" / "p", "vpd-wb-p", ["true"], pool="nowhere", est_gph=1)


# --- ledger -------------------------------------------------------------------------------------------------

def test_ledger_roundtrip_lock_and_errors(base, fake):
    lp = ledger.ledger_path()
    assert lp == base / "ledger.jsonl" and ledger.read() == []

    def writer(k):
        for i in range(25):
            ledger.append({"event": "submit", "name": f"vpd-wb-{k}-{i}", "run": f"/r/{k}/{i}", "gate": "wb",
                           "pt_id": f"pt-{k}{i}", "est": 0.01, "submitted_at": iso(T0)})

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    evs = ledger.read()
    assert len(evs) == 100 and len({e["run"] for e in evs}) == 100
    assert ledger.lock_path().exists()
    assert all(len(json.loads(line)) >= 7 for line in lp.read_text().splitlines())
    with pytest.raises(ValueError):
        ledger.append({"event": "bogus", "run": "/r"})
    with pytest.raises(ValueError):
        ledger.append({"event": "end"})
    with lp.open("a") as fh:
        fh.write("{not json\n")
    with pytest.raises(ValueError, match="malformed"):
        ledger.read()


def test_ledger_accounting(base, fake):
    E = [
        {"event": "submit", "name": "vpd-wb-a", "run": "/a", "gate": "wb", "pt_id": "pt-a", "est": 3.0},
        {"event": "submit", "name": "vpd-wb-b", "run": "/b", "gate": "wb", "pt_id": "pt-b", "est": 2.0},
        {"event": "end", "run": "/b", "pt_id": "pt-b", "state": "SUCCEEDED", "gph": 1.5},
        {"event": "delete", "run": "/b", "pt_id": "pt-b", "at": iso(T0)},
        {"event": "uncertain", "name": "vpd-e3vs-c", "run": "/c", "gate": "e3vs", "est": 4.0},
        {"event": "uncertain", "name": "vpd-tvr-d", "run": "/d", "gate": "tvr", "est": 0.4},
        {"event": "cleared", "run": "/d"},
        {"event": "end", "run": "/zz", "gph": 99.0},  # never submitted here: ignored
    ]
    for e in E:
        ledger.append(e)
    jobs = ledger.fold(ledger.read())
    assert jobs["/b"]["status"] == "ended" and jobs["/b"]["deleted_at"] == iso(T0)
    assert jobs["/c"]["status"] == "uncertain" and jobs["/d"]["status"] == "cleared" and "/zz" not in jobs
    s = ledger.summarise(ledger.read())
    wb, e3, tvr = s["by_gate"]["wb"], s["by_gate"]["e3vs"], s["by_gate"]["tvr"]
    assert (wb["spent"], wb["committed"], wb["open_jobs"], wb["ended_jobs"]) == (1.5, 3.0, 1, 1)
    assert (e3["spent"], e3["committed"]) == (0.0, 4.0) and (tvr["spent"], tvr["committed"]) == (0.0, 0.0)
    assert wb["remaining"] == 12.0 - 4.5 and s["total"]["spent"] == 1.5 and s["total"]["committed"] == 7.0
    # an open job running past its estimate is charged what it used so far
    s2 = ledger.summarise(ledger.read(), elapsed={"/a": 5.25})
    assert s2["by_gate"]["wb"]["committed"] == 5.25
    # an end without a measured gph is charged the estimate
    ledger.append({"event": "end", "run": "/a", "pt_id": "pt-a", "state": "FAILED", "gph": None})
    assert ledger.summarise(ledger.read())["by_gate"]["wb"]["spent"] == 4.5
    # resubmitting a cleared run counts again
    ledger.append({"event": "submit", "name": "vpd-tvr-d", "run": "/d", "gate": "tvr", "pt_id": "pt-d", "est": 0.4})
    assert ledger.summarise(ledger.read())["by_gate"]["tvr"]["committed"] == 0.4


def test_budget_check_at_cap(base, fake, monkeypatch):
    E = [{"event": "submit", "name": "vpd-wb-a", "run": "/a", "gate": "wb", "pt_id": "pt-a", "est": 3.0},
         {"event": "submit", "name": "vpd-wb-b", "run": "/b", "gate": "wb", "pt_id": "pt-b", "est": 9.0},
         {"event": "end", "run": "/b", "pt_id": "pt-b", "gph": 8.0}]
    assert ledger.budget_check(E, "wb", 1.0)[0]  # 8 + 3 + 1 = 12 = cap
    ok, msg = ledger.budget_check(E, "wb", 1.01)
    assert not ok and msg.startswith("BUDGET REFUSED")
    assert ledger.budget_check(E, "e3vs", 10.5)[0]  # the E3VS reservation is untouched by WB use
    assert not ledger.budget_check(E, "e3vs", 10.6)[0]
    assert ledger.budget_check(E, "tvr", 0.5)[0] and not ledger.budget_check(E, "tvr", 0.51)[0]
    assert not ledger.budget_check(E, "wb", 0.5, elapsed={"/a": 3.6})[0]  # /a over its estimate
    assert not ledger.budget_check(E, "wb", 0)[0] and not ledger.budget_check(E, "nope", 1)[0]
    monkeypatch.setenv("VPD_GPUH_CAP", "21")  # wb cap 10
    assert not ledger.budget_check(E, "wb", 0.1)[0]
    bad = E + [{"event": "submit", "name": "vpd-x", "run": "/x", "gate": None, "pt_id": "pt-x", "est": 1.0}]
    assert not ledger.budget_check(bad, "tvr", 0.1)[0]


def test_worker_log_span(base):
    log = base / "worker.log"
    log.write_text("[launch] host=h1 date=2026-11-08T10:00:00Z uid=0\nstuff\n"
                   "[launch] rc=0 date=2026-11-08T11:30:00Z\n")
    span = ledger.worker_log_span(log)
    assert span == ("2026-11-08T10:00:00Z", "2026-11-08T11:30:00Z") and ledger.hours_between(*span) == 1.5
    log.write_text("[launch] host=h1 date=2026-11-08T10:00:00Z uid=0\n")
    assert ledger.worker_log_span(log) is None and ledger.worker_log_span(base / "nope") is None


# --- submit guards ------------------------------------------------------------------------------------------

def test_submit_counts_live_jobs_against_budget(base, fake):
    """The J4 case: three pilot jobs submitted together each see the others' estimates."""
    clock = Clock()
    runs = [prep(base, f"vpd-wb-p{i}", est=3.4) for i in range(4)]
    pts = [do_submit(r, clock) for r in runs[:3]]
    assert pts == ["pt-fake0001", "pt-fake0002", "pt-fake0003"]
    with pytest.raises(sco.Refused, match="BUDGET REFUSED"):
        do_submit(runs[3], clock)  # 3 x 3.4 + 3.4 = 13.6 > 12
    assert len(fake.submits) == 3
    subs = [e for e in ledger.read() if e["event"] == "submit"]
    assert [e["pt_id"] for e in subs] == pts
    for e in subs:
        assert {"name", "run", "pt_id", "est", "submitted_at", "gate"} <= set(e) and e["est"] == 3.4
    assert clock.slept and all(abs(s - 2.0) < 1e-9 for s in clock.slept)  # >= 2 s between creates
    # the e3vs gate is unaffected by WB use
    e = prep(base, "vpd-e3vs-prep", est=2.0)
    assert do_submit(e, clock) == "pt-fake0004"
    # a finished job is charged its actual GPU-h (recorded once), freeing budget
    fake.set(pts[0], "SUCCEEDED", T0, T0 + datetime.timedelta(minutes=30))
    assert do_submit(runs[3], clock) == "pt-fake0005"  # 0.5 + 3 x 3.4 = 10.7 <= 12
    ends = [e for e in ledger.read() if e["event"] == "end"]
    assert len(ends) == 1 and ends[0]["gph"] == 0.5 and ends[0]["gph_source"] == "sco_start_to_complete"


def test_submit_refuses_twice_and_needs_gate_and_estimate(base, fake):
    r = prep(base, "vpd-wb-once", est=0.5)
    do_submit(r)
    with pytest.raises(sco.Refused, match="already submitted"):
        do_submit(r)
    t = prep(base, "vpd-test", est=0.5)
    with pytest.raises(sco.Refused, match="no gate"):
        do_submit(t)
    n = prep(base, "vpd-wb-noest", est=None)
    with pytest.raises(sco.Refused, match="estimate"):
        do_submit(n)
    assert do_submit(n, est_gph=0.2) == "pt-fake0002"
    with pytest.raises(sco.Refused, match="prepare"):
        do_submit(base / "jobs" / "never")
    assert len(fake.submits) == 2


def test_concurrency_refusal(base, fake):
    clock = Clock()
    runs = [prep(base, f"vpd-e3vs-c{i}", est=1.0) for i in range(6)]
    pts = [do_submit(r, clock) for r in runs[:4]]  # 0..3 others live: allowed
    with pytest.raises(sco.Refused, match="CONCURRENCY"):
        do_submit(runs[4], clock)
    fake.set(pts[1], "FAILED", T0, T0 + datetime.timedelta(minutes=6))
    assert do_submit(runs[4], clock) == "pt-fake0005"
    # a live downstream-vpd-* job of the user that is not in the ledger also counts
    fake.set(pts[2], "SUCCEEDED", T0, T0 + datetime.timedelta(minutes=6))
    fake.extra_list.append({"name": "pt-other", "display_name": "downstream-vpd-manual", "state": "RUNNING",
                            "ownership": {"user_name": "yinbaiqiao"}, "create_time": iso(T0)})
    fake.extra_list.append({"name": "pt-someone", "display_name": "downstream-vpd-x", "state": "RUNNING",
                            "ownership": {"user_name": "someone"}, "create_time": iso(T0)})
    with pytest.raises(sco.Refused, match="CONCURRENCY"):
        do_submit(runs[5], clock)
    fake.extra_list.clear()
    # live now: pts[0], pts[3], pt-fake0005.  An unreadable state (NOT_FOUND that the list does not confirm)
    # counts as live, and so does an uncertain create.
    fake.status_fail.add(pts[0])
    fake.list_ok = False
    rows, _ = sco.scan(ledger.read(), sco.list_page(), close=False)
    row0 = [r for r in rows if r["pt_id"] == pts[0]][0]
    assert row0["state"] == "NOT_FOUND" and row0["live"] is True
    fake.list_ok = True  # the list still shows pts[0]: still live
    rows, _ = sco.scan(ledger.read(), sco.list_page(), close=False)
    assert [r for r in rows if r["pt_id"] == pts[0]][0]["live"] is True
    ledger.append({"event": "uncertain", "name": "vpd-e3vs-u", "run": str(base / "jobs" / "u"), "gate": "e3vs",
                   "est": 1.0, "at": iso(T0)})
    with pytest.raises(sco.Refused, match="CONCURRENCY"):
        do_submit(runs[5], clock)
    fake.status_fail.clear()
    fake.set(pts[3], "STOPPED", T0, T0 + datetime.timedelta(minutes=6))
    assert do_submit(runs[5], clock) == "pt-fake0006"
    # a job that vanished from the list (confirmed absent) is no longer live
    fake.jobs.pop(pts[0])
    rows, _ = sco.scan(ledger.read(), sco.list_page(), close=False)
    assert [r for r in rows if r["pt_id"] == pts[0]][0]["live"] is False


def test_disk_refusals(base, fake):
    r = prep(base, "vpd-wb-disk", est=0.5)
    ok, msgs = sco.disk_check()
    assert ok and any("remaining 2.0 TB" in m for m in msgs)
    flag = sco.DISK_RUNS / "FLAG_PAUSE"
    flag.write_text("x")
    with pytest.raises(sco.Refused, match="DISK REFUSED"):
        do_submit(r)
    flag.unlink()
    sco.STOP_FLAG.write_text('{"remaining": 1.0}')
    with pytest.raises(sco.Refused, match="stop flag"):
        do_submit(r)
    sco.STOP_FLAG.unlink()
    text = sco.USAGE_CSV.read_text().replace(",2.0,1,", ",1.4,1,")
    sco.USAGE_CSV.write_text(text)
    with pytest.raises(sco.Refused, match="1.4 TB < 1.5"):
        do_submit(r)
    sco.USAGE_CSV.write_text("utc,quota_TB,table_used_TB,remaining_TB\n")
    with pytest.raises(sco.Refused, match="cannot read"):
        do_submit(r)
    sco.USAGE_CSV.unlink()
    with pytest.raises(sco.Refused, match="cannot read"):
        do_submit(r)
    sco.WARN_FLAG.write_text("x")
    sco.USAGE_CSV.write_text(text.replace(",1.4,1,", ",1.5,1,"))
    ok, msgs = sco.disk_check()
    assert ok and any("warning only" in m for m in msgs)  # WARN never blocks; exactly 1.5 TB passes
    assert fake.submits == []
    assert do_submit(r) == "pt-fake0001"


def test_submit_uncertain_is_never_retried(base, fake, capsys, monkeypatch):
    r = prep(base, "vpd-wb-unc", est=2.0)
    fake.uncertain_next = True
    with pytest.raises(sco_job.SubmitUncertain):
        do_submit(r)
    assert "FAKE: dns timeout" in capsys.readouterr().err  # submit.log printed
    assert len(fake.submits) == 1
    evs = ledger.read()
    assert [e["event"] for e in evs] == ["uncertain"]
    fake.uncertain_next = False
    with pytest.raises(sco.Refused, match="uncertain"):
        do_submit(r)
    assert len(fake.submits) == 1
    # the uncertain job is charged and counted as live
    s = ledger.summarise(ledger.read())
    assert s["by_gate"]["wb"]["committed"] == 2.0
    assert sco.main(["submit", "--run", str(r)]) == 2
    # resolve adopts the job found in the list; never creates one
    def fake_resolve(run, jobs):
        (Path(run) / "job" / "submission.json").write_text(json.dumps({"job_id": "pt-adopted"}))
        (Path(run) / "job" / sco_job.UNCERTAIN).unlink()
        fake.jobs["pt-adopted"] = {"state": "RUNNING", "start_time": iso(T0), "complete_time": None,
                                   "display_name": "downstream-vpd-wb-unc", "create_time": iso(T0)}
        return ("adopted", "pt-adopted")
    monkeypatch.setattr(sco_job, "resolve_uncertain", fake_resolve)
    assert sco.resolve(r) == ("adopted", "pt-adopted")
    job = ledger.fold(ledger.read())[str(r.resolve())]
    assert job["status"] == "submitted" and job["pt_id"] == "pt-adopted"
    assert len(fake.submits) == 1


def test_uncertain_main_exit_code(base, fake):
    r = prep(base, "vpd-wb-unc2", est=1.0)
    fake.uncertain_next = True
    assert sco.main(["submit", "--run", str(r)]) == 3


# --- status and delete --------------------------------------------------------------------------------------

def test_status_records_end_once(base, fake):
    r = prep(base, "vpd-tvr-smoke", est=0.5)
    pt = do_submit(r)
    st = sco.status()
    assert st["jobs"][0]["live"] and st["budget"]["by_gate"]["tvr"]["committed"] == 0.5
    fake.set(pt, "SUCCEEDED", T0, T0 + datetime.timedelta(minutes=12))
    st = sco.status()
    assert st["jobs"][0]["live"] is False and st["jobs"][0]["gph"] == 0.2
    sco.status()
    ends = [e for e in ledger.read() if e["event"] == "end"]
    assert len(ends) == 1
    assert sco.status()["jobs"] == [] and len(sco.status(include_ended=True)["jobs"]) == 1
    assert sco.status(include_ended=True)["budget"]["by_gate"]["tvr"]["spent"] == 0.2


def test_delete_rules(base, fake):
    clock = Clock()
    r1 = prep(base, "vpd-wb-d1", est=1.0)
    r2 = prep(base, "vpd-wb-d2", est=1.0)
    r3 = prep(base, "vpd-wb-d3", est=1.0)
    p1, p2, p3 = (do_submit(r, clock) for r in (r1, r2, r3))
    fake.set(p1, "RUNNING", T0)
    with pytest.raises(sco.Refused, match="force-live"):
        sco.delete(r1, sleep=clock.sleep)
    assert fake.deleted == [] and fake.stopped == []
    # a terminal job: deleted without stop, end + delete recorded
    fake.set(p1, "SUCCEEDED", T0, T0 + datetime.timedelta(minutes=45))
    out = sco.delete(r1, sleep=clock.sleep)
    assert "SUCCEEDED" in out and fake.deleted == [p1] and fake.stopped == []
    evs = [e for e in ledger.read() if e["run"] == str(r1.resolve())]
    assert [e["event"] for e in evs] == ["submit", "end", "delete"] and evs[1]["gph"] == 0.75
    # NOT_FOUND: only when the list confirms the job is gone
    fake.status_fail.add(p2)
    fake.list_ok = False
    with pytest.raises(sco.Refused):
        sco.delete(r2, sleep=clock.sleep)
    fake.list_ok = True
    with pytest.raises(sco.Refused):  # still in the list
        sco.delete(r2, sleep=clock.sleep)
    fake.jobs.pop(p2)
    (r2 / "job" / "worker.log").write_text("[launch] host=h date=2026-11-08T10:00:00Z uid=0\n"
                                           "[launch] rc=1 date=2026-11-08T10:06:00Z\n")
    assert "recorded as deleted" in sco.delete(r2, sleep=clock.sleep)
    end2 = [e for e in ledger.read() if e["run"] == str(r2.resolve()) and e["event"] == "end"][0]
    assert end2["gph"] == 0.1 and end2["gph_source"] == "worker_log" and end2["state"] == "NOT_FOUND"
    # a live job with --force-live: stop + delete through sco_job.delete
    fake.set(p3, "RUNNING", datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=30))
    sco.delete(r3, force_live=True, sleep=clock.sleep)
    assert fake.stopped == [p3]
    end3 = [e for e in ledger.read() if e["run"] == str(r3.resolve()) and e["event"] == "end"][0]
    assert end3["state"] == "FORCED_FROM_RUNNING" and 0.45 < end3["gph"] < 0.6
    s = ledger.summarise(ledger.read())["by_gate"]["wb"]
    assert s["open_jobs"] == 0 and s["ended_jobs"] == 3
    with pytest.raises(sco.Refused, match="nothing was submitted"):
        sco.delete(base / "jobs" / "none", sleep=clock.sleep)


# --- misc ---------------------------------------------------------------------------------------------------

def test_vpd_tree_digest(base):
    root = base / "tree"
    (root / "a" / "__pycache__").mkdir(parents=True)
    (root / "a" / "x.py").write_text("x = 1\n")
    (root / "y.sh").write_text("echo\n")
    (root / "a" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"junk")
    d1, n1 = sco.vpd_tree_digest(root)
    (root / "a" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"other")
    assert sco.vpd_tree_digest(root) == (d1, n1) and n1 == 2
    (root / "a" / "x.py").write_text("x = 2\n")
    assert sco.vpd_tree_digest(root)[0] != d1


def test_live_vpd_from_list_and_absent():
    page = [{"name": "pt-1", "display_name": "downstream-vpd-wb-a", "state": "RUNNING",
             "ownership": {"user_name": "yinbaiqiao"}, "create_time": "2026-11-08T10:00:00Z"},
            {"name": "pt-2", "display_name": "downstream-vpd-wb-b", "state": "SUCCEEDED",
             "ownership": {"user_name": "yinbaiqiao"}},
            {"name": "pt-3", "display_name": "downstream-ev-x", "state": "RUNNING",
             "ownership": {"user_name": "yinbaiqiao"}},
            {"name": "pt-4", "display_name": "downstream-vpd-wb-c", "state": "SUSPENDED",
             "ownership": {"user_name": "yinbaiqiao"}}]
    assert sorted(sco.live_vpd_from_list(page)) == ["pt-1", "pt-4"]
    assert sco.absent_confirmed("pt-9", "2026-11-08T11:00:00Z", page)
    assert not sco.absent_confirmed("pt-1", "2026-11-08T11:00:00Z", page)
    assert not sco.absent_confirmed("pt-9", "2026-11-08T11:00:00Z", None)


def test_cli_prepare_and_errors(base, fake, capsys):
    run = base / "jobs" / "cli"
    assert sco.main(["prepare", "--run", str(run), "--name", "vpd-test", "--", "echo", "hi"]) == 0
    assert "cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd" in (run / "job" / "launch.sh").read_text()
    assert sco.main(["prepare", "--run", str(base / "jobs" / "bad"), "--name", "ev-x", "--", "echo"]) == 2
    assert sco.main(["budget"]) == 0
    assert '"by_gate"' in capsys.readouterr().out
    assert sco.main(["disk"]) == 0


def test_import_is_the_vpd_worktree_copy():
    assert Path(sco_job.__file__).resolve() == (paths.WORKTREE / "downstream" / "sco_job.py").resolve()
    assert Path(sco_job.WORKTREE).resolve() == paths.WORKTREE.resolve()


# --- shell --------------------------------------------------------------------------------------------------

def test_shell_syntax():
    for f in ("common_env.sh", "preflight.sh"):
        res = subprocess.run(["bash", "-n", str(JOBS_DIR / f)], capture_output=True, text=True)
        assert res.returncode == 0, (f, res.stderr)


def _bash(script: str, run: Path, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VPD_")}
    env.update({"RUN": str(run)}, **(extra_env or {}))
    return subprocess.run(["bash", "-c", f"source {JOBS_DIR / 'common_env.sh'}\n{script}"], capture_output=True,
                          text=True, env=env, timeout=120)


def test_common_env_exports_and_steps(base):
    run = base / "run"
    res = _bash('echo "T=$OMP_NUM_THREADS/$MKL_NUM_THREADS/$OPENBLAS_NUM_THREADS/$NUMEXPR_NUM_THREADS"\n'
                'echo "E=$TOKENIZERS_PARALLELISM/$HF_HUB_OFFLINE/$TRANSFORMERS_OFFLINE/$PYTHONDONTWRITEBYTECODE"\n'
                'echo "P=$PYTHONPATH"\n'
                'vpd_threads 4; echo "V=$OMP_NUM_THREADS/$NUMEXPR_NUM_THREADS/$VPD_THREADS"\n'
                'echo "S=$-"\n'
                'vpd_step ok_step true\n'
                'vpd_step fail_step bash -c "exit 7" || echo "rc=$?"\n'
                'vpd_step "bad name" true || echo "badname=$?"\n'
                'vpd_chown_trap "$RUN"; echo chown_ok\n', run)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "T=2/2/2/2" in out and "E=false/1/1/1" in out and "P=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd" in out
    assert "V=4/4/4" in out and "rc=7" in out and "badname=2" in out and "chown_ok" in out
    flags = [ln for ln in out.splitlines() if ln.startswith("S=")][0]
    assert "e" in flags and "u" in flags
    steps = [json.loads(ln) for ln in (run / "steps.jsonl").read_text().splitlines()]
    assert [(s["name"], s["rc"]) for s in steps] == [("ok_step", 0), ("fail_step", 7)]
    assert all(set(s) == {"name", "start", "end", "secs", "rc"} for s in steps)
    # VPD_THREADS overrides the default
    res = _bash('echo "T=$OMP_NUM_THREADS"', run, {"VPD_THREADS": "1"})
    assert "T=1" in res.stdout


def test_common_env_strict_mode_and_exit_handlers(base):
    run = base / "run2"
    # a failing step stops the payload (set -e) and the EXIT handlers run in order, keeping the exit code
    res = _bash('vpd_on_exit "echo first-handler"\nvpd_on_exit "echo second-handler"\n'
                'vpd_step will_fail false\necho not-reached\n', run)
    assert res.returncode == 1
    assert "not-reached" not in res.stdout
    assert res.stdout.index("first-handler") < res.stdout.index("second-handler")
    res = _bash('echo "$UNSET_VARIABLE_X"\n', run)
    assert res.returncode != 0  # set -u
    res = _bash('vpd_step needs_run true', run, {"RUN": ""})
    assert res.returncode == 2


def test_node_facts(base):
    out = base / "facts" / "node.txt"
    res = _bash(f'vpd_node_facts {out}\necho done', base)
    assert res.returncode == 0 and "done" in res.stdout, res.stderr
    text = out.read_text()
    for sec in ("## id", "## nproc", "## cpu.max", "## df", "## free", "## nvidia-smi", "## devices"):
        assert sec in text


def test_disk_exception(base, fake, monkeypatch):
    out = base / "out"
    out.mkdir()
    monkeypatch.setattr(sco.paths, "OUT", out)
    monkeypatch.setattr(sco, "DISK_EXCEPTION", out / "DISK_EXCEPTION.json")
    r = prep(base, "vpd-wb-exc", est=0.5)
    sco.STOP_FLAG.write_text('{"remaining": 1.0}')
    text = sco.USAGE_CSV.read_text().replace(",2.0,1,", ",1.0,1,")
    sco.USAGE_CSV.write_text(text)
    with pytest.raises(sco.Refused, match="no user disk exception"):
        do_submit(r)
    sco.DISK_EXCEPTION.write_text(json.dumps({"gates": ["wb", "tvr"], "cap_gb": 10, "by": "test"}))
    (out / "big.bin").write_bytes(b"x" * 4096)
    assert sco.disk_exception_ok("e3vs")[0] is False
    assert sco.disk_exception_ok("wb")[0] is True
    monkeypatch.setattr(sco, "tree_bytes", lambda root: int(11e9))
    assert "exception cap" in sco.disk_exception_ok("wb")[1]
    monkeypatch.setattr(sco, "tree_bytes", lambda root: 4096)
    flag = sco.DISK_RUNS / "FLAG_PAUSE"
    flag.write_text("x")
    assert sco.disk_exception_ok("wb")[0] is False
    flag.unlink()
    sco.USAGE_CSV.write_text(text.replace(",1.0,1,", ",0.4,1,"))
    assert "hard floor" in sco.disk_exception_ok("wb")[1]
    sco.USAGE_CSV.write_text(text)
    assert fake.submits == []
    assert do_submit(r) == "pt-fake0001"


def test_no_cap_policy(base, fake, monkeypatch):
    """User 2026-11-09: BUDGET_POLICY.json with gpuh_cap null lifts the GPU-h cap; disk exception cap_gb null."""
    out = base / "out_nocap"
    out.mkdir()
    monkeypatch.setattr(ledger, "POLICY", out / "BUDGET_POLICY.json")
    monkeypatch.setattr(sco.paths, "OUT", out)
    monkeypatch.setattr(sco, "DISK_EXCEPTION", out / "DISK_EXCEPTION.json")
    ok, msg = ledger.budget_check([], "tvr", 5.0)
    assert not ok  # 5 > the 0.5 tvr reservation while capped
    ledger.POLICY.write_text(json.dumps({"gpuh_cap": None, "by": "test"}))
    ok, msg = ledger.budget_check([], "tvr", 5.0)
    assert ok and "no cap" in msg
    sco.DISK_EXCEPTION.write_text(json.dumps({"gates": ["wb", "tvr", "e3vs"], "cap_gb": None, "by": "test"}))
    monkeypatch.setattr(sco, "tree_bytes", lambda root: int(1e15))
    ok, msg = sco.disk_exception_ok("e3vs")
    assert ok and "no size cap" in msg
    flag = sco.DISK_RUNS / "FLAG_PAUSE"
    flag.write_text("x")
    assert sco.disk_exception_ok("e3vs")[0] is False  # other disk-line flags still refuse
    flag.unlink()
