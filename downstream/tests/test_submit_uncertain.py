"""Uncertain sco submissions (2026-10-30 22:45: sco's DNS lookup timed out while the watcher submitted C1's short job,
and the watcher, which let sco_job.submit's SystemExit through, stayed down until restarted by hand).  sco_job.submit
leaves a mark and then refuses the run; resolve_uncertain adopts the job from the job list, clears the mark ten
minutes after the attempt when no job appeared, and waits while the list is unreadable or names more than one job.
The watcher goes on, submits nothing more in that pass, and settles the mark in later passes."""
import datetime
import json
from types import SimpleNamespace

import pytest

from test_watch_and_tasks import FakeJobs, busy_view, sco_job, watcher

UTC = datetime.timezone.utc


def prepared_run(root, name="c1-short"):
    run = root / "run"
    (run / "job").mkdir(parents=True)
    (run / "job/prepared.json").write_text(json.dumps({"pool": "h800", "name": name, "spec": "N4lS.Iq.I80.8"}))
    return run


def mark(run, minutes_ago, name="downstream-c1-short"):
    t = datetime.datetime.now(UTC) - datetime.timedelta(minutes=minutes_ago)
    (run / "job" / sco_job.UNCERTAIN).write_text(json.dumps({"utc": t.isoformat(), "job_name": name}))
    return t


def listed(name, t, jid, state="RUNNING"):
    return {"name": jid, "display_name": name, "create_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "state": state}


def test_a_failed_create_leaves_a_mark_and_the_run_is_refused_until_settled(tmp_path, monkeypatch):
    run = prepared_run(tmp_path)
    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="dial tcp: lookup aec2: i/o timeout")

    monkeypatch.setattr(sco_job.subprocess, "run", fake_run)
    with pytest.raises(sco_job.SubmitUncertain) as e:
        sco_job.submit(run)
    assert isinstance(e.value, SystemExit) and e.value.run == run  # the command line still stops
    rec = json.loads((run / "job" / sco_job.UNCERTAIN).read_text())
    assert rec["job_name"] == "downstream-c1-short" and rec["returncode"] == 1 and rec["ids"] == []
    assert "i/o timeout" in (run / "job/submit.log").read_text()
    with pytest.raises(sco_job.SubmitUncertain):
        sco_job.submit(run)
    assert len(calls) == 1  # refused without asking sco again


def test_resolve_adopts_the_new_job_clears_after_ten_minutes_and_otherwise_waits(tmp_path):
    run = prepared_run(tmp_path / "a")
    t = mark(run, 2)
    old = listed("downstream-c1-short", t - datetime.timedelta(hours=3), "pt-old", "SUCCEEDED")  # an earlier one
    other = listed("downstream-other", t, "pt-other")
    assert sco_job.resolve_uncertain(run, None) == ("pending", None)  # no job list
    assert sco_job.resolve_uncertain(run, [old, other]) == ("pending", None)  # two minutes: wait
    new = listed("downstream-c1-short", t + datetime.timedelta(seconds=20), "pt-new")
    assert sco_job.resolve_uncertain(run, [new, dict(new, name="pt-new2")]) == ("ambiguous", ["pt-new", "pt-new2"])
    assert sco_job.resolve_uncertain(run, [old, new, other]) == ("adopted", "pt-new")
    assert json.loads((run / "job/submission.json").read_text())["job_id"] == "pt-new"
    assert not (run / "job" / sco_job.UNCERTAIN).exists()
    assert list((run / "job").glob("submit_uncertain_adopted_*.json"))
    assert sco_job.resolve_uncertain(run, []) is None  # nothing to settle

    run2 = prepared_run(tmp_path / "b")
    mark(run2, 11)
    assert sco_job.resolve_uncertain(run2, None) == ("pending", None)  # an unreadable list never clears
    assert sco_job.resolve_uncertain(run2, [old, other]) == ("cleared", None)
    assert not (run2 / "job" / sco_job.UNCERTAIN).exists() and not (run2 / "job/submission.json").exists()
    assert list((run2 / "job").glob("submit_uncertain_cleared_*.json"))


class UncertainJobs(FakeJobs):
    """sco fails once for the runs named in `fail`, leaving the mark as sco_job.submit does."""

    def __init__(self, fail=(), **kw):
        super().__init__(**kw)
        self.fail = set(fail)

    def submit(self, run):
        if run.name in self.fail:
            self.fail.discard(run.name)
            (run / "job" / sco_job.UNCERTAIN).write_text(json.dumps({"utc": sco_job.now(),
                                                                     "job_name": f"downstream-{run.name}"}))
            raise sco_job.SubmitUncertain(run, "submission uncertain; read job/submit.log")
        return super().submit(run)


ARGS = ("--exps", "DS-B-100-s1", "--pool-nodes", "h800:0", "--yield-pool-only")


def queued(tmp_path, jobs, names):
    (tmp_path / "queue").mkdir()
    out = []
    for i, n in enumerate(names):
        run = tmp_path / "x" / n
        run.mkdir(parents=True)
        jobs.prepare(run, 1, "h800", n, ["bash", "x.sh"])
        (tmp_path / f"queue/q{i}_{n}.json").write_text(json.dumps({"run": str(run), "cards": 1, "what": n,
                                                                   "pools": ["h800"]}))
        out.append(run)
    return out


def test_the_watcher_goes_on_and_adopts_the_job_it_finds(tmp_path):
    jobs = UncertainJobs(fail={"c1"}, views={"h800": busy_view(("10.0.0.1", [("other", 4)]))})
    c1, n1 = queued(tmp_path, jobs, ["c1", "n1"])
    w = watcher(tmp_path, *ARGS, jobs=jobs)
    w.tick()  # no exit; nothing more is submitted in this pass
    assert (c1 / "job" / sco_job.UNCERTAIN).exists() and not (c1 / "job/submission.json").exists()
    assert not (n1 / "job/submission.json").exists()
    assert w.state["uncertain"] == [str(c1)] and any("提交结果不明" in a for a in w.alerts)
    assert w.blocked(1, "h800") == "本轮有一次提交结果不明，暂停提交"

    w = watcher(tmp_path, *ARGS, jobs=jobs)  # next pass: the job list has nothing yet, the next entry goes
    w.tick()
    assert not (c1 / "job/submission.json").exists() and (n1 / "job/submission.json").exists()
    assert any("还在核对" in a for a in w.alerts) and any("上次提交结果不明" in n for n in w.notes)

    jobs.listing = [listed("downstream-c1", datetime.datetime.now(UTC), "pt-c1real")]
    w = watcher(tmp_path, *ARGS, jobs=jobs)  # sco did make it after all
    w.tick()
    assert json.loads((c1 / "job/submission.json").read_text())["job_id"] == "pt-c1real"
    assert w.state["uncertain"] == [] and any("已认领" in n for n in w.notes)


def test_without_a_job_ten_minutes_later_the_mark_is_cleared_and_the_run_submitted_again(tmp_path):
    jobs = UncertainJobs(fail={"c1"}, views={"h800": busy_view(("10.0.0.1", [("other", 4)]))})
    (c1,) = queued(tmp_path, jobs, ["c1"])
    watcher(tmp_path, *ARGS, jobs=jobs).tick()
    mark(c1, 11, name="downstream-c1")  # the attempt is eleven minutes old
    jobs.list_jobs = lambda: None  # an unreadable job list never clears it
    w = watcher(tmp_path, *ARGS, jobs=jobs)
    w.tick()
    assert (c1 / "job" / sco_job.UNCERTAIN).exists() and any("读不出" in a for a in w.alerts)
    jobs.list_jobs = lambda: []
    w = watcher(tmp_path, *ARGS, jobs=jobs)
    w.tick()
    assert json.loads((c1 / "job/submission.json").read_text())["job_id"] == "pt-c1"  # submitted again
    assert any("标记已清掉" in n for n in w.notes) and w.state["uncertain"] == []
