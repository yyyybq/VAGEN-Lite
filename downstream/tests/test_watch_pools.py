"""--pools (the user's instruction of 2026-11-02, relayed by the main window: every sco job not yet running, and every
new one, goes to zoetrope): a queue entry's "pools", --whole-pools and the one-card shards are cut down to --pools."""
import json

from test_watch_and_tasks import FakeJobs, busy_view, watcher

ROOM = busy_view(("10.0.0.1", [("other", 4)]))


def queued(tmp_path, jobs, pools):
    (tmp_path / "queue").mkdir()
    run = tmp_path / "x" / "n2"
    run.mkdir(parents=True)
    jobs.prepare(run, 1, "h800", "n2", ["bash", "x.sh"])
    (tmp_path / "queue/q0_n2.json").write_text(json.dumps({"run": str(run), "cards": 1, "what": "n2", "pools": pools}))
    return run


def test_allowed_cuts_a_pool_list_down_to_pools(tmp_path):
    w = watcher(tmp_path, "--pools", "zoetrope")
    assert w.allowed(["h800", "zoetrope"]) == ["zoetrope"]
    assert w.allowed(["h800"]) == ["zoetrope"]  # --whole-pools is h800 by default: nothing left there, so --pools
    assert watcher(tmp_path).allowed(["h800", "zoetrope"]) == ["h800", "zoetrope"]  # default: both pools


def test_a_queue_entry_goes_only_to_a_pool_of_pools(tmp_path):
    jobs = FakeJobs(views={"h800": ROOM, "zoetrope": ROOM})
    run = queued(tmp_path, jobs, ["h800", "zoetrope"])
    watcher(tmp_path, "--pool-nodes", "h800:0", "zoetrope:0", "--pools", "zoetrope", jobs=jobs).tick()
    assert (run / "job/submission.json").exists()
    assert json.loads((run / "job/prepared.json").read_text())["pool"] == "zoetrope"


def test_without_pools_the_entry_takes_its_first_pool_as_before(tmp_path):
    jobs = FakeJobs(views={"h800": ROOM, "zoetrope": ROOM})
    run = queued(tmp_path, jobs, ["h800", "zoetrope"])
    watcher(tmp_path, "--pool-nodes", "h800:0", "zoetrope:0", jobs=jobs).tick()
    assert json.loads((run / "job/prepared.json").read_text())["pool"] == "h800"
