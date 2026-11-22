"""k_racefix: plan "endpoint_race_fix" in train_chain.Chain._ppo_runs (q38, the user's request of 2026-11-21).

The real Chain._ppo_runs poll loop runs with job_sco.prepare / submit / status / delete stubbed (CPU only, no sco):
a fake sco (Sim) answers each job's state from a per-scenario script, a fake clock advances on every chain sleep, and
every call is logged with the poll it came in.

    cd k_racefix/wt && python -m pytest -q -p no:cacheprovider ../test_endpoint_race.py

(i)   gate on, endpoint COMPLETE, render service SUCCEEDED while sco still shows the trainer RUNNING: no retry, no
      submission, no deletion of a job sco shows alive; next poll the trainer is SUCCEEDED and the run ends normally.
(ii)  gate off (absent or false): the old behaviour, the renderer retry of 2026-11-21 18:01:01Z (the finished trainer's
      receipt deleted while sco shows it RUNNING, renderer and trainer submitted again).
(iii) gate on, no endpoint checkpoint, render service FAILED: the renderer retry as before (same calls as gate off).
(iv)  gate on, trainer FAILED before the endpoint: the trainer retry as before (same calls as gate off).
plus: the grace for a trainer sco keeps showing after its endpoint, the guard hole at the endpoint, and an endpoint
folder without COMPLETE (a save under way) left to the old behaviour.
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

HERE = Path(__file__).resolve().parent
WT = Path(os.environ.get("RACEFIX_WT", HERE / "wt"))
sys.path.insert(0, str(WT))

from data_gen.interiorgs_v10 import job_sco  # noqa: E402
from data_gen.interiorgs_v10 import train_chain as T  # noqa: E402

ENDPOINT = 300
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED", "KILLED"}
FLOWS = {"default": {}, "renderer_first": {"renderer_first": True}}


def log_line(step, entropy=0.1):
    return (f"(TaskRunner pid=1) step:{step} - episode/success_rate:0.5 - actor/entropy:{entropy} - "
            f"actor/kl_loss:0.01 - timing_s/step:60.0\n")


class Sim:
    """A fake sco.  ``behave(sim, run_dir, job_dir, n, calls)`` gives the state of the n-th job of ``job_dir``
    (``calls``: how many status calls that job has had, this one included); hooks run when a poll number is reached."""

    def __init__(self, root: Path, behave):
        self.root = root
        self.behave = behave
        self.poll = 0
        self.t = time.time()
        self.count = {}
        self.status_calls = {}
        self.calls = []
        self.live_deletes = []
        self.hooks = {}

    # the chain's sleep: one poll further, the clock with it
    def tick(self, seconds):
        self.t += seconds
        self.poll += 1
        for fn in self.hooks.pop(self.poll, []):
            fn()

    def at(self, poll, fn):
        self.hooks.setdefault(poll, []).append(fn)

    def _norm(self, args):
        return tuple(str(a).replace(str(self.root), "<root>") for a in args)

    def _state(self, run, job_dir, job_id, count=False):
        n = int(job_id.rsplit("-", 1)[1])
        key = (job_dir, n)
        if count:
            self.status_calls[key] = self.status_calls.get(key, 0) + 1
        return self.behave(self, run, job_dir, n, self.status_calls.get(key, 0))

    def prepare(self, run, gpus, args, job_dir="job", pool=None):
        (run / job_dir).mkdir(parents=True, exist_ok=True)
        self.calls.append((self.poll, "prepare", job_dir, gpus, self._norm(args)))

    def submit(self, run, job_dir="job"):
        n = self.count[job_dir] = self.count.get(job_dir, 0) + 1
        job_id = f"pt-{job_dir}-{n}"
        (run / job_dir).mkdir(parents=True, exist_ok=True)
        (run / job_dir / "submission.json").write_text(json.dumps({"job_id": job_id}))
        self.calls.append((self.poll, "submit", job_dir, job_id))

    def status(self, run, job_dir="job"):
        receipt = run / job_dir / "submission.json"
        if not receipt.exists():
            return {"state": "NOT_FOUND", "finished": False}
        job_id = json.loads(receipt.read_text())["job_id"]
        state = self._state(run, job_dir, job_id, count=True)
        return {"job_id": job_id, "state": state, "finished": state in FINISHED}

    def delete(self, run, job_dir="job"):
        receipt = run / job_dir / "submission.json"
        if not receipt.exists():
            self.calls.append((self.poll, "delete", job_dir, None))
            return None
        job_id = json.loads(receipt.read_text())["job_id"]
        state = self._state(run, job_dir, job_id)
        if state not in FINISHED:
            self.live_deletes.append((self.poll, job_dir, job_id, state))
        receipt.replace(receipt.with_name(f"submission_{job_id}_deleted.json"))
        self.calls.append((self.poll, "delete", job_dir, job_id))
        return {"delete_rc": 0}

    def submits(self, job_dir):
        return [c for c in self.calls if c[1] == "submit" and c[2] == job_dir]


def finish_trainer(sim, d, complete=True, entropy=0.1):
    """What the trainer job does at the endpoint: the checkpoint (with COMPLETE), its last log line, STOP_RENDERER."""
    ck = d / "checkpoints" / f"global_step_{ENDPOINT}"
    (ck / "actor").mkdir(parents=True, exist_ok=True)
    if complete:
        (ck / "COMPLETE").write_text(str(ENDPOINT))
        os.utime(ck / "COMPLETE", (sim.t, sim.t))
        (d / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(str(ENDPOINT))
        with open(d / "logs" / "rl-20261121T095952Z-pt-x-worker-0.log", "a") as f:
            f.write(log_line(ENDPOINT, entropy))
        (d / "STOP_RENDERER").touch()
        (d / "logs" / "rl_exit.json").write_text('{"mode":"rl","exit_status":0}')


def setup_run(sim, root, last_step, entropies=None):
    """A run in the middle of its training: render service and trainer submitted and running, the address written."""
    d = root / "rl_run_a"
    (d / "logs").mkdir(parents=True)
    entropies = entropies or {}
    (d / "logs" / "rl-20261121T095952Z-pt-x-worker-0.log").write_text(
        "".join(log_line(s, entropies.get(s, 0.1)) for s in range(1, last_step + 1)))
    sim.submit(d, "job_renderer")
    sim.submit(d, "job_rl")
    (d / "renderer_endpoint.txt").write_text("http://10.0.0.1:8915/render\n")
    sim.calls.clear()  # only the chain's own calls are compared
    return d


def renderer_service(sim, run, n, fail_first_at=None):
    """A render service: RUNNING (the n >= 2 ones write their address when first seen running) until STOP_RENDERER,
    then SUCCEEDED; ``fail_first_at``: the first one is FAILED from that poll on."""
    if n == 1 and fail_first_at is not None and sim.poll >= fail_first_at:
        return "FAILED"
    if (run / "STOP_RENDERER").exists():
        return "SUCCEEDED"
    if n >= 2 and not (run / "renderer_endpoint.txt").exists():
        (run / "renderer_endpoint.txt").write_text("http://10.0.0.2:8915/render\n")
    return "RUNNING"


def make_chain(root, sim, gate, rl=None, **extra):
    plan = {"tr": str(root), "sft_dir": str(root), "endpoint": ENDPOINT, "gs_root": "/ply", "retries": 2,
            "startup_retries_separate": True, "rl_runs": {"a": {"rl": dict(rl or {})}}}
    if gate is not None:
        plan["endpoint_race_fix"] = gate
    plan.update(extra)
    return T.Chain(plan, sleep=sim.tick, clock=lambda: sim.t)


def run_chain(chain, sim):
    with mock.patch.multiple(job_sco, prepare=sim.prepare, submit=sim.submit, status=sim.status, delete=sim.delete), \
            mock.patch.object(T, "ORDER_GAP_S", 0):
        chain._ppo_runs("rl", ENDPOINT)
    return chain.state["steps"]["rl"]["jobs"].get("a/ended")


def has_switch():
    return hasattr(T.Chain, "endpoint_done")


# ------------------------------------------------------------------ the race at the endpoint (i, ii)
def race_behave(trainer_lag):
    """The trainer reaches the endpoint between poll 0 and 1 (hook); sco shows it RUNNING for ``trainer_lag`` more
    polls; a later trainer ends at once ("[q38] already at the endpoint ... nothing to train")."""

    def behave(sim, run, job_dir, n, calls):
        if job_dir == "job_rl":
            if n == 1:
                return "RUNNING" if sim.poll < 1 + trainer_lag else "SUCCEEDED"
            return "SUCCEEDED"
        return renderer_service(sim, run, n)

    return behave


def race(tmp_path, flow, gate, trainer_lag=1, rl_extra=None, entropies=None, entropy_at_end=0.1, **extra):
    sim = Sim(tmp_path, race_behave(trainer_lag))
    d = setup_run(sim, tmp_path, ENDPOINT - 1, entropies)
    sim.at(1, lambda: finish_trainer(sim, d, entropy=entropy_at_end))
    chain = make_chain(tmp_path, sim, gate, {**FLOWS[flow], **(rl_extra or {})}, **extra)
    ended = run_chain(chain, sim)
    return sim, chain, d, ended


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_i_gate_on_race_waits_for_the_trainer_and_finishes(tmp_path, flow):
    assert has_switch(), "train_chain is not the patched copy"
    sim, chain, d, ended = race(tmp_path, flow, gate=True)
    assert not (chain.state.get("job_retries") or {}).get("rl/a")
    assert not chain.state.get("startup_retries", {}).get("a")
    assert [c for c in sim.calls if c[1] in ("prepare", "submit")] == []
    assert sim.live_deletes == []
    assert ended == "SUCCEEDED"
    # the poll of the race (1) did nothing; poll 2 saw the trainer SUCCEEDED; the step's clean-up deleted both receipts
    assert all(c[0] >= 2 for c in sim.calls), sim.calls
    assert sorted(c[2] for c in sim.calls if c[1] == "delete") == ["job_renderer", "job_rl"]
    assert (d / "STOP_RENDERER").exists()
    assert "a" not in (chain.state.get("guard_stops") or {})


@pytest.mark.parametrize("flow", sorted(FLOWS))
@pytest.mark.parametrize("gate", [None, False])
def test_ii_gate_off_keeps_the_old_renderer_retry(tmp_path, flow, gate):
    sim, chain, d, ended = race(tmp_path, flow, gate=gate)
    retries = chain.state["job_retries"]["rl/a"]
    assert [(r["what"], r["reason"]) for r in retries] == [("renderer", "render service ended SUCCEEDED")]
    # the finished trainer's receipt was deleted while sco still showed it RUNNING (pt-0cowhw80 / pt-i390zf0d)
    assert sim.live_deletes == [(1, "job_rl", "pt-job_rl-1", "RUNNING")]
    assert [c[3] for c in sim.submits("job_renderer")] == ["pt-job_renderer-2"]
    assert [c[3] for c in sim.submits("job_rl")] == ["pt-job_rl-2"]
    assert ended == "SUCCEEDED"


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_ii_absent_and_false_are_the_same(tmp_path, flow):
    a = race(tmp_path / "absent", flow, gate=None)[0].calls
    b = race(tmp_path / "false", flow, gate=False)[0].calls
    assert a == b


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_grace_for_a_trainer_sco_keeps_showing(tmp_path, flow):
    """Gate on, sco never shows the trainer's end: the run ends "endpoint_race_grace_min" (here 15) minutes after
    COMPLETE, without a retry; its job is deleted then by the step's clean-up, not before."""
    assert has_switch()
    sim, chain, d, ended = race(tmp_path, flow, gate=True, trainer_lag=10 ** 6)
    assert not (chain.state.get("job_retries") or {}).get("rl/a")
    assert [c for c in sim.calls if c[1] in ("prepare", "submit")] == []
    assert ended == f"endpoint {ENDPOINT} COMPLETE, trainer still RUNNING after 15 min"
    marker = (d / "checkpoints" / f"global_step_{ENDPOINT}" / "COMPLETE").stat().st_mtime
    (first_delete,) = [c for c in sim.calls if c[1] == "delete" and c[2] == "job_rl"]
    assert [x[:3] for x in sim.live_deletes] == [(first_delete[0], "job_rl", "pt-job_rl-1")]
    assert first_delete[0] >= 1 + 15  # poll 1 (marker) + 15 polls of 60 s
    assert marker <= sim.t


def test_grace_is_a_plan_value(tmp_path):
    assert has_switch()
    sim, chain, d, ended = race(tmp_path, "default", gate=True, trainer_lag=10 ** 6, endpoint_race_grace_min=3)
    assert ended == f"endpoint {ENDPOINT} COMPLETE, trainer still RUNNING after 3 min"
    (first_delete,) = [c for c in sim.calls if c[1] == "delete" and c[2] == "job_rl"]
    assert 1 + 3 <= first_delete[0] <= 1 + 4


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_guard_at_the_endpoint(tmp_path, flow):
    """A guard whose condition is met by the endpoint's own log line (here entropy > 0.3 in 5 updates in a row):
    gate off, the run that reached its endpoint is recorded as guard-stopped (state["guard_stops"], which later steps
    read as stopped); gate on, it ends as finished."""
    guards = {"entropy_max": 0.3, "entropy_window": 5, "kl_max": 1.0, "invalid_max": 1.0}
    high = {s: 0.35 for s in range(ENDPOINT - 4, ENDPOINT)}  # 4 high before the endpoint, the 5th at it
    before = "".join(log_line(s, high.get(s, 0.1)) for s in range(1, ENDPOINT))
    assert T.guard_reason(before, guards) is None
    assert T.guard_reason(before + log_line(ENDPOINT, 0.35), guards)
    kw = dict(rl_extra={"guards": guards}, entropies=high, entropy_at_end=0.35)
    sim, chain, d, ended = race(tmp_path / "off", flow, gate=None, **kw)
    assert "a" in chain.state["guard_stops"] and ended.startswith("stopped: entropy above 0.3")
    if has_switch():
        sim, chain, d, ended = race(tmp_path / "on", flow, gate=True, **kw)
        assert "a" not in (chain.state.get("guard_stops") or {})
        assert ended == "SUCCEEDED" and sim.live_deletes == []
        assert not (chain.state.get("job_retries") or {}).get("rl/a")


# ------------------------------------------------------------------ failures before the endpoint (iii, iv)
def second_trainer_finishes(sim, run, n, calls):
    """A resubmitted trainer: RUNNING at its first poll, then the endpoint (COMPLETE, STOP_RENDERER) and SUCCEEDED."""
    if calls >= 2:
        if not (run / "checkpoints" / f"global_step_{ENDPOINT}" / "COMPLETE").exists():
            finish_trainer(sim, run)
        return "SUCCEEDED"
    return "RUNNING"


def renderer_dies_behave(sim, run, job_dir, n, calls):
    if job_dir == "job_rl":
        return "RUNNING" if n == 1 else second_trainer_finishes(sim, run, n, calls)
    return renderer_service(sim, run, n, fail_first_at=1)


def trainer_fails_behave(sim, run, job_dir, n, calls):
    if job_dir == "job_rl":
        if n == 1:
            return "RUNNING" if sim.poll < 1 else "FAILED"
        return second_trainer_finishes(sim, run, n, calls)
    return renderer_service(sim, run, n)


def scenario(root, behave, flow, gate, hook=None):
    root.mkdir(parents=True, exist_ok=True)
    sim = Sim(root, behave)
    d = setup_run(sim, root, 100)
    if hook:
        sim.at(1, lambda: hook(sim, d))
    chain = make_chain(root, sim, gate, FLOWS[flow])
    ended = run_chain(chain, sim)
    return sim, chain, ended


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_iii_gate_on_dead_renderer_before_the_endpoint_is_retried_as_before(tmp_path, flow):
    assert has_switch()
    on = scenario(tmp_path / "on", renderer_dies_behave, flow, True)
    off = scenario(tmp_path / "off", renderer_dies_behave, flow, None)
    assert on[0].calls == off[0].calls
    assert [r["what"] for r in on[1].state["job_retries"]["rl/a"]] == ["renderer"]
    assert on[1].state["job_retries"]["rl/a"][0]["reason"] == "render service ended FAILED"
    assert len(on[0].submits("job_renderer")) == 1 and len(on[0].submits("job_rl")) == 1
    assert on[2] == off[2] == "SUCCEEDED"


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_iv_gate_on_failed_trainer_before_the_endpoint_is_retried_as_before(tmp_path, flow):
    assert has_switch()
    on = scenario(tmp_path / "on", trainer_fails_behave, flow, True)
    off = scenario(tmp_path / "off", trainer_fails_behave, flow, None)
    assert on[0].calls == off[0].calls
    assert [(r["what"], r["reason"]) for r in on[1].state["job_retries"]["rl/a"]] == [("trainer", "trainer ended FAILED")]
    assert len(on[0].submits("job_rl")) == 1 and not on[0].submits("job_renderer")
    assert on[0].live_deletes == []
    assert on[2] == off[2] == "SUCCEEDED"


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_endpoint_folder_without_complete_is_left_to_the_old_behaviour(tmp_path, flow):
    """Scope of the switch: an endpoint folder without COMPLETE (a save under way) whose render service dies is
    handled as before (the renderer retry), with the switch on or off."""
    assert has_switch()

    def behave(sim, run, job_dir, n, calls):
        if job_dir == "job_rl":
            return "RUNNING" if n == 1 else second_trainer_finishes(sim, run, n, calls)
        return renderer_service(sim, run, n, fail_first_at=1)

    hook = lambda sim, d: finish_trainer(sim, d, complete=False)  # noqa: E731
    on = scenario(tmp_path / "on", behave, flow, True, hook)
    off = scenario(tmp_path / "off", behave, flow, None, hook)
    assert on[0].calls == off[0].calls
    assert [r["what"] for r in on[1].state["job_retries"]["rl/a"]] == ["renderer"]
