"""Tests of the EXPLORATORY RL starts (PREREG_TVR_PILOT.md section 17, row 2026-11-12; DECISIONS G18): pilot_tvr's
explore mode (XJOBS, guards, block budget, expect, steps, xsmoke-verdict), the payloads tvr_xsmoke.sh and
tvr_train.sh / tvr_eval.sh with VPD_TVR_ROOT=explore/rl (dry runs and non-dry with fakes), vpd/tvr/raw_bf16.py (a tiny
model) and vpd/tvr/compare_rl.py (synthetic roots only; never the real explore or pilot root).

Hermetic like test_tvr_jobs.py (its fixtures and fakes are reused): no GPU, no network, no rendering, no 8B model, no
sco call.  Login node, 1 core:
    cd TREE && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 ... nice -n 19 env VPD_WORKTREE=TREE PYTHONPATH=TREE \
        $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=$SCRATCH/pt vpd/tests/test_tvr_explore.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vpd import paths  # noqa: E402
from vpd.tests import test_tvr_jobs as TJ  # noqa: E402
from vpd.tests.test_tvr_jobs import lay  # noqa: E402,F401  (the pytest fixture)
from vpd.tvr import compare_rl as CR  # noqa: E402
from vpd.tvr import compare_tvr as C  # noqa: E402
from vpd.tvr import pilot_cfg  # noqa: E402
from vpd.tvr import pilot_tvr as PT  # noqa: E402

JOBS_DIR = paths.VPD_PKG / "jobs"
wj, fsha, sha = TJ.wj, TJ.fsha, TJ.sha
XS = PT.EXPLORE_STARTS
KNOBS = TJ._KNOBS + ("VPD_TVR_ROOT",)


# ------------------------------------------------------------------------------------------ fixtures / builders
@pytest.fixture
def xlay(lay, monkeypatch):  # noqa: F811
    """test_tvr_jobs' layout (fake sco, fake prereg state) plus an explore-start check that needs no checkpoint dir
    (x_start_problems is tested on its own below)."""
    monkeypatch.setattr(PT, "x_start_problems", lambda s: [] if s in XS else [f"start {s!r} not explore"])
    return lay


def build_tiers(L: PT.Layout, cfg: dict) -> dict:
    """The tier files and a real tiers/v1/manifest.json (files: name -> sha256); cfg data.tiers_manifest_sha256 =
    its sha (build_data writes a placeholder manifest)."""
    files = {}
    for name in ("warmup0w20.jsonl", "demos_100.jsonl", "demos_400.jsonl", "demos_ALL.jsonl", "parity3.jsonl"):
        p = L.tiers / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{name} demos\n")
        files[name] = fsha(p)
    man = wj(L.tiers / "manifest.json", {"schema": "vpd.tvr.tiers/1", "files": files, **files})
    cfg["data"]["tiers_manifest_sha256"] = fsha(man)
    return files


def ready(L: PT.Layout, cfg: dict) -> dict:
    TJ.build_data(L, cfg)
    files = build_tiers(L, cfg)
    TJ.build_cal(L)
    return files


def xverdict(L: PT.Layout, v="PASS", name="tvr-xsmoke-202611121000") -> Path:
    return wj(L.xsmoke_dir(name) / "smoke_verdict.json", {"verdict": v, "failed": [] if v == "PASS" else ["x"]})


def build_xadapters(L: PT.Layout, s: str, cfg: dict, files: dict, root: Path = None) -> None:
    """The explore adapters of start s as the trainer writes their meta (seed 1001, the manifest's data sha, the init
    chain); under root/<S>/adapters (default L.xroot)."""
    root = Path(root or L.xroot)
    for tier, init in PT.train_tiers(s, cfg):
        d = root / s / "adapters" / tier
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter.pt").write_bytes(f"{s}-{tier}".encode())
        meta = {"start": s, "seed": 1001, "phase": PT.PHASE[tier], "optimizer_steps": 30,
                "optimizer_steps_planned": 30, "code_sha256": TJ.code_sha(), "attn": "sdpa",
                "data_sha256": files[PT.TIER_FILE[tier]], "init_adapter": None}
        if init:
            meta["init_adapter"] = {"sha256": fsha(root / s / "adapters" / init / "adapter.pt"), "start": s,
                                    "dir": str(root / s / "adapters" / init)}
        wj(d / "meta.json", meta)


def _edit(p: Path, fn) -> None:
    r = json.loads(Path(p).read_text())
    fn(r)
    Path(p).write_text(json.dumps(r))


# ------------------------------------------------------------------------------------------ job table, payloads
def test_explore_job_table_names_and_sco_rules():
    from vpd import ledger, sco

    assert PT.EXPLORE_STARTS == ("Q3RD", "Q3RS150") and not set(XS) & set(pilot_cfg.STARTS)
    assert len(PT.JOBS) == 13 and not set(PT.XJOBS) & set(PT.JOBS)  # the pilot table is unchanged
    assert sorted(PT.XJOBS) == sorted(["tvr-xsmoke", "tvr-xtrain-q3rd", "tvr-xtrain-q3rs150", "tvr-xeval-q3rd-top",
                                       "tvr-xeval-q3rd-low", "tvr-xeval-q3rs150-top", "tvr-xeval-q3rs150-low"])
    for job, spec in PT.XJOBS.items():
        assert spec["name"] == "vpd-" + job and spec["as_root"] is True
        sco.validate_name(spec["name"])
        assert ledger.gate_of(spec["name"]) == "tvr" and (JOBS_DIR / spec["script"]).is_file()
        assert PT.job_spec(job) is spec and pilot_cfg.missing(spec["base"], TJ.filled_cfg()) == []
    assert PT.XJOBS["tvr-xeval-q3rs150-low"]["start"] == "Q3RS150" and PT.XJOBS["tvr-xtrain-q3rd"]["kind"] == "xtrain"
    with pytest.raises(PT.Refused, match="explore jobs"):
        PT.job_spec("tvr-xtrain-q3b")


def test_explore_payloads_and_timeouts(tmp_path):
    run = tmp_path / "jobs" / "tvr-xeval-q3rd-top-202611121200"
    p = PT.payload_for("tvr-xeval-q3rd-top", run, 3.027)
    t = PT.job_timeouts("tvr-xeval-q3rd-top", 3.027)
    assert p == ["env", f"VPD_JOB_DEADLINE_S={t['soft_deadline_s']}", "VPD_TVR_ROOT=explore/rl", "timeout", "-k",
                 "120", str(t["outer_s"]), "bash", str(JOBS_DIR / "tvr_eval.sh"), "Q3RD", "top", str(run)]
    assert (t["outer_s"], t["soft_deadline_s"]) == (math.floor(3.027 * 3600) - 300, math.floor(3.027 * 3600) - 480)
    tr = PT.payload_for("tvr-xtrain-q3rs150", run, 2.759)
    assert tr[2] == "VPD_TVR_ROOT=explore/rl" and tr[-3:] == [str(JOBS_DIR / "tvr_train.sh"), "Q3RS150", str(run)]
    sm = PT.payload_for("tvr-xsmoke", run, PT.est_high("tvr-xsmoke")["est_high"])
    assert sm == ["env", "VPD_JOB_DEADLINE_S=3840", "timeout", "-k", "60", "4020", "bash",
                  str(JOBS_DIR / "tvr_xsmoke.sh"), str(run)]
    # the pilot payloads carry no VPD_TVR_ROOT
    assert not any("VPD_TVR_ROOT" in a for j in PT.JOBS for a in PT.payload_for(j, run, 3.0))
    from vpd import sco

    code = tmp_path / "code"
    (code / "vpd" / "jobs").mkdir(parents=True)
    (code / "vpd" / "jobs" / "tvr_eval.sh").write_text("#\n")
    new, rewrites = sco.snapshot_payload(p, code)
    assert new[8] == str(code / "vpd" / "jobs" / "tvr_eval.sh") and len(rewrites) == 1


def test_explore_est_high_is_the_start_independent_rule(xlay):
    cfg = TJ.filled_cfg()
    TJ.build_cal(xlay)
    for s in XS:
        assert PT.est_high(f"tvr-xtrain-{s.lower()}", xlay, cfg) == PT.derived_train("Q3S", xlay, cfg)
        for part in ("top", "low"):
            assert PT.est_high(f"tvr-xeval-{s.lower()}-{part}", xlay, cfg) == PT.derived_eval("Q3S", part, xlay, cfg)
    assert [t for t, _ in PT.train_tiers("Q3RD", cfg)] == ["0W", "100", "ALL"]
    assert PT.eval_plan("Q3RS150", "low", cfg) == [("0W", "c"), ("100", "c")]
    assert PT.est_high("tvr-xsmoke")["est_high"] == 1.2


# ------------------------------------------------------------------------------------------ guards
def test_prepare_xsmoke_records_both_budgets_and_parents(xlay, capsys):
    cfg = TJ.filled_cfg()
    ready(xlay, cfg)
    launch = PT.prepare("tvr-xsmoke", L=xlay, cfg=cfg, events=[])
    c = xlay.calls["prepare"][-1]
    assert c["name"] == "vpd-tvr-xsmoke" and c["as_root"] is True and c["gate"] == "tvr" and c["est_gph"] == 1.2
    info = json.loads((Path(c["run"]) / "prepared.json").read_text())
    assert info["exploratory"] is True and info["block_budget"]["ok"] and info["round_budget"]["ok"]
    assert info["block_budget"]["baseline"] == 48.5136 and info["block_budget"]["cap"] == 25.0
    assert info["facts"]["harness_code_sha256"].startswith(PT.HARNESS_SHA256_PREFIX)
    assert info["facts"]["prereg_row_sha256"] and launch == Path(c["run"]) / "job" / "launch.sh"
    for d in xlay.x_shared_parents():
        assert d.is_dir(), d
    assert not xlay.pilot_root.exists()  # an explore prepare never creates runs/pilot
    err = capsys.readouterr().err
    assert "block budget: used -48.5136 + open 0.0000 + new 1.2000" in err and "round budget" in err


def _ledger(spent: float, open_est: float):
    ev = [{"event": "submit", "name": "vpd-tvr-msmoke", "run": "/r/a", "gate": "tvr", "est": 1.0, "gpus": 1,
           "pt_id": "pt-a", "submitted_at": "2026-11-12T00:00:00Z"},
          {"event": "end", "run": "/r/a", "pt_id": "pt-a", "gph": spent, "gph_source": "sco_start_to_complete"}]
    if open_est:
        ev.append({"event": "submit", "name": "vpd-tvr-xtrain-q3rd", "run": "/r/b", "gate": "tvr", "est": open_est,
                   "gpus": 1, "pt_id": "pt-b", "submitted_at": "2026-11-12T01:00:00Z"})
    return ev


def test_block_budget_refusal_prints_the_three_numbers(xlay, capsys):
    cfg = TJ.filled_cfg()
    ready(xlay, cfg)
    bb = PT.block_budget(1.2, _ledger(60.0, 5.0))
    assert bb["block_used"] == round(60.0 - 48.5136, 4) and bb["open_est"] == 5.0 and bb["ok"] is True
    with pytest.raises(PT.Refused, match=r"BLOCK BUDGET: block_used 18\.4864 \+ open 5\.5180 \+ new 1\.2000 = "
                                         r"25\.2044 > 25\.0"):
        PT.prepare("tvr-xsmoke", L=xlay, cfg=cfg, events=_ledger(48.5136 + 18.4864, 5.518))
    assert not xlay.calls["prepare"]
    assert "block budget: used 18.4864 + open 5.5180 + new 1.2000 = 25.2044 GPU-h vs cap 25.0" in \
        capsys.readouterr().err
    PT.prepare("tvr-xsmoke", L=xlay, cfg=cfg, events=_ledger(48.5136 + 18.0, 5.518))  # 24.718 <= 25
    # the round-200 rule still applies (checked first: 210 - 8.8404 + 1.2 > 200)
    with pytest.raises(PT.Refused, match="ROUND BUDGET"):
        PT.prepare("tvr-xsmoke", L=xlay, cfg=cfg, new=True, events=_ledger(210.0, 0.0))
    assert len(xlay.calls["prepare"]) == 1


def test_block_budget_rechecked_at_submit(xlay):
    cfg = TJ.filled_cfg()
    ready(xlay, cfg)
    PT.prepare("tvr-xsmoke", L=xlay, cfg=cfg, events=[])
    with pytest.raises(PT.Refused, match="BLOCK BUDGET"):
        PT.submit("tvr-xsmoke", L=xlay, cfg=cfg, events=_ledger(48.5136 + 24.0, 0.0))
    assert PT.submit("tvr-xsmoke", L=xlay, cfg=cfg, events=[]) == "pt-fake1"


def test_xtrain_needs_the_explore_smoke_pass(xlay):
    cfg = TJ.filled_cfg()
    ready(xlay, cfg)
    TJ.verdict(xlay, "PASS")  # the pilot msmoke verdict does not count
    with pytest.raises(PT.Refused, match="no smoke_verdict.json under .*explore/rl/smoke"):
        PT.prepare("tvr-xtrain-q3rd", L=xlay, cfg=cfg, events=[])
    xverdict(xlay, "FAIL")
    with pytest.raises(PT.Refused, match="not PASS"):
        PT.prepare("tvr-xtrain-q3rd", L=xlay, cfg=cfg, events=[])
    xverdict(xlay, "PASS", name="tvr-xsmoke-202611121100")
    PT.prepare("tvr-xtrain-q3rd", L=xlay, cfg=cfg, events=[])
    c = xlay.calls["prepare"][-1]
    assert c["est_gph"] == PT.derived_train("Q3RD", xlay, cfg)["est_high"]
    assert c["payload"][2] == "VPD_TVR_ROOT=explore/rl" and c["payload"][-2:] == ["Q3RD", c["run"]]
    info = json.loads((Path(c["run"]) / "prepared.json").read_text())
    assert info["smoke_verdict"]["verdict"] == "PASS" and "tvr-xsmoke-202611121100" in info["smoke_verdict"]["path"]


def test_explore_guards_fingerprints_row_tiers_prereg(xlay, monkeypatch, tmp_path):
    cfg = TJ.filled_cfg()
    files = ready(xlay, cfg)
    xverdict(xlay)
    assert PT.prerequisites("tvr-xtrain-q3rs150", xlay, cfg) == []
    # the harness fingerprint
    real = PT.harness_sha256
    assert real().startswith("64edd575f36b")
    monkeypatch.setattr(PT, "harness_sha256", lambda: "0" * 64)
    assert any("harness code_sha256 000000000000" in b for b in PT.prerequisites("tvr-xtrain-q3rs150", xlay, cfg))
    monkeypatch.setattr(PT, "harness_sha256", real)
    # the trainer fingerprint
    cfg2 = copy.deepcopy(cfg)
    cfg2["recipe"]["trainer_code_sha256_prefix"] = "deadbeef0000"
    assert any("trainer code_sha256" in b for b in PT.prerequisites("tvr-xtrain-q3rd", xlay, cfg2))
    # the 2026-11-12 row: absent, or twice
    text = PT.prereg_path().read_text(encoding="utf-8")
    row = PT.xrow(text)
    assert row and row.startswith("| 2026-11-12 |") and "Q3RD" in row and "Q3RS150" in row
    p = tmp_path / "PREREG.md"
    p.write_text(text.replace(row + "\n", ""), encoding="utf-8")
    monkeypatch.setattr(PT, "prereg_path", lambda: p)
    assert any("no single section-17 row" in b for b in PT.prerequisites("tvr-xtrain-q3rd", xlay, cfg))
    p.write_text(text + "\n" + row + "\n", encoding="utf-8")
    assert PT.xrow(p.read_text(encoding="utf-8")) is None
    monkeypatch.setattr(PT, "prereg_path", lambda: paths.WORKTREE / PT.PREREG)
    # a tier file changed since the manifest; the manifest itself changed (data_problems)
    (xlay.tiers / "demos_100.jsonl").write_text("other demos\n")
    assert any("demos_100.jsonl: sha256" in b for b in PT.prerequisites("tvr-xtrain-q3rd", xlay, cfg))
    (xlay.tiers / "demos_100.jsonl").write_text("demos_100.jsonl demos\n")
    assert fsha(xlay.tiers / "demos_100.jsonl") == files["demos_100.jsonl"]
    _edit(xlay.tiers / "manifest.json", lambda m: m.update(extra=1))
    assert any("data.tiers_manifest_sha256" in b for b in PT.prerequisites("tvr-xsmoke", xlay, cfg))
    # the prereg must be committed and clean (as for the pilot jobs)
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": None, "committed": False, "status": "", "clean": True})
    assert any("is not committed" in b for b in PT.prerequisites("tvr-xsmoke", xlay, cfg))


def test_x_start_problems(monkeypatch, tmp_path):
    from vpd import starts

    assert PT.x_start_problems("Q3B") and PT.x_start_problems("QRL-s1")
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "config.json").write_text("{}")
    table = {"Q3RD": {"name": "Q3RD", "code": "c0q3_s1d_300", "path": str(ck), "family": "qwen3", "note": ""}}
    monkeypatch.setattr(starts, "resolve", lambda n: table[n])
    monkeypatch.setattr(starts, "processor_dir", lambda n: ck)
    assert PT.x_start_problems("Q3RD") == []
    monkeypatch.setattr(starts, "processor_dir", lambda n: tmp_path / "view")  # an eval view was built since
    assert any("processor would be read from" in b for b in PT.x_start_problems("Q3RD"))
    table["Q3RD"]["code"] = None
    assert any("no registry code" in b for b in PT.x_start_problems("Q3RD"))


def test_x_start_problems_on_the_real_starts():
    """The real starts.json: both explore starts have a code and no eval view, so the processor is read from the
    checkpoint dir (skipped where the checkpoints are not mounted)."""
    from vpd import starts

    for s in XS:
        if not os.path.lexists(Path(starts.resolve(s)["path"]) / "config.json"):
            pytest.skip("the RL checkpoints are not mounted here")
        assert PT.x_start_problems(s) == [], s


def test_xeval_needs_complete_adapters_with_seed_data_and_init_chain(xlay):
    cfg = TJ.filled_cfg()
    files = ready(xlay, cfg)
    xverdict(xlay)
    with pytest.raises(PT.Refused, match="adapter Q3RD/0W"):
        PT.prepare("tvr-xeval-q3rd-top", L=xlay, cfg=cfg, events=[])
    build_xadapters(xlay, "Q3RD", cfg, files)
    PT.prepare("tvr-xeval-q3rd-top", L=xlay, cfg=cfg, events=[])
    c = xlay.calls["prepare"][-1]
    assert c["payload"][-3:] == ["Q3RD", "top", c["run"]] and c["est_gph"] <= 5.5
    ad = xlay.xadapter("Q3RD", "100")
    for fn, want in ((lambda m: m.update(seed=1002), "meta seed 1002"),
                     (lambda m: m.update(data_sha256="d" * 64), "meta data_sha256"),
                     (lambda m: m.update(phase="tier400"), "meta phase"),
                     (lambda m: m["init_adapter"].update(start="Q3S"), "init_adapter start"),
                     (lambda m: m["init_adapter"].update(sha256="0" * 64), "retrain")):
        orig = (ad / "meta.json").read_text()
        _edit(ad / "meta.json", fn)
        probs = PT.prerequisites("tvr-xeval-q3rd-low", xlay, cfg)
        assert any(want in b for b in probs), (want, probs)
        (ad / "meta.json").write_text(orig)
    _edit(xlay.xadapter("Q3RD", "0W") / "meta.json", lambda m: m.update(init_adapter={"sha256": "x"}))
    assert any("starts from the fresh LoRA" in b for b in PT.prerequisites("tvr-xeval-q3rd-low", xlay, cfg))
    assert PT.prerequisites("tvr-xeval-q3rs150-low", xlay, cfg) and not xlay.pilot_root.exists()


# ------------------------------------------------------------------------------------------ steps / expect
def test_explore_required_steps():
    cfg = TJ.filled_cfg()
    sm = PT.required_steps("tvr-xsmoke", cfg)
    assert sm[:3] == ["guard", "prereg_committed", "merge_disk"] and sm[-1] == "verdict"
    assert sm[3:11] == [f"{x}_Q3RD" for x in ("load", "dry", "hfparity", "session", "train", "merge",
                                              "merge_parity", "eval")]
    assert PT.required_steps("tvr-xtrain-q3rs150", cfg) == ["guard", "prereg_committed", "train_0W", "train_100",
                                                            "train_ALL"]
    for s in XS:
        for part in ("top", "low"):
            assert PT.required_steps(f"tvr-xeval-{s.lower()}-{part}", cfg) == PT.required_steps(
                f"tvr-eval-q3s-{part}", cfg)


def test_steps_check_report_only_merge_parity_is_explore_only(tmp_path):
    cfg = TJ.filled_cfg()
    run = tmp_path / "run"
    run.mkdir()
    lines = [{"name": n, "rc": 0} for n in PT.required_steps("tvr-xeval-q3rd-top", cfg)]
    i = [x["name"] for x in lines].index("merge_parity_ALL")
    lines[i] = {"name": "merge_parity_ALL", "rc": 1}
    lines.insert(i + 1, {"name": "merge_parity_ALL:report_only", "rc": 0})
    (run / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert PT.steps_check("tvr-xeval-q3rd-top", run, cfg)["ok"]
    st = PT.steps_check("tvr-eval-q3s-top", run, cfg)  # the pilot rule is unchanged
    assert not st["ok"] and st["failed"] == ["merge_parity_ALL"]
    lines[i + 1] = {"name": "merge_parity_ALL:report_only", "rc": 1}
    (run / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert PT.steps_check("tvr-xeval-q3rd-top", run, cfg)["failed"] == ["merge_parity_ALL",
                                                                        "merge_parity_ALL:report_only"]
    lines[i] = {"name": "eval_ALL", "rc": 4}  # only merge parity is ever excused
    lines[i + 1] = {"name": "eval_ALL:report_only", "rc": 0}
    (run / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert "eval_ALL" in PT.steps_check("tvr-xeval-q3rd-top", run, cfg)["failed"]


def _xevals(L: PT.Layout, s: str, cfg: dict, part: str, parity_pass=False) -> None:
    wj(L.xcanary(s, part), {"pass": True, "n_frames": 27, "n_pass": 27})
    for tier, arm in PT.eval_plan(s, part, cfg):
        ad = L.xadapter(s, PT.EVAL_ADAPTER[tier])
        wj(L.xparity(s, tier), {"pass": parity_pass, "adapter_sha256": fsha(ad / "adapter.pt"), "mean_abs_dce": 0.009})
        d = L.xeval_dir(s, tier, arm)
        wj(d / "results.json", {"valid": True, "n_tasks": 500, "tasks_missing": [],
                                "adapter": {"adapter_sha256": fsha(ad / "adapter.pt")}})
        wj(d / "official_check.json", {"equal": True})


def test_expect_xtrain_and_xeval_report_only_parity(xlay):
    cfg = TJ.filled_cfg()
    files = ready(xlay, cfg)
    s = "Q3RS150"
    build_xadapters(xlay, s, cfg, files)
    run = xlay.jobs / "tvr-xtrain-q3rs150-202611121200"
    run.mkdir(parents=True)
    (run / "steps.jsonl").write_text("".join(json.dumps({"name": n, "rc": 0}) + "\n"
                                             for n in PT.required_steps("tvr-xtrain-q3rs150", cfg)))
    assert PT.expect("tvr-xtrain-q3rs150", run, xlay, cfg)["ok"]
    _edit(xlay.xadapter(s, "ALL") / "meta.json", lambda m: m.update(seed=7))
    assert PT.expect("tvr-xtrain-q3rs150", run, xlay, cfg)["failed"] == ["adapter_ALL"]
    _edit(xlay.xadapter(s, "ALL") / "meta.json", lambda m: m.update(seed=1001))
    for part in ("top", "low"):
        _xevals(xlay, s, cfg, part)  # merge parity FAILED (bf16): reported, not judged
        job = f"tvr-xeval-q3rs150-{part}"
        run = xlay.jobs / f"{job}-202611121300"
        run.mkdir(parents=True)
        (run / "steps.jsonl").write_text("".join(json.dumps({"name": n, "rc": 0}) + "\n"
                                                 for n in PT.required_steps(job, cfg)))
        rep = PT.expect(job, run, xlay, cfg)
        assert rep["ok"], rep["failed"]
        mp = [c for c in rep["items"] if c["name"].startswith("merge_parity_")]
        assert mp and all(c["detail"]["pass_reported_not_judged"] is False for c in mp)
        assert all(str(xlay.xroot) in c["path"] for c in rep["items"] if c.get("path"))
        paths_ = [Path(p) for p in (c["path"] for c in rep["items"] if c.get("path"))]
        assert all(xlay.pilot_root not in p.parents for p in paths_)
        assert any(p.parent.name.startswith("xrl__Q3RS150__") for p in paths_)
    # a parity json of another adapter (or none): a failure; an eval that served another adapter: a failure
    _edit(xlay.xparity(s, "ALL"), lambda r: r.update(adapter_sha256="0" * 64))
    assert "merge_parity_ALL" in PT.expect("tvr-xeval-q3rs150-top", run.parent / "tvr-xeval-q3rs150-top-202611121300",
                                           xlay, cfg)["failed"]
    _edit(xlay.xeval_dir(s, "0W", "c") / "results.json", lambda r: r["adapter"].update(adapter_sha256="1" * 64))
    assert "eval_0W_c" in PT.expect("tvr-xeval-q3rs150-low", run, xlay, cfg)["failed"]
    exp = PT.expected_outputs("tvr-xeval-q3rs150-low", run, xlay, cfg)
    assert xlay.xeval_dir(s, "100", "c") / "results.json" in exp and run / "DONE" in exp


def test_guard_in_job_writes_facts(xlay):
    cfg = TJ.filled_cfg()
    ready(xlay, cfg)
    run = xlay.jobs / "tvr-xsmoke-202611121400"
    run.mkdir(parents=True)
    rep = PT.guard("tvr-xsmoke", run, xlay, cfg)
    assert rep["ok"], rep["problems"]
    g = json.loads((run / "guard.json").read_text())
    assert g["facts"]["explore"] is True and g["facts"]["trainer_code_sha256"].startswith("0865dd75f1ee")
    assert g["facts"]["harness_code_sha256"].startswith("64edd575f36b")


def test_cli_explore_flags(xlay, capsys):
    TJ.build_cal(xlay)
    assert PT.main(["plan", "--explore"]) == 0
    out = capsys.readouterr().out
    assert "EXPLORATORY RL plan" in out and "[tvr-xsmoke]" in out and "[tvr-train-q3b]" not in out
    assert "block budget:" in out and "VPD_TVR_ROOT=explore/rl" in out
    assert PT.main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "[tvr-xsmoke]" not in out and "[tvr-train-q3b]" in out and "block budget" not in out
    assert PT.main(["served", "Q3RS150"]) == 0 and capsys.readouterr().out.strip() == "c0q3_s1s0_150"
    b = PT.budget(xlay, TJ.filled_cfg(), events=[], explore=True)
    assert set(b["est_high"]) == set(PT.XJOBS) and b["block_budget"]["ok"]
    assert "block_budget" not in PT.budget(xlay, TJ.filled_cfg(), events=[])


# ------------------------------------------------------------------------------------------ xsmoke verdict
def build_xsmoke(L: PT.Layout, sm: Path, cfg: dict, files: dict) -> None:
    rend = TJ.renderer_block(cfg)
    for s in XS:
        sd = sm / s
        wj(sd / "load" / "load.json", {"ok": True, "problems": [], "dtype": "bfloat16", "tensor_checks_ok": True})
        wj(sd / "dry" / "dry_run.json", {"per_conversation": [
            {"id": f"d{i}", "grid_set": [[1, 22, 40]], "images": 2 * (i + 2), "image_tokens": 440 * (i + 2)}
            for i in range(3)]})
        wj(sd / "dry" / "parity.json", {"verdict": "PASS", "problems": {}})
        ev = sd / "evals"
        wj(ev / f"smoke__{s}__raw__vllm_parity.json", {"equal": True, "n": 12})
        for a in ("c", "u", "blind"):
            r = {"valid": True, "n_tasks": 8, "tasks_missing": [], "code_sha256": PT.HARNESS_SHA256_PREFIX + "0" * 52,
                 "invalid": {"regex_mismatch": 0 if a != "u" else None, "replies": 90, "unparseable_steps": 3},
                 "perf": {"steps_per_s": 5.0}}
            if a != "blind":
                r["audit"] = {"prompt_parity": {"equal": True},
                              "image_tokens": {"tokens_per_image": [222], "grid_set": [[1, 22, 40]], "n": 30}}
                r["renderer"] = rend
            wj(ev / f"smoke__{s}__raw__{a}" / "results.json", r)
            wj(ev / f"smoke__{s}__raw__{a}" / "tasks" / "t0" / "trajectory.json",
               {"task_id": "t0", "result": {"success": None}})
        ad = sd / "adapters" / "0W"
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "adapter.pt").write_bytes(f"smoke {s}".encode())
        asha = fsha(ad / "adapter.pt")
        wj(ad / "meta.json", {"start": s, "seed": 1001, "phase": "0W", "optimizer_steps": 2,
                              "optimizer_steps_planned": 2, "conversations": 16, "epochs": 1,
                              "data_sha256": files["warmup0w20.jsonl"], "attn": "sdpa", "init_adapter": None,
                              "code_sha256": TJ.code_sha(), "loss_first": 0.9, "loss_last": 0.8,
                              "lora": {"wrapped": 252}})
        wj(sd / "parity" / "merge_0W.json", {"pass": False, "adapter_sha256": asha, "n_demos": 20,
                                             "mean_abs_dce": 0.0091, "reply_agreement": 0.96, "n_replies": 267,
                                             "merged": {"start": s, "adapter_sha256": asha, "merged": 252,
                                                        "processor_src": "/ck"}})
        wj(ev / f"xsmoke__{s}__0W__vllm_parity.json", {"equal": True, "n": 12})
        wj(ev / f"xsmoke__{s}__0W__c" / "results.json", {
            "valid": True, "n_tasks": 4, "tasks_missing": [], "code_sha256": PT.HARNESS_SHA256_PREFIX + "1" * 52,
            "invalid": {"regex_mismatch": 0, "replies": 40}, "adapter": {"adapter_sha256": asha},
            "audit": {"prompt_parity": {"equal": True}, "image_tokens": {"tokens_per_image": [222],
                                                                         "grid_set": [[1, 22, 40]]}},
            "renderer": rend, "perf": {"steps_per_s": 5.0}})


def test_xsmoke_verdict_pass_and_each_failure(xlay):
    cfg = TJ.filled_cfg()
    files = build_tiers(xlay, cfg)
    sm = xlay.xsmoke_dir("tvr-xsmoke-202611121500")
    build_xsmoke(xlay, sm, cfg, files)
    v = PT.xsmoke_verdict(sm, xlay, cfg)
    assert v["verdict"] == "PASS", v["failed"]
    assert json.loads((sm / "smoke_verdict.json").read_text())["schema"] == "vpd.tvr.xsmoke_verdict/1"
    mp = next(c for c in v["checks"] if c["name"] == "merge_parity_Q3RD")
    assert mp["ok"] and mp["detail"]["pass_reported_not_judged"] is False  # reported, never judged
    R, S = sm / "Q3RD", sm / "Q3RS150"
    breaks = [
        (R / "load" / "load.json", lambda r: r.update(ok=False), "load_Q3RD"),
        (S / "dry" / "dry_run.json", lambda r: r["per_conversation"][0].update(grid_set=[[1, 24, 40]]),
         "dry_Q3RS150"),
        (R / "dry" / "parity.json", lambda r: r.update(verdict="FAIL"), "hfparity_Q3RD"),
        (S / "evals" / "smoke__Q3RS150__raw__vllm_parity.json", lambda r: r.update(equal=False),
         "vllm_parity_Q3RS150"),
        (R / "evals" / "smoke__Q3RD__raw__u" / "results.json", lambda r: r.update(n_tasks=7), "runs_Q3RD"),
        (R / "evals" / "smoke__Q3RD__raw__blind" / "results.json", lambda r: r["invalid"].update(regex_mismatch=2),
         "regex_Q3RD"),
        (S / "evals" / "smoke__Q3RS150__raw__c" / "results.json",
         lambda r: r["audit"]["image_tokens"].update(tokens_per_image=[220]), "audit_Q3RS150"),
        (R / "evals" / "smoke__Q3RD__raw__u" / "results.json", lambda r: r["renderer"].update(lp=4), "renderer_Q3RD"),
        (R / "evals" / "smoke__Q3RD__raw__c" / "results.json", lambda r: r.update(code_sha256="ab" * 32),
         "harness_Q3RD"),
        (R / "adapters" / "0W" / "meta.json", lambda r: r.update(optimizer_steps=1), "train_Q3RD"),
        (S / "adapters" / "0W" / "meta.json", lambda r: r.update(seed=1002), "train_Q3RS150"),
        (S / "adapters" / "0W" / "meta.json", lambda r: r.update(data_sha256="e" * 64), "train_Q3RS150"),
        (R / "adapters" / "0W" / "meta.json", lambda r: r.update(loss_last=float("nan")), "train_Q3RD"),
        (R / "parity" / "merge_0W.json", lambda r: r["merged"].update(merged=251), "merge_Q3RD"),
        (S / "parity" / "merge_0W.json", lambda r: r.update(n_demos=2), "merge_parity_Q3RS150"),
        (S / "evals" / "xsmoke__Q3RS150__0W__vllm_parity.json", lambda r: r.update(equal=None),
         "merged_vllm_parity_Q3RS150"),
        (R / "evals" / "xsmoke__Q3RD__0W__c" / "results.json", lambda r: r.update(valid=False), "merged_runs_Q3RD"),
        (R / "evals" / "xsmoke__Q3RD__0W__c" / "results.json", lambda r: r["audit"]["prompt_parity"].update(
            equal=False), "merged_audit_Q3RD"),
        (S / "evals" / "xsmoke__Q3RS150__0W__c" / "results.json", lambda r: r["renderer"].update(workers_ok=False),
         "merged_renderer_Q3RS150"),
    ]
    for path, fn, name in breaks:
        orig = path.read_text()
        r = json.loads(orig)
        fn(r)
        path.write_text(json.dumps(r))
        v = PT.xsmoke_verdict(sm, xlay, cfg)
        assert v["verdict"] == "FAIL" and v["failed"] == [name], (name, v["failed"])
        path.write_text(orig)
    # the merged session served another adapter (the eval's results.adapter); a parity json of another adapter
    p = R / "evals" / "xsmoke__Q3RD__0W__c" / "results.json"
    orig = p.read_text()
    _edit(p, lambda r: r["adapter"].update(adapter_sha256="9" * 64))
    assert PT.xsmoke_verdict(sm, xlay, cfg)["failed"] == ["merged_served_Q3RD"]
    p.write_text(orig)
    p = S / "parity" / "merge_0W.json"
    orig = p.read_text()
    _edit(p, lambda r: r.update(adapter_sha256="9" * 64))
    assert PT.xsmoke_verdict(sm, xlay, cfg)["failed"] == ["merge_parity_Q3RS150"]
    p.write_text(orig)
    (R / "load" / "load.json").unlink()
    assert PT.main(["xsmoke-verdict", "--smoke", str(sm)]) == 1  # the CLI: FAIL is exit 1


def test_xsmoke_verdict_never_reads_a_success_field(xlay):
    cfg = TJ.filled_cfg()
    files = build_tiers(xlay, cfg)
    sm = xlay.xsmoke_dir("tvr-xsmoke-202611121501")
    build_xsmoke(xlay, sm, cfg, files)
    base = PT.xsmoke_verdict(sm, xlay, cfg)
    for p in sm.rglob("results.json"):
        _edit(p, lambda r: r.update(sr={"k": -1, "n": 0, "rate": 9.9}, by_category={"SR-easy": {"sr": "x"}},
                                    success=False))
    for p in sm.rglob("trajectory.json"):
        p.write_text(json.dumps({"task_id": "x", "result": {"success": "NOT A BOOL"}}))
    for d in sm.rglob("*__c"):
        (d / "per_task.jsonl").write_text("not json\n")
    again = PT.xsmoke_verdict(sm, xlay, cfg)
    assert again["verdict"] == base["verdict"] == "PASS"
    assert [c["detail"] for c in again["checks"]] == [c["detail"] for c in base["checks"]]
    text = (sm / "smoke_verdict.json").read_text()
    assert '"sr"' not in text and '"success"' not in text


# ------------------------------------------------------------------------------------------ payloads: dry runs
def xdry(tmp_path, script, *args, root="explore/rl", **env):
    run = tmp_path / "jobs" / "run-202611120000"
    e = {"VPD_DRYRUN": "1", "VPD_TVR_OUT": str(tmp_path / "tvr"), "VPD_MERGE_DIR": "/dev/shm/vpd_merged"}
    if root is not None:
        e["VPD_TVR_ROOT"] = root
    e.update(env)
    res = _bash([str(JOBS_DIR / script), *args, str(run)], e, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    return res, TJ._dry(res.stdout), run


def _bash(args, extra, cwd):
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    env.update(extra)
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env, cwd=str(cwd), timeout=300)


def test_payload_dry_xsmoke(tmp_path):
    res, lines, run = xdry(tmp_path, "tvr_xsmoke.sh", root=None)
    k = TJ._common_dry_asserts(lines, run, tmp_path, 3840)
    per = []
    for s in XS:
        per += ["rm", "raw_bf16:--start", "dry", "hfparity", "eval:raw:c,u,blind", f"reap:session_{s}", "rm",
                "train:0W", "rm", "merge:0W", "merge_parity:--start", "eval:0W:c", f"reap:eval_{s}", "rm"]
    assert k == ["prereg_committed", "guard"] + per + ["xsmoke-verdict", "expect", "DONE"], k
    SM = tmp_path / "tvr" / "runs" / "explore" / "rl" / "smoke" / run.name
    for s in XS:
        raw = f"/dev/shm/vpd_merged/xsmoke__{s}__raw"
        ld = next(ln for ln in lines if f"raw_bf16 --start {s} " in ln)
        assert ld.endswith(f"--out {raw} --report {SM}/{s}/load/load.json")
        sess = next(ln for ln in lines if f"--start {s} --tier raw" in ln)
        assert (f"--arms c,u,blind --task-set smoke8 --name-prefix smoke --out-root {SM}/{s}/evals --model-dir {raw} "
                f"--mechanism-only --vllm-parity {SM}/{s}/dry/dry_run.json --resume") in sess
        tr = next(ln for ln in lines if "n2_train_vpd.py" in ln and f"--start {s} " in ln and "--dry-run" not in ln)
        assert (f"--data {tmp_path}/tvr/tiers/v1/warmup0w20.jsonl --out {SM}/{s}/adapters/0W --limit 16 --epochs 1 "
                "--seed 1001 --phase 0W --threads 2 --num-workers 3") in tr
        mp = next(ln for ln in lines if "merge_parity" in ln and f"--start {s} " in ln)
        assert (f"--merged /dev/shm/vpd_merged/xsmoke__{s}__0W --demos {tmp_path}/tvr/tiers/v1/warmup0w20.jsonl --n 20 "
                f"--out {SM}/{s}/parity/merge_0W.json") in mp
        ev = next(ln for ln in lines if f"--start {s} --tier 0W" in ln)
        assert (f"--arms c --task-set smoke4 --name-prefix xsmoke --out-root {SM}/{s}/evals --model-dir "
                f"/dev/shm/vpd_merged/xsmoke__{s}__0W --adapter-meta {SM}/{s}/adapters/0W/meta.json --mechanism-only "
                f"--vllm-parity {SM}/{s}/dry/dry_run.json --resume") in ev
    assert "eval500" not in res.stdout and "runs/pilot" not in res.stdout and "replay --task-set" not in res.stdout
    assert f"guard tvr-xsmoke --run {run}" in res.stdout and f"xsmoke-verdict --smoke {SM}" in res.stdout


@pytest.mark.parametrize("start", XS)
def test_payload_dry_xtrain(tmp_path, start):
    res, lines, run = xdry(tmp_path, "tvr_train.sh", start)
    k = TJ._kinds(lines)
    assert k == ["prereg_committed", "guard", "adapter-ok:0W", "train:0W", "adapter-ok:100", "train:tier100",
                 "adapter-ok:ALL", "train:tierALL", "expect", "DONE"]
    P = tmp_path / "tvr" / "runs" / "explore" / "rl" / start
    tr = [ln for ln in lines if "n2_train_vpd.py" in ln]
    assert f"--data {tmp_path}/tvr/tiers/v1/demos_ALL.jsonl --out {P}/adapters/ALL" in tr[2]
    assert all(f"--start {start} " in ln and "--epochs auto --seed 1001" in ln and "--threads 2 --num-workers 3" in ln
               for ln in tr)
    assert tr[0].endswith("--phase 0W --threads 2 --num-workers 3") and tr[1].endswith(f"--init-adapter {P}/adapters/0W")
    assert f"guard tvr-xtrain-{start.lower()} --run {run}" in res.stdout and "s1002" not in res.stdout
    assert f"expect tvr-xtrain-{start.lower()} --run {run}" in res.stdout and "runs/pilot" not in res.stdout


@pytest.mark.parametrize("start,part", [("Q3RD", "top"), ("Q3RS150", "low")])
def test_payload_dry_xeval(tmp_path, start, part):
    res, lines, run = xdry(tmp_path, "tvr_eval.sh", start, part)
    k = TJ._common_dry_asserts(lines, run, tmp_path, 8160)
    head = ["prereg_committed", "guard", "replay:sft2", "reap:canary_replay", "render_parity:--replay"]
    if part == "top":
        want = head + ["adapter-ok:0W", "adapter-ok:ALL", "parity-ok", "rm", "merge:ALL", "merge_parity:--start",
                       "eval:ALL:c,u,blind", "reap:eval_ALL", "rm"] + ["score_tvr:official"] * 3
    else:
        sess = []
        for tier in ("0W", "100"):
            sess += ["parity-ok", "rm", f"merge:{tier}", "merge_parity:--start", f"eval:{tier}:c", f"reap:eval_{tier}",
                     "rm"]
        want = head + ["adapter-ok:0W", "adapter-ok:100"] + sess + ["score_tvr:official"] * 2
    assert k == want + ["expect", "DONE"], k
    P = tmp_path / "tvr" / "runs" / "explore" / "rl" / start
    evs = [ln for ln in lines if "run_tvr eval" in ln]
    assert all(f"--start {start} " in ln and "--task-set eval500 --name-prefix xrl" in ln
               and f"--out-root {P}/evals" in ln and "--model-dir /dev/shm/vpd_merged/xrl__" in ln for ln in evs)
    mps = [ln for ln in lines if "merge_parity" in ln]
    assert len(mps) == len(evs) and all("--n 20" in ln and f"--out {P}/parity/merge_" in ln for ln in mps)
    assert f"--run-dir {P}/canary/{part} --images-dir {P}/canary/{part}/images" in res.stdout
    offs = [ln for ln in lines if "score_tvr official" in ln]
    assert all(f"--run-dir {P}/evals/xrl__{start}__" in ln and f"--served <code:{start}>" in ln for ln in offs)
    assert f"guard tvr-xeval-{start.lower()}-{part} --run {run}" in res.stdout and "runs/pilot" not in res.stdout


@pytest.mark.parametrize("job", sorted(PT.XJOBS))
def test_explore_dry_run_records_every_required_step(tmp_path, job):
    spec = PT.XJOBS[job]
    args = [a for a in (spec["start"], spec["part"]) if a]
    res, lines, run = xdry(tmp_path, spec["script"], *args, root=None if job == "tvr-xsmoke" else "explore/rl",
                           VPD_DRY_NTOP="ALL")
    names = [json.loads(s)["name"] for s in (run / "steps.jsonl").read_text().splitlines()]
    absent = [r for r in PT.required_steps(job, TJ.filled_cfg()) if not any(n == r or n.startswith(r + ":")
                                                                           for n in names)]
    assert not absent, (job, absent, names)


def test_payload_root_and_start_refusals(tmp_path):
    run = str(tmp_path / "jobs" / "r")
    for script, args in (("tvr_train.sh", ["Q3RD", run]), ("tvr_eval.sh", ["Q3RS150", "top", run])):
        r = _bash([str(JOBS_DIR / script), *args], {"VPD_DRYRUN": "1"}, tmp_path)  # pilot root: pilot starts only
        assert r.returncode == 2 and "usage" in r.stderr and "Q3B Q3S E2" in r.stderr, script
    for script, args in (("tvr_train.sh", ["Q3B", run]), ("tvr_eval.sh", ["E2", "low", run])):
        r = _bash([str(JOBS_DIR / script), *args], {"VPD_DRYRUN": "1", "VPD_TVR_ROOT": "explore/rl"}, tmp_path)
        assert r.returncode == 2 and "Q3RD Q3RS150" in r.stderr, script
    for bad in ("pilot/../explore/rl", "explore", "/abs", "explore/rl/", ""):
        r = _bash([str(JOBS_DIR / "tvr_train.sh"), "Q3RD", run], {"VPD_DRYRUN": "1", "VPD_TVR_ROOT": bad}, tmp_path)
        if bad == "":  # empty = unset = pilot: Q3RD is not a pilot start
            assert r.returncode == 2 and "Q3B Q3S E2" in r.stderr
        else:
            assert r.returncode == 2 and "VPD_TVR_ROOT must be pilot or explore/rl" in r.stderr, bad
    r = _bash([str(JOBS_DIR / "tvr_xsmoke.sh")], {"VPD_DRYRUN": "1"}, tmp_path)
    assert r.returncode == 2 and "usage" in r.stderr


def test_payload_pilot_root_unchanged_when_set_explicitly(tmp_path):
    """VPD_TVR_ROOT=pilot is the default: the same commands as unset (the pilot tests run with it unset)."""
    def norm(lines, d):  # the seconds left before the deadline may differ by one between the two runs
        return [re.sub(r"(timeout -k 30|--deadline-s) \d+", r"\1 N", ln.replace(str(d), "X")) for ln in lines]

    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    a = xdry(tmp_path / "a", "tvr_eval.sh", "Q3B", "low", root=None)[1]
    b = xdry(tmp_path / "b", "tvr_eval.sh", "Q3B", "low", root="pilot")[1]
    assert norm(a, tmp_path / "a") == norm(b, tmp_path / "b")
    assert any("--name-prefix pilot" in ln and "/runs/pilot/Q3B/evals" in ln for ln in a)


def test_xsmoke_script_mode_shared_parents_and_hooks():
    f = JOBS_DIR / "tvr_xsmoke.sh"
    assert os.access(f, os.X_OK) and (f.stat().st_mode & 0o777) == 0o755
    assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0
    text = f.read_text()
    shared = {'"$TVR/runs"', '"$TVR/runs/explore"', '"$TVR/runs/explore/rl"', '"$TVR/runs/explore/rl/smoke"',
              '"$TVR/runs/pilot"', '"$CAL"'}
    for line in text.splitlines():
        if re.match(r"\s*mkdir ", line):
            args = shlex.split(line.strip(), posix=False)[2:] if "-p" in line else shlex.split(line.strip())[1:]
            assert not shared & set(args), line
    assert 'vpd_chown_trap "$RUN" "$SM"' in text and "runs/pilot" not in text
    code = [ln for ln in text.splitlines() if "VPD_TVR_OUT" in ln and not ln.lstrip().startswith("#")]
    assert len(code) == 2 and "TVR=$VPD_TVR_OUT" in code[1]
    # the shared helper block is the pilot scripts' (byte for byte)
    def block(t):
        return t[t.index("# ---- helpers (identical"):t.index("# ---- end of the shared helpers ----")]
    assert block(text) == block((JOBS_DIR / "tvr_mech_smoke.sh").read_text())


# ------------------------------------------------------------------------------------------ payloads: non-dry fakes
FAKE_PY2 = TJ.FAKE_PY.replace(
    '  *" vpd.tvr.pilot_tvr parity-ok "*) exit "${FAKE_PARITY_RC:-1}" ;;',
    '  *" vpd.tvr.pilot_tvr parity-ok "*)  # FAKE_PARITY_SEQ="1 4": successive answers (then the last one)\n'
    '    if [[ -n ${FAKE_PARITY_SEQ:-} ]]; then\n'
    '      n=$(cat "$FAKE_LOG.pn" 2>/dev/null || echo 0); echo $((n + 1)) > "$FAKE_LOG.pn"\n'
    '      read -r -a seq <<< "$FAKE_PARITY_SEQ"; i=$n; ((i < ${#seq[@]})) || i=$((${#seq[@]} - 1))\n'
    '      exit "${seq[$i]}"\n'
    '    fi\n'
    '    exit "${FAKE_PARITY_RC:-1}" ;;').replace(
    '  *" vpd.tvr.render_parity "*)',
    '  *" vpd.tvr.run_tvr eval "*)  # FAKE_EVAL_WRITES=1: a results.json per arm (so the official steps find one)\n'
    '    if [[ -n ${FAKE_EVAL_WRITES:-} ]]; then\n'
    '      prev=""; od=""; pf=""; st=""; ti=""; ar=""\n'
    '      for x in "$@"; do\n'
    '        case $prev in --out-root) od=$x ;; --name-prefix) pf=$x ;; --start) st=$x ;; --tier) ti=$x ;;\n'
    '          --arms) ar=$x ;; esac\n'
    '        prev=$x\n'
    '      done\n'
    '      for a in ${ar//,/ }; do mkdir -p "$od/${pf}__${st}__${ti}__$a" && echo \'{"valid": true}\' \\\n'
    '        > "$od/${pf}__${st}__${ti}__$a/results.json"; done\n'
    '    fi\n'
    '    exit 0 ;;\n'
    '  *" vpd.tvr.render_parity "*)')
assert "read -r -a seq" in FAKE_PY2 and "FAKE_EVAL_WRITES" in FAKE_PY2  # both anchors of TJ.FAKE_PY were found


def xfake_tree(tmp_path: Path, script: str):
    payload, env = TJ.fake_tree(tmp_path, script)
    (tmp_path / "fakepy").write_text(FAKE_PY2)
    return payload, dict(env, FAKE_EVAL_WRITES="1")


def test_nondry_xeval_merge_parity_report_only(tmp_path):
    payload, env = xfake_tree(tmp_path, "tvr_eval.sh")
    env = dict(env, VPD_TVR_ROOT="explore/rl", FAKE_ADAPTER_RC="0")
    P = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "rl" / "Q3RD"
    # merge parity runs and exits 1 (FAILED); its json is then current + failed (parity-ok 4): report only, DONE
    run = tmp_path / "OUT" / "jobs" / "tvr-xeval-q3rd-low-202611120000"
    res = _bash([str(payload), "Q3RD", "low", str(run)],
                dict(env, FAKE_FAIL="vpd.tvr.merge_parity ", FAKE_PARITY_SEQ="1 4 1 4"), tmp_path)
    assert res.returncode == 0, res.stderr[-1500:] + res.stdout[-1500:]
    st = [(s["name"], s["rc"]) for s in TJ._steps(run)]
    for tier in ("0W", "100"):
        assert (f"merge_parity_{tier}", 1) in st and (f"merge_parity_{tier}:report_only", 0) in st
    assert (run / "DONE").exists() and PT.steps_check("tvr-xeval-q3rd-low", run)["ok"]
    assert (P / "evals").is_dir() and not (tmp_path / "OUT" / "tvr" / "runs" / "pilot").exists()
    log = (tmp_path / "fake.log").read_text()
    assert "--name-prefix xrl" in log and f"--out-root {P}/evals" in log and "guard tvr-xeval-q3rd-low" in log
    # a crash without a json (parity-ok 1 after the step): a failure, no DONE
    (tmp_path / "fake.log.pn").unlink()
    run2 = tmp_path / "OUT" / "jobs" / "tvr-xeval-q3rd-low-202611120001"
    res = _bash([str(payload), "Q3RD", "low", str(run2)],
                dict(env, FAKE_FAIL="vpd.tvr.merge_parity ", FAKE_PARITY_SEQ="1"), tmp_path)
    assert res.returncode == 1 and not (run2 / "DONE").exists()
    assert "failed steps: merge_parity_0W merge_parity_100" in res.stdout
    # a current json that says FAILED (not rerun): report only, rc 0
    run3 = tmp_path / "OUT" / "jobs" / "tvr-xeval-q3rd-top-202611120002"
    res = _bash([str(payload), "Q3RD", "top", str(run3)], dict(env, FAKE_PARITY_RC="4"), tmp_path)
    assert res.returncode == 0 and ("merge_parity_ALL:report_only", 0) in [(s["name"], s["rc"])
                                                                          for s in TJ._steps(run3)]


def test_nondry_pilot_eval_merge_parity_still_gated(tmp_path):
    payload, env = xfake_tree(tmp_path, "tvr_eval.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-low-202611120000"
    res = _bash([str(payload), "Q3S", "low", str(run)],
                dict(env, FAKE_ADAPTER_RC="0", FAKE_FAIL="vpd.tvr.merge_parity ", FAKE_PARITY_SEQ="1 4 1 4"), tmp_path)
    assert res.returncode == 1 and not (run / "DONE").exists()
    names = [s["name"] for s in TJ._steps(run)]
    assert not any(n.endswith(":report_only") for n in names) and "failed steps: merge_parity_0W merge_parity_100" \
        in res.stdout
    run2 = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-top-202611120001"
    res = _bash([str(payload), "Q3S", "top", str(run2)], dict(env, FAKE_ADAPTER_RC="0", FAKE_PARITY_RC="4"), tmp_path)
    assert res.returncode == 1 and "merge_parity_ALL:failed_kept" in [s["name"] for s in TJ._steps(run2)]


def test_nondry_xtrain_writes_under_the_explore_root(tmp_path):
    payload, env = xfake_tree(tmp_path, "tvr_train.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-xtrain-q3rs150-202611120000"
    res = _bash([str(payload), "Q3RS150", str(run)], dict(env, VPD_TVR_ROOT="explore/rl"), tmp_path)
    assert res.returncode == 0, res.stderr[-1500:]
    assert (tmp_path / "OUT" / "tvr" / "runs" / "explore" / "rl" / "Q3RS150" / "adapters").is_dir()
    assert not (tmp_path / "OUT" / "tvr" / "runs" / "pilot").exists() and (run / "DONE").exists()
    log = (tmp_path / "fake.log").read_text()
    assert log.count("n2_train_vpd.py --start Q3RS150") == 3 and "guard tvr-xtrain-q3rs150" in log


def test_nondry_xsmoke_all_steps_and_failures(tmp_path):
    payload, env = xfake_tree(tmp_path, "tvr_xsmoke.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-xsmoke-202611120000"
    res = _bash([str(payload), str(run)], env, tmp_path)
    assert res.returncode == 0, res.stderr[-1500:] + res.stdout[-1500:]
    names = [s["name"] for s in TJ._steps(run)]
    assert (run / "DONE").exists() and PT.steps_check("tvr-xsmoke", run)["ok"]
    for s in XS:
        for x in ("load", "dry", "hfparity", "session", "reap_session", "rm_raw", "train", "merge", "merge_parity",
                  "eval", "reap_eval", "rm_merged"):
            assert f"{x}_{s}" in names, (x, s)
    assert names.index("rm_raw_Q3RD") < names.index("train_Q3RD") < names.index("load_Q3RS150")
    SM = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "rl" / "smoke" / run.name
    assert SM.is_dir() and names[-2:] == ["verdict", "expect"]
    log = (tmp_path / "fake.log").read_text()
    assert "vpd.tvr.pilot_tvr xsmoke-verdict --smoke " + str(SM) in log and "--task-set eval500" not in log
    # the training fails: no merge / merge parity / merged eval for that start; the other start still runs
    run2 = tmp_path / "OUT" / "jobs" / "tvr-xsmoke-202611120001"
    res = _bash([str(payload), str(run2)], dict(env, FAKE_FAIL="--start Q3RD --data"), tmp_path)
    assert res.returncode == 1 and not (run2 / "DONE").exists()
    names = [s["name"] for s in TJ._steps(run2)]
    assert "train_Q3RD" in names and "merge_Q3RD" not in names and "eval_Q3RD" not in names
    assert "eval_Q3RS150" in names and "verdict" in names
    # the bf16 load fails: no raw session for that start (recorded), training still runs
    run3 = tmp_path / "OUT" / "jobs" / "tvr-xsmoke-202611120002"
    res = _bash([str(payload), str(run3)], dict(env, FAKE_FAIL="raw_bf16 --start Q3RS150"), tmp_path)
    names = [s["name"] for s in TJ._steps(run3)]
    assert res.returncode == 1 and "session_Q3RS150:no_raw_folder" in names and "train_Q3RS150" in names
    # merge parity FAILED: report only
    run4 = tmp_path / "OUT" / "jobs" / "tvr-xsmoke-202611120003"
    res = _bash([str(payload), str(run4)], dict(env, FAKE_FAIL="vpd.tvr.merge_parity ", FAKE_PARITY_RC="4"),
                tmp_path)
    st = [(s["name"], s["rc"]) for s in TJ._steps(run4)]
    assert res.returncode == 0 and ("merge_parity_Q3RD:report_only", 0) in st and (run4 / "DONE").exists()


# ------------------------------------------------------------------------------------------ raw_bf16 (tiny model)
def test_raw_bf16_tiny_float32_checkpoint(tmp_path):
    """raw_bf16.save on a tiny float32 Qwen3-VL folder: the trainer's load (bf16, sdpa), merge.py's files, every
    check tensor == the checkpoint cast by .to(bfloat16); refusals."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if "starVLA" in sys.executable:
        pytest.skip("save_pretrained runs with PY_Q3 (merge.py's interpreter)")
    from safetensors.torch import save_file

    from vpd.tests import test_train as TT
    from vpd.tvr import raw_bf16 as RB

    model = TT._tiny_model(2048, torch.float32, seed=4)
    ck = tmp_path / "ck32"
    ck.mkdir()
    model.config.save_pretrained(str(ck))
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()},
              str(ck / "model.safetensors"), metadata={"format": "pt"})
    out = tmp_path / "raw"
    rep = RB.save("Q3S", out=out, model_path=ck, min_free_gb=0, _allow_shared_out=True)
    assert rep["ok"], rep["problems"]
    assert rep["checkpoint_dtypes"].keys() == {"F32"} and rep["saved_dtypes"].keys() == {"BF16"}
    assert rep["tensor_checks_ok"] and len(rep["tensor_checks"]) == 4
    assert {c["key"] for c in rep["tensor_checks"]} >= set(RB.FIXED_KEYS)
    assert rep["n_keys"]["checkpoint"] == rep["n_keys"]["saved"] and rep["dtype"] == "bfloat16"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg.get("dtype") == "bfloat16" and "attn_implementation" not in cfg
    assert json.loads((out / "generation_config.json").read_text())["do_sample"] is False
    for f in ("tokenizer_config.json", "preprocessor_config.json"):
        assert (out / f).is_file()
    assert json.loads((out / "vpd_raw_bf16.json").read_text())["ok"] is True
    with pytest.raises(SystemExit, match="not empty"):
        RB.save("Q3S", out=out, model_path=ck, min_free_gb=0, _allow_shared_out=True)
    from vpd.train import merge as MG

    with pytest.raises(SystemExit, match="refused"):
        MG.choose_out(Path("/mnt/umm/users/yinbaiqiao/x"), "r")


def test_raw_bf16_cli_help():
    res = subprocess.run([sys.executable, "-B", "-m", "vpd.tvr.raw_bf16", "--help"], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(paths.WORKTREE)), cwd=str(paths.WORKTREE), timeout=120)
    assert res.returncode == 0 and "usage" in res.stdout.lower()


# ------------------------------------------------------------------------------------------ compare_rl
N_TASKS = 24
CATS = (("ithor", "easy"), ("ithor", "hard"), ("procthor", "easy"), ("procthor", "hard"))
IDS = [f"t{i:03d}" for i in range(N_TASKS)]
SUCC = {("Q3RD", "0W", "c"): set(range(10)), ("Q3B", "0W", "c"): {8, 9, 10, 11}, ("Q3S", "0W", "c"): {0, 1},
        ("Q3RS150", "0W", "c"): {0, 1, 2}, ("E2", "0W", "c"): {5},
        ("Q3RD", "ALL", "c"): set(range(12)), ("Q3B", "ALL", "c"): set(range(6, 16)),
        ("Q3RS150", "ALL", "u"): {1, 2}, ("Q3S", "ALL", "u"): {2, 3, 4}, ("E2", "ALL", "c"): {0, 1, 2, 3},
        ("Q3RS150", "100", "c"): {7}}


def write_eval(d: Path, succ: set, start: str, tier: str, arm: str, adapter_sha: str = None) -> Path:
    for i, tid in enumerate(IDS):
        ds, diff = CATS[i % 4]
        wj(d / "tasks" / tid / "trajectory.json", {"task_id": tid, "dataset": ds, "difficulty": diff,
                                                    "result": {"success": i in succ, "stopped": True}})
    (d / "per_task.jsonl").write_text("".join(json.dumps({"task_id": tid, "success": int(i in succ)}) + "\n"
                                              for i, tid in enumerate(IDS)))
    wj(d / "results.json", {"valid": True, "n_tasks": N_TASKS, "tasks_missing": [], "start": start, "tier": tier,
                            "arm": arm, "mechanism_only": False, "code_sha256": PT.HARNESS_SHA256_PREFIX + "7" * 52,
                            "adapter": {"adapter_sha256": adapter_sha}, "sr": {"k": len(succ), "n": N_TASKS}})
    wj(d / "official_check.json", {"equal": True})
    return d


@pytest.fixture
def cworld(tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "EVAL500_N", N_TASKS)
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "c0ffee"})
    L = PT.Layout(tmp_path / "out")
    cfg = TJ.filled_cfg()
    files = build_tiers(L, cfg)
    xroot, proot = tmp_path / "explore_rl", tmp_path / "pilot"
    tasks_dir = tmp_path / "repo" / "data" / "tasks"
    wj(tasks_dir / "eval.json", [{"task_id": t, "dataset": CATS[i % 4][0], "difficulty": CATS[i % 4][1]}
                                 for i, t in enumerate(IDS)])
    for s in XS:
        build_xadapters(L, s, cfg, files, root=xroot)
        for tier, arm in C.cells():
            ad = xroot / s / "adapters" / PT.EVAL_ADAPTER[tier] / "adapter.pt"
            write_eval(C.eval_dir(xroot, s, tier, arm, "xrl"), SUCC.get((s, tier, arm), set()), s, tier, arm,
                       fsha(ad))
    for s in ("Q3B", "Q3S", "E2"):
        for tier, arm in C.cells():
            write_eval(C.eval_dir(proot, s, tier, arm), SUCC.get((s, tier, arm), set()), s, tier, arm, "p" * 64)
    return {"L": L, "xroot": xroot, "proot": proot, "tasks": tasks_dir, "cfg": cfg}


def crun(w, **kw):
    return CR.compare(w["xroot"], w["proot"], tasks_dir=w["tasks"], L=w["L"], **kw)


def test_compare_rl_cells_orientation_and_five_start_table(cworld):
    rep = crun(cworld)
    assert rep["schema"] == "vpd.tvr.compare_rl/1" and rep["exploratory"] is True and rep["validation"]["ok"]
    assert rep["cells"] == ["0W__c", "100__c", "ALL__c", "ALL__u", "ALL__blind"] and rep["z"] == 1.96
    assert rep["prereg"]["row_text"].startswith("| 2026-11-12 |") and rep["multiplicity_correction"] == "none"
    assert list(rep["comparisons"]) == ["Q3RD-Q3B", "Q3RS150-Q3B", "Q3RD-Q3S", "Q3RS150-Q3S"]
    assert [rep["comparisons"][k]["role"] for k in rep["comparisons"]] == ["primary", "primary", "supplement",
                                                                           "supplement"]
    assert rep["comparisons"]["Q3RS150-Q3B"]["b_is"] == "only Q3RS (Q3RS150) succeeds"
    want = {("Q3RD-Q3B", "0W__c"): (10, 4, 8, 2), ("Q3RD-Q3B", "ALL__c"): (12, 10, 6, 4),
            ("Q3RS150-Q3S", "ALL__u"): (2, 3, 1, 2), ("Q3RS150-Q3B", "100__c"): (1, 0, 1, 0),
            ("Q3RD-Q3S", "0W__c"): (10, 2, 8, 0), ("Q3RS150-Q3S", "ALL__blind"): (0, 0, 0, 0)}
    for (name, cell), (k1, k2, b, c) in want.items():
        x = rep["comparisons"][name]["cells"][cell]
        p = C.exact_p(b, c)
        assert (x["k_first"], x["k_second"], x["b"], x["c"], x["n"]) == (k1, k2, b, c, N_TASKS), (name, cell)
        assert x["p"] == float(p) and x["p_exact"] == f"{p.numerator}/{p.denominator}" and not x["missing"]
        assert x["wilson95_first"] == C.wilson(k1, N_TASKS, 1.96) and x["both"] + x["neither"] + b + c == N_TASKS
    t = rep["five_start_table"]
    assert len(t) == 25 and [r["start"] for r in t[:5]] == ["Q3B", "Q3S", "E2", "Q3RD", "Q3RS150"]
    assert [(r["tier"], r["arm"]) for r in t[::5]] == [tuple(c.split("__")) for c in rep["cells"]]
    e2 = next(r for r in t if r["start"] == "E2" and r["tier"] == "ALL" and r["arm"] == "c")
    assert e2["k"] == 4 and e2["source"] == "pilot" and not any("E2" in k for k in rep["comparisons"])
    assert rep["crosscheck"]["ok"] and rep["crosscheck"]["used_sides_all_checked"] and rep["missing_cells"] == []
    md = CR.render_md(rep)
    assert "## RL 对 Q3B（主报告）" in md and "## RL 对 Q3S（补充" in md and "### Q3RS (Q3RS150) 对 Q3B" in md
    assert "| 0W | c | 10/24" in md and "偏乐观" in md and "推断" in md and "全部一致" in md and "E2 只列绝对数" in md


def test_compare_rl_main_writes_beside_the_explore_root_never_the_pilot(cworld, monkeypatch, capsys):
    monkeypatch.setattr(PT, "LAYOUT", cworld["L"])
    args = ["--explore-root", str(cworld["xroot"]), "--pilot-root", str(cworld["proot"]), "--tasks-dir",
            str(cworld["tasks"])]
    assert CR.main(args) == 0
    out = cworld["xroot"] / "COMPARE_RL.json"
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == CR.SCHEMA and out.with_suffix(".md").is_file()
    assert not (cworld["proot"] / "COMPARE_RL.json").exists()
    assert CR.main(args + ["--out", str(cworld["proot"] / "x" / "C.json")]) == 2
    assert "under the pilot root" in capsys.readouterr().err and not (cworld["proot"] / "x").exists()


def test_compare_rl_refuses_until_all_ten_explore_evals_validate(cworld):
    xroot = cworld["xroot"]
    d = C.eval_dir(xroot, "Q3RS150", "ALL", "blind", "xrl")
    breaks = [
        (d / "results.json", lambda r: r.update(valid=False), "not valid"),
        (d / "results.json", lambda r: r.update(n_tasks=N_TASKS - 1), "not valid"),
        (d / "results.json", lambda r: r.update(mechanism_only=True), "mechanism_only True"),
        (d / "results.json", lambda r: r.update(arm="u"), "arm 'u' != 'blind'"),
        (d / "results.json", lambda r: r.update(code_sha256="0" * 64), "harness code_sha256"),
        (d / "results.json", lambda r: r["adapter"].update(adapter_sha256="0" * 64), "served adapter"),
        (d / "official_check.json", lambda r: r.update(equal=False), "official_check.json equal False"),
        (xroot / "Q3RS150" / "adapters" / "ALL" / "meta.json", lambda r: r.update(seed=1002), "meta seed 1002"),
        (xroot / "Q3RS150" / "adapters" / "100" / "meta.json", lambda r: r["init_adapter"].update(sha256="1" * 64),
         "retrain"),
    ]
    for path, fn, want in breaks:
        orig = path.read_text()
        _edit(path, fn)
        with pytest.raises(CR.Refused) as e:
            crun(cworld)
        assert "not all 10 explore evals" in e.value.message and want in e.value.message, (want, e.value.message)
        path.write_text(orig)
    shutil.rmtree(C.eval_dir(xroot, "Q3RD", "100", "c", "xrl"))
    with pytest.raises(CR.Refused) as e:  # Refused is SystemExit(2): the reason is in .message
        crun(cworld)
    assert "Q3RD/100__c (xrl__Q3RD__100__c)" in e.value.message


def test_compare_rl_row_crosscheck_and_missing(cworld, tmp_path):
    p = tmp_path / "PREREG.md"
    p.write_text("no rows here\n", encoding="utf-8")
    with pytest.raises(CR.Refused) as e:
        crun(cworld, prereg=p)
    assert "no single section-17 row" in e.value.message
    # a per_task.jsonl row that disagrees with its trajectory: a cross-check mismatch, numbers from trajectories
    pt = C.eval_dir(cworld["proot"], "Q3B", "ALL", "c") / "per_task.jsonl"
    lines = pt.read_text().splitlines()
    lines[0] = json.dumps({"task_id": IDS[0], "success": 1})
    pt.write_text("\n".join(lines) + "\n")
    rep = crun(cworld)
    assert not rep["crosscheck"]["ok"] and rep["crosscheck"]["mismatches"][0]["eval"] == "Q3B/ALL__c"
    assert rep["comparisons"]["Q3RD-Q3B"]["cells"]["ALL__c"]["k_second"] == 10
    # an E2 pilot eval without trajectories: its five-start row is missing (no comparison uses E2)
    shutil.rmtree(C.eval_dir(cworld["proot"], "E2", "0W", "c") / "tasks")
    rep = crun(cworld)
    assert rep["missing_cells"] == ["five-start table E2 0W__c"]
    assert all(not c["missing"] for comp in rep["comparisons"].values() for c in comp["cells"].values())


def test_compare_tvr_prefix_default_is_pilot(tmp_path):
    assert C.eval_dir(tmp_path, "Q3S", "ALL", "u") == tmp_path / "Q3S" / "evals" / "pilot__Q3S__ALL__u"
    assert C.eval_dir(tmp_path, "Q3RD", "0W", "c", "xrl") == tmp_path / "Q3RD" / "evals" / "xrl__Q3RD__0W__c"


@pytest.mark.parametrize("mod", ["compare_rl", "pilot_tvr"])
def test_cli_help(mod):
    res = subprocess.run([sys.executable, "-B", "-m", f"vpd.tvr.{mod}", "--help"], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(paths.WORKTREE)), cwd=str(paths.WORKTREE), timeout=120)
    assert res.returncode == 0 and "usage" in res.stdout.lower()


def test_readme_mentions_the_explore_files():
    text = (paths.VPD_PKG / "README.md").read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith("| `")]
    for f in ("tvr/compare_rl.py", "tvr/raw_bf16.py", "jobs/tvr_xsmoke.sh", "tests/test_tvr_explore.py"):
        assert any(f"`{f}`" in ln.split("|")[1] for ln in rows), f
    for d in ("tvr/runs/explore/rl/<起点>/", "tvr/runs/explore/rl/smoke/<RUN 名>/", "tvr/runs/explore/rl/"):
        assert f"`{d}" in text, d
