"""Tests of package tvrjobs (CONTRACT_wave2 sections 5.3, 7, 8.1, 8.2, 9): vpd/tvr/pilot_tvr.py, gate_tvr.py,
recheck_tvr.py, merge_parity.py and the payloads vpd/jobs/tvr_{data,cal,cal400,mech_smoke,train,eval}.sh.

Hermetic: no GPU, no network, no rendering, no model, no sco call (vpd.sco prepare / submit / delete are faked), no
git (prereg_state is faked).  Payloads run with VPD_DRYRUN=1 (they print their commands) and, for the failure
handling, non-dry in a temp code tree whose common_env.sh shim sources the real one and then points the interpreters
at a fake (a bash script that logs and answers each call) and VPD_OUT at a temp dir.  Login node, 1 core:
    cd TREE && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 ... nice -n 19 env VPD_WORKTREE=TREE PYTHONPATH=TREE \
        $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=$SCRATCH/pt vpd/tests/test_tvr_jobs.py
"""

from __future__ import annotations

import copy
import hashlib
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
from vpd.tvr import gate_tvr as G  # noqa: E402
from vpd.tvr import merge_parity as MP  # noqa: E402
from vpd.tvr import pilot_cfg  # noqa: E402
from vpd.tvr import pilot_tvr as PT  # noqa: E402
from vpd.tvr import recheck_tvr as RC  # noqa: E402
from vpd.walkerbench import pilot as WBP  # noqa: E402

JOBS_DIR = paths.VPD_PKG / "jobs"
SCRIPTS = ("tvr_data.sh", "tvr_cal.sh", "tvr_cal400.sh", "tvr_mech_smoke.sh", "tvr_train.sh", "tvr_eval.sh")
CATS = pilot_cfg.CATEGORIES


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wj(p: Path, obj) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
    return p


def fsha(p: Path) -> str:
    return sha(Path(p).read_bytes())


# ------------------------------------------------------------------------------------------ a filled pilot_cfg
def filled_cfg(n_top="ALL") -> dict:
    cfg = copy.deepcopy(pilot_cfg.FROZEN)
    r = cfg["render"]
    r.update(lvp_profile="envlib", lp_threads=2, replay_envs=5, eval_envs=5, recommended_measured=[6, 2],
             probe_run="OUT/jobs/tvr-probe-202611101501", probe_summary_sha256="5" * 64,
             ssim_definition_sha256="6" * 64, compare_verdict="same_rendering", deterministic=False)
    for k in cfg["data"]:
        cfg["data"][k] = "0" * 64 if k.endswith("sha256") else 1
    cfg["data"].update(kept=1599, dropped=1, F_eval500=0.0, F_source="stop0")
    cfg["tiers"]["n_top"] = n_top
    return cfg


def test_filled_cfg_has_nothing_missing():
    cfg = filled_cfg()
    for job in PT.JOBS:
        assert pilot_cfg.missing(job, cfg) == [], job
        assert PT.payload_constant_problems(cfg) == []
    cfg["data"]["kept"] = None  # independent of how far the real pilot_cfg is filled
    assert pilot_cfg.missing("tvr-train-q3b", cfg) == ["data.kept"] and pilot_cfg.missing("tvr-data", cfg) == []


def test_payload_constants_guard(lay):
    cfg = filled_cfg()
    assert PT.payload_constant_problems(cfg) == []
    cfg["tiers"]["replicate"]["seed"] = 1003
    cfg["gates"]["merge_parity"]["n_demos"] = 30
    probs = PT.payload_constant_problems(cfg)
    assert len(probs) == 2 and "tiers.replicate.seed" in probs[0] + probs[1]
    with pytest.raises(PT.Refused, match="the payloads use 1002"):
        PT.prepare("tvr-data", L=lay, cfg=cfg, events=[])


# ------------------------------------------------------------------------------------------ job table, timeouts
def test_job_table_names_gate_and_sco_rules():
    from vpd import ledger, sco

    assert len(PT.JOBS) == 13
    for job, spec in PT.JOBS.items():
        assert spec["name"] == "vpd-" + job
        sco.validate_name(spec["name"])
        assert ledger.gate_of(spec["name"]) == "tvr"
        assert (JOBS_DIR / spec["script"]).is_file()
    assert {j for j, s in PT.JOBS.items() if not s["as_root"]} == {"tvr-data"}
    assert PT.JOBS["tvr-eval-q3b-top"]["part"] == "top" and PT.JOBS["tvr-eval-e2-low"]["start"] == "E2"


@pytest.mark.parametrize("est", [1.0, 1.5, 2.5, 3.0, 3.4, 5.5])
def test_timeouts_reimplement_the_wb_rule(est):
    wb = WBP.job_timeouts("wb-q3b", est)
    for job in ("tvr-cal", "tvr-train-q3s", "tvr-eval-e2-top"):
        t = PT.job_timeouts(job, est)
        assert (t["outer_s"], t["soft_deadline_s"], t["kill_s"]) == (wb["outer_s"], wb["soft_deadline_s"], 120)
    for job in ("tvr-data", "tvr-msmoke"):
        assert PT.job_timeouts(job, est)["kill_s"] == 60


def test_timeouts_constants_of_section_7_and_refusals():
    want = {"tvr-data": (8700, 8520, 60), "tvr-cal": (19500, 19320, 120), "tvr-cal400": (5100, 4920, 120),
            "tvr-msmoke": (3300, 3120, 60)}
    for job, (outer, soft, kill) in want.items():
        t = PT.job_timeouts(job, PT.est_high(job)["est_high"])
        assert (t["outer_s"], t["soft_deadline_s"], t["kill_s"]) == (outer, soft, kill)
    with pytest.raises(PT.Refused, match="5.5 h job cap"):
        PT.job_timeouts("tvr-eval-q3b-low", 5.51)
    with pytest.raises(PT.Refused, match="outer timeout"):
        PT.job_timeouts("tvr-data", 0.2)


def test_payload_form_and_sco_snapshot_rewrite(tmp_path):
    from vpd import sco

    code = tmp_path / "code"
    for s in SCRIPTS:
        (code / "vpd" / "jobs").mkdir(parents=True, exist_ok=True)
        (code / "vpd" / "jobs" / s).write_text("#\n")
    run = tmp_path / "jobs" / "tvr-eval-q3s-top-202611111200"
    p = PT.payload_for("tvr-eval-q3s-top", run, 2.4)
    assert p[:6] == ["env", "VPD_JOB_DEADLINE_S=8160", "timeout", "-k", "120", "8340"]
    assert p[6:] == ["bash", str(JOBS_DIR / "tvr_eval.sh"), "Q3S", "top", str(run)]
    assert PT.payload_for("tvr-train-e2", run, 3.0)[-3:] == [str(JOBS_DIR / "tvr_train.sh"), "E2", str(run)]
    assert PT.payload_for("tvr-data", run, 2.5)[-2:] == [str(JOBS_DIR / "tvr_data.sh"), str(run)]
    new, rewrites = sco.snapshot_payload(p, code)
    assert new[7] == str(code / "vpd" / "jobs" / "tvr_eval.sh") and len(rewrites) == 1


def test_payload_scripts_mode_and_syntax():
    for s in SCRIPTS:
        f = JOBS_DIR / s
        assert os.access(f, os.X_OK) and (f.stat().st_mode & 0o777) == 0o755, s
        res = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert res.returncode == 0, (s, res.stderr)


def test_payloads_never_mkdir_a_shared_parent():
    shared = {'"$TVR/runs"', '"$TVR/runs/pilot"', '"$P"', '"$TVR/runs/smoke"', '"$CAL"', '"$TVR/runs/explore"',
              '"$TVR/runs/pilot/$START"'}
    for s in SCRIPTS:
        for line in (JOBS_DIR / s).read_text().splitlines():
            if re.match(r"\s*mkdir ", line):
                args = shlex.split(line.strip(), posix=False)[2:] if "-p" in line else shlex.split(line.strip())[1:]
                assert not shared & set(args), (s, line)
        text = (JOBS_DIR / s).read_text()
        assert "vpd_chown_trap" in text
        for bad in ('vpd_chown_trap "$RUN" "$P"\n', '"$TVR/runs"', 'chown -R ${VPD_JOB_UID}:${VPD_JOB_UID} "$CAL"'):
            assert bad not in text, (s, bad)


# ------------------------------------------------------------------------------------------ layout + fakes for prepare
@pytest.fixture
def lay(tmp_path, monkeypatch):
    L = PT.Layout(tmp_path / "out")
    monkeypatch.setattr(PT, "LAYOUT", L)
    monkeypatch.setenv("VPD_LEDGER", str(tmp_path / "out" / "ledger.jsonl"))
    monkeypatch.delenv("VPD_GPUH_CAP", raising=False)
    calls = {"prepare": [], "submit": [], "delete": []}
    from vpd import sco as vsco

    def fake_prepare(run, name, payload, pool="zoetrope", as_root=False, est_gph=None, gate=None):
        job = Path(run) / "job"
        job.mkdir(parents=True)
        (job / "prepared.json").write_text(json.dumps({"name": name, "payload": payload}))
        calls["prepare"].append({"run": str(run), "name": name, "payload": payload, "pool": pool, "as_root": as_root,
                                 "est_gph": est_gph, "gate": gate})
        return job / "launch.sh"

    def fake_submit(run, est_gph=None):
        (Path(run) / "job" / "submission.json").write_text(json.dumps({"job_id": "pt-fake1"}))
        calls["submit"].append(str(run))
        return "pt-fake1"

    monkeypatch.setattr(vsco, "prepare", fake_prepare)
    monkeypatch.setattr(vsco, "submit", fake_submit)
    monkeypatch.setattr(vsco, "delete", lambda run, force_live=False: calls["delete"].append(str(run)) or "deleted")
    monkeypatch.setattr(PT, "prereg_state", lambda: {"file": PT.PREREG, "commit": "c0ffee", "committed": True,
                                                     "status": "", "clean": True, "ok": True})
    monkeypatch.setattr(PT, "check_prereg", lambda cfg=None: [])
    L.calls = calls
    return L


def build_data(L: PT.Layout, cfg: dict, done=True) -> None:
    for key in PT.DATA_FILES:
        p = L.data_file(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if key == "floors_eval500_sha256":
            wj(p, {"F": {"rate": 0.0, "source": "stop0"}, "warnings": []})
        else:
            p.write_text(f"{key}\n")
        cfg["data"][key] = fsha(p)
    wj(L.demos / "manifest.json", {"kept": 1599, "dropped": 1, "kin_equal_all_kept": True})
    cfg["data"].update(kept=1599, dropped=1, F_eval500=0.0, F_source="stop0")
    if done:
        run = L.jobs / "tvr-data-202611101700"
        run.mkdir(parents=True, exist_ok=True)
        (run / "DONE").write_text("x")


def build_cal(L: PT.Layout, train_s=600.0, steps_per_s=6.0, mean_steps=13.0, tiers=("0W", "100", "ALL")) -> None:
    for t in tiers:
        wj(L.cal_adapter(t) / "meta.json", {"train_seconds": train_s, "load_seconds": 60.0})
        wj(L.cal_eval(t) / "results.json", {"valid": True, "n_tasks": 200, "tasks_missing": [],
                                            "sr": {"k": 100, "n": 200},
                                            "perf": {"steps_per_s": steps_per_s, "vllm_ready_s": 300.0,
                                                     "mean_steps_per_task": mean_steps}})
    wj(L.cal_shake("blind") / "results.json", {"valid": True, "perf": {"steps_per_s": 40.0, "vllm_ready_s": 280.0}})
    run = L.jobs / "tvr-cal-202611101800"
    run.mkdir(parents=True, exist_ok=True)
    (run / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in (
        {"name": "merge_ALL", "rc": 0, "secs": 95}, {"name": "merge_parity_ALL", "rc": 0, "secs": 420},
        {"name": "merge_0W", "rc": 0, "secs": 80}, {"name": "merge_parity_0W:skip", "rc": 0, "secs": 9999},
        {"name": "merge_100", "rc": 1, "secs": 9999})))
    (run / "DONE").write_text("x")
    wj(L.cal_decision, {"n_top": "ALL", "train_400": False, "futility": False})


def verdict(L: PT.Layout, v="PASS", name="tvr-msmoke-202611110000"):
    return wj(L.smoke_root / name / "smoke_verdict.json", {"verdict": v, "failed": [] if v == "PASS" else ["x"],
                                                           "checks": []})


def code_sha() -> str:
    from vpd.train import n2_train_vpd as T

    return T.code_fingerprint()["code_sha256"]


def build_adapters(L: PT.Layout, s: str, cfg: dict, tman: dict = None) -> None:
    for tier, init in PT.train_tiers(s, cfg):
        d = L.adapter(s, tier)
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter.pt").write_bytes(f"{s}-{tier}".encode())
        seed = 1002 if tier.endswith("_s1002") else 1001
        meta = {"start": s, "seed": seed, "phase": PT.PHASE[tier], "optimizer_steps": 30,
                "optimizer_steps_planned": 30, "code_sha256": code_sha(), "attn": "sdpa",
                "attn_sites": {"config": "sdpa", "model.language_model.config": "sdpa"}, "max_pixels": 1024000,
                "data_sha256": (tman or {}).get(PT.TIER_FILE[tier], "d" * 64)}
        if init:
            meta["init_adapter"] = {"sha256": fsha(L.adapter(s, init) / "adapter.pt")}
        wj(d / "meta.json", meta)


# ------------------------------------------------------------------------------------------ prepare / guards / budget
def test_prepare_data_creates_shared_parents_records_budget(lay):
    cfg = filled_cfg()
    launch = PT.prepare("tvr-data", L=lay, cfg=cfg, events=[])
    c = lay.calls["prepare"][-1]
    assert c["name"] == "vpd-tvr-data" and c["as_root"] is False and c["gate"] == "tvr" and c["est_gph"] == 2.5
    assert c["payload"][:6] == ["env", "VPD_JOB_DEADLINE_S=8520", "timeout", "-k", "60", "8700"]
    for d in lay.shared_parents():
        assert d.is_dir(), d
    run = Path(c["run"])
    info = json.loads((run / "prepared.json").read_text())
    assert info["round_budget"]["ok"] and info["round_budget"]["new_est_high"] == 2.5
    assert info["round_budget"]["baseline"] == 8.8404 and launch == run / "job" / "launch.sh"
    with pytest.raises(PT.Refused, match="prepared and not submitted"):
        PT.prepare("tvr-data", L=lay, cfg=cfg, events=[])


def test_prepare_refuses_missing_cfg_and_main_exit_2(lay, capsys, monkeypatch):
    cfg = copy.deepcopy(pilot_cfg.FROZEN)  # independent of the fill state of the real pilot_cfg
    cfg["render"]["lp_threads"] = None
    with pytest.raises(PT.Refused, match="pilot_cfg value missing for tvr-data: render.lp_threads"):
        PT.prepare("tvr-data", L=lay, cfg=cfg, events=[])
    assert not lay.calls["prepare"]
    # main(): a refusal is exit 2 with the reason on stderr (forced refusal: the real pilot_cfg may be filled)
    monkeypatch.setattr(pilot_cfg, "missing", lambda job, cfg=None: ["render.lp_threads"])
    assert PT.main(["prepare", "tvr-data"]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "render.lp_threads" in err and not lay.calls["prepare"]
    assert all(d.is_dir() for d in lay.shared_parents())  # created before any refusal


def _ledger(spent_tvr: float, open_est: float):
    ev = [{"event": "submit", "name": "vpd-tvr-probe", "run": "/r/a", "gate": "tvr", "est": 1.0, "gpus": 1,
           "pt_id": "pt-a", "submitted_at": "2026-11-10T00:00:00Z"},
          {"event": "end", "run": "/r/a", "pt_id": "pt-a", "gph": spent_tvr, "gph_source": "sco_start_to_complete"}]
    if open_est:
        ev.append({"event": "submit", "name": "vpd-wb-v2", "run": "/r/b", "gate": "wb", "est": open_est, "gpus": 1,
                   "pt_id": "pt-b", "submitted_at": "2026-11-10T01:00:00Z"})
    return ev


def test_round_budget_refusal_and_numbers(lay):
    cfg = filled_cfg()
    rb = PT.round_budget(2.5, _ledger(150.0, 40.0), cfg)
    assert rb["round_used"] == round(150.0 - 8.8404, 4) and rb["open_est"] == 40.0 and rb["ok"] is True
    with pytest.raises(PT.Refused, match="ROUND BUDGET"):
        PT.prepare("tvr-data", L=lay, cfg=cfg, events=_ledger(160.0, 49.0))  # 151.16 + 49 + 2.5 > 200
    assert not lay.calls["prepare"]
    PT.prepare("tvr-data", L=lay, cfg=cfg, events=_ledger(160.0, 40.0))  # 151.16 + 40 + 2.5 = 193.66
    info = json.loads((Path(lay.calls["prepare"][-1]["run"]) / "prepared.json").read_text())
    assert info["round_budget"]["total"] == round(160 - 8.8404 + 40 + 2.5, 4) and info["round_budget"]["cap"] == 200.0


def test_prepare_cal_needs_data_done_and_shas(lay):
    cfg = filled_cfg()
    with pytest.raises(PT.Refused, match="no tvr-data run with DONE"):
        PT.prepare("tvr-cal", L=lay, cfg=cfg, events=[])
    build_data(lay, cfg)
    PT.prepare("tvr-cal", L=lay, cfg=cfg, events=[])
    assert lay.calls["prepare"][-1]["as_root"] is True and lay.calls["prepare"][-1]["est_gph"] == 5.5
    lay.data_file("demos_all_sha256").write_text("changed\n")
    with pytest.raises(PT.Refused, match="data.demos_all_sha256"):
        PT.prepare("tvr-cal", L=lay, cfg=cfg, new=True, events=[])


def test_prepare_cal400_needs_train_400(lay):
    cfg = filled_cfg("400")
    build_data(lay, cfg)
    build_cal(lay)
    with pytest.raises(PT.Refused, match="train_400"):
        PT.prepare("tvr-cal400", L=lay, cfg=cfg, events=[])
    wj(lay.cal_decision, {"n_top": "400", "train_400": True, "futility": None})
    PT.prepare("tvr-cal400", L=lay, cfg=cfg, events=[])
    assert lay.calls["prepare"][-1]["payload"][1] == "VPD_JOB_DEADLINE_S=4920"


def test_prepare_msmoke_prereg_and_futility_guards(lay, monkeypatch):
    cfg = filled_cfg()
    build_data(lay, cfg)
    build_cal(lay)
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": None, "committed": False, "status": "", "clean": True})
    with pytest.raises(PT.Refused, match="is not committed"):
        PT.prepare("tvr-msmoke", L=lay, cfg=cfg, events=[])
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "c1", "committed": True, "status": " M x",
                                                     "clean": False})
    with pytest.raises(PT.Refused, match="uncommitted changes"):
        PT.prepare("tvr-msmoke", L=lay, cfg=cfg, events=[])
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "c1", "committed": True, "status": "", "clean": True})
    monkeypatch.setattr(PT, "check_prereg", lambda cfg=None: ["render.lp_threads: prereg None != pilot_cfg 2"])
    with pytest.raises(PT.Refused, match="block != pilot_cfg.FROZEN"):
        PT.prepare("tvr-msmoke", L=lay, cfg=cfg, events=[])
    monkeypatch.setattr(PT, "check_prereg", lambda cfg=None: [])
    wj(lay.cal_decision, {"n_top": "ALL", "train_400": False, "futility": True})
    with pytest.raises(PT.Refused, match="futility True"):
        PT.prepare("tvr-msmoke", L=lay, cfg=cfg, events=[])
    wj(lay.futility_override, {"by": "user", "decision": "continue"})
    PT.prepare("tvr-msmoke", L=lay, cfg=cfg, events=[])
    info = json.loads((Path(lay.calls["prepare"][-1]["run"]) / "prepared.json").read_text())
    assert info["prereg"]["commit"] == "c1" and lay.calls["prepare"][-1]["est_gph"] == 1.0


def test_check_prereg_uses_the_single_parser(tmp_path, monkeypatch):
    cfg = filled_cfg()
    p = tmp_path / "PREREG.md"
    p.write_text("x\n" + pilot_cfg.BLOCK_BEGIN + "\n```json\n" + json.dumps(cfg, indent=1) + "\n```\n"
                 + pilot_cfg.BLOCK_END + "\n", encoding="utf-8")
    monkeypatch.setattr(PT, "prereg_path", lambda: p)
    assert PT.check_prereg(cfg) == []
    cfg2 = filled_cfg("400")
    assert any("tiers.n_top" in d for d in PT.check_prereg(cfg2))
    # pilot_cfg.check_prereg's comparison (also with a test cfg): a type-only difference and a key on one side count
    cfg3 = filled_cfg()
    cfg3["power"]["n_sims"] = 2000.0
    cfg3["gates"]["extra"] = None
    diffs = PT.check_prereg(cfg3)
    assert any(d.startswith("power.n_sims") for d in diffs) and any(d.startswith("gates.extra") for d in diffs), diffs


def test_prepare_train_needs_smoke_pass_and_derives_est(lay):
    cfg = filled_cfg()
    build_data(lay, cfg)
    build_cal(lay)
    with pytest.raises(PT.Refused, match="no smoke_verdict.json"):
        PT.prepare("tvr-train-q3s", L=lay, cfg=cfg, events=[])
    verdict(lay, "FAIL")
    with pytest.raises(PT.Refused, match="not PASS"):
        PT.prepare("tvr-train-q3s", L=lay, cfg=cfg, events=[])
    verdict(lay, "PASS", name="tvr-msmoke-202611110100")
    PT.prepare("tvr-train-q3s", L=lay, cfg=cfg, events=[])
    want = math.ceil(round((1.3 * 3 * 660.0 / 3600 + 0.1) * 1000, 6)) / 1000
    assert lay.calls["prepare"][-1]["est_gph"] == want
    q3b = PT.derived_train("Q3B", lay, cfg)
    assert q3b["est_high"] == math.ceil(round((1.3 * 5 * 660.0 / 3600 + 0.1) * 1000, 6)) / 1000
    assert set(q3b["parts"]) == {"0W", "100", "ALL", "0W_s1002", "100_s1002"}
    assert q3b["parts"]["100_s1002"]["cal_tier"] == "100"


def test_derived_train_uses_cal400_for_n_top_400(lay):
    cfg = filled_cfg("400")
    build_cal(lay, tiers=("0W", "100", "ALL"))
    with pytest.raises(PT.Refused, match="adapters/400/meta.json"):
        PT.derived_train("E2", lay, cfg)
    wj(lay.cal_adapter("400") / "meta.json", {"train_seconds": 1800.0, "load_seconds": 60.0})
    e = PT.derived_train("E2", lay, cfg)
    assert e["parts"]["400"]["train_s"] == 1800.0


def test_derived_eval_formula_and_cap(lay):
    cfg = filled_cfg()
    build_cal(lay, steps_per_s=6.0, mean_steps=13.0)
    m = PT.cal_measurements(lay, cfg)
    assert (m["R"], m["R_blind"], m["vllm_ready_s_max"], m["merge_s"], m["parity_s"]) == (6.0, 40.0, 300.0, 95.0,
                                                                                          420.0)
    steps = min(17500, 1.5 * 500 * 13.0)
    per = 300.0 + 95.0 + 420.0
    top = (3 * per + 2 * steps / 6.0 + steps / 40.0) / 3600 + 0.15
    assert PT.derived_eval("Q3S", "top", lay, cfg)["est_high"] == math.ceil(round(top * 1000, 6)) / 1000
    low_q3b = (3 * (per + steps / 6.0)) / 3600 + 0.15
    e = PT.derived_eval("Q3B", "low", lay, cfg)
    assert e["est_high"] == math.ceil(round(low_q3b * 1000, 6)) / 1000
    assert [p["tier"] for p in e["parts"]] == ["0W", "100", "100s1002"] and e["parts"][2]["cal_tier"] == "100"
    # a slow calibration: the derived est_high exceeds 5.5 h -> prepare refuses (stop and ask)
    build_cal(lay, steps_per_s=1.0, mean_steps=40.0)
    build_data(lay, cfg)
    verdict(lay)
    for s in ("Q3S",):
        build_adapters(lay, s, cfg)
    with pytest.raises(PT.Refused, match="5.5 h job cap"):
        PT.prepare("tvr-eval-q3s-top", L=lay, cfg=cfg, events=[])


def test_prepare_eval_needs_complete_train(lay):
    cfg = filled_cfg()
    build_data(lay, cfg)
    build_cal(lay)
    verdict(lay)
    with pytest.raises(PT.Refused, match="adapter E2/0W"):
        PT.prepare("tvr-eval-e2-low", L=lay, cfg=cfg, events=[])
    build_adapters(lay, "E2", cfg)
    PT.prepare("tvr-eval-e2-low", L=lay, cfg=cfg, events=[])
    c = lay.calls["prepare"][-1]
    assert c["payload"][-3:] == ["E2", "low", c["run"]] and c["est_gph"] <= 5.5
    # 100 retrained from another 0W since: not complete any more
    (lay.adapter("E2", "0W") / "adapter.pt").write_bytes(b"new 0W")
    with pytest.raises(PT.Refused, match="retrain"):
        PT.prepare("tvr-eval-e2-top", L=lay, cfg=cfg, events=[])


def test_submit_rechecks_prereg_commit(lay, monkeypatch):
    cfg = filled_cfg()
    build_data(lay, cfg)
    build_cal(lay)
    PT.prepare("tvr-msmoke", L=lay, cfg=cfg, events=[])
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "other", "committed": True, "status": "",
                                                     "clean": True})
    with pytest.raises(PT.Refused, match="commit changed since prepare"):
        PT.submit("tvr-msmoke", L=lay, cfg=cfg, events=[])
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "c0ffee", "committed": True, "status": "",
                                                     "clean": True})
    assert PT.submit("tvr-msmoke", L=lay, cfg=cfg, events=[]) == "pt-fake1"


def test_prepare_refuses_n_top_other_than_the_calibration_decision(lay):
    """tiers.n_top (filled by hand) must be cal_decision.json's n_top (with_400 for 400): msmoke / train / eval."""
    cfg = filled_cfg("ALL")
    build_data(lay, cfg)
    build_cal(lay)
    verdict(lay)
    build_adapters(lay, "E2", cfg)
    wj(lay.cal_decision, {"n_top": "400", "train_400": True, "futility": False, "with_400": True})
    for job in ("tvr-msmoke", "tvr-train-e2", "tvr-eval-e2-low"):
        with pytest.raises(PT.Refused, match=r"tiers.n_top 'ALL' != .*cal_decision.json n_top '400'"):
            PT.prepare(job, L=lay, cfg=cfg, events=[])
    cfg4 = filled_cfg("400")
    build_data(lay, cfg4)
    wj(lay.cal_decision, {"n_top": "400", "train_400": True, "futility": None, "with_400": False})
    probs = PT.prerequisites("tvr-eval-q3s-top", lay, cfg4)
    assert any("with_400 False" in b for b in probs), probs
    lay.cal_decision.unlink()
    assert any("has no calibration decision" in b for b in PT.prerequisites("tvr-train-q3s", lay, cfg4))
    # an unfilled n_top is reported by missing() only
    unfilled = filled_cfg()
    unfilled["tiers"]["n_top"] = None
    assert PT.n_top_problems(lay, unfilled) == []
    wj(lay.cal_decision, {"n_top": "ALL", "train_400": False, "futility": False, "with_400": False})
    assert PT.n_top_problems(lay, cfg) == [] and not [b for b in PT.prerequisites("tvr-msmoke", lay, cfg)
                                                      if "n_top" in b]
    assert not lay.calls["prepare"]


def test_data_fill_prints_the_13_values(lay, capsys):
    cfg = filled_cfg()
    with pytest.raises(PT.Refused, match="has no DONE"):
        PT.data_fill(lay)
    build_data(lay, cfg)
    run = PT.done_run("tvr-data", lay)
    with pytest.raises(PT.Refused, match="expect.json is not ok"):
        PT.data_fill(lay)
    wj(run / "expect.json", {"ok": True, "failed": []})
    rep = PT.data_fill(lay)
    assert list(rep["data"]) == list(pilot_cfg.FROZEN["data"]) and rep["problems"] == []
    assert rep["data"] == cfg["data"]  # build_data filled cfg from the same files
    assert PT.LAYOUT is lay and PT.main(["data-fill"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["kept"] == 1599
    lay.data_file("bfs_dev200_sha256").unlink()
    assert PT.main(["data-fill"]) == 2


def test_guard_data_needs_the_probe_summary(lay, monkeypatch, tmp_path):
    from vpd.tvr import lvp

    cfg = filled_cfg()
    font = tmp_path / "font.ttf"
    font.write_bytes(b"font")
    cfg["tvrbench"]["files_sha256"] = {}
    cfg["render"]["badge_font_sha256"] = fsha(font)
    cfg["render"]["probe_run"] = "OUT/jobs/tvr-probe-202611101501"
    monkeypatch.setattr(lvp, "BADGE_FONT", str(font))
    monkeypatch.setattr(PT, "prerequisites", lambda job, L=None, cfg=None, in_job=False: [])
    run = lay.jobs / "tvr-data-202611111111"
    run.mkdir(parents=True)
    rep = PT.guard("tvr-data", run, lay, cfg)
    assert not rep["ok"] and any("summary.json: missing" in b for b in rep["problems"]), rep
    summ = wj(lay.jobs / "tvr-probe-202611101501" / "out" / "summary.json", {"recommended": {"E": 5, "L": 2}})
    cfg["render"]["probe_summary_sha256"] = fsha(summ)
    assert PT.guard("tvr-data", run, lay, cfg)["ok"]


# ------------------------------------------------------------------------------------------ cal-decide
def _cal_results(L, tier, k, n=200):
    wj(L.cal_eval(tier) / "results.json", {"valid": True, "n_tasks": n, "tasks_missing": [], "sr": {"k": k, "n": n}})


@pytest.mark.parametrize("k,top", [(140, "ALL"), (139, "ALL"), (141, "400"), (0, "ALL")])
def test_cal_decide_rule_at_070(lay, k, top):
    wj(lay.floors / "floors_dev200.json", {"F": {"rate": 0.0, "source": "stop0"}, "warnings": []})
    _cal_results(lay, "ALL", k)
    d = PT.cal_decide(lay.cal, lay, filled_cfg())
    assert d["n_top"] == top and d["train_400"] == (top == "400")
    if top == "ALL":
        assert d["futility"] is (PT._wilson(k, 200)[0] <= 0.0)
    else:
        assert d["futility"] is None and d["with_400"] is False


def test_cal_decide_futility_and_400_redecision(lay):
    wj(lay.floors / "floors_dev200.json", {"F": {"rate": 0.05, "source": "random"}, "warnings": []})
    _cal_results(lay, "ALL", 12)  # LB(12/200) ~ 0.035 <= 0.05 -> futility
    d = PT.cal_decide(lay.cal, lay, filled_cfg())
    assert d["n_top"] == "ALL" and d["futility"] is True
    assert PT.main(["cal-decide", "--cal-root", str(lay.cal), "--floors", str(lay.floors / "floors_dev200.json")]) == 0
    lay.cal_decision.unlink()
    _cal_results(lay, "ALL", 150)
    assert PT.cal_decide(lay.cal, lay, filled_cfg())["futility"] is None
    _cal_results(lay, "400", 120)
    d = PT.cal_decide(lay.cal, lay, filled_cfg())
    assert d["n_top"] == "400" and d["with_400"] is True and d["futility"] is False and d["top_dev"]["k"] == 120
    _cal_results(lay, "ALL", 100)  # n_top would flip to ALL: refused
    with pytest.raises(PT.Refused, match="refusing to change"):
        PT.cal_decide(lay.cal, lay, filled_cfg())


# ------------------------------------------------------------------------------------------ smoke verdict
def renderer_block(cfg: dict, envs: int = 5, n_workers: int = 2) -> dict:
    """run_tvr's results.renderer (Harness._renderer): the harness's request plus each worker's ready reply."""
    font = cfg["render"]["badge_font_sha256"]
    return {"profile": "envlib", "font_sha256": font, "lp": 2, "envs": envs, "n_workers_live": n_workers,
            "workers_ok": True,
            "workers": [{"wid": w, "profile": "envlib", "lp": "2", "font_mode": "pinned", "font_sha256": font,
                         "pid": 100 + w, "system_prompt_ok": True, "tvrbench_files_ok": True}
                        for w in range(n_workers)]}


def build_smoke(sm: Path, cfg: dict) -> None:
    for s in pilot_cfg.STARTS:
        wj(sm / s / "dry" / "dry_run.json", {"per_conversation": [
            {"id": f"d{i}", "grid_set": [[1, 22, 40]], "images": 2 * (i + 2), "image_tokens": 440 * (i + 2)}
            for i in range(3)]})
        wj(sm / s / "dry" / "parity.json", {"verdict": "PASS", "problems": {}})
        ev = sm / s / "evals"
        wj(ev / f"smoke__{s}__raw__vllm_parity.json", {"equal": True, "n": 12})
        for a in ("c", "u", "blind"):
            r = {"valid": True, "n_tasks": 8, "tasks_missing": [], "withheld": "mechanism smoke",
                 "invalid": {"regex_mismatch": 0 if a != "u" else None, "replies": 90, "unparseable_steps": 3},
                 "perf": {"steps_per_s": 5.0}}
            if a != "blind":
                r["audit"] = {"prompt_parity": {"equal": True},
                              "image_tokens": {"tokens_per_image": [222], "grid_set": [[1, 22, 40]], "n": 30}}
                r["renderer"] = renderer_block(cfg)
            wj(ev / f"smoke__{s}__raw__{a}" / "results.json", r)
            for i in range(8):
                wj(ev / f"smoke__{s}__raw__{a}" / "tasks" / f"t{i}" / "trajectory.json",
                   {"task_id": f"t{i}", "result": {"success": None, "stopped": True}})
    wj(sm / "replay2" / "render_parity.json", {"pass": True, "n_frames": 27, "n_pass": 27, "min_psnr_db": 45.0})


def test_smoke_verdict_pass_and_each_failure(lay):
    cfg = filled_cfg()
    sm = lay.smoke_root / "tvr-msmoke-202611110000"
    build_smoke(sm, cfg)
    v = PT.smoke_verdict(sm, lay, cfg)
    assert v["verdict"] == "PASS", v["failed"]
    assert json.loads((sm / "smoke_verdict.json").read_text())["verdict"] == "PASS"
    assert v["checks"][[c["name"] for c in v["checks"]].index("dry_Q3B")]["detail"]["tokens_per_image"] == 222
    breaks = [
        (sm / "Q3S" / "dry" / "parity.json", lambda r: r.update(verdict="FAIL"), "hfparity_Q3S"),
        (sm / "E2" / "evals" / "smoke__E2__raw__vllm_parity.json", lambda r: r.update(equal=False), "vllm_parity_E2"),
        (sm / "Q3B" / "evals" / "smoke__Q3B__raw__blind" / "results.json",
         lambda r: r["invalid"].update(regex_mismatch=1), "regex_Q3B"),
        (sm / "Q3B" / "evals" / "smoke__Q3B__raw__u" / "results.json", lambda r: r.update(valid=False), "runs_Q3B"),
        (sm / "Q3S" / "evals" / "smoke__Q3S__raw__c" / "results.json",
         lambda r: r["audit"]["image_tokens"].update(tokens_per_image=[220]), "audit_Q3S"),
        (sm / "E2" / "evals" / "smoke__E2__raw__u" / "results.json", lambda r: r["renderer"].update(lp=4),
         "renderer_E2"),
        # the workers' own ready replies (not the harness's request): workers_ok, the font, the pinned files / prompt
        (sm / "Q3S" / "evals" / "smoke__Q3S__raw__c" / "results.json", lambda r: r["renderer"].update(workers_ok=False),
         "renderer_Q3S"),
        (sm / "Q3B" / "evals" / "smoke__Q3B__raw__u" / "results.json",
         lambda r: r["renderer"]["workers"][1].update(font_mode="system"), "renderer_Q3B"),
        (sm / "E2" / "evals" / "smoke__E2__raw__c" / "results.json",
         lambda r: r["renderer"]["workers"][0].update(tvrbench_files_ok=False), "renderer_E2"),
        (sm / "E2" / "evals" / "smoke__E2__raw__c" / "results.json", lambda r: r["renderer"].update(workers=[]),
         "renderer_E2"),
        (sm / "E2" / "dry" / "dry_run.json", lambda r: r["per_conversation"][0].update(grid_set=[[1, 24, 40]]),
         "dry_E2"),
        (sm / "replay2" / "render_parity.json", lambda r: r.update(**{"pass": False}), "render_parity"),
    ]
    for path, fn, name in breaks:
        orig = path.read_text()
        r = json.loads(orig)
        fn(r)
        path.write_text(json.dumps(r))
        v = PT.smoke_verdict(sm, lay, cfg)
        assert v["verdict"] == "FAIL" and v["failed"] == [name], (name, v["failed"])
        path.write_text(orig)


def test_smoke_verdict_never_reads_a_success_field(lay):
    cfg = filled_cfg()
    sm = lay.smoke_root / "tvr-msmoke-202611110001"
    build_smoke(sm, cfg)
    base = PT.smoke_verdict(sm, lay, cfg)
    # poison every success-like field: an SR block, per-task successes, a per_task.jsonl; the verdict must not move
    for p in sm.rglob("results.json"):
        r = json.loads(p.read_text())
        r.update(sr={"k": -1, "n": 0, "rate": 9.9}, by_category={"SR-easy": {"sr": "garbage"}}, success=False)
        p.write_text(json.dumps(r))
    for p in sm.rglob("trajectory.json"):
        p.write_text(json.dumps({"task_id": "x", "result": {"success": "NOT A BOOL", "stopped": "?"}}))
    for d in sm.rglob("smoke__*__raw__c"):
        (d / "per_task.jsonl").write_text("not json at all\n")
    again = PT.smoke_verdict(sm, lay, cfg)
    assert again["verdict"] == base["verdict"] == "PASS"
    assert [c["detail"] for c in again["checks"]] == [c["detail"] for c in base["checks"]]
    text = (sm / "smoke_verdict.json").read_text()
    assert '"sr"' not in text and '"success"' not in text


# ------------------------------------------------------------------------------------------ expect / steps / parity-ok
def test_steps_check_rules(tmp_path):
    run = tmp_path / "run"
    req = PT.required_steps("tvr-cal")
    lines = [{"name": n, "rc": 0} for n in req]
    run.mkdir()
    (run / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert PT.steps_check("tvr-cal", run)["ok"]
    lines[4] = {"name": "shake", "rc": 1}
    lines.insert(5, {"name": "shake_fallback", "rc": 0})  # the O9 retry excuses the first attempt
    (run / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert PT.steps_check("tvr-cal", run)["ok"]
    lines[8] = {"name": "train_0W:skip", "rc": 0}
    lines.append({"name": "eval_100:no_time", "rc": 1})
    (run / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    st = PT.steps_check("tvr-cal", run)
    assert not st["ok"] and st["failed"] == ["eval_100:no_time"] and st["absent"] == []
    (run / "steps.jsonl").write_text(json.dumps({"name": "guard", "rc": 0}) + "\n")
    assert "merge_disk" in PT.steps_check("tvr-cal", run)["absent"]


def test_required_steps_per_job():
    cfg = filled_cfg()
    assert PT.required_steps("tvr-train-q3b", cfg)[-2:] == ["train_0W_s1002", "train_100_s1002"]
    assert PT.required_steps("tvr-train-e2", cfg)[-1] == "train_ALL"
    top = PT.required_steps("tvr-eval-q3s-top", cfg)
    assert top[-6:] == ["merge_ALL", "merge_parity_ALL", "eval_ALL", "official_ALL_c", "official_ALL_u",
                        "official_ALL_blind"]
    low = PT.required_steps("tvr-eval-q3b-low", filled_cfg("400"))
    assert "merge_parity_100s1002" in low and low[-1] == "official_100s1002_c" and "eval_400" not in low


def test_parity_state(tmp_path):
    ad = tmp_path / "ad"
    ad.mkdir()
    (ad / "adapter.pt").write_bytes(b"A")
    p = tmp_path / "merge_ALL.json"
    assert PT.parity_state(p, ad)[0] == 1
    wj(p, {"pass": True, "adapter_sha256": sha(b"A")})
    assert PT.parity_state(p, ad)[0] == 0 and PT.main(["parity-ok", str(p), "--adapter", str(ad)]) == 0
    wj(p, {"pass": False, "adapter_sha256": sha(b"A")})
    assert PT.parity_state(p, ad)[0] == 4
    (ad / "adapter.pt").write_bytes(b"B")
    assert PT.parity_state(p, ad)[0] == 1


def test_expect_train_and_eval(lay, tmp_path):
    cfg = filled_cfg()
    run = lay.jobs / "tvr-train-e2-202611111200"
    run.mkdir(parents=True)
    build_adapters(lay, "E2", cfg)
    (run / "steps.jsonl").write_text("".join(json.dumps({"name": n, "rc": 0}) + "\n"
                                             for n in PT.required_steps("tvr-train-e2", cfg)))
    rep = PT.expect("tvr-train-e2", run, lay, cfg)
    assert rep["ok"], rep["failed"]
    assert json.loads((run / "expect.json").read_text())["ok"] is True
    (lay.adapter("E2", "0W") / "adapter.pt").write_bytes(b"retrained")  # 100 / ALL now from another 0W
    assert set(PT.expect("tvr-train-e2", run, lay, cfg)["failed"]) == {"adapter_100", "adapter_ALL"}


# ------------------------------------------------------------------------------------------ merge_parity logic
def test_merge_parity_reply_rows_and_compare():
    # logits at positions 3, 4 predict the reply tokens 4, 5 (span (4, 6)); positions 7, 8 -> span (8, 10)
    rows = MP.reply_rows([0.5, 0.7, 1.0, 2.0], [11, 12, 13, 14], [3, 4, 7, 8], [(4, 6), (8, 10)])
    assert rows == [{"ce": 0.6, "argmax": [11, 12]}, {"ce": 1.5, "argmax": [13, 14]}]
    with pytest.raises(ValueError, match="positions scored"):
        MP.reply_rows([0.5], [11], [3], [(4, 6)])
    u = [rows, [{"ce": 0.1, "argmax": [1]}]]
    same = MP.compare(u, copy.deepcopy(u))
    assert same["pass"] and same["mean_abs_dce"] == 0.0 and same["reply_agreement"] == 1.0 and same["n_replies"] == 3
    m = copy.deepcopy(u)
    m[0][0]["ce"] += 0.012  # mean |dCE| = 0.004 <= 0.005
    r = MP.compare(u, m)
    assert r["pass"] and abs(r["mean_abs_dce"] - 0.004) < 1e-12 and abs(r["max_abs_dce"] - 0.012) < 1e-12
    m[0][1]["ce"] += 0.01  # mean 0.0073 > 0.005
    assert not MP.compare(u, m)["pass"]
    m = copy.deepcopy(u)
    m[1][0]["argmax"] = [2]  # 2 of 3 replies agree < 0.99
    r = MP.compare(u, m)
    assert not r["pass"] and abs(r["reply_agreement"] - 2 / 3) < 1e-12
    with pytest.raises(ValueError):
        MP.compare(u, u[:1])


def test_merge_parity_rows_on_tiny_tensors():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    labels = torch.tensor([0, 1, 1])
    ce = F.cross_entropy(logits, labels, reduction="none")
    rows = MP.reply_rows(ce.tolist(), logits.argmax(-1).tolist(), [0, 1, 4], [(1, 3), (5, 6)])
    assert rows[0]["argmax"] == [0, 1] and rows[1]["argmax"] == [2]
    assert abs(rows[0]["ce"] - float(ce[:2].mean())) < 1e-6


# ------------------------------------------------------------------------------------------ recheck_tvr
def _pos(x, z):
    return {"x": x, "y": 0.9, "z": z}


def _task(tid, scene, ds, diff, start, target):
    def pose(t):
        return {"position": _pos(t[0], t[1]), "rotation_y": float(t[2]), "horizon": float(t[3])}

    return {"task_id": tid, "dataset": ds, "scene": scene, "difficulty": diff, "seg_count": 1, "bfs_distance": 1,
            "start": pose(start), "target": pose(target)}


GRID = [_pos(round(-1.0 + 0.25 * i, 2), round(-1.0 + 0.25 * j, 2)) for i in range(9) for j in range(9)
        if (i, j) != (5, 5)]


def build_fake_tvr(root: Path, n_eval_per_cat: int = 2):
    """reach/v1 (index + scene files), data/tasks/{sft,eval}.json and a public sft_train.json for recheck floors."""
    sft, ev = [], []
    for ci, (ds, diff) in enumerate((("ithor", "easy"), ("ithor", "hard"), ("procthor", "easy"),
                                     ("procthor", "hard"))):
        scene = "FloorPlan2" if ds == "ithor" else "procthor_00002"
        sft.append(_task(f"{ds}_{scene}_{diff}_t00_s00", scene, ds, diff, (0.0, 0.0, 0, 0), (0.0, 0.5, 0, 0)))
        sft.append(_task(f"{ds}_{scene}_{diff}_t01_s00", scene, ds, diff, (0.0, 0.0, 90, 0), (0.25, 0.0, 135, 30)))
        for k in range(n_eval_per_cat):
            escene = f"FloorPlan{10 + k}" if ds == "ithor" else f"procthor_{k + 1:05d}"
            ev.append(_task(f"{ds}_{escene}_{diff}_t0{k}_s0{ci}", escene, ds, diff, (0.0, 0.0, 0, 0),
                            (0.25 * (k + 1), -0.5, 270, 30)))
    tasks_dir = root / "repo" / "data" / "tasks"
    wj(tasks_dir / "sft.json", sft)
    wj(tasks_dir / "eval.json", ev)
    dg = RC.digest(GRID)
    reach = root / "out" / "tvr" / "reach" / "v1"
    index = {"tasks": {}, "unstable_tasks": []}
    for t in sft + ev:
        index["tasks"][t["task_id"]] = {"scene": t["scene"], "digest": dg, "n": len(GRID), "stable": True}
        wj(reach / "scenes" / f"{t['scene']}.json", {"scene": t["scene"], "sets": {dg: GRID}})
    wj(reach / "index.json", index)
    public = []
    for t in sft:
        acts = ["MoveAhead", "MoveAhead"] if t["task_id"].endswith("t00_s00") else [
            "MoveRight", "RotateRight", "LookDown"]
        conv = []
        for a in acts + ["Stop"]:
            conv += [{"from": "human", "value": "<image><image>"}, {"from": "gpt", "value": f"Action: {a}"}]
        public.append({"system": "s", "conversations": conv,
                       "images": [f"images/{t['task_id']}/step_{i:02d}.png" for i in range(2 * len(conv))]})
    pub = root / "out" / "tvr" / "hf_meta" / "tvr-sft-va" / "sft_train.json"
    wj(pub, public)
    return {"tasks_dir": tasks_dir, "reach": reach, "public": pub, "eval": ev, "sft": sft, "digest": dg}


def test_recheck_transitions_by_hand():
    reach = RC.cells(GRID)
    p = (0.0, 0.0, 45, 0)
    assert RC.transition((0.0, 0.0, 0, 0), "MoveAhead", reach) == (0.0, 0.25, 0, 0)
    assert RC.transition(p, "MoveAhead", reach) == p  # the diagonal cell (0.25, 0.25) is blocked
    assert RC.transition(p, "MoveLeft", reach) == (-0.25, 0.25, 45, 0)
    assert RC.transition((0.0, 0.0, 0, -30), "LookUp", reach) == (0.0, 0.0, 0, -30)
    assert RC.transition((0.0, 0.0, 0, 0), "RotateLeft", reach) == (0.0, 0.0, 315, 0)
    assert RC.transition((0.0, 0.25, 0, 0), "MoveRight", reach) == (0.0, 0.25, 0, 0)  # blocked
    assert RC.transition((0.0, 0.25, 90, 0), "MoveAhead", reach) == (0.0, 0.25, 90, 0)  # (0.25, 0.25) blocked
    assert RC.valid_list((0.0, 0.25, 90, 30), reach) == ["MoveBack", "MoveLeft", "MoveRight", "RotateRight",
                                                         "RotateLeft", "LookUp", "Stop"]
    assert RC.pose({"position": {"x": -0.0, "y": 1, "z": 0.249999}, "rotation_y": 359.6, "horizon": -0.2}) == \
        (0.0, 0.25, 0, 0)
    assert RC.exact((0.0, 0.0, 0, 0), (0.005, 0.0, 0, 0)) and not RC.exact((0.0, 0.0, 0, 0), (0.0, 0.0, 45, 0))


def test_recheck_floors_and_imports_in_a_fresh_process(tmp_path):
    f = build_fake_tvr(tmp_path)
    rep = RC.floors(f["reach"], f["tasks_dir"], f["public"])
    assert rep["stop0"] == {"k": 0, "n": 8, "rate": 0.0}
    # MoveAhead x2 + Stop solves the 4 t00 sft tasks; nothing single-action solves the t01 ones
    assert rep["fixed"]["chosen"] == {"action": "MoveAhead", "k": 2} and rep["fixed"]["sft"]["k"] == 4
    assert rep["random"]["medians"] == {c: 2 for c in CATS} and len(rep["random"]["k"]) == 3
    assert rep["F"]["source"] in ("stop0", "fixed", "random") and RC.compare_floors(rep, json.loads(json.dumps(rep))) == []
    bad = copy.deepcopy(rep)
    bad["random"]["k"][0] += 1
    assert RC.compare_floors(rep, bad)
    code = ("import json, sys; from vpd.tvr import recheck_tvr as R; "
            f"R.main(['floors', '--reach', {str(f['reach'])!r}, '--tasks-dir', {str(f['tasks_dir'])!r}, "
            f"'--public', {str(f['public'])!r}, '--out', {str(tmp_path / 'F.json')!r}]); "
            "print(json.dumps(sorted(m for m in R.FORBIDDEN_IMPORTS if m in sys.modules)))")
    res = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(paths.WORKTREE)), cwd=str(paths.WORKTREE), timeout=120)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1]) == []
    out = json.loads((tmp_path / "F.json").read_text())
    assert out["F"] == rep["F"] and out["fixed"]["chosen"] == rep["fixed"]["chosen"]
    src = (paths.VPD_PKG / "tvr" / "recheck_tvr.py").read_text()
    for mod in ("kin_env", "floors", "bfs", "tasks", "score_tvr"):
        assert not re.search(rf"^\s*(from vpd\.tvr import .*\b{mod}\b|import vpd\.tvr\.{mod}|from vpd\.tvr\.{mod} )",
                             src, re.M), mod


FLOOR_CASES = [(0, 3, [3, 3, 3], 500), (0, 3, [0, 1, 8], 500), (0, 0, [0, 0, 0], 500), (2, 2, [2, 2, 2], 500),
               (0, 3, [3, 3, 4], 500), (0, 4, [3, 3, 3], 500), (1, 0, [1, 1, 1], 500), (0, 7, [7, 7, 7], 300),
               (0, 1, [1, 1, 1], 7)]


def _floors_py_choice(monkeypatch, k0, fe, ks, n):
    """floors.py's own F / source (floors_doc, as of 11f6d852) on given component counts: its component functions
    are stubbed, its max / tie code runs as written."""
    from fractions import Fraction

    from vpd.tvr import floors as FLR

    monkeypatch.setattr(FLR, "stop0", lambda tl: {"k": k0, "n": n, "rate": k0 / n})
    monkeypatch.setattr(FLR, "fixed_table", lambda tl, ri: {"MoveAhead:1": 1})
    monkeypatch.setattr(FLR, "choose_fixed", lambda tab: ("MoveAhead", 1, 1))
    monkeypatch.setattr(FLR, "fixed_score", lambda tl, ri, a, k: fe)
    monkeypatch.setattr(FLR, "random_floor", lambda tl, ri, med: {"k": list(ks), "mean_rate": sum(ks) / (3 * n),
                                                                   "_frac": Fraction(sum(ks), 3 * n)})
    task = {"task_id": "t", "start": {"position": _pos(0.0, 0.0), "rotation_y": 0.0, "horizon": 0.0},
            "target": {"position": _pos(0.0, 0.25), "rotation_y": 0.0, "horizon": 0.0}}

    class RI:
        sha256 = "r" * 64

        @staticmethod
        def reach(tid):
            return frozenset({(0.0, 0.0)})

    doc = FLR.floors_doc("eval500", [task] * n, [task], RI(), {})
    return doc["F"]["source"], doc["F"]["rate"]


@pytest.mark.parametrize("k0,fe,ks,n", FLOOR_CASES)
def test_recheck_floor_source_is_exact_like_floors_py(monkeypatch, k0, fe, ks, n):
    """F.source by exact fractions with floors.py's tie rule (first of stop0, fixed, random): a random mean that ties
    the fixed rate is 'fixed' (a float mean of the seed rates can be one ulp above it)."""
    src, F = RC.choose_floor(k0, fe, ks, n)
    assert (src, float(F)) == _floors_py_choice(monkeypatch, k0, fe, ks, n)
    if (k0, fe, ks, n) == (0, 3, [3, 3, 3], 500):
        assert src == "fixed" and sum(k / n for k in ks) / 3 > fe / n  # the float rule the recheck used before
    if (k0, fe, ks, n) == (0, 3, [0, 1, 8], 500):
        assert src == "fixed" and sum(k / n for k in ks) / 3 > fe / n


def test_recheck_floors_uses_the_exact_choice(tmp_path, monkeypatch):
    """RC.floors routes its components through choose_floor and writes floors.py's float fields (F.rate, mean_rate)."""
    f = build_fake_tvr(tmp_path)
    seen = []
    real = RC.choose_floor
    monkeypatch.setattr(RC, "choose_floor", lambda *a: seen.append(a) or real(*a))
    rep = RC.floors(f["reach"], f["tasks_dir"], f["public"])
    (k0, fe, ks, n), = seen
    assert (k0, fe, list(ks), n) == (rep["stop0"]["k"], rep["fixed"]["eval"]["k"], rep["random"]["k"], rep["n"])
    src, F = real(k0, fe, ks, n)
    assert rep["F"] == {"rate": float(F), "source": src, "k_equiv": math.ceil(F * n)}
    assert rep["random"]["mean_rate"] == sum(ks) / (len(ks) * n)


def test_recheck_counts(tmp_path):
    run = tmp_path / "run"
    for i, (ds, diff, ok) in enumerate((("ithor", "easy", True), ("ithor", "hard", False), ("procthor", "hard", True))):
        wj(run / "tasks" / f"t{i}" / "trajectory.json", {"task_id": f"t{i}", "dataset": ds, "difficulty": diff,
                                                          "result": {"success": ok, "stopped": True}})
    c = RC.counts(run)
    assert (c["k"], c["n"]) == (2, 3) and c["by_category"]["LR-hard"] == {"k": 1, "n": 1}
    assert c["by_category"]["LR-easy"] == {"k": 0, "n": 0} and c["success"] == {"t0": True, "t1": False, "t2": True}


# ------------------------------------------------------------------------------------------ gate_tvr on fake pilot roots
N_EVAL = 8
SUCC = {"0W__c": 0, "100__c": 3, "ALL__c": 6, "ALL__u": 5, "ALL__blind": 0, "100s1002__c": 3}


REQUEST_KEYS = ["max_tokens", "messages", "model", "temperature", "top_k", "top_p"]


def _settings(proto: dict, arm: str, mml: int = 32768) -> dict:
    """run_tvr's results.settings (Harness.settings): the protocol block plus the values this run used."""
    rx = arm in ("c", "blind")
    return dict(proto, arm=arm, max_tokens_used={"c": 16, "u": 64, "blind": 16}[arm], max_model_len_used=mml,
                regex_used=pilot_cfg.REGEX if rx else None, blind_concurrency_used=16 if arm == "blind" else None,
                envs=None if arm == "blind" else 5, lp=None if arm == "blind" else 2,
                request_fields=sorted(REQUEST_KEYS + (["structured_outputs"] if rx else [])))


def _results(cfg, s, key, n, k_by_cat, adapter_sha):
    tier, arm = key.split("__")
    proto = json.loads(json.dumps(cfg["protocol"]))
    r = {"schema": "vpd.tvr.results/1", "start": s, "tier": tier, "arm": arm, "task_set": "eval500", "n_tasks": n,
         "valid": True, "tasks_missing": [], "sr": {"k": sum(k_by_cat.values()), "n": n},
         "by_category": {c: {"sr": {"k": k_by_cat[c], "n": n // 4}} for c in CATS},
         "invalid": {"regex_mismatch": 0 if arm != "u" else None, "replies": 40, "unparseable_steps": 0,
                     "blocked_steps": 2},
         "kin": {"tasks_equal": n, "tasks_unequal": 0, "valid_list_mismatches": 0, "pose_or_success_mismatches": 0},
         "settings": _settings(proto, arm),
         # run_tvr's layout: server = {manifest: VLLMServer.manifest(), versions, max_model_len, ready_s, ...}
         "server": {"manifest": {"cfg": {"enforce_eager": True, "seed": 0, "max_images": 96, "max_model_len": 32768},
                                 "cmd": ["python", "-m", "vllm.entrypoints.openai.api_server", "--generation-config",
                                         "vllm", "--enforce-eager"],
                                 "extra_args": ["--structured-outputs-config", "{\"backend\": \"xgrammar\"}"],
                                 "mm_processor_kwargs": {"max_pixels": 1024000,
                                                         "size": {"longest_edge": 1024000, "shortest_edge": 65536}}},
                    "versions": {"vllm": "0.11.0", "xgrammar": "0.1.25", "transformers": "4.57.6", "torch": "2.8.0",
                                 "ai2thor": "5.0.0"}, "max_model_len": 32768, "ready_s": 300.0,
                    "served_model_name": f"code-{s}"},
         "adapter": {"adapter_sha256": adapter_sha}, "code": f"code-{s}", "code_sha256": "h" * 64,
         "code_fingerprint": {"code_sha256": "h" * 64, "files": {}},
         "reach": {"live_equal": n}, "spl": 0.1, "within1": {"k": 1, "n": n},
         "stop": {"stop_rate": 0.9, "fstop_given_stop": 0.5}}
    if arm != "blind":
        r["audit"] = {"prompt_parity": {"equal": True}, "image_tokens": {"tokens_per_image": [222],
                                                                          "grid_set": [[1, 22, 40]]}}
        r["renderer"] = renderer_block(cfg)
    return r


def write_power(L: PT.Layout, cfg: dict, ok: bool = True) -> Path:
    """A POWER.json that is current for this pilot root and made at the registered settings, in power_sim.run_tvr's
    layout (the inputs it records: per_task shas, pilot_cfg sha; registered, sigma, persistence, verdicts); the
    real writer runs in test_a4_reads_the_power_json_power_sim_writes and the integration pipeline."""
    files = {str(p): fsha(p) for p in sorted(L.pilot_root.glob("*/evals/pilot__*__c/per_task.jsonl"))
             if p.parent.name.split("__")[2] in ("100", pilot_cfg.n_top(cfg), "100s1002")}
    p = json.loads(json.dumps(cfg["power"]))
    return wj(L.pilot_root / "POWER.json", {
        "schema": "vpd.power_sim.tvr/3", "pass": ok, "fallback_pass": False,
        "registered": {"test": p["test"], "tier_corr": p["tier_corr"], "fallback": p["fallback"],
                       "persistence": p["persistence"], "n_sims_registered": p["n_sims"], "n_sims_used": p["n_sims"],
                       "seed": p["seed"], "backend": p["backend"], "allow_no_replicate": False},
        "sigma": {"sigma_hat": 0.01, "sigma_used": 0.03, "replicate_missing": False},
        # /3 layout: lambda_used = clip(lambda_hat + margin, 0, 1); no level / V / B / R / D (PERSISTENCE_KEYS)
        "persistence": {"lambda_hat": -0.03, "margin": 0.03, "lambda_used": 0.0, "replicate_missing": False},
        "verdicts": {"main": {"type1": 0.05, "type1_ok": True, "power_ok": ok, "power_tier_corr": 0.5, "pass": ok}},
        "inputs": {"files_sha256": files, "pilot_cfg_json_sha256": sha(pilot_cfg.as_json(cfg).encode())}})


def make_gate_root(tmp_path, monkeypatch, per_task: bool = True):
    """A complete, consistent fake pilot (3 starts, N_top ALL, 8 eval tasks) in run_tvr's results.json layout: every
    gate passes.  Each eval also gets its per_task.jsonl (6.4 columns; power_sim's input, POWER.json's shas)."""
    monkeypatch.setattr(PT, "EVAL500_N", N_EVAL)
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "c0ffee", "committed": True, "status": "",
                                                     "clean": True})
    f = build_fake_tvr(tmp_path)
    L = PT.Layout(tmp_path / "out")
    cfg = filled_cfg()
    # tiers manifest, floors (recheck's own numbers), demos, replay
    tman = {"warmup0w20.jsonl": "a" * 64, "demos_100.jsonl": "b" * 64, "demos_400.jsonl": "c" * 64,
            "demos_ALL.jsonl": "e" * 64, "nested": True}
    wj(L.tiers / "manifest.json", tman)
    cfg["data"]["tiers_manifest_sha256"] = fsha(L.tiers / "manifest.json")
    fl = RC.floors(f["reach"], f["tasks_dir"], f["public"])
    fl.pop("recheck")
    wj(L.floors / "floors_eval500.json", fl)
    cfg["data"].update(floors_eval500_sha256=fsha(L.floors / "floors_eval500.json"), F_eval500=fl["F"]["rate"],
                       F_source=fl["F"]["source"])
    wj(L.demos / "manifest.json", {"kept": 1599, "dropped": 1, "kin_equal_all_kept": True})
    wj(L.demos / "dropped.json", [{"task_id": "x", "reason": "move_failed@3", "detail": "blocked"}])
    wj(L.replay / "results.json", {"valid": True, "n_tasks": 1600, "tasks_missing": [], "code_sha256": "o" * 64,
                                   "renderer": renderer_block(cfg, n_workers=5)})
    wj(L.cal_decision, {"n_top": "ALL", "train_400": False, "futility": False, "with_400": False})
    sv = wj(L.smoke_root / "tvr-msmoke-202611110000" / "smoke_verdict.json", {"verdict": "PASS", "checks": [
        {"name": f"{n}_{s}", "ok": True, "detail": None} for s in pilot_cfg.STARTS
        for n in ("dry", "hfparity", "vllm_parity")]})
    wj(sv.parent / "replay2" / "render_parity.json", {"pass": True, "min_psnr_db": 45.0, "min_ssim": 0.995})
    for s in pilot_cfg.STARTS:
        build_adapters(L, s, cfg, tman)
        for part in ("top", "low"):
            wj(L.canary(s, part), {"pass": True, "min_psnr_db": 44.0, "min_ssim": 0.993, "n_frames": 27})
        for tier, arm in G.eval_list(s, cfg):
            key = f"{tier}__{arm}"
            ad = L.adapter(s, PT.EVAL_ADAPTER[tier])
            wj(L.parity(s, tier), {"pass": True, "adapter_sha256": fsha(ad / "adapter.pt")})
            d = L.eval_dir(s, tier, arm)
            k = SUCC[key]
            k_by = {c: 0 for c in CATS}
            for i, t in enumerate(f["eval"]):
                ok = i < k
                k_by[pilot_cfg.category_of(t)] += int(ok)
                wj(d / "tasks" / t["task_id"] / "trajectory.json",
                   {"task_id": t["task_id"], "dataset": t["dataset"], "scene": t["scene"],
                    "difficulty": t["difficulty"], "result": {"success": ok, "stopped": True},
                    "vpd": {"reach_digest": f["digest"]}})
            wj(d / "results.json", _results(cfg, s, key, N_EVAL, k_by, fsha(ad / "adapter.pt")))
            wj(d / "official_check.json", {"equal": True, "official_sr": k / N_EVAL, "our_sr": k / N_EVAL})
            if per_task:
                (d / "per_task.jsonl").write_text("".join(json.dumps(
                    {"task_id": t["task_id"], "category": pilot_cfg.category_of(t), "scene": t["scene"],
                     "success": int(i < k), "stopped": True}) + "\n" for i, t in enumerate(f["eval"])))
    write_power(L, cfg, True)
    rargs = {"tasks_dir": f["tasks_dir"], "public_path": f["public"]}
    return L, cfg, sv, rargs


@pytest.fixture
def gate_root(tmp_path, monkeypatch):
    return make_gate_root(tmp_path, monkeypatch)


def run_gate(gate_root, **kw):
    L, cfg, sv, rargs = gate_root
    return G.evaluate(L.pilot_root, L=L, cfg=cfg, smoke_verdict_path=sv, recheck_args=rargs, **kw)


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_gate_all_pass_files_and_absolute_only(gate_root, monkeypatch):
    rep = run_gate(gate_root)
    bad = [i for i in rep["G0"]["items"] if i["ok"] is not True]
    assert not bad, bad
    assert [i["name"] for i in rep["G0"]["items"]] == list(G.G0_ITEMS) and len(G.G0_ITEMS) == 17
    assert (rep["G1"]["status"], rep["G2"]["status"], rep["G3"]["status"], rep["A4"]["status"]) == \
        ("pass", "pass", "pass", "pass")
    assert rep["decision"] == "ruler" and rep["recheck"]["floors_equal"] is True and rep["recheck"]["counts_equal"]
    assert rep["A3"]["per_start"]["Q3S"] == {"b": 6, "c": 0, "n_paired": 8, "p": 0.03125,
                                             "images_better_p_lt_0.05": True}
    assert rep["G1"]["per_start"]["E2"]["k"] == 6 and rep["G1"]["per_start"]["E2"]["zero_w_rate"] == 0.0
    assert rep["starts"]["Q3B"]["evals"]["100s1002__c"]["sr"]["k"] == 3
    keys = set(_keys(rep))
    assert not [k for k in keys if re.search(r"diff|delta|minus|gap|vs_", str(k), re.I)], keys
    assert rep["note"] == "no between-start differences are computed"
    md = G.render_md(rep)
    assert md.startswith("# TVR 试点闸门结果") and "不算起点之间的差" in md and "| Q3S | ALL | c | 是 | 6/8" in md
    # the CLI writes GATE.json / GATE.md and exits 0
    L, cfg, sv, rargs = gate_root
    real = G.evaluate
    monkeypatch.setattr(G, "evaluate", lambda root, starts, smoke_verdict_path=None, power=False: real(
        root, starts, L=L, cfg=cfg, smoke_verdict_path=sv, recheck_args=rargs))
    assert G.main(["--pilot-root", str(L.pilot_root)]) == 0
    assert json.loads((L.pilot_root / "GATE.json").read_text())["decision"] == "ruler"
    assert (L.pilot_root / "GATE.md").read_text(encoding="utf-8").startswith("# TVR")


@pytest.mark.parametrize("b,c,p", [(6, 0, 0.03125), (8, 1, 0.0390625), (0, 0, 1.0), (3, 3, 1.0), (1, 0, 1.0)])
def test_a3_exact_binomial(b, c, p):
    assert G.binom_two_sided(b, c) == p


def _edit(path: Path, fn):
    r = json.loads(path.read_text())
    fn(r)
    path.write_text(json.dumps(r))


def _item(rep, name):
    return next(i for i in rep["G0"]["items"] if i["name"] == name)


def _only_fail(rep, name):
    failed = [i["name"] for i in rep["G0"]["items"] if i["ok"] is False]
    assert failed == [name], failed
    assert rep["decision"] == "engineering"


def test_g0_13_adapter_provenance(gate_root):
    L, cfg, sv, _ = gate_root
    _edit(L.adapter("Q3S", "100") / "meta.json", lambda m: m.update(seed=7))
    _only_fail(run_gate(gate_root), "adapter_provenance")
    _edit(L.adapter("Q3S", "100") / "meta.json", lambda m: m.update(seed=1001, data_sha256="z" * 64))
    _only_fail(run_gate(gate_root), "adapter_provenance")
    _edit(L.adapter("Q3S", "100") / "meta.json", lambda m: m.update(data_sha256="b" * 64))
    _edit(L.adapter("Q3B", "100_s1002") / "meta.json", lambda m: m.update(phase="tier100"))
    _only_fail(run_gate(gate_root), "adapter_provenance")


def test_g0_14_floors_file(gate_root):
    L, cfg, sv, _ = gate_root
    cfg["data"]["floors_eval500_sha256"] = "1" * 64
    _only_fail(run_gate(gate_root), "floors_file")


def test_g0_15_render_consistency(gate_root):
    L, cfg, sv, _ = gate_root
    _edit(L.canary("E2", "low"), lambda r: r.update(**{"pass": False}))
    _only_fail(run_gate(gate_root), "render_consistency")
    _edit(L.canary("E2", "low"), lambda r: r.update(**{"pass": True}))
    _edit(sv.parent / "replay2" / "render_parity.json", lambda r: r.update(**{"pass": False}))
    _only_fail(run_gate(gate_root), "render_consistency")
    _edit(sv.parent / "replay2" / "render_parity.json", lambda r: r.update(**{"pass": True}))
    # an earlier run's canary, moved aside by the payload, is not read (only canary/top and canary/low)
    wj(L.start_dir("E2") / "canary" / "low.prev-20261111T000000Z" / "render_parity.json", {"pass": False})
    assert _item(run_gate(gate_root), "render_consistency")["ok"] is True
    L.canary("Q3B", "top").unlink()
    rep = run_gate(gate_root)
    assert _item(rep, "render_consistency")["ok"] is None and rep["decision"] == "incomplete"


def test_g0_16_demos_replay(gate_root):
    L, cfg, sv, _ = gate_root
    _edit(L.demos / "manifest.json", lambda m: m.update(kept=1500, dropped=100))
    _only_fail(run_gate(gate_root), "demos_replay")
    _edit(L.demos / "manifest.json", lambda m: m.update(kept=1599, dropped=1))
    wj(L.demos / "dropped.json", [{"task_id": "x", "reason": "replay_error", "detail": ""}])
    _only_fail(run_gate(gate_root), "demos_replay")


def test_g0_17_independent_recompute(gate_root):
    L, cfg, sv, _ = gate_root
    d = L.eval_dir("Q3S", "100", "c")
    _edit(d / "results.json", lambda r: r["sr"].update(k=4))  # doctored results.json
    rep = run_gate(gate_root)
    _only_fail(rep, "independent_recompute")
    assert rep["recheck"]["counts_equal"] is False
    _edit(d / "results.json", lambda r: r["sr"].update(k=3))
    _edit(d / "results.json", lambda r: r["by_category"]["SR-easy"]["sr"].update(k=0))
    _only_fail(run_gate(gate_root), "independent_recompute")
    _edit(d / "results.json", lambda r: r["by_category"]["SR-easy"]["sr"].update(k=2))
    fp = L.floors / "floors_eval500.json"
    _edit(fp, lambda r: r["random"].update(k=[9, 9, 9]))
    cfg["data"]["floors_eval500_sha256"] = fsha(fp)
    rep = run_gate(gate_root)
    _only_fail(rep, "independent_recompute")
    assert rep["recheck"]["floors_equal"] is False


def test_g0_6_kin_tolerance(gate_root, monkeypatch):
    L, cfg, sv, _ = gate_root
    monkeypatch.setattr(PT, "EVAL500_N", N_EVAL)
    d = L.eval_dir("E2", "ALL", "u")
    _edit(d / "results.json", lambda r: r["kin"].update(valid_list_mismatches=1))
    _only_fail(run_gate(gate_root), "kin")
    # pose / success-type mismatches: tolerated up to 1% of the tasks (with 8 tasks, 1 is 12.5% -> fail)
    _edit(d / "results.json", lambda r: r["kin"].update(valid_list_mismatches=0, tasks_unequal=1))
    _only_fail(run_gate(gate_root), "kin")
    cfg["gates"]["kin_eval"]["task_mismatch_max_frac"] = 0.125
    assert _item(run_gate(gate_root), "kin")["ok"] is True
    cfg["gates"]["kin_eval"]["task_mismatch_max_frac"] = 0.01
    _edit(L.demos / "manifest.json", lambda m: m.update(kin_equal_all_kept=False))
    _edit(d / "results.json", lambda r: r["kin"].update(tasks_unequal=0))
    _only_fail(run_gate(gate_root), "kin")


def test_g0_other_items_fail_on_their_own(gate_root):
    L, cfg, sv, _ = gate_root
    cases = [
        (L.eval_dir("Q3B", "0W", "c") / "results.json", lambda r: r.update(n_tasks=7), "evals_valid"),
        (L.eval_dir("Q3B", "ALL", "c") / "results.json", lambda r: r["audit"]["prompt_parity"].update(equal=False),
         "templates"),
        (L.eval_dir("Q3B", "ALL", "u") / "results.json",
         lambda r: r["audit"]["image_tokens"].update(tokens_per_image=[220]), "image_tokens"),
        (L.eval_dir("Q3S", "0W", "c") / "results.json", lambda r: r["renderer"].update(envs=6), "renderer"),
        (L.replay / "results.json", lambda r: r["renderer"].update(envs=6), "renderer"),
        (L.eval_dir("E2", "100", "c") / "results.json",
         lambda r: (r.update(code_sha256="x" * 64), r["code_fingerprint"].update(code_sha256="x" * 64)), "code"),
        (L.eval_dir("E2", "100", "c") / "results.json", lambda r: r["code_fingerprint"].update(code_sha256="x" * 64),
         "code"),
        (L.eval_dir("Q3S", "ALL", "u") / "results.json",
         lambda r: r["server"]["manifest"]["mm_processor_kwargs"]["size"].update(longest_edge=16777216), "attn_pixels"),
        (L.eval_dir("Q3S", "ALL", "c") / "results.json",
         lambda r: r["server"]["manifest"]["cfg"].update(enforce_eager=False), "settings"),
        (L.eval_dir("Q3S", "100", "c") / "results.json",
         lambda r: r["server"]["manifest"].update(extra_args=[]), "settings"),
        (L.parity("E2", "100"), lambda r: r.update(**{"pass": False}), "merge_parity"),
        (L.eval_dir("E2", "0W", "c") / "results.json", lambda r: r["adapter"].update(adapter_sha256="0" * 64),
         "served_adapter"),
        (L.adapter("Q3S", "ALL") / "meta.json", lambda m: m["attn_sites"].update(x="eager"), "attn_pixels"),
        (L.eval_dir("Q3S", "ALL", "blind") / "results.json", lambda r: r["settings"].update(blind_concurrency=8),
         "settings"),
        (L.eval_dir("Q3S", "ALL", "c") / "results.json", lambda r: r["server"]["versions"].update(vllm="0.10.2"),
         "settings"),
        (L.eval_dir("Q3B", "100s1002", "c") / "results.json", lambda r: r["server"].update(max_model_len=128000),
         "settings"),
        (L.eval_dir("Q3B", "100", "c") / "tasks" / sorted(p.name for p in (L.eval_dir("Q3B", "100", "c") / "tasks"
                                                                          ).iterdir())[0] / "trajectory.json",
         lambda r: r["vpd"].update(reach_digest="live"), "reach_frozen"),
        (L.eval_dir("E2", "ALL", "blind") / "official_check.json", lambda r: r.update(equal=False),
         "official_aggregate"),
    ]
    for path, fn, name in cases:
        orig = path.read_text()
        _edit(path, fn)
        _only_fail(run_gate(gate_root), name)
        path.write_text(orig)
    # the 0W of one start with other training code: code (and nothing that checks the same value elsewhere)
    _edit(L.adapter("E2", "0W") / "meta.json", lambda m: m.update(code_sha256="0865dd75f1ee" + "0" * 52))
    assert _item(run_gate(gate_root), "code")["ok"] is False


def _cases_fail_alone(gate_root, cases):
    for path, fn, name, needle in cases:
        orig = path.read_text()
        _edit(path, fn)
        rep = run_gate(gate_root)
        _only_fail(rep, name)
        probs = _item(rep, name)["detail"]["problems"]
        assert any(needle in b for b in probs), (name, needle, probs)
        path.write_text(orig)


def test_g0_4_judges_the_workers_ready_replies(gate_root):
    """G0(4): renderer.workers_ok and each worker's own reply (profile, L, pinned font, system prompt, TVRBench
    files), in the data job's replay and every image eval; the harness's own profile / lp alone are not enough."""
    L, cfg, sv, _ = gate_root
    ev = L.eval_dir("Q3S", "ALL", "u") / "results.json"
    _cases_fail_alone(gate_root, [
        (ev, lambda r: r["renderer"].update(workers_ok=False), "renderer", "workers_ok False"),
        (ev, lambda r: r["renderer"]["workers"][1].update(font_mode="system"), "renderer", "'font_mode': 'system'"),
        (ev, lambda r: r["renderer"]["workers"][0].update(system_prompt_ok=False), "renderer", "system_prompt_ok"),
        (ev, lambda r: r["renderer"]["workers"][0].update(lp="4"), "renderer", "'lp': '4'"),
        (ev, lambda r: r["renderer"].update(workers=[]), "renderer", "no worker ready reply"),
        (L.replay / "results.json", lambda r: r["renderer"]["workers"][3].update(tvrbench_files_ok=False), "renderer",
         "'wid': 3"),
        (L.replay / "results.json", lambda r: r["renderer"].pop("workers_ok"), "renderer",
         "replay: renderer workers_ok"),
    ])
    # the blind arm renders nothing and is not judged here
    _edit(L.eval_dir("Q3S", "ALL", "blind") / "results.json", lambda r: r.update(renderer={"none": "kin_env"}))
    assert _item(run_gate(gate_root), "renderer")["ok"] is True


def test_g0_10_checks_the_values_each_run_used(gate_root):
    """G0(10): settings.*_used / request_fields (run_tvr.settings) against the protocol, per arm; results.settings'
    copy of the protocol block alone is equal by construction."""
    L, cfg, sv, _ = gate_root
    c, u, b = (L.eval_dir("E2", "ALL", a) / "results.json" for a in ("c", "u", "blind"))
    _cases_fail_alone(gate_root, [
        (c, lambda r: r["settings"].update(max_tokens_used=64), "settings", "max_tokens_used 64 != 16"),
        (u, lambda r: r["settings"].update(regex_used=pilot_cfg.REGEX), "settings", "regex_used"),
        (c, lambda r: r["settings"].update(regex_used=None), "settings", "regex_used None"),
        (b, lambda r: r["settings"].update(blind_concurrency_used=8), "settings", "blind_concurrency_used 8 != 16"),
        (u, lambda r: r["settings"].update(blind_concurrency_used=16), "settings", "blind_concurrency_used 16 != None"),
        (c, lambda r: r["settings"]["request_fields"].remove("structured_outputs"), "settings", "request_fields"),
        (u, lambda r: r["settings"]["request_fields"].append("seed"), "settings", "request_fields"),
        (b, lambda r: r["settings"].update(arm="c"), "settings", "settings.arm 'c' != 'blind'"),
        (L.eval_dir("Q3B", "100s1002", "c") / "results.json", lambda r: r["settings"].update(max_model_len_used=128000),
         "settings", "max_model_len_used 128000 != the served max_model_len 32768"),
    ])
    _edit(u, lambda r: r["settings"].pop("request_fields"))
    rep = run_gate(gate_root)
    it = _item(rep, "settings")
    assert it["ok"] is None and any("settings.request_fields" in m for m in it["detail"]["missing"])
    assert rep["decision"] == "incomplete"


def test_g0_13_n_top_is_the_calibration_decision(gate_root):
    L, cfg, sv, _ = gate_root
    wj(L.cal_decision, {"n_top": "400", "train_400": True, "futility": False, "with_400": True})
    rep = run_gate(gate_root)
    _only_fail(rep, "adapter_provenance")
    assert any("tiers.n_top 'ALL' != " in b for b in _item(rep, "adapter_provenance")["detail"]["problems"])
    L.cal_decision.unlink()
    rep = run_gate(gate_root)
    assert _item(rep, "adapter_provenance")["ok"] is None and rep["decision"] == "incomplete"


def test_g0_5_replicate_zero_w_code_and_harness_fingerprint(gate_root):
    """G0(5): 0W_s1002 (no init) is compared with its start's 0W too; a missing code_fingerprint is incomplete; the
    data job's replay (older run_tvr code, here another sha) is not part of the one-harness rule."""
    L, cfg, sv, _ = gate_root
    assert json.loads((L.replay / "results.json").read_text())["code_sha256"] != "h" * 64
    assert _item(run_gate(gate_root), "code")["ok"] is True
    m = L.adapter("Q3B", "0W_s1002") / "meta.json"
    orig = m.read_text()
    _edit(m, lambda x: x.update(code_sha256="0865dd75f1ee" + "1" * 52))
    rep = run_gate(gate_root)
    _only_fail(rep, "code")
    assert any("Q3B/0W_s1002: code sha" in b for b in _item(rep, "code")["detail"]["problems"])
    m.write_text(orig)
    _edit(L.eval_dir("E2", "0W", "c") / "results.json", lambda r: r.pop("code_fingerprint"))
    it = _item(run_gate(gate_root), "code")
    assert it["ok"] is None and it["detail"]["missing"] == ["E2/0W__c: results.code_fingerprint.code_sha256"]


def test_gate_reads_the_run_tvr_layout_and_missing_fields_are_incomplete(gate_root):
    """results.code is the served code string (not a fingerprint) and the server block nests the VLLMServer manifest;
    a results.json without the manifest / code_sha256 is incomplete, never a pass."""
    L, cfg, sv, _ = gate_root
    p = L.eval_dir("Q3B", "ALL", "c") / "results.json"
    assert isinstance(json.loads(p.read_text())["code"], str)
    _edit(p, lambda r: (r["server"].pop("manifest"), r.pop("code_sha256")))
    rep = run_gate(gate_root)
    for name in ("code", "attn_pixels", "settings"):
        it = _item(rep, name)
        assert it["ok"] is None and it["detail"]["missing"], (name, it)
    assert rep["decision"] == "incomplete"


def test_o9_fallback_must_be_registered_before_the_freeze(gate_root, monkeypatch, tmp_path):
    L, cfg, sv, _ = gate_root
    for p in L.pilot_root.glob("*/evals/*/results.json"):  # run_tvr --max-model-len 128000: server and settings
        _edit(p, lambda r: (r["server"].update(max_model_len=128000), r["settings"].update(max_model_len_used=128000)))
    pre = tmp_path / "PREREG_TVR_PILOT.md"
    # the prereg names O9 and 128000 by design: that alone registers nothing
    pre.write_text("17 偏差日志\n- O9：若服务起不来，max_model_len 回退到 128000\n", encoding="utf-8")
    monkeypatch.setattr(PT, "prereg_path", lambda: pre)
    texts = {}
    monkeypatch.setattr(PT, "prereg_text_at", lambda commit: texts.get(commit))
    _only_fail(run_gate(gate_root), "settings")  # no CAL/shake/fallback_used.json
    fb = wj(L.fallback, {"fallback_used": True, "max_model_len": 128000, "first_rc": 1})
    rep = run_gate(gate_root)
    _only_fail(rep, "settings")  # the file, but no entry quoting its sha256
    assert any("quoting the sha256" in b for b in _item(rep, "settings")["detail"]["problems"])
    pre.write_text(pre.read_text(encoding="utf-8") + f"- O9 已用：fallback_used.json sha256 {fsha(fb)}\n",
                   encoding="utf-8")
    _only_fail(run_gate(gate_root), "settings")  # no run records the prereg commit it was prepared with
    wj(L.jobs / "tvr-msmoke-202611120000" / "prepared.json", {"prereg": {"commit": "c1"}})
    wj(L.jobs / "tvr-eval-q3s-top-202611130000" / "prepared.json", {"prereg": {"commit": "c2"}})
    texts.update(c1="the frozen text before the entry", c2=pre.read_text(encoding="utf-8"))
    rep = run_gate(gate_root)
    _only_fail(rep, "settings")  # the smoke was prepared before the entry: registered after the freeze
    assert any("tvr-msmoke-202611120000" in b for b in _item(rep, "settings")["detail"]["problems"])
    texts["c1"] = texts["c2"]
    rep = run_gate(gate_root)
    assert rep["decision"] == "ruler" and _item(rep, "settings")["detail"]["facts"]["o9_fallback"]["registered"]
    # a registered fallback, but the evals ran at the frozen 32768
    for p in L.pilot_root.glob("*/evals/*/results.json"):
        _edit(p, lambda r: (r["server"].update(max_model_len=32768), r["settings"].update(max_model_len_used=32768)))
    rep = run_gate(gate_root)
    _only_fail(rep, "settings")
    assert any("fallback_used.json exists" in b for b in _item(rep, "settings")["detail"]["problems"])


def test_g1_g2_g3_a4_decisions(gate_root):
    L, cfg, sv, _ = gate_root

    def set_k(s, key, k):
        tier, arm = key.split("__")
        d = L.eval_dir(s, tier, arm)
        tr = sorted(d.glob("tasks/*/trajectory.json"))
        by = {c: 0 for c in CATS}
        for i, p in enumerate(tr):
            t = json.loads(p.read_text())
            t["result"]["success"] = i < k
            by[pilot_cfg.CATEGORY_OF[(t["dataset"], t["difficulty"])]] += int(i < k)
            p.write_text(json.dumps(t))
        _edit(d / "results.json", lambda r: (r["sr"].update(k=k), [r["by_category"][c]["sr"].update(k=by[c])
                                                                    for c in CATS]))
    # G1: no start's top LB beats max(F, own 0W)
    for s in pilot_cfg.STARTS:
        set_k(s, "0W__c", 6)
    rep = run_gate(gate_root)
    assert rep["G1"]["status"] == "fail" and rep["decision"] == "not_a_ruler" and rep["A4"]["status"] == "pending"
    assert G.render_md(rep)
    set_k("Q3S", "0W__c", 0)
    assert run_gate(gate_root)["decision"] == "ruler"  # one start is enough
    # G3: saturated top tier (7/8 = 0.875 > 0.80)
    set_k("E2", "ALL__c", 7)
    rep = run_gate(gate_root)
    assert rep["G3"]["status"] == "fail" and rep["decision"] == "not_a_ruler"
    set_k("E2", "ALL__c", 6)
    # G2: a regex mismatch on a blind eval
    _edit(L.eval_dir("Q3B", "ALL", "blind") / "results.json", lambda r: r["invalid"].update(regex_mismatch=2))
    rep = run_gate(gate_root)
    assert rep["G2"]["status"] == "fail" and rep["decision"] == "engineering"
    _edit(L.eval_dir("Q3B", "ALL", "blind") / "results.json", lambda r: r["invalid"].update(regex_mismatch=0))
    # A4: pending without POWER.json, fail with pass false
    (L.pilot_root / "POWER.json").unlink()
    rep = run_gate(gate_root)
    assert rep["A4"]["status"] == "pending" and rep["decision"] == "incomplete"
    write_power(L, cfg, False)
    assert run_gate(gate_root)["decision"] == "power_insufficient"
    # a stale POWER.json (an input changed since, or another pilot_cfg, or a hand-written one) is pending, not read
    write_power(L, cfg, True)
    assert run_gate(gate_root)["decision"] == "ruler"
    pt = L.eval_dir("Q3S", "ALL", "c") / "per_task.jsonl"
    pt.write_text(pt.read_text() + "\n")
    rep = run_gate(gate_root)
    assert rep["A4"]["status"] == "pending" and "stale POWER.json" in rep["A4"]["note"]
    write_power(L, cfg, True)
    cfg["power"]["n_sims"] = 40
    assert run_gate(gate_root)["A4"]["status"] == "pending"
    cfg["power"]["n_sims"] = 2000
    wj(L.pilot_root / "POWER.json", {"pass": True})
    assert run_gate(gate_root)["A4"]["status"] == "pending"


def test_a4_reads_only_a_power_json_made_at_the_registered_settings(gate_root):
    L, cfg, sv, _ = gate_root
    pf = L.pilot_root / "POWER.json"
    rep = run_gate(gate_root)
    assert rep["A4"]["status"] == "pass" and rep["A4"]["note"] is None
    assert rep["A4"]["power"] == {"pass": True, "fallback_pass": False, "type1": 0.05, "type1_ok": True,
                                  "power_ok": True, "power_tier_corr": 0.5, "sigma_used": 0.03, "lambda_used": 0.0,
                                  "n_sims_used": 2000}
    orig = pf.read_text()
    for fn, needle in (
            (lambda d: d["registered"].update(n_sims_used=40), "registered.n_sims_used 40 != power.n_sims 2000"),
            (lambda d: d["registered"].update(n_sims_registered=200), "registered.n_sims_registered 200"),
            (lambda d: d["registered"].update(test="seed_aware_cluster_t"), "registered.test 'seed_aware_cluster_t'"),
            (lambda d: d["registered"].update(allow_no_replicate=True), "registered.allow_no_replicate True"),
            (lambda d: d["sigma"].update(replicate_missing=True), "sigma.replicate_missing True"),
            (lambda d: d["persistence"].pop("replicate_missing"), "persistence.replicate_missing None"),
            (lambda d: d["registered"]["tier_corr"].update(power=0.0), "registered.tier_corr != power.tier_corr"),
            (lambda d: d["registered"]["fallback"]["seeds"].update(TOP=[1]), "registered.fallback"),
            (lambda d: d["registered"]["persistence"].update(range=[0.0, 0.5]), "registered.persistence"),
            (lambda d: d.pop("registered"), "no registered block")):
        d = json.loads(orig)
        fn(d)
        wj(pf, d)
        rep = run_gate(gate_root)
        assert rep["A4"]["status"] == "pending" and rep["decision"] == "incomplete", needle
        assert rep["A4"]["note"].startswith("POWER.json not made at the registered settings") and \
            needle in rep["A4"]["note"], (needle, rep["A4"]["note"])
        assert rep["A4"]["power"] is None
    assert "not made at the registered settings" in G.render_md(rep)
    pf.write_text(orig)
    assert run_gate(gate_root)["decision"] == "ruler"
    # the rule text comes from pilot_cfg power.*
    assert rep["A4"]["rule"] == ("after G0-G3 pass: power_sim (seed_aware_endpoint_t, Holm): power >= 0.8 at 8 pp for "
                                 "both contrasts at tier_corr 0.5, sigma_used, lambda_used; family-wise type-I <= "
                                 "0.06 at tier_corr 0.0 and 1.0 (the max)")
    cfg["power"]["tier_corr"]["type1"] = [0.0, 0.5, 1.0]
    assert "tier_corr 0.0 and 0.5 and 1.0" in G.a4_rule(cfg)


def test_a4_reads_the_power_json_power_sim_writes(gate_root):
    """The real writer (power_sim.run_tvr) on the fake pilot root: current and registered (the default backend,
    power.backend numpy) -> A4 is its pass; the test-only --n-sims or backend override, or allow_no_replicate ->
    pending."""
    from vpd.common import power_sim

    L, cfg, sv, _ = gate_root
    cfg["power"]["n_sims"] = 20  # the registered value of this test cfg (the run is the registered one)
    pf = L.pilot_root / "POWER.json"
    pw = power_sim.run_tvr(L.pilot_root, cfg=cfg, out=pf)
    assert pw["registered"]["backend"] == cfg["power"]["backend"] == "numpy"
    assert G.power_unregistered(pw, cfg) == [] and G.power_stale(pw, cfg, []) == []
    rep = run_gate(gate_root)
    assert rep["A4"]["status"] == ("pass" if pw["pass"] else "fail") and rep["A4"]["note"] is None
    assert rep["A4"]["power"]["n_sims_used"] == 20 and rep["A4"]["power"]["lambda_used"] == \
        pw["persistence"]["lambda_used"]
    power_sim.run_tvr(L.pilot_root, cfg=cfg, out=pf, n_sims=10)
    rep = run_gate(gate_root)
    assert rep["A4"]["status"] == "pending" and "registered.n_sims_used 10 != power.n_sims 20" in rep["A4"]["note"]
    power_sim.run_tvr(L.pilot_root, cfg=cfg, out=pf, allow_no_replicate=True)
    rep = run_gate(gate_root)
    assert rep["A4"]["status"] == "pending" and "allow_no_replicate True" in rep["A4"]["note"]
    # an explicit (test-only) backend is recorded as used and is not the registered run
    pw = power_sim.run_tvr(L.pilot_root, cfg=cfg, out=pf, backend="python")
    want = "registered.backend 'python' != power.backend 'numpy'"
    assert pw["registered"]["backend"] == "python" and G.power_unregistered(pw, cfg) == [want]
    rep = run_gate(gate_root)
    assert rep["A4"]["status"] == "pending" and want in rep["A4"]["note"]


def test_a3_alpha_from_pilot_cfg(gate_root):
    L, cfg, sv, _ = gate_root
    assert run_gate(gate_root)["A3"]["per_start"]["Q3S"]["images_better_p_lt_0.05"] is True  # p 0.03125
    cfg["gates"]["mcnemar_alpha"] = 0.01
    a3 = run_gate(gate_root)["A3"]
    assert a3["per_start"]["Q3S"]["images_better_p_lt_0.05"] is False and "p < 0.01" in a3["rule"]


def test_gate_incomplete_exit_3(gate_root, monkeypatch):
    L, cfg, sv, rargs = gate_root
    shutil.rmtree(L.eval_dir("E2", "ALL", "u"))
    rep = run_gate(gate_root)
    assert rep["decision"] == "incomplete" and _item(rep, "evals_valid")["ok"] is None
    real = G.evaluate
    monkeypatch.setattr(G, "evaluate", lambda root, starts, smoke_verdict_path=None, power=False: real(
        root, starts, L=L, cfg=cfg, smoke_verdict_path=sv, recheck_args=rargs))
    assert G.main(["--pilot-root", str(L.pilot_root)]) == 3


# ------------------------------------------------------------------------------------------ payloads: dry runs
_KNOBS = ("VPD_THREADS", "VPD_JOB_DEADLINE_S", "VPD_MIN_STEP_S", "VPD_MERGE_DIR", "VPD_RERUN_STALE", "VPD_DRYRUN",
          "RUN", "VPD_TVR_OUT", "VPD_DRY_NTOP", "VPD_WORKTREE", "VPD_TVR_PY")


def _bash(args, extra, cwd):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update(extra)
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env, cwd=str(cwd), timeout=300)


def _dry(out: str):
    return [ln[len("[dry] "):] for ln in out.splitlines() if ln.startswith("[dry] ")]


def _kinds(lines):
    out = []
    for ln in lines:
        m = re.search(r"-m vpd\.tvr\.(\w+) (\S+)", ln)
        if "n2_train_vpd.py" in ln:
            out.append("dry" if "--dry-run" in ln else "train:" + ln.split("--phase ")[1].split()[0])
        elif "parity_check.py" in ln:
            out.append("hfparity")
        elif "merge.py" in ln:
            out.append("merge:" + ln.split("--adapter ")[1].split()[0].rsplit("/", 1)[1])
        elif "rm -rf" in ln:
            out.append("rm")
        elif ln.startswith("reap_unity"):
            out.append("reap:" + ln.split()[1])
        elif "vpd.walkerbench.pilot adapter-ok" in ln:
            out.append("adapter-ok:" + ln.split("adapter-ok ")[1].split()[0].rsplit("/", 1)[1])
        elif m and m.group(1) == "run_tvr":
            if m.group(2) == "replay":
                out.append("replay:" + ln.split("--task-set ")[1].split()[0])
            else:
                out.append("eval:" + ln.split("--tier ")[1].split()[0] + ":" + ln.split("--arms ")[1].split()[0])
        elif m and m.group(1) == "pilot_tvr":
            out.append(m.group(2))
        elif m:
            out.append(m.group(1) + ":" + m.group(2))
        elif ln.startswith("vpd_snapshot_file"):
            out.append("prereg_committed")
        elif ln.startswith("date -u >"):
            out.append("DONE")
        else:
            out.append("?" + ln[:40])
    return out


def dry_run(tmp_path, script, *args, **env):
    run = tmp_path / "jobs" / "run-202611110000"
    e = {"VPD_DRYRUN": "1", "VPD_TVR_OUT": str(tmp_path / "tvr"), "VPD_MERGE_DIR": "/dev/shm/vpd_merged"}
    e.update(env)
    res = _bash([str(JOBS_DIR / script), *args, str(run)], e, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    return res, _dry(res.stdout), run


def _common_dry_asserts(lines, run, tmp_path, deadline):
    thor = [ln for ln in lines if "vpd.tvr.run_tvr" in ln or "vpd.tvr.reach dump" in ln]
    assert thor and all(ln.startswith("timeout -k 30 ") and re.search(r"--deadline-s (\d+)$", ln) for ln in thor)
    for ln in thor:
        t, dl = int(ln.split()[3]), int(re.search(r"--deadline-s (\d+)$", ln).group(1))
        assert t <= deadline and t - dl == 240, ln
    heavy = [ln for ln in lines if any(k in ln for k in ("n2_train_vpd", "merge.py", "merge_parity", "render_parity",
                                                         "parity_check.py", "score_tvr official"))]
    assert all(ln.startswith("timeout -k 30 ") for ln in heavy)
    steps = [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]
    assert all(s["rc"] == 0 for s in steps) and not (run / "DONE").exists()
    assert not (tmp_path / "tvr").exists()  # a dry run writes nothing under OUT/tvr
    k = _kinds(lines)
    for i, kind in enumerate(k):  # every THOR step is followed by its reap
        if kind.startswith(("eval:", "replay:")) or kind == "reach:dump":
            assert k[i + 1].startswith("reap:"), (i, k)
    return k


def test_payload_dry_data(tmp_path):
    res, lines, run = dry_run(tmp_path, "tvr_data.sh")
    k = _common_dry_asserts(lines, run, tmp_path, 8520)
    assert k == ["guard", "tasks:dev-select", "reach:houses", "reach:dump", "reap:reach_dump", "replay:sft1600",
                 "reap:replay", "demos:build", "tasks:tiers", "bfs:table", "floors:--task-sets", "expect", "DONE"]
    T = tmp_path / "tvr"
    rep = next(ln for ln in lines if "run_tvr replay" in ln)
    assert (f"--public {T}/hf_meta/tvr-sft-va/sft_train.json --run-dir {T}/replay/sft1600_lvp --images-dir "
            f"{T}/demos/sft1600_lvp/images --envs <render.replay_envs> --lp <render.lp_threads> "
            "--profile <render.lvp_profile> --resume --deadline-s") in rep
    dump = next(ln for ln in lines if "reach dump" in ln)
    assert f"--task-sets sft1600,eval500,dev200 --out {T}/reach/v1 --envs <render.replay_envs>" in dump
    assert next(ln for ln in lines if "pilot_tvr guard" in ln).endswith(f"guard tvr-data --run {run}")
    assert f"floors --task-sets eval500,dev200 --reach {T}/reach/v1 --out {T}/floors" in res.stdout
    assert "deadline=8520s" in res.stdout and str(paths.PY_EVAL) in lines[0]


def test_payload_dry_cal_and_fallback_branch(tmp_path):
    res, lines, run = dry_run(tmp_path, "tvr_cal.sh")
    k = _common_dry_asserts(lines, run, tmp_path, 19320)
    assert k == ["guard", "dry", "hfparity", "eval:raw:c,blind", "reap:shake", "replay:sft2", "reap:canary_replay",
                 "render_parity:--replay", "adapter-ok:0W", "train:0W", "adapter-ok:100", "train:tier100",
                 "adapter-ok:ALL", "train:tierALL", "parity-ok", "rm", "merge:ALL", "merge_parity:--start",
                 "eval:ALL:c", "reap:eval_ALL", "rm", "cal-decide", "rm", "merge:0W", "eval:0W:c", "reap:eval_0W",
                 "rm", "rm", "merge:100", "eval:100:c", "reap:eval_100", "rm", "expect", "DONE"]
    C = tmp_path / "tvr" / "runs" / "explore" / "cal" / "Q3B"
    shake = next(ln for ln in lines if "--tier raw" in ln)
    assert (f"--task-set smoke4 --name-prefix calshake --out-root {C}/shake --mechanism-only --vllm-parity "
            f"{C}/dry/dry_run.json --resume") in shake and "--max-model-len" not in res.stdout
    ev = next(ln for ln in lines if "--tier ALL" in ln)
    assert (f"--task-set dev200 --name-prefix cal --out-root {C}/evals --model-dir /dev/shm/vpd_merged/cal__Q3B__ALL "
            f"--adapter-meta {C}/adapters/ALL/meta.json --vllm-parity {C}/dry/dry_run.json --resume") in ev
    mp = next(ln for ln in lines if "merge_parity" in ln)
    assert (f"--adapter {C}/adapters/ALL --merged /dev/shm/vpd_merged/cal__Q3B__ALL --demos "
            f"{tmp_path}/tvr/tiers/v1/warmup0w20.jsonl --n 20 --out {C}/parity/merge_ALL.json") in mp
    dry = next(ln for ln in lines if "--dry-run" in ln)
    assert f"--data {tmp_path}/tvr/tiers/v1/parity3.jsonl --dry-run --limit 3 --expect-grid 22x40 --out {C}/dry" in dry
    tr = [ln for ln in lines if "n2_train_vpd.py" in ln and "--dry-run" not in ln]
    assert all("--epochs auto --seed 1001" in ln and "--threads 2 --num-workers 3" in ln for ln in tr)
    assert tr[1].endswith(f"--init-adapter {C}/adapters/0W") and tr[2].endswith(f"--init-adapter {C}/adapters/0W")
    # the O9 fallback file exists (decided by an earlier shakedown): every eval passes --max-model-len
    (C / "shake").mkdir(parents=True)
    (C / "shake" / "fallback_used.json").write_text("{}")
    res = _bash([str(JOBS_DIR / "tvr_cal.sh"), str(run)], {"VPD_DRYRUN": "1", "VPD_TVR_OUT": str(tmp_path / "tvr"),
                                                            "VPD_MERGE_DIR": "/dev/shm/vpd_merged"}, tmp_path)
    evs = [ln for ln in _dry(res.stdout) if "run_tvr eval" in ln]
    assert len(evs) == 4 and all("--max-model-len <protocol.max_model_len_fallback>" in ln for ln in evs)


def test_payload_dry_cal400(tmp_path):
    res, lines, run = dry_run(tmp_path, "tvr_cal400.sh")
    k = _common_dry_asserts(lines, run, tmp_path, 4920)
    assert k == ["guard", "replay:sft2", "reap:canary_replay", "render_parity:--replay", "adapter-ok:0W",
                 "adapter-ok:400", "train:tier400", "parity-ok", "rm", "merge:400", "merge_parity:--start",
                 "eval:400:c", "reap:eval_400", "rm", "cal-decide", "expect", "DONE"]
    C = tmp_path / "tvr" / "runs" / "explore" / "cal" / "Q3B"
    assert f"--run-dir {C}/canary400 --images-dir {C}/canary400/images" in res.stdout
    assert f"--data {tmp_path}/tvr/tiers/v1/demos_400.jsonl --out {C}/adapters/400" in res.stdout
    assert f"--init-adapter {C}/adapters/0W" in next(ln for ln in lines if "tier400" in ln)


def test_payload_dry_msmoke(tmp_path):
    res, lines, run = dry_run(tmp_path, "tvr_mech_smoke.sh")
    k = _common_dry_asserts(lines, run, tmp_path, 3120)
    per = ["dry", "hfparity", "eval:raw:c,u,blind"]
    assert k == ["prereg_committed", "guard"] + [x for s in ("Q3B", "Q3S", "E2") for x in per + [f"reap:session_{s}"]] \
        + ["replay:sft2", "reap:replay2", "render_parity:--replay", "smoke-verdict", "expect", "DONE"]
    SM = tmp_path / "tvr" / "runs" / "smoke" / run.name
    for s in ("Q3B", "Q3S", "E2"):
        sess = next(ln for ln in lines if f"--start {s} --tier raw" in ln)
        assert (f"--arms c,u,blind --task-set smoke8 --name-prefix smoke --out-root {SM}/{s}/evals --mechanism-only "
                f"--vllm-parity {SM}/{s}/dry/dry_run.json --resume") in sess
    rp = next(ln for ln in lines if "render_parity" in ln)
    assert (f"--replay {SM}/replay2 --ref {tmp_path}/tvr/replay/sft1600_lvp --ref-images "
            f"{tmp_path}/tvr/demos/sft1600_lvp/images --images {SM}/replay2/images --task-set sft2 --out "
            f"{SM}/replay2/render_parity.json") in rp
    assert "eval500" not in res.stdout and "merge.py" not in res.stdout


@pytest.mark.parametrize("start,ntop", [("Q3B", "ALL"), ("Q3S", "ALL"), ("E2", "400"), ("Q3B", "400")])
def test_payload_dry_train(tmp_path, start, ntop):
    res, lines, run = dry_run(tmp_path, "tvr_train.sh", start, VPD_DRY_NTOP=ntop)
    steps = [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]
    assert all(s["rc"] == 0 for s in steps)
    k = _kinds(lines)
    want = ["prereg_committed", "guard", "adapter-ok:0W", "train:0W", "adapter-ok:100", "train:tier100",
            f"adapter-ok:{ntop}", f"train:tier{ntop}"]
    if start == "Q3B":
        want += ["adapter-ok:0W_s1002", "train:0W_s1002", "adapter-ok:100_s1002", "train:tier100_s1002"]
    assert k == want + ["expect", "DONE"]
    P = tmp_path / "tvr" / "runs" / "pilot" / start
    tr = [ln for ln in lines if "n2_train_vpd.py" in ln]
    assert f"--data {tmp_path}/tvr/tiers/v1/demos_{ntop}.jsonl --out {P}/adapters/{ntop}" in tr[2]
    assert all("--epochs auto" in ln and "--threads 2 --num-workers 3" in ln for ln in tr)
    if start == "Q3B":
        assert "--seed 1002 --phase 0W_s1002" in tr[3] and tr[4].endswith(f"--init-adapter {P}/adapters/0W_s1002")
        assert "--seed 1002 --phase tier100_s1002" in tr[4]
    else:
        assert "s1002" not in res.stdout
    assert f"guard tvr-train-{start.lower()} --run {run}" in res.stdout


@pytest.mark.parametrize("start,part,ntop", [("Q3B", "top", "ALL"), ("Q3B", "low", "ALL"), ("E2", "low", "400"),
                                             ("Q3S", "top", "400")])
def test_payload_dry_eval(tmp_path, start, part, ntop):
    res, lines, run = dry_run(tmp_path, "tvr_eval.sh", start, part, VPD_DRY_NTOP=ntop)
    k = _common_dry_asserts(lines, run, tmp_path, 8160)
    head = ["prereg_committed", "guard", "replay:sft2", "reap:canary_replay", "render_parity:--replay"]
    if part == "top":
        want = head + ["adapter-ok:0W", f"adapter-ok:{ntop}", "parity-ok", "rm", f"merge:{ntop}",
                       "merge_parity:--start", f"eval:{ntop}:c,u,blind", f"reap:eval_{ntop}", "rm"] + \
            ["score_tvr:official"] * 3
    else:
        sess = []
        for tier, ad in (("0W", "0W"), ("100", "100")) + ((("100s1002", "100_s1002"),) if start == "Q3B" else ()):
            sess += ["parity-ok", "rm", f"merge:{ad}", "merge_parity:--start", f"eval:{tier}:c", f"reap:eval_{tier}",
                     "rm"]
        want = head + ["adapter-ok:0W", "adapter-ok:100"] + sess[:14]
        if start == "Q3B":
            want += ["adapter-ok:0W_s1002", "adapter-ok:100_s1002"] + sess[14:]
        want += ["score_tvr:official"] * (3 if start == "Q3B" else 2)
    assert k == want + ["expect", "DONE"]
    P = tmp_path / "tvr" / "runs" / "pilot" / start
    evs = [ln for ln in lines if "run_tvr eval" in ln]
    assert all("--task-set eval500 --name-prefix pilot" in ln and f"--out-root {P}/evals" in ln and "--resume" in ln
               and "--vllm-parity" not in ln and "--mechanism-only" not in ln for ln in evs)
    mps = [ln for ln in lines if "merge_parity" in ln]
    assert len(mps) == len(evs) and all("--n 20" in ln and f"--out {P}/parity/merge_" in ln for ln in mps)
    assert f"--run-dir {P}/canary/{part} --images-dir {P}/canary/{part}/images" in res.stdout
    offs = [ln for ln in lines if "score_tvr official" in ln]
    assert all(f"--served <code:{start}> --bfs {tmp_path}/tvr/bfs/eval500_opt.json" in ln for ln in offs)
    if part == "low" and start == "Q3B":
        assert f"--adapter-meta {P}/adapters/100_s1002/meta.json" in evs[2] and "--tier 100s1002" in evs[2]
    # merge parity runs before vLLM: every merge_parity line precedes its session's eval line
    for mp_line, ev_line in zip(mps, evs):
        assert lines.index(mp_line) < lines.index(ev_line)


def test_payload_usage_and_deadline_refusals(tmp_path):
    for script, args in (("tvr_data.sh", []), ("tvr_train.sh", ["Q3X"]), ("tvr_eval.sh", ["Q3B", "mid"]),
                         ("tvr_cal.sh", ["a", "b"])):
        r = _bash([str(JOBS_DIR / script), *args], {"VPD_DRYRUN": "1"}, tmp_path)
        assert r.returncode == 2 and "usage" in r.stderr, script
    for bad in ("0", "x"):
        r = _bash([str(JOBS_DIR / "tvr_cal.sh"), str(tmp_path / "r")], {"VPD_DRYRUN": "1",
                                                                         "VPD_JOB_DEADLINE_S": bad}, tmp_path)
        assert r.returncode == 2 and "VPD_JOB_DEADLINE_S" in r.stderr
    run = tmp_path / "jobs" / "short"
    r = _bash([str(JOBS_DIR / "tvr_eval.sh"), "E2", "top", str(run)],
              {"VPD_DRYRUN": "1", "VPD_TVR_OUT": str(tmp_path / "tvr"), "VPD_JOB_DEADLINE_S": "200",
               "VPD_MERGE_DIR": "/dev/shm/vpd_merged"}, tmp_path)
    assert r.returncode == 1
    names = [json.loads(s)["name"] for s in (run / "steps.jsonl").read_text().splitlines()]
    assert "canary_replay:no_time" in names and "eval_ALL:canary_failed" in names and not (run / "DONE").exists()


def test_dry_run_root_hook_only_in_dry_mode():
    for s in SCRIPTS:
        text = (JOBS_DIR / s).read_text()
        assert 'if [[ $DRY == 1 && -n ${VPD_TVR_OUT:-} ]]; then' in text, s
        code = [ln for ln in text.splitlines() if "VPD_TVR_OUT" in ln and not ln.lstrip().startswith("#")]
        assert len(code) == 2 and "TVR=$VPD_TVR_OUT" in code[1], (s, code)


# ------------------------------------------------------------------------------------------ payloads: non-dry with fakes
FAKE_PY = r"""#!/usr/bin/env bash
# fake PY_EVAL / PY_Q3 / PY_TRAIN / TVR_PY of the non-dry payload tests: log the call, answer it, touch nothing else
printf '%s\n' "$*" >> "$FAKE_LOG"
a=" $* "
if [[ -n ${FAKE_SLEEP_ON:-} && $a == *"$FAKE_SLEEP_ON"* ]]; then sleep "${FAKE_SLEEP_S:-3}"; fi
if [[ -n ${FAKE_FAIL:-} && $a == *"$FAKE_FAIL"* ]]; then exit "${FAKE_FAIL_RC:-1}"; fi
case $a in
  *" vpd.tvr.pilot_tvr cfg "*)
    out=()
    for k in "$@"; do
      case $k in
        render.replay_envs|render.eval_envs) out+=(5) ;;
        render.lp_threads) out+=(2) ;;
        render.lvp_profile) out+=(envlib) ;;
        render.build) out+=(thor-fake-build) ;;
        protocol.max_model_len_fallback) out+=(128000) ;;
        tiers.n_top) out+=("${FAKE_NTOP:-ALL}") ;;
      esac
    done
    echo "${out[*]}"; exit 0 ;;
  *" vpd.tvr.pilot_tvr served "*) echo fake-code; exit 0 ;;
  *" vpd.tvr.pilot_tvr guard "*) exit "${FAKE_GUARD_RC:-0}" ;;
  *" vpd.tvr.pilot_tvr expect "*) exit "${FAKE_EXPECT_RC:-0}" ;;
  *" vpd.tvr.pilot_tvr parity-ok "*) exit "${FAKE_PARITY_RC:-1}" ;;
  *" vpd.walkerbench.pilot adapter-ok "*) exit "${FAKE_ADAPTER_RC:-1}" ;;
  *" vpd.walkerbench.pilot eval-ok "*) exit "${FAKE_EVALOK_RC:-0}" ;;
  *" vpd/train/merge.py "*)
    if [[ -n ${FAKE_MERGE_WRITES:-} ]]; then  # the merged folder on job-local disk (only the TERM test)
      prev=""; od=""
      for x in "$@"; do [[ $prev == --out ]] && od=$x; prev=$x; done
      mkdir -p "$od" && echo x > "$od/model.safetensors"
    fi
    exit 0 ;;
  *" vpd.snapshot file "*) echo c0ffee; exit 0 ;;
  *" --tier raw "*"--name-prefix calshake "*)
    if [[ $a == *" --max-model-len "* ]]; then exit 0; fi
    if [[ -n ${FAKE_SHAKE_SERVER_ERROR:-} ]]; then  # what run_tvr writes when VLLMServer.__enter__ fails
      prev=""; od=""
      for x in "$@"; do [[ $prev == --out-root ]] && od=$x; prev=$x; done
      mkdir -p "$od" && echo '{"error": "RuntimeError: vLLM exited with 1", "max_model_len": 32768}' \
        > "$od/calshake__Q3B__raw__server_error.json"
    fi
    exit "${FAKE_SHAKE_RC:-0}" ;;
  *" vpd.tvr.render_parity "*) exit "${FAKE_CANARY_RC:-0}" ;;
esac
exit 0
"""


def fake_tree(tmp_path: Path, script: str):
    jobs = tmp_path / "code" / "vpd" / "jobs"
    jobs.mkdir(parents=True)
    shutil.copyfile(JOBS_DIR / script, jobs / script)
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("pgrep", "pkill"):  # no Unity player is ever alive here
        (bindir / tool).write_text("#!/usr/bin/env bash\nexit 1\n")
        (bindir / tool).chmod(0o755)
    q = shlex.quote
    (jobs / "common_env.sh").write_text(
        "# test shim: the real prelude, then fake interpreters and a temp OUT\n"
        f"source {q(str(JOBS_DIR / 'common_env.sh'))}\n"
        f"export VPD_OUT={q(str(tmp_path / 'OUT'))} PY_EVAL={q(str(fake))} PY_Q3={q(str(fake))} "
        f"PY_TRAIN={q(str(fake))}\n")
    env = {"VPD_WORKTREE": str(tmp_path / "code"), "VPD_TVR_PY": str(fake), "FAKE_LOG": str(tmp_path / "fake.log"),
           # a /dev/shm/vpd_merged path that never exists: the fake merge writes nothing, safe_rm_merged removes nothing real
           "VPD_MERGE_DIR": f"/dev/shm/vpd_merged/pytest-tvrjobs-{os.getpid()}-{tmp_path.name}", "PATH": f"{bindir}:{os.environ['PATH']}"}
    return jobs / script, env


def _steps(run: Path):
    return [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]


def test_nondry_cal_all_pass_writes_done(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_cal.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110000"
    res = _bash([str(payload), str(run)], env, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    names = [s["name"] for s in _steps(run)]
    assert all(s["rc"] == 0 for s in _steps(run)) and (run / "DONE").exists()
    for r in PT.required_steps("tvr-cal"):
        assert any(n == r or n.startswith(r + ":") for n in names), r
    cal = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "cal" / "Q3B"
    assert sorted(p.name for p in cal.iterdir()) == ["adapters", "canary", "dry", "evals", "parity", "shake"]
    log = (tmp_path / "fake.log").read_text()
    assert "--max-model-len" not in log and log.count("vpd.tvr.run_tvr eval") == 4


def test_nondry_cal_failed_step_keeps_going_no_done(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_cal.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110001"
    run.mkdir(parents=True)
    (run / "DONE").write_text("stale\n")
    res = _bash([str(payload), str(run)], dict(env, FAKE_FAIL="--phase tier100 "), tmp_path)
    assert res.returncode == 1
    st = {s["name"]: s["rc"] for s in _steps(run)}
    assert st["train_100"] == 1 and st["train_ALL"] == 0 and st["eval_ALL"] == 0 and st["cal_decide"] == 0
    assert st["merge_100:no_adapter"] == 1 and st["eval_0W"] == 0 and "expect" in st
    assert not (run / "DONE").exists() and "failed steps: train_100 merge_100" in res.stdout


def test_nondry_cal_mechanism_failure_stops_before_training(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_cal.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110002"
    res = _bash([str(payload), str(run)], dict(env, FAKE_CANARY_RC="1"), tmp_path)
    assert res.returncode == 1
    names = [s["name"] for s in _steps(run)]
    assert "canary_parity" in names and "after_mechanism:stopped" in names and names[-1] == "expect"
    assert not any(n.startswith(("train_", "merge_", "eval_")) for n in names if n != "merge_disk")
    assert "--phase" not in (tmp_path / "fake.log").read_text()  # no training call at all


def test_nondry_cal_o9_fallback(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_cal.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110003"
    res = _bash([str(payload), str(run)], dict(env, FAKE_SHAKE_RC="1", FAKE_SHAKE_SERVER_ERROR="1"), tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    st = [(s["name"], s["rc"]) for s in _steps(run)]
    assert ("shake", 1) in st and ("shake_fallback", 0) in st and (run / "DONE").exists()
    shake = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "cal" / "Q3B" / "shake"
    doc = json.loads((shake / "fallback_used.json").read_text())
    first = shake / "server_error_first_attempt.json"
    assert doc["max_model_len"] == 128000 and doc["first_rc"] == 1 and doc["first_server_error"] == str(first)
    assert doc["first_server_error_sha256"] == fsha(first) and "vLLM exited" in first.read_text()
    log = (tmp_path / "fake.log").read_text().splitlines()
    evs = [ln for ln in log if "run_tvr eval" in ln and "--tier raw" not in ln]
    assert len(evs) == 3 and all("--max-model-len 128000" in ln for ln in evs)
    fall = [ln for ln in log if "--name-prefix calshake" in ln]
    assert len(fall) == 2 and "--resume" not in fall[1] and "--max-model-len 128000" in fall[1]
    # steps_check excuses the first shake attempt
    run2 = tmp_path / "chk"
    run2.mkdir()
    shutil.copy(run / "steps.jsonl", run2 / "steps.jsonl")
    assert PT.steps_check("tvr-cal", run2)["ok"]


@pytest.mark.parametrize("rc,server_error", [("2", "1"), ("1", ""), ("4", "1")])
def test_nondry_cal_no_o9_retry_unless_the_server_refused(tmp_path, rc, server_error):
    """A refusal (2), a failure without this attempt's server_error.json, or a deadline (4) never switches the pilot
    to the fallback: no retry, no fallback_used.json, the mechanism stop."""
    payload, env = fake_tree(tmp_path, "tvr_cal.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110004"
    res = _bash([str(payload), str(run)], dict(env, FAKE_SHAKE_RC=rc, FAKE_SHAKE_SERVER_ERROR=server_error), tmp_path)
    assert res.returncode == 1
    names = [s["name"] for s in _steps(run)]
    assert "shake" in names and not any(n.startswith("shake_fallback") for n in names)
    assert "after_mechanism:stopped" in names and not (run / "DONE").exists()
    shake = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "cal" / "Q3B" / "shake"
    assert not (shake / "fallback_used.json").exists()
    assert "--max-model-len" not in (tmp_path / "fake.log").read_text()


def test_nondry_cal_old_server_error_does_not_trigger_o9(tmp_path):
    """A server_error.json left by an EARLIER attempt (older than this shake step) is not this attempt's."""
    payload, env = fake_tree(tmp_path, "tvr_cal.sh")
    shake = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "cal" / "Q3B" / "shake"
    old = wj(shake / "calshake__Q3B__raw__server_error.json", {"error": "old"})
    os.utime(old, (1_000_000_000, 1_000_000_000))
    run = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110005"
    res = _bash([str(payload), str(run)], dict(env, FAKE_SHAKE_RC="1"), tmp_path)
    assert res.returncode == 1 and not (shake / "fallback_used.json").exists()
    assert not any(s["name"].startswith("shake_fallback") for s in _steps(run))


def test_nondry_cal_failed_fallback_is_decided_again_on_the_rerun(tmp_path):
    """The O9 fallback attempt itself fails (rc 4): no fallback_used.json, the mechanism stop.  Its arm dirs hold the
    fallback's protocol; the next cal run moves them and the first attempt's copy aside and decides again (here the
    server starts at the frozen value: no fallback, DONE)."""
    payload, env = fake_tree(tmp_path, "tvr_cal.sh")
    shake = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "cal" / "Q3B" / "shake"
    run = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110006"
    res = _bash([str(payload), str(run)], dict(env, FAKE_SHAKE_RC="1", FAKE_SHAKE_SERVER_ERROR="1",
                                               FAKE_FAIL="--max-model-len 128000", FAKE_FAIL_RC="4"), tmp_path)
    assert res.returncode == 1 and not (run / "DONE").exists()
    st = [(s["name"], s["rc"]) for s in _steps(run)]
    assert ("shake", 1) in st and ("shake_fallback", 4) in st and ("after_mechanism:stopped", 1) in st
    keep = shake / "server_error_first_attempt.json"
    assert keep.is_file() and not (shake / "fallback_used.json").exists()
    for a in ("c", "blind"):  # what the failed fallback attempt of run_tvr left (its protocol)
        wj(shake / f"calshake__Q3B__raw__{a}" / "protocol.json", {"max_model_len_used": 128000})
    (tmp_path / "fake.log").write_text("")
    run2 = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110007"
    res = _bash([str(payload), str(run2)], env, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    names = [s["name"] for s in _steps(run2)]
    assert names.index("shake_aside:calshake__Q3B__raw__c") < names.index("shake")
    assert {"shake_aside:calshake__Q3B__raw__blind", "shake_aside:server_error_first_attempt.json"} <= set(names)
    assert not keep.exists() and len(list(shake.glob("server_error_first_attempt.json.prev-*"))) == 1
    prev = sorted(shake.glob("calshake__Q3B__raw__c.prev-*"))
    assert len(prev) == 1 and json.loads((prev[0] / "protocol.json").read_text())["max_model_len_used"] == 128000
    assert not (shake / "fallback_used.json").exists() and "--max-model-len" not in (tmp_path / "fake.log").read_text()
    assert (run2 / "DONE").exists() and PT.steps_check("tvr-cal", run2)["ok"]
    # a third run: nothing to move aside any more
    run3 = tmp_path / "OUT" / "jobs" / "tvr-cal-202611110008"
    assert _bash([str(payload), str(run3)], env, tmp_path).returncode == 0
    assert not any(s["name"].startswith("shake_aside") for s in _steps(run3))


def test_nondry_canary_kept_when_there_is_no_time_to_render(tmp_path):
    """A rerun without the time to render a canary (canary_replay:no_time) keeps the earlier canary in place (the
    one the existing evals were made under) instead of moving it aside and leaving canary/<part> empty."""
    payload, env = fake_tree(tmp_path, "tvr_eval.sh")
    P = tmp_path / "OUT" / "tvr" / "runs" / "pilot" / "Q3S"
    old = wj(P / "canary" / "top" / "render_parity.json", {"pass": True, "node": "an earlier run's node"})
    run = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-top-202611110400"
    res = _bash([str(payload), "Q3S", "top", str(run)],
                dict(env, FAKE_ADAPTER_RC="0", VPD_JOB_DEADLINE_S="250", VPD_MIN_STEP_S="60"), tmp_path)
    assert res.returncode == 1 and not (run / "DONE").exists()
    names = [s["name"] for s in _steps(run)]
    assert "canary_replay:no_time" in names and "canary_aside" not in names and "eval_ALL:canary_failed" in names
    assert old.is_file() and json.loads(old.read_text())["node"] and not list((P / "canary").glob("top.prev-*"))
    assert "replay --task-set sft2" not in (tmp_path / "fake.log").read_text()


def test_nondry_code_mismatch_stops_with_exit_3(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_train.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-train-q3s-202611110000"
    res = _bash([str(payload), "Q3S", str(run)], dict(env, FAKE_ADAPTER_RC="3"), tmp_path)
    assert res.returncode == 3
    names = [s["name"] for s in _steps(run)]
    assert names[-1] == "train_0W:code_mismatch" and not (run / "DONE").exists()
    assert "n2_train_vpd.py" not in (tmp_path / "fake.log").read_text()


def test_nondry_eval_skips_and_canary_failure(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_eval.sh")
    P = tmp_path / "OUT" / "tvr" / "runs" / "pilot" / "Q3S"
    for arm in ("c", "u", "blind"):
        wj(P / "evals" / f"pilot__Q3S__ALL__{arm}" / "results.json", {"valid": True})
    old = wj(P / "canary" / "top" / "render_parity.json", {"pass": True, "node": "an earlier run's node"})
    run = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-top-202611110000"
    res = _bash([str(payload), "Q3S", "top", str(run)], dict(env, FAKE_ADAPTER_RC="0", FAKE_PARITY_RC="0"), tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    names = [s["name"] for s in _steps(run)]
    assert {"merge_ALL:skip", "merge_parity_ALL:skip", "eval_ALL:skip"} <= set(names) and (run / "DONE").exists()
    assert names.count("official_ALL_c") == 1 and "merge.py" not in (tmp_path / "fake.log").read_text()
    # the earlier canary was moved aside and this run rendered its own (the replay ran without finding old frames)
    assert names.index("canary_aside") < names.index("canary_replay") and not old.exists()
    prev = sorted((P / "canary").glob("top.prev-*"))
    assert len(prev) == 1 and json.loads((prev[0] / "render_parity.json").read_text())["node"]
    assert (P / "canary" / "top").is_dir() and "replay --task-set sft2" in (tmp_path / "fake.log").read_text()
    # current but failed merge parity: not rerun, recorded as a failure
    run2 = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-top-202611110001"
    res = _bash([str(payload), "Q3S", "top", str(run2)], dict(env, FAKE_ADAPTER_RC="0", FAKE_PARITY_RC="4"), tmp_path)
    assert res.returncode == 1 and "merge_parity_ALL:failed_kept" in [s["name"] for s in _steps(run2)]
    # a failing canary: no eval runs, no DONE
    run3 = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-low-202611110000"
    res = _bash([str(payload), "Q3S", "low", str(run3)], dict(env, FAKE_ADAPTER_RC="0", FAKE_CANARY_RC="1"), tmp_path)
    assert res.returncode == 1
    names = [s["name"] for s in _steps(run3)]
    assert "eval_0W:canary_failed" in names and "eval_100:canary_failed" in names and not (run3 / "DONE").exists()
    assert "official_0W_c:no_results" in names


def test_nondry_eval_stale_kept_rerun_and_unreadable(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_eval.sh")
    P = tmp_path / "OUT" / "tvr" / "runs" / "pilot" / "Q3S"
    env = dict(env, FAKE_ADAPTER_RC="0", FAKE_PARITY_RC="0")

    def results():
        for arm in ("c", "u", "blind"):
            wj(P / "evals" / f"pilot__Q3S__ALL__{arm}" / "results.json", {"valid": True, "arm": arm})

    results()
    # stale (served other weights): kept, flagged, no eval, no DONE
    run = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-top-202611110100"
    res = _bash([str(payload), "Q3S", "top", str(run)], dict(env, FAKE_EVALOK_RC="3"), tmp_path)
    assert res.returncode == 1 and not (run / "DONE").exists()
    st = {s["name"]: s["rc"] for s in _steps(run)}
    assert st["eval_ALL:stale_kept"] == 1 and "eval_ALL" not in st
    assert "run_tvr eval" not in (tmp_path / "fake.log").read_text()
    assert (P / "evals" / "pilot__Q3S__ALL__c" / "results.json").is_file()
    # VPD_RERUN_STALE=1: every stale arm moved aside, the session runs
    run = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-top-202611110101"
    res = _bash([str(payload), "Q3S", "top", str(run)], dict(env, FAKE_EVALOK_RC="3", VPD_RERUN_STALE="1"), tmp_path)
    names = [s["name"] for s in _steps(run)]
    assert all(f"stale_aside_ALL_{a}" in names for a in ("c", "u", "blind")) and "eval_ALL" in names
    assert len(list((P / "evals").glob("pilot__Q3S__ALL__c.stale-*"))) == 1
    assert "--tier ALL --arms c,u,blind" in (tmp_path / "fake.log").read_text()
    # eval-ok crashed / unreadable results.json (rc 1): recorded with its rc and kept, even with VPD_RERUN_STALE=1
    results()
    (tmp_path / "fake.log").write_text("")
    run = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-top-202611110102"
    res = _bash([str(payload), "Q3S", "top", str(run)], dict(env, FAKE_EVALOK_RC="1", VPD_RERUN_STALE="1"), tmp_path)
    names = [s["name"] for s in _steps(run)]
    assert res.returncode == 1 and "eval_ALL_c:evalok_rc1" in names and "eval_ALL:stale_kept" in names
    assert not any(n.startswith("stale_aside") for n in names) and "run_tvr eval" not in (
        tmp_path / "fake.log").read_text()


def test_nondry_term_reaps_cleans_merged_and_exits_143(tmp_path):
    """SIGTERM while a step runs: the trap reaps and exits 143 when the step returns; the EXIT handlers remove the
    job-local merged folder; no later step, no DONE."""
    import signal
    import time

    payload, env = fake_tree(tmp_path, "tvr_eval.sh")
    if not os.access("/dev/shm", os.W_OK):
        pytest.skip("/dev/shm not writable here")
    merged = Path(env["VPD_MERGE_DIR"]) / "pilot__Q3S__ALL"
    shm_root_existed = merged.parent.parent.exists()  # /dev/shm/vpd_merged on this host
    try:
        run = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-top-202611110200"
        e = {k: v for k, v in os.environ.items() if k not in _KNOBS}
        e.update(env, FAKE_ADAPTER_RC="0", FAKE_SLEEP_ON="run_tvr eval", FAKE_SLEEP_S="4", FAKE_MERGE_WRITES="1")
        p = subprocess.Popen(["bash", str(payload), "Q3S", "top", str(run)], env=e, cwd=str(tmp_path),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log = tmp_path / "fake.log"
        t0 = time.time()
        while time.time() - t0 < 60 and not (log.is_file() and "run_tvr eval" in log.read_text()):
            time.sleep(0.2)
        assert "run_tvr eval" in log.read_text() and (merged / "model.safetensors").is_file()
        os.kill(p.pid, signal.SIGTERM)
        out, err = p.communicate(timeout=60)
        assert p.returncode == 143, (out[-2000:], err[-2000:])
        assert "TERM: reaping" in out and not (run / "DONE").exists()
        names = [s["name"] for s in _steps(run)]
        assert not any(n.startswith(("official_", "expect")) for n in names), names
        assert not merged.exists()
    finally:
        shutil.rmtree(merged.parent, ignore_errors=True)
        if not shm_root_existed:  # the test created it: leave the host's /dev/shm as it was
            try:
                merged.parent.parent.rmdir()
            except OSError:
                pass


def test_nondry_deadline_rc4_then_no_time(tmp_path):
    """run_tvr stopping at its deadline (exit 4) is a failed step; the next session has no time left (:no_time); no
    DONE.  Soft deadline 310 s: the canary needs >= 300 s (240 THOR reserve + 60), the 0W eval sleeps 12 s (so at
    most 298 s are left for the 100 eval even when every other step takes no time)."""
    payload, env = fake_tree(tmp_path, "tvr_eval.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-eval-q3s-low-202611110300"
    res = _bash([str(payload), "Q3S", "low", str(run)],
                dict(env, FAKE_ADAPTER_RC="0", VPD_JOB_DEADLINE_S="310", VPD_MIN_STEP_S="60",
                     FAKE_SLEEP_ON="--tier 0W --arms", FAKE_SLEEP_S="12", FAKE_FAIL="--tier 0W --arms",
                     FAKE_FAIL_RC="4"), tmp_path)
    assert res.returncode == 1 and not (run / "DONE").exists()
    st = {s["name"]: s["rc"] for s in _steps(run)}
    assert st["canary_replay"] == 0 and st["eval_0W"] == 4 and st["eval_100:no_time"] == 1, st
    assert "eval_100" not in st and "--tier 100 " not in (tmp_path / "fake.log").read_text()


def test_nondry_guard_failure_skips_model_steps(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_mech_smoke.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-msmoke-202611110000"
    res = _bash([str(payload), str(run)], dict(env, FAKE_GUARD_RC="2"), tmp_path)
    assert res.returncode == 1
    names = [s["name"] for s in _steps(run)]
    assert names == ["node", "prereg_committed", "guard", "steps:guard_failed", "expect"]
    assert (tmp_path / "OUT" / "tvr" / "runs" / "smoke" / run.name).is_dir()


def test_nondry_data_runs_every_step(tmp_path):
    payload, env = fake_tree(tmp_path, "tvr_data.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-data-202611110000"
    res = _bash([str(payload), str(run)], env, tmp_path)
    assert res.returncode == 0, res.stderr + res.stdout
    names = [s["name"] for s in _steps(run)]
    assert names == ["node", "guard", "dev_select", "houses", "reach_dump", "reap_reach_dump", "replay",
                     "reap_replay", "demos_build", "tiers", "bfs", "floors", "expect"]
    log = (tmp_path / "fake.log").read_text()
    assert "--envs 5 --lp 2 --profile envlib" in log
    res = _bash([str(payload), str(run)], dict(env, FAKE_FAIL="reach dump"), tmp_path)
    assert res.returncode == 1 and not (run / "DONE").exists()


# ------------------------------------------------------------------------------------------ CLIs
@pytest.mark.parametrize("mod", ["pilot_tvr", "gate_tvr", "recheck_tvr", "merge_parity"])
def test_cli_help(mod):
    res = subprocess.run([sys.executable, "-B", "-m", f"vpd.tvr.{mod}", "--help"], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(paths.WORKTREE)), cwd=str(paths.WORKTREE), timeout=120)
    assert res.returncode == 0 and "usage" in res.stdout.lower(), res.stderr


def test_cfg_and_served_helpers(capsys):
    assert PT.main(["cfg", "protocol.max_model_len", "render.build"]) == 0
    assert capsys.readouterr().out.split() == ["32768", pilot_cfg.FROZEN["render"]["build"]]
    rc = PT.main(["cfg", "render.lp_threads"])
    assert (rc == 2) == (pilot_cfg.FROZEN["render"]["lp_threads"] is None)
