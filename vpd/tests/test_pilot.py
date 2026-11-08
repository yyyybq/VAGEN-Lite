"""Tests of WP-WBPILOT: vpd.walkerbench.pilot_gate (G0-G3 of PREREG section 9 on synthetic results),
vpd.walkerbench.pilot (prepare refusals, optional-eval switch, estimates, collect/delete guards, adapter-ok, eval-ok,
smoke verdict, B5 request building) and the payloads vpd/jobs/wb_smoke.sh, wb_pilot_start.sh (bash -n, VPD_DRYRUN=1).

Login node, 1 core, no GPU, no network, no sco (vpd.sco.prepare / submit / delete are replaced by fakes).
Scratch: eval_external_assets/vpd/scratch/wbpilot/pytest_<rand>/ (env VPD_WBPILOT_TEST_DIR overrides), removed at
teardown.
    PYTHONPATH=$WT OMP_NUM_THREADS=1 PY_EVAL -m pytest -q -p no:cacheprovider vpd/tests/test_pilot.py
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from vpd import paths
from vpd.walkerbench import pilot as P
from vpd.walkerbench import pilot_gate as G

JOBS_DIR = paths.VPD_PKG / "jobs"
NAV = [f"nav_{i:04d}" for i in range(85)]
VIS = [f"vis_{i:04d}" for i in range(94)]
T0 = datetime.datetime(2026, 11, 8, 12, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture
def base():
    root = Path(os.environ.get("VPD_WBPILOT_TEST_DIR") or paths.scratch("wbpilot"))
    d = root / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def wj(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p


# --- synthetic pilot tree -----------------------------------------------------------------------------------

def baselines(nav_odo=24, vis_fwd=3) -> dict:
    odo = {s: {"success": int(i < nav_odo)} for i, s in enumerate(NAV)}
    odo.update({s: {"success": 0} for s in VIS})
    fwd = {s: {"success": int(i < vis_fwd)} for i, s in enumerate(VIS)}
    fwd.update({s: {"success": 0} for s in NAV})
    return {"floors": {"nav_ODO": {"k": nav_odo, "n": 85, "rate": nav_odo / 85},
                       "vis_FWD": {"k": vis_fwd, "n": 94, "rate": vis_fwd / 94},
                       "pooled_F": {"k": nav_odo + vis_fwd, "n": 179, "rate": (nav_odo + vis_fwd) / 179}},
            "policies": {"ODO": {"per_task": odo}, "FWD": {"per_task": fwd}},
            "survey_expected": {"per_kind": {"ODO": {"nav": 24}, "FWD": {"vis": 3}}}, "warnings": []}


def results(start, tier, res, mode, nav_k, vis_k, mismatch=0, tokens=None, parse_valid=99.0, no_log=0, mm=None):
    rows = [{"stem": s, "kind": "nav", "success": int(i < nav_k)} for i, s in enumerate(NAV)]
    rows += [{"stem": s, "kind": "vis", "success": int(i < vis_k)} for i, s in enumerate(VIS)]
    calls = 4000
    r = {"n_tasks": 179, "task_set": "s2test", "temperature": 0.0, "max_tokens": 512, "max_images": 80,
         "history_turns": 0, "blind": False, "lenient": False, "limit": None,
         "render_size": G.RES[res], "constrained": mode, "start": start,
         "pooled": {"k": nav_k + vis_k, "n": 179}, "tasks_without_log": no_log,
         "regex": {"answered": calls, "constrained_calls": 0 if mode == "off" else calls,
                   "regex_mismatch": 0 if mode == "off" else mismatch},
         "image_tokens": {"tokens_per_image": [tokens or G.TOKENS[res]], "prompt_parity": {"equal": True}},
         "parse_valid_pct": parse_valid, "invalid_move_crash": 1 if mode == "off" else 0, "mean_steps": 20.0,
         "adapter": None if tier == "raw" else {"tier": G.PHASE[tier], "start": start, "seed": 1001,
                                                "adapter_sha256": hashlib.sha256(ad_bytes(start, tier)).hexdigest()},
         "model_dir": None if tier == "raw" else f"/tmp/vpd_merged/{start}__{tier}",
         "server": {"mm_processor_kwargs": mm if mm is not None else
                    {"max_pixels": 1024000, "size": {"longest_edge": 1024000}}, "cfg": {"max_images": 80}}}
    return r, rows


def write_eval(root: Path, start, tier, res, mode, r, rows, invalid=False):
    d = G.eval_dir(root, start, tier, res, mode)
    d.mkdir(parents=True, exist_ok=True)
    (d / ("results_invalid.json" if invalid else "results.json")).write_text(json.dumps(r))
    (d / "per_task.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


def ad_bytes(start, tier) -> bytes:
    """The synthetic adapter.pt of START / TIER (distinct per tier, so a served sha can be checked)."""
    return f"{tier}-{start}".encode()


def write_adapters(root: Path, start, code="c0de"):
    ad = root / start / "adapters"
    (ad / "0W").mkdir(parents=True, exist_ok=True)
    (ad / "0W" / "adapter.pt").write_bytes(ad_bytes(start, "0W"))
    sha0 = hashlib.sha256(ad_bytes(start, "0W")).hexdigest()
    for tier in ("0W", "50", "200"):
        (ad / tier).mkdir(parents=True, exist_ok=True)
        if tier != "0W":
            (ad / tier / "adapter.pt").write_bytes(ad_bytes(start, tier))
        wj(ad / tier / "meta.json", {"start": start, "seed": 1001, "phase": G.PHASE[tier], "attn": "sdpa",
                                     "attn_sites": {"config": "sdpa", "model.visual.config": "sdpa"},
                                     "max_pixels": 1024000, "optimizer_steps": 30, "optimizer_steps_planned": 30,
                                     "code_sha256": code, "init_adapter": None if tier == "0W" else
                                     {"sha256": sha0, "dir": str(ad / "0W")}})


def smoke_ok() -> dict:
    return {"verdict": "PASS", "failed": [],
            "checks": [{"name": n, "ok": True, "detail": None} for n in
                       ("hf_parity", "vllm_parity", "render_parity", "train_max_pixels")]}


def make_pilot(root: Path, tier200=None, overrides=None, optional=False):
    """Three starts that pass every gate unless tier200 / overrides say otherwise.
    tier200: {start: (nav_k, vis_k)}; overrides: {(start, key): dict of results() kwargs}."""
    tier200 = tier200 or {}
    overrides = overrides or {}
    pr = root / "pilot"
    for s in G.STARTS:
        write_adapters(pr, s)
        for key, tier, res, mode, mand in G.EVALS:
            if not mand and not optional:
                continue
            nav_k, vis_k = tier200.get(s, (40, 20)) if tier == "200" else (10, 2)
            kw = dict(overrides.get((s, key), {}))
            nav_k, vis_k = kw.pop("nav_k", nav_k), kw.pop("vis_k", vis_k)
            if kw.pop("missing", False):
                continue
            invalid = kw.pop("invalid", False)
            r, rows = results(s, tier, res, mode, nav_k, vis_k, **kw)
            write_eval(pr, s, tier, res, mode, r, rows, invalid)
    wj(root / "baselines.json", baselines())
    wj(root / "smoke_verdict.json", smoke_ok())
    return pr


def gate(root: Path) -> dict:
    return G.evaluate(root / "pilot", root / "baselines.json", root / "smoke_verdict.json")


# --- pilot_gate ---------------------------------------------------------------------------------------------

def test_wilson_thresholds_of_prereg_example():
    nav, vis = 24 / 85, 3 / 94
    assert G.wilson_lb(32, 85) < nav < G.wilson_lb(33, 85)  # PREREG: >= 33/85 needed (LB 29.2%)
    assert round(100 * G.wilson_lb(33, 85), 1) == 29.2
    assert G.wilson_lb(6, 94) < vis < G.wilson_lb(7, 94)  # PREREG: >= 7/94 needed (LB 3.7%)
    assert round(100 * G.wilson_lb(7, 94), 1) == 3.7
    assert G.wilson_lb(0, 0) is None


def test_gate_all_pass_and_md(base):
    make_pilot(base)
    rep = gate(base)
    assert rep["gates"] == {"G0": "pass", "G1": "pass", "G2": "pass", "G3": "pass"}, \
        [c for c in rep["g0_checks"] + rep["g2_checks"] if c["ok"] is not True]
    assert rep["decision"].startswith("WalkerBench 过闸门")
    md = G.render_md(rep)
    assert "40/85" in md and "20/94" in md and "24/85" in md
    # absolute numbers only: no start-vs-start difference anywhere (no such key, no such text)
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(str(k).lower())
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(rep)
    assert not [k for k in keys if any(w in k for w in ("diff", "delta", "minus", "relative", "contrast"))]
    assert "差" not in md.replace("不算起点之间的差", "")


def test_g1_around_the_floors(base):
    # nav exactly at the boundary: 32/85 fails, 33/85 passes (floor 24/85); vis 6/94 fails, 7/94 passes (3/94)
    make_pilot(base, tier200={s: (32, 7) for s in G.STARTS})
    assert gate(base)["gates"]["G1"] == "fail"
    shutil.rmtree(base / "pilot")
    make_pilot(base, tier200={s: (33, 6) for s in G.STARTS})
    assert gate(base)["gates"]["G1"] == "fail"
    shutil.rmtree(base / "pilot")
    make_pilot(base, tier200={"Q3B": (33, 6), "Q3S": (32, 7), "E2": (0, 0)})  # both sides, but not one start
    assert gate(base)["gates"]["G1"] == "fail"
    shutil.rmtree(base / "pilot")
    make_pilot(base, tier200={"Q3B": (0, 0), "Q3S": (33, 7), "E2": (0, 0)})  # one start suffices
    rep = gate(base)
    assert rep["gates"]["G1"] == "pass" and rep["g1_per_start"]["Q3S"]["ok"] is True
    assert rep["g1_per_start"]["Q3B"]["ok"] is False


def test_g1_needs_the_floor_tasks(base):
    make_pilot(base)
    b = baselines()
    b["policies"]["ODO"]["per_task"].pop(NAV[0])
    b["policies"]["ODO"]["per_task"]["nav_9999"] = {"success": 0}
    wj(base / "baselines.json", b)
    rep = gate(base)
    assert rep["gates"]["G1"] == "incomplete"
    assert rep["g1_per_start"]["Q3B"]["nav"]["same_tasks"] is False


def test_g2_mechanism_and_unconstrained_not_blocking(base):
    # the old 'parse-valid >= 95%' rule is not a gate: unconstrained #5 at 94.9 or 95.0 never blocks (B3)
    for pv in (94.9, 95.0):
        make_pilot(base, overrides={(s, "5"): {"parse_valid": pv} for s in G.STARTS})
        rep = gate(base)
        assert rep["gates"]["G2"] == "pass"
        assert rep["secondary_unconstrained_200"]["Q3B"]["parse_valid_pct"] == pv
        shutil.rmtree(base / "pilot")
    # one regex mismatch in any constrained eval of any start fails G2
    make_pilot(base, overrides={("E2", "3"): {"mismatch": 1}})
    rep = gate(base)
    assert rep["gates"]["G2"] == "fail"
    assert [c["name"] for c in rep["g2_checks"] if c["ok"] is False] == ["g2_E2_#3"]
    shutil.rmtree(base / "pilot")
    # a task without a log in a constrained eval fails G2 (and G0)
    make_pilot(base, overrides={("Q3S", "2"): {"no_log": 1}})
    rep = gate(base)
    assert rep["gates"]["G2"] == "fail" and rep["gates"]["G0"] == "fail"


def test_g3_threshold(base):
    make_pilot(base, tier200={"Q3B": (85, 58)})  # 143/179 = 79.9% <= 80%
    assert gate(base)["gates"]["G3"] == "pass"
    shutil.rmtree(base / "pilot")
    make_pilot(base, tier200={"Q3B": (85, 59)})  # 144/179 = 80.4%
    rep = gate(base)
    assert rep["gates"]["G3"] == "fail"
    assert "饱和" in rep["decision"]


def test_g0_failures(base):
    make_pilot(base, overrides={("Q3S", "2"): {"tokens": 1003}})
    assert gate(base)["gates"]["G0"] == "fail"
    shutil.rmtree(base / "pilot")
    make_pilot(base)
    write_adapters(base / "pilot", "E2", code="other")  # B7: 0W code sha differs across starts
    rep = gate(base)
    assert rep["gates"]["G0"] == "fail"
    assert next(c for c in rep["g0_checks"] if c["name"] == "code_sha_0W_equal")["ok"] is False
    assert rep["decision"].startswith("G0 不过")
    shutil.rmtree(base / "pilot")
    make_pilot(base, overrides={("Q3B", "1"): {"invalid": True}})
    assert gate(base)["gates"]["G0"] == "fail"
    shutil.rmtree(base / "pilot")
    make_pilot(base)
    (base / "pilot" / "Q3B" / "adapters" / "0W" / "adapter.pt").write_bytes(b"changed")  # 50/200 not from this 0W
    assert gate(base)["gates"]["G0"] == "fail"
    shutil.rmtree(base / "pilot")
    make_pilot(base)
    (base / "pilot" / "Q3S" / "adapters" / "200" / "adapter.pt").write_bytes(b"retrained")  # #4/#5 now stale
    rep = gate(base)
    assert rep["gates"]["G0"] == "fail"
    stale = {c["name"] for c in rep["g0_checks"] if c["ok"] is False}
    assert stale == {"eval_Q3S_#4", "eval_Q3S_#5"}
    assert any("served adapter sha" in b for b in next(c for c in rep["g0_checks"]
                                                      if c["name"] == "eval_Q3S_#4")["detail"])
    shutil.rmtree(base / "pilot")
    make_pilot(base)
    sv = smoke_ok()
    sv["checks"][2]["ok"] = False  # render parity (B17)
    sv["verdict"], sv["failed"] = "FAIL", ["render_parity"]
    wj(base / "smoke_verdict.json", sv)
    assert gate(base)["gates"]["G0"] == "fail"
    shutil.rmtree(base / "pilot")
    make_pilot(base)
    b = baselines()
    b["warnings"] = ["ODO floor differs from the survey"]  # a floor file with warnings is not a valid floor
    wj(base / "baselines.json", b)
    rep = gate(base)
    assert rep["gates"]["G0"] == "fail"
    assert {c["name"] for c in rep["g0_checks"] if c["ok"] is False} == {"baselines_valid"}
    shutil.rmtree(base / "pilot")
    make_pilot(base, overrides={("E2", "3"): {"mm": {"max_pixels": 1024000}}})  # size.longest_edge missing (B16)
    rep = gate(base)
    assert {c["name"] for c in rep["g0_checks"] if c["ok"] is False} == {"eval_E2_#3"}


def test_g0_incomplete_and_optional(base):
    make_pilot(base, overrides={("Q3S", "5"): {"missing": True}})
    rep = gate(base)
    assert rep["gates"]["G0"] == "incomplete" and "不完整" in rep["decision"]
    shutil.rmtree(base / "pilot")
    make_pilot(base)  # optional #6 absent: fine
    assert gate(base)["gates"]["G2"] == "pass"
    shutil.rmtree(base / "pilot")
    make_pilot(base, optional=True, overrides={("Q3B", "6"): {"mismatch": 2}})  # if #6 runs it is gated (G2)
    assert gate(base)["gates"]["G2"] == "fail"


def test_gate_cli_writes_files(base):
    make_pilot(base)
    rc = G.main(["--pilot-root", str(base / "pilot"), "--baselines", str(base / "baselines.json"),
                 "--smoke-verdict", str(base / "smoke_verdict.json")])
    assert rc == 0
    rep = json.loads((base / "pilot" / "GATE.json").read_text())
    assert rep["gates"]["G1"] == "pass"
    assert (base / "pilot" / "GATE.md").read_text(encoding="utf-8").startswith("# WalkerBench 试点闸门结果")


# --- pilot: layout, prepare refusals, optional switch -------------------------------------------------------

@pytest.fixture
def lay(base, monkeypatch):
    L = P.Layout(base / "out")
    monkeypatch.setattr(P, "LAYOUT", L)
    monkeypatch.setenv("VPD_LEDGER", str(base / "out" / "ledger.jsonl"))
    monkeypatch.delenv("VPD_GPUH_CAP", raising=False)
    calls = {"prepare": [], "submit": [], "delete": []}
    from vpd import sco as vsco

    def fake_prepare(run, name, payload, pool="zoetrope", as_root=False, est_gph=None, gate=None):
        job = Path(run) / "job"
        job.mkdir(parents=True)
        (job / "prepared.json").write_text(json.dumps({"name": name, "payload": payload}))
        calls["prepare"].append({"run": str(run), "name": name, "payload": payload, "pool": pool,
                                 "as_root": as_root, "est_gph": est_gph, "gate": gate})
        return job / "launch.sh"

    def fake_submit(run, est_gph=None):
        (Path(run) / "job" / "submission.json").write_text(json.dumps({"job_id": "pt-fake1"}))
        calls["submit"].append(str(run))
        return "pt-fake1"

    def fake_delete(run, force_live=False):
        calls["delete"].append(str(run))
        return "deleted"

    monkeypatch.setattr(vsco, "prepare", fake_prepare)
    monkeypatch.setattr(vsco, "submit", fake_submit)
    monkeypatch.setattr(vsco, "delete", fake_delete)
    L.calls = calls
    return L


def git_fake(monkeypatch, log="abc123", status=""):
    def _git(*args):
        if args[0] == "log":
            return log
        if args[0] == "status":
            return status
        return None

    monkeypatch.setattr(P, "_git", _git)


def build_data(L: P.Layout):
    for p in (L.split, L.order(1001)):
        wj(p, {})
    L.demos_dir.mkdir(parents=True, exist_ok=True)
    L.demos_done.write_text("x")
    L.demos_all.write_text("")
    L.warmup.parent.mkdir(parents=True, exist_ok=True)
    L.warmup.write_text("")
    for n in (50, 200):
        p = L.tier(n)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps({"id": f"d{i}"}) + "\n" for i in range(n)))
    wj(L.baselines, baselines())
    L.s2test_tasks.mkdir(parents=True, exist_ok=True)


MEASURED = {"probe_tokens_per_s": 5000.0, "probe_load_s": 60.0, "zeroW_s": 180.0,
            "tier_tokens": {"50": 400_000, "200": 1_600_000}, "tier_epochs": {"50": 5, "200": 3},
            "merge_s": 240.0, "step_cap_total": 4582, "n_tasks": 179,
            "evals": {"640_empty": {"sps": 0.2, "mean_steps": 20.0, "ready_s": 150.0},
                      "1280_empty": {"sps": 0.4, "mean_steps": 20.0, "ready_s": 150.0},
                      "640_off": {"sps": 0.2, "mean_steps": 20.0, "ready_s": 150.0},
                      "640_merged": {"sps": 0.2, "mean_steps": 22.0, "ready_s": 150.0}}}


def write_verdict(L: P.Layout, verdict="PASS", measured=None, name="wb-smoke-202611081200"):
    return wj(L.smoke_root / name / "smoke_verdict.json",
              {"verdict": verdict, "failed": [] if verdict == "PASS" else ["eval_a"],
               "measured": MEASURED if measured is None else measured})


def test_estimate_from_smoke_math():
    e = P.estimate_from_smoke(MEASURED)
    tr = e["parts_s"]["train"]
    assert tr["0W"] == 180.0
    assert tr["50"] == 60.0 + 400_000 * 5 / 5000.0 == 460.0
    assert tr["200"] == 60.0 + 1_600_000 * 3 / 5000.0 == 1020.0
    steps_high = min(4582, 1.5 * 179 * 22.0)
    assert e["parts_s"]["steps_high"] == round(steps_high, 1)
    ev = MEASURED["evals"]

    def lo_e(k):
        return ev[k]["ready_s"] + 179 * ev[k]["mean_steps"] * ev[k]["sps"]

    def hi_e(k):
        return ev[k]["ready_s"] + steps_high * ev[k]["sps"]

    keys = ["640_empty", "640_merged", "640_off", "640_merged", "1280_empty"]  # raw640 200c 200u 50c raw1280
    lo = (180 + 460 + 1020 + 2 * 240) + sum(lo_e(k) for k in keys)
    hi = 1.25 * (180 + 460 + 1020 + 480) + sum(hi_e(k) for k in keys)
    assert e["per_start"]["mandatory"] == (round(lo / 3600 + 0.05, 3), round(hi / 3600 + 0.1, 3))
    assert e["per_start"]["optional"] == (round((240 + lo_e("640_merged")) / 3600, 3),
                                          round((1.25 * 240 + hi_e("640_merged")) / 3600, 3))
    with pytest.raises(ValueError):
        P.estimate_from_smoke({k: v for k, v in MEASURED.items() if k != "merge_s"})


def test_prepare_refuses_without_smoke_verdict_or_committed_prereg(lay, monkeypatch):
    build_data(lay)
    git_fake(monkeypatch)
    with pytest.raises(P.Refused, match="no smoke_verdict.json"):
        P.prepare("wb-q3b", L=lay)
    write_verdict(lay, "FAIL")
    with pytest.raises(P.Refused, match="not PASS"):
        P.prepare("wb-q3b", L=lay)
    write_verdict(lay, "PASS", name="wb-smoke-202611081300")  # the newest verdict counts
    git_fake(monkeypatch, log="", status="?? vpd/PREREG_WB_PILOT.md")
    with pytest.raises(P.Refused, match="is not committed"):
        P.prepare("wb-q3b", L=lay)
    git_fake(monkeypatch, log="abc123", status=" M vpd/PREREG_WB_PILOT.md")
    with pytest.raises(P.Refused, match="uncommitted changes"):
        P.prepare("wb-q3b", L=lay)
    assert lay.calls["prepare"] == []
    git_fake(monkeypatch)
    launch = P.prepare("wb-q3b", now=T0, L=lay)
    c = lay.calls["prepare"][-1]
    assert Path(c["run"]).name == "wb-q3b-202611081200" and launch.name == "launch.sh"
    assert c["name"] == "vpd-wb-q3b" and c["as_root"] is True and c["gate"] == "wb"
    assert c["payload"][:2] == ["env", "VPD_OPTIONAL=1"] and c["payload"][-2:] == ["Q3B", c["run"]]
    info = json.loads((Path(c["run"]) / "pilot.json").read_text())
    assert info["prereg"]["commit"] == "abc123" and info["optional"] == 1
    est = P.estimate_from_smoke(MEASURED)["per_start"]
    assert c["est_gph"] == round(est["mandatory"][1] + est["optional"][1], 3)
    # the time limits follow the est_high that is submitted (with the optional evals), not a fixed 5 h
    outer = int(round(c["est_gph"] * 3600, 6)) - 300
    assert c["payload"][2] == f"VPD_JOB_DEADLINE_S={outer - 180}"
    assert c["payload"][3:7] == ["timeout", "-k", "120", str(outer)]
    assert info["timeouts"]["outer_s"] == outer and info["timeouts"]["kill_s"] == 120
    assert outer + 120 <= c["est_gph"] * 3600 - 180
    with pytest.raises(P.Refused, match="prepared and not submitted"):  # no second run dir by accident
        P.prepare("wb-q3b", L=lay)


def test_prepare_smoke_and_demos_prerequisites(lay, monkeypatch):
    git_fake(monkeypatch)
    with pytest.raises(P.Refused, match="WP-WBDATA"):
        P.prepare("wb-demos", L=lay)
    with pytest.raises(P.Refused, match="job wb-demos"):
        P.prepare("wb-smoke", L=lay)
    build_data(lay)
    with pytest.raises(P.Refused, match="already built"):
        P.prepare("wb-demos", L=lay)
    shutil.rmtree(lay.root / "walkerbench" / "runs", ignore_errors=True)
    P.prepare("wb-smoke", now=T0, L=lay)  # no PREREG / verdict needed for the smoke
    # the shared parents exist (made by the login user) before the as_root smoke job could create them
    assert lay.smoke_root.is_dir() and lay.pilot_root.is_dir()
    assert lay.smoke_root.parent == lay.pilot_root.parent == lay.root / "walkerbench" / "runs"
    c = lay.calls["prepare"][-1]
    assert c["payload"][:6] == ["env", "VPD_SMOKE_DEADLINE_S=3120", "timeout", "-k", "60", "3300"]
    assert c["payload"][-1] == c["run"]
    assert c["as_root"] is True and c["est_gph"] == 1.0


def test_optional_switch(lay, monkeypatch):
    # the optional ceiling is the WB reserve itself (DECISIONS B2, PREREG 10.1); no extra 9 GPU-h rule
    assert P.OPTIONAL_MAX_PROJECTION is None
    assert P.projection(P.estimates(lay), [], L=lay)["optional_max"] == 12.0
    monkeypatch.setattr(P, "OPTIONAL_MAX_PROJECTION", 9.0)  # an extra ceiling would only ever lower it
    assert P.projection(P.estimates(lay), [], L=lay)["optional_max"] == 9.0
    monkeypatch.setattr(P, "OPTIONAL_MAX_PROJECTION", None)
    proj = {"wb_cap": 12.0, "optional_max": 12.0}
    assert P.decide_optional(dict(proj, mandatory_high=8.0, with_optional_high=12.0))["optional"] == 1
    d = P.decide_optional(dict(proj, mandatory_high=8.0, with_optional_high=12.01))
    assert d["optional"] == 0 and not d["stop"]
    d = P.decide_optional(dict(proj, mandatory_high=12.0, with_optional_high=13.0))
    assert d["optional"] == 0 and not d["stop"]
    d = P.decide_optional(dict(proj, mandatory_high=12.01, with_optional_high=13.0))
    assert d["stop"] and "hand to user" in d["reason"]
    # end to end: a slow smoke pushes the projection with optional over the 12 GPU-h reserve (mandatory alone
    # stays under it) -> VPD_OPTIONAL=0, decided once for all starts
    build_data(lay)
    git_fake(monkeypatch)
    slow = copy.deepcopy(MEASURED)
    for e in slow["evals"].values():
        e["sps"] = 0.42
    write_verdict(lay, measured=slow)
    est = P.estimates(lay)
    pj = P.projection(est, [], L=lay)
    assert pj["pending"] == ["wb-q3b", "wb-q3s", "wb-e2"]  # demos DONE, smoke verdict present
    assert pj["mandatory_high"] <= 12.0 < pj["with_optional_high"] and pj["optional_max"] == 12.0
    P.prepare("wb-q3s", now=T0, L=lay)
    assert lay.calls["prepare"][-1]["payload"][1] == "VPD_OPTIONAL=0"
    dec = json.loads(lay.optional_decision.read_text())
    assert dec["optional"] == 0
    write_verdict(lay, name="wb-smoke-202611081400")  # a faster later smoke does not flip the stored decision
    P.prepare("wb-e2", now=T0, L=lay)
    assert lay.calls["prepare"][-1]["payload"][1] == "VPD_OPTIONAL=0"
    # mandatory alone over the WB reserve: stop, nothing prepared
    huge = copy.deepcopy(MEASURED)
    for e in huge["evals"].values():
        e["sps"] = 5.0
    write_verdict(lay, measured=huge, name="wb-smoke-202611081500")
    n = len(lay.calls["prepare"])
    with pytest.raises(P.Refused, match="hand to user"):
        P.prepare("wb-q3b", redecide=True, L=lay)
    with pytest.raises(P.Refused, match="hand to user"):  # also when a stored decision exists
        P.prepare("wb-q3b", L=lay)
    assert len(lay.calls["prepare"]) == n


def test_submit_rechecks(lay, monkeypatch):
    build_data(lay)
    git_fake(monkeypatch)
    write_verdict(lay)
    with pytest.raises(P.Refused, match="no prepared"):
        P.submit("wb-q3b", L=lay)
    P.prepare("wb-q3b", now=T0, L=lay)
    git_fake(monkeypatch, log="def456")
    with pytest.raises(P.Refused, match="commit changed"):
        P.submit("wb-q3b", L=lay)
    git_fake(monkeypatch)
    assert P.submit("wb-q3b", L=lay) == "pt-fake1"
    assert len(lay.calls["submit"]) == 1


def test_collect_and_delete_guard(lay, monkeypatch):
    build_data(lay)
    git_fake(monkeypatch)
    write_verdict(lay)
    P.prepare("wb-e2", now=T0, L=lay)
    P.submit("wb-e2", L=lay)
    with pytest.raises(P.Refused, match="collect.json missing"):
        P.delete("wb-e2", L=lay)
    rep = P.collect("wb-e2", L=lay)
    assert not rep["ok"] and len(rep["missing"]) == 6 + 6 + 1  # 3 adapters x 2 files, 6 evals (optional on), DONE
    with pytest.raises(P.Refused, match="missing outputs"):
        P.delete("wb-e2", L=lay)
    run = Path(lay.calls["submit"][-1])
    for p in P.expected_outputs("wb-e2", run, lay):
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.name == "results.json" and "__raw__1280" in str(p):
            (p.parent / "results_invalid.json").write_text("{}")  # present, but flagged
        else:
            p.write_text("{}")
    rep = P.collect("wb-e2", L=lay)
    assert rep["ok"] and len(rep["invalid_evals"]) == 1
    (lay.adapter("E2", "50") / "meta.json").unlink()
    with pytest.raises(P.Refused, match="vanished"):
        P.delete("wb-e2", L=lay)
    (lay.adapter("E2", "50") / "meta.json").write_text("{}")
    assert P.delete("wb-e2", L=lay) == "deleted"
    assert lay.calls["delete"] == [str(run)]


def test_delete_allow_incomplete(lay, monkeypatch):
    build_data(lay)
    git_fake(monkeypatch)
    P.prepare("wb-smoke", now=T0, L=lay)
    P.submit("wb-smoke", L=lay)
    assert not P.collect("wb-smoke", L=lay)["ok"]
    assert P.delete("wb-smoke", allow_incomplete=True, L=lay) == "deleted"


def test_adapter_check(base):
    from vpd.train import n2_train_vpd as T

    d = base / "ad"
    assert P.adapter_check(d)[0] == 1
    d.mkdir()
    (d / "adapter.pt").write_bytes(b"x")
    cur = T.code_fingerprint()["code_sha256"]
    wj(d / "meta.json", {"optimizer_steps": 10, "optimizer_steps_planned": 30, "code_sha256": cur})
    assert P.adapter_check(d)[0] == 1  # unfinished
    wj(d / "meta.json", {"optimizer_steps": 30, "optimizer_steps_planned": 30, "code_sha256": cur})
    assert P.adapter_check(d)[0] == 0
    wj(d / "meta.json", {"optimizer_steps": 30, "optimizer_steps_planned": 30, "code_sha256": "old"})
    assert P.adapter_check(d)[0] == 3
    assert P.main(["adapter-ok", str(d)]) == 3
    # --init: a 50/200 adapter must have started from the 0W adapter.pt now on disk, else it is retrained (1)
    z = base / "ad0W"
    z.mkdir()
    (z / "adapter.pt").write_bytes(b"0W-v1")
    sha_v1 = hashlib.sha256(b"0W-v1").hexdigest()
    wj(d / "meta.json", {"optimizer_steps": 30, "optimizer_steps_planned": 30, "code_sha256": cur,
                         "init_adapter": {"sha256": sha_v1}})
    assert P.adapter_check(d, z)[0] == 0 and P.main(["adapter-ok", str(d), "--init", str(z)]) == 0
    (z / "adapter.pt").write_bytes(b"0W-v2")  # 0W retrained since
    rc, msg = P.adapter_check(d, z)
    assert rc == 1 and "retrain" in msg
    assert P.main(["adapter-ok", str(d), "--init", str(z)]) == 1
    (z / "adapter.pt").unlink()
    assert P.adapter_check(d, z)[0] == 1
    wj(d / "meta.json", {"optimizer_steps": 30, "optimizer_steps_planned": 30, "code_sha256": "old",
                         "init_adapter": {"sha256": "zz"}})
    assert P.adapter_check(d, z)[0] == 3  # other code still stops the job first


def test_eval_check(base):
    ev, ad = base / "ev", base / "ad200"
    assert P.eval_check(ev)[0] == 1 and P.main(["eval-ok", str(ev)]) == 1
    ad.mkdir()
    (ad / "adapter.pt").write_bytes(b"200-v1")
    wj(ev / "results.json", {"adapter": None})
    assert P.eval_check(ev)[0] == 0  # raw eval: results.json is enough
    assert P.eval_check(ev, ad)[0] == 3  # merged eval without a served sha: not judged fresh
    wj(ev / "results.json", {"adapter": {"adapter_sha256": hashlib.sha256(b"200-v1").hexdigest()}})
    assert P.eval_check(ev, ad)[0] == 0 and P.main(["eval-ok", str(ev), "--adapter", str(ad)]) == 0
    (ad / "adapter.pt").write_bytes(b"200-v2")  # the tier was retrained after this eval
    rc, msg = P.eval_check(ev, ad)
    assert rc == 3 and "stale" in msg
    assert P.main(["eval-ok", str(ev), "--adapter", str(ad)]) == 3


def test_plan_prints_five_jobs(lay, monkeypatch, capsys):
    git_fake(monkeypatch)
    P.plan(L=lay)
    out = capsys.readouterr().out
    for job, spec in P.JOBS.items():
        assert f"[{job}] sco name downstream-{spec['name']}" in out
    assert "wb_pilot_start.sh Q3B" in out and "VPD_OPTIONAL=<0|1>" in out and "wb_smoke.sh" in out
    assert "est 2.05-3.40 GPU-h (+0.30-0.50 optional" in out
    assert "budget (ledger, gate wb): cap 12.00" in out


# --- smoke verdict ------------------------------------------------------------------------------------------

def build_smoke(L: P.Layout, smoke: Path, bad_regex=False, mm_no_size=False):
    build_data(L)
    for i in range(3):
        wj(L.s2test_tasks / f"t{i}.json", {"visual_path": [0] * 10})
    per = [{"id": f"d{i}", "tokens": [1000]} for i in range(200)]
    wj(smoke / "dry_200" / "dry_run.json", {"max_pixels": 1024000, "per_conversation": per})
    wj(smoke / "dry_200" / "parity.json", {"verdict": "PASS", "prefixes_checked": 31})
    wj(smoke / "vllm_parity.json", {"equal": True, "n": 9, "demos": ["a", "b", "c"], "rows": []})
    wj(smoke / "render_parity.json", {"equal": True, "per_size": [], "probes": []})
    sites = {"config": "sdpa", "model.language_model.config": "sdpa", "model.visual.config": "sdpa"}
    wj(smoke / "train_probe" / "probe_train.json", {"rc": 0, "optimizer_steps": 1, "attn_sites": sites,
                                                    "tokens_per_s": 5000.0, "load_seconds": 60.0,
                                                    "peak_mem_gib": 50.0})
    wj(smoke / "train_probe" / "probe_load_Q3RD.json", {"verdict": "PASS", "attn_sites": sites})
    wj(smoke / "adapters" / "Q3S_0W" / "meta.json", {"optimizer_steps": 30, "optimizer_steps_planned": 30,
                                                     "attn": "sdpa", "load_seconds": 60, "train_seconds": 120,
                                                     "lora": {"wrapped": 252}})
    wj(smoke / "merge" / "vpd_merge.json", {"merged": 252, "seconds": 240.0})
    for key, (model, tier, res, mode, k) in P.SMOKE_EVALS.items():
        calls = 150
        wj(P.smoke_eval_dir(smoke, key) / "results.json", {
            "n_tasks": 6, "tasks_without_log": 0, "max_images": 80, "history_turns": k, "constrained": mode,
            "render_size": [int(x) for x in P.RES[res].split("x")], "model_dir": "/tmp/m" if tier == "0W" else None,
            "regex": {"answered": calls, "constrained_calls": 0 if mode == "off" else calls,
                      "regex_mismatch": int(bad_regex and key == "a")},
            "image_tokens": {"tokens_per_image": [P.TOKENS_PER_IMAGE[res]], "prompt_parity": {"equal": True}},
            "server": {"cfg": {"max_images": 80}, "ready_seconds": 150.0,
                       "mm_processor_kwargs": {"max_pixels": 1024000} if mm_no_size and key == "b" else
                       {"max_pixels": 1024000, "size": {"longest_edge": 1024000, "shortest_edge": 65536}}},
            "seconds_per_step": 0.5, "mean_steps": 20.0, "wall_seconds": 60.0})


def test_smoke_verdict_pass_and_fail(lay, base):
    smoke = lay.smoke_root / "wb-smoke-202611081200"
    build_smoke(lay, smoke)
    v = P.smoke_verdict(smoke, L=lay, local_disk=())
    assert v["verdict"] == "PASS", v["failed"]
    m = v["measured"]
    assert m["tier_tokens"] == {"50": 50_000, "200": 200_000} and m["tier_epochs"] == {"50": 5, "200": 3}
    assert m["step_cap_total"] == 45 and m["zeroW_s"] == 180 and m["merge_s"] == 240.0
    assert "per_start" in v["estimate"]
    assert json.loads((smoke / "smoke_verdict.json").read_text())["verdict"] == "PASS"
    # the pilot estimates now come from the smoke
    assert P.estimates(lay)["source"] == "smoke"
    build_smoke(lay, smoke, bad_regex=True)
    (smoke / "vllm_parity.json").unlink()
    v = P.smoke_verdict(smoke, L=lay, local_disk=())
    assert v["verdict"] == "FAIL" and set(v["failed"]) == {"eval_a", "vllm_parity"}
    # B16: max_pixels alone does not cap in vLLM 0.11 / tf 4.57; size.longest_edge is required, as in pilot_gate
    build_smoke(lay, smoke, mm_no_size=True)
    v = P.smoke_verdict(smoke, L=lay, local_disk=())
    assert v["verdict"] == "FAIL" and set(v["failed"]) == {"eval_b"}


# --- B5 request building, B17 helpers -----------------------------------------------------------------------

def test_pick_turns():
    assert P.pick_turns([1, 3, 5], 4) == [1, 3, 5]
    t = P.pick_turns(list(range(1, 32, 2)), 4)
    assert len(t) == 4 and t[0] == 1 and t[1] == 3 and t[-1] == 31


def test_vllm_parity_with_fake_server(base):
    imgs = []
    for i in range(3):
        p = base / f"turn_{i:02d}.jpg"
        p.write_bytes(bytes([i]) * 10)
        imgs.append(str(p))
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(3):
        msgs += [{"role": "user", "content": f"step {i}\n<image>"},
                 {"role": "assistant", "content": '{"THOUGHT": "", "ACTION": "MOVE(1)"}'}]
    rec = {"id": "demo1", "messages": msgs, "images": imgs}
    data = base / "demos.jsonl"
    data.write_text(json.dumps(rec) + "\n")
    prefixes = [{"turn": 2, "n_images": 1, "prompt_tokens": 1044}, {"turn": 4, "n_images": 2, "prompt_tokens": 1578},
                {"turn": 6, "n_images": 3, "prompt_tokens": 2100}]
    dry = wj(base / "dry_run.json", {"start": "Q3S", "data": str(data), "history_k": 0, "blind": False,
                                     "parity": [{"id": "demo1", "prefixes": prefixes}]})
    sent = []

    def post(url, body):
        sent.append(body)
        n_img = sum(1 for m in body["messages"] if isinstance(m["content"], list)
                    for part in m["content"] if part["type"] == "image_url")
        return {"usage": {"prompt_tokens": {1: 1044, 2: 1578, 3: 2101}[n_img]}}

    class Srv:
        base_url = "http://fake/v1"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    rep = P.vllm_parity("Q3S", dry, base / "vp.json", demos=1, post=post, server=Srv())
    assert rep["n"] == 3 and [r["equal"] for r in rep["rows"]] == [True, True, False] and rep["equal"] is False
    b = sent[1]
    assert b["model"] == "c0q3_sft_e8" and b["max_tokens"] == 1 and b["messages"][0] == msgs[0]
    user = b["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "step 0\n"} and user[1]["type"] == "image_url"
    assert user[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert b["messages"][-1]["role"] == "user" and len(b["messages"]) == 4
    with pytest.raises(ValueError):
        P.vllm_parity("Q3B", dry, base / "vp2.json", post=post, server=Srv())


def test_step_cap_total(base):
    wj(base / "t" / "a.json", {"visual_path": [0] * 15})  # round(22.5) = 22
    wj(base / "t" / "b.json", {"max_steps": 12})
    wj(base / "t" / "c.json", {})
    assert P.step_cap_total(base / "t") == 22 + 12 + 30


# --- shell payloads -----------------------------------------------------------------------------------------

# knobs of the payloads that the caller's shell may have set (e.g. VPD_THREADS=1 in a 1-core test runner): the
# payload tests must see the payloads' own defaults
_PAYLOAD_KNOBS = ("VPD_THREADS", "VPD_OPTIONAL", "VPD_JOB_DEADLINE_S", "VPD_SMOKE_DEADLINE_S", "VPD_MIN_STEP_S",
                  "VPD_MERGE_DIR", "VPD_RERUN_STALE", "VPD_RERUN_INVALID", "VPD_DRYRUN", "RUN")


def _bash(args, env_extra, cwd):
    env = {k: v for k, v in os.environ.items() if k not in _PAYLOAD_KNOBS}
    env.update(env_extra)
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env, cwd=str(cwd), timeout=300)


def _timeout_of(payload):
    """(kill_s, outer_s, env dict) of a payload `[env K=V ...] timeout -k KILL OUTER bash ...`."""
    i = payload.index("timeout")
    assert payload[i + 1] == "-k" and payload[i + 4] == "bash"
    envs = dict(x.split("=", 1) for x in payload[1:i]) if payload[0] == "env" else {}
    return int(payload[i + 2]), int(payload[i + 3]), envs


def test_payload_timeouts_bound_the_charge(lay, monkeypatch):
    """Audit 2026-11-08 (B2): every payload's outer timeout + kill window <= the est_high it is submitted at,
    with >= 180 s left for container start / teardown, and the soft deadline 180 s before the outer timeout."""
    run = lay.jobs / "x-202611081200"
    cases = [(j, None, P.PLAN_EST[j][1]) for j in ("wb-demos", "wb-smoke")]
    est = P.estimate_from_smoke(MEASURED)["per_start"]
    for j in P.PILOT_JOBS:
        cases += [(j, 0, P.PLAN_EST["pilot"][1]), (j, 0, est["mandatory"][1]),
                  (j, 1, round(est["mandatory"][1] + est["optional"][1], 3)), (j, 1, 3.123)]
    for job, opt, hi in cases:
        kill, outer, envs = _timeout_of(P.payload_for(job, run, opt, hi))
        assert outer + kill <= hi * 3600 - 180, (job, hi, outer, kill)
        assert kill == (120 if job in P.PILOT_JOBS else 60)
        t = P.job_timeouts(job, hi)
        assert (t["outer_s"], t["kill_s"]) == (outer, kill)
        if job == "wb-demos":
            assert envs == {} and t["soft_deadline_s"] is None
        elif job == "wb-smoke":
            assert envs == {"VPD_SMOKE_DEADLINE_S": str(outer - 180)}
        else:
            assert envs == {"VPD_OPTIONAL": str(opt), "VPD_JOB_DEADLINE_S": str(outer - 180)}
    # the plan numbers of the audit
    assert _timeout_of(P.payload_for("wb-demos", run, None, 0.3))[:2] == (60, 780)
    assert _timeout_of(P.payload_for("wb-smoke", run, None, 1.0))[:2] == (60, 3300)
    assert _timeout_of(P.payload_for("wb-q3b", run, 0, 3.4))[:2] == (120, 11940)
    with pytest.raises(ValueError):
        P.payload_for("wb-q3b", run, 0)  # no est_high: no payload
    with pytest.raises(P.Refused):
        P.job_timeouts("wb-demos", 0.2)  # 720 - 300 = 420 s < MIN_OUTER_S
    with pytest.raises(P.Refused):
        P.job_timeouts("wb-smoke", float("nan"))
    # prepare uses the est_high it records, and stores the timeouts in pilot.json
    build_data(lay)
    lay.demos_done.unlink()
    git_fake(monkeypatch)
    P.prepare("wb-demos", now=T0, L=lay)
    c = lay.calls["prepare"][-1]
    assert c["payload"][:4] == ["timeout", "-k", "60", "780"] and c["est_gph"] == 0.3
    info = json.loads((Path(c["run"]) / "pilot.json").read_text())
    assert info["timeouts"]["outer_s"] == 780 and info["payload"] == c["payload"]


@pytest.mark.parametrize("script", ["wb_smoke.sh", "wb_pilot_start.sh"])
def test_bash_syntax(script):
    res = subprocess.run(["bash", "-n", str(JOBS_DIR / script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def _dry_lines(out: str):
    return [ln[len("[dry] "):] for ln in out.splitlines() if ln.startswith("[dry] ")]


@pytest.mark.parametrize("optional", ["0", "1"])
def test_pilot_start_dryrun_order(base, optional):
    run = base / "jobs" / "wb-q3s-202611081200"
    res = _bash([str(JOBS_DIR / "wb_pilot_start.sh"), "Q3S", str(run)],
                {"VPD_DRYRUN": "1", "VPD_OPTIONAL": optional, "VPD_MERGE_DIR": "/dev/shm/vpd_merged"}, base)
    assert res.returncode == 0, res.stderr + res.stdout
    lines = _dry_lines(res.stdout)
    py = [ln for ln in lines if "n2_train_vpd.py" in ln or "merge.py" in ln or "run_vpd" in ln or "rm -rf" in ln]
    kinds = []
    for ln in py:
        if "n2_train_vpd.py" in ln:
            kinds.append("train_" + ln.split("--phase ")[1].split()[0])
        elif "merge.py" in ln:
            kinds.append("merge_" + ln.split("/adapters/")[1].split()[0])
        elif "rm -rf" in ln:
            kinds.append("rm_" + ln.rsplit("__", 1)[1])
        else:
            kinds.append("eval_" + ln.split("--run-name pilot__Q3S__")[1].split()[0])
    want = ["train_0W", "train_tier50", "train_tier200", "eval_raw__640__empty", "rm_200", "merge_200",
            "eval_200__640__empty", "eval_200__640__off", "rm_200", "rm_50", "merge_50", "eval_50__640__empty",
            "rm_50", "eval_raw__1280__empty"]
    if optional == "1":
        want += ["rm_0W", "merge_0W", "eval_0W__640__empty", "rm_0W"]
    assert kinds == want
    t50 = next(ln for ln in py if "--phase tier50" in ln)
    assert "--init-adapter /" in t50 and t50.endswith("/pilot/Q3S/adapters/0W")
    assert "--epochs auto --seed 1001" in t50 and "--threads 2 --num-workers 3" in t50
    assert "/tiers/seed1001/demos_50.jsonl" in t50
    ev = next(ln for ln in py if "pilot__Q3S__200__640__off" in ln)
    assert "--constrained off" in ev and "--model-dir /dev/shm/vpd_merged/Q3S__200" in ev
    assert "--adapter-meta" in ev and "--workers 8" in ev and "--task-set s2test" in ev
    assert "--render-size 1280x800" in next(ln for ln in py if "raw__1280" in ln)
    steps = [json.loads(x) for x in (run / "steps.jsonl").read_text().splitlines()]
    assert all(s["rc"] == 0 for s in steps) and not (run / "DONE").exists()
    # every train / merge / eval runs under the soft deadline (default 11760 s = plan est_high 3.4)
    assert all(ln.startswith("timeout -k 30 ") for ln in py if "rm -rf" not in ln)
    assert all(int(ln.split()[3]) <= 11760 for ln in py if "rm -rf" not in ln)
    assert "deadline=11760s" in res.stdout


def test_pilot_start_soft_deadline(base):
    """A job that reaches its soft deadline starts no further GPU step, records each as NAME:no_time, still runs its
    cleanup, exits 1 and writes no DONE (a resubmission continues: every step is idempotent)."""
    run = base / "jobs" / "wb-q3s-202611081202"
    res = _bash([str(JOBS_DIR / "wb_pilot_start.sh"), "Q3S", str(run)],
                {"VPD_DRYRUN": "1", "VPD_MERGE_DIR": "/dev/shm/vpd_merged", "VPD_JOB_DEADLINE_S": "30"}, base)
    assert res.returncode == 1, res.stderr + res.stdout
    lines = _dry_lines(res.stdout)
    assert not any("n2_train_vpd.py" in ln or "merge.py" in ln or "run_vpd" in ln for ln in lines)
    names = [json.loads(x)["name"] for x in (run / "steps.jsonl").read_text().splitlines()]
    assert {"train_0W:no_time", "eval_raw_640_empty:no_time", "eval_raw_1280_empty:no_time"} <= set(names)
    assert "merge_200:no_adapter" in names and "merge_50:no_adapter" in names
    assert (run / "df_end.txt").exists() and not (run / "DONE").exists()
    assert "failed steps: train_0W" in res.stdout
    for bad in ("0", "abc", "-5"):
        res = _bash([str(JOBS_DIR / "wb_pilot_start.sh"), "Q3S", str(run)],
                    {"VPD_DRYRUN": "1", "VPD_JOB_DEADLINE_S": bad}, base)
        assert res.returncode == 2 and "VPD_JOB_DEADLINE_S" in res.stderr


def test_pilot_start_usage(base):
    res = _bash([str(JOBS_DIR / "wb_pilot_start.sh"), "Q3X", str(base / "r")], {"VPD_DRYRUN": "1"}, base)
    assert res.returncode == 2 and "usage" in res.stderr
    res = _bash([str(JOBS_DIR / "wb_pilot_start.sh"), "E2"], {"VPD_DRYRUN": "1", "RUN": ""}, base)
    assert res.returncode == 2


def test_smoke_dryrun(base):
    run = base / "jobs" / "wb-smoke-202611081200"
    res = _bash([str(JOBS_DIR / "wb_smoke.sh"), str(run)], {"VPD_DRYRUN": "1", "VPD_MERGE_DIR": "/dev/shm/vpd_merged"},
                base)
    assert res.returncode == 0, res.stderr + res.stdout
    lines = _dry_lines(res.stdout)
    order = []
    for ln in lines:
        for key in ("--dry-run --expect-grid", "parity_check.py", "render-parity", "vllm-parity", "wb_train_probe.py",
                    "--phase 0W", "merge.py", "smoke-verdict"):
            if key in ln:
                order.append(key)
        if "run_vpd" in ln:
            order.append("eval:" + ln.split("--run-name ")[1].split()[0])
    assert order == ["--dry-run --expect-grid", "parity_check.py", "render-parity", "vllm-parity", "wb_train_probe.py",
                     "--phase 0W",
                     "eval:smoke__Q3B__raw__640__empty", "eval:smoke__Q3B__raw__1280__empty",
                     "eval:smoke__Q3B__raw__640__off", "eval:smoke__Q3B__raw__640__empty__k2", "merge.py",
                     "eval:smoke__Q3S__0W__640__empty", "smoke-verdict"]
    smoke = "/walkerbench/runs/smoke/wb-smoke-202611081200"
    probe = next(ln for ln in lines if "wb_train_probe.py" in ln)
    assert "--start Q3S --n 8 --load-only Q3RD" in probe and smoke + "/train_probe" in probe
    assert all(ln.startswith("timeout -k 30 ") for ln in lines if "run_vpd" in ln)
    evc = next(ln for ln in lines if "smoke__Q3S__0W" in ln)
    assert "--model-dir /dev/shm/vpd_merged/Q3S__smoke_0W" in evc and "--limit 6" in evc
    assert "--history-turns 2" in next(ln for ln in lines if "__k2" in ln)
    steps = [json.loads(x) for x in (run / "steps.jsonl").read_text().splitlines()]
    assert [s["name"] for s in steps][-1] == "verdict" and all(s["rc"] == 0 for s in steps)
    assert not Path(paths.WB_RUNS / "smoke" / run.name).exists()  # the dry run writes nothing under OUT


def test_smoke_soft_deadline(base):
    run = base / "jobs" / "wb-smoke-202611081201"
    res = _bash([str(JOBS_DIR / "wb_smoke.sh"), str(run)],
                {"VPD_DRYRUN": "1", "VPD_MERGE_DIR": "/dev/shm/vpd_merged", "VPD_SMOKE_DEADLINE_S": "100"}, base)
    assert res.returncode == 1  # every timed step is skipped and reported
    steps = [json.loads(x) for x in (run / "steps.jsonl").read_text().splitlines()]
    assert any(s["name"].endswith(":no_time") for s in steps) and steps[-1]["name"] == "verdict"


def test_prereg_text_matches_the_code():
    """Audit 2026-11-08: the registered text names what the code checks (B16 mm kwargs, G0 provenance, B9 threads,
    the rerun switches, the frozen hashes that already exist)."""
    pr = (paths.VPD_PKG / "PREREG_WB_PILOT.md").read_text(encoding="utf-8")
    for s in ('"size": {"longest_edge": 1024000', "size.longest_edge", "init_adapter.sha256",
              "results.adapter.adapter_sha256", "attn_sites", "check_eval_settings", "VPD_RERUN_STALE",
              "VPD_RERUN_INVALID", "OMP = 2", "job_timeouts", "`warnings` 为空列表",
              "f063f93d2705a6d0ccc87423f8aa05dd531b57be8b8fd4fe023a1079c9c8dee1",
              "67707f8aefa7f3c802fd03f8f5f1f1d65d589a8ab0a948d74db34ab5ed8e8f3b",
              "682a15ae254d3b11157fbb126c540f83d29c5dfc000778f23def3c379a97d4cd"):
        assert s in pr, s
    assert "OMP = 1" not in pr and "〔提交前填〕" not in pr
    # the hashes in the table are the files' (when the frozen outputs exist on this machine)
    for rel, sha in (("walkerbench/split/split_s2.json", "f063f93d2705a6d0"),
                     ("walkerbench/tiers/order_seed1001.json", "67707f8aefa7f3c8"),
                     ("walkerbench/baselines/baselines_s2test.json", "682a15ae254d3b11")):
        p = paths.OUT / rel
        if p.is_file():
            assert P.sha256_file(p).startswith(sha), rel
