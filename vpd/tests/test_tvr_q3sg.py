"""Tests of the third EXPLORATORY start Q3SG (genc0, sft_q3genc0_e8; PREREG_TVR_PILOT.md section 17, row 2026-11-13;
DECISIONS G20): vpd/starts.json's Q3SG entry and the fingerprints (the trainer: this tree 119da3c84b8d, with the
pilot's starts.json 0865dd75f1ee; the run_tvr harness 64edd575f36b unchanged); pilot_tvr's Q3SG jobs (tvr-xsmoke-q3sg,
tvr-xtrain-q3sg, tvr-xeval-q3sg-top / -low: payloads, est_high, block budget, guards: the trainer prefix and
invariant, the 2026-11-13 row, the Q3SG smoke verdict, the processor dir; steps, expect, xsmoke-verdict); the
per-start trainer rule (the pilot and RL jobs still require 0865dd75f1ee, so they refuse on this tree); tvr_xsmoke.sh's
start list (its default unchanged) and tvr_train.sh / tvr_eval.sh with Q3SG; vpd/tvr/compare_q3sg.py on synthetic
roots (cells, predictions, refusals; COMPARE_RL unchanged).

Hermetic like test_tvr_explore.py (its fixtures and builders are reused): no GPU, no network, no rendering, no 8B
model, no sco call; real files are only read (this tree, its prereg, the explore / pilot adapter metas and the Q3SG
checkpoint when mounted, `git show` of the old starts.json).  Login node, 1 core:
    cd TREE && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 ... nice -n 19 env VPD_WORKTREE=TREE PYTHONPATH=TREE \
        $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=$SCRATCH/pt vpd/tests/test_tvr_q3sg.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vpd import paths  # noqa: E402
from vpd.tests import test_tvr_explore as TX  # noqa: E402
from vpd.tests import test_tvr_jobs as TJ  # noqa: E402
from vpd.tests.test_tvr_explore import cworld  # noqa: E402,F401  (the pytest fixture)
from vpd.tests.test_tvr_jobs import lay  # noqa: E402,F401  (the pytest fixture)
from vpd.train import n2_train_vpd as T  # noqa: E402
from vpd.tvr import compare_q3sg as CQ  # noqa: E402
from vpd.tvr import compare_rl as CR  # noqa: E402
from vpd.tvr import compare_tvr as C  # noqa: E402
from vpd.tvr import pilot_cfg  # noqa: E402
from vpd.tvr import pilot_tvr as PT  # noqa: E402

REAL_FP = T.code_fingerprint  # the real one (lay answers the pilot's fingerprint for the pilot / RL tests)
JOBS_DIR = paths.VPD_PKG / "jobs"
wj, fsha = TJ.wj, TJ.fsha
Q = "Q3SG"
QJOBS = ("tvr-xsmoke-q3sg", "tvr-xtrain-q3sg", "tvr-xeval-q3sg-top", "tvr-xeval-q3sg-low")
OLD_COMMIT = "216656b9"  # the RL explore mode's last commit: vpd/starts.json before the Q3SG entry
TRAINER_NEW, TRAINER_OLD = "119da3c84b8d", "0865dd75f1ee"


@pytest.fixture
def qlay(lay, monkeypatch):  # noqa: F811
    """test_tvr_jobs' layout (fake sco, fake prereg state) with THIS tree's trainer fingerprint (the Q3SG one) and an
    explore-start check that needs no checkpoint dir (x_start_problems is tested on its own below)."""
    monkeypatch.setattr(T, "code_fingerprint", REAL_FP)
    monkeypatch.setattr(PT, "x_start_problems", lambda s: [] if s in PT.EXPLORE_STARTS else [f"start {s!r} not "
                                                                                              "explore"])
    return lay


def qverdict(L: PT.Layout, v="PASS", name="tvr-xsmoke-q3sg-202611131000", starts=(Q,)) -> Path:
    return wj(L.xsmoke_dir(name) / "smoke_verdict.json", {"verdict": v, "failed": [] if v == "PASS" else ["x"],
                                                          "starts": list(starts)})


def _steps_file(run: Path, job: str, cfg: dict) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "steps.jsonl").write_text("".join(json.dumps({"name": n, "rc": 0}) + "\n"
                                             for n in PT.required_steps(job, cfg)))


def trainer_tree(root: Path, starts_json: bytes = None, edit_py: str = None) -> Path:
    """ROOT = a copy of the trainer's CODE_FILES of this tree; STARTS_JSON replaces vpd/starts.json; EDIT_PY gets one
    more line."""
    for rel in T.CODE_FILES:
        dst = Path(root) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(T.WORKTREE / rel, dst)
    if starts_json is not None:
        (Path(root) / PT.STARTS_JSON).write_bytes(starts_json)
    if edit_py:
        with open(Path(root) / edit_py, "a") as f:
            f.write("# changed\n")
    return Path(root)


def git_old_starts_json():
    """`git show 216656b9:vpd/starts.json` from this tree or the real worktree (None where neither has it)."""
    for root in dict.fromkeys((paths.WORKTREE, paths.GIT_WORKTREE)):
        try:
            r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(root), "show",
                                f"{OLD_COMMIT}:vpd/starts.json"], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout:
            return r.stdout
    return None


# ------------------------------------------------------------------------------------------ starts.json, fingerprints
def test_starts_json_only_gains_the_q3sg_entry():
    from vpd import starts

    p = paths.VPD_PKG / "starts.json"
    new = p.read_bytes()
    assert TJ.sha(new) == PT.STARTS_JSON_SHA256_Q3SG
    table = json.loads(new)
    assert list(table)[-1] == Q and list(starts.load_table()) == list(table)
    e = starts.resolve(Q)
    assert e["code"] == "sft_q3genc0_e8" and e["family"] == "qwen3"
    assert e["path"] == ("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_q3_20261027/"
                         "gen3b_sft_20261107/c0g/model/checkpoint-3976")
    old = PT.starts_json_without(p)
    assert TJ.sha(old) == PT.STARTS_JSON_SHA256_PILOT and json.loads(old) == {k: v for k, v in table.items() if k != Q}
    # appended: the old bytes up to their closing brace, then the one new entry
    assert new.startswith(old[:-3]) and new[len(old) - 3:].startswith(b',\n "Q3SG": {')


def test_trainer_fingerprint_this_tree_and_the_old_starts_json(tmp_path):
    """This tree: 119da3c84b8d (the 5 .py shas are the pilot's, starts.json the Q3SG one); the same trainer files with
    the old starts.json (git show 216656b9, == this file without the Q3SG entry): 0865dd75f1ee, the pilot's."""
    fp = REAL_FP()
    assert fp["code_sha256"] == PT.Q3SG_TRAINER_SHA256 and fp["code_sha256"].startswith(TRAINER_NEW)
    assert PT.Q3SG_TRAINER_PREFIX == TRAINER_NEW and PT.trainer_fingerprint() == fp
    assert {k: v for k, v in fp["code_files"].items() if k != PT.STARTS_JSON} == PT.TRAINER_PY_SHA256
    assert fp["code_files"][PT.STARTS_JSON] == PT.STARTS_JSON_SHA256_Q3SG and PT.q3sg_trainer_problems() == []
    derived = PT.starts_json_without(T.WORKTREE / PT.STARTS_JSON)
    old = git_old_starts_json()
    if old is not None:
        assert old == derived and TJ.sha(old) == PT.STARTS_JSON_SHA256_PILOT
    ofp = REAL_FP(trainer_tree(tmp_path / "old", starts_json=old if old is not None else derived))
    assert ofp["code_sha256"] == PT.PILOT_TRAINER_SHA256 and ofp["code_sha256"].startswith(TRAINER_OLD)
    assert ofp["code_sha256"].startswith(pilot_cfg.FROZEN["recipe"]["trainer_code_sha256_prefix"])
    assert {k: v for k, v in ofp["code_files"].items() if k != PT.STARTS_JSON} == PT.TRAINER_PY_SHA256
    if old is None:
        pytest.skip("git show 216656b9:vpd/starts.json unavailable here (checked against the derived bytes only)")


def test_pinned_shas_are_the_recorded_adapter_ones():
    """TRAINER_PY_SHA256 / STARTS_JSON_SHA256_PILOT / PILOT_TRAINER_SHA256 are what the RL and pilot adapters recorded
    (skipped where OUT is not mounted)."""
    metas = [PT.LAYOUT.xadapter("Q3RD", "0W") / "meta.json", PT.LAYOUT.xadapter("Q3RS150", "ALL") / "meta.json",
             PT.LAYOUT.adapter("Q3B", "0W") / "meta.json", PT.LAYOUT.adapter("Q3S", "ALL") / "meta.json"]
    if not all(p.is_file() for p in metas):
        pytest.skip("the explore / pilot adapters are not mounted here")
    for p in metas:
        m = json.loads(p.read_text())
        assert m["code_sha256"] == PT.PILOT_TRAINER_SHA256, p
        assert m["code_files"] == dict(PT.TRAINER_PY_SHA256, **{PT.STARTS_JSON: PT.STARTS_JSON_SHA256_PILOT}), p


def test_harness_fingerprint_and_pilot_cfg_unchanged():
    from vpd.tvr import run_tvr

    assert run_tvr.code_sha256()["code_sha256"].startswith("64edd575f36b") and PT.harness_problems() == []
    assert PT.HARNESS_SHA256_PREFIX == "64edd575f36b"
    assert pilot_cfg.FROZEN["recipe"]["trainer_code_sha256_prefix"] == TRAINER_OLD  # pilot_cfg.py not edited
    assert "vpd/tvr/pilot_cfg.py" in run_tvr.CODE_FILES
    assert not {"vpd/tvr/pilot_tvr.py", "vpd/tvr/compare_rl.py", "vpd/tvr/compare_q3sg.py", PT.STARTS_JSON,
                "vpd/jobs/tvr_xsmoke.sh"} & set(run_tvr.CODE_FILES)


def test_q3sg_trainer_rule_on_changed_trees(tmp_path):
    table = json.loads((T.WORKTREE / PT.STARTS_JSON).read_text())

    def dumps(t):
        return (json.dumps(t, indent=1) + "\n").encode()

    assert PT.q3sg_trainer_problems(root=trainer_tree(tmp_path / "same")) == []
    # one more entry after Q3SG: the fingerprint changes and starts.json without Q3SG is no longer the pilot's
    probs = PT.q3sg_trainer_problems(root=trainer_tree(tmp_path / "extra", dumps(dict(table, X={
        "code": None, "path": "/x", "family": "qwen3", "note": ""}))))
    assert any("!= the Q3SG prefix" in b for b in probs) and any("changed beyond the Q3SG entry" in b for b in probs)
    # a pilot entry edited
    t = copy.deepcopy(table)
    t["Q3B"]["note"] += "!"
    probs = PT.q3sg_trainer_problems(root=trainer_tree(tmp_path / "q3b", dumps(t)))
    assert any("changed beyond the Q3SG entry" in b for b in probs)
    # only the Q3SG entry edited: the old bytes still come back, the fingerprint does not
    t = copy.deepcopy(table)
    t[Q]["note"] += "!"
    probs = PT.q3sg_trainer_problems(root=trainer_tree(tmp_path / "q3sg", dumps(t)))
    assert any("!= the Q3SG prefix" in b for b in probs) and not any("changed beyond" in b for b in probs)
    assert any(f"code_files {PT.STARTS_JSON}" in b for b in probs)
    # the pilot's starts.json (no Q3SG entry): the pilot's fingerprint, not the Q3SG one
    probs = PT.q3sg_trainer_problems(root=trainer_tree(tmp_path / "pilot", PT.starts_json_without(
        T.WORKTREE / PT.STARTS_JSON)))
    assert any(f"{TRAINER_OLD} != the Q3SG prefix" in b for b in probs) and not any("changed beyond" in b for b in probs)
    # a trainer .py changed
    probs = PT.q3sg_trainer_problems(root=trainer_tree(tmp_path / "py", edit_py="vpd/walkerbench/history.py"))
    assert any("code_files vpd/walkerbench/history.py" in b for b in probs) and any("Q3SG prefix" in b for b in probs)
    # code_files_problems on an adapter's record
    good = dict(PT.TRAINER_PY_SHA256, **{PT.STARTS_JSON: PT.STARTS_JSON_SHA256_Q3SG})
    assert PT.code_files_problems(good) == [] and PT.code_files_problems(None)
    assert PT.code_files_problems(dict(good, extra="0" * 64)) == ["code_files has other files ['extra']"]
    assert PT.code_files_problems(good, PT.STARTS_JSON_SHA256_PILOT)


# ------------------------------------------------------------------------------------------ job table, payloads, est
def test_q3sg_jobs_table_payloads_and_est_high(tmp_path):
    from vpd import ledger, sco

    for job in QJOBS:
        spec = PT.XJOBS[job]
        assert spec["name"] == "vpd-" + job and spec["as_root"] is True and spec["starts"] == (Q,)
        sco.validate_name(spec["name"])
        assert ledger.gate_of(spec["name"]) == "tvr" and (JOBS_DIR / spec["script"]).is_file()
        assert pilot_cfg.missing(spec["base"], TJ.filled_cfg()) == []
    sm = PT.XJOBS["tvr-xsmoke-q3sg"]
    assert sm["kind"] == "xsmoke" and sm["xlist"] == Q and sm["script"] == "tvr_xsmoke.sh"
    assert (PT.XJOBS["tvr-xtrain-q3sg"]["start"], PT.XJOBS["tvr-xeval-q3sg-low"]["part"]) == (Q, "low")
    # est_high of the smokes: n starts x 1,815 s + 300 + 180 s, rounded up to 0.1 h
    for job, n in (("tvr-xsmoke", 2), ("tvr-xsmoke-q3sg", 1)):
        assert PT.est_high(job)["est_high"] == math.ceil((n * 1815 + 300 + 180) / 360) / 10
    assert PT.est_high("tvr-xsmoke-q3sg")["est_high"] == 0.7 and PT.KILL_S["tvr-xsmoke-q3sg"] == 60
    t = PT.job_timeouts("tvr-xsmoke-q3sg", 0.7)
    assert (t["outer_s"], t["soft_deadline_s"], t["kill_s"]) == (2220, 2040, 60) and t["soft_deadline_s"] >= 1815
    run = tmp_path / "jobs" / "tvr-xsmoke-q3sg-202611131200"
    assert PT.payload_for("tvr-xsmoke-q3sg", run, 0.7) == [
        "env", "VPD_JOB_DEADLINE_S=2040", "timeout", "-k", "60", "2220", "bash", str(JOBS_DIR / "tvr_xsmoke.sh"), Q,
        str(run)]
    # tvr-xsmoke's payload is unchanged: no start list (the script's default)
    assert PT.payload_for("tvr-xsmoke", run, 1.2) == [
        "env", "VPD_JOB_DEADLINE_S=3840", "timeout", "-k", "60", "4020", "bash", str(JOBS_DIR / "tvr_xsmoke.sh"),
        str(run)]
    tr = PT.payload_for("tvr-xtrain-q3sg", run, 2.759)
    assert tr[2] == "VPD_TVR_ROOT=explore/rl" and tr[-3:] == [str(JOBS_DIR / "tvr_train.sh"), Q, str(run)]
    ev = PT.payload_for("tvr-xeval-q3sg-low", run, 3.0)
    assert ev[2] == "VPD_TVR_ROOT=explore/rl" and ev[-4:] == [str(JOBS_DIR / "tvr_eval.sh"), Q, "low", str(run)]


def test_q3sg_est_high_is_the_start_independent_rule(qlay):
    cfg = TJ.filled_cfg()
    TJ.build_cal(qlay)
    assert PT.est_high("tvr-xtrain-q3sg", qlay, cfg) == PT.est_high("tvr-xtrain-q3rd", qlay, cfg) == \
        PT.derived_train("Q3S", qlay, cfg)
    for part in ("top", "low"):
        assert PT.est_high(f"tvr-xeval-q3sg-{part}", qlay, cfg) == PT.derived_eval("Q3S", part, qlay, cfg)
    assert [t for t, _ in PT.train_tiers(Q, cfg)] == ["0W", "100", "ALL"]
    assert PT.eval_plan(Q, "low", cfg) == [("0W", "c"), ("100", "c")]
    assert PT.eval_plan(Q, "top", cfg) == [("ALL", "c"), ("ALL", "u"), ("ALL", "blind")]


def test_q3sg_prepare_records_budgets_facts_and_parents(qlay, capsys):
    cfg = TJ.filled_cfg()
    TX.ready(qlay, cfg)
    launch = PT.prepare("tvr-xsmoke-q3sg", L=qlay, cfg=cfg, events=[])
    c = qlay.calls["prepare"][-1]
    assert c["name"] == "vpd-tvr-xsmoke-q3sg" and c["as_root"] is True and c["est_gph"] == 0.7
    assert c["payload"][-2:] == [Q, c["run"]] and launch == Path(c["run"]) / "job" / "launch.sh"
    info = json.loads((Path(c["run"]) / "prepared.json").read_text())
    bb = info["block_budget"]
    assert (bb["baseline"], bb["cap"], bb["ok"]) == (48.5136, 25.0, True) and info["round_budget"]["ok"]
    f = info["facts"]
    assert f["trainer_code_sha256"] == PT.Q3SG_TRAINER_SHA256 and f["trainer_problems"] == []
    assert f["prereg_row_date"] == "2026-11-13" and f["harness_code_sha256"].startswith("64edd575f36b")
    row = PT.xrow(PT.prereg_path().read_text(encoding="utf-8"), PT.Q3SG_ROW_DATE, PT.Q3SG_ROW_NEEDLE)
    assert row and f["prereg_row_sha256"] == TJ.sha(row.encode())
    assert (qlay.xroot / Q).is_dir() and not qlay.pilot_root.exists()
    assert "block budget: used -48.5136 + open 0.0000 + new 0.7000" in capsys.readouterr().err
    # the same block as the RL starts: 24.5 used + 0.7 > 25
    with pytest.raises(PT.Refused, match=r"BLOCK BUDGET: block_used 24\.5000 \+ open 0\.0000 \+ new 0\.7000 = "
                                         r"25\.2000 > 25\.0"):
        PT.prepare("tvr-xsmoke-q3sg", L=qlay, cfg=cfg, new=True, events=TX._ledger(48.5136 + 24.5, 0.0))


def test_pilot_and_rl_jobs_refuse_on_this_tree(qlay):
    """This tree's trainer is the Q3SG one: every pilot and RL job (recipe prefix 0865dd75f1ee) refuses, the Q3SG
    jobs do not (DECISIONS G20: intended, the pilot is closed and the RL block is done)."""
    cfg = TJ.filled_cfg()
    TX.ready(qlay, cfg)
    TX.xverdict(qlay)
    qverdict(qlay)
    want = f"trainer code_sha256 {TRAINER_NEW} != frozen prefix {TRAINER_OLD}"
    for job in ("tvr-data", "tvr-cal", "tvr-msmoke", "tvr-train-q3b", "tvr-eval-q3s-low", "tvr-xsmoke",
                "tvr-xtrain-q3rd", "tvr-xeval-q3rs150-top"):
        assert want in PT.prerequisites(job, qlay, cfg), job
    for job in ("tvr-xsmoke-q3sg", "tvr-xtrain-q3sg"):
        assert PT.prerequisites(job, qlay, cfg) == [], job
    assert PT.x_trainer_problems("tvr-xeval-q3sg-top") == [] and PT.x_trainer_problems("tvr-xsmoke", cfg) == [want]
    with pytest.raises(PT.Refused, match="frozen prefix 0865dd75f1ee"):
        PT.prepare("tvr-xtrain-q3rd", L=qlay, cfg=cfg, events=[])
    with pytest.raises(PT.Refused, match="frozen prefix 0865dd75f1ee"):
        PT.prepare("tvr-train-q3b", L=qlay, cfg=cfg, events=[])


# ------------------------------------------------------------------------------------------ guards
def test_q3sg_guards_smoke_verdict_trainer_and_row(qlay, monkeypatch, tmp_path):
    cfg = TJ.filled_cfg()
    TX.ready(qlay, cfg)
    assert PT.prerequisites("tvr-xsmoke-q3sg", qlay, cfg) == []
    # xtrain: the newest verdict whose starts include Q3SG; the RL smoke's verdicts never count
    TX.xverdict(qlay, "PASS")
    wj(qlay.xsmoke_dir("tvr-xsmoke-202611131100") / "smoke_verdict.json",
       {"verdict": "PASS", "failed": [], "starts": list(PT.RL_STARTS)})
    assert any("whose starts include Q3SG (job tvr-xsmoke-q3sg)" in b
               for b in PT.prerequisites("tvr-xtrain-q3sg", qlay, cfg))
    qverdict(qlay, "FAIL", name="tvr-xsmoke-q3sg-202611131000")
    assert any("not PASS" in b for b in PT.prerequisites("tvr-xtrain-q3sg", qlay, cfg))
    qverdict(qlay, "PASS", name="tvr-xsmoke-q3sg-202611131030")
    assert PT.prerequisites("tvr-xtrain-q3sg", qlay, cfg) == []
    qverdict(qlay, "FAIL", name="tvr-xsmoke-q3sg-202611131200")  # a newer FAIL: the newest counts
    assert any("tvr-xsmoke-q3sg-202611131200" in b and "not PASS" in b
               for b in PT.prerequisites("tvr-xtrain-q3sg", qlay, cfg))
    shutil.rmtree(qlay.xsmoke_dir("tvr-xsmoke-q3sg-202611131200"))
    assert PT.latest_xverdict(qlay, Q)[0].parent.name == "tvr-xsmoke-q3sg-202611131030"
    assert PT.latest_xverdict(qlay, "Q3RD")[0].parent.name == "tvr-xsmoke-202611131100"  # the RL rule, unchanged
    assert PT.latest_xverdict(qlay)[0] == PT.latest_xverdict(qlay, "Q3RS150")[0]
    PT.prepare("tvr-xtrain-q3sg", L=qlay, cfg=cfg, events=[])
    c = qlay.calls["prepare"][-1]
    info = json.loads((Path(c["run"]) / "prepared.json").read_text())
    assert "tvr-xsmoke-q3sg-202611131030" in info["smoke_verdict"]["path"] and info["smoke_verdict"]["verdict"] == "PASS"
    assert c["payload"][2] == "VPD_TVR_ROOT=explore/rl" and c["payload"][-2:] == [Q, c["run"]]
    assert c["est_gph"] == PT.derived_train(Q, qlay, cfg)["est_high"]
    # the trainer: prefix and invariant (this tree's fingerprint replaced)
    fp = REAL_FP()
    monkeypatch.setattr(PT, "trainer_fingerprint", lambda root=None: dict(fp, code_sha256=PT.PILOT_TRAINER_SHA256))
    assert any(f"{TRAINER_OLD} != the Q3SG prefix {TRAINER_NEW}" in b
               for b in PT.prerequisites("tvr-xtrain-q3sg", qlay, cfg))
    monkeypatch.setattr(PT, "trainer_fingerprint", lambda root=None: dict(fp, code_files=dict(
        fp["code_files"], **{"vpd/paths.py": "0" * 64})))
    assert any("code_files vpd/paths.py" in b for b in PT.prerequisites("tvr-xeval-q3sg-low", qlay, cfg))
    monkeypatch.setattr(PT, "trainer_fingerprint", lambda root=None: REAL_FP(root or T.WORKTREE))
    assert PT.prerequisites("tvr-xtrain-q3sg", qlay, cfg) == []
    # the row: the Q3SG jobs need exactly one 2026-11-13 row; the RL jobs the 2026-11-12 one
    text = PT.prereg_path().read_text(encoding="utf-8")
    q, r = PT.xrow(text, PT.Q3SG_ROW_DATE, PT.Q3SG_ROW_NEEDLE), PT.xrow(text)
    assert q and q.startswith("| 2026-11-13 |") and "Q3SG" in q and "sft_q3genc0_e8" in q and r
    p = tmp_path / "PREREG.md"
    monkeypatch.setattr(PT, "prereg_path", lambda: p)
    p.write_text(text.replace(q + "\n", ""), encoding="utf-8")
    assert any("'| 2026-11-13 |" in b and "third explore start Q3SG" in b
               for b in PT.prerequisites("tvr-xsmoke-q3sg", qlay, cfg))
    assert not any("2026-11-1" in b for b in PT.prerequisites("tvr-xsmoke", qlay, cfg))
    p.write_text(text.replace(r + "\n", ""), encoding="utf-8")
    assert PT.prerequisites("tvr-xsmoke-q3sg", qlay, cfg) == []
    assert any("'| 2026-11-12 |" in b for b in PT.prerequisites("tvr-xsmoke", qlay, cfg))
    p.write_text(text + "\n" + q + "\n", encoding="utf-8")
    assert any("no single section-17 row '| 2026-11-13 |" in b for b in PT.prerequisites("tvr-xtrain-q3sg", qlay, cfg))


def test_x_start_problems_q3sg(monkeypatch, tmp_path):
    from vpd import starts

    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "config.json").write_text("{}")
    table = {Q: {"name": Q, "code": "sft_q3genc0_e8", "path": str(ck), "family": "qwen3", "note": ""}}
    monkeypatch.setattr(starts, "resolve", lambda n: table[n])
    monkeypatch.setattr(starts, "processor_dir", lambda n: ck)
    assert PT.x_start_problems(Q) == []
    monkeypatch.setattr(starts, "processor_dir", lambda n: tmp_path / "view")  # an eval view was built since
    assert any("processor would be read from" in b for b in PT.x_start_problems(Q))
    monkeypatch.setattr(starts, "processor_dir", lambda n: ck)
    (ck / "config.json").unlink()
    assert any("config.json missing" in b for b in PT.x_start_problems(Q))


def test_x_start_problems_q3sg_on_the_real_starts(capsys):
    """The real starts.json: Q3SG has a code and no eval view, so the processor is read from checkpoint-3976 (skipped
    where the checkpoint is not mounted)."""
    from vpd import starts

    e = starts.resolve(Q)
    assert PT.main(["served", Q]) == 0 and capsys.readouterr().out.strip() == "sft_q3genc0_e8"
    if not os.path.lexists(Path(e["path"]) / "config.json"):
        pytest.skip("the Q3SG checkpoint is not mounted here")
    assert PT.x_start_problems(Q) == [] and Path(starts.processor_dir(Q)) == Path(e["path"])
    assert not os.path.lexists(starts.view_dir(Q) / starts.PROCESSOR_MARKER)


def test_q3sg_xeval_adapter_checks(qlay):
    cfg = TJ.filled_cfg()
    files = TX.ready(qlay, cfg)
    qverdict(qlay)
    with pytest.raises(PT.Refused, match="adapter Q3SG/0W"):
        PT.prepare("tvr-xeval-q3sg-top", L=qlay, cfg=cfg, events=[])
    TX.build_xadapters(qlay, Q, cfg, files)
    assert PT.prerequisites("tvr-xeval-q3sg-low", qlay, cfg) == []
    PT.prepare("tvr-xeval-q3sg-top", L=qlay, cfg=cfg, events=[])
    c = qlay.calls["prepare"][-1]
    assert c["payload"][-3:] == [Q, "top", c["run"]] and c["est_gph"] <= 5.5
    ad = qlay.xadapter(Q, "ALL") / "meta.json"
    for fn, want in ((lambda m: m.update(code_sha256=PT.PILOT_TRAINER_SHA256), f"not the Q3SG prefix {TRAINER_NEW}"),
                     (lambda m: m["code_files"].update({"downstream/train/lora.py": "1" * 64}),
                      "code_files downstream/train/lora.py 111111111111"),
                     (lambda m: m["code_files"].update({PT.STARTS_JSON: PT.STARTS_JSON_SHA256_PILOT}),
                      f"code_files {PT.STARTS_JSON} 86bb356e61d6"),
                     (lambda m: m.pop("code_files"), "code_files NoneType"),
                     (lambda m: m["code_files"].update(extra="2" * 64), "other files ['extra']"),
                     (lambda m: m.update(seed=1002), "meta seed 1002")):
        orig = ad.read_text()
        TX._edit(ad, fn)
        probs = PT.prerequisites("tvr-xeval-q3sg-low", qlay, cfg)
        assert any("adapter Q3SG/ALL" in b and want in b for b in probs), (want, probs)
        ad.write_text(orig)
    assert PT.prerequisites("tvr-xeval-q3sg-low", qlay, cfg) == []


def test_rl_adapters_judged_by_their_registered_code(qlay):
    """On this (Q3SG) tree an RL adapter trained by the pilot's trainer is still complete for x_adapter_check, so
    compare_rl keeps validating the RL adapters; walkerbench adapter_check (the current tree) says other code."""
    from vpd.walkerbench.pilot import adapter_check

    cfg = TJ.filled_cfg()
    files = TX.ready(qlay, cfg)
    TX.build_xadapters(qlay, "Q3RD", cfg, files)
    for tier, init in PT.train_tiers("Q3RD", cfg):
        rc, msg = PT.x_adapter_check("Q3RD", tier, init, qlay, cfg)
        assert rc == 0 and f"code {TRAINER_OLD}" in msg, (tier, msg)
    assert adapter_check(qlay.xadapter("Q3RD", "0W"))[0] == 3
    TX._edit(qlay.xadapter("Q3RD", "100") / "meta.json", lambda m: m.update(code_sha256=PT.Q3SG_TRAINER_SHA256))
    rc, msg = PT.x_adapter_check("Q3RD", "100", "0W", qlay, cfg)
    assert rc == 3 and f"not the registered prefix {TRAINER_OLD}" in msg


# ------------------------------------------------------------------------------------------ steps / expect / guard
def test_q3sg_required_steps_and_expect(qlay):
    cfg = TJ.filled_cfg()
    files = TX.build_tiers(qlay, cfg)
    assert PT.required_steps("tvr-xsmoke-q3sg", cfg) == ["guard", "prereg_committed", "merge_disk"] + [
        f"{x}_Q3SG" for x in ("load", "dry", "hfparity", "session", "train", "merge", "merge_parity", "eval")] + [
        "verdict"]
    assert PT.required_steps("tvr-xtrain-q3sg", cfg) == PT.required_steps("tvr-xtrain-q3rd", cfg)
    for part in ("top", "low"):
        assert PT.required_steps(f"tvr-xeval-q3sg-{part}", cfg) == PT.required_steps(f"tvr-eval-q3s-{part}", cfg)
    # the smoke's expect: a PASS verdict of exactly its starts
    run = qlay.jobs / "tvr-xsmoke-q3sg-202611131300"
    _steps_file(run, "tvr-xsmoke-q3sg", cfg)
    vp = wj(qlay.xsmoke_dir(run) / "smoke_verdict.json", {"verdict": "PASS", "failed": [], "starts": [Q]})
    assert PT.expect("tvr-xsmoke-q3sg", run, qlay, cfg)["ok"]
    wj(vp, {"verdict": "PASS", "failed": [], "starts": list(PT.RL_STARTS)})
    assert PT.expect("tvr-xsmoke-q3sg", run, qlay, cfg)["failed"] == ["smoke_verdict"]
    assert vp in PT.expected_outputs("tvr-xsmoke-q3sg", run, qlay, cfg)
    # the training's expect: the adapters by the Q3SG trainer rule
    TX.build_xadapters(qlay, Q, cfg, files)
    run = qlay.jobs / "tvr-xtrain-q3sg-202611131300"
    _steps_file(run, "tvr-xtrain-q3sg", cfg)
    assert PT.expect("tvr-xtrain-q3sg", run, qlay, cfg)["ok"]
    m = qlay.xadapter(Q, "100") / "meta.json"
    orig = m.read_text()
    TX._edit(m, lambda x: x.update(code_sha256=PT.PILOT_TRAINER_SHA256))
    assert PT.expect("tvr-xtrain-q3sg", run, qlay, cfg)["failed"] == ["adapter_100"]
    m.write_text(orig)
    # the evals' expect: under runs/explore/rl/Q3SG, prefix xrl, merge parity reported (not judged)
    for part in ("top", "low"):
        TX._xevals(qlay, Q, cfg, part)
        job = f"tvr-xeval-q3sg-{part}"
        run = qlay.jobs / f"{job}-202611131300"
        _steps_file(run, job, cfg)
        rep = PT.expect(job, run, qlay, cfg)
        assert rep["ok"], rep["failed"]
        paths_ = [Path(c["path"]) for c in rep["items"] if c.get("path")]
        assert paths_ and all(qlay.xroot / Q in p.parents for p in paths_)
        assert any(p.parent.name.startswith("xrl__Q3SG__") for p in paths_)


def test_guard_in_job_q3sg_writes_facts(qlay):
    cfg = TJ.filled_cfg()
    TX.ready(qlay, cfg)
    run = qlay.jobs / "tvr-xsmoke-q3sg-202611131400"
    run.mkdir(parents=True)
    rep = PT.guard("tvr-xsmoke-q3sg", run, qlay, cfg)
    assert rep["ok"], rep["problems"]
    f = json.loads((run / "guard.json").read_text())["facts"]
    assert f["explore"] is True and f["trainer_code_sha256"] == PT.Q3SG_TRAINER_SHA256 and f["trainer_problems"] == []
    assert f["trainer_code_files"][PT.STARTS_JSON] == PT.STARTS_JSON_SHA256_Q3SG and "Q3SG" in f["trainer_rule"]
    assert f["harness_code_sha256"].startswith("64edd575f36b") and f["prereg_row_date"] == "2026-11-13"
    rep = PT.guard("tvr-xsmoke", run, qlay, cfg)  # the RL smoke on this tree: refused by its trainer rule
    assert not rep["ok"] and any(f"frozen prefix {TRAINER_OLD}" in b for b in rep["problems"])


def test_cli_plan_and_budget_show_the_q3sg_jobs(qlay, capsys):
    TJ.build_cal(qlay)
    assert PT.main(["plan", "--explore"]) == 0
    out = capsys.readouterr().out
    for job in QJOBS:
        assert f"[{job}]" in out, job
    assert "Q3SG: row 2026-11-13" in out and "est_high 0.70 GPU-h" in out and "outer 2220 s, soft 2040 s, kill 60 s" in out
    assert "tvr_xsmoke.sh Q3SG " in out
    b = PT.budget(qlay, TJ.filled_cfg(), events=[], explore=True)
    assert set(QJOBS) <= set(b["est_high"]) and b["est_high"]["tvr-xsmoke-q3sg"] == 0.7


# ------------------------------------------------------------------------------------------ xsmoke verdict
def test_xsmoke_verdict_q3sg(qlay):
    cfg = TJ.filled_cfg()
    files = TX.build_tiers(qlay, cfg)
    sm = qlay.xsmoke_dir("tvr-xsmoke-q3sg-202611131500")
    TX.build_xsmoke(qlay, sm, cfg, files, starts=(Q,))
    v = PT.xsmoke_verdict(sm, qlay, cfg, job="tvr-xsmoke-q3sg")
    assert v["verdict"] == "PASS", v["failed"]
    assert v["starts"] == [Q] and v["job"] == "tvr-xsmoke-q3sg" and all(c["name"].endswith("_Q3SG") for c in v["checks"])
    tr = next(c for c in v["checks"] if c["name"] == "train_Q3SG")
    assert tr["detail"]["code_sha256"] == TRAINER_NEW and tr["detail"]["code_problems"] == []
    meta = sm / Q / "adapters" / "0W" / "meta.json"
    for fn in (lambda m: m.update(code_sha256=PT.PILOT_TRAINER_SHA256),
               lambda m: m["code_files"].update({"vpd/starts.py": "3" * 64}),
               lambda m: m["code_files"].update({PT.STARTS_JSON: PT.STARTS_JSON_SHA256_PILOT})):
        orig = meta.read_text()
        TX._edit(meta, fn)
        assert PT.xsmoke_verdict(sm, qlay, cfg, job="tvr-xsmoke-q3sg")["failed"] == ["train_Q3SG"]
        meta.write_text(orig)
    with pytest.raises(PT.Refused, match="not an explore smoke job"):
        PT.xsmoke_verdict(sm, qlay, cfg, job="tvr-xtrain-q3sg")
    # the CLI passes --job (the frozen cfg: only the parsing and the starts are checked here)
    PT.main(["xsmoke-verdict", "--smoke", str(sm), "--job", "tvr-xsmoke-q3sg"])
    w = json.loads((sm / "smoke_verdict.json").read_text())
    assert (w["job"], w["starts"]) == ("tvr-xsmoke-q3sg", [Q])
    with pytest.raises(SystemExit):
        PT.main(["xsmoke-verdict", "--smoke", str(sm), "--job", "tvr-xtrain-q3sg"])
    # the default job (the RL starts) on this smoke dir: FAIL, nothing of theirs is there
    v = PT.xsmoke_verdict(sm, qlay, cfg)
    assert v["verdict"] == "FAIL" and v["starts"] == list(PT.RL_STARTS) and v["job"] == "tvr-xsmoke"


# ------------------------------------------------------------------------------------------ payloads
def test_payload_dry_xsmoke_q3sg(tmp_path):
    res, lines, run = TX.xdry(tmp_path, "tvr_xsmoke.sh", Q, root=None)
    k = TJ._common_dry_asserts(lines, run, tmp_path, 3840)
    per = ["rm", "raw_bf16:--start", "dry", "hfparity", "eval:raw:c,u,blind", "reap:session_Q3SG", "rm", "train:0W",
           "rm", "merge:0W", "merge_parity:--start", "eval:0W:c", "reap:eval_Q3SG", "rm"]
    assert k == ["prereg_committed", "guard"] + per + ["xsmoke-verdict", "expect", "DONE"], k
    SM = tmp_path / "tvr" / "runs" / "explore" / "rl" / "smoke" / run.name
    assert f"guard tvr-xsmoke-q3sg --run {run}" in res.stdout and f"expect tvr-xsmoke-q3sg --run {run}" in res.stdout
    assert any(ln.endswith(f"xsmoke-verdict --smoke {SM} --job tvr-xsmoke-q3sg") for ln in lines)
    assert "Q3RD" not in res.stdout and "Q3RS150" not in res.stdout and "eval500" not in res.stdout
    ld = next(ln for ln in lines if "raw_bf16 --start Q3SG " in ln)
    assert ld.endswith(f"--out /dev/shm/vpd_merged/xsmoke__Q3SG__raw --report {SM}/Q3SG/load/load.json")
    tr = next(ln for ln in lines if "n2_train_vpd.py" in ln and "--dry-run" not in ln)
    assert (f"--start Q3SG --data {tmp_path}/tvr/tiers/v1/warmup0w20.jsonl --out {SM}/Q3SG/adapters/0W --limit 16 "
            "--epochs 1 --seed 1001 --phase 0W --threads 2 --num-workers 3") in tr
    ev = next(ln for ln in lines if "--start Q3SG --tier 0W" in ln)
    assert (f"--name-prefix xsmoke --out-root {SM}/Q3SG/evals --model-dir /dev/shm/vpd_merged/xsmoke__Q3SG__0W "
            f"--adapter-meta {SM}/Q3SG/adapters/0W/meta.json --mechanism-only") in ev


def test_xsmoke_default_start_list_unchanged_and_refusals(tmp_path):
    def norm(lines, d):  # the seconds left before the deadline may differ by one between two runs
        return [re.sub(r"(timeout -k 30|--deadline-s) \d+", r"\1 N", ln.replace(str(d), "X")) for ln in lines]

    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    ra, a, run = TX.xdry(tmp_path / "a", "tvr_xsmoke.sh", root=None)
    rb, b, _ = TX.xdry(tmp_path / "b", "tvr_xsmoke.sh", "Q3RD,Q3RS150", root=None)
    assert norm(a, tmp_path / "a") == norm(b, tmp_path / "b")
    SM = tmp_path / "a" / "tvr" / "runs" / "explore" / "rl" / "smoke" / run.name
    assert any(ln.endswith(f"xsmoke-verdict --smoke {SM}") for ln in a) and "--job" not in ra.stdout
    assert f"guard tvr-xsmoke --run {run}" in ra.stdout and "Q3SG" not in ra.stdout
    for bad in (["Q3SG,Q3RD"], ["Q3RD"], ["q3sg"], ["Q3B"], ["Q3RS150,Q3RD"], ["Q3SG", "x"], [""]):
        r = TX._bash([str(JOBS_DIR / "tvr_xsmoke.sh"), *bad, str(tmp_path / "jobs" / "r")], {"VPD_DRYRUN": "1"},
                     tmp_path)
        assert r.returncode == 2 and "usage: tvr_xsmoke.sh [Q3RD,Q3RS150|Q3SG] RUN" in r.stderr, bad
    r = TX._bash([str(JOBS_DIR / "tvr_xsmoke.sh"), Q, ""], {"VPD_DRYRUN": "1"}, tmp_path)
    assert r.returncode == 2 and "usage" in r.stderr


def test_payload_dry_xtrain_and_xeval_q3sg(tmp_path):
    res, lines, run = TX.xdry(tmp_path, "tvr_train.sh", Q)
    assert TJ._kinds(lines) == ["prereg_committed", "guard", "adapter-ok:0W", "train:0W", "adapter-ok:100",
                                "train:tier100", "adapter-ok:ALL", "train:tierALL", "expect", "DONE"]
    P = tmp_path / "tvr" / "runs" / "explore" / "rl" / Q
    tr = [ln for ln in lines if "n2_train_vpd.py" in ln]
    assert all(f"--start {Q} " in ln and "--epochs auto --seed 1001" in ln for ln in tr)
    assert f"--out {P}/adapters/ALL" in tr[2] and tr[1].endswith(f"--init-adapter {P}/adapters/0W")
    assert f"guard tvr-xtrain-q3sg --run {run}" in res.stdout and "runs/pilot" not in res.stdout
    for part in ("top", "low"):
        d = tmp_path / part
        d.mkdir()
        res, lines, run = TX.xdry(d, "tvr_eval.sh", Q, part)
        P = d / "tvr" / "runs" / "explore" / "rl" / Q
        evs = [ln for ln in lines if "run_tvr eval" in ln]
        assert evs and all(f"--start {Q} " in ln and "--task-set eval500 --name-prefix xrl" in ln
                           and f"--out-root {P}/evals" in ln for ln in evs)
        assert f"guard tvr-xeval-q3sg-{part} --run {run}" in res.stdout and f"--served <code:{Q}>" in res.stdout
        assert f"--run-dir {P}/canary/{part}" in res.stdout and "runs/pilot" not in res.stdout
    run = str(tmp_path / "jobs" / "r")
    for script, args in (("tvr_train.sh", [Q, run]), ("tvr_eval.sh", [Q, "top", run])):
        r = TX._bash([str(JOBS_DIR / script), *args], {"VPD_DRYRUN": "1"}, tmp_path)  # the pilot root: never Q3SG
        assert r.returncode == 2 and "Q3B Q3S E2" in r.stderr, script


def test_nondry_xsmoke_q3sg(tmp_path):
    payload, env = TX.xfake_tree(tmp_path, "tvr_xsmoke.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-xsmoke-q3sg-202611130000"
    res = TX._bash([str(payload), Q, str(run)], env, tmp_path)
    assert res.returncode == 0, res.stderr[-1500:] + res.stdout[-1500:]
    names = [s["name"] for s in TJ._steps(run)]
    assert (run / "DONE").exists() and PT.steps_check("tvr-xsmoke-q3sg", run)["ok"]
    assert "train_Q3SG" in names and not any("Q3RD" in n or "Q3RS150" in n for n in names)
    SM = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "rl" / "smoke" / run.name
    log = (tmp_path / "fake.log").read_text()
    assert f"vpd.tvr.pilot_tvr xsmoke-verdict --smoke {SM} --job tvr-xsmoke-q3sg" in log
    assert "guard tvr-xsmoke-q3sg --run" in log and "expect tvr-xsmoke-q3sg --run" in log and SM.is_dir()


# ------------------------------------------------------------------------------------------ compare_q3sg
SUCC_Q = {("Q3SG", "0W", "c"): set(range(5)), ("Q3B", "0W", "c"): {8, 9, 10, 11}, ("Q3S", "0W", "c"): {0, 1},
          # 100 c: Q3SG 2 < Q3B 12 (b 0, c 10, p 1/512); Q3SG - Q3S b 0 c 1 (p 1) -> 动作任务本身
          ("Q3SG", "100", "c"): {0, 1}, ("Q3B", "100", "c"): set(range(12)), ("Q3S", "100", "c"): {0, 1, 2},
          # ALL c: Q3SG 3 < Q3B 4 (p 1) -> 介于两者之间，不判
          ("Q3SG", "ALL", "c"): {0, 1, 2}, ("Q3B", "ALL", "c"): {0, 1, 2, 3}, ("Q3S", "ALL", "c"): {5},
          ("Q3SG", "ALL", "u"): {1, 2}, ("Q3S", "ALL", "u"): {2, 3, 4}, ("Q3SG", "ALL", "blind"): {7}}


@pytest.fixture
def qworld(tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "EVAL500_N", TX.N_TASKS)
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "c0ffee"})
    L = PT.Layout(tmp_path / "out")
    cfg = TJ.filled_cfg()
    files = TX.build_tiers(L, cfg)
    xroot, proot = tmp_path / "explore_rl", tmp_path / "pilot"
    tasks_dir = tmp_path / "repo" / "data" / "tasks"
    wj(tasks_dir / "eval.json", [{"task_id": t, "dataset": TX.CATS[i % 4][0], "difficulty": TX.CATS[i % 4][1]}
                                 for i, t in enumerate(TX.IDS)])
    TX.build_xadapters(L, Q, cfg, files, root=xroot)
    for tier, arm in C.cells():
        ad = xroot / Q / "adapters" / PT.EVAL_ADAPTER[tier] / "adapter.pt"
        TX.write_eval(C.eval_dir(xroot, Q, tier, arm, "xrl"), SUCC_Q.get((Q, tier, arm), set()), Q, tier, arm,
                      fsha(ad))
    for s in ("Q3B", "Q3S"):
        for tier, arm in C.cells():
            TX.write_eval(C.eval_dir(proot, s, tier, arm), SUCC_Q.get((s, tier, arm), set()), s, tier, arm, "p" * 64)
        for t in ("0W", "100", "ALL"):  # the pilot adapters' records (the pilot's trainer)
            wj(proot / s / "adapters" / t / "meta.json", {"start": s, **TX.trainer_code(s)})
    return {"L": L, "xroot": xroot, "proot": proot, "tasks": tasks_dir, "cfg": cfg}


def qrun(w, **kw):
    return CQ.compare(w["xroot"], w["proot"], tasks_dir=w["tasks"], L=w["L"], **kw)


def test_compare_q3sg_cells_predictions_and_table(qworld):
    rep = qrun(qworld)
    assert rep["schema"] == "vpd.tvr.compare_q3sg/1" and rep["exploratory"] is True
    assert rep["validation"]["ok"] and rep["trainer_invariant"]["ok"] and rep["multiplicity_correction"] == "none"
    assert list(rep["validation"]["evals"]) == [f"Q3SG/{t}__{a}" for t, a in C.cells()]
    assert rep["prereg"]["row_text"].startswith("| 2026-11-13 |") and "第三个探索起点 Q3SG" in rep["prereg"]["row_text"]
    assert rep["cells"] == ["0W__c", "100__c", "ALL__c", "ALL__u", "ALL__blind"]
    assert rep["registered_cells"] == ["100__c", "ALL__c"]
    assert list(rep["comparisons"]) == ["Q3SG-Q3B", "Q3SG-Q3S"]
    for name, role in (("Q3SG-Q3B", "primary"), ("Q3SG-Q3S", "diagnostic")):
        comp = rep["comparisons"][name]
        assert comp["role"] == role and comp["b_is"] == "only Q3SG succeeds"
        assert {k: c["use"] for k, c in comp["cells"].items()} == {
            "0W__c": "description", "100__c": role, "ALL__c": role, "ALL__u": "description",
            "ALL__blind": "description"}
    want = {("Q3SG-Q3B", "100__c"): (2, 12, 0, 10), ("Q3SG-Q3S", "100__c"): (2, 3, 0, 1),
            ("Q3SG-Q3B", "ALL__c"): (3, 4, 0, 1), ("Q3SG-Q3B", "0W__c"): (5, 4, 5, 4),
            ("Q3SG-Q3S", "ALL__u"): (2, 3, 1, 2), ("Q3SG-Q3B", "ALL__blind"): (1, 0, 1, 0)}
    for (name, cell), (k1, k2, b, c) in want.items():
        x = rep["comparisons"][name]["cells"][cell]
        p = C.exact_p(b, c)
        assert (x["k_first"], x["k_second"], x["b"], x["c"], x["n"]) == (k1, k2, b, c, TX.N_TASKS), (name, cell)
        assert x["p"] == float(p) and x["p_exact"] == f"{p.numerator}/{p.denominator}" and not x["missing"]
        assert x["wilson95_first"] == C.wilson(k1, TX.N_TASKS, 1.96)
    pr = rep["predictions"]
    assert list(pr) == ["100__c", "ALL__c"]
    assert pr["100__c"]["verdict"] == "动作任务本身" and pr["100__c"]["p_vs_Q3B_exact"] == "1/512"
    assert (pr["100__c"]["k_Q3SG"], pr["100__c"]["k_Q3B"], pr["100__c"]["k_Q3S"]) == (2, 12, 3)
    assert pr["ALL__c"]["verdict"] == "介于两者之间，不判" and pr["ALL__c"]["rule"] == "otherwise"
    t = rep["start_table"]
    assert len(t) == 15 and [r["start"] for r in t[:3]] == ["Q3B", "Q3S", "Q3SG"]
    assert [r["source"] for r in t[:3]] == ["pilot", "pilot", "explore"]
    assert rep["crosscheck"]["ok"] and rep["crosscheck"]["used_sides_all_checked"] and rep["missing_cells"] == []
    inv = rep["trainer_invariant"]
    assert inv["pilot_reference"]["py_equals_pinned"] is True and inv["this_tree"]["problems"] == []
    assert set(inv["adapters"]) == {"0W", "100", "ALL"} and all(a["ok"] for a in inv["adapters"].values())
    md = CQ.render_md(rep)
    assert "## 预判（登记的两格，分别判，不合并）" in md and "| 100 c | 2/24 | 12/24 | 0.001953 | 3/24 | 1 | 动作任务本身 |" in md
    assert "| ALL | c | 3/24" in md and "主（登记）" in md and "诊断（登记）" in md and "描述" in md
    assert "偏乐观" in md and "推断" in md and "全部一致" in md and "同口径" in md
    # compare_rl's own sides never include Q3SG
    assert CR.RL_STARTS == ("Q3RD", "Q3RS150") and CR.side_root(Q, qworld["xroot"], qworld["proot"]) == (
        qworld["proot"], "pilot")


def _pair(k1, k2, p):
    return {"missing": False, "k_first": k1, "k_second": k2, "n": 500, "p": float(Fraction(p)), "p_exact": p}


@pytest.mark.parametrize("q3b,q3s,verdict", [
    ((10, 10, "1/1"), (10, 12, "1/2"), "遗忘"),               # k equal: forgetting
    ((11, 10, "1/2"), (11, 30, "1/1000"), "遗忘"),
    ((25, 20, "1/1000"), None, "遗忘"),                      # decidable without the Q3S side
    ((5, 20, "1/1000"), (5, 6, "1/20"), "动作任务本身"),       # p vs Q3S exactly 0.05: not significant
    ((5, 20, "1/1000"), (5, 6, "51/1000"), "动作任务本身"),
    ((5, 20, "1/20"), (5, 6, "1/2"), "介于两者之间，不判"),     # p vs Q3B exactly 0.05: not significant
    ((5, 20, "49/1000"), (5, 6, "1/2"), "动作任务本身"),
    ((5, 20, "1/1000"), (5, 30, "1/1000"), "介于两者之间，不判"),  # both significant
    ((5, 6, "1/2"), (5, 6, "1/2"), "介于两者之间，不判"),
    ((5, 20, "1/1000"), None, None),                         # needs the Q3S side
])
def test_prediction_rule(q3b, q3s, verdict):
    got = CQ.prediction(_pair(*q3b), {"missing": True} if q3s is None else _pair(*q3s))
    assert got["verdict"] == verdict and (got["rule"] is None) == (verdict is None)
    if verdict is not None:
        assert got["rule"] == CQ.RULES[verdict]
    assert CQ.prediction({"missing": True}, _pair(1, 1, "1/1"))["verdict"] is None


def test_compare_q3sg_refuses_until_the_five_q3sg_evals_validate(qworld):
    xroot = qworld["xroot"]
    d = C.eval_dir(xroot, Q, "ALL", "u", "xrl")
    breaks = [
        (d / "results.json", lambda r: r.update(valid=False), "not valid"),
        (d / "results.json", lambda r: r.update(n_tasks=TX.N_TASKS - 1), "not valid"),
        (d / "results.json", lambda r: r.update(mechanism_only=True), "mechanism_only True"),
        (d / "results.json", lambda r: r.update(start="Q3S"), "start 'Q3S' != 'Q3SG'"),
        (d / "results.json", lambda r: r.update(code_sha256="0" * 64), "harness code_sha256"),
        (d / "results.json", lambda r: r["adapter"].update(adapter_sha256="0" * 64), "served adapter"),
        (d / "official_check.json", lambda r: r.update(equal=False), "official_check.json equal False"),
        (xroot / Q / "adapters" / "ALL" / "meta.json", lambda r: r.update(seed=1002), "meta seed 1002"),
        (xroot / Q / "adapters" / "100" / "meta.json", lambda r: r.update(code_sha256=PT.PILOT_TRAINER_SHA256),
         f"not the Q3SG prefix {TRAINER_NEW}"),
        (xroot / Q / "adapters" / "0W" / "meta.json", lambda r: r["code_files"].update({"vpd/paths.py": "9" * 64}),
         "code_files vpd/paths.py"),
    ]
    for path, fn, want in breaks:
        orig = path.read_text()
        TX._edit(path, fn)
        with pytest.raises(CQ.Refused) as e:
            qrun(qworld)
        assert "not all 5 Q3SG evals" in e.value.message and want in e.value.message, (want, e.value.message)
        path.write_text(orig)
    shutil.rmtree(C.eval_dir(xroot, Q, "100", "c", "xrl"))
    with pytest.raises(CQ.Refused) as e:
        qrun(qworld)
    assert "Q3SG/100__c (xrl__Q3SG__100__c)" in e.value.message


def test_compare_q3sg_trainer_invariant_refusals(qworld):
    """The Q3SG adapters pass x_adapter_check (the pinned shas), but the invariant compares them with what the pilot
    Q3B / Q3S adapters recorded."""
    proot = qworld["proot"]
    metas = [proot / s / "adapters" / t / "meta.json" for s in ("Q3B", "Q3S") for t in ("0W", "100", "ALL")]
    origs = {p: p.read_text() for p in metas}

    def refused(want):
        with pytest.raises(CQ.Refused) as e:
            qrun(qworld)
        assert "trainer invariant does not hold" in e.value.message and want in e.value.message, e.value.message
        for p, t in origs.items():
            p.write_text(t)

    for p in metas:  # the pilot recorded another n2_train_vpd.py: Q3SG's .py shas are not the pilot's
        TX._edit(p, lambda m: m["code_files"].update({"vpd/train/n2_train_vpd.py": "4" * 64}))
    refused("code_files vpd/train/n2_train_vpd.py 6c2b4056b337 != the pilot adapters' 444444444444")
    TX._edit(metas[4], lambda m: m["code_files"].update({"vpd/paths.py": "5" * 64}))  # one pilot adapter differs
    refused("recorded different code_files")
    for p in metas:  # the pilot's starts.json is not the registered old one
        TX._edit(p, lambda m: m["code_files"].update({PT.STARTS_JSON: "6" * 64}))
    refused("STARTS_JSON_SHA256_PILOT")
    TX._edit(metas[0], lambda m: m.pop("code_files"))
    refused(f"{metas[0]}: no code_files")
    assert qrun(qworld)["trainer_invariant"]["ok"]


def test_compare_q3sg_row_missing_cells_and_main(qworld, monkeypatch, capsys, tmp_path):
    p = tmp_path / "PREREG.md"
    p.write_text(PT.prereg_path().read_text(encoding="utf-8").replace(PT.Q3SG_ROW_NEEDLE, "x"), encoding="utf-8")
    with pytest.raises(CQ.Refused) as e:
        qrun(qworld, prereg=p)
    assert "no single section-17 row '| 2026-11-13 |" in e.value.message
    monkeypatch.setattr(PT, "LAYOUT", qworld["L"])
    args = ["--explore-root", str(qworld["xroot"]), "--pilot-root", str(qworld["proot"]), "--tasks-dir",
            str(qworld["tasks"])]
    assert CQ.main(args) == 0
    out = qworld["xroot"] / "COMPARE_Q3SG.json"
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["schema"] == CQ.SCHEMA and out.with_suffix(".md").is_file()
    assert not (qworld["xroot"] / "COMPARE_RL.json").exists() and not (qworld["proot"] / "COMPARE_Q3SG.json").exists()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["predictions"] == {
        "100__c": "动作任务本身", "ALL__c": "介于两者之间，不判"}
    assert CQ.main(args + ["--out", str(qworld["proot"] / "x" / "C.json")]) == 2
    assert "under the pilot root" in capsys.readouterr().err and not (qworld["proot"] / "x").exists()
    # a Q3S eval missing: its diagnostic cell is missing, so is the 100 c verdict (k(Q3SG) < k(Q3B)); exit 3
    shutil.rmtree(C.eval_dir(qworld["proot"], "Q3S", "100", "c"))
    rep = qrun(qworld)
    assert rep["predictions"]["100__c"]["verdict"] is None and rep["predictions"]["ALL__c"]["verdict"] is not None
    assert "Q3SG-Q3S 100__c" in rep["missing_cells"] and "start table Q3S 100__c" in rep["missing_cells"]
    assert "不判（Q3SG - Q3S missing" in CQ.render_md(rep)
    assert CQ.main(args) == 3


def test_compare_rl_ignores_q3sg_evals(cworld):  # noqa: F811
    """COMPARE_RL is unchanged by Q3SG evals and adapters under the same explore root (its RL starts only)."""
    a = TX.crun(cworld)
    files = PT.tier_manifest_files(cworld["L"])
    TX.build_xadapters(cworld["L"], Q, cworld["cfg"], files, root=cworld["xroot"])
    for tier, arm in C.cells():
        TX.write_eval(C.eval_dir(cworld["xroot"], Q, tier, arm, "xrl"), {0}, Q, tier, arm, "q" * 64)
    b = TX.crun(cworld)
    a.pop("utc"), b.pop("utc")
    assert a == b and len(a["validation"]["evals"]) == 10


def test_cli_help_compare_q3sg():
    res = subprocess.run([sys.executable, "-B", "-m", "vpd.tvr.compare_q3sg", "--help"], capture_output=True,
                         text=True, env=dict(os.environ, PYTHONPATH=str(paths.WORKTREE)), cwd=str(paths.WORKTREE),
                         timeout=120)
    assert res.returncode == 0 and "usage" in res.stdout.lower() and "COMPARE_Q3SG" in res.stdout


def test_readme_mentions_the_q3sg_files():
    text = (paths.VPD_PKG / "README.md").read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith("| `")]
    for f in ("tvr/compare_q3sg.py", "tests/test_tvr_q3sg.py"):
        assert any(f"`{f}`" in ln.split("|")[1] for ln in rows), f
    for w in ("COMPARE_Q3SG", "vpd-tvr-xsmoke-q3sg", "Q3SG"):
        assert w in text, w
