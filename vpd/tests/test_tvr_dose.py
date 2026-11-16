"""Tests of the EXPLORATORY dose starts Q3S2 and Q3S4 (Q3S's own C0 action SFT run lf_q3formal at epochs 2 and 4:
checkpoint-308 / -616, codes c0q3_sft_e2 / c0q3_sft_e4; PREREG_TVR_PILOT.md section 17, row pilot_tvr.DOSE_ROW_DATE):
vpd/starts.json's two entries and the fingerprints (the trainer: this tree d694cddb942b; with the starts.json before
the dose entries, `git show ea5196e8`, the Q3SG one 119da3c84b8d; with the pilot's 0865dd75f1ee); the checkpoints'
identity with checkpoint-1232; the start groups' allowlist (XGROUPS); pilot_tvr's dose jobs (tvr-xsmoke-dose,
tvr-xtrain-q3s2 / -q3s4, tvr-xeval-q3s2 / -q3s4-top / -low: payloads, est_high, the dose budget (25 GPU-h, the dose
jobs only), guards: the trainer prefix and invariant, the harness (the registered 64edd575f36b, as every explore
job), the dose row, the dose smoke verdict, the processor dir; steps, expect, guard facts,
xsmoke-verdict); the Q3SG, RL and pilot jobs refuse on this tree, their adapters are still judged by their registered
code; tvr_xsmoke.sh's dose start list and tvr_train.sh / tvr_eval.sh with Q3S2 / Q3S4; vpd/tvr/compare_dose.py on
synthetic roots (cells, the dose table, H1, refusals; COMPARE_RL unchanged).

The dose row is not in the prereg until the merge: the guards' tests use a copy of this tree's prereg with one test row
(prereg_with_dose_row); test_real_prereg_dose_row checks the real one once it is there (skipped before).

Hermetic like test_tvr_q3sg.py (test_tvr_explore's and test_tvr_jobs' fixtures and builders are reused): no GPU, no
network, no rendering, no 8B model, no sco call; real files are only read (this tree, its prereg, the Q3S / Q3S2 / Q3S4
checkpoints' small files when mounted, `git show` of the old starts.json).  Login node, 1 core:
    cd TREE && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 ... nice -n 19 env VPD_WORKTREE=TREE PYTHONPATH=TREE \
        $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=$SCRATCH/pt vpd/tests/test_tvr_dose.py
"""

from __future__ import annotations

import copy
import json
import math
import os
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
from vpd.tvr import compare_dose as CD  # noqa: E402
from vpd.tvr import compare_q3sg as CQ  # noqa: E402
from vpd.tvr import compare_rl as CR  # noqa: E402
from vpd.tvr import compare_tvr as C  # noqa: E402
from vpd.tvr import pilot_cfg  # noqa: E402
from vpd.tvr import pilot_tvr as PT  # noqa: E402

REAL_FP = T.code_fingerprint  # the real one (lay answers the pilot's fingerprint for the pilot / RL tests)
JOBS_DIR = paths.VPD_PKG / "jobs"
wj, fsha = TJ.wj, TJ.fsha
D = ("Q3S2", "Q3S4")
DJOBS = ("tvr-xsmoke-dose", "tvr-xtrain-q3s2", "tvr-xtrain-q3s4", "tvr-xeval-q3s2-top", "tvr-xeval-q3s2-low",
         "tvr-xeval-q3s4-top", "tvr-xeval-q3s4-low")
OLD_COMMIT = "ea5196e8"  # the last commit before the dose entries: its vpd/starts.json is the Q3SG tree's
TRAINER_NEW, TRAINER_Q3SG, TRAINER_OLD = "d694cddb942b", "119da3c84b8d", "0865dd75f1ee"
# the user's pre-readout prediction, verbatim (compare_dose.H1_TEXT and the row carry it unchanged)
USER_H1 = ("H1（剂量）：ALL c 的 Q3Sn − Q3B 随轮数单调下降（第 2 轮 > 第 4 轮 > 第 8 轮，差值从接近 0 走向 −9.2 个百分点）。"
           "若第 2 轮的 Q3S2 − Q3B 不显著或为正，如实写'轻量动作训练不损害迁移，重训才损害'，这是对第 3.5 节的修正，"
           "不是对命题的支持。")
TEST_ROW = (f"| {PT.DOSE_ROW_DATE} | 探索（EXPLORATORY；不进闸门、决策表、A4 和确认轮）：{PT.DOSE_ROW_NEEDLE}（测试用的行；"
            "真的那一行合并时登记） | 测试 | 测试 |")


def prereg_with_dose_row(tmp_path: Path) -> Path:
    """A copy of this tree's prereg with one dose row after the Q3SG row (the real one if it is there already)."""
    text = PT.prereg_path().read_text(encoding="utf-8")
    if PT.xrow(text, PT.DOSE_ROW_DATE, PT.DOSE_ROW_NEEDLE) is None:
        q = PT.xrow(text, PT.Q3SG_ROW_DATE, PT.Q3SG_ROW_NEEDLE)
        assert q
        text = text.replace(q + "\n", q + "\n" + TEST_ROW + "\n", 1)
    p = Path(tmp_path) / "prereg" / "PREREG_TVR_PILOT.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def dlay(lay, monkeypatch, tmp_path):  # noqa: F811
    """test_tvr_jobs' layout (fake sco, fake prereg state) with THIS tree's trainer fingerprint (the dose one), a
    prereg copy with the dose row and an explore-start check that needs no checkpoint dir (x_start_problems: tested
    below)."""
    TJ.use_trainer_tree(monkeypatch, T.WORKTREE, PT.DOSE_TRAINER_SHA256)
    monkeypatch.setattr(PT, "x_start_problems", lambda s: [] if s in PT.EXPLORE_STARTS else [f"start {s!r} not "
                                                                                              "explore"])
    p = prereg_with_dose_row(tmp_path)
    monkeypatch.setattr(PT, "prereg_path", lambda: p)
    return lay


def dverdict(L: PT.Layout, v="PASS", name="tvr-xsmoke-dose-202611151000", starts=D) -> Path:
    return wj(L.xsmoke_dir(name) / "smoke_verdict.json", {"verdict": v, "failed": [] if v == "PASS" else ["x"],
                                                          "starts": list(starts)})


def _steps_file(run: Path, job: str, cfg: dict) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "steps.jsonl").write_text("".join(json.dumps({"name": n, "rc": 0}) + "\n"
                                             for n in PT.required_steps(job, cfg)))


def trainer_tree(root: Path, starts_json: bytes, edit_py: str = None) -> Path:
    """ROOT = a copy of the trainer's CODE_FILES of this tree, vpd/starts.json = STARTS_JSON; EDIT_PY gets one more
    line."""
    for rel in T.CODE_FILES:
        dst = Path(root) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(T.WORKTREE / rel, dst)
    (Path(root) / PT.STARTS_JSON).write_bytes(starts_json)
    if edit_py:
        with open(Path(root) / edit_py, "a") as f:
            f.write("# changed\n")
    return Path(root)


def dumps(t) -> bytes:
    return (json.dumps(t, indent=1) + "\n").encode()


def git_starts_json(commit: str):
    """`git show COMMIT:vpd/starts.json` from this tree or the real worktree (None where neither has it)."""
    for root in dict.fromkeys((paths.WORKTREE, paths.GIT_WORKTREE)):
        try:
            r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(root), "show", f"{commit}:vpd/starts.json"],
                               capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout:
            return r.stdout
    return None


# ------------------------------------------------------------------------------------------ starts.json, fingerprints
def test_starts_json_only_gains_the_dose_entries():
    from vpd import starts

    p = paths.VPD_PKG / "starts.json"
    new = p.read_bytes()
    assert TJ.sha(new) == PT.STARTS_JSON_SHA256_DOSE and new.isascii() and dumps(json.loads(new)) == new
    table = json.loads(new)
    assert list(table)[-3:] == ["Q3SG", "Q3S2", "Q3S4"] and list(starts.load_table()) == list(table)
    q3s = Path(starts.resolve("Q3S")["path"])
    assert q3s.name == "checkpoint-1232"
    for s, ep, step in (("Q3S2", 2, 308), ("Q3S4", 4, 616)):
        e = starts.resolve(s)
        assert e["code"] == f"c0q3_sft_e{ep}" and e["family"] == "qwen3" and e["path"] == str(q3s.parent /
                                                                                              f"checkpoint-{step}")
        for w in (f"epoch {ep} / {step} steps", "checkpoint-1232", "trainer_state.json", "ckpt_check.json",
                  "no eval view", "EXPLORATORY"):
            assert w in e["note"], (s, w)
    old = PT.starts_json_without(p, D)
    assert TJ.sha(old) == PT.STARTS_JSON_SHA256_Q3SG
    assert json.loads(old) == {k: v for k, v in table.items() if k not in D}
    assert TJ.sha(PT.starts_json_without(p, PT.Q3SG_STARTS + D)) == PT.STARTS_JSON_SHA256_PILOT
    # appended: the old bytes up to their closing brace, then the two new entries
    assert new.startswith(old[:-3]) and new[len(old) - 3:].startswith(b',\n "Q3S2": {')
    head = git_starts_json(OLD_COMMIT)
    if head is None:
        pytest.skip(f"git show {OLD_COMMIT}:vpd/starts.json unavailable here (checked against the derived bytes only)")
    assert head == old


def test_trainer_fingerprint_this_tree_and_the_old_starts_json(tmp_path):
    """This tree: d694cddb942b (the 5 .py shas are the pilot's, starts.json the dose one); the same trainer files with
    the starts.json before the dose entries (git show ea5196e8, == this file without them): 119da3c84b8d, the Q3SG
    one; with the pilot's starts.json: 0865dd75f1ee."""
    fp = REAL_FP()
    assert fp["code_sha256"] == PT.DOSE_TRAINER_SHA256 and fp["code_sha256"].startswith(TRAINER_NEW)
    assert PT.DOSE_TRAINER_PREFIX == TRAINER_NEW and PT.trainer_fingerprint() == fp and PT.trainer_root() == T.WORKTREE
    assert {k: v for k, v in fp["code_files"].items() if k != PT.STARTS_JSON} == PT.TRAINER_PY_SHA256
    assert fp["code_files"][PT.STARTS_JSON] == PT.STARTS_JSON_SHA256_DOSE and PT.dose_trainer_problems() == []
    derived = PT.starts_json_without(T.WORKTREE / PT.STARTS_JSON, D)
    old = git_starts_json(OLD_COMMIT)
    if old is not None:
        assert old == derived and TJ.sha(old) == PT.STARTS_JSON_SHA256_Q3SG
    head = trainer_tree(tmp_path / "head", old if old is not None else derived)
    ofp = REAL_FP(head)
    assert ofp["code_sha256"] == PT.Q3SG_TRAINER_SHA256 and ofp["code_sha256"].startswith(TRAINER_Q3SG)
    assert {k: v for k, v in ofp["code_files"].items() if k != PT.STARTS_JSON} == PT.TRAINER_PY_SHA256
    assert PT.q3sg_trainer_problems(root=head) == [] and PT.dose_trainer_problems(root=head)
    pfp = REAL_FP(TJ.pilot_trainer_root(tmp_path / "pilot"))
    assert pfp["code_sha256"] == PT.PILOT_TRAINER_SHA256 and pfp["code_sha256"].startswith(TRAINER_OLD)
    assert pfp["code_sha256"].startswith(pilot_cfg.FROZEN["recipe"]["trainer_code_sha256_prefix"])
    if old is None:
        pytest.skip(f"git show {OLD_COMMIT}:vpd/starts.json unavailable here (checked against the derived bytes only)")


def test_dose_checkpoints_are_q3s_runs_epochs_2_and_4():
    """The registered identity (OUT/scratch/orch/r5/dose/ckpt_check.json): the same 21 files as checkpoint-1232, the
    13 non-weight files byte-identical, the 7 shards of equal size, only trainer_state.json differs (global_step 308 /
    616 of 1,540, epoch 2.0 / 4.0 of 10).  Skipped where the checkpoints are not mounted."""
    from vpd import starts

    ref = Path(starts.resolve("Q3S")["path"])
    cks = {s: Path(starts.resolve(s)["path"]) for s in D}
    if not all(os.path.lexists(p / "config.json") for p in [ref, *cks.values()]):
        pytest.skip("the Q3S / Q3S2 / Q3S4 checkpoints are not mounted here")
    same = ("added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json", "training_args.bin", "video_preprocessor_config.json", "vocab.json")
    names = sorted(p.name for p in ref.iterdir())
    assert len(names) == 21 and set(same) < set(names)
    for (s, ck), step in zip(cks.items(), (308, 616)):
        assert sorted(p.name for p in ck.iterdir()) == names, s
        for f in same:
            assert fsha(ck / f) == fsha(ref / f), (s, f)
        for f in (n for n in names if n.endswith(".safetensors")):
            assert (ck / f).stat().st_size == (ref / f).stat().st_size, (s, f)
        assert fsha(ck / "trainer_state.json") != fsha(ref / "trainer_state.json")
        st = json.loads((ck / "trainer_state.json").read_text())
        assert (st["global_step"], st["epoch"], st["max_steps"], st["num_train_epochs"]) == (step, step / 154, 1540,
                                                                                              10), s


def test_harness_fingerprint_and_pilot_cfg_unchanged():
    """The run_tvr harness is the registered 64edd575f36b (no run_tvr CODE_FILE changes): the dose evals use it as
    every other eval; pilot_cfg and the run_tvr CODE_FILES are not touched by the dose code."""
    from vpd.tvr import run_tvr

    assert run_tvr.code_sha256()["code_sha256"].startswith("64edd575f36b") and PT.harness_problems() == []
    assert PT.HARNESS_SHA256_PREFIX == "64edd575f36b"
    assert pilot_cfg.FROZEN["recipe"]["trainer_code_sha256_prefix"] == TRAINER_OLD  # pilot_cfg.py not edited
    assert not {"vpd/tvr/pilot_tvr.py", "vpd/tvr/compare_rl.py", "vpd/tvr/compare_dose.py", PT.STARTS_JSON,
                "vpd/jobs/tvr_xsmoke.sh", "vpd/jobs/tvr_train.sh", "vpd/jobs/tvr_eval.sh"} & set(run_tvr.CODE_FILES)


def test_dose_trainer_rule_on_changed_trees(tmp_path):
    table = json.loads((T.WORKTREE / PT.STARTS_JSON).read_text())
    assert PT.dose_trainer_problems(root=trainer_tree(tmp_path / "same", dumps(table))) == []
    # one more entry after Q3S4: the fingerprint changes and starts.json without the dose entries is not the Q3SG one
    probs = PT.dose_trainer_problems(root=trainer_tree(tmp_path / "extra", dumps(dict(table, X={
        "code": None, "path": "/x", "family": "qwen3", "note": ""}))))
    assert any(f"!= the dose prefix {TRAINER_NEW} (PREREG section 17, {PT.DOSE_ROW_DATE})" in b for b in probs)
    assert any("changed beyond the Q3S2 and Q3S4 entries" in b for b in probs)
    # the Q3SG entry edited
    t = copy.deepcopy(table)
    t["Q3SG"]["note"] += "!"
    probs = PT.dose_trainer_problems(root=trainer_tree(tmp_path / "q3sg_edit", dumps(t)))
    assert any("!= the Q3SG tree's 8cd21cfa6076 (starts.json changed beyond" in b for b in probs)
    # only a dose entry edited: the Q3SG bytes still come back, the fingerprint does not
    t = copy.deepcopy(table)
    t["Q3S4"]["note"] += "!"
    probs = PT.dose_trainer_problems(root=trainer_tree(tmp_path / "dose_edit", dumps(t)))
    assert any("!= the dose prefix" in b for b in probs) and not any("changed beyond" in b for b in probs)
    assert any(f"code_files {PT.STARTS_JSON}" in b for b in probs)
    # the Q3SG tree's starts.json (HEAD before the dose entries): the Q3SG fingerprint, not the dose one
    probs = PT.dose_trainer_problems(root=trainer_tree(tmp_path / "q3sg", PT.starts_json_without(
        T.WORKTREE / PT.STARTS_JSON, D)))
    assert any(f"{TRAINER_Q3SG} != the dose prefix" in b for b in probs)
    assert not any("changed beyond" in b for b in probs)
    # the Q3SG rule on this tree: its prefix and its starts.json invariant both fail (intended: the Q3SG line is done)
    probs = PT.q3sg_trainer_problems(root=T.WORKTREE)
    assert any(f"trainer code_sha256 {TRAINER_NEW} != the Q3SG prefix {TRAINER_Q3SG}" in b for b in probs)
    assert any("changed beyond the Q3SG entry" in b for b in probs) and PT.q3sg_trainer_problems() == probs
    # a trainer .py changed
    probs = PT.dose_trainer_problems(root=trainer_tree(tmp_path / "py", dumps(table), edit_py="vpd/paths.py"))
    assert any("code_files vpd/paths.py" in b for b in probs) and any("dose prefix" in b for b in probs)
    # code_files_problems on an adapter's record, with the dose sha
    good = dict(PT.TRAINER_PY_SHA256, **{PT.STARTS_JSON: PT.STARTS_JSON_SHA256_DOSE})
    assert PT.code_files_problems(good, PT.STARTS_JSON_SHA256_DOSE) == [] and PT.code_files_problems(good)


# ------------------------------------------------------------------------------------------ the start groups, jobs
def test_start_groups_allowlist():
    assert list(PT.XGROUPS) == ["Q3SG", "dose"] and PT.XGROUP_OF == {"Q3SG": "Q3SG", "Q3S2": "dose", "Q3S4": "dose"}
    assert PT.DOSE_STARTS == D and PT.EXPLORE_STARTS == PT.RL_STARTS + PT.Q3SG_STARTS + D
    assert not set(PT.XGROUP_OF) & set(PT.RL_STARTS) and not set(PT.EXPLORE_STARTS) & set(pilot_cfg.STARTS)
    g = PT.XGROUPS["dose"]
    assert (g["trainer"], g["starts_json"], g["base_json"], g["cap"]) == (
        PT.DOSE_TRAINER_SHA256, PT.STARTS_JSON_SHA256_DOSE, PT.STARTS_JSON_SHA256_Q3SG, 25.0)
    assert PT.XGROUPS["Q3SG"]["base_json"] == PT.STARTS_JSON_SHA256_PILOT and PT.XGROUPS["Q3SG"]["cap"] is None
    for job in PT.XJOBS:
        want = "dose" if job.endswith("-dose") or "-q3s2" in job or "-q3s4" in job else (
            "Q3SG" if "-q3sg" in job else None)
        assert PT.job_group(job) == want, job
    assert sorted(j for j in PT.XJOBS if PT.job_group(j) == "dose") == sorted(DJOBS)
    assert PT.xrow_key("Q3S2") == PT.xrow_key("Q3S4") == (PT.DOSE_ROW_DATE, PT.DOSE_ROW_NEEDLE)
    assert PT.xrow_key("Q3SG") == (PT.Q3SG_ROW_DATE, PT.Q3SG_ROW_NEEDLE) and PT.xrow_key("Q3RD") == (
        PT.XROW_DATE, PT.XROW_NEEDLE)
    assert PT.DOSE_ROW_NEEDLE == "剂量起点 Q3S2、Q3S4 按试点完全相同的配方训练和评测"


def test_dose_jobs_table_payloads_and_est_high(tmp_path):
    from vpd import ledger, sco

    for job in DJOBS:
        spec = PT.XJOBS[job]
        assert spec["name"] == "vpd-" + job and spec["as_root"] is True
        sco.validate_name(spec["name"])
        assert ledger.gate_of(spec["name"]) == "tvr" and (JOBS_DIR / spec["script"]).is_file()
        assert pilot_cfg.missing(spec["base"], TJ.filled_cfg()) == []
    sm = PT.XJOBS["tvr-xsmoke-dose"]
    assert sm["kind"] == "xsmoke" and sm["xlist"] == "Q3S2,Q3S4" and sm["starts"] == D and sm["start"] is None
    for s in D:
        assert PT.XJOBS[f"tvr-xtrain-{s.lower()}"]["starts"] == (s,)
        assert (PT.XJOBS[f"tvr-xeval-{s.lower()}-low"]["start"], PT.XJOBS[f"tvr-xeval-{s.lower()}-low"]["part"]) == (
            s, "low")
    # est_high of the dose smoke: 2 starts x 1,815 s + 300 + 180 s, rounded up to 0.1 h (as tvr-xsmoke)
    assert PT.est_high("tvr-xsmoke-dose")["est_high"] == math.ceil((2 * 1815 + 300 + 180) / 360) / 10 == 1.2
    assert PT.KILL_S["tvr-xsmoke-dose"] == 60
    t = PT.job_timeouts("tvr-xsmoke-dose", 1.2)
    assert (t["outer_s"], t["soft_deadline_s"], t["kill_s"]) == (4020, 3840, 60)
    run = tmp_path / "jobs" / "tvr-xsmoke-dose-202611151200"
    assert PT.payload_for("tvr-xsmoke-dose", run, 1.2) == [
        "env", "VPD_JOB_DEADLINE_S=3840", "timeout", "-k", "60", "4020", "bash", str(JOBS_DIR / "tvr_xsmoke.sh"),
        "Q3S2,Q3S4", str(run)]
    tr = PT.payload_for("tvr-xtrain-q3s4", run, 2.759)
    assert tr[2] == "VPD_TVR_ROOT=explore/rl" and tr[-3:] == [str(JOBS_DIR / "tvr_train.sh"), "Q3S4", str(run)]
    ev = PT.payload_for("tvr-xeval-q3s2-top", run, 3.027)
    assert ev[2] == "VPD_TVR_ROOT=explore/rl" and ev[-4:] == [str(JOBS_DIR / "tvr_eval.sh"), "Q3S2", "top", str(run)]
    # the RL and Q3SG smokes' payloads are unchanged
    assert PT.payload_for("tvr-xsmoke", run, 1.2)[-2:] == [str(JOBS_DIR / "tvr_xsmoke.sh"), str(run)]
    assert PT.payload_for("tvr-xsmoke-q3sg", run, 0.7)[-3:] == [str(JOBS_DIR / "tvr_xsmoke.sh"), "Q3SG", str(run)]


def test_dose_est_high_is_the_start_independent_rule(dlay):
    cfg = TJ.filled_cfg()
    TJ.build_cal(dlay)
    for s in D:
        assert PT.est_high(f"tvr-xtrain-{s.lower()}", dlay, cfg) == PT.derived_train("Q3S", dlay, cfg)
        for part in ("top", "low"):
            assert PT.est_high(f"tvr-xeval-{s.lower()}-{part}", dlay, cfg) == PT.derived_eval("Q3S", part, dlay, cfg)
        assert [t for t, _ in PT.train_tiers(s, cfg)] == ["0W", "100", "ALL"]
        assert PT.eval_plan(s, "low", cfg) == [("0W", "c"), ("100", "c")]
        assert PT.eval_plan(s, "top", cfg) == [("ALL", "c"), ("ALL", "u"), ("ALL", "blind")]
    b = PT.budget(dlay, cfg, events=[], explore=True)
    gb = b["group_budgets"]["dose"]
    assert set(b["group_budgets"]) == {"dose"} and gb["cap"] == 25.0 and gb["ok"]
    assert gb["est_high_total"] == round(sum(b["est_high"][j] for j in DJOBS), 3) and set(gb["jobs"]) == {
        "vpd-" + j for j in DJOBS}


# ------------------------------------------------------------------------------------------ budget
def _events(*recs):
    """Ledger events: (name, est, gph or None (open)) per record."""
    ev = []
    for i, (name, est, gph) in enumerate(recs):
        run = f"/r/{i}"
        ev.append({"event": "submit", "name": name, "run": run, "gate": "tvr", "est": est, "gpus": 1,
                   "pt_id": f"pt-{i}", "submitted_at": "2026-11-15T00:00:00Z"})
        if gph is not None:
            ev.append({"event": "end", "run": run, "pt_id": f"pt-{i}", "gph": gph,
                       "gph_source": "sco_start_to_complete"})
    return ev


def test_dose_budget_counts_only_the_dose_jobs(dlay, capsys):
    cfg = TJ.filled_cfg()
    TX.ready(dlay, cfg)
    ev = _events(("vpd-tvr-msmoke", 1.0, 30.0), ("vpd-tvr-xtrain-q3s2", 2.759, 2.1),
                 ("vpd-tvr-xeval-q3s2-top", 3.027, None), ("vpd-tvr-xtrain-q3sg", 2.759, 2.0))
    gb = PT.group_budget("dose", 2.0, ev)
    assert (gb["spent"], gb["open_est"], gb["total"], gb["ok"]) == (2.1, 3.027, 7.127, True)
    with pytest.raises(ValueError, match="no cap of its own"):
        PT.group_budget("Q3SG", 1.0, ev)
    # the G18 block would refuse here (30 GPU-h of other jobs since its baseline); the dose cap does not count them
    assert not PT.block_budget(1.2, _events(("vpd-tvr-msmoke", 1.0, 48.5136 + 30.0)))["ok"]
    PT.prepare("tvr-xsmoke-dose", L=dlay, cfg=cfg, events=_events(("vpd-tvr-msmoke", 1.0, 48.5136 + 30.0)))
    assert "dose budget: spent 0.0000 + open 0.0000 + new 1.2000 = 1.2000 GPU-h vs cap 25.0" in capsys.readouterr().err
    # 22.7 spent + 1.0 open + 1.2 = 24.9 <= 25; 23.0 + 1.0 + 1.2 = 25.2 > 25
    rb, bb = PT._budget_checks("tvr-xsmoke-dose", 1.2, _events(
        ("vpd-tvr-xtrain-q3s2", 2.759, 22.7), ("vpd-tvr-xeval-q3s4-low", 1.0, None)), cfg)
    assert bb["ok"] and bb["total"] == 24.9 and rb["ok"]
    with pytest.raises(PT.Refused, match=r"DOSE BUDGET: spent 23\.0000 \+ open 1\.0000 \+ new 1\.2000 = 25\.2000 > "
                                         r"25\.0 GPU-h \(the user's cap for Q3S2 and Q3S4 together"):
        PT.prepare("tvr-xsmoke-dose", L=dlay, cfg=cfg, new=True, events=_events(
            ("vpd-tvr-xtrain-q3s2", 2.759, 23.0), ("vpd-tvr-xeval-q3s4-low", 1.0, None)))
    # rechecked at submit
    with pytest.raises(PT.Refused, match="DOSE BUDGET"):
        PT.submit("tvr-xsmoke-dose", L=dlay, cfg=cfg, events=_events(("vpd-tvr-xeval-q3s2-low", 24.0, 24.0)))
    # the round-200 rule still applies (checked first)
    with pytest.raises(PT.Refused, match="ROUND BUDGET"):
        PT.prepare("tvr-xsmoke-dose", L=dlay, cfg=cfg, new=True, events=_events(("vpd-tvr-msmoke", 1.0, 210.0)))
    # the Q3SG and RL jobs keep the G18 block
    rb, bb = PT._budget_checks("tvr-xsmoke-q3sg", 0.7, [], cfg)
    assert bb["baseline"] == 48.5136 and bb["cap"] == 25.0 and "group" not in bb
    with pytest.raises(PT.Refused, match="BLOCK BUDGET"):
        PT._budget_checks("tvr-xsmoke", 1.2, _events(("vpd-tvr-msmoke", 1.0, 48.5136 + 24.5)), cfg)


# ------------------------------------------------------------------------------------------ guards
def test_dose_prepare_records_budget_facts_and_parents(dlay, capsys):
    cfg = TJ.filled_cfg()
    TX.ready(dlay, cfg)
    launch = PT.prepare("tvr-xsmoke-dose", L=dlay, cfg=cfg, events=[])
    c = dlay.calls["prepare"][-1]
    assert c["name"] == "vpd-tvr-xsmoke-dose" and c["as_root"] is True and c["est_gph"] == 1.2
    assert c["payload"][-2:] == ["Q3S2,Q3S4", c["run"]] and launch == Path(c["run"]) / "job" / "launch.sh"
    info = json.loads((Path(c["run"]) / "prepared.json").read_text())
    bb = info["block_budget"]
    assert (bb["group"], bb["cap"], bb["ok"], bb["total"]) == ("dose", 25.0, True, 1.2) and "baseline" not in bb
    assert info["round_budget"]["ok"] and info["exploratory"] is True
    f = info["facts"]
    assert f["trainer_code_sha256"] == PT.DOSE_TRAINER_SHA256 and f["trainer_problems"] == []
    assert f["trainer_rule"] == (f"dose: prefix {TRAINER_NEW}, 5 .py shas == the pilot's, starts.json == the Q3SG "
                                 "tree's + the Q3S2 and Q3S4 entries")
    assert f["trainer_code_files"][PT.STARTS_JSON] == PT.STARTS_JSON_SHA256_DOSE
    assert f["prereg_row_date"] == PT.DOSE_ROW_DATE and f["harness_code_sha256"].startswith("64edd575f36b")
    assert "harness_rule" not in f  # the harness is every explore job's (harness_problems)
    row = PT.xrow(PT.prereg_path().read_text(encoding="utf-8"), PT.DOSE_ROW_DATE, PT.DOSE_ROW_NEEDLE)
    assert row and f["prereg_row_sha256"] == TJ.sha(row.encode())
    assert all((dlay.xroot / s).is_dir() for s in D) and not dlay.pilot_root.exists()
    err = capsys.readouterr().err
    assert "dose budget: spent 0.0000 + open 0.0000 + new 1.2000" in err and "block budget" not in err


def test_q3sg_rl_and_pilot_jobs_refuse_on_this_tree(dlay):
    """This tree's trainer is the dose one: every pilot and RL job (recipe prefix 0865dd75f1ee) and every Q3SG job
    (prefix 119da3c84b8d, starts.json == the pilot's + the Q3SG entry) refuses; the dose jobs do not."""
    cfg = TJ.filled_cfg()
    TX.ready(dlay, cfg)
    TX.xverdict(dlay)
    wj(dlay.xsmoke_dir("tvr-xsmoke-q3sg-202611131000") / "smoke_verdict.json",
       {"verdict": "PASS", "failed": [], "starts": ["Q3SG"]})
    dverdict(dlay)
    want = f"trainer code_sha256 {TRAINER_NEW} != frozen prefix {TRAINER_OLD}"
    for job in ("tvr-data", "tvr-msmoke", "tvr-train-q3b", "tvr-eval-q3s-low", "tvr-xsmoke", "tvr-xtrain-q3rd"):
        assert want in PT.prerequisites(job, dlay, cfg), job
    for job in ("tvr-xsmoke-q3sg", "tvr-xtrain-q3sg", "tvr-xeval-q3sg-low"):
        probs = PT.prerequisites(job, dlay, cfg)
        assert any(f"trainer code_sha256 {TRAINER_NEW} != the Q3SG prefix {TRAINER_Q3SG}" in b for b in probs), job
        assert any("changed beyond the Q3SG entry" in b for b in probs), job
    for job in ("tvr-xsmoke-dose", "tvr-xtrain-q3s2", "tvr-xtrain-q3s4"):
        assert PT.prerequisites(job, dlay, cfg) == [], job
    assert PT.x_trainer_problems("tvr-xeval-q3s4-top") == [] and PT.x_trainer_problems("tvr-xsmoke", cfg) == [want]
    with pytest.raises(PT.Refused, match="the Q3SG prefix 119da3c84b8d"):
        PT.prepare("tvr-xtrain-q3sg", L=dlay, cfg=cfg, events=[])


def test_dose_guards_smoke_verdict_trainer_harness_and_row(dlay, monkeypatch, tmp_path):
    cfg = TJ.filled_cfg()
    TX.ready(dlay, cfg)
    assert PT.prerequisites("tvr-xsmoke-dose", dlay, cfg) == []
    # xtrain / xeval: the newest verdict whose starts include the start; the RL and Q3SG smokes' verdicts never count
    TX.xverdict(dlay, "PASS")
    wj(dlay.xsmoke_dir("tvr-xsmoke-q3sg-202611151100") / "smoke_verdict.json",
       {"verdict": "PASS", "failed": [], "starts": ["Q3SG"]})
    for s in D:
        assert any(f"whose starts include {s} (job tvr-xsmoke-dose)" in b
                   for b in PT.prerequisites(f"tvr-xtrain-{s.lower()}", dlay, cfg)), s
    dverdict(dlay, "FAIL", name="tvr-xsmoke-dose-202611151000")
    assert any("not PASS" in b for b in PT.prerequisites("tvr-xtrain-q3s4", dlay, cfg))
    dverdict(dlay, "PASS", name="tvr-xsmoke-dose-202611151030")
    assert PT.prerequisites("tvr-xtrain-q3s2", dlay, cfg) == PT.prerequisites("tvr-xtrain-q3s4", dlay, cfg) == []
    dverdict(dlay, "FAIL", name="tvr-xsmoke-dose-202611151200")  # a newer FAIL: the newest counts
    assert any("tvr-xsmoke-dose-202611151200" in b and "not PASS" in b
               for b in PT.prerequisites("tvr-xtrain-q3s2", dlay, cfg))
    shutil.rmtree(dlay.xsmoke_dir("tvr-xsmoke-dose-202611151200"))
    assert PT.latest_xverdict(dlay, "Q3S4")[0].parent.name == "tvr-xsmoke-dose-202611151030"
    assert PT.latest_xverdict(dlay, "Q3SG")[0].parent.name == "tvr-xsmoke-q3sg-202611151100"  # unchanged
    assert PT.latest_xverdict(dlay, "Q3RD")[0].parent.name == "tvr-xsmoke-202611121000"
    PT.prepare("tvr-xtrain-q3s2", L=dlay, cfg=cfg, events=[])
    c = dlay.calls["prepare"][-1]
    info = json.loads((Path(c["run"]) / "prepared.json").read_text())
    assert "tvr-xsmoke-dose-202611151030" in info["smoke_verdict"]["path"]
    assert info["smoke_verdict"]["verdict"] == "PASS"
    assert c["payload"][2] == "VPD_TVR_ROOT=explore/rl" and c["payload"][-2:] == ["Q3S2", c["run"]]
    assert c["est_gph"] == PT.derived_train("Q3S2", dlay, cfg)["est_high"]
    # the trainer: prefix and invariant (this tree's fingerprint replaced)
    real_tfp = PT.trainer_fingerprint
    fp = real_tfp()
    monkeypatch.setattr(PT, "trainer_fingerprint", lambda root=None: dict(fp, code_sha256=PT.Q3SG_TRAINER_SHA256))
    assert any(f"{TRAINER_Q3SG} != the dose prefix {TRAINER_NEW}" in b
               for b in PT.prerequisites("tvr-xtrain-q3s4", dlay, cfg))
    monkeypatch.setattr(PT, "trainer_fingerprint", lambda root=None: dict(fp, code_files=dict(
        fp["code_files"], **{"vpd/paths.py": "0" * 64})))
    assert any("code_files vpd/paths.py" in b for b in PT.prerequisites("tvr-xsmoke-dose", dlay, cfg))
    monkeypatch.setattr(PT, "trainer_fingerprint", real_tfp)
    assert PT.prerequisites("tvr-xtrain-q3s4", dlay, cfg) == []
    # the harness: the registered one, as every explore job (another run_tvr CODE_FILE refuses the dose jobs too)
    real_h = PT.harness_sha256
    monkeypatch.setattr(PT, "harness_sha256", lambda: "0" * 64)
    for job in ("tvr-xsmoke-dose", "tvr-xtrain-q3s2", "tvr-xeval-q3s4-low"):
        assert any("harness code_sha256 000000000000 != the pilot evals' 64edd575f36b" in b
                   for b in PT.prerequisites(job, dlay, cfg)), job
    with pytest.raises(PT.Refused, match="harness code_sha256 000000000000"):
        PT.prepare("tvr-xtrain-q3s4", L=dlay, cfg=cfg, events=[])
    monkeypatch.setattr(PT, "harness_sha256", real_h)
    assert PT.prerequisites("tvr-xtrain-q3s4", dlay, cfg) == []
    # the row: the dose jobs need exactly one dose row; the RL and Q3SG jobs their own
    text = PT.prereg_path().read_text(encoding="utf-8")
    row = PT.xrow(text, PT.DOSE_ROW_DATE, PT.DOSE_ROW_NEEDLE)
    assert row and row.startswith(f"| {PT.DOSE_ROW_DATE} |")
    p = tmp_path / "PREREG2.md"
    monkeypatch.setattr(PT, "prereg_path", lambda: p)
    p.write_text(text.replace(row + "\n", ""), encoding="utf-8")
    assert any(f"'| {PT.DOSE_ROW_DATE} |" in b and "the dose starts Q3S2 and Q3S4" in b
               for b in PT.prerequisites("tvr-xsmoke-dose", dlay, cfg))
    assert not any(PT.DOSE_ROW_DATE in b for b in PT.prerequisites("tvr-xsmoke", dlay, cfg))
    p.write_text(text + "\n" + row + "\n", encoding="utf-8")
    assert any(f"no single section-17 row '| {PT.DOSE_ROW_DATE} |" in b
               for b in PT.prerequisites("tvr-xtrain-q3s2", dlay, cfg))


def test_x_start_problems_dose_on_the_real_starts(capsys):
    """The real starts.json: Q3S2 / Q3S4 have a code and no eval view, so the processor is read from the checkpoint
    dir (skipped where the checkpoints are not mounted)."""
    from vpd import starts

    for s, code in (("Q3S2", "c0q3_sft_e2"), ("Q3S4", "c0q3_sft_e4")):
        assert PT.main(["served", s]) == 0 and capsys.readouterr().out.strip() == code
    for s in D:
        e = starts.resolve(s)
        if not os.path.lexists(Path(e["path"]) / "config.json"):
            pytest.skip("the dose checkpoints are not mounted here")
        assert PT.x_start_problems(s) == [] and Path(starts.processor_dir(s)) == Path(e["path"])
        assert not os.path.lexists(starts.view_dir(s) / starts.PROCESSOR_MARKER)


def test_dose_xeval_adapter_checks(dlay):
    cfg = TJ.filled_cfg()
    files = TX.ready(dlay, cfg)
    dverdict(dlay)
    with pytest.raises(PT.Refused, match="adapter Q3S2/0W"):
        PT.prepare("tvr-xeval-q3s2-top", L=dlay, cfg=cfg, events=[])
    for s in D:
        TX.build_xadapters(dlay, s, cfg, files)
    assert PT.prerequisites("tvr-xeval-q3s2-low", dlay, cfg) == PT.prerequisites("tvr-xeval-q3s4-top", dlay, cfg) == []
    PT.prepare("tvr-xeval-q3s4-top", L=dlay, cfg=cfg, events=[])
    c = dlay.calls["prepare"][-1]
    assert c["payload"][-3:] == ["Q3S4", "top", c["run"]] and c["est_gph"] <= 5.5
    ad = dlay.xadapter("Q3S2", "ALL") / "meta.json"
    for fn, want in ((lambda m: m.update(code_sha256=PT.Q3SG_TRAINER_SHA256), f"not the dose prefix {TRAINER_NEW}"),
                     (lambda m: m["code_files"].update({"downstream/train/lora.py": "1" * 64}),
                      "code_files downstream/train/lora.py 111111111111"),
                     (lambda m: m["code_files"].update({PT.STARTS_JSON: PT.STARTS_JSON_SHA256_Q3SG}),
                      f"code_files {PT.STARTS_JSON} 8cd21cfa6076"),
                     (lambda m: m.pop("code_files"), "code_files NoneType"),
                     (lambda m: m.update(seed=1002), "meta seed 1002"),
                     (lambda m: m.update(start="Q3S4"), "meta start 'Q3S4'")):
        orig = ad.read_text()
        TX._edit(ad, fn)
        probs = PT.prerequisites("tvr-xeval-q3s2-top", dlay, cfg)
        assert any("adapter Q3S2/ALL" in b and want in b for b in probs), (want, probs)
        ad.write_text(orig)
    assert PT.prerequisites("tvr-xeval-q3s2-top", dlay, cfg) == []


def test_q3sg_and_rl_adapters_still_judged_by_their_registered_code(dlay, tmp_path):
    """On this (dose) tree the Q3SG and RL adapters are still complete for x_adapter_check (so compare_q3sg and
    compare_rl keep validating them); compare_q3sg's trainer invariant still holds, the Q3SG rule on this tree is
    recorded (this_tree problems), not judged."""
    cfg = TJ.filled_cfg()
    files = TX.ready(dlay, cfg)
    for s, code in (("Q3SG", TRAINER_Q3SG), ("Q3RD", TRAINER_OLD)):
        TX.build_xadapters(dlay, s, cfg, files)
        for tier, init in PT.train_tiers(s, cfg):
            rc, msg = PT.x_adapter_check(s, tier, init, dlay, cfg)
            assert rc == 0 and f"code {code}" in msg, (s, tier, msg)
    proot = tmp_path / "pilot"
    for s in ("Q3B", "Q3S"):
        for t in ("0W", "100", "ALL"):
            wj(proot / s / "adapters" / t / "meta.json", {"start": s, **TX.trainer_code(s)})
    inv = CQ.trainer_invariant(dlay.xroot, proot, cfg)
    assert inv["ok"] and inv["this_tree"]["trainer_code_sha256"] == PT.DOSE_TRAINER_SHA256
    assert any("the Q3SG prefix" in b for b in inv["this_tree"]["problems"])


# ------------------------------------------------------------------------------------------ steps / expect / guard
def test_dose_required_steps_and_expect(dlay):
    cfg = TJ.filled_cfg()
    files = TX.build_tiers(dlay, cfg)
    assert PT.required_steps("tvr-xsmoke-dose", cfg) == ["guard", "prereg_committed", "merge_disk"] + [
        f"{x}_{s}" for s in D for x in ("load", "dry", "hfparity", "session", "train", "merge", "merge_parity",
                                        "eval")] + ["verdict"]
    for s in D:
        assert PT.required_steps(f"tvr-xtrain-{s.lower()}", cfg) == PT.required_steps("tvr-xtrain-q3rd", cfg)
        for part in ("top", "low"):
            assert PT.required_steps(f"tvr-xeval-{s.lower()}-{part}", cfg) == PT.required_steps(
                f"tvr-eval-q3s-{part}", cfg)
    # the smoke's expect: a PASS verdict of exactly its starts
    run = dlay.jobs / "tvr-xsmoke-dose-202611151300"
    _steps_file(run, "tvr-xsmoke-dose", cfg)
    vp = dverdict(dlay, name=run.name)
    assert PT.expect("tvr-xsmoke-dose", run, dlay, cfg)["ok"]
    wj(vp, {"verdict": "PASS", "failed": [], "starts": ["Q3S2"]})
    assert PT.expect("tvr-xsmoke-dose", run, dlay, cfg)["failed"] == ["smoke_verdict"]
    assert vp in PT.expected_outputs("tvr-xsmoke-dose", run, dlay, cfg)
    for s in D:
        TX.build_xadapters(dlay, s, cfg, files)
        job = f"tvr-xtrain-{s.lower()}"
        run = dlay.jobs / f"{job}-202611151300"
        _steps_file(run, job, cfg)
        assert PT.expect(job, run, dlay, cfg)["ok"]
        m = dlay.xadapter(s, "100") / "meta.json"
        orig = m.read_text()
        TX._edit(m, lambda x: x.update(code_sha256=PT.Q3SG_TRAINER_SHA256))
        assert PT.expect(job, run, dlay, cfg)["failed"] == ["adapter_100"]
        m.write_text(orig)
        for part in ("top", "low"):
            TX._xevals(dlay, s, cfg, part)
            job = f"tvr-xeval-{s.lower()}-{part}"
            run = dlay.jobs / f"{job}-202611151300"
            _steps_file(run, job, cfg)
            rep = PT.expect(job, run, dlay, cfg)
            assert rep["ok"], rep["failed"]
            paths_ = [Path(c["path"]) for c in rep["items"] if c.get("path")]
            assert paths_ and all(dlay.xroot / s in p.parents for p in paths_)
            assert any(p.parent.name.startswith(f"xrl__{s}__") for p in paths_)


def test_guard_in_job_dose_writes_facts(dlay, monkeypatch):
    cfg = TJ.filled_cfg()
    TX.ready(dlay, cfg)
    run = dlay.jobs / "tvr-xsmoke-dose-202611151400"
    run.mkdir(parents=True)
    rep = PT.guard("tvr-xsmoke-dose", run, dlay, cfg)
    assert rep["ok"], rep["problems"]
    f = json.loads((run / "guard.json").read_text())["facts"]
    assert f["explore"] is True and f["trainer_code_sha256"] == PT.DOSE_TRAINER_SHA256 and f["trainer_problems"] == []
    assert f["trainer_code_files"][PT.STARTS_JSON] == PT.STARTS_JSON_SHA256_DOSE and "dose" in f["trainer_rule"]
    assert f["prereg_row_date"] == PT.DOSE_ROW_DATE and f["harness_code_sha256"].startswith("64edd575f36b")
    rep = PT.guard("tvr-xsmoke-q3sg", run, dlay, cfg)  # the Q3SG smoke on this tree: refused by its trainer rule
    assert not rep["ok"] and any("the Q3SG prefix" in b for b in rep["problems"])


def test_cli_plan_and_budget_show_the_dose_jobs(dlay, capsys):
    TJ.build_cal(dlay)
    assert PT.main(["plan", "--explore"]) == 0
    out = capsys.readouterr().out
    for job in DJOBS:
        assert f"[{job}]" in out, job
    assert f"Q3S2, Q3S4: row {PT.DOSE_ROW_DATE}" in out and "Q3SG: row 2026-11-13" in out
    assert "tvr_xsmoke.sh Q3S2,Q3S4 " in out and "outer 4020 s, soft 3840 s, kill 60 s" in out
    assert "dose budget: spent 0.0000 + open 0.00 = 0.00 of 25.0 GPU-h; the dose jobs' est_high sum" in out
    b = PT.budget(dlay, TJ.filled_cfg(), events=[], explore=True)
    assert set(DJOBS) <= set(b["est_high"]) and b["est_high"]["tvr-xsmoke-dose"] == 1.2


# ------------------------------------------------------------------------------------------ xsmoke verdict
def test_xsmoke_verdict_dose(dlay, monkeypatch):
    cfg = TJ.filled_cfg()
    files = TX.build_tiers(dlay, cfg)
    sm = dlay.xsmoke_dir("tvr-xsmoke-dose-202611151500")
    TX.build_xsmoke(dlay, sm, cfg, files, starts=D)
    v = PT.xsmoke_verdict(sm, dlay, cfg, job="tvr-xsmoke-dose")
    assert v["verdict"] == "PASS", v["failed"]
    assert next(c for c in v["checks"] if c["name"] == "harness_Q3S4")["detail"] == PT.HARNESS_SHA256_PREFIX
    assert v["starts"] == list(D) and v["job"] == "tvr-xsmoke-dose"
    assert {c["name"].rsplit("_", 1)[1] for c in v["checks"]} == set(D)
    tr = next(c for c in v["checks"] if c["name"] == "train_Q3S4")
    assert tr["detail"]["code_sha256"] == TRAINER_NEW and tr["detail"]["code_problems"] == []
    meta = sm / "Q3S2" / "adapters" / "0W" / "meta.json"
    for fn in (lambda m: m.update(code_sha256=PT.Q3SG_TRAINER_SHA256),
               lambda m: m["code_files"].update({PT.STARTS_JSON: PT.STARTS_JSON_SHA256_Q3SG})):
        orig = meta.read_text()
        TX._edit(meta, fn)
        assert PT.xsmoke_verdict(sm, dlay, cfg, job="tvr-xsmoke-dose")["failed"] == ["train_Q3S2"]
        meta.write_text(orig)
    # a session of another harness: its harness check fails (the registered harness, as every explore smoke)
    TX._edit(sm / "Q3S4" / "evals" / "smoke__Q3S4__raw__u" / "results.json", lambda r: r.update(code_sha256="0" * 64))
    assert PT.xsmoke_verdict(sm, dlay, cfg, job="tvr-xsmoke-dose")["failed"] == ["harness_Q3S4"]
    # the CLI passes --job
    PT.main(["xsmoke-verdict", "--smoke", str(sm), "--job", "tvr-xsmoke-dose"])
    w = json.loads((sm / "smoke_verdict.json").read_text())
    assert (w["job"], w["starts"]) == ("tvr-xsmoke-dose", list(D))
    # the default job (the RL starts) on this smoke dir: FAIL, nothing of theirs is there
    v = PT.xsmoke_verdict(sm, dlay, cfg)
    assert v["verdict"] == "FAIL" and v["starts"] == list(PT.RL_STARTS) and v["job"] == "tvr-xsmoke"


# ------------------------------------------------------------------------------------------ payloads
def test_payload_dry_xsmoke_dose(tmp_path):
    res, lines, run = TX.xdry(tmp_path, "tvr_xsmoke.sh", "Q3S2,Q3S4", root=None)
    k = TJ._common_dry_asserts(lines, run, tmp_path, 3840)
    per = []
    for s in D:
        per += ["rm", "raw_bf16:--start", "dry", "hfparity", "eval:raw:c,u,blind", f"reap:session_{s}", "rm",
                "train:0W", "rm", "merge:0W", "merge_parity:--start", "eval:0W:c", f"reap:eval_{s}", "rm"]
    assert k == ["prereg_committed", "guard"] + per + ["xsmoke-verdict", "expect", "DONE"], k
    SM = tmp_path / "tvr" / "runs" / "explore" / "rl" / "smoke" / run.name
    assert f"guard tvr-xsmoke-dose --run {run}" in res.stdout and f"expect tvr-xsmoke-dose --run {run}" in res.stdout
    assert any(ln.endswith(f"xsmoke-verdict --smoke {SM} --job tvr-xsmoke-dose") for ln in lines)
    for w in ("Q3RD", "Q3RS150", "Q3SG", "eval500"):
        assert w not in res.stdout, w
    for s in D:
        ld = next(ln for ln in lines if f"raw_bf16 --start {s} " in ln)
        assert ld.endswith(f"--out /dev/shm/vpd_merged/xsmoke__{s}__raw --report {SM}/{s}/load/load.json")
        tr = next(ln for ln in lines if "n2_train_vpd.py" in ln and "--dry-run" not in ln and f"--start {s} " in ln)
        assert (f"--start {s} --data {tmp_path}/tvr/tiers/v1/warmup0w20.jsonl --out {SM}/{s}/adapters/0W --limit 16 "
                "--epochs 1 --seed 1001 --phase 0W --threads 2 --num-workers 3") in tr
        ev = next(ln for ln in lines if f"--start {s} --tier 0W" in ln)
        assert (f"--name-prefix xsmoke --out-root {SM}/{s}/evals --model-dir /dev/shm/vpd_merged/xsmoke__{s}__0W "
                f"--adapter-meta {SM}/{s}/adapters/0W/meta.json --mechanism-only") in ev


def test_xsmoke_dose_start_list_refusals(tmp_path):
    for bad in (["Q3S4,Q3S2"], ["Q3S2"], ["Q3S4"], ["Q3S2,Q3S4,Q3SG"], ["q3s2,q3s4"], ["Q3S2, Q3S4"],
                ["Q3S2,Q3S4", "x"]):
        r = TX._bash([str(JOBS_DIR / "tvr_xsmoke.sh"), *bad, str(tmp_path / "jobs" / "r")], {"VPD_DRYRUN": "1"},
                     tmp_path)
        assert r.returncode == 2 and "usage: tvr_xsmoke.sh [Q3RD,Q3RS150|Q3SG|Q3S2,Q3S4] RUN" in r.stderr, bad


def test_payload_dry_xtrain_and_xeval_dose(tmp_path):
    for s in D:
        d = tmp_path / s
        d.mkdir()
        res, lines, run = TX.xdry(d, "tvr_train.sh", s)
        assert TJ._kinds(lines) == ["prereg_committed", "guard", "adapter-ok:0W", "train:0W", "adapter-ok:100",
                                    "train:tier100", "adapter-ok:ALL", "train:tierALL", "expect", "DONE"]
        P = d / "tvr" / "runs" / "explore" / "rl" / s
        tr = [ln for ln in lines if "n2_train_vpd.py" in ln]
        assert all(f"--start {s} " in ln and "--epochs auto --seed 1001" in ln for ln in tr)
        assert f"--out {P}/adapters/ALL" in tr[2] and tr[1].endswith(f"--init-adapter {P}/adapters/0W")
        assert f"guard tvr-xtrain-{s.lower()} --run {run}" in res.stdout and "runs/pilot" not in res.stdout
        for part in ("top", "low"):
            dp = d / part
            dp.mkdir()
            res, lines, run = TX.xdry(dp, "tvr_eval.sh", s, part)
            P = dp / "tvr" / "runs" / "explore" / "rl" / s
            evs = [ln for ln in lines if "run_tvr eval" in ln]
            assert evs and all(f"--start {s} " in ln and "--task-set eval500 --name-prefix xrl" in ln
                               and f"--out-root {P}/evals" in ln for ln in evs)
            assert f"guard tvr-xeval-{s.lower()}-{part} --run {run}" in res.stdout
            assert f"--served <code:{s}>" in res.stdout and "runs/pilot" not in res.stdout
    run = str(tmp_path / "jobs" / "r")
    for script, args in (("tvr_train.sh", ["Q3S2", run]), ("tvr_eval.sh", ["Q3S4", "top", run])):
        r = TX._bash([str(JOBS_DIR / script), *args], {"VPD_DRYRUN": "1"}, tmp_path)  # the pilot root: never a dose
        assert r.returncode == 2 and "Q3B Q3S E2" in r.stderr, script
    for script, args in (("tvr_train.sh", ["Q3S8", run]), ("tvr_eval.sh", ["Q3S6", "top", run])):
        r = TX._bash([str(JOBS_DIR / script), *args], {"VPD_DRYRUN": "1", "VPD_TVR_ROOT": "explore/rl"}, tmp_path)
        assert r.returncode == 2 and "(START: Q3RD Q3RS150 Q3SG Q3S2 Q3S4)" in r.stderr, script


def test_nondry_xsmoke_dose(tmp_path):
    payload, env = TX.xfake_tree(tmp_path, "tvr_xsmoke.sh")
    run = tmp_path / "OUT" / "jobs" / "tvr-xsmoke-dose-202611150000"
    res = TX._bash([str(payload), "Q3S2,Q3S4", str(run)], env, tmp_path)
    assert res.returncode == 0, res.stderr[-1500:] + res.stdout[-1500:]
    names = [s["name"] for s in TJ._steps(run)]
    assert (run / "DONE").exists() and PT.steps_check("tvr-xsmoke-dose", run)["ok"]
    assert "train_Q3S2" in names and "train_Q3S4" in names and not any("Q3RD" in n or "Q3SG" in n for n in names)
    SM = tmp_path / "OUT" / "tvr" / "runs" / "explore" / "rl" / "smoke" / run.name
    log = (tmp_path / "fake.log").read_text()
    assert f"vpd.tvr.pilot_tvr xsmoke-verdict --smoke {SM} --job tvr-xsmoke-dose" in log
    assert "guard tvr-xsmoke-dose --run" in log and "expect tvr-xsmoke-dose --run" in log and SM.is_dir()


# ------------------------------------------------------------------------------------------ compare_dose
# ALL c: Q3B 12; Q3S2 12 (d 0, b 1 c 1, p 1: not significant -> the light clause), Q3S4 9 (d -1/8), Q3S 6 (d -1/4):
# monotone.  100 c: Q3B 10, Q3S2 3 (b 0 c 7, p 1/64), Q3S4 5, Q3S 4 (not monotone; description only)
SUCC_D = {("Q3B", "ALL", "c"): set(range(12)), ("Q3S2", "ALL", "c"): set(range(1, 13)),
          ("Q3S4", "ALL", "c"): set(range(9)), ("Q3S", "ALL", "c"): set(range(6)),
          ("Q3B", "100", "c"): set(range(10)), ("Q3S2", "100", "c"): {0, 1, 2}, ("Q3S4", "100", "c"): set(range(5)),
          ("Q3S", "100", "c"): {0, 1, 2, 3}, ("Q3S2", "0W", "c"): {0}, ("Q3S4", "ALL", "u"): {1, 2},
          ("Q3S", "ALL", "u"): {2, 3, 4}, ("Q3S2", "ALL", "blind"): {7}}


@pytest.fixture
def dworld(tmp_path, monkeypatch):
    TJ.use_trainer_tree(monkeypatch, T.WORKTREE, PT.DOSE_TRAINER_SHA256)  # this tree (compare_dose's this_tree)
    monkeypatch.setattr(PT, "EVAL500_N", TX.N_TASKS)
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "c0ffee"})
    L = PT.Layout(tmp_path / "out")
    cfg = TJ.filled_cfg()
    files = TX.build_tiers(L, cfg)
    xroot, proot = tmp_path / "explore_rl", tmp_path / "pilot"
    tasks_dir = tmp_path / "repo" / "data" / "tasks"
    wj(tasks_dir / "eval.json", [{"task_id": t, "dataset": TX.CATS[i % 4][0], "difficulty": TX.CATS[i % 4][1]}
                                 for i, t in enumerate(TX.IDS)])
    for s in D:
        TX.build_xadapters(L, s, cfg, files, root=xroot)
        for tier, arm in C.cells():
            ad = xroot / s / "adapters" / PT.EVAL_ADAPTER[tier] / "adapter.pt"
            TX.write_eval(C.eval_dir(xroot, s, tier, arm, "xrl"), SUCC_D.get((s, tier, arm), set()), s, tier, arm,
                          fsha(ad))  # the registered harness (write_eval: 64edd575f36b...), as every eval
    for s in ("Q3B", "Q3S"):
        for tier, arm in C.cells():
            TX.write_eval(C.eval_dir(proot, s, tier, arm), SUCC_D.get((s, tier, arm), set()), s, tier, arm, "p" * 64)
        for t in ("0W", "100", "ALL"):  # the pilot adapters' records (the pilot's trainer)
            wj(proot / s / "adapters" / t / "meta.json", {"start": s, **TX.trainer_code(s)})
    return {"L": L, "xroot": xroot, "proot": proot, "tasks": tasks_dir, "cfg": cfg,
            "prereg": prereg_with_dose_row(tmp_path)}


def drun(w, **kw):
    kw.setdefault("prereg", w["prereg"])
    return CD.compare(w["xroot"], w["proot"], tasks_dir=w["tasks"], L=w["L"], **kw)


def test_compare_dose_cells_dose_table_h1_and_md(dworld):
    rep = drun(dworld)
    assert rep["schema"] == "vpd.tvr.compare_dose/1" and rep["exploratory"] is True
    assert rep["validation"]["ok"] and rep["trainer_invariant"]["ok"]
    assert list(rep["validation"]["evals"]) == [f"{s}/{t}__{a}" for s in D for t, a in C.cells()]
    assert rep["prereg"]["row_text"].startswith(f"| {PT.DOSE_ROW_DATE} |") and PT.DOSE_ROW_NEEDLE in rep["prereg"][
        "row_text"]
    assert rep["cells"] == ["0W__c", "100__c", "ALL__c", "ALL__u", "ALL__blind"]
    assert rep["registered_cells"] == ["100__c", "ALL__c"] and rep["multiplicity_correction"] == "none"
    assert list(rep["comparisons"]) == ["Q3S2-Q3B", "Q3S4-Q3B", "Q3S2-Q3S", "Q3S4-Q3S"]
    for name, comp in rep["comparisons"].items():
        assert comp["role"] == "registered" and comp["b_is"] == f"only {comp['first']} succeeds"
        assert {k: c["use"] for k, c in comp["cells"].items()} == {
            "0W__c": "description", "100__c": "registered", "ALL__c": "registered", "ALL__u": "description",
            "ALL__blind": "description"}, name
    want = {("Q3S2-Q3B", "ALL__c"): (12, 12, 1, 1), ("Q3S4-Q3B", "ALL__c"): (9, 12, 0, 3),
            ("Q3S2-Q3B", "100__c"): (3, 10, 0, 7), ("Q3S2-Q3S", "100__c"): (3, 4, 0, 1),
            ("Q3S4-Q3S", "ALL__c"): (9, 6, 3, 0), ("Q3S4-Q3S", "ALL__u"): (2, 3, 1, 2),
            ("Q3S2-Q3B", "ALL__blind"): (1, 0, 1, 0), ("Q3S2-Q3B", "0W__c"): (1, 0, 1, 0)}
    for (name, cell), (k1, k2, b, c) in want.items():
        x = rep["comparisons"][name]["cells"][cell]
        p = C.exact_p(b, c)
        assert (x["k_first"], x["k_second"], x["b"], x["c"], x["n"]) == (k1, k2, b, c, TX.N_TASKS), (name, cell)
        assert x["p"] == float(p) and x["p_exact"] == f"{p.numerator}/{p.denominator}" and not x["missing"]
        assert x["wilson95_first"] == C.wilson(k1, TX.N_TASKS, 1.96)
    # the dose table: ALL c then 100 c, epochs 2, 4, 8, each minus Q3B
    dt = rep["dose_table"]
    assert [(r["cell"], r["start"], r["epoch"], r["source"]) for r in dt] == [
        (c, s, e, src) for c in ("ALL__c", "100__c")
        for s, e, src in (("Q3S2", 2, "explore"), ("Q3S4", 4, "explore"), ("Q3S", 8, "pilot"))]
    q3s = next(r for r in dt if r["start"] == "Q3S" and r["cell"] == "ALL__c")
    assert (q3s["k_first"], q3s["k_second"], q3s["b"], q3s["c"], q3s["p_exact"]) == (6, 12, 0, 6, "1/32")
    assert dt[0]["k_first"] == rep["comparisons"]["Q3S2-Q3B"]["cells"]["ALL__c"]["k_first"] and "use" not in dt[0]
    h1 = rep["h1"]
    assert (h1["tier"], h1["arm"], h1["order"], h1["epochs"]) == ("ALL", "c", ["Q3S2", "Q3S4", "Q3S"], [2, 4, 8])
    assert h1["differences_exact"] == {"Q3S2": "0/1", "Q3S4": "-1/8", "Q3S": "-1/4"}
    assert h1["monotone_decreasing"] is True and h1["light_clause"] is True and h1["light_text"] == CD.LIGHT_TEXT
    assert h1["q3s2_vs_q3b"] == {"k_Q3S2": 12, "k_Q3B": 12, "n": TX.N_TASKS, "p": 1.0, "p_exact": "1/1",
                                 "not_significant": True, "positive": False}
    assert h1["text"] == USER_H1 == CD.H1_TEXT
    t = rep["start_table"]
    assert len(t) == 20 and [r["start"] for r in t[:4]] == ["Q3B", "Q3S", "Q3S2", "Q3S4"]
    assert [r["source"] for r in t[:4]] == ["pilot", "pilot", "explore", "explore"]
    assert rep["crosscheck"]["ok"] and rep["crosscheck"]["used_sides_all_checked"] and rep["missing_cells"] == []
    assert {h[:12] for h in rep["harness_code_sha256"]} == {"64edd575f36b"}
    inv = rep["trainer_invariant"]
    assert inv["pilot_reference"]["py_equals_pinned"] is True and inv["this_tree"]["problems"] == []
    assert set(inv["adapters"]) == {f"{s}/{t}" for s in D for t in ("0W", "100", "ALL")}
    md = CD.render_md(rep)
    for w in ("## H1（剂量，读数之前登记，原文）", USER_H1, "| Q3S2 | 2 | 12/24 | 12/24 | +0.0 | 1 |",
              "| Q3S | 8 | 6/24 | 12/24 | -25.0 | 0.03125 |", "单调下降（第 2 轮 > 第 4 轮 > 第 8 轮，差值按精确分数比，严格）：是",
              f"按登记如实写：**{CD.LIGHT_TEXT}**——这是对第 3.5 节的修正，不是对命题的支持。",
              "## 剂量表（描述：同一次训练的第 2、4、8 轮，各减 Q3B）", "## Q3S2 对 Q3B（登记：100、ALL 的 c 臂；其余三格作描述）",
              "## Q3S4 对 Q3S（登记", "| ALL | c | 9/24", "登记 |", "描述 |", "偏乐观", "推断", "全部一致", "同口径",
              "评测框架指纹（results.code_sha256）：64edd575f36b"):
        assert w in md, w
    # compare_rl's and compare_q3sg's own sides never include the dose starts
    for s in D:
        assert CR.side_root(s, dworld["xroot"], dworld["proot"]) == (dworld["proot"], "pilot")
        assert CR.side_root(s, dworld["xroot"], dworld["proot"], CQ.Q3SG_STARTS) == (dworld["proot"], "pilot")


def _pair(k1, k2, p, n=500):
    d = Fraction(k1 - k2, n)
    return {"missing": False, "k_first": k1, "k_second": k2, "n": n, "p": float(Fraction(p)), "p_exact": p,
            "rate_difference": float(d), "rate_difference_exact": f"{d.numerator}/{d.denominator}"}


@pytest.mark.parametrize("q3s2,q3s4,q3s,monotone,light", [
    ((100, 100, "1/1"), (90, 100, "1/20"), (54, 100, "1/1000"), True, True),       # 0 > -2 > -9.2 pp; p 1
    ((101, 100, "1/1000"), (90, 100, "1/20"), (54, 100, "1/1000"), True, True),    # positive: light even if p < 0.05
    ((80, 100, "1/20"), (90, 100, "1/2"), (54, 100, "1/1000"), False, True),       # p exactly 0.05: not significant
    ((80, 100, "49/1000"), (70, 100, "1/2"), (54, 100, "1/1000"), True, False),    # significant and negative
    ((90, 100, "1/2"), (90, 100, "1/2"), (54, 100, "1/1000"), False, True),        # equal: not strictly decreasing
    ((90, 100, "1/2"), None, (54, 100, "1/1000"), None, True),                     # a pair missing: monotone null
    (None, (90, 100, "1/2"), (54, 100, "1/1000"), None, None),
])
def test_h1_readout(q3s2, q3s4, q3s, monotone, light):
    pairs = {s: {"missing": True} if x is None else _pair(*x)
             for s, x in (("Q3S2", q3s2), ("Q3S4", q3s4), ("Q3S", q3s))}
    got = CD.h1_readout(pairs, "ALL", "c")
    assert got["monotone_decreasing"] is monotone and got["light_clause"] is light
    assert got["light_text"] == (CD.LIGHT_TEXT if light else None)
    assert (got["differences"]["Q3S2"] is None) == (q3s2 is None)


def test_compare_dose_refuses_without_the_row(dworld, tmp_path):
    p = tmp_path / "PREREG_norow.md"
    text = dworld["prereg"].read_text(encoding="utf-8")
    p.write_text(text.replace(PT.DOSE_ROW_NEEDLE, "x"), encoding="utf-8")
    with pytest.raises(CD.Refused) as e:
        drun(dworld, prereg=p)
    assert f"no single section-17 row '| {PT.DOSE_ROW_DATE} |" in e.value.message
    p.write_text(text + "\n" + PT.xrow(text, PT.DOSE_ROW_DATE, PT.DOSE_ROW_NEEDLE) + "\n", encoding="utf-8")
    with pytest.raises(CD.Refused) as e:  # twice
        drun(dworld, prereg=p)
    assert f"no single section-17 row '| {PT.DOSE_ROW_DATE} |" in e.value.message


def test_compare_dose_refuses_until_the_ten_dose_evals_validate(dworld):
    xroot = dworld["xroot"]
    d = C.eval_dir(xroot, "Q3S4", "ALL", "u", "xrl")
    breaks = [
        (d / "results.json", lambda r: r.update(valid=False), "not valid"),
        (d / "results.json", lambda r: r.update(mechanism_only=True), "mechanism_only True"),
        (d / "results.json", lambda r: r.update(start="Q3S"), "start 'Q3S' != 'Q3S4'"),
        (d / "results.json", lambda r: r.update(code_sha256="0" * 64), "harness code_sha256 000000000000"),
        (d / "results.json", lambda r: r["adapter"].update(adapter_sha256="0" * 64), "served adapter"),
        (d / "official_check.json", lambda r: r.update(equal=False), "official_check.json equal False"),
        (xroot / "Q3S2" / "adapters" / "ALL" / "meta.json", lambda r: r.update(seed=1002), "meta seed 1002"),
        (xroot / "Q3S2" / "adapters" / "100" / "meta.json", lambda r: r.update(code_sha256=PT.Q3SG_TRAINER_SHA256),
         f"not the dose prefix {TRAINER_NEW}"),
        (xroot / "Q3S4" / "adapters" / "0W" / "meta.json",
         lambda r: r["code_files"].update({PT.STARTS_JSON: PT.STARTS_JSON_SHA256_Q3SG}), "code_files vpd/starts.json"),
    ]
    for path, fn, want in breaks:
        orig = path.read_text()
        TX._edit(path, fn)
        with pytest.raises(CD.Refused) as e:
            drun(dworld)
        assert "not all 10 dose evals" in e.value.message and want in e.value.message, (want, e.value.message)
        path.write_text(orig)
    assert drun(dworld)["validation"]["ok"]
    shutil.rmtree(C.eval_dir(xroot, "Q3S4", "100", "c", "xrl"))
    with pytest.raises(CD.Refused) as e:
        drun(dworld)
    assert "Q3S4/100__c (xrl__Q3S4__100__c)" in e.value.message


def test_compare_dose_trainer_invariant_refusals(dworld):
    """The dose adapters pass x_adapter_check (the pinned shas), but the invariant compares them with what the pilot
    Q3B / Q3S adapters recorded."""
    proot = dworld["proot"]
    metas = [proot / s / "adapters" / t / "meta.json" for s in ("Q3B", "Q3S") for t in ("0W", "100", "ALL")]
    origs = {p: p.read_text() for p in metas}

    def refused(want):
        with pytest.raises(CD.Refused) as e:
            drun(dworld)
        assert "trainer invariant does not hold" in e.value.message and want in e.value.message, e.value.message
        for p, t in origs.items():
            p.write_text(t)

    for p in metas:  # the pilot recorded another n2_train_vpd.py: the dose adapters' .py shas are not the pilot's
        TX._edit(p, lambda m: m["code_files"].update({"vpd/train/n2_train_vpd.py": "4" * 64}))
    refused("code_files vpd/train/n2_train_vpd.py 6c2b4056b337 != the pilot adapters' 444444444444")
    TX._edit(metas[4], lambda m: m["code_files"].update({"vpd/paths.py": "5" * 64}))
    refused("recorded different code_files")
    for p in metas:
        TX._edit(p, lambda m: m["code_files"].update({PT.STARTS_JSON: "6" * 64}))
    refused("STARTS_JSON_SHA256_PILOT")
    assert drun(dworld)["trainer_invariant"]["ok"]


def test_compare_dose_main_and_missing_cells(dworld, monkeypatch, capsys):
    monkeypatch.setattr(PT, "LAYOUT", dworld["L"])
    args = ["--explore-root", str(dworld["xroot"]), "--pilot-root", str(dworld["proot"]), "--tasks-dir",
            str(dworld["tasks"]), "--prereg", str(dworld["prereg"])]
    assert CD.main(args) == 0
    out = dworld["xroot"] / "COMPARE_DOSE.json"
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["schema"] == CD.SCHEMA and out.with_suffix(".md").is_file()
    for other in ("COMPARE_RL.json", "COMPARE_Q3SG.json"):
        assert not (dworld["xroot"] / other).exists()
    assert not (dworld["proot"] / "COMPARE_DOSE.json").exists()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["h1"] == {
        "monotone_decreasing": True, "light_clause": True}
    assert CD.main(args + ["--out", str(dworld["proot"] / "x" / "C.json")]) == 2
    assert "under the pilot root" in capsys.readouterr().err and not (dworld["proot"] / "x").exists()
    # a Q3S ALL c trajectory without a bool result.success (a failed task: the cross-checks still agree): the Q3S
    # cells there and H1's monotone part are missing (never imputed); exit 3
    TX._edit(C.eval_dir(dworld["proot"], "Q3S", "ALL", "c") / "tasks" / TX.IDS[20] / "trajectory.json",
             lambda t: t["result"].update(success=None))
    rep = drun(dworld)
    assert rep["crosscheck"]["ok"] and rep["crosscheck"]["used_sides_all_checked"]
    assert rep["h1"]["monotone_decreasing"] is None and rep["h1"]["light_clause"] is True
    for m in ("Q3S2-Q3S ALL__c", "Q3S4-Q3S ALL__c", "dose table Q3S-Q3B ALL__c", "start table Q3S ALL__c"):
        assert m in rep["missing_cells"], m
    assert "：缺（不判）" in CD.render_md(rep)
    assert CD.main(args) == 3


def test_compare_rl_and_q3sg_ignore_dose_evals(cworld):  # noqa: F811
    """COMPARE_RL is unchanged by dose evals and adapters under the same explore root (its RL starts only)."""
    a = TX.crun(cworld)
    files = PT.tier_manifest_files(cworld["L"])
    for s in D:
        TX.build_xadapters(cworld["L"], s, cworld["cfg"], files, root=cworld["xroot"])
        for tier, arm in C.cells():
            TX.write_eval(C.eval_dir(cworld["xroot"], s, tier, arm, "xrl"), {0}, s, tier, arm, "q" * 64)
    b = TX.crun(cworld)
    a.pop("utc"), b.pop("utc")
    assert a == b and len(a["validation"]["evals"]) == 10
    assert CR.validate(cworld["xroot"], cworld["L"], cworld["cfg"]) == a["validation"]


def test_real_prereg_dose_row():
    """The registered row (once the merge commits it): one row of DOSE_ROW_DATE with the needle, the two starts and
    their checkpoints, the trainer fingerprint, the comparisons and the user's H1 verbatim."""
    text = PT.prereg_path().read_text(encoding="utf-8")
    row = PT.xrow(text, PT.DOSE_ROW_DATE, PT.DOSE_ROW_NEEDLE)
    if row is None:
        pytest.skip("the dose row is not in vpd/PREREG_TVR_PILOT.md yet (a draft until the merge)")
    for w in ("Q3S2", "Q3S4", "checkpoint-308", "checkpoint-616", "c0q3_sft_e2", "c0q3_sft_e4", TRAINER_NEW,
              TRAINER_Q3SG, "runs/explore/rl/Q3S2/", "runs/explore/rl/Q3S4/", "Q3Sn − Q3B", "Q3Sn − Q3S", USER_H1):
        assert w in row, w


def test_cli_help_compare_dose():
    res = subprocess.run([sys.executable, "-B", "-m", "vpd.tvr.compare_dose", "--help"], capture_output=True,
                         text=True, env=dict(os.environ, PYTHONPATH=str(paths.WORKTREE)), cwd=str(paths.WORKTREE),
                         timeout=120)
    assert res.returncode == 0 and "usage" in res.stdout.lower() and "COMPARE_DOSE" in res.stdout


def test_readme_mentions_the_dose_files():
    text = (paths.VPD_PKG / "README.md").read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith("| `")]
    for f in ("tvr/compare_dose.py", "tests/test_tvr_dose.py"):
        assert any(f"`{f}`" in ln.split("|")[1] for ln in rows), f
    for w in ("COMPARE_DOSE", "vpd-tvr-xsmoke-dose", "Q3S2", "Q3S4", TRAINER_NEW):
        assert w in text, w
