"""vagen/q38/aoss_offload.py: selection rule, guards and the archive -> delete flow on a fake chain and a fake AOSS.

Run (login node, 1 thread):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    nice -n 19 /mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python -B -m pytest vagen/q38/tests/test_aoss_offload.py \
        -q -x -p no:cacheprovider --basetemp=/mnt/umm/users/yinbaiqiao/q38/tmp_tests/e_cleanup/bt
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from vagen.q38 import aoss_offload as O


class FakeS3:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": self.objects[Key]}

    def upload_file(self, path, bucket, key, **_kw):
        assert bucket == O.BUCKET
        self.objects[key] = os.path.getsize(path)


VALS = {50: 0.30, 100: 0.40, 150: 0.35}  # best 100, last 150: only 50 goes


def make_chain(base: Path, evaluate="done", steps=(50, 100, 150), name="a"):
    tr = base / "q38" / "runs" / "chainx"
    val = tr / "eval_val"
    run_dir = tr / f"rl_run_{name}"
    for s in steps:
        d = run_dir / "checkpoints" / f"global_step_{s}"
        (d / "actor" / "huggingface").mkdir(parents=True)
        (d / "actor" / "huggingface" / "model.safetensors").write_bytes(b"w" * (1000 + s))
        (d / "actor" / "optim_world_size_8_rank_0.pt").write_bytes(b"o" * 3000)
        (d / "data.pt").write_bytes(b"d" * 10)
        (d / "COMPLETE").write_text("")
        res = val / f"{name}_{s:03d}"
        res.mkdir(parents=True)
        (res / "done.json").write_text("{}")
        (res / "results.jsonl").write_text("{}\n")
    (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(str(max(steps)))
    plan = {"tr": str(tr), "sft_dir": str(tr / "sft"), "rl_runs": {name: {}}, "eval_sets": {"val": str(val)},
            "eval": {"every": 50}, "select_by": "val"}
    plan_path = tr / "plan.json"
    plan_path.write_text(json.dumps(plan))
    steps_state = {k: {"state": "done"} for k in ("rl", "rl_extend")}
    steps_state["evaluate"] = {"state": evaluate}
    (tr / "progress.json").write_text(json.dumps({"plan": plan, "steps": steps_state}))
    return plan_path, run_dir


def scorer(folder: Path):
    return VALS.get(int(folder.name.split("_")[-1]))


def make_ctx(base: Path, jobs=None, s3=None) -> O.Ctx:
    (base / "proc").mkdir(exist_ok=True)
    prot = base / "PROTECTED_PATHS.tsv"
    prot.write_text("pattern\towner\tlisted_by\twhy\n" + str(base / "elsewhere") + "\tx\ty\tz\n")
    s3 = s3 if s3 is not None else FakeS3()
    return O.Ctx(home=str(base), scope=(str(base / "q38" / "runs"),), never=(str(base / "q38" / "models"),),
                 protected_tsv=str(prot), spec_dir=str(base / "specs"), records_dir=str(base / "records"),
                 ref_globs=(str(base / "q38" / "plans" / "*.json"),),
                 progress_globs=(str(base / "q38" / "runs" / "*" / "progress.json"),),
                 proc_root=str(base / "proc"), min_age_s=3600, uid=os.getuid(),
                 jobs_fn=(lambda: jobs if jobs is not None else []), client_fn=lambda *_a: s3,
                 clock=lambda: time.time() + 7200)


def test_mentions_is_a_whole_path():
    assert O.mentions("model: /x/global_step_50/actor", "/x/global_step_50")
    assert O.mentions('"/x/global_step_50"', "/x/global_step_50")
    assert not O.mentions("/x/global_step_50", "/x/global_step_5")
    assert not O.mentions("/x/rl_run_ab/checkpoints", "/x/rl_run_a")


def test_select_default_keeps_only_the_last(tmp_path):
    # 定 5 (user 2026-11-20): locally only the checkpoint the decision uses, the last (PREREG_Q38 4)
    plan, run_dir = make_chain(tmp_path)
    items, report = O.select(str(plan), make_ctx(tmp_path), scorer=scorer)
    assert [(it["kind"], it["step"]) for it in items] == [("rl_ckpt", 50), ("rl_ckpt", 100)]
    assert report["runs"]["a"]["kept"] == [150]
    items, report = O.select(str(plan), make_ctx(tmp_path), scorer=scorer, keep_rule="best")
    assert [it["step"] for it in items] == [50, 150] and report["runs"]["a"]["kept"] == [100]


def test_select_keeps_best_and_last_and_writes_nothing(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    ctx = make_ctx(tmp_path)
    items, report = O.select(str(plan), ctx, scorer=scorer, keep_rule="best_and_last")
    assert [(it["kind"], it["step"]) for it in items] == [("rl_ckpt", 50)]
    assert report["runs"]["a"]["best"] == 100 and report["runs"]["a"]["last"] == 150
    assert O.cmd_select(str(plan), ctx, [], "keep", write=False, scorer=scorer, keep_rule="best_and_last") is None
    assert not (tmp_path / "specs").exists() and not (tmp_path / "records").exists()


def test_nothing_before_evaluate_is_done(tmp_path):
    plan, _ = make_chain(tmp_path, evaluate="running")
    items, report = O.select(str(plan), make_ctx(tmp_path), scorer=scorer, keep_rule="best_and_last")
    assert items == [] and "not done" in report["notes"][0]


def test_incomplete_results_skip_the_run(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    (Path(json.loads(plan.read_text())["eval_sets"]["val"]) / "a_050" / "done.json").unlink()
    items, report = O.select(str(plan), make_ctx(tmp_path), scorer=scorer, keep_rule="best_and_last")
    assert items == [] and "50" in report["runs"]["a"]["skipped"]


def test_keep_and_references(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    ctx = make_ctx(tmp_path)
    step50 = str((run_dir / "checkpoints" / "global_step_50").resolve())
    items, _ = O.select(str(plan), ctx, keep=[step50], scorer=scorer, keep_rule="best_and_last")
    assert items == []
    (tmp_path / "q38" / "plans").mkdir(parents=True)
    other = tmp_path / "q38" / "plans" / "q5.json"
    other.write_text(json.dumps({"rl_start_model": step50[:-1] + "/actor/huggingface"}))  # global_step_5: not 50
    assert len(O.select(str(plan), ctx, scorer=scorer, keep_rule="best_and_last")[0]) == 1
    other.write_text(json.dumps({"rl_start_model": step50 + "/actor/huggingface"}))
    items, report = O.select(str(plan), ctx, scorer=scorer, keep_rule="best_and_last")
    assert items == [] and any("named by" in n for n in report["notes"])


def test_guards_unfinished_jobs_and_unreadable_list(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    (run_dir / "trainer").mkdir()
    (run_dir / "trainer" / "submission.json").write_text(json.dumps({"job_id": "pt-run1"}))
    items, report = O.select(str(plan), make_ctx(tmp_path), scorer=scorer, keep_rule="best_and_last")
    rows = [r for it in items for r in O.enumerate_item(it)]
    ctx = make_ctx(tmp_path, jobs=[{"name": "pt-run1", "state": "RUNNING"}])
    assert any("unfinished job pt-run1" in why for _, why in O.guards(items, rows, ctx, report["chain"]))
    ctx = make_ctx(tmp_path, jobs=[{"name": "pt-run1", "state": "SUCCEEDED"}])
    assert O.guards(items, rows, ctx, report["chain"]) == []
    ctx = make_ctx(tmp_path, jobs=[{"name": "pt-eval", "state": "QUEUEING", "cmd": items[0]["root"] + "/actor"}])
    assert any("pt-eval mentions it" in why for _, why in O.guards(items, rows, ctx, report["chain"]))
    ctx = make_ctx(tmp_path)
    ctx.jobs_fn = lambda: None
    assert any("could not be read" in why for _, why in O.guards(items, rows, ctx, report["chain"]))


def test_guards_scope_open_files_age_protected(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    ctx = make_ctx(tmp_path)
    items, report = O.select(str(plan), ctx, scorer=scorer, keep_rule="best_and_last")
    rows = [r for it in items for r in O.enumerate_item(it)]
    (tmp_path / "proc" / "123").mkdir()
    os.symlink(items[0]["root"] + "/actor", tmp_path / "proc" / "123" / "cwd")
    assert any("open on the login node" in why for _, why in O.guards(items, rows, ctx, report["chain"]))
    os.unlink(tmp_path / "proc" / "123" / "cwd")
    ctx.clock = time.time
    assert any("changed in the last" in why for _, why in O.guards(items, rows, ctx, report["chain"]))
    ctx.clock = lambda: time.time() + 7200
    ctx.scope = (str(tmp_path / "somewhere_else"),)
    assert any("outside the q38 scope" in why for _, why in O.guards(items, rows, ctx, report["chain"]))
    ctx.scope = (str(tmp_path / "q38" / "runs"),)
    Path(ctx.protected_tsv).write_text("pattern\to\tl\tw\n" + str(run_dir) + "\to\tl\tw\n")
    assert any("PROTECTED" in why for _, why in O.guards(items, rows, ctx, report["chain"]))


def test_archive_then_delete(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    s3 = FakeS3()
    ctx = make_ctx(tmp_path, s3=s3)
    spec = O.cmd_select(str(plan), ctx, [], "keep", write=True, batch="q38_test_b1", scorer=scorer, keep_rule="best_and_last")
    assert spec and Path(spec).is_file()
    assert O.cmd_archive(spec, ctx, execute=False) == 0 and s3.objects == {}
    with pytest.raises(SystemExit):  # nothing archived yet: delete refuses
        O.cmd_delete(spec, ctx, yes=True, scorer=scorer)
    assert O.cmd_archive(spec, ctx, execute=True) == 0
    step50 = run_dir / "checkpoints" / "global_step_50"
    data = sorted(k for k in s3.objects if not k.startswith(O.MANIFEST_PREFIX))
    assert data == sorted("vagen_lite_archive/q38/runs/chainx/rl_run_a/checkpoints/global_step_50/" + p for p in
                          ("COMPLETE", "data.pt", "actor/optim_world_size_8_rank_0.pt",
                           "actor/huggingface/model.safetensors"))
    assert {O.MANIFEST_PREFIX + "/q38_test_b1/plan.tsv", O.MANIFEST_PREFIX + "/q38_test_b1/manifest.jsonl"} <= set(s3.objects)
    assert O.cmd_delete(spec, ctx, yes=False, scorer=scorer) == 0 and step50.is_dir()  # dry run
    n = O.cmd_delete(spec, ctx, yes=True, scorer=scorer)
    assert n == 4 and not step50.exists()
    assert (run_dir / "checkpoints" / "global_step_100" / "actor" / "optim_world_size_8_rank_0.pt").is_file()
    assert (run_dir / "checkpoints" / "global_step_150" / "COMPLETE").is_file()
    rec = tmp_path / "records" / "q38_test_b1"
    assert (rec / "deleted.tsv").read_text().count("\n") == 5
    assert (run_dir / "aoss_offload" / "global_step_50.q38_test_b1.txt").is_file()
    assert "global_step_50" in (run_dir / "aoss_offload" / "LEDGER.tsv").read_text()
    assert O.MANIFEST_PREFIX + "/q38_test_b1/deleted.tsv" in s3.objects


def test_delete_refuses_changed_file_or_missing_object(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    s3 = FakeS3()
    ctx = make_ctx(tmp_path, s3=s3)
    spec = O.cmd_select(str(plan), ctx, [], "keep", write=True, batch="q38_test_b2", scorer=scorer, keep_rule="best_and_last")
    O.cmd_archive(spec, ctx, execute=True)
    victim = run_dir / "checkpoints" / "global_step_50" / "data.pt"
    key = ctx.key_of(str(victim))
    size = s3.objects.pop(key)
    with pytest.raises(SystemExit):
        O.cmd_delete(spec, ctx, yes=True, scorer=scorer)
    s3.objects[key] = size
    os.utime(victim, (time.time() - 10, time.time() - 10))
    with pytest.raises(SystemExit):
        O.cmd_delete(spec, ctx, yes=True, scorer=scorer)
    assert victim.is_file()


def test_delete_refuses_when_no_longer_selected(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    s3 = FakeS3()
    ctx = make_ctx(tmp_path, s3=s3)
    spec = O.cmd_select(str(plan), ctx, [], "keep", write=True, batch="q38_test_b3", scorer=scorer, keep_rule="best_and_last")
    O.cmd_archive(spec, ctx, execute=True)
    VALS[50] = 0.9  # step 50 became the best (e.g. a re-evaluation): it stays
    try:
        with pytest.raises(SystemExit):
            O.cmd_delete(spec, ctx, yes=True, scorer=scorer)
    finally:
        VALS[50] = 0.30
    assert (run_dir / "checkpoints" / "global_step_50").is_dir()


def test_resume_state_keeps_hf(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    s3 = FakeS3()
    ctx = make_ctx(tmp_path, s3=s3)
    spec = O.cmd_select(str(plan), ctx, [], "archive", write=True, batch="q38_test_b4", scorer=scorer, keep_rule="best_and_last")
    kinds = [(it["kind"], it["step"]) for it in json.loads(Path(spec).read_text())["items"]]
    assert kinds == [("rl_ckpt", 50), ("resume_state", 150)]
    O.cmd_archive(spec, ctx, execute=True)
    O.cmd_delete(spec, ctx, yes=True, scorer=scorer)
    last = run_dir / "checkpoints" / "global_step_150"
    assert (last / "actor" / "huggingface" / "model.safetensors").is_file() and (last / "COMPLETE").is_file()
    assert not (last / "actor" / "optim_world_size_8_rank_0.pt").exists() and not (last / "data.pt").exists()


def test_login_node_core_limit(tmp_path):
    with pytest.raises(SystemExit):
        O.main(["archive", str(tmp_path / "none.json"), "--files", "4", "--threads", "2"], ctx=make_ctx(tmp_path))


def test_delete_reselects_with_the_specs_rule(tmp_path):
    plan, run_dir = make_chain(tmp_path)
    ctx = make_ctx(tmp_path)
    spec = O.cmd_select(str(plan), ctx, [], "keep", write=True, batch="q38_test_b5", scorer=scorer)
    assert json.loads(Path(spec).read_text())["keep_rule"] == "last"
    again, _ = O.select(json.loads(Path(spec).read_text())["plan_file"], ctx, scorer=scorer,
                        keep_rule=json.loads(Path(spec).read_text())["keep_rule"])
    assert [it["step"] for it in again] == [50, 100]
