from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from data_gen.interiorgs_v10 import qa_eval
from data_gen.interiorgs_v10 import train_chain as TC
from vagen.q38.base_zs import make_plan as MP
from vagen.q38s4.tools import eval_plan as EP

HEAD = ["nfp_head.fc1.weight", "nfp_head.fc1.bias", "nfp_head.fc2.weight", "nfp_head.fc2.bias"]
TRUNK = ["model.language_model.layers.0.mlp.up_proj.weight", "lm_head.weight"]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def bf16_copy(runs: Path, arm: str, step: int, head: bool) -> Path:
    copy = runs / arm / "ckpt_bf16" / f"checkpoint-{step}"
    copy.mkdir(parents=True)
    (copy / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    keys = TRUNK + (HEAD if head else [])
    (copy / EP.INDEX).write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}))
    (copy / "bf16_manifest.json").write_text(json.dumps({"source": str(runs / arm / "ckpt" / f"checkpoint-{step}")}))
    return copy


def stripped(copy: Path, folder: Path, over=None) -> Path:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text((copy / "config.json").read_text())
    (folder / EP.INDEX).write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in TRUNK}}))
    man = {"source": str(copy), "dst": str(folder), "check": {"ok": True},
           "source_bf16_manifest_md5": md5(copy / "bf16_manifest.json"), **(over or {})}
    (folder / "strip_manifest.json").write_text(json.dumps(man))
    return folder


def test_easi_jobs_are_qa_evals():
    assert tuple(qa_eval.EASI_8_JOBS) == EP.EASI_JOBS


def test_check_model_nonfp_is_its_bf16_copy(tmp_path):
    runs = tmp_path / "runs"
    copy = bf16_copy(runs, "c_nonfp", 1000, head=False)
    got = EP.check_model("c_nonfp", 1000, copy, runs)
    assert got["bf16_copy"] == str(copy) and got["bf16_manifest_md5"] == md5(copy / "bf16_manifest.json")
    other = tmp_path / "elsewhere"
    other.mkdir()
    for f in copy.iterdir():
        (other / f.name).write_bytes(f.read_bytes())
    with pytest.raises(SystemExit, match="itself"):
        EP.check_model("c_nonfp", 1000, other, runs)
    with pytest.raises(SystemExit, match="no bf16_manifest"):
        EP.check_model("c_nonfp", 2000, copy, runs)  # step 2000's copy does not exist


def test_check_model_nfp_needs_the_strip_copy(tmp_path):
    runs = tmp_path / "runs"
    copy = bf16_copy(runs, "c_nfp", 500, head=True)
    with pytest.raises(SystemExit, match="nfp_head"):
        EP.check_model("c_nfp", 500, copy, runs)
    dst = stripped(copy, tmp_path / "evalprep/c_nfp_500_nohead")
    got = EP.check_model("c_nfp", 500, dst, runs)
    assert got["strip_manifest_md5"] == md5(dst / "strip_manifest.json")
    # a strip copy of another step, a failed check, another bf16 copy, a moved copy
    for i, over in enumerate(({"source": str(runs / "c_nfp/ckpt_bf16/checkpoint-1000")}, {"check": {"ok": False}},
                              {"source_bf16_manifest_md5": "0" * 32}, {"dst": str(tmp_path / "x")})):
        bad = stripped(copy, tmp_path / f"bad{i}", over)
        with pytest.raises(SystemExit):
            EP.check_model("c_nfp", 500, bad, runs)
    nomanifest = tmp_path / "noman"
    nomanifest.mkdir()
    for f in ("config.json", EP.INDEX):
        (nomanifest / f).write_bytes((dst / f).read_bytes())
    with pytest.raises(SystemExit, match="strip_manifest"):
        EP.check_model("c_nfp", 500, nomanifest, runs)


def test_check_model_type_and_source(tmp_path):
    runs = tmp_path / "runs"
    copy = bf16_copy(runs, "c_nonfp", 1000, head=False)
    (copy / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    with pytest.raises(SystemExit, match="model_type"):
        EP.check_model("c_nonfp", 1000, copy, runs)
    (copy / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (copy / "bf16_manifest.json").write_text(json.dumps({"source": str(runs / "c_nonfp/ckpt/checkpoint-500")}))
    with pytest.raises(SystemExit, match="source"):
        EP.check_model("c_nonfp", 1000, copy, runs)


def test_weight_keys_single_file(tmp_path):
    save_file({"a.weight": torch.zeros(2), "nfp_head.fc1.bias": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
    assert sorted(EP.weight_keys(tmp_path)) == ["a.weight", "nfp_head.fc1.bias"]
    with pytest.raises(SystemExit):
        EP.weight_keys(tmp_path / "none")


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """Tiny v2 / screened sources and the base's copies made by make_plan.write_set, patched into make_plan."""
    src = tmp_path / "src"
    src.mkdir()

    def jl(name, rows):
        p = src / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    val = [{"task_id": f"v{i}", "x": i} for i in range(5)]
    test = [{"task_id": f"t{i}", "x": i} for i in range(4)]
    new = [{"task_id": f"n{i}", "x": i} for i in range(59)]
    p_val, p_test = jl("val.jsonl", val), jl("test.jsonl", test)
    p_st, p_sv = jl("st.jsonl", test[:2] + new), jl("sv.jsonl", val[:3])
    sha = lambda p: MP.sha(p)[:12]  # noqa: E731
    monkeypatch.setattr(MP, "SOURCES", {"q38bz_eval_val_v2": (p_val, sha(p_val), 5),
                                        "q38bz_eval_test_v2": (p_test, sha(p_test), 4)})
    monkeypatch.setattr(MP, "SCREENED_TEST", (p_st, sha(p_st), 61))
    monkeypatch.setattr(MP, "SCREENED_VAL", (p_sv, sha(p_sv), 3))
    base = tmp_path / "base"
    MP.write_set(base, "q38bz_eval_val_v2", p_val.read_bytes(), {"source": str(p_val)})
    MP.write_set(base, "q38bz_eval_test_v2", p_test.read_bytes(), {"source": str(p_test)})
    MP.write_set(base, MP.NEW59, new, {"source": str(p_st)})
    return base


def test_write_sets_match_the_base(tmp_path, sources):
    tr = tmp_path / "tr"
    got = EP.write_sets(tr, "full", "c_nfp", 1000, ["val", "test", "new59"], base=sources)
    assert set(got) == {"val", "test", "new59"}
    for key, rec in got.items():
        f = Path(rec["dir"]) / "tasks.jsonl"
        assert rec["same_as_base"] and rec["md5"] == rec["base_md5"] == md5(f)
        assert Path(rec["dir"]).name == f"c4f_c_nfp_1000_" + EP.SET_KEYS[key][len("q38bz_"):]
        assert not (f.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))  # read-only
        assert json.loads((Path(rec["dir"]) / "source.json").read_text())["sha256"] == MP.sha(f)
    assert len(MP.rows(Path(got["new59"]["dir"]) / "tasks.jsonl")) == 59


def test_write_sets_stop_on_a_differing_base(tmp_path, sources):
    f = sources / "sets/q38bz_eval_test_v2/tasks.jsonl"
    os.chmod(f, 0o644)
    f.write_text(f.read_text() + "\n")
    with pytest.raises(SystemExit, match="differs from the base"):
        EP.write_sets(tmp_path / "tr", "full", "c_nonfp", 2000, ["val", "test", "new59"], base=sources)


def test_write_sets_check_the_sources(tmp_path, sources):
    p, _, n = MP.SOURCES["q38bz_eval_val_v2"]
    MP.SOURCES["q38bz_eval_val_v2"] = (p, "000000000000", n)
    with pytest.raises(SystemExit, match="sha256"):
        EP.write_sets(tmp_path / "tr", "quick", "c_nfp", 3000, ["val"], base=sources)


def test_set_names_new_for_every_run():
    names = {EP.set_name(k, a, s, key) for k in EP.STEPS for a in EP.ARMS for s in EP.STEPS[k] for key in EP.SET_KEYS}
    assert len(names) == sum(len(v) for v in EP.STEPS.values()) * len(EP.ARMS) * len(EP.SET_KEYS)
    assert not names & set(EP.SET_KEYS.values())


def chain_jobs(plan, tmp_path):
    """The evaluate step's jobs of a plan, by the training chain's own code (nothing submitted)."""
    chain = TC.Chain(json.loads(json.dumps(plan)), sleep=lambda s: None)
    got = {}
    chain.eval_jobs = lambda step, jobs: got.update(jobs)
    chain.tables = lambda: None
    chain.evaluate()
    return got


def fake_sets(tr, kind, arm, step):
    out = {}
    for key in EP.SET_KEYS:
        d = tr / "sets" / EP.set_name(kind, arm, step, key)
        d.mkdir(parents=True)
        out[key] = {"dir": str(d)}
    return out


def test_length_plan_jobs(tmp_path):
    tr, model = tmp_path / "len", tmp_path / "m"
    plan = EP.build("length", "c_nfp", 500, model, "abcdef12", tr)
    assert plan["eval_only"] and plan["pool"] == "h800" and plan["model_type"] == "qwen3_5"
    jobs = chain_jobs(plan, tmp_path)
    assert set(jobs) == {f"qa_{b}__c_nfp_500" for b in EP.EASI_JOBS}
    for name, job in jobs.items():
        out, result, m, _, args, mode, *cards = job
        assert result == "c_nfp_500" and m == str(model) and mode == "qa" and out == tr / "qa" / out.name
        assert args[:2] == ["easi", "--benchmark"] and ["--limit", "50"] == args[args.index("--limit"):][:2]
        assert "--no-stack-dump" in args
        assert cards == ([2] if out.name == "site_video" else [])
        assert ("--cards" in args) == (out.name == "site_video")


def test_quick_plan_jobs(tmp_path):
    tr, model = tmp_path / "quick", tmp_path / "m"
    sets = fake_sets(tr, "quick", "c_nonfp", 3000)
    plan = EP.build("quick", "c_nonfp", 3000, model, "abcdef12", tr, sets)
    assert EP.TABLE_CMD.format(eval_dir=sets["val"]["dir"]) in plan["note"]  # the chain makes no table for val
    jobs = chain_jobs(plan, tmp_path)
    assert set(jobs) == {"val__c_nonfp_3000_t08", "qa_probe__c_nonfp_3000"}
    val = jobs["val__c_nonfp_3000_t08"]
    assert val[0] == Path(sets["val"]["dir"]) and val[1] == "c_nonfp_3000_t08" and list(val[4]) == MP.T08
    probe = jobs["qa_probe__c_nonfp_3000"]
    assert probe[4][:3] == ["probe", "--probe", str(MP.PROBE)] and "--max-tokens" not in probe[4]
    assert "--limit" not in probe[4]


def test_full_plan_jobs(tmp_path):
    tr, model = tmp_path / "full", tmp_path / "m"
    sets = fake_sets(tr, "full", "c_nfp", 3441)
    jobs = chain_jobs(EP.build("full", "c_nfp", 3441, model, "abcdef12", tr, sets), tmp_path)
    name = "c_nfp_3441"
    want = ({f"{k}__{name}" for k in EP.SET_KEYS} | {f"{k}__{name}_t08" for k in EP.SET_KEYS}
            | {f"qa_{b}__{name}" for b in EP.EASI_JOBS} | {f"qa_p0__{name}", f"qa_probe__{name}",
                                                            f"qa_probe1024__{name}"})
    assert set(jobs) == want and len(jobs) == 18
    for k in EP.SET_KEYS:
        assert list(jobs[f"{k}__{name}"][4]) == MP.GREEDY and list(jobs[f"{k}__{name}_t08"][4]) == MP.T08
        assert jobs[f"{k}__{name}"][0] == Path(sets[k]["dir"])
    for b in EP.EASI_JOBS:
        args = jobs[f"qa_{b}__{name}"][4]
        assert "--no-stack-dump" in args and "--limit" not in args
    p0 = jobs[f"qa_p0__{name}"][4]
    assert p0[:3] == ["p0", "--selection", str(MP.P0_SEL)] and p0[p0.index("--res") + 1] == "256"
    assert jobs[f"qa_probe1024__{name}"][4][-2:] == ["--max-tokens", "1024"]
    assert jobs[f"qa_probe1024__{name}"][0] == tr / "qa/probe1024"


@pytest.mark.skipif(not (EP.BASE / "plan.json").is_file(), reason="the base zero-shot plan (cluster only)")
def test_plans_match_the_base_options():
    base = json.loads((EP.BASE / "plan.json").read_text())
    assert base["eval"]["models"]["q38base"]["options"] == MP.GREEDY
    assert base["eval"]["variants"][0]["options"] == MP.T08
    assert base["qa_eval"]["p0"] == str(MP.P0_SEL) and base["qa_eval"]["probe"] == str(MP.PROBE)
    easi = json.loads(Path("/mnt/umm/users/yinbaiqiao/q38/runs/q38-base-zs-easi/plan.json").read_text())
    assert easi["qa_eval"]["easi"] == "easi_8" and easi["qa_eval"]["cards"] == {"site_video": 2}


def test_check_qa_file(tmp_path):
    f, done = tmp_path / "p.jsonl", tmp_path / "done.json"
    f.write_text('{"a": 1}\n{"a": 2}\n')
    done.write_text("{}")
    os.utime(f, (1000, 1000))
    os.utime(done, (2000, 2000))
    assert EP.check_qa_file(f, 2, done)["sha256"] == MP.sha(f)
    with pytest.raises(SystemExit, match="rows"):
        EP.check_qa_file(f, 3, done)
    os.utime(f, (3000, 3000))
    with pytest.raises(SystemExit, match="modified after"):
        EP.check_qa_file(f, 2, done)


@pytest.mark.parametrize("argv,msg", [
    (["full", "--arm", "c_nfp", "--step", "1500"], "not in"),
    (["quick", "--arm", "c_nfp", "--step", "1000"], "not in"),
    (["length", "--arm", "c_nonfp", "--step", "500"], "C-NFP"),
    (["full", "--arm", "c_nfp", "--step", "1000", "--commit", "deadbeef"], "HEAD"),
])
def test_main_refuses(argv, msg, tmp_path):
    if "--commit" not in argv:
        argv = argv + ["--commit", "deadbeef"]
    with pytest.raises(SystemExit, match=msg):
        EP.main(argv + ["--model", str(tmp_path)])
