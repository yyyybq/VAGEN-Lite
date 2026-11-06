"""reward.make_plans_p3 on a fake reward_runs root (temp folders only)."""

import json

import pytest

from reward import make_plans_p3 as mp


def _root(tmp_path):
    runs = tmp_path / "runs"
    guards = {"kl_max": 0.2, "invalid_max": 0.1, "invalid_window": 10, "keep_fill_min": 1.0}
    base = {"tr": "x", "commit": "abc", "rl": {"standard": "v10_p1_S0", "seed": 20261019, "guards": guards},
            "eval_sets": {"val": str(tmp_path / "eval_sets/eval_val_v2"), "test": str(tmp_path / "eval_sets/eval_test_v2")},
            "endpoint": 300, "pool": "queue", "run_pools": {"rw_g": "zoetrope"}, "rl_name": "rw_b", "note": "n"}
    v4 = {"rl": {"standard": "v10_p1_Dcond", "group_filter": "keep_by_return_std", "keep_std_threshold": 0.3318,
                 "keep_std_floor": 1.0, "progress_reward_scale": 10 / 3}}
    for name, rl_runs in (("rw_b_w2", {"rw_g": v4, "rw_ok": {"rl": {}}}), ("rw_b_w1", {"rw_s0": {}, "rw_o": {"rl": {}}})):
        (runs / name).mkdir(parents=True)
        (runs / name / "plan.json").write_text(json.dumps({**base, "rl_runs": rl_runs}))
    common = {"model_type": "qwen3_vl", "sft_dir": "/sft", "retries": 4, "suspended_grace_min": 10, "eval_only": True,
              "pool": "queue", "job_prefix": "rw"}
    for name, extra in (("rw_p2_st1_p1", {"diagnostics": {"probes": {"P1": "/p1_probe.jsonl"}}}),
                        ("rw_mix_qa_ctl", {"diagnostics": {"downstream_qa": {"sets": {"x0": "/x0"}, "models_per_job": 6}}}),
                        ("rw_p2_st1_easi", {"qa_eval": {"out": "/easi8", "easi": "easi_8"}})):
        (runs / name).mkdir(parents=True)
        (runs / name / "plan.json").write_text(json.dumps({**common, **extra}))
    return tmp_path


def test_d_plans_change_only_the_listed_keys(tmp_path):
    root = _root(tmp_path)
    w2 = json.loads((root / "runs/rw_b_w2/plan.json").read_text())
    for arm, (kind, seed) in mp.D_ARMS.items():
        p = mp.d_plan(root, arm)
        assert p["rl"]["seed"] == seed and list(p["rl_runs"]) == [arm] and "run_pools" not in p
        assert p["commit"] == "COMMIT_PLACEHOLDER" and p["tr"].endswith(mp.D_CHAIN[arm])
        assert p["rl"]["guards"] == w2["rl"]["guards"] and p["endpoint"] == 300
        assert set(p["eval_sets"]) == {"val", "test", "scr59"}
        changed = {k for k in set(p) | set(w2) if p.get(k) != w2.get(k)}
        assert changed <= {"tr", "commit", "rl", "rl_runs", "run_pools", "rl_name", "eval_sets", "note"}
        assert {k for k in set(p["rl"]) | set(w2["rl"]) if p["rl"].get(k) != w2["rl"].get(k)} <= {"seed"}
        spec = p["rl_runs"][arm]
        if kind == "v4":
            assert spec == w2["rl_runs"]["rw_g"]
        elif kind == "s0":
            assert spec == {}
        else:
            assert spec == {"rl": {"invalid_penalty": -1.0}} and spec["rl"]["invalid_penalty"] < 0


def test_a_plans(tmp_path):
    root = _root(tmp_path)
    plans = mp.a_plans(root, "/c0/checkpoint-1232")
    diag, easi = plans["rw_p3a_diag"], plans["rw_p3a_easi"]
    assert len(diag["eval"]["models"]) == 12 and diag["eval"]["models"][mp.C0_NAME]["model"] == "/c0/checkpoint-1232"
    assert diag["diagnostics"]["probes"] == {"P1": "/p1_probe.jsonl"} and "downstream_qa" in diag["diagnostics"]
    assert set(easi["eval"]["models"]) == set(mp.A_EASI) and easi["qa_eval"]["easi"] == "easi_8"
    assert all(m["model"].endswith("/actor/huggingface") for n, m in easi["eval"]["models"].items())
    assert diag["eval_only"] is True and easi["eval_only"] is True


def test_write_refuses_and_scr59(tmp_path):
    root = _root(tmp_path)
    (root / "eval_sets/eval_test_v2").mkdir(parents=True)
    (root / "eval_sets/eval_test_v2/tasks.jsonl").write_text(
        "".join(json.dumps({"task_id": f"t{i}"}) + "\n" for i in range(5)))
    scr = tmp_path / "screened"
    scr.mkdir()
    scr.joinpath("tasks.jsonl").write_text(
        "".join(json.dumps({"task_id": t}) + "\n" for t in ("t1", "n1", "t3", "n2")))
    info = mp.scr59(root, scr)
    assert info["rows"] == 2 and info["source_rows"] == 4
    got = [json.loads(l)["task_id"] for l in (root / "eval_sets/eval_test_v2_screened59/tasks.jsonl").read_text().splitlines()]
    assert got == ["n1", "n2"]
    with pytest.raises(SystemExit):
        mp.scr59(root, scr)
    mp.main(["d", "--root", str(root), "--arm", "rw_s2g"])
    with pytest.raises(SystemExit):
        mp.main(["d", "--root", str(root), "--arm", "rw_s2g"])
