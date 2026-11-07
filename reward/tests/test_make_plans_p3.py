"""reward.make_plans_p3 on a fake reward_runs root (temp folders only)."""

import json

import pytest

from reward import make_plans_p3 as mp


def _root(tmp_path):
    runs = tmp_path / "runs"
    guards = {"kl_max": 0.2, "invalid_max": 0.1, "invalid_window": 10, "keep_fill_min": 1.0}
    base = {"tr": "x", "commit": "abc", "rl": {"standard": "v10_p1_S0", "seed": 20261019, "guards": guards},
            "eval_sets": {"val": str(tmp_path / "eval_sets/eval_val_v2"), "test": str(tmp_path / "eval_sets/eval_test_v2")},
            "endpoint": 300, "pool": "queue", "run_pools": {"rw_g": "zoetrope"}, "rl_name": "rw_b", "note": "n",
            "select_by": "val", "eval": {"every": 50, "options": [], "stop_invalid": {"set": "val", "max": 0.1}}}
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


def test_a2_plans(tmp_path):
    root = _root(tmp_path)
    plans = mp.a2_plans(root)
    diag, easi = plans["rw_p3a2_diag"], plans["rw_p3a2_easi"]
    assert set(diag["eval"]["models"]) == set(mp.A2_CKPTS) and mp.C0_NAME not in diag["eval"]["models"]
    assert set(easi["eval"]["models"]) == set(mp.A2_EASI) and "rw_s2s0_300" not in easi["eval"]["models"]
    assert diag["tr"].endswith("rw_p3a2_diag") and easi["tr"].endswith("rw_p3a2_easi")
    assert "downstream_qa" in diag["diagnostics"] and easi["qa_eval"]["easi"] == "easi_8"
    assert all(m["model"].endswith("/actor/huggingface") for m in diag["eval"]["models"].values())


def test_c_plans(tmp_path):
    root = _root(tmp_path)
    w2 = json.loads((root / "runs/rw_b_w2/plan.json").read_text())
    smoke, arm = mp.c_plan(root, "smoke"), mp.c_plan(root, "arm")
    assert smoke["endpoint"] == 10 and arm["endpoint"] == 300 and list(smoke["rl_runs"]) == ["rw_mx_smoke"]
    for p, name in ((smoke, "rw_mx_smoke"), (arm, "rw_mx_g")):
        spec = p["rl_runs"][name]["rl"]
        um = spec.pop("und_mix")
        assert spec == w2["rl_runs"]["rw_g"]["rl"] and um["batch_size"] == 128 and um["pool"].endswith(mp.C_POOL)
        assert p["rl"] == w2["rl"] and "run_pools" not in p and p["commit"] == "COMMIT_PLACEHOLDER"
    assert "scr59" in arm["eval_sets"] and smoke["eval_sets"] == {} and "select_by" not in smoke
    assert "stop_invalid" not in smoke["eval"] and arm["eval"] == w2["eval"] and arm["select_by"] == "val"
    changed = {k for k in set(arm) | set(w2) if arm.get(k) != w2.get(k)}
    assert changed <= {"tr", "commit", "rl_runs", "run_pools", "rl_name", "eval_sets", "note"}
    assert "und_mix" not in w2["rl_runs"]["rw_g"]["rl"]  # the source plan is not touched


def test_b_plans(tmp_path):
    root = _root(tmp_path)
    w2 = json.loads((root / "runs/rw_b_w2/plan.json").read_text())
    w1 = json.loads((root / "runs/rw_b_w1/plan.json").read_text())
    check, smoke, s0, g = (mp.b_plan(root, w) for w in ("check", "smoke", "s0", "g"))
    assert (check["endpoint"], smoke["endpoint"], s0["endpoint"], g["endpoint"]) == (3, 10, 300, 300)
    assert set(check["rl_runs"]) == {"rw_dgc_s0", "rw_dgc_g"} and list(g["rl_runs"]) == ["rw_dg_g"]
    for p in (check, smoke, s0, g):
        assert p["rl"] == w2["rl"] and "run_pools" not in p and p["commit"] == "COMMIT_PLACEHOLDER"
        for name, spec in p["rl_runs"].items():
            dag = spec["rl"].pop("dagger")
            assert dag["tables"].endswith(mp.B_TABLES) and dag["lam"] == 0.5 and dag["cap"] == 96
            assert dag["audit_only"] is (p is check)
            want = w1["rl_runs"]["rw_s0"].get("rl", {}) if name.endswith("s0") else w2["rl_runs"]["rw_g"]["rl"]
            assert spec["rl"] == want
    for p in (check, smoke):
        assert p["eval_sets"] == {} and "select_by" not in p and "stop_invalid" not in p["eval"]
    for p in (s0, g):
        assert "scr59" in p["eval_sets"] and p["eval"] == w2["eval"] and p["select_by"] == "val"
        changed = {k for k in set(p) | set(w2) if p.get(k) != w2.get(k)}
        assert changed <= {"tr", "commit", "rl_runs", "run_pools", "rl_name", "eval_sets", "note"}
    assert mp.b_plan(root, "smoke", lam=0.25)["rl_runs"]["rw_dgs_g"]["rl"]["dagger"]["lam"] == 0.25
    assert "dagger" not in w2["rl_runs"]["rw_g"]["rl"]
    s2s0, s2g = mp.b_plan(root, "s2s0"), mp.b_plan(root, "s2g")
    assert s2s0["rl"]["seed"] == s2g["rl"]["seed"] == 20261021 and list(s2g["rl_runs"]) == ["rw_s2dg_g"]
    assert {k for k in set(s2g["rl"]) | set(w2["rl"]) if s2g["rl"].get(k) != w2["rl"].get(k)} == {"seed"}
    assert "endpoint" not in g["note"] and "rl seed" in s2g["note"] and "rw_b_w1 rw_s0" in s0["note"]


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
