"""T series: the LoRA experiment option, the short-run check, the NavigateKitchen manifest, plan and analysis."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("train", "demos", "x", "eval"):
    sys.path.insert(0, str(ROOT / sub))
import closed_loop  # noqa: E402
import make_experiment as M  # noqa: E402
import subsets  # noqa: E402
import t5_analyze  # noqa: E402


def test_lora_option():
    assert M.parse_lora("rank=16,alpha=32,dropout=0.05,lr=1e-4") == {"rank": 16, "alpha": 32.0, "dropout": 0.05,
                                                                     "lr": 1e-4}
    assert M.parse_lora("") == {} and M.parse_lora("rank=8,alpha=16,lr=2e-4")["dropout"] == 0.0
    for bad in ("rank=16,alpha=32", "rank=16,alpha=32,lr=1e-4,beta=2"):
        with pytest.raises(ValueError):
            M.parse_lora(bad)


def data_dir(tmp_path):
    d = tmp_path / "data"
    (d / "train").mkdir(parents=True)
    (d / "split_manifest.json").write_text(json.dumps({"datasets": [{"name": "NavigateKitchen"}]}))
    (d / "mixture_spec.json").write_text(json.dumps([["NavigateKitchen/lerobot", 1.0, "panda_omron_robocasa365"]]))
    return d


def test_build_config_puts_lora_in_the_trainer_and_refuses_freeze_modules_with_it(tmp_path):
    recipe = {"framework": {"qwenvl": {}}, "datasets": {"vla_data": {}}, "trainer": {}}
    kw = dict(exp_dir=tmp_path / "EXP", exp_id="T2-B-030-s1", start_path="/start", data_dir=data_dir(tmp_path), seed=1,
              steps=100000, save_interval=25000, log_every=100, full_state_interval=25000)
    lora = M.parse_lora("rank=16,alpha=32,dropout=0.05,lr=1e-4")
    cfg = M.build_config(recipe, lora=lora, **kw)
    assert cfg["trainer"]["lora"] == lora and "freeze_modules" not in cfg["trainer"]
    assert "lora" not in M.build_config(recipe, **kw)["trainer"]
    with pytest.raises(ValueError):
        M.build_config(recipe, lora=lora, freeze_modules="qwen_vl_interface", **kw)


def test_single_task_manifest_is_the_task_row_unchanged(tmp_path):
    row = {"task": "NavigateKitchen", "group": "atomic_seen", "train_ids": [0, 1, 2], "val_ids": [],
           "frames": {"train": 450, "val": 0}}
    src = tmp_path / "target50_100.json"
    src.write_text(json.dumps({"name": "target50_100", "data_root": "/d", "rule": "all human demonstrations",
                               "tasks": [{"task": "OpenDrawer", "train_ids": [5], "frames": {"train": 9}}, row]}))
    out = tmp_path / "navigate_100.json"
    res = subsets.single("NavigateKitchen", src, out, "navigate_100")
    m = json.loads(out.read_text())
    assert m["tasks"] == [row] and m["totals"] == {"tasks": 1, "episodes": 3, "frames": 450} and res["episodes"] == 3
    assert m["name"] == "navigate_100" and len(m["source_manifest_sha256"]) == 64
    with pytest.raises(ValueError):
        subsets.single("NoSuchTask", src, out, "x")


def test_t5_plan_is_navigatekitchen_100_seeds_with_the_recorder():
    eps, p = closed_loop.eval_plan_episodes(ROOT / "x/t5_plan.json")
    assert len(eps) == 100 and {e["task"] for e in eps} == {"NavigateKitchen"}
    assert [e["seed"] for e in eps] == list(range(1000, 1100)) and {e["max_steps"] for e in eps} == {450}
    assert p["end_state"] and p["split"] == "target" and p["execute_steps"] == 16 and p["stable_counter_regions"]
    target = {t["task"]: t["horizon"] for t in json.loads((ROOT / "demos/target50_tasks.json").read_text())["tasks"]}
    assert target["NavigateKitchen"] == 450


def nav_row(seed, success, target_dist, other_dist, tied_other=False):
    fixtures = {"sink": {"type": "Sink", "dist": target_dist}, "stove": {"type": "Stove", "dist": other_dist}}
    if tied_other:
        fixtures["hood"] = {"type": "Hood", "dist": target_dist + 0.01}
    return {"task": "NavigateKitchen", "seed": seed, "status": "complete", "success": success,
            "end_state": {"start": {"target": "sink", "src": "stove"}, "end": {"fixtures": fixtures}}}


def write_eval(exp: Path, step: int, rows):
    d = exp / f"evals/steps_{step}/out/episodes"
    d.mkdir(parents=True)
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_t5_analysis(tmp_path):
    exps = tmp_path / "experiments"
    write_eval(exps / "T5-B-nav-s1", 7500, [nav_row(1000, True, 0.1, 2.0), nav_row(1001, False, 0.4, 2.0),
                                           nav_row(1002, False, 3.0, 0.2), nav_row(1003, False, 3.0, 2.5)])
    write_eval(exps / "T5-RS-nav-s1", 7500, [nav_row(1000, True, 0.1, 2.0), nav_row(1001, True, 0.15, 2.0, True),
                                            nav_row(1002, True, 0.1, 2.0), nav_row(1003, False, 0.3, 2.0)])
    write_eval(exps / "T5-S-nav-s1", 2500, [nav_row(1000, False, 3.0, 3.0)])  # an earlier point only:
    (exps / "T5-S-nav-s1/evals/steps_7500").mkdir(parents=True)  # the last point has no results yet
    res = t5_analyze.analyze(exps, tmp_path / "out")
    b, rs = res["starts"]["B"], res["starts"]["RS"]
    assert b["success"]["k"] == 1 and b["near_target"]["k"] == 2 and b["at_other"]["k"] == 1 and b["episodes"] == 4
    assert rs["success"]["k"] == 3 and rs["near_target"]["k"] == 4 and rs["correct"]["k"] == 4  # the hood shares the spot
    assert res["vs_B"]["RS"]["success"] == {"pairs": 4, "only_this": 2, "only_B": 0, "p": 0.5}
    assert sorted(res["missing"]) == ["RD", "S"]
    assert "到达指令设施附近" in (tmp_path / "out/results.md").read_text(encoding="utf-8")


def test_lora_check_on_a_small_full_state(tmp_path):
    torch = pytest.importorskip("torch")
    import lora_check

    exp, scale = tmp_path / "EXP", 2.0
    run = exp / "run"
    fs = run / "full_state/steps_100/pytorch_model"
    fs.mkdir(parents=True)
    (fs.parent / "latest").write_text("pytorch_model")
    torch.manual_seed(0)
    p = "qwen.model.model.language_model.layers.0.self_attn.q_proj"
    w, a, b = (torch.randn(4, 3).to(torch.bfloat16), torch.randn(2, 3).to(torch.bfloat16),
               (0.01 * torch.randn(4, 2)).to(torch.bfloat16))
    head = torch.randn(5)
    torch.save({"module": {p + ".weight": w, p + ".lora_A": a, p + ".lora_B": b, "action_model.w": head}},
               str(fs / "mp_rank_00_model_states.pt"))
    merged = {p + ".weight": (w.float() + scale * (b.float() @ a.float())).to(torch.bfloat16), "action_model.w": head}
    (run / "checkpoints").mkdir(parents=True)
    torch.save(merged, str(run / "checkpoints/steps_100_pytorch_model.pt"))
    (run / "final_model").mkdir()
    torch.save(merged, str(run / "final_model/pytorch_model.pt"))
    plain = tmp_path / "plain.pt"
    torch.save({k: v.clone() for k, v in merged.items()}, str(plain))
    (run / "metrics.jsonl").write_text("".join(json.dumps({"step": i, "timing/model": 0.5}) + "\n" for i in range(30)))
    (exp / "gpu.csv").write_text("t, 0, 30000 MiB, 90 %\nt, 1, 31000 MiB, 91 %\n")
    res = lora_check.check(exp, plain, scale=scale)
    assert res["ok"] and res["lora_layers"] == 1 and res["max_abs_B"] > 0 and res["gpu_peak_mib"] == 31000
    assert res["speed"]["median_model_seconds"] == 0.5 and res["plain_layout"] and res["final_plain_layout"]
    bad = dict(merged, **{p + ".weight": w})  # not merged: the base weight as it was
    torch.save(bad, str(run / "checkpoints/steps_100_pytorch_model.pt"))
    assert not lora_check.check(exp, plain, scale=scale)["merge_within_bf16_step"]
