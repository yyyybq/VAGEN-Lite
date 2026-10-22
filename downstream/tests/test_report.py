"""Tests of downstream/report (statistics by hand-computed values; the whole build on simulated runs)."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")
from downstream.report import build, collect, simulate, stats  # noqa: E402


def test_wilson_and_exact_tests_match_hand_values():
    r, lo, hi = stats.wilson(10, 20)
    assert r == 0.5 and abs(lo - 0.2993) < 1e-3 and abs(hi - 0.7007) < 1e-3
    assert all(math.isnan(v) for v in stats.wilson(0, 0))
    assert stats.binom_two_sided(0, 5) == 2 / 32
    assert stats.mcnemar_exact(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(stats.holm([0.01, math.nan])[1])


def test_paired_counts_only_common_episodes():
    a = {("T", 1, 1): True, ("T", 2, 1): True, ("T", 3, 1): False, ("T", 4, 1): True, ("T", 9, 1): True}
    b = {("T", 1, 1): True, ("T", 2, 1): False, ("T", 3, 1): False, ("T", 4, 1): False}
    r = stats.paired(a, b)
    assert (r["n"], r["both"], r["only_a"], r["only_b"], r["neither"]) == (4, 1, 2, 0, 1)
    assert r["diff"] == 0.5 and r["rate_a"] == 0.75 and r["rate_b"] == 0.25
    assert r["low"] < 0.5 < r["high"] and r["p"] == 0.5


def test_seed_spread():
    m, s, n = stats.seed_spread([0.2, 0.4, 0.6])
    assert (round(m, 6), round(s, 6), n) == (0.4, 0.2, 3)
    assert math.isnan(stats.seed_spread([0.3])[1])


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    simulate.simulate(root, seed=1)
    return root


def test_simulated_tree_reads_back(sim):
    coll = collect.collect(sim)
    assert len(coll.experiments) == 36 and coll.starts() == ["B", "S", "RS", "RD"]
    assert len(coll.first()) == 4 * 8 + 8 + 24 and len([e for e in coll.evals if e.kind == "repeat"]) == 2
    assert all(len(e.episodes) == 950 for e in coll.first())
    assert stats.common_steps(coll, "100", [1])[-1] == 200000
    cmp = stats.compare(coll, "100", 200000, [1, 2, 3])
    assert cmp["groups"]["B"]["all"][1] == 3 * 950 and cmp["groups"]["B"]["spatial"][1] == 3 * 450
    rs = cmp["vs_reference"]["RS"]
    assert rs["groups"]["all"]["n"] == 2850 and len(rs["tasks"]) == 34
    assert all(0 <= t["p_holm"] <= 1 for t in rs["tasks"].values())


def test_build_writes_a_complete_page(sim, tmp_path):
    out = build.build(sim, tmp_path / "report")
    assert out["simulated"] and out["evaluations"] == 64 and out["examples"] >= 6
    assert set(out["figures"]) == {"loss", "checkpoints", "learning", "heatmap", "forest", "agreement", "efficiency",
                                   "noise"}
    page = (tmp_path / "report/index.html").read_text(encoding="utf-8")
    for anchor in ("summary", "metrics", "training", "checkpoints", "efficiency", "tasks", "agreement", "noise",
                   "examples", "sources"):
        assert f'id="{anchor}"' in page
    assert "模拟数据" in page and page.startswith("<title>RoboCasa 起点对照</title>")
    assert out["html_bytes"] < 16 * 1024 * 1024
    results = json.loads((tmp_path / "report/results.json").read_text(encoding="utf-8"))
    assert results["main_label"].startswith("第三步，100% 数据") and results["noise"][0]["compared"] == 950
    assert results["d5_step"] == 200000 and len(results["summary_lines"]) >= 8


def test_collect_merges_one_card_shards(tmp_path):
    exp = tmp_path / "experiments/DS-RS-100-s1"
    (exp / "run").mkdir(parents=True)
    (exp / "experiment.json").write_text("{}")
    for i in (1, 2):
        d = exp / f"evals/steps_25000/shards/s{i}of2/out/episodes"
        d.mkdir(parents=True)
        rows = [{"task": "OpenDrawer", "seed": 1000 + 2 * i + j, "success": j == 0, "steps": 100, "max_steps": 750,
                 "wall_seconds": 50.0 * i, "status": "complete"} for j in (0, 1)]
        (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "summary.json").write_text(json.dumps({"per_task": {"OpenDrawer": {"n": 2, "successes": 1}},
                                                    "workers": 12, "wall_seconds": 100.0 * i}))
    evs = collect.evaluations(exp)
    assert len(evs) == 1 and evs[0].kind == "first" and evs[0].step == 25000 and len(evs[0].episodes) == 4
    assert evs[0].successes == 2 and evs[0].summary["shards"] == 2 and evs[0].summary["episode_wall_mean_seconds"] == 75.0
    (exp / "evals/steps_25000/shards/s2of2/out/episodes/summary.json").unlink()
    assert collect.evaluations(exp) == []  # incomplete until every shard has a result


def test_build_before_any_evaluation_shows_losses_only(tmp_path):
    root = tmp_path / "runs"
    exp = root / "experiments/DS-B-100-s1"
    (exp / "run").mkdir(parents=True)
    (exp / "experiment.json").write_text("{}")
    (exp / "run/metrics.jsonl").write_text("".join(json.dumps({"step": s, "action_dit_loss": 1 / s}) + "\n"
                                                   for s in range(100, 5001, 100)))
    out = build.build(root, tmp_path / "report")
    assert out["figures"] == ["loss"] and out["examples"] == 0 and not out["simulated"]
    page = (tmp_path / "report/index.html").read_text(encoding="utf-8")
    assert "还没有完成的闭环评测" in page and "模拟数据" not in page


def test_task_level_pairs_by_task():
    r = stats.task_level([0.1, 0.2, 0.3, 0.4, 0.5, -0.1, 0.0, math.nan])
    assert (r["tasks"], r["tasks_up"], r["tasks_down"]) == (7, 5, 1)
    assert r["task_mean_diff"] == pytest.approx(1.4 / 7) and r["sign_p"] == pytest.approx(14 / 64)
    assert r["task_low"] < r["task_mean_diff"] < r["task_high"]
    assert stats.task_level([])["tasks"] == 0 and math.isnan(stats.task_level([])["task_mean_diff"])


def test_noise_floor_of_a_repeated_checkpoint(sim):
    floor = stats.noise_floor(collect.collect(sim))
    assert len(floor) == 2 and all(f["episodes_compared"] == 950 for f in floor)
    f = floor[0]
    assert f["task_level"]["tasks"] == 34 and 0 <= f["task_mean_abs_diff"] <= f["task_max_abs_diff"] <= 1


def test_stage0_build_reads_the_stopped_runs(tmp_path):
    import shutil

    root = tmp_path / "sim"
    simulate.simulate(root, seed=2)
    for exp in (root / "experiments").glob("DS-*"):
        if not exp.name.endswith("-100-s1"):
            shutil.rmtree(exp)
            continue
        (exp / "run").rename(exp / build.STAGE0_RUN)
        ev = exp / f"evals_{build.STAGE0_RUN}"
        ev.mkdir()
        for d in (exp / "evals").glob("steps_*"):
            step = int(d.name.split("_")[1])
            if step <= 100000:
                d.rename(ev / d.name)
        shutil.rmtree(exp / "evals")
    assert build.build(root, tmp_path / "plain")["evaluations"] == 0  # the normal report does not read them
    out = build.build(root, tmp_path / "report0", stage0_mode=True)
    assert out["evaluations"] == 16 and "checkpoints" in out["figures"] and "learning" not in out["figures"]
    results = json.loads((tmp_path / "report0/results.json").read_text(encoding="utf-8"))
    s0 = results["stage0"]
    assert s0["steps"] == [25000, 50000, 75000, 100000] and s0["latest_common"] == 100000
    assert s0["starts"] == ["B", "S", "RS", "RD"] and len(s0["spatial"]) == 9
    assert all(x["in_band"] == (0.1 <= x["mean"] <= 0.6) for x in s0["spatial"])
    assert all((r["from"], r["to"]) == (75000, 100000) and r["all"]["n"] == 950 and r["spatial"]["n"] == 450
               for r in s0["flat"].values())
    assert results["finals"] == {} and results["main_label"].startswith("阶段 0") and len(results["noise_floor"]) == 2
    page = (tmp_path / "report0/index.html").read_text(encoding="utf-8")
    assert 'id="stage0"' in page and "最后两个检查点之差" in page and "评测噪声下限" in page
