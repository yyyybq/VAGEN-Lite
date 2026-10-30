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
    assert all(len(e.episodes) == 943 for e in coll.first())  # 950 minus the 7 excluded (10-23, OpenDrawer 1012 on 10-30)
    assert stats.common_steps(coll, "100", [1])[-1] == 200000
    cmp = stats.compare(coll, "100", 200000, [1, 2, 3])
    assert cmp["groups"]["B"]["all"][1] == 3 * 943 and cmp["groups"]["B"]["spatial"][1] == 3 * 446
    rs = cmp["vs_reference"]["RS"]
    assert rs["groups"]["all"]["n"] == 3 * 943 and len(rs["tasks"]) == 34
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
    assert results["main_label"].startswith("第三步，100% 数据") and results["noise"][0]["compared"] == 943
    assert results["d5_step"] is None and len(results["summary_lines"]) >= 8  # 200k is the final step, not "D5"
    assert set(results["e2"]) == {"B", "S", "RS", "RD"} and set(results["e2"]["RS"]) == {"010", "030"}


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
    assert len(floor) == 2 and all(f["episodes_compared"] == 943 for f in floor)
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
    assert all((r["from"], r["to"]) == (75000, 100000) and r["all"]["n"] == 943 and r["spatial"]["n"] == 446
               for r in s0["flat"].values())
    assert results["finals"] == {} and results["main_label"].startswith("阶段 0") and len(results["noise_floor"]) == 2
    rule = s0["step_rule"]
    assert rule["complete"] and set(rule["starts"]) == {"B", "S", "RS", "RD"} and rule["steps"] in (100000, 200000)
    for r in rule["starts"].values():
        assert r["n"] == 943 and r["within"] == (r["wilcoxon_p"] > 0.05 and abs(r["task_mean_diff"]) < 0.03)
    assert (rule["steps"] == 100000) == all(r["within"] for r in rule["starts"].values())
    assert all({"mean_all", "low_all", "high_all", "in_band_all", "band_certain", "per_step"} <= set(x)
               for x in s0["spatial"])
    assert len(s0["evaluations"]) == 16 + 2 and not any(r["missing"] for r in s0["evaluations"])  # the 2 repeats at 25k
    fl = s0["fluctuation"]
    assert len(fl["within"]) == 4 * 3 and len(fl["between"]) == 3 * 4 and fl["summary"]["within_all"]["pairs"] == 12
    page = (tmp_path / "report0/index.html").read_text(encoding="utf-8")
    assert 'id="stage0"' in page and "最后两个检查点之差" in page and "评测噪声下限" in page and "阶段 1 的步数规则" in page
    assert "检查点之间的波动" in page and "缺回合的评测" not in page


def test_step_rule_by_hand(tmp_path):
    """Two tasks; B and S equal at 50k and 100k (within), RS and RD gain 40 points on both tasks (not within)."""
    root = tmp_path / "runs"
    tasks = {"OpenDrawer": 50, "CloseFridge": 20}
    gain = {"B": 0.0, "S": 0.0, "RS": 0.4, "RD": 0.4}
    for start, g in gain.items():
        exp = root / f"experiments/DS-{start}-100-s1"
        exp.mkdir(parents=True)
        (exp / "experiment.json").write_text("{}")
        for step, rate in ((50000, 0.2), (100000, 0.2 + g)):
            d = exp / f"evals_{build.STAGE0_RUN}/steps_{step}/out/episodes"
            d.mkdir(parents=True)
            rows = [{"task": t, "seed": 1000 + i, "success": i < round(rate * n), "steps": 10, "max_steps": 100,
                     "status": "complete"} for t, n in tasks.items() for i in range(n)]
            (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (d / "summary.json").write_text(json.dumps({"per_task": {t: {"n": n} for t, n in tasks.items()}}))
    coll = collect.collect(root, evals_name=f"evals_{build.STAGE0_RUN}", run=build.STAGE0_RUN)
    rule = build.step_rule(coll, coll.starts())
    assert rule["complete"] and rule["steps"] == 200000 and rule["pooled_steps"] == 200000
    assert rule["starts"]["B"]["within"] and not rule["starts"]["RS"]["within"]
    assert rule["starts"]["RS"]["task_mean_diff"] == pytest.approx(0.4) and rule["starts"]["RS"]["p"] < 0.05


def test_noise_floor_and_task_diffs_use_only_common_episodes(tmp_path):
    """The repeat lacks one episode that succeeded in the first evaluation and is identical otherwise: every
    difference is 0 (not -1/20 on that task, as when each side's task rate used all its own episodes)."""
    assert stats.common_task_diffs({("T", 1): True, ("T", 2): False, ("U", 1): True},
                                   {("T", 1): True, ("T", 2): True, ("U", 2): False}) == [0.5]
    exp = tmp_path / "experiments/DS-B-100-s1"
    exp.mkdir(parents=True)
    (exp / "experiment.json").write_text("{}")
    for name, skip in (("steps_100000", None), ("steps_100000_repeat", ("TurnOnMicrowave", 1000))):
        d = exp / f"evals/{name}/out/episodes"
        d.mkdir(parents=True)
        rows = [{"task": t, "seed": 1000 + i, "success": i < k, "steps": 10, "max_steps": 100, "status": "complete"}
                for t, k in (("TurnOnMicrowave", 1), ("CloseFridge", 5)) for i in range(20) if (t, 1000 + i) != skip]
        (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "summary.json").write_text(json.dumps({"per_task": {}}))
    coll = collect.collect(tmp_path)
    f = stats.noise_floor(coll)[0]
    assert (f["episodes_compared"], f["episodes_changed"]) == (39, 0)
    assert f["pooled_diff"] == 0 and f["task_max_abs_diff"] == 0 and f["task_mean_abs_diff"] == 0
    assert build.noise_rows(coll)[0]["rate_diff"] == 0


def _write_eval(d: Path, tasks: dict, summary: bool):
    """tasks: name -> (successes, done, planned); seeds 1000.., the first `successes` succeed."""
    d.mkdir(parents=True)
    (d / "plan.json").write_text(json.dumps({"episodes": [{"task": t, "seed": 1000 + i}
                                                          for t, (_, _, n) in tasks.items() for i in range(n)]}))
    rows = [{"task": t, "seed": 1000 + i, "success": i < k, "steps": 10, "max_steps": 100, "status": "complete"}
            for t, (k, done, _) in tasks.items() for i in range(done)]
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if summary:
        (d / "summary.json").write_text(json.dumps({"per_task": {}}))


def test_partial_evaluations_count_their_completed_episodes(tmp_path):
    """collect(partial=True): an evaluation without summary.json (its job deleted, or still queued) counts with the
    episodes it completed; every evaluation lists the planned episodes it lacks, never the excluded ones."""
    exp = tmp_path / "experiments/DS-RD-100-s1"
    exp.mkdir(parents=True)
    (exp / "experiment.json").write_text("{}")
    ev = exp / "evals"
    _write_eval(ev / "steps_25000/out/episodes", {"OpenDrawer": (2, 3, 4), "DeliverStraw": (0, 0, 7)}, summary=True)
    _write_eval(ev / "steps_75000/out/episodes", {"OpenDrawer": (1, 2, 4)}, summary=False)
    _write_eval(ev / "steps_50000/shards/s1of2/out/episodes", {"OpenDrawer": (1, 2, 2)}, summary=True)
    _write_eval(ev / "steps_50000/shards/s2of2/out/episodes", {"CloseFridge": (1, 1, 3)}, summary=False)
    evs = collect.evaluations(exp)
    assert [(e.step, e.partial, len(e.episodes)) for e in evs] == [(25000, False, 3)]
    # DeliverStraw 1006 is on the exclusion list: missing in the plan's sense, but it never counts
    assert evs[0].missing == [("OpenDrawer", 1003)] + [("DeliverStraw", 1000 + i) for i in range(7) if i != 6]
    evs = {e.step: e for e in collect.evaluations(exp, partial=True)}
    assert sorted(evs) == [25000, 50000, 75000]
    e = evs[75000]
    assert e.partial and e.successes == 1 and e.missing == [("OpenDrawer", 1002), ("OpenDrawer", 1003)]
    assert e.summary["per_task"] == {"OpenDrawer": {"n": 2, "successes": 1}}
    e = evs[50000]
    assert e.partial and len(e.episodes) == 3 and e.missing == [("CloseFridge", 1001), ("CloseFridge", 1002)]
    assert e.summary["shards"] == 2 and e.summary["per_task"]["CloseFridge"] == {"n": 1, "successes": 1}


def test_stage0_band_bounds_and_fluctuation(tmp_path):
    """RD 50k still lacks 20 OpenDrawer episodes: it counts with its 30; OpenDrawer's all-checkpoint average could
    end on either side of 10%; the adjacent-checkpoint and between-start differences match hand counts."""
    from downstream.report import page

    root = tmp_path / "runs"
    for s in ("B", "S", "RS", "RD"):
        exp = root / f"experiments/DS-{s}-100-s1"
        exp.mkdir(parents=True)
        (exp / "experiment.json").write_text("{}")
        for step in (25000, 50000):
            tasks = {"OpenDrawer": (4, 50, 50), "CloseFridge": (10, 20, 20)}
            if (s, step) == ("B", 50000):
                tasks["CloseFridge"] = (14, 20, 20)
            if (s, step) == ("RD", 50000):
                tasks["OpenDrawer"] = (6, 30, 50)
            _write_eval(exp / f"evals_{build.STAGE0_RUN}/steps_{step}/out/episodes", tasks,
                        summary=(s, step) != ("RD", 50000))
    coll = collect.collect(root, evals_name=f"evals_{build.STAGE0_RUN}", run=build.STAGE0_RUN, partial=True)
    assert len(coll.first()) == 8 and [e.partial for e in coll.first()].count(True) == 1
    s0 = build.stage0(coll)
    od = next(x for x in s0["spatial"] if x["task"] == "OpenDrawer")
    assert s0["latest_common"] == 50000 and od["mean"] == pytest.approx(0.11) and od["in_band"]
    assert od["mean_all"] == pytest.approx(0.095) and not od["in_band_all"]
    assert od["low_all"] == pytest.approx(0.085) and od["high_all"] == pytest.approx(0.135) and not od["band_certain"]
    assert od["per_step"] == pytest.approx({25000: 0.08, 50000: 0.11}) and od["steps_in_band"] == 1
    rd = next(r for r in s0["evaluations"] if (r["start"], r["step"]) == ("RD", 50000))
    assert rd["partial"] and (rd["episodes"], rd["planned"], len(rd["missing"])) == (50, 70, 20)
    assert rd["all"]["low"] == pytest.approx(16 / 70) and rd["all"]["high"] == pytest.approx(36 / 70)
    assert rd["spatial"]["missing"] == 20 and rd["spatial"]["rate"] == pytest.approx(0.2)
    fl = s0["fluctuation"]
    assert len(fl["within"]) == 4 and len(fl["between"]) == 6
    b = next(r for r in fl["within"] if r["start"] == "B")
    assert b["all"]["diff"] == pytest.approx(4 / 70) and b["all"]["task_mean_diff"] == pytest.approx(0.1)
    rd_b = next(r for r in fl["between"] if (r["start"], r["step"]) == ("RD", 50000))
    assert rd_b["all"]["n"] == 50 and rd_b["all"]["diff"] == pytest.approx(-2 / 50)
    # per task on the 30 OpenDrawer scenes both have (B: 4 of them succeed), not B's 4/50
    assert rd_b["all"]["task_mean_diff"] == pytest.approx(((6 - 4) / 30 + (10 - 14) / 20) / 2)
    assert fl["summary"]["within_all"]["pooled_abs_max"] == pytest.approx(4 / 70)
    # the last two checkpoints (s0["flat"]) are the last within pair, same numbers
    assert s0["flat"]["RD"]["all"]["task_task_mean_diff"] == pytest.approx(
        next(r for r in fl["within"] if r["start"] == "RD")["all"]["task_mean_diff"])
    assert s0["flat"]["RD"]["all"]["task_task_mean_diff"] == pytest.approx(((6 - 4) / 30 + 0) / 2)
    lines = "".join(build.stage0_lines(s0))
    assert "RD 50k（未完成） 50/70" in lines and "可能改变归类的：OpenDrawer" in lines
    html_text = page.stage0_html(s0)
    assert "缺回合的评测" in html_text and "检查点之间的波动" in html_text and "补完后 22.9%–51.4%" in html_text


def test_compare_mean_over_three_checkpoints_by_hand(tmp_path):
    """E1's main numbers (the user's decision 2 of 2026-10-23 evening): the mean over 150k, 175k and 200k.  B is flat;
    S gains 5 OpenDrawer successes per checkpoint; CloseFridge is equal throughout."""
    from downstream.report import page

    root = tmp_path / "runs"
    succ = {"B": (10, 10, 10), "S": (15, 20, 25)}
    for s, ks in succ.items():
        exp = root / f"experiments/DS-{s}-100-s1"
        exp.mkdir(parents=True)
        (exp / "experiment.json").write_text("{}")
        for step, k in zip(build.E1_POINTS, ks):
            _write_eval(exp / f"evals/steps_{step}/out/episodes", {"OpenDrawer": (k, 50, 50), "CloseFridge": (5, 20, 20)},
                        summary=True)
    coll = collect.collect(root)
    m = stats.compare_mean(coll, "100", build.E1_POINTS, [1])
    assert m["starts"] == ["B", "S"]
    assert m["groups"]["S"]["all"]["rates"] == pytest.approx([20 / 70, 25 / 70, 30 / 70])
    assert m["groups"]["S"]["all"]["mean"] == pytest.approx(25 / 70) and m["groups"]["B"]["spatial"]["mean"] == pytest.approx(0.2)
    assert m["tasks"]["S"]["OpenDrawer"] == pytest.approx(0.4)
    d = m["vs_reference"]["S"]["groups"]
    # per task: OpenDrawer (0.1 + 0.2 + 0.3) / 3 = 0.2, CloseFridge 0
    assert d["all"]["task_mean_diff"] == pytest.approx(0.1) and d["spatial"]["task_mean_diff"] == pytest.approx(0.2)
    assert d["all"]["pooled_mean_diff"] == pytest.approx(10 / 70)
    assert [x["step"] for x in d["all"]["pooled_by_step"]] == list(build.E1_POINTS)
    e1 = build.e1_points(coll)
    assert e1["starts"] == ["B", "S"] and len(e1["fluctuation"]["within"]) == 2 * 3  # 150->175, 175->200, 150->200
    s_span = next(r for r in e1["fluctuation"]["within"] if (r["start"], r["from"], r["to"]) == ("S", 150000, 200000))
    assert s_span["all"]["diff"] == pytest.approx(10 / 70)
    lines = "".join(build.e1_lines(e1))
    assert "第 150k、175k、200k 步三点平均" in lines and "S 35.7%（三点 28.6% / 35.7% / 42.9%）" in lines
    html_text = page.e1_html(e1)
    assert "三点平均" in html_text and "训练末尾的起伏" in html_text and "<b>35.7%</b>" in html_text
    # the E3 rule on the three-point means: OpenDrawer (0.2 + 0.4) / 2 = 30%, inside 10%-60%, on stage 0's list
    assert e1["band"] == [{"task": "OpenDrawer", "rates": {"B": pytest.approx(0.2), "S": pytest.approx(0.4)},
                           "mean": pytest.approx(0.3), "in_band": True, "listed": True}]
    assert ("E3 任务清单按同一规则复核" in lines and "OpenDrawer 30.0%；和阶段 0 的清单相同" in lines
            and "CoffeeSetupMug、SlideDishwasherRack、TurnOffStove 没有数据" in lines)
    assert "E3 任务清单复核" in html_text and "<b>30.0%</b>" in html_text


def test_e2_compares_each_smaller_ratio_with_100_percent_within_a_start(tmp_path):
    """The user's question of 2026-10-26: is 30% above 100% for every start or only for RS?  Same start, same final
    step, same scenes: S gains 10 OpenDrawer successes at 30%, CloseFridge is unchanged."""
    from downstream.report import page

    root = tmp_path / "runs"
    for ratio, drawer in (("100", 10), ("030", 20)):
        exp = root / f"experiments/DS-S-{ratio}-s1"
        exp.mkdir(parents=True)
        (exp / "experiment.json").write_text("{}")
        _write_eval(exp / f"evals/steps_{build.FINAL_STEP}/out/episodes",
                    {"OpenDrawer": (drawer, 50, 50), "CloseFridge": (5, 20, 20)}, summary=True)
    rv = build.ratio_vs_full(collect.collect(root))
    assert list(rv) == ["S"] and list(rv["S"]) == ["030"]
    d = rv["S"]["030"]
    assert d["all"]["rate_b"] == pytest.approx(15 / 70) and d["all"]["rate_a"] == pytest.approx(25 / 70)
    assert d["all"]["task_mean_diff"] == pytest.approx(0.1) and d["spatial"]["task_mean_diff"] == pytest.approx(0.2)
    lines = "".join(build.e2_lines(rv))
    assert "E2 30% 数据减 100% 数据" in lines and "S +10.0" in lines and "21.4% → 35.7%" in lines
    assert "30% − 100%" in page.e2_html(rv) and "0.36 个 epoch" in page.e2_html(rv)


def test_the_e3_rule_reports_tasks_that_enter_or_leave_the_band(tmp_path):
    root = tmp_path / "runs"
    for s, (drawer, bowls) in {"B": (60, 6), "S": (70, 8)}.items():  # OpenDrawer 65% leaves, StackBowlsCabinet 14% enters
        exp = root / f"experiments/DS-{s}-100-s1"
        exp.mkdir(parents=True)
        (exp / "experiment.json").write_text("{}")
        for step in build.E1_POINTS:
            _write_eval(exp / f"evals/steps_{step}/out/episodes",
                        {"OpenDrawer": (drawer, 100, 100), "StackBowlsCabinet": (bowls, 50, 50)}, summary=True)
    e1 = build.e1_points(collect.collect(root))
    assert [(r["task"], r["in_band"], r["listed"]) for r in e1["band"]] == [
        ("OpenDrawer", False, True), ("StackBowlsCabinet", True, False)]
    assert "和阶段 0 的清单不同：新进区间 StackBowlsCabinet；移出区间 OpenDrawer" in "".join(build.e1_lines(e1))


def test_the_main_report_leads_with_e1s_three_point_mean(sim, tmp_path):
    out = build.build(sim, tmp_path / "report")
    results = json.loads((tmp_path / "report/results.json").read_text(encoding="utf-8"))
    e1 = results["e1"]
    assert e1["points"] == list(build.E1_POINTS) and e1["starts"] == ["B", "S", "RS", "RD"]
    assert set(e1["mean"]["vs_reference"]) == {"S", "RS", "RD"}
    assert len(e1["fluctuation"]["within"]) == 4 * 3 and len(e1["fluctuation"]["between"]) == 3 * 3
    assert results["summary_lines"][0].startswith("E1（100% 数据、训练种子 1）第 150k、175k、200k 步三点平均")
    # the final step is reported once, as the final one, not again as an intermediate "D5" checkpoint
    assert not any(x.startswith("D5 第 200k 步") or x.startswith("这是中间检查点") for x in results["summary_lines"])
    assert results["d5_step"] is None
    page_text = (tmp_path / "report/index.html").read_text(encoding="utf-8")
    assert 'id="e1"' in page_text and "训练末尾的起伏" in page_text and out["html_bytes"] < 16 * 1024 * 1024
