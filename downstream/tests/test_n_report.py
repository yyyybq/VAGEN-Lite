"""N1's report (downstream/n/n_report.py; the user's decision 3 of 2026-10-31): rates with Wilson intervals, the
pairing against the same generation's base, the near-threshold split, behaviour, missing and failed episodes."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import n_report as R  # noqa: E402


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def ep(i, success, **kw):
    return dict({"instance_id": i, "success": success, "turns": 4, "steps": 8, "invalid_turns": 1, "blocked": 2,
                 "max_shaping_score": 0.4, "end": "success" if success else "turns"}, **kw)


def test_report_pairs_each_model_with_its_generations_base(tmp_path):
    gen, ev = tmp_path / "gen", tmp_path / "eval"
    ids = [f"S{k}_{t}" for t in ("projective_relations", "equidistance") for k in range(4)]
    write(gen / "instances_eval.jsonl", [{"instance_id": i, "task_type": i.split("_", 1)[1],
                                         "near_threshold": ["visible"] if i.startswith("S0") else []} for i in ids])
    write(ev / "B/episodes.jsonl", [ep(i, i.startswith("S1")) for i in ids])
    write(ev / "RS/episodes.jsonl", [ep(i, i.startswith(("S1", "S2"))) for i in ids[:-1]]
          + [{"instance_id": ids[-1], "error": "RuntimeError: boom"}])
    write(ev / "Q3B/episodes.jsonl", [ep(i, False) for i in ids])
    write(ev / "Q3RS/episodes.jsonl", [ep(i, True) for i in ids[:6]])
    rep = R.build(gen, ev, ["B", "RS", "Q3B", "Q3RS"])
    assert rep["instances"] == 8 and rep["per_type"] == {"projective_relations": 4, "equidistance": 4}
    b, rs, q3rs = rep["models"]["B"], rep["models"]["RS"], rep["models"]["Q3RS"]
    assert b["rates"]["all"]["k"] == 2 and b["rates"]["all"]["n"] == 8 and "versus" not in b
    assert rs["errors"] == [ids[-1]] and rs["rates"]["all"]["n"] == 7 and rs["rates"]["all"]["k"] == 4
    v = rs["versus"]
    assert v["base"] == "B" and v["all"]["n"] == 7 and v["all"]["only_a"] == 2 and v["all"]["only_b"] == 0
    assert v["all"]["diff"] == pytest.approx(2 / 7)
    assert set(v["per_type"]) == {"projective_relations", "equidistance"} and "p_holm" in v["per_type"]["equidistance"]
    assert v["near"]["n"] == 2 and v["clear"]["n"] == 5  # S0 instances are near the threshold
    assert q3rs["versus"]["base"] == "Q3B" and q3rs["missing"] == ids[6:] and q3rs["versus"]["all"]["only_a"] == 6
    beh = b["behaviour"]
    assert beh["unreadable_per_turn"] == pytest.approx(0.25) and beh["blocked_per_action"] == pytest.approx(0.25)
    assert beh["ends"] == {"turns": 6, "success": 2} and beh["max_shaping_mean"] == pytest.approx(0.4)
    md = R.markdown(rep)
    assert "| RS | B |" in md and "| Q3RS | Q3B |" in md and "左右关系" in md and "全部" in md
    assert math.isnan(R.build(gen, tmp_path / "none", ["B"])["models"]["B"]["rates"]["all"]["rate"])


def test_rl_starts_pair_with_the_sft_start_they_grew_from(tmp_path):
    gen, ev = tmp_path / "gen", tmp_path / "eval"
    ids = [f"S{k}_fov_inclusion" for k in range(6)]
    write(gen / "instances_eval.jsonl", [{"instance_id": i, "task_type": "fov_inclusion", "near_threshold": []}
                                         for i in ids])
    write(ev / "S/episodes.jsonl", [ep(i, k < 4) for k, i in enumerate(ids)])
    write(ev / "RS/episodes.jsonl", [ep(i, k < 1) for k, i in enumerate(ids)])
    write(ev / "Q3RD/episodes.jsonl", [ep(i, True) for i in ids])
    rep = R.build(gen, ev, ["B", "S", "RS", "Q3RD"])
    rs = rep["models"]["RS"]
    assert rs["versus_sft"]["base"] == "S" and rs["versus_sft"]["all"]["only_b"] == 3
    assert rs["versus_sft"]["all"]["diff"] == pytest.approx(-3 / 6)
    assert "versus" not in rs  # B has no episodes
    assert "versus_sft" not in rep["models"]["S"] and "versus_sft" not in rep["models"]["Q3RD"]  # no Q3S
    assert R.sft_of("RD") == "S" and R.sft_of("Q3RS") == "Q3S" and R.sft_of("B") is None
    assert "RL 和 SFT 配对" in R.markdown(rep) and "| RS | S |" in R.markdown(rep)


def two_sets(tmp_path, second_ids=None):
    sets = []
    for name, rs_wins in (("one", 3), ("two", 0)):
        gen, ev = tmp_path / f"gen_{name}", tmp_path / f"eval_{name}"
        ids = second_ids if (second_ids and name == "two") else [f"{name}{k}_equidistance" for k in range(10)]
        write(gen / "instances_eval.jsonl", [{"instance_id": i, "task_type": "equidistance",
                                             "near_threshold": ["margin"] if k < 2 else []} for k, i in enumerate(ids)])
        write(ev / "S/episodes.jsonl", [ep(i, k < 2) for k, i in enumerate(ids)])
        write(ev / "RS/episodes.jsonl", [ep(i, k < 2 + rs_wins) for k, i in enumerate(ids)])
        sets.append((name, gen, ev))
    return sets


def test_sets_are_reported_apart_and_merged(tmp_path):
    sets = two_sets(tmp_path)
    rep = R.build_sets(sets, ["S", "RS"])
    assert set(rep["sets"]) == {"one", "two"} and rep["combined"]["instances"] == 20
    assert rep["combined"]["near_threshold"] == 4
    comb = rep["combined"]["models"]["RS"]
    assert comb["rates"]["all"]["k"] == 7 and comb["versus_sft"]["all"]["only_a"] == 3
    whole = comb["versus_sft"]["all"]
    assert whole["scene_low"] <= whole["diff"] <= whole["scene_high"]
    assert rep["sets"]["one"]["models"]["RS"]["versus_sft"]["all"]["diff"] == pytest.approx(0.3)
    assert rep["sets"]["two"]["models"]["RS"]["versus_sft"]["all"]["diff"] == 0
    a = rep["agreement"]["RS"]["versus_sft"]
    assert a["gap"] == pytest.approx(0.3) and 0 < a["p"] < 1
    assert rep["agreement"]["S"] == {}
    md = R.markdown_sets(rep)
    assert md.startswith("# N 系列合并结果（20 个实例：one 10 个、two 10 个）")
    assert "两套之差" in md and "## 合并的 20 个实例" in md and "### 成功率" in md
    assert R.agree({"n": 5, "only_a": 0, "only_b": 0, "diff": 0.0},
                   {"n": 5, "only_a": 0, "only_b": 0, "diff": 0.0})["p"] == 1.0
    assert math.isnan(R.agree({"n": 0}, {"n": 5})["gap"])


def test_scene_interval_resamples_whole_scenes():
    insts = {f"{s}_{k}": {"scene_id": s} for s in ("a", "b", "c") for k in range(2)}
    ref = {i: False for i in insts}
    assert R.scene_interval({i: True for i in insts}, ref, insts, list(insts)) == (1.0, 1.0)
    succ = {i: i.startswith("a") for i in insts}  # one scene in three solved, both of its instances
    lo, hi = R.scene_interval(succ, ref, insts, list(insts), draws=500)
    assert lo == 0.0 and (hi == pytest.approx(2 / 3) or hi == 1.0)  # a draw without scene a is common, 2+ of it not rare
    assert R.scene_interval(succ, ref, insts, list(insts), draws=500) == (lo, hi)  # fixed seed
    assert all(math.isnan(x) for x in R.scene_interval({}, ref, insts, list(insts)))


def test_an_instance_in_two_sets_is_an_error(tmp_path):
    sets = two_sets(tmp_path, second_ids=[f"one{k}_equidistance" for k in range(10)])
    with pytest.raises(ValueError, match="10 instance ids in two sets"):
        R.build_sets(sets, ["S", "RS"])


def test_main_with_two_sets_needs_out_and_writes_the_merged_report(tmp_path):
    sets = two_sets(tmp_path)
    gens, evs = [str(s[1]) for s in sets], [str(s[2]) for s in sets]
    with pytest.raises(SystemExit):
        R.main(["--gen", *gens, "--eval", *evs, "--models", "S", "RS"])
    with pytest.raises(SystemExit):
        R.main(["--gen", *gens, "--eval", evs[0], "--models", "S", "RS", "--out", str(tmp_path / "o")])
    out = tmp_path / "merged"
    assert R.main(["--gen", *gens, "--eval", *evs, "--names", "N1", "N1b", "--models", "S", "RS",
                   "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert set(rep["sets"]) == {"N1", "N1b"} and rep["combined"]["models"]["RS"]["rates"]["all"]["n"] == 20
    assert (out / "report.md").read_text().startswith("# N 系列合并结果（20 个实例：N1 10 个、N1b 10 个）")
    assert not (sets[0][2] / "report.json").exists()


def test_main_writes_the_report(tmp_path):
    gen, ev = tmp_path / "gen", tmp_path / "eval"
    write(gen / "instances_eval.jsonl", [{"instance_id": "a_fov_inclusion", "task_type": "fov_inclusion",
                                         "near_threshold": []}])
    write(ev / "B/episodes.jsonl", [ep("a_fov_inclusion", True)])
    assert R.main(["--gen", str(gen), "--eval", str(ev), "--models", "B"]) == 0
    assert json.loads((ev / "report.json").read_text())["models"]["B"]["rates"]["all"]["k"] == 1
    assert (ev / "report.md").read_text().startswith("# N1 结果")
