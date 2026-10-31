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


def test_main_writes_the_report(tmp_path):
    gen, ev = tmp_path / "gen", tmp_path / "eval"
    write(gen / "instances_eval.jsonl", [{"instance_id": "a_fov_inclusion", "task_type": "fov_inclusion",
                                         "near_threshold": []}])
    write(ev / "B/episodes.jsonl", [ep("a_fov_inclusion", True)])
    assert R.main(["--gen", str(gen), "--eval", str(ev), "--models", "B"]) == 0
    assert json.loads((ev / "report.json").read_text())["models"]["B"]["rates"]["all"]["k"] == 1
    assert (ev / "report.md").read_text().startswith("# N1 结果")
