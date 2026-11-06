"""g_report.py (tier g, design G_DESIGN_20261106 section 5; the user's decisions of 2026-11-06): with history against
the recorded runs without, per model and per pairing, the flagged instances apart, error rows out of both sides, the
(g-c) condition."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import g_report as G  # noqa: E402


def write(d: Path, model: str, rows: list):
    (d / model).mkdir(parents=True, exist_ok=True)
    (d / model / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def ep(i, ok, scene=None, err=False, turns=()):
    r = {"instance_id": f"I{i}", "scene_id": scene or f"K{i // 2}", "task_type": "equidistance", "success": ok,
         "turn_log": [{"turn": t, "prompt_tokens": n} for t, n in turns]}
    if err:
        r["error"] = "server error at turn 3"
    return r


def test_levels_pairs_flagged_errors_and_the_gc_condition(tmp_path):
    g, d = tmp_path / "g", tmp_path / "d"
    n = 20
    # without history: B never, S succeeds on 0-3, RS on 0-1.  With history: B never, S on 0-3, RS on 0-5
    write(d, "B", [ep(i, False) for i in range(n)])
    write(d, "S", [ep(i, i < 4) for i in range(n)])
    write(d, "RS", [ep(i, i < 2) for i in range(n)])
    write(g, "B", [ep(i, False, turns=[(1, 700), (12, 3200)]) for i in range(n)])
    write(g, "S", [ep(i, i < 4) for i in range(n)])
    write(g, "RS", [ep(i, i < 6) for i in range(n - 1)] + [ep(n - 1, False, err=True)])  # I19: an error row
    rep = G.build([g], [d], flagged={"I0", "I1"}, models=["B", "S", "RS"])
    assert rep["left_out_for_errors"] == ["I19"] and rep["errors"]["g"]["RS"] == ["I19"]
    s = rep["subsets"]["all"]
    assert s["n"] == 19
    rs = s["levels"]["RS"]
    assert rs["only_a"] == 4 and rs["only_b"] == 0 and math.isclose(rs["scene"]["mean"], 4 / 19)
    p = s["pairs"]["RS-S"]
    assert math.isclose(p["with"]["mean"], 2 / 19) and math.isclose(p["without"]["mean"], -2 / 19)
    assert math.isclose(p["did"]["mean"], 4 / 19) and p["did"]["low"] >= 0  # the gain sits in 2 of 10 scenes
    assert set(s["pairs"]) == {"S-B", "RS-B", "RS-S"}
    assert s["rules"]["RS-S comes back"] is (p["with"]["low"] > 0)
    assert s["gc_condition"]["parts"]["RS-S with history >= +3"] is True and s["gc_condition"]["met"] is True
    assert s["gc_condition"]["parts"]["S-B moved more than 3"] is False
    w = rep["subsets"]["without flagged"]
    assert w["n"] == 17
    assert rep["flagged"] == ["I0", "I1"] and rep["flagged_episodes"]["RS"]["I0"] == {"g": True, "d": True}
    assert rep["prompt_tokens"]["B"]["12"] == {"calls": 20, "mean": 3200, "max": 3200}  # every row of B (the error is RS's)
    md = G.markdown(rep)
    assert "去掉可疑场景的实例（17 个）" in md and "| RS-S | 19 |" in md and "(g-c) 的条件" in md


def test_scene_bootstrap_keeps_a_scene_together_and_gives_a_two_sided_p():
    per = {"a1": 1, "a2": 1, "b1": 0, "b2": 0}
    c = G.scene_boot(per, {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}, draws=400, seed=1)
    assert c["mean"] == 0.5 and c["low"] in (0.0, 0.5) and c["high"] == 1.0
    assert 0 < c["p"] <= 1
    z = G.scene_boot({"x": 0, "y": 0}, {}, draws=50)
    assert z["mean"] == 0 and z["p"] == 1.0 and not z["p_floor"]
    allup = G.scene_boot({f"i{k}": 1 for k in range(6)}, {}, draws=200)  # never at or below 0: p under the resolution
    assert allup["p_floor"] and allup["p"] == 1 / 200
    assert G.position({"low": 0.01, "high": 0.2}) == "above" and G.position({"low": -0.1, "high": 0.2}) == "across"


def test_cli_writes_the_report(tmp_path):
    g, d = tmp_path / "g", tmp_path / "d"
    for x in (g, d):
        write(x, "B", [ep(i, False) for i in range(4)])
        write(x, "S", [ep(i, i < 2) for i in range(4)])
    flag = tmp_path / "flag.jsonl"
    flag.write_text(json.dumps({"instance_id": "I3"}) + "\n")
    out = tmp_path / "out"
    assert G.main(["--g", str(g), "--d", str(d), "--flag", str(flag), "--models", "B", "S", "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["subsets"]["without flagged"]["n"] == 3 and (out / "report.md").exists()
