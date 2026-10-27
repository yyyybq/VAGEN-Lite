"""X4 analysis: rates per model and set, the E1-vs-start pairing (McNemar, bootstrap interval, "kept")."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x"))
import x4_analyze  # noqa: E402


def write(d: Path, outcomes, types=None):
    d.mkdir(parents=True)
    rows = [{"task_id": f"t{i}", "task_type": (types or ["a"] * len(outcomes))[i], "success": ok}
            for i, ok in enumerate(outcomes)]
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "garbage\n")
    (d / "done.json").write_text(json.dumps({"episodes": len(rows)}))


def test_rates_pairing_and_kept(tmp_path):
    x4 = tmp_path / "X4"
    start = [True] * 30 + [False] * 10
    worse = [False] * 20 + [True] * 10 + [False] * 10  # loses 20 of the start's 30, gains none
    write(x4 / "eval_val_v2/orig-S", start, ["a"] * 20 + ["b"] * 20)
    write(x4 / "eval_val_v2/E1-S-200k", worse)
    write(x4 / "eval_val_v2/orig-B", start)
    write(x4 / "eval_val_v2/E1-B-200k", start[:39] + [True])  # gains one
    res = x4_analyze.analyze(x4)
    s = res["retention"]["S"]["val"]
    assert s["tasks"] == 40 and s["only_e1"] == 0 and s["only_start"] == 20 and s["diff"] == -0.5
    assert s["p"] < 0.001 and not s["kept"] and s["ci95"][1] < 0
    b = res["retention"]["B"]["val"]
    assert b["only_e1"] == 1 and b["kept"] and b["diff"] == 0.025
    m = res["models"]["orig-S"]["val"]
    assert m["k"] == 30 and m["n"] == 40 and m["finished"] and m["expected"] == 413
    assert m["by_type"]["a"]["k"] == 20 and m["by_type"]["b"]["k"] == 10
    assert "test" not in res["retention"]["S"]  # no test results: not paired
    md = (x4 / "results/results.md").read_text(encoding="utf-8")
    assert "| S | val | 40 | -50.0" in md and "| 否 |" in md and "| orig-RD | — | — |" in md
