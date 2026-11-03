"""Paper figures (downstream/report/n_figures.py; the user's decision 4 of 2026-11-03): the Wilson interval, and the
three figures and figures.json from stand-in result files."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))
import n_figures as F  # noqa: E402


def test_wilson_matches_the_reports():
    assert F.wilson(0, 100)[2] == pytest.approx(0.03699480747600191)  # N1 merged, B on left / right
    p, lo, hi = F.wilson(240, 300)
    assert p == 0.8 and lo == pytest.approx(0.7510701139033206) and hi == pytest.approx(0.8413438280980382)  # G0, B


def fake_runs(root: Path) -> Path:
    models = [m for _, ms in F.GENERATIONS for m in ms]
    (root / "n/n1_merged_20261102").mkdir(parents=True)
    comb = {"models": {m: {"rates": {t: {"k": 3 + i + j, "n": 100} for j, t in enumerate(F.FAMILIES)}}
                       for i, m in enumerate(models)}}
    (root / "n/n1_merged_20261102/report.json").write_text(json.dumps({"combined": comb}))
    qa = root / "x/QA_20261030/results"
    qa.mkdir(parents=True)
    for _, fname, _ in F.QA_SETS:
        n = 300 if fname == "g0.json" else 510
        qa.joinpath(fname).write_text(json.dumps({m: {"all": {"correct": n // 2 + 10 * i, "n": n}}
                                                  for i, m in enumerate(models)}))
    (root / "v4_recheck").mkdir()
    (root / "v4_recheck/report.json").write_text(json.dumps({"points": [150000, 175000, 200000], "counts": {
        k: [[300 - 10 * i + d, 943] for d in (0, 5, 2)] for i, (k, *_rest) in enumerate(F.V4_LINES)}}))
    return root


def test_three_figures_and_their_numbers(tmp_path):
    pytest.importorskip("matplotlib")
    runs = fake_runs(tmp_path / "runs")
    assert F.main(["--out", str(tmp_path / "fig"), "--runs", str(runs)]) == 0
    names = sorted(p.name for p in (tmp_path / "fig").iterdir())
    assert names == ["figures.json", "n1_by_type.png", "n1_by_type.svg", "qa_heatmap.png", "qa_heatmap.svg",
                     "v4_curve.png", "v4_curve.svg"]
    rec = json.loads((tmp_path / "fig/figures.json").read_text())
    assert rec["n1_by_type"]["B"]["all"]["k"] == sum(3 + j for j in range(6)) and rec["n1_by_type"]["B"]["all"]["n"] == 600
    assert rec["qa_heatmap"]["G0"]["Q3RD"] == {"k": 220, "n": 300, "rate": 220 / 300,
                                               "wilson": list(F.wilson(220, 300)[1:])}
    assert [c["k"] for c in rec["v4_curve"]["counts"]["Q3DCKS2"]] == [270, 275, 272]
