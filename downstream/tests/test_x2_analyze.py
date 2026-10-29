"""downstream/x/x2_analyze.py: rates by group, task-level pairing with B on common episodes, the exclusion list,
and which checkpoint stands for each ratio (X2's only one; E1 / E2 at 200,000 steps)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "x"))
import x2_analyze as X  # noqa: E402

GROUP_OF = {"OpenDrawer": "atomic_seen", "CloseFridge": "atomic_seen", "MakeTea": "composite_seen"}
SPATIAL = {"OpenDrawer"}


def write(experiments: Path, name: str, step: int, outcomes):
    d = experiments / name / f"evals/steps_{step}/out/episodes"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "episodes.jsonl", "w") as f:
        for task, seed, ok in outcomes:
            f.write(json.dumps({"task": task, "seed": seed, "success": ok, "status": "complete"}) + "\n")


def outcomes(spec):
    """spec: {task: [success flags by seed 1000..]}"""
    return [(t, 1000 + i, ok) for t, flags in spec.items() for i, ok in enumerate(flags)]


def test_rates_pairing_and_checkpoints(tmp_path):
    exp = tmp_path / "experiments"
    write(exp, "X2-B-001-s1", 10368, outcomes({"OpenDrawer": [1, 0, 0, 0], "CloseFridge": [1, 1, 0, 0],
                                               "MakeTea": [0, 0, 0, 0]}))
    write(exp, "X2-RS-001-s1", 10368, outcomes({"OpenDrawer": [1, 1, 1, 0], "CloseFridge": [1, 1, 1, 1],
                                                "MakeTea": [0, 1, 0, 0]}))
    write(exp, "DS-B-030-s1", 175000, outcomes({"OpenDrawer": [0, 0, 0, 0]}))  # not the final: ignored
    write(exp, "DS-B-030-s1", 200000, outcomes({"OpenDrawer": [1, 1, 0, 0], "CloseFridge": [1, 0, 0, 0]}))
    excluded = {("CloseFridge", 1003)}
    res = X.analyze(exp, GROUP_OF, SPATIAL, excluded)
    b, rs = res["rates"]["001"]["B"], res["rates"]["001"]["RS"]
    assert (b["all"]["k"], b["all"]["n"]) == (3, 11)  # 12 episodes minus the excluded one
    assert (rs["all"]["k"], rs["all"]["n"]) == (7, 11) and (rs["spatial"]["k"], rs["spatial"]["n"]) == (3, 4)
    assert (rs["composite_seen"]["k"], rs["atomic_seen"]["k"]) == (1, 6)
    assert res["rates"]["030"]["B"]["spatial"]["k"] == 2  # the 200,000-step evaluation, not 175,000
    v = res["vs_B"]["001"]["RS"]
    assert v["all"]["tasks"] == 3 and v["spatial"]["tasks"] == 1
    # per-task RS - B: OpenDrawer +0.5, CloseFridge (3 common seeds) +1/3, MakeTea +0.25
    assert abs(v["all"]["task_mean_diff"] - (0.5 + 1 / 3 + 0.25) / 3) < 1e-3
    md = X.markdown(res)
    assert "## 1%：成功率" in md and "| 1% | RS |" in md and "| 30% |" not in md  # no RS at 30%: no pairing row
