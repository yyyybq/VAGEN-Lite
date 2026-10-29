"""downstream/x/e3_analyze.py: before / after RL on common episodes, McNemar counts, the start's change against B's
task by task, and the rollouts' success parsed from the RLinf log of the run that reached the most updates."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "x"))
import e3_analyze as E  # noqa: E402


def write(path: Path, outcomes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for task, seed, ok in outcomes:
            f.write(json.dumps({"task": task, "seed": seed, "success": ok, "status": "complete"}) + "\n")


def test_before_after_and_against_b(tmp_path):
    exp = tmp_path / "experiments"
    # B: CoffeeSetupMug 2/4 -> 1/4, OpenDrawer 1/4 -> 1/4
    write(E.before_file(exp, "B"), [("CoffeeSetupMug", s, s < 1002) for s in range(1000, 1004)]
          + [("OpenDrawer", s, s == 1000) for s in range(1000, 1004)])
    write(E.after_file(exp, "B"), [("CoffeeSetupMug", s, s == 1003) for s in range(1000, 1004)]
          + [("OpenDrawer", s, s == 1000) for s in range(1000, 1004)])
    # RS: CoffeeSetupMug 1/4 -> 3/4, OpenDrawer 2/4 -> 2/4, one excluded episode left out
    write(E.before_file(exp, "RS"), [("CoffeeSetupMug", s, s == 1000) for s in range(1000, 1004)]
          + [("OpenDrawer", s, s < 1002) for s in range(1000, 1004)] + [("PreSoakPan", 1016, True)])
    write(E.after_file(exp, "RS"), [("CoffeeSetupMug", s, s != 1003) for s in range(1000, 1004)]
          + [("OpenDrawer", s, s < 1002) for s in range(1000, 1004)] + [("PreSoakPan", 1016, False)])
    run = exp / "E3-RS"
    (run / "rlinf_train_old.log").write_text("Global Step: 1/100 | x\nsuccess_once=0.5\n")  # a shorter (hung) run
    (run / "rlinf_train_new.log").write_text(
        "".join(f"Global Step: {i}/100 |\r success_once={0.01 * (i > 90)}\n" for i in range(1, 101)))
    res = E.analyze(exp, spatial={"OpenDrawer"}, excluded={("PreSoakPan", 1016)})
    rs = res["starts"]["RS"]
    assert (rs["all"]["n"], rs["all"]["before"], rs["all"]["after"]) == (8, 3, 5)
    assert (rs["rl_tasks"]["gained"], rs["rl_tasks"]["lost"]) == (2, 0) and rs["spatial"]["diff"] == 0.0
    assert res["starts"]["B"]["rl_tasks"]["diff"] == -0.25
    # per task, RS change minus B change: CoffeeSetupMug +0.5 - (-0.25) = +0.75, OpenDrawer 0 - 0 = 0
    v = res["vs_B"]["RS"]
    assert v["all"]["tasks"] == 2 and abs(v["all"]["task_mean_diff"] - 0.375) < 1e-9
    assert v["rl_tasks"]["tasks"] == 1 and abs(v["rl_tasks"]["task_mean_diff"] - 0.75) < 1e-9
    tr = res["training"]["RS"]
    assert tr["updates"] == 100 and tr["success_once"] == {"1-10": 0.0, "41-60": 0.0, "91-100": 0.01}
    md = E.markdown(res)
    assert "## RL 的 3 个任务" in md and "| RS | 100 |" in md
