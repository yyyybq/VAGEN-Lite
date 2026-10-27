"""X5 evaluation: the task file from the list files, the plan (tasks from that file), the analysis, the run script."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x"))
sys.path.insert(0, str(ROOT / "eval"))
import closed_loop  # noqa: E402
import x5_analyze  # noqa: E402
import x5_tasks  # noqa: E402


def lists(tmp_path):
    a = {"kind": "atomic", "list": ["OpenOven", "CloseDrawer"],
         "tried": [{"task": "OpenOven", "ok": True, "horizon": 300}, {"task": "Broken", "ok": False, "error": "x"},
                   {"task": "CloseDrawer", "ok": True, "horizon": 450}]}
    c = {"kind": "composite", "list": ["SweetenCoffee"], "tried": [{"task": "SweetenCoffee", "ok": True, "horizon": 1950}]}
    pa, pc = tmp_path / "atomic.json", tmp_path / "composite.json"
    pa.write_text(json.dumps(a))
    pc.write_text(json.dumps(c))
    return pa, pc


def test_task_file_from_the_list_files_and_a_plan_that_reads_it(tmp_path):
    pa, pc = lists(tmp_path)
    out = tmp_path / "x5_tasks.json"
    assert x5_tasks.main(["--rows-from", str(pa), str(pc), "--out", str(out)]) == 0
    rows = json.loads(out.read_text())["tasks"]
    assert rows == [{"task": "OpenOven", "group": "unseen_atomic", "horizon": 300},
                    {"task": "CloseDrawer", "group": "unseen_atomic", "horizon": 450},
                    {"task": "SweetenCoffee", "group": "unseen_composite", "horizon": 1950}]
    plan = json.loads((ROOT / "x/x5_plan.json").read_text())
    plan["tasks_file"] = str(out)  # absolute: replaces the worktree path
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    eps, p = closed_loop.eval_plan_episodes(tmp_path / "plan.json")
    assert len(eps) == 3 * 20 and {e["max_steps"] for e in eps if e["task"] == "SweetenCoffee"} == {1950}
    assert p["split"] == "target" and p["execute_steps"] == 16 and p["stable_counter_regions"]
    assert closed_loop.apply_exclusions(eps, closed_loop.excluded_episodes()["keys"]) == eps
    with pytest.raises(SystemExit):
        x5_tasks.main(["--out", str(out)])  # neither --kind nor --rows-from


def test_the_real_x5_plan_has_32_tasks_and_640_episodes():
    if not (ROOT / "x/x5_tasks.json").is_file():
        pytest.skip("x5_tasks.json is generated on the cluster")
    eps, _ = closed_loop.eval_plan_episodes(ROOT / "x/x5_plan.json")
    tasks = json.loads((ROOT / "x/x5_tasks.json").read_text())["tasks"]
    assert len(tasks) == 32 and sum(t["group"] == "unseen_atomic" for t in tasks) == 16 and len(eps) == 640
    target = {t["task"] for t in json.loads((ROOT / "demos/target50_tasks.json").read_text())["tasks"]}
    assert not target & {t["task"] for t in tasks}  # none of them in our training data


def episodes(path: Path, outcomes: dict):
    path.mkdir(parents=True)
    rows = [{"task": t, "seed": 1000 + i, "status": "complete", "success": ok}
            for t, oks in outcomes.items() for i, ok in enumerate(oks)]
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_analysis_compares_with_b_per_task_and_skips_a_group_nobody_can_do(tmp_path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"tasks": [{"task": "A1", "group": "unseen_atomic", "horizon": 300},
                                           {"task": "A2", "group": "unseen_atomic", "horizon": 300},
                                           {"task": "C1", "group": "unseen_composite", "horizon": 900}]}))
    x5 = tmp_path / "X5"
    episodes(x5 / "B/out/episodes", {"A1": [True, False, False, False], "A2": [False] * 4, "C1": [False] * 4})
    episodes(x5 / "RS/out/episodes", {"A1": [True, True, True, False], "A2": [True, False, False, False],
                                      "C1": [False] * 4})
    res = x5_analyze.analyze(x5, tasks)
    assert res["starts"]["B"]["unseen_atomic"]["k"] == 1 and res["starts"]["RS"]["unseen_atomic"]["k"] == 4
    assert res["starts"]["RS"]["unseen_atomic"]["per_task"]["A1"] == {"k": 3, "n": 4}
    v = res["vs_B"]["RS"]["unseen_atomic"]
    assert v["tasks"] == 2 and v["tasks_up"] == 2 and v["task_mean_diff"] == pytest.approx((0.5 + 0.25) / 2)
    assert res["nobody"] == {"unseen_atomic": False, "unseen_composite": True}
    assert "unseen_composite" not in res["vs_B"]["RS"]
    md = (x5 / "results/results.md").read_text(encoding="utf-8")
    assert "未见任务上都不会" in md and "| RS | 2 |" in md


def test_run_script_evaluates_the_e1_final_checkpoint_with_the_x5_plan(tmp_path):
    runs = tmp_path / "experiments"
    ckpt = runs / "DS-S-100-s1/run/final_model/pytorch_model.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("w")
    fake = tmp_path / "eval.sh"
    fake.write_text('echo "$EVAL_PLAN|$1|$2|$3|$4" > "$(dirname "$0")/called"\n')
    env = dict(os.environ, X5_RUNS=str(runs), X5_EVAL=str(fake), X5_PY=sys.executable)
    res = subprocess.run(["bash", str(ROOT / "x/run_x5.sh"), "S", str(tmp_path / "X5")], capture_output=True,
                         text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    plan, ck, out, servers, workers = (tmp_path / "called").read_text().strip().split("|")
    assert plan.endswith("downstream/x/x5_plan.json") and ck == str(ckpt) and out == str(tmp_path / "X5/S/out")
    assert (servers, workers) == ("8", "96")
