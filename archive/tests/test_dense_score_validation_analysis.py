from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_active_spatial_dense_score_validation.py"
SPEC = importlib.util.spec_from_file_location("dense_score_validation_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(task: str, scene: str, success: float, final: float, *, bad: float = 0.0) -> dict:
    row = {
        "task_id": task,
        "scene_id": scene,
        "traj_success": success,
        "final_score": final,
        "score_improvement": final - 0.01,
        "reward": final + 1.0,
        "episode_length": 4.0,
        "n_primitive_steps": 5.0,
        "collision_termination": 0.0,
        "invalid_action": bad,
        "strict_parse_success_rate": 1.0 - bad,
        "renderer_failure": 0.0,
        "empty_action": bad,
        "contradictory_action": 0.0,
        "missing_action_tag": bad,
        "unknown_action_name": 0.0,
        "truncated_before_action": 0.0,
        "multiple_action_tag": 0.0,
        "strict_format_correct_rate": 1.0 - bad,
        "format_penalty_rate": bad,
        "low_info_termination": 0.0,
        "trajectory_hash": f"{task}:{success}:{final}:{bad}",
    }
    return row


def _make_run(root: Path) -> Path:
    run = root / "run"
    tasks = (("t0", "scene-a"), ("t1", "scene-b"))
    for branch in MODULE.BRANCHES:
        validation = run / "branches" / branch / "validation"
        validation.mkdir(parents=True)
        for step in MODULE.STEPS:
            learned = max(0, step - 60) / 90
            bonus = {"S0": 0.0, "S1": 0.1 * learned, "S5": 0.2 * learned}[branch]
            rows = []
            for task, scene in tasks:
                for rollout in range(4):
                    final = 0.1 + 0.2 * learned + bonus + rollout * 0.01
                    success = float(final >= 0.3)
                    bad = float(branch == "S1" and step == 150 and rollout == 0)
                    rows.append(_row(task, scene, success, final, bad=bad))
            with (validation / f"{step}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
    return run


def _args(run: Path, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_dir=run,
        output_dir=output,
        steps=list(MODULE.STEPS),
        rollouts_per_task=4,
        critic_warmup=60,
        bootstrap_samples=200,
        seed=17,
    )


def test_analysis_is_deterministic_and_warmup_aware(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    first = MODULE.analyze(_args(run, tmp_path / "out-a"))
    second = MODULE.analyze(_args(run, tmp_path / "out-b"))

    assert first["status"] == "PASS"
    assert first["formal_ood_status"] == "NOT_RUN"
    assert first["checkpoint_to_actor_updates"] == {"0": 0, "50": 0, "100": 40, "150": 90}
    assert first["contrasts"] == second["contrasts"]
    assert first["contrasts"]["S1"]["actor_update_auc"]["final_score"]["estimate"] > 0
    assert first["validation"]["aggregate"]["S1"]["150"]["invalid_action"] == 0.25
    for name in ("validation_analysis.json", "validation_curves.csv", "training_curves.csv", "validation_learning_curves.svg", "REPORT.md"):
        assert (tmp_path / "out-a" / name).is_file()


def test_task_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    path = run / "branches" / "S1" / "validation" / "100.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows[:4]:
        row["task_id"] = "unexpected-task"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="task set differs"):
        MODULE.analyze(_args(run, tmp_path / "out"))


def test_missing_metric_fails_closed(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    path = run / "branches" / "S5" / "validation" / "50.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del rows[0]["final_score"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        MODULE.analyze(_args(run, tmp_path / "out"))
