from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_batch_shape_and_distinct_scene_selection(tmp_path: Path) -> None:
    benchmark = load_module(
        "active_spatial_renderer_microbenchmark",
        ROOT / "scripts" / "active_spatial_renderer_microbenchmark.py",
    )
    manifest = tmp_path / "train.jsonl"
    rows = [
        {
            "scene_id": f"scene-{index}",
            "init_camera": {
                "extrinsics": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        }
        for index in range(12)
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows + rows[:2]) + "\n", encoding="utf-8")
    selected = benchmark.load_fixed_scenes(manifest)
    assert [row["scene_id"] for row in selected] == [f"scene-{index}" for index in range(12)]
    assert benchmark.EXPECTED_REQUESTS == 12 * 4 * 8 == 384


def test_comparison_selects_smallest_passing_topology(tmp_path: Path, monkeypatch) -> None:
    compare = load_module(
        "compare_active_spatial_renderer_microbenchmarks",
        ROOT / "scripts" / "compare_active_spatial_renderer_microbenchmarks.py",
    )
    results = tmp_path / "results"
    results.mkdir()
    for gpu_count, status in ((1, "BLOCKED"), (2, "PASS"), (4, "PASS")):
        (results / f"{gpu_count}gpu_12workers.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "measured": {"requests": 384, "ordered_png_digest_sha256": "same"},
                    "safety": {"optimizer_step_called": False, "checkpoint_written": False},
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["compare", "--run", str(tmp_path), "--output", str(output)])
    assert compare.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_topology"] == {"gpu_count": 2, "max_workers": 12, "max_inflight": 12}
    assert report["training_submission_allowed"] is True


def test_trainer_observability_is_fail_closed() -> None:
    source = (ROOT / "examples" / "train" / "active_spatial" / "sco_dense_score_reward_only_parallel.sh").read_text(
        encoding="utf-8"
    )
    for required in (
        "PYTHONUNBUFFERED=1",
        "DENSE_SCORE_RAY_TEMP_DIR",
        "ray_logs_latest.tar.gz",
        "gpu_resources.csv",
        "host_resources.csv",
        "process_snapshots.log",
        "train_status=${PIPESTATUS[0]}",
    ):
        assert required in source
