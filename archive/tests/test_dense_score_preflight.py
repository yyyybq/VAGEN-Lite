import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "active_spatial_dense_score_preflight",
    ROOT / "scripts/active_spatial_dense_score_preflight.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_manifest_gate_matches_explicit_frozen_contract(tmp_path):
    row = {
        "task_id": "contract-only-fixture",
        "task_type": "projective_relations",
        "canonical_task_metric_version": "canonical_spatial_task_h1_v1",
        "camera_model_version": "canonical_h1_from_frozen_candidate_intrinsics_and_pose",
        "collision_convention": {
            "version": "interiorgs_structure_label_alignment_v1",
            "structure_y_sign": 1.0,
        },
    }
    manifest = tmp_path / "fixture.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report, _ = MODULE.manifest_report(
        manifest,
        MODULE.asdict(MODULE.ActiveSpatialEnvConfig()),
        {
            "canonical_task_metric_version": row["canonical_task_metric_version"],
            "camera_model_version": row["camera_model_version"],
            "collision_convention": row["collision_convention"],
        },
    )
    assert report["canonical_gate"]["status"] == "PASS"
    assert report["canonical_gate"]["camera_model_version_ok"] == 1
    assert report["canonical_gate"]["frozen_collision_convention_ok"] == 1


def test_manifest_gate_rejects_contract_drift(tmp_path):
    row = {
        "task_id": "contract-drift-fixture",
        "task_type": "projective_relations",
        "canonical_task_metric_version": "canonical_spatial_task_h1_v1",
        "camera_model_version": "wrong-camera",
        "collision_convention": {
            "version": "interiorgs_structure_label_alignment_v1",
            "structure_y_sign": 1.0,
        },
    }
    manifest = tmp_path / "fixture.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report, _ = MODULE.manifest_report(
        manifest,
        MODULE.asdict(MODULE.ActiveSpatialEnvConfig()),
        {
            "canonical_task_metric_version": "canonical_spatial_task_h1_v1",
            "camera_model_version": "canonical_h1_from_frozen_candidate_intrinsics_and_pose",
            "collision_convention": row["collision_convention"],
        },
    )
    assert report["canonical_gate"]["status"] == "BLOCKED"
    assert report["canonical_gate"]["camera_model_version_ok"] == 0
