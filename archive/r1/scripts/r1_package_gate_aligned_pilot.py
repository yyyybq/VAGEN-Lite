#!/usr/bin/env python3
"""Publish an immutable, isolated eight-update gate-aligned reward pilot."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "exps/vagen_active_spatial/R1-task-context-v1-acceptance-20261003"
RUN = ROOT / "exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005"
OVERLAYS = (
    "vagen/envs/active_spatial/canonical_task_metrics.py",
    "vagen/envs/active_spatial/env.py",
    "vagen/envs/active_spatial/prompt.py",
    "vagen/r1_clean_projective_ppo.py",
    "scripts/r1_gate_aligned_reward_replay.py",
    "examples/train/active_spatial/sco_r1_gate_aligned_pilot.sh",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_hashes(directory: Path) -> None:
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{digest(path)}  {path.name}\n"
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        )
    )


def rewrite_env_yaml(source: Path, target: Path, train_jsonl: Path) -> None:
    document = yaml.safe_load(source.read_text())
    config = document["envs"][0]["config"]
    config.update(
        jsonl_path=str(train_jsonl),
        enable_potential_shaping_reward=True,
        enable_near_success_reward=False,
        enable_visibility_shaping_reward=False,
        near_success_bonus=0.0,
        format_reward=0.0,
        invalid_format_penalty=-0.1,
        potential_field_gamma=0.95,
    )
    target.write_text(yaml.safe_dump(document, sort_keys=False))


def main() -> None:
    if RUN.exists():
        raise FileExistsError(f"immutable pilot already exists: {RUN}")
    stage = Path(tempfile.mkdtemp(prefix=".gate-aligned-pilot-", dir=RUN.parent))
    frozen = stage / "frozen"
    package = stage / "package"
    frozen.mkdir()
    package.mkdir()

    for name in (
        "audit_only.jsonl",
        "data_gate.json",
        "development_regression_manifest.jsonl",
        "eval_policy_rows.jsonl",
        "policy_input_rows.jsonl",
        "train.jsonl",
    ):
        shutil.copy2(SOURCE_RUN / "frozen" / name, frozen / name)

    rewrite_env_yaml(SOURCE_RUN / "frozen/train.yaml", frozen / "train.yaml", RUN / "frozen/train.jsonl")
    rewrite_env_yaml(SOURCE_RUN / "frozen/val.yaml", frozen / "val.yaml", RUN / "frozen/train.jsonl")

    pilot = yaml.safe_load((SOURCE_RUN / "frozen/smoke.yaml").read_text())
    pilot["data"]["train_files"] = str(RUN / "frozen/train.yaml")
    pilot["data"]["val_files"] = str(RUN / "frozen/val.yaml")
    pilot["trainer"].update(
        experiment_name=RUN.name,
        rollout_data_dir=str(RUN / "pilot/rollout_data"),
        validation_data_dir=str(RUN / "pilot/validation"),
        default_local_dir=str(RUN / "pilot/checkpoints"),
        save_freq=2,
        ckpt_milestones=[2, 4, 8],
        max_actor_ckpt_to_keep=3,
        max_critic_ckpt_to_keep=3,
    )
    # Evaluation-only snapshots: retain the merged actor weights but avoid
    # replicating 200+ GiB of optimizer/critic state three times.
    pilot["actor_rollout_ref"]["actor"]["checkpoint"]["save_contents"] = ["hf_model"]
    pilot["actor_rollout_ref"]["actor"]["checkpoint"]["load_contents"] = []
    pilot["critic"]["checkpoint"]["save_contents"] = []
    pilot["critic"]["checkpoint"]["load_contents"] = []
    (frozen / "pilot.yaml").write_text(yaml.safe_dump(pilot, sort_keys=False))

    protocol = json.loads((SOURCE_RUN / "frozen/eval_protocol.json").read_text())
    protocol["audit_manifest"]["path"] = str(frozen / "development_regression_manifest.jsonl").replace(str(stage), str(RUN))
    protocol["policy_input"]["path"] = str(frozen / "policy_input_rows.jsonl").replace(str(stage), str(RUN))
    (frozen / "eval_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_hashes(frozen)

    replay = json.loads((SOURCE_RUN / "gate_aligned_reward_replay.json").read_text())
    if replay["status"] != "PASS" or not all(replay["checks"].values()):
        raise RuntimeError("offline reward replay did not pass")
    (stage / "gate_aligned_reward_replay.json").write_text(json.dumps(replay, indent=2) + "\n")

    base_archive = SOURCE_RUN / "package/source.tar.gz"
    with tarfile.open(base_archive) as archive:
        files = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    changes = {}
    for name in OVERLAYS:
        value = (ROOT / name).read_bytes()
        if name.endswith(".py"):
            compile(value, name, "exec")
        changes[name] = {
            "before": hashlib.sha256(files[name]).hexdigest() if name in files else None,
            "after": hashlib.sha256(value).hexdigest(),
        }
        files[name] = value
    with tarfile.open(package / "source.tar.gz", "w:gz") as archive:
        for name, value in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    shutil.copy2(SOURCE_RUN / "package/owner.sh", package / "owner.sh")

    for mode, limit in (("renderer", "14h"), ("training", "14h")):
        (package / f"launch_{mode}.sh").write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
PACKAGE={RUN}/package
RUN={RUN}
if [[ $(id -u) == 0 ]]; then exec bash "${{PACKAGE}}/owner.sh" bash "$0" owner; fi
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
cd "${{PACKAGE}}"; sha256sum -c SHA256SUMS
WORK=$(mktemp -d /tmp/r1_gate_aligned.XXXXXXXX)
tar -xzf source.tar.gz -C "${{WORK}}"; cd "${{WORK}}"
exec timeout --signal=TERM --kill-after=120s {limit} bash examples/train/active_spatial/sco_r1_gate_aligned_pilot.sh {mode} "${{RUN}}"
"""
        )
    provenance = {
        "role": "gate_aligned_reward_pilot",
        "base_archive": str(base_archive),
        "base_archive_sha256": digest(base_archive),
        "overlays": changes,
        "updates": 8,
        "evaluation_steps": [2, 4, 8],
        "fresh_pretrained_initialization": True,
        "historical_step1_checkpoint_mutated": False,
        "canonical_success_gate_changed": False,
        "offline_replay_status": "PASS",
        "snapshot_contents": "actor_hf_model_only; no optimizer resume",
        "ppo_gamma": 0.95,
        "potential_gamma": 0.95,
    }
    (package / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_hashes(package)

    os.rename(stage, RUN)
    print(json.dumps({"run": str(RUN), "package_sha256": digest(RUN / "package/source.tar.gz")}, indent=2))


if __name__ == "__main__":
    main()
