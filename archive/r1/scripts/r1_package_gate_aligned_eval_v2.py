#!/usr/bin/env python3
"""Publish an immutable reward-contract-aligned fixed-32 evaluation package."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005"
SOURCE = RUN / "eval_package_v3"
OUTPUT = RUN / "eval_package_v4"
OVERLAYS = (
    "vagen/envs/active_spatial/env.py",
    "vagen/envs/active_spatial/env_config.py",
    "vagen/envs/active_spatial/prompt.py",
    "vagen/envs/active_spatial/reward_trace.py",
    "scripts/r1_run_canonical_dev_eval32.py",
    "scripts/r1_vllm_eval_preflight.py",
    "scripts/r1_compare_gate_aligned_pilot.py",
    "examples/train/active_spatial/sco_r1_gate_aligned_eval.sh",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    OUTPUT.mkdir(exist_ok=False)
    with tarfile.open(SOURCE / "source.tar.gz") as archive:
        files = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
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
    with tarfile.open(OUTPUT / "source.tar.gz", "w:gz") as archive:
        for name, value in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    shutil.copy2(SOURCE / "owner.sh", OUTPUT / "owner.sh")
    for mode, limit in (("preflight", "4h"), ("renderer", "16h"), ("evaluation", "16h")):
        (OUTPUT / f"launch_{mode}.sh").write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
PACKAGE={OUTPUT}
RUN={RUN}
if [[ $(id -u) == 0 ]]; then exec bash "${{PACKAGE}}/owner.sh" bash "$0" owner; fi
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
cd "${{PACKAGE}}"; sha256sum -c SHA256SUMS
WORK=$(mktemp -d /tmp/r1_gate_eval_v2.XXXXXXXX); tar -xzf source.tar.gz -C "${{WORK}}"; cd "${{WORK}}"
export R1_EVAL_TAG=_v2
exec timeout --signal=TERM --kill-after=120s {limit} bash examples/train/active_spatial/sco_r1_gate_aligned_eval.sh {mode} "${{RUN}}"
"""
        )
    provenance = {
        "role": "independent_fixed32_evaluation_reward_contract_recheck",
        "supersedes": str(SOURCE),
        "source_archive_sha256_before": digest(SOURCE / "source.tar.gz"),
        "models": ["base", "step2", "step4", "step8"],
        "tasks_per_model": 32,
        "output_root": "paired_eval_v2",
        "preflight_root": "eval_preflight_v2",
        "reward_contract": {
            "potential": "canonical_gate_aligned_shaping_v1",
            "gamma": 0.95,
            "near_success_bonus": 0.0,
            "visibility_shaping": False,
            "format_reward": 0.0,
            "invalid_format_penalty": -0.1,
        },
        "comparison_addition": "collision_attempt_rate_not_worse",
        "overlays": changes,
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (OUTPUT / "SHA256SUMS").write_text(
        "".join(
            f"{digest(path)}  {path.name}\n"
            for path in sorted(OUTPUT.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        )
    )
    print(
        json.dumps(
            {"package": str(OUTPUT), "source_sha256": digest(OUTPUT / "source.tar.gz")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
