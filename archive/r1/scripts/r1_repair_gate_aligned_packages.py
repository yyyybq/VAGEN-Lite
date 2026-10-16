#!/usr/bin/env python3
"""Publish package_v2 with the reward ledger dependency omitted by v1."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def repack(source: Path, destination: Path, launch_names: tuple[str, ...]) -> None:
    destination.mkdir(exist_ok=False)
    with tarfile.open(source / "source.tar.gz") as archive:
        files = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    for dependency in (
        "vagen/envs/active_spatial/reward_trace.py",
        "vagen/envs/active_spatial/env_config.py",
    ):
        value = (ROOT / dependency).read_bytes()
        compile(value, dependency, "exec")
        files[dependency] = value
    with tarfile.open(destination / "source.tar.gz", "w:gz") as archive:
        for name, payload in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    shutil.copy2(source / "owner.sh", destination / "owner.sh")
    for name in launch_names:
        text = (source / name).read_text().replace(str(source), str(destination))
        (destination / name).write_text(text)
    provenance = json.loads((source / "provenance.json").read_text())
    provenance.update(
        {
            "supersedes": str(source),
            "repair": "include reward_trace.py and matching env_config.py",
            "source_archive_sha256_before": digest(source / "source.tar.gz"),
        }
    )
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (destination / "SHA256SUMS").write_text(
        "".join(
            f"{digest(path)}  {path.name}\n"
            for path in sorted(destination.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        )
    )


def main() -> None:
    repack(
        RUN / "package_v2",
        RUN / "package_v3",
        ("launch_renderer.sh", "launch_training.sh"),
    )
    repack(
        RUN / "eval_package_v2",
        RUN / "eval_package_v3",
        ("launch_preflight.sh", "launch_renderer.sh", "launch_evaluation.sh"),
    )
    print(
        json.dumps(
            {
                "package_v3_sha256": digest(RUN / "package_v3/source.tar.gz"),
                "eval_package_v3_sha256": digest(RUN / "eval_package_v3/source.tar.gz"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
