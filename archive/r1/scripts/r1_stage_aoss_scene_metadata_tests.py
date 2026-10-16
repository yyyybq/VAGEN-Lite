#!/usr/bin/env python3
from pathlib import Path

from r1_stage_aoss_scene_metadata import METADATA_INCLUDE_PATTERN, metadata_sync_command


def test_metadata_sync_command_is_directory_scoped_and_filtered() -> None:
    command = metadata_sync_command(
        Path("/opt/ads-cli"),
        "s3://redacted@bucket.endpoint/InteriorGS/scene",
        Path("/tmp/staging"),
    )

    assert command[-3:] == ["sync", "s3://redacted@bucket.endpoint/InteriorGS/scene/", "/tmp/staging/"]
    assert command[command.index("--include") + 1] == METADATA_INCLUDE_PATTERN
    assert "cp" not in command
