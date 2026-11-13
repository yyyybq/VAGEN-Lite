"""Observation key-name translation between the official evaluator and the openpi R1Pro config.

The evaluator's default robot config names the robot "robot_r1" (OmniGibson/omnigibson/eval/r1pro.yaml),
so keys look like "robot_r1::robot_r1:zed_link:Camera:0::rgb" and "robot_r1::proprio". The openpi fork's
R1Pro config (src/openpi/configs/robots/b1k.py) reads "robot::robot:zed_link:Camera:0::rgb" and
"robot::proprio". Only key NAMES are rewritten; array values are passed through untouched.
No openpi imports here, so it is unit-testable outside the openpi venv.
"""

from __future__ import annotations

from typing import Optional


def rename_keys(obs: dict, src: Optional[str], dst: str) -> dict:
    if not src or src == dst:
        return obs
    out = {}
    for k, v in obs.items():
        if isinstance(k, str) and k.startswith(f"{src}::"):
            k = f"{dst}::" + k[len(src) + 2 :].replace(f"{src}:", f"{dst}:", 1)
        out[k] = v
    return out
