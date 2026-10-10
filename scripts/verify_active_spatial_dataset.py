#!/usr/bin/env python3
"""Replay canonical candidates in the real env; publish a non-overwriting contract.

--evidence is JSONL: {"task_id": ..., "actions": ["turn_left", ...],
                    "initial_rgb": "/absolute/path/to/audited.png"}.
Each actions entry is one env turn; use | to join a multi-action turn.
No private evidence is inserted into policy observations or source rows.
"""
import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify(manifest, evidence, env_yaml, output):
    import numpy as np
    from PIL import Image
    from omegaconf import OmegaConf
    from data_gen.active_spatial_pipeline.splits import read_rows
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
    from vagen.envs.active_spatial.dataset_contract import (
        CONTRACT_VERSION, protocol, row_digest, validate_canonical_row,
    )
    if Path(output).exists():
        raise FileExistsError(f"refusing to replace certificate: {output}")
    raw = OmegaConf.to_container(OmegaConf.load(env_yaml), resolve=True)
    raw = raw["envs"][0]["config"] if "envs" in raw else raw.get("env", raw)
    keys = {f.name for f in fields(ActiveSpatialEnvConfig)}
    config = ActiveSpatialEnvConfig(**{k: v for k, v in raw.items() if k in keys})
    config.jsonl_path = str(manifest)
    config.include_task_types, config.exclude_task_types = None, []
    config.total_lines = -1
    config.require_verified_dataset = False  # This command PRODUCES the certificate.
    if config.render_backend not in ("local", "http", "client") or not config.render_fail_fast:
        raise ValueError("verification requires a real fail-fast renderer")
    rows, witnesses = read_rows(manifest), read_rows(evidence)
    lookup = {w["task_id"]: w for w in witnesses}
    if not rows or len(lookup) != len(witnesses):
        raise ValueError("empty manifest or duplicate evidence task ids")
    if len({r.get("task_id") for r in rows}) != len(rows) or any(not r.get("task_id") for r in rows):
        raise ValueError("canonical manifest needs unique task ids")
    for row in rows:
        validate_canonical_row(row)
        if row["task_id"] not in lookup:
            raise ValueError(f"missing evidence for {row['task_id']}")
    certified = {}
    env = ActiveSpatialEnv(config)
    try:
        for index, row in enumerate(rows):
            witness = lookup[row["task_id"]]
            observation, _ = env.reset(seed=index)
            if env.current_item != row or env._calculate_canonical_metric()["success"]:
                raise ValueError("wrong episode or initial state already successful")
            images = [im for values in observation["multi_modal_data"].values() for im in values]
            if len(images) != 1:
                raise ValueError("expected one initial runtime RGB frame")
            reference = Path(witness["initial_rgb"])
            old = np.asarray(Image.open(reference).convert("RGB"), dtype=np.int16)
            new = np.asarray(images[0].convert("RGB"), dtype=np.int16)
            if old.shape != new.shape:
                raise ValueError("audited/runtime RGB resolutions differ")
            error = np.abs(old-new)
            mae, p99 = float(error.mean()), float(np.quantile(error, .99))
            if mae > .5 or p99 > 2:
                raise ValueError(f"audited/runtime RGB differs: MAE={mae}, P99={p99}")
            steps = 0
            done = False
            for action in witness["actions"]:
                _, reward, done, _ = env.step("<action>" + action + "</action>")
                steps += 1
                if env.collision_count or env.invalid_action_count or not np.isfinite(reward):
                    raise ValueError("certificate replay collided or produced nonfinite reward")
                if done:
                    break
            success = bool(env._calculate_canonical_metric()["success"])
            if not steps or not success or not done:
                raise ValueError(f"no runtime-reachable successful terminal for {row['task_id']}")
            certified[row_digest(row)] = {"success": True, "rgb_match": True, "steps": steps,
                                          "mae": mae, "p99": p99,
                                          "rgb_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    finally:
        env.close()
    document = {"version": CONTRACT_VERSION, "status": "PASS", "protocol": protocol(config),
                "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                "evidence_sha256": hashlib.sha256(Path(evidence).read_bytes()).hexdigest(), "rows": certified}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x") as handle:
        json.dump(document, handle, indent=2)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "evidence", "env_yaml", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    result = verify(**vars(parser.parse_args()))
    print(f"PASS: {len(result['rows'])} certified rows")
