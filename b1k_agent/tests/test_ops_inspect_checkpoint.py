"""scripts/inspect_checkpoint.py on a SYNTHETIC openpi checkpoint tree (no real weights; layout as written by
openpi training/checkpoints.py save_state: params/, train_state/, assets/<asset_id>/norm_stats.json)."""

import json
import os
import subprocess
import sys

from conftest import ROOT

from agent.fields import ACTION_SLICES
from scripts import inspect_checkpoint as ic


def _norm_stats(dim_state, dim_actions):
    def entry(n):
        return {"mean": [0.1 * i for i in range(n)], "std": [1.0] * n, "q01": [-1.0] * n, "q99": [1.0] * n}

    return {"norm_stats": {"state": entry(dim_state), "actions": entry(dim_actions)}}


def make_ckpt(root, dim_state=23, dim_actions=23, asset_id="behavior-1k/2026-challenge-demos", step=None):
    base = os.path.join(root, str(step)) if step is not None else root
    os.makedirs(os.path.join(base, "params", "d"), exist_ok=True)
    for name in ("manifest.ocdbt", "_METADATA", "_sharding", os.path.join("d", "0001"), os.path.join("d", "0002")):
        with open(os.path.join(base, "params", name), "wb") as f:
            f.write(b"\0" * 1024)
    os.makedirs(os.path.join(base, "train_state"), exist_ok=True)
    with open(os.path.join(base, "_CHECKPOINT_METADATA"), "w") as f:
        json.dump({"item_handlers": {"params": "PyTreeCheckpointHandler"}, "metrics": {}, "init_timestamp_nsecs": 1}, f)
    asset_dir = os.path.join(base, "assets", *asset_id.split("/"))
    os.makedirs(asset_dir, exist_ok=True)
    with open(os.path.join(asset_dir, "norm_stats.json"), "w") as f:
        json.dump(_norm_stats(dim_state, dim_actions), f)
    return base


def test_layout_copy_matches_agent_fields():
    assert [(n, a, b) for n, a, b in ic.ACTION_LAYOUT_R1PRO] == [(k, s.start, s.stop) for k, s in ACTION_SLICES.items()]
    assert ic.STATE_LAYOUT_OPENPI_B1K[-1][2] == 23


def test_inspect_synthetic_checkpoint(tmp_path):
    root = str(tmp_path / "pi05_turn_on_the_radio")
    make_ckpt(root)
    s = ic.inspect(root, depth=3)
    inf = s["inferred"]
    assert inf["action_dim"] == 23 and inf["state_dim"] == 23
    assert inf["repo_id_for_server"] == "behavior-1k/2026-challenge-demos"
    assert inf["policy_dir"] == os.path.abspath(root)
    assert inf["params_files"] == 5 and inf["format"].startswith("orbax")
    assert inf["train_config_recorded"] is None and "does not store" in inf["train_config_note"]
    assert "R1Pro layout" in inf["action_layout"] and "left_gripper_width 14" in inf["state_layout"]
    ns = s["norm_stats"][0]
    assert ns["keys"]["actions"]["mean"]["len"] == 23 and ns["keys"]["actions"]["mean"]["preview"] == [0.0, 0.1, 0.2, 0.3]
    assert any(m["path"] == "_CHECKPOINT_METADATA" and "item_handlers" in m["json_keys"] for m in s["metadata_files"])
    text = ic.render(s)
    assert "asset id `behavior-1k/2026-challenge-demos`" in text and "params/" in text


def test_experiment_dir_with_step_dirs_and_odd_dims(tmp_path):
    exp = tmp_path / "outputs" / "checkpoints" / "pi05_b1k" / "radio_exp"
    make_ckpt(str(exp), dim_state=61, dim_actions=32, asset_id="turning_on_radio", step=30000)
    s = ic.inspect(str(exp))
    inf = s["inferred"]
    assert inf["step_dirs"] == [30000] and inf["policy_dir"].endswith(os.path.join("radio_exp", "30000"))
    assert inf["train_config_path_hint"]["config"] == "pi05_b1k"
    assert "raw evaluator proprio" in inf["state_layout"] and "padded" in inf["action_layout"]
    assert s["norm_stats"][0]["checkpoint_subdir"] == "30000"


def test_cli_writes_json_and_handles_missing_dir(tmp_path):
    root = str(tmp_path / "ck")
    make_ckpt(root)
    out = str(tmp_path / "ck.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "inspect_checkpoint.py"), root, "--out", out],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.load(open(out))["inferred"]["action_dim"] == 23
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "inspect_checkpoint.py"), str(tmp_path / "nope")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "is not a directory" in r.stdout


def test_path_hint_needs_a_numeric_step(tmp_path):
    """Regression: <...>/checkpoints/<name>/<step> (a copied checkpoint, as on 105) was reported as train config
    <name>; only outputs/checkpoints/<config>/<exp>/<step> is openpi's pattern."""
    hint = ic.path_hint("/nas/baiqiao/checkpoints/pi05_turn_on_the_radio/49999")
    assert hint["config"] is None and hint["name_fragment"] == "pi05_turn_on_the_radio"
    hint = ic.path_hint("/x/outputs/checkpoints/pi05_b1k/radio_exp/30000")
    assert (hint["config"], hint["exp"], hint["step"]) == ("pi05_b1k", "radio_exp", "30000")
    ck = make_ckpt(str(tmp_path / "checkpoints" / "pi05_turn_on_the_radio"), step=49999)
    inf = ic.inspect(os.path.dirname(ck))["inferred"]
    assert inf["policy_dir"] == ck and inf["train_config_path_hint"]["config"] is None


def test_cli_writes_json_even_if_stdout_closes_early(tmp_path):
    root = str(tmp_path / "ck")
    make_ckpt(root)
    for i in range(40):  # a long tree listing
        for j in range(40):
            p = os.path.join(root, "params", f"shard_{i:03d}", f"chunk_{j:04d}.bin")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").close()
    out = str(tmp_path / "ck.json")
    script = os.path.join(ROOT, "scripts", "inspect_checkpoint.py")
    subprocess.run(f"'{sys.executable}' '{script}' '{root}' --out '{out}' | head -n 2", shell=True,
                   capture_output=True, text=True, timeout=120)
    assert json.load(open(out))["inferred"]["action_dim"] == 23
