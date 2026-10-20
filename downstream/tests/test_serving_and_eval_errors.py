"""The step-2 serving-contract failure and its fixes (CPU).

The policy server reads run/config.yaml, which StarVLA saves with only the keys training read; a 200-step run never
reads obs_image_size, so the server reported training_obs_image_size None and every simulator worker failed its
contract check -- and a failing Pool initializer makes multiprocessing respawn workers forever.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))
import closed_loop  # noqa: E402
import fill_serving_config  # noqa: E402


def test_fill_adds_only_missing_serving_keys():
    accessed = {"datasets": {"vla_data": {"data_mix": "x", "per_device_batch_size": 8}},
                "framework": {"action_model": {"action_dim": 12}}, "trainer": {"max_train_steps": 200}}
    full = {"datasets": {"vla_data": {"data_mix": "OTHER", "obs_image_size": [224, 224], "include_state": False}},
            "framework": {"action_model": {"action_dim": 12, "action_horizon": 16}, "qwenvl": {"base_vlm": "/b"}},
            "trainer": {"max_train_steps": 200, "eval_interval": 2000}}
    added = fill_serving_config.fill(accessed, full)
    assert sorted(added) == ["datasets.vla_data.include_state", "datasets.vla_data.obs_image_size",
                             "framework.action_model.action_horizon", "framework.qwenvl.base_vlm"]
    assert accessed["datasets"]["vla_data"]["data_mix"] == "x"  # existing values are never changed
    assert "eval_interval" not in accessed["trainer"]  # only the serving sections


def test_failed_worker_init_returns_error_rows_instead_of_hanging(monkeypatch):
    def boom(*a, **kw):
        raise TypeError("'NoneType' object is not iterable")

    monkeypatch.setattr(closed_loop, "connect", boom)
    closed_loop._STATE.clear()
    closed_loop._init_worker("/ckpt.pt", ["127.0.0.1:5678"], "/starvla")  # must not raise
    row = closed_loop.run_episode({"task": "OpenDrawer", "seed": 1, "max_steps": 750})
    assert row["status"] == "error" and "worker init" in row["error"] and row["task"] == "OpenDrawer"
    closed_loop._STATE.clear()


def test_mark_for_serving_puts_obs_image_size_into_the_saved_config(tmp_path):
    pytest.importorskip("omegaconf")
    tracker = pytest.importorskip("starVLA.training.trainer_utils.config_tracker")
    import train_entry
    from omegaconf import OmegaConf

    cfg = tracker.wrap_config(OmegaConf.create({
        "datasets": {"vla_data": {"obs_image_size": [224, 224], "include_state": False, "data_mix": "m"}},
        "framework": {"action_model": {"action_dim": 12}, "qwenvl": {"base_vlm": "/b"}},
        "trainer": {"max_train_steps": 2}}))
    _ = cfg.trainer.max_train_steps
    train_entry.mark_for_serving(cfg)
    cfg.save_accessed_config(tmp_path / "config.yaml", use_original_values=False)
    saved = OmegaConf.load(tmp_path / "config.yaml")
    assert list(saved.datasets.vla_data.obs_image_size) == [224, 224]
    assert saved.datasets.vla_data.include_state is False
