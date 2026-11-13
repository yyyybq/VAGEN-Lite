"""The privileged critic of the per-turn PPO (trainer.critic_source "features", vagen/feature_critic.py; the user's
supplement of 2026-11-13, gate 1 of docs/PREREG_PPO2.md) and the env's ground truth it reads
(vagen/envs/active_spatial/critic_features.py)."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from vagen import feature_critic as FC
from vagen import turn_gae as TG


def feat(d=3, bearing="front", condition=None, family="fov_inclusion", near=False, table=True):
    return {"v": FC.FEATURE_VERSION, "d": d, "near": near, "table": table, "bearing": bearing,
            "condition": condition, "family": family}


def test_encoding():
    n = FC.dim(12)
    assert n == 29
    x = FC.encode(feat(d=3, bearing="right", condition="ok", family="absolute_positioning"), 2, 12)
    assert len(x) == n
    assert x[:2] == [3 / 12, 0.0]
    assert x[2:7] == [0.0, 1.0, 0.0, 0.0, 0.0]          # front right back left none
    assert x[7:11] == [0.0, 1.0, 0.0, 0.0]               # too_close ok too_far none
    assert x[11:23] == [0.0, 0.0, 1.0] + [0.0] * 9       # turn index 2
    assert x[23:] == [1.0, 0, 0, 0, 0, 0]
    # no distance: 1.0 and the flag; no bearing / condition: the none slots; a turn past the limit: the last slot
    y = FC.encode(feat(d=None, bearing=None, condition=None), 20, 12)
    assert y[:2] == [1.0, 1.0] and y[6] == 1.0 and y[10] == 1.0 and y[22] == 1.0
    assert FC.encode(feat(d=40), 0, 12)[0] == 1.0  # capped at D_CAP
    for bad, word in ((None, "no observation"), (dict(feat(), v=0), "version"), (feat(family="x"), "family"),
                      (feat(bearing="up"), "bearing"), (feat(condition="near"), "condition")):
        with pytest.raises(ValueError, match=word):
            FC.encode(bad, 0, 12)


def test_record_and_kept_turns():
    rec = TG.turn_record([[0, 3], [5, 8], [10, 12]], [-0.01, 0.5, 5.0])
    with pytest.raises(ValueError, match="observations"):
        FC.add_to_record(dict(rec), [feat()], True)
    FC.add_to_record(rec, [feat(d=4), feat(d=2), feat(d=1), feat(d=0)], True)  # one more (after the last step)
    assert [f["d"] for f in rec["features"]] == [4, 2, 1] and rec["success"] is True
    mask = torch.zeros(12)
    mask[0:3] = 1; mask[5:8] = 1; mask[10:12] = 1
    rows = FC.episode_rows(rec, mask, 12)
    assert rows["anchors"] == [0, 5, 10] and rows["turns_left"] == [3.0, 2.0, 1.0]
    assert rows["returns"] == pytest.approx([5.49, 5.5, 5.0]) and rows["success"] == 1.0
    assert [r[0] for r in rows["x"]] == pytest.approx([4 / 12, 2 / 12, 1 / 12])
    # a third turn cut off by the response length: its reward folds into turn 2 (as turn_gae), its state is dropped
    cut = TG.turn_record([[0, 4], [6, 10], [12, 15]], [0.1, 0.2, 0.3])
    FC.add_to_record(cut, [feat(d=5), feat(d=3), feat(d=2)], False)
    m = torch.zeros(12)
    m[0:4] = 1; m[6:10] = 1
    kept = FC.kept_turns(cut, m)
    assert kept == [(0, 0, pytest.approx(0.1)), (6, 1, pytest.approx(0.5))]
    rows = FC.episode_rows(cut, m, 12)
    assert rows["anchors"] == [0, 6] and rows["turns_left"] == [2.0, 1.0] and rows["success"] == 0.0
    assert rows["returns"] == pytest.approx([0.6, 0.5])
    # an empty first turn gives its reward to the next kept turn; the kept turn keeps its own index
    first = {"spans": [[0, 0], [0, 2], [4, 6]], "rewards": [0.5, 1.0, 2.0]}
    FC.add_to_record(first, [feat(d=6), feat(d=5), feat(d=4)], False)
    m2 = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
    assert FC.kept_turns(first, m2) == [(0, 1, 1.5), (4, 2, 2.0)]
    with pytest.raises(KeyError, match="observation features"):
        FC.episode_rows(TG.turn_record([[0, 3]], [1.0]), torch.tensor([1, 1, 1]), 12)


def _proto(n_ep=4, seed=0):
    """n_ep concat episodes of 3 turns (positions 0-2, 5-7, 10-11), success when the first turn's d is small."""
    from verl import DataProto

    rng = np.random.default_rng(seed)
    mask = torch.zeros(n_ep, 12)
    mask[:, 0:3] = 1; mask[:, 5:8] = 1; mask[:, 10:12] = 1
    recs = np.empty(n_ep, dtype=object)
    rewards = torch.zeros(n_ep, 12)
    for i in range(n_ep):
        d0 = int(rng.integers(1, 9))
        ok = d0 <= 4
        rw = [-0.01, -0.01, 5.0 if ok else -0.01]
        rec = TG.turn_record([[0, 3], [5, 8], [10, 12]], rw)
        FC.add_to_record(rec, [feat(d=d0), feat(d=max(d0 - 2, 0)), feat(d=max(d0 - 4, 0)), feat(d=0)], ok)
        recs[i] = rec
        rewards[i, 11] = sum(rw)
    return DataProto.from_dict(tensors={"response_mask": mask, "token_level_rewards": rewards,
                                        "responses": torch.zeros(n_ep, 12, dtype=torch.long)},
                               non_tensors={TG.RECORD_KEY: recs, "uid": np.array([f"u{i}" for i in range(n_ep)], dtype=object)})


def test_values_update_and_learning():
    critic = FC.FeatureCritic({"max_turns": 12, "epochs": 20, "mini_batch": 64}, seed=3)
    data = _proto(64)
    values, m = critic.values(data)
    assert values.shape == (64, 12) and float(values.abs().max()) == 0.0  # the read-out starts at 0
    assert m["critic/feature/anchors"] == 192 and m["critic/feature/d_missing_rate"] == 0.0
    um = critic.update(data)
    for key in ("bce_before", "bce_after", "ev_mc_before", "ev_mc_buffer_after", "readout_a", "readout_b", "readout_c",
                "rows_new", "buffer_rows", "buffer_updates", "seconds", "success_rate_new"):
        assert "critic/feature/" + key in um, key
    assert um["critic/feature/rows_new"] == 192 and um["critic/feature/buffer_updates"] == 1
    assert um["critic/feature/bce_after"] < um["critic/feature/bce_before"]
    for k in range(8):
        um = critic.update(_proto(64, seed=k + 1))
    assert um["critic/feature/buffer_updates"] == 8 and um["critic/feature/buffer_rows"] == 8 * 192  # the last 8
    # learnable from d: values only at the anchors, and they track the return
    test = _proto(64, seed=99)
    values, _ = critic.values(test)
    mask = test.batch["response_mask"]
    off = values[(mask == 0) | torch.tensor([[p not in (0, 5, 10) for p in range(12)]] * 64)]
    assert float(off.abs().max()) == 0.0
    rows = critic.batch_rows(test)
    v = torch.tensor([float(values[i, a]) for i, r in enumerate(rows) for a in r["anchors"]])
    g = torch.tensor([x for r in rows for x in r["returns"]])
    assert FC.explained_variance(v, g) > 0.8
    # into the estimator: the returns at the anchors are lambda-returns from these values
    test.batch["values"] = values
    adv, ret, _ = TG.compute(test.batch["token_level_rewards"], values, mask, list(test.non_tensor_batch[TG.RECORD_KEY]),
                             1.0, 0.95)
    assert torch.isfinite(adv).all() and (ret[:, 0] != TG.IGNORE_VALUE).all()


def test_checkpoint_round_trip_and_determinism():
    a = FC.FeatureCritic({"max_turns": 12}, seed=5)
    b = FC.FeatureCritic({"max_turns": 12}, seed=5)
    for k in range(3):
        a.update(_proto(16, seed=k)); b.update(_proto(16, seed=k))
    va, _ = a.values(_proto(16, seed=50))
    vb, _ = b.values(_proto(16, seed=50))
    assert torch.equal(va, vb)  # the same seed, the same updates: the same values
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / FC.CHECKPOINT_FILE)
        a.save(path)
        c = FC.FeatureCritic({"max_turns": 12}, seed=5)
        c.load(path)
        a.update(_proto(16, seed=7)); c.update(_proto(16, seed=7))
        assert torch.equal(a.values(_proto(16, seed=8))[0], c.values(_proto(16, seed=8))[0])
        assert c.updates == a.updates == 4
        with pytest.raises(ValueError, match="cfg"):
            FC.FeatureCritic({"max_turns": 12, "hidden": 64}, seed=5).load(path)
    # the driver's global RNG is untouched by building and training a critic
    torch.manual_seed(123)
    want = torch.rand(3)
    torch.manual_seed(123)
    d = FC.FeatureCritic({"max_turns": 12}, seed=9)
    d.update(_proto(8))
    assert torch.equal(torch.rand(3), want)


def test_settings_and_config_refusals():
    from omegaconf import OmegaConf

    s = FC.settings({"max_turns": 12})
    assert s == dict(FC.DEFAULTS, max_turns=12)
    for bad, word in (({}, "max_turns"), ({"max_turns": 12, "depth": 2}, "unknown"), ({"max_turns": 12, "epochs": 0}, "epochs"),
                      ({"max_turns": 12, "hidden": 2.5}, "hidden"), ({"max_turns": 12, "lr": 0}, "lr"),
                      ({"max_turns": 12, "ridge": -1}, "ridge"), ({"max_turns": True}, "max_turns")):
        with pytest.raises(ValueError, match=word):
            FC.settings(bad)

    def cfg(**over):
        base = {"algorithm": {"adv_estimator": "turn_gae"}, "critic": {"enable": False}, "data": {"seed": 7},
                "filter": {"enable": True},
                "trainer": {"critic_source": "features", "critic_all_candidates": True,
                            "feature_critic": {"max_turns": 12}}}
        for path, value in over.items():
            node = base
            keys = path.split("__")
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value
        return OmegaConf.create(base)

    critic = FC.from_config(cfg())
    assert isinstance(critic, FC.FeatureCritic) and critic.seed == 7
    assert FC.from_config(cfg(trainer__critic_source="model", trainer__feature_critic=None)) is None
    assert FC.from_config(OmegaConf.create({"trainer": {}})) is None  # every existing config
    for over, word in (({"algorithm__adv_estimator": "gae"}, "turn_gae"), ({"critic__enable": True}, "critic.enable"),
                       ({"critic__enable": None}, "critic.enable"), ({"trainer__critic_all_candidates": False}, "every candidate"),
                       ({"filter__enable": False}, "filter"), ({"trainer__critic_replay_updates": 8}, "critic_replay"),
                       ({"trainer__critic_overfit_steps": 50}, "critic_overfit"), ({"trainer__critic_source": "table"}, "one of"),
                       ({"trainer__critic_source": "model"}, "is not features")):
        with pytest.raises(ValueError, match=word):
            FC.from_config(cfg(**over))


def _c2w(x, y, facing):
    """A level OpenCV c2w at (x, y, 1.5): forward (facing) on the floor, down = -Z, right = down x forward."""
    fwd = np.array([math.cos(math.radians(facing)), math.sin(math.radians(facing)), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = np.cross(down, fwd), down, fwd, [x, y, 1.5]
    return c2w


def _box(x, y, oid="1"):
    return {"id": oid, "label": "box", "bbox_min": [x - 0.2, y - 0.2, 0.0], "bbox_max": [x + 0.2, y + 0.2, 0.8],
            "center": [x, y, 0.4]}


def test_env_features_from_a_table():
    from data_gen.interiorgs_v10 import difficulty as D
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.grounding_probe import truth
    from vagen.envs.active_spatial import critic_features as CF

    item = {"task_id": "t_fc", "task_type": "absolute_positioning", "target_object": {"objects": [_box(3.0, 0.0)]},
            "target_region": {"type": "footprint_distance_band", "params": {"distance_m": 2.5, "object_a_id": "1"}}}
    here, other = _c2w(0.0, 0.0, 0.0), _c2w(1.0, 1.0, 90.0)
    key = int(D.contract_keys(*D._split(here))[0])
    with tempfile.TemporaryDirectory() as tmp:
        DT.save(DT.table_path(Path(tmp), "t_fc"), {"keys": np.array([key], dtype=np.int64), "dist": np.array([3]),
                                                   "depth_of": np.array([0]), "task_id": "t_fc"})
        CF._TABLES.clear()
        f = CF.features(item, here, tmp)
        gt = truth(item, here)
        assert f == {"v": CF.VERSION, "d": 3, "near": False, "table": True, "bearing": "front",
                     "condition": gt["condition"], "family": "absolute_positioning"}
        assert gt["bearing"] == "front" and f["condition"] in FC.CONDITIONS
        # 1 cm off the table's cell: the neighbour's distance
        shifted = here.copy()
        shifted[0, 3] += 0.01
        g = CF.features(item, shifted, tmp)
        assert (g["d"], g["near"]) == (3, True)
        # a pose far off the table: no distance; facing +Y the object at +X is to the right
        h = CF.features(item, other, tmp)
        assert (h["d"], h["near"], h["bearing"]) == (None, False, "right")
        # another family has no condition; inside the footprint no bearing; a task without a table
        fov = dict(item, task_type="fov_inclusion", task_id="none")
        k = CF.features(fov, here, tmp)
        assert (k["condition"], k["table"], k["d"], k["family"]) == (None, False, None, "fov_inclusion")
        inside = dict(fov, target_object={"objects": [_box(0.05, 0.0)]})
        assert CF.features(inside, here, tmp)["bearing"] is None
        assert FC.encode(f, 0, 12)[:2] == [0.25, 0.0]  # the encoding reads what the env gives
        with pytest.raises(ValueError, match="critic_feature_tables_dir"):
            CF.features(item, here, None)
    CF._TABLES.clear()


def test_env_config_default_off():
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    cfg = ActiveSpatialEnvConfig()
    assert cfg.critic_features is False and cfg.critic_feature_tables_dir is None


def test_env_reset_and_step_wiring():
    """The env adds info["critic_features"] only with the switch (read from the source: the env needs a renderer)."""
    import inspect

    from vagen.envs.active_spatial.env import ActiveSpatialEnv

    for fn in (ActiveSpatialEnv.reset, ActiveSpatialEnv.step):
        src = inspect.getsource(fn)
        assert src.count('info["critic_features"] = self._critic_features()') == 1, fn
        assert 'if self.config.get("critic_features", False):' in src, fn


def test_agent_loop_records_the_features():
    import inspect

    from vagen.agent_loop.gym_agent_loop import AgentData, GymAgentLoop

    d = AgentData(messages=[], image_data=[], metrics={}, request_id="r", env=None, response_limit=100, env_name="e")
    assert d.obs_features == []
    src = inspect.getsource(GymAgentLoop.run)
    assert 'if info and "critic_features" in info:' in src and "add_to_record(" in src
    assert 'agent_data.obs_features.append((info or {}).get("critic_features"))' in inspect.getsource(
        GymAgentLoop._handle_env_state)


def test_trainer_wiring():
    """The fit loop's wiring (read from the source: the loop needs Ray workers): values from the feature critic only
    when there is no value network, its update on the every-candidate batch, the warm-up counts it, the checkpoint
    holds its state and a resume needs it."""
    import inspect

    from vagen.ray_trainer import RayPPOTrainer

    fit = inspect.getsource(RayPPOTrainer.fit)
    assert fit.count("self.feature_critic.values(batch)") == 1 and fit.count("self.feature_critic.update(pre_filter_batch)") == 1
    i, j = fit.index("values = self.critic_wg.compute_values(batch)"), fit.index("self.feature_critic.values(batch)")
    assert i < j and "elif self.feature_critic is not None:" in fit[i:j]
    assert "if (self.use_critic or self.feature_critic is not None) else 0" in fit
    save = inspect.getsource(RayPPOTrainer._save_checkpoint)
    assert "self.feature_critic.save(feature_critic_path)" in save and 'checkpoint_metadata.get("has_feature_critic", True)' in save
    load = inspect.getsource(RayPPOTrainer._load_checkpoint)
    assert "self.feature_critic.load(feature_critic_path)" in load and "cannot resume" in load
    init = inspect.getsource(RayPPOTrainer.__init__)
    assert "self.feature_critic = _feature_critic_from_config(self.config)" in init
