"""Seed tests of downstream/train/train_entry.py (decision D2).

Same seed, different start -> same action-head init and same sample ids of the first 20 batches;
different seed -> both change.  CPU; the sampler test uses StarVLA's own LeRobotMixtureDataset.sample_step and is
skipped where StarVLA cannot be imported.
"""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import train_entry as te  # noqa: E402


class FakeFramework(torch.nn.Module):
    """Loads a 'backbone' (fixed values from the start path, no RNG use) and creates a random action head."""

    def __init__(self, cfg):
        super().__init__()
        value = float(len(cfg["base_vlm"]))
        self.backbone = torch.nn.Parameter(torch.full((4,), value))
        self.action_model = torch.nn.Linear(8, 12)


def head_sha(cfg, seed):
    model, _ = te.seeded_build(FakeFramework, cfg, seed)
    return te.tensor_sha(model.action_model.parameters())


def test_same_seed_other_start_same_head_other_seed_other_head():
    a = head_sha({"base_vlm": "/starts/B"}, 1)
    b = head_sha({"base_vlm": "/starts/RS_longer_path"}, 1)
    c = head_sha({"base_vlm": "/starts/B"}, 2)
    assert a == b and a != c


def test_seed_dataset_passes_the_seed_and_can_be_reapplied():
    calls = []
    module = types.SimpleNamespace(get_vla_dataset=lambda *a, **kw: calls.append(kw) or kw)
    te.seed_dataset(7, module)
    te.seed_dataset(8, module)  # wrapping twice does not stack: the latest seed wins
    assert module.get_vla_dataset(data_cfg="x")["seed"] == 8
    assert calls == [{"data_cfg": "x", "seed": 8}]


def test_force_bf16_loading_overrides_auto():
    class FakeModel:
        seen = []

        @classmethod
        def from_pretrained(cls, path, **kw):
            cls.seen.append(kw)
            return cls

    te.force_bf16_loading(FakeModel)
    te.force_bf16_loading(FakeModel)  # idempotent
    FakeModel.from_pretrained("/w", torch_dtype="auto", attn_implementation="sdpa")
    assert FakeModel.seen == [{"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}]


def mixture(seed):
    datasets_mod = pytest.importorskip("starVLA.dataloader.gr00t_lerobot.datasets")
    import numpy as np

    ds = object.__new__(datasets_mod.LeRobotMixtureDataset)

    class Single:
        def __init__(self, n):
            self.trajectory_ids = np.arange(n) * 3
            self.trajectory_lengths = np.full(n, 40)

    ds.datasets = [Single(5), Single(7), Single(9)]
    ds.mode, ds.epoch, ds.seed = "train", 0, seed
    ds.drop_incomplete_action_chunks = True
    ds._valid_step_bounds = [(0, 15)] * 3
    ds._dataset_sampling_weights = np.full(3, 1 / 3)
    ds._trajectory_sampling_weights = [np.full(len(d.trajectory_ids), 1 / len(d.trajectory_ids)) for d in ds.datasets]
    return ds


def test_sample_ids_follow_the_seed_only():
    if str(Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA")) not in sys.path:
        sys.path.insert(0, "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA")
    first, sha1 = te.sample_ids(mixture(1), 20, 64)
    again, sha1b = te.sample_ids(mixture(1), 20, 64)  # a second run (any start) with the same seed
    other, sha2 = te.sample_ids(mixture(2), 20, 64)
    assert len(first) == 20 and len(first[0]) == 64
    assert first == again and sha1 == sha1b
    assert sha1 != sha2
