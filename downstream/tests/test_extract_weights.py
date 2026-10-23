"""Tests of downstream/train/extract_weights.py on a small DeepSpeed-like full state (CPU)."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import extract_weights as X  # noqa: E402


def full_state(tmp_path, step=150000):
    """RUN/full_state/steps_<N> as train_entry.save_full leaves it, and the run directory's config files."""
    run = tmp_path / "run"
    fs = run / "full_state" / f"steps_{step}"
    (fs / "pytorch_model").mkdir(parents=True)
    (fs / "latest").write_text("pytorch_model")
    module = {"qwen.layer.weight": torch.randn(4, 3).to(torch.bfloat16), "action_model.bias": torch.randn(5),
              "qwen.embed.weight": torch.randn(6, 2).to(torch.bfloat16)}
    torch.save({"module": module, "ds_version": "0.16", "param_shapes": [{"x": torch.Size([4, 3])}], "global_steps": step},
               str(fs / "pytorch_model" / "mp_rank_00_model_states.pt"))
    (fs / "pytorch_model" / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"not read")
    (fs / "downstream_state.json").write_text(json.dumps({"completed_steps": step, "batches_consumed": step + 37}))
    (run / "config.yaml").write_text("trainer:\n  max_train_steps: 200000\n")
    (run / "dataset_statistics.json").write_text("{}")
    return fs, run, module


def test_extract_writes_a_starvla_checkpoint_in_a_run_layout(tmp_path):
    fs, run, module = full_state(tmp_path)
    dest = tmp_path / "evals_src/steps_150000"
    meta = X.extract(fs, run, dest)
    ckpt = dest / "checkpoints/steps_150000_pytorch_model.pt"
    assert meta["step"] == 150000 and meta["checkpoint"] == str(ckpt) and ckpt.is_file()
    got = torch.load(str(ckpt), map_location="cpu", weights_only=True)  # a plain state dict, as StarVLA saves it
    assert set(got) == set(module) and all(torch.equal(got[k], module[k]) and got[k].dtype == module[k].dtype
                                           for k in module)
    # StarVLA finds the config two levels above the checkpoint (read_mode_config: CKPT.parents[1])
    assert (ckpt.parents[1] / "config.yaml").read_text() == (run / "config.yaml").read_text()
    assert (ckpt.parents[1] / "dataset_statistics.json").is_file()
    assert meta["weights"]["tensors"] == 3 and meta["weights"]["dtypes"] == {"torch.bfloat16": 2, "torch.float32": 1}
    assert meta["downstream_state"]["batches_consumed"] == 150037 and not list(dest.rglob("*.partial"))
    assert json.loads((dest / "source.json").read_text())["checkpoint_bytes"] == ckpt.stat().st_size
    again = X.extract(fs, run, dest)  # done already: nothing read or written
    assert again["already"] and again["checkpoint"] == str(ckpt)


def test_dry_run_reads_and_checks_without_writing(tmp_path):
    fs, run, module = full_state(tmp_path)
    like = tmp_path / "ref.pt"
    torch.save({k: v.clone() for k, v in module.items()}, str(like))
    dest = tmp_path / "evals_src/steps_150000"
    meta = X.extract(fs, run, dest, like=like, dry=True)
    assert not dest.exists() and meta["weights"]["tensors"] == 3 and abs(meta["sum"]) < 1e6
    torch.save({"qwen.layer.weight": torch.zeros(4, 4)}, str(like))  # another layout
    with pytest.raises(ValueError, match="differ"):
        X.extract(fs, run, dest, like=like, dry=True)


def test_a_full_state_without_weights_is_refused(tmp_path):
    fs, run, _ = full_state(tmp_path)
    (fs / "pytorch_model" / "mp_rank_00_model_states.pt").unlink()
    with pytest.raises(FileNotFoundError):
        X.extract(fs, run, tmp_path / "dest")
    assert not (tmp_path / "dest" / "source.json").exists()
