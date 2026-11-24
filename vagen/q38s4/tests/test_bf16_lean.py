"""CPU tests of vagen.q38s4.tools.bf16_lean (to_bf16 without the resume states) and of lf_nfp's switch to it."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vagen.q38.sft import to_bf16 as B
from vagen.q38s4.tools import bf16_lean as BL

RESUME = ["optimizer.bin", "pytorch_model_fsdp.bin", "rng_state_0.pth", "rng_state_1.pth", "rng_state_10.pth",
          "scheduler.pt"]
SIDE = ["chat_template.jinja", "config.json", "tokenizer.json", "trainer_state.json", "training_args.bin"]
CODE = Path(BL.__file__).resolve().parents[3]  # the folder holding vagen/


def fake_ckpt(d: Path) -> dict:
    """An fp32 checkpoint in two shards with LF's side files and, as with save_only_model false, the resume states."""
    from safetensors.torch import save_file

    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    ts = {"a.weight": torch.randn(64, 32, generator=g), "c.A_log": torch.randn(8, generator=g),
          "e.frozen": torch.randn(16, 4, generator=g).bfloat16()}
    wm = {}
    for i, part in enumerate((["a.weight"], ["c.A_log", "e.frozen"])):
        f = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file({k: ts[k] for k in part}, str(d / f), metadata={"format": "pt"})
        wm.update({k: f for k in part})
    (d / B.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    for n in SIDE:
        (d / n).write_text("{}")
    for n in RESUME:
        (d / n).write_bytes(b"x" * 1000)
    (d / "pytorch_model_fsdp_0").mkdir()  # a sharded-state folder (SHARDED_STATE_DICT): to_bf16 copies files only
    return ts


@pytest.fixture
def lean(monkeypatch):
    """bf16_lean installed for the test; to_bf16's names restored after it."""
    monkeypatch.setattr(B, "copy_side_files", B.copy_side_files)
    monkeypatch.setattr(B, "write_copy", B.write_copy)
    BL.install()
    return BL


def tensors(d: Path) -> dict:
    from safetensors import safe_open
    out = {}
    for f in B.weight_files(d):
        with safe_open(str(d / f), framework="pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys()})
    return out


@pytest.mark.parametrize("name,resume", [
    ("optimizer.bin", True), ("optimizer.pt", True), ("pytorch_model_fsdp.bin", True), ("rng_state_12.pth", True),
    ("rng_state.pth", True), ("scheduler.pt", True), ("scaler.pt", True), ("trainer_state.json", False),
    ("training_args.bin", False), ("config.json", False), ("model-00001-of-00003.safetensors", False),
    ("processor_config.json", False), ("tokenizer.json", False), ("generation_config.json", False)])
def test_is_resume_state(name, resume):
    assert BL.is_resume_state(name) is resume


def test_to_bf16_alone_copies_the_resume_states(tmp_path):
    """What S1 showed: to_bf16 copies optimizer.bin etc. into the bf16 copy."""
    fake_ckpt(tmp_path / "ck")
    m = B.convert(tmp_path / "ck", tmp_path / "bf", max_shard_gb=1e-6)
    assert m["verify"]["ok"] and set(RESUME) <= set(m["side_files"])
    assert all((tmp_path / "bf" / n).exists() for n in RESUME)


def test_sibling_copy_leaves_out_the_resume_states(tmp_path, lean):
    ts = fake_ckpt(tmp_path / "ck")
    m = B.convert(tmp_path / "ck", tmp_path / "bf", max_shard_gb=1e-6)
    assert m["verify"]["ok"] and m["side_files"] == SIDE
    assert m["skipped_resume_state"] == RESUME and m["wrapper"] == BL.MODULE
    names = set(os.listdir(tmp_path / "bf"))
    assert not names & set(RESUME) and set(SIDE) <= names and B.MANIFEST in names and B.INDEX in names
    assert "pytorch_model_fsdp_0" not in names
    man = json.loads((tmp_path / "bf" / B.MANIFEST).read_text())
    assert man["skipped_resume_state"] == RESUME and man["side_files"] == SIDE and man["verify"]["ok"]
    out = tensors(tmp_path / "bf")
    assert out["a.weight"].dtype == torch.bfloat16 and torch.equal(out["a.weight"], ts["a.weight"].bfloat16())
    assert out["c.A_log"].dtype == torch.float32 and torch.equal(out["c.A_log"], ts["c.A_log"])
    assert torch.equal(out["e.frozen"], ts["e.frozen"])
    assert all((tmp_path / "ck" / n).exists() for n in RESUME + SIDE)  # the fp32 checkpoint keeps everything
    assert tensors(tmp_path / "ck")["a.weight"].dtype == torch.float32


def test_same_shards_as_to_bf16(tmp_path, monkeypatch):
    """Only the side files differ: the bf16 shards are byte for byte to_bf16's."""
    fake_ckpt(tmp_path / "ck")
    B.convert(tmp_path / "ck", tmp_path / "plain", max_shard_gb=1e-6)
    monkeypatch.setattr(B, "copy_side_files", B.copy_side_files)
    monkeypatch.setattr(B, "write_copy", B.write_copy)
    BL.install()
    B.convert(tmp_path / "ck", tmp_path / "lean", max_shard_gb=1e-6)
    for f in B.weight_files(tmp_path / "plain") + [B.INDEX]:
        assert (tmp_path / "plain" / f).read_bytes() == (tmp_path / "lean" / f).read_bytes(), f


def test_in_place_keeps_the_resume_states(tmp_path, lean):
    fake_ckpt(tmp_path / "ck")
    m = B.convert(tmp_path / "ck", None, in_place=True)
    assert m["verify"]["ok"] and m["skipped_resume_state"] == RESUME
    assert all((tmp_path / "ck" / n).read_bytes() == b"x" * 1000 for n in RESUME)
    assert not (tmp_path / "ck" / ".bf16_tmp").exists()
    assert tensors(tmp_path / "ck")["a.weight"].dtype == torch.bfloat16


def test_command_line(tmp_path):
    """python -m vagen.q38s4.tools.bf16_lean convert SRC DST, as the callback starts it (a new process)."""
    fake_ckpt(tmp_path / "ck")
    env = dict(os.environ, PYTHONPATH=str(CODE), OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-B", "-m", BL.MODULE, "convert", str(tmp_path / "ck"), str(tmp_path / "bf"),
                        "--max-shard-gb", "1e-6"], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout)["ok"]
    assert not any((tmp_path / "bf" / n).exists() for n in RESUME)
    assert json.loads((tmp_path / "bf" / B.MANIFEST).read_text())["skipped_resume_state"] == RESUME


# ------------------------------------------------------------------------------------------------ lf_nfp's switch
def test_lf_nfp_bf16_command_runs_the_lean_copy(monkeypatch):
    from vagen.q38.sft import lf_launcher
    from vagen.q38s4 import lf_nfp

    orig = lf_launcher.bf16_command
    monkeypatch.setattr(lf_launcher, "bf16_command", orig)
    lf_nfp.lean_bf16(lf_launcher)
    ck = Path("/x/ckpt/checkpoint-500")
    for mode in ("sibling", "in_place"):
        a, b = orig(ck, mode), lf_launcher.bf16_command(ck, mode)
        assert lf_nfp.BF16_TOOL in a and lf_nfp.BF16_TOOL not in b and BL.MODULE in b
        assert b == [BL.MODULE if c == lf_nfp.BF16_TOOL else c for c in a]
    assert lf_launcher.bf16_command(ck, "sibling")[-1] == "/x/ckpt_bf16/checkpoint-500"
    assert lf_nfp.BF16_LEAN == BL.MODULE


def test_lf_nfp_refuses_an_unknown_bf16_command(monkeypatch):
    from vagen.q38.sft import lf_launcher
    from vagen.q38s4 import lf_nfp

    monkeypatch.setattr(lf_launcher, "bf16_command", lambda ckpt, mode: ["python", "-m", "something.else"])
    with pytest.raises(RuntimeError, match="to_bf16"):
        lf_nfp.lean_bf16(lf_launcher)


def test_lf_nfp_main_switches_before_the_run(monkeypatch):
    from vagen.q38.sft import lf_launcher
    from vagen.q38s4 import lf_nfp

    orig = lf_launcher.bf16_command
    monkeypatch.setattr(lf_launcher, "bf16_command", orig)
    seen = {}
    monkeypatch.setattr(lf_nfp, "settings_for", lambda argv: SimpleNamespace())
    monkeypatch.setattr(lf_nfp, "install", lambda s: seen.setdefault("install", True))
    monkeypatch.setattr(lf_launcher, "main", lambda: seen.setdefault(
        "cmd", lf_launcher.bf16_command(Path("/r/ckpt/checkpoint-1"), "sibling")))
    lf_nfp.main()
    assert seen["install"] and BL.MODULE in seen["cmd"] and lf_nfp.BF16_TOOL not in seen["cmd"]
