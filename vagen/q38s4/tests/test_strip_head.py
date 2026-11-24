"""CPU tests of vagen.q38s4.tools.strip_head (a C-NFP bf16 copy without the NFP head, for vLLM)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vagen.q38s4.tools import strip_head as S
from vagen.q38s4.tests.tiny import cpu_gdn, needs_models  # noqa: F401 (cpu_gdn is a fixture)

CODE = Path(S.__file__).resolve().parents[3]  # the folder holding vagen/
SIDE = ["chat_template.jinja", "config.json", "generation_config.json", "processor_config.json", "tokenizer.json",
        "tokenizer_config.json", "trainer_state.json", "training_args.bin"]
SMOKE_COPY = Path("/mnt/umm/users/yinbaiqiao/q38/s4/runs/smoke/s1v2_nfp/ckpt_bf16/checkpoint-50")


def head_tensors(g):
    return {"nfp_head.fc1.weight": torch.randn(8, 8, generator=g).bfloat16(),
            "nfp_head.fc1.bias": torch.randn(8, generator=g).bfloat16(),
            "nfp_head.fc2.weight": torch.randn(3, 8, generator=g).bfloat16(),
            "nfp_head.fc2.bias": torch.randn(3, generator=g).bfloat16()}


def fake_copy(d: Path, shards=None, side=SIDE, extra=()) -> dict:
    """A bf16 copy as bf16_lean writes it: shards, index with total_size, side files, the bf16 manifest."""
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    ts = {"model.a.weight": torch.randn(16, 8, generator=g).bfloat16(),
          "model.b.A_log": torch.randn(5, generator=g),                       # fp32 kept, as to_bf16 does
          "model.c.weight": torch.randn(7, 3, generator=g).bfloat16(),        # 21 items: an odd byte count / 2
          "model.d.A_log": torch.randn(3, generator=g),
          "model.e.weight": torch.randn(4, 4, generator=g).bfloat16()}
    ts.update(head_tensors(g))
    shards = shards or [["model.a.weight", "model.b.A_log"], ["model.c.weight"],
                        ["model.e.weight", "nfp_head.fc1.bias", "model.d.A_log", "nfp_head.fc1.weight",
                         "nfp_head.fc2.bias", "nfp_head.fc2.weight"]]
    wm, total = {}, 0
    for i, part in enumerate(shards):
        f = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: ts[k] for k in part}, str(d / f), metadata={"format": "pt"})
        wm.update({k: f for k in part})
        total += sum(ts[k].numel() * ts[k].element_size() for k in part)
    (d / S.INDEX).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": wm}))
    for n in side:
        (d / n).write_text(json.dumps({"name": n}))
    (d / S.BF16_MANIFEST).write_text('{"verify": {"ok": true}}')
    for n in extra:
        (d / n).write_bytes(b"x" * 100)
    return ts


def read_all(d: Path) -> dict:
    out = {}
    for f in sorted(p for p in d.iterdir() if p.name.endswith(".safetensors")):
        with safe_open(str(f), framework="pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys()})
    return out


def tree_digest(d: Path) -> dict:
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


def bytes_of(t: torch.Tensor) -> bytes:
    return t.reshape(-1).view(torch.uint8).numpy().tobytes()


# ------------------------------------------------------------------------------------------------ make
def test_make_strips_exactly_the_head(tmp_path):
    ts = fake_copy(tmp_path / "src", extra=["optimizer.bin", "rng_state_0.pth"])
    before = tree_digest(tmp_path / "src")
    m = S.make(tmp_path / "src", tmp_path / "dst")
    dst = tmp_path / "dst"
    assert m["check"]["ok"] and m["check"]["problems"] == [] and not (tmp_path / "dst.partial").exists()
    assert m["linked"] == ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors"]
    assert list(m["rewritten"]) == ["model-00003-of-00003.safetensors"] and m["dropped_shards"] == []
    assert m["rewritten"]["model-00003-of-00003.safetensors"]["dropped"] == sorted(S.HEAD_KEYS)
    for f in m["linked"]:
        assert (dst / f).is_symlink() and os.readlink(dst / f) == str((tmp_path / "src" / f).resolve())
    assert not (dst / "model-00003-of-00003.safetensors").is_symlink()
    out = read_all(dst)
    assert set(out) == set(ts) - set(S.HEAD_KEYS)
    for k, v in out.items():
        assert v.dtype == ts[k].dtype and v.shape == ts[k].shape and bytes_of(v) == bytes_of(ts[k]), k
    with safe_open(str(dst / "model-00003-of-00003.safetensors"), framework="pt") as h:
        assert h.metadata() == {"format": "pt"}
    idx = json.loads((dst / S.INDEX).read_text())
    head_bytes = sum(ts[k].numel() * ts[k].element_size() for k in S.HEAD_KEYS)
    src_idx = json.loads((tmp_path / "src" / S.INDEX).read_text())
    assert m["head_bytes"] == head_bytes
    assert idx["metadata"]["total_size"] == src_idx["metadata"]["total_size"] - head_bytes
    assert idx["metadata"]["total_size"] == sum(v.numel() * v.element_size() for v in out.values())
    assert set(idx["weight_map"]) == set(out) and not any(k.startswith("nfp_head") for k in idx["weight_map"])
    names = set(os.listdir(dst))
    assert set(SIDE) <= names and S.MANIFEST in names
    assert not names & {"optimizer.bin", "rng_state_0.pth", S.BF16_MANIFEST}
    assert all((dst / n).read_text() == (tmp_path / "src" / n).read_text() for n in SIDE)
    man = json.loads((dst / S.MANIFEST).read_text())
    assert man["skipped"] == [S.BF16_MANIFEST, "optimizer.bin", "rng_state_0.pth"] and man["check"]["ok"]
    assert man["source_bf16_manifest_md5"] == hashlib.md5((tmp_path / "src" / S.BF16_MANIFEST).read_bytes()).hexdigest()
    assert tree_digest(tmp_path / "src") == before  # SRC only read
    assert S.check(tmp_path / "src", dst)["ok"]


def test_rewritten_shard_is_aligned_and_kept_in_order(tmp_path):
    """fp32 first, then bf16, each class in SRC's order; every offset a multiple of its item size."""
    fake_copy(tmp_path / "src", shards=[["model.c.weight", "nfp_head.fc1.bias", "model.b.A_log", "model.a.weight",
                                         "nfp_head.fc1.weight", "nfp_head.fc2.bias", "nfp_head.fc2.weight",
                                         "model.d.A_log", "model.e.weight"]])
    S.make(tmp_path / "src", tmp_path / "dst")
    f = "model-00001-of-00001.safetensors"
    start, meta, ts = S.read_header(tmp_path / "dst" / f)
    assert start % 8 == 0 and meta == {"format": "pt"}
    order = sorted(ts, key=lambda k: ts[k]["data_offsets"][0])
    _, _, src_ts = S.read_header(tmp_path / "src" / f)
    src_order = [k for k in sorted(src_ts, key=lambda k: src_ts[k]["data_offsets"][0]) if k not in S.HEAD_KEYS]
    f32 = [k for k in src_order if src_ts[k]["dtype"] == "F32"]
    assert order == f32 + [k for k in src_order if k not in f32]
    assert all(ts[k]["data_offsets"][0] % S.item_bytes(ts[k]["dtype"]) == 0 for k in ts)


def test_a_shard_of_only_head_tensors_is_dropped(tmp_path):
    ts = fake_copy(tmp_path / "src", shards=[["model.a.weight", "model.b.A_log", "model.c.weight", "model.d.A_log",
                                              "model.e.weight"], sorted(S.HEAD_KEYS)])
    m = S.make(tmp_path / "src", tmp_path / "dst")
    assert m["dropped_shards"] == ["model-00002-of-00002.safetensors"] and m["rewritten"] == {}
    assert not (tmp_path / "dst" / "model-00002-of-00002.safetensors").exists()
    idx = json.loads((tmp_path / "dst" / S.INDEX).read_text())
    assert set(idx["weight_map"].values()) == {"model-00001-of-00002.safetensors"}
    assert set(read_all(tmp_path / "dst")) == set(ts) - set(S.HEAD_KEYS)


def test_plan_reads_headers_only(tmp_path):
    fake_copy(tmp_path / "src")
    p = S.source_plan(tmp_path / "src")
    assert p["shards"] == {"model-00001-of-00003.safetensors": "link", "model-00002-of-00003.safetensors": "link",
                           "model-00003-of-00003.safetensors": "rewrite"}
    assert p["head_keys"] == sorted(S.HEAD_KEYS) and p["side_files"] == SIDE and p["skipped"] == [S.BF16_MANIFEST]
    assert not (tmp_path / "dst").exists()


# ------------------------------------------------------------------------------------------------ refusals
def test_refuses_a_copy_without_the_head(tmp_path):
    fake_copy(tmp_path / "src", shards=[["model.a.weight", "model.b.A_log"], ["model.c.weight", "model.d.A_log",
                                                                              "model.e.weight"]])
    with pytest.raises(S.StripError, match="C-noNFP"):
        S.make(tmp_path / "src", tmp_path / "dst")
    assert not (tmp_path / "dst").exists() and not (tmp_path / "dst.partial").exists()


@pytest.mark.parametrize("keys", [sorted(S.HEAD_KEYS)[:3], sorted(S.HEAD_KEYS) + ["nfp_head.fc3.weight"]])
def test_refuses_other_head_keys(tmp_path, keys):
    d = tmp_path / "src"
    d.mkdir()
    g = torch.Generator().manual_seed(1)
    ts = {"model.a.weight": torch.randn(4, 4, generator=g).bfloat16()}
    ts.update({k: torch.randn(3, generator=g).bfloat16() for k in keys})
    save_file(ts, str(d / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (d / S.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors"
                                                                         for k in ts}}))
    with pytest.raises(S.StripError, match="expected exactly"):
        S.make(d, tmp_path / "dst")


def test_never_overwrites(tmp_path):
    fake_copy(tmp_path / "src")
    (tmp_path / "dst").mkdir()
    with pytest.raises(S.StripError, match="never overwrites"):
        S.make(tmp_path / "src", tmp_path / "dst")
    (tmp_path / "dst").rmdir()
    (tmp_path / "dst.partial").mkdir()
    with pytest.raises(S.StripError, match="never overwrites"):
        S.make(tmp_path / "src", tmp_path / "dst")


def test_refuses_a_stray_weight_file(tmp_path):
    fake_copy(tmp_path / "src")
    save_file({"x": torch.zeros(2)}, str(tmp_path / "src" / "extra.safetensors"))
    with pytest.raises(S.StripError, match="weight files"):
        S.make(tmp_path / "src", tmp_path / "dst")


def test_refuses_an_index_that_disagrees_with_the_headers(tmp_path):
    fake_copy(tmp_path / "src")
    idx = json.loads((tmp_path / "src" / S.INDEX).read_text())
    idx["weight_map"]["model.a.weight"] = "model-00002-of-00003.safetensors"
    (tmp_path / "src" / S.INDEX).write_text(json.dumps(idx))
    with pytest.raises(S.StripError, match="header keys differ"):
        S.make(tmp_path / "src", tmp_path / "dst")


# ------------------------------------------------------------------------------------------------ check
def flip_a_byte(p: Path, key: str) -> None:
    start, _, ts = S.read_header(p)
    off = start + ts[key]["data_offsets"][0]
    b = bytearray(p.read_bytes())
    b[off] ^= 0x01
    p.write_bytes(bytes(b))


def test_check_finds_a_changed_byte(tmp_path):
    fake_copy(tmp_path / "src")
    S.make(tmp_path / "src", tmp_path / "dst")
    flip_a_byte(tmp_path / "dst" / "model-00003-of-00003.safetensors", "model.e.weight")
    r = S.check(tmp_path / "src", tmp_path / "dst")
    assert not r["ok"] and any("model.e.weight: bytes differ" in p for p in r["problems"])


def test_check_finds_a_wrong_symlink_and_a_head_key(tmp_path):
    fake_copy(tmp_path / "src")
    fake_copy(tmp_path / "other")
    S.make(tmp_path / "src", tmp_path / "dst")
    f = "model-00001-of-00003.safetensors"
    os.remove(tmp_path / "dst" / f)
    os.symlink(tmp_path / "other" / f, tmp_path / "dst" / f)
    r = S.check(tmp_path / "src", tmp_path / "dst")
    assert not r["ok"] and any(p.startswith(f"{f}: symlink to") for p in r["problems"])
    os.remove(tmp_path / "dst" / f)
    os.symlink((tmp_path / "src" / f).resolve(), tmp_path / "dst" / f)
    assert S.check(tmp_path / "src", tmp_path / "dst")["ok"]
    idx = json.loads((tmp_path / "dst" / S.INDEX).read_text())
    idx["weight_map"]["nfp_head.fc1.bias"] = f
    (tmp_path / "dst" / S.INDEX).write_text(json.dumps(idx))
    r = S.check(tmp_path / "src", tmp_path / "dst")
    assert not r["ok"] and any(p.startswith("keys: 1 extra") for p in r["problems"])


def test_a_failed_check_leaves_the_partial_folder(tmp_path, monkeypatch):
    fake_copy(tmp_path / "src")
    monkeypatch.setattr(S, "check", lambda src, dst: {"ok": False, "problems": ["injected"]})
    with pytest.raises(S.StripError, match="check failed"):
        S.make(tmp_path / "src", tmp_path / "dst")
    assert not (tmp_path / "dst").exists() and (tmp_path / "dst.partial" / S.MANIFEST).is_file()


# ------------------------------------------------------------------------------------------------ real classes
@needs_models
def test_hf_loads_the_stripped_copy_with_no_stray_keys(tmp_path, cpu_gdn):
    """A tiny qwen3_5 with lf_nfp's head attached, saved by save_pretrained in shards: the copy with the head has
    unexpected nfp_head keys (vLLM refuses them), the stripped one loads with none missing or unexpected."""
    from transformers import Qwen3_5ForConditionalGeneration

    from vagen.q38s4.nfp import attach_head
    from vagen.q38s4.tests.tiny import build

    m = build(seed=0, untied=True)
    attach_head(m)
    m.save_pretrained(tmp_path / "src", max_shard_size="200KB", safe_serialization=True)
    idx = json.loads((tmp_path / "src" / S.INDEX).read_text())
    assert set(S.HEAD_KEYS) <= set(idx["weight_map"])
    _, info = Qwen3_5ForConditionalGeneration.from_pretrained(tmp_path / "src", output_loading_info=True)
    assert sorted(info["unexpected_keys"]) == sorted(S.HEAD_KEYS)
    S.make(tmp_path / "src", tmp_path / "dst")
    m2, info2 = Qwen3_5ForConditionalGeneration.from_pretrained(tmp_path / "dst", output_loading_info=True)
    assert not info2["missing_keys"] and not info2["unexpected_keys"] and not info2["mismatched_keys"]
    want = {k: v for k, v in m.state_dict().items() if not k.startswith("nfp_head")}
    got = m2.state_dict()
    assert set(got) == set(want)
    assert all(torch.equal(got[k], want[k]) for k in want)


@pytest.mark.skipif(not (SMOKE_COPY / S.INDEX).is_file(), reason="smoke bf16 copy not readable")
def test_plan_on_the_smoke_copy():
    """The real 27B layout: 11 shards, the four head tensors (bf16) at the end of the last one."""
    p = S.source_plan(SMOKE_COPY)
    assert p["head_keys"] == sorted(S.HEAD_KEYS) and p["tensors"] == 1188
    assert [f for f, w in p["shards"].items() if w != "link"] == ["model-00011-of-00011.safetensors"]
    assert p["shards"]["model-00011-of-00011.safetensors"] == "rewrite"
    assert p["head_bytes"] == 2 * (5120 * 5120 + 5120 + 1152 * 5120 + 1152)
    assert {"config.json", "tokenizer.json", "chat_template.jinja", "processor_config.json"} <= set(p["side_files"])


def test_command_line(tmp_path):
    fake_copy(tmp_path / "src")
    env = dict(os.environ, PYTHONPATH=str(CODE), OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    run = lambda *a: subprocess.run([sys.executable, "-B", "-m", S.MODULE, *map(str, a)], env=env,
                                    capture_output=True, text=True, timeout=300)
    r = run("plan", tmp_path / "src")
    assert r.returncode == 0 and json.loads(r.stdout)["ok"], r.stderr[-2000:]
    r = run("make", tmp_path / "src", tmp_path / "dst")
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert out["ok"] and out["rewritten"] == ["model-00003-of-00003.safetensors"] and out["check"]["ok"]
    r = run("check", tmp_path / "src", tmp_path / "dst")
    assert r.returncode == 0 and json.loads(r.stdout)["ok"]
    r = run("make", tmp_path / "src", tmp_path / "dst")
    assert r.returncode == 1 and "never overwrites" in json.loads(r.stdout)["error"]
