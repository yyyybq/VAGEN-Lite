"""Unit tests for the parts every eval_external table depends on: the Wilson interval, model views, manifests.

    python -m pytest -q eval_external/tests
"""

import json

from eval_external.adapters import models as M
from eval_external.common.manifest import write_manifest
from eval_external.common.stats import fmt_pct_ci, wilson


def test_wilson_known_values():
    p, lo, hi = wilson(1, 20)
    assert abs(p - 0.05) < 1e-12
    assert abs(lo - 0.0089) < 5e-4 and abs(hi - 0.2361) < 5e-4
    p, lo, hi = wilson(82, 150)  # SAT-Real at ~54.7%
    assert lo < p < hi and abs((hi - lo) / 2 - 0.079) < 0.002
    assert wilson(0, 30)[1] == 0.0 and wilson(30, 30)[2] == 1.0
    assert fmt_pct_ci(1, 20).startswith("5.0 [0.9, 23.6]")


def _fake_checkpoint(tmp_path, dtype="float32"):
    ckpt = tmp_path / "exps" / "run" / "ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["Qwen2_5_VLForConditionalGeneration"],
                                                  "dtype": dtype}))
    (ckpt / "generation_config.json").write_text(json.dumps({"do_sample": True, "repetition_penalty": 1.05,
                                                             "temperature": 1e-6, "eos_token_id": [1, 2],
                                                             "pad_token_id": 2}))
    (ckpt / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (ckpt / "tokenizer.json").write_text("{}")
    (ckpt / "chat_template.jinja").write_text("{{ messages }}")
    (ckpt / "optimizer.pt").write_bytes(b"x")
    reg = {"hf_home": str(tmp_path / "hf"), "root": str(tmp_path / "root"), "vagen_exps": str(tmp_path / "exps"),
           "models": {"ours-X": {"source": "local", "path": "run/ckpt", "family": "qwen2_5_vl", "params": "7B"}}}
    return ckpt, reg


def test_view_is_bf16_greedy_and_leaves_checkpoint_alone(tmp_path):
    ckpt, reg = _fake_checkpoint(tmp_path)
    before = {p.name: p.read_bytes() for p in ckpt.iterdir()}
    view = M.build_view("ours-X", reg)
    cfg = json.loads((view / "config.json").read_text())
    assert cfg["dtype"] == "bfloat16" and cfg["torch_dtype"] == "bfloat16"
    gen = json.loads((view / "generation_config.json").read_text())
    assert gen == {"eos_token_id": [1, 2], "pad_token_id": 2, "do_sample": False, "repetition_penalty": 1.0}
    assert (view / "model-00001-of-00001.safetensors").is_symlink()
    assert not (view / "optimizer.pt").exists()
    assert {p.name: p.read_bytes() for p in ckpt.iterdir()} == before
    d = M.describe("ours-X", reg)
    assert d["weights_sha256"] == M.weight_hash("ours-X", reg)["sha256"]
    assert d["chat_template_sha256"] is not None


def test_view_rejects_other_architectures(tmp_path):
    ckpt, reg = _fake_checkpoint(tmp_path)
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["LlavaForConditionalGeneration"]}))
    try:
        M.build_view("ours-X", reg)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_manifest_fields(tmp_path):
    path = write_manifest(tmp_path / "run", models=[{"code": "m"}], benchmark={"name": "b"}, backend={"name": "x"},
                          prompt={"post": "p"}, frames=32, decoding={"do_sample": False})
    m = json.loads(path.read_text())
    for key in ("utc", "models", "benchmark", "backend", "subset", "prompt", "frames", "decoding", "eval_external"):
        assert key in m
    assert m["frames"] == 32 and m["subset"]["kind"] == "full"
