"""CPU tests of the pure helpers of the step-1 load test and of the start registry."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "load_test"))
import load_in_starvla as lis  # noqa: E402


@pytest.mark.parametrize("key,expected", [
    ("model.layers.0.mlp.down_proj.weight", "LM.layers.0.mlp.down_proj.weight"),
    ("model.language_model.layers.0.mlp.down_proj.weight", "LM.layers.0.mlp.down_proj.weight"),
    ("model.embed_tokens.weight", "LM.embed_tokens.weight"),
    ("model.language_model.embed_tokens.weight", "LM.embed_tokens.weight"),
    ("visual.blocks.0.attn.qkv.weight", "VIS.blocks.0.attn.qkv.weight"),
    ("model.visual.blocks.0.attn.qkv.weight", "VIS.blocks.0.attn.qkv.weight"),
    ("qwen_vl_interface.model.model.visual.merger.mlp.0.weight", "VIS.merger.mlp.0.weight"),
    ("lm_head.weight", "lm_head.weight"),
])
def test_norm_key_maps_both_layouts_to_one_name(key, expected):
    assert lis.norm_key(key) == expected


def test_fake_examples_shapes():
    ex = lis.fake_examples(2, 16, 12, seed=0)
    assert len(ex) == 2 and len(ex[0]["image"]) == 3
    assert ex[0]["image"][0].size == (224, 224)
    assert ex[0]["action"].shape == (16, 12)


def test_start_registry():
    starts = json.loads((ROOT / "starts.json").read_text())
    main = {k: v for k, v in starts.items() if not k.startswith("_") and v["role"] == "main"}
    assert sorted(main) == ["B", "RD", "RS", "S"]
    for name, entry in starts.items():
        if name.startswith("_"):
            continue
        assert entry["role"] in {"main", "spare"} and entry["path"].startswith("/mnt/umm/")
