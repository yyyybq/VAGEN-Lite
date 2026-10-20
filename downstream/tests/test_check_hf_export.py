"""CPU tests of downstream/load_test/check_hf_export.py on tiny fake exports (no torch, no transformers)."""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "load_test"))
import check_hf_export as chk  # noqa: E402

VOCAB, HIDDEN = 10, 4


def write_safetensors(path: Path, tensors: dict):
    """tensors: name -> (dtype, shape); writes a valid header and zero bytes for the data."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = chk.numel(shape) * chk.DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def make_export(root: Path, dtype="BF16", scheme="old", drop=None, lora=False):
    root.mkdir(parents=True)
    lm = "model." if scheme == "old" else "model.language_model."
    vis = "visual." if scheme == "old" else "model.visual."
    shard1 = {lm + "embed_tokens.weight": (dtype, [VOCAB, HIDDEN]), vis + "patch_embed.proj.weight": (dtype, [4, 3])}
    shard2 = {lm + "layers.0.mlp.down_proj.weight": (dtype, [HIDDEN, HIDDEN]), "lm_head.weight": (dtype, [VOCAB, HIDDEN])}
    write_safetensors(root / "model-00001-of-00002.safetensors", shard1)
    write_safetensors(root / "model-00002-of-00002.safetensors", shard2)
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen2_5_vl", "torch_dtype": "bfloat16" if dtype == "BF16" else "float32",
        "hidden_size": HIDDEN, "vocab_size": VOCAB, "num_hidden_layers": 1, "tie_word_embeddings": False,
        "vision_config": {"depth": 1, "out_hidden_size": HIDDEN}}))
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json"):
        (root / name).write_text("{}")
    if lora:
        (root / "adapter_config.json").write_text("{}")
    if drop:
        (root / drop).unlink()
    return root


def test_complete_export_counts_params_by_dtype(tmp_path):
    r = chk.inspect_dir(make_export(tmp_path / "w"), tokenizer_check=False)
    assert r["problems"] == [] and r["format"] == "complete HF export"
    assert r["shards"] == 2 and r["tensors"] == 4
    assert r["params_total"] == VOCAB * HIDDEN * 2 + 12 + HIDDEN * HIDDEN
    assert r["per_dtype"]["BF16"]["bytes"] == r["params_total"] * 2
    assert r["key_scheme"].startswith("old")
    assert r["params_by_group_and_dtype"]["vision"]["BF16"] == 12
    assert r["config"]["torch_dtype"] == "bfloat16"


def test_new_key_scheme_and_fp32(tmp_path):
    r = chk.inspect_dir(make_export(tmp_path / "w", dtype="F32", scheme="new"), tokenizer_check=False)
    assert r["problems"] == []
    assert r["key_scheme"].startswith("new")
    assert set(r["per_dtype"]) == {"F32"}
    assert r["params_by_group_and_dtype"]["language"]["F32"] == VOCAB * HIDDEN + HIDDEN * HIDDEN


def test_missing_shard_and_processor_are_problems(tmp_path):
    r = chk.inspect_dir(make_export(tmp_path / "a", drop="model-00002-of-00002.safetensors"), tokenizer_check=False)
    assert any("shards missing" in p for p in r["problems"])
    r = chk.inspect_dir(make_export(tmp_path / "b", drop="preprocessor_config.json"), tokenizer_check=False)
    assert any("preprocessor_config" in p for p in r["problems"])


def test_lora_and_config_only(tmp_path):
    r = chk.inspect_dir(make_export(tmp_path / "a", lora=True), tokenizer_check=False)
    assert r["format"] == "LoRA adapter"
    cfg_only = tmp_path / "b"
    cfg_only.mkdir()
    (cfg_only / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    r = chk.inspect_dir(cfg_only, tokenizer_check=False)
    assert r["format"].startswith("config only")


def test_embedding_shape_mismatch(tmp_path):
    root = make_export(tmp_path / "w")
    cfg = json.loads((root / "config.json").read_text())
    cfg["vocab_size"] = VOCAB + 1
    (root / "config.json").write_text(json.dumps(cfg))
    r = chk.inspect_dir(root, tokenizer_check=False)
    assert any("embed_tokens.weight shape" in p for p in r["problems"])


def test_main_writes_report(tmp_path):
    root = make_export(tmp_path / "w")
    out = tmp_path / "r.json"
    assert chk.main(["--start", f"B={root}", "--out", str(out), "--no-tokenizer"]) == 0
    assert json.loads(out.read_text())["B"]["format"] == "complete HF export"
