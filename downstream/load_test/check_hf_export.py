#!/usr/bin/env python3
"""Is each start weight a complete Hugging Face export that StarVLA can load? (CPU only, reads headers only.)

    python downstream/load_test/check_hf_export.py --start B=/path --start RS=/path ... --out report.json

Per weight directory it records, without reading tensor data:
  * the files a Qwen2.5-VL HF export needs (config, weight index and every shard it names, tokenizer, image
    processor, chat template) and whether the directory is instead a LoRA adapter or an FSDP / DCP shard set;
  * from the safetensors headers: tensor count, parameter count and bytes per dtype, the key naming scheme,
    the embedding / lm_head shapes against the config's vocab and hidden size;
  * from config.json: model_type, torch_dtype (what ``from_pretrained(torch_dtype="auto")`` loads as), sizes;
  * if ``transformers`` is importable: whether the QwenOFT action query "🔍" is one token (16 in a row -> 16).

Exit code 0 even when a weight fails a check; the report says which check failed.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1}
TOKENIZER_SETS = (("tokenizer.json",), ("vocab.json", "merges.txt"))
FSDP_PATTERNS = ("model_world_size_*_rank_*.pt", "*.distcp", "__*_*.distcp")


def read_safetensors_header(path: Path) -> dict:
    """The JSON header of a .safetensors file (8-byte little-endian length, then JSON); no tensor data."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length <= 0 or length > 100_000_000:
            raise ValueError(f"implausible safetensors header length {length} in {path}")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def key_group(key: str) -> str:
    """vision / language / lm_head / other, for both the old (model.layers.*, visual.*) and new
    (model.language_model.*, model.visual.*) Qwen2.5-VL key layouts."""
    if key.startswith(("visual.", "model.visual.")):
        return "vision"
    if key.startswith("lm_head."):
        return "lm_head"
    if key.startswith(("model.language_model.", "model.layers.", "model.embed_tokens.", "model.norm.")):
        return "language"
    return "other"


def key_scheme(keys) -> str:
    if any(k.startswith("model.language_model.") for k in keys):
        return "new (model.language_model.* / model.visual.*)"
    if any(k.startswith("model.layers.") for k in keys):
        return "old (model.layers.* / visual.*)"
    return "unknown"


def inspect_dir(root: Path, tokenizer_check: bool = True) -> dict:
    out = {"path": str(root), "exists": root.is_dir(), "checks": {}, "problems": []}
    if not root.is_dir():
        out["problems"].append("directory missing")
        return out
    files = sorted(p.name for p in root.iterdir() if p.is_file())
    out["files"] = files
    if (root / "adapter_config.json").exists() or any(f.startswith("adapter_model") for f in files):
        out["format"] = "LoRA adapter"
        out["problems"].append("LoRA adapter only: needs merging into its base (a new full-size copy)")
    fsdp = sorted({str(p.relative_to(root)) for pat in FSDP_PATTERNS for p in root.glob(pat)})
    if fsdp:
        out["fsdp_or_dcp_files"] = len(fsdp)
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        out["problems"].append("config.json missing")
        out.setdefault("format", "not an HF export (no config.json)")
        return out
    cfg = json.loads(cfg_path.read_text())
    text = cfg.get("text_config", {}) if isinstance(cfg.get("text_config"), dict) else {}
    vision = cfg.get("vision_config", {}) if isinstance(cfg.get("vision_config"), dict) else {}

    def pick(name):
        return text.get(name, cfg.get(name))

    out["config"] = {
        "model_type": cfg.get("model_type"), "architectures": cfg.get("architectures"),
        "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype") or text.get("torch_dtype") or text.get("dtype"),
        "transformers_version": cfg.get("transformers_version"), "hidden_size": pick("hidden_size"),
        "num_hidden_layers": pick("num_hidden_layers"), "vocab_size": pick("vocab_size"),
        "tie_word_embeddings": pick("tie_word_embeddings"), "vision_depth": vision.get("depth"),
        "vision_out_hidden_size": vision.get("out_hidden_size"),
    }
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif (root / "model.safetensors").is_file():
        weight_map, shards = None, ["model.safetensors"]
    else:
        out["problems"].append("no model.safetensors(.index.json)")
        out.setdefault("format", "config only (weights missing)")
        return out
    out["shards"] = len(shards)
    per_dtype = defaultdict(lambda: {"tensors": 0, "params": 0, "bytes": 0})
    per_group = defaultdict(lambda: defaultdict(int))
    keys, shapes, missing_shards, bad_index = [], {}, [], 0
    for name in shards:
        shard = root / name
        if not shard.is_file() or shard.stat().st_size == 0:
            missing_shards.append(name)
            continue
        header = read_safetensors_header(shard)
        if weight_map is not None:
            wanted = {k for k, v in weight_map.items() if v == name}
            bad_index += len(wanted - set(header))
        for key, meta in header.items():
            n = numel(meta["shape"])
            dtype = meta["dtype"]
            per_dtype[dtype]["tensors"] += 1
            per_dtype[dtype]["params"] += n
            per_dtype[dtype]["bytes"] += n * DTYPE_BYTES.get(dtype, 0)
            per_group[key_group(key)][dtype] += n
            keys.append(key)
            if key.endswith("embed_tokens.weight") or key == "lm_head.weight":
                shapes[key] = meta["shape"]
    if missing_shards:
        out["problems"].append(f"shards missing or empty: {missing_shards}")
    if bad_index:
        out["problems"].append(f"{bad_index} index keys not found in their shard headers")
    out["tensors"] = len(keys)
    out["params_total"] = sum(v["params"] for v in per_dtype.values())
    out["bytes_total"] = sum(v["bytes"] for v in per_dtype.values())
    out["per_dtype"] = {k: dict(v) for k, v in sorted(per_dtype.items())}
    out["params_by_group_and_dtype"] = {g: dict(v) for g, v in sorted(per_group.items())}
    out["key_scheme"] = key_scheme(keys)
    out["embedding_shapes"] = shapes
    vocab, hidden = out["config"]["vocab_size"], out["config"]["hidden_size"]
    for key, shape in shapes.items():
        if [vocab, hidden] != list(shape):
            out["problems"].append(f"{key} shape {shape} != [vocab_size {vocab}, hidden_size {hidden}]")
    if "lm_head.weight" not in shapes and not out["config"]["tie_word_embeddings"]:
        out["problems"].append("lm_head.weight missing while tie_word_embeddings is false")
    tok = next((s for s in TOKENIZER_SETS if all((root / f).is_file() for f in s)), None)
    out["checks"]["tokenizer_files"] = list(tok) if tok else None
    if not tok:
        out["problems"].append("tokenizer files missing")
    out["checks"]["tokenizer_config"] = (root / "tokenizer_config.json").is_file()
    out["checks"]["image_processor"] = (root / "preprocessor_config.json").is_file()
    if not out["checks"]["image_processor"]:
        out["problems"].append("preprocessor_config.json missing")
    tcfg = json.loads((root / "tokenizer_config.json").read_text()) if out["checks"]["tokenizer_config"] else {}
    out["checks"]["chat_template"] = ("chat_template.json" if (root / "chat_template.json").is_file() else
                                      "chat_template.jinja" if (root / "chat_template.jinja").is_file() else
                                      "tokenizer_config.json" if "chat_template" in tcfg else None)
    if tokenizer_check:
        out["action_token"] = action_token_check(root)
        if out["action_token"].get("single_token") is False:
            out["problems"].append("the action query 🔍 is not one token")
    out.setdefault("format", "complete HF export" if not out["problems"] else "HF export with problems")
    return out


def action_token_check(root: Path) -> dict:
    """QwenOFT puts 16 '🔍' into the prompt and reads their hidden states; each must be one token."""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # the check is optional on a machine without transformers
        return {"skipped": f"transformers not importable: {type(exc).__name__}"}
    try:
        tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
        one = tok.encode("🔍", add_special_tokens=False)
        many = tok.encode("🔍" * 16, add_special_tokens=False)
        return {"token_ids": one, "single_token": len(one) == 1 and many == one * 16, "len_tokenizer": len(tok)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def parse_start(value: str):
    name, sep, path = value.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError("use NAME=/absolute/path")
    return name, Path(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", action="append", type=parse_start, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the transformers tokenizer check")
    args = ap.parse_args(argv)
    report = {name: inspect_dir(path, tokenizer_check=not args.no_tokenizer) for name, path in args.start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for name, r in report.items():
        print(f"{name:6s} {r.get('format')}  params={r.get('params_total')}  dtypes={sorted(r.get('per_dtype', {}))}"
              f"  config_dtype={r.get('config', {}).get('torch_dtype')}  problems={r['problems']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
