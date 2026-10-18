"""Model adapters: one code name -> one read-only "view" directory that every backend loads the same way.

    python -m eval_external.adapters.models prepare base-7B ours-RL-150      # build views and weight hashes
    python -m eval_external.adapters.models show ours-RL-150                  # what a manifest records

A view is ROOT/model_views/<code>/:
- symlinks to the checkpoint's weight, tokenizer and processor files (the checkpoint is never written to);
- config.json with dtype bfloat16 (our RL checkpoints are stored in float32);
- generation_config.json reduced to greedy decoding (do_sample false, repetition_penalty 1.0), so that every model
  decodes the same way whatever a backend leaves unset (Qwen's own file says repetition_penalty 1.05).
The weight hash is the sha256 of the sorted "file sha256" lines of the *.safetensors files; it is cached in
ROOT/model_hashes/<code>.json and recomputed when a file's size or mtime changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
REGISTRY = HERE / "registry.yaml"
ASSETS = EVAL_ROOT / "assets.yaml"
FAMILY_ARCH = {"qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration"}
VIEW_DTYPE = "bfloat16"
GREEDY_KEYS = ("bos_token_id", "eos_token_id", "pad_token_id")
TRAINING_STATE = ("optimizer", "scheduler", "rng_state", "trainer_state", "training_args")  # not linked into views


def load_registry(path: Path = REGISTRY) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def entry(code: str, reg: dict | None = None) -> dict:
    reg = reg or load_registry()
    if code not in reg["models"]:
        raise KeyError(f"unknown model code {code!r}; known: {list(reg['models'])}")
    return reg["models"][code]


def source_dir(code: str, reg: dict | None = None) -> Path:
    reg = reg or load_registry()
    e = entry(code, reg)
    if e["source"] == "hf_cache":
        name = "models--" + e["repo"].replace("/", "--")
        return Path(reg["hf_home"]) / "hub" / name / "snapshots" / e["revision"]
    if e["source"] == "local":
        return Path(reg["vagen_exps"]) / e["path"]
    if e["source"] == "asset":
        with open(ASSETS) as f:
            assets = yaml.safe_load(f)
        return Path(assets["root"]) / assets["models"][e["asset"]]["dir"]
    raise ValueError(f"unknown source {e['source']!r} for {code}")


def origin(code: str, reg: dict | None = None) -> str:
    """repo@revision for downloaded models, the path for ours."""
    reg = reg or load_registry()
    e = entry(code, reg)
    if e["source"] == "hf_cache":
        return f"{e['repo']}@{e['revision']}"
    if e["source"] == "asset":
        with open(ASSETS) as f:
            a = yaml.safe_load(f)["models"][e["asset"]]
        return f"{a['repo']}@{a['revision']}"
    return str(source_dir(code, reg))


def view_dir(code: str, reg: dict | None = None) -> Path:
    reg = reg or load_registry()
    return Path(reg["root"]) / "model_views" / code


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def build_view(code: str, reg: dict | None = None) -> Path:
    reg = reg or load_registry()
    e = entry(code, reg)
    src = source_dir(code, reg)
    if not (src / "config.json").exists():
        raise FileNotFoundError(f"{code}: no config.json in {src} (not downloaded yet?)")
    cfg = json.loads((src / "config.json").read_text())
    arch = (cfg.get("architectures") or [None])[0]
    if arch != FAMILY_ARCH[e["family"]]:
        raise ValueError(f"{code}: architecture {arch} is not {FAMILY_ARCH[e['family']]}")
    view = view_dir(code, reg)
    view.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.name in ("config.json", "generation_config.json") or p.is_dir() or p.name.startswith("."):
            continue
        if p.name.startswith(TRAINING_STATE):
            continue
        link = view / p.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p.resolve())
    cfg["torch_dtype"] = VIEW_DTYPE
    cfg["dtype"] = VIEW_DTYPE
    _write_json(view / "config.json", cfg)
    gen = {}
    if (src / "generation_config.json").exists():
        old = json.loads((src / "generation_config.json").read_text())
        gen = {k: old[k] for k in GREEDY_KEYS if k in old}
    gen.update({"do_sample": False, "repetition_penalty": 1.0})
    _write_json(view / "generation_config.json", gen)
    return view


def _sha256(path: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def weight_hash(code: str, reg: dict | None = None) -> dict:
    reg = reg or load_registry()
    src = source_dir(code, reg)
    cache = Path(reg["root"]) / "model_hashes" / f"{code}.json"
    files = sorted(p for p in src.iterdir() if p.name.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"{code}: no *.safetensors in {src}")
    stamp = {p.name: [p.stat().st_size, int(p.stat().st_mtime)] for p in files}
    if cache.exists():
        old = json.loads(cache.read_text())
        if old.get("source") == str(src) and old.get("stamp") == stamp:
            return old
    per_file = {p.name: _sha256(p.resolve()) for p in files}
    combined = hashlib.sha256("".join(f"{n} {h}\n" for n, h in sorted(per_file.items())).encode()).hexdigest()
    out = {"code": code, "source": str(src), "stamp": stamp, "files": per_file, "sha256": combined}
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cache, out)
    return out


def chat_template_sha(path: Path) -> str | None:
    if (path / "chat_template.jinja").exists():
        text = (path / "chat_template.jinja").read_text()
    elif (path / "chat_template.json").exists():
        text = json.loads((path / "chat_template.json").read_text())["chat_template"]
    elif (path / "tokenizer_config.json").exists():
        text = json.loads((path / "tokenizer_config.json").read_text()).get("chat_template")
    else:
        text = None
    return hashlib.sha256(text.encode()).hexdigest() if text else None


def describe(code: str, reg: dict | None = None, with_hash: bool = True) -> dict:
    """What a run manifest records about a model."""
    reg = reg or load_registry()
    e = entry(code, reg)
    view = view_dir(code, reg)
    out = {
        "code": code,
        "origin": origin(code, reg),
        "source_dir": str(source_dir(code, reg)),
        "view_dir": str(view),
        "family": e["family"],
        "params": e["params"],
        "view_dtype": VIEW_DTYPE,
        "chat_template_sha256": chat_template_sha(view if view.exists() else source_dir(code, reg)),
        "note": e.get("note"),
    }
    if with_hash:
        out["weights_sha256"] = weight_hash(code, reg)["sha256"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("prepare", "show"))
    ap.add_argument("codes", nargs="+")
    args = ap.parse_args()
    reg = load_registry()
    for code in args.codes:
        if args.action == "prepare":
            build_view(code, reg)
        print(json.dumps(describe(code, reg), indent=1))


if __name__ == "__main__":
    main()
