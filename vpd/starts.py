"""The VPD start table (vpd/starts.json) and the one way every VPD process loads a start.

    resolve("Q3S")            -> {"name", "code", "path", "family", "note"}
    processor_dir("Q3S")      -> the folder the processor/tokenizer is read from
    load_processor("Q3S")     -> AutoProcessor with the shared max_pixels cap (DECISIONS B16)
    load_model("Q3S", device="cuda", attn="sdpa")  -> (model, processor), bf16, attention asserted (B10)

starts.json maps NAME -> {code, path, family, note}.  code is the eval_external registry code (its view is
eval_external_assets/model_views/<code>), null when the start has none; path is the checkpoint folder, read by path
only.  The codes and paths were checked against eval_external/adapters/registry_matrix_v3.yaml at bf4b2553.

torch/transformers are imported inside the loaders only, so the json helpers run in any interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vpd import paths

ATTN_ALIASES = {"fa2": "flash_attention_2", "flash_attention_2": "flash_attention_2", "sdpa": "sdpa",
                "eager": "eager"}
PROCESSOR_MARKER = "preprocessor_config.json"


def load_table(path: Path | str | None = None) -> dict:
    with open(path or Path(__file__).with_name("starts.json")) as f:
        table = json.load(f)
    for name, e in table.items():
        missing = {"code", "path", "family", "note"} - set(e)
        if missing:
            raise ValueError(f"starts.json: {name} lacks {sorted(missing)}")
    return table


def names() -> list[str]:
    return list(load_table())


def resolve(name: str) -> dict:
    table = load_table()
    if name not in table:
        raise KeyError(f"unknown start {name!r}; known: {list(table)}")
    return {"name": name, **table[name]}


def view_dir(name: str) -> Path | None:
    """The eval view of the start's code (same folder as eval_external.adapters.models.view_dir), or None."""
    code = resolve(name)["code"]
    return None if code is None else paths.MODEL_VIEWS / code


def processor_dir(name: str) -> Path:
    """Where the processor and tokenizer are read from: the start's eval view when it exists and holds processor
    files (what vLLM serves, so train and eval read the same files), else the checkpoint folder.  Uses
    os.path.lexists so that a root-only blob behind a symlink still counts as present."""
    v = view_dir(name)
    if v is not None and os.path.lexists(v / PROCESSOR_MARKER):
        return v
    return Path(resolve(name)["path"])


def norm_attn(attn: str) -> str:
    if attn not in ATTN_ALIASES:
        raise ValueError(f"attn must be one of {sorted(ATTN_ALIASES)}, got {attn!r}")
    return ATTN_ALIASES[attn]


def load_processor(name: str | None = None, path: Path | str | None = None,
                   max_pixels: int | None = paths.MAX_PIXELS):
    """AutoProcessor from `path` if it holds processor files, else from processor_dir(name).  max_pixels (default
    paths.MAX_PIXELS, the value vLLM gets via --mm-processor-kwargs) caps the image size; None keeps the files'
    own limits.  The chosen folder and cap are stored on the processor as vpd_processor_dir / vpd_max_pixels."""
    from transformers import AutoProcessor

    if path is not None and os.path.lexists(Path(path) / PROCESSOR_MARKER):
        pdir = Path(path)
    elif name is not None:
        pdir = processor_dir(name)
    else:
        raise ValueError("load_processor needs a start name or a path with processor files")
    proc = AutoProcessor.from_pretrained(str(pdir))
    if max_pixels is not None:
        set_max_pixels(proc, int(max_pixels))
    proc.vpd_processor_dir = str(pdir)
    proc.vpd_max_pixels = max_pixels
    return proc


def set_max_pixels(proc, max_pixels: int) -> None:
    """Cap the image processor at max_pixels.  AutoProcessor.from_pretrained(dir, max_pixels=X) is NOT enough:
    with transformers 4.57.0 (starVLA) it only sets an unused attribute, size.longest_edge stays 16,777,216 and a
    2000x1200 image is not downscaled (probe 2026-11-08).  smart_resize reads size['longest_edge'], so set that."""
    ip = getattr(proc, "image_processor", proc)
    size = dict(getattr(ip, "size", None) or {})
    if "longest_edge" not in size:
        raise RuntimeError(f"{type(ip).__name__} has no size.longest_edge: {size}")
    size["longest_edge"] = int(max_pixels)
    ip.size = size
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = int(max_pixels)
    got = processor_max_pixels(proc)
    if got != int(max_pixels):
        raise RuntimeError(f"processor max_pixels is {got}, asked {max_pixels}")


def processor_max_pixels(proc) -> int | None:
    """The image processor's effective pixel cap (size.longest_edge, or max_pixels on older processors)."""
    ip = getattr(proc, "image_processor", proc)
    size = getattr(ip, "size", None)
    if isinstance(size, dict) and size.get("longest_edge") is not None:
        return int(size["longest_edge"])
    mp = getattr(ip, "max_pixels", None)
    return None if mp is None else int(mp)


def _get(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# Where an attention implementation lives on a Qwen3-VL model (transformers 4.57).  The verl RL exports write
# "attn_implementation": "eager" at the top level and float32 in text_config / vision_config, so the top-level
# check alone is not enough (DECISIONS B10).
ATTN_SITES = ("config", "config.text_config", "config.vision_config",
              "model.language_model.config", "model.visual.config")
REQUIRED_SITES = ("config", "model.language_model.config", "model.visual.config")


def attn_report(model) -> dict:
    """{site: _attn_implementation} for every site that exists on the model."""
    out = {}
    for site in ATTN_SITES:
        cfg = _get(model, site)
        if cfg is not None:
            out[site] = getattr(cfg, "_attn_implementation", None)
    return out


def check_attn(model, attn: str) -> dict:
    """Assert that the model, its language model and its vision tower all run `attn`; return the report."""
    attn = norm_attn(attn)
    rep = attn_report(model)
    missing = [s for s in REQUIRED_SITES if s not in rep]
    if missing:
        raise AssertionError(f"attention sites not found on {type(model).__name__}: {missing}")
    bad = {s: v for s, v in rep.items() if v != attn}
    if bad:
        raise AssertionError(f"attention is not {attn!r} everywhere: {bad}")
    return rep


def check_dtype(model, dtype) -> dict:
    """Assert every floating parameter has `dtype`; return {dtype name: parameter count}."""
    counts: dict = {}
    for p in model.parameters():
        if p.is_floating_point():
            counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + p.numel()
    if set(counts) != {str(dtype)}:
        raise AssertionError(f"parameters not all {dtype}: {counts}")
    return counts


def load_model(name: str, device: str | None = "cuda", attn: str = "sdpa", path: Path | str | None = None,
               max_pixels: int | None = paths.MAX_PIXELS, log=print):
    """(model, processor) of a start in bfloat16 with attention `attn` ('sdpa' | 'fa2' | 'eager').

    path overrides the weights folder (e.g. a merged copy); the processor then comes from that folder if it holds
    processor files, else from processor_dir(name).  Attention is asserted on model.config, the language model
    and the vision tower; dtype on every floating parameter.  model.vpd_load_info records what was loaded."""
    import torch
    from transformers import AutoModelForImageTextToText

    e = resolve(name)
    wdir = Path(path) if path is not None else Path(e["path"])
    impl = norm_attn(attn)
    model = AutoModelForImageTextToText.from_pretrained(str(wdir), torch_dtype=torch.bfloat16,
                                                        attn_implementation=impl)
    if device is not None:
        model = model.to(device)
    rep = check_attn(model, impl)
    dtypes = check_dtype(model, torch.bfloat16)
    proc = load_processor(name, path=path, max_pixels=max_pixels)
    info = {"start": name, "code": e["code"], "path": str(wdir), "attn": impl, "attn_sites": rep,
            "dtype": "bfloat16", "params": sum(dtypes.values()), "processor_dir": proc.vpd_processor_dir,
            "max_pixels": max_pixels, "class": type(model).__name__}
    model.vpd_load_info = info
    if log is not None:
        log(f"[vpd.starts] loaded {name} from {wdir}: attn={impl} at {sorted(rep)}; processor {proc.vpd_processor_dir}")
    return model, proc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Show the VPD start table.")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args(argv)
    for n in args.names or names():
        e = resolve(n)
        v = view_dir(n)
        print(json.dumps({**e, "processor_dir": str(processor_dir(n)), "view": None if v is None else str(v),
                          "path_is_dir": os.path.isdir(e["path"])}))


if __name__ == "__main__":
    main()
