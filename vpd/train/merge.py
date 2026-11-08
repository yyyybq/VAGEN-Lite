#!/usr/bin/env python3
"""Merge a VPD LoRA adapter into its start and save a servable HF folder on JOB-LOCAL disk (WP-TRAIN; plan 3.4;
DECISIONS B10, B11).  Runs inside sco jobs only, with PY_Q3 (transformers 4.57.6), as a subprocess before vLLM.

    PY_Q3 vpd/train/merge.py --start Q3S --adapter RUNS/pilot/Q3S/adapters/200 [--out LOCAL] [--device cpu]

- Loads vpd.starts.load_model(start, attn='sdpa') in bf16 (attention asserted on model / language model / vision
  tower) and adds alpha/rank * B A to the 252 wrapped projections (merge_adapter copied from n2_train.py); the merged
  count must equal the adapter's meta['lora']['wrapped'], and meta['start'] must be the start.
- save_pretrained(LOCAL, safe_serialization=True, max_shard_size='5GB'); then copies tokenizer*, vocab*, merges*,
  chat_template*, added_tokens.json, special_tokens_map.json, preprocessor_config.json and
  video_preprocessor_config.json from the start's eval view (or its checkpoint folder: vpd.starts.processor_dir);
  config.json is rewritten as eval_external.adapters.models.build_view does (dtype bfloat16 at the top AND in
  text_config / vision_config, no attn_implementation keys: B10); generation_config.json is greedy (bos/eos/pad
  from the start's file, do_sample false, repetition_penalty 1.0).
- LOCAL: --out, else $VPD_MERGE_DIR/<start>__<adapter name>, else /tmp/vpd_merged/... if /tmp has >= 40 GB free,
  else /dev/shm/vpd_merged/... if /dev/shm has >= 40 GB free, else it fails (B11).  The chosen target's free space
  is always checked.  LOCAL under /mnt/umm is refused: Q3B-derived weights would become a user-readable copy of
  root-only weights.  An existing non-empty LOCAL is refused (the payload removes merged dirs after each eval).
- Re-reads one merged tensor from the saved shards and compares it with memory; writes LOCAL/vpd_merge.json; the
  last stdout line is LOCAL.
Merging real weights is a stop-and-ask item on the login node (DECISIONS A3): there it is only tested on a tiny model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from vpd.train import n2_train_vpd as T  # noqa: E402

MIN_FREE_GB = 40.0
FORBIDDEN_ROOTS = ("/mnt/umm",)
FALLBACK_BASES = ("/tmp/vpd_merged", "/dev/shm/vpd_merged")
COPY_PATTERNS = ("tokenizer*", "vocab*", "merges*", "chat_template*", "added_tokens.json", "special_tokens_map.json",
                 "preprocessor_config.json", "video_preprocessor_config.json")
REQUIRED_COPIES = ("tokenizer_config.json", "preprocessor_config.json")
ATTN_KEYS = ("attn_implementation", "_attn_implementation")  # as eval_external/adapters/models.py
GREEDY_KEYS = ("bos_token_id", "eos_token_id", "pad_token_id")
VIEW_DTYPE = "bfloat16"


def free_gb(path) -> float:
    """Free space (GB, 1e9) of the filesystem holding path or its nearest existing parent."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def is_forbidden(path) -> bool:
    rp = os.path.realpath(str(path))
    return any(rp == r or rp.startswith(r.rstrip("/") + "/") for r in FORBIDDEN_ROOTS)


def choose_out(out, run_name: str, min_free_gb: float = MIN_FREE_GB, allow_shared: bool = False) -> tuple[Path, dict]:
    """(LOCAL, facts) per the module docstring; raises with every candidate's free space when none fits."""
    facts = {"min_free_gb": min_free_gb, "candidates": []}
    if out is not None:
        cands, how = [Path(out)], "--out"
    elif os.environ.get("VPD_MERGE_DIR"):
        cands, how = [Path(os.environ["VPD_MERGE_DIR"]) / run_name], "VPD_MERGE_DIR"
    else:
        cands, how = [Path(b) / run_name for b in FALLBACK_BASES], "auto"
    for c in cands:
        if is_forbidden(c) and not allow_shared:
            raise SystemExit(f"[merge] refused: {c} is under {FORBIDDEN_ROOTS} (merged weights stay on job-local disk)")
        g = free_gb(c)
        facts["candidates"].append({"path": str(c), "free_gb": round(g, 1)})
        if g >= min_free_gb:
            facts["chosen_by"] = how
            return c, facts
    raise SystemExit(f"[merge] no target with >= {min_free_gb} GB free: {facts['candidates']}")


def run_name_of(start: str, adapter: Path) -> str:
    parts = [p for p in Path(adapter).resolve().parts[-2:] if p not in ("/", "")]
    return "__".join([start] + parts).replace(" ", "_")


def view_config(cfg: dict) -> tuple[dict, dict]:
    """config.json as build_view writes it: dtype bfloat16 at every level, attention keys dropped."""
    changes = {}
    for prefix, holder in (("", cfg), ("text_config.", cfg.get("text_config")),
                           ("vision_config.", cfg.get("vision_config"))):
        if not isinstance(holder, dict):
            continue
        for key in ("torch_dtype", "dtype"):
            if (key in holder or key == "dtype") and holder.get(key) != VIEW_DTYPE:
                changes[prefix + key] = [holder.get(key), VIEW_DTYPE]
                holder[key] = VIEW_DTYPE
        for key in ATTN_KEYS:
            if key in holder:
                changes[prefix + key] = [holder.pop(key), None]
    return cfg, changes


def greedy_generation_config(src_dirs) -> dict:
    gen = {}
    for d in src_dirs:
        f = Path(d) / "generation_config.json"
        if os.path.lexists(f):
            old = json.loads(f.read_text())
            gen = {k: old[k] for k in GREEDY_KEYS if k in old}
            break
    gen.update({"do_sample": False, "repetition_penalty": 1.0})
    return gen


def copy_processor_files(src: Path, dst: Path) -> list:
    copied = []
    for pat in COPY_PATTERNS:
        for f in sorted(Path(src).glob(pat)):
            if f.is_dir():
                continue
            shutil.copyfile(f, dst / f.name)  # follows the view's symlinks
            copied.append(f.name)
    missing = [r for r in REQUIRED_COPIES if r not in copied]
    if missing:
        raise RuntimeError(f"{src}: processor/tokenizer files missing: {missing}")
    if not any(c.startswith("chat_template") for c in copied):
        cfg = json.loads((dst / "tokenizer_config.json").read_text())
        if not cfg.get("chat_template"):
            raise RuntimeError(f"{src}: no chat template (no chat_template* file and none in tokenizer_config.json)")
    return sorted(set(copied))


def check_saved(model, out: Path, name: str) -> float:
    """max |saved - memory| of one merged weight, read back from the safetensors shards."""
    from safetensors import safe_open

    key = name + ".weight"
    idx = out / "model.safetensors.index.json"
    shard = json.loads(idx.read_text())["weight_map"][key] if idx.is_file() else "model.safetensors"
    with safe_open(str(out / shard), framework="pt") as f:
        saved = f.get_tensor(key)
    mem = model.get_submodule(name).weight.detach().cpu()
    return float((saved.float() - mem.float()).abs().max())


def merge(start: str, adapter, out=None, model_path=None, device: str = "cpu", min_free_gb: float = MIN_FREE_GB,
          _allow_shared_out: bool = False) -> dict:
    """The merge (see the module docstring).  _allow_shared_out is for the tiny-model tests only; the CLI never
    sets it."""
    import torch

    from vpd import starts

    t0 = time.time()
    adapter = Path(adapter)
    meta = json.loads((adapter / "meta.json").read_text())
    if meta.get("start", meta.get("model")) != start:
        raise SystemExit(f"[merge] {adapter} was trained on {meta.get('start', meta.get('model'))!r}, not {start!r}")
    scale = float(meta["alpha"]) / int(meta["rank"])
    want = int(meta["lora"]["wrapped"])
    local, facts = choose_out(out, run_name_of(start, adapter), min_free_gb, _allow_shared_out)
    if local.exists() and any(local.iterdir()):
        raise SystemExit(f"[merge] refused: {local} exists and is not empty")
    model, proc = starts.load_model(start, device=device, attn="sdpa", path=model_path)
    n = T.merge_adapter(model, adapter / "adapter.pt", scale)
    if n != want:
        raise AssertionError(f"[merge] merged {n} modules, adapter meta says {want}")
    local.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(local), safe_serialization=True, max_shard_size="5GB")
    src = starts.processor_dir(start) if model_path is None or not os.path.lexists(
        Path(model_path) / "preprocessor_config.json") else Path(model_path)
    copied = copy_processor_files(Path(src), local)
    cfg, changes = view_config(json.loads((local / "config.json").read_text()))
    (local / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    weights_dir = Path(model_path) if model_path is not None else Path(starts.resolve(start)["path"])
    gen = greedy_generation_config([src, weights_dir])
    (local / "generation_config.json").write_text(json.dumps(gen, indent=1) + "\n")
    first = sorted(torch.load(adapter / "adapter.pt", map_location="cpu"))[0]
    diff = check_saved(model, local, first)
    if diff != 0.0:
        raise AssertionError(f"[merge] saved {first} differs from memory by {diff}")
    rep = {"start": start, "adapter": str(adapter), "adapter_sha256": T.sha256_file(adapter / "adapter.pt"),
           "adapter_meta": {k: meta.get(k) for k in ("start", "code", "path", "data", "epochs", "phase", "seed",
                                                     "code_sha256", "init_adapter")},
           "scale": scale, "merged": n, "weights": str(weights_dir), "processor_src": str(src), "copied": copied,
           "config_changes": changes, "generation_config": gen, "load_info": model.vpd_load_info,
           "out": str(local), "out_facts": facts, "check_tensor": first, "seconds": round(time.time() - t0, 1),
           "versions": T.versions(), **T.code_fingerprint()}
    (local / "vpd_merge.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    del model
    return rep


def main(argv=None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--adapter", type=Path, required=True, help="DIR with adapter.pt and meta.json")
    ap.add_argument("--out", type=Path, default=None, help="LOCAL target (job-local disk; never /mnt/umm)")
    ap.add_argument("--model-path", type=Path, default=None, help="weights folder overriding the start's path")
    ap.add_argument("--device", default="cpu", help="cpu (default: no GPU memory held before vLLM) or cuda")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    rep = merge(args.start, args.adapter, args.out, args.model_path, args.device, args.min_free_gb)
    print(f"[merge] {rep['merged']} modules merged (scale {rep['scale']}) in {rep['seconds']} s; "
          f"config changes {sorted(rep['config_changes'])}", flush=True)
    print(rep["out"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
