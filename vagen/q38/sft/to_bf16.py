"""q38: a bf16 copy of an fp32 Hugging Face checkpoint (LLaMA-Factory's per-epoch save of the 27B SFT).

LLaMA-Factory has no option to save bf16 without dropping the fp32 master weights (pure_bf16 trains in bf16;
SFT_OPTIONS.md 6), so every trainable tensor of a checkpoint is fp32 (27B: about 109 GB) while config.json says
bfloat16.  vLLM loads such a checkpoint into bf16 parameters (the evaluation and the rollout see bf16 weights anyway)
and the RL actor loads it into fp32 (verl's model_dtype fp32), so a bf16 copy changes neither the evaluation nor the
rollout and moves the actor's initial weights by less than half a bf16 ulp (SFT_OPTIONS.md 6; to be confirmed on
readings before the fp32 files are dropped, P1_REPORT decision 4).

    python -m vagen.q38.sft.to_bf16 convert SRC DST [--max-shard-gb 5] [--fp32-modules A_log]
    python -m vagen.q38.sft.to_bf16 convert SRC --in-place      # replaces SRC's fp32 shards (deletes them)
    python -m vagen.q38.sft.to_bf16 verify SRC DST [--fp32-modules A_log]

fp32 modules (user OK 2026-11-20, P2_FIX_REPORT question 4b): float32 tensors whose names contain one of
--fp32-modules (comma list, default A_log; '' for none) stay float32, bit-identical; the same substring rule and
default as verl's checkpoint.hf_fp32_modules, so SFT and RL exports keep the 48 Gated DeltaNet A_log in fp32, the
recipe the bf16 readout check verified (PASS_IDENTICAL, 2026-11-20).

convert: every other float32 tensor -> torch.bfloat16 (round to nearest even, torch's cast, the cast vLLM's loader does
when it copies an fp32 tensor into a bf16 parameter); other dtypes unchanged; shards of at most --max-shard-gb,
streamed tensor by tensor (memory: one output shard); model.safetensors.index.json rewritten; every other file of SRC
copied (config, tokenizer, processor, chat template, trainer_state.json, ...).  The copy is written to DST.partial
and renamed to DST only after verify passed: same tensor names and shapes, each tensor bit-identical to the cast of
the fp32 one, |bf16 - fp32| <= 2**-8 |fp32| (+ the bf16 subnormal step).  DST/bf16_manifest.json records the sizes,
counts, dtypes and the check.  --in-place writes the copy next to SRC, verifies it, then swaps the shards and the
index into SRC (the fp32 shards are deleted; only with that flag).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WEIGHT_SUFFIX = ".safetensors"
INDEX = "model.safetensors.index.json"
MANIFEST = "bf16_manifest.json"
REL_TOL = 2.0 ** -8  # half a bf16 ulp is 2**-9 relative; 2**-8 leaves room for the binade edge
ABS_TOL = 2.0 ** -133  # bf16's smallest subnormal step
FP32_MODULES = ("A_log",)  # = verl CheckpointConfig.hf_fp32_modules default


def keep_fp32(name: str, fp32_modules) -> bool:
    return any(m in name for m in fp32_modules)


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def weight_files(ckpt: Path) -> List[str]:
    idx = ckpt / INDEX
    if idx.is_file():
        return sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    single = ckpt / "model.safetensors"
    if single.is_file():
        return [single.name]
    raise FileNotFoundError(f"no {INDEX} or model.safetensors in {ckpt}")


def tensor_plan(ckpt: Path) -> List[Tuple[str, str, List[int], str]]:
    """(file, tensor name, shape, dtype) of every tensor, read from the safetensors headers only."""
    import struct

    out = []
    for f in weight_files(ckpt):
        with open(ckpt / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            out.append((f, name, list(meta["shape"]), meta["dtype"]))
    return out


DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def out_bytes(shape: List[int], dtype: str, fp32: bool = False) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * (2 if dtype == "F32" and not fp32 else DTYPE_BYTES[dtype])


def shard_names(groups: int) -> List[str]:
    if groups == 1:
        return ["model.safetensors"]
    return [f"model-{i + 1:05d}-of-{groups:05d}.safetensors" for i in range(groups)]


def plan_shards(plan, max_bytes: int, fp32_modules=FP32_MODULES) -> List[List[Tuple[str, str, List[int], str]]]:
    groups, cur, size = [], [], 0
    for item in plan:
        b = out_bytes(item[2], item[3], keep_fp32(item[1], fp32_modules))
        if cur and size + b > max_bytes:
            groups.append(cur)
            cur, size = [], 0
        cur.append(item)
        size += b
    if cur:
        groups.append(cur)
    return groups


def copy_side_files(src: Path, dst: Path, weights: List[str]) -> List[str]:
    copied = []
    for p in sorted(src.iterdir()):
        if p.name in weights or p.name == INDEX or p.name == MANIFEST or p.name.startswith(".bf16"):
            continue
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            copied.append(p.name)
    return copied


def write_copy(src: Path, dst: Path, max_shard_gb: float, fp32_modules=FP32_MODULES) -> Dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    plan = tensor_plan(src)
    weights = weight_files(src)
    groups = plan_shards(plan, int(max_shard_gb * (1 << 30)), fp32_modules)
    names = shard_names(len(groups))
    dst.mkdir(parents=True, exist_ok=False)
    handles: Dict[str, object] = {}
    weight_map, total = {}, 0
    counts = {"cast_f32_to_bf16": 0, "kept_fp32": 0, "kept": 0}
    kept_dtypes: Dict[str, int] = {}
    try:
        for name, group in zip(names, groups):
            tensors = {}
            for f, t, _, dt in group:
                if f not in handles:
                    handles[f] = safe_open(str(src / f), framework="pt")
                x = handles[f].get_tensor(t)
                if x.dtype == torch.float32 and keep_fp32(t, fp32_modules):
                    counts["kept_fp32"] += 1
                elif x.dtype == torch.float32:
                    x = x.to(torch.bfloat16)
                    counts["cast_f32_to_bf16"] += 1
                else:
                    counts["kept"] += 1
                    kept_dtypes[dt] = kept_dtypes.get(dt, 0) + 1
                tensors[t] = x.contiguous()
                weight_map[t] = name
                total += x.numel() * x.element_size()
            save_file(tensors, str(dst / name), metadata={"format": "pt"})
            del tensors
    finally:
        handles.clear()
    if len(names) > 1:
        (dst / INDEX).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
    side = copy_side_files(src, dst, weights)
    return {"tensors": len(plan), "shards": names, "bytes": total, "counts": counts, "kept_dtypes": kept_dtypes,
            "fp32_modules": list(fp32_modules), "side_files": side}


def verify(src: Path, dst: Path, fp32_modules=FP32_MODULES) -> Dict:
    """Every tensor of DST against SRC: names, shapes, dtype bf16 where SRC is fp32 (fp32 and bit-identical for
    fp32_modules), bit-identical to SRC's cast."""
    import torch
    from safetensors import safe_open

    a = {t: f for f, t, _, _ in tensor_plan(src)}
    b = {t: f for f, t, _, _ in tensor_plan(dst)}
    problems: List[str] = []
    if set(a) != set(b):
        problems.append(f"tensor names differ: {len(set(a) - set(b))} missing, {len(set(b) - set(a))} extra")
    ha: Dict[str, object] = {}
    hb: Dict[str, object] = {}
    worst_rel = 0.0
    checked = 0
    for t in sorted(set(a) & set(b)):
        fa, fb = a[t], b[t]
        ha.setdefault(fa, safe_open(str(src / fa), framework="pt"))
        hb.setdefault(fb, safe_open(str(dst / fb), framework="pt"))
        x, y = ha[fa].get_tensor(t), hb[fb].get_tensor(t)
        if list(x.shape) != list(y.shape):
            problems.append(f"{t}: shape {list(x.shape)} vs {list(y.shape)}")
            continue
        if x.dtype == torch.float32 and keep_fp32(t, fp32_modules):
            if not (y.dtype == torch.float32 and torch.equal(x, y)):
                problems.append(f"{t}: an fp32 module tensor is not the unchanged float32 one ({y.dtype})")
        elif x.dtype == torch.float32:
            if y.dtype != torch.bfloat16:
                problems.append(f"{t}: dtype {y.dtype}, not bfloat16")
                continue
            if not torch.equal(x.to(torch.bfloat16).view(torch.int16), y.view(torch.int16)):
                problems.append(f"{t}: not the round-to-nearest-even cast of the fp32 tensor")
            err = (y.float() - x).abs()
            bound = x.abs() * REL_TOL + ABS_TOL
            if bool((err > bound).any()):
                problems.append(f"{t}: |bf16 - fp32| above 2**-8 relative")
            nz = x.abs() > 0
            if bool(nz.any()):
                worst_rel = max(worst_rel, float((err[nz] / x.abs()[nz]).max()))
        elif not (x.dtype == y.dtype and torch.equal(x, y)):
            problems.append(f"{t}: a {x.dtype} tensor changed")
        checked += 1
    return {"tensors_checked": checked, "worst_relative_error": worst_rel, "problems": problems[:50],
            "ok": not problems and checked == len(a)}


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.iterdir() if f.is_file())


def convert(src: Path, dst: Optional[Path], max_shard_gb: float = 5.0, in_place: bool = False,
            fp32_modules=FP32_MODULES) -> Dict:
    src = src.resolve()
    if in_place:
        if dst is not None:
            raise ValueError("--in-place takes no DST")
        work = src / ".bf16_tmp"
    else:
        if dst is None:
            raise ValueError("give DST or --in-place")
        if dst.exists():
            raise SystemExit(f"{dst} exists")
        work = Path(str(dst) + ".partial")
    if work.exists():
        shutil.rmtree(work)  # a partial copy of an earlier, interrupted run of this tool
    t0 = time.time()
    src_bytes = sum((src / f).stat().st_size for f in weight_files(src))
    info = write_copy(src, work, max_shard_gb, fp32_modules)
    t_write = time.time() - t0
    check = verify(src, work, fp32_modules)
    manifest = {"utc": utc(), "tool": "vagen.q38.sft.to_bf16", "source": str(src), "source_weight_bytes": src_bytes,
                "bf16_weight_bytes": info["bytes"], "write_seconds": round(t_write, 1),
                "verify_seconds": round(time.time() - t0 - t_write, 1), "in_place": in_place, **info, "verify": check}
    if not check["ok"]:
        (work / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        raise SystemExit(f"verify failed, the copy is left in {work}: {check['problems'][:5]}")
    if in_place:
        old = weight_files(src)
        for f in old:
            (src / f).unlink()
        if (src / INDEX).exists():
            (src / INDEX).unlink()
        for f in info["shards"] + ([INDEX] if (work / INDEX).exists() else []):
            os.replace(work / f, src / f)
        manifest["fp32_deleted"] = old
        (src / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        shutil.rmtree(work)
        manifest["output"] = str(src)
    else:
        (work / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        os.replace(work, dst)
        manifest["output"] = str(dst)
    return manifest


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("src", type=Path)
    c.add_argument("dst", type=Path, nargs="?")
    c.add_argument("--max-shard-gb", type=float, default=5.0)
    c.add_argument("--in-place", action="store_true", help="replace SRC's fp32 shards (deletes them)")
    v = sub.add_parser("verify")
    v.add_argument("src", type=Path)
    v.add_argument("dst", type=Path)
    for p in (c, v):
        p.add_argument("--fp32-modules", default=",".join(FP32_MODULES),
                       help="comma list of name substrings kept float32 (default A_log; '' for none)")
    a = ap.parse_args(argv)
    mods = tuple(m for m in a.fp32_modules.split(",") if m)
    if a.cmd == "convert":
        m = convert(a.src, a.dst, a.max_shard_gb, a.in_place, mods)
        print(json.dumps({k: m[k] for k in ("output", "source_weight_bytes", "bf16_weight_bytes", "write_seconds",
                                             "verify_seconds", "counts")} | {"ok": m["verify"]["ok"]}, indent=1))
    else:
        r = verify(a.src, a.dst, mods)
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
