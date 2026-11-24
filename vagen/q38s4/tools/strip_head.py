"""C line (q38 s4): an evaluation folder of a C-NFP bf16 copy without the NFP head (PREREG_v1 section 8).

    python -m vagen.q38s4.tools.strip_head plan  SRC          # what make would do; reads headers only
    python -m vagen.q38s4.tools.strip_head make  SRC DST
    python -m vagen.q38s4.tools.strip_head check SRC DST

vLLM 0.28's qwen3_5 loader stops at weights it does not know, and a C-NFP copy holds the head's four tensors
nfp_head.fc{1,2}.{weight,bias} (in the smoke copies all four sit at the end of the last shard).  make builds DST in
DST.partial and renames it to DST only after the check passed:
- a weight shard without a head tensor becomes a symlink to SRC's shard (absolute path);
- a shard holding head tensors is written again with only its other tensors: the same header metadata, each tensor's
  dtype, shape and bytes copied verbatim from the file (no load, no cast), in SRC's order within each alignment class
  (the larger item sizes first, as safetensors' own writer does, so every tensor stays aligned); a shard holding only
  head tensors is left out;
- model.safetensors.index.json without the head keys, total_size reduced by their bytes;
- the side files (config, tokenizer, chat template, processor and generation configs, trainer_state.json, ...) are
  copied; the Trainer's resume states (bf16_lean.is_resume_state) and SRC's bf16_manifest.json are not (the latter's
  md5 goes into DST's strip_manifest.json, which lists everything make did and the check).
check (also the last step of make): DST's keys = SRC's keys minus exactly the four head keys; the index names for each
key the shard whose header holds it; every symlink resolves to SRC's shard of that name; every tensor of a rewritten
shard has SRC's dtype and shape and the same bytes, compared in the files and again as read back by safetensors;
no other weight file in DST.  SRC is only read.  A SRC without the head (C-noNFP) or with other nfp_head keys is
refused: C-noNFP copies are evaluated as they are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vagen.q38s4.tools.bf16_lean import is_resume_state

MODULE = "vagen.q38s4.tools.strip_head"
HEAD_KEYS = ("nfp_head.fc1.bias", "nfp_head.fc1.weight", "nfp_head.fc2.bias", "nfp_head.fc2.weight")
HEAD_PREFIX = "nfp_head."
INDEX = "model.safetensors.index.json"
BF16_MANIFEST = "bf16_manifest.json"
MANIFEST = "strip_manifest.json"
WEIGHT_SUFFIX = ".safetensors"
ITEM_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
              "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
MAX_HEADER = 100 * 2 ** 20
CHUNK = 16 * 2 ** 20


class StripError(RuntimeError):
    pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ safetensors files
def read_header(path: Path) -> Tuple[int, Optional[Dict], Dict[str, Dict]]:
    """(byte offset of the data region, __metadata__ or None, {key: {dtype, shape, data_offsets}})."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise StripError(f"{path}: shorter than a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if n > MAX_HEADER:
            raise StripError(f"{path}: header of {n} bytes")
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return 8 + n, meta, h


def item_bytes(dtype: str) -> int:
    if dtype not in ITEM_BYTES:
        raise StripError(f"unknown safetensors dtype {dtype}")
    return ITEM_BYTES[dtype]


def nbytes(info: Dict) -> int:
    b, e = info["data_offsets"]
    return int(e) - int(b)


def check_layout(path: Path, start: int, tensors: Dict[str, Dict]) -> None:
    """The data region is the tensors back to back, each of the size its dtype and shape give."""
    end = 0
    for k, v in sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        b, e = v["data_offsets"]
        n = item_bytes(v["dtype"])
        for d in v["shape"]:
            n *= int(d)
        if b != end or e - b != n:
            raise StripError(f"{path}: tensor {k} at {v['data_offsets']}, expected [{end}, {end + n}]")
        end = e
    if start + end != path.stat().st_size:
        raise StripError(f"{path}: data region ends at {start + end}, file has {path.stat().st_size} bytes")


def kept_order(tensors: Dict[str, Dict], drop) -> List[str]:
    """The kept keys, larger item sizes first, SRC's order within each size: every offset a multiple of its item."""
    keys = [k for k in tensors if k not in drop]
    return sorted(keys, key=lambda k: (-item_bytes(tensors[k]["dtype"]), tensors[k]["data_offsets"][0]))


def copy_range(fi, fo, offset: int, n: int) -> None:
    fi.seek(offset)
    while n:
        b = fi.read(min(CHUNK, n))
        if not b:
            raise StripError("source shard ended early")
        fo.write(b)
        n -= len(b)


def rewrite_shard(src: Path, dst: Path, drop) -> Dict:
    """dst = src without the keys in drop; the other tensors' bytes copied verbatim."""
    start, meta, ts = read_header(src)
    check_layout(src, start, ts)
    keep = kept_order(ts, drop)
    new: Dict = {} if meta is None else {"__metadata__": meta}
    off = 0
    for k in keep:
        n = nbytes(ts[k])
        new[k] = {"dtype": ts[k]["dtype"], "shape": ts[k]["shape"], "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(new, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)  # data region at a multiple of 8, as safetensors writes it
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for k in keep:
            copy_range(fi, fo, start + ts[k]["data_offsets"][0], nbytes(ts[k]))
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    return {"kept": len(keep), "dropped": sorted(set(ts) & set(drop)), "bytes": dst.stat().st_size,
            "source_bytes": src.stat().st_size}


# ------------------------------------------------------------------------------------------------ the copy
def load_index(d: Path) -> Dict:
    p = d / INDEX
    if not p.is_file():
        raise StripError(f"{d}: no {INDEX}")
    idx = json.loads(p.read_text())
    if not isinstance(idx.get("weight_map"), dict) or not idx["weight_map"]:
        raise StripError(f"{p}: no weight_map")
    return idx


def shard_keys(wm: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k, f in wm.items():
        out.setdefault(f, []).append(k)
    return out


def source_plan(src: Path) -> Dict:
    """Reads SRC's index and shard headers and says what make would do; refuses a SRC it cannot strip."""
    src = Path(src)
    idx = load_index(src)
    wm = idx["weight_map"]
    head = sorted(k for k in wm if k.startswith(HEAD_PREFIX))
    if not head:
        raise StripError(f"{src}: no {HEAD_PREFIX}* weights (a C-noNFP copy is evaluated as it is)")
    if head != sorted(HEAD_KEYS):
        raise StripError(f"{src}: head keys {head}, expected exactly {sorted(HEAD_KEYS)}")
    per = shard_keys(wm)
    on_disk = sorted(p.name for p in src.iterdir() if p.name.endswith(WEIGHT_SUFFIX))
    if on_disk != sorted(per):
        raise StripError(f"{src}: weight files {on_disk} != the index's {sorted(per)}")
    head_bytes, shards = 0, {}
    for f in sorted(per):
        start, _, ts = read_header(src / f)
        check_layout(src / f, start, ts)
        if set(ts) != set(per[f]):
            raise StripError(f"{src / f}: header keys differ from the index's ({len(ts)} vs {len(per[f])})")
        hk = sorted(set(ts) & set(HEAD_KEYS))
        head_bytes += sum(nbytes(ts[k]) for k in hk)
        shards[f] = "link" if not hk else ("drop" if len(hk) == len(ts) else "rewrite")
    side, skipped = [], []
    for p in sorted(src.iterdir(), key=lambda p: p.name):
        if p.name.endswith(WEIGHT_SUFFIX) or p.name == INDEX:
            continue
        if p.is_dir() or p.name == BF16_MANIFEST or is_resume_state(p.name) or not p.is_file():
            skipped.append(p.name)
        else:
            side.append(p.name)
    return {"source": str(src.resolve()), "tensors": len(wm), "head_keys": head, "head_bytes": head_bytes,
            "shards": shards, "side_files": side, "skipped": skipped, "index_metadata": idx.get("metadata")}


def make(src: Path, dst: Path) -> Dict:
    src, dst = Path(src).resolve(), Path(dst).absolute()
    part = dst.with_name(dst.name + ".partial")
    for p in (dst, part):
        if os.path.lexists(p):
            raise StripError(f"{p} exists; make never overwrites")
    t0 = time.time()
    plan = source_plan(src)
    idx = load_index(src)
    part.mkdir(parents=True)
    rewritten, linked, dropped = {}, [], []
    for f, what in plan["shards"].items():
        if what == "link":
            os.symlink(src / f, part / f)
            linked.append(f)
        elif what == "rewrite":
            rewritten[f] = rewrite_shard(src / f, part / f, set(HEAD_KEYS))
        else:
            dropped.append(f)
    meta = dict(idx.get("metadata") or {})
    if "total_size" in meta:
        meta["total_size"] = int(meta["total_size"]) - plan["head_bytes"]
    wm = {k: f for k, f in idx["weight_map"].items() if k not in HEAD_KEYS}
    (part / INDEX).write_text(json.dumps({"metadata": meta, "weight_map": dict(sorted(wm.items()))}, indent=2) + "\n")
    for n in plan["side_files"]:
        shutil.copy2(src / n, part / n)
    res = check(src, part)
    man = {"utc": utc(), "tool": MODULE, "source": str(src), "dst": str(dst), "head_keys": plan["head_keys"],
           "head_bytes": plan["head_bytes"], "linked": linked, "rewritten": rewritten, "dropped_shards": dropped,
           "side_files": plan["side_files"], "skipped": plan["skipped"], "source_index_md5": md5(src / INDEX),
           "source_bf16_manifest_md5": md5(src / BF16_MANIFEST) if (src / BF16_MANIFEST).is_file() else None,
           "seconds": round(time.time() - t0, 1), "check": res}
    (part / MANIFEST).write_text(json.dumps(man, indent=1) + "\n")
    if not res["ok"]:
        raise StripError(f"check failed, {part} left for inspection: {res['problems'][:5]}")
    os.rename(part, dst)
    return man


def same_bytes(fa, oa: int, fb, ob: int, n: int) -> bool:
    fa.seek(oa)
    fb.seek(ob)
    while n:
        k = min(CHUNK, n)
        a, b = fa.read(k), fb.read(k)
        if a != b or len(a) != k:
            return False
        n -= k
    return True


def readback_equal(src_file: Path, dst_file: Path, keys: List[str]) -> List[str]:
    """The tensors as safetensors reads them back, compared as bytes; returns the keys that differ."""
    import torch
    from safetensors import safe_open

    bad = []
    with safe_open(str(src_file), framework="pt") as a, safe_open(str(dst_file), framework="pt") as b:
        for k in keys:
            x, y = a.get_tensor(k), b.get_tensor(k)
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(
                    x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8)):
                bad.append(k)
    return bad


def check(src: Path, dst: Path, readback: bool = True) -> Dict:
    src, dst = Path(src).resolve(), Path(dst)
    problems: List[str] = []
    swm = load_index(src)["weight_map"]
    dwm = load_index(dst)["weight_map"]
    want = set(swm) - set(HEAD_KEYS)
    if set(dwm) != want:
        problems.append(f"keys: {len(set(dwm) - want)} extra, {len(want - set(dwm))} missing")
    if set(HEAD_KEYS) - set(swm):
        problems.append("SRC lacks some of the head keys")
    per = shard_keys(dwm)
    on_disk = sorted(p.name for p in dst.iterdir() if p.name.endswith(WEIGHT_SUFFIX))
    if on_disk != sorted(per):
        problems.append(f"weight files {on_disk} != the index's {sorted(per)}")
    checked, linked, rewritten = 0, 0, 0
    for f, keys in sorted(per.items()):
        p = dst / f
        if not p.exists():
            problems.append(f"{f}: missing")
            continue
        if p.is_symlink():
            if not (src / f).exists() or not os.path.samefile(p, src / f):
                problems.append(f"{f}: symlink to {os.path.realpath(p)}, not SRC's {f}")
                continue
            _, _, ts = read_header(p)
            if set(ts) != set(keys):
                problems.append(f"{f}: header keys differ from the index's")
            if set(ts) & set(HEAD_KEYS):
                problems.append(f"{f}: linked shard holds head tensors")
            linked += 1
            checked += len(ts)
            continue
        rewritten += 1
        try:
            dstart, _, dts = read_header(p)
            check_layout(p, dstart, dts)
        except StripError as e:
            problems.append(str(e))
            continue
        if set(dts) != set(keys):
            problems.append(f"{f}: header keys differ from the index's")
        if not (src / f).is_file():
            problems.append(f"{f}: rewritten shard has no SRC shard of that name")
            continue
        sstart, _, sts = read_header(src / f)
        with open(src / f, "rb") as fs, open(p, "rb") as fd:
            for k in sorted(dts):
                v = dts[k]
                if v["data_offsets"][0] % item_bytes(v["dtype"]):
                    problems.append(f"{f}:{k}: offset {v['data_offsets'][0]} not aligned")
                s = sts.get(k)
                if s is None or s["dtype"] != v["dtype"] or s["shape"] != v["shape"]:
                    problems.append(f"{f}:{k}: not SRC's dtype/shape")
                    continue
                if not same_bytes(fs, sstart + s["data_offsets"][0], fd, dstart + v["data_offsets"][0], nbytes(v)):
                    problems.append(f"{f}:{k}: bytes differ from SRC")
                checked += 1
        if readback and not problems:
            problems += [f"{f}:{k}: read back differently" for k in readback_equal(src / f, p, sorted(dts))]
    return {"ok": not problems, "tensors_checked": checked, "linked_shards": linked, "rewritten_shards": rewritten,
            "problems": problems}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog=MODULE, description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("src", type=Path)
    for name in ("make", "check"):
        p = sub.add_parser(name)
        p.add_argument("src", type=Path)
        p.add_argument("dst", type=Path)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    try:
        if a.cmd == "plan":
            out = source_plan(a.src)
            out["ok"] = True
        elif a.cmd == "make":
            m = make(a.src, a.dst)
            out = {"ok": True, "dst": m["dst"], "linked": len(m["linked"]), "rewritten": list(m["rewritten"]),
                   "dropped_shards": m["dropped_shards"], "check": {k: v for k, v in m["check"].items()
                                                                     if k != "problems"}, "seconds": m["seconds"]}
        else:
            out = check(a.src, a.dst)
    except StripError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
    print(json.dumps(out, indent=1))
    if not out.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
