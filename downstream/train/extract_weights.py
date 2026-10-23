#!/usr/bin/env python3
"""Copy the model weights out of a full-state checkpoint into a StarVLA weights-only checkpoint, laid out like a run
directory, so the evaluation loads it unchanged (StarVLA reads CKPT.parents[1]/config.yaml and dataset_statistics.json):

    DEST/checkpoints/steps_<N>_pytorch_model.pt     the module state dict (bf16), what VLATrainer._save_checkpoint
                                                     writes (torch.save of the unwrapped model's state_dict)
    DEST/config.yaml, DEST/dataset_statistics.json  copied from the run directory
    DEST/source.json                                where it came from; written last, so it marks a finished copy

A full state (downstream/train/train_entry.py, accelerate save_state with DeepSpeed ZeRO-2) keeps the bf16 weights of
the whole model in <tag>/mp_rank_00_model_states.pt under "module" (<tag> from its file `latest`); the fp32 master
weights and AdamW moments are in the optimizer files, which are not read.

The user's decision 2 of 2026-10-23 (evening): E1 is evaluated at 150k, 175k and 200k steps.  The runs keep only
their newest full state, so the watcher copies the weights out of the 150k and 175k states before the next state
replaces them (downstream/eval/watch.py).  Run with the StarVLA environment's Python (torch + deepspeed: the
DeepSpeed file may hold DeepSpeed objects next to the weights).

    python downstream/train/extract_weights.py FULL_STATE RUN_DIR DEST            # copy (no-op if already done)
    python downstream/train/extract_weights.py FULL_STATE RUN_DIR DEST --dry      # read and check only, no writing
    ... --like CKPT    also require the same tensor names, shapes and dtypes as an existing StarVLA checkpoint
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

CONFIG_FILES = ("config.yaml", "dataset_statistics.json")


def model_states_file(full_state: Path) -> Path:
    tag = (full_state / "latest").read_text().strip() if (full_state / "latest").is_file() else "pytorch_model"
    path = full_state / tag / "mp_rank_00_model_states.pt"
    if not path.is_file():
        raise FileNotFoundError(f"no {path}")
    return path


def checkpoint_path(dest: Path, step: int) -> Path:
    return dest / "checkpoints" / f"steps_{step}_pytorch_model.pt"


def done(dest: Path):
    """source.json of a finished copy in DEST, or None."""
    src = dest / "source.json"
    if not src.is_file():
        return None
    meta = json.loads(src.read_text())
    return meta if checkpoint_path(dest, int(meta["step"])).is_file() else None


def describe(state_dict) -> dict:
    """Tensor count, total elements, bytes and dtypes, and a fingerprint of names, shapes and dtypes."""
    h = hashlib.sha256()
    numel, nbytes, dtypes = 0, 0, {}
    for name in sorted(state_dict):
        t = state_dict[name]
        h.update(f"{name}:{tuple(t.shape)}:{t.dtype};".encode())
        numel += t.numel()
        nbytes += t.numel() * t.element_size()
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
    return {"tensors": len(state_dict), "numel": numel, "bytes": nbytes, "dtypes": dtypes,
            "layout_sha256": h.hexdigest()}


def load_module(path: Path):
    import torch

    blob = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(blob, dict) or "module" not in blob:
        raise ValueError(f"{path} has no 'module' entry (keys: {list(blob)[:10] if isinstance(blob, dict) else type(blob)})")
    return blob["module"]


def load_plain(path: Path):
    import torch

    return torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)


def finite_sum(state_dict) -> float:
    """Reads every tensor once (the whole file through the page cache); nan / inf would show in the sum."""
    total = 0.0
    for t in state_dict.values():
        if t.is_floating_point():
            total += float(t.float().sum())
    return total


def extract(full_state: Path, run_dir: Path, dest: Path, like: Path = None, dry: bool = False,
            now=datetime.datetime.now) -> dict:
    full_state, run_dir, dest = Path(full_state), Path(run_dir), Path(dest)
    prior = done(dest)
    if prior and not dry:
        return dict(prior, already=True)
    state = json.loads((full_state / "downstream_state.json").read_text())
    step = int(state["completed_steps"])
    src = model_states_file(full_state)
    src_stat = src.stat()
    t0 = time.time()
    module = load_module(src)
    info = describe(module)
    if like is not None:
        ref = describe(load_plain(Path(like)))
        if ref["layout_sha256"] != info["layout_sha256"]:
            raise ValueError(f"tensor names / shapes / dtypes differ from {like}: {ref} vs {info}")
    meta = {"step": step, "full_state": str(full_state), "model_states": str(src), "model_states_bytes": src_stat.st_size,
            "model_states_mtime": src_stat.st_mtime, "downstream_state": state, "weights": info,
            "like": str(like) if like else None}
    if dry:
        meta["sum"] = finite_sum(module)
        meta["read_seconds"] = round(time.time() - t0, 1)
        return meta
    import torch

    ckpt = checkpoint_path(dest, step)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_name(ckpt.name + ".partial")
    torch.save(module, str(tmp))
    if not src.is_file() or src.stat().st_mtime != src_stat.st_mtime:  # replaced or pruned while we read it
        tmp.unlink()
        raise RuntimeError(f"{src} changed or vanished during the copy; nothing kept")
    os.replace(tmp, ckpt)
    for name in CONFIG_FILES:
        shutil.copy2(run_dir / name, dest / name)
    meta.update(checkpoint=str(ckpt), checkpoint_bytes=ckpt.stat().st_size, seconds=round(time.time() - t0, 1),
                utc=now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    (dest / "source.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("full_state", type=Path, help="RUN/full_state/steps_<N> (complete, not .partial)")
    ap.add_argument("run_dir", type=Path, help="the run directory (config.yaml, dataset_statistics.json)")
    ap.add_argument("dest", type=Path, help="EXP/evals_src/steps_<N>")
    ap.add_argument("--like", type=Path, help="a StarVLA checkpoint the copy must match in names, shapes, dtypes")
    ap.add_argument("--dry", action="store_true", help="read and check only")
    args = ap.parse_args(argv)
    print(json.dumps(extract(args.full_state, args.run_dir, args.dest, args.like, args.dry)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
