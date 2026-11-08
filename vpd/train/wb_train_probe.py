#!/usr/bin/env python3
"""In-job train probe for the WalkerBench smoke job (WP-TRAIN; plan 3.4; DECISIONS B9, B10).  GPU, PY_TRAIN.

    PY_TRAIN vpd/train/wb_train_probe.py --start Q3S [--demos demos_all.jsonl] [--n 8] [--out DIR]
    PY_TRAIN vpd/train/wb_train_probe.py --load-only Q3RD [--out DIR]
    PY_TRAIN vpd/train/wb_train_probe.py --start Q3S --n 8 --load-only Q3RD      (both, train first)

Train: ONE optimizer step (epochs 1, accum N) of vpd/train/n2_train_vpd.py on the N longest demos (most images,
then most text), with the trainer's defaults (sdpa, 3 loader workers, max_pixels cap); writes OUT/train/ (the
trainer's adapter.pt, meta.json, log.jsonl) and OUT/probe_train.json: peak memory, tokens, tokens/s, step and
loader-wait seconds, the attention sites, max_len and the demo ids.
--load-only NAME: loads the start with vpd.starts.load_model(attn='sdpa') on the GPU (asserting sdpa on the model,
its language model and its vision tower, and bf16 everywhere: the verl RL exports say eager / float32) and writes
OUT/probe_load_<NAME>.json; no training step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from vpd import paths  # noqa: E402
from vpd.train import n2_train_vpd as T  # noqa: E402

DEFAULT_DEMOS = paths.WB_DEMOS / "s2train_640x400" / "demos_all.jsonl"
DEFAULT_OUT = paths.WB_RUNS / "smoke" / "train_probe"


def demo_size(rec: dict) -> tuple:
    return (len(rec.get("images") or []), sum(len(m["content"]) for m in rec["messages"]))


def longest(records: list, n: int) -> list:
    """The n largest demos by (images, characters), ties by id; in the data's order."""
    ranked = sorted(range(len(records)), key=lambda i: (demo_size(records[i]), str(records[i].get("id"))),
                    reverse=True)[:n]
    return [records[i] for i in sorted(ranked)]


def probe_train(args) -> dict:
    recs = longest(T.load_records(args.demos), args.n)
    if len(recs) < args.n:
        raise SystemExit(f"[probe] {args.demos}: only {len(recs)} demos")
    out = args.out / "train"
    out.mkdir(parents=True, exist_ok=True)
    data = args.out / "probe_demos.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in recs))
    argv = ["--start", args.start, "--data", str(data), "--out", str(out), "--epochs", "1",
            "--accum", str(args.n), "--seed", str(args.seed), "--attn", args.attn, "--phase", "probe",
            "--num-workers", str(args.num_workers), "--device", args.device]
    if args.threads:
        argv += ["--threads", str(args.threads)]
    if args.model_path:
        argv += ["--model-path", str(args.model_path)]
    t0 = time.time()
    rc = T.main(argv)
    meta = json.loads((out / "meta.json").read_text())
    steps = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines() if x.strip()]
    rep = {"rc": rc, "start": args.start, "attn": meta["attn"], "attn_sites": meta["attn_sites"],
           "demos": [r.get("id") for r in recs], "images": [len(r.get("images") or []) for r in recs],
           "optimizer_steps": meta["optimizer_steps"], "tokens": meta["tokens_total"], "max_len": meta["max_len"],
           "tokens_per_s": meta["tokens_per_s"], "peak_mem_gib": meta["peak_mem_gib"],
           "step_seconds": [s["step_seconds"] for s in steps], "wait_seconds": [s["wait_seconds"] for s in steps],
           "load_seconds": meta["load_seconds"], "train_seconds": meta["train_seconds"],
           "skipped": meta["skipped"], "wall_seconds": round(time.time() - t0, 1), "versions": meta["versions"]}
    (args.out / "probe_train.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[probe] train {args.start}: {rep['tokens']} tokens, {rep['tokens_per_s']} tok/s, peak "
          f"{rep['peak_mem_gib']} GiB, attn {sorted(set(rep['attn_sites'].values()))}", flush=True)
    return rep


def probe_load(args, name: str) -> dict:
    import gc

    import torch

    from vpd import starts

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.time()
    model, _ = starts.load_model(name, device=args.device, attn=args.attn, path=args.load_only_path)
    info = dict(model.vpd_load_info)
    raw = json.loads((Path(info["path"]) / "config.json").read_text())
    rep = {"start": name, **info, "config_file_attn": raw.get("attn_implementation"),
           "config_file_sub_dtypes": {k: (raw.get(k) or {}).get("dtype") for k in ("text_config", "vision_config")},
           "load_seconds": round(time.time() - t0, 1),
           "gpu_mem_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
           "verdict": "PASS"}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    (args.out / f"probe_load_{name}.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"[probe] load-only {name}: attn {rep['attn_sites']} (file says {rep['config_file_attn']}), "
          f"{rep['load_seconds']} s", flush=True)
    return rep


def main(argv=None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=None)
    ap.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--load-only", default=None, metavar="NAME")
    ap.add_argument("--attn", default="sdpa", choices=("sdpa", "fa2"))
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--num-workers", type=int, default=T.NUM_WORKERS)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-path", type=Path, default=None, help="train probe: weights override (tests)")
    ap.add_argument("--load-only-path", type=Path, default=None, help="load-only: weights override (tests)")
    args = ap.parse_args(argv)
    if not args.start and not args.load_only:
        ap.error("give --start (train probe) and/or --load-only NAME")
    args.out.mkdir(parents=True, exist_ok=True)
    rc = 0
    if args.start:
        rc = probe_train(args)["rc"]
    if args.load_only:
        probe_load(args, args.load_only)
    return rc


if __name__ == "__main__":
    sys.exit(main())
