"""The raw start as the trainer loads it, saved as a servable bf16 folder on JOB-LOCAL disk (EXPLORATORY; DECISIONS
G18).  The explore smoke (vpd/jobs/tvr_xsmoke.sh) serves this folder to vLLM for its raw-start mechanism checks: the
RL starts Q3RD / Q3RS150 have no eval view (their checkpoints are verl float32 exports), and the pilot smoke served
each start's eval view.  Not a run_tvr CODE_FILE and not a trainer code file.

    PY_Q3 -B -m vpd.tvr.raw_bf16 --start Q3RD --out $VPD_MERGE_DIR/xsmoke__Q3RD__raw --report SM/Q3RD/load/load.json

- vpd.starts.load_model(start, device=cpu, attn='sdpa'): the trainer's (and merge.py's) load: from_pretrained with
  torch_dtype bfloat16, so every float32 checkpoint tensor is cast to bf16; attention sdpa asserted on the model, its
  language model and its vision tower; every floating parameter asserted bf16.
- save_pretrained(safe_serialization, 5 GB shards) and the processor / tokenizer files, config.json and
  generation_config.json exactly as vpd/train/merge.py writes a merged folder, with merge.py's own functions
  (choose_out: never under /mnt/umm, >= 40 GB free; copy_processor_files from vpd.starts.processor_dir; view_config;
  greedy_generation_config): merge.py's folder without an adapter added.
- Conversion check: the check tensors (the first language-model q_proj, the embeddings, the first vision weight,
  lm_head) re-read from the saved shards are bit-equal to the checkpoint's tensor cast by torch .to(bfloat16) (what
  from_pretrained does); the saved keys == the checkpoint's keys; the checkpoint's dtype census (safetensors headers).
- Report (--report, and LOCAL/vpd_raw_bf16.json): ok iff the loaded dtype is bfloat16, sdpa at every attention site,
  the processor came from vpd.starts.processor_dir, every check tensor is equal and no key is missing or extra.
  Exit 0 ok, 1 not ok (the report is written either way), 2 bad call.
An 8B load: inside sco jobs only, never on the login node (tests use a tiny model).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

SCHEMA = "vpd.tvr.raw_bf16/1"
FIXED_KEYS = ("model.language_model.layers.0.self_attn.q_proj.weight", "model.language_model.embed_tokens.weight",
              "lm_head.weight")


def st_header(path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def st_index(d) -> dict:
    """{tensor key: shard path} of a safetensors folder (index.json, else model.safetensors)."""
    d = Path(d)
    idx = d / "model.safetensors.index.json"
    if os.path.lexists(idx):
        return {k: d / v for k, v in json.loads(idx.read_text())["weight_map"].items()}
    one = d / "model.safetensors"
    return {k: one for k in st_header(one)}


def dtype_census(d) -> dict:
    """{dtype: [tensors, elements]} over the folder's safetensors headers."""
    out: dict = {}
    for shard in sorted(set(st_index(d).values())):
        for v in st_header(shard).values():
            n = 1
            for x in v["shape"]:
                n *= int(x)
            c = out.setdefault(v["dtype"], [0, 0])
            c[0] += 1
            c[1] += n
    return out


def check_keys(keys) -> list:
    """FIXED_KEYS that exist, plus the first vision weight (sorted): the tensors compared bit for bit."""
    ks = set(keys)
    out = [k for k in FIXED_KEYS if k in ks]
    vis = sorted(k for k in ks if k.startswith("model.visual.") and k.endswith(".weight"))
    return out + vis[:1]


def save(start: str, out=None, model_path=None, device: str = "cpu", min_free_gb: float = None,
         _allow_shared_out: bool = False) -> dict:
    """The folder and report (see the module docstring).  model_path overrides the weights folder (tests: a tiny
    model); _allow_shared_out is for the tiny-model tests only, the CLI never sets it."""
    import torch
    from safetensors import safe_open

    from vpd import starts
    from vpd.train import merge as MG
    from vpd.train import n2_train_vpd as T

    t0 = time.time()
    e = starts.resolve(start)
    ck = Path(model_path) if model_path is not None else Path(e["path"])
    local, facts = MG.choose_out(out, f"{start}__raw_bf16", MG.MIN_FREE_GB if min_free_gb is None else min_free_gb,
                                 _allow_shared_out)
    if local.exists() and any(local.iterdir()):
        raise SystemExit(f"[raw_bf16] refused: {local} exists and is not empty")
    model, _proc = starts.load_model(start, device=device, attn="sdpa", path=model_path, log=None)
    info = dict(model.vpd_load_info)
    local.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(local), safe_serialization=True, max_shard_size="5GB")
    del model
    src = Path(starts.processor_dir(start)) if model_path is None or not os.path.lexists(
        Path(model_path) / "preprocessor_config.json") else Path(model_path)  # merge.py's rule
    copied = MG.copy_processor_files(src, local)
    cfg, changes = MG.view_config(json.loads((local / "config.json").read_text()))
    (local / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    gen = MG.greedy_generation_config([src, ck])
    (local / "generation_config.json").write_text(json.dumps(gen, indent=1) + "\n")
    ci, si = st_index(ck), st_index(local)
    only_ck, only_saved = sorted(set(ci) - set(si)), sorted(set(si) - set(ci))
    checks = []
    for k in check_keys(set(ci) & set(si)):
        with safe_open(str(ci[k]), framework="pt") as f:
            a = f.get_tensor(k)
        with safe_open(str(si[k]), framework="pt") as f:
            b = f.get_tensor(k)
        exp = a.to(torch.bfloat16)
        checks.append({"key": k, "checkpoint_dtype": str(a.dtype), "saved_dtype": str(b.dtype),
                       "shape": list(b.shape), "equal": bool(b.dtype == torch.bfloat16 and b.shape == exp.shape
                                                             and torch.equal(b, exp))})
    problems = []
    if info.get("dtype") != "bfloat16":
        problems.append(f"loaded dtype {info.get('dtype')!r}, not bfloat16")
    sites = info.get("attn_sites") or {}
    if not sites or any(v != "sdpa" for v in sites.values()):
        problems.append(f"attention sites {sites}")
    if Path(str(info.get("processor_dir"))) != src:
        problems.append(f"processor read from {info.get('processor_dir')}, not {src}")
    if not checks or not all(c["equal"] for c in checks):
        problems.append(f"check tensors not equal to the checkpoint cast to bf16: "
                        f"{[c['key'] for c in checks if not c['equal']] or 'none compared'}")
    if only_ck or only_saved:
        problems.append(f"keys only in the checkpoint {len(only_ck)}, only in the saved folder {len(only_saved)}")
    rep = {"schema": SCHEMA, "start": start, "code": e["code"], "checkpoint": str(ck), "out": str(local),
           "out_facts": facts, "dtype": info.get("dtype"), "attn_sites": sites,
           "processor_dir": info.get("processor_dir"), "processor_src": str(src), "copied": copied,
           "config_changes": changes, "generation_config": gen, "checkpoint_dtypes": dtype_census(ck),
           "saved_dtypes": dtype_census(local), "n_keys": {"checkpoint": len(ci), "saved": len(si)},
           "keys_only_in_checkpoint": only_ck[:20], "keys_only_in_saved": only_saved[:20],
           "tensor_checks": checks, "tensor_checks_ok": bool(checks) and all(c["equal"] for c in checks),
           "load_info": info, "ok": not problems, "problems": problems, "seconds": round(time.time() - t0, 1),
           "versions": T.versions(), **T.code_fingerprint()}
    (local / "vpd_raw_bf16.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep


def write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, p)


def main(argv=None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", type=Path, required=True, help="LOCAL target (job-local disk; never /mnt/umm)")
    ap.add_argument("--report", type=Path, required=True, help="the report json (e.g. SM/<S>/load/load.json)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    rep = save(args.start, args.out, device=args.device)
    write_json(args.report, rep)
    print(json.dumps({k: rep[k] for k in ("start", "ok", "problems", "dtype", "processor_dir", "seconds")}),
          flush=True)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
