"""Merged weights == LoRA, teacher-forced (CONTRACT_wave2 5.3.2, G0 item 7; brief A1 "合并后的模型和 LoRA 一致").

    PY_TRAIN -B -m vpd.tvr.merge_parity --start S --adapter A/<tier> --merged LOCAL \
        --demos OUT/tvr/tiers/v1/warmup0w20.jsonl --n 20 --out P/parity/merge_<tier>.json

Run inside a job for EVERY merged pilot eval (0W, 100, TOP, Q3B 100s1002; in the calibration for ALL and 400), after
vpd/train/merge.py and before vLLM starts (it needs the GPU alone).  On the first --n demos, teacher-forced, bf16
autocast, model.eval():
  (1) unmerged: vpd.starts.load_model(start) + lora.wrap(rank 16, alpha 32, dropout 0.05) + the LoRA masters in
      float32 (as the trainer) + n2_train_vpd.load_init_adapter(A);
  (2) merged: vpd.starts.load_model(start, path=LOCAL);
loaded one after the other (GPU memory freed in between).  Per reply (n2_train_vpd.encode of the whole conversation,
every reply span labelled; response positions only, as n2_train_vpd.response_loss): the mean token CE, and the
argmax token at every position of the reply; a reply "agrees" when the two models' argmax tokens are equal at EVERY
position of it.  Report: {"mean_abs_dce", "max_abs_dce", "reply_agreement", "n_replies", "n_demos", "pass"
(mean_abs_dce <= gates.merge_parity.max_mean_abs_dce and reply_agreement >= min_reply_agreement), "adapter_sha256",
"merged" (LOCAL/vpd_merge.json), "seconds", ...}.  Exit 0 pass, 1 fail, 2 bad call.  The trainer modules are
imported read-only (frozen).  Threads: --threads (default OMP_NUM_THREADS or 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

SCHEMA = "vpd.tvr.merge_parity/1"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thresholds(cfg=None) -> dict:
    from vpd.tvr import pilot_cfg

    mp = pilot_cfg.get("gates.merge_parity", cfg)
    return {"max_mean_abs_dce": float(mp["max_mean_abs_dce"]), "min_reply_agreement": float(mp["min_reply_agreement"])}


# --- comparison logic (pure; tested without a model) -----------------------------------------------------------

def reply_rows(ce_tokens, argmax_tokens, positions, spans) -> list:
    """Per reply span (a, b) (token positions a..b-1 of the reply in input_ids): {"ce": mean token CE, "argmax":
    [argmax tokens]} from per-position lists: positions[j] = p means CE ce_tokens[j] / argmax argmax_tokens[j]
    belong to the prediction of token p + 1 (logits at p predict p + 1)."""
    by_tok = {int(p) + 1: (float(c), int(a)) for p, c, a in zip(positions, ce_tokens, argmax_tokens)}
    rows = []
    for a, b in spans:
        toks = [by_tok[t] for t in range(int(a), int(b)) if t in by_tok]
        if len(toks) != int(b) - int(a):
            raise ValueError(f"reply span {a}..{b}: {len(toks)} of {int(b) - int(a)} positions scored")
        rows.append({"ce": sum(c for c, _ in toks) / len(toks), "argmax": [x for _, x in toks]})
    return rows


def compare(unmerged: list, merged: list, cfg=None) -> dict:
    """unmerged / merged: per demo the reply rows of reply_rows, in the same order."""
    th = thresholds(cfg)
    if len(unmerged) != len(merged):
        raise ValueError(f"{len(unmerged)} unmerged demos vs {len(merged)} merged")
    d, agree, n = [], 0, 0
    for du, dm in zip(unmerged, merged):
        if len(du) != len(dm):
            raise ValueError(f"demo with {len(du)} unmerged replies vs {len(dm)} merged")
        for ru, rm in zip(du, dm):
            if len(ru["argmax"]) != len(rm["argmax"]):
                raise ValueError("reply lengths differ between the two runs")
            d.append(abs(rm["ce"] - ru["ce"]))
            agree += int(ru["argmax"] == rm["argmax"])
            n += 1
    if not n:
        raise ValueError("no replies compared")
    mean_abs = sum(d) / n
    agreement = agree / n
    finite = all(math.isfinite(x) for x in d)
    return {"mean_abs_dce": mean_abs, "max_abs_dce": max(d), "reply_agreement": agreement, "n_replies": n,
            "n_demos": len(unmerged), "replies_agreeing": agree, "thresholds": th,
            "pass": bool(finite and mean_abs <= th["max_mean_abs_dce"] and agreement >= th["min_reply_agreement"])}


# --- model side (job only; PY_TRAIN) ---------------------------------------------------------------------------

def score_demos(model, processor, records, device_type: str = "cuda") -> list:
    """Teacher-forced reply rows of every record (whole conversation, every reply labelled)."""
    import torch
    import torch.nn.functional as F

    from vpd.train import n2_train_vpd as T

    out = []
    for rec in records:
        s = T.samples(rec, 0, False)[0]
        inputs, labels, spans, _ = T.encode(processor, s)
        T.check_spans(rec.get("id"), s["messages"], spans)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        labels = labels.to(model.device)
        idx = (labels[0, 1:] != -100).nonzero().squeeze(1)
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(**inputs, logits_to_keep=idx).logits[0].float()
        ce = F.cross_entropy(logits, labels[0, idx + 1], reduction="none")
        out.append(reply_rows(ce.tolist(), logits.argmax(-1).tolist(), idx.tolist(), spans))
    return out


def _release() -> None:
    """Collect garbage and return cached CUDA blocks.  The caller must first drop EVERY reference to the model (a
    `del` inside a helper only drops the helper's own name), so this runs after `del model, ...` in run()."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _allocated_gb():
    import torch

    return round(torch.cuda.memory_allocated() / 2 ** 30, 3) if torch.cuda.is_available() else None


def run(start: str, adapter: Path, merged: Path, demos: Path, n: int, device: str = "cuda") -> dict:
    import torch

    from vpd import starts
    from vpd.train import n2_train_vpd as T
    from vpd.tvr import pilot_cfg

    import lora  # downstream/train (n2_train_vpd put it on sys.path)

    rec = pilot_cfg.FROZEN["recipe"]
    t0 = time.time()
    records = T.load_records(demos, n)
    if len(records) < n:
        raise SystemExit(f"[merge_parity] {demos}: {len(records)} demos < --n {n}")
    meta = json.loads((adapter / "meta.json").read_text())
    if meta.get("start") != start:
        raise SystemExit(f"[merge_parity] {adapter} was trained on {meta.get('start')!r}, not {start!r}")
    dtype = "cuda" if str(device).startswith("cuda") else "cpu"
    model, proc = starts.load_model(start, device=device, attn=rec["attn"])
    info = lora.wrap(model, int(rec["rank"]), float(rec["alpha"]), float(rec["dropout"]), int(meta.get("seed") or 0))
    for m in model.modules():
        if isinstance(m, lora.LoRALinear):
            m.lora_A.data, m.lora_B.data = m.lora_A.data.float(), m.lora_B.data.float()
    m = None  # the loop variable would keep the last submodule alive past `del model`
    init = T.load_init_adapter(model, adapter, int(rec["rank"]), float(rec["alpha"]))
    model.eval()
    t1 = time.time()
    rows_u = score_demos(model, proc, records, dtype)
    t_u = time.time() - t1
    mem = {"after_unmerged_gb": _allocated_gb()}
    # free the unmerged model before the merged one is loaded (5.3.2): drop every reference held in this frame
    # (info / init are plain facts; the LoRA modules live inside the model), then collect
    del model, proc
    _release()
    mem["after_free_gb"] = _allocated_gb()
    model2, proc2 = starts.load_model(start, device=device, attn=rec["attn"], path=merged)
    model2.eval()
    t2 = time.time()
    rows_m = score_demos(model2, proc2, records, dtype)
    t_m = time.time() - t2
    del model2, proc2
    _release()
    rep = compare(rows_u, rows_m)
    vm = merged / "vpd_merge.json"
    rep.update({"schema": SCHEMA, "start": start, "adapter": str(adapter),
                "adapter_sha256": sha256_file(adapter / "adapter.pt"), "adapter_phase": meta.get("phase"),
                "init": init, "lora": info, "merged_dir": str(merged),
                "merged": json.loads(vm.read_text()) if vm.is_file() else None,
                "merged_vpd_merge_sha256": sha256_file(vm) if vm.is_file() else None,
                "demos": str(demos), "demos_sha256": sha256_file(demos), "demo_ids": [r.get("id") for r in records],
                "seconds": round(time.time() - t0, 1), "score_seconds": {"unmerged": round(t_u, 1),
                                                                         "merged": round(t_m, 1)},
                "torch": torch.__version__, "device": str(device), "cuda_allocated_gb": mem,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return rep


def write_json(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, p)


def main(argv=None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--demos", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS") or 2))
    args = ap.parse_args(argv)
    if args.n < 1:
        ap.error("--n must be >= 1")
    import torch

    torch.set_num_threads(args.threads)
    rep = run(args.start, args.adapter, args.merged, args.demos, args.n, args.device)
    write_json(args.out, rep)
    print(json.dumps({k: rep[k] for k in ("mean_abs_dce", "max_abs_dce", "reply_agreement", "n_replies", "pass")}),
          flush=True)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
