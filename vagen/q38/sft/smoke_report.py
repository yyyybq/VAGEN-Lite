"""q38: the numbers and pass lines of a 27B SFT smoke (SMOKE_PLAN 2c), read-only from its run folder.

    python -m vagen.q38.sft.smoke_report RUN [--out RUN/smoke_report.json]

RUN is the job's run folder (job_sco --run): RUN/logs (the job log sft-*.log, sft_exit.json, gpu_mem_sft.csv) and
RUN/lf (the LLaMA-Factory folder: model/ with trainer_log.jsonl and checkpoint-*, model_bf16/, sft_metrics/ of
vagen.q38.sft.lf_launcher).  Prints one JSON report; writes it to --out when given.

Pass lines (all must hold):
  exit_0           sft_exit.json exit_status 0, no Traceback / out-of-memory in the job log;
  version_check    "Version checking has been disabled" in the log (datasets 5.0.1 under DISABLE_VERSION_CHECK=1);
  upcast           "Upcasting trainable params to float32" in the log;
  trainable        LF's "trainable params" line reports 26,940,837,120 trainable of 27,356,728,560 (no MTP);
  steps            trainer_log.jsonl reaches total_steps (400 records / 64 = 7 steps, the last one partial);
  loss_finite      every logged loss finite; loss_down: mean of the last 3 logged losses < mean of the first 3;
  eval             an eval_loss logged at the end of the epoch;
  checkpoint       model/checkpoint-<last> holds config.json, safetensors and trainer_state.json;
  bf16_copy        model_bf16/checkpoint-<last>/bf16_manifest.json verify ok, about half the fp32 bytes;
  gpu_memory       peak nvidia-smi memory per card < 81,559 MiB and torch's peak reserved < 79 GiB on every rank;
  host_memory      peak MemTotal - MemAvailable < 1,600 GB (P1_REPORT decision 7's line).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TRAINABLE = 26_940_837_120
ALL = 27_356_728_560
HOST_LINE_GB = 1600.0
CARD_MIB = 81559
RESERVED_LINE_GIB = 79.0


def jsonl(p: Path) -> List[dict]:
    if not p.is_file():
        return []
    out = []
    for l in p.read_text(errors="ignore").splitlines():
        try:
            out.append(json.loads(l))
        except ValueError:
            continue
    return out


def dir_gb(p: Path) -> Optional[float]:
    if not p.is_dir():
        return None
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9, 2)


def median(xs: List[float]) -> Optional[float]:
    return round(statistics.median(xs), 2) if xs else None


def report(run: Path) -> Dict[str, Any]:
    lf = run / "lf"
    model = lf / "model"
    logs = sorted((run / "logs").glob("sft-*.log"))
    text = "\n".join(p.read_text(errors="ignore") for p in logs)
    exit_rec = json.loads((run / "logs" / "sft_exit.json").read_text()) if (run / "logs" / "sft_exit.json").is_file() else {}
    r: Dict[str, Any] = {"run": str(run), "job_logs": [str(p) for p in logs], "exit": exit_rec}
    m = re.search(r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)", text)
    r["trainable_params"] = [int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))] if m else None
    tl = jsonl(model / "trainer_log.jsonl")
    losses = [x["loss"] for x in tl if "loss" in x and x.get("loss") is not None]
    evals = [x["eval_loss"] for x in tl if x.get("eval_loss") is not None]
    if not evals:
        evals = [float(v) for v in re.findall(r"'eval_loss': ([0-9.eE+-]+)", text)]
    r["trainer_log"] = {"entries": len(tl), "total_steps": tl[-1].get("total_steps") if tl else None,
                        "last_step": tl[-1].get("current_steps") if tl else None, "losses": losses, "eval_loss": evals,
                        "elapsed": tl[-1].get("elapsed_time") if tl else None}
    ranks = {}
    for p in sorted((lf / "sft_metrics").glob("rank*.jsonl")):
        rows = [x for x in jsonl(p) if x.get("event") == "step"]
        ranks[p.stem] = {"steps": len(rows), "step_s": [x["step_s"] for x in rows],
                         "max_allocated_gib": max((x["max_allocated_gib"] or 0) for x in rows) if rows else None,
                         "max_reserved_gib": max((x["max_reserved_gib"] or 0) for x in rows) if rows else None,
                         "rss_gb_max": max((x["rss_gb"] or 0) for x in rows) if rows else None}
    r0 = ranks.get("rank0", {})
    steady = r0.get("step_s", [])[1:-1] or r0.get("step_s", [])
    r["step_seconds"] = {"all_rank0": r0.get("step_s"), "median_steady": median(steady),
                         "mean_steady": round(sum(steady) / len(steady), 2) if steady else None}
    r["ranks"] = {k: {kk: vv for kk, vv in v.items() if kk != "step_s"} for k, v in ranks.items()}
    host = []
    for p in sorted((lf / "sft_metrics").glob("host_mem_sft-*.jsonl")):
        host += jsonl(p)
    peaks = [x.get("peak", {}) for x in host if "peak" in x]
    r["host_memory"] = {"samples": len(host), "used_gb_peak": max((p.get("used_gb", 0) for p in peaks), default=None),
                        "cgroup_gb_peak": max((p.get("cgroup_gb", 0) for p in peaks), default=None),
                        "shmem_gb_peak": max((p.get("Shmem_gb", 0) for p in peaks), default=None)}
    gpu: Dict[str, int] = {}
    csv = run / "logs" / "gpu_mem_sft.csv"
    if csv.is_file():
        for line in csv.read_text(errors="ignore").splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) == 3 and parts[2].isdigit():
                gpu[parts[1]] = max(gpu.get(parts[1], 0), int(parts[2]))
    r["nvidia_smi_peak_mib"] = gpu
    ckpts = sorted(model.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    last = ckpts[-1] if ckpts else None
    r["checkpoints"] = {p.name: dir_gb(p) for p in ckpts}
    r["final_model_gb"] = round(sum(f.stat().st_size for f in model.glob("*.safetensors")) / 1e9, 2) if model.is_dir() else None
    bf = (lf / "model_bf16" / last.name) if last else None
    bman = json.loads((bf / "bf16_manifest.json").read_text()) if bf and (bf / "bf16_manifest.json").is_file() else None
    r["bf16"] = {"dir": str(bf) if bf else None, "gb": dir_gb(bf) if bf else None,
                 "verify_ok": (bman or {}).get("verify", {}).get("ok"),
                 "worst_relative_error": (bman or {}).get("verify", {}).get("worst_relative_error"),
                 "write_seconds": (bman or {}).get("write_seconds"), "verify_seconds": (bman or {}).get("verify_seconds"),
                 "bytes_fp32": (bman or {}).get("source_weight_bytes"), "bytes_bf16": (bman or {}).get("bf16_weight_bytes")}
    copies = lf / "sft_metrics" / "bf16_copies.json"
    r["bf16_copies"] = json.loads(copies.read_text()) if copies.is_file() else None
    checks = {
        "exit_0": exit_rec.get("exit_status") == 0 and "Traceback" not in text and "OutOfMemoryError" not in text,
        "version_check": "Version checking has been disabled" in text,
        "upcast": "Upcasting trainable params to float32" in text,
        "trainable": r["trainable_params"] == [TRAINABLE, ALL],
        "steps": bool(tl) and tl[-1].get("current_steps") == tl[-1].get("total_steps"),
        "loss_finite": bool(losses) and all(math.isfinite(x) for x in losses),
        "loss_down": len(losses) >= 4 and sum(losses[-3:]) / 3 < sum(losses[:3]) / 3,
        "eval": bool(evals),
        "checkpoint": bool(last) and (last / "config.json").is_file() and (last / "trainer_state.json").is_file()
                      and any(last.glob("*.safetensors")),
        "bf16_copy": bool(bman) and bman["verify"]["ok"] and 0.45 < bman["bf16_weight_bytes"] / bman["source_weight_bytes"] < 0.55,
        "gpu_memory": bool(gpu) and max(gpu.values()) < CARD_MIB and bool(ranks)
                      and all((v["max_reserved_gib"] or 0) < RESERVED_LINE_GIB for v in ranks.values()),
        "host_memory": r["host_memory"]["used_gb_peak"] is not None and r["host_memory"]["used_gb_peak"] < HOST_LINE_GB,
    }
    r["checks"] = checks
    r["ok"] = all(checks.values())
    return r


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    r = report(a.run)
    if a.out:
        a.out.write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r, indent=1))
    print("ok", r["ok"])


if __name__ == "__main__":
    main(sys.argv[1:])
