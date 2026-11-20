"""q38: LLaMA-Factory main's SFT entry with this line's callback, run by accelerate in every process:

    accelerate launch --config_file vagen/q38/configs/sft/accelerate_fsdp_qwen35.yaml -m vagen.q38.sft.lf_launcher train.yaml

(sco_interiorgs_v10_train.sh lf_train with VAGEN_LITE_MODEL_TYPE qwen3_5: vagen/models/model_types.py names this
module and that config).  It is ``llamafactory.launcher``'s own ``__main__`` (``run_exp()``) with one callback added;
the training itself (data, model, loss, optimizer, schedule, saves) is LLaMA-Factory's, unchanged.

The callback (Q38SFTCallback), never raising into the trainer:
- per optimizer step, every rank: the step's seconds (on_step_begin -> on_step_end) and the time since the previous
  step's end, torch's peak allocated / reserved GPU memory of the step (GiB, reset after each step), the process RSS;
  rank 0 adds the machine's used memory (MemTotal - MemAvailable) and the cgroup's memory.current.  One JSON line per
  step in <metrics dir>/rank<R>.jsonl;
- local rank 0 runs vagen.q38.host_mem_log's sampler every VAGEN_Q38_SFT_HOST_MEM_S seconds (default 10; 0: off) into
  <metrics dir>/host_mem_sft-<UTC>.jsonl: the peaks between steps (model loading, the fp32 gather of a save);
- after each save, rank 0 starts a bf16 copy of the checkpoint in a background process (vagen.q38.sft.to_bf16,
  nice 10, 4 threads), VAGEN_Q38_SFT_BF16 (or "bf16_copy" of the folder's q38_sft.json, see apply_folder_settings):
    sibling  (default) <output_dir>_bf16/checkpoint-<step>; the fp32 checkpoint stays;
    in_place the checkpoint's fp32 shards are replaced by the verified bf16 ones (deletes fp32; only when asked);
    off      nothing;
  rank 0 waits for the copies before the process exits (after LLaMA-Factory's final save_model) and records each
  one's exit status and log in <metrics dir>/bf16_copies.json.
The metrics dir is VAGEN_Q38_SFT_METRICS_DIR, default <output_dir>/../sft_metrics.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import TrainerCallback

GIB = float(1 << 30)
BF16_MODES = ("sibling", "in_place", "off")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rss_gb() -> Optional[float]:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / (1024.0 * 1024.0), 2)
    except OSError:
        pass
    return None


def _machine_gb() -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"used_gb": None, "cgroup_current_gb": None}
    try:
        info = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l}
        out["used_gb"] = round((info["MemTotal"] - info["MemAvailable"]) / (1024.0 * 1024.0), 1)
    except (OSError, KeyError, ValueError, IndexError):
        pass
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            out["cgroup_current_gb"] = round(int(Path(p).read_text().strip()) / GIB, 1)
            break
        except (OSError, ValueError):
            continue
    return out


def bf16_mode() -> str:
    mode = os.environ.get("VAGEN_Q38_SFT_BF16", "sibling").strip() or "sibling"
    if mode not in BF16_MODES:
        raise ValueError(f"VAGEN_Q38_SFT_BF16 must be one of {BF16_MODES}, not {mode!r}")
    return mode


def bf16_command(ckpt: Path, mode: str) -> List[str]:
    cmd = ["nice", "-n", "10", sys.executable, "-B", "-m", "vagen.q38.sft.to_bf16", "convert", str(ckpt)]
    if mode == "in_place":
        cmd.append("--in-place")
    else:
        cmd.append(str(Path(str(ckpt.parent) + "_bf16") / ckpt.name))
    return cmd


class Q38SFTCallback(TrainerCallback):
    def __init__(self, metrics_dir: Optional[Path] = None, rank: Optional[int] = None, local_rank: Optional[int] = None):
        self.metrics_dir = metrics_dir
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else local_rank
        self.mode = bf16_mode()
        self.t_begin: Optional[float] = None
        self.t_last_end: Optional[float] = None
        self.copies: List[Dict[str, Any]] = []
        self.host_logger = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _dir(self, args) -> Path:
        if self.metrics_dir is None:
            env = os.environ.get("VAGEN_Q38_SFT_METRICS_DIR")
            self.metrics_dir = Path(env) if env else Path(args.output_dir).resolve().parent / "sft_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        return self.metrics_dir

    def _write(self, args, rec: Dict[str, Any]) -> None:
        try:
            with open(self._dir(args) / f"rank{self.rank}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # never into the trainer
            print(f"[q38 sft callback] metrics write failed: {e!r}", file=sys.stderr, flush=True)

    @staticmethod
    def _gpu() -> Dict[str, Optional[float]]:
        try:
            import torch

            if torch.cuda.is_available():
                out = {"max_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
                       "max_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2)}
                torch.cuda.reset_peak_memory_stats()
                return out
        except Exception:
            pass
        return {"max_allocated_gib": None, "max_reserved_gib": None}

    # ------------------------------------------------------------------ trainer events
    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import torch, transformers, llamafactory  # noqa: E401

            rec = {"event": "train_begin", "utc": utc(), "rank": self.rank, "bf16_copy": self.mode,
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "llamafactory": getattr(llamafactory, "__version__", None), "max_steps": state.max_steps,
                   "rss_gb": _rss_gb(), **self._gpu()}
            if self.rank == 0:
                rec.update(_machine_gb())
            self._write(args, rec)
            interval = float(os.environ.get("VAGEN_Q38_SFT_HOST_MEM_S", "10") or 0)
            if self.local_rank == 0 and interval > 0 and self.host_logger is None:
                from vagen.q38.host_mem_log import HostMemLogger

                self.host_logger = HostMemLogger(self._dir(args) / f"host_mem_sft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl", interval)
                self.host_logger.start()
        except Exception as e:
            print(f"[q38 sft callback] train_begin: {e!r}", file=sys.stderr, flush=True)
        self.t_last_end = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self.t_begin = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        rec = {"event": "step", "step": state.global_step, "epoch": state.epoch, "utc": utc(), "rank": self.rank,
               "step_s": round(now - self.t_begin, 3) if self.t_begin else None,
               "since_last_step_end_s": round(now - self.t_last_end, 3) if self.t_last_end else None,
               "rss_gb": _rss_gb(), **self._gpu()}
        if self.rank == 0:
            rec.update(_machine_gb())
        self.t_last_end = now
        self._write(args, rec)

    def on_save(self, args, state, control, **kwargs):
        if self.rank != 0:
            return
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        rec = {"event": "save", "step": state.global_step, "utc": utc(), "checkpoint": str(ckpt), **_machine_gb()}
        try:
            rec["checkpoint_bytes"] = sum(p.stat().st_size for p in ckpt.iterdir() if p.is_file())
        except OSError:
            rec["checkpoint_bytes"] = None
        self._write(args, rec)
        if self.mode == "off":
            return
        try:
            log = self._dir(args) / f"bf16_{ckpt.name}.log"
            env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       NUMEXPR_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
            cmd = bf16_command(ckpt, self.mode)
            proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
            with self.lock:
                self.copies.append({"checkpoint": str(ckpt), "command": cmd, "log": str(log), "started": utc(),
                                    "_proc": proc})
        except Exception as e:
            print(f"[q38 sft callback] bf16 copy of {ckpt} not started: {e!r}", file=sys.stderr, flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        self._write(args, {"event": "train_end", "utc": utc(), "rank": self.rank, "step": state.global_step,
                           "rss_gb": _rss_gb(), **self._gpu(), **(_machine_gb() if self.rank == 0 else {})})

    # ------------------------------------------------------------------ after run_exp
    def finish(self, args_output_dir: Optional[str] = None, timeout_s: float = 6 * 3600) -> bool:
        """Rank 0: wait for the bf16 copies and record them; stop the host-memory sampler.  True if all exited 0."""
        ok = True
        deadline = time.time() + timeout_s
        with self.lock:
            copies = list(self.copies)
        for c in copies:
            proc = c.pop("_proc")
            try:
                c["exit_status"] = proc.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                c["exit_status"] = "timeout"
            c["ended"] = utc()
            ok = ok and c["exit_status"] == 0
        if self.host_logger is not None:
            try:
                self.host_logger.stop()
            except Exception:
                pass
        if copies and self.metrics_dir is not None:
            (self.metrics_dir / "bf16_copies.json").write_text(json.dumps(copies, indent=1) + "\n")
        return ok


FOLDER_SETTINGS = "q38_sft.json"  # next to train.yaml: {"bf16_copy": ..., "host_mem_s": ..., "metrics_dir": ...}


def apply_folder_settings(argv: List[str]) -> Dict[str, Any]:
    """The LLaMA-Factory folder's q38_sft.json (written by vagen.q38.sft.lf_data) as defaults of the VAGEN_Q38_SFT_*
    variables; a variable already set wins (job_sco does not pass these on, so the folder is how a job gets them)."""
    if len(argv) < 2 or not argv[1].endswith((".yaml", ".yml")):
        return {}
    path = Path(argv[1]).resolve().parent / FOLDER_SETTINGS
    if not path.is_file():
        return {}
    cfg = json.loads(path.read_text())
    for key, var in (("bf16_copy", "VAGEN_Q38_SFT_BF16"), ("host_mem_s", "VAGEN_Q38_SFT_HOST_MEM_S"),
                     ("metrics_dir", "VAGEN_Q38_SFT_METRICS_DIR")):
        if key in cfg and cfg[key] is not None and not os.environ.get(var):
            os.environ[var] = str(cfg[key])
    return cfg


def main() -> None:
    from llamafactory.train.tuner import run_exp

    apply_folder_settings(sys.argv)
    cb = Q38SFTCallback()
    run_exp(callbacks=[cb])
    ok = cb.finish()
    if not ok:
        print("[q38 sft] a bf16 copy failed; see sft_metrics/bf16_copies.json", file=sys.stderr, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
