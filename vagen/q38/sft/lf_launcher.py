"""q38: LLaMA-Factory main's SFT entry with this line's callback, run by accelerate in every process:

    accelerate launch --config_file vagen/q38/configs/sft/accelerate_fsdp_qwen35.yaml -m vagen.q38.sft.lf_launcher train.yaml

(sco_interiorgs_v10_train.sh lf_train with VAGEN_LITE_MODEL_TYPE qwen3_5: vagen/models/model_types.py names this
module and that config).  It is ``llamafactory.launcher``'s own ``__main__`` (``run_exp()``) with one callback added;
the training itself (data, model, loss, optimizer, schedule, saves) is LLaMA-Factory's, unchanged.

The callback (Q38SFTCallback), never raising into the trainer:
- per optimizer step, every rank: the step's seconds (on_step_begin -> on_step_end) and the time since the previous
  step's end, torch's peak allocated / reserved GPU memory of the step (GiB, reset after each step), the process RSS;
  rank 0 adds the machine's used memory (MemTotal - MemAvailable) and the cgroup's memory.current.  One JSON line per
  step in <metrics dir>/rank<R>.jsonl.  Each line also holds the allocated / reserved GPU memory at the step's end
  (allocated_gib / reserved_gib);
- per micro-batch of the first VAGEN_Q38_SFT_SUBSTEP_STEPS optimizer steps (default 1; 0: none), every rank: a
  "substep" line after each micro-batch but the step's last (on_substep_end) with the allocated / reserved memory and
  the step's peak so far (not reset), so a growth from micro-batch to micro-batch (FSDP no_sync's unsharded gradients)
  shows in the first step;
- at train_begin, every rank: the gradient accumulation settings the trainer uses -- TrainingArguments'
  accelerator_config.gradient_accumulation_kwargs and accelerate's GradientState.plugin_kwargs, whose
  "sync_each_batch" is what transformers' Trainer._run_epoch reads to skip FSDP's no_sync on the micro-batches before
  the step's last (transformers 5.10.4 trainer.py 1733-1743) -- and the CUDA allocator settings (the variables and
  torch's own reading); rank 0 prints them on one line of the job log too;
- VAGEN_Q38_SFT_MEM_HISTORY=1 (or "mem_history" of q38_sft.json; default off): rank 0 records the CUDA allocator's
  history from train_begin (python stacks, at most VAGEN_Q38_SFT_MEM_HISTORY_ENTRIES events, default 100000) and
  writes it to <metrics dir>/mem_history_rank0_step1.pickle at the end of the first optimizer step, then stops
  recording; a CUDA out-of-memory error while it records writes mem_history_rank0_oom.pickle instead (view either at
  https://pytorch.org/memory_viz);
- a CUDA out-of-memory error out of the training (main()): every rank writes an "oom" line with the allocated /
  reserved memory before the error goes on;
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

The CUDA caching allocator gets expandable segments (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True), set when this
module is imported, before anything imports torch, unless PYTORCH_ALLOC_CONF or PYTORCH_CUDA_ALLOC_CONF is set
already (the 27B smoke's OOM, q38-smoke-27b-sft 2026-11-20: 4.4-6.8 GiB reserved but unallocated per rank).  Only
this line's SFT runs this module (model_types: qwen3_5); the other model types' SFT launchers are unchanged.
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
from types import SimpleNamespace
from typing import Any, Dict, List, MutableMapping, Optional

ALLOC_CONF = "expandable_segments:True"
ALLOC_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")  # torch >= 2.9 reads the first, then the second


def set_alloc_conf(environ: Optional[MutableMapping[str, str]] = None) -> Optional[str]:
    """PYTORCH_CUDA_ALLOC_CONF=ALLOC_CONF unless an allocator variable is set already; the value set (None: none)."""
    env = os.environ if environ is None else environ
    if any(env.get(v) for v in ALLOC_VARS):
        return None
    env["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF
    return ALLOC_CONF


ALLOC_CONF_SET = set_alloc_conf()  # before the transformers import below (which imports torch)

from transformers import TrainerCallback  # noqa: E402

GIB = float(1 << 30)
MEM_HISTORY_STEP1 = "mem_history_rank0_step1.pickle"
MEM_HISTORY_OOM = "mem_history_rank0_oom.pickle"
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


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)


def grad_acc_settings(args) -> Dict[str, Any]:
    """The gradient accumulation settings the trainer runs with: TrainingArguments' and accelerate's GradientState's
    (read from its shared state without creating one).  sync_each_batch is GradientState.plugin_kwargs'
    "sync_each_batch" (False when absent), the value transformers' Trainer._run_epoch reads."""
    out: Dict[str, Any] = {"gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
                           "args_gradient_accumulation_kwargs": None, "plugin_kwargs": None, "sync_each_batch": None}
    try:
        ac = getattr(args, "accelerator_config", None)
        kw = ac.get("gradient_accumulation_kwargs") if isinstance(ac, dict) else getattr(ac, "gradient_accumulation_kwargs", None)
        out["args_gradient_accumulation_kwargs"] = dict(kw) if kw is not None else None
        from accelerate.state import GradientState

        shared = GradientState._shared_state
        if "plugin_kwargs" in shared:
            out["plugin_kwargs"] = dict(shared["plugin_kwargs"] or {})
            out["sync_each_batch"] = bool(out["plugin_kwargs"].get("sync_each_batch", False))
    except Exception as e:
        out["error"] = repr(e)
    return out


def alloc_settings() -> Dict[str, Any]:
    out: Dict[str, Any] = {v: os.environ.get(v) for v in ALLOC_VARS}
    out["set_by_launcher"] = ALLOC_CONF_SET
    try:
        import torch

        out["torch"] = torch._C._accelerator_getAllocatorSettings()
    except Exception as e:
        out["torch"] = repr(e)
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
        self.args = None
        self.micro = 0  # micro-batches of the current optimizer step done
        try:
            self.substep_steps = int(os.environ.get("VAGEN_Q38_SFT_SUBSTEP_STEPS", "1") or 0)
        except ValueError:
            self.substep_steps = 1
        self.mem_history = False  # rank 0 is recording the allocator history

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
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:  # never into the trainer
            print(f"[q38 sft callback] metrics write failed: {e!r}", file=sys.stderr, flush=True)

    @staticmethod
    def _gpu(reset: bool = True) -> Dict[str, Optional[float]]:
        """torch's peak allocated / reserved memory since the last reset (reset again unless reset=False) and the
        allocated / reserved memory now, GiB."""
        try:
            import torch

            if torch.cuda.is_available():
                out = {"max_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
                       "max_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2),
                       "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 2),
                       "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 2)}
                if reset:
                    torch.cuda.reset_peak_memory_stats()
                return out
        except Exception:
            pass
        return {"max_allocated_gib": None, "max_reserved_gib": None, "allocated_gib": None, "reserved_gib": None}

    def _start_mem_history(self) -> None:
        if self.rank != 0 or not _env_on("VAGEN_Q38_SFT_MEM_HISTORY"):
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            n = int(os.environ.get("VAGEN_Q38_SFT_MEM_HISTORY_ENTRIES", "100000") or 100000)
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=n)
            self.mem_history = True
            print(f"[q38 sft callback] rank 0 records the CUDA allocator history ({n} events)", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[q38 sft callback] memory history not started: {e!r}", file=sys.stderr, flush=True)

    def _dump_mem_history(self, name: str) -> Optional[str]:
        """Rank 0: the recorded history into <metrics dir>/name, then recording stops.  The path (None: nothing)."""
        if not self.mem_history:
            return None
        path = None
        try:
            import torch

            d = self._dir(self.args) if self.args is not None else self.metrics_dir
            if d is not None:
                path = str(Path(d) / name)
                torch.cuda.memory._dump_snapshot(path)
        except Exception as e:
            print(f"[q38 sft callback] memory history not written: {e!r}", file=sys.stderr, flush=True)
            path = None
        try:
            import torch

            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:
            pass
        self.mem_history = False
        return path

    # ------------------------------------------------------------------ trainer events
    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import torch, transformers, llamafactory  # noqa: E401

            rec = {"event": "train_begin", "utc": utc(), "rank": self.rank, "bf16_copy": self.mode,
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   "llamafactory": getattr(llamafactory, "__version__", None), "max_steps": state.max_steps,
                   "rss_gb": _rss_gb(), **self._gpu(), "grad_acc": grad_acc_settings(args),
                   "cuda_alloc": alloc_settings()}
            if self.rank == 0:
                rec.update(_machine_gb())
                print(f"[q38 sft callback] gradient accumulation {json.dumps(rec['grad_acc'], default=str)}; "
                      f"CUDA allocator {json.dumps(rec['cuda_alloc'], default=str)}", file=sys.stderr, flush=True)
            self._write(args, rec)
            interval = float(os.environ.get("VAGEN_Q38_SFT_HOST_MEM_S", "10") or 0)
            if self.local_rank == 0 and interval > 0 and self.host_logger is None:
                from vagen.q38.host_mem_log import HostMemLogger

                self.host_logger = HostMemLogger(self._dir(args) / f"host_mem_sft-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl", interval)
                self.host_logger.start()
        except Exception as e:
            print(f"[q38 sft callback] train_begin: {e!r}", file=sys.stderr, flush=True)
        self.args = args
        self._start_mem_history()
        self.t_last_end = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self.t_begin = time.time()
        self.micro = 0

    def on_substep_end(self, args, state, control, **kwargs):
        self.micro += 1
        if state.global_step >= self.substep_steps:
            return
        self._write(args, {"event": "substep", "step": state.global_step + 1, "micro_batch": self.micro,
                           "utc": utc(), "rank": self.rank,
                           "s_since_step_begin": round(time.time() - self.t_begin, 3) if self.t_begin else None,
                           **self._gpu(reset=False)})

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        rec = {"event": "step", "step": state.global_step, "epoch": state.epoch, "utc": utc(), "rank": self.rank,
               "step_s": round(now - self.t_begin, 3) if self.t_begin else None,
               "since_last_step_end_s": round(now - self.t_last_end, 3) if self.t_last_end else None,
               "micro_batches": self.micro + 1, "rss_gb": _rss_gb(), **self._gpu()}
        if self.rank == 0:
            rec.update(_machine_gb())
        self.t_last_end = now
        self._write(args, rec)
        if self.mem_history:
            path = self._dump_mem_history(MEM_HISTORY_STEP1)
            self._write(args, {"event": "mem_history", "step": state.global_step, "utc": utc(), "rank": self.rank,
                               "path": path})

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
    def on_exception(self, exc: BaseException) -> None:
        """main(): an error out of run_exp.  A CUDA OOM gets an "oom" line (every rank) and, on rank 0 while it
        records, the allocator history; never raises."""
        try:
            if not _is_oom(exc):
                return
            path = self._dump_mem_history(MEM_HISTORY_OOM)
            rec = {"event": "oom", "utc": utc(), "rank": self.rank, "error": str(exc)[:500],
                   "mem_history": path, **self._gpu(reset=False)}
            if self.args is not None or self.metrics_dir is not None:
                self._write(self.args if self.args is not None else SimpleNamespace(output_dir=None), rec)
        except Exception as e:
            print(f"[q38 sft callback] oom record: {e!r}", file=sys.stderr, flush=True)

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


FOLDER_SETTINGS = "q38_sft.json"  # next to train.yaml: {"bf16_copy": ..., "host_mem_s": ..., "metrics_dir": ..., "mem_history": ...}


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
                     ("metrics_dir", "VAGEN_Q38_SFT_METRICS_DIR"), ("mem_history", "VAGEN_Q38_SFT_MEM_HISTORY")):
        if key in cfg and cfg[key] is not None and not os.environ.get(var):
            os.environ[var] = str(cfg[key])
    return cfg


def main() -> None:
    from llamafactory.train.tuner import run_exp

    apply_folder_settings(sys.argv)
    cb = Q38SFTCallback()
    try:
        run_exp(callbacks=[cb])
    except BaseException as e:
        cb.on_exception(e)
        raise
    ok = cb.finish()
    if not ok:
        print("[q38 sft] a bf16 copy failed; see sft_metrics/bf16_copies.json", file=sys.stderr, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
