"""q38: periodic host-memory log of a trainer job (DESIGN_ASYNC 5.4 Z1).

Smoke Z1 has two host-memory pass lines: the machine's peak used memory (provisional threshold 1,700 GB; pear's r6 was
killed at 1839/1920 GB by Ray's memory monitor, STACK_SUPPORT 4) and each rank's RSS.  The trainer's own
``actor/perf/cpu_memory_used_gb`` is psutil's ``used`` (verl/verl/workers/engine_workers.py:222), sampled once an
update on one rank; it leaves out shared memory (/dev/shm, Ray's object store) and the peaks between updates (model
loading, checkpoint saves).  This module samples the whole machine from the job's main process instead:

- ``/proc/meminfo``: ``used_gb`` = MemTotal - MemAvailable (what Ray's memory monitor compares against its threshold),
  ``psutil_used_gb`` = MemTotal - MemFree - Buffers - Cached - SReclaimable (psutil's ``used``, the trainer's metric),
  Shmem, and the rest in GB;
- the cgroup's own counter when there is one (``memory.current`` / ``memory.max`` of cgroup v2, or v1's
  ``memory.usage_in_bytes`` / ``memory.limit_in_bytes``);
- every process's RSS from ``/proc/<pid>/status`` (VmRSS, RssAnon, RssShmem), summed by kind (Ray's process titles:
  ``ray::WorkerDict...`` = the FSDP ranks, vLLM, the agent-loop workers, TransferQueue storage, Ray's own daemons) and
  listed in full for the WorkerDict processes (the per-rank RSS) plus the largest others.

One JSON line per sample is appended to ``<run>/logs/host_mem_rl-<UTC start>.jsonl`` (a new file per trainer job, so
a resumed run does not overwrite the first one), each line with the running peaks.  Switched on by the config key
``vagen.host_mem_log.interval_s`` (seconds, > 0); absent or 0: nothing runs.  Reading /proc costs a few milliseconds a
sample; the thread is a daemon and never raises into the trainer (a failed sample is written as an ``error`` line).
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

KB_PER_GB = 1024.0 * 1024.0  # /proc reports kB (KiB); GB here = GiB, as psutil / verl's metric divide by 1024**3
TOP_OTHERS = 12

# process kind: (title prefixes, substrings of the command line).  Ray sets a worker's title to
# "ray::<Class>.<method>" (the FSDP ranks are "ray::WorkerDict...", pear r6's OOM report, STACK_SUPPORT 4); vLLM's
# engine processes call themselves "VLLM::EngineCore" / "VLLM::Worker..."
KINDS = (
    ("worker_dict", ("ray::WorkerDict",), ()),
    ("vllm", ("ray::vLLMHttpServer", "ray::vLLM", "VLLM::", "ray::RayWorkerWrapper"), ("EngineCore",)),
    ("agent_loop", ("ray::AgentLoopWorker", "ray::AgentLoop"), ()),
    ("transfer_queue", ("ray::SimpleStorageUnit", "ray::TransferQueue", "ray::StorageUnit"), ()),
    ("task_runner", ("ray::Q38TaskRunner", "ray::TaskRunner"), ()),
    ("driver", (), ("vagen.q38.ppo_v1",)),
    ("ray_core", ("ray::IDLE", "ray::_"), ("raylet", "gcs_server", "/ray/core/", "dashboard", "log_monitor",
                                           "ray_client_server", "runtime_env")),
)


def classify(title: str) -> str:
    for kind, prefixes, substrings in KINDS:
        if any(title.startswith(p) for p in prefixes) or any(s in title for s in substrings):
            return kind
    return "other"


def _kb_field(text: str, name: str) -> Optional[int]:
    for line in text.splitlines():
        if line.startswith(name + ":"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def read_meminfo(path: str = "/proc/meminfo") -> Dict[str, float]:
    """MemTotal, MemAvailable, ... in GB, plus used_gb (total - available) and psutil_used_gb."""
    text = Path(path).read_text()
    kb = {}
    for name in ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SReclaimable", "Shmem", "SwapTotal",
                 "SwapFree", "AnonPages", "Mlocked"):
        v = _kb_field(text, name)
        if v is not None:
            kb[name] = v
    out = {f"{k}_gb": round(v / KB_PER_GB, 3) for k, v in kb.items()}
    if "MemTotal" in kb and "MemAvailable" in kb:
        out["used_gb"] = round((kb["MemTotal"] - kb["MemAvailable"]) / KB_PER_GB, 3)
    if all(k in kb for k in ("MemTotal", "MemFree", "Buffers", "Cached")):
        used = kb["MemTotal"] - kb["MemFree"] - kb["Buffers"] - kb["Cached"] - kb.get("SReclaimable", 0)
        out["psutil_used_gb"] = round(used / KB_PER_GB, 3)
    return out


def read_cgroup(root: str = "/sys/fs/cgroup") -> Dict[str, float]:
    """The cgroup's memory counter in GB (v2 memory.current / memory.max, else v1), {} when there is none."""
    r = Path(root)
    pairs = (("memory.current", "memory.max"), ("memory/memory.usage_in_bytes", "memory/memory.limit_in_bytes"))
    for cur, lim in pairs:
        try:
            current = int((r / cur).read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        out = {"current_gb": round(current / 1024 ** 3, 3)}
        try:
            raw = (r / lim).read_text().split()[0]
            if raw != "max" and int(raw) < 1 << 60:
                out["limit_gb"] = round(int(raw) / 1024 ** 3, 3)
        except (OSError, ValueError, IndexError):
            pass
        return out
    return {}


def _title(pid_dir: Path) -> str:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return ""
    parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
    if not parts:
        try:
            return "[" + (pid_dir / "comm").read_text().strip() + "]"
        except OSError:
            return ""
    return " ".join(parts)[:160]


def scan_processes(proc: str = "/proc") -> List[dict]:
    """Every process with a resident set: pid, rss_gb (VmRSS), anon_gb, shmem_gb, kind, title."""
    rows = []
    for d in Path(proc).iterdir():
        if not d.name.isdigit():
            continue
        try:
            status = (d / "status").read_text()
        except OSError:  # gone, or not ours to read
            continue
        rss = _kb_field(status, "VmRSS")
        if not rss:
            continue
        title = _title(d)
        rows.append({"pid": int(d.name), "rss_gb": round(rss / KB_PER_GB, 3),
                     "anon_gb": round((_kb_field(status, "RssAnon") or 0) / KB_PER_GB, 3),
                     "shmem_gb": round((_kb_field(status, "RssShmem") or 0) / KB_PER_GB, 3),
                     "kind": classify(title), "title": title})
    rows.sort(key=lambda r: -r["rss_gb"])
    return rows


def sample(proc: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> dict:
    """One record: meminfo, cgroup, RSS by kind, the WorkerDict processes, the largest others."""
    record = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "t": round(time.time(), 1),
              "mem": read_meminfo(os.path.join(proc, "meminfo")), "cgroup": read_cgroup(cgroup_root)}
    procs = scan_processes(proc)
    by_kind: Dict[str, dict] = {}
    for p in procs:
        k = by_kind.setdefault(p["kind"], {"n": 0, "rss_gb": 0.0, "anon_gb": 0.0})
        k["n"] += 1
        k["rss_gb"] = round(k["rss_gb"] + p["rss_gb"], 3)
        k["anon_gb"] = round(k["anon_gb"] + p["anon_gb"], 3)
    record["by_kind"] = by_kind
    record["rss_sum_gb"] = round(sum(p["rss_gb"] for p in procs), 3)
    record["anon_sum_gb"] = round(sum(p["anon_gb"] for p in procs), 3)
    record["worker_dict"] = [[p["pid"], p["rss_gb"], p["anon_gb"], p["title"][:80]] for p in procs
                             if p["kind"] == "worker_dict"]
    record["top_other"] = [[p["pid"], p["rss_gb"], p["anon_gb"], p["kind"], p["title"][:80]] for p in procs
                           if p["kind"] != "worker_dict"][:TOP_OTHERS]
    return record


class HostMemLogger(threading.Thread):
    """Daemon thread: a sample every ``interval_s`` seconds, one JSON line each, with the running peaks."""

    def __init__(self, path: Path, interval_s: float, proc: str = "/proc", cgroup_root: str = "/sys/fs/cgroup"):
        super().__init__(name="q38-host-mem-log", daemon=True)
        self.path = Path(path)
        self.interval_s = float(interval_s)
        self.proc = proc
        self.cgroup_root = cgroup_root
        self._stop_event = threading.Event()
        self.samples = 0
        self.peak = {"used_gb": 0.0, "psutil_used_gb": 0.0, "Shmem_gb": 0.0, "cgroup_gb": 0.0,
                     "worker_dict_rss_max_gb": 0.0, "worker_dict_rss_sum_gb": 0.0, "used_utc": None}

    def _update_peak(self, rec: dict) -> None:
        mem, peak = rec.get("mem", {}), self.peak
        if mem.get("used_gb", 0.0) > peak["used_gb"]:
            peak["used_gb"], peak["used_utc"] = mem["used_gb"], rec["utc"]
        for key in ("psutil_used_gb", "Shmem_gb"):
            peak[key] = max(peak[key], mem.get(key, 0.0))
        peak["cgroup_gb"] = max(peak["cgroup_gb"], rec.get("cgroup", {}).get("current_gb", 0.0))
        wd = [row[1] for row in rec.get("worker_dict", [])]
        if wd:
            peak["worker_dict_rss_max_gb"] = max(peak["worker_dict_rss_max_gb"], max(wd))
            peak["worker_dict_rss_sum_gb"] = max(peak["worker_dict_rss_sum_gb"], round(sum(wd), 3))

    def sample_once(self) -> dict:
        try:
            rec = sample(self.proc, self.cgroup_root)
            self._update_peak(rec)
            rec["peak"] = dict(self.peak)
        except Exception as e:  # never into the trainer
            rec = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "error": f"{type(e).__name__}: {e}"}
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except OSError:
            pass
        self.samples += 1
        return rec

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.interval_s)

    def stop(self, final_sample: bool = True) -> None:
        self._stop_event.set()
        if final_sample:
            self.sample_once()


def log_path(run_dir: Path, now: Optional[datetime] = None) -> Path:
    now = now or datetime.now(timezone.utc)
    return Path(run_dir) / "logs" / f"host_mem_rl-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"


def interval_from_config(cfg) -> float:
    """``vagen.host_mem_log.interval_s`` as a float; 0 when absent, null or not positive."""
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, "vagen.host_mem_log.interval_s", default=None)
    try:
        value = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        raise ValueError(f"vagen.host_mem_log.interval_s must be a number of seconds, got {value!r}")
    return value if value > 0 else 0.0


def start_from_config(cfg, run_dir) -> Optional[HostMemLogger]:
    """Start the logger when the config asks for it and the run directory is known; returns it (or None)."""
    interval = interval_from_config(cfg)
    if not interval or not run_dir:
        return None
    path = log_path(Path(run_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = HostMemLogger(path, interval)
    logger.start()
    print(f"[q38] host memory every {interval:g} s -> {path}", flush=True)
    return logger
