"""GPU discovery and selection for evaluation launches. Read-only: nothing here signals any process.

Server 105 discipline (CLAUDE.md "GPU discipline"): check nvidia-smi before every launch; a GPU is free
when memory.used < 2048 MiB AND utilization.gpu < 10 %; use only free GPUs, always through an explicit
CUDA_VISIBLE_DEVICES; never touch other users' processes. ``--query-compute-apps`` is parsed for
ownership information only (who runs what on which board).

nvidia-smi numbers boards in PCI bus order, CUDA's default order is "fastest first"; ``cuda_env`` also
sets CUDA_DEVICE_ORDER=PCI_BUS_ID so that CUDA_VISIBLE_DEVICES=<nvidia-smi index> names the same board.

Status: written and unit-tested on captured nvidia-smi output (tests/test_eval_run_gpu.py); not yet run
on 105.

    python -m eval.gpu              # every GPU: free?, memory, utilization, compute processes and owners
    python -m eval.gpu --pick 2     # print two free GPU indices (comma-separated) or exit 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pwd
import subprocess
import sys
import time
from typing import Callable, Iterable, Optional, Sequence

GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total,utilization.gpu,name,uuid",
    "--format=csv,noheader,nounits",
)
APPS_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
    "--format=csv,noheader,nounits",
)
MAX_MEM_MIB = 2048
MAX_UTIL = 10


class GpuError(RuntimeError):
    pass


class GpuQueryError(GpuError):
    pass


class NoFreeGpuError(GpuError):
    pass


@dataclasses.dataclass(frozen=True)
class GpuInfo:
    index: int
    memory_used_mib: Optional[float]
    memory_total_mib: Optional[float]
    utilization_pct: Optional[float]
    name: str
    uuid: Optional[str] = None

    def is_free(self, max_mem_mib: float = MAX_MEM_MIB, max_util: float = MAX_UTIL) -> bool:
        """Unknown values ([N/A], [Not Supported]) count as busy."""
        return (
            self.memory_used_mib is not None
            and self.utilization_pct is not None
            and self.memory_used_mib < max_mem_mib
            and self.utilization_pct < max_util
        )


@dataclasses.dataclass(frozen=True)
class GpuProcess:
    pid: int
    gpu_uuid: str
    used_memory_mib: Optional[float]
    process_name: str


def _num(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_gpu_csv(text: str) -> list[GpuInfo]:
    """Parse ``--query-gpu=index,memory.used,memory.total,utilization.gpu,name[,uuid]`` csv/noheader/nounits."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("index"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            raise GpuQueryError(f"unexpected nvidia-smi --query-gpu line: {line!r}")
        uuid = None
        if len(parts) >= 6 and parts[-1].startswith(("GPU-", "MIG-")):
            uuid, name_parts = parts[-1], parts[4:-1]
        else:
            name_parts = parts[4:]
        out.append(
            GpuInfo(
                index=int(parts[0]),
                memory_used_mib=_num(parts[1]),
                memory_total_mib=_num(parts[2]),
                utilization_pct=_num(parts[3]),
                name=", ".join(name_parts),
                uuid=uuid,
            )
        )
    return out


def parse_compute_apps_csv(text: str) -> list[GpuProcess]:
    """Parse ``--query-compute-apps=pid,gpu_uuid,used_memory,process_name`` csv/noheader/nounits."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("pid") or line.startswith("No running"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            raise GpuQueryError(f"unexpected nvidia-smi --query-compute-apps line: {line!r}")
        out.append(GpuProcess(int(parts[0]), parts[1], _num(parts[2]), ", ".join(parts[3:])))
    return out


def _run(cmd: Sequence[str], runner: Callable = subprocess.run) -> str:
    try:
        res = runner(list(cmd), capture_output=True, text=True, timeout=60)
    except FileNotFoundError as e:
        raise GpuQueryError("nvidia-smi not found: is this server 105?") from e
    except subprocess.TimeoutExpired as e:
        raise GpuQueryError("nvidia-smi timed out") from e
    if res.returncode != 0:
        raise GpuQueryError(f"nvidia-smi exited {res.returncode}: {(res.stderr or res.stdout).strip()[:300]}")
    return res.stdout


def query_gpus(runner: Callable = subprocess.run) -> list[GpuInfo]:
    return parse_gpu_csv(_run(GPU_QUERY, runner))


def query_compute_apps(runner: Callable = subprocess.run) -> list[GpuProcess]:
    return parse_compute_apps_csv(_run(APPS_QUERY, runner))


def _table(gpus: Iterable[GpuInfo]) -> str:
    return "; ".join(f"GPU{g.index} {g.memory_used_mib} MiB {g.utilization_pct}%" for g in gpus)


def pick_free_gpus(
    n: int,
    max_mem_mib: float = MAX_MEM_MIB,
    max_util: float = MAX_UTIL,
    *,
    gpus: Optional[Sequence[GpuInfo]] = None,
    exclude: Iterable[int] = (),
) -> list[int]:
    """The ``n`` lowest-index free GPUs (not in ``exclude``); NoFreeGpuError if there are fewer."""
    gpus = query_gpus() if gpus is None else list(gpus)
    exclude = {int(i) for i in exclude}
    free = [g.index for g in sorted(gpus, key=lambda g: g.index) if g.index not in exclude and g.is_free(max_mem_mib, max_util)]
    if len(free) < n:
        raise NoFreeGpuError(
            f"need {n} free GPU(s) (memory < {max_mem_mib} MiB, utilization < {max_util}%), found {free}: {_table(gpus)}"
        )
    return free[:n]


def check_gpus_free(
    ids: Sequence[int], max_mem_mib: float = MAX_MEM_MIB, max_util: float = MAX_UTIL, *, gpus: Optional[Sequence[GpuInfo]] = None
) -> None:
    """Raise NoFreeGpuError unless every GPU in ``ids`` exists and is free."""
    gpus = query_gpus() if gpus is None else list(gpus)
    by_index = {g.index: g for g in gpus}
    bad = [i for i in ids if int(i) not in by_index or not by_index[int(i)].is_free(max_mem_mib, max_util)]
    if bad:
        raise NoFreeGpuError(f"GPU(s) {bad} not free (memory < {max_mem_mib} MiB, utilization < {max_util}%): {_table(gpus)}")


def wait_for_free_gpus(
    n: int,
    max_mem_mib: float = MAX_MEM_MIB,
    max_util: float = MAX_UTIL,
    *,
    poll_s: float = 60.0,
    timeout_s: Optional[float] = None,
    query: Callable[[], Sequence[GpuInfo]] = query_gpus,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr, flush=True),
) -> list[int]:
    """Poll until ``n`` GPUs are free. Only used when the caller explicitly asked to wait (--wait-gpu)."""
    deadline = None if timeout_s is None else clock() + timeout_s
    while True:
        try:
            return pick_free_gpus(n, max_mem_mib, max_util, gpus=query())
        except NoFreeGpuError as e:
            if deadline is not None and clock() >= deadline:
                raise NoFreeGpuError(f"gave up after {timeout_s:.0f} s: {e}") from e
            log(f"waiting for {n} free GPU(s): {e}")
            sleep(poll_s)


def cuda_env(ids: Sequence) -> dict[str, str]:
    """Environment that pins a child process to ``ids`` (nvidia-smi indices); empty ``ids`` hides all GPUs."""
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in ids)}


def process_owner(pid: int) -> Optional[str]:
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def describe(gpus: Sequence[GpuInfo], apps: Sequence[GpuProcess], max_mem_mib: float = MAX_MEM_MIB, max_util: float = MAX_UTIL,
             owner: Callable[[int], Optional[str]] = process_owner) -> list[dict]:
    """One dict per GPU with its free flag and the compute processes on it (pid, owner, memory, name)."""
    out = []
    for g in sorted(gpus, key=lambda g: g.index):
        procs = [
            {"pid": a.pid, "owner": owner(a.pid), "used_memory_mib": a.used_memory_mib, "process_name": a.process_name}
            for a in apps
            if g.uuid and a.gpu_uuid == g.uuid
        ]
        out.append({**dataclasses.asdict(g), "free": g.is_free(max_mem_mib, max_util), "processes": procs})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick", type=int, default=None, help="print N free GPU indices, exit 1 if there are fewer")
    ap.add_argument("--max-mem-mib", type=float, default=MAX_MEM_MIB)
    ap.add_argument("--max-util", type=float, default=MAX_UTIL)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        gpus = query_gpus()
        if args.pick is not None:
            print(",".join(str(i) for i in pick_free_gpus(args.pick, args.max_mem_mib, args.max_util, gpus=gpus)))
            return 0
        rows = describe(gpus, query_compute_apps(), args.max_mem_mib, args.max_util)
    except GpuError as e:
        print(f"eval.gpu: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows:
        print(f"GPU{r['index']} {'FREE' if r['free'] else 'busy'} {r['memory_used_mib']}/{r['memory_total_mib']} MiB "
              f"{r['utilization_pct']}% {r['name']}")
        for p in r["processes"]:
            print(f"    pid {p['pid']} {p['owner'] or '?'} {p['used_memory_mib']} MiB {p['process_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
