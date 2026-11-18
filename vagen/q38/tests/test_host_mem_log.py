"""vagen/q38/host_mem_log.py: the host-memory log of smoke Z1 (DESIGN_ASYNC 5.4) on a fake /proc tree.

Run (login node, 1 thread):
    OMP_NUM_THREADS=1 ... nice -n 19 python -B -m pytest vagen/q38/tests/test_host_mem_log.py -q -x -p no:cacheprovider
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vagen.q38 import host_mem_log as H
from vagen.q38.tests.sampler_fixtures import work_dir

GB_KB = 1024 * 1024  # kB in a GB (GiB)

MEMINFO = """\
MemTotal:       {total} kB
MemFree:        {free} kB
MemAvailable:   {avail} kB
Buffers:        {buffers} kB
Cached:         {cached} kB
SwapCached:            0 kB
Shmem:          {shmem} kB
SReclaimable:   {srecl} kB
SwapTotal:             0 kB
SwapFree:              0 kB
"""


def _fake_proc(root: Path, procs, *, total=1920, free=1000, avail=1300, buffers=10, cached=280, shmem=200,
               srecl=20):
    root.mkdir(parents=True, exist_ok=True)
    (root / "meminfo").write_text(MEMINFO.format(total=total * GB_KB, free=free * GB_KB, avail=avail * GB_KB,
                                                 buffers=buffers * GB_KB, cached=cached * GB_KB,
                                                 shmem=shmem * GB_KB, srecl=srecl * GB_KB))
    for pid, title, rss_gb, anon_gb, shm_gb in procs:
        d = root / str(pid)
        d.mkdir(exist_ok=True)
        (d / "status").write_text(f"Name:\tpython\nVmRSS:\t{int(rss_gb * GB_KB)} kB\nRssAnon:\t{int(anon_gb * GB_KB)} kB\n"
                                  f"RssFile:\t0 kB\nRssShmem:\t{int(shm_gb * GB_KB)} kB\n")
        (d / "cmdline").write_bytes(title.encode() + b"\0")
    (root / "self").mkdir(exist_ok=True)  # not a pid: skipped
    kthread = root / "2"
    kthread.mkdir(exist_ok=True)
    (kthread / "status").write_text("Name:\tkthreadd\n")  # no VmRSS: skipped
    (kthread / "cmdline").write_bytes(b"")


PROCS = [
    (101, "ray::WorkerDict.actor_update_actor", 180.0, 170.0, 5.0),
    (102, "ray::WorkerDict.actor_update_actor", 175.5, 165.0, 5.0),
    (103, "ray::WorkerDict", 20.0, 15.0, 1.0),
    (201, "VLLM::EngineCore", 8.0, 6.0, 0.5),
    (202, "ray::vLLMHttpServer.generate", 3.0, 2.0, 0.1),
    (301, "ray::AgentLoopWorkerTQ.generate_sequences", 2.0, 1.5, 0.0),
    (401, "ray::SimpleStorageUnit.put", 4.5, 4.0, 0.0),
    (501, "/usr/bin/python -m vagen.q38.ppo_v1 --config RL/ppo.yaml --endpoint 5", 1.0, 0.8, 0.0),
    (601, "/x/site-packages/ray/core/src/ray/raylet/raylet --flags", 0.5, 0.4, 0.0),
    (701, "bash launch.sh", 0.01, 0.01, 0.0),
]


def test_classify():
    assert H.classify("ray::WorkerDict.actor_compute_log_prob") == "worker_dict"
    assert H.classify("VLLM::Worker_TP0") == "vllm"
    assert H.classify("python -m vllm.v1.engine EngineCore_0") == "vllm"
    assert H.classify("ray::AgentLoopWorkerTQ.run") == "agent_loop"
    assert H.classify("ray::SimpleStorageUnit") == "transfer_queue"
    assert H.classify("ray::Q38TaskRunner.run") == "task_runner"
    assert H.classify("python -m vagen.q38.ppo_v1 --config x") == "driver"
    assert H.classify("/env/lib/python3.12/site-packages/ray/core/src/ray/gcs/gcs_server --x") == "ray_core"
    assert H.classify("ray::IDLE") == "ray_core"
    assert H.classify("nvidia-smi --query-gpu") == "other"


def test_meminfo_and_sample():
    with work_dir("hostmem") as d:
        proc, cg = Path(d) / "proc", Path(d) / "cgroup"
        _fake_proc(proc, PROCS)
        cg.mkdir()
        (cg / "memory.current").write_text(str(700 * 1024 ** 3) + "\n")
        (cg / "memory.max").write_text("max\n")
        mem = H.read_meminfo(str(proc / "meminfo"))
        assert mem["MemTotal_gb"] == 1920 and mem["used_gb"] == 620  # total - available
        assert mem["psutil_used_gb"] == 1920 - 1000 - 10 - 280 - 20  # psutil's used (verl's cpu_memory_used_gb)
        assert mem["Shmem_gb"] == 200
        rec = H.sample(str(proc), str(cg))
        assert rec["cgroup"] == {"current_gb": 700.0}
        assert rec["by_kind"]["worker_dict"]["n"] == 3
        assert rec["by_kind"]["worker_dict"]["rss_gb"] == pytest.approx(375.5)
        assert rec["by_kind"]["vllm"]["n"] == 2 and rec["by_kind"]["driver"]["n"] == 1
        assert rec["by_kind"]["ray_core"]["n"] == 1 and rec["by_kind"]["other"]["n"] == 1
        assert [r[0] for r in rec["worker_dict"]] == [101, 102, 103]  # largest first
        assert all(r[3] != "worker_dict" for r in rec["top_other"]) and rec["top_other"][0][0] == 201
        assert rec["rss_sum_gb"] == pytest.approx(sum(p[2] for p in PROCS), abs=0.01)
        json.dumps(rec)  # one JSON line


def test_cgroup_v1_and_absent():
    with work_dir("hostmem") as d:
        root = Path(d)
        assert H.read_cgroup(str(root)) == {}
        (root / "memory").mkdir()
        (root / "memory" / "memory.usage_in_bytes").write_text(str(3 * 1024 ** 3))
        (root / "memory" / "memory.limit_in_bytes").write_text(str(1 << 62))  # "unlimited"
        assert H.read_cgroup(str(root)) == {"current_gb": 3.0}
        (root / "memory" / "memory.limit_in_bytes").write_text(str(1800 * 1024 ** 3))
        assert H.read_cgroup(str(root)) == {"current_gb": 3.0, "limit_gb": 1800.0}


def test_logger_peaks_lines_and_errors():
    with work_dir("hostmem") as d:
        proc = Path(d) / "proc"
        _fake_proc(proc, PROCS, avail=1300)
        path = Path(d) / "logs" / "host_mem_rl-x.jsonl"
        path.parent.mkdir()
        log = H.HostMemLogger(path, 3600, proc=str(proc), cgroup_root=str(Path(d) / "none"))
        log.sample_once()
        _fake_proc(proc, PROCS[:1], avail=1100)  # used 820 GB; pids 102.. still there (dirs kept): same RSS
        log.sample_once()
        _fake_proc(proc, PROCS[:1], avail=1500)  # lower again: the peak stays
        rec = log.sample_once()
        assert rec["peak"]["used_gb"] == 820 and rec["mem"]["used_gb"] == 420
        assert rec["peak"]["worker_dict_rss_max_gb"] == 180.0
        assert rec["peak"]["worker_dict_rss_sum_gb"] == pytest.approx(375.5)
        (proc / "meminfo").unlink()  # a failed sample is an error line, not an exception
        bad = log.sample_once()
        assert "error" in bad
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(lines) == 4 and "error" in lines[-1] and lines[1]["peak"]["used_utc"] == lines[1]["utc"]


def test_start_from_config_thread():
    with work_dir("hostmem") as d:
        off = OmegaConf.create({"vagen": {}})
        assert H.interval_from_config(off) == 0.0 and H.start_from_config(off, d) is None
        zero = OmegaConf.create({"vagen": {"host_mem_log": {"interval_s": 0}}})
        assert H.start_from_config(zero, d) is None
        with pytest.raises(ValueError):
            H.interval_from_config(OmegaConf.create({"vagen": {"host_mem_log": {"interval_s": "often"}}}))
        on = OmegaConf.create({"vagen": {"host_mem_log": {"interval_s": 0.05}}})
        assert H.start_from_config(on, None) is None  # no run directory: nothing
        log = H.start_from_config(on, d)  # the real /proc of this machine
        try:
            assert log is not None and log.daemon
            assert log.path.parent == Path(d) / "logs" and log.path.name.startswith("host_mem_rl-")
            deadline = time.time() + 10
            while log.samples < 2 and time.time() < deadline:
                time.sleep(0.05)
        finally:
            log.stop()
            log.join(timeout=5)
        lines = [json.loads(x) for x in log.path.read_text().splitlines()]
        assert len(lines) >= 3 and all("error" not in x for x in lines)
        assert lines[-1]["mem"]["MemTotal_gb"] > 0 and lines[-1]["peak"]["used_gb"] > 0
        assert lines[-1]["by_kind"]  # at least this test's own process


def test_entry_starts_and_stops_the_logger():
    # vagen/q38/ppo_v1.py main: started after the contract check (run dir from its report), stopped after Ray
    src = (Path(H.__file__).parent / "ppo_v1.py").read_text()
    assert "host_mem_log.start_from_config(cfg, report[\"derived\"][\"run_dir\"])" in src
    assert "mem_log.stop()" in src
