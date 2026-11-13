"""eval/gpu.py: nvidia-smi parsing and free-GPU picking on CAPTURED-STYLE sample output (no GPU here)."""

import ast
import os
import subprocess

import pytest

from conftest import ROOT
from eval import gpu

# --query-gpu=index,memory.used,memory.total,utilization.gpu,name,uuid --format=csv,noheader,nounits
SAMPLE_GPUS = """\
0, 1, 81920, 0, NVIDIA A100-SXM4-80GB, GPU-00000000-1111-2222-3333-444444444444
1, 30210, 81920, 97, NVIDIA A100-SXM4-80GB, GPU-aaaaaaaa-1111-2222-3333-444444444444
2, 2047, 81920, 9, NVIDIA A100-SXM4-80GB, GPU-bbbbbbbb-1111-2222-3333-444444444444
3, 2048, 81920, 0, NVIDIA A100-SXM4-80GB, GPU-cccccccc-1111-2222-3333-444444444444
4, 15, 81920, 10, NVIDIA A100-SXM4-80GB, GPU-dddddddd-1111-2222-3333-444444444444
5, [N/A], 81920, [N/A], NVIDIA A100-SXM4-80GB, GPU-eeeeeeee-1111-2222-3333-444444444444
"""
# the exact five-column query from the task description (no uuid)
SAMPLE_GPUS_5COL = "0, 523, 24564, 3, NVIDIA GeForce RTX 4090\n1, 20311, 24564, 88, NVIDIA GeForce RTX 4090\n"
# --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader,nounits
SAMPLE_APPS = """\
41234, GPU-aaaaaaaa-1111-2222-3333-444444444444, 30000, /home/other/.venv/bin/python
41300, GPU-aaaaaaaa-1111-2222-3333-444444444444, 200, /usr/bin/python3
"""


def test_parse_gpu_csv_six_columns():
    gpus = gpu.parse_gpu_csv(SAMPLE_GPUS)
    assert [g.index for g in gpus] == [0, 1, 2, 3, 4, 5]
    g1 = gpus[1]
    assert (g1.memory_used_mib, g1.memory_total_mib, g1.utilization_pct) == (30210.0, 81920.0, 97.0)
    assert g1.name == "NVIDIA A100-SXM4-80GB" and g1.uuid.startswith("GPU-aaaaaaaa")
    assert gpus[5].memory_used_mib is None and gpus[5].utilization_pct is None


def test_parse_gpu_csv_five_columns_and_header():
    gpus = gpu.parse_gpu_csv("index, memory.used [MiB], memory.total [MiB], utilization.gpu [%], name\n" + SAMPLE_GPUS_5COL)
    assert [(g.index, g.memory_used_mib, g.utilization_pct, g.uuid) for g in gpus] == [(0, 523.0, 3.0, None), (1, 20311.0, 88.0, None)]
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"


def test_parse_gpu_csv_rejects_garbage():
    with pytest.raises(gpu.GpuQueryError):
        gpu.parse_gpu_csv("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.")


def test_free_thresholds_are_strict_and_unknown_is_busy():
    free = {g.index: g.is_free() for g in gpu.parse_gpu_csv(SAMPLE_GPUS)}
    assert free == {0: True, 1: False, 2: True, 3: False, 4: False, 5: False}


def test_pick_free_gpus():
    gpus = gpu.parse_gpu_csv(SAMPLE_GPUS)
    assert gpu.pick_free_gpus(2, gpus=gpus) == [0, 2]
    assert gpu.pick_free_gpus(1, gpus=gpus, exclude=[0]) == [2]
    assert gpu.pick_free_gpus(1, max_mem_mib=4096, max_util=50, gpus=gpus, exclude=[0, 2]) == [3]
    with pytest.raises(gpu.NoFreeGpuError, match="need 3 free GPU"):
        gpu.pick_free_gpus(3, gpus=gpus)


def test_check_gpus_free():
    gpus = gpu.parse_gpu_csv(SAMPLE_GPUS)
    gpu.check_gpus_free([0, 2], gpus=gpus)
    with pytest.raises(gpu.NoFreeGpuError):
        gpu.check_gpus_free([0, 1], gpus=gpus)
    with pytest.raises(gpu.NoFreeGpuError):
        gpu.check_gpus_free([9], gpus=gpus)


def test_parse_compute_apps_and_describe():
    apps = gpu.parse_compute_apps_csv(SAMPLE_APPS)
    assert [(a.pid, a.used_memory_mib) for a in apps] == [(41234, 30000.0), (41300, 200.0)]
    assert gpu.parse_compute_apps_csv("") == []
    rows = gpu.describe(gpu.parse_gpu_csv(SAMPLE_GPUS), apps, owner=lambda pid: "other" if pid == 41234 else "me")
    assert rows[1]["free"] is False and [p["owner"] for p in rows[1]["processes"]] == ["other", "me"]
    assert rows[0]["free"] is True and rows[0]["processes"] == []


def test_query_uses_nvidia_smi_and_reports_failures():
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=SAMPLE_GPUS, stderr="")

    assert len(gpu.query_gpus(runner)) == 6
    assert calls[0][0] == "nvidia-smi" and calls[0][1].startswith("--query-gpu=index,memory.used,memory.total,utilization.gpu,name")

    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    with pytest.raises(gpu.GpuQueryError, match="not found"):
        gpu.query_gpus(missing)
    with pytest.raises(gpu.GpuQueryError, match="exited 9"):
        gpu.query_gpus(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 9, stdout="", stderr="driver mismatch"))


def test_wait_for_free_gpus_polls_then_times_out():
    busy = gpu.parse_gpu_csv("0, 30000, 81920, 99, A100, GPU-a\n")
    free = gpu.parse_gpu_csv("0, 10, 81920, 0, A100, GPU-a\n")
    seq = [busy, busy, free]
    sleeps = []
    got = gpu.wait_for_free_gpus(1, query=lambda: seq.pop(0), sleep=sleeps.append, log=lambda m: None, poll_s=5)
    assert got == [0] and sleeps == [5, 5]
    t = [0.0]
    with pytest.raises(gpu.NoFreeGpuError, match="gave up"):
        gpu.wait_for_free_gpus(1, query=lambda: busy, sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0],
                               timeout_s=10, poll_s=4, log=lambda m: None)


def test_cuda_env_pins_pci_order():
    assert gpu.cuda_env([3, 1]) == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "3,1"}
    assert gpu.cuda_env([])["CUDA_VISIBLE_DEVICES"] == ""


def test_gpu_module_never_signals_processes():
    tree = ast.parse(open(os.path.join(ROOT, "eval", "gpu.py"), encoding="utf-8").read())
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not names & {"kill", "killpg", "send_signal", "terminate", "signal"}
