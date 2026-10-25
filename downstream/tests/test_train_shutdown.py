"""CPU test of downstream/train/train_entry.end_process_group: a failed NCCL teardown after the run is saved and
recorded must not fail the job (E1 B, 2026-10-25 03:47: training exited 1 and run_segmented.sh skipped the
evaluation of the final checkpoint)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import train_entry as te  # noqa: E402


class FakeDist:
    def __init__(self, fail=False, initialized=True):
        self.fail, self.initialized, self.calls = fail, initialized, []

    def is_initialized(self):
        return self.initialized

    def barrier(self):
        self.calls.append("barrier")

    def destroy_process_group(self):
        self.calls.append("destroy")
        if self.fail:
            raise RuntimeError("NCCL error in: NCCLUtils.cpp:133, unhandled cuda error; Cuda failure 'out of memory'")


def test_a_failed_teardown_ends_the_process_at_once_with_zero(capsys):
    exits, d = [], FakeDist(fail=True)
    te.end_process_group(d, exit_now=exits.append)
    assert d.calls == ["barrier", "destroy"] and exits == [0]
    assert "teardown failed after the run was saved and recorded (RuntimeError" in capsys.readouterr().err


def test_a_clean_teardown_returns_and_a_missing_group_is_left_alone():
    exits, d = [], FakeDist()
    te.end_process_group(d, exit_now=exits.append)
    assert d.calls == ["barrier", "destroy"] and exits == []
    d = FakeDist(initialized=False)
    te.end_process_group(d, exit_now=exits.append)
    assert d.calls == [] and exits == []
