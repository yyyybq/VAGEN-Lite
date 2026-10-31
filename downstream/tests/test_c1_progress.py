"""C1's progress every 25k steps (downstream/train/c1_progress.py; the user's decision 1 of 2026-10-31): windows of a
run with a crash and a relaunch from an earlier full state, the stale rows of the crashed launch left out, the skip
counter restarting with each launch, memory from both sources, the card-hour projection."""
import datetime
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import c1_progress as CP  # noqa: E402


def stamp(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S.000")


def iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run(tmp_path: Path) -> Path:
    exp = tmp_path / "C1"
    (exp / "run").mkdir(parents=True)
    (exp / "job").mkdir()
    (exp / "experiment.json").write_text(json.dumps({"steps": 4000}))
    rows = []
    for s in range(100, 2501, 100):  # launch 0: 2 s a step, one skip in 500 steps; crashes after 2,500
        rows.append({"step": s, "time": 1000 + 2.0 * s, "action_dit_loss": 0.3, "text_loss": 0.2,
                     "text_loss_traj": 0.1, "text_loss_qa": 0.3 if s != 200 else float("nan"),
                     "mem_peak_gib_max": 76.5 if s == 2300 else 70.0, "text_too_long_total": s // 500})
    rows.append({"event": "not a step row"})
    t1 = 1000 + 2.0 * 2500 + 600
    for s in range(2100, 3101, 100):  # launch 1: resumed from the full state at 2,000, 2.5 s a step
        rows.append({"step": s, "time": t1 + 2.5 * (s - 2000), "action_dit_loss": 0.1, "text_loss": 0.4,
                     "text_loss_traj": 0.2, "text_loss_qa": 0.6, "mem_peak_gib_max": 74.0 if s == 2600 else 71.0,
                     "text_too_long_total": (s - 2000) // 400})
    (exp / "run/metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (exp / "gpu.csv").write_text(f"{stamp(2000)}, 0, 72000 MiB, 99 %\n{stamp(2500)}, 3, 60000 MiB, 99 %\n"
                                 f"{stamp(7000)}, 5, 79000 MiB, 99 %\nbad line\n")
    (exp / "job/worker.log").write_text(f"[launch] host=a date={iso(960)} uid=0\nnoise\n[launch] rc=1 date={iso(6000)}\n"
                                        f"[launch] host=b date={iso(6560)} uid=0\n")
    ref = tmp_path / "REF"
    (ref / "run").mkdir(parents=True)
    (ref / "run/metrics.jsonl").write_text("\n".join(json.dumps({"step": s, "time": float(s), "action_dit_loss": 0.2})
                                                     for s in range(100, 4001, 100)) + "\n")
    return exp


def test_windows_leave_out_the_restart_and_the_stale_rows(tmp_path):
    exp = make_run(tmp_path)
    now = 1000 + 2.0 * 2500 + 600 + 2.5 * 1100 + 50
    p = CP.progress(exp, every=1000, reference=tmp_path / "REF", eval_card_hours=65, approved=910, now=now)
    w0, w1, w2 = p["windows"]
    assert [w["window"] for w in p["windows"]] == [[0, 1000], [1000, 2000], [2000, 3000]]
    assert w0["s_per_step"] == pytest.approx(2.0) and w1["s_per_step"] == pytest.approx(2.0)
    assert w2["s_per_step"] == pytest.approx(2.5) and w2["s_per_step_median"] == pytest.approx(2.5)  # restart left out
    assert w2["action_dit_loss"] == pytest.approx(0.1)  # launch 0's rows past 2,000 are stale
    assert w2["mem_alloc_peak_gib"] == pytest.approx(74.0) and w2["launches"] == [1]
    assert w0["text_loss_qa"] == pytest.approx(0.3)  # a NaN row is left out of the mean
    assert w0["reference_action_dit_loss"] == pytest.approx(0.2)
    assert (w0["skipped"], w1["skipped"], w2["skipped"]) == (2, 2, 2)  # the counter restarts with launch 1
    assert w0["skipped_share"] == pytest.approx(2 / (4 * 1000 + 2))
    assert w0["mem_used_peak_gib"] == pytest.approx(72000 / 1024)  # the 79,000 MiB sample is in window 2's time
    assert w2["mem_used_peak_gib"] == pytest.approx(79000 / 1024)
    assert p["last_logged_step"] == 3100 and p["launches"] == 2 and p["job_runs"] == 2
    spent = 8 * ((6000 - 960) + (now - 6560)) / 3600
    assert p["card_hours_spent"] == pytest.approx(spent)
    assert p["card_hours_projected"] == pytest.approx(spent + 8 * 900 * 2.5 / 3600 + 65)
    md = CP.markdown(p)
    assert "| 2k–3k |" in md and "2.50" in md and "6.7%" in md


def test_counter_going_back_marks_a_launch_without_a_step_going_back(tmp_path):
    rows = [{"step": s, "time": 2.0 * s, "text_too_long_total": c} for s, c in
            [(100, 1), (200, 3), (300, 4), (400, 0), (500, 2)]]  # a new process at 400, right after the last row
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    kept = CP.kept(CP.read_rows(path))
    assert [r["_launch"] for r in kept] == [0, 0, 0, 1, 1]
    (w,) = CP.windows(kept, 500, [], [])
    assert w["skipped"] == 1 + 2 + 1 + 0 + 2  # launch 0: 1, +2, +1; launch 1 counts from 0: 0, +2; never negative


def test_main_writes_json_and_markdown_and_handles_a_run_not_started(tmp_path):
    exp = make_run(tmp_path)
    assert CP.main(["--exp", str(exp), "--every", "1000", "--reference", str(tmp_path / "REF")]) == 0
    p = json.loads((exp / "progress/progress.json").read_text())
    assert len(p["windows"]) == 3 and (exp / "progress/progress.md").read_text().startswith("# C1 进度")
    empty = tmp_path / "EMPTY"
    (empty / "run").mkdir(parents=True)
    (empty / "experiment.json").write_text(json.dumps({"steps": 200000}))
    q = CP.progress(empty, reference=tmp_path / "nothing")
    assert q["windows"] == [] and q["card_hours_projected"] is None and q["last_logged_step"] == 0
    assert "None" not in CP.markdown(q)
    assert math.isnan(CP.mean([None, float("nan")]))
