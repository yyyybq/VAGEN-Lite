"""Corrupted episodes out and played again (downstream/n/n_repair.py, run_n2_repair.sh; 2026-11-03)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import n_repair as NR  # noqa: E402

SMOOTH = np.tile(np.arange(16)[None, :, None] * 8, (16, 1, 3)).astype(np.uint8)
NOISE = np.random.default_rng(0).integers(0, 256, (16, 16, 3)).astype(np.uint8)


def episode_dir(d: Path, frames: dict):
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "episodes.jsonl").write_text("".join(json.dumps({"instance_id": i, "success": False}) + "\n"
                                              for i in list(frames) + ["no_strip"]))
    for i, tiles in frames.items():
        Image.fromarray(np.concatenate(tiles, axis=1)).save(d / "frames" / f"{i}.png")


def test_drop_moves_corrupted_episodes_and_their_strips_aside(tmp_path):
    d = tmp_path / "eval/100"
    episode_dir(d, {"a": [SMOOTH, SMOOTH], "b": [SMOOTH, NOISE], "c": [NOISE]})
    assert NR.drop(d) == {"dropped": 2, "kept": 2}
    assert [json.loads(x)["instance_id"] for x in (d / "episodes.jsonl").read_text().splitlines()] == ["a", "no_strip"]
    gone = [json.loads(x) for x in (d / "episodes_corrupted.jsonl").read_text().splitlines()]
    assert [(g["instance_id"], g["corrupted_tiles"], g["tiles"]) for g in gone] == [("b", 1, 2), ("c", 1, 1)]
    assert sorted(p.name for p in (d / "frames_corrupted").iterdir()) == ["b.0.png", "c.0.png"]
    assert not (d / "frames/b.png").exists() and (d / "frames/a.png").exists()
    Image.fromarray(NOISE).save(d / "frames/b.png")  # replayed and broken again: kept apart under a new number
    (d / "episodes.jsonl").write_text((d / "episodes.jsonl").read_text() + json.dumps({"instance_id": "b"}) + "\n")
    assert NR.drop(d) == {"dropped": 1, "kept": 2} and (d / "frames_corrupted/b.1.png").exists()
    assert NR.drop(tmp_path / "nothing") == {"dropped": 0, "kept": 0}


EPISODES_STUB = r"""#!/usr/bin/env bash
printf '%s|%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" "${N_ADAPTER:-}" "${N_VAL_ENV:-}" >> "$STUB_LOG"
echo '{"instance_id": "b", "success": true}' >> "$1/episodes.jsonl"
"$STUB_PY" -c 'import sys; from PIL import Image; import numpy as np; Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(sys.argv[1])' "$1/frames/b.png"
"""


def test_repair_job_replays_only_tiers_with_corrupted_episodes(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    for f in ("run_n2_repair.sh", "n_repair.py", "n_frames.py"):
        shutil.copy(ROOT / "n" / f, wt / "downstream/n" / f)
    (wt / "downstream/n/run_n_episodes.sh").write_text(EPISODES_STUB)
    run = tmp_path / "run"
    episode_dir(run / "eval/100", {"a": [SMOOTH], "b": [NOISE]})
    episode_dir(run / "eval/300", {"a": [SMOOTH], "b": [SMOOTH]})
    env = {k: v for k, v in os.environ.items() if k != "N_VAL_ENV"}
    env.update(N_VLM_PY=sys.executable, STUB_PY=sys.executable, STUB_LOG=str(tmp_path / "log"),
               N_VAL_ENV_QWEN3="/q3.yaml")
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n2_repair.sh"), str(run), "Q3S", "/i.jsonl"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "log").read_text().split() == [
        f"{run}/eval/100|/i.jsonl|Q3S|4|{run}/adapters/100/adapter.pt|/q3.yaml"]
    log = (run / "n2.log").read_text()
    assert 'tier 100: {"dropped": 1, "kept": 2}' in log and 'tier 300: {"dropped": 0, "kept": 3}' in log
    scan = json.loads((run / "frames_scan_repaired.json").read_text())
    assert [v["corrupted_tiles"] for v in scan["evals"].values()] == [0]
