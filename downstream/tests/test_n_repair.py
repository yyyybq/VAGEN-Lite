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


DARK = np.full((16, 16, 3), 4, np.uint8)


def test_drop_can_take_near_black_and_named_episodes_too(tmp_path):
    """2026-11-05 (the user's task book, item 2): near-black tiles (n_frames' broken rule) and instances judged broken
    by eye can go out too; without the options nothing changes."""
    d = tmp_path / "S"
    episode_dir(d, {"a": [SMOOTH, SMOOTH], "b": [SMOOTH, NOISE], "c": [SMOOTH, DARK], "d": [SMOOTH]})
    assert NR.drop(d) == {"dropped": 1, "kept": 4}  # colour noise only, as before
    assert NR.drop(d, broken=True, instances=["d", "no_strip"]) == {"dropped": 3, "kept": 1}
    assert [json.loads(x)["instance_id"] for x in (d / "episodes.jsonl").read_text().splitlines()] == ["a"]
    gone = {g["instance_id"]: g for g in map(json.loads, (d / "episodes_corrupted.jsonl").read_text().splitlines())}
    assert gone["c"]["broken"] == {"near_black": [1]} and gone["c"]["corrupted_tiles"] == 0
    assert gone["d"]["named"] is True and "broken" not in gone["d"]
    assert gone["no_strip"]["named"] is True and gone["no_strip"]["tiles"] == 0  # no strip to move
    assert sorted(p.name for p in (d / "frames_corrupted").iterdir()) == ["b.0.png", "c.0.png", "d.0.png"]
    assert NR.main(["drop", "--out", str(d), "--broken", "--instances", "a"]) == 0
    assert (d / "episodes.jsonl").read_text() == ""


N1_STUB = r"""#!/usr/bin/env bash
printf '%s|%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" "$5" "${N_VAL_ENV:-}" >> "$STUB_LOG"
"$STUB_PY" - "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
out, inst = Path(sys.argv[1]), Path(sys.argv[2])
have = {json.loads(x)["instance_id"] for x in (out / "episodes.jsonl").read_text().splitlines() if x.strip()}
with open(out / "episodes.jsonl", "a") as f:
    for line in inst.read_text().splitlines():
        i = json.loads(line)["instance_id"]
        if i not in have:
            f.write(json.dumps({"instance_id": i, "success": True}) + "\n")
            Image.fromarray(np.tile(np.arange(16)[None, :, None] * 8, (16, 1, 3)).astype(np.uint8)).save(out / "frames" / f"{i}.png")
EOF
"""


def test_n1_repair_replays_noise_near_black_and_named_episodes_per_set(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    for f in ("run_n1_repair.sh", "n_repair.py", "n_frames.py"):
        shutil.copy(ROOT / "n" / f, wt / "downstream/n" / f)
    (wt / "downstream/n/run_n_episodes.sh").write_text(N1_STUB)
    root = tmp_path / "n"
    for rel, ids in (("n1_20261031/instances_eval.jsonl", "abcd"), ("n1b_20261102/instances_final.jsonl", "ef")):
        (root / rel).parent.mkdir(parents=True)
        (root / rel).write_text("".join(json.dumps({"instance_id": i}) + "\n" for i in ids))
    episode_dir(root / "n1_eval_20261031/S", {"a": [SMOOTH], "b": [NOISE], "c": [SMOOTH, DARK], "d": [SMOOTH]})
    episode_dir(root / "n1b_eval_20261102/Q3S", {"e": [SMOOTH], "f": [SMOOTH]})
    named = tmp_path / "named.txt"
    named.write_text("n1b_eval_20261102 Q3S f\nn1_eval_20261031 Q3S a\n")
    env = {k: v for k, v in os.environ.items() if k != "N_VAL_ENV"}
    env.update(N_VLM_PY=sys.executable, STUB_PY=sys.executable, STUB_LOG=str(tmp_path / "log"), N1_ROOT=str(root),
               N_VAL_ENV_QWEN3="/q3.yaml")
    for model in ("S", "Q3S"):
        res = subprocess.run(["bash", str(wt / "downstream/n/run_n1_repair.sh"), model, "4", str(named)],
                             capture_output=True, text=True, env=env, timeout=120)
        assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "log").read_text().split() == [
        f"{root}/n1_eval_20261031/S|{root}/n1_20261031/instances_eval.jsonl|S|4|0|",
        f"{root}/n1b_eval_20261102/Q3S|{root}/n1b_20261102/instances_final.jsonl|Q3S|4|0|/q3.yaml"]
    s = root / "n1_eval_20261031/S"
    log = (s / "n1_repair.log").read_text()
    assert 'n1_eval_20261031 S drop: {"dropped": 2, "kept": 3} (named: none)' in log
    assert "replay rc=0: 5 episodes of 4" in log  # the stub's row for the strip-less one stays
    assert sorted(json.loads(x)["instance_id"] for x in (s / "episodes_corrupted.jsonl").read_text().splitlines()) == ["b", "c"]
    scan = list(json.loads((s / "frames_scan_repaired.json").read_text())["evals"].values())[0]
    assert scan["corrupted_tiles"] == 0 and scan["broken"] == {} and not (s / ".scan").exists()
    q = root / "n1b_eval_20261102/Q3S"
    assert '(named: f)' in (q / "n1_repair.log").read_text()
    assert [json.loads(x)["instance_id"] for x in (q / "episodes_corrupted.jsonl").read_text().splitlines()] == ["f"]
    assert not (root / "n1_eval_20261031/Q3S").exists()  # no folder for this model in the first set: skipped
