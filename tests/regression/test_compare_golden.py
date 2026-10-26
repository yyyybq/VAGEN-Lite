"""compare_golden.py, the tool of the cleanup's golden-output comparison: identical files, files that differ only in
wall-time fields, and real differences."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).with_name("compare_golden.py")


def _write(root: Path, files: dict) -> None:
    lines = []
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        lines.append(f"{hashlib.sha256(text.encode()).hexdigest()}  {name}")
    (root / "SHA256SUMS.txt").write_text("\n".join(sorted(lines, key=lambda l: l.split()[1])) + "\n")


def _run(a: Path, b: Path):
    return subprocess.run([sys.executable, str(TOOL), str(a), str(b)], capture_output=True, text=True)


def test_identical_and_wall_time_only(tmp_path):
    base = {"same.jsonl": '{"a": 1}\n',
            "scene.json": json.dumps({"x": 1, "timings": {"load": 0.2}, "summary": {"seconds": 3.0, "n": 2}}),
            "per_scene.csv": "scene_id,n_tasks,seconds\n0001,14,361.65\n"}
    after = dict(base)
    after["scene.json"] = json.dumps({"x": 1, "timings": {"load": 0.3}, "summary": {"seconds": 4.5, "n": 2}})
    after["per_scene.csv"] = "scene_id,n_tasks,seconds\n0001,14,401.80\n"
    _write(tmp_path / "a", base)
    _write(tmp_path / "b", after)
    res = _run(tmp_path / "a", tmp_path / "b")
    assert res.returncode == 0, res.stdout
    assert "3 files: 1 identical, 2 only wall-time, 0 real differences" in res.stdout


def test_real_difference_fails(tmp_path):
    _write(tmp_path / "a", {"scene.json": json.dumps({"x": 1, "timings": {}})})
    _write(tmp_path / "b", {"scene.json": json.dumps({"x": 2, "timings": {}})})
    res = _run(tmp_path / "a", tmp_path / "b")
    assert res.returncode == 1
    assert "DIFFERS" in res.stdout


def test_missing_file_fails(tmp_path):
    _write(tmp_path / "a", {"one.md": "x", "two.md": "y"})
    _write(tmp_path / "b", {"one.md": "x"})
    res = _run(tmp_path / "a", tmp_path / "b")
    assert res.returncode == 1
    assert "only in before" in res.stdout
