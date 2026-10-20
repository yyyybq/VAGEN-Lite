"""CPU tests of downstream/verify_starts.py and downstream/starvla_patch.py (a throw-away git repository)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import starvla_patch  # noqa: E402
import verify_starts  # noqa: E402


def fake_start(path: Path):
    path.mkdir(parents=True)
    (path / "model-00001-of-00002.safetensors").write_bytes(b"a" * 10)
    (path / "model-00002-of-00002.safetensors").write_bytes(b"b" * 20)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "x": "model-00001-of-00002.safetensors", "y": "model-00002-of-00002.safetensors"}}))
    (path / "config.json").write_text("{}")
    return path


def test_lock_then_verify_detects_changes(tmp_path, monkeypatch):
    starts = {"B": {"role": "main", "desc": "b", "path": str(fake_start(tmp_path / "B"))},
              "R": {"role": "spare", "desc": "r", "path": str(fake_start(tmp_path / "R"))}}
    (tmp_path / "starts.json").write_text(json.dumps(starts))
    monkeypatch.setattr(verify_starts, "STARTS", tmp_path / "starts.json")
    lock = tmp_path / "lock.json"
    assert verify_starts.main(["lock", "--lock-file", str(lock)]) == 0
    assert sorted(json.loads(lock.read_text())) == ["B"]  # role main only by default
    assert verify_starts.verify(None, lock) == {"B": []}
    (tmp_path / "B/model-00002-of-00002.safetensors").write_bytes(b"b" * 21)
    assert verify_starts.verify(None, lock)["B"] == ["shard model-00002-of-00002.safetensors: size 20 -> 21"]
    (tmp_path / "B/model.safetensors.index.json").write_text("{\"weight_map\": {}}")
    problems = verify_starts.verify(None, lock)["B"]
    assert any(p.startswith("index_sha256") for p in problems)
    assert verify_starts.main(["verify", "--lock-file", str(lock)]) == 1


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_patch_export_and_reverse_check(tmp_path, monkeypatch):
    repo = tmp_path / "starVLA"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    git(repo, "add", "a.py")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base")
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.yaml").write_text("k: v\n")
    (repo / "playground").mkdir()
    (repo / "playground/big.pt").write_bytes(b"0")
    monkeypatch.setattr(starvla_patch, "PATCH_DIR", tmp_path / "patches")
    assert starvla_patch.untracked(repo) == ["new.yaml"]  # playground/ is never part of the patch
    assert starvla_patch.main(["export", "--repo", str(repo)]) == 0
    result = starvla_patch.check(repo)
    assert result["ok"] and result["reverse_check"] and result["working_tree_equals_patch"]
    text = Path(result["patch"]).read_text()
    assert "new.yaml" in text and "a.py" in text and "big.pt" not in text
    (repo / "a.py").write_text("x = 3\n")  # a change after export is caught
    assert not starvla_patch.check(repo)["ok"]
