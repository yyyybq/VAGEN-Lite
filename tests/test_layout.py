"""Test layout of the cleaned repository (cleanup phase 2, 2026-10): every test file is under tests/<area>/.

tests/ has no __init__.py anywhere: pytest imports each test file by its file name (so names must be unique), and a
package called "tests" would clash with verl's tests package on the same path.  A branch that still adds test files
in an old place (data_gen/interiorgs_v10/tests/, tests/active_spatial/, ...) fails here instead of being skipped
silently, because pytest.ini only collects tests/.
"""
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "setup.py").is_file())
AREAS = ("envs", "datagen", "training", "evaluation", "downstream", "regression")
TOP = {"tests/test_config.py", "tests/test_layout.py"}
# test files that stay where they are, and why
ELSEWHERE = {
    "data_gen/active_spatial_qa/": "another session's QA code (decision D15); suite B runs two of its tests in place",
    "data_gen/active_spatial_sft/test_path_finder.py": "the legacy SFT generator's own test; archived with it later",
}


def _tracked():
    res = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip("not a git checkout")
    return [f for f in res.stdout.split() if not f.startswith(("archive/", "third_party/", "verl/"))]


def _tests():
    return [f for f in _tracked() if Path(f).name.startswith("test_") and f.endswith(".py")]


def test_every_test_file_is_under_tests():
    stray = [f for f in _tests() if not f.startswith("tests/") and not any(f.startswith(k) for k in ELSEWHERE)]
    assert not stray, f"move these into tests/<area>/ with an area of {AREAS}: {stray}"


def test_every_test_file_is_in_an_area():
    bad = []
    for f in _tests():
        if not f.startswith("tests/") or f in TOP:
            continue
        parts = f.split("/")
        if len(parts) != 3 or parts[1] not in AREAS:
            bad.append(f)
    assert not bad, f"tests/<area>/test_*.py with an area of {AREAS}: {bad}"


def test_test_file_names_are_unique():
    names = Counter(Path(f).name for f in _tests() if f.startswith("tests/"))
    assert not [n for n, c in names.items() if c > 1]


def test_no_package_markers_under_tests():
    assert not [f for f in _tracked() if f.startswith("tests/") and f.endswith("__init__.py")]
