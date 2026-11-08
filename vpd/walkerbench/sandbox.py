"""A private, patched copy of the WalkerBench checkout: the one helper used by the demo builder and the eval.

    copy, diff = make_sandbox(RUN, OUT/walkerbench/split/s2test/tasks, render_size=(640, 400))

Mirrors eval_external/closed_loop/walkerbench/run.py:make_copy (bf4b2553, lines 84-99), which is not edited:
- the checkout (REPO_WB, b197f51) is copied to dst/WalkerBench without .git, logs and temp_images;
- the task JSONs are copied into the copy at tasks/<label> and config/settings.py points TASKS_DIR there;
- PANORAMAS_DIR points at the unpacked panoramas;
- in addition, Settings.RENDER_OUTPUT_SIZE (1280, 800) becomes render_size, so the engine renders the agent's view
  directly at that size (py360convert e2p, hFOV 90, JPEG); no resize step exists anywhere (plan section 1.1).
Each rewrite must hit exactly once.  The returned unified diff of settings.py goes into the manifests.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
from pathlib import Path

from vpd import paths

MARKER = ".vpd_sandbox.json"
IGNORE = ("logs", "temp_images", ".git")  # make_copy's ignore patterns (any depth)
# Opt-in (exclude=LEAN_EXCLUDE): checkout paths a sandbox never reads once PANORAMAS_DIR / TASKS_DIR are rewritten.
# They are ~1,400 of the ~1,550 files; on the shared FS a full copy takes minutes.
LEAN_EXCLUDE = ("data/panoramas_sample", "tasks/full", "tasks/sample")


def _ignore(repo: Path, exclude: tuple[str, ...]):
    ex = {Path(e).as_posix().strip("/") for e in exclude}

    def ignore(dirpath: str, names: list[str]) -> set[str]:
        rel = Path(dirpath).relative_to(repo).as_posix()
        out = {n for n in names if n in IGNORE}
        out |= {n for n in names if (n if rel == "." else f"{rel}/{n}") in ex}
        return out

    return ignore
RE_PANOS = r"(?m)^PANORAMAS_DIR = .*$"
RE_TASKS = r"(?m)^TASKS_DIR = .*$"
RE_SIZE = r"(?m)^(\s*RENDER_OUTPUT_SIZE: Tuple\[int, int\] = )\(1280, 800\)"


def task_label(tasks_dir: Path) -> str:
    """tasks/<label> name inside the copy: vpd_<split> for .../<split>/tasks, else vpd_<dir name>."""
    tasks_dir = Path(tasks_dir)
    base = tasks_dir.parent.name if tasks_dir.name == "tasks" else tasks_dir.name
    label = "vpd_" + re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    return label


def patch_settings(old: str, tasks_label: str, render_size: tuple[int, int], panoramas: Path) -> tuple[str, str]:
    """(new settings.py text, unified diff).  Raises if any of the three rewrites does not hit exactly once."""
    w, h = (int(x) for x in render_size)
    if w <= 0 or h <= 0:
        raise ValueError(f"bad render size {render_size}")
    new, n1 = re.subn(RE_PANOS, lambda m: f'PANORAMAS_DIR = Path("{panoramas}")', old)
    new, n2 = re.subn(RE_TASKS, lambda m: f'TASKS_DIR = BASE_DIR / "tasks" / "{tasks_label}"', new)
    new, n3 = re.subn(RE_SIZE, lambda m: f"{m.group(1)}({w}, {h})", new)
    if (n1, n2, n3) != (1, 1, 1):
        raise RuntimeError(f"config/settings.py layout changed: PANORAMAS_DIR / TASKS_DIR / RENDER_OUTPUT_SIZE "
                           f"matched {(n1, n2, n3)} times, expected (1, 1, 1)")
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "settings.py", "settings.py"))
    return new, diff


def repo_commit(repo: Path = paths.REPO_WB) -> str | None:
    """HEAD sha of a checkout, read from .git without running git (None if unreadable)."""
    git = Path(repo) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(None, 1)[1]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        return None
    return None


def make_sandbox(dst: Path | str, tasks_dir: Path | str, render_size: tuple[int, int] = (640, 400),
                 panoramas: Path | str = paths.PANOS, repo: Path | str = paths.REPO_WB,
                 exclude: tuple[str, ...] = ()) -> tuple[Path, str]:
    """Copy the WalkerBench checkout to dst/WalkerBench and patch it; return (copy_dir, settings_diff).

    exclude: checkout-relative paths not copied (default none = exactly make_copy; LEAN_EXCLUDE for speed).
    An existing dst/WalkerBench is replaced only if it is an earlier sandbox (it carries MARKER)."""
    dst, tasks_dir, panoramas, repo = Path(dst), Path(tasks_dir), Path(panoramas), Path(repo)
    if not tasks_dir.is_dir() or not any(tasks_dir.glob("*.json")):
        raise FileNotFoundError(f"no task JSONs in {tasks_dir}")
    if not panoramas.is_dir():
        raise FileNotFoundError(f"panoramas dir {panoramas} missing")
    copy = dst / "WalkerBench"
    if copy.exists():
        if not (copy / MARKER).exists():
            raise FileExistsError(f"{copy} exists and is not a vpd sandbox; refusing to replace it")
        shutil.rmtree(copy)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo, copy, ignore=_ignore(repo, tuple(exclude)))
    label = task_label(tasks_dir)
    target = copy / "tasks" / label
    if target.exists():
        raise FileExistsError(f"{target} exists in the checkout; pick another task dir name")
    shutil.copytree(tasks_dir, target)
    settings = copy / "config" / "settings.py"
    new, diff = patch_settings(settings.read_text(), label, render_size, panoramas)
    settings.write_text(new)
    meta = {"repo": str(repo), "commit": repo_commit(repo), "tasks_dir": str(tasks_dir), "tasks_label": label,
            "n_tasks": len(list(target.glob("*.json"))), "render_size": [int(x) for x in render_size],
            "panoramas": str(panoramas), "exclude": list(exclude), "settings_diff": diff}
    (copy / MARKER).write_text(json.dumps(meta, indent=1) + "\n")
    return copy, diff


def parse_size(text: str) -> tuple[int, int]:
    """'640x400' -> (640, 400)."""
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not m:
        raise ValueError(f"size must look like 640x400, got {text!r}")
    return int(m.group(1)), int(m.group(2))
