"""E3VS-Bench pilot (gate 3): episode files, the pinned repo and the action vocabulary.

E3VS-Bench: https://github.com/K0uya/E3VS-BENCH, pinned at commit bb7e603ccdcf2dd77fbeedd15660a09310a8a881
(paths.E3VS_COMMIT).  Read from it: ``dataset/episode_info_{train,val,test}.jsonl``; the official collision code
``e3vs/eval/metrics.py`` (class Collisions), which mesh_collision.py executes by file path after asserting the
commit and the file's sha256; for parity, render.py's official loader / renderer; in the tests,
``e3vs/environments/camera_geometry.py`` and the prompt files.  Nothing is imported as a package (no __init__).

Where the repo is read from (``repo_dir()``):
  1. ``$VPD_E3VS_REPO`` if set (an explicit choice, e.g. the earlier-fetched read-only copy
     ``LOGIN_TMP_REPO`` while its provenance is pending the user's OK, DECISIONS section D);
  2. otherwise ``paths.E3VS_REPO`` = OUT/e3vs/repo/E3VS-BENCH (created by ``download_splats.sh repo`` after
     the user's OK).
There is no silent fallback to the login_tmp copy: outputs that rest on it must be produced on purpose.

Stdlib + numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from vpd import paths

E3VS_COMMIT = paths.E3VS_COMMIT
E3VS_URL = "https://github.com/K0uya/E3VS-BENCH"

# Read-only copy fetched during the survey (plan/login_tmp); commit and md5s were checked on 2026-11-08.
LOGIN_TMP_REPO = paths.PLAN_DIR / "login_tmp" / "vpd_e3vs" / "repo"

SPLITS = ("train", "val", "test")
EPISODE_MD5 = {
    "train": "d21ba21fafb1846699bd8d2779f56a40",
    "val": "6a6df2992f3f9f549738a0b8f0718288",
    "test": "c537eb6100ec31b371091258b3533737",
}
EPISODE_COUNTS = {"train": 1406, "val": 231, "test": 377}
SCENE_COUNTS = {"train": 68, "val": 10, "test": 21}

# The 11 official actions, in the order of e3vs/configs/qwen_3.0_vl_8B_instruct/single_frame.yaml.
ACTIONS = (
    "move_forward", "move_backward", "move_up", "move_down", "move_left", "move_right",
    "turn_left", "turn_right", "look_up", "look_down", "stop",
)
# The 10 non-stop actions in the fixed BFS / floor enumeration order (survey prototype bfs.py).  Ties between
# equally short BFS paths and between equally good blind sequences break by this order.
SEARCH_ORDER = (
    "move_forward", "move_backward", "move_left", "move_right", "move_up", "move_down",
    "turn_left", "turn_right", "look_up", "look_down",
)
TRANSLATIONS = frozenset(a for a in ACTIONS if a.startswith("move_"))

# Official environment constants (e3vs/environments/constraint.py, single_frame.yaml).
STEP_M = 0.25
ANGLE_DEG = 30.0
IMAGE_W = IMAGE_H = 512
FOV_DEG = 90.0
MAX_DECISIONS = 25
# Viewpoint success tolerance (report v2 section 3.2): position <= 0.25 m AND view-direction angle <= 20 deg.
SUCCESS_POS_M = 0.25
SUCCESS_ANG_DEG = 20.0


def repo_dir() -> Path:
    """The E3VS-Bench checkout to read (see module docstring)."""
    env = os.environ.get("VPD_E3VS_REPO")
    p = Path(env) if env else paths.E3VS_REPO
    if not (p / "dataset").is_dir():
        raise FileNotFoundError(
            f"E3VS repo not found at {p}. Run `bash vpd/e3vs/download_splats.sh repo` (needs the user's OK), "
            f"or set VPD_E3VS_REPO={LOGIN_TMP_REPO} to read the earlier-fetched read-only copy."
        )
    return p


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def repo_commit(repo: Path) -> str | None:
    """HEAD commit of a checkout, read from .git without running git (None if not a git checkout)."""
    git = repo / ".git"
    if git.is_file():  # linked worktree: ".git" holds "gitdir: <path>"
        s = git.read_text().strip()
        if not s.startswith("gitdir:"):
            return None
        git = Path(s.split(":", 1)[1].strip())
        if not git.is_absolute():
            git = (repo / git).resolve()
    head = git / "HEAD"
    if not head.is_file():
        return None
    common = git
    if (git / "commondir").is_file():
        common = (git / (git / "commondir").read_text().strip()).resolve()
    s = head.read_text().strip()
    if not s.startswith("ref:"):
        return s
    ref = s.split(None, 1)[1]
    for base in (git, common):
        loose = base / ref
        if loose.is_file():
            return loose.read_text().strip()
    for base in (git, common):
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    return None


def check_repo(repo: Path | None = None, splits=SPLITS) -> dict:
    """Assert the checkout is the pinned commit and the episode files have the pinned md5s."""
    repo = Path(repo) if repo else repo_dir()
    commit = repo_commit(repo)
    if commit is not None and commit != E3VS_COMMIT:
        raise AssertionError(f"{repo}: HEAD {commit} != pinned {E3VS_COMMIT}")
    md5 = {}
    for sp in splits:
        got = md5_file(episodes_path(sp, repo))
        if got != EPISODE_MD5[sp]:
            raise AssertionError(f"{sp}: md5 {got} != pinned {EPISODE_MD5[sp]}")
        md5[sp] = got
    return {"repo": str(repo), "commit": commit, "episode_md5": md5}


def episodes_path(split: str, repo: Path | None = None) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return Path(repo or repo_dir()) / "dataset" / f"episode_info_{split}.jsonl"


def _c2w_table(d: dict) -> dict[int, np.ndarray]:
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in d.items()}


def load_episodes(split: str, repo: Path | None = None, check: bool = True) -> list[dict]:
    """Episodes of one split, in file order.

    Each dict: split, episode_index, scene_id, question_type, question, answer, instance_id, instance_name,
    start_viewpoint (int template id), start_c2w (4x4 float64), goals {vid: c2w} (answerable viewpoints),
    unans {vid: c2w} (unanswerable viewpoints; the start is one of them).
    """
    repo = Path(repo) if repo else repo_dir()
    if check:
        got = md5_file(episodes_path(split, repo))
        if got != EPISODE_MD5[split]:
            raise AssertionError(f"{split}: md5 {got} != pinned {EPISODE_MD5[split]}")
    out = []
    with open(episodes_path(split, repo)) as f:
        for line in f:
            r = json.loads(line)
            cp = r["camera_pose_info"]
            out.append({
                "split": split,
                "episode_index": int(r["episode_index"]),
                "scene_id": r["scene_id"],
                "question_type": r["question_type"],
                "question": r["question"],
                "answer": r["answer"],
                "instance_id": int(r["instance_id"]),
                "instance_name": r["instance_name"],
                "start_viewpoint": int(r["start_viewpoint"]),
                "start_c2w": np.asarray(cp["start_viewpoint"]["c2w"], dtype=np.float64),
                "goals": _c2w_table(cp["answerable_viewpoints"]["c2w"]),
                "unans": _c2w_table(cp["unanswerable_viewpoints"]["c2w"]),
            })
    return out


def thread_env(n: int = 1) -> dict:
    """Thread-count environment for helper processes (login-node rule: 1 thread per process)."""
    v = str(int(n))
    return {"OMP_NUM_THREADS": v, "MKL_NUM_THREADS": v, "OPENBLAS_NUM_THREADS": v, "NUMEXPR_NUM_THREADS": v,
            "TOKENIZERS_PARALLELISM": "false"}


def _snapshot_commit(root: Path) -> str | None:
    """The commit of a job's code snapshot (vpd.snapshot, RUN/code/SNAPSHOT.json), None when root is not one."""
    from vpd import snapshot

    try:
        st = snapshot.snapshot_state(root)
    except snapshot.SnapshotError:
        return None
    return st.get("commit") if st.get("mode") == "snapshot" else None


def code_meta(*files: Path) -> dict:
    """sha256 of the given source files plus the worktree HEAD (read from .git, no git command).  In a sco job the
    code is the snapshot RUN/code, which has no .git: worktree_head is then the snapshot's commit and
    worktree_head_source is "snapshot"."""
    head = repo_commit(paths.WORKTREE)
    meta = {"worktree_head": head}
    if head is None:
        snap = _snapshot_commit(paths.WORKTREE)
        if snap is not None:
            meta = {"worktree_head": snap, "worktree_head_source": "snapshot"}
    meta["files"] = {str(Path(f).relative_to(paths.WORKTREE)) if str(f).startswith(str(paths.WORKTREE)) else str(f):
                     sha256_file(Path(f)) for f in files}
    return meta
