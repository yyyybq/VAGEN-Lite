"""C line (q38 s4): the exploratory numbers of PREREG_v1 section v1.2 on one nfp_heldout result -- r of the three
fixed groups of held-out videos (another video of the same room, house or apartment in training NFP: none found /
overlap / cannot tell), r over the distinct windows, and the within-source splits of scannet and procthor.  None of
them is a verdict: verdict 1 is the 240-row r of section 3 in nfp_heldout's summary.json, whatever these numbers are.

  python -m vagen.q38s4.tools.nfp_groups --heldout HELDOUT_STEP_1000 --out OUT --allow-uncommitted   (the check)
  python -m vagen.q38s4.tools.nfp_groups --heldout HELDOUT_STEP_1000 --out OUT --expect-commit HASH
  python -m vagen.q38s4.tools.nfp_groups --heldout HELDOUT --out OUT --expect-commit HASH --same-code-as PREV/groups.json

- HELDOUT is an nfp_heldout result folder (rows.jsonl, summary.json).  Nothing runs on the model: every number comes
  from the rows' sums through nfp_heldout.pooled (token means pooled over the rows' pairs, r = (copy_cos - nfp_cos)
  / copy_cos) and nfp_heldout.bootstrap (the set's videos resampled with replacement, 2,000 draws, numpy seed 0,
  2.5 and 97.5 % quantiles, not stratified by source): the functions behind verdict 1 and its interval.  Every row
  counts as verdict 1 counts it (duplicate windows included), except in the distinct-window number.
- The check comes first: nfp_heldout.summarize on the rows must give summary.json's 240-row r, cos means, interval
  and every source's r (the draws and seed are summary.json's, which must be 2,000 and 0).  The rows' videos must be
  the held-out list (built/s4_v2/heldout_videos.txt), 4 rows each with u = 0.125, 0.375, 0.625, 0.875 (data_build's
  (k + 0.5) / 4).  Otherwise the run stops before any window or group number is computed and writes nothing.
- The groups (v1.2, fixed; never replaced): NONE_FOUND by source and name (18), UNDETERMINED = every arkitscenes video
  (25), OVERLAP = the rest (17: scannet 12, procthor 3, adt 2).  A name matches a held-out video of its source whose
  path, without .mp4, ends with /<name>; each name must match exactly one video, and the counts and the overlap
  group's sources are checked.
- Distinct windows: a row's window is its frame indices, lf_video.nfp_indices(total frames, average fps, u,
  video_fps, video_maxlen), from the stream's header (lf_video.stream_info on the first video stream, as the training
  plugin opens it; no frame is decoded) and summary.json's media settings.  The unit count must be the row's.  The
  first row of each (video, indices) is kept, in file order; 128 windows are required.
- The splits are four r without an interval (2-12 videos a side): scannet and procthor, overlap vs none found.
- Every number comes with its sizes: videos, rows, distinct windows, prediction pairs.
- The code (v1.2: "第 2,000、3,441 步都用这个 commit"): a run needs --expect-commit HASH (the full 40-hex hash, not
  an abbreviation, HEAD or a branch name); the file running must be this one (not a copy under another name), and
  this file, nfp_heldout.py and lf_video.py must be tracked, clean, imported from this checkout, byte for byte their
  blobs at HASH, and those blobs must be the ones at this file's
  first commit (the oldest commit adding it: the commit v1.2 asks for after the check), so no later commit can
  start another chain.  Steps 2000 and 3441 (no other) also need --same-code-as PREV/groups.json: step 2000 the
  committed step-1000 run, step 3441 step 2000's run, each with the same HASH, first commit and md5s, the same
  held-out list and media settings and the same run's copies, so the three runs form one chain on one commit.
  --allow-uncommitted (step 1000 only) is the check run before the commit.  Changing the files after the commit
  needs a new v1.x first (and a new version of this guard).  Threads: 1 (OMP/MKL/OPENBLAS/NUMEXPR set, not just
  defaulted, before numpy and torch load; torch.set_num_threads(1)); groups.json records them and the versions.

Output: OUT/groups.json (OUT must not exist).  Reads HELDOUT, the held-out list, the videos' headers and git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # the login node's rule, whatever the shell exports; before numpy and torch load
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vagen.q38s4.tools import nfp_heldout as H  # noqa: E402

HELDOUT_LIST = Path("/mnt/umm/users/yinbaiqiao/q38/s4/built/s4_v2/heldout_videos.txt")
ROOT = Path(__file__).resolve().parents[3]
THIS = "vagen/q38s4/tools/nfp_groups.py"
CODE = (THIS, "vagen/q38s4/tools/nfp_heldout.py", "vagen/q38s4/lf_video.py")  # every number goes through these
NONE_FOUND: Dict[str, Tuple[str, ...]] = {  # PREREG_v1 v1.2, "没查到组"
    "scannet": ("scene0433_00", "scene0617_00"),
    "procthor": ("001082/raw_navigation_camera__1", "001224/raw_navigation_camera__0"),
    "s3dis": ("Area_3_hallway_6", "Area_4_conferenceRoom_2", "Area_4_hallway_11"),
    "scannetppv2": ("05d3c8c6f0", "0caa1ae59a", "192ab15daf", "3c8d535d49", "4e2db5058c", "52599ae063",
                    "5b38982d25", "5c21213f16", "9ad3639fbe", "c842edbdf5", "eee7eef8e5"),
}
UNDETERMINED_SOURCE = "arkitscenes"
GROUPS = ("none_found", "overlap", "undetermined")
EXPECT_GROUPS = {"none_found": 18, "overlap": 17, "undetermined": 25}
EXPECT_OVERLAP = {"adt": 2, "procthor": 3, "scannet": 12}
EXPECT_WINDOWS = 128
ROWS_PER_VIDEO = H.EXPECT_ROWS // H.EXPECT_VIDEOS  # 4
U_VALUES = tuple((k + 0.5) / ROWS_PER_VIDEO for k in range(ROWS_PER_VIDEO))  # data_build: u = (k + 0.5) / windows
HASH = re.compile(r"[0-9a-f]{40}")  # full: a name can follow another commit, a short hash can lose to a branch name
SPLITS = ("scannet", "procthor")
PARTLY_SEEN_STEP = 1000  # v1.2 was written after seeing part of step 1000's numbers
CHAIN = {2000: 1000, 3441: 2000}  # v1.2's steps after 1000, each with the step whose committed run it follows
NOTE = "exploratory (PREREG_v1 v1.2), no thresholds; not a verdict: verdict 1 is the 240-row r in summary.json (section 3)"
READING = ("v1.2 怎么读: none_found = no other video of the same room found in training NFP, not shown to be absent "
           "(没查到不等于没有); the groups' sources differ, so none_found above or below overlap says nothing about "
           "whether overlap matters; the splits are numbers only, no conclusion; conclusions use X8's wording; "
           "all_rows reproduces summary.json's 240-row r for the check, verdict 1 is read from summary.json")
NEEDED = ("video", "source", "u", "units", "pairs", "copy", "head", "random")
SUMS = ("cos_sum", "mse_sum")
TOL = 1e-12


# ---------------------------------------------------------------------------------------------------- inputs
def read_rows(path: Path) -> List[Dict[str, Any]]:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for i, r in enumerate(recs):
        miss = [k for k in NEEDED if k not in r] + [f"{k}.{s}" for k in ("copy", "head", "random") if k in r
                                                   for s in SUMS if s not in r[k]]
        if miss:
            raise SystemExit(f"{path} line {i + 1}: missing {miss} (ask the user before going on: PREREG v1.2)")
        if r["source"] != H.source_of(r["video"]):  # summary.json's by_source uses the field, the groups the path
            raise SystemExit(f"{path} line {i + 1}: source {r['source']!r} but the path says {H.source_of(r['video'])!r}")
    return recs


def step_of(summary: Mapping[str, Any]) -> Optional[int]:
    """The step from the copy's folder name (.../ckpt_bf16/checkpoint-N); None if the name is not checkpoint-N."""
    m = re.fullmatch(r"checkpoint-(\d+)", Path(str(summary.get("meta", {}).get("model", ""))).name)
    return int(m.group(1)) if m else None


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def relative(video: str) -> str:
    return video.split("/VSI-590K/", 1)[1] if "/VSI-590K/" in video else video


def check_list(recs: Sequence[Mapping[str, Any]], listed: Sequence[str]) -> None:
    have, want = {relative(r["video"]) for r in recs}, {l.strip() for l in listed if l.strip()}
    if have != want:
        raise SystemExit(f"rows' videos are not the held-out list: {len(have - want)} extra "
                         f"{sorted(have - want)[:3]}, {len(want - have)} missing {sorted(want - have)[:3]}")
    per = Counter(r["video"] for r in recs)
    odd = {relative(v): n for v, n in per.items() if n != ROWS_PER_VIDEO}
    if odd:
        raise SystemExit(f"{len(odd)} videos without {ROWS_PER_VIDEO} rows: {sorted(odd.items())[:3]}")
    us: Dict[str, List[float]] = {}
    for r in recs:
        us.setdefault(r["video"], []).append(float(r["u"]))
    off = {relative(v): sorted(u) for v, u in us.items() if sorted(u) != list(U_VALUES)}
    if off:
        raise SystemExit(f"{len(off)} videos without the u values {U_VALUES}: {sorted(off.items())[:3]}")


# ---------------------------------------------------------------------------------------------------- the check
def _diffs(a: Any, b: Any, where: str, out: List[Tuple[str, float]]) -> None:
    if a is None or b is None:
        if a is not b:
            out.append((where, float("inf")))
    else:
        out.append((where, abs(float(a) - float(b))))


def check_summary(recs: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> Dict[str, Any]:
    """nfp_heldout.summarize on the rows against summary.json: counts, 240-row r and cos, interval, per-source r."""
    b = summary["bootstrap"]
    if (b["draws"], b["seed"]) != (H.DRAWS, H.SEED):
        raise SystemExit(f"summary.json's interval used draws {b['draws']}, seed {b['seed']}; v1.2 needs "
                         f"{H.DRAWS}, {H.SEED}")
    s = H.summarize(recs, draws=b["draws"], seed=b["seed"])
    if (s["rows"], s["videos"], s["pairs"]) != (summary["rows"], summary["videos"], summary["pairs"]):
        raise SystemExit(f"rows/videos/pairs {s['rows']}/{s['videos']}/{s['pairs']} vs summary.json "
                         f"{summary['rows']}/{summary['videos']}/{summary['pairs']}")
    if sorted(s["by_source"]) != sorted(summary["by_source"]):
        raise SystemExit(f"sources {sorted(s['by_source'])} vs summary.json {sorted(summary['by_source'])}")
    d: List[Tuple[str, float]] = []
    _diffs(s["r"]["head"], summary["r"]["head"], "r.head", d)
    for k in ("copy", "head"):
        _diffs(s[k]["cos"], summary[k]["cos"], f"{k}.cos", d)
    for k in ("lo", "hi"):
        _diffs(s["bootstrap"]["head"][k], b["head"][k], f"bootstrap.head.{k}", d)
    for src, v in summary["by_source"].items():
        _diffs(s["by_source"][src]["r"]["head"], v["r"]["head"], f"by_source.{src}.r.head", d)
    bad = [x for x in d if not x[1] <= TOL]  # a NaN is bad too
    if bad:
        raise SystemExit(f"rows.jsonl does not give summary.json: {bad[0][0]} differs by {bad[0][1]}"
                         + (f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""))
    return {"ok": True, "compared": len(d), "max_abs_diff": max(x[1] for x in d), "tolerance": TOL}


# ---------------------------------------------------------------------------------------------------- groups
def stem(video: str) -> str:
    return video[:-4] if video.endswith(".mp4") else video


def groups_of(videos: Sequence[str]) -> Dict[str, str]:
    """{video: group} with the v1.2 groups; stops on a name that does not match exactly one video or wrong counts."""
    vids = sorted(set(videos))
    group = {v: ("undetermined" if H.source_of(v) == UNDETERMINED_SOURCE else "overlap") for v in vids}
    for src, names in NONE_FOUND.items():
        for name in names:
            hits = [v for v in vids if H.source_of(v) == src and stem(v).endswith("/" + name)]
            if len(hits) != 1:
                raise SystemExit(f"{src} {name}: {len(hits)} held-out videos match, expected 1 ({hits[:3]})")
            if group[hits[0]] != "overlap":
                raise SystemExit(f"{src} {name}: {hits[0]} is already {group[hits[0]]}")
            group[hits[0]] = "none_found"
    counts = {g: sum(1 for v in vids if group[v] == g) for g in GROUPS}
    if counts != EXPECT_GROUPS:
        raise SystemExit(f"group sizes {counts}, expected {EXPECT_GROUPS}")
    over = dict(Counter(H.source_of(v) for v in vids if group[v] == "overlap"))
    if over != EXPECT_OVERLAP:
        raise SystemExit(f"overlap group by source {over}, expected {EXPECT_OVERLAP}")
    return group


# ---------------------------------------------------------------------------------------------------- windows
def video_header(path: str) -> Tuple[int, float]:
    """(total frames, average fps) of the first video stream, as lf_plugin._open_video_stream + stream_info read
    them; no decoding.  (lf_plugin itself is not imported: it loads LLaMA-Factory.)"""
    import av

    from vagen.q38s4.lf_video import stream_info

    container = av.open(path, "r")
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise SystemExit(f"{path}: no video stream")
        info = stream_info(container, stream)
    finally:
        container.close()
    return int(info["total_frames"]), float(info["average_fps"])


def window_keys(recs: Sequence[Mapping[str, Any]], media: Mapping[str, Any],
                header: Callable[[str], Tuple[int, float]] = video_header) -> List[Tuple[str, Tuple[int, ...]]]:
    from vagen.q38s4.lf_video import nfp_indices

    fps, n_max = float(media["video_fps"]), int(media["video_maxlen"])
    cache: Dict[str, Tuple[int, float]] = {}
    keys = []
    for r in recs:
        v = r["video"]
        if v not in cache:
            cache[v] = header(v)
        total, avg = cache[v]
        idx = nfp_indices(total, avg, float(r["u"]), fps, n_max)
        if len(idx) // 2 != int(r["units"]):
            raise SystemExit(f"row {r.get('row')}: {len(idx)} frames = {len(idx) // 2} units, the row has {r['units']}")
        keys.append((v, tuple(idx)))
    return keys


def first_per_key(keys: Sequence[Tuple[str, Tuple[int, ...]]]) -> List[int]:
    """Positions of the first row of each distinct (video, frame indices), in file order."""
    seen, out = set(), []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out


# ---------------------------------------------------------------------------------------------------- numbers
def number(recs: Sequence[Mapping[str, Any]], keys: Sequence[Tuple[str, Tuple[int, ...]]], sel: Sequence[int],
           draws: int, seed: int, interval: bool = True) -> Dict[str, Any]:
    """Sizes, pooled r of the head (with its two cos means) and, if asked, the video bootstrap interval of the rows
    at ``sel``."""
    sub = [recs[i] for i in sel]
    p = H.pooled(sub, names=("head",))
    out: Dict[str, Any] = {"videos": len({r["video"] for r in sub}), "rows": len(sub),
                           "windows": len({keys[i] for i in sel}), "pairs": int(sum(r["pairs"] for r in sub)),
                           "r": p["r"]["head"], "nfp_cos": p["head"]["cos"], "copy_cos": p["copy"]["cos"]}
    if interval:
        b = H.bootstrap(sub, names=("head",), draws=draws, seed=seed)
        out["interval"] = {"lo": b["head"]["lo"], "hi": b["head"]["hi"], "draws": draws, "seed": seed,
                           "videos": b["videos"]}
    return out


def compute(recs: Sequence[Mapping[str, Any]], keys: Sequence[Tuple[str, Tuple[int, ...]]], draws: int = H.DRAWS,
            seed: int = H.SEED, expect_windows: int = EXPECT_WINDOWS) -> Dict[str, Any]:
    if len(keys) != len(recs):
        raise SystemExit(f"{len(keys)} windows for {len(recs)} rows")
    group = groups_of([r["video"] for r in recs])
    distinct = first_per_key(keys)
    if len(distinct) != expect_windows:
        raise SystemExit(f"{len(distinct)} distinct windows, expected {expect_windows}")
    every = list(range(len(recs)))
    out: Dict[str, Any] = {"all_rows": number(recs, keys, every, draws, seed),
                           "distinct_windows": number(recs, keys, distinct, draws, seed), "groups": {},
                           "splits": {}, "members": {}}
    for g in GROUPS:
        sel = [i for i in every if group[recs[i]["video"]] == g]
        src = dict(sorted(Counter(H.source_of(v) for v in {recs[i]["video"] for i in sel}).items()))
        out["groups"][g] = {**number(recs, keys, sel, draws, seed), "videos_by_source": src}
        out["members"][g] = sorted({relative(recs[i]["video"]) for i in sel})
    for s in SPLITS:
        out["splits"][s] = {g: number(recs, keys, [i for i in every if H.source_of(recs[i]["video"]) == s
                                                   and group[recs[i]["video"]] == g], draws, seed, interval=False)
                            for g in ("overlap", "none_found")}
    out["splits"]["note"] = "numbers only, no conclusion (v1.2)"
    return out


# ---------------------------------------------------------------------------------------------------- the code
def _git_env() -> Dict[str, str]:
    return {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}  # read-only: git status does not refresh the index


def _git(root: Path, *args: str, ok: bool = False) -> Optional[str]:
    p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, env=_git_env())
    if p.returncode != 0:
        if ok:
            return None
        raise SystemExit(f"git {' '.join(args)} failed in {root}: {p.stderr.strip()}")
    return p.stdout.rstrip("\n")  # a status line keeps its leading space (" M": changed, not staged)


def _blob_md5(root: Path, rev_path: str) -> Optional[str]:
    """md5 of the blob's bytes (git cat-file), None if there is no such blob."""
    p = subprocess.run(["git", "cat-file", "blob", rev_path], cwd=root, capture_output=True, env=_git_env())
    return hashlib.md5(p.stdout).hexdigest() if p.returncode == 0 else None


def first_commit(root: Path, path: str) -> Optional[str]:
    """The oldest commit in HEAD's history that adds ``path``; for this file, the commit v1.2 asks for after the
    check (the tool comes as its own commit before the step-2000 held-out job)."""
    got = _git(root, "log", "--reverse", "--diff-filter=A", "--format=%H", "HEAD", "--", path, ok=True)
    return got.splitlines()[0] if got else None


def provenance(expect_commit: Optional[str] = None, root: Path = ROOT, files: Sequence[str] = CODE) -> Dict[str, Any]:
    """git state of the files every number goes through: tracked, clean, blob at HEAD, the last commit that touched
    each; with ``expect_commit`` (the full hash), whether each file's bytes on disk and its blob at HEAD are its blob
    there (on disk too, since git status trusts the index: skip-worktree or assume-unchanged hide a change), and
    whether that blob is the one at the first commit of ``files[0]`` (this tool)."""
    if expect_commit and not HASH.fullmatch(expect_commit):
        raise SystemExit(f"--expect-commit {expect_commit!r}: give the commit's full hash (40 lowercase hex), not an "
                         f"abbreviation or a name such as HEAD or a branch (PREREG v1.2 fixes one commit)")
    top = _git(root, "rev-parse", "--show-toplevel")
    if Path(top).resolve() != Path(root).resolve():
        raise SystemExit(f"{root} is not the top of its git checkout ({top}): the paths are relative to the top")
    out: Dict[str, Any] = {"head": _git(root, "rev-parse", "HEAD"), "files": {}}
    for f in files:
        tracked = bool(_git(root, "ls-files", "--", f))
        status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", f)
        info: Dict[str, Any] = {"tracked": tracked, "status": status or ("clean" if tracked else "untracked"),
                                "md5": md5(Path(root) / f) if (Path(root) / f).is_file() else None}
        if info["status"] == "clean":
            info["blob"] = _git(root, "rev-parse", f"HEAD:{f}")
            info["last_commit"] = _git(root, "log", "-1", "--format=%H", "--", f)
        out["files"][f] = info
    out["committed"] = all(i["status"] == "clean" for i in out["files"].values())
    out["expect_commit"], out["expect_commit_arg"], out["first_commit"] = None, expect_commit, None
    if expect_commit:
        c = _git(root, "rev-parse", "--verify", "--quiet", f"{expect_commit}^{{commit}}", ok=True)
        if not c:
            raise SystemExit(f"--expect-commit {expect_commit}: no such commit in {root}")
        first = first_commit(root, files[0])
        if not first:
            raise SystemExit(f"{files[0]}: no commit in HEAD's history adds it ({root})")
        out["expect_commit"], out["first_commit"] = c, first
        out["differ"], out["differ_why"] = [], {}
        for f, i in out["files"].items():
            i["md5_at_expect"] = _blob_md5(root, f"{c}:{f}")
            blob_c = _git(root, "rev-parse", f"{c}:{f}", ok=True)
            i["blob_at_first"] = _git(root, "rev-parse", f"{first}:{f}", ok=True)
            if i["md5_at_expect"] is None:
                why = "not in that commit"
            elif i.get("blob") is None or blob_c != i["blob"]:
                why = "its blob at HEAD is not the one there"
            elif i["md5_at_expect"] != i["md5"]:
                why = "the bytes on disk are not its blob there, though git status says clean (skip-worktree?)"
            elif i["blob_at_first"] is None or blob_c != i["blob_at_first"]:
                why = (f"its blob there is not the one at {files[0]}'s first commit {first[:8]}, the commit v1.2 "
                       f"fixes")
            else:
                continue
            out["differ"].append(f)
            out["differ_why"][f] = why
    return out


def check_this_file() -> str:
    """The file running is the registered one (ROOT/THIS), not a copy under another name next to it, which would
    pass every check on the registered file."""
    here = Path(__file__).resolve()
    if here != (ROOT / THIS).resolve():
        raise SystemExit(f"this is {here}, not {ROOT / THIS}: only the registered file runs (PREREG v1.2)")
    return str(here)


def check_imports(root: Path = ROOT, files: Sequence[str] = CODE) -> Dict[str, str]:
    """The modules Python runs are the checkout's files that provenance looked at."""
    import importlib.util

    got = {}
    for f in files:
        name = f[:-3].replace("/", ".")
        spec = importlib.util.find_spec(name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin != (Path(root) / f).resolve():
            raise SystemExit(f"{name} is imported from {origin}, not {Path(root) / f}")
        got[name] = str(origin)
    return got


def check_code(code: Mapping[str, Any], step: Optional[int], allow_uncommitted: bool) -> None:
    if allow_uncommitted:
        if step != PARTLY_SEEN_STEP:
            raise SystemExit(f"--allow-uncommitted is for step {PARTLY_SEEN_STEP} only (the check before the commit, "
                             f"PREREG v1.2); this is step {step}")
        return
    dirty = [f"{f} ({i['status']})" for f, i in code["files"].items() if i["status"] != "clean"]
    if dirty:
        raise SystemExit(f"not committed and clean at {code['head'][:8]}: {', '.join(dirty)} (PREREG v1.2: the "
                         f"numbers come from the committed script)")
    if not code.get("expect_commit"):
        raise SystemExit("--expect-commit COMMIT is needed: the commit PREREG v1.2 fixes for steps 2000 and 3441")
    if code["differ"]:
        why = code.get("differ_why", {})
        raise SystemExit(f"{', '.join(code['differ'])} differ from {code['expect_commit'][:8]} ("
                         + "; ".join(f"{f}: {why.get(f, '?')}" for f in code["differ"])
                         + "): run with the commit's own files; a change after the commit needs a new v1.x first "
                           "(PREREG v1.2)")


def check_same_code(code: Mapping[str, Any], step: Optional[int], prev_path: Optional[str],
                    run: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The chain of committed runs, one commit throughout: step 1000 needs none; step 2000 follows the committed
    step-1000 run, step 3441 step 2000's run (itself following step 1000 with the same commit); same first commit
    and md5s; with ``run`` ({model, media, heldout_list_md5} of this run), the same held-out list, media settings and
    run folder of the copies."""
    if step == PARTLY_SEEN_STEP:
        if prev_path is not None:
            raise SystemExit(f"step {PARTLY_SEEN_STEP} takes no --same-code-as: its committed run starts the chain "
                             f"(PREREG v1.2)")
        return None
    if step not in CHAIN:
        raise SystemExit(f"step {step}: v1.2 registers the group numbers at steps {PARTLY_SEEN_STEP} (partly seen) "
                         f"and {', '.join(map(str, CHAIN))} only")
    if prev_path is None:
        raise SystemExit(f"--same-code-as PREV/groups.json is needed at step {step}: the committed step-{CHAIN[step]} "
                         f"run (PREREG v1.2 uses one commit for steps 2000 and 3441)")
    p = Path(prev_path)
    try:
        prev = json.loads(p.read_text(encoding="utf-8"))["meta"]
        pc = prev["code"]
        pc["files"].items()
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise SystemExit(f"{p}: not a groups.json of this tool ({type(e).__name__}: {e})")
    if not (pc.get("committed") and pc.get("expect_commit")) or prev.get("allow_uncommitted"):
        raise SystemExit(f"{p}: not a committed run (--expect-commit)")
    if pc["expect_commit"] != code["expect_commit"]:
        raise SystemExit(f"{p} used commit {pc['expect_commit'][:8]}, this run {code['expect_commit'][:8]}: "
                         f"PREREG v1.2 uses one commit")
    if prev.get("step") != CHAIN[step]:
        raise SystemExit(f"{p} is step {prev.get('step')}: PREV of step {step} must be the committed step-"
                         f"{CHAIN[step]} run")
    if prev["step"] in CHAIN and (prev.get("same_code_as") or {}).get("expect_commit") != pc["expect_commit"]:
        raise SystemExit(f"{p}: step {prev['step']}'s run does not follow a run with the same commit")
    md5s = {f: i["md5"] for f, i in code["files"].items()}
    if {f: i.get("md5") for f, i in pc["files"].items()} != md5s:
        raise SystemExit(f"{p}: its files' md5s are not this run's")
    if pc.get("first_commit") != code.get("first_commit"):
        raise SystemExit(f"{p}: its first commit {str(pc.get('first_commit'))[:8]} is not this run's "
                         f"{str(code.get('first_commit'))[:8]}")
    if run is not None:
        if prev.get("heldout_list_md5") != run["heldout_list_md5"]:
            raise SystemExit(f"{p}: another held-out list (md5 {prev.get('heldout_list_md5')}, this run "
                             f"{run['heldout_list_md5']})")
        if prev.get("media") != run["media"]:
            raise SystemExit(f"{p}: other media settings ({prev.get('media')}, this run {run['media']})")
        if Path(str(prev.get("model"))).parent != Path(str(run["model"])).parent:
            raise SystemExit(f"{p}: a copy of another run ({prev.get('model')}; this run {run['model']})")
    return {"path": str(p.resolve()), "md5": md5(p), "step": prev["step"], "expect_commit": pc["expect_commit"]}


# ---------------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heldout", required=True, help="nfp_heldout result folder (rows.jsonl, summary.json)")
    ap.add_argument("--out", required=True, help="output folder (must not exist)")
    ap.add_argument("--heldout-list", default=str(HELDOUT_LIST))
    ap.add_argument("--expect-commit", default=None, help="the hash of the commit v1.2 fixes for the script (needed "
                                                          "unless --allow-uncommitted)")
    ap.add_argument("--same-code-as", default=None, help="an earlier step's committed groups.json (needed after "
                                                         f"step {PARTLY_SEEN_STEP})")
    ap.add_argument("--allow-uncommitted", action="store_true", help=f"step {PARTLY_SEEN_STEP} only")
    return ap.parse_args(argv)


def main(argv=None, header: Callable[[str], Tuple[int, float]] = video_header,
         prov: Callable[[Optional[str]], Dict[str, Any]] = provenance,
         expect_windows: int = EXPECT_WINDOWS) -> Dict[str, Any]:
    a = parse_args(argv)
    check_this_file()
    if a.allow_uncommitted and a.same_code_as:
        raise SystemExit("--allow-uncommitted takes no --same-code-as: the check before the commit is not in the chain")
    import numpy
    import torch

    torch.set_num_threads(1)
    env = {"python": platform.python_version(), "numpy": numpy.__version__, "torch": torch.__version__,
           "threads": {k: os.environ.get(k) for k in THREAD_VARS}, "torch_threads": torch.get_num_threads()}
    out, src, lst = Path(a.out), Path(a.heldout), Path(a.heldout_list)
    if out.exists():
        raise SystemExit(f"{out} exists")
    summary = json.loads((src / "summary.json").read_text())
    step = step_of(summary)
    code = prov(a.expect_commit)
    check_code(code, step, a.allow_uncommitted)
    run = {"model": summary["meta"].get("model"), "media": summary["meta"].get("media"), "heldout_list_md5": md5(lst)}
    same = None if a.allow_uncommitted else check_same_code(code, step, a.same_code_as, run)
    imports = check_imports()
    recs = read_rows(src / "rows.jsonl")
    check_list(recs, lst.read_text().splitlines())
    check = check_summary(recs, summary)  # before any group number
    media = summary["meta"]["media"]
    keys = window_keys(recs, media, header)
    res = compute(recs, keys, draws=summary["bootstrap"]["draws"], seed=summary["bootstrap"]["seed"],
                  expect_windows=expect_windows)
    meta = {"note": NOTE, "reading": READING, "heldout": str(src.resolve()), "model": summary["meta"].get("model"),
            "step": step, "partly_seen": step == PARTLY_SEEN_STEP, "summary_md5": md5(src / "summary.json"),
            "rows_md5": md5(src / "rows.jsonl"), "heldout_list": str(lst.resolve()), "heldout_list_md5": md5(lst),
            "expect_windows": expect_windows, "allow_uncommitted": a.allow_uncommitted, "media": media, "code": code,
            "same_code_as": same, "imports": imports, "env": env}
    if step == PARTLY_SEEN_STEP:
        meta["partly_seen_note"] = "PREREG v1.2 was written after seeing part of this step's numbers"
    result = {"meta": meta, "check": check, **res}
    out.mkdir(parents=True)
    (out / "groups.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[groups] step {step}: check ok (max diff {check['max_abs_diff']:.2e}); wrote {out / 'groups.json'}",
          flush=True)
    return result


if __name__ == "__main__":
    main()
