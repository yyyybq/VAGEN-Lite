"""Inventory of a BEHAVIOR-1K datasets root: what is there, how big, and which challenge tasks it covers.

    python scripts/inventory_datasets.py [ROOT] [--out datasets.json] [--md datasets.md]
        [--task-data $B1K_ROOT/docs/challenge/task_data.json] [--max-files 300000] [--time-limit-s 600]

ROOT defaults to $OMNIGIBSON_DATA_PATH, else /nas/baiqiao/BEHAVIOR-1K/datasets. Read-only, stdlib only. A top-level
entry that is a symlink to a directory is measured at its target (recorded); symlinks below it are not followed.
Every walk stops at --max-files files per top-level directory and at the global --time-limit-s; sizes from a
truncated walk are lower bounds and marked so. The instance scan runs before the size walk, so a slow size walk
cannot starve the task coverage; a truncated instance scan is reported separately (its counts are lower bounds).

Reported:
  top level      each entry with bytes / files (os.walk + lstat, like du -sb), and a file-type histogram
  known items    the paths OmniGibson reads (OmniGibson/omnigibson/utils/asset_utils.py, eval/utils/eval_utils.py):
                 2026-challenge-task-instances/metadata/{B100_task_misc.csv, task.jsonl, available_tasks.yaml,
                 task_custom_lists.json}, the instance splits scenes/ (train), scene_test/public (public_test),
                 scene_test/private (hidden_test), behavior-1k-assets, omnigibson-robot-assets, og_dataset (older
                 name), 2025-challenge-task-instances, omnigibson.key (existence only, never read)
  demos          every LeRobot root (a directory with meta/info.json, e.g. 2026-challenge-demos): meta files,
                 info.json counts, data/chunk-* (chunk index = task id, docs/challenge/baselines.md), video keys,
                 annotations/task-*
  task coverage  which of the 100 challenge tasks (docs/challenge/task_data.json) have instance files per split
                 and demo / annotation data. Task ids map to names through B100_task_misc.csv when present, else
                 through the task_data.json order (equal to the official ids per configs/tasks/README.md; labelled).

Status: written and unit-tested on a synthetic tree (tests/test_ops_inventory_datasets.py); not yet run on 105.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
import time
from typing import Optional

DEFAULT_ROOT = "/nas/baiqiao/BEHAVIOR-1K/datasets"
INSTANCES = "2026-challenge-task-instances"
METADATA = os.path.join(INSTANCES, "metadata")
SPLITS = {"train": os.path.join(INSTANCES, "scenes"),
          "public_test": os.path.join(INSTANCES, "scene_test", "public"),
          "hidden_test": os.path.join(INSTANCES, "scene_test", "private")}
KNOWN_ITEMS = {
    "task_instances_2026": INSTANCES,
    "B100_task_misc.csv": os.path.join(METADATA, "B100_task_misc.csv"),
    "task.jsonl": os.path.join(METADATA, "task.jsonl"),
    "available_tasks.yaml": os.path.join(METADATA, "available_tasks.yaml"),
    "task_custom_lists.json": os.path.join(METADATA, "task_custom_lists.json"),
    "instances_train": SPLITS["train"],
    "instances_public_test": SPLITS["public_test"],
    "instances_hidden_test": SPLITS["hidden_test"],
    "task_instances_2025": "2025-challenge-task-instances",
    "behavior-1k-assets": "behavior-1k-assets",
    "omnigibson-robot-assets": "omnigibson-robot-assets",
    "og_dataset (older asset dir name)": "og_dataset",
    "omnigibson.key": "omnigibson.key",
    "2026-challenge-demos": "2026-challenge-demos",
}
VERSION_FILES = ("behavior-1k-assets", "omnigibson-robot-assets")
NO_READ = {"omnigibson.key"}
LEROBOT_META = ("info.json", "tasks.jsonl", "tasks.parquet", "stats.json", "episodes")
_INSTANCE_ID = re.compile(r"_(\d+)_(\d+)_template")
_CHUNK = re.compile(r"^chunk-(\d+)$")
_TASK_DIR = re.compile(r"^task-(\d+)$")


class Budget:
    """Global wall-clock limit shared by every walk."""

    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds
        self.hit = False

    def over(self) -> bool:
        if time.monotonic() > self.deadline:
            self.hit = True
        return self.hit


def ext_of(name: str) -> str:
    base, ext = os.path.splitext(name)
    if ext in (".gz", ".bz2", ".xz", ".zst") and os.path.splitext(base)[1]:
        ext = os.path.splitext(base)[1] + ext
    return ext.lower() or "(none)"


def walk_size(path: str, max_files: int, budget: Budget, hist: Optional[collections.Counter] = None,
              hist_bytes: Optional[collections.Counter] = None, lerobot_roots: Optional[set] = None) -> dict:
    """du -sb-like size of ``path`` (a symlinked ``path`` itself is followed once), capped by files and time."""
    link = os.readlink(path) if os.path.islink(path) else None
    if link is not None and not os.path.isdir(path):
        return {"bytes": 0, "files": 0, "dirs": 0, "truncated": False, "symlink": link}
    if os.path.isfile(path):
        size = os.lstat(path).st_size
        if hist is not None:
            hist[ext_of(path)] += 1
            hist_bytes[ext_of(path)] += size
        return {"bytes": size, "files": 1, "dirs": 0, "truncated": False}
    total = files = dirs = 0
    truncated = False
    for root, dnames, names in os.walk(path):
        dirs += 1
        if lerobot_roots is not None and os.path.basename(root) == "meta" and "info.json" in names:
            lerobot_roots.add(os.path.dirname(root))
        for name in names:
            if files >= max_files or (files % 1000 == 0 and budget.over()):
                truncated = True
                break
            try:
                size = os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
            total += size
            files += 1
            if hist is not None:
                hist[ext_of(name)] += 1
                hist_bytes[ext_of(name)] += size
        if truncated:
            break
    out = {"bytes": total, "files": files, "dirs": dirs, "truncated": truncated}
    if link is not None:
        out["symlink"] = link
    return out


def human(n: Optional[float]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def load_task_list(path: Optional[str]) -> list[str]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    return [t["id"] if isinstance(t, dict) else str(t) for t in tasks]


def load_task_misc(path: str) -> dict[int, str]:
    """B100_task_misc.csv "Task ID" -> "Task" (eval_utils._load_challenge_task_misc)."""
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["Task ID"]): row["Task"] for row in csv.DictReader(f) if row.get("Task ID", "").strip().isdigit()}


def task_of_name(stem: str, scene: Optional[str], known: set) -> Optional[str]:
    """Task name in an instance file / dir name: <scene>_task_<task>_instances or <scene>_task_<task>_0_<id>_template..."""
    if scene and stem.startswith(scene + "_task_"):
        rest = stem[len(scene) + len("_task_"):]
    elif "_task_" in stem:
        rest = stem.split("_task_", 1)[1]
    else:
        return None
    if rest.endswith("_instances"):
        return rest[: -len("_instances")]
    m = _INSTANCE_ID.search(rest)
    if m:
        return rest[: m.start()]
    candidates = [t for t in known if rest == t or rest.startswith(t + "_")]
    return max(candidates, key=len) if candidates else None


def scan_instances(split_dir: str, known: set, budget: Budget, max_files: int) -> tuple[dict, bool]:
    """({task: {"files", "ids", "scenes"}}, truncated) for one split directory (<split>/<scene>/json/...)."""
    out: dict = collections.defaultdict(lambda: {"files": 0, "ids": set(), "scenes": set()})
    seen, truncated = 0, False
    for scene in sorted(os.listdir(split_dir)) if os.path.isdir(split_dir) else []:
        scene_dir = os.path.join(split_dir, scene)
        if truncated or not os.path.isdir(scene_dir):
            continue
        for root, dnames, names in os.walk(scene_dir):
            for name in list(dnames) + names:
                stem = name.removesuffix(".json")
                task = task_of_name(stem, scene, known)
                if task is None:
                    continue
                rec = out[task]
                rec["scenes"].add(scene)
                if name in names:
                    rec["files"] += 1
                    m = _INSTANCE_ID.search(stem)
                    if m:
                        rec["ids"].add(int(m.group(2)))
            seen += len(names)
            if seen > max_files or budget.over():
                truncated = True
                break
    tasks = {t: {"files": r["files"], "ids": sorted(r["ids"]), "scenes": sorted(r["scenes"])} for t, r in out.items()}
    return tasks, truncated


def find_lerobot_roots(root: str, extra: set) -> list[str]:
    found = set(extra)
    for pattern in ("meta/info.json", "*/meta/info.json", "*/*/meta/info.json", "*/*/*/meta/info.json"):
        for p in glob.glob(os.path.join(root, pattern)):
            found.add(os.path.dirname(os.path.dirname(p)))
    return sorted(os.path.abspath(p) for p in found)


def read_jsonl(path: str, limit: int = 100_000) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def scan_lerobot(path: str, known: set) -> dict:
    rec: dict = {"path": path, "meta": {m: os.path.exists(os.path.join(path, "meta", m)) for m in LEROBOT_META}}
    info_path = os.path.join(path, "meta", "info.json")
    try:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        rec["info"] = {k: info.get(k) for k in ("codebase_version", "robot_type", "fps", "total_episodes",
                                                 "total_frames", "total_tasks", "chunks_size")}
        rec["info"]["features"] = sorted((info.get("features") or {}).keys())
    except (OSError, ValueError) as e:
        rec["info_error"] = f"{type(e).__name__}: {e}"
    tasks_jsonl = os.path.join(path, "meta", "tasks.jsonl")
    if os.path.isfile(tasks_jsonl):
        lines = read_jsonl(tasks_jsonl)
        named = {}
        for r in lines:
            for key in ("task_name", "task"):
                if isinstance(r.get(key), str) and r[key] in known:
                    named[r[key]] = r.get("task_index")
        rec["tasks_jsonl"] = {"lines": len(lines), "known_task_names": named,
                              "sample": lines[:3]}
    data_dir = os.path.join(path, "data")
    chunks = {}
    if os.path.isdir(data_dir):
        for d in sorted(os.listdir(data_dir)):
            m = _CHUNK.match(d)
            if m and os.path.isdir(os.path.join(data_dir, d)):
                chunks[int(m.group(1))] = len(os.listdir(os.path.join(data_dir, d)))
    rec["data_chunks"] = chunks
    videos_dir = os.path.join(path, "videos")
    rec["video_keys"] = sorted(d for d in os.listdir(videos_dir)) if os.path.isdir(videos_dir) else []
    ann_dir = os.path.join(path, "annotations")
    ann = {}
    if os.path.isdir(ann_dir):
        for d in sorted(os.listdir(ann_dir)):
            m = _TASK_DIR.match(d)
            if m and os.path.isdir(os.path.join(ann_dir, d)):
                ann[int(m.group(1))] = len(os.listdir(os.path.join(ann_dir, d)))
    rec["annotation_tasks"] = ann
    parts = os.path.normpath(path).split(os.sep)
    rec["task_from_path"] = next((p for p in reversed(parts[-2:]) if p in known), None)
    return rec


def inventory(root: str, task_data: Optional[str] = None, max_files: int = 300_000, time_limit_s: float = 600.0) -> dict:
    root = os.path.abspath(os.path.expanduser(root))
    budget = Budget(time_limit_s)
    out: dict = {"root": root, "inventoried_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "exists": os.path.isdir(root), "limits": {"max_files_per_top_level": max_files,
                                                           "time_limit_s": time_limit_s}}
    if not out["exists"]:
        out["error"] = f"{root} is not a directory"
        return out
    tasks = load_task_list(task_data)
    known = set(tasks)
    out["task_data"] = {"path": task_data, "tasks": len(tasks)}

    # task coverage first: the size walk below may use up the time budget on multi-TB asset trees
    out["instances"], instances_truncated = {}, False
    for split, rel in SPLITS.items():
        out["instances"][split], cut = scan_instances(os.path.join(root, rel), known, budget, max_files)
        instances_truncated = instances_truncated or cut
    out["instances_truncated"] = instances_truncated

    hist, hist_bytes, lerobot = collections.Counter(), collections.Counter(), set()
    top = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        top.append({"name": name, "kind": "link" if os.path.islink(p) else ("dir" if os.path.isdir(p) else "file"),
                    **walk_size(p, max_files, budget, hist, hist_bytes, lerobot)})
    out["top_level"] = top
    out["file_types"] = [{"ext": e, "files": n, "bytes": hist_bytes[e]} for e, n in hist.most_common(40)]

    known_items = {}
    for label, rel in KNOWN_ITEMS.items():
        p = os.path.join(root, rel)
        item = {"path": rel, "exists": os.path.exists(p)}
        if item["exists"] and label not in NO_READ:
            item["kind"] = "dir" if os.path.isdir(p) else "file"
            if item["kind"] == "file":
                item["bytes"] = os.path.getsize(p)
        if label in VERSION_FILES and os.path.isfile(os.path.join(p, "VERSION")):
            with open(os.path.join(p, "VERSION"), encoding="utf-8") as f:
                item["version"] = f.read().strip()[:80]
        known_items[label] = item
    out["known_items"] = known_items

    misc = os.path.join(root, KNOWN_ITEMS["B100_task_misc.csv"])
    if os.path.isfile(misc):
        id_to_name, mapping = load_task_misc(misc), "B100_task_misc.csv"
    else:
        id_to_name, mapping = dict(enumerate(tasks)), "task_data.json order (official file absent; assumed equal)"
    out["id_mapping"] = {"source": mapping, "n": len(id_to_name)}
    out["lerobot_roots"] = [scan_lerobot(p, known) for p in find_lerobot_roots(root, lerobot)]

    def covers(lr: dict, key: str, tid: int, name: str) -> bool:
        # a per-task root (.../<task>/meta/info.json) holds only that task; its chunk numbers are not task ids
        return lr["task_from_path"] == name if lr.get("task_from_path") else tid in lr[key]

    coverage = []
    for tid, name in sorted(id_to_name.items()):
        row = {"task_id": tid, "task": name}
        for split in SPLITS:
            r = out["instances"][split].get(name)
            row[split] = len(r["ids"]) if r else 0
        row["demo_chunks"] = sum(1 for lr in out["lerobot_roots"] if covers(lr, "data_chunks", tid, name))
        row["annotations"] = sum(1 for lr in out["lerobot_roots"] if covers(lr, "annotation_tasks", tid, name))
        coverage.append(row)
    listed = set(id_to_name.values())
    out["coverage"] = coverage
    out["coverage_summary"] = {
        "tasks_listed": len(listed),
        "instance_scan_truncated": instances_truncated,
        **{f"tasks_with_{split}_instances": sum(1 for r in coverage if r[split]) for split in SPLITS},
        "tasks_with_demo_data": sum(1 for r in coverage if r["demo_chunks"]),
        "tasks_with_annotations": sum(1 for r in coverage if r["annotations"]),
        "instance_tasks_not_in_task_list": sorted({t for s in out["instances"].values() for t in s} - listed),
    }
    out["truncated"] = budget.hit or instances_truncated or any(t.get("truncated") for t in top)
    return out


def render(inv: dict) -> str:
    lines = [f"# Datasets inventory: {inv['root']}", "", f"inventoried {inv['inventoried_at']}"]
    if not inv.get("exists"):
        return "\n".join(lines + ["", f"ERROR: {inv.get('error')}"]) + "\n"
    if inv["truncated"]:
        lines.append("**Some walks hit the file or time cap: sizes marked >= are lower bounds.**")
    if inv.get("instances_truncated"):
        lines.append("**The instance scan hit the file or time cap: instance counts in the coverage table are lower "
                     "bounds (raise --max-files / --time-limit-s).**")
    lines += ["", "## Top level", "", "| entry | kind | size | files |", "|---|---|---:|---:|"]
    for t in inv["top_level"]:
        ge = ">= " if t.get("truncated") else ""
        kind = f"link -> {t['symlink']}" if t.get("symlink") else t["kind"]
        lines.append(f"| {t['name']} | {kind} | {ge}{human(t['bytes'])} | {ge}{t['files']} |")
    lines += ["", "## File types (top 40 by count)", "", "| ext | files | size |", "|---|---:|---:|"]
    lines += [f"| {e['ext']} | {e['files']} | {human(e['bytes'])} |" for e in inv["file_types"]]
    lines += ["", "## Known items", "", "| item | path | present | note |", "|---|---|---|---|"]
    for label, item in inv["known_items"].items():
        note = item.get("version") or (human(item["bytes"]) if "bytes" in item else "")
        lines.append(f"| {label} | {item['path']} | {'yes' if item['exists'] else 'NO'} | {note} |")
    lines += ["", "## LeRobot (demo) roots"]
    if not inv["lerobot_roots"]:
        lines.append("none found (no meta/info.json within 4 levels)")
    for lr in inv["lerobot_roots"]:
        info = lr.get("info") or {}
        lines.append(f"- {lr['path']}: meta {', '.join(k for k, v in lr['meta'].items() if v) or 'none'}; "
                     f"episodes {info.get('total_episodes')}, frames {info.get('total_frames')}, fps {info.get('fps')}, "
                     f"tasks {info.get('total_tasks')}; data chunks {sorted(lr['data_chunks'])[:20]}"
                     f"{' ...' if len(lr['data_chunks']) > 20 else ''}; videos {len(lr['video_keys'])} keys; "
                     f"annotation tasks {len(lr['annotation_tasks'])}"
                     + (f"; per-task root for {lr['task_from_path']}" if lr.get("task_from_path") else ""))
    cs = inv["coverage_summary"]
    lines += ["", f"## Task coverage (ids -> names via {inv['id_mapping']['source']})", ""]
    lines += [f"- {k}: {v}" for k, v in cs.items()]
    lines += ["", "| id | task | train | public_test | hidden_test | demo | annotations |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in inv["coverage"]:
        lines.append(f"| {r['task_id']} | {r['task']} | {r['train']} | {r['public_test']} | {r['hidden_test']} | "
                     f"{r['demo_chunks']} | {r['annotations']} |")
    return "\n".join(lines) + "\n"


def default_task_data(root: str) -> Optional[str]:
    cands = []
    if os.environ.get("B1K_ROOT"):
        cands.append(os.path.join(os.environ["B1K_ROOT"], "docs", "challenge", "task_data.json"))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(root)), "docs", "challenge", "task_data.json"))
    return next((c for c in cands if os.path.isfile(c)), None)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=os.environ.get("OMNIGIBSON_DATA_PATH") or DEFAULT_ROOT)
    ap.add_argument("--task-data", default=None, help="docs/challenge/task_data.json (default $B1K_ROOT/..., <root>/../...)")
    ap.add_argument("--out", default=None, help="JSON output")
    ap.add_argument("--md", default=None, help="markdown output (default: next to --out)")
    ap.add_argument("--max-files", type=int, default=300_000, help="files counted per top-level entry")
    ap.add_argument("--time-limit-s", type=float, default=600.0)
    args = ap.parse_args(argv)
    task_data = args.task_data or default_task_data(args.root)
    inv = inventory(args.root, task_data=task_data, max_files=args.max_files, time_limit_s=args.time_limit_s)
    text = render(inv)
    # files first: takeover_check.sh pipes stdout through head, which may close the pipe early
    written = []
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(inv, f, indent=2, sort_keys=True, default=str)
        written.append(args.out)
    md = args.md or (os.path.splitext(args.out)[0] + ".md" if args.out else None)
    if md:
        with open(md, "w", encoding="utf-8") as f:
            f.write(text)
        written.append(md)
    print(text, end="")
    if written:
        print("wrote " + " and ".join(written))
    return 0 if inv.get("exists") else 1


if __name__ == "__main__":
    sys.exit(main())
