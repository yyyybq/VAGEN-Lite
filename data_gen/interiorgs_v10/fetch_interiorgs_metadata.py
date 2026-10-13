"""Download only the InteriorGS metadata the generator needs (no Gaussian PLY files).

The generator reads, per scene, ``labels.json`` and ``structure.json``
(occupancy files are kept if present). Two sources:

* ``--source hf`` (default): Hugging Face dataset spatialverse/InteriorGS.
  Written without network access to Hugging Face; run ``--list-only`` first.
* ``--source aoss``: the cluster's AOSS copy (``fj:s3://baiqiao/InteriorGS``,
  all 1,000 scenes). Needs a Python with ``aoss_client`` (on fujian:
  ``.conda/envs/probe-spatial``). The internal endpoint is not reachable through
  the login-node HTTP proxy, so proxy variables are dropped in this mode.

    export HF_TOKEN=...            # or --token-file path/to/.hf_token
    python -m data_gen.interiorgs_v10.fetch_interiorgs_metadata --out /data/InteriorGS_meta --list-only
    python -m data_gen.interiorgs_v10.fetch_interiorgs_metadata --out /data/InteriorGS_meta --workers 8
    python -m data_gen.interiorgs_v10.fetch_interiorgs_metadata --source aoss --out /mnt/umm/users/yinbaiqiao/InteriorGS_meta

The output mirrors the repository layout, so ``run_all --scenes-root OUT`` finds
every directory that contains a labels.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional

API = "https://huggingface.co/api/datasets/{repo}/tree/{rev}"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
WANTED_NAMES = {"labels.json", "structure.json"}
WANTED_PREFIXES = ("occupancy",)


def _token(args: argparse.Namespace) -> Optional[str]:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    for cand in [args.token_file, ".hf_token", str(Path.home() / ".cache/huggingface/token")]:
        if cand and Path(cand).is_file():
            return Path(cand).read_text().strip()
    return None


def _get(url: str, token: Optional[str], timeout: float = 60.0) -> urllib.request.addinfourl:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=timeout)


def list_tree(repo: str, rev: str, token: Optional[str], path: str = "") -> Iterator[Dict]:
    """Recursive listing with cursor pagination (Link: <...>; rel="next")."""
    url = API.format(repo=repo, rev=rev) + (("/" + urllib.parse.quote(path)) if path else "") + "?recursive=true"
    while url:
        with _get(url, token) as resp:
            entries = json.loads(resp.read().decode())
            link = resp.headers.get("Link") or ""
        for e in entries:
            yield e
        url = ""
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")


def wanted(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in WANTED_NAMES or name.startswith(WANTED_PREFIXES)


def download(repo: str, rev: str, token: Optional[str], rel: str, size: Optional[int], out: Path,
             retries: int = 4) -> str:
    target = out / rel
    if target.is_file() and (size is None or target.stat().st_size == size):
        return "skip"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = RESOLVE.format(repo=repo, rev=rev, path=urllib.parse.quote(rel))
    for attempt in range(retries):
        try:
            with _get(url, token, timeout=120.0) as resp:
                data = resp.read()
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)
            return "ok"
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                return f"error:{exc}"
            time.sleep(2 ** attempt)
    return "error"


AOSS_FILES = ("labels.json", "structure.json", "occupancy.json", "occupancy.png")
_PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")


def _aoss_client(conf: str):
    for var in _PROXY_VARS:
        os.environ.pop(var, None)
    from aoss_client.client import Client  # only installed on the cluster
    return Client(conf)


def aoss_download(client, root: str, rel: str, out: Path, retries: int = 4) -> str:
    target = out / rel
    if target.is_file() and target.stat().st_size > 0:
        return "skip"
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            data = client.get(f"{root}/{rel}")
            if data is None:
                return "missing"
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(bytes(data))
            tmp.replace(target)
            return "ok"
        except Exception as exc:  # aoss_client raises its own exception types
            if attempt == retries - 1:
                return f"error:{type(exc).__name__}"
            time.sleep(2 ** attempt)
    return "error"


def aoss_jobs(scenes: List[str], files: Optional[List[str]] = None,
              only: Optional[List[str]] = None) -> List[str]:
    """Relative paths to fetch: ``files`` (default AOSS_FILES) of every scene, or of ``only`` those scenes."""
    if only is not None:
        keep = set(only)
        missing = sorted(keep - set(scenes))
        if missing:
            raise ValueError(f"scenes not on AOSS: {missing[:5]}")
        scenes = [s for s in scenes if s in keep]
    return [f"{s}/{name}" for s in scenes for name in (files or list(AOSS_FILES))]


def main_aoss(args: argparse.Namespace) -> int:
    root = args.aoss_root.rstrip("/")
    out = Path(args.out)
    client = _aoss_client(args.aoss_conf)
    scenes = sorted(n.rstrip("/") for n in client.list(root + "/")
                    if n.endswith("/") and n[:4].isdigit())
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    files = [f for f in (args.files or "").split(",") if f] or list(AOSS_FILES)
    only = Path(args.scenes_file).read_text().split() if args.scenes_file else None
    jobs = aoss_jobs(scenes, files, only)
    print(json.dumps({"scenes": len(scenes) if only is None else len(set(only)), "example": scenes[:3],
                      "files_per_scene": files}), flush=True)
    if args.list_only:
        return 0
    counts: Dict[str, int] = {}
    missing_required: List[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(aoss_download, client, root, rel, out): rel for rel in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            counts[r.split(":", 1)[0]] = counts.get(r.split(":", 1)[0], 0) + 1
            if r != "ok" and r != "skip" and futs[f].rsplit("/", 1)[-1] in WANTED_NAMES:
                missing_required.append(f"{futs[f]}:{r}")
            if i % 400 == 0:
                print(f"[fetch] {i}/{len(futs)} {counts}", flush=True)
    print(json.dumps({"done": counts, "missing_required": sorted(missing_required)}, indent=1))
    return 0 if not missing_required else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("hf", "aoss"), default="hf")
    ap.add_argument("--repo", default="spatialverse/InteriorGS")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--aoss-root", default="fj:s3://baiqiao/InteriorGS")
    ap.add_argument("--aoss-conf", default="/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--token-file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--limit-scenes", type=int)
    ap.add_argument("--scenes-file", help="aoss: only these scene ids (one per line)")
    ap.add_argument("--files", help="aoss: comma-separated file names per scene (default: the metadata files; "
                                    "add 3dgs_compressed.ply for rendering)")
    args = ap.parse_args(argv)
    if args.source == "aoss":
        return main_aoss(args)
    token = _token(args)
    out = Path(args.out)
    files = []
    other_ext: Dict[str, int] = {}
    for e in list_tree(args.repo, args.revision, token):
        if e.get("type") != "file":
            continue
        p = e["path"]
        if wanted(p):
            files.append((p, e.get("size")))
        else:
            ext = p.rsplit(".", 1)[-1] if "." in p else "<none>"
            other_ext[ext] = other_ext.get(ext, 0) + 1
    scenes = sorted({p.rsplit("/", 1)[0] for p, _ in files if p.endswith("labels.json")})
    if args.limit_scenes:
        keep = set(scenes[: args.limit_scenes])
        files = [(p, s) for p, s in files if p.rsplit("/", 1)[0] in keep]
    print(json.dumps({"metadata_files": len(files), "scenes_with_labels": len(scenes),
                      "other_files_by_extension": other_ext, "example": files[:5]}, indent=1), flush=True)
    if not scenes and other_ext:
        print("No per-scene labels.json found: the dataset may ship archives; inspect other_files_by_extension.",
              file=sys.stderr)
    if args.list_only:
        return 0
    counts: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(download, args.repo, args.revision, token, p, s, out) for p, s in files]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            key = r.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
            if i % 200 == 0:
                print(f"[fetch] {i}/{len(futs)} {counts}", flush=True)
    print(json.dumps({"done": counts}, indent=1))
    return 0 if not counts.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
