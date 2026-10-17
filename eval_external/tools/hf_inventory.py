"""Size, revision and access of Hugging Face repos, without downloading them (step 0 of eval_external).

    python eval_external/tools/hf_inventory.py --out eval_external/step0/hf_inventory.json \
        dataset:nyu-visionx/VSI-Bench model:kagakouko/Spatial-Interactor-Qwen2.5-VL-7B ...

Each item is TYPE:REPO (TYPE is dataset or model).  For each repo it records the commit sha, the last change,
whether it is gated, the number of files, the total size and the size of each top-level folder.  It reads
HF_ENDPOINT (default https://huggingface.co); a repo that cannot be read is recorded with its error.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def fetch(kind: str, repo: str) -> dict:
    url = f"{ENDPOINT}/api/{kind}s/{repo}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "eval_external-inventory"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def summarise(kind: str, repo: str) -> dict:
    try:
        info = fetch(kind, repo)
    except urllib.error.HTTPError as e:
        return {"kind": kind, "repo": repo, "error": f"HTTP {e.code}"}
    except Exception as e:  # network errors are recorded, not raised
        return {"kind": kind, "repo": repo, "error": repr(e)}
    files = info.get("siblings", [])
    by_top = collections.Counter()
    total = 0
    for f in files:
        size = f.get("size") or (f.get("lfs") or {}).get("size") or 0
        total += size
        top = f["rfilename"].split("/", 1)[0] if "/" in f["rfilename"] else "."
        by_top[top] += size
    return {
        "kind": kind,
        "repo": repo,
        "sha": info.get("sha"),
        "last_modified": info.get("lastModified"),
        "gated": info.get("gated"),
        "private": info.get("private"),
        "n_files": len(files),
        "total_gb": round(total / 1e9, 3),
        "top_level_gb": {k: round(v / 1e9, 3) for k, v in by_top.most_common(15)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+", help="dataset:REPO or model:REPO")
    ap.add_argument("--out", help="write the list as JSON here")
    args = ap.parse_args()
    rows = []
    for item in args.items:
        kind, _, repo = item.partition(":")
        if kind not in ("dataset", "model") or not repo:
            sys.exit(f"bad item {item!r}: use dataset:REPO or model:REPO")
        row = summarise(kind, repo)
        rows.append(row)
        if "error" in row:
            print(f"{kind:7s} {repo:55s} ERROR {row['error']}")
        else:
            print(f"{kind:7s} {repo:55s} {row['total_gb']:9.2f} GB  files={row['n_files']:<6d} "
                  f"gated={row['gated']} sha={str(row['sha'])[:10]} modified={row['last_modified']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="\n") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
