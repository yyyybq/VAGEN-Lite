"""Download the datasets and models listed in eval_external/assets.yaml at their pinned revisions.

    python eval_external/tools/fetch_assets.py datasets sat_real vstibench
    python eval_external/tools/fetch_assets.py models SI-7B SpaceR-7B
    python eval_external/tools/fetch_assets.py models --all

Each finished item gets ROOT/<dir>/.fetch_receipt.json (repo, revision, files with sizes, UTC time).  An item
whose receipt already names the same revision is skipped.  Uses HF_ENDPOINT if set (e.g. https://hf-mirror.com).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent.parent
ATTEMPTS = 5


def load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch(root: Path, kind: str, name: str, spec: dict) -> dict:
    target = root / spec["dir"]
    receipt = target / ".fetch_receipt.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("revision") == spec["revision"]:
        print(f"skip {name}: already at {spec['revision'][:10]}")
        return json.loads(receipt.read_text())
    target.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):  # the mirror drops a connection now and then; downloads resume
        try:
            snapshot_download(
                repo_id=spec["repo"],
                repo_type="dataset" if kind == "datasets" else "model",
                revision=spec["revision"],
                local_dir=str(target),
                allow_patterns=spec.get("allow"),
                max_workers=4,
            )
            break
        except Exception as e:
            if attempt == ATTEMPTS:
                raise
            print(f"{name}: attempt {attempt} failed ({type(e).__name__}); retrying in {30 * attempt}s", flush=True)
            time.sleep(30 * attempt)
    files = sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name != receipt.name)
    out = {
        "name": name,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": {str(p.relative_to(target)): p.stat().st_size for p in files},
        "total_gb": round(sum(p.stat().st_size for p in files) / 1e9, 3),
    }
    with open(receipt, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"done {name}: {out['total_gb']} GB in {target}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("datasets", "models"))
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--assets", type=Path, default=HERE / "assets.yaml")
    args = ap.parse_args()
    cfg = load(args.assets)
    items = cfg[args.kind]
    names = list(items) if args.all else args.names
    unknown = [n for n in names if n not in items]
    if unknown or not names:
        raise SystemExit(f"unknown or no names {unknown}; known: {list(items)}")
    for n in names:
        fetch(Path(cfg["root"]), args.kind, n, items[n])


if __name__ == "__main__":
    main()
