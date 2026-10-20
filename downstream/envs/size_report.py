#!/usr/bin/env python3
"""Sum the download size of what `pip install --dry-run --report REPORT.json` would install (HEAD requests only).

    python downstream/envs/size_report.py REPORT.json --out sizes.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            value = resp.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    report = json.loads(args.report.read_text())
    rows = []
    for item in report.get("install", []):
        meta = item.get("metadata", {})
        url = item.get("download_info", {}).get("url", "")
        rows.append({"name": meta.get("name"), "version": meta.get("version"), "url": url,
                     "bytes": head_size(url) if url.startswith("http") else None})
    rows.sort(key=lambda r: -(r["bytes"] or 0))
    total = sum(r["bytes"] or 0 for r in rows)
    unknown = [r["name"] for r in rows if r["bytes"] is None]
    out = {"packages": len(rows), "download_bytes": total, "download_gb": round(total / 1e9, 3),
           "unknown_size": unknown, "rows": rows}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("packages", "download_gb", "unknown_size")}))
    for r in rows[:15]:
        print(f"{(r['bytes'] or 0) / 1e6:9.1f} MB  {r['name']}=={r['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
