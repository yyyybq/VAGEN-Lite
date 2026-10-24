"""Model line, step 2 (the user's decision 1 of 2026-10-24): download Cambrian-S-7B and SigLIP2 so400m-patch14-384
from hf-mirror.com into an HF cache layout under the shared model directory, then check every file against the
manifest the HF API gives (size for every file, sha256 for the LFS ones).  Only these two repositories.

    python -I fetch_cambrian.py manifest --out RECORD_DIR      (huggingface.co's API, through the login node's proxy)
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python -I fetch_cambrian.py download|verify --out RECORD_DIR
                                                             (without proxy variables: hf-mirror is reached directly,
                                                              the proxy gives HF ~0.1 MB/s; hf-mirror's API redirects)
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ENDPOINT = "https://hf-mirror.com"
API = "https://huggingface.co"
CACHE = "/mnt/umm/shared_model/vagen-lite-models/hub"
REPOS = {"nyu-visionx/Cambrian-S-7B": "556e72fded09352f6aa4e9c100c9f39e6fe0fba7",
         "google/siglip2-so400m-patch14-384": None}  # None: the revision the API names now (recorded)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def api(repo, rev):
    url = f"{API}/api/models/{repo}" + (f"/revision/{rev}" if rev else "") + "?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def cmd_manifest(out: Path):
    man = {"utc": now(), "manifest_from": API, "download_from": ENDPOINT, "cache": CACHE, "repos": {}}
    for repo, rev in REPOS.items():
        d = api(repo, rev)
        files = [{"name": s["rfilename"], "size": s.get("size"),
                  "sha256": (s.get("lfs") or {}).get("sha256"), "blob_id": s.get("blobId")}
                 for s in d.get("siblings") or []]
        man["repos"][repo] = {"revision": d["sha"], "asked_revision": rev, "files": files,
                              "total_bytes": sum(f["size"] or 0 for f in files)}
        if rev and d["sha"] != rev:
            raise SystemExit(f"{repo}: the API names revision {d['sha']}, not {rev}")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({r: (v["revision"], len(v["files"]), v["total_bytes"]) for r, v in man["repos"].items()}))


def cmd_download(out: Path):
    from huggingface_hub import snapshot_download

    man = json.loads((out / "manifest.json").read_text())
    log = {"started": now(), "repos": {}}
    for repo, info in man["repos"].items():
        t0 = datetime.datetime.now()
        path = snapshot_download(repo_id=repo, revision=info["revision"], cache_dir=CACHE, max_workers=8)
        log["repos"][repo] = {"snapshot": path, "seconds": (datetime.datetime.now() - t0).total_seconds()}
        print(repo, path, flush=True)
    log["ended"] = now()
    (out / "download.json").write_text(json.dumps(log, indent=1))


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def cmd_verify(out: Path):
    man = json.loads((out / "manifest.json").read_text())
    report, ok = {"utc": now(), "repos": {}}, True
    for repo, info in man["repos"].items():
        snap = Path(CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / info["revision"]
        rows = []
        for f in info["files"]:
            p = snap / f["name"]
            row = {"name": f["name"], "exists": p.is_file()}
            if row["exists"]:
                row["size"] = p.stat().st_size
                row["size_ok"] = f["size"] is None or row["size"] == f["size"]
                if f["sha256"]:
                    row["sha256"] = sha256(p)
                    row["sha256_ok"] = row["sha256"] == f["sha256"]
            good = row["exists"] and row.get("size_ok", False) and row.get("sha256_ok", True)
            ok &= good
            rows.append(row)
        extra = sorted({str(q.relative_to(snap)) for q in snap.rglob("*") if q.is_file()} - {f["name"] for f in info["files"]})
        report["repos"][repo] = {"snapshot": str(snap), "files": rows, "unexpected_files": extra,
                                 "all_ok": all(r["exists"] and r.get("size_ok") and r.get("sha256_ok", True) for r in rows)}
    report["all_ok"] = ok
    (out / "verify.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({r: v["all_ok"] for r, v in report["repos"].items()}), "all_ok", ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "download", "verify"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd != "manifest":
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            if os.environ.get(k):
                raise SystemExit(f"unset {k} first (hf-mirror is reached directly)")
        if os.environ.get("HF_ENDPOINT") != ENDPOINT or os.environ.get("HF_HUB_DISABLE_XET") != "1":
            raise SystemExit(f"run with HF_ENDPOINT={ENDPOINT} HF_HUB_DISABLE_XET=1")
    a.out.mkdir(parents=True, exist_ok=True)
    {"manifest": cmd_manifest, "download": cmd_download, "verify": cmd_verify}[a.cmd](a.out)
