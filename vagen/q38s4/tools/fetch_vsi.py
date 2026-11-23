"""C line (q38 s4): download, verify and unpack VSI-590K's 9 media archives (235.65 GB; the user's yes of 2026-11-23).

  python -m vagen.q38s4.tools.fetch_vsi probe   --out DIR          # can this machine reach the mirror? speed
  python -m vagen.q38s4.tools.fetch_vsi fetch   --out DIR [--only robotics,s3dis] [--conns 8]
  python -m vagen.q38s4.tools.fetch_vsi extract --out DIR
  python -m vagen.q38s4.tools.fetch_vsi check   --out DIR
  python -m vagen.q38s4.tools.fetch_vsi all     --out DIR          # probe, fetch, extract, check

- Source: hf-mirror.com (no proxy; the configured proxy gives 0.2 MB/s, STAGE0_REPORT section 1), dataset
  nyu-visionx/VSI-590K; sizes and sha256 (the LFS oids) are fixed in FILES.  Smallest archive first.
- fetch: each archive is cut into --chunk-mb pieces fetched by --conns parallel range requests into
  DIR/_fetch/<name>.part (written in place; finished pieces in <name>.chunks.json, so a stopped run resumes).  A
  complete .part is hashed; only a matching sha256 renames it to DIR/<name>.  A mismatch stops (the .part stays).
- extract: each archive is unpacked into DIR/_extract/<name>.<n>; when its only top-level entry is a folder named
  after its source, that folder is moved to DIR/<source>, else the whole unpacked tree becomes DIR/<source>.  An existing DIR/<source> is
  never written into.  The archives stay (whether to delete them is the user's call).
- check: every video / image path of DIR/vsi_590k.jsonl must exist under DIR; per source 20 videos (seeded) are
  opened with PyAV and one frame decoded, 20 images opened with PIL.  DIR/_fetch/check.json; exit 1 on a miss.
- Nothing is ever deleted or overwritten; DIR/_fetch/status.json records each step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

BASE = "https://hf-mirror.com/datasets/nyu-visionx/VSI-590K/resolve/main"


@dataclass(frozen=True)
class Archive:
    name: str
    size: int
    sha256: str

    @property
    def source(self) -> str:
        return self.name.split(".", 1)[0]


FILES = [Archive(*x) for x in sorted([
    ("arkitscenes.tar.gz", 90_154_741_781, "4c54a042dd9c3ca808a61d05536598878d93a1b395b668551daec19895da9728"),
    ("ytb_roomtour.tar.gz", 55_261_730_714, "98699f63d65223e04034b372d56ff53919819ab90885e2a3f0c7ce317d3e0946"),
    ("scannet.tar.gz", 27_632_592_849, "8a8a07b37ad3e52c286bdc60118c6445a6543f19b9423007d60a9485912f9ab2"),
    ("adt.tar.gz", 23_895_961_889, "bf9fa6b79ad9e529210eb57df114372de2ea452fe0871d569036132e05f87c1b"),
    ("scannetppv2.tar.gz", 21_438_045_428, "3ef0b2f7aa0cc30ad2508a19a52af7239e0160be809d7341ee57198777d7ea3b"),
    ("hypersim.tar.gz", 6_775_393_985, "8376d276022beac3ab2a8af8a4ef23890a784c9c7da126cba6e1eca1a763a847"),
    ("procthor.tar.gz", 6_523_652_708, "881d7be20b7d67b4a4bd76f23082c48623c2e68cf02ebb661eeaa1f1bffb7bc6"),
    ("s3dis.tar.gz", 2_865_916_732, "c97c0dfd60430bef6365f4adfc376c1a94ff0e0ce50909ca84df0e2396ff1b53"),
    ("robotics.tar.gz", 1_104_873_829, "9bae84fac90fcfa7f86d4a7dfb75f3798da241d7211ec10a512d2e3c01ba5816"),
], key=lambda x: x[1])]
TOTAL_BYTES = 235_652_909_915  # STAGE0_REPORT first said 235,653,909,915 (an addition slip)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never the configured proxy
PIECE = 4 << 20


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[fetch_vsi {utc()}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ state
class Status:
    def __init__(self, out: Path):
        self.path = out / "_fetch" / "status.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def set(self, name: str, **kw) -> None:
        self.data.setdefault(name, {}).update(kw, utc=utc())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1) + "\n")
        os.replace(tmp, self.path)


def chunk_plan(size: int, chunk: int) -> List[tuple]:
    return [(i, s, min(s + chunk, size) - 1) for i, s in enumerate(range(0, size, chunk))]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


# ------------------------------------------------------------------------------------------------ fetch
def get_range(url: str, start: int, end: int, fd: int, timeout: float) -> int:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "fetch-vsi/1"})
    got = 0
    with OPENER.open(req, timeout=timeout) as r:
        if r.status != 206:
            raise IOError(f"HTTP {r.status} for a range request")
        while True:
            b = r.read(PIECE)
            if not b:
                break
            if start + got + len(b) > end + 1:
                raise IOError("server sent more than the range")
            os.pwrite(fd, b, start + got)
            got += len(b)
    if got != end - start + 1:
        raise IOError(f"short range: {got} of {end - start + 1} bytes")
    return got


def fetch_one(a: Archive, out: Path, base: str, conns: int, chunk: int, retries: int = 8, timeout: float = 120,
              sleep: Callable[[float], None] = time.sleep) -> Path:
    final = out / a.name
    st = Status(out)
    if final.exists():
        if st.data.get(a.name, {}).get("verified") != a.sha256:
            raise SystemExit(f"{final} exists but status.json has no verified sha256 for it: check by hand")
        log(f"{a.name}: verified earlier")
        return final
    part = out / "_fetch" / (a.name + ".part")
    done_path = out / "_fetch" / (a.name + ".chunks.json")
    plan = chunk_plan(a.size, chunk)
    done = set(json.loads(done_path.read_text())["done"]) if done_path.is_file() else set()
    if done_path.is_file() and json.loads(done_path.read_text())["chunk"] != chunk:
        raise SystemExit(f"{done_path}: written with another --chunk-mb")
    fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, a.size)
    lock = threading.Lock()
    url = f"{base}/{a.name}"
    todo = [c for c in plan if c[0] not in done]
    t0, got0 = time.time(), [0]
    log(f"{a.name}: {len(todo)} of {len(plan)} chunks to fetch, {conns} connections")

    def save_done():
        tmp = done_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"chunk": chunk, "done": sorted(done)}))
        os.replace(tmp, done_path)

    def work(c):
        i, s, e = c
        for k in range(retries):
            try:
                n = get_range(url, s, e, fd, timeout)
                with lock:
                    done.add(i)
                    got0[0] += n
                    save_done()
                    if len(done) % 20 == 0 or len(done) == len(plan):
                        dt = max(time.time() - t0, 1e-6)
                        log(f"{a.name}: {len(done)}/{len(plan)} chunks, {got0[0] / dt / 1e6:.1f} MB/s")
                return
            except Exception as ex:  # noqa: BLE001  network errors of any kind: retry
                log(f"{a.name}: chunk {i} try {k + 1} failed: {type(ex).__name__}: {ex}")
                sleep(min(60, 2 ** k))
        raise IOError(f"{a.name}: chunk {i} failed {retries} times")

    try:
        with ThreadPoolExecutor(conns) as ex:
            list(ex.map(work, todo))
        os.fsync(fd)
    finally:
        os.close(fd)
    log(f"{a.name}: hashing")
    digest = sha256_of(part)
    if digest != a.sha256:
        st.set(a.name, sha256_mismatch=digest)
        raise SystemExit(f"{a.name}: sha256 {digest} != {a.sha256} (the .part stays in _fetch/)")
    os.rename(part, final)
    st.set(a.name, verified=a.sha256, bytes=a.size)
    log(f"{a.name}: sha256 ok")
    return final


# ------------------------------------------------------------------------------------------------ extract
def extract_one(a: Archive, out: Path) -> Path:
    st = Status(out)
    tar, dest = out / a.name, out / a.source
    if st.data.get(a.name, {}).get("extracted"):
        log(f"{a.name}: extracted earlier")
        return dest
    if st.data.get(a.name, {}).get("verified") != a.sha256:
        raise SystemExit(f"{a.name}: not verified")
    if dest.exists():
        raise SystemExit(f"{dest} exists and status.json does not say it was extracted here: check by hand")
    n = 0
    while (out / "_extract" / f"{a.name}.{n}").exists():  # an earlier attempt stays as it is
        n += 1
    work = out / "_extract" / f"{a.name}.{n}"
    work.mkdir(parents=True)
    log(f"{a.name}: extracting into {work}")
    gz = ["-I", "pigz"] if shutil.which("pigz") else ["-z"]
    subprocess.run(["tar", *gz, "-x", "--no-same-owner", "-f", str(tar), "-C", str(work)], check=True)
    tops = sorted(os.listdir(work))
    log(f"{a.name}: top level {tops[:5]}{' ...' if len(tops) > 5 else ''}")
    if tops == [a.source] and (work / a.source).is_dir():
        os.rename(work / a.source, dest)
        work.rmdir()  # empty now
        layout = "folder"
    else:
        os.rename(work, dest)
        layout = "root"
    st.set(a.name, extracted=str(dest), layout=layout, top_level=tops[:20])
    log(f"{a.name}: -> {dest} ({layout})")
    return dest


# ------------------------------------------------------------------------------------------------ check
def check(out: Path, sample: int = 20, seed: int = 0) -> Dict:
    rows = [json.loads(x) for x in open(out / "vsi_590k.jsonl")]
    vids, imgs = {}, {}
    for r in rows:
        for key, bucket in (("video", vids), ("image", imgs)):
            if r.get(key):
                p = r[key]
                bucket.setdefault(p.split("/", 1)[0], set()).add(p)
    res: Dict = {"utc": utc(), "rows": len(rows), "sources": {}, "missing_examples": [], "decode_failures": []}
    rng = random.Random(seed)
    for kind, bucket in (("video", vids), ("image", imgs)):
        for src, paths in sorted(bucket.items()):
            miss = sorted(p for p in paths if not (out / p).is_file())
            res["sources"][f"{kind}/{src}"] = {"distinct": len(paths), "missing": len(miss)}
            res["missing_examples"] += miss[:5]
            have = sorted(paths - set(miss))
            for p in rng.sample(have, min(sample, len(have))):
                try:
                    if kind == "video":
                        import llamafactory.data.mm_plugin  # noqa: F401  before PyAV (see lf_plugin)
                        import av

                        with av.open(str(out / p)) as c:
                            next(c.decode(video=0))
                    else:
                        from PIL import Image

                        with Image.open(out / p) as im:
                            im.load()
                except Exception as ex:  # noqa: BLE001
                    res["decode_failures"].append(f"{p}: {type(ex).__name__}: {ex}")
    res["ok"] = not res["missing_examples"] and not res["decode_failures"]
    (out / "_fetch").mkdir(exist_ok=True)
    (out / "_fetch" / "check.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


# ------------------------------------------------------------------------------------------------ probe
def probe(base: str, mb: int = 32) -> Dict:
    a = FILES[0]
    t0 = time.time()
    req = urllib.request.Request(f"{base}/{a.name}", headers={"Range": f"bytes=0-{mb * 1048576 - 1}",
                                                              "User-Agent": "fetch-vsi/1"})
    with OPENER.open(req, timeout=60) as r:
        n = len(r.read())
        status = r.status
    dt = time.time() - t0
    res = {"file": a.name, "status": status, "bytes": n, "seconds": round(dt, 2), "MB_s": round(n / dt / 1e6, 2)}
    log(f"probe {json.dumps(res)}")
    if status != 206 or n != mb * 1048576:
        raise SystemExit(f"probe failed: {res}")
    return res


def selected(only: Optional[str]) -> List[Archive]:
    if not only:
        return FILES
    names = {x.strip() for x in only.split(",")}
    pick = [a for a in FILES if a.source in names or a.name in names]
    if len(pick) != len(names):
        raise SystemExit(f"--only {only!r}: unknown names")
    return pick


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("probe", "fetch", "extract", "check", "all"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--conns", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if not (out / "vsi_590k.jsonl").is_file():
        raise SystemExit(f"{out}/vsi_590k.jsonl missing")
    files = selected(a.only)
    if a.cmd in ("probe", "all"):
        probe(a.base)
    if a.cmd in ("fetch", "all"):
        for f in files:
            fetch_one(f, out, a.base, a.conns, a.chunk_mb << 20)
    if a.cmd in ("extract", "all"):
        for f in files:
            extract_one(f, out)
    if a.cmd in ("check", "all"):
        res = check(out)
        log(f"check ok={res['ok']} " + json.dumps(res["sources"]))
        if not res["ok"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
