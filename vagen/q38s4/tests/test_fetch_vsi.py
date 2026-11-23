"""CPU tests of vagen.q38s4.tools.fetch_vsi: ranged fetch from a local server, resume, sha256, extract, check."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vagen.q38s4.tools import fetch_vsi as F


class RangeHandler(SimpleHTTPRequestHandler):
    fail_once: set = set()

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = Path(self.directory) / self.path.lstrip("/")
        data = p.read_bytes()
        rng = self.headers.get("Range")
        s, e = (int(x) for x in rng.split("=")[1].split("-"))
        e = min(e, len(data) - 1)
        if (self.path, s) not in RangeHandler.fail_once and "flaky" in self.path:
            RangeHandler.fail_once.add((self.path, s))
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Length", str(e - s + 1))
        self.end_headers()
        self.wfile.write(data[s:e + 1])


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(root)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def make_tar(path: Path, top: str, files: dict) -> F.Archive:
    src = path.parent / ("src_" + path.name)
    for rel, data in files.items():
        q = src / top / rel if top else src / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(data)
    with tarfile.open(path, "w:gz") as t:
        for x in ([src / top] if top else sorted(src.iterdir())):
            t.add(x, arcname=x.name)
    b = path.read_bytes()
    return F.Archive(path.name, len(b), hashlib.sha256(b).hexdigest())


def test_files_table():
    assert sum(a.size for a in F.FILES) == F.TOTAL_BYTES and len(F.FILES) == 9
    assert [a.size for a in F.FILES] == sorted(a.size for a in F.FILES)
    assert F.FILES[0].source == "robotics" and {a.source for a in F.FILES} >= {"scannet", "arkitscenes", "hypersim"}
    assert F.chunk_plan(10, 4) == [(0, 0, 3), (1, 4, 7), (2, 8, 9)]


def test_fetch_resume_and_retry(server, tmp_path):
    root, base = server
    out = tmp_path / "out"
    out.mkdir()
    blob = os.urandom(300_000)
    (root / "flaky.tar.gz").write_bytes(blob)
    a = F.Archive("flaky.tar.gz", len(blob), hashlib.sha256(blob).hexdigest())
    # a stopped run: chunk 1 already marked done with the right bytes in place
    (out / "_fetch").mkdir()
    part = out / "_fetch" / "flaky.tar.gz.part"
    part.write_bytes(b"\0" * 65536 + blob[65536:131072])
    (out / "_fetch" / "flaky.tar.gz.chunks.json").write_text(json.dumps({"chunk": 65536, "done": [1]}))
    got = F.fetch_one(a, out, base, conns=3, chunk=65536, sleep=lambda s: None)
    assert got == out / a.name and got.read_bytes() == blob and not part.exists()
    assert json.loads((out / "_fetch" / "status.json").read_text())[a.name]["verified"] == a.sha256
    assert F.fetch_one(a, out, base, conns=3, chunk=65536) == got  # verified earlier: nothing fetched


def test_fetch_sha_mismatch_keeps_part(server, tmp_path):
    root, base = server
    out = tmp_path / "out"
    out.mkdir()
    (root / "x.tar.gz").write_bytes(b"abc" * 1000)
    a = F.Archive("x.tar.gz", 3000, "0" * 64)
    with pytest.raises(SystemExit, match="sha256"):
        F.fetch_one(a, out, base, conns=2, chunk=1000)
    assert (out / "_fetch" / "x.tar.gz.part").is_file() and not (out / "x.tar.gz").exists()
    st = json.loads((out / "_fetch" / "status.json").read_text())
    assert st["x.tar.gz"]["sha256_mismatch"] == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_fetch_refuses_unverified_final(tmp_path):
    (tmp_path / "x.tar.gz").write_bytes(b"1")
    with pytest.raises(SystemExit, match="no verified"):
        F.fetch_one(F.Archive("x.tar.gz", 1, "0" * 64), tmp_path, "http://unused", 1, 1)


def verified(out: Path, a: F.Archive) -> None:
    F.Status(out).set(a.name, verified=a.sha256)


def test_extract_folder_and_root_layouts(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    a = make_tar(out / "scannet.tar.gz", "scannet", {"s0.mp4": b"v0", "sub/s1.mp4": b"v1"})
    b = make_tar(out / "hypersim.tar.gz", "", {"ai_001/f.jpg": b"i0", "ai_002/f.jpg": b"i1"})
    for x in (a, b):
        verified(out, x)
    assert F.extract_one(a, out) == out / "scannet" and (out / "scannet" / "sub" / "s1.mp4").read_bytes() == b"v1"
    assert F.extract_one(b, out) == out / "hypersim" and (out / "hypersim" / "ai_002" / "f.jpg").is_file()
    st = json.loads((out / "_fetch" / "status.json").read_text())
    assert st[a.name]["layout"] == "folder" and st[b.name]["layout"] == "root"
    assert (out / "scannet.tar.gz").is_file()  # archives stay
    assert F.extract_one(a, out) == out / "scannet"  # extracted earlier


def test_extract_refuses(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    a = make_tar(out / "adt.tar.gz", "adt", {"v.mp4": b"v"})
    with pytest.raises(SystemExit, match="not verified"):
        F.extract_one(a, out)
    verified(out, a)
    (out / "adt").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        F.extract_one(a, out)


def test_check_paths_and_decode(tmp_path):
    import llamafactory.data.mm_plugin  # noqa: F401
    import av
    import numpy as np
    from PIL import Image

    out = tmp_path
    (out / "scannet").mkdir()
    (out / "hypersim" / "a").mkdir(parents=True)
    c = av.open(str(out / "scannet" / "v.mp4"), "w")
    s = c.add_stream("mpeg4", rate=10)
    s.width = s.height = 32
    s.pix_fmt = "yuv420p"
    for _ in range(5):
        for p in s.encode(av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    Image.new("RGB", (8, 8)).save(out / "hypersim" / "a" / "f.jpg")
    (out / "scannet" / "bad.mp4").write_bytes(b"not a video")
    rows = [{"video": "scannet/v.mp4"}, {"video": "scannet/v.mp4"}, {"image": "hypersim/a/f.jpg"}]
    (out / "vsi_590k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = F.check(out)
    assert r["ok"] and r["sources"] == {"video/scannet": {"distinct": 1, "missing": 0},
                                        "image/hypersim": {"distinct": 1, "missing": 0}}
    rows += [{"video": "scannet/bad.mp4"}, {"video": "scannet/gone.mp4"}]
    (out / "vsi_590k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = F.check(out)
    assert not r["ok"] and r["missing_examples"] == ["scannet/gone.mp4"]
    assert len(r["decode_failures"]) == 1 and r["decode_failures"][0].startswith("scannet/bad.mp4")
    assert json.loads((out / "_fetch" / "check.json").read_text())["ok"] is False


def test_selected_and_job_script():
    assert [a.source for a in F.selected("s3dis,robotics")] == ["robotics", "s3dis"]
    with pytest.raises(SystemExit):
        F.selected("nope")
    sh = Path(F.__file__).with_name("fetch_vsi_job.sh")
    subprocess.run(["bash", "-n", str(sh)], check=True)
    assert "unset http_proxy" in sh.read_text()
