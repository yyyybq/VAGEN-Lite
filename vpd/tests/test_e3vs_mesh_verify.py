"""Tests of vpd.e3vs.mesh_verify on synthetic PLY files (stdlib; login node, 1 core).

    nice -n 19 PY -B -m pytest -q -p no:cacheprovider vpd/tests/test_e3vs_mesh_verify.py
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest

import vpd.e3vs as E
from vpd import paths
from vpd.e3vs import mesh_verify as MV


def write_ply(path, nv=4, nf=2, fmt="binary_little_endian", truncate=0, quad=False, color=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    props = "property float x\nproperty float y\nproperty float z\n"
    if color:
        props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    hdr = (f"ply\nformat {fmt} 1.0\ncomment synthetic\nelement vertex {nv}\n{props}"
           f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n").encode()
    if fmt == "ascii":
        body = "".join(f"{i} 0 0" + (" 1 2 3" if color else "") + "\n" for i in range(nv))
        body += "".join("3 0 1 2\n" for _ in range(nf))
        data = hdr + body.encode()
    else:
        e = "<" if fmt == "binary_little_endian" else ">"
        body = b"".join(struct.pack(e + "fff", float(i), 0.0, 0.0) + (bytes([1, 2, 3]) if color else b"")
                        for i in range(nv))
        for _ in range(nf):
            body += (struct.pack(e + "Biiii", 4, 0, 1, 2, 3) if quad else struct.pack(e + "Biii", 3, 0, 1, 2))
        data = hdr + body
    if truncate:
        data = data[:-truncate]
    path.write_bytes(data)
    return path


def make_tree(tmp_path, ids, **kw):
    root = tmp_path / "mesh_dl"
    for s in ids:
        write_ply(root / s / MV.MESH_REL, **kw)
    idf = tmp_path / "ids.txt"
    idf.write_text("".join(f"{s}\n" for s in ids))
    return root, idf


IDS = ["0a184cf634", "511061232", "ffeeddccbb"]


def test_ok_and_manifest(tmp_path):
    root, idf = make_tree(tmp_path, IDS)
    out = tmp_path / "out"
    out.mkdir()
    res = MV.verify(root, idf, out, expected=IDS)
    assert res["ok"], res["problems"]
    assert res["n_ok"] == 3 and res["n_vertices_total"] == 12 and res["n_faces_total"] == 6
    s0 = res["scenes"][IDS[0]]
    assert s0["size"] == s0["expected_size_triangles"] and s0["first_face_n"] == 3
    assert len(res["mesh_set_sha256"]) == 64
    man = json.loads((out / MV.MANIFEST).read_text())
    assert man["mesh_set_sha256"] == res["mesh_set_sha256"]
    assert set(man["code"]["e3vs_files"]) == set(E.CODE_FILES)  # recording only
    lines = (out / MV.SHA_LIST).read_text().splitlines()
    assert len(lines) == 3 and all(len(x.split("  ")[0]) == 64 for x in lines)
    assert MV.summary_line(res).startswith("mesh verify OK (3 scenes")


def test_set_key_depends_on_content(tmp_path):
    root, idf = make_tree(tmp_path, IDS)
    a = MV.verify(root, idf, None, expected=IDS)["mesh_set_sha256"]
    write_ply(root / IDS[1] / MV.MESH_REL, nv=5)
    b = MV.verify(root, idf, None, expected=IDS)["mesh_set_sha256"]
    assert a != b


@pytest.mark.parametrize("kw,needle", [
    ({"truncate": 5}, "truncated"),
    ({"quad": True}, "non-triangle"),
    ({"nv": 0}, "vertex count 0"),
])
def test_bad_mesh(tmp_path, kw, needle):
    root, idf = make_tree(tmp_path, IDS)
    write_ply(root / IDS[2] / MV.MESH_REL, **kw)
    res = MV.verify(root, idf, None, expected=IDS)
    assert not res["ok"] and res["mesh_set_sha256"] is None
    assert any(IDS[2] in p and needle in p for p in res["problems"]), res["problems"]
    assert MV.summary_line(res).startswith("mesh verify FAILED")


def test_missing_file_and_empty(tmp_path):
    root, idf = make_tree(tmp_path, IDS[:2])
    res = MV.verify(root, idf.with_name("ids.txt"), None, expected=IDS)
    assert not res["ok"]
    assert any("missing" in p and IDS[2] in p for p in res["problems"])
    (root / IDS[0] / MV.MESH_REL).write_bytes(b"")
    res = MV.verify(root, idf, None, expected=IDS[:2])
    assert any("empty file" in p for p in res["problems"])


def test_id_list_must_match(tmp_path):
    root, idf = make_tree(tmp_path, IDS)
    idf.write_text(f"{IDS[0]}\n{IDS[1]}\n{IDS[1]}\n")
    res = MV.verify(root, idf, None, expected=IDS)
    assert not res["ok"]
    assert any("duplicate" in p for p in res["problems"])
    assert any("missing ['ffeeddccbb']" in p for p in res["problems"])


def test_ascii_and_big_endian(tmp_path):
    root, idf = make_tree(tmp_path, IDS, fmt="ascii")
    res = MV.verify(root, idf, None, expected=IDS)
    assert res["ok"] and "expected_size_triangles" not in res["scenes"][IDS[0]]
    root2, idf2 = make_tree(tmp_path / "be", IDS, fmt="binary_big_endian", color=False)
    assert MV.verify(root2, idf2, None, expected=IDS)["ok"]


def test_header_errors(tmp_path):
    p = tmp_path / "x.ply"
    p.write_bytes(b"plx\nformat ascii 1.0\nend_header\n")
    with pytest.raises(MV.PlyError, match="magic"):
        MV.parse_header(p)
    p.write_bytes(b"ply\nformat ascii 2.0\nend_header\n")
    with pytest.raises(MV.PlyError, match="format"):
        MV.parse_header(p)
    p.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 3\n")
    with pytest.raises(MV.PlyError, match="end_header"):
        MV.parse_header(p)
    write_ply(p, color=False)
    t = p.read_bytes().replace(b"property float z\n", b"property uchar z\n")
    p.write_bytes(t)
    with pytest.raises(MV.PlyError, match="not float"):
        MV.check_mesh(p, with_sha=False)


def test_notes_for_extras(tmp_path):
    root, idf = make_tree(tmp_path, IDS)
    (root / IDS[0] / "scans" / "mesh_aligned_0.05_semantic.ply").write_bytes(b"x")
    (root / "zzz").mkdir()
    res = MV.verify(root, idf, None, expected=IDS)
    assert res["ok"]
    assert any("semantic" in n for n in res["notes"]) and any("zzz" in n for n in res["notes"])


def test_code_files_are_the_tracked_e3vs_code():
    """CODE_FILES (recorded as code.e3vs_files) is exactly the tracked vpd/e3vs/*.py and vpd/jobs/e3vs_*.sh."""
    shas = E.code_files_sha256()
    assert list(shas) == list(E.CODE_FILES) and all(v and len(v) == 64 for v in shas.values())
    assert shas["vpd/e3vs/mesh_verify.py"] == E.sha256_file(paths.WORKTREE / "vpd" / "e3vs" / "mesh_verify.py")
    m = E.code_meta(Path(MV.__file__), e3vs_files=True)
    assert m["e3vs_files"] == shas and "e3vs_files" not in E.code_meta(Path(MV.__file__))
    assert E.code_files_sha256(Path("/nonexistent_vpd_tree"))["vpd/e3vs/run.py"] is None
    try:
        r = subprocess.run(["git", "-C", str(paths.WORKTREE), "ls-files", "--", "vpd/e3vs/*.py", "vpd/jobs/e3vs_*.sh"],
                           capture_output=True, text=True, timeout=60)
    except OSError:
        r = None
    if r is None or r.returncode != 0:
        pytest.skip("no git here (a job's code snapshot)")
    assert sorted(r.stdout.split()) == sorted(E.CODE_FILES)


def test_expected_ids_are_the_99_episode_scenes():
    ids = MV.expected_scene_ids()
    assert len(ids) == 99 and "511061232" in ids
    if MV.MESH_IDS.is_file():  # the list the user downloads from must be exactly these
        assert sorted(MV.read_ids(MV.MESH_IDS)) == ids
