"""Tests of vpd.e3vs.mesh_verify on synthetic PLY files (stdlib; login node, 1 core).

The synthetic meshes have the vertex / face properties of the real files, the semantic version the user downloaded
(2026-11-13): binary little endian; vertex x, y, z float, red, green, blue uchar, label int; face
`list uchar int vertex_indices`.  Each tree is a tmp mesh root (the copies), a tmp source dir with the same bytes
(the user's download) and a tmp data dir whose listing is the scene list (scene dirs, SHA256SUMS, a hidden .cache).
No real mesh, download or data dir is read; only the pinned episode files (test_expected_ids_...).

    nice -n 19 PY -B -m pytest -q -p no:cacheprovider vpd/tests/test_e3vs_mesh_verify.py
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

import vpd.e3vs as E
from vpd import paths
from vpd.e3vs import mesh_verify as MV

# The layout of the real files (orchestrator's check of all 99 on 2026-11-13: one vertex and one face property set,
# binary_little_endian, no comment lines).
SEMANTIC_VERTEX = (("x", "float"), ("y", "float"), ("z", "float"), ("red", "uchar"), ("green", "uchar"),
                   ("blue", "uchar"), ("label", "int"))
SEMANTIC_FACE = ("vertex_indices", ("uchar", "int"))


def semantic_header(nv: int, nf: int) -> bytes:
    return ("ply\nformat binary_little_endian 1.0\n" f"element vertex {nv}\n"
            + "".join(f"property {t} {n}\n" for n, t in SEMANTIC_VERTEX)
            + f"element face {nf}\nproperty list {SEMANTIC_FACE[1][0]} {SEMANTIC_FACE[1][1]} {SEMANTIC_FACE[0]}\n"
            + "end_header\n").encode()


def write_semantic_ply(path, verts, faces, colors=None, labels=None) -> Path:
    """A PLY in the exact layout of the real semantic files: per vertex x/y/z float32, rgb uint8, label int32;
    triangles as (3, i, j, k) with a uint8 count and int32 indices."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(verts)
    if colors is None:
        colors = [((7 * i) % 256, (13 * i + 5) % 256, (29 * i + 11) % 256) for i in range(n)]
    if labels is None:
        labels = [(i % 7) - 1 for i in range(n)]  # -1 (unlabelled) included
    body = b"".join(struct.pack("<fffBBBi", *map(float, v), *c, int(lab)) for v, c, lab in zip(verts, colors, labels))
    body += b"".join(struct.pack("<Biii", 3, *map(int, f)) for f in faces)
    path.write_bytes(semantic_header(n, len(faces)) + body)
    return path


def write_ply(path, nv=4, nf=2, fmt="binary_little_endian", truncate=0, quad=False, color=True, label=True):
    """A small PLY; by default with the semantic files' extra vertex columns (rgb uchar, label int)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    props = "property float x\nproperty float y\nproperty float z\n"
    if color:
        props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    if label:
        props += "property int label\n"
    hdr = (f"ply\nformat {fmt} 1.0\ncomment synthetic\nelement vertex {nv}\n{props}"
           f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n").encode()
    if fmt == "ascii":
        body = "".join(f"{i} 0 0" + (" 1 2 3" if color else "") + (f" {i - 1}" if label else "") + "\n"
                       for i in range(nv))
        body += "".join("3 0 1 2\n" for _ in range(nf))
        data = hdr + body.encode()
    else:
        e = "<" if fmt == "binary_little_endian" else ">"
        body = b"".join(struct.pack(e + "fff", float(i), 0.0, 0.0) + (bytes([1, 2, 3]) if color else b"")
                        + (struct.pack(e + "i", i - 1) if label else b"") for i in range(nv))
        for _ in range(nf):
            body += (struct.pack(e + "Biiii", 4, 0, 1, 2, 3) if quad else struct.pack(e + "Biii", 3, 0, 1, 2))
        data = hdr + body
    if truncate:
        data = data[:-truncate]
    path.write_bytes(data)
    return path


def data_listing(data: Path, ids) -> Path:
    """A data dir like OUT/e3vs/data/scannetpp_v2_mcmc_3dgs: one dir per scene, the file SHA256SUMS, a hidden dir."""
    for s in ids:
        (data / s).mkdir(parents=True, exist_ok=True)
    (data / "SHA256SUMS").write_text("")
    (data / ".cache").mkdir(exist_ok=True)
    return data


def put(base: Path, scene: str, writer=write_ply, *a, **kw) -> Path:
    """(Re)write one scene's mesh in base/mesh and the same bytes in base/mesh_dl (the source)."""
    p = writer(base / "mesh" / scene / MV.MESH_REL, *a, **kw)
    s = base / "mesh_dl" / scene / MV.MESH_REL
    s.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, s)
    return p


def make_tree(base, ids, **kw):
    """base/mesh (root), base/mesh_dl (source), base/data (listing), base/ids.txt -> (root, ids file, verify kw)."""
    base = Path(base)
    for s in ids:
        put(base, s, **kw)
    idf = base / "ids.txt"
    idf.write_text("".join(f"{s}\n" for s in ids))
    return base / "mesh", idf, {"source": base / "mesh_dl", "data_dir": data_listing(base / "data", ids)}


IDS = ["0a184cf634", "511061232", "ffeeddccbb"]


def test_ok_and_manifest(tmp_path):
    root, idf, kw = make_tree(tmp_path, IDS)
    res = MV.verify(root, idf, root, expected=IDS, **kw)  # the manifest goes into the mesh root, as in production
    assert res["ok"], res["problems"]
    assert res["n_ok"] == 3 and res["n_vertices_total"] == 12 and res["n_faces_total"] == 6
    s0 = res["scenes"][IDS[0]]
    assert s0["size"] == s0["expected_size_triangles"] and s0["first_face_n"] == 3
    # the MANIFEST fields of the user's decision (scene, path, sha256, counts, size, the source and its sha256)
    assert s0["scene"] == IDS[0] and s0["path"] == str(root / IDS[0] / MV.MESH_REL) and len(s0["sha256"]) == 64
    assert s0["source"] == str(kw["source"] / IDS[0] / MV.MESH_REL) and s0["source_sha256"] == s0["sha256"]
    assert s0["n_vertices"] == 4 and s0["n_faces"] == 2 and s0["format"] == "binary_little_endian"
    assert s0["vertex_props"][-1] == "label" and s0["vertex_prop_types"]["label"] == "int" and s0["mtime_utc"]
    assert len(res["mesh_set_sha256"]) == 64
    man = json.loads((root / MV.MANIFEST).read_text())
    assert man["mesh_set_sha256"] == res["mesh_set_sha256"] and man["source_root"] == str(kw["source"])
    assert man["mesh_rel"] == "scans/mesh_aligned_0.05_semantic.ply" and man["data_dir"] == str(kw["data_dir"])
    assert set(man["code"]["e3vs_files"]) == set(E.CODE_FILES)  # recording only
    assert list(man["code"]["files"].values()) == [E.sha256_file(Path(MV.__file__))]
    lines = (root / MV.SHA_LIST).read_text().splitlines()
    assert len(lines) == 3 and all(len(x.split("  ")[0]) == 64 for x in lines)
    assert MV.summary_line(res).startswith("mesh verify OK (3 scenes")
    # the root holds non-scene files (manifest, sha list, validation, frame ack): no notes, and the key is the same
    (root / "validation.json").write_text("{}")
    (root / "frame_ack.json").write_text("{}")
    again = MV.verify(root, idf, root, expected=IDS, **kw)
    assert again["ok"] and again["notes"] == [] and again["mesh_set_sha256"] == res["mesh_set_sha256"]


def test_set_key_depends_on_content(tmp_path):
    root, idf, kw = make_tree(tmp_path, IDS)
    a = MV.verify(root, idf, None, expected=IDS, **kw)["mesh_set_sha256"]
    put(tmp_path, IDS[1], nv=5)
    b = MV.verify(root, idf, None, expected=IDS, **kw)["mesh_set_sha256"]
    assert a and b and a != b


@pytest.mark.parametrize("kw,needle", [
    ({"truncate": 5}, "truncated"),
    ({"quad": True}, "non-triangle"),
    ({"nv": 0}, "vertex count 0"),
])
def test_bad_mesh(tmp_path, kw, needle):
    root, idf, side = make_tree(tmp_path, IDS)
    put(tmp_path, IDS[2], **kw)
    res = MV.verify(root, idf, None, expected=IDS, **side)
    assert not res["ok"] and res["mesh_set_sha256"] is None
    assert any(IDS[2] in p and needle in p for p in res["problems"]), res["problems"]
    assert MV.summary_line(res).startswith("mesh verify FAILED")


def test_missing_file_and_empty(tmp_path):
    root, idf, kw = make_tree(tmp_path, IDS[:2])
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert not res["ok"]
    assert any("missing" in p and IDS[2] in p for p in res["problems"])
    (root / IDS[0] / MV.MESH_REL).write_bytes(b"")
    res = MV.verify(root, idf, None, expected=IDS[:2], **kw)
    assert any("empty file" in p for p in res["problems"])


def test_id_list_must_match(tmp_path):
    root, idf, kw = make_tree(tmp_path, IDS)
    idf.write_text(f"{IDS[0]}\n{IDS[1]}\n{IDS[1]}\n")
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert not res["ok"]
    assert any("duplicate" in p for p in res["problems"])
    assert any("episode scenes (missing ['ffeeddccbb']" in p for p in res["problems"])
    assert any("directory listing" in p and "missing ['ffeeddccbb']" in p for p in res["problems"])


def test_scene_list_is_the_data_dir_listing(tmp_path):
    """User 2026-11-13: the scene list is the directory listing of the splat data dir without SHA256SUMS and hidden
    entries; mesh_scene_ids.txt must equal it (and the episode scenes)."""
    root, idf, kw = make_tree(tmp_path, IDS)
    data = kw["data_dir"]
    assert MV.data_scene_ids(data) == sorted(IDS)  # SHA256SUMS and .cache are not scenes
    (data / ".partial").write_text("")
    assert MV.verify(root, idf, None, expected=IDS, **kw)["ok"]
    (data / "abcdef0123").mkdir()  # a scene dir that is not in the list
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert not res["ok"] and any("directory listing" in p and "missing ['abcdef0123']" in p for p in res["problems"])
    (data / "abcdef0123").rmdir()
    (data / "notes.txt").write_text("")  # any other entry counts
    assert not MV.verify(root, idf, None, expected=IDS, **kw)["ok"]
    (data / "notes.txt").unlink()
    shutil.rmtree(data / IDS[0])  # a listed scene without its data dir
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert any("directory listing" in p and f"extra ['{IDS[0]}']" in p for p in res["problems"])
    res = MV.verify(root, idf, None, expected=IDS, source=kw["source"], data_dir=tmp_path / "no_data")
    assert not res["ok"] and any("data dir" in p and "missing" in p for p in res["problems"])
    # the default is paths.E3VS_DATA, read at call time
    assert MV.verify.__kwdefaults__["data_dir"] is None


def test_copies_must_equal_the_source(tmp_path):
    """Every copy's sha256 must equal its source file's (the user's download); a missing or different source file is
    a problem; a symlink to the source is fine (its target is recorded)."""
    root, idf, kw = make_tree(tmp_path, IDS)
    src = kw["source"]
    with pytest.raises(TypeError):
        MV.verify(root, idf, None, expected=IDS)  # no default source: never the real download by accident
    with open(src / IDS[1] / MV.MESH_REL, "ab") as f:
        f.write(b"\0")
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert not res["ok"] and res["n_ok"] == 2 and res["mesh_set_sha256"] is None
    assert any(IDS[1] in p and "of the source" in p for p in res["problems"]), res["problems"]
    (src / IDS[1] / MV.MESH_REL).unlink()
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert not res["ok"] and any(IDS[1] in p and "source" in p and "missing" in p for p in res["problems"])
    # copy back, then make one copy a symlink to its source
    shutil.copyfile(root / IDS[1] / MV.MESH_REL, src / IDS[1] / MV.MESH_REL)
    (root / IDS[2] / MV.MESH_REL).unlink()
    (root / IDS[2] / MV.MESH_REL).symlink_to(src / IDS[2] / MV.MESH_REL)
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert res["ok"], res["problems"]
    assert res["scenes"][IDS[2]]["resolved"] == str((src / IDS[2] / MV.MESH_REL).resolve())
    # without sha256 the sizes are compared
    with open(src / IDS[0] / MV.MESH_REL, "ab") as f:
        f.write(b"\0")
    res = MV.verify(root, idf, None, expected=IDS, with_sha=False, **kw)
    assert not res["ok"] and any("size" in p and "of the source" in p for p in res["problems"])


def test_ascii_and_big_endian(tmp_path):
    root, idf, kw = make_tree(tmp_path, IDS, fmt="ascii")
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert res["ok"] and "expected_size_triangles" not in res["scenes"][IDS[0]]
    root2, idf2, kw2 = make_tree(tmp_path / "be", IDS, fmt="binary_big_endian", color=False, label=False)
    assert MV.verify(root2, idf2, None, expected=IDS, **kw2)["ok"]


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
    root, idf, kw = make_tree(tmp_path, IDS)
    (root / IDS[0] / "scans" / MV.OFFICIAL_MESH_REL.name).write_bytes(b"x")
    (root / "zzz").mkdir()
    res = MV.verify(root, idf, None, expected=IDS, **kw)
    assert res["ok"]
    assert any("mesh_aligned_0.05.ply" in n for n in res["notes"]) and any("zzz" in n for n in res["notes"])


def test_semantic_layout_of_the_real_files(tmp_path):
    """The fixture writer has the real files' layout: for the counts of scene 00a231a370 (meshchk 2026-11-13: 1019235
    vertices, 2074643 faces, header 260 bytes, size 46336084) header and size come out exactly; mesh_verify accepts
    such a file and records the label column."""
    nv, nf = 1019235, 2074643
    assert len(semantic_header(nv, nf)) == 260
    assert 260 + nv * (3 * 4 + 3 + 4) + nf * (1 + 3 * 4) == 46336084
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    p = write_semantic_ply(tmp_path / "s" / MV.MESH_REL, verts, faces)
    info = MV.check_mesh(p)
    assert info["vertex_props"] == [n for n, _ in SEMANTIC_VERTEX] and info["face_props"] == ["vertex_indices"]
    assert info["vertex_prop_types"] == dict(SEMANTIC_VERTEX) and info["format"] == "binary_little_endian"
    assert info["size"] == info["expected_size_triangles"] == len(semantic_header(4, 4)) + 4 * 19 + 4 * 13
    assert info["n_vertices"] == 4 and info["n_faces"] == 4 and info["first_face_n"] == 3
    root, idf, kw = make_tree(tmp_path / "t", IDS[:1])
    put(tmp_path / "t", IDS[0], write_semantic_ply, verts, faces)
    assert MV.verify(root, idf, None, expected=IDS[:1], **kw)["ok"]


def test_paths_of_the_semantic_copies(monkeypatch, capsys):
    """User 2026-11-13: semantic files, copies in OUT/e3vs/mesh, MANIFEST.json there, the download is the source."""
    assert MV.MESH_REL == Path("scans") / "mesh_aligned_0.05_semantic.ply"
    assert MV.OFFICIAL_MESH_REL == Path("scans") / "mesh_aligned_0.05.ply"  # the official configs' name
    assert MV.MESH_ROOT == paths.E3VS_OUT / "mesh" and MV.MESH_DL == paths.E3VS_OUT / "mesh_dl"
    assert MV.MANIFEST == "MANIFEST.json" and MV.SHA_LIST == "mesh_sha256.txt"
    assert MV.verify.__defaults__[:3] == (MV.MESH_ROOT, MV.MESH_IDS, MV.MESH_ROOT)
    # the CLI defaults (verify replaced: nothing is read)
    got = []
    monkeypatch.setattr(MV, "verify", lambda *a, **k: got.append((a, k)) or {
        "ok": False, "problems": [], "notes": [], "n_ok": 0, "n_expected": 99})
    assert MV.main([]) == 1 and "FAILED" in capsys.readouterr().out
    assert got == [((MV.MESH_ROOT, MV.MESH_IDS, MV.MESH_ROOT), {"source": MV.MESH_DL, "data_dir": None})]


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
    if paths.E3VS_DATA.is_dir():  # ... and the user's definition: the data dir's listing
        assert MV.data_scene_ids() == ids
