"""Verify the ScanNet++ collision meshes of the 99 E3VS scenes (user 2026-11-13, item 1 and supplement 2).

The user downloaded the SEMANTIC version of the official ScanNet++ mesh, one file per scene:
    OUT/e3vs/mesh_dl/<scene>/scans/mesh_aligned_0.05_semantic.ply     (the user's download: SOURCE, never written)
Same geometry as mesh_aligned_0.05.ply (the official configs' collision_mesh_path), plus per-vertex label
columns; the collision model uses the geometry only (x/y/z and the face index lists, read by the official
loader o3d.io.read_triangle_mesh; the extra columns are ignored).  The pipeline reads copies (or symlinks) of them:
    OUT/e3vs/mesh/<scene>/scans/mesh_aligned_0.05_semantic.ply       (MESH_ROOT, the collision mesh root)
and every copy's sha256 must equal its source file's (--source, default OUT/e3vs/mesh_dl; a missing or different
source file is a problem: refused).
Scene list (user 2026-11-13): the directory names under OUT/e3vs/data/scannetpp_v2_mcmc_3dgs/ (paths.E3VS_DATA)
without the file SHA256SUMS and without hidden entries (.cache, ...).  OUT/e3vs/mesh_scene_ids.txt must equal it,
and both must equal the scene ids of the pinned episode files (68 train + 10 val + 21 test = 99).

Checks per scene: the file is there and non-empty; the PLY header parses (ply, format 1.0, vertex element with
x/y/z, face element with a vertex index list); vertex and face counts > 0; for binary files the size equals
header + vertices + triangle faces exactly (catches a truncated download) and the first face has 3 indices;
size and sha256; the source file exists and has the same sha256.  Writes OUT/e3vs/mesh/MANIFEST.json (per-scene
facts: scene, path, sha256, n_vertices, n_faces, size, format, props, mtime, source path and source sha256;
problems; mesh_set_sha256 = sha256 of the sorted "<scene> <sha256>" lines, the key of the mesh set; code: the
sha256 of the E3VS code, recording only) and OUT/e3vs/mesh/mesh_sha256.txt (sha256sum format) beside it.  The
mesh root may hold such non-scene files (MANIFEST.json, mesh_sha256.txt, validation.json, frame_ack.json).
The last line printed is "mesh verify OK (99 scenes ...)" or "mesh verify FAILED ..."; the exit code is 0 only
when OK.

Read-only on the meshes and on the source.  Never touches credentials and never downloads anything.  Stdlib only
(plus the episode reader of vpd.e3vs).  Login node: 1 core, nice -n 19; reading 2 x 99 meshes takes a few minutes.

    nice -n 19 PY -B -m vpd.e3vs.mesh_verify [--root DIR] [--source DIR] [--ids FILE] [--data DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from vpd import paths

MESH_DL = paths.E3VS_OUT / "mesh_dl"  # the user's download (the source of the copies; read only)
MESH_ROOT = paths.E3VS_OUT / "mesh"  # the collision meshes: copies (or symlinks) of the source files
MESH_IDS = paths.E3VS_OUT / "mesh_scene_ids.txt"
MESH_REL = Path("scans") / "mesh_aligned_0.05_semantic.ply"  # the semantic version (user 2026-11-13); geometry only
OFFICIAL_MESH_REL = Path("scans") / "mesh_aligned_0.05.ply"  # the official configs' name: same geometry, no labels
MANIFEST = "MANIFEST.json"  # in MESH_ROOT
SHA_LIST = "mesh_sha256.txt"  # beside MANIFEST
DATA_SKIP = ("SHA256SUMS",)  # the one non-scene entry of the data dir (hidden entries are skipped too)
ROOT_FILES = (MANIFEST, SHA_LIST, "validation.json", "frame_ack.json")  # non-scene files of MESH_ROOT

PLY_SIZES = {
    "char": 1, "int8": 1, "uchar": 1, "uint8": 1, "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
    "int": 4, "int32": 4, "uint": 4, "uint32": 4, "float": 4, "float32": 4, "double": 8, "float64": 8,
}
FORMATS = ("ascii", "binary_little_endian", "binary_big_endian")
INDEX_NAMES = ("vertex_indices", "vertex_index")


class PlyError(ValueError):
    pass


def expected_scene_ids() -> list[str]:
    """Scene ids of the pinned episode files (md5-checked), sorted; the per-split counts must match."""
    import vpd.e3vs as E

    ids = set()
    for sp in E.SPLITS:
        s = {ep["scene_id"] for ep in E.load_episodes(sp)}
        if len(s) != E.SCENE_COUNTS[sp]:
            raise AssertionError(f"{sp}: {len(s)} scenes != {E.SCENE_COUNTS[sp]}")
        ids |= s
    return sorted(ids)


def read_ids(path: Path) -> list[str]:
    lines = [x.strip() for x in Path(path).read_text().splitlines()]
    return [x for x in lines if x]


def data_scene_ids(data_dir: Path | None = None) -> list[str]:
    """The user's scene list: the entries of the splat data dir (paths.E3VS_DATA) without SHA256SUMS and without
    hidden entries, sorted (any other entry counts, so a stray file shows up as an extra id)."""
    d = Path(data_dir) if data_dir is not None else paths.E3VS_DATA
    return sorted(e.name for e in d.iterdir() if not e.name.startswith(".") and e.name not in DATA_SKIP)


def parse_header(path: Path, max_bytes: int = 1 << 16) -> dict:
    """PLY header: format, elements [(name, count, [props])], header length in bytes."""
    with open(path, "rb") as f:
        head = f.read(max_bytes)
    if not head.startswith(b"ply\n") and not head.startswith(b"ply\r\n"):
        raise PlyError("no 'ply' magic line")
    i = head.find(b"end_header")
    if i < 0:
        raise PlyError(f"no end_header in the first {max_bytes} bytes")
    j = head.find(b"\n", i)
    if j < 0:
        raise PlyError("end_header line not terminated")
    hlen = j + 1
    try:
        lines = head[:hlen].decode("ascii").splitlines()
    except UnicodeDecodeError as e:
        raise PlyError(f"header is not ascii: {e}") from None
    fmt, elements = None, []
    for ln in lines[1:]:
        t = ln.split()
        if not t or t[0] in ("comment", "obj_info"):
            continue
        if t[0] == "format":
            if len(t) != 3 or t[1] not in FORMATS or t[2] != "1.0":
                raise PlyError(f"bad format line {ln!r}")
            fmt = t[1]
        elif t[0] == "element":
            if len(t) != 3 or not t[2].isdigit():
                raise PlyError(f"bad element line {ln!r}")
            elements.append({"name": t[1], "count": int(t[2]), "props": []})
        elif t[0] == "property":
            if not elements:
                raise PlyError("property before any element")
            if len(t) == 5 and t[1] == "list":
                if t[2] not in PLY_SIZES or t[3] not in PLY_SIZES:
                    raise PlyError(f"bad list property {ln!r}")
                elements[-1]["props"].append({"name": t[4], "list": (t[2], t[3])})
            elif len(t) == 3 and t[1] in PLY_SIZES:
                elements[-1]["props"].append({"name": t[2], "type": t[1]})
            else:
                raise PlyError(f"bad property line {ln!r}")
        elif t[0] == "end_header":
            break
        else:
            raise PlyError(f"unknown header line {ln!r}")
    if fmt is None:
        raise PlyError("no format line")
    return {"format": fmt, "elements": elements, "header_bytes": hlen}


def check_mesh(path: Path, with_sha: bool = True) -> dict:
    """Facts of one mesh file; raises PlyError on any problem."""
    st = path.stat()
    if st.st_size == 0:
        raise PlyError("empty file")
    h = parse_header(path)
    el = {e["name"]: e for e in h["elements"]}
    if "vertex" not in el or "face" not in el:
        raise PlyError(f"elements {[e['name'] for e in h['elements']]} lack vertex or face")
    v, fc = el["vertex"], el["face"]
    vnames = {p["name"]: p for p in v["props"]}
    for c in ("x", "y", "z"):
        if c not in vnames or vnames[c].get("type") not in ("float", "float32", "double", "float64"):
            raise PlyError(f"vertex property {c} missing or not float")
    idx = [p for p in fc["props"] if p["name"] in INDEX_NAMES and "list" in p]
    if len(idx) != 1:
        raise PlyError("face element has no vertex index list")
    if v["count"] <= 0 or fc["count"] <= 0:
        raise PlyError(f"vertex count {v['count']}, face count {fc['count']}")
    out = {
        "size": st.st_size, "format": h["format"], "header_bytes": h["header_bytes"],
        "n_vertices": v["count"], "n_faces": fc["count"],
        "vertex_props": [p["name"] for p in v["props"]],
        "vertex_prop_types": {p["name"]: p.get("type") or "list " + " ".join(p["list"]) for p in v["props"]},
        "face_props": [p["name"] for p in fc["props"]],
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
    }
    if h["format"] != "ascii":
        expect, ok = h["header_bytes"], True
        for e in h["elements"]:
            stride = 0
            for p in e["props"]:
                if "list" in p:
                    if e["name"] != "face" or p is not idx[0]:
                        ok = False  # another list property: the size cannot be computed
                        break
                    ct, it = p["list"]
                    stride += PLY_SIZES[ct] + 3 * PLY_SIZES[it]
                else:
                    stride += PLY_SIZES[p["type"]]
            if not ok:
                break
            expect += e["count"] * stride
        if ok:
            out["expected_size_triangles"] = expect
            if expect != st.st_size:
                raise PlyError(f"size {st.st_size} != {expect} expected for {v['count']} vertices and "
                               f"{fc['count']} triangles (truncated download, or non-triangle faces)")
            # the first face record starts after the vertex block when face is the element right after vertex
            names = [e["name"] for e in h["elements"]]
            if names[:2] == ["vertex", "face"] and fc["props"] and fc["props"][0] is idx[0]:
                vstride = sum(PLY_SIZES[p["type"]] for p in v["props"])
                ct = idx[0]["list"][0]
                with open(path, "rb") as f:
                    f.seek(h["header_bytes"] + v["count"] * vstride)
                    b = f.read(PLY_SIZES[ct])
                n0 = int.from_bytes(b, "little" if h["format"] == "binary_little_endian" else "big")
                if n0 != 3:
                    raise PlyError(f"first face has {n0} indices, not 3")
                out["first_face_n"] = n0
        else:
            out["expected_size_triangles"] = None
    if with_sha:
        out["sha256"] = _sha256(path)
    return out


def mesh_set_sha256(shas: dict[str, str]) -> str:
    """Key of a mesh set: sha256 of the sorted "<scene> <sha256>\\n" lines."""
    s = "".join(f"{k} {shas[k]}\n" for k in sorted(shas))
    return hashlib.sha256(s.encode()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    hs = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            hs.update(b)
    return hs.hexdigest()


def _check_source(sc: str, info: dict, src: Path, with_sha: bool) -> str | None:
    """Record the source file of one copy (path, sha256) in info; the problem when it is missing or differs."""
    info["source"] = str(src)
    if not src.is_file():
        return f"{sc}: source {src} missing (the copy cannot be checked against the download)"
    if with_sha:
        info["source_sha256"] = _sha256(src)
        if info["source_sha256"] != info["sha256"]:
            return (f"{sc}: sha256 {info['sha256'][:12]} of {info['path']} != {info['source_sha256'][:12]} of the "
                    f"source {src}")
    elif src.stat().st_size != info["size"]:
        return f"{sc}: size {info['size']} of {info['path']} != {src.stat().st_size} of the source {src}"
    return None


def verify(root: Path = MESH_ROOT, ids_file: Path = MESH_IDS, out_dir: Path | None = MESH_ROOT,
           expected: list[str] | None = None, with_sha: bool = True, *, source: Path,
           data_dir: Path | None = None) -> dict:
    """Verify the meshes under root against the source download and the scene lists (module docstring); write
    MANIFEST.json and mesh_sha256.txt to out_dir (None: write nothing).  `source` has no default here (the CLI's
    is MESH_DL), so that no caller reads the user's download by accident; data_dir None = paths.E3VS_DATA."""
    root, ids_file, source = Path(root), Path(ids_file), Path(source)
    data_dir = Path(data_dir) if data_dir is not None else paths.E3VS_DATA
    problems, notes, scenes = [], [], {}
    if expected is None:
        expected = expected_scene_ids()
    ids = read_ids(ids_file) if ids_file.is_file() else []
    if not ids_file.is_file():
        problems.append(f"scene list {ids_file} missing")
    if len(ids) != len(set(ids)):
        problems.append("scene list has duplicate ids")
    if set(ids) != set(expected):
        miss, extra = sorted(set(expected) - set(ids)), sorted(set(ids) - set(expected))
        problems.append(f"scene list != the episode scenes (missing {miss[:5]}, extra {extra[:5]})")
    listing = data_scene_ids(data_dir) if data_dir.is_dir() else None
    if listing is None:
        problems.append(f"data dir {data_dir} missing (the scene list is its directory listing)")
    elif set(ids) != set(listing):
        miss, extra = sorted(set(listing) - set(ids)), sorted(set(ids) - set(listing))
        problems.append(f"scene list {ids_file.name} != the directory listing of {data_dir} (missing {miss[:5]}, "
                        f"extra {extra[:5]})")
    for sc in sorted(set(expected)):
        p = root / sc / MESH_REL
        if not p.is_file():
            problems.append(f"{sc}: {p} missing")
            continue
        try:
            info = check_mesh(p, with_sha=with_sha)
        except (PlyError, OSError) as e:
            problems.append(f"{sc}: {e}")
            continue
        info["scene"] = sc
        info["path"] = str(p)
        if p.is_symlink():
            info["resolved"] = str(p.resolve())
        try:
            prob = _check_source(sc, info, source / sc / MESH_REL, with_sha)
        except OSError as e:
            prob = f"{sc}: source {source / sc / MESH_REL}: {e}"
        if prob:
            problems.append(prob)
            continue
        scenes[sc] = info
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.name.startswith(".") or (d.name in ROOT_FILES and d.is_file()):
                continue  # the manifest, the validation and their temporary files
            if d.name not in expected:
                notes.append(f"extra entry {d.name} in {root}")
            elif (d / "scans").is_dir():
                for f in sorted((d / "scans").iterdir()):
                    if f.name != MESH_REL.name:
                        notes.append(f"{d.name}: extra file scans/{f.name}")
    ok = not problems and len(scenes) == len(set(expected))
    shas = {k: v["sha256"] for k, v in scenes.items() if "sha256" in v}
    res = {
        "ok": ok,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root), "ids_file": str(ids_file), "source_root": str(source), "data_dir": str(data_dir),
        "mesh_rel": str(MESH_REL),
        "n_expected": len(set(expected)), "n_ok": len(scenes),
        "total_bytes": sum(v["size"] for v in scenes.values()),
        "n_vertices_total": sum(v["n_vertices"] for v in scenes.values()),
        "n_faces_total": sum(v["n_faces"] for v in scenes.values()),
        "mesh_set_sha256": mesh_set_sha256(shas) if ok and with_sha else None,
        "problems": problems, "notes": notes, "scenes": scenes,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        try:
            import vpd.e3vs as E
            res["code"] = E.code_meta(Path(__file__), e3vs_files=True)
        except Exception as e:  # noqa: BLE001 - provenance only
            res["code"] = {"error": str(e)}
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(out_dir / MANIFEST, json.dumps(res, indent=1, sort_keys=True) + "\n")
        _write_atomic(out_dir / SHA_LIST, "".join(f"{shas[k]}  {scenes[k]['path']}\n" for k in sorted(shas)))
    return res


def summary_line(res: dict) -> str:
    if res["ok"]:
        gb = res["total_bytes"] / 1e9
        key = (res["mesh_set_sha256"] or "no-sha")[:12]
        return (f"mesh verify OK ({res['n_ok']} scenes, {gb:.2f} GB, {res['n_vertices_total']} vertices, "
                f"{res['n_faces_total']} faces, set sha {key})")
    return f"mesh verify FAILED ({len(res['problems'])} problems, {res['n_ok']}/{res['n_expected']} ok)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=MESH_ROOT, help="the collision meshes (default OUT/e3vs/mesh)")
    ap.add_argument("--source", type=Path, default=MESH_DL,
                    help="the user's download the copies must equal (default OUT/e3vs/mesh_dl; read only)")
    ap.add_argument("--ids", type=Path, default=MESH_IDS)
    ap.add_argument("--data", type=Path, default=None,
                    help="the dir whose listing is the scene list (default OUT/e3vs/data/scannetpp_v2_mcmc_3dgs)")
    ap.add_argument("--out-dir", type=Path, default=MESH_ROOT, help="MANIFEST.json + mesh_sha256.txt (default "
                                                                     "OUT/e3vs/mesh)")
    a = ap.parse_args(argv)
    res = verify(a.root, a.ids, a.out_dir, source=a.source, data_dir=a.data)
    for p in res["problems"][:30]:
        print("PROBLEM", p)
    if len(res["problems"]) > 30:
        print(f"... {len(res['problems']) - 30} more problems in {a.out_dir / MANIFEST}")
    for n in res["notes"][:10]:
        print("note", n)
    print(summary_line(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
