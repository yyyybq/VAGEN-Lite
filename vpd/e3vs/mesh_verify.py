"""Verify the user-downloaded ScanNet++ collision meshes of the 99 E3VS scenes (user 2026-11-13, item 1).

Layout (official ScanNet++ structure, one file per scene, not the semantic version):
    OUT/e3vs/mesh_dl/<scene>/scans/mesh_aligned_0.05.ply
Scene list: OUT/e3vs/mesh_scene_ids.txt, which must equal the scene ids of the pinned episode files
(68 train + 10 val + 21 test = 99).

Checks per scene: the file is there and non-empty; the PLY header parses (ply, format 1.0, vertex element with
x/y/z, face element with a vertex index list); vertex and face counts > 0; for binary files the size equals
header + vertices + triangle faces exactly (catches a truncated download) and the first face has 3 indices;
size and sha256.  Writes OUT/e3vs/mesh_manifest.json (per-scene facts, problems, and mesh_set_sha256 =
sha256 of the sorted "<scene> <sha256>" lines, the key of the mesh set) and OUT/e3vs/mesh_sha256.txt
(sha256sum format).  The last line printed is "mesh verify OK (99 scenes ...)" or "mesh verify FAILED ...";
the exit code is 0 only when OK.

Read-only on the meshes.  Never touches credentials and never downloads anything.  Stdlib only (plus the
episode reader of vpd.e3vs).  Login node: 1 core, nice -n 19; reading 99 meshes takes a few minutes.

    nice -n 19 PY -B -m vpd.e3vs.mesh_verify [--root DIR] [--ids FILE] [--out-dir DIR]
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

MESH_DL = paths.E3VS_OUT / "mesh_dl"
MESH_IDS = paths.E3VS_OUT / "mesh_scene_ids.txt"
MESH_REL = Path("scans") / "mesh_aligned_0.05.ply"
MANIFEST = "mesh_manifest.json"
SHA_LIST = "mesh_sha256.txt"

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
        hs = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                hs.update(b)
        out["sha256"] = hs.hexdigest()
    return out


def mesh_set_sha256(shas: dict[str, str]) -> str:
    """Key of a mesh set: sha256 of the sorted "<scene> <sha256>\\n" lines."""
    s = "".join(f"{k} {shas[k]}\n" for k in sorted(shas))
    return hashlib.sha256(s.encode()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def verify(root: Path = MESH_DL, ids_file: Path = MESH_IDS, out_dir: Path | None = paths.E3VS_OUT,
           expected: list[str] | None = None, with_sha: bool = True) -> dict:
    root, ids_file = Path(root), Path(ids_file)
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
        info["path"] = str(p)
        if p.is_symlink():
            info["resolved"] = str(p.resolve())
        scenes[sc] = info
    if root.is_dir():
        for d in sorted(root.iterdir()):
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
        "root": str(root), "ids_file": str(ids_file),
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
            res["code"] = E.code_meta(Path(__file__))
        except Exception as e:  # noqa: BLE001 - provenance only
            res["code"] = {"error": str(e)}
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
    ap.add_argument("--root", type=Path, default=MESH_DL)
    ap.add_argument("--ids", type=Path, default=MESH_IDS)
    ap.add_argument("--out-dir", type=Path, default=paths.E3VS_OUT)
    a = ap.parse_args(argv)
    res = verify(a.root, a.ids, a.out_dir)
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
