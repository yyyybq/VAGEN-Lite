"""q38s4 stage 5 (PREREG_v1 section 4): SITE's score from its two EASI jobs, by EASI's own code.  qa_eval runs SITE as
two jobs (site_image, site_video), each in its own folder; EASI's LmmsEvalAdapter._extract_site_scores combines the
two parts' per-sample caa_num / caa_den (chance-adjusted accuracy) when both samples files are in one folder.  This
puts links to each part's newest samples file into such a folder and calls that method (EASI v0.2.2 at EASI_ROOT,
loaded from its files), for the combined CAA and its sub-scores, and again on each part alone.

    python -m vagen.q38s4.tools.site_caa --qa QA --name NAME --out OUT.json [--easi EASI_ROOT]

QA/site_image/NAME and QA/site_video/NAME must each have done.json (qa_eval's mark of a finished job) and exactly one
newest samples file (lmms-eval's <timestamp>_samples_<task>.jsonl under models__*/; the newest by name, as EASI picks
them).  The links go to OUT's sibling folder OUT.stem + "_join" (combined) and "_image" / "_video" (each part); it
must not exist.  Used for both arms and for the base (q38/runs/q38-base-zs/qa, q38base).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EASI_ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/EASI")
PKG = "_q38s4_easi_backends"
PARTS = {"site_image": "*_samples_site_bench_image.jsonl", "site_video": "*_samples_site_bench_video*.jsonl"}


def load_easi(root: Path = EASI_ROOT):
    """EASI's lmmseval backend module, loaded from root's files under a private package name (no sys.path change, so
    no clash with this worktree's own scripts/ folder)."""
    name = f"{PKG}.lmmseval"
    if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent == (
            root / "scripts/submissions/backends").resolve():
        return sys.modules[name]
    for k in [k for k in sys.modules if k == PKG or k.startswith(PKG + ".")]:
        del sys.modules[k]
    pkg_dir = root / "scripts/submissions/backends"
    spec = importlib.util.spec_from_file_location(PKG, pkg_dir / "__init__.py",
                                                  submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(name)


def newest(folder: Path, pattern: str) -> Path:
    found = sorted(folder.glob(f"models__*/{pattern}"), key=lambda p: p.name, reverse=True)
    if not found:
        raise SystemExit(f"{folder}: no samples file {pattern}")
    if len(found) > 1 and found[0].name == found[1].name:
        raise SystemExit(f"{folder}: two newest samples files named {found[0].name}: {found[:2]}")
    return found[0]


def part_files(qa: Path, name: str) -> Dict[str, Path]:
    out = {}
    for part, pattern in PARTS.items():
        folder = qa / part / name
        if not (folder / "done.json").is_file():
            raise SystemExit(f"{folder}: no done.json (the job has not finished)")
        out[part] = newest(folder, pattern)
    return out


def scores(mod, folder: Path, files: List[Path]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    folder.mkdir(parents=True)
    for f in files:
        os.symlink(f, folder / f.name)
    return mod.LmmsEvalAdapter()._extract_site_scores(folder, mod._METRIC_MAP["site"])


def site(qa: Path, name: str, out: Path, root: Path = EASI_ROOT) -> Dict[str, Any]:
    files = part_files(qa, name)
    base = out.parent / out.stem
    dirs = {k: Path(f"{base}_{k}") for k in ("join", "image", "video")}
    taken = [str(d) for d in dirs.values() if d.exists() or d.is_symlink()]
    if taken:
        raise SystemExit(f"exists: {taken}")
    mod = load_easi(root)
    overall, subs = scores(mod, dirs["join"], [files["site_image"], files["site_video"]])
    img, _ = scores(mod, dirs["image"], [files["site_image"]])
    vid, _ = scores(mod, dirs["video"], [files["site_video"]])
    res = {"qa": str(qa), "name": name, "files": {k: str(v) for k, v in files.items()},
           "easi": str(root), "site_caa": overall, "site_image_caa": img, "site_video_caa": vid, "sub_scores": subs}
    out.write_text(json.dumps(res, indent=1) + "\n")
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--easi", type=Path, default=EASI_ROOT)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    res = site(a.qa, a.name, a.out, a.easi)
    print(json.dumps({k: res[k] for k in ("name", "site_caa", "site_image_caa", "site_video_caa")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
