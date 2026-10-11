"""Build families/canonical_task_metrics_v2.patch (runtime v2 dispatch).

The patch (apply from the repo root with ``patch -p1``):
  * adds ``vagen/envs/active_spatial/canonical_task_metrics_v2.py`` = a
    byte-identical copy of ``families/judges_v2.py``;
  * edits ``vagen/envs/active_spatial/canonical_task_metrics.py`` so that
    ``score_canonical_task`` / ``uses_canonical_backend`` accept
    ``canonical_spatial_task_h1_v2`` rows.  v1 rows take exactly the old code
    path; v2 projective / fov rows take the same v1 functions; v2 family rows
    are routed to ``canonical_task_metrics_v2.score_v2_family``.

Usage:
    python3 -m data_gen.interiorgs_v10.families.make_patch          # rewrite
    python3 -m data_gen.interiorgs_v10.families.make_patch --check  # exit 1 if stale
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

from . import judges_v2 as J

V1_REL = "vagen/envs/active_spatial/canonical_task_metrics.py"
V2_REL = "vagen/envs/active_spatial/canonical_task_metrics_v2.py"
HERE = Path(__file__).resolve().parent
PATCH_PATH = HERE / "canonical_task_metrics_v2.patch"
REPO_ROOT = HERE.parents[2]

_OLD_CONSTANTS = 'SUPPORTED_TASK_TYPES = frozenset(("projective_relations", "fov_inclusion"))\n'
_NEW_CONSTANTS = _OLD_CONSTANTS + (
    "\n"
    "# canonical_spatial_task_h1_v2: projective_relations / fov_inclusion keep the\n"
    "# frozen v1 judges bit-identically; the new families are judged in\n"
    "# canonical_task_metrics_v2.py (generated from data_gen/interiorgs_v10/families).\n"
    'CANONICAL_TASK_METRIC_VERSION_V2 = "canonical_spatial_task_h1_v2"\n'
    "V2_FAMILY_TASK_TYPES = frozenset((\n"
    + "".join(f'    "{t}",\n' for t in sorted(J.NEW_TASK_TYPES))
    + "))\n"
    "SUPPORTED_TASK_TYPES_V2 = SUPPORTED_TASK_TYPES | V2_FAMILY_TASK_TYPES\n"
    "_SUPPORTED_TASK_TYPES_BY_VERSION = {\n"
    "    CANONICAL_TASK_METRIC_VERSION: SUPPORTED_TASK_TYPES,\n"
    "    CANONICAL_TASK_METRIC_VERSION_V2: SUPPORTED_TASK_TYPES_V2,\n"
    "}\n"
    "\n"
    "\n"
    "def _supported_task_types(version: Any) -> frozenset[str] | None:\n"
    "    return _SUPPORTED_TASK_TYPES_BY_VERSION.get(version) if isinstance(version, str) else None\n"
)

_OLD_SCORE = (
    '    if item.get("canonical_task_metric_version") != CANONICAL_TASK_METRIC_VERSION:\n'
    '        raise ValueError("item does not request the R1 canonical metric backend")\n'
    '    task_type = str(item.get("task_type") or "")\n'
)
_NEW_SCORE = (
    '    version = item.get("canonical_task_metric_version")\n'
    "    if _supported_task_types(version) is None:\n"
    '        raise ValueError("item does not request the R1 canonical metric backend")\n'
    '    task_type = str(item.get("task_type") or "")\n'
    "    if version == CANONICAL_TASK_METRIC_VERSION_V2 and task_type in V2_FAMILY_TASK_TYPES:\n"
    "        from .canonical_task_metrics_v2 import score_v2_family\n"
    "\n"
    "        return score_v2_family(item, np.asarray(c2w, dtype=float))\n"
)

_OLD_USES = (
    '    if item and item.get("canonical_task_metric_version"):\n'
    '        if item["canonical_task_metric_version"] != CANONICAL_TASK_METRIC_VERSION:\n'
    '            raise ValueError("unknown canonical_task_metric_version; refusing legacy fallback")\n'
    '        if item.get("task_type") not in SUPPORTED_TASK_TYPES:\n'
    '            raise ValueError("canonical metric version does not support this task type")\n'
    "    return bool(\n"
    "        item\n"
    '        and item.get("canonical_task_metric_version") == CANONICAL_TASK_METRIC_VERSION\n'
    '        and item.get("task_type") in SUPPORTED_TASK_TYPES\n'
    "    )\n"
)
_NEW_USES = (
    '    if item and item.get("canonical_task_metric_version"):\n'
    '        supported = _supported_task_types(item["canonical_task_metric_version"])\n'
    "        if supported is None:\n"
    '            raise ValueError("unknown canonical_task_metric_version; refusing legacy fallback")\n'
    '        if item.get("task_type") not in supported:\n'
    '            raise ValueError("canonical metric version does not support this task type")\n'
    "    if not item:\n"
    "        return False\n"
    '    supported = _supported_task_types(item.get("canonical_task_metric_version"))\n'
    '    return bool(supported is not None and item.get("task_type") in supported)\n'
)

REPLACEMENTS = ((_OLD_CONSTANTS, _NEW_CONSTANTS), (_OLD_SCORE, _NEW_SCORE), (_OLD_USES, _NEW_USES))


def patched_v1_source(original: str) -> str:
    out = original
    for old, new in REPLACEMENTS:
        if out.count(old) != 1:
            raise RuntimeError(f"{V1_REL} changed upstream; cannot locate block:\n{old}")
        out = out.replace(old, new)
    return out


def v2_module_source() -> str:
    return (HERE / "judges_v2.py").read_text(encoding="utf-8")


def build_patch(repo_root: Path = REPO_ROOT) -> str:
    original = (Path(repo_root) / V1_REL).read_text(encoding="utf-8")
    patched = patched_v1_source(original)
    parts = list(difflib.unified_diff(
        original.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile=f"a/{V1_REL}", tofile=f"b/{V1_REL}", n=3))
    parts += list(difflib.unified_diff(
        [], v2_module_source().splitlines(keepends=True),
        fromfile="/dev/null", tofile=f"b/{V2_REL}", n=3))
    text = "".join(parts)
    if not text.endswith("\n"):  # pragma: no cover
        text += "\n"
    return text


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    text = build_patch()
    if "--check" in argv:
        current = PATCH_PATH.read_text(encoding="utf-8") if PATCH_PATH.exists() else ""
        if current != text:
            print(f"{PATCH_PATH} is stale; rerun make_patch", file=sys.stderr)
            return 1
        print("patch up to date")
        return 0
    PATCH_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {PATCH_PATH} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
