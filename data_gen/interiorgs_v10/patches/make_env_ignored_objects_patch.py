"""Generate patches/env_ignored_objects.patch (apply after env_structure_override.patch).

The RL env must drop the same labels.json instances as the generator
(configs/ignored_objects_*.json, confirmed annotation errors), otherwise a
room whose floor is covered by a mislabelled box is a collision everywhere.
Each v10 task row lists them in ``collision_convention.ignored_object_ids``.

    * CollisionDetector(ignored_object_ids={scene: [ins_id, ...]}): no collision
      box for those instances.  Their corners still feed the room-boundary
      estimate and the label centres still feed the structure-sign check, so
      every other collision result is unchanged.
    * VisibilityChecker: no occluder box for those instances.
    * env.reset: registers the row's ids for its scene before loading it.

    python data_gen/interiorgs_v10/patches/make_env_ignored_objects_patch.py [--check]
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PATCH_PATH = Path(__file__).with_name("env_ignored_objects.patch")
BASE_PATCH = Path(__file__).with_name("env_structure_override.patch")
REL = "vagen/envs/active_spatial"

EDITS = {
    "collision_detector.py": [
        ("        structure_y_sign_overrides: Optional[Dict[str, float]] = None,\n    ):",
         "        structure_y_sign_overrides: Optional[Dict[str, float]] = None,\n"
         "        ignored_object_ids: Optional[Dict[str, List[str]]] = None,\n    ):"),
        ("        self.structure_convention_override_source: Optional[str] = None\n",
         "        self.structure_convention_override_source: Optional[str] = None\n"
         "        # Per scene: labels.json ins_id values that are annotation errors (InteriorGS v10\n"
         "        # configs/ignored_objects_*.json); no collision box is built for them.\n"
         "        self.ignored_object_ids: Dict[str, set] = {\n"
         "            str(key): {str(v) for v in values} for key, values in (ignored_object_ids or {}).items()\n"
         "        }\n"),
        ("                self.object_boxes = []\n                self._label_xy_centers = []\n"
         "                all_points = []\n                \n                for obj in labels_data:\n",
         "                self.object_boxes = []\n                self._label_xy_centers = []\n"
         "                all_points = []\n"
         "                ignored_here = self.ignored_object_ids.get(\n"
         "                    str(scene_id) if scene_id is not None else scene_path.name, set())\n"
         "                \n                for obj in labels_data:\n"),
        ("                    expanded = aabb.expand(self.camera_radius + self.safety_margin)\n"
         "                    self.object_boxes.append(expanded)\n                    \n"
         "                    # Collect all points for room boundary estimation\n",
         "                    expanded = aabb.expand(self.camera_radius + self.safety_margin)\n"
         "                    if str(ins_id) not in ignored_here:\n"
         "                        self.object_boxes.append(expanded)\n                    \n"
         "                    # Collect all points for room boundary estimation\n"),
        ("                self.object_boxes = []\n                self._label_xy_centers = []\n"
         "                all_points = []\n                for obj in labels_data:\n",
         "                self.object_boxes = []\n                self._label_xy_centers = []\n"
         "                all_points = []\n"
         "                ignored_here = self.ignored_object_ids.get(str(scene_id), set())\n"
         "                for obj in labels_data:\n"),
        ("                    expanded = aabb.expand(self.camera_radius + self.safety_margin)\n"
         "                    self.object_boxes.append(expanded)\n                    for corner in bbox:\n",
         "                    expanded = aabb.expand(self.camera_radius + self.safety_margin)\n"
         "                    if str(ins_id) not in ignored_here:\n"
         "                        self.object_boxes.append(expanded)\n                    for corner in bbox:\n"),
    ],
    "visibility_checker.py": [
        ("        self.occluder_boxes: List[Dict] = []\n",
         "        self.occluder_boxes: List[Dict] = []\n"
         "        # Per scene: labels.json ins_id values that are annotation errors (not occluders).\n"
         "        self.ignored_object_ids: Dict[str, set] = {}\n"),
        ("                self.occluder_boxes = []\n                for obj in labels_data:\n",
         "                self.occluder_boxes = []\n"
         "                ignored_here = self.ignored_object_ids.get(str(scene_id), set()) if scene_id is not None else set()\n"
         "                for obj in labels_data:\n"),
        ("                    if len(bbox) != 8:\n                        continue\n                    \n"
         "                    # Skip non-occluding objects\n",
         "                    if len(bbox) != 8:\n                        continue\n"
         "                    if str(obj.get('ins_id', '')) in ignored_here:\n                        continue\n"
         "                    \n                    # Skip non-occluding objects\n"),
    ],
    "env.py": [
        ("            self.collision_detector.load_scene_from_gs_root(self.config.gs_root, scene_id)\n",
         "            # InteriorGS v10 rows list labels.json instances that are annotation errors (e.g. one\n"
         "            # cabinet box covering a whole room); the generator ignored them, so must the env.\n"
         "            row_convention = item.get(\"collision_convention\") if isinstance(item, dict) else None\n"
         "            ignored = row_convention.get(\"ignored_object_ids\") if isinstance(row_convention, dict) else None\n"
         "            if ignored:\n"
         "                ids = {str(v) for v in ignored}\n"
         "                previous_ids = self.collision_detector.ignored_object_ids.get(str(scene_id))\n"
         "                if previous_ids is not None and previous_ids != ids:\n"
         "                    raise RuntimeError(f\"conflicting ignored_object_ids for {scene_id}\")\n"
         "                if previous_ids is None:\n"
         "                    self.collision_detector.ignored_object_ids[str(scene_id)] = ids\n"
         "                    if getattr(self.collision_detector, \"_current_scene_id\", None) == scene_id:\n"
         "                        self.collision_detector._current_scene_id = None  # force reload without them\n"
         "                    if self.visibility_checker is not None:\n"
         "                        self.visibility_checker.ignored_object_ids[str(scene_id)] = ids\n"
         "                        if getattr(self.visibility_checker, \"_current_scene_id\", None) == scene_id:\n"
         "                            self.visibility_checker._current_scene_id = None\n"
         "            self.collision_detector.load_scene_from_gs_root(self.config.gs_root, scene_id)\n"),
    ],
}


def _base_sources() -> dict:
    """The three env files with env_structure_override.patch applied (the base of this patch)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in EDITS:
            dst = tmp / REL / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((REPO / REL / name).read_bytes())
        res = subprocess.run(["patch", "-p1", "--batch", "--forward", "--quiet"], cwd=tmp,
                             input=BASE_PATCH.read_bytes(), capture_output=True)
        if res.returncode != 0:
            raise RuntimeError("env_structure_override.patch does not apply: " + res.stdout.decode())
        return {name: (tmp / REL / name).read_text() for name in EDITS}


def build() -> str:
    out = []
    for name, src in _base_sources().items():
        new = src
        for old, rep in EDITS[name]:
            if new.count(old) != 1:
                raise RuntimeError(f"{name}: anchor found {new.count(old)} times: {old[:60]!r}")
            new = new.replace(old, rep)
        out.extend(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                        fromfile=f"a/{REL}/{name}", tofile=f"b/{REL}/{name}"))
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if the committed patch is out of date")
    args = ap.parse_args(argv)
    text = build()
    if args.check:
        return 0 if PATCH_PATH.read_text() == text else 1
    PATCH_PATH.write_text(text)
    print(f"wrote {PATCH_PATH} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
