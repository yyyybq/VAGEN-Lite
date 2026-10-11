"""Tests for workstream G: calibration v2, scene loading, fast legality, scene identity."""

import contextlib
import io
import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import coordinates as CO
from data_gen.interiorgs_v10.contract import pose_from_xy_yaw
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.scene_identity import fingerprint, group_same_source, similarity
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


def _detector_v1(labels, structure, tmp: Path):
    from vagen.envs.active_spatial.collision_detector import CollisionDetector

    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "labels.json").write_text(json.dumps(labels))
    (tmp / "structure.json").write_text(json.dumps(structure))
    det = CollisionDetector()
    with contextlib.redirect_stdout(io.StringIO()):
        det.load_scene(tmp, scene_id=tmp.name)
    return det.convention_record()


def _transform(labels, k, mirror, t, rng=None, jitter=0.0, drop=0, relabel=0):
    out = []
    for it in labels:
        bb = []
        for c in it["bounding_box"]:
            x, y = c["x"], (-c["y"] if mirror else c["y"])
            for _ in range(k):
                x, y = -y, x
            bb.append({"x": x + t[0], "y": y + t[1], "z": c["z"]})
        out.append(dict(it, bounding_box=bb))
    if rng is not None:
        for it in out:
            dx, dy = rng.normal(0.0, jitter, 2)
            for c in it["bounding_box"]:
                c["x"] += dx
                c["y"] += dy
        for _ in range(drop):
            out.pop(int(rng.integers(len(out))))
        for _ in range(relabel):
            out[int(rng.integers(len(out)))]["label"] = "unknown thing"
    return out


class GeometryTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = write_synthetic_scenes(cls.root)
        cls.truth = {k: json.loads((p / "_truth.json").read_text()) for k, p in cls.paths.items()}
        cls.labels = {k: json.loads((p / "labels.json").read_text()) for k, p in cls.paths.items()}
        cls.structure = {k: json.loads((p / "structure.json").read_text()) for k, p in cls.paths.items()}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class CalibrationTest(GeometryTestBase):
    def test_synthetic_scenes(self):
        expect = {"two_rooms": "v1_frozen", "symmetric_ambiguous": "v2_evidence", "dense_duplicates": "v2_evidence"}
        for name, rule in expect.items():
            rec = CO.calibrate_structure_y_sign(self.labels[name], self.structure[name])
            self.assertEqual(rec["version"], "interiorgs_structure_label_alignment_v2")
            self.assertEqual(rec["status"], "frozen", (name, rec["reason"]))
            self.assertEqual(rec["sign"], self.truth[name]["structure_y_sign"], name)
            self.assertEqual(rec["rule"], rule, name)
            for key in ("label_centres", "corner_containment", "wall_alignment", "openings", "edge_crossing"):
                self.assertIn("+1", rec["evidence"][key]["score"])
                self.assertIn("-1", rec["evidence"][key]["score"])
        self.assertEqual(CO.calibrate_structure_y_sign(self.labels["symmetric_ambiguous"],
                                                       self.structure["symmetric_ambiguous"])["v1"]["status"],
                         "ambiguous")

    def test_v1_replica_matches_collision_detector(self):
        rng = np.random.default_rng(7)
        cases = [(self.labels[k], self.structure[k]) for k in self.paths]
        for i in range(45):
            kind = ("generic", "hull_symmetric", "notch", "sparse", "wall_heavy")[i % 5]
            lab, st, _ = CO.synthetic_layout(rng, kind)
            cases.append((lab, st))
        for j, (lab, st) in enumerate(cases):
            ref = _detector_v1(lab, st, self.root / f"v1check_{j}")
            mine = CO.v1_record(lab, st)
            self.assertEqual(mine["status"], ref["status"], j)
            self.assertEqual(mine["sign"], ref["structure_y_sign"], j)
            self.assertEqual(mine["alignment_scores"], ref["alignment_scores"], j)

    def test_benchmark_and_v1_agreement(self):
        bench = CO.run_benchmark(n=240, seed=3, unrelated=20)
        # v2's own rule never freezes the wrong sign; it resolves >= 97 % of identifiable layouts
        self.assertEqual(bench["frozen_wrong"]["v2_own"], 0, bench["confusion"])
        self.assertGreaterEqual(bench["accuracy_identifiable"]["v2_own"], 0.97, bench["confusion"])
        # exactly mirror-symmetric structures are never frozen
        self.assertEqual(bench["confusion"]["v2_own"]["no"].get("frozen", 0), 0)
        # the final decision never departs from a v1-frozen scene; all final errors are inherited from v1
        self.assertEqual(bench["frozen_wrong"]["final"], bench["frozen_wrong"]["v1"])
        for row in bench["rows"]:
            if row["rule"] == "v1_frozen":
                self.assertEqual(row["status"], "frozen")
        # v2-own on unrelated structure: almost never frozen
        self.assertLessEqual(bench["unrelated_structure"]["v2_own"].get("frozen", 0), 2)

    def test_never_disagrees_with_v1_frozen(self):
        rng = np.random.default_rng(11)
        n_frozen = 0
        for i in range(120):
            lab, st, _ = CO.synthetic_layout(rng, ("generic", "wall_heavy", "notch")[i % 3])
            rec = CO.calibrate_structure_y_sign(lab, st)
            if rec["v1"]["status"] == "frozen":
                n_frozen += 1
                self.assertEqual(rec["status"], "frozen")
                self.assertEqual(rec["sign"], rec["v1"]["sign"])
                self.assertFalse(rec["env_override_required"])
                self.assertEqual(rec["usable"], not rec["conflict_with_v1"])
        self.assertGreater(n_frozen, 20)

    def test_small_scene_with_clear_evidence(self):
        # 4 objects + 4 walls, clearly inside the -1 interpretation only
        lab = [CO._box(str(i), "cabinet", 1 + i, 1, 0, 1.6 + i, 1.5, 0.9) for i in range(4)]
        lab += [CO._box("10", "wall", -0.1, -0.1, 0, 6.1, 0.0, 2.8), CO._box("11", "wall", -0.1, 4.0, 0, 6.1, 4.1, 2.8),
                CO._box("12", "wall", -0.1, 0.0, 0, 0.0, 4.0, 2.8), CO._box("13", "wall", 6.0, 0.0, 0, 6.1, 4.0, 2.8)]
        st = {"rooms": [{"name": "study", "profile": [[0, 0], [6, 0], [6, -4], [0, -4]]}]}
        rec = CO.calibrate_structure_y_sign(lab, st)
        self.assertEqual(rec["v1"]["status"], "ambiguous")
        self.assertEqual((rec["status"], rec["sign"]), ("frozen", -1.0), rec["reason"])

    def test_no_structure_or_no_labels(self):
        self.assertEqual(CO.calibrate_structure_y_sign([], {})["status"], "ambiguous")
        self.assertEqual(CO.calibrate_structure_y_sign(self.labels["two_rooms"], {})["status"], "ambiguous")

    def test_env_overrides(self):
        recs = {k: CO.calibrate_structure_y_sign(self.labels[k], self.structure[k]) for k in self.paths}
        self.assertEqual(CO.env_overrides(recs), {"dense_duplicates": -1.0, "symmetric_ambiguous": 1.0})
        self.assertEqual(CO.env_overrides(recs, include_v1_frozen=True)["two_rooms"], -1.0)
        as_list = [dict(r, scene_id=k) for k, r in recs.items()]
        self.assertEqual(CO.env_overrides(as_list), CO.env_overrides(recs))

    def test_cli_report(self):
        out = self.root / "report.json"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            CO.main(["--scenes-root", str(self.root), "--out", str(out)])
        rep = json.loads(out.read_text())["calibration"]
        self.assertEqual(rep["n_scenes"], 3)
        self.assertEqual(rep["v1_status_counts"], {"frozen": 1, "ambiguous": 2})
        self.assertEqual(rep["v2_status_counts"], {"frozen": 3})
        self.assertEqual(len(rep["disagreements"]), 2)
        self.assertEqual(set(rep["structure_y_sign_overrides"]), {"synth_symmetric_ambiguous", "synth_dense_duplicates"})
        self.assertIn("v1_to_v2", buf.getvalue())


class SceneTest(GeometryTestBase):
    def test_load_scene_passes_override_to_detector(self):
        from vagen.envs.active_spatial.collision_detector import CollisionDetector

        for name, path in self.paths.items():
            sc = load_scene(path)
            self.assertEqual(sc.convention["status"], "frozen", name)
            self.assertEqual(sc.convention["structure_y_sign"], self.truth[name]["structure_y_sign"], name)
            self.assertEqual(sc.convention_v2["applied_sign"], sc.convention["structure_y_sign"])
            # an env detector built with the generator's overrides agrees with the scene's detector
            env_det = CollisionDetector(structure_y_sign_overrides=CO.env_overrides([sc.convention_v2]))
            with contextlib.redirect_stdout(io.StringIO()):
                env_det.load_scene(path, scene_id=sc.scene_id)
            self.assertEqual(env_det.convention_record()["status"], "frozen")
            self.assertEqual(env_det.structure_y_sign, sc.detector.structure_y_sign)
            self.assertEqual(len(env_det.wall_segments), len(sc.detector.wall_segments))
        sc = load_scene(self.paths["two_rooms"])
        self.assertEqual(sc.convention_v2["applied_source"], "detector_v1")
        self.assertIsNone(sc.convention["override_source"])
        sc = load_scene(self.paths["two_rooms"], y_sign_override=1.0)
        self.assertEqual(sc.convention["structure_y_sign"], 1.0)

    @staticmethod
    def _samples(scene, n, rng):
        pts = np.concatenate([r.polygon for r in scene.rooms])
        lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
        a = rng.uniform(lo, hi, size=(n, 2))
        k = n // 5
        a[:k] = np.round(a[:k] / 0.05) * 0.05   # grid points: exact ties with axis-aligned walls/boxes
        return a

    def test_legal_mask_and_transitions_match_reference(self):
        rng = np.random.default_rng(0)
        n = 20000
        for name, path in self.paths.items():
            sc = load_scene(path)
            xy = self._samples(sc, n, rng)
            t0 = time.perf_counter()
            fast = sc.legal_mask(xy)
            t_fast = time.perf_counter() - t0
            slow = np.array([sc.is_pose_legal(pose_from_xy_yaw(float(x), float(y), 0.0))[0] for x, y in xy])
            self.assertEqual(int((fast != slow).sum()), 0, name)
            self.assertGreater(int(slow.sum()), 1000)
            self.assertGreater(int((~slow).sum()), 1000)
            # 0.3 m moves in random and 20-degree-lattice directions, plus zero / long moves
            a = self._samples(sc, n, rng)
            ang = rng.uniform(0.0, 2 * math.pi, n)
            ang[: n // 3] = np.radians(rng.integers(0, 18, n // 3) * 20.0)
            L = np.full(n, 0.3)
            L[-n // 20:] = rng.choice([0.0, 0.8, 2.0], n // 20)
            b = a + np.stack([np.cos(ang), np.sin(ang)], 1) * L[:, None]
            fast_t = sc.transitions_legal(a, b)
            slow_t = np.array([not sc.detector.check_collision(np.array([q[0], q[1], 1.5]),
                                                               previous_position=np.array([p[0], p[1], 1.5])).has_collision
                               for p, q in zip(a, b)])
            self.assertEqual(int((fast_t != slow_t).sum()), 0, name)
            self.assertGreater(int((~slow_t).sum()), 500)
            self.assertLess(t_fast, 2.0)

    def test_rooms_of_and_api(self):
        sc = load_scene(self.paths["two_rooms"])
        rng = np.random.default_rng(1)
        xy = rng.uniform([-1, -6], [11, 1], size=(3000, 2))
        ref = np.array([-1 if sc.room_of(x, y) is None else sc.room_of(x, y) for x, y in xy])
        np.testing.assert_array_equal(sc.rooms_of(xy), ref)
        self.assertEqual(sc.legal_mask(np.zeros((0, 2))).shape, (0,))
        single = sc.transitions_legal(np.array([2.0, -2.0]), np.array([[2.3, -2.0], [2.0, -2.3]]))
        self.assertEqual(single.shape, (2,))
        ok, _ = sc.transition_legal(pose_from_xy_yaw(2.0, -2.0, 0.0), pose_from_xy_yaw(2.0, -2.0, 1.0))
        self.assertTrue(ok)
        # changing the clearance invalidates the cached kernel
        before = sc.legal_mask(xy).sum()
        sc.min_wall_distance = 0.2
        self.assertGreaterEqual(sc.legal_mask(xy).sum(), before)


class IdentityTest(GeometryTestBase):
    def test_invariance_and_discrimination(self):
        rng = np.random.default_rng(0)
        base = fingerprint(self.labels["two_rooms"])
        for k in range(4):
            for mirror in (False, True):
                copy = _transform(self.labels["two_rooms"], k, mirror, (13.7, -4.2), rng, jitter=0.02, drop=1, relabel=1)
                self.assertGreater(similarity(base, fingerprint(copy)), 0.9, (k, mirror))
        self.assertGreater(similarity(base, fingerprint(_transform(self.labels["two_rooms"], 1, True, (3, 3)))), 0.999)
        self.assertLess(similarity(base, fingerprint(self.labels["dense_duplicates"])), 0.3)
        self.assertLess(similarity(base, fingerprint(self.labels["symmetric_ambiguous"])), 0.3)
        sc = load_scene(self.paths["two_rooms"])
        self.assertGreater(similarity(fingerprint(sc), base), 0.999)
        json.dumps(base)

    def test_group_same_source(self):
        rng = np.random.default_rng(4)
        fps = {}
        for i in range(12):
            lab, _, _ = CO.synthetic_layout(rng, "generic")
            fps[f"s{i:02d}"] = fingerprint(lab)
            if i in (2, 7):
                dup = _transform(lab, int(rng.integers(4)), True, rng.uniform(-20, 20, 2), rng, 0.02, 2, 2)
                fps[f"s{i:02d}_copy"] = fingerprint(dup)
        groups = group_same_source(fps, threshold=0.6)
        self.assertEqual(groups, [["s02", "s02_copy"], ["s07", "s07_copy"]])
        ids = sorted(k for k in fps if not k.endswith("copy"))
        worst = max(similarity(fps[a], fps[b]) for i, a in enumerate(ids) for b in ids[i + 1:])
        self.assertLess(worst, 0.4)


if __name__ == "__main__":
    unittest.main()
