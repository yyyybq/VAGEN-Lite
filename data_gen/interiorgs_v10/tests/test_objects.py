"""Tests for objects.py / referring.py (stdlib unittest, synthetic scenes only)."""

import contextlib
import io
import json
import random
import re
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_gen.interiorgs_v10 import objects as O
from data_gen.interiorgs_v10 import referring as R
from data_gen.interiorgs_v10.contract import Obj
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

VIEWPOINT_RE = re.compile(r"\b(left|right|front|behind|back|near|far|closer|nearest|farthest|facing)\b")
PRESET_NAMES = ("v9_strict", "v10_relaxed")


def _box(i, label, x0, y0, z0, x1, y1, z1, room=None):
    return Obj(id=str(i), label=label, aabb_min=np.array([x0, y0, z0], float),
               aabb_max=np.array([x1, y1, z1], float), room_index=room)


def _room(i, name, x0, y0, x1, y1):
    return SimpleNamespace(index=i, name=name,
                           polygon=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))


def mini_scene():
    """Three rooms; exercises the room, size (volume and height) and anchor
    methods, the hypernym rule and the exclusions."""
    rooms = [_room(0, "LivingRoom", 0, 0, 5, 5), _room(1, "MasterBedroom", 5, 0, 10, 5),
             _room(2, "room_3", 0, 5, 5, 10)]
    objs = [
        _box("a1", "armchair", 1.0, 1.0, 0, 1.8, 1.8, 0.9, room=0),
        _box("a2", "armchair", 6.0, 1.0, 0, 6.8, 1.8, 0.9, room=1),
        _box("c1", "cabinet", 0.5, 5.5, 0, 1.7, 6.0, 0.9, room=2),
        _box("c2", "cabinet", 3.0, 9.0, 0, 4.0, 9.6, 1.8, room=2),
        _box("b1", "bookshelf", 0.3, 8.0, 0, 1.3, 8.4, 1.0, room=2),
        _box("b2", "bookshelf", 3.8, 6.0, 0, 4.6, 6.3, 2.0, room=2),
        _box("s1", "stool", 3.0, 3.0, 0, 3.4, 3.4, 0.45, room=0),
        _box("s2", "stool", 3.6, 3.0, 0, 4.0, 3.4, 0.45, room=0),
        _box("s3", "stool", 3.0, 3.6, 0, 3.4, 4.0, 0.45, room=0),
        _box("dt", "dining table", 3.0, 4.3, 0, 4.2, 4.9, 0.75, room=0),
        _box("t", "table", 0.5, 3.5, 0, 1.5, 4.5, 0.75, room=0),
        _box("m", "mattress", 7.0, 2.0, 0, 9.0, 3.5, 0.25, room=1),
        _box("d", "door", 4.95, 2.0, 0, 5.05, 3.0, 2.1, room=None),
        _box("w", "window", 9.95, 1.0, 0.9, 10.05, 2.5, 2.2, room=1),
        _box("fl", "floor lamp", 9.3, 0.3, 0, 9.7, 0.7, 1.7, room=1),
        _box("wl", "wall lamp", 9.8, 4.0, 1.8, 9.95, 4.3, 2.1, room=1),
        _box("r", "rug", 1.0, 1.0, 0, 3.0, 3.0, 0.02, room=0),
        _box("pj", "projector", 2.0, 2.0, 2.5, 2.4, 2.4, 2.7, room=0),
        _box("v", "vase", 8.0, 0.3, 0, 8.2, 0.5, 0.3, room=1),
    ]
    return SimpleNamespace(scene_id="mini", scene_dir=None, objects=objs, rooms=rooms, convention={})


class _SyntheticBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10.scene import load_scene

        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = write_synthetic_scenes(cls.root)
        cls.scenes = {name: load_scene(p, name) for name, p in cls.paths.items()}
        cls.results = {}
        for name in ("dense_duplicates", "two_rooms"):
            for preset in PRESET_NAMES:
                sc = cls.scenes[name]
                objs, f_obj = O.candidate_objects(sc, preset)
                pairs, f_pair = O.select_pairs(sc, objs, preset)
                singles, f_single = O.select_singles(sc, objs, preset)
                cls.results[(name, preset)] = SimpleNamespace(
                    objs=objs, pairs=pairs, singles=singles, f_obj=f_obj, f_pair=f_pair, f_single=f_single)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class ReferringOnSyntheticScenes(_SyntheticBase):
    def test_every_ref_resolves_to_exactly_one_object(self):
        for name in ("dense_duplicates", "two_rooms"):
            sc = self.scenes[name]
            res = self.results[(name, "v10_relaxed")]
            self.assertTrue(res.objs)
            for o in res.objs:
                hits = R.resolve(sc, sc.objects, o.ref)
                self.assertEqual([h.id for h in hits], [o.id], f"{name}: {o.ref!r} -> {[h.id for h in hits]}")

    def test_no_viewpoint_words(self):
        for (name, preset), res in self.results.items():
            for o in res.objs:
                self.assertIsNotNone(o.ref)
                self.assertIsNone(VIEWPOINT_RE.search(o.ref), f"{name}/{preset}: {o.ref!r}")
                self.assertFalse(R.has_viewpoint_word(o.ref))

    def test_v10_yields_strictly_more_than_v9(self):
        for name in ("dense_duplicates", "two_rooms"):
            v9 = self.results[(name, "v9_strict")]
            v10 = self.results[(name, "v10_relaxed")]
            self.assertGreater(len(v10.objs), len(v9.objs), name)
            self.assertGreater(len(v10.pairs), len(v9.pairs), name)
            self.assertGreaterEqual(len(v10.singles), len(v9.singles), name)

    def test_dense_duplicates_details(self):
        sc = self.scenes["dense_duplicates"]
        v9 = self.results[("dense_duplicates", "v9_strict")]
        self.assertEqual(sorted(o.id for o in v9.objs), ["1", "10", "6", "9"])
        v10 = {o.id: o for o in self.results[("dense_duplicates", "v10_relaxed")].objs}
        self.assertEqual(v10["7"].ref, "the cabinet next to the kitchen counter")
        self.assertEqual(v10["7"].ref_method, "anchor")
        # four identical chairs around the table and the far cabinet: no unambiguous expression
        for cid in ("2", "3", "4", "5", "8"):
            self.assertNotIn(cid, v10)
        f = self.results[("dense_duplicates", "v10_relaxed")].f_obj.to_dict()
        self.assertEqual(f["objects"]["reject_no_ref"], 5)
        self.assertEqual(f["referring"]["no_ref"], 5)
        self.assertEqual(f["referring"]["anchor"], 1)
        self.assertIn("anchor:margin_too_small", f["referring.no_ref_detail"])
        self.assertIn("size:not_exactly_two", f["referring.no_ref_detail"])
        # raw labels are kept; display text only in ref
        self.assertEqual(v10["9"].label, "cooking bench")
        self.assertEqual(v10["9"].ref, "the kitchen counter")
        # the scene itself is never mutated
        self.assertTrue(all(o.ref is None for o in sc.objects))

    def test_two_rooms_details(self):
        v10 = {o.id: o for o in self.results[("two_rooms", "v10_relaxed")].objs}
        v9 = {o.id: o for o in self.results[("two_rooms", "v9_strict")].objs}
        self.assertEqual(v10["12"].ref, "the bedside table next to the wardrobe")
        self.assertNotIn("11", v10)  # tied between bed sides, nothing else nearby
        self.assertNotIn("6", v10)
        self.assertNotIn("7", v10)
        self.assertIn("3", v10)  # TV: volume 0.07 m^3 < v9's 0.1
        self.assertNotIn("3", v9)
        self.assertEqual(v10["1"].ref, "the sofa")  # multi person sofa
        self.assertEqual(v10["4"].ref, "the coffee table")  # tea table
        pair_ids = {frozenset((a.id, b.id)) for a, b in self.results[("two_rooms", "v10_relaxed")].pairs}
        self.assertNotIn(frozenset(("2", "3")), pair_ids)  # TV on its cabinet
        fp = self.results[("two_rooms", "v10_relaxed")].f_pair.to_dict()["pairs"]
        self.assertEqual(fp["reject_stacked_or_contained"], 1)
        self.assertGreater(fp["reject_different_rooms"], 0)
        for a, b in self.results[("two_rooms", "v10_relaxed")].pairs:
            self.assertEqual(a.room_index, b.room_index)
            self.assertFalse(R.same_category(a.label, b.label))

    def test_funnel_reasons_present(self):
        f9 = self.results[("dense_duplicates", "v9_strict")].f_obj.to_dict()["objects"]
        for key in ("input", "reject_blacklist", "pass_single_filters", "reject_not_unique_in_scene", "kept"):
            self.assertIn(key, f9)
        self.assertEqual(f9["input"], 14)
        self.assertEqual(f9["pass_single_filters"], 10)
        self.assertEqual(f9["kept"], 4)
        f10 = self.results[("two_rooms", "v10_relaxed")].f_obj.to_dict()["objects"]
        for key in ("input", "reject_structural", "reject_floor_covering", "pass_single_filters",
                    "reject_no_ref", "kept"):
            self.assertIn(key, f10)
        rejected = sum(v for k, v in f10.items() if k.startswith("reject_"))
        self.assertEqual(f10["input"], rejected + f10["kept"])
        for (name, preset), res in self.results.items():
            fp = res.f_pair.to_dict()["pairs"]
            total = sum(v for k, v in fp.items() if k.startswith("reject_") and k != "reject_object_without_ref")
            self.assertEqual(fp["input_combinations"], total + fp["admissible"], (name, preset))
            self.assertEqual(fp["selected"] + fp.get("over_budget", 0), fp["admissible"])
            fs = res.f_single.to_dict()["singles"]
            self.assertEqual(fs["input"], len(res.objs))

    def test_v9_pairs_capped_at_12_and_legacy_order(self):
        cfg = O.get_preset("v9_strict")
        self.assertEqual(cfg.max_pairs_per_scene, 12)
        sc = self.scenes["two_rooms"]
        objs, _ = O.candidate_objects(sc, "v9_strict")
        pairs, f = O.select_pairs(sc, objs, "v9_strict", max_pairs=2)
        everything, _ = O.select_pairs(sc, objs, "v9_strict", max_pairs=None)
        self.assertEqual([(a.id, b.id) for a, b in pairs], [(a.id, b.id) for a, b in everything[:2]])
        self.assertEqual(f.to_dict()["pairs"]["over_budget"], len(everything) - 2)

    def test_adjacent_rooms_flag_adds_cross_room_pairs(self):
        sc = self.scenes["two_rooms"]
        objs = self.results[("two_rooms", "v10_relaxed")].objs
        cfg = O.get_preset("v10_relaxed", pair_allow_adjacent_rooms=True)
        self.assertEqual(O.adjacent_rooms(sc, cfg.adjacent_room_max_gap_m), frozenset({frozenset((0, 1))}))
        pairs, _ = O.select_pairs(sc, objs, cfg, max_pairs=1000)
        self.assertTrue(any(a.room_index != b.room_index for a, b in pairs))
        self.assertGreater(len(pairs), len(self.results[("two_rooms", "v10_relaxed")].pairs))


class LegacyEquivalence(_SyntheticBase):
    """v9_strict single/pair rules reproduce the legacy ObjectSelector."""

    def test_blacklist_and_thresholds_match_legacy_config(self):
        from data_gen.active_spatial_pipeline.config import ObjectSelectionConfig

        legacy = ObjectSelectionConfig()
        cfg = O.get_preset("v9_strict")
        self.assertEqual(set(O.LEGACY_BLACKLIST), set(legacy.blacklist))
        self.assertEqual((cfg.min_dim_component, cfg.max_dim_component, cfg.min_volume, cfg.min_aspect_ratio),
                         (legacy.min_dim_component, legacy.max_dim_component, legacy.min_volume,
                          legacy.min_aspect_ratio))
        for ours, theirs in (("legacy_min_pair_dist", "min_pair_dist"), ("legacy_max_pair_dist", "max_pair_dist"),
                             ("legacy_dyn_min_mult", "dyn_min_mult"), ("legacy_dyn_max_mult", "dyn_max_mult"),
                             ("legacy_max_pair_dim_ratio", "max_pair_dim_ratio"),
                             ("legacy_max_pair_dim_diff", "max_pair_dim_diff"),
                             ("legacy_require_same_room", "require_same_room"),
                             ("legacy_pair_fov_deg", "pair_fov_deg"),
                             ("legacy_pair_fov_margin_deg", "pair_fov_margin_deg"),
                             ("legacy_min_viewing_distance_mult", "min_viewing_distance_mult")):
            self.assertEqual(getattr(cfg, ours), getattr(legacy, theirs), ours)

    def test_single_and_pair_filters_match_object_selector(self):
        from data_gen.active_spatial_pipeline.config import ObjectSelectionConfig
        from data_gen.active_spatial_pipeline.object_selector import ObjectSelector

        sel = ObjectSelector(ObjectSelectionConfig())
        cfg = O.get_preset("v9_strict")
        n_pairs = 0
        for name, path in self.paths.items():
            sc = self.scenes[name]
            items = json.loads((path / "labels.json").read_text())
            legacy = {}
            for item in items:
                lo = sel.parse_object(item, [])
                legacy[lo.id] = lo
            passing = []
            for o in sc.objects:
                ok, why = sel.filter_single_object(legacy[o.id])
                ours = O.single_object_rejection(o, cfg)
                self.assertEqual(ok, ours is None, (name, o.id, o.label, why, ours))
                if not ok:
                    self.assertEqual(why, ours)
                else:
                    passing.append(o)
            for a, b in combinations(passing, 2):
                la, lb = legacy[a.id], legacy[b.id]
                la.room_index, lb.room_index = a.room_index, b.room_index
                ok, why = sel.filter_object_pair(la, lb)
                ours = O.pair_rejection(a, b, cfg, a.room_index, b.room_index)
                self.assertEqual(ok, ours is None, (name, a.id, b.id, why, ours))
                if not ok:
                    self.assertTrue(why.startswith(ours), (why, ours))
                n_pairs += 1
        self.assertGreater(n_pairs, 20)


class MiniSceneMethods(unittest.TestCase):
    def setUp(self):
        self.scene = mini_scene()
        self.objs, self.funnel = O.candidate_objects(self.scene, "v10_relaxed")
        self.by_id = {o.id: o for o in self.objs}

    def test_expected_expressions(self):
        expected = {
            "a1": ("the armchair in the living room", "room"),
            "a2": ("the armchair in the bedroom", "room"),
            "c1": ("the smaller cabinet", "size"),
            "c2": ("the larger cabinet", "size"),
            "b1": ("the shorter bookshelf", "size"),
            "b2": ("the taller bookshelf", "size"),
            "s3": ("the stool next to the dining table", "anchor"),
            "dt": ("the dining table", "unique_label"),
            "m": ("the mattress", "unique_label"),
            "fl": ("the floor lamp", "unique_label"),
        }
        self.assertEqual({k: (o.ref, o.ref_method) for k, o in self.by_id.items()}, expected)
        for o in self.objs:
            self.assertEqual([h.id for h in R.resolve(self.scene, None, o.ref)], [o.id], o.ref)

    def test_exclusion_reasons(self):
        f = self.funnel.to_dict()
        objs = f["objects"]
        self.assertEqual(objs["reject_door_window"], 2)
        self.assertEqual(objs["reject_light_fixture"], 1)  # wall lamp; floor lamp kept
        self.assertEqual(objs["reject_floor_covering"], 1)
        self.assertEqual(objs["reject_ceiling_mounted"], 1)  # projector, bottom at 2.5 m
        self.assertEqual(objs["reject_longest_edge_too_small"], 1)  # 0.3 m vase
        self.assertEqual(objs["reject_no_ref"], 3)  # s1, s2, generic "table"
        detail = f["referring.no_ref_detail"]
        self.assertIn("room:not_unique_in_room_type", detail)
        self.assertIn("anchor:margin_too_small", detail)
        self.assertIn("size:ratio_below_threshold", detail)  # table vs dining table
        self.assertIn("size:not_exactly_two", detail)  # three stools

    def test_hypernym_makes_generic_label_ambiguous(self):
        self.assertEqual(sorted(o.id for o in R.resolve(self.scene, None, "the table")), ["dt", "t"])
        self.assertEqual([o.id for o in R.resolve(self.scene, None, "the dining table")], ["dt"])
        self.assertNotIn("t", self.by_id)

    def test_size_requires_same_room_flag(self):
        sc = mini_scene()
        sc.objects[3].room_index = 0  # move c2's room tag elsewhere
        cfg = O.get_preset("v10_relaxed")
        objs, f = O.candidate_objects(sc, cfg)
        self.assertNotIn("c1", {o.id for o in objs})
        self.assertIn("size:other_room", f.to_dict()["referring.no_ref_detail"])
        relaxed = O.get_preset(cfg, referring=R.ReferringConfig(size_requires_same_room=False))
        objs2, _ = O.candidate_objects(sc, relaxed)
        self.assertEqual({o.id: o.ref for o in objs2}["c1"], "the smaller cabinet")

    def test_singles_budget_and_category_cap(self):
        cfg = O.get_preset("v10_relaxed", max_singles_per_scene=4, max_singles_per_category=1)
        singles, f = O.select_singles(self.scene, self.objs, cfg)
        self.assertEqual(len(singles), 4)
        cats = [R.display_label(o.label) for o in singles]
        self.assertEqual(len(cats), len(set(cats)))
        fs = f.to_dict()["singles"]
        self.assertEqual(fs["selected"] + fs.get("reject_category_cap", 0) + fs.get("reject_over_budget", 0),
                         fs["input"])
        again, _ = O.select_singles(self.scene, list(reversed(self.objs)), cfg)
        self.assertEqual([o.id for o in singles], [o.id for o in again])


class FuzzConsistency(unittest.TestCase):
    """Random dense scenes: every emitted expression resolves to exactly its
    object and contains no viewpoint word."""

    LABELS = ["chair", "dining chair", "armchair", "table", "dining table", "side table", "tea table",
              "teatable", "cabinet", "tv cabinet", "sofa", "multi person sofa", "bed", "bed combination",
              "bedside table", "plant", "green plants", "stool", "desk", "wardrobe", "tv", "floor lamp",
              "door", "window", "rug", "mattress"]
    NAMES = ["LivingRoom", "MasterBedroom", "SecondBedroom", "Kitchen", "room_7", "OtherRoom"]

    def _scene(self, rng):
        rooms = [_room(k, rng.choice(self.NAMES), 5 * (k // 2), 5 * (k % 2), 5 * (k // 2) + 5, 5 * (k % 2) + 5)
                 for k in range(4)]
        objs = []
        for i in range(rng.randint(5, 40)):
            w, d, h = rng.uniform(0.2, 2.5), rng.uniform(0.2, 2.5), rng.uniform(0.05, 2.2)
            x, y, z = rng.uniform(-0.3, 10), rng.uniform(-0.3, 10), rng.choice([0, 0, 0, 0.8, 2.3])
            o = _box(i, rng.choice(self.LABELS), x, y, z, x + w, y + d, z + h)
            c = o.center
            o.room_index = next((r.index for r in rooms if 0 <= c[0] - r.polygon[0, 0] < 5
                                 and 0 <= c[1] - r.polygon[0, 1] < 5), None)
            objs.append(o)
        return SimpleNamespace(scene_id="fuzz", scene_dir=None, objects=objs, rooms=rooms, convention={})

    def test_generated_refs_resolve_uniquely(self):
        rng = random.Random(7)
        n, methods = 0, set()
        for _ in range(150):
            sc = self._scene(rng)
            objs, _ = O.candidate_objects(sc, "v10_relaxed")
            resolver = R.Resolver(sc)
            for o in objs:
                self.assertEqual([h.id for h in resolver(o.ref)], [o.id], o.ref)
                self.assertFalse(R.has_viewpoint_word(o.ref), o.ref)
                methods.add(o.ref_method)
                n += 1
            pairs, _ = O.select_pairs(sc, objs, "v10_relaxed")
            for a, b in pairs:
                self.assertFalse(R.same_category(a.label, b.label))
        self.assertGreater(n, 500)
        self.assertEqual(methods, {"unique_label", "room", "anchor", "size"})


class PairSelection(unittest.TestCase):
    def _grid_scene(self):
        labels = ["sofa", "tv cabinet", "armchair", "bookshelf", "desk", "plant", "piano", "fish tank",
                  "dresser", "stool", "wardrobe", "refrigerator"]
        objs = []
        for k, lab in enumerate(labels):
            x, y = 0.5 + 1.6 * (k % 4), 0.5 + 1.6 * (k // 4)
            objs.append(_box(k, lab, x, y, 0, x + 0.6, y + 0.6, 0.8, room=0))
        return SimpleNamespace(scene_id="grid", scene_dir=None, objects=objs,
                               rooms=[_room(0, "living room", 0, 0, 7, 5)], convention={})

    def test_deterministic_input_order_independent_and_stratified(self):
        sc = self._grid_scene()
        objs, _ = O.candidate_objects(sc, "v10_relaxed")
        self.assertEqual(len(objs), 12)
        p1, f = O.select_pairs(sc, objs, "v10_relaxed", max_pairs=8, seed=3)
        shuffled = list(objs)
        random.Random(1).shuffle(shuffled)
        p2, _ = O.select_pairs(sc, shuffled, "v10_relaxed", max_pairs=8, seed=3)
        key = lambda ps: sorted(tuple(sorted((a.id, b.id))) for a, b in ps)
        self.assertEqual(key(p1), key(p2))
        self.assertEqual(len(p1), 8)
        edges = O.get_preset("v10_relaxed").distance_band_edges_m
        bands = {int(np.searchsorted(edges, O.center_distance_xy(a, b), side="right")) for a, b in p1}
        self.assertGreaterEqual(len(bands), 3)
        fp = f.to_dict()["pairs"]
        self.assertEqual(fp["selected"], 8)
        self.assertEqual(fp["over_budget"], fp["admissible"] - 8)
        p3, _ = O.select_pairs(sc, objs, "v10_relaxed", max_pairs=8, seed=4)
        self.assertEqual(len(p3), 8)

    def test_expand_ordered(self):
        a, b = _box("a", "sofa", 0, 0, 0, 1, 1, 1), _box("b", "desk", 2, 0, 0, 3, 1, 1)
        self.assertEqual([(x.id, y.id) for x, y in O.expand_ordered([(a, b)])], [("a", "b"), ("b", "a")])


class LabelTaxonomy(unittest.TestCase):
    def test_display_labels(self):
        self.assertEqual(R.display_label("teatable"), "coffee table")
        self.assertEqual(R.display_label("tea table"), "coffee table")
        self.assertEqual(R.display_label("washing_machine"), "washing machine")
        self.assertEqual(R.display_label("tv"), "TV")
        self.assertEqual(R.display_label("tv cabinet"), "TV cabinet")
        self.assertEqual(R.display_label("bed combination"), "bed")
        self.assertEqual(R.display_label("smoke lampblack machine"), "range hood")
        self.assertEqual(R.display_label("the mahjong table"), "mahjong table")
        self.assertEqual(R.display_label("dining chair"), "dining chair")
        for lab, disp in R.DISPLAY_LABELS.items():
            self.assertFalse(R.has_viewpoint_word(disp), lab)

    def test_same_category(self):
        self.assertTrue(R.same_category("tea table", "teatable"))
        self.assertTrue(R.same_category("chair", "dining chair"))
        self.assertTrue(R.same_category("cabinet", "tv cabinet"))
        self.assertFalse(R.same_category("armchair", "chair"))
        self.assertFalse(R.same_category("tv", "tv cabinet"))

    def test_room_types(self):
        self.assertEqual(R.room_type("MasterBedroom"), "bedroom")
        self.assertEqual(R.room_type("second_bedroom_2"), "bedroom")
        self.assertEqual(R.room_type("LivingDiningRoom"), "living room")
        self.assertEqual(R.room_type("kitchen"), "kitchen")
        self.assertEqual(R.room_type("主卧"), "bedroom")
        self.assertIsNone(R.room_type("room_3"))
        self.assertIsNone(R.room_type("OtherRoom"))
        self.assertIsNone(R.room_type(""))

    def test_semantic_exclusions(self):
        cfg = O.get_preset("v10_relaxed")
        for lab in O.LEGACY_BLACKLIST:
            self.assertIsNotNone(O.semantic_exclusion_reason(lab, cfg), lab)
        cases = {
            "door": "door_window", "sliding door": "door_window", "window combination": "door_window",
            "wall": "structural", "ceiling combination": "structural", "shower room": "structural",
            "floor mold": "structural", "rug": "floor_covering", "doormat": "floor_covering",
            "wall lamp": "light_fixture", "table lamp": "light_fixture", "decorative pendant": "light_fixture",
            "venetian blinds": "curtain_blind", "bathroom curtain": "curtain_blind",
            "other": "generic_or_wall_fixture", "switch panel": "generic_or_wall_fixture",
            "blanket": "soft_furnishing", "pillow combination": "soft_furnishing",
            "book": "small_clutter", "": "empty_label",
        }
        for lab, why in cases.items():
            self.assertEqual(O.semantic_exclusion_reason(lab, cfg), why, lab)
        for lab in ("floor lamp", "mattress", "wall cabinet", "chair", "tv", "shower head", "person"):
            self.assertIsNone(O.semantic_exclusion_reason(lab, cfg), lab)
        self.assertIsNotNone(O.semantic_exclusion_reason("door", O.get_preset("v10_relaxed")))
        self.assertIsNone(O.semantic_exclusion_reason("door", O.get_preset("v10_relaxed",
                                                                           exclude_doors_windows=False)))
        # v9: exact legacy blacklist only
        self.assertIsNone(O.semantic_exclusion_reason("door", "v9_strict"))
        self.assertEqual(O.semantic_exclusion_reason("lamp", "v9_strict"), "blacklist")

    def test_presets(self):
        self.assertEqual(O.get_preset("v10_relaxed").max_pairs_per_scene, 60)
        self.assertEqual(O.get_preset("v10_relaxed", max_pairs_per_scene=5).max_pairs_per_scene, 5)
        with self.assertRaises(ValueError):
            O.get_preset("nope")
        json.dumps(O.get_preset("v10_relaxed").to_json())


class Cli(_SyntheticBase):
    def test_cli_writes_stats(self):
        out = self.root / "stats.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = O.main(["--scenes-root", str(self.root), "--out", str(out)])
        self.assertEqual(rc, 0)
        data = json.loads(out.read_text())
        s = data["summary"]
        self.assertEqual(s["scenes_total"], 3)
        for preset in PRESET_NAMES:
            self.assertIn("pass_single_filters", s[preset]["totals"])
            self.assertIn("funnel", s[preset])
        self.assertGreater(s["v10_relaxed"]["totals"]["objects"], s["v9_strict"]["totals"]["objects"])
        rec = data["per_scene"]["synth_dense_duplicates"]["presets"]["v9_strict"]
        self.assertEqual((rec["pass_single_filters"], rec["objects"]), (10, 4))
        self.assertIn("v9_strict", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
