"""Tests for workstream A: admission, QA states and release (stdlib unittest)."""

import copy
import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_gen.interiorgs_v10 import admission as A
from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import qa_states as Q
from data_gen.interiorgs_v10 import release as R
from data_gen.interiorgs_v10 import sampling
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

ROOT = Path(__file__).resolve().parents[3]
SEQS = [list(s) for n in (2, 3) for s in itertools.product(C.ACTIONS, repeat=n)]
# (scene, A id, B id, relation, terminal x, y, yaw_deg, certificate) found by search on the synthetic scenes;
# d* of each initial state equals the certificate length (verified again in _episode).
FIXTURES = (("two_rooms", "2", "4", "left", 0.6, 3.0, 0, ["move_left", "move_left", "move_left"]),
            ("dense_duplicates", "1", "6", "left", 4.8, 1.2, 120, ["move_forward", "move_forward", "turn_left"]))


def _episode(scene, scene_id, a_id, b_id, rel, x, y, yaw_deg, seq0, policy):
    objs = {o.id: copy.deepcopy(o) for o in scene.objects}
    oa, ob = objs[a_id], objs[b_id]
    oa.ref, ob.ref = f"the {oa.label} in the {scene_id.replace('_', ' ')}", f"the {ob.label}"
    term = C.pose_from_xy_yaw(x, y, math.radians(yaw_deg))
    legal = Q.scene_legal_fn(scene)
    for seq in [seq0] + SEQS:
        init = term.copy()
        for a in reversed(seq):
            init = C.apply_action(init, C.INVERSE_ACTION[a])
        row = sampling.projective_item(scene_id, oa, ob, rel, init)
        row.update(task_id=f"{scene_id}_{a_id}_{b_id}_{'_'.join(seq)}",
                   difficulty={"d_star": len(seq), "lower_bound_complete": True, "certificate": list(seq)},
                   initial_observability={})
        if not A.precheck_task(row, scene, policy)["ok"]:
            continue
        d, _ = Q.optimal_first_actions(row, init, max_depth=len(seq), legal_fn=legal)
        if d == len(seq):
            return row
    raise AssertionError(f"no fixture episode for {scene_id}")


def _review(row, state_ids, frames=None, task_level=None, kind="vlm"):
    slots = [s for s, _ in Q.task_slots(row)]
    rv = A.review_template(row, state_ids)
    rv["reviewer"] = {"kind": kind, "name": "fixture"}
    rv["task_level"] = {"label_matches": {s: True for s in slots}, "referring_unambiguous": {s: True for s in slots},
                        "task_possible": True, "notes": ""}
    rv["task_level"].update(task_level or {})
    for sid in state_ids:
        rv["frames"][sid] = {"render_ok": True, "targets_identifiable": {s: True for s in slots},
                             "box_alignment_ok": {s: True for s in slots}, "notes": ""}
        rv["frames"][sid].update((frames or {}).get(sid, {}))
    return rv


class AdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        paths = write_synthetic_scenes(cls.root / "scenes")
        cls.policy = A.load_policy()
        cls.scenes = {}
        cls.rows = {}
        for scene_id, a_id, b_id, rel, x, y, yaw, seq in FIXTURES:
            cls.scenes[scene_id] = load_scene(paths[scene_id], scene_id)
            cls.rows[scene_id] = _episode(cls.scenes[scene_id], scene_id, a_id, b_id, rel, x, y, yaw, seq, cls.policy)
        cls.qa = {sid: Q.generate_qa(row, scene=cls.scenes[sid], config=cls.policy["qa_generation"])
                  for sid, row in cls.rows.items()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _frames(self, row):
        return list(A.required_review_frames(row, self.policy, self.qa[row["scene_id"]]))

    # ------------------------------------------------------------ admission
    def test_initial_frame_with_one_cropped_target_is_admitted(self):
        row = self.rows["two_rooms"]
        rv = _review(row, self._frames(row), frames={"s0": {"targets_identifiable": {"A": True, "B": False},
                                                            "box_alignment_ok": {"A": True, "B": None}}})
        res = A.admit([row], [rv], self.policy)
        self.assertEqual(res["rl"]["decisions"][row["task_id"]]["status"], "ADMIT")
        self.assertEqual(res["sft"]["decisions"][row["task_id"]]["status"], "ADMIT")

    def test_no_identifiable_initial_target_is_hold_and_qa_still_admitted(self):
        row = self.rows["two_rooms"]
        rv = _review(row, self._frames(row), frames={"s0": {"targets_identifiable": {"A": False, "B": False}}})
        res = A.admit([row], [rv], self.policy, qa_items=self.qa["two_rooms"])
        dec = res["rl"]["decisions"][row["task_id"]]
        self.assertEqual((dec["status"], dec["reasons"]), ("HOLD", ["initial_no_identifiable_target"]))
        later = [q for q in self.qa["two_rooms"] if q["state_id"] != "s0"]
        self.assertTrue(later)
        self.assertTrue(all(res["qa"]["decisions"][q["sample_id"]]["status"] == "ADMIT" for q in later))

    def test_terminal_target_missing_and_missing_review_are_hold(self):
        row = self.rows["two_rooms"]
        d = len(row["difficulty"]["certificate"])
        rv = _review(row, self._frames(row), frames={f"s{d}": {"targets_identifiable": {"A": True, "B": False}}})
        res = A.admit([row], [rv], self.policy)
        self.assertEqual(res["rl"]["decisions"][row["task_id"]]["reasons"], ["terminal_target_not_identifiable"])
        res = A.admit([row], [], self.policy)
        self.assertEqual(res["rl"]["decisions"][row["task_id"]], {"status": "HOLD", "reasons": ["review_missing"],
                                                                   "scene_id": "two_rooms"})

    def test_task_invalid_blocks_rl_sft_and_qa(self):
        row = self.rows["two_rooms"]
        rv = _review(row, self._frames(row), task_level={"referring_unambiguous": {"A": True, "B": False}})
        res = A.admit([row], [rv], self.policy, qa_items=self.qa["two_rooms"])
        for m in ("rl", "sft"):
            self.assertEqual(res[m]["decisions"][row["task_id"]]["reasons"], ["task_invalid:referring_ambiguous"])
        self.assertEqual(res["qa"]["admitted"], [])
        self.assertEqual(res["qa"]["reason_counts"], {"qa_parent_task_invalid": len(self.qa["two_rooms"])})

    def test_box_misaligned_on_terminal_invalidates_task(self):
        row = self.rows["two_rooms"]
        d = len(row["difficulty"]["certificate"])
        rv = _review(row, self._frames(row), frames={f"s{d}": {"box_alignment_ok": {"A": False, "B": True}}})
        res = A.admit([row], [rv], self.policy)
        self.assertEqual(res["rl"]["decisions"][row["task_id"]]["status"], "REJECT")
        self.assertIn(row["task_id"], res["task_invalid"])

    def test_qa_from_rejected_but_valid_parent_is_admitted(self):
        row = self.rows["two_rooms"]
        d = len(row["difficulty"]["certificate"])
        rv = _review(row, self._frames(row), frames={f"s{d}": {"render_ok": False}})
        res = A.admit([row], [rv], self.policy, qa_items=self.qa["two_rooms"])
        self.assertEqual(res["rl"]["decisions"][row["task_id"]]["status"], "REJECT")
        self.assertEqual(res["rl"]["decisions"][row["task_id"]]["reasons"], ["render_artifact_terminal"])
        self.assertNotIn(row["task_id"], res["task_invalid"])
        s0 = [q for q in self.qa["two_rooms"] if q["state_id"] == "s0"]
        self.assertTrue(s0 and all(res["qa"]["decisions"][q["sample_id"]]["status"] == "ADMIT" for q in s0))
        term = [q for q in self.qa["two_rooms"] if q["state_id"] == f"s{d}"]
        self.assertTrue(all(res["qa"]["decisions"][q["sample_id"]]["reasons"] == ["qa_render_artifact"] for q in term))

    def test_qa_item_needs_its_referenced_targets(self):
        row = self.rows["two_rooms"]
        pair = [q for q in self.qa["two_rooms"] if q["requires_identifiable_mode"] == "all"
                and len(q["requires_identifiable"]) == 2]
        self.assertTrue(pair)
        q = pair[0]
        rv = _review(row, self._frames(row), frames={q["state_id"]: {"targets_identifiable": {"A": True, "B": False}}})
        res = A.admit([row], [rv], self.policy, qa_items=[q])
        self.assertEqual(res["qa"]["decisions"][q["sample_id"]]["reasons"], ["qa_target_not_identifiable"])

    def test_human_review_overrides_vlm(self):
        row = self.rows["two_rooms"]
        frames = self._frames(row)
        bad = _review(row, frames, frames={"s0": {"targets_identifiable": {"A": False, "B": False}}})
        human = {"review_schema_version": A.REVIEW_SCHEMA_VERSION, "task_id": row["task_id"],
                 "reviewer": {"kind": "human"}, "frames": {"s0": {"targets_identifiable": {"A": True}}}}
        res = A.admit([row], [bad, human], self.policy)
        self.assertEqual(res["rl"]["decisions"][row["task_id"]]["status"], "ADMIT")

    def test_reviewer_prompt_and_template(self):
        row = self.rows["two_rooms"]
        text = A.reviewer_prompt(row, ["s0", "s2"], self.policy)
        self.assertIn("STILL identifiable", text)
        self.assertIn(Q.ref_text(row["target_object"]["objects"][0]), text)
        self.assertNotIn("{", text.split("Answer with JSON only")[0])
        self.assertEqual(A.validate_review(A.review_template(row, ["s0"]), row), [])

    def test_precheck(self):
        row = self.rows["two_rooms"]
        self.assertTrue(A.precheck_task(row, self.scenes["two_rooms"], self.policy)["ok"])
        bad = copy.deepcopy(row)
        bad["init_camera"]["extrinsics"] = Q.certificate_poses(row)[-1].tolist()
        self.assertEqual(A.precheck_task(bad, self.scenes["two_rooms"], self.policy)["reason"],
                         "precheck:initial_already_successful")
        bad = copy.deepcopy(row)
        bad["difficulty"]["certificate"] = row["difficulty"]["certificate"][:-1]
        bad["difficulty"]["d_star"] = 1
        self.assertIn(A.precheck_task(bad, self.scenes["two_rooms"], self.policy)["reason"],
                      ("precheck:certificate_not_successful",))

    def test_chunking(self):
        acts = ["turn_left"] * 2 + ["move_right"] * 7
        self.assertEqual([len(c) for c in A.chunk_certificate(acts, "action_runs", 5)], [2, 5, 2])
        self.assertEqual([len(c) for c in A.chunk_certificate(acts, "greedy", 5)], [5, 4])
        self.assertEqual([len(c) for c in A.chunk_certificate(acts, "balanced", 5)], [5, 4])

    # ---------------------------------------------------------------- QA
    def test_qa_labels_are_geometric(self):
        row = self.rows["two_rooms"]
        qa = self.qa["two_rooms"]
        kinds = {q["qa_kind"] for q in qa}
        self.assertTrue({"read_goal", "act_choice", "act_yesno", "predict_shift", "read_inside_frame"} <= kinds, kinds)
        legal = Q.scene_legal_fn(self.scenes["two_rooms"])
        poses = {f"s{k}": p for k, p in enumerate(Q.certificate_poses(row))}
        cert = row["difficulty"]["certificate"]
        for q in qa:
            self.assertIn(q["private_answer"], ("Yes", "No") if q["answer_format"] == "yes_no" else
                          [q["private_answer"]])
            self.assertNotIn("<image>", q["question"])
            c2w = np.asarray(q["state_pose_c2w"])
            if q["qa_kind"] == "read_left_of":
                p = Q.project_entries(row, c2w)
                x, y = q["requires_identifiable"]
                self.assertEqual(q["private_answer"] == "Yes", p[x]["center_uv"][0] < p[y]["center_uv"][0])
            if q["qa_kind"] == "predict_shift":
                a = q["_audit"]["action"]
                u0 = Q.project_entries(row, c2w)[q["requires_identifiable"][0]]["center_uv"][0]
                u1 = Q.project_entries(row, C.apply_action(c2w, a))[q["requires_identifiable"][0]]["center_uv"][0]
                left = "left edge" in q["question"]
                self.assertEqual(q["private_answer"] == "Yes", (u1 < u0) == left)
            if q["qa_kind"] == "act_choice":
                k = q["state_index"]
                self.assertIn(cert[k], q["accepted_answers"])
                _, best = Q.optimal_first_actions(row, poses[q["state_id"]], max_depth=3, legal_fn=legal)
                self.assertEqual(set(q["accepted_answers"]), best)
            if q["qa_kind"] == "read_goal":
                self.assertEqual(q["private_answer"] == "Yes", bool(sampling.score(row, c2w)["success"]))

    def test_near_boundary_states_are_excluded_and_counted(self):
        row = self.rows["two_rooms"]
        f = C.Funnel()
        cfg = dict(self.policy["qa_generation"], left_of_min_sep_px=1e6, closer_min_abs_diff_m=1e6)
        qa = Q.generate_qa(row, scene=self.scenes["two_rooms"], config=cfg, funnel=f)
        self.assertFalse([q for q in qa if q["qa_kind"] in ("read_left_of", "read_closer")])
        self.assertGreater(f.counts["qa_gen"].get("near_boundary_separation", 0), 0)
        self.assertGreater(f.counts["qa_gen"].get("near_boundary_depth", 0), 0)

    def test_balance_yes_no(self):
        recs = [{"sample_id": str(i), "qa_kind": "k", "split": "train", "answer_format": "yes_no",
                 "private_answer": "Yes" if i < 7 else "No"} for i in range(10)]
        kept = Q.balance_yes_no(recs)
        self.assertEqual(sorted(r["private_answer"] for r in kept), ["No"] * 3 + ["Yes"] * 3)

    # ------------------------------------------------------------ release
    def test_sft_export_has_no_leakage(self):
        row = R.finalize_task_row(self.rows["two_rooms"], "train")
        d = len(row["difficulty"]["certificate"])
        images = {f"{row['task_id']}:s{k}": str(self.root / f"x{k}.png") for k in range(d + 1)}
        rec, why = R.build_sft_record(row, images, self.policy)
        self.assertIsNone(why)
        self.assertEqual(R.scan_leakage(rec, self.policy), [])
        for turn in rec["conversations"]:
            if turn["role"] == "assistant":
                self.assertRegex(turn["content"], r"^<action>([a-z_]+\|)+</action>$")
            if turn["role"] == "user":
                text = turn["content"].replace(rec["task_description"], "")
                self.assertNotRegex(text, r"\d+(\.\d+)?\s*(m|meters?|degrees?|°|px)\b")
                self.assertNotRegex(text.lower(), r"score|reward|distance")
        self.assertEqual(sum(t["role"] == "user" for t in rec["conversations"]), rec["trajectory_steps"])
        leaked = copy.deepcopy(rec)
        leaked["conversations"][1]["content"] += "\nDistance to target: 1.25 m"
        self.assertTrue(R.scan_leakage(leaked, self.policy))

    def test_split_groups_reserved_and_frozen(self):
        scenes = [f"s{i:02d}" for i in range(20)]
        m = R.assign_splits(scenes, groups=[["s00", "s01", "s02"]], reserved={"s05": "test"}, seed="x")
        look = R.split_lookup(m)
        self.assertEqual(len({look["s00"], look["s01"], look["s02"]}), 1)
        self.assertEqual(look["s05"], "test")
        self.assertGreaterEqual(len(m["train_scenes"]), 14)
        m2 = R.assign_splits(scenes + ["s99"], groups=[["s00", "s01", "s02"]], frozen=m, seed="other")
        look2 = R.split_lookup(m2)
        self.assertTrue(all(look2[s] == look[s] for s in scenes))
        with self.assertRaises(ValueError):
            R.assign_splits(["a", "b"], groups=[["a", "b"]], reserved={"a": "train", "b": "test"})

    def test_split_requires_explicit_groups(self):
        scenes = [f"s{i:02d}" for i in range(10)]
        with self.assertRaises(ValueError):
            R.same_source_groups(scenes)
        with self.assertRaises(ValueError):
            R.assign_splits(scenes)  # groups=None no longer means "no groups"
        self.assertEqual(R.assign_splits(scenes, groups=[])["physical_groups"], 10)
        path = self.root / "groups.json"
        path.write_text(json.dumps({"version": "scene_identity_v1", "groups": [["s01", "s07"]], "edges": []}))
        groups = R.load_groups(path)
        self.assertEqual(groups, [["s01", "s07"]])
        look = R.split_lookup(R.assign_splits(scenes, groups=groups))
        self.assertEqual(look["s01"], look["s07"])

    def test_regeneration_planner(self):
        stats = {
            "full": {"admitted_rl": 12, "attempted": 20},
            "short": {"admitted_rl": 3, "attempted": 10, "reasons": {"initial_no_identifiable_target": 4,
                                                                      "task_invalid:wrong_object": 1}},
            "empty": {"admitted_rl": 0, "attempted": 0},
            "dead": {"admitted_rl": 0, "attempted": 30, "exhausted": True},
        }
        plan = R.plan_regeneration(stats, self.policy)
        by = {r["scene_id"]: r for r in plan["requests"]}
        self.assertNotIn("dead", by)
        self.assertEqual(plan["requests"][0]["scene_id"], "empty")
        self.assertGreaterEqual(by["short"]["deficit"], 8)
        self.assertGreater(by["short"]["request_candidates"], by["short"]["deficit"])
        self.assertEqual(by["short"]["hints"].get("resample_initial"), 4)
        self.assertFalse(plan["status"]["targets_met"])
        small = dict(self.policy, regeneration=dict(self.policy["regeneration"], target_total_tasks=20,
                                                    target_scenes=2))
        done = R.plan_regeneration({"a": {"admitted_rl": 11, "attempted": 11},
                                    "b": {"admitted_rl": 11, "attempted": 11}}, small)
        self.assertTrue(done["status"]["targets_met"])
        self.assertEqual(done["requests"], [])

    def test_tiny_release_passes_prepare_validation(self):
        from PIL import Image

        rows = list(self.rows.values())
        qa = [q for sid in self.rows for q in self.qa[sid]]
        reqs = A.review_requests(rows, qa, self.policy)
        img_dir = self.root / "renders"
        img_dir.mkdir(exist_ok=True)
        renders = {}
        for i, fr in enumerate(reqs["frames"]):
            path = img_dir / f"{i:04d}.png"
            Image.new("RGB", (8, 8), (i % 255, 40, 90)).save(path)
            renders[fr["request_id"]] = str(path)
            A.draw_review_overlay(path, fr["overlay"], self.root / "overlay" / path.name)
        reviews = [_review(r, t["frames"]) for r, t in zip(rows, reqs["tasks"])]
        # one-target-cropped initial frame must not block anything
        reviews[0]["frames"]["s0"]["targets_identifiable"]["B"] = False
        out = self.root / "release"
        summary = R.build_release(out, rows, reviews, self.policy, qa_items=qa, renders=renders, groups=[],
                                  reserved={"two_rooms": "train", "dense_duplicates": "val"})
        self.assertEqual(summary["rl"]["total"], 2)
        self.assertEqual(summary["sft"]["total"], 2)
        self.assertGreater(summary["qa"]["total"], 0)

        # Raw RL rows are bound to the registry by hash and split.
        sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
        from active_spatial_release_contract import bind_raw, load_registry, scene_partitions
        from prepare_active_spatial_sft import prepare

        manifest = json.loads((out / "split_manifest.json").read_text())
        registry = load_registry(out / "task_registry.jsonl", manifest)
        lookup = scene_partitions(manifest)
        for sp in ("train", "val"):
            for line in (out / f"rl_tasks_{sp}.jsonl").read_text().splitlines():
                bind_raw(json.loads(line), registry, lookup)

        model = self.root / "model"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
        args = SimpleNamespace(out=str(self.root / "mix"), model=str(model),
                               trajectory=[str(out / "sft_trajectories.jsonl")],
                               qa_bank=[str(out / "qa_bank.jsonl")], qa_probability=0.5, val_fraction=0.1, seed=42,
                               split_manifest=str(out / "split_manifest.json"),
                               task_registry=str(out / "task_registry.jsonl"), cutoff_len=2048)
        result = prepare(args)
        self.assertEqual(result["counts"]["trajectory/train"], 1)
        self.assertEqual(result["counts"]["trajectory/val"], 1)
        self.assertGreater(result["counts"]["qa/train"], 0)
        self.assertGreater(result["counts"]["qa/val"], 0)
        self.assertEqual(result["excluded"], {})
        traj = [json.loads(x) for x in (self.root / "mix" / "trajectory_train.jsonl").read_text().splitlines()]
        self.assertTrue(all(R._ACTION_ONLY_RE.fullmatch(m["content"]) for m in traj[0]["messages"]
                            if m["role"] == "assistant"))


if __name__ == "__main__":
    unittest.main()
