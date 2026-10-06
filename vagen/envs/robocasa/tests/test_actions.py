#!/usr/bin/env python3
"""Unit tests for RoboCasa action parse/format/to_gym_action.

Must NOT import robocasa or gymnasium. Run with:

    python vagen/envs/robocasa/tests/test_actions.py
"""

from __future__ import annotations

import os
import sys
import unittest

# Make the repo importable without installing the package.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class TestActions(unittest.TestCase):
    def test_no_robocasa_import(self):
        self.assertNotIn("robocasa", sys.modules)
        self.assertNotIn("gymnasium", sys.modules)
        from vagen.envs.robocasa.utils import actions as act
        from vagen.envs.robocasa.utils import parse as parse_mod

        self.assertNotIn("robocasa", sys.modules)
        self.assertNotIn("gymnasium", sys.modules)
        self.assertEqual(act.ACTION_DIM, 12)
        self.assertTrue(hasattr(parse_mod, "parse_response"))

    def test_format_and_parse_vector(self):
        from vagen.envs.robocasa.utils.actions import format_action_text, parse_vector

        vec = [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05, 0.0, 0.0, 0.0]
        text = format_action_text(vec)
        self.assertTrue(text.startswith("[") and text.endswith("]"))
        got = parse_vector(text)
        self.assertEqual(len(got), 12)
        for a, b in zip(vec, got):
            self.assertAlmostEqual(a, b, places=4)

    def test_clip_action(self):
        from vagen.envs.robocasa.utils.actions import clip_action

        clipped = clip_action([2.0] * 12)
        self.assertTrue(all(x == 1.0 for x in clipped))
        clipped = clip_action([-3.0] * 12)
        self.assertTrue(all(x == -1.0 for x in clipped))

    def test_parse_named_action(self):
        from vagen.envs.robocasa.utils.actions import parse_named_action

        vec = parse_named_action("eef_dx=0.2, gripper_close=1, control_mode=1")
        self.assertEqual(len(vec), 12)
        self.assertAlmostEqual(vec[0], 0.2)
        self.assertAlmostEqual(vec[6], 1.0)
        self.assertAlmostEqual(vec[11], 1.0)
        self.assertAlmostEqual(vec[1], 0.0)

    def test_parse_action_block_multi(self):
        from vagen.envs.robocasa.utils.actions import format_action_text, parse_action_block

        a = [0.1] + [0.0] * 11
        b = [0.0, 0.2] + [0.0] * 10
        block = format_action_text(a) + "\n" + format_action_text(b)
        parsed = parse_action_block(block, max_actions=8)
        self.assertEqual(len(parsed), 2)
        self.assertAlmostEqual(parsed[0][0], 0.1, places=4)
        self.assertAlmostEqual(parsed[1][1], 0.2, places=4)

    def test_to_gym_action(self):
        from vagen.envs.robocasa.utils.actions import to_gym_action

        vec = [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 0.7, 0.0]
        gym = to_gym_action(vec)
        self.assertEqual(gym["action.end_effector_position"], [0.01, 0.02, 0.03])
        self.assertEqual(gym["action.end_effector_rotation"], [0.1, 0.2, 0.3])
        self.assertEqual(gym["action.gripper_close"], [1.0])
        self.assertEqual(gym["action.base_motion"], [0.4, 0.5, 0.6, 0.7])
        self.assertEqual(gym["action.control_mode"], [0.0])

    def test_from_demo_row_flat_and_mapped(self):
        from vagen.envs.robocasa.utils.actions import from_demo_row

        flat = {"action": [0.01 * i for i in range(12)]}
        self.assertEqual(from_demo_row(flat)[3], 0.03)
        mapped = {
            "action.end_effector_position": [0.1, 0.2, 0.3],
            "action.end_effector_rotation": [0.0, 0.0, 0.0],
            "action.gripper_close": [1.0],
            "action.base_motion": [0.0, 0.0, 0.0, 0.0],
            "action.control_mode": [0.0],
        }
        got = from_demo_row(mapped)
        self.assertAlmostEqual(got[0], 0.1)
        self.assertAlmostEqual(got[6], 1.0)

    def test_parse_response_free_think(self):
        from vagen.envs.robocasa.utils.actions import format_action_text
        from vagen.envs.robocasa.utils.parse import parse_response

        vec = [0.05] + [0.0] * 11
        text = f"<think>reach the mug</think><action>{format_action_text(vec)}</action>"
        parsed = parse_response(text, prompt_format="free_think", max_actions=4)
        self.assertTrue(parsed["format_correct"])
        self.assertEqual(parsed["think"], "reach the mug")
        self.assertEqual(len(parsed["actions"]), 1)

        missing_think = f"<action>{format_action_text(vec)}</action>"
        parsed2 = parse_response(missing_think, prompt_format="free_think")
        self.assertFalse(parsed2["format_correct"])
        self.assertEqual(len(parsed2["actions"]), 1)

        parsed3 = parse_response(missing_think, prompt_format="no_think")
        self.assertTrue(parsed3["format_correct"])

    def test_remap_lerobot_layout(self):
        from vagen.envs.robocasa.utils.actions import (
            ROBOCASA_LEROBOT_SLICES,
            from_demo_row,
            remap_stored_action,
        )

        # modality.json order: base(4) + control(1) + eef_pos(3) + eef_rot(3) + grip(1)
        stored = [0.4, 0.5, 0.6, 0.7, 1.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.9]
        canon = remap_stored_action(stored, ROBOCASA_LEROBOT_SLICES)
        self.assertAlmostEqual(canon[0], 0.1)
        self.assertAlmostEqual(canon[3], 0.01)
        self.assertAlmostEqual(canon[6], 0.9)
        self.assertAlmostEqual(canon[7], 0.4)
        self.assertAlmostEqual(canon[11], 1.0)
        got = from_demo_row({"action": stored}, action_layout=ROBOCASA_LEROBOT_SLICES)
        self.assertEqual(got, canon)
        # Without layout, keep stored order (unit-test / already-canonical rows).
        raw = from_demo_row({"action": stored})
        self.assertEqual(raw, stored)

    def test_resolve_tasks(self):
        from vagen.envs.robocasa.utils.tasks import ATOMIC_SEEN, resolve_tasks

        self.assertEqual(resolve_tasks(task="PickPlaceCounterToCabinet")[0], "PickPlaceCounterToCabinet")
        self.assertEqual(len(resolve_tasks(task_set="atomic_seen")), len(ATOMIC_SEEN))

    def test_starvla_action_contract(self):
        from vagen.envs.robocasa.utils.actions import (
            ACTION_DIM,
            CANONICAL_SLICES,
            GYM_SLICES,
            STARVLA_ACTION_DIM,
            STARVLA_ACTION_KEY_DIMS,
            STARVLA_ACTION_KEYS,
            STARVLA_STATE_DIM,
            STARVLA_STATE_KEY_DIMS,
            STARVLA_STATE_KEYS,
        )

        self.assertEqual(STARVLA_ACTION_DIM, 12)
        self.assertEqual(STARVLA_ACTION_DIM, ACTION_DIM)
        self.assertEqual(STARVLA_STATE_DIM, 16)
        self.assertEqual(sum(STARVLA_ACTION_KEY_DIMS[k] for k in STARVLA_ACTION_KEYS), 12)
        self.assertEqual(sum(STARVLA_STATE_KEY_DIMS[k] for k in STARVLA_STATE_KEYS), 16)
        offset = 0
        for key in STARVLA_ACTION_KEYS:
            name = key.split(".", 1)[1]
            dim = STARVLA_ACTION_KEY_DIMS[key]
            self.assertEqual(GYM_SLICES[key], (offset, offset + dim))
            self.assertEqual(CANONICAL_SLICES[name], (offset, offset + dim))
            offset += dim


if __name__ == "__main__":
    unittest.main(verbosity=2)
