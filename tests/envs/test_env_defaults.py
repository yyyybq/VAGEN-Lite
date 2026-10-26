"""Env defaults for v10 training (user decision 2026-10-13): occlusion_alignment hints off by default."""

import unittest
from types import SimpleNamespace

from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

OA_ROW = {
    "task_type": "occlusion_alignment",
    "task_description": "Move so that the chair is hidden behind the wardrobe from your viewpoint.",
    "target_object": {"objects": [{"id": "1", "label": "chair", "ref": "the chair"},
                                  {"id": "2", "label": "wardrobe", "ref": "the wardrobe"}]},
    "target_region": {"type": "occlusion_shadow",
                      "params": {"object_a_center": [1.0, 2.0, 0.5], "object_b_center": [2.0, 2.0, 1.0]}},
}


class OcclusionHintsTest(unittest.TestCase):
    def test_hints_default_off(self):
        cfg = ActiveSpatialEnvConfig()
        self.assertFalse(cfg.enable_occlusion_task_guidance)
        self.assertFalse(cfg.enable_occlusion_occluder_hint)

    def test_prompt_is_the_task_description_only(self):
        stub = SimpleNamespace(config=ActiveSpatialEnvConfig())
        prompt = ActiveSpatialEnv._build_task_prompt(stub, OA_ROW)
        self.assertEqual(prompt, OA_ROW["task_description"])
        self.assertNotIn("Occluder hint", prompt)


if __name__ == "__main__":
    unittest.main()
