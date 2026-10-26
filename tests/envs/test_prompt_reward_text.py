"""System prompt reward text (round 9, next data version): "legacy" stays the text R1 and v10 trained with; "actual"
lists the reward terms the env applies (S0 has no progress reward, which the legacy lines promise)."""

import unittest

from vagen.envs.active_spatial import prompt as P

S0 = dict(step_translation=0.3, step_rotation_deg=20.0, action_space="strafe", enable_explicit_done=False,
          max_actions_per_step=5, format_reward=0.0, invalid_format_penalty=-0.1, success_reward=5.0)


def rewards(text):
    return text.split("Rewards:\n", 1)[1].split("\n\nHints:", 1)[0]


class RewardTextTest(unittest.TestCase):
    def test_legacy_is_the_default_and_unchanged(self):
        for task in ("projective_relations", "fov_inclusion"):
            default = P.system_prompt(format="no_think", task_type=task, **S0)
            self.assertEqual(default, P.system_prompt(format="no_think", task_type=task, reward_text="legacy", **S0))
        self.assertEqual(rewards(P.system_prompt(format="no_think", task_type="fov_inclusion", **S0)),
                         "- Valid format is required but gives no positive reward; invalid format: -0.1\n"
                         "- Progress toward target pose: continuous reward based on distance and orientation improvement\n"
                         "- Reaching target pose: +5")
        self.assertIn("- Gate-aligned projective progress",
                      rewards(P.system_prompt(format="no_think", task_type="projective_relations", **S0)))

    def test_actual_lists_the_s0_terms(self):
        text = rewards(P.system_prompt(format="no_think", task_type="projective_relations", reward_text="actual", **S0))
        self.assertEqual(text.splitlines(), [
            "- Invalid format (no action can be read): -0.1; 3 in a row end the episode",
            "- Each response (step): -0.01",
            "- Each blocked move (the camera stays where it was): -0.15; 3 blocked moves in a row end the episode (-0.5 more)",
            "- Nearly uniform view (e.g. facing a wall): -0.2; 3 in a row end the episode",
            "- Reaching the requested view: +5, and the episode ends",
            "- There is no reward for partial progress"])
        free = rewards(P.system_prompt(format="no_think", task_type="fov_inclusion", reward_text="actual",
                                       max_consecutive_collisions=0, **S0))
        self.assertIn("- Each blocked move (the camera stays where it was): -0.15\n", free)  # switch 3: no ending

    def test_sft_export_follows_the_setting(self):
        from data_gen.interiorgs_v10 import release as R

        policy = {"sft": {"export": {"prompt_format": "no_think", "system_prompt": dict(S0, reward_text="actual")}}}
        self.assertEqual(R.protocol_system_prompt("equidistance", policy),
                         P.system_prompt(format="no_think", task_type="equidistance", reward_text="actual", **S0))


if __name__ == "__main__":
    unittest.main()
