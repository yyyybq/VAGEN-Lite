"""Round 12 freeze (the user's 3.1.1 of 2026-10-22): the prompt's format line shows real action names instead of the
placeholders action1|action2|... that RL runs copied into their answers.  The default ("placeholder") keeps every
prompt byte for byte; the SFT export, the RL env, rl-config, the evaluation and the chain pass the same setting."""

import unittest
from argparse import Namespace

from data_gen.interiorgs_v10 import evaluation as E
from data_gen.interiorgs_v10 import release as R
from data_gen.interiorgs_v10 import train_chain as TC
from data_gen.interiorgs_v10 import training_prep as TP
from vagen.envs.active_spatial import prompt as P
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

S0 = dict(step_translation=0.3, step_rotation_deg=20.0, action_space="strafe", enable_explicit_done=False,
          max_actions_per_step=5, format_reward=0.0, invalid_format_penalty=-0.1, success_reward=5.0,
          reward_text="actual")
LEGACY_LINE = "<action>action1|action2|...|</action>"


class FormatExampleTest(unittest.TestCase):
    def test_default_is_the_old_text(self):
        for name, fn in P.format_prompt.items():
            for first in (True, False):
                default = fn(max_actions_per_step=5, action_sep="|", add_example=first)
                self.assertEqual(default, fn(max_actions_per_step=5, action_sep="|", add_example=first,
                                             format_example="placeholder"), name)
                self.assertIn("\n" + LEGACY_LINE + "\n", default, name)
        for fmt in P.FORMAT_CONFIGS:
            default = P.system_prompt(format=fmt, task_type="fov_inclusion", **S0)
            self.assertEqual(default, P.system_prompt(format=fmt, task_type="fov_inclusion",
                                                      format_example="placeholder", **S0))
            self.assertIn("Response format: " + P.FORMAT_CONFIGS[fmt]["format"] + "\n", default)

    def test_action_names(self):
        line = "<action>move_forward|turn_left|</action>"
        for name, fn in P.format_prompt.items():
            text = fn(max_actions_per_step=5, action_sep="|", add_example=True, format_example="action_names")
            self.assertIn("\n" + line + "\n", text, name)
            self.assertNotIn("action1", text)
            self.assertNotIn("...", text)
            # only the format line changes
            old = fn(max_actions_per_step=5, action_sep="|", add_example=True)
            self.assertEqual(text.replace(line, LEGACY_LINE), old)
        self.assertEqual(P.action_format_line("|", 1, "action_names"), "<action>move_forward|</action>")
        no_think = P.system_prompt(format="no_think", task_type="equidistance", format_example="action_names", **S0)
        self.assertIn("Response format: " + line + "\n", no_think)
        self.assertEqual(no_think.replace("Response format: " + line, "Response format: <action>...</action>"),
                         P.system_prompt(format="no_think", task_type="equidistance", **S0))
        free = P.system_prompt(format="free_think", task_type="equidistance", format_example="action_names", **S0)
        self.assertIn("Response format: <think>...</think>" + line + "\n", free)
        with self.assertRaises(ValueError):
            P.action_format_line("|", 5, "names")

    def test_env_and_sft_export_agree(self):
        cfg = ActiveSpatialEnvConfig()
        self.assertEqual(cfg.prompt_format_example, "placeholder")
        self.assertEqual(cfg.system_prompt_arguments("fov_inclusion")["format_example"], "placeholder")
        export = {"prompt_format": "no_think", "max_actions_per_turn": 5, "system_prompt": dict(S0)}
        old = {"sft": {"export": dict(export)}}
        new = {"sft": {"export": dict(export, format_example="action_names")}}
        self.assertEqual(R.format_example(old), "placeholder")
        self.assertEqual(R._format_block(old, True), P.format_prompt["no_think"](max_actions_per_step=5, action_sep="|",
                                                                                 add_example=True))
        self.assertEqual(R._format_block(new, False), P.format_prompt["no_think"](
            max_actions_per_step=5, action_sep="|", add_example=False, format_example="action_names"))
        env = dict(prompt_format="no_think", step_translation=0.3, step_rotation_deg=20.0, action_space="strafe",
                   enable_explicit_done=False, max_actions_per_step=5, format_reward=0.0, invalid_format_penalty=-0.1,
                   success_reward=5.0, prompt_reward_text="actual")
        for t in ("projective_relations", "fov_inclusion"):
            self.assertEqual(TP.env_system_prompt(env, t), R.protocol_system_prompt(t, old))
            self.assertEqual(TP.env_system_prompt(dict(env, prompt_format_example="action_names"), t),
                             R.protocol_system_prompt(t, new))
            self.assertNotEqual(TP.env_system_prompt(env, t), R.protocol_system_prompt(t, new))
        self.assertEqual(TP.prompt_mismatches(dict(env, prompt_format_example="action_names"), ["fov_inclusion"], old),
                         ["fov_inclusion"])
        self.assertEqual(TP.prompt_mismatches(dict(env, prompt_format_example="action_names"), ["fov_inclusion"],
                                              TP.with_format_example(old, "action_names")), [])
        self.assertNotIn("format_example", old["sft"]["export"])  # with_format_example copies

    def test_evaluation_and_chain_pass_the_setting(self):
        self.assertEqual(E.switch_overrides(Namespace(format_example="action_names")),
                         {"prompt_format_example": "action_names"})
        self.assertEqual(E.switch_overrides(Namespace(format_example=None)), {})
        self.assertEqual(TC.eval_options_for(["--samples", "4"], "action_names"),
                         ["--samples", "4", "--format-example", "action_names"])
        self.assertEqual(TC.eval_options_for(["--samples", "4"], "placeholder"), ["--samples", "4"])
        self.assertEqual(TC.eval_options_for([], None), [])
        self.assertEqual(TC.eval_options_for(["--format-example", "action_names"], "action_names"),
                         ["--format-example", "action_names"])
        with self.assertRaises(TC.ChainError):
            TC.eval_options_for(["--format-example", "placeholder"], "action_names")


if __name__ == "__main__":
    unittest.main()
