"""Cambrian-S on the concat path (the model line, step 2 of 2026-10-24): the expansion of the compact <image> tokens
for the trainer (vagen/agent_loop/cambrian_concat.py) and the trainer's switch that picks its manager."""
import unittest
from pathlib import Path

import torch


def t(*rows, dtype=torch.long):
    return torch.tensor([list(rows)], dtype=dtype)


class ExpandSampleTest(unittest.TestCase):
    # a toy tokenizer: <image> = 7, its expanded stand-in 9, three positions per frame, pad 0
    KW = dict(pad_id=0, image_id=7, alias=9, per_image=3)

    def sample(self):
        # prompt [pad pad | a <image> b]; response [gen gen | user: <image> c | pad]
        return dict(prompt_ids=t(0, 0, 1, 7, 2), prompt_attention=t(0, 0, 1, 1, 1),
                    response_ids=t(3, 4, 7, 5, 0), response_attention=t(1, 1, 1, 1, 0),
                    response_mask=t(1, 1, 0, 0, 0), response_logprobs=t(-0.1, -0.2, 0.0, 0.0, 0.0, dtype=torch.float32))

    def test_expanded_form(self):
        from vagen.agent_loop.cambrian_concat import expand_sample

        e = expand_sample(**self.sample(), prompt_length=8, response_length=10, **self.KW)
        self.assertEqual(e["prompt_ids"].tolist(), [[0, 0, 0, 1, 9, 9, 9, 2]])
        self.assertEqual(e["response_ids"].tolist(), [[3, 4, 9, 9, 9, 5, 0, 0, 0, 0]])
        self.assertEqual(e["input_ids"].tolist(), [[0, 0, 0, 1, 9, 9, 9, 2, 3, 4, 9, 9, 9, 5, 0, 0, 0, 0]])
        self.assertEqual(e["attention_mask"].tolist(), [[0, 0, 0] + [1] * 11 + [0] * 4])
        self.assertEqual(e["position_ids"].tolist(), [[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10]])
        self.assertEqual(e["response_mask"].tolist(), [[1, 1] + [0] * 8])  # frames are never trained on
        self.assertEqual([round(x, 3) for x in e["response_logprobs"][0].tolist()], [-0.1, -0.2] + [0.0] * 8)
        self.assertEqual((e["frames_prompt"], e["frames_response"]), (1, 1))
        self.assertEqual(e["where"], [0, 1, 2, 5, 6])
        self.assertEqual(e["response_logprobs"].dtype, torch.float32)
        no_lp = expand_sample(**dict(self.sample(), response_logprobs=None), prompt_length=8, response_length=10,
                              **self.KW)
        self.assertIsNone(no_lp["response_logprobs"])

    def test_refuses_what_does_not_fit(self):
        from vagen.agent_loop.cambrian_concat import expand_sample

        with self.assertRaisesRegex(ValueError, "expanded prompt"):
            expand_sample(**self.sample(), prompt_length=4, response_length=10, **self.KW)
        with self.assertRaisesRegex(ValueError, "max_images"):
            expand_sample(**self.sample(), prompt_length=8, response_length=5, **self.KW)
        with self.assertRaisesRegex(ValueError, "marked as generated"):
            expand_sample(**dict(self.sample(), response_mask=t(1, 1, 1, 0, 0)), prompt_length=8, response_length=10,
                          **self.KW)

    def test_spans_move_with_their_tokens(self):
        from vagen.agent_loop.cambrian_concat import move_spans

        where = [0, 1, 2, 5, 6]
        self.assertEqual(move_spans({"spans": [[0, 2], [3, 4]]}, where), {"spans": [[0, 2], [5, 6]]})
        self.assertEqual(move_spans({"spans": [[3, 9]]}, where), {"spans": [[5, 6]]})  # clipped to the response
        self.assertEqual(move_spans({"spans": []}, where), {"spans": []})

    def test_worker_expands_an_agent_loop_output(self):
        from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, _InternalAgentLoopOutput

        from vagen.agent_loop import cambrian_concat as C

        class Proc:
            def preprocess_images(self, images):
                return {"pixel_values": torch.stack([torch.full((3, 2, 2), float(i)) for i in images])}

        worker = object.__new__(C.CambrianAgentLoopWorkerBase)
        worker.tokenizer = type("Tok", (), {"pad_token_id": 0})()
        worker.processor = Proc()
        s = ExpandSampleTest.sample(self)
        img, pad = C.IMAGE_TOKEN_ID, C.IMAGE_PAD_ID
        prompt_ids = t(0, 0, 1, img, 2)
        response_ids = t(3, 4, img, 5, 0)
        out = _InternalAgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response_ids, input_ids=torch.cat([prompt_ids, response_ids], 1),
            position_ids=torch.zeros(1, 10, dtype=torch.long), response_mask=s["response_mask"],
            attention_mask=torch.cat([s["prompt_attention"], s["response_attention"]], 1),
            response_logprobs=s["response_logprobs"], multi_modal_data={"image": [10, 11, 12]}, reward_score=1.0,
            num_turns=3, metrics=AgentLoopMetrics(),
            extra_fields={"invalid_turn_spans": {"spans": [[3, 4]]}, "image_data": [10, 11, 12]})
        # the toy sample at Cambrian's 756 positions per frame: the prompt just fits, the response has one pad
        n = C.TOKENS_PER_IMAGE
        worker.expanded_prompt_length, worker.expanded_response_length = 2 + n, 4 + n
        e = worker.expand_output(out)
        self.assertEqual(e.input_ids.shape[1], 2 + n + 4 + n)
        self.assertEqual(e.prompt_ids[0].tolist(), [1] + [pad] * n + [2])
        self.assertEqual(e.response_ids[0].tolist(), [3, 4] + [pad] * n + [5, 0])
        self.assertEqual(int((e.input_ids == img).sum()), 0)
        self.assertEqual(e.response_mask[0].tolist(), [1, 1] + [0] * (n + 2))
        # two frames in the sequence: the first two of the episode's three (a cut response drops the trailing ones)
        self.assertEqual(e.multi_modal_inputs["pixel_values"][:, 0, 0, 0].tolist(), [10.0, 11.0])
        self.assertEqual(e.extra_fields["invalid_turn_spans"], {"spans": [[2 + n, 3 + n]]})
        self.assertEqual((e.reward_score, e.num_turns), (1.0, 3))
        with self.assertRaisesRegex(ValueError, "frames"):
            worker.expand_output(out.model_copy(update={"multi_modal_data": {"image": [10]}}))


class SwitchesTest(unittest.TestCase):
    def test_expanded_lengths_need_the_alias_and_the_frames(self):
        from omegaconf import OmegaConf

        from vagen.agent_loop.cambrian_concat import expanded_lengths

        cfg = {"actor_rollout_ref": {"model": {"override_config": {"image_token_alias": 151655}},
                                     "rollout": {"prompt_length": 4096 + 756 * 12, "response_length": 4096}},
               "cambrian_concat": {"max_images": 12}}
        self.assertEqual(expanded_lengths(OmegaConf.create(cfg)),
                         {"max_images": 12, "prompt": 4096 + 756 * 12, "response": 4096 + 755 * 12})
        bad = OmegaConf.create(cfg)
        bad.actor_rollout_ref.model.override_config = {}
        with self.assertRaisesRegex(ValueError, "image_token_alias"):
            expanded_lengths(bad)
        bad = OmegaConf.create(cfg)
        del bad["cambrian_concat"]
        with self.assertRaisesRegex(ValueError, "max_images"):
            expanded_lengths(bad)

    def test_trainer_picks_the_named_manager(self):
        from omegaconf import OmegaConf
        from verl.experimental.agent_loop import AgentLoopManager

        from vagen.agent_loop.cambrian_concat import CambrianAgentLoopManager
        from vagen.ray_trainer import concat_agent_loop_manager

        self.assertIs(concat_agent_loop_manager(OmegaConf.create({"concat_multi_turn": True})), AgentLoopManager)
        self.assertIs(concat_agent_loop_manager(OmegaConf.create(
            {"agent_loop_manager": "vagen.agent_loop.cambrian_concat.CambrianAgentLoopManager"})),
            CambrianAgentLoopManager)
        self.assertTrue(issubclass(CambrianAgentLoopManager, AgentLoopManager))


CAMBRIAN = Path("/mnt/umm/shared_model/vagen-lite-models/hub/models--nyu-visionx--Cambrian-S-7B/snapshots/"
                "556e72fded09352f6aa4e9c100c9f39e6fe0fba7")


@unittest.skipUnless((CAMBRIAN / "tokenizer_config.json").is_file(), "needs the Cambrian-S-7B download")
class EvaluationRequestTest(unittest.TestCase):
    def test_the_evaluator_sends_what_the_agent_loop_builds(self):
        """data_gen/interiorgs_v10/evaluation.py with a Cambrian-S model: the chat through the model's template, one
        compact <image> per frame (as GymAgentLoop with CambrianProcessorWrapper), the frames in order."""
        from transformers import AutoTokenizer

        from data_gen.interiorgs_v10.evaluation import cambrian_request
        from vagen.models.cambrian_processor import CambrianProcessorWrapper

        wrap = CambrianProcessorWrapper(AutoTokenizer.from_pretrained(str(CAMBRIAN)))
        req = cambrian_request(wrap, "SYS", [("[Initial Observation]:\n<image>\nTask: x", "f0", "<action>a|</action>")],
                               "[Observation]:\n<image>\npose", "f1")
        self.assertEqual(req["multi_modal_data"], {"image": ["f0", "f1"]})
        ids = req["prompt_token_ids"]
        self.assertEqual(sum(1 for i in ids if i == 151665), 2)
        self.assertEqual(wrap.tokenizer.decode(ids),
                         "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\n[Initial Observation]:\n<image>\nTask: x"
                         "<|im_end|>\n<|im_start|>assistant\n<action>a|</action><|im_end|>\n<|im_start|>user\n"
                         "[Observation]:\n<image>\npose<|im_end|>\n<|im_start|>assistant\n")
        # the agent loop's compact prompt for the same first turn (system + first observation)
        agent = wrap(text=[wrap.apply_chat_template(
            [{"role": "system", "content": [{"type": "text", "text": "SYS"}]},
             {"role": "user", "content": [{"type": "text", "text": "[Initial Observation]:\n"}, {"type": "image"},
                                          {"type": "text", "text": "\nTask: x"}]}],
            tokenize=False, add_generation_prompt=True)], return_tensors=None)["input_ids"][0]
        first = cambrian_request(wrap, "SYS", [], "[Initial Observation]:\n<image>\nTask: x", "f0")
        self.assertEqual(first["prompt_token_ids"], agent)


if __name__ == "__main__":
    unittest.main()
