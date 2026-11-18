"""Qwen3-VL-8B (model_type qwen3_vl, deepstack) on the q38 stack, CPU only (DESIGN_ASYNC 5.3, P8-2 / P8-3 prerequisites).

The 8B parity runs start from the 8B C0 SFT checkpoint (transformers 4.57.6 format) in the q38 environment
(transformers 5.10.4, the q38 verl tree).  Checked here without GPU and without loading the 8B weights:

- the checkpoint's config loads in transformers 5 (rope_scaling of 4.57 -> rope_parameters), and the parameter names
  of a meta-device Qwen3VLForConditionalGeneration equal the checkpoint's safetensors index (the FSDP engine loads the
  checkpoint and naive-syncs these names into vLLM);
- verl's hf_processor binds Qwen3VLModel.get_rope_index to the 8B processor, and AgentLoopBase._compute_position_ids
  gives the (1, 4, L) position ids the agent loop stores (text row + 3 M-RoPE rows);
- verl's qwen3_vl monkey patch (forward_with_normal_backend, qwen3_vl_base_forward, deepstack) runs on a tiny random
  Qwen3-VL built from the 8B config, and the engine's remove-padding input (two packed sequences, attention_mask None,
  position_ids (4, 1, total)) gives the same logits as each sequence alone, with attn_implementation eager (the 8B RL
  setting, decision 20 of the model line) and with sdpa; eager and sdpa agree with each other;
- vLLM 0.28 registers Qwen3VLForConditionalGeneration and maps the HF names (model.visual., model.language_model.,
  lm_head.) to its own modules.

Skipped where the 8B checkpoint is not readable (Q38_TEST_MODEL_8B overrides the path).
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

MODEL_8B = os.environ.get(
    "Q38_TEST_MODEL_8B",
    "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232",
)
HAVE_8B = os.path.isfile(os.path.join(MODEL_8B, "config.json"))

# a tiny text and vision stack with the 8B's token ids, vocabulary and patch geometry
TINY_TEXT = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 16}
TINY_MROPE_SECTION = [3, 3, 2]  # sums to head_dim / 2 like the 8B's [24, 20, 20] for head_dim 128
TINY_VISION = {"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
               "deepstack_visual_indexes": [0, 1]}


def _tiny_config(attn_implementation: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_8B, attn_implementation=attn_implementation)
    for k, v in TINY_TEXT.items():
        setattr(cfg.text_config, k, v)
    rope = getattr(cfg.text_config, "rope_parameters", None) or getattr(cfg.text_config, "rope_scaling", None)
    rope["mrope_section"] = list(TINY_MROPE_SECTION)
    for k, v in TINY_VISION.items():
        setattr(cfg.vision_config, k, v)
    cfg.hidden_size = TINY_TEXT["hidden_size"]
    return cfg


def _processor():
    from verl.utils.tokenizer import hf_processor

    return hf_processor(MODEL_8B)


def _sample(processor, text: str, seed: int):
    """One prompt with one 256x256 frame through the 8B's own chat template, as the agent loop renders a turn."""
    import numpy as np
    from PIL import Image

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    input_ids = inputs.pop("input_ids")
    attention_mask = inputs.pop("attention_mask")
    mm = dict(inputs)
    position_ids = AgentLoopWorker._compute_position_ids(SimpleNamespace(processor=processor), input_ids,
                                                      attention_mask, dict(mm))
    return input_ids, position_ids, mm


@unittest.skipUnless(HAVE_8B, "8B C0 checkpoint not readable")
class Qwen3VLStackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_checkpoint_names_match_transformers5_model(self):
        from transformers import AutoConfig, Qwen3VLForConditionalGeneration

        cfg = AutoConfig.from_pretrained(MODEL_8B)
        self.assertEqual(cfg.model_type, "qwen3_vl")
        self.assertEqual(list(cfg.vision_config.deepstack_visual_indexes), [8, 16, 24])
        rope = getattr(cfg.text_config, "rope_parameters", None) or getattr(cfg.text_config, "rope_scaling", None)
        self.assertEqual(list(rope["mrope_section"]), [24, 20, 20])
        with torch.device("meta"):
            model = Qwen3VLForConditionalGeneration(cfg)
        names = set(model.state_dict().keys())
        index = json.loads(Path(MODEL_8B, "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(sorted(names - set(index)), [])
        self.assertEqual(sorted(set(index) - names), [])

    def test_processor_binding_and_position_ids(self):
        processor = _processor()
        self.assertEqual(processor.__class__.__name__, "Qwen3VLProcessor")
        self.assertTrue(hasattr(processor, "get_rope_index"))
        input_ids, position_ids, mm = _sample(processor, "Where is the chair?", 0)
        n = input_ids.shape[1]
        self.assertEqual(tuple(position_ids.shape), (1, 4, n))
        self.assertEqual(int((input_ids == 151655).sum()), 64)  # 256 px / (16 x 2) = 8 -> 64 image tokens
        self.assertEqual(mm["image_grid_thw"].tolist(), [[1, 16, 16]])
        # text row: 0..n-1; the M-RoPE rows advance less than the text row across the image
        self.assertEqual(position_ids[0, 0].tolist(), list(range(n)))
        self.assertLess(int(position_ids[0, 1].max()), n - 1)

    def _model_pair(self):
        from transformers import Qwen3VLForConditionalGeneration

        from verl.models.transformers.monkey_patch import apply_monkey_patch

        torch.manual_seed(0)
        eager = Qwen3VLForConditionalGeneration._from_config(_tiny_config("eager"), attn_implementation="eager")
        sdpa = Qwen3VLForConditionalGeneration._from_config(_tiny_config("sdpa"), attn_implementation="sdpa")
        eager, sdpa = eager.float(), sdpa.float()  # the config says bfloat16; compare in fp32 as the actor trains
        sdpa.load_state_dict(eager.state_dict())
        for m in (eager, sdpa):
            m.eval()
            apply_monkey_patch(model=m, ulysses_sp_size=1, use_remove_padding=True)
        self.assertEqual(eager.config.text_config._attn_implementation, "eager")
        self.assertEqual(sdpa.config.text_config._attn_implementation, "sdpa")
        return eager, sdpa

    @staticmethod
    def _logits(model, input_ids, position_ids, mm):
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=None, position_ids=position_ids, use_cache=False, **mm)
        return out.logits.float()

    def test_packed_forward_equals_unpacked_eager_and_sdpa(self):
        processor = _processor()
        a = _sample(processor, "Where is the chair?", 1)
        b = _sample(processor, "Turn left until the lamp is in view, then move forward.", 2)
        eager, sdpa = self._model_pair()

        # the engine's remove-padding input (transformer_impl.py prepare_model_inputs): (1, total) ids,
        # position ids (4, 1, total), attention_mask None, the images of all sequences concatenated
        packed_ids = torch.cat([a[0], b[0]], dim=1)
        packed_pos = torch.cat([a[1], b[1]], dim=2).transpose(0, 1)  # (1, 4, L) -> (4, 1, L)
        packed_mm = {"pixel_values": torch.cat([a[2]["pixel_values"], b[2]["pixel_values"]], dim=0),
                     "image_grid_thw": torch.cat([a[2]["image_grid_thw"], b[2]["image_grid_thw"]], dim=0)}
        la = a[0].shape[1]

        results = {}
        for name, model in (("eager", eager), ("sdpa", sdpa)):
            alone_a = self._logits(model, a[0], a[1].transpose(0, 1), a[2])
            alone_b = self._logits(model, b[0], b[1].transpose(0, 1), b[2])
            packed = self._logits(model, packed_ids, packed_pos, packed_mm)
            err_a = (packed[:, :la] - alone_a).abs().max().item()
            err_b = (packed[:, la:] - alone_b).abs().max().item()
            print(f"{name}: packed vs alone max |diff| {err_a:.3g} / {err_b:.3g}")
            self.assertLess(err_a, 1e-4, f"{name}: packed first sequence differs from alone ({err_a})")
            self.assertLess(err_b, 1e-4, f"{name}: packed second sequence attends the first ({err_b})")
            results[name] = packed
        err = (results["eager"] - results["sdpa"]).abs().max().item()
        print(f"eager vs sdpa max |diff| {err:.3g}")
        self.assertLess(err, 1e-4, f"eager vs sdpa ({err})")

    def test_vllm_registers_qwen3_vl(self):
        try:
            from vllm.model_executor.models.registry import ModelRegistry
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"vllm not importable here: {type(e).__name__}: {e}")
        self.assertIn("Qwen3VLForConditionalGeneration", ModelRegistry.get_supported_archs())
        from vllm.model_executor.models import qwen3_vl

        mapper = qwen3_vl.Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
        mapped = mapper.apply_list(["model.visual.blocks.0.attn.qkv.weight",
                                    "model.language_model.layers.0.self_attn.q_proj.weight", "lm_head.weight"])
        self.assertEqual(mapped, ["visual.blocks.0.attn.qkv.weight",
                                  "language_model.model.layers.0.self_attn.q_proj.weight",
                                  "language_model.lm_head.weight"])


if __name__ == "__main__":
    unittest.main()
