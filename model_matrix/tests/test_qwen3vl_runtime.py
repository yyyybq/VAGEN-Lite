"""Qwen3-VL checks that need transformers >= 4.57 (the runtime venv, .venv-qwen3vl-runtime) and the model's processor
files; skipped elsewhere (the vagen-lite env has 4.56.1).  CPU only.

    .venv-qwen3vl-runtime/bin/python -m pytest -q -p no:cacheprovider model_matrix/tests/test_qwen3vl_runtime.py
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

QWEN3VL = os.environ.get("QWEN3VL_MODEL", "/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
                         "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")


def _runtime() -> bool:
    try:
        import transformers
        from packaging.version import Version

        return Version(transformers.__version__) >= Version("4.57") and (Path(QWEN3VL) / "tokenizer.json").is_file()
    except Exception:
        return False


def _frames(n, tmp):
    from PIL import Image

    paths = []
    for k in range(n):
        p = Path(tmp) / f"f{k}.png"
        Image.new("RGB", (256, 256), (20 * k, 40, 200 - 20 * k)).save(p)
        paths.append(str(p))
    return paths


def _conversation(n_images, tmp):
    paths = _frames(n_images, tmp)
    conv = [{"role": "system", "content": "You are a spatial navigation agent.\n"}]
    for k in range(n_images):
        conv.append({"role": "user", "content": f"[{'Initial ' if k == 0 else ''}Observation]:\n<image>\nTurn {k}.\n"})
        conv.append({"role": "assistant", "content": "<action>move_forward|turn_left|</action>"})
    return {"id": "synthetic", "conversations": conv, "conversation_image_paths": paths}


@unittest.skipUnless(_runtime(), "needs transformers >= 4.57 and the Qwen3-VL processor files")
class Qwen3VLRuntimeTest(unittest.TestCase):
    def test_concat_position_ids_qwen2_vl_function_equals_qwen3_vl_function(self):
        """verl's agent loop computes the concat path's 3-D position ids with the Qwen2-VL function for any processor
        whose image processor is a Qwen2VLImageProcessor (Qwen3-VL's is Qwen2VLImageProcessorFast).  For images (one
        frame each: no time axis) it must give exactly Qwen3-VL's own positions, on a whole multi-turn episode."""
        import torch
        from model_matrix.sft.check_lf_qwen3vl import hf_encode
        from verl.models.transformers.qwen2_vl import get_rope_index as rope2
        from verl.models.transformers.qwen3_vl import get_rope_index as rope3

        with tempfile.TemporaryDirectory() as tmp:
            ids, grids, processor, _ = hf_encode(QWEN3VL, _conversation(12, tmp))
            self.assertIn("Qwen2VLImageProcessor", type(processor.image_processor).__name__)
            input_ids = torch.tensor(ids)
            mask = torch.ones_like(input_ids)
            mask[:3] = 0  # left padding, as a prompt batch has it
            thw = torch.tensor(grids)
            a = rope2(processor, input_ids=input_ids, image_grid_thw=thw, attention_mask=mask)
            b = rope3(processor, input_ids=input_ids, image_grid_thw=thw, attention_mask=mask)
            self.assertEqual(a.shape, (3, len(ids)))
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(sum(1 for t in ids if t == processor.image_token_id), 12 * 64)

    def test_lf_template_matches_qwen3vl_chat_template(self):
        """LLaMA-Factory with the registered template encodes like Qwen3-VL's processor, supervises only the answers,
        and writes the processor's number of image placeholders (model_matrix/sft/check_lf_qwen3vl.py)."""
        from model_matrix.sft.check_lf_qwen3vl import check

        with tempfile.TemporaryDirectory() as tmp:
            result = check(QWEN3VL, _conversation(3, tmp))
        self.assertTrue(result["ids_identical"], json.dumps(result.get("context")))
        self.assertEqual(result["image_tokens_per_image_lf"], [64, 64, 64])
        self.assertTrue(result["supervised_text_is_answers"] and result["labels_equal_ids_where_supervised"])
        self.assertEqual(len(result["supervised_spans"]), 3)  # one span per answer, nothing from the user turns
        self.assertFalse(result["think_in_template"])

    def test_registration_is_idempotent_and_freezes_the_vision_encoder(self):
        from llamafactory.data.template import TEMPLATES
        from llamafactory.model.model_utils.visual import COMPOSITE_MODELS
        from model_matrix.sft.lf_qwen3vl import TEMPLATE, register

        register()
        register()
        self.assertIn(TEMPLATE, TEMPLATES)
        cm = COMPOSITE_MODELS["qwen3_vl"]
        self.assertEqual(cm.projector_key, "visual.merger")
        self.assertEqual(cm.vision_model_keys, ["visual.pos_embed", "visual.patch_embed", "visual.blocks",
                                                "visual.deepstack_merger_list"])
        self.assertEqual(TEMPLATES[TEMPLATE].default_system, "")


if __name__ == "__main__":
    unittest.main()
