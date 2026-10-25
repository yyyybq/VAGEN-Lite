"""Cambrian-S's SFT in the bundled LLaMA-Factory (model_matrix/sft/lf_cambrian.py; the model line, step 2 of
2026-10-24): the registrations, the loss over the accumulated batch, the processor that saves nothing, and the
encoding against the RL path (model_matrix/sft/check_lf_cambrian.py)."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CAMBRIAN_SRC = "/mnt/umm/users/yinbaiqiao/cambrian-s"
CACHE = "/mnt/umm/shared_model/vagen-lite-models/hub"
CAMBRIAN = Path(CACHE) / "models--nyu-visionx--Cambrian-S-7B/snapshots/556e72fded09352f6aa4e9c100c9f39e6fe0fba7"
RECORDS = Path("/mnt/umm/users/yinbaiqiao/models_runs/sft_v10_newformat_20261024/sft_trajectories.jsonl")
HAVE = (importlib.util.find_spec("llamafactory") is not None and Path(CAMBRIAN_SRC).is_dir()
        and (CAMBRIAN / "config.json").is_file())


@unittest.skipUnless(HAVE, "needs LLaMA-Factory, the Cambrian source and the Cambrian-S-7B download")
class LfCambrianTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # register() sets CAMBRIAN_SRC in this process; the environment is given back afterwards, so later tests
        # (job_sco's launch scripts pass CAMBRIAN_SRC on) see the one they started with
        cls._env = mock.patch.dict(os.environ, {"DISABLE_VERSION_CHECK": "1"})
        cls._env.start()
        from model_matrix.sft import lf_cambrian

        lf_cambrian.register()
        lf_cambrian.register()  # idempotent
        cls.L = lf_cambrian

    @classmethod
    def tearDownClass(cls):
        cls._env.stop()

    def test_registrations(self):
        from llamafactory.data.template import TEMPLATES
        from llamafactory.model.model_utils.visual import COMPOSITE_MODELS, get_forbidden_modules
        from transformers import AutoConfig, AutoModelForCausalLM

        from vagen.models import cambrian_register as R

        t = TEMPLATES["cambrian_qwen"]
        self.assertEqual((t.mm_plugin.image_token, t.mm_plugin.video_token), ("<|image_pad|>", None))
        self.assertEqual(t.stop_words, ["<|im_end|>"])
        c = COMPOSITE_MODELS["cambrian_qwen"]
        self.assertEqual((c.projector_key, c.vision_model_keys), ("model.mm_projector", ["model.vision_tower_aux_list"]))
        config = AutoConfig.from_pretrained(str(CAMBRIAN))
        args = mock.Mock(freeze_vision_tower=True, freeze_multi_modal_projector=False, freeze_language_model=False)
        self.assertEqual(get_forbidden_modules(config, args), {"model.vision_tower_aux_list"})  # the recipe's rule
        cls = AutoModelForCausalLM._model_mapping[type(config)]
        self.assertEqual(cls.__name__, "CambrianForCausalLMAdapter")  # the saved architecture the vLLM plugin knows
        self.assertTrue(issubclass(cls, R.CambrianForCausalLMAdapter) and cls is not R.CambrianForCausalLMAdapter)

    def test_loss_over_the_accumulated_batch(self):
        """The Trainer passes num_items_in_batch (the label tokens of the whole accumulated batch); the loss is
        Transformers' causal-LM loss with it, as Qwen2.5-VL's own forward computes it."""
        import torch
        from transformers.loss.loss_utils import ForCausalLMLoss
        from transformers.modeling_outputs import CausalLMOutputWithPast

        from vagen.models import cambrian_register as R

        cls = self.L.model_class()
        logits = torch.randn(1, 6, 11)
        labels = torch.tensor([[-100, -100, 3, 4, -100, 7]])
        seen = {}

        def parent_forward(self, input_ids=None, labels=None, **kw):
            seen.update(kw, labels=labels)
            return CausalLMOutputWithPast(logits=logits)

        model = object.__new__(cls)
        model.__dict__["config"] = mock.Mock(vocab_size=11)
        with mock.patch.object(R.CambrianForCausalLMAdapter, "forward", parent_forward):
            out = cls.forward(model, input_ids=torch.zeros(1, 6, dtype=torch.long), labels=labels,
                              num_items_in_batch=torch.tensor(9), pixel_values="pv")
            self.assertTrue(torch.allclose(out.loss, ForCausalLMLoss(logits, labels, 11, num_items_in_batch=torch.tensor(9))))
            self.assertEqual((seen["labels"], seen["pixel_values"], seen["return_dict"]), (None, "pv", True))
            self.assertNotIn("num_items_in_batch", seen)
            mean = cls.forward(model, input_ids=torch.zeros(1, 6, dtype=torch.long), labels=labels).loss
            self.assertTrue(torch.allclose(mean, ForCausalLMLoss(logits, labels, 11)))  # no count: the mean
            self.assertIsNone(cls.forward(model, input_ids=torch.zeros(1, 6, dtype=torch.long)).loss)

    def test_register_needs_no_cambrian_src(self):
        """An SFT job's environment may lack CAMBRIAN_SRC; register() sets this cluster's source path first."""
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[2]
        env = {k: v for k, v in os.environ.items() if k != "CAMBRIAN_SRC"}
        env.update(PYTHONPATH=str(root), DISABLE_VERSION_CHECK="1", HF_HUB_OFFLINE="1")
        out = subprocess.run([sys.executable, "-c", "from model_matrix.sft.lf_cambrian import register; register(); "
                              "import os; print(os.environ['CAMBRIAN_SRC'])"], capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], CAMBRIAN_SRC)

    def test_processor_saves_nothing(self):
        with mock.patch("vagen.models.cambrian_processor.SIGLIP_CACHE", CACHE):
            proc = self.L.CambrianSFTProcessor.from_pretrained(str(CAMBRIAN), use_fast=True)
        self.assertEqual(type(proc.image_processor).__name__, "SiglipImageProcessor")
        self.assertEqual(proc.image_processor.size, {"height": 384, "width": 384})
        self.assertNotIn("<image>", proc.tokenizer.get_vocab())  # the SFT text uses <|image_pad|>
        with tempfile.TemporaryDirectory() as tmp:
            proc.save_pretrained(tmp)
            self.assertEqual(os.listdir(tmp), [])  # no preprocessor file for verl or vLLM to take as the processor

    @unittest.skipUnless(RECORDS.is_file(), "needs the new-format SFT conversations")
    def test_encoding_matches_the_rl_path(self):
        from model_matrix.sft.check_lf_cambrian import check
        from model_matrix.sft.check_lf_qwen3vl import load_records

        with mock.patch("vagen.models.cambrian_processor.SIGLIP_CACHE", CACHE):
            r = check(str(CAMBRIAN), load_records(str(RECORDS), ["sft_v10_c269344785dd21218e7a415e"])[0])
        self.assertTrue(r["ok"], {k: r[k] for k in r if k != "supervised_spans"})
        self.assertEqual((r["frames"], r["image_tokens_per_frame_lf"]), (2, [756, 756]))


if __name__ == "__main__":
    unittest.main()
