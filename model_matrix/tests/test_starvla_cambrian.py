"""Cambrian-S as StarVLA's backbone (model_matrix/downstream/starvla_cambrian.py, cambrian_entry.py; the model line,
step 2 of 2026-10-24): the views' positions and features, the prompt, the batch, the embedding, the registration and
the entry, on the CPU (StarVLA itself faked where it is needed)."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = "/mnt/umm/shared_model/vagen-lite-models/hub"
CAMBRIAN = Path(CACHE) / "models--nyu-visionx--Cambrian-S-7B/snapshots/556e72fded09352f6aa4e9c100c9f39e6fe0fba7"


class ViewFeaturesTest(unittest.TestCase):
    def test_miv_and_si(self):
        import torch.nn.functional as F

        from model_matrix.downstream.starvla_cambrian import tokens_per_view, view_features

        self.assertEqual((tokens_per_view("miv"), tokens_per_view("si")), (72, 756))
        feats = torch.randn(2, 729, 3)
        newline = torch.tensor([7.0, 8.0, 9.0])
        out = view_features(feats, newline, 8)
        self.assertEqual(tuple(out.shape), (2, 72, 3))
        grid = F.interpolate(feats.unflatten(1, (27, 27)).permute(0, 3, 1, 2), size=(8, 8), mode="bilinear",
                             align_corners=False)
        rows = out.reshape(2, 8, 9, 3)
        self.assertTrue(torch.equal(rows[:, :, 8], newline.expand(2, 8, 3)))  # the newline closes every row
        self.assertTrue(torch.allclose(rows[:, :, :8], grid.permute(0, 2, 3, 1)))
        si = view_features(feats, newline, 27).reshape(2, 27, 28, 3)
        self.assertTrue(torch.equal(si[:, :, :27].reshape(2, 729, 3), feats))  # no resizing at 27


def stub_interface(form="miv", tokenizer=None, image_processor=None, cot=None):
    from omegaconf import OmegaConf

    from model_matrix.downstream import starvla_cambrian as C

    it = object.__new__(C.CambrianVLInterface)
    torch.nn.Module.__init__(it)
    it.form = form
    it.processor = SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)
    it.config = OmegaConf.create({"datasets": {"vla_data": {"CoT_prompt": cot} if cot else {}}})
    return it


@unittest.skipUnless((CAMBRIAN / "tokenizer_config.json").is_file(), "needs the Cambrian-S-7B download")
class PromptAndBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoImageProcessor, AutoTokenizer

        cls.tok = AutoTokenizer.from_pretrained(str(CAMBRIAN))
        cls.tok.padding_side = "left"
        cls.ip = AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384", cache_dir=CACHE,
                                                    local_files_only=True)

    def test_prompt_is_cambrians_chat_with_the_views_behind_one_image(self):
        from model_matrix.downstream.starvla_cambrian import IMAGE_TOKEN_INDEX

        it = stub_interface(tokenizer=self.tok)
        ids = it.prompt_ids("open the drawer <action>🔍🔍<action>.", 3)
        self.assertEqual(sum(1 for t in ids if t == IMAGE_TOKEN_INDEX), 3 * 72)
        self.assertEqual(ids.count(146663), 2)  # the action token, as for Qwen2.5-VL
        first = ids.index(IMAGE_TOKEN_INDEX)
        self.assertEqual(self.tok.decode(ids[:first]),
                         "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n")
        self.assertEqual(self.tok.decode(ids[first + 216:]),
                         "\nopen the drawer <action>🔍🔍<action>.<|im_end|>\n<|im_start|>assistant\n")
        si = stub_interface(form="si", tokenizer=self.tok).prompt_ids("x", 2)
        self.assertEqual(sum(1 for t in si if t == IMAGE_TOKEN_INDEX), 2 * 756)

    def test_batch(self):
        from PIL import Image

        from model_matrix.downstream.starvla_cambrian import IMAGE_TOKEN_INDEX

        it = stub_interface(tokenizer=self.tok, image_processor=self.ip, cot="Task: {instruction}")
        it.model = torch.nn.Linear(1, 1)  # stands for the weights' device
        views = [Image.new("RGB", (224, 224), (i * 40, 0, 0)) for i in range(5)]
        b = it.build_qwenvl_inputs([views[:3], views[3:]], ["a", "bb cc"])
        ids, att = b["input_ids"], b["attention_mask"]
        self.assertEqual(ids.shape, att.shape)
        self.assertEqual([int((r == IMAGE_TOKEN_INDEX).sum()) for r in ids], [216, 144])
        self.assertEqual(b["views_per_sample"].tolist(), [3, 2])
        self.assertEqual(tuple(b["pixel_values"].shape), (5, 3, 384, 384))
        for r, a in zip(ids, att):  # left padding
            n = int(a.sum())
            self.assertTrue(bool((a[-n:] == 1).all()) and bool((r[:-n] == self.tok.pad_token_id).all()))
        self.assertIn("Task: bb cc", self.tok.decode([t for t in ids[1].tolist() if t >= 0]))
        with self.assertRaises(NotImplementedError):
            it.build_qwenvl_inputs([views[:1]], ["a"], solutions=["x"])


class EmbedTest(unittest.TestCase):
    def test_views_fill_the_image_positions_in_order(self):
        from model_matrix.downstream.starvla_cambrian import IMAGE_TOKEN_INDEX

        d = 4
        inner = SimpleNamespace(embed_tokens=torch.nn.Embedding(10, d),
                                mm_projector=torch.nn.Sequential(torch.nn.Linear(2, d)),
                                image_newline=torch.nn.Parameter(torch.full((d,), 5.0)))
        feats = torch.randn(3, 729, 2)
        model = SimpleNamespace(get_model=lambda: inner, encode_images=lambda xs: [feats])
        it = stub_interface()
        it.__dict__["model"] = model
        ids = torch.tensor([[1, 2] + [IMAGE_TOKEN_INDEX] * 144 + [3], [0] * 73 + [4] + [IMAGE_TOKEN_INDEX] * 72 + [5]])
        out = it.embed(ids, torch.zeros(3, 3, 384, 384))
        from model_matrix.downstream.starvla_cambrian import view_features

        want = view_features(inner.mm_projector(feats), inner.image_newline, 8).reshape(-1, d)
        self.assertTrue(torch.allclose(out[0, 2:146], want[:144]))
        self.assertTrue(torch.allclose(out[1, 74:146], want[144:]))
        self.assertTrue(torch.allclose(out[0, :2], inner.embed_tokens(torch.tensor([1, 2]))))
        with self.assertRaisesRegex(ValueError, "image positions"):
            it.embed(ids[:1], torch.zeros(3, 3, 384, 384))


@unittest.skipUnless(Path("/mnt/umm/users/yinbaiqiao/cambrian-s").is_dir(), "needs the Cambrian source")
class LanguageModelBatchTest(unittest.TestCase):
    """A small randomly initialised Cambrian language model: Qwen2Model.forward equals the Cambrian source's own loop
    on one sample, and a left-padded batch of two equals the two samples run alone."""

    def test_batch_equals_single_samples(self):
        os.environ.setdefault("CAMBRIAN_SRC", "/mnt/umm/users/yinbaiqiao/cambrian-s")
        from vagen.models import cambrian_register as R

        from model_matrix.downstream.starvla_cambrian import language_model_forward

        torch.manual_seed(0)
        cfg = R.CambrianQwenConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=50, max_position_embeddings=64,
                                   connector_only=True, use_cache=False)
        cfg._attn_implementation = "eager"
        from cambrian.model.language_model.cambrian_qwen2 import CambrianQwenModel

        inner = CambrianQwenModel(cfg).eval()
        inner._attn_implementation = "eager"  # as the adapter's constructor sets it
        R.CambrianForCausalLMAdapter._patch_qwen2_decoder_layers(inner)
        a, b = torch.randn(1, 5, 32), torch.randn(1, 8, 32)
        with torch.no_grad():
            own = inner(inputs_embeds=b, use_cache=False, return_dict=True).last_hidden_state.reshape(1, 8, 32)
            alone_a = language_model_forward(inner, a, None).last_hidden_state
            alone_b = language_model_forward(inner, b, None)
            self.assertTrue(torch.allclose(alone_b.last_hidden_state, own, atol=1e-5))  # one sample: the same
            x = torch.cat([torch.cat([torch.zeros(1, 3, 32), a], 1), b], 0)
            mask = torch.tensor([[0, 0, 0] + [1] * 5, [1] * 8])
            both = language_model_forward(inner, x, mask)
        self.assertTrue(torch.allclose(both.last_hidden_state[0, 3:], alone_a[0], atol=1e-5))
        self.assertTrue(torch.allclose(both.last_hidden_state[1], alone_b.last_hidden_state[0], atol=1e-5))
        self.assertTrue(torch.allclose(both.hidden_states[-1][1], alone_b.hidden_states[-1][0], atol=1e-5))
        self.assertEqual(len(both.hidden_states), cfg.num_hidden_layers + 1)


FAKE_STARVLA = {
    "starVLA/__init__.py": "",
    "starVLA/model/__init__.py": "",
    "starVLA/model/modules/__init__.py": "",
    "starVLA/model/modules/vlm/__init__.py": "def get_vlm_model(config):\n    return 'original'\n",
    "starVLA/model/framework/__init__.py": "",
    "starVLA/model/framework/VLM4A/__init__.py": "",
    "starVLA/model/framework/VLM4A/QwenOFT.py": "from starVLA.model.modules.vlm import get_vlm_model\n",
}


class RegisterAndEntryTest(unittest.TestCase):
    def fake(self, tmp):
        for rel, text in FAKE_STARVLA.items():
            p = Path(tmp) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def test_register_dispatches_by_the_starts_type(self):
        from omegaconf import OmegaConf

        from model_matrix.downstream import starvla_cambrian as C

        with tempfile.TemporaryDirectory() as tmp:
            self.fake(tmp)
            start = Path(tmp) / "cambrian"
            start.mkdir()
            (start / "config.json").write_text(json.dumps({"model_type": "cambrian_qwen"}))
            saved = {k: v for k, v in sys.modules.items() if k.startswith("starVLA")}
            for k in saved:
                del sys.modules[k]
            sys.path.insert(0, tmp)
            try:
                with mock.patch.object(C, "CambrianVLInterface", lambda config: "cambrian"):
                    C.register()
                    C.register()  # once
                    import starVLA.model.framework.VLM4A.QwenOFT as oft
                    import starVLA.model.modules.vlm as vlm

                    cfg = lambda p: OmegaConf.create({"framework": {"qwenvl": {"base_vlm": str(p)}}})
                    self.assertEqual(oft.get_vlm_model(cfg(start)), "cambrian")
                    self.assertEqual(vlm.get_vlm_model(cfg(Path(tmp) / "qwen")), "original")
                    self.assertEqual(vlm.get_vlm_model._original(cfg(start)), "original")
            finally:
                sys.path.remove(tmp)
                for k in [k for k in sys.modules if k.startswith("starVLA")]:
                    del sys.modules[k]
                sys.modules.update(saved)

    def test_entry_runs_the_script_as_python_would(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fake(tmp)
            script = Path(tmp) / "work" / "job.py"
            script.parent.mkdir()
            (script.parent / "sibling.py").write_text("X = 1\n")
            script.write_text(textwrap.dedent("""
                import json, sys
                import sibling, ezcolorlog, open_clip
                import starVLA.model.modules.vlm as vlm
                print(json.dumps({"argv": sys.argv, "name": __name__, "path0": sys.path[0],
                                  "shim": ezcolorlog.__file__, "patched": hasattr(vlm.get_vlm_model, "_original")}))
            """))
            env = dict(os.environ, PYTHONPATH=tmp)
            out = subprocess.run([sys.executable, str(ROOT / "model_matrix/downstream/cambrian_entry.py"), str(script),
                                  "--a", "1"], capture_output=True, text=True, env=env, cwd=tmp)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            rec = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(rec["argv"], [str(script), "--a", "1"])
            self.assertEqual((rec["name"], rec["path0"], rec["patched"]), ("__main__", str(script.parent), True))
            import importlib.util

            installed = importlib.util.find_spec("ezcolorlog") is not None  # an installed package wins over the stand-in
            self.assertEqual(rec["shim"].endswith("model_matrix/downstream/shims/ezcolorlog.py"), not installed)

    def test_stand_ins(self):
        import importlib.util

        shims = ROOT / "model_matrix" / "downstream" / "shims"
        spec = importlib.util.spec_from_file_location("ezcolorlog_stand_in", shims / "ezcolorlog.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with self.assertLogs("cambrian", level="WARNING") as logs:
            mod.root_logger.warning_once("once")
            mod.root_logger.warning_once("once")
            mod.root_logger.info("info")
        self.assertEqual(logs.output, ["WARNING:cambrian:once"])
        spec = importlib.util.spec_from_file_location("open_clip_stand_in", shims / "open_clip" / "__init__.py")
        oc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oc)
        with self.assertRaises(ImportError):
            oc.create_model_from_pretrained("ViT-B-32")

    def test_scripts_switch_only_for_cambrian(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, mt in (("c", "cambrian_qwen"), ("q", "qwen3_vl")):
                (Path(tmp) / name).mkdir()
                (Path(tmp) / name / "config.json").write_text(json.dumps({"model_type": mt}))
            probe = (f"source {ROOT}/model_matrix/downstream/cambrian_env.sh; for s in {tmp}/c {tmp}/q; do "
                     "unset SIGLIP_CACHE; starvla_entry $s >/dev/null; echo \"${#ENTRY[@]} ${SIGLIP_CACHE:-none}\"; done")
            env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}")
            out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, env=env).stdout.split("\n")
            self.assertEqual(out[:2], [f"1 {CACHE}", "0 none"])
        for f in ("backbone_load.sh", "backbone_smoke.sh", "cambrian_env.sh"):
            self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / "model_matrix" / "downstream" / f)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
