"""The Transformers compatibility names of the Qwen3-VL SFT registration (model_matrix/sft/lf_qwen3vl.py) and the
small-model SFT check's guard (model_matrix/sft/tiny_sft_check.py)."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class CompatTest(unittest.TestCase):
    def test_is_jieba_available_given_back_when_missing(self):
        import transformers.utils as tu

        from model_matrix.sft.lf_qwen3vl import transformers_compat

        saved = tu.__dict__.pop("is_jieba_available", None)
        try:
            transformers_compat()  # as in Transformers 4.57.6, where the name is gone
            self.assertEqual(tu.is_jieba_available(), importlib.util.find_spec("jieba") is not None)
            mine = object()
            tu.is_jieba_available = mine
            transformers_compat()  # an existing name is left as it is
            self.assertIs(tu.is_jieba_available, mine)
        finally:
            if saved is not None:
                tu.is_jieba_available = saved
            else:
                tu.__dict__.pop("is_jieba_available", None)

    def test_tiny_check_refuses_visible_gpus(self):
        from model_matrix.sft import tiny_sft_check as T

        with tempfile.TemporaryDirectory() as tmp:
            for env in ({"CUDA_VISIBLE_DEVICES": "0"}, {}):
                with mock.patch.dict(os.environ, env, clear=False):
                    if not env:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    with self.assertRaisesRegex(SystemExit, "CUDA_VISIBLE_DEVICES"):
                        T.run("qwen3_vl", Path(tmp), "/m", Path(tmp) / "out")
            self.assertFalse((Path(tmp) / "out").exists())
        self.assertEqual(sorted(T.BUILD), ["cambrian_qwen", "qwen3_vl"])


if __name__ == "__main__":
    unittest.main()
