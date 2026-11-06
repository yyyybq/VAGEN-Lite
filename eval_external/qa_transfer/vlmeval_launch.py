"""Run VLMEvalKit's run.py with one extra model class, Qwen2VLChatSDPA (config "class": "Qwen2VLChatSDPA").

    python -m eval_external.qa_transfer.vlmeval_launch --config CFG --work-dir DIR --mode all ...

Qwen2VLChat loads the model with attn_implementation="flash_attention_2" written into its code.  The flash-attn
wheels for torch 2.8 need glibc 2.32 and the sco image has 2.31 (checked 2026-10-18); our own training env runs
without flash-attn too.  Qwen2VLChatSDPA is Qwen2VLChat with only that argument changed to "sdpa" (PyTorch's
built-in attention): prompts, frame sampling, pixels and decoding stay Qwen2VLChat's own.  VLMEvalKit's files
are not modified; the class is registered into vlmeval.vlm before run.py starts.
"""

from __future__ import annotations

import contextlib
import json
import os
import runpy
import sys
from pathlib import Path

import transformers
import vlmeval.vlm
from vlmeval.vlm import Qwen2VLChat

from eval_external.qa_transfer.benchmarks import MINDCUBE_TINY1050

VLMEVAL_RUN = Path(vlmeval.__file__).resolve().parent.parent / "run.py"
# Qwen3VLChat (2026-11-06) loads through AutoModelForImageTextToText, also with flash_attention_2 written into its
# code; Qwen3VLForConditionalGeneration exists from transformers 4.57 on (the eval-external-q3 env).
MODEL_CLASSES = tuple(c for c in (transformers.Qwen2_5_VLForConditionalGeneration,
                                  transformers.Qwen2VLForConditionalGeneration,
                                  transformers.AutoModelForImageTextToText,
                                  getattr(transformers, "Qwen3VLForConditionalGeneration", None)) if c is not None)


@contextlib.contextmanager
def attention_override(attn: str):
    """While active, from_pretrained(..., attn_implementation="flash_attention_2") loads with `attn` instead."""
    own = {cls: cls.__dict__.get("from_pretrained") for cls in MODEL_CLASSES}
    for cls in MODEL_CLASSES:
        original = cls.from_pretrained  # bound to cls

        def from_pretrained(*args, _original=original, **kwargs):
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = attn
            return _original(*args, **kwargs)

        cls.from_pretrained = staticmethod(from_pretrained)
    try:
        yield
    finally:
        for cls, saved in own.items():
            if saved is None:
                del cls.from_pretrained
            else:
                cls.from_pretrained = saved


class Qwen2VLChatSDPA(Qwen2VLChat):
    ATTN = "sdpa"

    def __init__(self, *args, **kwargs):
        with attention_override(self.ATTN):
            super().__init__(*args, **kwargs)
        self.attn_implementation = self.ATTN
        model = getattr(self, "model", None)
        print(f"[eval_external] {self.model_path}: attention {getattr(getattr(model, 'config', None), '_attn_implementation', None)}, "
              f"dtype {getattr(model, 'dtype', None)}", flush=True)


vlmeval.vlm.Qwen2VLChatSDPA = Qwen2VLChatSDPA

if hasattr(vlmeval.vlm, "Qwen3VLChat"):
    class Qwen3VLChatSDPA(vlmeval.vlm.Qwen3VLChat):
        """Qwen3VLChat (EASI v0.2.2) with SDPA attention instead of flash_attention_2; nothing else changes."""
        ATTN = "sdpa"

        def __init__(self, *args, **kwargs):
            with attention_override(self.ATTN):
                super().__init__(*args, **kwargs)
            model = getattr(self, "model", None)
            print(f"[eval_external] {self.model_path}: attention "
                  f"{getattr(getattr(model, 'config', None), '_attn_implementation', None)}, "
                  f"dtype {getattr(model, 'dtype', None)}, do_sample "
                  f"{getattr(getattr(model, 'generation_config', None), 'do_sample', None)}", flush=True)

    vlmeval.vlm.Qwen3VLChatSDPA = Qwen3VLChatSDPA


def register_tables() -> None:
    """Our extra question tables (benchmarks.MINDCUBE_TINY1050): VLMEvalKit finds a table by its class's
    DATASET_URL / DATASET_MD5; the file is already in $LMUData, so the URL is never fetched."""
    from vlmeval.dataset.mindcubebench import MindCubeBench

    lmu = Path(os.environ.get("LMUData", "/mnt/umm/users/yinbaiqiao/eval_external_assets/LMUData"))
    info = lmu / f"{MINDCUBE_TINY1050}.json"
    if info.exists():
        MindCubeBench.DATASET_URL[MINDCUBE_TINY1050] = f"file://{lmu / (MINDCUBE_TINY1050 + '.tsv')}"
        MindCubeBench.DATASET_MD5[MINDCUBE_TINY1050] = json.loads(info.read_text())["md5"]


register_tables()

if __name__ == "__main__":
    from eval_external.qa_transfer import pixel_log

    pixel_log.install()  # only when EE_PIXEL_LOG is set (run_qa sets it)
    sys.argv = [str(VLMEVAL_RUN)] + sys.argv[1:]
    sys.path.insert(0, str(VLMEVAL_RUN.parent))
    runpy.run_path(str(VLMEVAL_RUN), run_name="__main__")
