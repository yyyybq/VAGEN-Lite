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
import runpy
import sys
from pathlib import Path

import transformers
import vlmeval.vlm
from vlmeval.vlm import Qwen2VLChat

VLMEVAL_RUN = Path(vlmeval.__file__).resolve().parent.parent / "run.py"
MODEL_CLASSES = (transformers.Qwen2_5_VLForConditionalGeneration, transformers.Qwen2VLForConditionalGeneration)


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

if __name__ == "__main__":
    from eval_external.qa_transfer import pixel_log

    pixel_log.install()  # only when EE_PIXEL_LOG is set (run_qa sets it)
    sys.argv = [str(VLMEVAL_RUN)] + sys.argv[1:]
    sys.path.insert(0, str(VLMEVAL_RUN.parent))
    runpy.run_path(str(VLMEVAL_RUN), run_name="__main__")
