"""Run lmms-eval exactly as `python -m lmms_eval ...`, with the processor pixel log (pixel_log.py) installed first.

    python -m eval_external.qa_transfer.lmms_launch --model qwen2_5_vl --model_args ... --tasks ...
"""

from __future__ import annotations

import runpy
import sys

from eval_external.qa_transfer import pixel_log

if __name__ == "__main__":
    pixel_log.install()
    sys.argv = ["lmms_eval"] + sys.argv[1:]
    runpy.run_module("lmms_eval", run_name="__main__", alter_sys=True)
