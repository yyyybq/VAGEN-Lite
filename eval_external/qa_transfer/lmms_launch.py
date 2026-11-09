"""Run lmms-eval exactly as `python -m lmms_eval ...`, with the processor pixel log (pixel_log.py) installed first.

    python -m eval_external.qa_transfer.lmms_launch --model qwen2_5_vl --model_args ... --tasks ...

2026-11-06: also every reply as the model produced it, before any task filter, goes to EE_RAW_LOG (run_qa sets
RUN_DIR/raw_responses.jsonl: task, doc_id, text).  lmms-eval's own sample log keeps only the filtered answer when
the two differ in some tasks (SAT's filter keeps the first word, and the raw reply is dropped), so a lenient
re-score needs this file.  Only the Qwen-VL model classes are wrapped; their outputs are not changed.
"""

from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
import threading

from eval_external.qa_transfer import pixel_log

RAW_CLASSES = [("lmms_eval.models.chat.qwen3_vl", "Qwen3_VL"), ("lmms_eval.models.chat.qwen2_5_vl", "Qwen2_5_VL"),
               ("lmms_eval.models.simple.qwen3_vl", "Qwen3_VL"), ("lmms_eval.models.simple.qwen2_5_vl", "Qwen2_5_VL")]
_LOCK = threading.Lock()


def install_raw_log(path: str | None = None) -> list[str]:
    path = path or os.environ.get("EE_RAW_LOG")
    if not path:
        return []
    done = []
    for mod, name in RAW_CLASSES:
        try:
            cls = getattr(importlib.import_module(mod), name)
        except Exception:  # e.g. qwen3_vl module unusable under an older transformers
            continue
        if "generate_until" not in cls.__dict__ or getattr(cls.generate_until, "_ee_raw_log", False):
            continue
        original = cls.generate_until

        def generate_until(self, requests, _original=original):
            out = _original(self, requests)
            with _LOCK, open(path, "a") as f:
                for req, o in zip(requests, out):
                    text = o.text if hasattr(o, "text") else o
                    f.write(json.dumps({"task": req.task_name, "doc_id": req.doc_id, "text": text},
                                       ensure_ascii=False) + "\n")
            return out

        generate_until._ee_raw_log = True
        cls.generate_until = generate_until
        done.append(f"{mod}.{name}")
    return done


if __name__ == "__main__":
    pixel_log.install()
    install_raw_log()
    sys.argv = ["lmms_eval"] + sys.argv[1:]
    runpy.run_module("lmms_eval", run_name="__main__", alter_sys=True)
