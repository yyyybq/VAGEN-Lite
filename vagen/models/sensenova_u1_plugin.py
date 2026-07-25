"""
vLLM general plugin bootstrap for SenseNova-U1.

This imports the SenseNova-U1 package inside vLLM worker subprocesses so
Transformers knows the ``neo_chat`` / ``neo_vision`` config classes.  Full vLLM
execution still depends on the installed vLLM version supporting NEOChatModel
or on adding a dedicated U1 vLLM model wrapper.
"""

from __future__ import annotations

import os
import sys


def register() -> None:
    src = os.environ.get("SENSENOVA_U1_SRC", "/nas/baiqiao/SenseNova-U1/src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import sensenova_u1  # noqa: F401
