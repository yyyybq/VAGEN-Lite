"""Placeholder for the SenseNova-U1 vLLM plugin (cleanup, 2026-10).

The SenseNova-U1 adapter moved to archive/side_models/sensenova_u1/ (decision D4).  The vagen-lite environment was
installed from setup.py while it still named this module as a vLLM plugin, and vLLM imports every registered plugin
in each of its processes; this module keeps that import working and registers nothing.  Delete it after the
environment is reinstalled (pip install -e .), which drops the entry point.
"""


def register() -> None:
    return None
