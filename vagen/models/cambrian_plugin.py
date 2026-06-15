"""
vagen/models/cambrian_plugin.py

vLLM general plugin that registers CambrianQwenForCausalLM in every process
that vLLM spawns (including the EngineCore_DP* subprocesses).

Registered via the `vllm.general_plugins` entry_point in setup.py so that
`vllm.plugins.load_general_plugins()` automatically imports + calls `register()`
in every worker subprocess.
"""


def register() -> None:
    """Register Cambrian-S vLLM model class with the ModelRegistry."""
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "CambrianQwenForCausalLM",
        "vagen.models.cambrian_vllm:CambrianVLLMForCausalLM",
    )
