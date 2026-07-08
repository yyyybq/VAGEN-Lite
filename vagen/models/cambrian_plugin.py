"""
vagen/models/cambrian_plugin.py

vLLM general plugin that registers CambrianQwenForCausalLM in every process
that vLLM spawns (including the EngineCore_DP* subprocesses).

Registered via the `vllm.general_plugins` entry_point in setup.py so that
`vllm.plugins.load_general_plugins()` automatically imports + calls `register()`
in every worker subprocess.
"""


def register() -> None:
    """Register Cambrian-S with both transformers AutoConfig and vLLM ModelRegistry."""
    # 1. Register a minimal CambrianQwenConfig with transformers so that
    #    vllm's early config-validation (AutoConfig.from_pretrained) doesn't
    #    fail with "does not recognize this architecture".
    #    We use Qwen2Config as the base — the actual field values are loaded
    #    from the checkpoint's config.json at model-load time.
    from transformers import AutoConfig, Qwen2Config

    class _CambrianQwenConfig(Qwen2Config):
        model_type = "cambrian_qwen"

    try:
        AutoConfig.register("cambrian_qwen", _CambrianQwenConfig)
    except ValueError:
        pass  # already registered (e.g. cambrian_register was imported first)

    # 2. Register the vLLM model class so vllm can instantiate it.
    #    The checkpoint config.json may list either class name depending on how
    #    it was saved, so register both to be safe.
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "CambrianQwenForCausalLM",
        "vagen.models.cambrian_vllm:CambrianVLLMForCausalLM",
    )
    ModelRegistry.register_model(
        "CambrianForCausalLMAdapter",
        "vagen.models.cambrian_vllm:CambrianVLLMForCausalLM",
    )
