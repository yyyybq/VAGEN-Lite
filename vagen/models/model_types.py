"""Model types (the model line, step 2 of 2026-10-24): what differs between the vision-language models the
framework trains, kept in one place.  Qwen2.5-VL is the default, and with it every path stays as it was.

A model type may differ only in how frames become model inputs (its image processing), its chat template, its stop
tokens and its length limits; and in what runs it: the Python of its runtime, the LLaMA-Factory template and
launcher of its SFT, the FSDP wrap classes of that SFT.  The prompt texts, the answer parser, the rewards and the
evaluator are shared.

The type is a field of the training chain's plan ("model_type", checked against the model's config.json); the chain
hands it to the jobs as VAGEN_LITE_MODEL_TYPE with the type's Python as VAGEN_LITE_PYTHON and its environment
(job_sco.PASS_ENV); training_prep rl-config --model-type adds its PPO-config overrides (rl_overrides).

    python -m vagen.models.model_types sft-launch TYPE   # the job script: "ACCELERATE_CONFIG LAUNCHER_MODULE"
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT = "qwen2_5_vl"
QWEN3VL_RUNTIME_PYTHON = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime/bin/python"
# the line's download of Cambrian-S-7B and SigLIP2 (decision 1 of 2026-10-24; model_matrix/weights/fetch_cambrian.py):
# the Cambrian adapter loads its vision tower and both image processors by repository name from this cache
CAMBRIAN_CACHE = "/mnt/umm/shared_model/vagen-lite-models/hub"
CAMBRIAN_TOKENS_PER_IMAGE = 756  # 27 x (27 + 1): SigLIP2's 729 patches and a newline per row


@dataclass(frozen=True)
class ModelType:
    name: str  # the model's config.json "model_type"
    python: Optional[str]  # the jobs' Python; None: the job script's default (the vagen-lite env)
    lf_template: Optional[str]  # the LLaMA-Factory template of its SFT (None: no SFT tool yet)
    lf_launcher: Optional[str]  # the module accelerate runs for its SFT
    sft_accelerate_config: Optional[str]  # repository-relative accelerate FSDP config of its SFT
    stop_token: str = "<|im_end|>"  # the end of an assistant turn in its chat template
    env: Tuple[Tuple[str, str], ...] = ()  # extra environment of its jobs (job_sco.PASS_ENV)


TYPES: Dict[str, ModelType] = {
    "qwen2_5_vl": ModelType("qwen2_5_vl", None, "qwen2_vl", "llamafactory.launcher",
                            "examples/train/active_spatial/accelerate_fsdp_qwen25vl.yaml"),
    # Qwen3-VL-8B-Instruct: transformers >= 4.57 (the runtime venv: vagen-lite + transformers 4.57.6); the local
    # LLaMA-Factory (ca75f1e) gets the template and the vision-tower freeze rule from model_matrix.sft.lf_qwen3vl
    "qwen3_vl": ModelType("qwen3_vl", QWEN3VL_RUNTIME_PYTHON, "qwen3_vl_nothink", "model_matrix.sft.lf_launcher",
                          "model_matrix/sft/accelerate_fsdp_qwen3vl.yaml"),
    # Cambrian-S-7B: the vagen-lite env, the repository's adapter (vagen/models/cambrian_register.py, FSDP) and vLLM
    # model (cambrian_vllm.py); one <image> token per frame in the prompt, 756 positions in the model
    "cambrian_qwen": ModelType("cambrian_qwen", None, None, None, None,
                               env=(("HF_HUB_CACHE", CAMBRIAN_CACHE), ("SIGLIP_CACHE", CAMBRIAN_CACHE))),
}


def get(name: Optional[str] = None) -> ModelType:
    key = name or DEFAULT
    if key not in TYPES:
        raise ValueError(f"unknown model type {key!r}; known: {sorted(TYPES)}")
    return TYPES[key]


def of_checkpoint(model: str) -> Optional[str]:
    """The model_type of a Hugging Face model folder (None without a readable config.json)."""
    path = Path(model) / "config.json"
    try:
        return json.loads(path.read_text()).get("model_type")
    except (OSError, ValueError):
        return None


def check(name: Optional[str], model: str) -> ModelType:
    """The type, after checking it is the model's own (a model folder whose config.json cannot be read passes: the
    jobs read it as root)."""
    kind = get(name)
    found = of_checkpoint(model)
    if found is not None and found != kind.name:
        raise ValueError(f"{model} is a {found} model, not {kind.name}")
    return kind


def rl_overrides(name: Optional[str], max_turns: int, prompt_length: int) -> Dict[str, Any]:
    """The PPO-config overrides of a model type (dotted keys; none for the Qwen types, whose configs stay as they
    are).  Cambrian-S on the concat path: the adapter as verl's external library, image positions marked by
    <|image_pad|> (151655, the adapter's image_token_alias), the agent-loop manager that expands the compact <image>
    tokens for the trainer (vagen/agent_loop/cambrian_concat.py) with room for ``max_turns`` frames, and the prompt
    budget raised by 756 tokens per frame: vLLM's max_model_len is prompt_length + response_length and counts the
    expanded tokens (verl's vllm_async_server)."""
    kind = get(name)
    if kind.name != "cambrian_qwen":
        return {}
    return {"actor_rollout_ref.model.external_lib": "vagen.models.cambrian_register",
            "actor_rollout_ref.model.trust_remote_code": True,
            "actor_rollout_ref.model.override_config.image_token_alias": 151655,
            "trainer.agent_loop_manager": "vagen.agent_loop.cambrian_concat.CambrianAgentLoopManager",
            "cambrian_concat.max_images": int(max_turns),
            "data.max_prompt_length": int(prompt_length) + CAMBRIAN_TOKENS_PER_IMAGE * int(max_turns)}


def main(argv=None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "sft-launch":
        kind = get(args[1])
        if not kind.lf_launcher:
            raise SystemExit(f"no SFT tool for {kind.name} yet")
        print(kind.sft_accelerate_config, kind.lf_launcher)
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
