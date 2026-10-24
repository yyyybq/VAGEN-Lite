"""Model types (the model line, step 2 of 2026-10-24): what differs between the vision-language models the
framework trains, kept in one place.  Qwen2.5-VL is the default, and with it every path stays as it was.

A model type may differ only in how frames become model inputs (its image processing), its chat template, its stop
tokens and its length limits; and in what runs it: the Python of its runtime, the LLaMA-Factory template and
launcher of its SFT, the FSDP wrap classes of that SFT.  The prompt texts, the answer parser, the rewards and the
evaluator are shared.

The type is a field of the training chain's plan ("model_type", checked against the model's config.json); the chain
hands it to the jobs as VAGEN_LITE_MODEL_TYPE with the type's Python as VAGEN_LITE_PYTHON (job_sco.PASS_ENV).

    python -m vagen.models.model_types sft-launch TYPE   # the job script: "ACCELERATE_CONFIG LAUNCHER_MODULE"
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DEFAULT = "qwen2_5_vl"
QWEN3VL_RUNTIME_PYTHON = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime/bin/python"


@dataclass(frozen=True)
class ModelType:
    name: str  # the model's config.json "model_type"
    python: Optional[str]  # the jobs' Python; None: the job script's default (the vagen-lite env)
    lf_template: str  # the LLaMA-Factory template of its SFT
    lf_launcher: str  # the module accelerate runs for its SFT
    sft_accelerate_config: str  # repository-relative accelerate FSDP config of its SFT
    stop_token: str = "<|im_end|>"  # the end of an assistant turn in its chat template


TYPES: Dict[str, ModelType] = {
    "qwen2_5_vl": ModelType("qwen2_5_vl", None, "qwen2_vl", "llamafactory.launcher",
                            "examples/train/active_spatial/accelerate_fsdp_qwen25vl.yaml"),
    # Qwen3-VL-8B-Instruct: transformers >= 4.57 (the runtime venv: vagen-lite + transformers 4.57.6); the local
    # LLaMA-Factory (ca75f1e) gets the template and the vision-tower freeze rule from model_matrix.sft.lf_qwen3vl
    "qwen3_vl": ModelType("qwen3_vl", QWEN3VL_RUNTIME_PYTHON, "qwen3_vl_nothink", "model_matrix.sft.lf_launcher",
                          "model_matrix/sft/accelerate_fsdp_qwen3vl.yaml"),
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


def main(argv=None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "sft-launch":
        kind = get(args[1])
        print(kind.sft_accelerate_config, kind.lf_launcher)
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
