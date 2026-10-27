"""LLaMA-Factory's launcher with the model line's registrations first (accelerate runs it in every process):
``accelerate launch ... -m model_matrix.sft.lf_launcher train.yaml`` (sco_interiorgs_v10_train.sh lf_train with
VAGEN_LITE_MODEL_TYPE set; vagen/models/model_types.py names this module)."""
from model_matrix.sft.lf_qwen3vl import register

register()

from llamafactory.launcher import launch  # noqa: E402  (after the registrations)

if __name__ == "__main__":
    launch()
