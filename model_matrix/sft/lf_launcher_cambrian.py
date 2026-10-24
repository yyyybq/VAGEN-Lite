"""LLaMA-Factory's launcher with Cambrian-S's registrations first (accelerate runs it in every process):
``accelerate launch ... -m model_matrix.sft.lf_launcher_cambrian train.yaml`` (sco_interiorgs_v10_train.sh lf_train
with VAGEN_LITE_MODEL_TYPE=cambrian_qwen; vagen/models/model_types.py names this module).  A launcher of its own, so
the Qwen3-VL one (model_matrix.sft.lf_launcher, run in the Qwen3-VL runtime) never imports the Cambrian code."""
from model_matrix.sft.lf_cambrian import register

register()

from llamafactory.launcher import launch  # noqa: E402  (after the registrations)

if __name__ == "__main__":
    launch()
