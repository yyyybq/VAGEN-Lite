"""A two-step SFT through the bundled LLaMA-Factory with a small randomly initialised model of a type, on the CPU (the
model line, step 2): the training path of a model type's SFT -- its registrations, the data, the collator, the forward
and loss, the backward with gradient checkpointing, the optimizer step -- before an 8-card job spends its place in
the queue (the first Qwen3-VL smoke SFT stopped at an import, 2026-10-25 05:31).

The small model keeps the real model's tokenizer, processor, vision settings and token ids, with a two-layer language
model (Qwen3-VL: also a two-block vision encoder; Cambrian-S: the real SigLIP2 tower, frozen as in the recipe).  The
recipe is the smoke's train.yaml with only: the small model, 2 steps, at most 4 conversations, no evaluation, no
saving, fp32, one process.  GPUs are hidden (GPU work goes through sco).

    CUDA_VISIBLE_DEVICES= python -m model_matrix.sft.tiny_sft_check --type qwen3_vl --lf-dir LF --real MODEL --out OUT
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def tiny_qwen3vl(real: str, out: Path) -> None:
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(real)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.rope_scaling = dict(t.rope_scaling, mrope_section=[4, 2, 2])  # sums to head_dim / 2
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 32, 64, 2
    v.out_hidden_size, v.deepstack_visual_indexes = 64, [0]
    torch.manual_seed(0)
    Qwen3VLForConditionalGeneration(cfg).save_pretrained(out)
    AutoProcessor.from_pretrained(real).save_pretrained(out)


def tiny_cambrian(real: str, out: Path) -> None:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from model_matrix.sft.lf_cambrian import register

    register()
    from vagen.models import cambrian_register as R

    cfg = AutoConfig.from_pretrained(real)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    torch.manual_seed(0)
    R.CambrianForCausalLMAdapter(cfg).save_pretrained(out)  # the real SigLIP2 tower, a small language model
    AutoTokenizer.from_pretrained(real).save_pretrained(out)


BUILD = {"qwen3_vl": tiny_qwen3vl, "cambrian_qwen": tiny_cambrian}


def run(kind: str, lf_dir: Path, real: str, out: Path) -> dict:
    import yaml

    from vagen.models.model_types import get

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("run with CUDA_VISIBLE_DEVICES= (GPU work goes through sco)")
    out.mkdir(parents=True, exist_ok=False)
    model = out / "tiny_model"
    t0 = time.time()
    BUILD[kind](real, model)
    built = time.time() - t0
    launcher = get(kind).lf_launcher  # its registrations, as the job's launcher runs them
    __import__(launcher)
    from llamafactory.train.tuner import run_exp

    recipe = yaml.safe_load((lf_dir / "train.yaml").read_text())
    recipe.update(model_name_or_path=str(model), output_dir=str(out / "run"), max_steps=2, max_samples=4,
                  do_eval=False, eval_strategy="no", save_strategy="no", bf16=False, report_to="none",
                  gradient_accumulation_steps=1, logging_steps=1, overwrite_output_dir=True,
                  dataloader_num_workers=0, preprocessing_num_workers=1, overwrite_cache=True)
    recipe.pop("eval_dataset", None)
    recipe.pop("eval_steps", None)
    t0 = time.time()
    run_exp(recipe)
    log = json.loads((out / "run" / "trainer_state.json").read_text())["log_history"] \
        if (out / "run" / "trainer_state.json").is_file() else []
    report = {"type": kind, "lf_dir": str(lf_dir), "real": real, "build_seconds": round(built, 1),
              "train_seconds": round(time.time() - t0, 1),
              "losses": [h["loss"] for h in log if "loss" in h], "log": log, "ok": True}
    report["ok"] = len(report["losses"]) == 2 and all(l == l and l < 1e4 for l in report["losses"])
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=sorted(BUILD), required=True)
    ap.add_argument("--lf-dir", type=Path, required=True, help="the smoke's LLaMA-Factory folder (train.yaml, data)")
    ap.add_argument("--real", required=True, help="the real model (tokenizer, processor, vision settings)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rep = run(a.type, a.lf_dir, a.real, a.out)
    (a.out / "tiny_sft_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("type", "ok", "losses", "build_seconds", "train_seconds")}))


if __name__ == "__main__":
    main()
