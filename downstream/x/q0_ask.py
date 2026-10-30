#!/usr/bin/env python3
"""Ask one model one or more question sets zero-shot, and optionally take its hidden states (the user's decisions of
2026-10-30: Q0, G0, X0b and the linear probe, on the four Qwen2.5 starts and four Qwen3 models).

    python downstream/x/q0_ask.py --model Q3B --ask Q.jsonl=ANSWERS.jsonl [--ask Q2.jsonl=A2.jsonl ...]
                                  [--hidden Q.jsonl=HIDDEN.npz] [--smoke 2]

A question line needs "id", "prompt" and "image_path" (or "images": a list of paths); "max_new_tokens" is optional
(default 16, as X0).  Models (MODELS; read by path, never copied): the Qwen2.5-VL-7B starts B, S, RS, RD from
downstream/starts.json, and Qwen3-VL-8B-Instruct Q3B (the shared snapshot), Q3S (SFT epoch 8), Q3RS (S0 update 100),
Q3RD (V4 seed 20261019 update 300), the paths the main window gave on 2026-10-28.  The processor comes from the
model's own folder when it has a preprocessor_config.json, otherwise from its family's base (B or Q3B); either way
the folder used is written to ANSWERS.meta.json.  Greedy decoding; each answer line is {"id", "reply"}; ids already
in ANSWERS are skipped, so a rerun asks only what is missing.  --hidden writes, per question, the last prompt token's
hidden state after the layers at a quarter, half, three quarters and all of the language model's depth (float16,
HIDDEN.npz: ids, layers, h [n, 4, d]).  --smoke N first asks the first N questions of the first set and stops with
exit 3 if the model cannot answer.  Run with the StarVLA environment's Python (transformers 4.57) on one GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent

QWEN3_ROOT = "/mnt/umm/users/yinbaiqiao/models_runs/chains"
QWEN3 = {
    "Q3B": "/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
           "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "Q3S": f"{QWEN3_ROOT}/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232",
    "Q3RS": f"{QWEN3_ROOT}/m_qwen3vl8b_formal_rl_20261026/rl_run_q3s0/checkpoints/global_step_100/actor/huggingface",
    "Q3RD": f"{QWEN3_ROOT}/m_qwen3vl8b_formal_rl_20261026/rl_run_q3d/checkpoints/global_step_300/actor/huggingface",
}
QWEN25 = ("B", "S", "RS", "RD")
MODELS = QWEN25 + tuple(QWEN3)
DEPTHS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_NEW_TOKENS = 16


def family(name: str) -> str:
    if name in QWEN3:
        return "qwen3"
    if name in QWEN25:
        return "qwen2.5"
    raise KeyError(f"unknown model {name}; known: {', '.join(MODELS)}")


def model_path(name: str, starts: Path = DOWNSTREAM / "starts.json") -> str:
    if family(name) == "qwen3":
        return QWEN3[name]
    return json.loads(Path(starts).read_text())[name]["path"]


def processor_dir(name: str, starts: Path = DOWNSTREAM / "starts.json") -> str:
    """The model's own folder if it has its image processor settings, otherwise its family's base model."""
    own = Path(model_path(name, starts))
    if (own / "preprocessor_config.json").is_file():
        return str(own)
    return model_path("Q3B" if family(name) == "qwen3" else "B", starts)


def layer_indices(n_layers: int, depths=DEPTHS) -> list:
    """Indices into the hidden_states tuple (0: embeddings, i: after layer i) at the given fractions of depth."""
    return [max(1, min(n_layers, round(f * n_layers))) for f in depths]


def load_questions(path: Path) -> list:
    out = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for q in out:
        if "id" not in q:
            raise ValueError(f"{path}: a question without an id")
    return out


def images_of(q: dict) -> list:
    return list(q.get("images") or [q["image_path"]])


def done_ids(path: Path) -> set:
    if not Path(path).is_file():
        return set()
    return {json.loads(line)["id"] for line in Path(path).read_text().splitlines() if line.strip()}


def load(name: str):
    import torch
    from transformers import AutoProcessor

    proc_dir = processor_dir(name)
    processor = AutoProcessor.from_pretrained(proc_dir)
    if family(name) == "qwen2.5":  # the class X0 used for these four
        from transformers import Qwen2_5_VLForConditionalGeneration as cls
    else:
        from transformers import AutoModelForImageTextToText as cls
    model = cls.from_pretrained(model_path(name), torch_dtype=torch.bfloat16).to("cuda").eval()
    return processor, model, proc_dir


def inputs_for(processor, model, q: dict):
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in images_of(q)]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": q["prompt"]}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    return processor(text=[text], images=images, return_tensors="pt").to(model.device)


def ask(processor, model, q: dict) -> str:
    import torch

    inputs = inputs_for(processor, model, q)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(q.get("max_new_tokens", DEFAULT_NEW_TOKENS)),
                             do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def hidden(processor, model, q: dict, layers: list):
    import torch

    inputs = inputs_for(processor, model, q)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    return torch.stack([hs[i][0, -1, :].float() for i in layers]).cpu().numpy().astype("float16")


def n_text_layers(model) -> int:
    cfg = model.config
    text = getattr(cfg, "text_config", None) or cfg
    return int(text.num_hidden_layers)


def pair(spec: str):
    q, _, out = spec.partition("=")
    if not q or not out:
        raise argparse.ArgumentTypeError(f"expected QUESTIONS=OUT, got {spec!r}")
    return Path(q), Path(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--ask", type=pair, action="append", default=[], help="QUESTIONS.jsonl=ANSWERS.jsonl")
    ap.add_argument("--hidden", type=pair, action="append", default=[], help="QUESTIONS.jsonl=HIDDEN.npz")
    ap.add_argument("--smoke", type=int, default=0, help="ask the first N questions of the first set first")
    args = ap.parse_args(argv)
    import numpy as np

    t0 = time.time()
    processor, model, proc_dir = load(args.model)
    meta = {"model": args.model, "path": model_path(args.model), "processor": proc_dir, "family": family(args.model)}
    print(f"[q0_ask] {args.model}: loaded in {time.time() - t0:.0f} s (processor {proc_dir})", flush=True)
    if args.smoke and args.ask:
        qs = load_questions(args.ask[0][0])[:args.smoke]
        try:
            replies = [ask(processor, model, q) for q in qs]
        except Exception as exc:
            print(f"[q0_ask] {args.model}: smoke test failed: {type(exc).__name__}: {exc}", flush=True)
            return 3
        print(f"[q0_ask] {args.model}: smoke replies {replies}", flush=True)
    for qpath, out in args.ask:
        questions = load_questions(qpath)
        done = done_ids(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + ".meta.json").write_text(json.dumps(dict(meta, questions=str(qpath)), indent=1) + "\n")
        t = time.time()
        with open(out, "a") as f:
            for q in questions:
                if q["id"] in done:
                    continue
                f.write(json.dumps({"id": q["id"], "reply": ask(processor, model, q)}) + "\n")
                f.flush()
        print(f"[q0_ask] {args.model}: {qpath.name} -> {out} ({len(questions) - len(done)} asked, "
              f"{time.time() - t:.0f} s)", flush=True)
    for qpath, out in args.hidden:
        if out.is_file():
            print(f"[q0_ask] {args.model}: {out} exists, skipped", flush=True)
            continue
        questions = load_questions(qpath)
        layers = layer_indices(n_text_layers(model))
        t = time.time()
        h = np.stack([hidden(processor, model, q, layers) for q in questions])
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".partial.npz")
        np.savez(tmp, ids=np.array([q["id"] for q in questions]), layers=np.array(layers), h=h,
                 meta=json.dumps(dict(meta, questions=str(qpath))))
        tmp.rename(out)
        print(f"[q0_ask] {args.model}: hidden states {h.shape} at layers {layers} -> {out} ({time.time() - t:.0f} s)",
              flush=True)
    print(f"[q0_ask] {args.model}: done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
