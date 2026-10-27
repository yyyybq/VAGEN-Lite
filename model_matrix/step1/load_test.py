"""Step 1 of the model line: one-card load test (the brief's 2.6).

Loads one model on one card, shows it one environment frame and the first user turn of one new-format SFT
conversation (format example "action_names"), and records what it answers: one greedy answer and 4 answers sampled
as the RL rollout samples (temperature 0.8, top-p 0.92), each read with the env's strict reading.  Nothing is
trained, nothing is written outside --out.

    python load_test.py --kind qwen3vl|cambrian --backend hf|vllm --model PATH --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

RECORDS = ("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r12_20261019/"
           "format_sft_v2a/sft_trajectories_format.jsonl")
RECORD_ID = "sft_v10_c269344785dd21218e7a415e"  # val, absolute_positioning, d* 2
MAX_NEW_TOKENS = 64
SAMPLES = 4


def load_case(path: str, record_id: str):
    from PIL import Image

    for line in open(path):
        rec = json.loads(line)
        if rec["id"] == record_id:
            break
    else:
        raise SystemExit(f"record {record_id} not in {path}")
    conv = rec["conversations"]
    system, user, reference = conv[0]["content"], conv[1]["content"], conv[2]["content"]
    image = Image.open(rec["conversation_image_paths"][0]).convert("RGB")
    content = []
    for seg in re.split(r"(<image>)", user):  # as the agent loop's convert_obs_to_content
        if seg == "<image>":
            content.append({"type": "image"})
        elif seg:
            content.append({"type": "text", "text": seg})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    return rec, messages, image, reference


ENV_TEMPLATE = Path(__file__).resolve().parents[2] / "data_gen" / "interiorgs_v10" / "configs" / "rl_env_v10_S0.yaml"


def allowed_actions(template: Path = ENV_TEMPLATE) -> set:
    """The action names the RL / evaluation env accepts, worked out as ActiveSpatialEnv.__init__ does: the
    template's action_space preset, without "done" when explicit done is off."""
    import yaml

    from vagen.envs.active_spatial.utils import ACTION_SPACE_PRESETS

    cfg = yaml.safe_load(template.read_text())["envs"][0]["config"]
    names = set(ACTION_SPACE_PRESETS.get(cfg.get("action_space", "legacy"), ACTION_SPACE_PRESETS["legacy"]))
    if not cfg.get("enable_explicit_done", True):
        names.discard("done")
    return names


def strict_read(text: str) -> dict:
    """The env's strict reading (ActiveSpatialEnv._strict_parse_func) without building an env."""
    try:
        from vagen.envs.active_spatial.env import ActiveSpatialEnv

        class _Bare:
            _allowed_actions = allowed_actions()

        r = ActiveSpatialEnv._strict_parse_func(_Bare(), text, action_sep="|", max_actions=5)
        return {"readable": bool(r["actions"]) and bool(r["format_correct"]), "actions": r["actions"],
                "parse_error": r["parse_error"], "reader": "env"}
    except Exception as exc:  # the env module did not import: say so, read nothing
        return {"readable": None, "actions": [], "parse_error": f"reader unavailable: {exc!r}"[:300],
                "reader": None}


def gpu_state() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {"cuda": True, "device": torch.cuda.get_device_name(0), "used_gib": round((total - free) / 2**30, 2),
            "total_gib": round(total / 2**30, 2), "torch_peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def answers(greedy: str, samples) -> dict:
    return {"greedy": {"text": greedy, **strict_read(greedy)},
            "samples": [{"text": s, **strict_read(s)} for s in samples]}


# ----------------------------------------------------------------------------------------------- Qwen3-VL
def qwen3vl_prompt(path, messages, image):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(path)
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[text], images=[image], return_tensors="pt")
    pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    info = {"processor": type(proc).__name__, "image_processor": type(proc.image_processor).__name__,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "image_tokens": int((inputs["input_ids"] == pad).sum()),
            "processor_keys": sorted(inputs.keys()),
            "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None,
            "eos": proc.tokenizer.eos_token, "pad": proc.tokenizer.pad_token}
    return proc, text, inputs, info


def run_qwen3vl_hf(path, messages, image, out):
    import torch
    from transformers import AutoModelForImageTextToText

    proc, text, inputs, info = qwen3vl_prompt(path, messages, image)
    out["prompt"] = info
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda:0")
    model.eval()
    out["load_s"] = round(time.time() - t0, 1)
    out["model_class"] = type(model).__name__
    out["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
    out["after_load"] = gpu_state()
    inputs = inputs.to("cuda:0")
    n = inputs["input_ids"].shape[1]
    with torch.no_grad():
        t0 = time.time()
        g = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        out["greedy_s"] = round(time.time() - t0, 2)
        s = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.8, top_p=0.92,
                           top_k=0, num_return_sequences=SAMPLES)
    dec = lambda ids: proc.tokenizer.decode(ids[n:], skip_special_tokens=True)
    out["answers"] = answers(dec(g[0]), [dec(x) for x in s])
    out["after_generate"] = gpu_state()


def run_qwen3vl_vllm(path, messages, image, out):
    from vllm import LLM, SamplingParams

    proc, text, inputs, info = qwen3vl_prompt(path, messages, image)
    out["prompt"] = info
    t0 = time.time()
    llm = LLM(model=path, tensor_parallel_size=1, gpu_memory_utilization=0.5, max_model_len=16384,
              limit_mm_per_prompt={"image": 13}, enforce_eager=True, seed=0)
    out["load_s"] = round(time.time() - t0, 1)
    out["after_load"] = gpu_state()
    req = {"prompt": text, "multi_modal_data": {"image": [image]}}
    t0 = time.time()
    g = llm.generate([req], SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))[0]
    out["greedy_s"] = round(time.time() - t0, 2)
    s = llm.generate([req], SamplingParams(n=SAMPLES, temperature=0.8, top_p=0.92, max_tokens=MAX_NEW_TOKENS,
                                           seed=0))[0]
    out["vllm_prompt_tokens"] = len(g.prompt_token_ids)
    out["answers"] = answers(g.outputs[0].text, [o.text for o in s.outputs])


# ----------------------------------------------------------------------------------------------- Cambrian-S
def cambrian_prompt(path, messages, image, out):
    from transformers import AutoTokenizer

    from vagen.models.cambrian_processor import IMAGE_TOKEN_INDEX, TOKENS_PER_IMAGE, CambrianProcessorWrapper

    tok = AutoTokenizer.from_pretrained(path)
    wrap = CambrianProcessorWrapper(tok)
    text = wrap.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    compact = tok.encode(text, add_special_tokens=False)
    expanded = []
    for t in compact:
        expanded.extend([IMAGE_TOKEN_INDEX] * TOKENS_PER_IMAGE if t == wrap.image_token_id else [t])
    pixel = None
    try:
        pixel = wrap.preprocess_images([image])["pixel_values"]
        out["siglip_processor"] = "AutoImageProcessor google/siglip2-so400m-patch14-384 (cache)"
    except Exception as exc:
        out["siglip_processor_error"] = repr(exc)[:400]
        # Labelled stand-in, only so the rest of the path can be tested: the images are already resized to 384 x 384
        # (preprocess_siglip_images), so what is left is SigLIP2's rescale 1/255 and normalisation (mean = std = 0.5),
        # which SiglipImageProcessor's defaults do too.
        from transformers import SiglipImageProcessor

        from vagen.models.cambrian_processor import preprocess_siglip_images

        pixel = preprocess_siglip_images([image], SiglipImageProcessor(size={"height": 384, "width": 384}))
        out["siglip_processor"] = "STAND-IN SiglipImageProcessor(384, mean=std=0.5), not the checkpoint's own"
    out["prompt"] = {"compact_tokens": len(compact), "expanded_tokens": len(expanded),
                     "image_token_id": wrap.image_token_id, "tokens_per_image": TOKENS_PER_IMAGE,
                     "eos": tok.eos_token, "pad": tok.pad_token,
                     "pixel_values": list(pixel.shape) if pixel is not None else None}
    return tok, wrap, text, compact, expanded, pixel


def run_cambrian_hf(path, messages, image, out):
    import torch

    import vagen.models.cambrian_register  # noqa: F401  (registers the adapter; needs CAMBRIAN_SRC)
    from transformers import AutoModelForCausalLM

    tok, wrap, text, compact, expanded, pixel = cambrian_prompt(path, messages, image, out)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    model.to("cuda:0").eval()
    out["load_s"] = round(time.time() - t0, 1)
    out["model_class"] = type(model).__name__
    out["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
    out["after_load"] = gpu_state()
    if pixel is None:
        raise RuntimeError("no SigLIP pixel values (see siglip_processor_error); generation skipped")
    ids = torch.tensor([expanded], device="cuda:0")
    pv = pixel.to("cuda:0", dtype=torch.bfloat16)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    # step 2: the adapter's generate() fails in the Cambrian source's KV-cache code with these Transformers
    # (DynamicCache.get_usable_length, an older API; training never uses the cache), so decode greedily by full
    # forwards (no cache) -- the training path -- and keep each token's log-probability for the comparison with vLLM
    tokens, logprobs = [], []
    with torch.no_grad():
        t0 = time.time()
        cur = ids
        for _ in range(MAX_NEW_TOKENS):
            logits = model(input_ids=cur, pixel_values=pv, use_cache=False).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            t = int(lp.argmax())
            tokens.append(t)
            logprobs.append(float(lp[t]))
            if t in eos:
                break
            cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], dim=1)
        out["greedy_s"] = round(time.time() - t0, 2)
    out["greedy_tokens"], out["greedy_logprobs"] = tokens, logprobs
    out["sampling"] = "not run: the adapter's cached generation fails (see greedy_s); greedy by full forwards"
    out["answers"] = answers(tok.decode(tokens, skip_special_tokens=True), [])
    out["after_generate"] = gpu_state()


def run_cambrian_vllm(path, messages, image, out):
    from vllm import LLM, SamplingParams

    tok, wrap, text, compact, expanded, pixel = cambrian_prompt(path, messages, image, out)
    if "siglip_processor_error" in out:
        # Without SigLIP2's preprocessor_config.json the wrapper falls back to SiglipImageProcessor's default 224 x 224
        # (256 patches against the vision tower's 729; the first run of this test stopped there).  The labelled
        # stand-in of cambrian_prompt, in this process: VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps the engine here.
        from transformers import SiglipImageProcessor

        import vagen.models.cambrian_vllm as cv

        cv._get_siglip_image_processor = lambda: SiglipImageProcessor(size={"height": 384, "width": 384})
        out["vllm_siglip_stand_in"] = True
        out["vllm_v1_multiprocessing"] = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    t0 = time.time()
    llm = LLM(model=path, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.6,
              max_model_len=16384, limit_mm_per_prompt={"image": 13}, enforce_eager=True, seed=0)
    out["load_s"] = round(time.time() - t0, 1)
    out["after_load"] = gpu_state()
    req = {"prompt_token_ids": compact, "multi_modal_data": {"image": [image]}}
    t0 = time.time()
    g = llm.generate([req], SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, logprobs=0))[0]
    out["greedy_s"] = round(time.time() - t0, 2)
    # step 2: each greedy token's log-probability, for the comparison with the Transformers stage
    out["greedy_tokens"] = list(g.outputs[0].token_ids)
    out["greedy_logprobs"] = [float(step[t].logprob) for t, step in zip(g.outputs[0].token_ids, g.outputs[0].logprobs or [])]
    s = llm.generate([req], SamplingParams(n=SAMPLES, temperature=0.8, top_p=0.92, max_tokens=MAX_NEW_TOKENS,
                                           seed=0))[0]
    out["vllm_prompt_tokens"] = len(g.prompt_token_ids)
    out["answers"] = answers(g.outputs[0].text, [o.text for o in s.outputs])


RUNNERS = {("qwen3vl", "hf"): run_qwen3vl_hf, ("qwen3vl", "vllm"): run_qwen3vl_vllm,
           ("cambrian", "hf"): run_cambrian_hf, ("cambrian", "vllm"): run_cambrian_vllm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["qwen3vl", "cambrian"], required=True)
    ap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", default=RECORDS)
    ap.add_argument("--record-id", default=RECORD_ID)
    args = ap.parse_args()
    out = {"kind": args.kind, "backend": args.backend, "model": args.model, "utc_start": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename, "env_python": sys.executable}
    try:
        import importlib.metadata as md

        out["versions"] = {p: md.version(p) for p in ("torch", "transformers", "vllm") if _has(md, p)}
        rec, messages, image, reference = load_case(args.records, args.record_id)
        out["case"] = {"record": rec["id"], "task_type": rec["task_type"], "split": rec["split"],
                       "image": rec["conversation_image_paths"][0], "image_size": list(image.size),
                       "reference_answer": reference, "format_example": rec.get("format_example")}
        RUNNERS[(args.kind, args.backend)](args.model, messages, image, out)
        out["ok"] = True
    except Exception as exc:
        out["ok"] = False
        out["error"] = repr(exc)[:600]
        out["traceback_tail"] = traceback.format_exc()[-4000:]
    out["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out.get(k) for k in ("kind", "backend", "ok", "error", "load_s")}, ensure_ascii=False))


def _has(md, p):
    try:
        md.version(p)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
