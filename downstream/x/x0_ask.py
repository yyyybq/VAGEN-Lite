#!/usr/bin/env python3
"""X0: ask one start model the questions zero-shot (greedy, one image and one two-choice question at a time).

    python downstream/x/x0_ask.py --start B --questions Q.jsonl --out answers_B.jsonl [--limit 20]

The model is read by path from downstream/starts.json (never copied); its processor comes from the same folder, or
from B's when the folder has none.  Writes one line per question {"id", "reply", "letter"} (letter: A / B / null,
x0_qa.parse_letter) and skips ids already in OUT, so a rerun only asks what is missing.  Run with the StarVLA
environment's Python (transformers with Qwen2.5-VL) on one GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
sys.path.insert(0, str(HERE))
from x0_qa import parse_letter  # noqa: E402

MAX_NEW_TOKENS = 16


def start_path(name: str) -> str:
    return json.loads((DOWNSTREAM / "starts.json").read_text())[name]["path"]


def load(model_dir: str, fallback: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    try:
        processor = AutoProcessor.from_pretrained(model_dir)
    except Exception:  # a training checkpoint without its processor files: the base model's
        processor = AutoProcessor.from_pretrained(fallback)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda").eval()
    return processor, model


def ask(processor, model, image_path: str, prompt: str) -> str:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="a name in downstream/starts.json")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="ask only the first N questions (a smoke test)")
    args = ap.parse_args(argv)
    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    for q in questions:
        q.setdefault("id", f"{q['task']}_s{q['seed']}_{q['camera']}_{q['kind']}_{q['a']['key']}_{q['b']['key']}")
    if args.limit:
        questions = questions[:args.limit]
    out = Path(args.out)
    done = set()
    if out.is_file():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    t0 = time.time()
    processor, model = load(start_path(args.start), start_path("B"))
    print(f"[x0_ask] {args.start}: loaded in {time.time() - t0:.0f} s; {len(questions) - len(done)} to ask", flush=True)
    with open(out, "a") as f:
        for q in questions:
            if q["id"] in done:
                continue
            reply = ask(processor, model, q["image_path"], q["prompt"])
            f.write(json.dumps({"id": q["id"], "reply": reply, "letter": parse_letter(reply)}) + "\n")
            f.flush()
    print(f"[x0_ask] {args.start}: done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
