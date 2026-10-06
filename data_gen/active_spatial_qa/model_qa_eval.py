#!/usr/bin/env python3
"""Frozen VLM Yes/No inference over one fixed observable bank."""
from __future__ import annotations
import argparse, base64, io, json, os
from pathlib import Path
from PIL import Image


def _image_data_url(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def load_predictions(path: Path) -> dict:
    """Read completed prediction rows. A truncated final line is ignored."""
    done = {}
    if not path.exists() or path.stat().st_size == 0:
        return done
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
        sid = str(rec.get("sample_id") or "")
        if sid and "prediction" in rec:
            done[sid] = rec
    return done


def select_eligible(rows, eligible_ids_path: str | None, limit: int):
    eligible = [
        r for r in rows
        if r.get("observability_validity") == "valid" and r.get("public_observation", {}).get("image_path")
    ]
    if eligible_ids_path:
        payload = json.loads(Path(eligible_ids_path).read_text())
        order = [str(x) for x in payload["eligible_ids"]]
        by_id = {str(r["sample_id"]): r for r in eligible}
        missing = [sid for sid in order if sid not in by_id]
        if missing:
            raise SystemExit("eligible ids missing from bank: n=%s example=%s" % (len(missing), missing[:3]))
        eligible = [by_id[sid] for sid in order]
    if limit:
        eligible = eligible[:limit]
    return eligible


def merge_records(eligible, existing: dict, new_by_id: dict) -> list:
    ordered = []
    seen = set()
    for row in eligible:
        sid = str(row["sample_id"])
        if sid in new_by_id:
            ordered.append(new_by_id[sid])
        elif sid in existing:
            ordered.append(existing[sid])
        else:
            continue
        seen.add(sid)
    for sid, rec in existing.items():
        if sid not in seen:
            ordered.append(rec)
    return ordered


def write_jsonl_atomic(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _hf_generate(checkpoint: str, pending: list, args) -> dict:
    """Run one-image-at-a-time Qwen2.5-VL inference without vLLM/Triton."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    generated = {}
    for row in pending:
        image_path = Path(row["public_observation"]["image_path"])
        if not image_path.is_absolute():
            image_path = Path(args.bank).parent / image_path
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": row["question"]},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=8)
        prompt_len = inputs["input_ids"].shape[1]
        text = processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
        generated[str(row["sample_id"])] = {
            "sample_id": row["sample_id"],
            "prediction": text,
            "model_identity": args.checkpoint,
            "decode": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8, "backend": "transformers"},
        }
    return generated


def run(args):
    rows = [json.loads(x) for x in Path(args.bank).read_text().splitlines() if x.strip()]
    eligible = select_eligible(rows, args.eligible_ids, args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not eligible:
        if not args.resume:
            out.write_text("")
        summary = {"bank": args.bank, "checkpoint": args.checkpoint, "candidate_count": len(rows), "eligible_count": 0, "predictions": 0, "status": "EMPTY_BLOCKED"}
        Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return summary
    existing = load_predictions(out) if args.resume else {}
    pending = [r for r in eligible if str(r["sample_id"]) not in existing]
    if args.resume and not pending:
        summary = {
            "bank": args.bank,
            "checkpoint": args.checkpoint,
            "candidate_count": len(rows),
            "eligible_count": len(eligible),
            "predictions": len(existing),
            "pending": 0,
            "status": "SKIPPED_COMPLETE",
        }
        Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return summary

    if args.backend == "transformers":
        new_by_id = _hf_generate(args.checkpoint, pending, args)
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.checkpoint,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=4096,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8)
        messages = []
        for row in pending:
            image_path = Path(row["public_observation"]["image_path"])
            if not image_path.is_absolute():
                image_path = Path(args.bank).parent / image_path
            image = Image.open(image_path).convert("RGB")
            messages.append([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                {"type": "text", "text": row["question"]},
            ]}])
        outputs = llm.chat(messages, sampling_params=sampling, use_tqdm=True)
        new_by_id = {}
        for row, result in zip(pending, outputs):
            text = result.outputs[0].text if result.outputs else ""
            sid = str(row["sample_id"])
            new_by_id[sid] = {"sample_id": row["sample_id"], "prediction": text,
                              "model_identity": args.checkpoint,
                              "decode": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8}}
    records = merge_records(eligible, existing if args.resume else {}, new_by_id)
    write_jsonl_atomic(out, records)
    status = "RESUMED" if existing else "OK"
    summary = {
        "bank": args.bank,
        "checkpoint": args.checkpoint,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "predictions": len(records),
        "pending": len(pending),
        "status": status,
    }
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eligible-ids", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
