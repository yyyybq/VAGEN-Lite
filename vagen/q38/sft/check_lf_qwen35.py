"""q38: LLaMA-Factory main's SFT encoding of Qwen3.8-27B against the model's own chat template, CPU only
(port of model_matrix/sft/check_lf_qwen3vl.py; SFT_OPTIONS.md 2.3 and 7, TEMPLATE_CHECK.md, SMOKE_PLAN 2b).

Modes (the folder is one written by vagen.q38.sft.lf_data, whose build_manifest.json maps each record back to its line
of the 8B source file):

  tokens    a seeded sample of records.  LLaMA-Factory (template qwen3_8, enable_thinking false, mask_history false,
            SupervisedDatasetProcessor as the SFT job runs it) encodes the converted record; Qwen3.8's processor encodes
            the SOURCE record (not stripped: what the RL agent loop renders, the template trims it) with
            ``apply_chat_template(enable_thinking=False)``, the images where the <image> placeholders stand, as
            content blocks (gym_agent_loop_v1.attach_images).  Per record:
              ids_identical          LF ids == processor ids, one by one;
              labels                 the supervised tokens decode to every (stripped) answer + "<|im_end|>\\n" and
                                     nothing else, one span per assistant turn, labels == ids there;
              think_block            the 4 tokens before every span are <think> \\n\\n </think> \\n\\n (248068 271
                                     248069 271) and not supervised (TEMPLATE_CHECK 4);
              generation_prefix      for every assistant turn the processor's generation prompt (add_generation_prompt,
                                     enable_thinking False) of the turns before it is a token prefix of the full ids
                                     and ends where LF's span of that turn starts (what RL's prompt is);
              image tokens           <|image_pad|> runs per image == t*h*w/merge**2 of the processor's grid;
              no_reasoning_text      no "Reasoning effort" text, no "<think>" outside the empty blocks;
            and for the first --pixels records the pixel values and grids of LF's mm plugin == the processor's, and
            for the first --positions records the M-RoPE position ids of LF's SFT collator (model_type qwen3_5, a
            weightless model on the meta device) == the model's own get_rope_index on the processor's inputs.
  lengths   every record of every dataset of the folder: LF's untruncated length (cutoff 10**9), how many exceed the
            recipe's cutoff_len (16384), quantiles, the longest records.
  pipeline  LLaMA-Factory's own loader (get_train_args on the folder's train.yaml + get_dataset, max_samples N): the
            datasets library of the q38 env (5.0.1, DISABLE_VERSION_CHECK=1) loads, converts and tokenizes the first N
            records of each split; they must equal the direct encoding of those records.
  fsdp      a weightless model (meta device, bf16 as loaded), LLaMA-Factory's freeze rule (freeze_vision_tower true,
            freeze_multi_modal_projector false) and fp32 upcast of trainable weights; every class of the accelerate
            config's fsdp_transformer_layer_cls_to_wrap exists in the model, and every FSDP1 unit (the nearest wrapped
            ancestor of each parameter, or the root) holds one dtype; frozen / trainable parameter counts.

    cd VAGEN-Lite-q38 && DISABLE_VERSION_CHECK=1 OMP_NUM_THREADS=1 ... PYTHONPATH=$PWD nice -n 19 \\
      /mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python -B -m vagen.q38.sft.check_lf_qwen35 tokens \\
      --lf-dir LF --model MODEL --dataset trajectory_train --n 200 --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IGNORE = -100
THINK_IDS = [248068, 271, 248069, 271]  # <think> \n\n </think> \n\n (TEMPLATE_CHECK 4)
ASSISTANT_HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE = "qwen3_8"
CUTOFF = 16384


# ----------------------------------------------------------------------------------------------------------- inputs
def read_manifest(lf_dir: Path) -> Dict[str, Any]:
    return json.loads((lf_dir / "build_manifest.json").read_text())


def dataset_file(lf_dir: Path, name: str) -> Path:
    info = json.loads((lf_dir / "dataset_info.json").read_text())
    return lf_dir / info[name]["file_name"]


def read_lines(path: Path) -> List[str]:
    return [l for l in path.read_text().splitlines() if l.strip()]


def source_records(manifest: Dict[str, Any], name: str, lf_lines: int) -> Tuple[Path, List[int]]:
    """(the 8B source file, its line number for each line of the converted file)."""
    d = manifest["datasets"][name]
    sel = d["selected_lines"]
    lines = list(range(d["source_records"])) if sel == "all" else list(sel)
    if len(lines) != lf_lines:
        raise SystemExit(f"{name}: the manifest lists {len(lines)} records, the file has {lf_lines}")
    return Path(d["source"]), lines


# ----------------------------------------------------------------------------------------------------------- LF side
class LFEncoder:
    """LLaMA-Factory's tokenizer, processor, qwen3_8 template and SupervisedDatasetProcessor, as run_sft builds them."""

    def __init__(self, model: str, cutoff_len: int = CUTOFF, enable_thinking: bool = False):
        from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments, ModelArguments
        from llamafactory.model import load_tokenizer

        self.model_args = ModelArguments(model_name_or_path=model)
        tm = load_tokenizer(self.model_args)
        self.tokenizer, self.processor = tm["tokenizer"], tm["processor"]
        self.data_args = DataArguments(template=TEMPLATE, cutoff_len=cutoff_len, mask_history=False,
                                       enable_thinking=enable_thinking)
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.proc = SupervisedDatasetProcessor(template=self.template, tokenizer=self.tokenizer,
                                               processor=self.processor, data_args=self.data_args)

    def encode(self, rec: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        msgs = rec["messages"]
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else None
        turns = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"]
        return self.proc._encode_data_example(turns[:-1], turns[-1:], system, None, list(rec.get("images") or []),
                                              list(rec.get("videos") or []), [])

    def mm_inputs(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """The pixel values and grids the SFT collator gets from the qwen3_vl plugin (mm_plugin._get_mm_inputs)."""
        return self.template.mm_plugin._get_mm_inputs(list(rec.get("images") or []), [], [], self.processor)


# ----------------------------------------------------------------------------------------------------------- HF side
def to_blocks(messages: List[Dict[str, Any]], image_paths: List[str]):
    """Messages with each user turn's <image> placeholders as {"type": "image"} blocks (the RL agent loop's layout),
    and the PIL images in order."""
    from PIL import Image

    out, images, it = [], [], iter(image_paths)
    for m in messages:
        if m["role"] != "user":
            out.append({"role": m["role"], "content": m["content"]})
            continue
        content = []
        for seg in re.split(r"(<image>)", m["content"]):
            if seg == "<image>":
                content.append({"type": "image"})
                images.append(Image.open(next(it)).convert("RGB"))
            elif seg:
                content.append({"type": "text", "text": seg})
        out.append({"role": "user", "content": content})
    return out, images


def hf_encode(processor, messages, images, add_generation_prompt: bool = False):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         enable_thinking=False)
    out = processor(text=[text], images=images or None, return_tensors="pt")
    return text, out


def spans(labels: List[int]) -> List[List[int]]:
    runs, start = [], None
    for i, v in enumerate(list(labels) + [IGNORE]):
        if v != IGNORE and start is None:
            start = i
        elif v == IGNORE and start is not None:
            runs.append([start, i])
            start = None
    return runs


def image_runs(ids: List[int], pad: int) -> List[int]:
    return [len(m) for m in re.findall(r"P+", "".join("P" if t == pad else "." for t in ids))]


def check_record(enc: LFEncoder, conv: Dict[str, Any], src: Dict[str, Any], prefix: bool = True) -> Dict[str, Any]:
    tok, processor = enc.tokenizer, enc.processor
    ids, labels = enc.encode(conv)
    blocks, images = to_blocks(src["messages"], list(src.get("images") or []))
    hf_text, out = hf_encode(processor, blocks, images)
    hf_ids = out["input_ids"][0].tolist()
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    merge = processor.image_processor.merge_size
    grids = out["image_grid_thw"].tolist() if "image_grid_thw" in out else []
    want_per_image = [t * h * w // (merge * merge) for t, h, w in grids]
    first_diff = next((i for i, (a, b) in enumerate(zip(ids, hf_ids)) if a != b), None)
    if first_diff is None and len(ids) != len(hf_ids):
        first_diff = min(len(ids), len(hf_ids))
    answers = [m["content"].strip() for m in src["messages"] if m["role"] == "assistant"]
    sp = spans(labels)
    supervised = [t for t, lab in zip(ids, labels) if lab != IGNORE]
    think_ok = len(sp) == len(answers) and all(
        ids[s - 4:s] == THINK_IDS and all(l == IGNORE for l in labels[s - 4:s])
        and tok.decode(ids[s - 7:s]) == ASSISTANT_HEAD for s, _ in sp)
    span_text_ok = len(sp) == len(answers) and all(
        tok.decode(ids[s:e]) == a + "<|im_end|>\n" for (s, e), a in zip(sp, answers))
    r: Dict[str, Any] = {
        "assistant_turns": len(answers), "images": len(images), "tokens_lf": len(ids), "tokens_hf": len(hf_ids),
        "ids_identical": ids == hf_ids, "first_difference": first_diff,
        "image_tokens_per_image_lf": image_runs(ids, pad), "image_tokens_per_image_processor": want_per_image,
        "supervised_tokens": len(supervised), "supervised_spans": len(sp),
        "supervised_text_is_answers": tok.decode(supervised) == "".join(a + "<|im_end|>\n" for a in answers),
        "spans_are_answers": span_text_ok,
        "labels_equal_ids_where_supervised": all(lab == t for t, lab in zip(ids, labels) if lab != IGNORE),
        "think_block": think_ok,
        "no_reasoning_text": "Reasoning effort" not in hf_text
                             and hf_text.count("<think>") == len(answers) and tok.decode(ids).count("<think>") == len(answers),
    }
    r["image_counts_ok"] = r["image_tokens_per_image_lf"] == want_per_image
    if prefix:
        ok, ends = True, []
        msgs = src["messages"]
        a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for k, i in enumerate(a_idx):
            pb, pim = to_blocks(msgs[:i], list(src.get("images") or [])[: sum(m["content"].count("<image>") for m in msgs[:i] if m["role"] == "user")])
            _, pout = hf_encode(processor, pb, pim, add_generation_prompt=True)
            p_ids = pout["input_ids"][0].tolist()
            ends.append(len(p_ids))
            ok = ok and p_ids == hf_ids[: len(p_ids)] and k < len(sp) and sp[k][0] == len(p_ids)
        r["generation_prefix"] = ok
        r["generation_prompt_ends"] = ends
    else:
        r["generation_prefix"] = None
    if first_diff is not None:
        r["context"] = {"lf": tok.decode(ids[max(0, first_diff - 20):first_diff + 20]),
                        "hf": tok.decode(hf_ids[max(0, first_diff - 20):first_diff + 20])}
    r["ok"] = bool(r["ids_identical"] and r["image_counts_ok"] and r["supervised_text_is_answers"] and span_text_ok
                   and r["labels_equal_ids_where_supervised"] and think_ok and r["no_reasoning_text"]
                   and r["generation_prefix"] is not False)
    r["_hf"] = (out, hf_ids)
    return r


def check_pixels(enc: LFEncoder, conv: Dict[str, Any], hf_out) -> Dict[str, Any]:
    import torch

    lf = enc.mm_inputs(conv)
    same_grid = torch.equal(lf["image_grid_thw"], hf_out["image_grid_thw"])
    same_px = lf["pixel_values"].shape == hf_out["pixel_values"].shape and torch.equal(lf["pixel_values"], hf_out["pixel_values"])
    return {"pixels_identical": bool(same_px), "grid_identical": bool(same_grid),
            "pixel_shape": list(lf["pixel_values"].shape)}


_META_MODEL = None


def meta_model(model: str):
    """Qwen3_5ForConditionalGeneration on the meta device (no weights), bf16 as LF loads it."""
    global _META_MODEL
    if _META_MODEL is None:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText

        config = AutoConfig.from_pretrained(model)
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                _META_MODEL = AutoModelForImageTextToText.from_config(config)
        finally:
            torch.set_default_dtype(old)
    return _META_MODEL


def check_positions(enc: LFEncoder, model: str, ids: List[int], labels: List[int], conv: Dict[str, Any], hf_out) -> Dict[str, Any]:
    """M-RoPE positions of LF's SFT collator (as run_sft builds it) vs the model's own get_rope_index on the
    processor's inputs (what the forward pass computes when no position_ids are given)."""
    import torch
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask

    m = meta_model(model)
    coll = SFTDataCollatorWith4DAttentionMask(template=enc.template, model=m, pad_to_multiple_of=8,
                                              label_pad_token_id=IGNORE, block_diag_attn=False, neat_packing=False,
                                              attn_implementation="sdpa", compute_dtype=torch.bfloat16,
                                              tokenizer=enc.tokenizer, processor=enc.processor)
    feats = [{"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
              "images": list(conv.get("images") or []), "videos": [], "audios": []}]
    batch = coll(feats)
    lf_pos = batch["position_ids"][..., : len(ids)]
    ref_pos, _ = m.model.get_rope_index(input_ids=hf_out["input_ids"], mm_token_type_ids=hf_out["mm_token_type_ids"],
                                        image_grid_thw=hf_out.get("image_grid_thw"),
                                        attention_mask=hf_out["attention_mask"])
    # LF passes 4 rows (transformers issue 39400): row 0 is a copy of the temporal row and becomes the text model's
    # text_position_ids, which only reach create_causal_mask; rows 1-3 are the M-RoPE positions.  The model's own
    # forward (no position_ids) computes the 3 rows with get_rope_index and no text row.
    mrope = lf_pos[1:] if lf_pos.shape[0] == 4 else lf_pos
    same = bool(mrope.shape == ref_pos.shape and torch.equal(mrope.cpu(), ref_pos.cpu()))
    row0_is_t = bool(lf_pos.shape[0] == 4 and torch.equal(lf_pos[0], lf_pos[1]))
    # the text row changes nothing in the mask while an attention_mask is given (packed-sequence detection runs only
    # without one, transformers masking_utils): the causal mask with LF's text row == the mask without it
    from transformers.masking_utils import create_causal_mask

    cfg = m.config.get_text_config()
    cfg._attn_implementation = "sdpa"
    emb = torch.zeros(1, len(ids), 8, dtype=torch.bfloat16)
    am = batch["attention_mask"][:, : len(ids)]
    with_row = create_causal_mask(config=cfg, inputs_embeds=emb, attention_mask=am, past_key_values=None,
                                  position_ids=lf_pos[0])
    without = create_causal_mask(config=cfg, inputs_embeds=emb, attention_mask=am, past_key_values=None, position_ids=None)
    mask_same = (with_row is None and without is None) or (
        with_row is not None and without is not None and torch.equal(with_row, without))
    return {"positions_identical": same and row0_is_t and bool(mask_same), "mrope_rows_identical": same,
            "text_row_is_temporal_row": row0_is_t, "causal_mask_unchanged_by_text_row": bool(mask_same),
            "position_shape": list(lf_pos.shape), "collator_keys": sorted(batch.keys())}


def mode_tokens(a) -> Dict[str, Any]:
    lf_dir = Path(a.lf_dir)
    manifest = read_manifest(lf_dir)
    conv_lines = read_lines(dataset_file(lf_dir, a.dataset))
    src_path, src_idx = source_records(manifest, a.dataset, len(conv_lines))
    src_lines = read_lines(src_path)
    pick = sorted(random.Random(a.seed).sample(range(len(conv_lines)), min(a.n, len(conv_lines))))
    enc = LFEncoder(a.model)
    results, t0 = [], time.time()
    for j, i in enumerate(pick):
        conv, src = json.loads(conv_lines[i]), json.loads(src_lines[src_idx[i]])
        r = check_record(enc, conv, src, prefix=not a.no_prefix)
        out, hf_ids = r.pop("_hf")
        if j < a.pixels:
            r.update(check_pixels(enc, conv, out))
            r["ok"] = r["ok"] and r["pixels_identical"] and r["grid_identical"]
        if j < a.positions:
            ids, labels = enc.encode(conv)
            r.update(check_positions(enc, a.model, ids, labels, conv, out))
            r["ok"] = r["ok"] and r["positions_identical"]
        r.update(line=i + 1, source_line=src_idx[i] + 1)
        results.append(r)
        if (j + 1) % 25 == 0:
            print(f"[tokens] {j + 1}/{len(pick)} ok so far {sum(x['ok'] for x in results)} ({time.time() - t0:.0f} s)", flush=True)
    keys = ("ids_identical", "image_counts_ok", "supervised_text_is_answers", "spans_are_answers",
            "labels_equal_ids_where_supervised", "think_block", "no_reasoning_text", "generation_prefix",
            "pixels_identical", "grid_identical", "positions_identical")
    tally = {k: [sum(1 for r in results if r.get(k) is True), sum(1 for r in results if k in r and r[k] is not None)] for k in keys}
    return {"mode": "tokens", "lf_dir": str(lf_dir), "dataset": a.dataset, "model": a.model, "seed": a.seed,
            "records_checked": len(results), "records_ok": sum(r["ok"] for r in results), "tally_pass_of_checked": tally,
            "seconds": round(time.time() - t0, 1), "failures": [r for r in results if not r["ok"]][:20],
            "results": results, "ok": all(r["ok"] for r in results) and len(results) >= min(a.n, len(conv_lines))}


def mode_lengths(a) -> Dict[str, Any]:
    lf_dir = Path(a.lf_dir)
    info = json.loads((lf_dir / "dataset_info.json").read_text())
    enc = LFEncoder(a.model, cutoff_len=10 ** 9)
    report: Dict[str, Any] = {"mode": "lengths", "lf_dir": str(lf_dir), "model": a.model, "cutoff_len": a.cutoff, "datasets": {}}
    t0 = time.time()
    for name in info:
        lines = read_lines(lf_dir / info[name]["file_name"])
        if a.limit:
            lines = lines[: a.limit]
        lens, sup, rows = [], [], []
        for i, l in enumerate(lines):
            rec = json.loads(l)
            ids, labels = enc.encode(rec)
            lens.append(len(ids))
            sup.append(sum(1 for x in labels if x != IGNORE))
            rows.append((len(ids), i + 1, len(rec.get("images") or []), sum(1 for m in rec["messages"] if m["role"] == "assistant")))
            if (i + 1) % 1000 == 0:
                print(f"[lengths] {name} {i + 1}/{len(lines)} ({time.time() - t0:.0f} s)", flush=True)
        s = sorted(lens)
        q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))] if s else None
        report["datasets"][name] = {
            "records": len(lens), "over_cutoff": sum(1 for x in lens if x > a.cutoff), "max": max(lens) if lens else None,
            "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "mean": round(sum(lens) / len(lens), 1) if lens else None,
            "supervised_tokens_total": sum(sup), "tokens_total": sum(lens),
            "longest": [{"tokens": t, "line": ln, "images": im, "assistant_turns": at} for t, ln, im, at in sorted(rows, reverse=True)[:10]]}
    report["seconds"] = round(time.time() - t0, 1)
    report["ok"] = all(d["over_cutoff"] == 0 for d in report["datasets"].values())
    return report


def mode_pipeline(a) -> Dict[str, Any]:
    """LF's own loader on the folder's train.yaml (CPU: bf16 off, no output written but the cache folder given)."""
    import yaml
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    lf_dir = Path(a.lf_dir)
    cfg = yaml.safe_load((lf_dir / "train.yaml").read_text())
    cfg.update(model_name_or_path=a.model, max_samples=a.n, bf16=False, output_dir=a.scratch, overwrite_cache=True,
               preprocessing_num_workers=None, report_to="none")
    cfg.pop("flash_attn", None)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(cfg)
    tm = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tm["tokenizer"], data_args)
    t0 = time.time()
    dm = get_dataset(template, model_args, data_args, training_args, stage="sft", **tm)
    enc = LFEncoder(a.model)
    import datasets as hf_datasets

    out: Dict[str, Any] = {"mode": "pipeline", "lf_dir": str(lf_dir), "datasets_version": hf_datasets.__version__,
                           "template_enable_thinking": template.enable_thinking,
                           "template_preserve_thinking": template.preserve_thinking, "splits": {}}
    for split, name in (("train_dataset", cfg["dataset"]), ("eval_dataset", cfg.get("eval_dataset"))):
        ds = dm.get(split)
        if ds is None or name is None:
            continue
        lines = read_lines(dataset_file(lf_dir, name))
        same = 0
        for i in range(len(ds)):
            ids, labels = enc.encode(json.loads(lines[i]))
            same += int(list(ds[i]["input_ids"]) == ids and list(ds[i]["labels"]) == labels)
        out["splits"][split] = {"dataset": name, "rows": len(ds), "identical_to_direct_encoding": same,
                                "columns": list(ds.column_names)}
    out["seconds"] = round(time.time() - t0, 1)
    out["ok"] = bool(out["splits"]) and all(s["rows"] == min(a.n, len(read_lines(dataset_file(lf_dir, s["dataset"]))))
                                            and s["identical_to_direct_encoding"] == s["rows"] for s in out["splits"].values()) \
        and template.enable_thinking is False
    return out


def wrap_classes(accelerate_yaml: Path) -> List[str]:
    import yaml

    cfg = yaml.safe_load(accelerate_yaml.read_text())
    return [c.strip() for c in cfg["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"].split(",") if c.strip()]


def mode_fsdp(a) -> Dict[str, Any]:
    import torch
    from llamafactory.model.adapter import _setup_full_tuning
    from llamafactory.model.model_utils.visual import get_forbidden_modules
    from llamafactory.hparams import FinetuningArguments

    m = meta_model(a.model)
    fa = FinetuningArguments(finetuning_type="full", freeze_vision_tower=True, freeze_multi_modal_projector=False)
    forbidden = sorted(get_forbidden_modules(m.config, fa))
    _setup_full_tuning(m, fa, is_trainable=True, cast_trainable_params_to_fp32=True)
    classes = wrap_classes(Path(a.accelerate_config))
    present = {c: 0 for c in classes}
    owner: Dict[str, str] = {}
    for name, mod in m.named_modules():
        if type(mod).__name__ in present:
            present[type(mod).__name__] += 1
    units: Dict[str, set] = {}
    wrapped = [(n, mod) for n, mod in m.named_modules() if type(mod).__name__ in present]
    wrapped_names = sorted((n for n, _ in wrapped), key=len, reverse=True)
    for pname, p in m.named_parameters():
        unit = next((w for w in wrapped_names if w and (pname.startswith(w + "."))), "<root>")
        owner[pname] = unit
        units.setdefault(unit, set()).add(str(p.dtype))
    mixed = {u: sorted(d) for u, d in units.items() if len(d) > 1}
    frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    trainable_dtypes = sorted({str(p.dtype) for p in m.parameters() if p.requires_grad})
    frozen_dtypes = sorted({str(p.dtype) for p in m.parameters() if not p.requires_grad})
    merger = [n for n, p in m.named_parameters() if ".visual.merger." in n]
    root_params = sorted(n for n, u in owner.items() if u == "<root>")
    return {"mode": "fsdp", "model": a.model, "accelerate_config": a.accelerate_config, "wrap_classes": present,
            "missing_classes": [c for c, k in present.items() if k == 0], "forbidden_modules": forbidden,
            "units": len(units), "mixed_dtype_units": mixed, "root_params": root_params,
            "frozen_params": frozen, "trainable_params": trainable, "trainable_dtypes": trainable_dtypes,
            "frozen_dtypes": frozen_dtypes, "merger_trainable": all(dict(m.named_parameters())[n].requires_grad for n in merger) and bool(merger),
            "keep_in_fp32_modules": getattr(m, "_keep_in_fp32_modules", None),
            "ok": not mixed and all(k > 0 for k in present.values()) and trainable_dtypes == ["torch.float32"]
                  and frozen_dtypes == ["torch.bfloat16"] and bool(merger)}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("--lf-dir", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--dataset", default="trajectory_train")
    t.add_argument("--n", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--pixels", type=int, default=20)
    t.add_argument("--positions", type=int, default=20)
    t.add_argument("--no-prefix", action="store_true")
    t.add_argument("--out", required=True)
    l = sub.add_parser("lengths")
    l.add_argument("--lf-dir", required=True)
    l.add_argument("--model", required=True)
    l.add_argument("--cutoff", type=int, default=CUTOFF)
    l.add_argument("--limit", type=int, default=0)
    l.add_argument("--out", required=True)
    p = sub.add_parser("pipeline")
    p.add_argument("--lf-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--scratch", required=True, help="LF's output_dir for get_train_args (nothing is trained)")
    p.add_argument("--out", required=True)
    f = sub.add_parser("fsdp")
    f.add_argument("--model", required=True)
    f.add_argument("--accelerate-config", required=True)
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = {"tokens": mode_tokens, "lengths": mode_lengths, "pipeline": mode_pipeline, "fsdp": mode_fsdp}[a.mode](a)
    report["env"] = {"DISABLE_VERSION_CHECK": os.environ.get("DISABLE_VERSION_CHECK")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    brief = {k: v for k, v in report.items() if k not in ("results", "failures", "root_params")}
    print(json.dumps(brief, indent=1, ensure_ascii=False, default=str)[:6000])
    print("ok", report["ok"])


if __name__ == "__main__":
    main(sys.argv[1:])
