#!/usr/bin/env python3
"""Train/eval processor parity (WP-TRAIN; plan 3.4; DECISIONS B5, B16, C6).

    PY_TRAIN vpd/train/n2_train_vpd.py --start Q3S --data D.jsonl --dry-run --limit 3 --expect-grid 24x40 --out DIR
    PY_Q3    vpd/train/parity_check.py --dry-run-json DIR/dry_run.json [--out DIR/parity.json]

Re-renders the dry run's first demos (dry_run.json['parity']) with the RUNNING interpreter's processor (meant for
PY_Q3: transformers 4.57.6, the eval's vLLM env) and the same start, processor folder and max_pixels cap, and asserts
equal sha256 of: the decoded RGB images, the rendered chat text, input_ids, labels, image_grid_thw and pixel_values,
and of every step's prompt prefix (context + generation prompt; B5).  It also renders the text from OpenAI-style
messages (system / assistant as plain strings, user as [text, image] parts: what vLLM's chat endpoint receives) and
asserts it equals the training text.  Mismatch details are printed (first differing character / token, grid,
pixel max |diff| against the saved first image).  Exit 0 = PASS, 1 = FAIL.

Fallback when it FAILS (plan 3.4, DECISIONS C6): train with PY_Q3 instead of PY_TRAIN (torch 2.8, sdpa), and
record that in the run's meta.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from vpd.train import n2_train_vpd as T  # noqa: E402

FALLBACK = "FALLBACK: train with PY_Q3 (eval-external-q3, transformers 4.57.6) instead of PY_TRAIN and record it"


def oai_messages(messages: list) -> list:
    """The chat as an OpenAI-format client sends it: plain-string system/assistant content, user content as
    text/image parts in the placeholder order (vLLM turns image_url parts into image placeholders)."""
    out = []
    for m in messages:
        if m["role"] != "user" or T.IMAGE not in m["content"]:
            out.append({"role": m["role"], "content": m["content"]})
            continue
        out.append({"role": "user", "content": T.chat_messages([m])[0]["content"]})
    return out


def first_diff(a, b) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def check_entry(processor, rec: dict, want: dict, history_k: int, blind: bool, dry_dir: Path) -> list:
    """The list of mismatches (empty = equal) between the dry run's entry and this interpreter's rendering."""
    import torch

    bad = []
    s = T.samples(rec, history_k, blind)[want.get("sample", 0)]
    got_rgb = [T.sha256_bytes(im.tobytes()) for im in T.load_images(s["images"])]
    if got_rgb != want["image_rgb_sha256"]:
        idx = [i for i, (x, y) in enumerate(zip(got_rgb, want["image_rgb_sha256"])) if x != y]
        bad.append(f"decoded RGB images differ at {idx[:5]} (PIL/libjpeg decode, not the processor)")
    inputs, labels, spans, text = T.encode(processor, s)
    if T.sha256_bytes(text.encode("utf-8")) != want["text_sha256"]:
        bad.append(f"chat text differs ({len(text)} vs {want['text_chars']} chars)")
    oai = processor.apply_chat_template(oai_messages(s["messages"]), tokenize=False, add_generation_prompt=False)
    if oai != text:
        i = first_diff(oai, text)
        bad.append(f"OpenAI-format text != training text at char {i}: {oai[max(0, i - 40):i + 40]!r} vs "
                   f"{text[max(0, i - 40):i + 40]!r}")
    ids = inputs["input_ids"]
    if T.sha256_ids(ids) != want["input_ids_sha256"]:
        bad.append(f"input_ids differ ({int(ids.shape[1])} vs {want['n_tokens']} tokens)")
    if T.sha256_ids(labels) != want["labels_sha256"]:
        bad.append(f"labels differ; spans {spans[:3]}... vs {want['spans'][:3]}...")
    grid = None if inputs.get("image_grid_thw") is None else inputs["image_grid_thw"].tolist()
    if grid != want["grid"]:
        bad.append(f"image_grid_thw differ: {grid[:2] if grid else grid} vs "
                   f"{want['grid'][:2] if want['grid'] else want['grid']}")
    pv = inputs.get("pixel_values")
    got_pv = None if pv is None else T.sha256_tensor(pv)
    if got_pv != want["pixel_values_sha256"]:
        detail = f"pixel_values differ (shape {None if pv is None else list(pv.shape)} vs {want['pixel_values_shape']})"
        px0 = want.get("px0_file")
        if px0 and pv is not None and (dry_dir / px0).is_file():
            ref = torch.load(dry_dir / px0, map_location="cpu")
            cur = pv[:ref.shape[0]].float()
            if cur.shape == ref.shape:
                detail += f"; first image max|diff| {float((cur - ref).abs().max()):.3g}"
        bad.append(detail)
    want_pref = {p["turn"]: p for p in want.get("prefixes", [])}
    for p, pin, ptext in T.prompt_prefixes(processor, rec, history_k, blind):
        w = want_pref.pop(p["turn"], None)
        if w is None:
            bad.append(f"prefix turn {p['turn']} not in the dry run")
            continue
        n = int(pin["input_ids"].shape[1])
        if T.sha256_ids(pin["input_ids"]) != w["input_ids_sha256"] or n != w["prompt_tokens"]:
            bad.append(f"prompt prefix at turn {p['turn']} differs ({n} vs {w['prompt_tokens']} tokens)")
    if want_pref:
        bad.append(f"dry-run prefixes not re-rendered: turns {sorted(want_pref)}")
    return bad


def main(argv=None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run-json", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="override the dry run's data path")
    ap.add_argument("--out", type=Path, default=None, help="report (default: parity.json next to dry_run.json)")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    from vpd import starts

    dry = json.loads(args.dry_run_json.read_text())
    dry_dir = args.dry_run_json.parent
    data = args.data or Path(dry["data"])
    recs = {r.get("id"): r for r in T.load_records(data)}
    processor = starts.load_processor(dry["start"], path=dry.get("model_path"), max_pixels=dry["max_pixels"])
    problems = {}
    if processor.vpd_processor_dir != dry["processor_dir"]:
        problems["_processor_dir"] = [f"processor folder {processor.vpd_processor_dir} != {dry['processor_dir']}"]
    if T.sha256_file(data) != dry["data_sha256"]:
        problems["_data"] = [f"{data} sha256 differs from the dry run's data"]
    for want in dry["parity"]:
        rec = recs.get(want["id"])
        if rec is None:
            problems[want["id"]] = ["demo not found in the data"]
            continue
        bad = check_entry(processor, rec, want, dry["history_k"], dry["blind"], dry_dir)
        if bad:
            problems[want["id"]] = bad
    here = T.versions()
    ok = not problems and bool(dry["parity"])
    rep = {"verdict": "PASS" if ok else "FAIL", "dry_run": str(args.dry_run_json), "demos": [w["id"] for w in
           dry["parity"]], "train_versions": dry.get("versions"), "check_versions": here,
           "processor_dir": processor.vpd_processor_dir, "max_pixels": dry["max_pixels"],
           "prefixes_checked": sum(len(w.get("prefixes", [])) for w in dry["parity"]), "problems": problems,
           "fallback": None if ok else FALLBACK}
    out = args.out or dry_dir / "parity.json"
    out.write_text(json.dumps(rep, indent=1) + "\n")
    for k, v in problems.items():
        for line in v:
            print(f"[parity] MISMATCH {k}: {line}", flush=True)
    print(f"[parity] {rep['verdict']}: {len(dry['parity'])} demos, {rep['prefixes_checked']} prompt prefixes; "
          f"train transformers {(dry.get('versions') or {}).get('transformers')} vs here {here.get('transformers')}"
          f" -> {out}", flush=True)
    if not ok:
        print(f"[parity] {FALLBACK}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
