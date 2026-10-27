#!/usr/bin/env python3
"""T2's short run, checked (CPU, the StarVLA environment's Python; downstream/train/run_t2_short.sh calls it).

The short run saves at step S (its first save point) a weights-only checkpoint, written with LoRA merged, and a full
state, where LoRA stays apart.  This compares the two at S:
  merge             every LoRA layer's merged weight equals W + alpha / rank * B A computed from the full state
                    (float32, then the weight's dtype), to within one bf16 rounding step (the training process
                    multiplies B A on the GPU, this on the CPU)
  others            every other tensor of the checkpoint equals the full state's
  trained           the B matrices are no longer zero (B starts at zero: LoRA learned something)
  plain layout      the checkpoint's names, shapes and dtypes are those of a plain StarVLA checkpoint (PLAIN, e.g.
                    DS-B-030-s1's final model), and so are the final model's
plus the training speed (median timing/model after the first 20 logged steps, metrics.jsonl) and the GPU memory peak
(gpu.csv).  Writes EXP/lora_check.json; exit 1 when a check fails.

    python downstream/train/lora_check.py --exp EXP --plain PLAIN_CKPT
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_weights as X  # noqa: E402


def layout(sd) -> dict:
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items()}


def gpu_peak(path: Path):
    peak = None
    if path.is_file():
        for row in csv.reader(path.read_text(errors="replace").replace("\0", "").splitlines()):
            try:
                mib = int(row[2].split()[0])
            except (IndexError, ValueError):
                continue
            peak = mib if peak is None else max(peak, mib)
    return peak


def speed(path: Path, skip: int = 20):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []
    t = [r["timing/model"] for r in rows if "timing/model" in r][skip:]
    return {"logged": len(rows), "median_model_seconds": round(statistics.median(t), 4) if t else None,
            "mean_model_seconds": round(statistics.mean(t), 4) if t else None}


def check(exp: Path, plain_path: Path, scale: float = None) -> dict:
    import torch

    run = exp / "run"
    states = sorted((p for p in (run / "full_state").glob("steps_*") if p.name[6:].isdigit()), key=lambda p: int(p.name[6:]))
    if not states:
        raise FileNotFoundError(f"no full state in {run / 'full_state'}")
    fs = states[0]
    step = int(fs.name[6:])
    if scale is None:
        import yaml

        lora = yaml.safe_load((exp / "config.yaml").read_text())["trainer"]["lora"]
        scale = float(lora["alpha"]) / int(lora["rank"])
    module = X.load_module(X.model_states_file(fs))
    merged = X.load_plain(run / "checkpoints" / f"steps_{step}_pytorch_model.pt")
    layers = sorted(k[: -len(".lora_A")] for k in module if k.endswith(".lora_A"))
    max_diff, max_b, within, differing = 0.0, 0.0, True, 0
    for p in layers:
        w, a, b = module[p + ".weight"], module[p + ".lora_A"], module[p + ".lora_B"]
        want = (w.float() + scale * (b.float() @ a.float())).to(w.dtype).float()
        diff = (merged[p + ".weight"].float() - want).abs()
        max_diff = max(max_diff, float(diff.max()))
        within = within and bool((diff <= want.abs() * 2.0 ** -7).all())
        differing += int((diff > 0).sum())
        max_b = max(max_b, float(b.float().abs().max()))
    lora_keys = {p + s for p in layers for s in (".lora_A", ".lora_B")}
    merged_names = {p + ".weight" for p in layers}
    others = [k for k in merged if k not in merged_names]
    others_equal = all(torch.equal(merged[k], module[k]) for k in others)
    plain = layout(X.load_plain(plain_path))
    final = run / "final_model" / "pytorch_model.pt"
    res = {"step": step, "full_state": str(fs), "lora_layers": len(layers), "scale": scale,
           "max_merge_diff": max_diff, "merge_within_bf16_step": within, "merge_elements_differing": differing,
           "max_abs_B": max_b, "others_equal": others_equal,
           "names_are_full_state_minus_lora": set(merged) == set(module) - lora_keys,
           "plain_layout": layout(merged) == plain,
           "final_plain_layout": layout(X.load_plain(final)) == plain if final.is_file() else None,
           "speed": speed(run / "metrics.jsonl"), "gpu_peak_mib": gpu_peak(exp / "gpu.csv")}
    res["ok"] = bool(layers) and within and max_b > 0 and others_equal and res["names_are_full_state_minus_lora"] \
        and res["plain_layout"] and res["final_plain_layout"] is not False
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--plain", type=Path, required=True, help="a plain StarVLA checkpoint of the same model")
    args = ap.parse_args(argv)
    res = check(args.exp, args.plain)
    (args.exp / "lora_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
