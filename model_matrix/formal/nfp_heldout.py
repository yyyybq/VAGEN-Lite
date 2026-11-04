"""Decision 48 甲 (the user, 2026-11-04: "都按甲执行"): a checkpoint's next-frame prediction head on a fixed held-out
set, so checkpoints compare on the same frames (the training batches move with the policy's own episodes, and csnfp's
per-update statistics were lost: STATUS 1zq).

The set: the formal SFT's validation conversations (lf_cformal/trajectory_val.jsonl: 1,293 episodes of 1 to 10
frames, 2,118 consecutive pairs; the 327 with one frame have none).  Each episode is one sequence as the RL trainer's
concat path holds one: the chat template over the whole conversation with one <image> per frame
(CambrianProcessorWrapper), every <image> expanded to 756 <|image_pad|> positions (cambrian_concat.expand_sample),
positions 0..n-1.  The checkpoint's adapter (bf16, eval mode, no gradient) embeds it as its training forward does (the
alias as image positions, _embed_multimodal_batch keeping the frames' SigLIP features), its language model gives the
last hidden states, and every consecutive pair is scored as in training (cambrian_nfp.concat_nfp_pairs /
concat_nfp_loss, the config's weights): the head's mean squared error and cosine distance to the next frame's 8 x 8
SigLIP grid, and the same for the copy baseline (the current frame's grid as the prediction).  Unlike training, the
frames and the answers between them are the SFT trajectories', not the policy's own.

    python -m model_matrix.formal.nfp_heldout --model CKPT/actor/huggingface --out OUT [--limit N]

OUT/pairs.jsonl: one line per pair (episode, k: frame k predicts frame k + 1, the answer between them, mse, cos,
copy_mse, copy_cos, and loss / copy_loss weighted as in training); OUT/summary.json: the counts, the means over the
pairs, the share of pairs where the head's loss is below the copy baseline's, quantiles, and the checks (the head's
and a vision-tower weight are the checkpoint's: a head left at its initialisation would score as an untrained one).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

TRAJ_VAL = Path("/mnt/umm/users/yinbaiqiao/models_runs/chains/m_cambrian7b_formal_sft_20261026/lf_cformal/"
                "trajectory_val.jsonl")
IMAGE_TOKEN_ID = 151665  # the compact <image> (CambrianProcessorWrapper)
IMAGE_PAD_ID = 151655  # <|image_pad|>, the adapter's image_token_alias
TOKENS_PER_IMAGE = 756  # 27 x (27 + 1)
KEYS = ("mse", "cos", "copy_mse", "copy_cos", "loss", "copy_loss")
CHECKED = ("model.nfp_head.2.weight", "model.vision_tower_aux_list.0.vision_tower.vision_model.embeddings."
           "patch_embedding.weight")


def expand(ids: Sequence[int], image_id: int = IMAGE_TOKEN_ID, alias: int = IMAGE_PAD_ID,
           per_image: int = TOKENS_PER_IMAGE) -> List[int]:
    """The compact ids with every <image> as per_image alias positions (cambrian_concat's expansion)."""
    out: List[int] = []
    for t in ids:
        out.extend([alias] * per_image if t == image_id else [int(t)])
    return out


def episode_ids(processor, messages) -> List[int]:
    """One conversation as compact ids: the chat template over all its turns (the answers included, no generation
    prompt), one <image> token per frame."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return processor.tokenizer.encode(text, add_special_tokens=False)


def answers(messages) -> List[str]:
    """The assistant answers in order: answer k comes between frame k and frame k + 1."""
    return [m["content"] for m in messages if m.get("role") == "assistant"]


def score_episode(model, ids, pixel_values) -> List[Dict[str, float]]:
    """The statistics of every consecutive frame pair of one expanded episode (ids (1, L) with the alias, pixel_values
    (frames, 3, H, W)), as the adapter's training forward computes its loss (cambrian_register), one pair at a time."""
    import torch

    from vagen.models import cambrian_nfp as N

    cfg = model.config
    side = int(cfg.si_token_len ** 0.5)
    newline = getattr(cfg, "mm_use_im_newline_token", True)
    per_image = side * (side + 1) if newline else side * side
    mse_w = float(getattr(cfg, "nfp_mse_loss_weight", 0.1))
    cos_w = float(getattr(cfg, "nfp_cosine_loss_weight", 0.1))
    mapped = ids.masked_fill(ids == int(cfg.image_token_alias), N.IMAGE_TOKEN_INDEX)
    pos = torch.arange(ids.shape[1], device=ids.device)[None]
    embeds, raw = model._embed_multimodal_batch(mapped, pixel_values, return_raw=True)
    hidden = model.model(inputs_embeds=embeds, position_ids=pos, use_cache=False, return_dict=True)[0]
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    rows = []
    for pair in N.concat_nfp_pairs(mapped, pos, per_image):
        loss, s = N.concat_nfp_loss(hidden, [pair], raw, model.model.nfp_head, side=side, newline=newline,
                                    mse_weight=mse_w, cos_weight=cos_w)
        rows.append({**{k: v for k, v in s.items() if k != "pairs"}, "loss": float(loss),
                     "copy_loss": mse_w * s["copy_mse"] + cos_w * s["copy_cos"]})
    return rows


def quantiles(values: Sequence[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> Dict[str, float]:
    v = sorted(values)
    return {f"q{int(q * 100)}": v[min(len(v) - 1, int(q * len(v)))] for q in qs} if v else {}


def summarize(rows: List[dict], episodes: int, one_frame: int) -> dict:
    """The counts, the means over the pairs and over the episodes (each episode's pairs averaged first, as one
    training micro-batch holds one), the head against the copy baseline, the quantiles."""
    out = {"episodes": episodes, "one_frame_episodes": one_frame, "pairs": len(rows)}
    if not rows:
        return out
    out["mean"] = {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}
    by_ep: Dict[int, List[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["episode"], []).append(r)
    out["episode_mean"] = {k: sum(sum(r[k] for r in rs) / len(rs) for rs in by_ep.values()) / len(by_ep)
                           for k in KEYS}
    out["head_below_copy"] = {k: sum(r[k] < r[f"copy_{k}"] for r in rows) / len(rows) for k in ("loss", "mse", "cos")}
    still = [r for r in rows if r["copy_mse"] < 1e-6]  # the frame did not change (e.g. only blocked moves)
    out["unchanged_frame_pairs"] = len(still)
    out["quantiles"] = {k: quantiles([r[k] for r in rows]) for k in ("loss", "copy_loss")}
    return out


def weight_checks(params: Dict[str, "object"], path: Path, keys: Iterable[str] = CHECKED) -> Dict[str, object]:
    """Whether the loaded parameters ``keys`` equal the checkpoint's tensors (after the load's cast); None where the
    model or the checkpoint has no such key."""
    import torch
    from safetensors import safe_open

    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    out: Dict[str, object] = {}
    for key in keys:
        if key not in params or key not in index:
            out[key] = None
            continue
        got = params[key].detach().cpu()
        with safe_open(str(path / index[key]), framework="pt") as f:
            want = f.get_tensor(key).to(got.dtype)
        out[key] = bool(torch.equal(got, want))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="a checkpoint's actor/huggingface (with model.nfp_head)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=TRAJ_VAL)
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes (a check)")
    a = ap.parse_args()

    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    import vagen.models.cambrian_register  # noqa: F401  (registers the adapter; needs CAMBRIAN_SRC)
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vagen.models.cambrian_processor import CambrianProcessorWrapper

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = CambrianProcessorWrapper(AutoTokenizer.from_pretrained(a.model))
    if proc.image_token_id != IMAGE_TOKEN_ID:
        raise SystemExit(f"<image> is token {proc.image_token_id}, not {IMAGE_TOKEN_ID}")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    if not getattr(model.config, "model_line_nfp", False) or not hasattr(model.model, "nfp_head"):
        raise SystemExit("no model-line NFP head in this checkpoint (config model_line_nfp)")
    checks = weight_checks(dict(model.named_parameters()), a.model)
    if checks[CHECKED[0]] is not True:  # the head must be the trained one; the vision tower is only recorded
        raise SystemExit(f"the loaded head is not the checkpoint's: {checks}")
    model.to("cuda:0").eval()
    load_s = time.time() - t0

    rows: List[dict] = []
    episodes = one_frame = 0
    with open(a.data) as src, open(a.out / "pairs.jsonl", "w") as dst:
        for i, line in enumerate(src):
            if a.limit is not None and i >= a.limit:
                break
            r = json.loads(line)
            ids = episode_ids(proc, r["messages"])
            frames = sum(t == IMAGE_TOKEN_ID for t in ids)
            if frames != len(r["images"]):
                raise SystemExit(f"episode {i}: {frames} <image> tokens, {len(r['images'])} frames")
            episodes += 1
            if frames < 2:
                one_frame += 1
                continue
            images = []
            for p in r["images"]:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            pix = proc.preprocess_images(images)["pixel_values"].to("cuda:0", dtype=torch.bfloat16)
            with torch.no_grad():
                scored = score_episode(model, torch.tensor([expand(ids)], device="cuda:0"), pix)
            if len(scored) != frames - 1:
                raise SystemExit(f"episode {i}: {len(scored)} pairs scored for {frames} frames")
            said = answers(r["messages"])
            for k, s in enumerate(scored):
                row = {"episode": i, "k": k, "answer": said[k], **s}
                rows.append(row)
                dst.write(json.dumps(row) + "\n")
    summary = {"model": str(a.model), "data": str(a.data), "limit": a.limit, "checks": checks,
               "attn_implementation": getattr(model.config, "_attn_implementation", None),
               "load_s": round(load_s, 1), "score_s": round(time.time() - t0 - load_s, 1),
               **summarize(rows, episodes, one_frame)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary.get(k) for k in ("episodes", "pairs", "mean", "head_below_copy", "checks")}))


if __name__ == "__main__":
    main()
