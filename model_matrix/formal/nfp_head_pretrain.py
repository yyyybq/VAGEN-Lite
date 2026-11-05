"""Decision 54 丙 (the user, 2026-11-05: "54 丙；55，甲"): Cambrian-S's next-frame prediction head trained alone, the
model frozen, on the formal SFT's training frame pairs, before an RL arm starts from it.

Why: the three RL arms with the prediction loss (csnfp, csnfp01, csnfp01lr) started from a freshly initialised head,
which on the held-out pairs never reached the average-frame baseline (STATUS 1zs-1zx).  Whether the fresh head is why
they drifted early is what the RL arm from this head tests.

The pairs: lf_cformal/trajectory_train.jsonl (9,802 episodes, 15,441 consecutive frame pairs), each episode one
sequence as nfp_heldout builds it (the RL trainer's concat path).  The SFT model (checkpoint-770, bf16, eval mode, no
gradient) gives every pair the head's input -- the last hidden states at the current frame's 27 x 27 patch positions,
average-pooled to 8 x 8 -- and its target, the next frame's 8 x 8 SigLIP grid, exactly as cambrian_nfp.concat_nfp_loss
takes them (pair_tensors).  They stay in GPU memory; nothing is cached on disk.  A new head (cambrian_nfp.build_head,
its initialisation, a fixed seed) is trained on them with the training loss (0.1 x MSE + 0.1 x cosine distance): AdamW,
mini-batches of pairs, a cosine learning-rate decay; 5% of the training episodes (a fixed hash of the episode's index)
are held back to pick the epoch.  The held-out set (trajectory_val.jsonl, 2,118 pairs, the set every RL checkpoint's
head was scored on) is scored once at the end, with the copy and average-frame baselines, in nfp_heldout's format; the
gate is the user's: the head's mean loss below the average-frame baseline's.

    python -m model_matrix.formal.nfp_head_pretrain --out OUT [--limit N] [--epochs E] [--lr LR] ...

OUT/head.safetensors (model.nfp_head.* in float32); OUT/history.json (per epoch: the training loss, the pick set's
loss, the learning rate); OUT/heldout/pairs.jsonl + summary.json (as nfp_heldout's, with "gate" and the run's
settings); and when the gate passes OUT/start: the RL arm's start model -- checkpoint-770's files as symbolic links
(no weight copied), its config with model_line_nfp, the head as one more shard in the index (start_dir).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from model_matrix.formal import nfp_heldout as H

SFT = Path("/mnt/umm/users/yinbaiqiao/models_runs/chains/m_cambrian7b_formal_sft_20261026/lf_cformal/model/"
           "checkpoint-770")
TRAJ_TRAIN = H.TRAJ_VAL.with_name("trajectory_train.jsonl")
HEAD_SHARD = "model-nfp-head.safetensors"
HEAD_PREFIX = "model.nfp_head."


def pair_tensors(hidden, pairs: Sequence[Tuple[int, int, int, int]], raw, side: int = 27, newline: bool = True):
    """The head's input and target of every pair, as cambrian_nfp.concat_nfp_loss takes them: the pooled hidden
    states (P, 64, h) in the hidden states' dtype, the next frame's grid (P, 64, d) and the current frame's (the copy
    baseline's prediction), both float32."""
    import torch

    from vagen.models import cambrian_nfp as N

    offs = N.patch_offsets(side, newline).to(hidden.device)
    rows = torch.tensor([p[0] for p in pairs], device=hidden.device)
    starts = torch.tensor([p[1] for p in pairs], device=hidden.device)
    pooled = N._pool_hidden(hidden[rows.unsqueeze(1), starts.unsqueeze(1) + offs.unsqueeze(0)], side)
    raw = raw.detach()
    cur = torch.tensor([p[2] for p in pairs], device=raw.device)
    nxt = torch.tensor([p[3] for p in pairs], device=raw.device)
    return pooled, N._grid_targets(raw[nxt], side), N._grid_targets(raw[cur], side)


def held_back(episode: int, share: float = 0.05) -> bool:
    """Whether training episode ``episode`` is held back to pick the epoch (a fixed hash of its index)."""
    return int(hashlib.md5(str(episode).encode()).hexdigest()[:8], 16) % 10000 < share * 10000


def mean_loss(head, x, y, weights=(0.1, 0.1), chunk: int = 256) -> float:
    """The training loss over pairs (x (P, 64, h), y (P, 64, d)): every pair weighs the same, as every cell does."""
    import torch

    from vagen.models import cambrian_nfp as N

    total = 0.0
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            mse, cos = N._losses(head(x[i:i + chunk].float()).float(), y[i:i + chunk])
            total += (weights[0] * float(mse) + weights[1] * float(cos)) * min(chunk, x.shape[0] - i)
    return total / x.shape[0]


def train_head(head, train: Tuple, pick: Tuple, *, epochs: int, lr: float, batch: int, patience: int,
               weights=(0.1, 0.1), warmup: int = 100, seed: int = 0, log=print):
    """AdamW on the (x, y) pairs of ``train`` with a linear warm-up and a cosine decay to a tenth; after every epoch
    the pick set's loss.  Returns the best epoch's state dict (on the CPU) and the history; stops after ``patience``
    epochs without a better pick loss."""
    import torch

    from vagen.models import cambrian_nfp as N

    x, y = train
    n = x.shape[0]
    per_epoch = math.ceil(n / batch)
    total = epochs * per_epoch
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))))
    gen = torch.Generator().manual_seed(seed)
    best, best_loss, bad, history = None, float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n, generator=gen).to(x.device)
        seen = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            mse, cos = N._losses(head(x[idx].float()).float(), y[idx])
            loss = weights[0] * mse + weights[1] * cos
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            seen += float(loss.detach()) * len(idx)
        head.eval()
        pick_loss = mean_loss(head, *pick, weights=weights)
        history.append({"epoch": epoch, "train_loss": seen / n, "pick_loss": pick_loss,
                        "lr": opt.param_groups[0]["lr"]})
        log(json.dumps(history[-1]))
        if pick_loss < best_loss:
            best_loss, bad = pick_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return best, history


def heldout_rows(head, x, y, copy, meta: List[dict], weights=(0.1, 0.1)) -> List[dict]:
    """nfp_heldout's rows for the pairs (x, y, copy) with their (episode, k, answer): the head's and the copy
    baseline's losses (the average-frame baseline is added by nfp_heldout.avg_baseline)."""
    import torch

    from vagen.models import cambrian_nfp as N

    rows = []
    with torch.no_grad():
        for i, m in enumerate(meta):
            mse, cos = N._losses(head(x[i:i + 1].float()).float(), y[i:i + 1])
            cmse, ccos = N._losses(copy[i:i + 1], y[i:i + 1])
            rows.append({**m, "mse": float(mse), "cos": float(cos), "copy_mse": float(cmse), "copy_cos": float(ccos),
                         "loss": weights[0] * float(mse) + weights[1] * float(cos),
                         "copy_loss": weights[0] * float(cmse) + weights[1] * float(ccos)})
    return rows


def gate(summary: dict) -> bool:
    """The user's condition for the RL arm: the head's mean loss on the held-out pairs below the average frame's."""
    mean = summary.get("mean") or {}
    return "avg_loss" in mean and mean["loss"] < mean["avg_loss"]


def start_dir(src: Path, head_state: Dict[str, "object"], out: Path) -> Path:
    """The RL arm's start model in ``out``: every file of ``src`` as a symbolic link except config.json (written with
    model_line_nfp true, so the adapter builds the head and loads it) and model.safetensors.index.json (the source's
    map and the head's keys on one more shard); the head (model.nfp_head.*, float32) in HEAD_SHARD.  No weight of the
    source is copied; ``out`` must not exist and the source must have no head."""
    from safetensors.torch import save_file

    index = json.loads((src / "model.safetensors.index.json").read_text())
    if any(k.startswith(HEAD_PREFIX) for k in index["weight_map"]):
        raise ValueError(f"{src} already has a head")
    out.mkdir(parents=True)  # FileExistsError if it is there
    for p in sorted(src.iterdir()):
        if p.is_file() and p.name not in ("config.json", "model.safetensors.index.json"):
            (out / p.name).symlink_to(p.resolve())
    cfg = json.loads((src / "config.json").read_text())
    cfg["model_line_nfp"] = True
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    tensors = {HEAD_PREFIX + k: v.detach().float().contiguous().cpu() for k, v in head_state.items()}
    save_file(tensors, str(out / HEAD_SHARD), metadata={"format": "pt"})
    index["weight_map"].update({k: HEAD_SHARD for k in tensors})
    meta = index.setdefault("metadata", {})
    meta["total_size"] = int(meta.get("total_size", 0)) + sum(t.numel() * t.element_size() for t in tensors.values())
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    return out


def extract(model, proc, path: Path, limit: Optional[int], device: str, keep_copy: bool = True, log=print) -> dict:
    """Every consecutive pair of ``path``'s episodes (as nfp_heldout's main reads them) through the frozen model:
    x (P, 64, h) bf16, y (P, 64, d) float32 and, with ``keep_copy``, the copy baseline's copy (P, 64, d) on
    ``device``; meta [(episode, k, answer)]; the counts."""
    import torch
    from PIL import Image

    xs, ys, cs, meta = [], [], [], []
    episodes = one_frame = 0
    t0 = time.time()
    with open(path) as src:
        for i, line in enumerate(src):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            ids = H.episode_ids(proc, r["messages"])
            frames = sum(t == H.IMAGE_TOKEN_ID for t in ids)
            if frames != len(r["images"]):
                raise SystemExit(f"{path.name} episode {i}: {frames} <image> tokens, {len(r['images'])} frames")
            episodes += 1
            if frames < 2:
                one_frame += 1
                continue
            images = []
            for p in r["images"]:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            pix = proc.preprocess_images(images)["pixel_values"].to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                hidden, raw, pairs, side, newline = H.episode_forward(model, torch.tensor([H.expand(ids)], device=device),
                                                                      pix)
                if len(pairs) != frames - 1:
                    raise SystemExit(f"{path.name} episode {i}: {len(pairs)} pairs for {frames} frames")
                x, y, c = pair_tensors(hidden, pairs, raw, side, newline)
            xs.append(x)
            ys.append(y)
            if keep_copy:
                cs.append(c)
            said = H.answers(r["messages"])
            meta.extend({"episode": i, "k": k, "answer": said[k]} for k in range(len(pairs)))
            if episodes % 1000 == 0:
                log(f"{path.name}: {episodes} episodes, {len(meta)} pairs, {time.time() - t0:.0f} s")
    if not xs:
        raise SystemExit(f"{path.name}: no frame pairs in the first {limit} episodes")
    return {"x": torch.cat(xs), "y": torch.cat(ys), "copy": torch.cat(cs) if keep_copy else None, "meta": meta,
            "episodes": episodes, "one_frame": one_frame, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=SFT, help="the SFT start (no head)")
    ap.add_argument("--train", type=Path, default=TRAJ_TRAIN)
    ap.add_argument("--heldout", type=Path, default=H.TRAJ_VAL)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes of each file (a check)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64, help="pairs per step")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--build-start", action="store_true",
                    help="build OUT/start even if the gate fails (a check of the whole path; never for the RL arm)")
    a = ap.parse_args()

    os.environ.update(H.training_env())  # before anything imports huggingface_hub or the Cambrian source
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    import vagen.models.cambrian_register  # noqa: F401  (registers the adapter; needs CAMBRIAN_SRC)
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from vagen.models import cambrian_nfp as N
    from vagen.models.cambrian_processor import CambrianProcessorWrapper

    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    proc = CambrianProcessorWrapper(AutoTokenizer.from_pretrained(a.model))
    if proc.image_token_id != H.IMAGE_TOKEN_ID:
        raise SystemExit(f"<image> is token {proc.image_token_id}, not {H.IMAGE_TOKEN_ID}")
    cfg = AutoConfig.from_pretrained(a.model)
    cfg.model_line_nfp = True  # the adapter's own path to the head's input (its fresh head is not used here)
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, torch_dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.to("cuda:0").eval()
    weights = H.loss_weights(model.config)
    hidden_size, mm_hidden = int(cfg.hidden_size), int(cfg.mm_hidden_size)
    train = extract(model, proc, a.train, a.limit, "cuda:0", keep_copy=False, log=lambda s: print(s, flush=True))
    held = extract(model, proc, a.heldout, a.limit, "cuda:0", log=lambda s: print(s, flush=True))
    del model
    torch.cuda.empty_cache()
    extract_s = time.time() - t0
    back = torch.tensor([held_back(m["episode"]) for m in train["meta"]], device="cuda:0")
    if not int(back.sum()) or int(back.sum()) == len(back):
        raise SystemExit(f"{int(back.sum())} of {len(back)} training pairs held back: raise --limit")
    x, y = train.pop("x"), train.pop("y")
    fit, pick = (x[~back], y[~back]), (x[back], y[back])
    del x, y
    torch.cuda.empty_cache()
    print(json.dumps({"train_episodes": train["episodes"], "train_pairs": int((~back).sum()),
                      "pick_pairs": int(back.sum()), "heldout_pairs": len(held["meta"]),
                      "extract_s": round(extract_s, 1)}), flush=True)

    torch.manual_seed(a.seed)
    head = N.build_head(hidden_size, mm_hidden).to("cuda:0")
    first = {"pick_loss_at_init": mean_loss(head, *pick, weights=weights)}
    best, history = train_head(head, fit, pick, epochs=a.epochs, lr=a.lr, batch=a.batch, patience=a.patience,
                               weights=weights, seed=a.seed, log=lambda s: print(s, flush=True))
    head.load_state_dict(best)
    head.eval()
    save_file({HEAD_PREFIX + k: v.float().contiguous() for k, v in best.items()}, str(a.out / "head.safetensors"),
              metadata={"format": "pt"})
    (a.out / "history.json").write_text(json.dumps({**first, "epochs": history}, indent=1))

    rows = heldout_rows(head, held["x"], held["y"], held["copy"], held["meta"], weights)
    H.avg_baseline(rows, [t.cpu() for t in held["y"]], *weights)
    out = a.out / "heldout"
    out.mkdir()
    with open(out / "pairs.jsonl", "w") as dst:
        for row in rows:
            dst.write(json.dumps(row) + "\n")
    summary = {"model": str(a.model), "data": str(a.heldout), "train": str(a.train), "limit": a.limit,
               "settings": {k: getattr(a, k) for k in ("epochs", "patience", "lr", "batch", "seed")},
               "picked_epoch": min(history, key=lambda h: h["pick_loss"])["epoch"], "train_pairs": int((~back).sum()),
               "pick_pairs": int(back.sum()), "extract_s": round(extract_s, 1),
               "train_s": round(time.time() - t0 - extract_s, 1), **H.summarize(rows, held["episodes"], held["one_frame"])}
    summary["gate"] = gate(summary)
    summary["start_built"] = bool(summary["gate"] or a.build_start)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if summary["start_built"]:
        start_dir(a.model, best, a.out / "start")
    print(json.dumps({k: summary.get(k) for k in ("picked_epoch", "pairs", "mean", "head_below_copy", "head_below_avg",
                                                  "gate")}), flush=True)


if __name__ == "__main__":
    main()
