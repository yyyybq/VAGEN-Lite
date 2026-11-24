"""C line (q38 s4): verdict 1 of PREREG_v1 section 3 -- the NFP head of a C-NFP bf16 copy on the 240 held-out NFP
windows (60 videos x 4), against the copy baseline and against a fresh random head.

  python -m vagen.q38s4.tools.nfp_heldout --model BF16_COPY --run RUN --tokenized TOK --out OUT [--device cuda:0]

- The head: the four ``nfp_head.fc{1,2}.{weight,bias}`` tensors are read explicitly (safetensors) from the shards
  BF16_COPY's index names, loaded into an nfp.NFPHead of their dtype (one dtype for all four, the shapes read from
  the file) and compared with the file bit for bit; any difference stops the run.  So a loader that skips these keys
  (transformers reports them as unexpected and leaves them out) can never leave a random head in their place.  A copy
  without them (C-noNFP) or with other nfp_head keys is refused.  The comparison runs again after the last row.
- The model: Qwen3_5ForConditionalGeneration from BF16_COPY on one card (bf16, sdpa, eval, inference mode).  The
  load must report nothing missing or mismatched and exactly the four head keys unexpected.
- The control: a new NFPHead with training's initialisation (trunc_normal std 0.02, biases 0) made on the CPU in
  fp32 after torch.manual_seed(--seed, 0), cast to the trained head's dtype and device.
- The data: RUN/train.yaml's template, cutoff and media settings (2 fps, up to 128 units, 65,536 pixels per frame:
  the training's; RUN is the formal run), the tokenized held-out set TOK (LF tokenized with the same settings), each
  row one NFP video.  One row per forward (batch size 1), as training forwards an NFP row: no labels, logits_to_keep
  1.  The video is decoded by the run's S4VideoPlugin in the collator, as in training.
- Per row, from that one forward: the targets (nfp.merge_targets of the encoder's blocks[-2]), the pairs
  (nfp.unit_pairs: unit t's tokens predict unit t+1's targets) and nfp.error_sums for the trained head, the random
  head and the copy baseline (prediction = unit t's own target), plus per unit the mean 1 - cos of each.
- Summary (summary.json): token means pooled over all rows (sum over the rows' pairs / number of pairs, the training
  log's nfp_cos / copy_cos); r = (copy_cos - nfp_cos) / copy_cos, verdict 1 "learned" when r >= 0.20; recorded only:
  nfp_mse and copy_mse, the mean over rows of each row's r, a 95% interval of r from 2,000 resamples of the videos
  (with replacement, numpy seed 0), the random head's r, and all of it per source (the folder under VSI-590K).

Output: OUT/rows.jsonl (one line per row) and OUT/summary.json.  OUT must not exist.  Reads BF16_COPY, RUN/train.yaml,
the base model's tokenizer and processor (train.yaml's model_name_or_path), TOK and the videos; writes nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from vagen.q38s4 import nfp as N
from vagen.q38s4.tools.strip_head import HEAD_KEYS, INDEX

SINGLE = "model.safetensors"
PREFIX = "nfp_head."
THRESHOLD = 0.20  # PREREG_v1 section 3
DRAWS = 2000
SEED = 0
EXPECT_ROWS, EXPECT_VIDEOS = 240, 60
HEADS = ("head", "random")  # the trained head and the control; "copy" is the baseline


# ---------------------------------------------------------------------------------------------------- the head
def weight_files(model_dir: Path) -> Dict[str, str]:
    """{tensor name: shard file} from the index, or every tensor of a single model.safetensors."""
    model_dir = Path(model_dir)
    if (model_dir / INDEX).is_file():
        return dict(json.loads((model_dir / INDEX).read_text())["weight_map"])
    if (model_dir / SINGLE).is_file():
        from safetensors import safe_open

        with safe_open(str(model_dir / SINGLE), "pt") as f:
            return {k: SINGLE for k in f.keys()}
    raise SystemExit(f"{model_dir}: neither {INDEX} nor {SINGLE}")


def read_head(model_dir: Path) -> Dict[str, torch.Tensor]:
    """The four head tensors as stored (CPU, the file's dtype).  Refuses a copy without them or with other
    nfp_head keys."""
    from safetensors import safe_open

    wm = weight_files(model_dir)
    keys = sorted(k for k in wm if k.startswith(PREFIX))
    if not keys:
        raise SystemExit(f"{model_dir}: no {PREFIX}* tensors (a C-noNFP copy has no head to judge)")
    if keys != sorted(HEAD_KEYS):
        raise SystemExit(f"{model_dir}: head keys {keys}, expected exactly {sorted(HEAD_KEYS)}")
    out: Dict[str, torch.Tensor] = {}
    for k in HEAD_KEYS:
        with safe_open(str(Path(model_dir) / wm[k]), "pt", device="cpu") as f:
            out[k] = f.get_tensor(k)
    return out


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and bytes (so also NaN payloads and the sign of zero)."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def compare_head(head: nn.Module, tensors: Mapping[str, torch.Tensor]) -> None:
    """Every parameter of ``head`` bit for bit the file's tensor of the same name, and nothing else; SystemExit
    otherwise."""
    sd = {PREFIX + k: v for k, v in head.state_dict().items()}
    if sorted(sd) != sorted(tensors):
        raise SystemExit(f"head parameters {sorted(sd)} against file tensors {sorted(tensors)}")
    bad = [k for k in sorted(sd) if not same_bits(sd[k], tensors[k])]
    if bad:
        raise SystemExit(f"head parameters differ from the file: {bad}")


def head_from_tensors(tensors: Mapping[str, torch.Tensor], device="cpu", in_dim: Optional[int] = None,
                      out_dim: Optional[int] = None) -> N.NFPHead:
    """An NFPHead of the file's shapes and dtype holding the file's values (checked bit for bit)."""
    dtypes = {t.dtype for t in tensors.values()}
    if len(dtypes) != 1:
        raise SystemExit(f"head tensors in several dtypes: { {k: str(v.dtype) for k, v in tensors.items()} }")
    dtype = dtypes.pop()
    w1, b1 = tensors[PREFIX + "fc1.weight"], tensors[PREFIX + "fc1.bias"]
    w2, b2 = tensors[PREFIX + "fc2.weight"], tensors[PREFIX + "fc2.bias"]
    hidden, d_in = w1.shape
    d_out = w2.shape[0]
    if (tuple(b1.shape), tuple(w2.shape), tuple(b2.shape)) != ((hidden,), (d_out, hidden), (d_out,)):
        raise SystemExit(f"head shapes do not chain: {[tuple(t.shape) for t in (w1, b1, w2, b2)]}")
    if (in_dim is not None and d_in != in_dim) or (out_dim is not None and d_out != out_dim):
        raise SystemExit(f"head {d_in} -> {d_out}, model text hidden {in_dim}, vision width {out_dim}")
    head = N.NFPHead(d_in, d_out, hidden, dtype=dtype, device="cpu")
    head.load_state_dict({k[len(PREFIX):]: v for k, v in tensors.items()}, strict=True)
    head = head.to(device).eval()
    compare_head(head, tensors)
    return head


def random_head(like: N.NFPHead, seed: int = SEED) -> N.NFPHead:
    """Training's initialisation (nfp.NFPHead: trunc_normal 0.02, biases 0), fp32 on the CPU after
    torch.manual_seed(seed), then cast to ``like``'s dtype and device."""
    p = like.fc1.weight
    torch.manual_seed(seed)
    h = N.NFPHead(like.fc1.in_features, like.fc2.out_features, like.fc1.out_features, dtype=torch.float32,
                  device="cpu")
    return h.to(device=p.device, dtype=p.dtype).eval()


def check_loading(info: Mapping[str, Sequence[str]]) -> None:
    """transformers' loading info: nothing missing or mismatched, exactly the four head keys unexpected."""
    missing, mism = list(info.get("missing_keys") or []), list(info.get("mismatched_keys") or [])
    unexp = sorted(info.get("unexpected_keys") or [])
    if missing or mism or unexp != sorted(HEAD_KEYS):
        raise SystemExit(f"model load: missing {missing[:8]}, mismatched {mism[:8]}, unexpected {unexp[:8]} "
                         f"(expected only {sorted(HEAD_KEYS)})")


# ---------------------------------------------------------------------------------------------------- the tap
class HeldoutTap(N.NFPTap):
    """nfp.NFPTap whose model hook runs several heads (name -> module) and the copy baseline on the same targets and
    hidden states of a forward with exactly one NFP video; ``out`` then holds the sums (nfp.error_sums over all
    pairs, as training) and the per-unit mean 1 - cos of each."""

    def __init__(self, model: nn.Module, heads: Mapping[str, nn.Module]):
        super().__init__(model, head=None)
        self.heads = dict(heads)
        self.out: Optional[Dict[str, Any]] = None

    def arm(self, nfp_video) -> None:
        super().arm(nfp_video)
        self.out = None

    def _on_model(self, module, args, kwargs, output):
        if not self._armed:
            return
        grid, flags = kwargs.get("video_grid_thw"), self._flags
        if grid is None or flags is None or len(flags) != 1 or not bool(flags[0]) or len(grid) != 1:
            raise ValueError("nfp_heldout: a forward must hold exactly one video, an NFP one")
        if self._vision_out is None:
            raise ValueError("nfp_heldout: the encoder's target block did not run")
        input_ids = kwargs.get("input_ids")
        targets = N.merge_targets(self._vision_out, self.merge)
        vid_mask = input_ids == self.video_token_id
        if int(vid_mask.sum()) != targets.shape[0]:
            raise ValueError(f"{int(vid_mask.sum())} video tokens, {targets.shape[0]} encoder targets")
        src, dst = N.unit_pairs(grid, flags, self.merge)
        t, h, w = (int(x) for x in grid[0].tolist())
        n = h * w // (self.merge * self.merge)
        if int(src.numel()) != (t - 1) * n or t < 2:
            raise ValueError(f"grid {t}x{h}x{w}: {int(src.numel())} pairs")
        src, dst = src.to(targets.device), dst.to(targets.device)
        y = targets[dst]
        hidden = self._hidden[vid_mask]  # row-major = the order masked_scatter filled the video embeddings
        preds = {"copy": targets[src]}
        for name, head in self.heads.items():
            preds[name] = head(hidden[src.to(hidden.device)])
        res: Dict[str, Any] = {"units": t, "tokens_per_unit": n, "pairs": int(src.numel())}
        for name, p in preds.items():
            yy = y.to(p.device)
            mse, cos = N.error_sums(p, yy)
            unit = [float(N.error_sums(p[k * n:(k + 1) * n], yy[k * n:(k + 1) * n])[1]) / n for k in range(t - 1)]
            res[name] = {"mse_sum": float(mse), "cos_sum": float(cos), "unit_cos": unit}
        self.out = res


# ---------------------------------------------------------------------------------------------------- summary
def source_of(path: str) -> str:
    """The VSI-590K source folder of a video path (else its parent folder's name)."""
    parts = Path(path).parts
    if "VSI-590K" in parts and parts.index("VSI-590K") + 1 < len(parts) - 1:
        return parts[parts.index("VSI-590K") + 1]
    return Path(path).parent.name


def _r(copy: float, pred: float) -> Optional[float]:
    return None if copy == 0 else (copy - pred) / copy


def pooled(recs: Sequence[Mapping[str, Any]], names: Sequence[str] = HEADS) -> Dict[str, Any]:
    """Token means over all rows' pairs, r of each head against the copy baseline (cos; mse recorded too)."""
    P = sum(r["pairs"] for r in recs)
    out: Dict[str, Any] = {"rows": len(recs), "videos": len({r["video"] for r in recs}), "pairs": P}
    if P == 0:
        return out
    for k in ("copy", *names):
        out[k] = {"cos": sum(r[k]["cos_sum"] for r in recs) / P, "mse": sum(r[k]["mse_sum"] for r in recs) / P}
    out["r"] = {k: _r(out["copy"]["cos"], out[k]["cos"]) for k in names}
    out["r_mse"] = {k: _r(out["copy"]["mse"], out[k]["mse"]) for k in names}
    return out


def row_mean_r(recs: Sequence[Mapping[str, Any]], names: Sequence[str] = HEADS) -> Dict[str, float]:
    return {k: float(np.mean([_r(r["copy"]["cos_sum"] / r["pairs"], r[k]["cos_sum"] / r["pairs"]) for r in recs]))
            for k in names}


def bootstrap(recs: Sequence[Mapping[str, Any]], names: Sequence[str] = HEADS, draws: int = DRAWS,
              seed: int = SEED) -> Dict[str, Any]:
    """95% interval of the pooled r over ``draws`` resamples of the videos (with replacement, numpy seed)."""
    vids = sorted({r["video"] for r in recs})
    pos = {v: i for i, v in enumerate(vids)}
    cols = ["pairs", "copy", *names]
    arr = np.zeros((len(vids), len(cols)))
    for r in recs:
        i = pos[r["video"]]
        arr[i, 0] += r["pairs"]
        for j, k in enumerate(cols[1:], 1):
            arr[i, j] += r[k]["cos_sum"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), size=(draws, len(vids)))
    s = arr[idx].sum(1)  # draws x cols
    copy = s[:, 1] / s[:, 0]
    out: Dict[str, Any] = {"draws": draws, "seed": seed, "videos": len(vids)}
    for j, k in enumerate(names, 2):
        r = (copy - s[:, j] / s[:, 0]) / copy
        out[k] = {"lo": float(np.quantile(r, 0.025)), "hi": float(np.quantile(r, 0.975))}
    return out


def summarize(recs: Sequence[Mapping[str, Any]], draws: int = DRAWS, seed: int = SEED,
              threshold: float = THRESHOLD) -> Dict[str, Any]:
    s = pooled(recs)
    s["row_mean_r"] = row_mean_r(recs)
    s["bootstrap"] = bootstrap(recs, draws=draws, seed=seed)
    s["by_source"] = {}
    for src in sorted({r["source"] for r in recs}):
        sub = [r for r in recs if r["source"] == src]
        s["by_source"][src] = {**pooled(sub), "row_mean_r": row_mean_r(sub)}
    r = s["r"]["head"]
    s["verdict_1"] = {"r": r, "threshold": threshold, "learned": bool(r is not None and r >= threshold),
                      "rule": "r = (copy_cos - nfp_cos) / copy_cos over all held-out pairs; learned iff r >= 0.20"}
    return s


# ---------------------------------------------------------------------------------------------------- main
def setup(run: Path, model_dir: Path, tokenized: Path, device: str):
    """tools.logits_probe.setup with the weights from ``model_dir`` and the data from ``tokenized``; also returns
    transformers' loading info."""
    import datasets
    import yaml
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer
    from transformers import Qwen3_5ForConditionalGeneration

    from vagen.q38s4.lf_nfp import IGNORE_INDEX, make_collator_class, make_get_template

    cfg = yaml.safe_load((run / "train.yaml").read_text())
    media = {k: cfg[k] for k in ("video_fps", "video_maxlen", "video_max_pixels", "image_max_pixels") if k in cfg}
    margs = ModelArguments(model_name_or_path=cfg["model_name_or_path"], **media)
    tm = load_tokenizer(margs)
    dargs = DataArguments(template=cfg["template"], cutoff_len=int(cfg["cutoff_len"]),
                          mask_history=bool(cfg.get("mask_history", False)),
                          enable_thinking=bool(cfg.get("enable_thinking", True)))
    template = make_get_template(get_template_and_fix_tokenizer)(tm["tokenizer"], dargs)
    t0 = time.time()
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device},
        output_loading_info=True)
    model.eval()
    print(f"[heldout] model loaded from {model_dir} in {time.time() - t0:.0f} s", flush=True)
    collator = make_collator_class()(template=template, model=model, pad_to_multiple_of=8,
                                     label_pad_token_id=IGNORE_INDEX, block_diag_attn=False,
                                     attn_implementation="sdpa", compute_dtype=torch.bfloat16, **tm)
    ds = datasets.load_from_disk(str(tokenized))["train"]
    return cfg, media, model, info, collator, ds


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="C-NFP bf16 copy with the head (ckpt_bf16/checkpoint-N)")
    ap.add_argument("--run", required=True, help="run folder whose train.yaml gives template and media settings")
    ap.add_argument("--tokenized", required=True, help="tokenized held-out set (tok/s4_v2_nfp_heldout)")
    ap.add_argument("--out", required=True, help="output folder (must not exist)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=SEED, help="random head's torch seed and the bootstrap's numpy seed")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--expect-rows", type=int, default=EXPECT_ROWS, help="0: no check")
    ap.add_argument("--expect-videos", type=int, default=EXPECT_VIDEOS, help="0: no check")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    model_dir = Path(a.model).resolve()
    tensors = read_head(model_dir)  # before the 27B loads: a copy without the head stops here
    from vagen.q38s4.lf_video import is_nfp, parse

    cfg, media, model, info, collator, ds = setup(Path(a.run).resolve(), model_dir, Path(a.tokenized), a.device)
    check_loading(info)
    head = head_from_tensors(tensors, a.device, int(model.config.text_config.hidden_size),
                             int(model.config.vision_config.hidden_size))
    rand = random_head(head, a.seed)
    rows: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        vids = ds[i].get("videos") or []
        if len(vids) != 1 or not is_nfp(vids[0]):
            raise SystemExit(f"row {i}: videos {vids} (expected one NFP video)")
        path, spec = parse(vids[0])
        rows.append({"row": i, "video": path, "source": source_of(path), "u": spec.u})
    n_vid = len({r["video"] for r in rows})
    if (a.expect_rows and len(rows) != a.expect_rows) or (a.expect_videos and n_vid != a.expect_videos):
        raise SystemExit(f"{len(rows)} rows of {n_vid} videos, expected {a.expect_rows} of {a.expect_videos}")
    dev = torch.device(a.device)
    meta = {"model": str(model_dir), "run": str(Path(a.run).resolve()), "tokenized": str(Path(a.tokenized).resolve()),
            "base": cfg["model_name_or_path"], "media": media, "template": cfg["template"], "rows": len(rows),
            "videos": n_vid, "head_dtype": str(head.fc1.weight.dtype), "seed": a.seed, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev),
            "unexpected_keys": sorted(info.get("unexpected_keys") or [])}
    print(f"[heldout] {json.dumps(meta)}", flush=True)
    out.mkdir(parents=True)
    tap = HeldoutTap(model, {"head": head, "random": rand})
    recs: List[Dict[str, Any]] = []
    t_all = time.time()
    with torch.inference_mode(), open(out / "rows.jsonl", "w") as f:
        for n_, r in enumerate(rows):
            t0 = time.time()
            b = collator([dict(ds[r["row"]])])
            nfp = b.pop("nfp_video")
            b.pop("labels", None)
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            tap.arm(nfp)
            try:
                model(**b, logits_to_keep=1)
                res = tap.out
            finally:
                tap.disarm()
            if res is None:
                raise SystemExit(f"row {r['row']}: the tap computed nothing")
            rec = {**r, **res, "seconds": round(time.time() - t0, 2)}
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[heldout] {n_ + 1}/{len(rows)} {r['source']} u {r['u']:.3f} units {res['units']}: head cos "
                  f"{res['head']['cos_sum'] / res['pairs']:.4f}, copy {res['copy']['cos_sum'] / res['pairs']:.4f}, "
                  f"random {res['random']['cos_sum'] / res['pairs']:.4f}, {rec['seconds']} s", flush=True)
    tap.remove()
    compare_head(head, tensors)  # the head was not changed by the run
    summary = {"meta": {**meta, "seconds": round(time.time() - t_all, 1), "head_rechecked": True},
               **summarize(recs, a.draws, a.seed)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[heldout] verdict 1: {json.dumps(summary['verdict_1'])}; bootstrap {json.dumps(summary['bootstrap'])}",
          flush=True)


if __name__ == "__main__":
    main()
