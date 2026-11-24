"""C line (q38 s4): where logits "all" and "targets" part, measured on the 27B's initial weights (one card).

D2 (STAGE3_REPORT section 8): with the S1 settings, step 2 is a forward of the initial weights (step 1's learning
rate is 0) and deterministic with logits "targets" (S1 and D1 bit for bit), yet logits "all" gives a text loss 5.5e-3
higher (relative; the NFP terms bit for bit).  The two paths differ in the LM-head GEMM only: "all" multiplies every
position's final hidden state (M = T rows, LF's path), "targets" only the kept positions (M = K + 1, lf_nfp.at_targets);
both round the logits to bf16 before the fp32 cross entropy.

For each text row of the run's tokenized LF data (NFP rows and rows without a target skipped), in dataset order, the
probe runs the model once (bf16, sdpa, eval, no grad) on lf_nfp.at_targets' layout, capturing the final hidden states
(a forward hook on model.model) and the LM head's output, then recomputes the logits at the kept positions:

  all        F.linear(h, W)[:, keep]          (M = T)
  targets    F.linear(h[:, keep], W)          (M = K + 1)
  ref        float64 h[:, keep] @ W.T         (the same bf16 h and W, the product not rounded)

the two bf16 ones once more with torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction False ("_nored";
PyTorch's default, used in training, is True), and the cross entropy of every target token as the model's loss reads
it (logit k against label k + 1, fp32; float64 for ref).  Checks per row: the LM head's output equals "targets" bit
for bit, the model's own loss equals the mean of "targets"; for T <= --own-all-max also a second forward on the full
labels (LF's path): its loss against the mean of "all", its hidden states against the first forward's.  The first
row first runs --first-calls times (the process's first forward against the later ones: does a first call differ, as
S1 against D1 at step 1?).

Training's text loss (lf_nfp: the mean over the QA micro-batches of LF's loss, which with transformers'
average_tokens_across_devices is the token sum over the global token count) moves with the sum of the token losses
of the step's QA rows, so the summary draws --draws random subsets of --subset rows (D2's step 2 had 16 - 7 NFP = 9
QA rows) and gives the relative difference of those sums, all against targets (and each against ref), placed against
--observed (D2 step 2: 0.5738064886 / 0.5706624463 - 1).

  python -m vagen.q38s4.tools.logits_probe --run RUN_DIR --out OUT [--rows N] [--subset 9] [--own-all-max 4096]

Output: OUT/rows.jsonl (a line per row, the token losses of every variant included) and OUT/summary.json.  OUT must
not exist.  Reads RUN_DIR/train.yaml (model, template, media settings, tokenized_path); writes nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vagen.q38s4.lf_nfp import IGNORE_INDEX, at_targets

FLAG = "allow_bf16_reduced_precision_reduction"
D2_STEP2 = 0.5738064886277748 / 0.5706624462683167 - 1.0  # logits "all" (D2) over "targets" (S1 = D1), step 2
PAIRS = (("all", "targets"), ("all", "ref"), ("targets", "ref"), ("all_nored", "targets_nored"),
         ("all", "all_nored"), ("targets", "targets_nored"))


# ---------------------------------------------------------------------------------------------------- pieces
def select_rows(ds, limit: int = 0) -> List[int]:
    """Indices of the rows with a text target (labels[1:] not all IGNORE_INDEX) and no NFP video, in order."""
    from vagen.q38s4.lf_video import is_nfp

    out: List[int] = []
    for i in range(len(ds)):
        row = ds[i]
        if any(is_nfp(v) for v in (row.get("videos") or [])):
            continue
        if not any(int(t) != IGNORE_INDEX for t in row["labels"][1:]):
            continue
        out.append(i)
        if limit and len(out) >= limit:
            break
    return out


def media_of(row: Dict[str, Any]) -> str:
    return "video" if row.get("videos") else "image" if row.get("images") else "text"


def target_logits(h: torch.Tensor, weight: torch.Tensor, keep: torch.Tensor, flags: Sequence[bool] = (True, False),
                  ref: bool = True, chunk: int = 32768) -> Dict[str, torch.Tensor]:
    """The kept positions' logits by each variant (module docstring); the reduced-precision flag is set only on a
    GPU and restored afterwards."""
    out: Dict[str, torch.Tensor] = {}
    m = torch.backends.cuda.matmul
    old = getattr(m, FLAG)
    try:
        for f in flags:
            if h.is_cuda:
                setattr(m, FLAG, bool(f))
            tag = "" if f else "_nored"
            full = F.linear(h, weight)
            out["all" + tag] = full[:, keep].clone()
            del full
            out["targets" + tag] = F.linear(h[:, keep], weight)
    finally:
        setattr(m, FLAG, old)
    if ref:
        hk = h[:, keep].double()
        out["ref"] = torch.cat([hk @ weight[i:i + chunk].double().T for i in range(0, weight.shape[0], chunk)], -1)
    return out


def token_ce(logits: torch.Tensor, new_labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy of every target as the causal-LM loss reads at_targets' layout: logit k against label k + 1, in
    fp32 (float64 for a float64 input), row-major over the targets."""
    lab = new_labels[:, 1:]
    mask = lab != IGNORE_INDEX
    dt = torch.float64 if logits.dtype == torch.float64 else torch.float32
    return F.cross_entropy(logits[:, :-1][mask].to(dt), lab[mask], reduction="none")


def logit_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Over the read positions (the last kept one is never read): share of unequal logits, max / mean |a - b|."""
    a, b = a[:, :-1], b[:, :-1]
    d = (a.double() - b.double()).abs()
    unequal = (a != b) if a.dtype == b.dtype else (d > 0)
    return {"unequal_frac": float(unequal.double().mean()), "max_abs": float(d.max()), "mean_abs": float(d.mean())}


class Capture:
    """Forward hooks on model.model (final hidden states) and model.lm_head (its output)."""

    def __init__(self, model):
        self.h: Optional[torch.Tensor] = None
        self.head: Optional[torch.Tensor] = None
        self._hooks = [model.model.register_forward_hook(self._body), model.lm_head.register_forward_hook(self._lm)]

    def _body(self, mod, args, out):
        self.h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def _lm(self, mod, args, out):
        self.head = out

    def remove(self):
        for hk in self._hooks:
            hk.remove()


def forward(model, batch: Dict[str, Any]):
    cap = Capture(model)
    try:
        out = model(**batch)
    finally:
        cap.remove()
    return out, cap


def first_calls(model, batch: Dict[str, Any], n: int = 3) -> Dict[str, Any]:
    """The same targets forward n times: hidden states and loss of each call against the first and the last."""
    lay = at_targets(batch)
    hs, losses = [], []
    for _ in range(n):
        out, cap = forward(model, lay)
        hs.append(cap.h.clone())
        losses.append(float(out.loss))
    return {"calls": n, "loss": losses,
            "h_equal_first": [bool(torch.equal(hs[0], x)) for x in hs],
            "h_equal_last": [bool(torch.equal(hs[-1], x)) for x in hs],
            "h_max_abs_vs_last": [float((hs[-1].double() - x.double()).abs().max()) for x in hs]}


def probe_row(model, batch: Dict[str, Any], flags: Sequence[bool] = (True, False), own_all_max: int = 4096,
              ref: bool = True) -> Dict[str, Any]:
    """One row (batch size 1, no nfp_video): the variants' token losses and the checks (module docstring)."""
    lay = at_targets(batch)
    keep, new = lay["logits_to_keep"], lay["labels"]
    out, cap = forward(model, lay)
    h, weight = cap.h, model.lm_head.weight
    lg = target_logits(h, weight, keep, flags, ref)
    ce = {k: token_ce(v, new) for k, v in lg.items()}
    own_t = float(out.loss)
    rec: Dict[str, Any] = {
        "T": int(batch["input_ids"].shape[1]), "K": int(new[:, 1:].ne(IGNORE_INDEX).sum()),
        "head_equals_targets": bool(torch.equal(cap.head, lg["targets"])),
        "own_targets_loss": own_t, "own_targets_minus_ce_mean": own_t - float(ce["targets"].double().mean()),
        "own_all_loss": None, "own_all_minus_ce_mean": None, "h_equal_second_forward": None,
    }
    if rec["T"] <= own_all_max:
        out_a, cap_a = forward(model, batch)
        rec["own_all_loss"] = float(out_a.loss)
        rec["own_all_minus_ce_mean"] = rec["own_all_loss"] - float(ce["all"].double().mean())
        rec["h_equal_second_forward"] = bool(torch.equal(cap_a.h, h))
        del out_a, cap_a
    rec["ce"] = {k: [float(x) for x in v.tolist()] for k, v in ce.items()}
    rec["ce_sum"] = {k: float(v.double().sum()) for k, v in ce.items()}
    rec["logits"] = {f"{a}~{b}": logit_diff(lg[a], lg[b]) for a, b in PAIRS if a in lg and b in lg}
    # the size of a bf16 step at the target logits (the loss moves by up to about this when a target logit flips)
    if "targets" in lg:
        lab = new[:, 1:]
        tgt = lg["targets"][:, :-1].gather(-1, lab.clamp_min(0).unsqueeze(-1)).squeeze(-1)[lab != IGNORE_INDEX]
        rec["target_logit_abs_mean"] = float(tgt.double().abs().mean())
    del lg, out, cap
    return rec


# ---------------------------------------------------------------------------------------------------- summary
def _q(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {}
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "p05": float(np.quantile(x, .05)),
            "p50": float(np.quantile(x, .5)), "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def summarize(rows: List[Dict[str, Any]], subset: int = 9, draws: int = 20000, seed: int = 0,
              observed: float = D2_STEP2) -> Dict[str, Any]:
    names = sorted(set.intersection(*[set(r["ce_sum"]) for r in rows])) if rows else []
    sums = {k: np.array([r["ce_sum"][k] for r in rows]) for k in names}
    tok = {k: np.concatenate([np.array(r["ce"][k]) for r in rows]) if rows else np.zeros(0) for k in names}
    s: Dict[str, Any] = {"rows": len(rows), "tokens": int(sum(r["K"] for r in rows)),
                         "T": _q(np.array([r["T"] for r in rows], dtype=float)),
                         "K": _q(np.array([r["K"] for r in rows], dtype=float)),
                         "checks": {
                             "head_equals_targets": int(sum(r["head_equals_targets"] for r in rows)),
                             "own_targets_max_abs": max((abs(r["own_targets_minus_ce_mean"]) for r in rows), default=0.0),
                             "own_all_rows": int(sum(r["own_all_loss"] is not None for r in rows)),
                             "own_all_max_abs": max((abs(r["own_all_minus_ce_mean"]) for r in rows
                                                     if r["own_all_minus_ce_mean"] is not None), default=None),
                             "h_equal_second_forward": [int(sum(bool(r["h_equal_second_forward"]) for r in rows
                                                                if r["h_equal_second_forward"] is not None)),
                                                        int(sum(r["h_equal_second_forward"] is not None for r in rows))]},
                         "total_ce": {k: float(v.sum()) for k, v in sums.items()},
                         "pairs": {}}
    rng = np.random.default_rng(seed)
    n = len(rows)
    idx = np.stack([rng.choice(n, subset, replace=False) for _ in range(draws)]) if n >= subset > 0 else None
    for a, b in PAIRS:
        if a not in sums or b not in sums:
            continue
        dt = tok[a] - tok[b]
        p: Dict[str, Any] = {
            "rel_total": float(sums[a].sum() / sums[b].sum() - 1.0),
            "tokens_unequal": int((dt != 0).sum()),
            "token_abs": _q(np.abs(dt)),
            "token_signed_mean": float(dt.mean()) if dt.size else 0.0,
            "row_rel": _q(sums[a] / sums[b] - 1.0),
            "logits_unequal_frac": _q(np.array([r["logits"][f"{a}~{b}"]["unequal_frac"] for r in rows])),
            "logits_max_abs": _q(np.array([r["logits"][f"{a}~{b}"]["max_abs"] for r in rows])),
        }
        if idx is not None:
            rel = sums[a][idx].sum(1) / sums[b][idx].sum(1) - 1.0
            p["subset"] = {"size": subset, "draws": draws, **_q(rel), "abs_p95": float(np.quantile(np.abs(rel), .95)),
                           "share_abs_ge_observed": float((np.abs(rel) >= abs(observed)).mean())}
        s["pairs"][f"{a}~{b}"] = p
    s["observed"] = observed
    return s


# ---------------------------------------------------------------------------------------------------- main
def to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def setup(run: Path, device: str):
    import yaml
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer
    from transformers import Qwen3_5ForConditionalGeneration

    from vagen.q38s4.lf_nfp import make_collator_class, make_get_template

    cfg = yaml.safe_load((run / "train.yaml").read_text())
    media = {k: cfg[k] for k in ("video_fps", "video_maxlen", "video_max_pixels", "image_max_pixels") if k in cfg}
    margs = ModelArguments(model_name_or_path=cfg["model_name_or_path"], **media)
    tm = load_tokenizer(margs)
    dargs = DataArguments(template=cfg["template"], cutoff_len=int(cfg["cutoff_len"]),
                          mask_history=bool(cfg.get("mask_history", False)),
                          enable_thinking=bool(cfg.get("enable_thinking", True)))
    template = make_get_template(get_template_and_fix_tokenizer)(tm["tokenizer"], dargs)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg["model_name_or_path"], dtype=torch.bfloat16,
                                                            attn_implementation="sdpa", device_map={"": device})
    model.eval()
    print(f"[probe] model loaded in {time.time() - t0:.0f} s: lm_head {tuple(model.lm_head.weight.shape)} "
          f"{model.lm_head.weight.dtype}, tied {model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()}",
          flush=True)
    collator = make_collator_class()(template=template, model=model, pad_to_multiple_of=8,
                                     label_pad_token_id=IGNORE_INDEX, block_diag_attn=False,
                                     attn_implementation="sdpa", compute_dtype=torch.bfloat16, **tm)
    import datasets

    ds = datasets.load_from_disk(cfg["tokenized_path"])["train"]
    return cfg, model, collator, ds


def batch_of(collator, ds, i: int, device) -> Dict[str, Any]:
    b = collator([dict(ds[i])])
    nfp = b.pop("nfp_video", None)
    if nfp is not None and bool(nfp.any()):
        raise RuntimeError(f"row {i} has an NFP video")
    return to_device(b, device)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="run folder with train.yaml (e.g. runs/smoke/d2_s1_all)")
    ap.add_argument("--out", required=True, help="output folder (must not exist)")
    ap.add_argument("--rows", type=int, default=0, help="at most this many text rows (0: all)")
    ap.add_argument("--subset", type=int, default=9, help="rows per drawn subset (D2 step 2: 9 QA rows)")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--observed", type=float, default=D2_STEP2)
    ap.add_argument("--own-all-max", type=int, default=4096, help="LF's full-label forward only up to this length")
    ap.add_argument("--first-calls", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    run = Path(a.run).resolve()
    cfg, model, collator, ds = setup(run, a.device)
    rows = select_rows(ds, a.rows)
    out.mkdir(parents=True)
    meta = {"run": str(run), "model": cfg["model_name_or_path"], "tokenized_path": cfg["tokenized_path"],
            "dataset_rows": len(ds), "text_rows": len(rows), "torch": torch.__version__,
            "flag_default": bool(getattr(torch.backends.cuda.matmul, FLAG)),
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32), "device": torch.cuda.get_device_name(a.device)}
    print(f"[probe] {json.dumps(meta)}", flush=True)
    with torch.inference_mode():
        first = first_calls(model, batch_of(collator, ds, rows[0], a.device), a.first_calls) if rows else None
        print(f"[probe] first calls on row {rows[0] if rows else None}: {json.dumps(first)}", flush=True)
        recs: List[Dict[str, Any]] = []
        with open(out / "rows.jsonl", "w") as f:
            for n, i in enumerate(rows):
                t0 = time.time()
                rec = {"row": i, "media": media_of(ds[i]),
                       **probe_row(model, batch_of(collator, ds, i, a.device), own_all_max=a.own_all_max)}
                rec["seconds"] = round(time.time() - t0, 2)
                recs.append(rec)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                p = rec["logits"]["all~targets"]
                print(f"[probe] {n + 1}/{len(rows)} row {i} {rec['media']} T {rec['T']} K {rec['K']}: "
                      f"sum all-targets {rec['ce_sum']['all'] - rec['ce_sum']['targets']:+.3e}, logits unequal "
                      f"{p['unequal_frac']:.2e}, head==targets {rec['head_equals_targets']}, "
                      f"{rec['seconds']} s", flush=True)
    summary = {"meta": meta, "first_calls": first,
               **summarize(recs, a.subset, a.draws, observed=a.observed)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[probe] summary: {json.dumps(summary['pairs'].get('all~targets', {}))}", flush=True)


if __name__ == "__main__":
    main()
