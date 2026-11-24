"""C line (q38 s4): a training step's QA rows replayed on one card through the LM head and the loss in both layouts
(logits "all" and "targets"), with inference numerics and with the training's, next to the logged text_loss.

Why (STAGE3_REPORT section 8): at D2's step 2 (S1 settings, initial weights) logits "all" logged a text loss 5.5e-3
(relative) above "targets", which is bit for bit repeatable (S1 = D1).  tools.logits_probe found the two layouts'
logits and token losses bit for bit equal on one card in inference.  It left two things open, both tested here:

1. The loss on the full sequence.  LF's path (logits "all") takes the cross entropy of a T x V fp32 tensor; on the
   video rows (T 9,216-9,672, V 248,320) that is more than 2^31 elements (T > 8,648), with the answer tokens at the
   end, past element 2^31.  logits_probe compared the model's own full-label loss only on rows up to 4,096 tokens.
   (a) synthetic: bf16 logits of T rows, targets at a few rows (the last ones included); the fp32 cross entropy of
       the whole tensor at those rows against that of those rows alone and against float64;
   (b) real: each replayed row's loss by transformers' ForCausalLMLoss on the full logits (the model's "all" loss,
       reduction mean as Qwen3_5ForConditionalGeneration calls it), also summed over two halves of the rows (each
       under 2^31 elements), against the at_targets layout (the model's "targets" loss), from the same hidden states.
2. The training's numerics.  Accelerate's FSDP1 with mixed_precision bf16 computes with every parameter and buffer in
   bf16 (A_log, dt_bias and the rotary inv_freq included; loaded on one card they stay fp32) inside
   torch.autocast(bf16); with LF's reentrant gradient checkpointing the decoder layers' forward runs without grad,
   the final norm, LM head and loss with grad.  Mode "train" does that on one card: model.to(bfloat16), train(),
   use_cache False, the body under autocast without grad, the head and loss under autocast with grad on h.  Mode
   "infer" is tools.logits_probe's (the model as loaded, eval, inference mode).

The rows: the run's data order is rebuilt (accelerate's SeedableRandomSampler: randperm of the dataset seeded
seed + epoch; BatchSamplerShard: a global batch is --world consecutive indices, rank r the r-th, an epoch's last
batch filled from the epoch's start) and checked against --check-run's logged nfp_videos at every step; then the QA
rows (no NFP video) of --steps.  Training's text_loss at a step is the mean over its QA rows of each row's mean token
loss (Qwen3_5ForConditionalGeneration does not pass num_items_in_batch to its loss; lf_nfp averages the micro-batch
losses over ranks), so the replayed step loss is that mean, printed next to the logged text_loss of --compare runs.

  python -m vagen.q38s4.tools.head_loss_probe --run RUN --check-run RUN_S1 --compare RUN_S1 RUN_D1 RUN_D2 --out OUT

Output: OUT/rows.jsonl (a line per row and mode) and OUT/summary.json (order check, synthetic, step losses).  OUT
must not exist.  Reads RUN/train.yaml (model, template, media settings, tokenized_path) and the runs'
trainer_state.json; writes nothing else.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from vagen.q38s4.lf_nfp import IGNORE_INDEX, at_targets

TWO31 = 2 ** 31
SYNTHETIC_T = (4096, 8640, 8656, 9328, 9672)


# ---------------------------------------------------------------------------------------------------- data order
def data_order(n: int, world: int, seed: int, epochs: int) -> List[List[int]]:
    """The global batches (dataset indices in rank order) of `epochs` epochs, batch size 1 per rank."""
    steps: List[List[int]] = []
    for e in range(epochs):
        g = torch.Generator()
        g.manual_seed(seed + e)
        perm = torch.randperm(n, generator=g).tolist()
        for s in range(-(-n // world)):
            b = perm[s * world:(s + 1) * world]
            if len(b) < world:
                b = b + perm[:world - len(b)]
            steps.append(b)
    return steps


def log_steps(run: Path) -> Dict[int, Dict[str, Any]]:
    """{step: logged entry} from RUN/ckpt/trainer_state.json (the training-step entries)."""
    with open(Path(run) / "ckpt" / "trainer_state.json") as f:
        hist = json.load(f)["log_history"]
    return {int(e["step"]): e for e in hist if "step" in e and "loss" in e and "train_runtime" not in e}


def check_order(steps: Sequence[Sequence[int]], nfp: Sequence[bool], logged: Dict[int, Dict[str, Any]]) -> Dict:
    """Rebuilt order against the logged nfp_videos (one video per NFP row) at every logged step it covers."""
    common = [s for s in sorted(logged) if s <= len(steps) and logged[s].get("nfp_videos") is not None]
    bad = [s for s in common if sum(bool(nfp[i]) for i in steps[s - 1]) != int(logged[s]["nfp_videos"])]
    return {"steps": len(common), "match": len(common) - len(bad), "mismatch": bad}


# ---------------------------------------------------------------------------------------------------- synthetic CE
def synthetic_ce(T: int, V: int, device, seed: int = 0, n_tail: int = 4) -> Dict[str, Any]:
    """Part 1(a): fp32 cross entropy of a whole T x V tensor (bf16 logits upcast, as ForCausalLMLoss does) at a few
    target rows against the same rows alone and float64.  Targets: row 1, row T // 2 and the last n_tail rows; half
    of them confident (target logit 25), half not."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(T, V, device=device, dtype=torch.bfloat16, generator=g).mul_(3)
    rows = torch.tensor(sorted({1, T // 2, *range(T - n_tail, T)}), device=device)
    tgt = torch.randint(0, V, (rows.numel(),), device=device, generator=g)
    conf = torch.arange(rows.numel(), device=device) % 2 == 0
    x[rows[conf], tgt[conf]] = 25.0
    labels = torch.full((T,), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[rows] = tgt
    xf = x.float()
    full_none = F.cross_entropy(xf, labels, ignore_index=IGNORE_INDEX, reduction="none")[rows]
    full_mean = F.cross_entropy(xf, labels, ignore_index=IGNORE_INDEX, reduction="mean")
    del xf
    sub = x[rows].float()
    sub_none = F.cross_entropy(sub, tgt, reduction="none")
    xd = x[rows].double()
    ref = torch.logsumexp(xd, -1) - xd.gather(-1, tgt[:, None]).squeeze(-1)
    out = {"T": T, "V": V, "elements": T * V, "over_2_31": T * V > TWO31,
           "rows": rows.tolist(), "rows_past_2_31": [int(r) for r in rows.tolist() if int(r) * V >= TWO31],
           "full_equals_rows": bool(torch.equal(full_none, sub_none)),
           "full_minus_rows_max_abs": float((full_none.double() - sub_none.double()).abs().max()),
           "rows_minus_ref_max_abs": float((sub_none.double() - ref).abs().max()),
           "full_mean_minus_rows_mean": float(full_mean.double() - sub_none.double().mean()),
           "loss": [float(v) for v in sub_none.tolist()]}
    del x, sub, xd
    return out


# ---------------------------------------------------------------------------------------------------- replay
def _autocast(device: torch.device, on: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if on else contextlib.nullcontext()


def replay(h: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, vocab: int, train: bool) -> Dict[str, Any]:
    """Part 1(b): from the same final hidden states h (1 x T x H) and LM-head weight, the loss of both layouts as
    the model computes them (lm_head = F.linear, then transformers' ForCausalLMLoss, reduction mean); with train,
    under autocast(bf16) with grad on h, as in training."""
    from transformers.loss.loss_utils import ForCausalLMLoss

    lay = at_targets({"labels": labels})
    keep, new = lay["logits_to_keep"], lay["labels"]
    lab = labels[:, 1:]
    mask = lab != IGNORE_INDEX
    gm = torch.enable_grad() if train else torch.inference_mode()
    with gm, _autocast(h.device, train):
        hh = h.detach().clone().requires_grad_(train)
        lt = F.linear(hh[:, keep], weight)
        loss_t = ForCausalLMLoss(lt, new, vocab)
        la = F.linear(hh, weight)
        loss_a = ForCausalLMLoss(la, labels, vocab)
        head_equal = bool(torch.equal(la[:, keep], lt))
        unequal = float((la[:, keep][:, :-1] != lt[:, :-1]).double().mean())
        # the full row's CE summed over two halves (each under 2^31 elements) and over the target rows alone
        T = la.shape[1]
        cut = T // 2
        k = int(mask.sum())
        with torch.no_grad():
            shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:]
            halves = [F.cross_entropy(la[0, a:b].float(), shifted[0, a:b], ignore_index=IGNORE_INDEX,
                                      reduction="sum") for a, b in ((0, cut), (cut, T))]
            loss_halves = (halves[0] + halves[1]) / k
            sel = F.cross_entropy(la[0, :-1][mask[0]].float(), lab[0][mask[0]], reduction="none")
            tok_t = F.cross_entropy(lt[0, :-1].float(), new[0, 1:], reduction="none")
        loss_a, loss_t = float(loss_a.detach()), float(loss_t.detach())
        rec = {"T": T, "K": k, "dtype_h": str(h.dtype), "dtype_logits": str(la.dtype),
               "all": loss_a, "targets": loss_t, "all_halves": float(loss_halves),
               "all_selected": float(sel.double().mean()),
               "head_equal": head_equal, "logits_unequal_frac": unequal,
               "all_equals_targets": bool(loss_a == loss_t),
               "tokens_all": [float(v) for v in sel.tolist()], "tokens_targets": [float(v) for v in tok_t.tolist()]}
        del la, lt, loss_a, loss_t, halves, sel, tok_t, hh
    return rec


# ---------------------------------------------------------------------------------------------------- model side
def prepare_train(model) -> Dict[str, int]:
    """Mode "train": every parameter and buffer in bf16 (FSDP1 mixed precision's compute dtypes), train mode,
    use_cache False (as LF sets it for training)."""
    before = {"fp32_params": sum(p.dtype == torch.float32 for p in model.parameters()),
              "fp32_buffers": sum(b.dtype == torch.float32 for b in model.buffers())}
    model.to(torch.bfloat16)
    model.train()
    for c in (model.config, getattr(model.config, "text_config", None)):
        if c is not None:
            c.use_cache = False
    before.update(fp32_params_after=sum(p.dtype == torch.float32 for p in model.parameters()),
                  fp32_buffers_after=sum(b.dtype == torch.float32 for b in model.buffers()))
    return before


def capture(model, batch: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """The at_targets forward in a mode: final hidden states (on the CPU), the LM head's output there and the
    model's own loss."""
    from vagen.q38s4.tools.logits_probe import forward

    lay = at_targets(batch)
    train = mode == "train"
    if train:  # FSDP1 casts the root module's floating forward inputs to the compute dtype
        lay = {k: v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in lay.items()}
    gm = torch.no_grad() if train else torch.inference_mode()
    dev = batch["input_ids"].device
    with gm, _autocast(dev, train):
        out, cap = forward(model, lay)
    rec = {"h": cap.h.detach().to("cpu", copy=True), "head": cap.head.detach().to("cpu", copy=True),
           "own_targets": float(out.loss)}
    del out, cap
    return rec


def head_check(c: Dict[str, Any], weight: torch.Tensor, device, train: bool) -> bool:
    """The model's own LM-head output (captured) equals F.linear of the captured h at the kept positions."""
    device = torch.device(device)
    keep = at_targets({"labels": c["labels"]})["logits_to_keep"].to(device)
    with torch.no_grad(), _autocast(device, train):
        return bool(torch.equal(c["head"].to(device), F.linear(c["h"].to(device)[:, keep], weight)))


def release(model) -> None:
    """Free the model's GPU memory although other objects (the collator) still hold the module."""
    for t in list(model.parameters()) + list(model.buffers()):
        t.data = torch.empty(0, dtype=t.dtype, device=t.device)


def step_losses(recs: Dict[int, Dict[str, Any]], step_rows: Dict[int, List[int]],
                keys: Sequence[str] = ("all", "targets", "all_halves", "all_selected", "own_targets")) -> Dict:
    """Per step, per key: the mean over the step's QA rows of the row loss (training's text_loss)."""
    out: Dict[int, Dict[str, float]] = {}
    for s, rows in step_rows.items():
        if rows and all(i in recs for i in rows):
            out[s] = {k: sum(recs[i][k] for i in rows) / len(rows) for k in keys if all(k in recs[i] for i in rows)}
    return out


def _rel(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None or b == 0 else a / b - 1.0


# ---------------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="run folder with train.yaml (model, data)")
    ap.add_argument("--out", required=True, help="output folder (must not exist)")
    ap.add_argument("--check-run", required=True, help="run whose logged nfp_videos check the rebuilt order")
    ap.add_argument("--compare", nargs="*", default=[], help="runs whose logged text_loss is printed per step")
    ap.add_argument("--steps", default="1,2", help="steps whose QA rows are replayed")
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4, help="epochs of rebuilt order (for the check)")
    ap.add_argument("--synthetic", default=",".join(map(str, SYNTHETIC_T)), help="T values of part 1(a) ('' none)")
    ap.add_argument("--modes", default="infer,train")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    modes = [m for m in a.modes.split(",") if m]
    if any(m not in ("infer", "train") for m in modes) or (modes and modes[0] == "train" and "infer" in modes):
        raise SystemExit("--modes: infer and/or train, infer first (train converts the model)")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from vagen.q38s4.lf_video import is_nfp
    from vagen.q38s4.tools import logits_probe as LP

    run = Path(a.run).resolve()
    out.mkdir(parents=True)
    dev = torch.device(a.device)
    summary: Dict[str, Any] = {"run": str(run), "torch": torch.__version__, "device": torch.cuda.get_device_name(dev)}

    # part 1(a), before the model takes the memory
    syn = []
    for T in [int(t) for t in a.synthetic.split(",") if t]:
        t0 = time.time()
        syn.append(synthetic_ce(T, 248320, dev))
        torch.cuda.empty_cache()
        r = syn[-1]
        print(f"[ce] synthetic T {T}: {r['elements']:,} elements (> 2^31: {r['over_2_31']}), rows past 2^31 "
              f"{r['rows_past_2_31']}, full == rows {r['full_equals_rows']} (max {r['full_minus_rows_max_abs']:.2e}),"
              f" rows - fp64 max {r['rows_minus_ref_max_abs']:.2e}, {time.time() - t0:.1f} s", flush=True)
    summary["synthetic"] = syn

    cfg, model, collator, ds = LP.setup(run, a.device)
    n = len(ds)
    nfp = [any(is_nfp(v) for v in (ds[i].get("videos") or [])) for i in range(n)]
    order = data_order(n, a.world, a.seed, a.epochs)
    summary["order_check"] = check_order(order, nfp, log_steps(Path(a.check_run)))
    print(f"[ce] order check against {a.check_run}: {json.dumps(summary['order_check'])}", flush=True)
    if summary["order_check"]["mismatch"] or not summary["order_check"]["steps"]:
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        raise SystemExit("rebuilt data order does not match the logged nfp_videos")
    steps = [int(s) for s in a.steps.split(",") if s]
    step_rows = {s: [i for i in order[s - 1] if not nfp[i]] for s in steps}
    rows = list(dict.fromkeys(i for s in steps for i in step_rows[s]))
    summary["step_rows"] = {str(s): {"rows": order[s - 1], "qa": step_rows[s],
                                     "media": [LP.media_of(ds[i]) for i in step_rows[s]]} for s in steps}
    print(f"[ce] steps {steps}: QA rows {json.dumps(step_rows)}", flush=True)

    caps: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for mode in modes:
        if mode == "train":
            summary["train_dtypes"] = prepare_train(model)
            print(f"[ce] train mode: {json.dumps(summary['train_dtypes'])}", flush=True)
        caps[mode] = {}
        for n_, i in enumerate(rows):
            t0 = time.time()
            batch = LP.batch_of(collator, ds, i, a.device)
            caps[mode][i] = capture(model, batch, mode)
            if n_ == 0:  # the process's first forward in this mode against a second one
                again = capture(model, batch, mode)
                summary[f"first_call_{mode}"] = {"row": i, "h_equal": bool(torch.equal(again["h"], caps[mode][i]["h"])),
                                                 "loss": [caps[mode][i]["own_targets"], again["own_targets"]]}
                del again
            caps[mode][i]["labels"] = batch["labels"].to("cpu")
            del batch
            print(f"[ce] {mode} forward {n_ + 1}/{len(rows)} row {i} T {caps[mode][i]['h'].shape[1]}: own targets "
                  f"loss {caps[mode][i]['own_targets']!r}, {time.time() - t0:.1f} s", flush=True)
    weight = model.lm_head.weight.detach().to(torch.bfloat16).clone()
    vocab = int(model.config.text_config.vocab_size)
    release(model)
    del model, collator
    torch.cuda.empty_cache()
    print(f"[ce] model released: {torch.cuda.memory_allocated(dev) / 2 ** 30:.2f} GiB allocated", flush=True)

    recs: Dict[str, Dict[int, Dict[str, Any]]] = {}
    with open(out / "rows.jsonl", "w") as f:
        for mode in modes:
            recs[mode] = {}
            for i in rows:
                c = caps[mode][i]
                t0 = time.time()
                r = replay(c["h"].to(dev), weight, c["labels"].to(dev), vocab, train=(mode == "train"))
                r.update(row=i, mode=mode, media=LP.media_of(ds[i]), own_targets=c["own_targets"],
                         own_equals_targets=bool(c["own_targets"] == r["targets"]),
                         head_equals_replay=head_check(c, weight, dev, mode == "train"))
                recs[mode][i] = r
                f.write(json.dumps(r) + "\n")
                f.flush()
                torch.cuda.empty_cache()
                print(f"[ce] {mode} replay row {i} {r['media']} T {r['T']} K {r['K']}: all {r['all']!r} targets "
                      f"{r['targets']!r} halves {r['all_halves']!r} own {r['own_targets']!r}; head equal "
                      f"{r['head_equal']}, {time.time() - t0:.1f} s", flush=True)
    if "train" in recs and "infer" in recs:
        summary["train_vs_infer_targets"] = {str(i): _rel(recs["train"][i]["targets"], recs["infer"][i]["targets"])
                                             for i in rows}
    logged = {Path(r).name: log_steps(Path(r)) for r in a.compare}
    summary["steps"] = {}
    for mode in modes:
        sl = step_losses(recs[mode], step_rows)
        for s, v in sl.items():
            row = summary["steps"].setdefault(str(s), {})
            row[mode] = v
            row["logged"] = {k: h.get(s, {}).get("text_loss") for k, h in logged.items()}
            print(f"[ce] step {s} {mode}: " + ", ".join(f"{k} {x!r}" for k, x in v.items())
                  + "; logged " + ", ".join(f"{k} {x!r}" for k, x in row["logged"].items()), flush=True)
    summary["rows"] = {mode: {"all_equals_targets": sum(r["all_equals_targets"] for r in recs[mode].values()),
                              "head_equal": sum(r["head_equal"] for r in recs[mode].values()),
                              "own_equals_targets": sum(r["own_equals_targets"] for r in recs[mode].values()),
                              "head_equals_replay": sum(r["head_equals_replay"] for r in recs[mode].values()),
                              "n": len(recs[mode]),
                              "max_abs_all_minus_targets": max(abs(r["all"] - r["targets"]) for r in recs[mode].values()),
                              "max_abs_halves_minus_all": max(abs(r["all_halves"] - r["all"]) for r in recs[mode].values())}
                       for mode in modes}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[ce] rows: {json.dumps(summary['rows'])}", flush=True)


if __name__ == "__main__":
    main()
