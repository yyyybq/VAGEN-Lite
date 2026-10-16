#!/usr/bin/env python3
"""Phase 2B diagnostics for the U1 critic fixed batch."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(ROOT / "verl"))

from tools.u1_critic_one_batch_overfit import (  # noqa: E402
    DEFAULT_OUT,
    choose_trainable,
    load_model,
    param_l2_delta,
    prepare_sample,
    tensor_checksum,
)


def ev(preds, targets):
    p = torch.tensor(preds, dtype=torch.float32)
    t = torch.tensor(targets, dtype=torch.float32)
    if t.numel() < 2:
        return None
    return float(1.0 - torch.var(p - t, unbiased=False) / (torch.var(t, unbiased=False) + 1e-5))


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def value_forward(model, sample, return_hidden=False):
    out = model(**sample["batch"], use_cache=False, return_dict=True)
    logits = out.logits.squeeze(-1)
    readout_idx = sample["prompt_len"] - 1
    pred = logits[0, readout_idx].float()
    if not return_hidden:
        return pred
    hidden = out.hidden_states[-1] if isinstance(out.hidden_states, (tuple, list)) else out.hidden_states
    return pred, hidden[0, readout_idx].float()


def clipping_branch(pred, old, target, cliprange):
    clipped_pred = torch.clamp(pred, old - cliprange, old + cliprange)
    lu = 0.5 * (pred - target).pow(2)
    lc = 0.5 * (clipped_pred - target).pow(2)
    pred_v = float(pred.detach().cpu())
    clipped_v = float(clipped_pred.detach().cpu())
    lu_v = float(lu.detach().cpu())
    lc_v = float(lc.detach().cpu())
    clamp_active = abs(pred_v - clipped_v) > 1e-6
    if not clamp_active:
        branch = "tie_inside_clip"
        active_gradient = True
    elif lc_v > lu_v:
        branch = "clipped_zero_grad"
        active_gradient = False
    else:
        branch = "unclipped"
        active_gradient = True
    return clipped_pred, lu, lc, branch, active_gradient, clamp_active


def eval_model(model, samples, old_values, cliprange, loss_mode, initial_named, trainable, step):
    model.eval()
    preds, targets, rows = [], [], []
    unclipped_losses, clipped_losses, selected_losses = [], [], []
    with torch.no_grad():
        for i, s in enumerate(samples):
            pred = value_forward(model, s)
            target = torch.tensor(s["target"], device=pred.device, dtype=torch.float32)
            old = torch.tensor(old_values[i], device=pred.device, dtype=torch.float32)
            clipped_pred, lu, lc, branch, active_gradient, clamp_active = clipping_branch(
                pred, old, target, cliprange
            )
            if loss_mode == "unclipped":
                selected = lu
                branch = "unclipped"
                active_gradient = True
            else:
                selected = torch.maximum(lu, lc)
            delta = float((pred - old).detach().cpu())
            lower = float((old - cliprange).detach().cpu())
            upper = float((old + cliprange).detach().cpu())
            boundary = abs(delta - cliprange) <= 2e-2 or abs(delta + cliprange) <= 2e-2
            preds.append(float(pred.detach().cpu()))
            targets.append(float(target.detach().cpu()))
            unclipped_losses.append(float(lu.detach().cpu()))
            clipped_losses.append(float(lc.detach().cpu()))
            selected_losses.append(float(selected.detach().cpu()))
            rows.append(
                {
                    "sample_id": s["row"]["sample_id"],
                    "target": targets[-1],
                    "old_value": float(old.detach().cpu()),
                    "prediction": preds[-1],
                    "prediction_minus_old": delta,
                    "clip_lower": lower,
                    "clip_upper": upper,
                    "at_boundary": boundary,
                    "unclipped_squared_error": float(2.0 * lu.detach().cpu()),
                    "clipped_squared_error": float(2.0 * lc.detach().cpu()),
                    "clamp_active": clamp_active,
                    "selected_branch": branch,
                    "active_gradient": active_gradient,
                }
            )
    return {
        "step": step,
        "loss_mode": loss_mode,
        "vf_loss_unclipped": float(np.mean(unclipped_losses)),
        "vf_loss_clipped": float(np.mean(clipped_losses)),
        "vf_loss_selected": float(np.mean(selected_losses)),
        "value_clip_fraction": float(np.mean([r["clamp_active"] for r in rows])),
        "value_boundary_saturation_ratio": float(np.mean([r["at_boundary"] for r in rows])),
        "value_active_gradient_ratio": float(np.mean([r["active_gradient"] for r in rows])),
        "prediction_mean": float(np.mean(preds)),
        "prediction_std": float(np.std(preds)),
        "return_mean": float(np.mean(targets)),
        "return_std": float(np.std(targets)),
        "return_min": float(min(targets)),
        "return_max": float(max(targets)),
        "full_batch_ev": ev(preds, targets),
        "parameter_delta": param_l2_delta(initial_named, trainable),
        "value_head_parameter_delta": param_l2_delta(initial_named, [(n, p) for n, p in trainable if "score" in n]),
        "per_sample": rows,
    }


def train_run(name, args, samples, loss_mode, steps, lr, refresh_old=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer, processor = load_model(args.model_dir, device)
    trainable = choose_trainable(model, args.train_scope)
    params = [p for _, p in trainable]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    initial_named = {n: p.detach().float().cpu().clone() for n, p in trainable}
    with torch.no_grad():
        old_values = [float(value_forward(model, s).detach().cpu()) for s in samples]
    records = [eval_model(model, samples, old_values, args.cliprange_value, loss_mode, initial_named, trainable, 0)]
    log_path = args.out / f"{name}_metrics.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
        log_steps = {1, max(1, steps // 4), max(1, steps // 2), steps}
        for step in range(1, steps + 1):
            if refresh_old:
                model.eval()
                with torch.no_grad():
                    old_values = [float(value_forward(model, s).detach().cpu()) for s in samples]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            clipped_selected = []
            active = []
            for i, s in enumerate(samples):
                pred = value_forward(model, s)
                target = torch.tensor(s["target"], device=device, dtype=torch.float32)
                old = torch.tensor(old_values[i], device=device, dtype=torch.float32)
                if loss_mode == "unclipped":
                    loss = 0.5 * (pred - target).pow(2)
                    clipped_selected.append(False)
                    active.append(True)
                else:
                    clipped_pred, lu, lc, branch, active_gradient, clamp_active = clipping_branch(
                        pred, old, target, args.cliprange_value
                    )
                    loss = torch.maximum(lu, lc)
                    clipped_selected.append(clamp_active)
                    active.append(active_gradient)
                (loss / len(samples)).backward()
                losses.append(float(loss.detach().cpu()))
            total_gn = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
            optimizer.step()
            if step in log_steps:
                rec = eval_model(model, samples, old_values, args.cliprange_value, loss_mode, initial_named, trainable, step)
                rec.update(
                    {
                        "critic_grad_norm": total_gn,
                        "learning_rate": lr,
                        "refresh_old_value": refresh_old,
                        "step_clipped_selected_ratio": float(np.mean(clipped_selected)),
                        "step_active_gradient_ratio": float(np.mean(active)),
                    }
                )
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    ck = tensor_checksum(params)
    del model, optimizer
    torch.cuda.empty_cache()
    return {"name": name, "steps": steps, "lr": lr, "loss_mode": loss_mode, "refresh_old_value": refresh_old, "checksum": ck, "records": records}


def hidden_probe(args, samples):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer, processor = load_model(args.model_dir, device)
    model.eval()
    hs, targets, sample_ids = [], [], []
    readout_rows = []
    with torch.no_grad():
        for s in samples:
            pred, h = value_forward(model, s, return_hidden=True)
            hs.append(h.detach().cpu().float())
            targets.append(float(s["target"]))
            sample_ids.append(s["row"]["sample_id"])
            readout_rows.append(
                {
                    "sample_id": s["row"]["sample_id"],
                    "prompt_tokens": s["prompt_len"],
                    "response_tokens": s["response_len"],
                    "target_response_index": 0,
                    "critic_readout_full_index": s["prompt_len"] - 1,
                    "critic_readout_semantics": "last prompt token predicts first response value, matching dp_critic values[:, -response_length-1:-1]",
                }
            )
    H = torch.stack(hs, dim=0)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    Hn = F.normalize(H, dim=1)
    cosine = Hn @ Hn.T
    dist = torch.cdist(H, H)
    svals = torch.linalg.svdvals(H - H.mean(dim=0, keepdim=True))
    rank = int(torch.linalg.matrix_rank(H - H.mean(dim=0, keepdim=True), tol=1e-4).item())
    stats = {
        "hidden_shape": list(H.shape),
        "readout": readout_rows,
        "hidden_norms": [{"sample_id": sid, "norm": float(h.norm().item())} for sid, h in zip(sample_ids, H)],
        "pairwise_cosine": cosine.tolist(),
        "pairwise_l2": dist.tolist(),
        "feature_matrix_rank_centered": rank,
        "singular_values": [float(x) for x in svals.tolist()],
        "near_duplicate_pairs_cosine_gt_0_999": [
            [sample_ids[i], sample_ids[j], float(cosine[i, j])]
            for i in range(len(sample_ids))
            for j in range(i + 1, len(sample_ids))
            if float(cosine[i, j]) > 0.999
        ],
    }
    H_mean = H.mean(dim=0, keepdim=True)
    H_std = H.std(dim=0, keepdim=True).clamp_min(1e-6)
    X = (H - H_mean) / H_std
    y_mean = y.mean(dim=0, keepdim=True)
    solution = torch.linalg.lstsq(X, y - y_mean).solution
    closed_pred = X @ solution + y_mean
    closed_mse = F.mse_loss(closed_pred, y)
    stats["closed_form_linear_probe"] = {
        "feature": "standardized hidden state, centered target, explicit target mean restored",
        "mse": float(closed_mse.item()),
        "full_batch_ev": ev(closed_pred.squeeze(1).tolist(), targets),
        "prediction_return_pairs": [
            {"sample_id": sid, "prediction": float(p), "return": float(t)}
            for sid, p, t in zip(sample_ids, closed_pred.squeeze(1).tolist(), targets)
        ],
        "weight_norm": float(solution[:-1].norm().item()),
        "bias": float(y_mean.item()),
    }
    (args.out / "hidden_state_statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    del model
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    probe = nn.Linear(H.shape[1], 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=0.0)
    before = [p.detach().clone() for p in probe.parameters()]
    log_path = args.out / "linear_probe_metrics.jsonl"
    records = []
    with log_path.open("w", encoding="utf-8") as f:
        for step in range(args.probe_steps + 1):
            with torch.no_grad():
                pred = probe(H)
                loss = F.mse_loss(pred, y)
                rec = {
                    "step": step,
                    "mse": float(loss.item()),
                    "full_batch_ev": ev(pred.squeeze(1).tolist(), targets),
                    "prediction_return_pairs": [
                        {"sample_id": sid, "prediction": float(p), "return": float(t)}
                        for sid, p, t in zip(sample_ids, pred.squeeze(1).tolist(), targets)
                    ],
                }
            if step in {0, 1, 10, 50, 100, args.probe_steps}:
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step == args.probe_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(probe(H), y)
            loss.backward()
            opt.step()
    with torch.no_grad():
        delta = math.sqrt(sum(float((p.detach() - b).pow(2).sum()) for p, b in zip(probe.parameters(), before)))
    stats["linear_probe"] = {"initial": records[0], "final": records[-1], "parameter_delta": delta}
    (args.out / "hidden_state_statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats["linear_probe"]


def write_audit(args, prod):
    final = prod["records"][-1]
    rows = final["per_sample"]
    audit = {
        "phase2a_status_relabel": "INCONCLUSIVE",
        "reason": [
            "forward/backward/optimizer/save/reload passed",
            "fixed-batch overfit did not pass",
            "Phase 2A did not isolate value clipping and used reconstructed targets, not saved GAE tensors",
        ],
        "prediction_stuck_at_old_value_plus_minus_clip": bool(final["value_boundary_saturation_ratio"] > 0.5),
        "boundary_saturation_ratio": final["value_boundary_saturation_ratio"],
        "clamp_active_ratio": final["value_clip_fraction"],
        "clipped_branch_selected_ratio": final["value_clip_fraction"],
        "zero_gradient_from_clipped_branch_ratio": 1.0 - final["value_active_gradient_ratio"],
        "prediction_delta_min_median_max": [
            float(np.min([r["prediction_minus_old"] for r in rows])),
            float(np.median([r["prediction_minus_old"] for r in rows])),
            float(np.max([r["prediction_minus_old"] for r in rows])),
        ],
        "implementation": {
            "file": "verl/verl/trainer/ppo/core_algos.py",
            "formula": "vf_loss = 0.5 * agg_loss(max((vpreds-returns)^2, (clip(vpreds, old_values±cliprange)-returns)^2), valid_value_mask)",
            "old_values_fixed_during_ppo_epoch": True,
            "diagnostic_old_values": "initial predictions from the same fixed step-32 critic before each A/B run",
            "valid_mask": "response_mask & value_mask & returns != -100",
            "reduction_denominator": "valid target count under token-mean agg_loss, not total tokens",
            "bf16_fp32": "model forward is BF16, loss/clipping scalars are evaluated in FP32 tensors in this diagnostic",
        },
    }
    (args.out / "phase2b_value_clipping_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (args.out / "value_clipping_per_sample.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return audit


def update_final(args, final_report):
    path = args.out.parent / "reports" / "FINAL_DELIVERY.json"
    data = json.loads(path.read_text())
    data["phase2a_production_clipped_overfit"] = {
        "status": "INCONCLUSIVE",
        "reason": "update/reload passed but clipping/capacity/target-source effects were not isolated",
        "report": "../critic_only/phase2_critic_only_report.json",
    }
    data["phase2_final"] = {
        "status": final_report["phase2_status"],
        "report": "../critic_only/phase2_final_report.json",
        "critic_capacity": final_report["phase2_critic_capacity"],
        "production_clipping_behavior": final_report["phase2_production_clipping_behavior"],
        "allow_phase3": final_report["allow_phase3"],
    }
    data["overall_status"] = "NOT_READY_FOR_TRAINING" if final_report["phase2_status"] != "PASS" else "NOT_READY_FOR_TRAINING"
    data["next_gate"] = "phase3_fm_only" if final_report["allow_phase3"] else "blocked_on_phase2b"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_OUT / "step32_critic_merged_hf")
    ap.add_argument("--batch", type=Path, default=DEFAULT_OUT / "fixed_critic_batch.jsonl")
    ap.add_argument("--train-scope", default="score_and_last_layer")
    ap.add_argument("--cliprange-value", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--prod-steps", type=int, default=20)
    ap.add_argument("--unclipped-steps", type=int, default=100)
    ap.add_argument("--refreshed-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--probe-steps", type=int, default=500)
    ap.add_argument("--probe-lr", type=float, default=1e-2)
    ap.add_argument("--only", choices=["all", "refreshed", "probe", "finalize"], default="all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    samples = None
    if args.only != "finalize":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rows = [json.loads(x) for x in args.batch.read_text().splitlines() if x.strip()]
        model, tokenizer, processor = load_model(args.model_dir, device)
        samples = [prepare_sample(processor, tokenizer, r, device) for r in rows]
        del model
        torch.cuda.empty_cache()

    if args.only == "refreshed":
        refreshed = train_run("ab_refreshed_old_value", args, samples, "production_clipped", args.refreshed_steps, args.lr, True)
        print(json.dumps({"mode": "refreshed", "final": refreshed["records"][-1]}, indent=2, ensure_ascii=False))
        return
    if args.only == "probe":
        probe = hidden_probe(args, samples)
        print(json.dumps({"mode": "probe", "final": probe["final"]}, indent=2, ensure_ascii=False))
        return
    if args.only == "all":
        prod = train_run("ab_production_clipped", args, samples, "production_clipped", args.prod_steps, args.lr, False)
        audit = write_audit(args, prod)
        unclipped = train_run("ab_unclipped", args, samples, "unclipped", args.unclipped_steps, args.lr, False)
        refreshed = train_run("ab_refreshed_old_value", args, samples, "production_clipped", args.refreshed_steps, args.lr, True)
        probe = hidden_probe(args, samples)
    else:
        def load_records(name):
            p = args.out / name
            return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

        prod = {"records": load_records("ab_production_clipped_metrics.jsonl")}
        unclipped = {"records": load_records("ab_unclipped_metrics.jsonl")}
        refreshed = {"records": load_records("ab_refreshed_old_value_metrics.jsonl")}
        hidden_stats = json.loads((args.out / "hidden_state_statistics.json").read_text())
        probe = hidden_stats["linear_probe"]
        audit = json.loads((args.out / "phase2b_value_clipping_audit.json").read_text())

    u0 = unclipped["records"][0]
    uf = unclipped["records"][-1]
    hidden_stats = json.loads((args.out / "hidden_state_statistics.json").read_text())
    closed_probe = hidden_stats.get("closed_form_linear_probe", {})
    adam_probe_ok = probe["final"]["mse"] < 0.05 * max(probe["initial"]["mse"], 1e-8)
    closed_probe_ok = (closed_probe.get("mse", float("inf")) < 1e-4) and (
        closed_probe.get("full_batch_ev", -999) > 0.99
    )
    probe_ok = bool(adam_probe_ok or closed_probe_ok)
    capacity_ok = uf["vf_loss_selected"] < 0.5 * u0["vf_loss_selected"] and (uf["full_batch_ev"] or -999) > (u0["full_batch_ev"] or -999) + 0.2
    clipping_expected = audit["clipped_branch_selected_ratio"] > 0.5 or audit["boundary_saturation_ratio"] > 0.25
    clipping_label = "EXPECTED_SATURATION" if clipping_expected else "VERIFIED_NOT_PRIMARY_BLOCKER"
    phase2_pass = bool(capacity_ok and probe_ok)
    report = {
        "phase2_status": "PASS" if phase2_pass else "FAIL",
        "phase2_critic_capacity": "PASS" if capacity_ok else "FAIL",
        "phase2_production_clipping_behavior": clipping_label,
        "allow_phase3": bool(phase2_pass),
        "production_clipped": {"initial": prod["records"][0], "final": prod["records"][-1]},
        "unclipped": {"initial": u0, "final": uf},
        "refreshed_old_value": {"initial": refreshed["records"][0], "final": refreshed["records"][-1]},
        "linear_probe": probe,
        "closed_form_linear_probe": closed_probe,
        "value_readout_position": "last prompt token predicts first valid response token; first response token target is shifted by dp_critic slicing",
        "target_limit": "fixed rollout dump lacks saved old values/bootstrap/GAE tensors; targets are reconstructed capacity targets from real rollout score/reward",
        "production_ev": "dp_critic full-valid EV excludes sentinel and uses unbiased=False; microbatch=1 EV is not meaningful",
    }
    (args.out / "phase2_final_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    update_final(args, report)
    print(json.dumps({"phase2_status": report["phase2_status"], "allow_phase3": report["allow_phase3"], "report": str(args.out / "phase2_final_report.json")}, indent=2))


if __name__ == "__main__":
    main()
