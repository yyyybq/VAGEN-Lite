#!/usr/bin/env python3
"""Fixed-batch offline diagnostics for U1 Active Spatial training chain.

Does NOT resample. Re-run modes against the same JSONL for strict A/B.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/fixed_batch/fixed_batch_8.jsonl"
DEFAULT_OUT = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/reports"


def _add_sys_path():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "verl"))


def classify_format(output: str) -> str:
    has_tc = "<tool_call>" in (output or "")
    has_act = "<action>" in (output or "")
    if has_tc and has_act:
        return "tool_call_and_action"
    if has_tc:
        return "tool_call_only"
    if has_act:
        return "action_only"
    return "missing_or_invalid"


def parse_with_env(response: str):
    _add_sys_path()
    from vagen.envs.active_spatial import env as envmod

    obj = SimpleNamespace(
        _allowed_actions={
            "move_forward", "turn_left", "turn_right", "look_up", "look_down",
            "move_backward", "move_left", "move_right", "done",
        }
    )
    return envmod.ActiveSpatialEnv._default_parse_func(obj, response or "")


def load_batch(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_protocol(batch, out_dir: Path):
    _add_sys_path()
    samples = []
    for i, s in enumerate(batch):
        out = s.get("output") or ""
        inp = s.get("input") or ""
        parsed = parse_with_env(out)
        fmt = classify_format(out)
        tools_in_input = ("<tools>" in inp.lower()) or ("# tools" in inp.lower())
        samples.append(
            {
                "idx": i,
                "task_type": s.get("task_type"),
                "traj_success": s.get("traj_success"),
                "score": s.get("score"),
                "format_class": fmt,
                "parser": {
                    "format_correct": parsed.get("format_correct"),
                    "actions": parsed.get("actions"),
                    "parse_error": parsed.get("parse_error"),
                    "has_tool_call": parsed.get("has_tool_call"),
                    "fallback_parse": parsed.get("fallback_parse"),
                    "strict_parse_success": parsed.get("strict_parse_success"),
                },
                "prompt_has_tools_block": tools_in_input,
                "prompt_requires_action_tag": "<action>" in inp,
                "raw_output_preview": out[:400],
                "input_preview": inp[:400],
            }
        )

    n = max(len(samples), 1)
    rates = {
        "strict_action_tag_rate": sum(1 for x in samples if x["format_class"] == "action_only") / n,
        "tool_call_rate": sum(1 for x in samples if "tool_call" in x["format_class"]) / n,
        "fallback_parse_rate": sum(1 for x in samples if x["parser"].get("fallback_parse")) / n,
        "missing_action_rate": sum(1 for x in samples if x["format_class"] == "missing_or_invalid") / n,
        "strict_parse_success_rate": sum(1 for x in samples if x["parser"].get("strict_parse_success")) / n,
        "format_correct_rate_post_fix": sum(1 for x in samples if x["parser"].get("format_correct")) / n,
        "tools_declared_in_stored_prompts": any(x["prompt_has_tools_block"] for x in samples),
    }
    report = {
        "samples": samples,
        "rates": rates,
        "chat_template_tools_gate": {
            "tools_passed_in_short_run": False,
            "system_prompt_target": "<think>...</think><action>...</action>",
            "conclusion": (
                "<tool_call> NOT induced by tools= in this run; spontaneous model prior / format confusion. "
                "Jinja WOULD induce tool_call IF tools provided. After Phase6 fix, tool_call+action is "
                "executable fallback but format_correct=False (invalid_format_penalty)."
            ),
        },
        "actor_mask_note": (
            "Policy loss uses dense response_mask over all response LLM tokens; tool_call tokens are NOT "
            "masked out of actor loss. Protocol fix does not silently expand action-span mask to tool_call."
        ),
    }
    (out_dir / "phase2_protocol_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = [
        "# Phase2 protocol",
        json.dumps(rates, indent=2),
        "",
        report["chat_template_tools_gate"]["conclusion"],
        "",
        report["actor_mask_note"],
    ]
    (out_dir / "phase2_protocol_report.md").write_text("\n".join(md))
    print("[protocol]", json.dumps(rates))


def run_sentinel_doc(out_dir: Path):
    report = {
        "definition_file": "vagen/ray_trainer.py + vagen/custom_advantage/no_concat_gae.py",
        "gae_file": "vagen/custom_advantage/no_concat_gae.py",
        "ignore_value": -100,
        "meaning": (
            "returns init to ignore_value=-100; GAE writes real return only on first valid response token "
            "per unique turn. sentinel_ratio=(response_tokens - valid_return_tokens)/response_tokens."
        ),
        "numerator": "count of response tokens whose return == ignore_value (-100)",
        "denominator": "count of response_mask tokens (response tokens, not full sequence)",
        "sentinel_token_id": None,
        "semantic": "ignored critic target placeholder, NOT an image/pad tokenizer id",
        "stage": "after GAE / advantage computation (critic target construction)",
        "not": [
            "Not a tokenizer sentinel ID",
            "Not image placeholder count",
            "~0.995 expected for long responses with first-token value targets",
        ],
        "critic_implication": "Critic has sparse targets (~1/turn). Low explained_var can be capacity/data, not zero grads.",
        "actor_implication": "Actor uses dense response_mask; sentinel_ratio does NOT zero actor supervision.",
    }
    (out_dir / "phase3_sentinel_definition.json").write_text(json.dumps(report, indent=2))
    print("[sentinel] wrote definition")


def run_phase4_rewards(batch, out_dir: Path):
    scores = [float(s.get("score") or 0.0) for s in batch]
    successes = [float(bool(s.get("traj_success"))) for s in batch]
    by_fmt = Counter(classify_format(s.get("output") or "") for s in batch)
    rows = []
    for i, s in enumerate(batch):
        rows.append(
            {
                "idx": i,
                "task_type": s.get("task_type"),
                "format": classify_format(s.get("output") or ""),
                "score": s.get("score"),
                "traj_success": s.get("traj_success"),
                "initial_score": s.get("initial_score"),
                "final_score": s.get("final_score"),
                "invalid_action": s.get("invalid_action"),
                "missing_action_tag": s.get("missing_action_tag"),
                "n_primitive_steps": s.get("n_primitive_steps"),
                "parser_post_fix": {
                    k: parse_with_env(s.get("output") or "").get(k)
                    for k in (
                        "format_correct",
                        "actions",
                        "parse_error",
                        "fallback_parse",
                        "strict_parse_success",
                    )
                },
            }
        )

    def stats(xs):
        if not xs:
            return {}
        import statistics as st

        return {
            "n": len(xs),
            "mean": float(st.mean(xs)),
            "std": float(st.pstdev(xs)) if len(xs) > 1 else 0.0,
            "min": float(min(xs)),
            "max": float(max(xs)),
        }

    report = {
        "note": (
            "Fixed-batch rollout dumps store env score/success, not full GAE tensors. "
            "Production step31/32 logs prove advantage/return variance and critic targets; see "
            "phase5_production_update_evidence.json."
        ),
        "format_counts": dict(by_fmt),
        "score_stats": stats(scores),
        "success_rate": sum(successes) / max(len(successes), 1),
        "samples": rows,
        "production_step32_advantage": {
            "mean": -0.025390625,
            "max": 2.9375,
            "min": -0.86328125,
            "nonzero": True,
        },
        "production_step32_returns": {
            "valid_count": 16,
            "sentinel_ratio": 0.996362809729484,
            "valid_mean": 0.515625,
            "valid_min": -0.94921875,
            "valid_max": 6.03125,
        },
    }
    (out_dir / "phase4_reward_advantage_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("[phase4]", json.dumps({"score_stats": report["score_stats"], "format_counts": report["format_counts"]}))


def run_update_probe(out_dir: Path, fm_backprop: str):
    import torch
    import torch.nn as nn

    os.environ["U1_FM_BACKPROP"] = str(fm_backprop)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Linear(8, 8)
            self.critic = nn.Linear(8, 1)
            self.fm = nn.Linear(8, 8)

        def forward(self, x):
            return self.actor(x).sum()

    def checksum(m):
        h = hashlib.sha256()
        with torch.no_grad():
            for p in m.parameters():
                h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def l2delta(before, m):
        num = 0.0
        with torch.no_grad():
            for (n, p), b in zip(m.named_parameters(), before):
                num += float((p.detach().cpu() - b).pow(2).sum().item())
        return num ** 0.5

    m = Toy()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    ret = torch.randn(4, 1)
    before = [p.detach().cpu().clone() for p in m.parameters()]
    cksum0 = checksum(m)
    logits = m.actor(x)
    pg = ((logits - y) ** 2).mean()
    vf = ((m.critic(x) - ret) ** 2).mean()
    fm_raw = ((m.fm(x) - y) ** 2).mean()
    fm_back = os.environ.get("U1_FM_BACKPROP", "1") != "0"
    total = pg + vf
    if fm_back:
        total = total + 0.1 * fm_raw
    opt.zero_grad()
    total.backward()
    g_actor = float(m.actor.weight.grad.norm()) if m.actor.weight.grad is not None else None
    g_critic = float(m.critic.weight.grad.norm()) if m.critic.weight.grad is not None else None
    g_fm = float(m.fm.weight.grad.norm()) if m.fm.weight.grad is not None else None
    opt.step()
    report = {
        "U1_FM_BACKPROP_env": fm_backprop,
        "fm_backprop_resolved_bool": fm_back,
        "pg_loss_raw": float(pg.detach()),
        "value_loss_raw": float(vf.detach()),
        "fm_loss_raw": float(fm_raw.detach()),
        "fm_in_total_loss": bool(fm_back),
        "total_loss": float(total.detach()),
        "grad_norm_actor": g_actor,
        "grad_norm_critic": g_critic,
        "grad_norm_fm": g_fm,
        "fm_grad_is_none": g_fm is None,
        "actor_param_l2delta": l2delta(before, m),
        "critic_param_l2delta": float((m.critic.weight.detach().cpu() - before[2]).pow(2).sum().sqrt())
        if False
        else None,
        "checksum_before": cksum0,
        "checksum_after": checksum(m),
        "optimizer_step": True,
        "note": "Toy modules only. Real U1 proof: phase5_production_update_evidence.json + FM code path.",
    }
    # proper critic/fm deltas
    named = list(m.named_parameters())
    before_map = {n: b for (n, _), b in zip(named, before)}
    with torch.no_grad():
        report["critic_param_l2delta"] = float(
            (m.critic.weight.cpu() - before_map["critic.weight"]).pow(2).sum().sqrt()
        )
        report["fm_param_l2delta"] = float((m.fm.weight.cpu() - before_map["fm.weight"]).pow(2).sum().sqrt())
        report["actor_param_l2delta"] = float(
            (m.actor.weight.cpu() - before_map["actor.weight"]).pow(2).sum().sqrt()
        )
    (out_dir / f"phase5_toy_update_fm{fm_backprop}.json").write_text(json.dumps(report, indent=2))
    print("[toy_update]", json.dumps({k: report[k] for k in [
        "fm_backprop_resolved_bool","fm_in_total_loss","grad_norm_fm","fm_param_l2delta","actor_param_l2delta","critic_param_l2delta"
    ]}))


def run_fm_dataflow_doc(out_dir: Path):
    report = {
        "path": (
            "rollout/sample → u1_gen_pixel_values/grid_hw/valid → _prepare_single_gen_target "
            "→ FM forward (_compute_fm_aux_split) → fm_loss → (BACKPROP=0: detach into model.loss/aux; "
            "BACKPROP=1: stash → PG backward → compute_pending_fm_aux_loss → separate backward) "
            "→ optimizer → checkpoint (FM weights inside actor shards)"
        ),
        "condition": "current und prefix / optional detached und KV; gen pixel values; timestep; noise scale",
        "target": "next-frame / gen image patches → flow-matching velocity v in patch/latent space (MSE on v_pred vs image_gen_v)",
        "U1_FM_BACKPROP=0": {
            "forward": "yes under torch.no_grad",
            "enters_model_loss_tensor": "yes but detached (+ 0*input_embeds.sum())",
            "added_to_policy_loss_scalar": "yes via aux_loss * coef (value visible in logs)",
            "FM_param_grads": "no",
            "pending_twopass_backward": "no",
        },
        "U1_FM_BACKPROP=1": {
            "forward_stash": "yes",
            "pg_backward_first": "yes",
            "fm_separate_backward": "yes via compute_pending_fm_aux_loss",
            "FM_param_grads": "yes (gen Vit / mot_gen / fm_head)",
            "und_full_sequence_grads_via_FM": "no when und KV detached / und_kv=0",
        },
        "production_short_run": {
            "env": "U1_FM_BACKPROP=0",
            "logged_nfp_loss_active": 1.0,
            "interpretation": "metrics-only FM; nfp_loss value logged; FM params should not learn from FM term",
        },
        "overfit_status": "NOT YET RUN on real U1 weights in this gated pass; toy A/B only",
    }
    (out_dir / "phase7_fm_dataflow.json").write_text(json.dumps(report, indent=2))
    print("[fm_dataflow] wrote")


def write_final_delivery(out_dir: Path):
    prod = json.loads((out_dir / "phase5_production_update_evidence.json").read_text()) if (
        out_dir / "phase5_production_update_evidence.json"
    ).exists() else []
    delivery = {
        "A_root_cause": {
            "tool_call_source": (
                "Not induced by tools= / system prompt in short run; model spontaneously emits <tool_call> "
                "(often with <action>). Chat template would induce tool_call IF tools were passed."
            ),
            "protocol_vs_reward": (
                "BEFORE fix: tool_call+valid <action> => format_correct=True, format_reward=0.0, no penalty, "
                "task rewards still applied. AFTER fix: executable fallback + invalid_format_penalty, "
                "format_correct=False."
            ),
            "protocol_vs_actor_mask": (
                "Actor trains full response_mask; tool_call tokens were/are in policy loss (not masked out)."
            ),
            "sentinel_ratio": (
                "Ignored critic return tokens / response tokens (~0.995). Expected with first-token-per-turn GAE."
            ),
            "critic_not_fitting": (
                "Not 'optimizer dead': vf_loss & grad_norm nonzero. Sparse valid targets + hard returns → "
                "explained_var ~0/negative. step32 explained_var_valid≈-330 is a red flag for EV numerical "
                "stability on tiny valid_count microbatches."
            ),
            "pg_value_kl_logging": (
                "NOT a pure logging bug for late steps: train.log step31/32 shows pg_loss, kl_loss, vf_loss."
            ),
            "FM_BACKPROP_0": (
                "FM forward under no_grad; value may appear in aux/nfp metrics; FM term contributes ~0 grads; "
                "twopass pending backward disabled."
            ),
        },
        "B_training_chain_evidence": {
            "actor": {
                "source": "production train.log step31/32",
                "pg_loss": prod[-1].get("actor/pg_loss") if prod else None,
                "grad_norm": prod[-1].get("actor/grad_norm") if prod else None,
                "update_performed": prod[-1].get("trainer/actor_update_performed") if prod else None,
                "param_delta_checksum_reload": "NOT YET measured on FSDP reload in this pass",
                "one_batch_overfit": "NOT YET",
            },
            "critic": {
                "vf_loss": prod[-1].get("critic/vf_loss") if prod else None,
                "grad_norm": prod[-1].get("critic/grad_norm") if prod else None,
                "valid_count_micro": prod[-1].get("critic/valid_count") if prod else None,
                "returns_valid_count": prod[-1].get("critic/returns/valid_count") if prod else None,
                "explained_var_valid": prod[-1].get("critic/explained_var_valid") if prod else None,
                "one_batch_overfit": "NOT YET",
            },
            "fm": {
                "short_run_backprop": 0,
                "nfp_loss_logged": prod[-1].get("actor/nfp_loss") if prod else None,
                "real_FM_param_update_proof": "toy only + code path; real U1 overfit PENDING",
            },
        },
        "C_fm_evidence": "see phase7_fm_dataflow.json and phase5_toy_update_fm0/1.json",
        "D_code_changes": [
            {
                "file": "vagen/envs/active_spatial/env.py",
                "change": "tool_call => format_correct=False, executable fallback, format penalty, protocol metrics",
                "changes_formal_training": True,
            },
            {
                "file": "vagen/agent_loop/gym_agent_loop_no_concat.py",
                "change": "log strict_action_tag_rate/tool_call_rate/fallback_parse_rate/...",
                "changes_formal_training": False,
            },
            {
                "file": "vagen/ray_trainer.py",
                "change": "COMPLETE marker + whole-folder rotation; disable role-wise shard deletion",
                "changes_formal_training": True,
            },
            {
                "file": "examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_short.sh",
                "change": "max_actor/critic_ckpt_to_keep default 5 (trainer rotation uses this as keep_n)",
                "changes_formal_training": True,
            },
            {
                "file": "verl/verl/workers/actor/dp_actor.py",
                "change": "pg_loss_raw/weighted + total_loss + entropy_loss aliases",
                "changes_formal_training": False,
            },
            {
                "file": "tools/debug_u1_training_batch.py",
                "change": "fixed-batch offline diagnostic entry",
                "changes_formal_training": False,
            },
        ],
        "E_controlled_smoke": {
            "started": False,
            "reason": "Gates incomplete: real FM overfit, actor/critic one-batch overfit, protocol smoke not run",
        },
        "F_decision": "NOT_READY_FOR_TRAINING",
        "F_evidence": [
            "Protocol unify coded but not live-validated",
            "Actor/critic grads exist in production logs, but one-batch overfit + param checksum/reload not done",
            "FM_BACKPROP=1 real U1 overfit not done (required before joint training)",
            "explained_var_valid pathology on tiny critic valid_count needs attention before long run",
        ],
    }
    (out_dir / "FINAL_DELIVERY.json").write_text(json.dumps(delivery, indent=2, ensure_ascii=False))
    (out_dir / "FINAL_DELIVERY.md").write_text(
        "# U1 gated diagnostic delivery\n\n"
        f"**Decision: {delivery['F_decision']}**\n\n"
        + "\n".join(f"- {x}" for x in delivery["F_evidence"])
        + "\n\nSee FINAL_DELIVERY.json for full A–F.\n"
    )
    print("[final]", delivery["F_decision"])


def expand_to_10(batch_path, roll_dir, out_path):
    # kept for compatibility; prefer existing fixed_batch_10
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--mode",
        choices=["protocol", "sentinel", "phase4", "toy_update", "fm_doc", "final", "all_offline"],
        default="all_offline",
    )
    ap.add_argument("--rollouts", type=Path, default=DEFAULT_BATCH.parent.parent / "rollouts")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    batch_path = args.batch
    ten = args.out / "fixed_batch_10.jsonl"
    if ten.exists():
        batch_path = ten
    batch = load_batch(batch_path)
    if args.mode in ("protocol", "all_offline"):
        run_protocol(batch, args.out)
    if args.mode in ("sentinel", "all_offline"):
        run_sentinel_doc(args.out)
    if args.mode in ("phase4", "all_offline"):
        run_phase4_rewards(batch, args.out)
    if args.mode in ("toy_update", "all_offline"):
        run_update_probe(args.out, "0")
        run_update_probe(args.out, "1")
    if args.mode in ("fm_doc", "all_offline"):
        run_fm_dataflow_doc(args.out)
    if args.mode in ("final", "all_offline"):
        write_final_delivery(args.out)


if __name__ == "__main__":
    main()
