"""The model line's formal runs (step 3; the user's approval of 2026-10-26): the training-chain plans, written once
(an existing plan is never overwritten; ``--recommit`` gives an unstarted one this commit and pool).

    python -m model_matrix.formal.plans --commit HEAD sft            # the two formal SFT chains
    python -m model_matrix.formal.plans --commit HEAD sdpa-control   # the Qwen3-VL SDPA log-probability control
    python -m model_matrix.formal.plans --commit HEAD diagnostics    # decision 19: where the gap comes from
    python -m model_matrix.formal.plans --commit HEAD --date D rl-qwen3 [--seed S] [--variant V4]   # a formal RL pair
    python -m model_matrix.formal.plans --commit HEAD --date D rl-cambrian-dense   # Cambrian's dense arm again
    python -m model_matrix.formal.plans --commit HEAD --date D rl-cambrian-nfp     # decision 47 乙: S0 + NFP loss
    python -m model_matrix.formal.plans --commit HEAD --date D rl-cambrian-nfp01   # decision 49 甲: coefficient 0.1

- m_<model>_formal_sft_<date> (Qwen3-VL-8B, Cambrian-S-7B): SFT and its evaluation only ("sft_only"): C0's
  conversations in the new format (train 9,802, validation 1,293), C0's recipe, 10 epochs, each epoch evaluated
  greedily on validation (413) and test (292), the epoch picked by validation (C0's rule); then the acceptance of
  the picked epoch on the main line's acceptance set (200 reserve tasks, eval_accept_reserve200, the set the main
  line accepted its new-format SFT on): sampled (4 episodes per task, temperature 0.8, top-p 0.92: readable answers
  at least 99%, no name outside the action table) and, the user's decision 2 of 2026-10-26, greedy, both with the
  answers' token entropy (model_matrix/formal/acceptance.py).  Result names sft_<group>_e<k> (groups q3formal,
  cformal), sft_<group>_e<k>_s4 and _greedy.  No checkpoint is deleted (no "retain").  RL goes in its own plans,
  from the picked epoch ("rl_start_model").
- m_qwen3vl8b_sdpa_control_<date> (decision 1): the step-2 Qwen3-VL RL smoke again from its SFT checkpoint, the
  same V4 settings, seed and 10 updates, the trainer on SDPA attention (plan "trainer_attn") and every response
  token's two log-probabilities written (rl "logprob_tokens", model_matrix/logprob/breakdown.py); the update-10
  checkpoint evaluated on both sets (result q3v4sdpa_010).
- m_<model>_formal_rl_<date> (rl-qwen3, rl-cambrian): the formal pair, S0 and the dense arm (rl_plan).  Another seed
  (the user's decision 2 of 2026-10-27: Qwen3-VL's second seed, 20261021) gets the tag suffix _s<seed> and the run
  names <prefix>s2s0 / <prefix>s2d, everything else the first seed's.  ``--variant`` names the dense variant
  instead of the main line's best (decision 2: V4, as the first seed).
- m_cambrian7b_formal_rl_<date>_dense (rl-cambrian-dense; decision 3 of 2026-10-27): Cambrian's dense arm alone
  again (run crd), V4, seed 20261019 (the existing S0 arm's), from the same SFT epoch, on zoetrope; decision 7's
  fill rule off (no keep_fill_min: the fills are still logged, filter/keep/filled), every other stop line kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_matrix.smoke import plans as smoke

RUNS = smoke.RUNS
E = smoke.E
ACCEPT = RUNS / "eval_sets" / "eval_accept_reserve200"
ACCEPT_SRC = smoke.V2 / "eval_accept_reserve200"
FORMAT = ["--format-example", "action_names"]  # variants pass their options as given (train_chain.variant_jobs)
ACCEPT_SAMPLED = ["--samples", "4", "--temperature", "0.8", "--top-p", "0.92", "--entropy", *FORMAT]
ACCEPT_GREEDY = ["--entropy", *FORMAT]
SMOKE_Q3_SFT = RUNS / "chains" / "m_qwen3vl8b_smoke_20261024"
SMOKE_Q25_SFT = RUNS / "chains" / "m_qwen25vl7b_smoke_20261024"
REFRESHED = smoke.REFRESHED


def accept_set() -> None:
    ACCEPT.mkdir(parents=True, exist_ok=True)
    for f in ("tasks.jsonl", "selection.json"):
        if not (ACCEPT / f).exists():
            (ACCEPT / f).symlink_to(ACCEPT_SRC / f)


def sft_plan(tag: str, model: str, model_type: str, group: str, commit: str) -> dict:
    return {"tr": str(RUNS / "chains" / tag), "sft_dir": str(smoke.SFT_DIR), "commit": commit, "model": model,
            "model_type": model_type, "job_prefix": "models", "gs_root": "/mnt/umm/users/yinbaiqiao/InteriorGS_ply",
            "scenes_root": "/mnt/umm/users/yinbaiqiao/InteriorGS_meta", "keep_text": True, "sft_records": False,
            "sft_epochs": 10, "sft_groups": {group: "sft_trajectories.jsonl"}, "sft_only": True,
            "eval_sets": {k: str(v) for k, v in smoke.EVAL_SETS.items()}, "select_by": "val", "max_eval_jobs": 6,
            "pool": smoke.POOL, "gpu_log": True, "retries": 2, "endpoint": 300,
            "eval": {"options": [], "extra_sets": {"accept": str(ACCEPT)},
                     "variants": [{"suffix": "_s4", "sets": ["accept"], "options": ACCEPT_SAMPLED, "models": ["sft_pick"]},
                                  {"suffix": "_greedy", "sets": ["accept"], "options": ACCEPT_GREEDY,
                                   "models": ["sft_pick"]}]}}


def picked_model(chain: Path) -> str:
    """The SFT checkpoint a chain picked (its progress file)."""
    state = json.loads((chain / "progress.json").read_text())
    return state["steps"]["pick_sft"]["decision"]["model"]


def sdpa_control_plan(tag: str, commit: str, start: str) -> dict:
    # the smoke's plan as it was (its SFT group stays: rl-config checks the RL tasks against those conversations and
    # reads their format example; rl_start_model skips the SFT steps)
    plan = smoke.smoke(tag, smoke.QWEN3VL, "qwen3_vl", "q3smoke", "q3v4sdpa", commit)
    plan.update(rl_start_model=start, trainer_attn="sdpa", rl_name="m_q3v4sdpa")
    plan["rl"]["logprob_tokens"] = True
    return plan


def diagnostic_plans(date: str, commit: str, q25_start: str, q3_start: str) -> dict:
    """The user's decision 19 (甲) of 2026-10-26, where the Qwen3-VL probability gap comes from; RL only, no
    evaluation, both with the per-token files:
    - m_qwen25vl7b_tokens: the Qwen2.5-VL smoke's RL again from its SFT checkpoint, 3 updates, eager attention (the
      main line's), the same V4 arm and seed: Qwen2.5-VL's per-token differences, by token probability;
    - m_qwen3vl8b_restart: Qwen3-VL from the SDPA control's update-10 checkpoint, 2 updates, SDPA: in its first
      update vLLM reads the weights from disk instead of receiving them from the trainer."""
    out = {}
    for tag, model, kind, run, start, attn, endpoint in (
            (f"m_qwen25vl7b_tokens_{date}", smoke.QWEN25VL, None, "q25v4tok", q25_start, None, 3),
            (f"m_qwen3vl8b_restart_{date}", smoke.QWEN3VL, "qwen3_vl", "q3v4restart", q3_start, "sdpa", 2)):
        group = "q25smoke" if kind is None else "q3smoke"
        plan = smoke.smoke(tag, model, kind, group, run, commit)
        plan.pop("eval")  # RL only: the evaluate step is not planned
        plan.update(rl_start_model=start, endpoint=endpoint, rl_name=f"m_{run}")
        plan["rl"].update(logprob_tokens=True, checkpoint_every=endpoint)
        if attn:
            plan["trainer_attn"] = attn
        out[tag] = plan
    return out


C0_E9 = E / "interiorgs_v10_train_r11_20261016" / "lf_formal" / "model" / "checkpoint-1386"  # C0's picked epoch
VARIANTS = smoke.V2 / "analysis" / "variants" / "variants.json"  # the main line's dense-variant summary
VARIANT_RL = {"V4": {"standard": "v10_p1_Dcond", "group_filter": "keep_by_return_std", "keep_std_threshold": 0.3318,
                     "keep_std_floor": 1.0, "progress_reward_scale": 10.0 / 3.0}}  # as the main line's dv4s2 run
# the main line's formal RL (interiorgs_v10_train_r12y_20261024, V4's second seed), the format example aside
RL_GUARDS = {"kl_max": 0.2, "invalid_max": 0.1, "invalid_window": 10, "success_drop": 0.1, "drop_window": 50,
             "start_updates": 5, "keep_fill_min": 1.0, "keep_fill_at": 50}


def best_variant(path: Path = VARIANTS) -> str:
    """The dense variant with the highest pooled success in the main line's summary (the user's rule 9 / 1.6)."""
    variants = json.loads(Path(path).read_text())["variants"]
    best = max(variants.values(), key=lambda v: v["summary"]["success_pooled"])
    return best["variant"]


def rl_plan(tag: str, model: str, model_type: str, start: str, group: str, prefix: str, commit: str, variant: str,
            seed: int = 20261019, endpoint: int = 300, arms: tuple = ("s0", "d"), guards: dict = None,
            extra_arms: dict = None) -> dict:
    """A formal RL pair (the user's 2.1-2.2 of 2026-10-26): S0 and the dense arm from the picked SFT epoch, the main
    line's settings (reserve tasks, 300 updates, a checkpoint and its evaluation every 50, eager attention, the
    stop lines of decisions 2, 6, 7), the new format example; the S0 arm on h800, the dense arm on zoetrope, each
    render service with its trainer (rule 1.3); evaluations to the pool with the shorter queue.  No checkpoint is
    deleted (no "retain").  ``arms`` plans only some of the two (runs <prefix>s0, <prefix>d); ``guards`` replaces
    the stop lines (RL_GUARDS); ``extra_arms`` {suffix: (pool, rl settings)} adds S0 runs <prefix><suffix> with
    those settings over the plan's."""
    if variant not in VARIANT_RL:
        raise SystemExit(f"no RL settings for dense variant {variant}; known {sorted(VARIANT_RL)}")
    rl = {"standard": "v10_p1_S0", "seed": seed, "task_pool": "reserve", "rows": smoke.RL_BASE["rows"],
          "exclude": smoke.RL_BASE["exclude"], "format_example": "action_names",
          "guards": dict(RL_GUARDS if guards is None else guards),
          "startup": {"retries": 5, "stall_minutes": 40}, "renderer_workers": 32, "renderer_first": True,
          "renderer_repack": True}
    arm_runs = {"s0": (f"{prefix}s0", "h800", {}), "d": (f"{prefix}d", "zoetrope", {"rl": dict(VARIANT_RL[variant])})}
    runs = {arm_runs[a][0]: arm_runs[a] for a in arms}
    for suffix, (pool, settings) in (extra_arms or {}).items():
        runs[f"{prefix}{suffix}"] = (f"{prefix}{suffix}", pool, {"rl": dict(settings)})
    return {"tr": str(RUNS / "chains" / tag), "sft_dir": str(smoke.SFT_DIR), "commit": commit, "model": model,
            "model_type": model_type, "rl_start_model": start, "job_prefix": "models",
            "gs_root": "/mnt/umm/users/yinbaiqiao/InteriorGS_ply", "scenes_root": "/mnt/umm/users/yinbaiqiao/InteriorGS_meta",
            "keep_text": True, "sft_records": False, "sft_groups": {group: "sft_trajectories.jsonl"},
            "eval_sets": {k: str(v) for k, v in smoke.EVAL_SETS.items()}, "select_by": "val", "max_eval_jobs": 6,
            "pool": smoke.POOL, "run_pools": {n: pool for n, (_, pool, _) in runs.items()}, "gpu_log": True,
            "retries": 2, "startup_retries_separate": True, "record_nodes": True, "endpoint": endpoint,
            "rl_name": f"m_{prefix}rl", "eval": {"every": 50, "options": [], "stop_invalid": {"set": "val", "max": 0.10}},
            "dense_variant": variant, "rl": rl, "rl_runs": {n: spec for n, (_, _, spec) in runs.items()}}


FORMAL_RL = {"qwen3": ("m_qwen3vl8b", smoke.QWEN3VL, "qwen3_vl", "q3formal", "q3"),
             "cambrian": ("m_cambrian7b", smoke.CAMBRIAN, "cambrian_qwen", "cformal", "c")}
NO_FILL_RULE = {k: v for k, v in RL_GUARDS.items() if not k.startswith("keep_fill")}  # decision 3 of 2026-10-27
# decision 33 of 2026-10-29: two S0 arms that change one setting each against the drift of cs0 (the same seed), the
# KL penalty or the learning rate (the p1 standard's 0.05 and 2e-6; training_prep --kl-loss-coef / --actor-lr)
STAB_ARMS = {"skl": ("h800", {"kl_loss_coef": 0.15}), "slr": ("zoetrope", {"actor_lr": 1e-6})}
LATER_RULES = {"pool": "queue:h800", "suspended_grace_min": 10}  # decisions 28 (evaluations on h800) and 25
# decision 47 乙 of 2026-11-03: one S0 arm as cs0 (the same start, seed and settings) with Cambrian-S's next-frame
# prediction loss trained alongside (rl-config --nfp-loss-coef, vagen/models/cambrian_nfp.py; coefficient 1.0 as the
# August runs); the user's choice of 11-03: the trainer, its render service and the evaluations all on h800
NFP_ARMS = {"snfp": ("h800", {"nfp_loss_coef": 1.0})}
# decision 49 甲 of 2026-11-04: the same arm with the coefficient 0.1, nothing else changed (with cs0's 0 and csnfp's
# 1.0 a dose comparison; csnfp drifted from update 44); everything on h800 as csnfp (the user: "都按甲执行")
NFP01_ARMS = {"snfp01": ("h800", {"nfp_loss_coef": 0.1})}


def formal_rl_plans(which: str, date: str, commit: str, start: str, variant: str, seed: int = 20261019) -> dict:
    """{tag: plan} of the rl-* commands: rl-qwen3 / rl-cambrian the pair (another seed: tag _s<seed>, runs
    <prefix>s2s0 / <prefix>s2d), rl-cambrian-dense Cambrian's dense arm alone without the fill rule (run crd).
    Decision 33 of 2026-10-29, both with the later rules (LATER_RULES): rl-cambrian-s2 Cambrian's pair for another
    seed with the dense arm as crd (the fill rule off; runs cs2s0 / cs2d), rl-cambrian-stab the two S0 arms of
    STAB_ARMS (runs cskl / cslr, tag _stab).  Decision 47 乙 of 2026-11-03, with the later rules: rl-cambrian-nfp
    the S0 arm of NFP_ARMS (run csnfp, tag _nfp), its render service queued with its trainer (no renderer_first: one
    run, so the service goes in just before the trainer instead of the trainer waiting for a running service).
    Decision 49 甲 of 2026-11-04: rl-cambrian-nfp01 the same with NFP01_ARMS (run csnfp01, tag _nfp01)."""
    if which in ("rl-cambrian-nfp", "rl-cambrian-nfp01"):
        base, model, kind, group, prefix = FORMAL_RL["cambrian"]
        dose = "01" if which.endswith("01") else ""
        tag = f"{base}_formal_rl_{date}_nfp{dose}"
        plan = rl_plan(tag, model, kind, start, group, prefix, commit, variant, seed=seed, arms=(),
                       guards=NO_FILL_RULE, extra_arms=NFP01_ARMS if dose else NFP_ARMS)
        plan["rl"]["renderer_first"] = False
        return {tag: {**plan, **LATER_RULES, "rl_name": f"m_{prefix}nfp{dose}rl"}}
    if which == "rl-cambrian-s2":
        if seed == 20261019:
            raise SystemExit("rl-cambrian-s2 is the pair for another seed (--seed)")
        base, model, kind, group, prefix = FORMAL_RL["cambrian"]
        tag = f"{base}_formal_rl_{date}_s{seed}"
        return {tag: {**rl_plan(tag, model, kind, start, group, f"{prefix}s2", commit, variant, seed=seed,
                                guards=NO_FILL_RULE), **LATER_RULES}}
    if which == "rl-cambrian-stab":
        base, model, kind, group, prefix = FORMAL_RL["cambrian"]
        tag = f"{base}_formal_rl_{date}_stab"
        plan = rl_plan(tag, model, kind, start, group, prefix, commit, variant, seed=seed, arms=(),
                       guards=NO_FILL_RULE, extra_arms=STAB_ARMS)
        return {tag: {**plan, **LATER_RULES, "rl_name": f"m_{prefix}stabrl"}}
    if which == "rl-cambrian-dense":
        base, model, kind, group, _ = FORMAL_RL["cambrian"]
        tag = f"{base}_formal_rl_{date}_dense"
        return {tag: rl_plan(tag, model, kind, start, group, "cr", commit, variant, seed=seed, arms=("d",),
                             guards=NO_FILL_RULE)}
    base, model, kind, group, prefix = FORMAL_RL[which[3:]]
    suffix = "" if seed == 20261019 else f"_s{seed}"
    tag = f"{base}_formal_rl_{date}{suffix}"
    return {tag: rl_plan(tag, model, kind, start, group, prefix if not suffix else f"{prefix}s2", commit, variant,
                         seed=seed)}


def accept_epoch_plans(date: str, commit: str, epochs: dict) -> dict:
    """The acceptance did not pass (2026-10-26; the user's 2.1: stop and report every epoch's success, unreadable
    share and entropy): evaluation-only chains on the acceptance set --
    - every other epoch of both formal SFTs greedily with the answers' entropy (results sft_<group>_e<k>_greedy, the
      picked epoch's own run being there already);
    - Qwen2.5-VL's C0 epoch 9 (the main line's picked SFT, its own placeholder format) sampled x4 and greedily, both
      with entropy (results c0_sft_formal_e9_s4, c0_sft_formal_e9_greedy), for the side-by-side the user asked for.
    ``epochs``: {chain tag: (model type, group, picked epoch, {epoch: checkpoint})}."""
    def base(tag: str, model_type, models: dict) -> dict:
        plan = {"tr": str(RUNS / "chains" / tag), "sft_dir": str(smoke.SFT_DIR), "commit": commit, "eval_only": True,
                "job_prefix": "models", "pool": smoke.POOL, "max_eval_jobs": 6, "eval_sets": {"accept": str(ACCEPT)},
                "eval": {"options": [], "models": models}}
        if model_type:
            plan["model_type"] = model_type
        return plan

    out = {}
    for tag, (model_type, group, picked, ckpts) in epochs.items():
        models = {f"sft_{group}_e{e}_greedy": {"model": str(p), "options": ACCEPT_GREEDY}
                  for e, p in sorted(ckpts.items()) if e != picked}
        out[tag] = base(tag, model_type, models)
    c0 = f"m_qwen25vl7b_c0_accept_{date}"
    out[c0] = base(c0, None, {
        "c0_sft_formal_e9_s4": {"model": str(C0_E9), "options": ["--samples", "4", "--temperature", "0.8",
                                                                  "--top-p", "0.92", "--entropy"]},
        "c0_sft_formal_e9_greedy": {"model": str(C0_E9), "options": ["--entropy"]}})
    return out


def epoch_checkpoints(chain: Path, group: str) -> dict:
    """{epoch: checkpoint} of a formal SFT chain (checkpoint-<step>, 154 steps per epoch, in step order)."""
    model = chain / f"lf_{group}" / "model"
    steps = sorted(int(p.name.split("-")[1]) for p in model.glob("checkpoint-*") if p.name.split("-")[1].isdigit())
    return {i + 1: model / f"checkpoint-{s}" for i, s in enumerate(steps)}


def write(all_plans: dict, recommit: bool = False) -> None:
    smoke.write(all_plans, recommit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--date", default="20261026")
    ap.add_argument("--recommit", action="store_true", help="give plans not yet started this commit and pool")
    ap.add_argument("which", choices=("sft", "sdpa-control", "diagnostics", "accept-epochs", "rl-qwen3", "rl-cambrian",
                                      "rl-cambrian-dense", "rl-cambrian-s2", "rl-cambrian-stab", "rl-cambrian-nfp",
                                      "rl-cambrian-nfp01"))
    ap.add_argument("--sft-date", default="20261026", help="rl-*: the formal SFT chain's date")
    ap.add_argument("--seed", type=int, default=20261019, help="rl-*: the RL seed")
    ap.add_argument("--variant", default=None, help="rl-*: the dense variant (default: the main line's best)")
    a = ap.parse_args()
    if a.which.startswith("rl-"):
        base = FORMAL_RL["cambrian" if a.which.startswith("rl-cambrian") else a.which[3:]][0]
        start = picked_model(RUNS / "chains" / f"{base}_formal_sft_{a.sft_date}")
        variant = a.variant or best_variant()
        plans = formal_rl_plans(a.which, a.date, a.commit, start, variant, a.seed)
        write(plans, a.recommit)
        print(*plans, "start", start, "dense variant", variant)
    elif a.which == "accept-epochs":
        epochs = {}
        for tag, kind, group in ((f"m_qwen3vl8b_accept_epochs_{a.date}", "qwen3_vl", "q3formal"),
                                 (f"m_cambrian7b_accept_epochs_{a.date}", "cambrian_qwen", "cformal")):
            chain = RUNS / "chains" / tag.replace("accept_epochs", "formal_sft")
            best = json.loads((chain / "progress.json").read_text())["steps"]["pick_sft"]["decision"]["best"]
            epochs[tag] = (kind, group, int(best.rsplit("_e", 1)[1]), epoch_checkpoints(chain, group))
        write(accept_epoch_plans(a.date, a.commit, epochs), a.recommit)
    elif a.which == "diagnostics":
        q3 = RUNS / "chains" / "m_qwen3vl8b_sdpa_control_20261026" / "rl_run_q3v4sdpa" / "checkpoints" / \
            "global_step_10" / "actor" / "huggingface"
        write(diagnostic_plans(a.date, a.commit, picked_model(SMOKE_Q25_SFT), str(q3)), a.recommit)
    elif a.which == "sft":
        accept_set()
        write({f"m_qwen3vl8b_formal_sft_{a.date}": sft_plan(f"m_qwen3vl8b_formal_sft_{a.date}", smoke.QWEN3VL,
                                                             "qwen3_vl", "q3formal", a.commit),
               f"m_cambrian7b_formal_sft_{a.date}": sft_plan(f"m_cambrian7b_formal_sft_{a.date}", smoke.CAMBRIAN,
                                                              "cambrian_qwen", "cformal", a.commit)}, a.recommit)
    else:
        tag = f"m_qwen3vl8b_sdpa_control_{a.date}"
        write({tag: sdpa_control_plan(tag, a.commit, picked_model(SMOKE_Q3_SFT))}, a.recommit)


if __name__ == "__main__":
    main()
