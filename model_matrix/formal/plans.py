"""The model line's formal runs (step 3; the user's approval of 2026-10-26): the training-chain plans, written once
(an existing plan is never overwritten; ``--recommit`` gives an unstarted one this commit and pool).

    python -m model_matrix.formal.plans --commit HEAD sft            # the two formal SFT chains
    python -m model_matrix.formal.plans --commit HEAD sdpa-control   # the Qwen3-VL SDPA log-probability control
    python -m model_matrix.formal.plans --commit HEAD diagnostics    # decision 19: where the gap comes from

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
    ap.add_argument("which", choices=("sft", "sdpa-control", "diagnostics", "accept-epochs"))
    a = ap.parse_args()
    if a.which == "accept-epochs":
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
