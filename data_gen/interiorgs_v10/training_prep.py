"""Training data for InteriorGS v10 (user decisions of 2026-10-14).

SFT, first version: the formal tasks of train (training data) and val (validation data), one multi-turn
conversation per task along the shortest path the task certifies, actions only, no QA mixed in; test is
not used.  The conversations are built by ``release.build_sft_record`` (the RL env's system prompt, the
"Current camera pose" line in every user turn, at most three actions per turn).

    # CPU: frames to render (turn starts and the final state), published rows, task registry, split manifest
    python -m data_gen.interiorgs_v10.training_prep sft-frames --run OUT --split-manifest SPLIT --out SFT
    # GPU (one card, zoetrope): render the frames
    python data_gen/interiorgs_v10/pilot_sco.py prepare --pilot SFT --gs-root PLY_ROOT   (then submit / status / delete)
    # CPU: the conversations
    python -m data_gen.interiorgs_v10.training_prep sft-records --dir SFT
    # CPU: RL env YAMLs (configs/rl_env_v10_S0.yaml) and the PPO config (dense-score S0 recipe)
    python -m data_gen.interiorgs_v10.training_prep rl-config --rows SFT/rows.jsonl --ppo-template S0.yaml         --gs-root PLY_ROOT --model SFT_MODEL --name NAME --run-root RUN --out RL
    # CPU: LLaMA-Factory datasets and recipe (no QA; trajectories are not model-reviewed this round)
    python scripts/prepare_active_spatial_sft.py --trajectory SFT/sft_trajectories.jsonl \\
        --split-manifest SFT/split_manifest.json --task-registry SFT/task_registry.jsonl \\
        --qa-probability 0 --allow-unreviewed-trajectories --model MODEL --out LF

Round 12: ``rl-config --standard NAME`` takes a named standard configuration (configs/rl_standards.json: the
round-11 optimisation package with the sparse or the dense reward, v10_p1_S0 / v10_p1_D, and the candidate p2 with
48 candidate tasks per update) with the repository's PPO template (configs/ppo_template_v10.yaml); ``--seed N``
sets the run's task order (data.seed; the template's 20260930 otherwise); vLLM's sampling seed stays 0 (this
verl's rollout config has no seed field: setting one stops the trainer at start-up).

Round 12 batch: ``subsets --trajectories SFT/sft_trajectories.jsonl --rows ROWS --sizes 50,100,200,400
--equal-tasks-of 200 --link-from SFT --out DIR`` draws nested train-scene subsets (a fixed seed; each the first n
scenes of one shuffled order) with all formal tasks of their scenes for SFT and the reserve tasks of their scenes
for RL, plus an arm with the same task count over every scene.

The Qwen3.8-27B rerun (docs/q38/DESIGN_ASYNC.md): ``rl-config ... --stack q38`` writes, in place of the 8B PPO config,
the complete config of verl's V1 trainer (vagen.q38.ppo_v1): verl's generated default + configs/ppo_q38_delta.yaml +
the run's keys from the same 8B path (q38_ppo_config).  Without --stack (or --stack 8b) nothing changes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SFT_SPLITS = ("train", "val")


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def sft_states(row: Dict[str, Any], policy: Dict[str, Any]) -> List[int]:
    """State indices whose frames a conversation needs: every turn start and the final state."""
    from .admission import turn_boundaries

    return sorted(set(turn_boundaries(row, policy)))


def sft_frames(rows: Sequence[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Frames in the render job's format (pilot.py render), plain images only (no overlay)."""
    from .pilot import camera_item
    from .qa_states import certificate_poses

    out = []
    for row in rows:
        poses = certificate_poses(row)
        for k in sft_states(row, policy):
            out.append({"frame_id": f"{row['task_id']}__s{k}", "kind": "sft", "task_id": row["task_id"],
                        "scene_id": row["scene_id"], "state_id": f"s{k}", "c2w": np.asarray(poses[k]).tolist(),
                        "camera_item": camera_item(row), "overlay": {}})
    return out


def plan_frames(rows: Sequence[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Round 11: the frames of the rows' ``blocked_plan`` paths (the escaped pose, every turn start, the end), as
    states ``x<k>`` of the task, in the render job's format."""
    from .pilot import camera_item
    from .release import plan_frame_states

    out = []
    for row in rows:
        plan = row.get("blocked_plan")
        if not plan:
            continue
        for k in plan_frame_states(plan, policy):
            out.append({"frame_id": f"{row['task_id']}__x{k}", "kind": "sft", "task_id": row["task_id"],
                        "scene_id": row["scene_id"], "state_id": f"x{k}", "c2w": plan["poses"][k],
                        "camera_item": camera_item(row), "overlay": {}})
    return out


def _plan_scene(job: Tuple[str, str, List[Dict[str, Any]], str, int]) -> List[Tuple[str, Any, bool]]:
    """(task_id, plan or None, selected) for the rows of one scene (a worker of ``plan_blocked``)."""
    from .admission import load_policy
    from .release import blocked_selected, plan_blocked_escape
    from .scene import load_scene

    scenes_root, scene_id, rows, wall_model, max_actions = job
    policy = load_policy()
    selected = [r for r in rows if blocked_selected(r, policy)]
    if not selected:
        return [(r["task_id"], None, False) for r in rows]
    scene = load_scene(Path(scenes_root) / scene_id, scene_id, wall_model=wall_model)
    out = []
    for r in rows:
        if not blocked_selected(r, policy):
            out.append((r["task_id"], None, False))
            continue
        out.append((r["task_id"], plan_blocked_escape(r, policy, scene, max_actions=max_actions), True))
    return out


def plan_blocked(rows: Sequence[Dict[str, Any]], scenes_root: Path, wall_model: str, max_actions: int,
                 workers: int = 1) -> Dict[str, Any]:
    """Round 11: ``release.plan_blocked_escape`` for every row, scene by scene (``workers`` processes).  Returns
    {"plans": {task_id: plan}, "selected": n, "no_candidate": n}."""
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_scene.setdefault(str(r["scene_id"]), []).append(r)
    jobs = [(str(scenes_root), sid, rs, wall_model, int(max_actions)) for sid, rs in sorted(by_scene.items())]
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_plan_scene, jobs, chunksize=1)
    else:
        results = [_plan_scene(j) for j in jobs]
    plans, selected, missing = {}, 0, 0
    for res in results:
        for tid, plan, sel in res:
            selected += int(sel)
            if plan is not None:
                plans[tid] = plan
            elif sel:
                missing += 1
    return {"plans": plans, "selected": selected, "no_candidate": missing}


def _template_env(env_template: Optional[str]) -> Dict[str, Any]:
    import yaml

    template = Path(env_template or Path(__file__).with_name("configs") / "rl_env_v10_S0.yaml")
    return dict(yaml.safe_load(template.read_text())["envs"][0]["config"])


def blocked_report(rows: Sequence[Dict[str, Any]], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Counts of the inserted blocked turns of the SFT rows and their share of all assistant actions."""
    from .release import conversation_actions

    plans = [r["blocked_plan"] for r in rows if r.get("blocked_plan")]
    actions = sum(conversation_actions(r, policy) for r in rows)
    return {"blocked_turns": len(plans), "conversations": len(rows), "assistant_actions": actions,
            "blocked_action_share": round(len(plans) / actions, 5) if actions else 0.0,
            "blocked_moves": dict(Counter(p["move"] for p in plans)),
            "escape_moves": dict(Counter(p["escape"] for p in plans)),
            "escape_is_opposite": sum(1 for p in plans if p["escape"] == {"move_forward": "move_backward",
                                                                           "move_backward": "move_forward",
                                                                           "move_left": "move_right",
                                                                           "move_right": "move_left"}[p["move"]]),
            "blocked_by": dict(Counter(p["blocked_by"].split(" by ", 1)[-1] if p["blocked_by"].endswith("wall")
                                       else "object" for p in plans)),
            "path_lengths": dict(Counter(len(p["path"]) for p in plans)),
            "by_family": dict(Counter(r["task_type"] for r in rows if r.get("blocked_plan")))}


def check_blocked_share(report: Dict[str, Any], policy: Dict[str, Any]) -> None:
    """Decision 5 of round 11: the blocked moves stay at most ``blocked_turn_max_action_share`` of all SFT actions."""
    cap = float(policy["sft"]["export"].get("blocked_turn_max_action_share", 1.0))
    if report["blocked_action_share"] > cap:
        raise SystemExit(f"blocked moves are {report['blocked_action_share']:.2%} of the SFT actions, above the "
                         f"{cap:.0%} cap: lower sft.export.blocked_turn_share")


def cmd_sft_frames(args: argparse.Namespace) -> None:
    from .admission import load_policy
    from .release import finalize_task_row, registry_row, split_lookup

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.split_manifest).read_text())
    lookup = split_lookup(manifest)
    splits = tuple(s for s in args.splits.split(",") if s)
    policy = load_policy()
    excluded = set()
    if getattr(args, "exclude", None):  # round 10: tasks flagged by the rendered-depth check
        excluded = set(json.loads(Path(args.exclude).read_text())["flagged_task_ids"])
    sources = [("formal", Path(args.run) / "tasks.jsonl")]
    if getattr(args, "include_reserve", False):  # round 10: the reserve tasks too, marked, for the larger SFT set
        sources.append(("reserve", Path(args.run) / "tasks_reserve.jsonl"))
    rows, skipped = [], Counter()
    for kind, path in sources:
        for raw in _jsonl(path):
            split = lookup.get(str(raw["scene_id"]))
            if split is None:
                skipped["scene_not_in_manifest"] += 1
                continue
            if split not in splits:
                skipped[f"split_{split}"] += 1
                continue
            if raw["task_id"] in excluded:
                skipped[f"excluded_{kind}_{raw['task_type']}"] += 1
                continue
            rows.append(dict(finalize_task_row(raw, split), reserve=(kind == "reserve")))
    if args.limit:
        rows = rows[: args.limit]
    exp = policy["sft"]["export"]
    planned = None
    if exp.get("blocked_turn_style", "continue") == "escape" and getattr(args, "scenes_root", None):
        # round 11: the blocked turns are planned here, so the frames of their new paths get rendered
        env = _template_env(getattr(args, "env_template", None))
        planned = plan_blocked([r for r in rows if not r.get("reserve")], Path(args.scenes_root),
                               env.get("wall_model", "room_edges"), int(env.get("max_episode_steps", 12)),
                               int(getattr(args, "workers", 1) or 1))
        for r in rows:
            if r["task_id"] in planned["plans"]:
                r["blocked_plan"] = planned["plans"][r["task_id"]]
    frames = sft_frames(rows, policy) + plan_frames(rows, policy)
    _write_jsonl(out / "rows.jsonl", rows)
    _write_jsonl(out / "frames.jsonl", frames)
    # RL trains on the formal tasks of train (the reserve tasks are SFT-only)
    _write_jsonl(out / "task_registry.jsonl",
                 [registry_row(r, {"sft": True, "rl": r["split"] == "train" and not r.get("reserve")}) for r in rows])
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    report = {"tasks": len(rows), "by_split": dict(Counter(r["split"] for r in rows)),
              "by_family": dict(Counter(r["task_type"] for r in rows)), "frames": len(frames),
              "scenes": len({r["scene_id"] for r in rows}), "skipped": dict(skipped),
              "reserve": sum(1 for r in rows if r.get("reserve")), "excluded": len(excluded)}
    if planned is not None:
        report["blocked"] = dict(blocked_report(rows, policy), selected=planned["selected"],
                                 selected_without_escape=planned["no_candidate"],
                                 share=exp.get("blocked_turn_share"), seed=exp.get("blocked_turn_seed"))
    (out / "sft_frames_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if planned is not None:
        check_blocked_share(report["blocked"], policy)


def cmd_sft_records(args: argparse.Namespace) -> None:
    from .admission import load_policy
    from .pilot import safe_name
    from .release import build_sft_record

    d = Path(args.dir).resolve()
    policy = load_policy()
    rows = _jsonl(d / "rows.jsonl")
    images: Dict[str, str] = {}
    for f in _jsonl(d / "frames.jsonl"):
        path = d / "frames" / "plain" / safe_name(f["frame_id"])
        if path.is_file():
            images[f"{f['task_id']}:{f['state_id']}"] = str(path)
    # round 10 (blocked_turn_style "continue"): the scene's collision detector (the env's wall model) for the
    # inserted blocked turns; round 11 ("escape") plans them in sft-frames, so the rows carry them
    scenes_root = Path(args.scenes_root) if getattr(args, "scenes_root", None) else None
    if policy["sft"]["export"].get("blocked_turn_style", "continue") != "continue":
        scenes_root = None
    wall_model = "room_edges"
    if scenes_root is not None:
        import yaml

        template = Path(getattr(args, "env_template", None) or Path(__file__).with_name("configs") / "rl_env_v10_S0.yaml")
        wall_model = yaml.safe_load(template.read_text())["envs"][0]["config"].get("wall_model", "room_edges")
    detectors: Dict[str, Any] = {}

    def detector_for(scene_id: str):
        if scenes_root is None:
            return None
        if scene_id not in detectors:
            from .scene import load_scene

            detectors.clear()  # rows come sorted by scene; keep one scene loaded
            detectors[scene_id] = load_scene(scenes_root / scene_id, scene_id, wall_model=wall_model).detector
        return detectors[scene_id]

    records, reasons = [], Counter()
    for row in rows:
        rec, reason = build_sft_record(row, images, policy, detector=detector_for(str(row["scene_id"])))
        if rec is None:
            reasons[reason] += 1
            continue
        rec["reserve"] = bool(row.get("reserve"))
        records.append(rec)
    _write_jsonl(d / "sft_trajectories.jsonl", records)
    formal = [r for r in records if not r["reserve"]]
    if len(formal) < len(records):  # the smaller SFT set: formal tasks only
        _write_jsonl(d / "sft_trajectories_formal.jsonl", formal)
    report = {"tasks": len(rows), "records": len(records), "dropped": dict(reasons),
              "by_split": dict(Counter(r["split"] for r in records)),
              "turns": dict(Counter(r["trajectory_steps"] for r in records)),
              "images": len(images), "reserve_records": len(records) - len(formal),
              "blocked_turns": sum(1 for r in records if r.get("blocked_insert")),
              "blocked_moves": dict(Counter(r["blocked_insert"]["move"] for r in records if r.get("blocked_insert"))),
              "wall_model": wall_model if scenes_root is not None else None}
    actions = sum(int(r["total_actions"]) for r in records)
    report["assistant_actions"] = actions
    report["blocked_action_share"] = round(report["blocked_turns"] / actions, 5) if actions else 0.0
    (d / "sft_records_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    check_blocked_share(report, policy)  # round 11: at most 3% of the SFT actions are blocked moves


# ---------------------------------------------------------------------------
# RL: env YAMLs and the PPO config (dense-score S0 recipe)
# ---------------------------------------------------------------------------
# One card, a handful of updates: does the data load, does the env render, are rewards and success sane,
# is a checkpoint written?  (The formal recipe needs 8 cards for the trainer plus one for the renderer.)
TRIAL_PPO: Dict[str, Any] = {
    "trainer.n_gpus_per_node": 1,
    "trainer.save_freq": 1,
    "trainer.test_freq": 2,
    "trainer.critic_warmup": 0,
    "trainer.ckpt_milestones": [],
    # retention at trial scale: every 2nd checkpoint kept, of the others the latest one
    "trainer.ckpt_keep_every": 2,
    "trainer.max_actor_ckpt_to_keep": 1,
    "trainer.max_critic_ckpt_to_keep": 1,
    "trainer.logger": ["console"],
    "data.train_batch_size": 4,
    "data.val_batch_size": 4,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 4,
    "actor_rollout_ref.rollout.n": 2,
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.agent.num_workers": 4,
    # one card holds vLLM, the render service and, during its optimizer step, the critic's fp32 weights,
    # gradients and Adam state (offloaded otherwise): vLLM gets 20% instead of 30% (12% leaves no KV cache)
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.2,
    # the critic's update holds fp32 weights, gradients and Adam state (~60 GB for a 3.75B model): on one card
    # the trial keeps weights in bf16 (the formal 8-card run keeps fp32 master weights)
    "actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
    "actor_rollout_ref.ref.fsdp_config.model_dtype": "bf16",
    "critic.model.fsdp_config.model_dtype": "bf16",
}


# The formal run (decisions of 2026-10-14): a checkpoint every 25 updates; every 50th is kept, of the others only
# the latest 2 (vagen/interiorgs_v10_ppo.py retained_steps); validation at every checkpoint; a restart resumes
# from the latest checkpoint.  The trial overrides apply on top.
FORMAL_PPO: Dict[str, Any] = {
    "trainer.save_freq": 25,
    "trainer.test_freq": 25,
    "trainer.resume_mode": "auto",
    "trainer.ckpt_keep_every": 50,
    "trainer.max_actor_ckpt_to_keep": 2,
    "trainer.max_critic_ckpt_to_keep": 2,
    "trainer.ckpt_milestones": [],
}


# Round 10 (2026-10-15): how the model's input carries the earlier turns of an episode ("history", the fifth
# default rule).  "none": as rounds 6-9, each turn a separate sample with the current frame only; "no_concat":
# each turn a separate sample carrying every earlier turn (the env setting history_window_size = the turn limit;
# the no-concat agent loop); "concat": one sequence per episode as in SFT (verl's multi-turn: the concat agent
# loop, trainer.concat_multi_turn, GAE over the whole sequence; the response budget must hold every later turn).
HISTORY_MODES = ("none", "no_concat", "concat")
CONCAT_RESPONSE_LENGTH = 4096  # 12 turns x (81 image tokens + ~150 text + the answer) stay well below this


# Round 11: the reward presets of the RL env.  "S0": the template's sparse reward (success +5, step -0.01, blocked
# move -0.15, invalid answer -0.1, near-uniform view -0.2).  "dense": the dense-score pilot's S5 branch on top of
# it (exps/.../dense_score_reward_only_pilot_parallel_r3_20261009/frozen/configs/S5.train.yaml): progress
# shaping on the canonical metric's 0..1 progress (potential mode, gamma 0.99), a bonus while that progress is
# at least near_success_threshold, and the visibility shaping.  DENSE_ADJUST holds the changes made after the
# round-11 checks of the reward in the env (shortest paths of 120 formal train tasks, no rendering):
#   potential_field_reward_scale 10   at scale 1 the progress term adds +0.014 on average over a whole shortest
#                                     path (at most 0.37), below 1% of the success reward 5 and of the order of
#                                     the step penalty; at 10: +0.14 on average, up to 3.7, turning in place -0.16;
#   enable_near_success_reward off    the bonus needs the potential (half the 0..1 progress, which is capped while
#                                     the task is unmet) at 0.55 or more, never reached before success: it never
#                                     pays; lowered, it would pay every turn spent next to the goal (hovering);
#   enable_visibility_shaping_reward off  inactive for v10 tasks (it needs a single target centre): off says so.
# The system prompt stays the SFT one for both (its reward lines describe the sparse terms), so the two runs
# differ only in the reward.
REWARD_PRESETS = ("S0", "dense", "D_cond")
TASK_POOLS = ("formal", "reserve")  # which train tasks RL takes (round 11: reserve, the tasks SFT never saw)

# Round 11, the optimisation package (the user's decision after R-S0 / R-D left the model where it started: KL to
# the start model 0.002 at update 150, about 3/4 of the tasks all succeeding or all failing).  On top of the run's
# settings: GRPO (the episodes of one task compared with each other) instead of the value network; only tasks
# whose episodes both succeed and fail are trained on (vagen's mixed_outcome filter; each update draws
# ``candidates`` tasks, the first ``target_groups`` are the nominal batch and the extra ones refill dropped tasks);
# ``rollouts`` episodes per task; the learning rate, KL penalty and entropy bonus below.  p1, with the reasons:
#   rollouts 8, candidates 24, target_groups 12: from the epoch-9 start about 24% of tasks are mixed with 8
#       episodes (a beta-binomial fit to its 4-episode outcomes), so 24 candidates (192 episodes, twice the
#       current generation, about 2.5 minutes per update) keep about 6 tasks (48 episodes, all informative,
#       against about 24 informative of the current 96);
#   lr 2e-6 (5e-7 before): the per-update policy change was about 1e-4 and the KL stalled at 0.002; 4x the step;
#   kl_loss_coef 0.05 (0.3 before): the KL penalty held the policy at the start model;
#   entropy_coeff 0.01 (0.005 before): entropy fell from 0.077 to 0.046 with the old step; the larger step needs
#       a stronger bonus to keep it above 70% of the start.
# Round 12, p2 (the candidate standard): p1 with 48 candidates.  With 24, P-S0 / P-D kept about 6 tasks per update
# (about 4 dropped as all success, 5 as all failure, 3 refilled, 6 short of 12); 48 candidates fill the 12 tasks
# (at about 25% mixed, 48 x 0.25 = 12) for twice the generation (about 4 minutes per update instead of about 2).
# The main line's PPO supplement, p1gae (the user's decision 甲 of 2026-11-08): p1 with GAE and a value network
# instead of GRPO; nothing else changes (rollouts, candidates, the mixed_outcome filter, lr, KL, entropy as p1).  The
# value network is the template's critic block (started from the RL start model: lr 2e-5 constant, fp32, parameters
# and optimizer offloaded, cliprange_value 0.5) with the template's gamma 0.95 / lam 0.95, as the round-11 R-S0 PPO.
# critic_warmup 11: verl counts its updates from 1 and updates the actor at every update >= critic_warmup
# (vagen/ray_trainer.py), so the value network trains alone for 10 updates and endpoint 310 gives 300 actor
# updates.  A package's estimator and warm-up are applied with its other keys, last (package_overrides; the
# defaults "grpo" and 0 keep p1 / p2 as they were).
OPT_PACKAGES: Dict[str, Dict[str, Any]] = {
    "p1": {"rollouts": 8, "candidates": 24, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
           "entropy_coeff": 0.01},
    "p2": {"rollouts": 8, "candidates": 48, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
           "entropy_coeff": 0.01},
    "p1gae": {"rollouts": 8, "candidates": 24, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
              "entropy_coeff": 0.01, "adv_estimator": "gae", "critic_warmup": 11},
    # the main line's per-turn PPO (the user's decision 1A of 2026-11-09): p1gae with GAE over the episode's turns
    # (vagen/turn_gae.py: each turn's env reward on that turn, the value at the turn's first token, gamma / lam
    # counted per turn: 1.0, so the return is the undiscounted episode reward GRPO compares, and 0.95); verl starts
    # the value network only for the estimators it knows, so the package turns it on (critic.enable)
    "p1tgae": {"rollouts": 8, "candidates": 24, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
               "entropy_coeff": 0.01, "adv_estimator": "turn_gae", "critic_warmup": 11, "gamma": 1.0, "lam": 0.95},
    # the critic pilot (the user's supplement of 2026-11-10, section 2): p1tgae's value network did not learn (its
    # explained variance at the turns' first tokens stayed about 0 over 300 updates), so this package changes its four
    # settings and nothing else: critic lr 2e-5 -> 5e-5 (critic.optim.lr), no value clipping (critic.cliprange_value
    # NO_VALUE_CLIP: verl's clip_by_value leaves every prediction inside so wide a range, so the loss is the plain
    # squared error), two passes of the value network over each update's batch (critic.ppo_epochs 2; the actor keeps
    # its one) and critic_warmup 30 (the value network alone for 29 updates).  A new package, default off: p1tgae and
    # the standards naming it keep their configs
    "p1tgae_c": {"rollouts": 8, "candidates": 24, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
                 "entropy_coeff": 0.01, "adv_estimator": "turn_gae", "critic_warmup": 30, "gamma": 1.0, "lam": 0.95,
                 "critic_lr": 5e-5, "critic_value_clip": False, "critic_ppo_epochs": 2},
    # fallback (a) of PPO_REPORT appendix A (the user's decision of 2026-11-11, after p1tgae_c failed level 1a):
    # p1tgae_c with the value network's data and steps changed and nothing else: it updates on every candidate
    # episode before the group filter (trainer.critic_all_candidates: 192, the all-success and all-fail groups too,
    # against about 56 kept), in mini-batches of 3 groups (critic.ppo_mini_batch_size 3: 8 optimiser steps a pass,
    # 16 an update with its two passes).  The actor's batch, filter, whitening, rewards and stop lines are p1tgae_c's
    "p1tgae_c2": {"rollouts": 8, "candidates": 24, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
                  "entropy_coeff": 0.01, "adv_estimator": "turn_gae", "critic_warmup": 30, "gamma": 1.0, "lam": 0.95,
                  "critic_lr": 5e-5, "critic_value_clip": False, "critic_ppo_epochs": 2,
                  "critic_all_candidates": True, "critic_mini_batch_groups": 3},
    # the critic overfit check (the user's supplement of 2026-11-11, item 2, after p1tgae_c2 failed level 1a too):
    # p1tgae_c2 with trainer.critic_overfit_steps 50: the trainer's first update runs as p1tgae_c2's, and its value-
    # network batch (every candidate, 192) is kept fixed; updates 2-50 train only the value network on that batch
    # (no new rollouts, no actor update: the actor is never updated in this mode), logging the readings on it at
    # every update and once more after the 50th (critic/overfit/final/*).  A diagnostic, not a training recipe
    "p1tgae_c2_ofit": {"rollouts": 8, "candidates": 24, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
                       "entropy_coeff": 0.01, "adv_estimator": "turn_gae", "critic_warmup": 30, "gamma": 1.0,
                       "lam": 0.95, "critic_lr": 5e-5, "critic_value_clip": False, "critic_ppo_epochs": 2,
                       "critic_all_candidates": True, "critic_mini_batch_groups": 3, "critic_overfit_steps": 50},
    # the critic replay pilot (the user's supplement of 2026-11-12, item 4: "critic 不再修" lifted for this one time;
    # the user's choice of the same day, option 1: 8 updates, one pass): p1tgae_c2 with trainer.critic_replay_updates 8
    # and critic.ppo_epochs 1 (p1tgae_c2: 2): every update the value network trains one pass on the every-candidate
    # batches of the last 8 updates (this one included, each with the returns of its own update) instead of two passes
    # on this update's alone, so each episode gets 8 passes over its life (p1tgae_c2: 2).  The actor, its batch, the
    # filter, whitening, rewards and stop lines are p1tgae_c2's
    "p1tgae_c2_rb": {"rollouts": 8, "candidates": 24, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
                     "entropy_coeff": 0.01, "adv_estimator": "turn_gae", "critic_warmup": 30, "gamma": 1.0,
                     "lam": 0.95, "critic_lr": 5e-5, "critic_value_clip": False, "critic_ppo_epochs": 1,
                     "critic_all_candidates": True, "critic_mini_batch_groups": 3, "critic_replay_updates": 8},
    # the privileged critic (the user's supplement of 2026-11-13, gate 1 of docs/PREREG_PPO2.md): p1tgae_c2's actor,
    # batch, filter, warm-up 30 and every-candidate value data, with the value from a three-layer MLP in the trainer
    # on the env's ground truth instead of the value network (vagen/feature_critic.py: the remaining minimum number of
    # actions from the task's distance table, the bearing sector, the distance condition, the turn's index, the
    # family; success cross-entropy plus the turns-left regression, a least-squares read-out to the return with the
    # env's potential phi as one more read-out term (readout_phi: a dense reward's return-to-go carries -scale * phi);
    # a buffer of the last 8 updates, 4 passes an update): no critic worker (critic.enable false); the env gives the
    # truth (critic_features on, its tables from --critic-feature-tables, usually the standard's).  gamma 1 / lam 0.95
    "p1tgae_fc": {"rollouts": 8, "candidates": 24, "target_groups": 12, "lr": 2e-6, "kl_loss_coef": 0.05,
                  "entropy_coeff": 0.01, "adv_estimator": "turn_gae", "critic_warmup": 30, "gamma": 1.0, "lam": 0.95,
                  "critic_all_candidates": True, "critic_source": "features",
                  "feature_critic": {"hidden": 128, "lr": 1e-3, "epochs": 4, "mini_batch": 256, "buffer_updates": 8,
                                     "steps_weight": 1.0, "ridge": 1e-4, "threads": 4, "readout_phi": True}},
}
NO_VALUE_CLIP = 1.0e6  # critic.cliprange_value of a package without value clipping (far beyond any return)
CRITIC_KEYS = ("critic_lr", "critic_value_clip", "critic_ppo_epochs")  # a package's value-network settings
CRITIC_BATCH_KEYS = ("critic_all_candidates", "critic_mini_batch_groups")  # the value network's data (p1tgae_c2)
CRITIC_CHECK_KEYS = ("critic_overfit_steps",)  # the critic overfit check (p1tgae_c2_ofit)
CRITIC_REPLAY_KEYS = ("critic_replay_updates",)  # the critic replay pilot (p1tgae_c2_rb)
CRITIC_SOURCE_KEYS = ("critic_source", "feature_critic")  # the privileged critic (p1tgae_fc)
PACKAGE_ESTIMATORS = ("grpo", "gae", "turn_gae")  # the advantage estimators a package may name (default grpo)
CRITIC_ESTIMATORS = ("gae", "turn_gae")  # the ones with a value network (their warm-up is part of the package)

# Round 12 (the user's handover for the experiment matrix): named standard RL configurations, rl-config --standard
# NAME.  Each fixes the rl-config options that make the recipe (configs/rl_standards.json); options a standard does
# not name stay free (model, rows, run folders, seed), and an explicit option that contradicts the standard is
# refused.  Template paths are relative to this package.
STANDARDS_FILE = Path(__file__).with_name("configs") / "rl_standards.json"
STANDARD_KEYS = ("history_mode", "task_pool", "save_freq", "no_internal_val", "opt_package", "reward_preset",
                 "ppo_template", "env_template", "group_filter", "keep_std_threshold", "keep_std_floor",
                 "format_example", "lenient_parse", "invalid_penalty", "invalid_turn_nonpositive",
                 "turn_footprint_radius",  # experiment 3A (the user's decision 甲 of 2026-11-06)
                 "critic_feature_tables")  # the privileged critic (2026-11-13): the train tasks' distance tables
# Round 12, the dense-reward first batch (D-keep, the user's decision of 2026-10-21): which group filter the
# package's batch goes through.  mixed_outcome (the package's) trains only on tasks whose episodes both succeed and
# fail; keep_by_return_std (vagen/custom_filter/filter.py) fills the places still free with all-fail tasks whose
# episode returns spread at least keep_std_threshold (standard deviation, n - 1), their group comparison dividing
# by at least keep_std_floor.
GROUP_FILTERS = ("mixed_outcome", "keep_by_return_std", "nominal")  # nominal: reward line 2026-11-01, every outcome


def load_standards(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """{name: rl-config options} of the named standard configurations (keys starting with "_" are notes;
    "chain_defaults" is the standard's chain settings, load_chain_defaults)."""
    data = json.loads(Path(path or STANDARDS_FILE).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_") and k != "chain_defaults"}


def load_chain_defaults(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Round 12 freeze (the user's decision 3 of 2026-10-22): {"rl": run settings, "plan": plan settings} the
    training chain gives every run whose plan rl names a standard, under the plan's own values."""
    data = json.loads(Path(path or STANDARDS_FILE).read_text(encoding="utf-8"))
    return dict(data.get("chain_defaults") or {})


def apply_standard(args: argparse.Namespace, name: str, standards: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Fill ``args`` with the standard's options; an option given explicitly with another value is refused."""
    standards = standards if standards is not None else load_standards()
    if name not in standards:
        raise SystemExit(f"--standard must be one of {sorted(standards)}, not {name!r}")
    std = {k: v for k, v in standards[name].items() if k in STANDARD_KEYS}
    for key, value in std.items():
        if key in ("ppo_template", "env_template"):
            value = str((Path(__file__).parent / value).resolve()) if not Path(value).is_absolute() else value
        current = getattr(args, key, None)
        if current in (None, False):
            setattr(args, key, value)
        elif (str(Path(current).resolve()) if key.endswith("_template") else current) != value:
            raise SystemExit(f"--standard {name} sets {key} to {value!r}; the command gives {current!r}")
    return std


def package_overrides(name: str, candidates: Optional[int] = None, target_groups: Optional[int] = None,
                      rollouts: Optional[int] = None) -> Dict[str, Any]:
    """PPO-config overrides of an optimisation package (dotted keys), with optional sizes (one-card pilots)."""
    if name not in OPT_PACKAGES:
        raise ValueError(f"optimisation package must be one of {sorted(OPT_PACKAGES)}, not {name!r}")
    p = dict(OPT_PACKAGES[name])
    for key, value in (("candidates", candidates), ("target_groups", target_groups), ("rollouts", rollouts)):
        if value:
            p[key] = int(value)
    if p["candidates"] < p["target_groups"]:
        raise ValueError(f"candidates ({p['candidates']}) must be at least target_groups ({p['target_groups']})")
    estimator = p.get("adv_estimator", "grpo")
    if estimator not in PACKAGE_ESTIMATORS:
        raise ValueError(f"package {name}: adv_estimator must be one of {PACKAGE_ESTIMATORS}, not {estimator!r}")
    # norm_adv_by_std_in_grpo stays True: GAE ignores it
    out = {"algorithm.adv_estimator": estimator, "algorithm.norm_adv_by_std_in_grpo": True,
           "actor_rollout_ref.rollout.n": p["rollouts"], "data.train_batch_size": p["candidates"],
           "actor_rollout_ref.actor.ppo_mini_batch_size": p["target_groups"],
           "filter.enable": True, "filter.name": "mixed_outcome",
           "filter.filter_kwargs": {"target_groups": p["target_groups"]},
           "actor_rollout_ref.actor.optim.lr": p["lr"], "actor_rollout_ref.actor.kl_loss_coef": p["kl_loss_coef"],
           "actor_rollout_ref.actor.entropy_coeff": p["entropy_coeff"],
           "trainer.critic_warmup": int(p.get("critic_warmup", 0))}
    if estimator == "turn_gae":  # its own gamma / lam (per turn) and the value network verl would not start
        if "gamma" not in p or "lam" not in p:
            raise ValueError(f"package {name}: turn_gae needs its own gamma and lam (counted per turn)")
        out.update({"algorithm.gamma": float(p["gamma"]), "algorithm.lam": float(p["lam"]), "critic.enable": True})
    named = [k for k in CRITIC_KEYS + CRITIC_BATCH_KEYS + CRITIC_CHECK_KEYS + CRITIC_REPLAY_KEYS + CRITIC_SOURCE_KEYS
             if k in p]  # the critic pilot (2026-11-10): the value network's own settings
    if named and estimator not in CRITIC_ESTIMATORS:
        raise ValueError(f"package {name}: {named} set the value network, which {estimator!r} does not have")
    if "critic_source" in p or "feature_critic" in p:  # the privileged critic (2026-11-13): no critic worker
        if p.get("critic_source") != "features" or not isinstance(p.get("feature_critic"), dict):
            raise ValueError(f"package {name}: critic_source can only be 'features', with its feature_critic settings")
        if estimator != "turn_gae":
            raise ValueError(f"package {name}: the privileged critic is for turn_gae, not {estimator!r}")
        if p.get("critic_all_candidates") is not True:
            raise ValueError(f"package {name}: the privileged critic trains on every candidate (critic_all_candidates)")
        worker = [k for k in CRITIC_KEYS + ("critic_mini_batch_groups",) + CRITIC_CHECK_KEYS + CRITIC_REPLAY_KEYS
                  if k in p]
        if worker:
            raise ValueError(f"package {name}: {worker} set the value network, which the privileged critic replaces")
        from vagen.feature_critic import settings as _feature_settings

        _feature_settings(dict(p["feature_critic"], max_turns=1))  # the settings' check (max_turns from rl-config)
        out.update({"critic.enable": False, "trainer.critic_source": "features",
                    "trainer.feature_critic": dict(p["feature_critic"])})
    if "critic_lr" in p:
        if isinstance(p["critic_lr"], bool) or not float(p["critic_lr"]) > 0:
            raise ValueError(f"package {name}: critic_lr must be positive, not {p['critic_lr']!r}")
        out["critic.optim.lr"] = float(p["critic_lr"])
    if "critic_value_clip" in p:  # only turning the clip off is a setting (the template's 0.5 is the default)
        if p["critic_value_clip"] is not False:
            raise ValueError(f"package {name}: critic_value_clip can only be False (no value clipping), "
                             f"not {p['critic_value_clip']!r}")
        out["critic.cliprange_value"] = NO_VALUE_CLIP
    if "critic_ppo_epochs" in p:
        epochs = p["critic_ppo_epochs"]
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"package {name}: critic_ppo_epochs must be a whole number >= 1, not {epochs!r}")
        out["critic.ppo_epochs"] = int(epochs)
    if "critic_all_candidates" in p:  # fallback (a), 2026-11-11: only turning it on is a setting (off is the default)
        if p["critic_all_candidates"] is not True:
            raise ValueError(f"package {name}: critic_all_candidates can only be True, not {p['critic_all_candidates']!r}")
        out["trainer.critic_all_candidates"] = True
    if "critic_mini_batch_groups" in p:  # in groups (verl multiplies by the rollouts and divides by the cards)
        groups = p["critic_mini_batch_groups"]
        if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1 or p["candidates"] % groups:
            raise ValueError(f"package {name}: critic_mini_batch_groups must be a whole number >= 1 dividing the "
                             f"{p['candidates']} candidates, not {groups!r}")
        out["critic.ppo_mini_batch_size"] = int(groups)
    if "critic_overfit_steps" in p:  # the overfit check, 2026-11-11: on the fixed batch of every candidate only
        steps = p["critic_overfit_steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError(f"package {name}: critic_overfit_steps must be a whole number >= 2, not {steps!r}")
        if p.get("critic_all_candidates") is not True:
            raise ValueError(f"package {name}: critic_overfit_steps needs critic_all_candidates (the fixed batch is "
                             "every candidate)")
        out["trainer.critic_overfit_steps"] = int(steps)
    if "critic_replay_updates" in p:  # the replay pilot, 2026-11-12: the value network on the last K updates' batches
        k = p["critic_replay_updates"]
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"package {name}: critic_replay_updates must be a whole number >= 1, not {k!r}")
        if p.get("critic_all_candidates") is not True:
            raise ValueError(f"package {name}: critic_replay_updates needs critic_all_candidates (the buffer keeps "
                             "every candidate)")
        if "critic_overfit_steps" in p:
            raise ValueError(f"package {name}: critic_replay_updates cannot be combined with critic_overfit_steps")
        out["trainer.critic_replay_updates"] = int(k)
    return out
DENSE_S5 = {"enable_potential_shaping_reward": True, "potential_field_gamma": 0.99,
            "enable_near_success_reward": True, "near_success_bonus": 0.5, "near_success_threshold": 0.55,
            "enable_visibility_shaping_reward": True}
DENSE_ADJUST: Dict[str, Any] = {"potential_field_reward_scale": 10.0, "enable_near_success_reward": False,
                                "enable_visibility_shaping_reward": False}


# Round 12 freeze (the user's decision of 2026-10-22): D_cond is the dense reward with the grouped, gated progress
# (target_visibility.PROGRESS_VERSIONS, docs/dense_reward_dcond.md): same potential, same scale and discount.
DENSE_COND: Dict[str, Any] = {"progress_version": "grouped_gated"}


def reward_overrides(preset: str) -> Dict[str, Any]:
    """Env-config overrides of a reward preset."""
    if preset not in REWARD_PRESETS:
        raise ValueError(f"reward preset must be one of {REWARD_PRESETS}, not {preset!r}")
    if preset == "S0":
        return {}
    return {**DENSE_S5, **DENSE_ADJUST, **(DENSE_COND if preset == "D_cond" else {})}


def _three_numbers(text: Any, name: str) -> List[float]:
    values = [float(x) for x in (text.split(",") if isinstance(text, str) else text)]
    if len(values) != 3 or min(values) < 0:
        raise SystemExit(f"{name} must be three non-negative numbers (frame, face, relation), not {text!r}")
    return values


def design_variant_overrides(args: argparse.Namespace, env: Dict[str, Any]) -> Dict[str, Any]:
    """Round 12 freeze, the dense-reward design variants (the user's 10th supplement of 2026-10-23): env-config
    overrides on top of the reward preset's (``env``, already merged): --progress-version (grouped_soft, V3),
    --progress-weights (V1 / V2), --progress-reward-scale (the potential's scale, V4 / V5), --milestone-rewards (V6).
    Weights and milestones need a grouped progress version and a dense preset."""
    out: Dict[str, Any] = {}
    if getattr(args, "progress_version", None):
        out["progress_version"] = args.progress_version
    version = out.get("progress_version", env.get("progress_version", "product"))
    if getattr(args, "progress_weights", None):
        out["progress_weights"] = _three_numbers(args.progress_weights, "--progress-weights")
    if getattr(args, "milestone_rewards", None):
        out["milestone_rewards"] = _three_numbers(args.milestone_rewards, "--milestone-rewards")
    if getattr(args, "progress_reward_scale", None) is not None:
        out["potential_field_reward_scale"] = float(args.progress_reward_scale)
    if version == "optimal" or getattr(args, "distance_tables", None):  # round 13: the optimal dense reward (B)
        if version != "optimal" or not getattr(args, "distance_tables", None):
            raise SystemExit("--progress-version optimal and --distance-tables go together")
        if not Path(args.distance_tables).is_dir():
            raise SystemExit(f"--distance-tables {args.distance_tables} is not a directory")
        out["distance_tables_dir"] = str(Path(args.distance_tables).resolve())
    if getattr(args, "table_near", False):  # reward line (phase 2): the table's neighbour-cell fallback
        if version != "optimal":
            raise SystemExit("--table-near needs --progress-version optimal --distance-tables")
        out["optimal_progress_near"] = True
    if out and not env.get("enable_potential_shaping_reward"):
        # reward line (2026-11-01): the per-turn credit reads the table's progress score with the S0 reward (the env's
        # progress reward off); only the score's version, tables and lookup, nothing that shapes a reward
        if not (getattr(args, "turn_credit", None)
                and set(out) <= {"progress_version", "distance_tables_dir", "optimal_progress_near"}):
            raise SystemExit("the progress options change the dense reward: give a dense --reward-preset (dense, D_cond)")
    if ("progress_weights" in out or "milestone_rewards" in out) and version == "product":
        raise SystemExit("--progress-weights / --milestone-rewards need a grouped progress version (D_cond, or "
                         "--progress-version grouped_gated / grouped_soft)")
    return out


def history_overrides(mode: str, max_turns: int) -> Dict[str, Dict[str, Any]]:
    """{"env": env-config overrides, "ppo": PPO-config overrides} for a history mode."""
    if mode not in HISTORY_MODES:
        raise ValueError(f"history mode must be one of {HISTORY_MODES}, not {mode!r}")
    if mode == "none":
        return {"env": {}, "ppo": {}}
    if mode == "no_concat":
        return {"env": {"history_window_size": int(max_turns)}, "ppo": {}}
    return {"env": {}, "ppo": {"trainer.concat_multi_turn": True, "algorithm.adv_estimator": "gae",
                               "actor_rollout_ref.rollout.agent.agent_loop_config_path": "vagen/configs/agent.yaml",
                               "data.max_response_length": CONCAT_RESPONSE_LENGTH}}


def _set(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def env_system_prompt(env_cfg: Dict[str, Any], task_type: str) -> str:
    """The system prompt the RL env shows for this env config (the arguments env.system_prompt passes)."""
    from vagen.envs.active_spatial import prompt as P
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    keys = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    c = ActiveSpatialEnvConfig(**{k: v for k, v in env_cfg.items() if k in keys})
    return P.system_prompt(**c.system_prompt_arguments(task_type))  # every setting the env itself passes


def sft_format_examples(paths: Sequence[Path], max_actions: int = 5, action_sep: str = "|") -> List[str]:
    """Which format example (prompt.FORMAT_EXAMPLES) the SFT conversations show: the format line of the first user
    turn of each file's first record.  Round 12 freeze (the user's 3.1.1): an RL run must show the text its start
    model trained with.  Files whose first turn shows neither are left out."""
    from vagen.envs.active_spatial import prompt as P

    found = set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            line = fh.readline()
        if not line.strip():
            continue
        turns = json.loads(line).get("conversations") or []
        text = next((str(t.get("content", "")) for t in turns if t.get("role") == "user"), "")
        found |= {f for f in P.FORMAT_EXAMPLES if P.action_format_line(action_sep, max_actions, f) in text}
    return sorted(found)


def with_format_example(policy: Dict[str, Any], format_example: str) -> Dict[str, Any]:
    """The policy with its SFT export's format example set (the prompt check after the SFT data decided it)."""
    import copy

    policy = copy.deepcopy(policy)
    policy["sft"]["export"]["format_example"] = format_example
    return policy


def prompt_mismatches(env_cfg: Dict[str, Any], task_types: Sequence[str], policy: Dict[str, Any]) -> List[str]:
    """Task types whose RL system prompt differs from the SFT conversations' (release.protocol_system_prompt)."""
    from .release import protocol_system_prompt

    return [t for t in sorted(set(task_types)) if env_system_prompt(env_cfg, t) != protocol_system_prompt(t, policy)]


# ---------------------------------------------------------------------------
# The q38 stack (the Qwen3.8-27B rerun, docs/q38/DESIGN_ASYNC.md sections 2.2 and 3.1)
# ---------------------------------------------------------------------------
# rl-config --stack q38 writes the complete config of verl's V1 trainer (entry vagen.q38.ppo_v1) in place of the 8B
# one: verl's generated default (this tree's verl/verl/trainer/config/_generated_ppo_trainer.yaml) + the recipe
# (configs/ppo_q38_delta.yaml: the keys that differ from that default) + the run's keys.  The run's keys are the ones
# the 8B path computes for the run (files, model, names, folders, seed, the package's sizes and coefficients, the group
# filter, a run's own overrides): they are taken from the 8B config under the same names (Q38_RUN_KEYS) or a V1 name
# (Q38_RUN_RENAMED), so the two stacks get them from the same code.  Everything before the 8B config is written (rows,
# checks, env YAMLs, system prompt check) is the 8B path unchanged; the stack only replaces the dict it writes.
STACKS = ("8b", "q38")
Q38_DELTA = Path(__file__).with_name("configs") / "ppo_q38_delta.yaml"
Q38_VERL_DEFAULT = (Path(__file__).resolve().parents[2] / "verl" / "verl" / "trainer" / "config"
                    / "_generated_ppo_trainer.yaml")
Q38_MISSING = "???"  # a delta key every run sets
Q38_RUN_KEYS = (
    # the run's files, model, names and folders
    "data.train_files", "data.val_files", "data.seed", "actor_rollout_ref.model.path", "critic.model.path",
    "critic.model.tokenizer_path", "trainer.project_name", "trainer.experiment_name", "trainer.default_local_dir",
    "trainer.validation_data_dir",
    # the package's sizes and coefficients and the run's own overrides of them (--actor-lr, --kl-loss-coef, ...)
    "data.train_batch_size", "data.val_batch_size", "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.actor.ppo_mini_batch_size", "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.actor.kl_loss_coef", "actor_rollout_ref.actor.entropy_coeff", "algorithm.adv_estimator",
    "algorithm.norm_adv_by_std_in_grpo", "trainer.critic_warmup",
    # the checkpoints (--save-freq / --checkpoint-every): every save_freq-th is a milestone the callback keeps
    "trainer.save_freq", "trainer.ckpt_keep_every", "trainer.ckpt_prune_to_hf",
    # the group filter (S0 mixed_outcome, V4 keep_by_return_std with its rows), read by the sampler (${filter})
    "filter",
)
# 8B keys whose value the V1 config keeps under another name: the 8B trainer's rollout dump is the sampler's
Q38_RUN_RENAMED = {"trainer.rollout_data_dir": "trainer.v1.sampler.sampler_kwargs.rollout_data_dir"}
# keys of verl's default the q38 config must not have: actor.grad_clip is deprecated and unread in V1 (the clip is
# actor.optim.clip_grad, transformer_impl.py:859-876); a config showing it would suggest it acts
Q38_REMOVED = ("actor_rollout_ref.actor.grad_clip",)
Q38_PACKAGE_ESTIMATORS = ("grpo",)  # phase 1 (turn_credit, turn_gae and the critic come in phases 2-3)
# phase 1 runs package p1 only: the runtime contract (vagen/q38/contract.py EXPECTED) fixes its shape, 24 candidates a
# step (data.train_batch_size, Y8:291), 8 episodes a task and 12 kept groups; p2 (48 candidates) is phase 2 and would
# be refused at start-up, so rl-config refuses it (and --candidates / --target-groups / --rollouts off p1) up front
Q38_PACKAGES = ("p1",)
Q38_PACKAGE_SHAPE = ("candidates", "target_groups", "rollouts")
# P1 decision 4 (docs/q38/VERL_PATCHES.md P1): the recipe's HF export in bf16 with A_log in fp32
# (actor.checkpoint.hf_dtype / hf_fp32_modules) was checked on readings for Qwen3.5 only (runs/q38-bf16check,
# PASS_IDENTICAL on the 27B), so rl-config keeps the two keys for these start-model types and drops them for any other
# type or an unreadable config.json (e.g. the 8B parity runs on qwen3_vl), whose export then stays fp32 as before
Q38_HF_EXPORT_TYPES = ("qwen3_5",)
Q38_HF_EXPORT_KEYS = ("hf_dtype", "hf_fp32_modules")  # under actor_rollout_ref.actor.checkpoint
Q38_HF_EXPORT = {"hf_dtype": "bfloat16", "hf_fp32_module": "A_log"}  # what a qwen3_5 config must hold
# The 27B main-line RL (q38/tmp_tests/k_mainrl, 2026-11-20).  A qwen3_5 start model's jobs run the V1 entry
# vagen.q38.ppo_v1 (vagen/models/model_types.py: the type's V10_PPO_MODULE), which reads only a V1 config, so rl-config
# writes the q38 stack for these types when --stack is not given, and refuses --stack 8b for them; every other type
# keeps the 8B default
Q38_STACK_TYPES = ("qwen3_5",)
# The card layout of a start-model type (configs/ppo_q38_layout_<type>.yaml), merged over the recipe after the run's
# keys, for these types only (start model's config.json, like q38_hf_export); any other type, or an unreadable
# config.json, gets the recipe's values as before.  A layout file may set only Q38_LAYOUT_KEYS: memory, parallelism,
# the attention implementation and timeouts, which smoke Z1 / Z2 settled on the 27B (runs/q38-smoke-27b-z1/rl/
# ppo.yaml, README section 3; P1_REPORT 1.2 and 3.1) and the delta leaves at the 8B values ("the smoke decides")
Q38_LAYOUTS = {"qwen3_5": Path(__file__).with_name("configs") / "ppo_q38_layout_qwen3_5.yaml"}
Q38_LAYOUT_KEYS = (
    "actor_rollout_ref.model.override_config.attn_implementation",
    "actor_rollout_ref.nccl_timeout",
    "actor_rollout_ref.actor.strategy",
    "actor_rollout_ref.actor.fsdp_config.strategy",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload_step",
    "actor_rollout_ref.ref.fsdp_config.strategy",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.max_num_seqs",
    "actor_rollout_ref.rollout.max_model_len",
    "actor_rollout_ref.rollout.agent.num_workers",
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
)


def _q38_flat(node: Any, prefix: str = "") -> Dict[str, Any]:
    """{dotted key: leaf value} of a nested dict (an empty dict is a leaf)."""
    if isinstance(node, dict) and node:
        out: Dict[str, Any] = {}
        for k, v in node.items():
            out.update(_q38_flat(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: node}


def _q38_get(cfg: Dict[str, Any], dotted: str) -> Tuple[bool, Any]:
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return False, None
        node = node[k]
    return True, node


def _q38_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """``base`` with ``over`` merged in, key by key (dicts recursively; any other value replaces)."""
    import copy

    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _q38_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def q38_check_options(args: argparse.Namespace) -> None:
    """The rl-config options the q38 stack refuses (after the standard filled them): the 8B trainer's features the V1
    trainer does not have, and recipes of the later phases."""
    refused = {"trial": "has no one-card trial (the 4B smoke config is written by hand)",
               "curriculum_easy_updates": "has no curriculum sampler (data.sampler; the step sampler draws the tasks)",
               "invalid_turn_nonpositive": "has no invalid-turn advantage rule (vagen/invalid_turns.py: the 8B trainer)",
               "progress_adv_weight": "has no split progress advantage (vagen/split_reward_adv.py is the 8B trainer's)",
               "logprob_tokens": "has no per-token log-probability files (vagen/logprob_tokens.py is the 8B trainer's)",
               "greedy_val": "trains without validation inside training (val_kwargs are unused)"}
    for key, why in refused.items():
        if getattr(args, key, None):
            raise SystemExit(f"--stack q38 {why}: leave out --{key.replace('_', '-')}")
    if (getattr(args, "history_mode", None) or "none") != "concat":
        raise SystemExit("--stack q38 trains one sequence per episode: it needs --history-mode concat (the standards "
                         "set it)")
    package = getattr(args, "opt_package", None)
    if not package:
        raise SystemExit("--stack q38 needs an optimisation package (--opt-package or a --standard naming one)")
    estimator = OPT_PACKAGES.get(package, {}).get("adv_estimator", "grpo")
    if estimator not in Q38_PACKAGE_ESTIMATORS:
        raise SystemExit(f"--stack q38 has no {estimator!r} yet (phase 1: {Q38_PACKAGE_ESTIMATORS}): package {package}")
    if package not in Q38_PACKAGES:
        raise SystemExit(f"--stack q38 runs the packages {Q38_PACKAGES} in phase 1, not {package!r} (its shape is not "
                         f"the contract's: vagen/q38/contract.py)")
    for key in Q38_PACKAGE_SHAPE:
        value = getattr(args, key, None)
        if value and int(value) != OPT_PACKAGES[package][key]:
            raise SystemExit(f"--stack q38 phase 1 keeps package {package}'s {key} {OPT_PACKAGES[package][key]} (the "
                             f"contract fixes it): leave out --{key.replace('_', '-')} {value}")
    if getattr(args, "test_freq", None) or not getattr(args, "no_internal_val", False):
        raise SystemExit("--stack q38 trains without validation inside training: --no-internal-val (the standards set "
                         "it) and no --test-freq")


def q38_ppo_config(ppo8: Dict[str, Any], n_train: int, type_overrides: Optional[Dict[str, Any]] = None,
                   default_path: Optional[Path] = None,
                   delta_path: Optional[Path] = None,
                   layouts: Optional[Dict[str, Path]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(the complete q38 V1 config, its report) for the run whose 8B config is ``ppo8`` (``n_train`` train rows).
    Refuses an 8B config the V1 trainer would not run as the 8B one ran."""
    import copy
    import hashlib

    import yaml

    default_path = Path(default_path or Q38_VERL_DEFAULT)
    delta_path = Path(delta_path or Q38_DELTA)
    if not default_path.is_file():
        raise SystemExit(f"--stack q38 needs verl's V1 config default {default_path} (the q38 worktree's verl)")
    base = yaml.safe_load(default_path.read_text())
    if not ((base.get("trainer") or {}).get("use_v1") and "v1" in (base.get("trainer") or {})):
        raise SystemExit(f"{default_path} is not a V1 trainer's config (no trainer.use_v1 / trainer.v1)")
    delta = yaml.safe_load(delta_path.read_text())
    if type_overrides:
        raise SystemExit(f"--stack q38 has no model-type overrides of the 8B config ({sorted(type_overrides)})")
    trainer8 = ppo8.get("trainer") or {}
    if not trainer8.get("concat_multi_turn"):
        raise SystemExit("--stack q38 needs the 8B config of the concat history mode")
    if trainer8.get("test_freq", -1) != -1 or trainer8.get("val_before_train"):
        raise SystemExit("--stack q38 trains without validation inside training (trainer.test_freq -1, "
                         "val_before_train false)")
    cfg = _q38_merge(base, delta)
    for key in Q38_RUN_KEYS:
        found, value = _q38_get(ppo8, key)
        if not found:
            raise SystemExit(f"the 8B config has no {key}, which the q38 config takes from the run")
        _set(cfg, key, copy.deepcopy(value))
    for key8, key in Q38_RUN_RENAMED.items():
        found, value = _q38_get(ppo8, key8)
        if not found or value in (None, ""):
            raise SystemExit(f"the 8B config has no {key8}, which the q38 config keeps as {key}")
        _set(cfg, key, copy.deepcopy(value))
    for key in Q38_REMOVED:
        found, _ = _q38_get(cfg, key)
        if found:
            parent, last = key.rsplit(".", 1)
            _q38_get(cfg, parent)[1].pop(last)
    hf_export = q38_hf_export(cfg)
    layout = q38_layout(cfg, Q38_LAYOUTS if layouts is None else layouts)
    # the learning-rate horizon is the recipe's (V1 fixes it at start-up, TB:792-813); the 8B template must agree
    horizon = int(cfg["trainer"]["total_training_steps"])
    if trainer8.get("total_training_steps") != horizon:
        raise SystemExit(f"the 8B config's trainer.total_training_steps {trainer8.get('total_training_steps')!r} is "
                         f"not the q38 recipe's {horizon}")
    # epochs: V1 counts len(dataset) // train_batch_size steps an epoch (TB:790) and fit ends after total_epochs of
    # them; one epoch more than the horizon needs, as the step dispatched ahead takes a batch too (DESIGN_ASYNC 3.1)
    batch = int(cfg["data"]["train_batch_size"])
    steps_per_epoch = int(n_train) // batch
    if steps_per_epoch < 1:
        raise SystemExit(f"{n_train} train rows give no step of {batch} tasks")
    cfg["trainer"]["total_epochs"] = math.ceil(horizon / steps_per_epoch) + 1
    # vagen.ga_8b, the 8B update's gradient accumulation (the KL metric's conversion, DESIGN_ASYNC 2.5.2): the rows of
    # an 8B mini-batch over its ranks and micro-batch (0.6.1 dp_actor normalises the mini-batch by the ranks)
    actor8 = ppo8["actor_rollout_ref"]["actor"]
    rows8 = int(actor8["ppo_mini_batch_size"]) * int(ppo8["actor_rollout_ref"]["rollout"]["n"])
    ranks8 = int(trainer8.get("n_gpus_per_node", 8)) * int(trainer8.get("nnodes", 1))
    micro8 = int(actor8.get("ppo_micro_batch_size_per_gpu") or 1)
    if rows8 % (ranks8 * micro8):
        raise SystemExit(f"the 8B mini-batch of {rows8} rows does not split into {ranks8} ranks x micro {micro8}")
    cfg["vagen"]["ga_8b"] = rows8 // (ranks8 * micro8)
    flat_delta = _q38_flat(delta)
    flat = _q38_flat(cfg)
    unset = sorted(k for k, v in flat_delta.items() if v == Q38_MISSING
                   and _q38_get(cfg, k)[1] in (None, Q38_MISSING, ""))
    if unset:
        raise SystemExit(f"the q38 config leaves {unset} unset")
    # P1 decision 7: every q38 config logs the host memory (vagen/q38/host_mem_log.py), which the training chain's
    # host-memory guard (plan rl guards host_mem_max_gb) reads; the recipe file sets it, a config without it is refused
    interval = ((cfg.get("vagen") or {}).get("host_mem_log") or {}).get("interval_s")
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval <= 0:
        raise SystemExit(f"the q38 config needs vagen.host_mem_log.interval_s > 0 (the training chain's host-memory "
                         f"guard reads that log), not {interval!r}")
    # the run's keys that differ from the recipe file's values (a run's own overrides), for the report
    overrides = {k: {"recipe": flat_delta[k], "run": flat[k]} for k in sorted(flat_delta)
                 if flat_delta[k] != Q38_MISSING and k in flat and flat[k] != flat_delta[k]
                 and k not in (layout.get("keys") or {})}
    report = {"verl_default": str(default_path),
              "verl_default_sha256": hashlib.sha256(default_path.read_bytes()).hexdigest(),
              "delta": str(delta_path), "delta_sha256": hashlib.sha256(delta_path.read_bytes()).hexdigest(),
              "trainer_mode": cfg["trainer"]["v1"]["trainer_mode"], "lr_horizon": horizon,
              "steps_per_epoch": steps_per_epoch, "total_epochs": cfg["trainer"]["total_epochs"],
              "ga_8b": cfg["vagen"]["ga_8b"], "removed": [k for k in Q38_REMOVED],
              "host_mem_log_interval_s": interval, "hf_export": hf_export, "layout": layout,
              "run_keys_over_recipe": overrides or None}
    return cfg, report


def q38_hf_export(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """P1 decision 4, gated by the start model's type (Q38_HF_EXPORT_TYPES): a qwen3_5 config keeps the recipe's
    actor.checkpoint.hf_dtype bfloat16 with A_log in hf_fp32_modules (refused otherwise); any other type (or a model
    folder without a readable config.json) gets both keys removed, so its config is the one without them.  Changes
    ``cfg`` in place; returns what the report records."""
    from vagen.models.model_types import of_checkpoint

    model_type = of_checkpoint(str(cfg["actor_rollout_ref"]["model"]["path"]))
    ckpt = cfg["actor_rollout_ref"]["actor"].get("checkpoint") or {}
    if model_type not in Q38_HF_EXPORT_TYPES:
        dropped = [k for k in Q38_HF_EXPORT_KEYS if k in ckpt]
        for k in dropped:
            ckpt.pop(k)
        return {"model_type": model_type, "hf_dtype": None, "dropped": dropped}
    dtype, fp32_modules = ckpt.get("hf_dtype"), ckpt.get("hf_fp32_modules")
    if dtype != Q38_HF_EXPORT["hf_dtype"] or not isinstance(fp32_modules, list) \
            or sorted(map(str, fp32_modules)) != [Q38_HF_EXPORT["hf_fp32_module"]]:
        raise SystemExit(f"a {model_type} q38 config saves its HF export in bf16 with A_log in fp32 (P1 decision 4): "
                         f"actor_rollout_ref.actor.checkpoint.hf_dtype {Q38_HF_EXPORT['hf_dtype']} and hf_fp32_modules "
                         f"[{Q38_HF_EXPORT['hf_fp32_module']}], not {dtype!r} / {fp32_modules!r}")
    return {"model_type": model_type, "hf_dtype": dtype, "hf_fp32_modules": list(fp32_modules)}


def q38_layout(cfg: Dict[str, Any], layouts: Dict[str, Path]) -> Dict[str, Any]:
    """The card layout of the start model's type (``layouts``: {model_type: yaml file}, Q38_LAYOUTS) merged into
    ``cfg`` in place; a file key outside Q38_LAYOUT_KEYS is refused.  Returns what the report records: the type, the
    file and its sha256, and {key: {"recipe": the value before, "layout": the value set}} (file None: no layout)."""
    import hashlib

    import yaml

    from vagen.models.model_types import of_checkpoint

    model_type = of_checkpoint(str(cfg["actor_rollout_ref"]["model"]["path"]))
    path = layouts.get(model_type) if model_type else None
    if path is None:
        return {"model_type": model_type, "file": None}
    path = Path(path)
    flat = _q38_flat(yaml.safe_load(path.read_text()) or {})
    outside = sorted(k for k in flat if k not in Q38_LAYOUT_KEYS)
    if outside:
        raise SystemExit(f"the {model_type} layout {path} sets {outside}, which are not layout keys "
                         f"(training_prep.Q38_LAYOUT_KEYS)")
    keys = {}
    for key in sorted(flat):
        found, before = _q38_get(cfg, key)
        keys[key] = {"recipe": before if found else "<absent>", "layout": flat[key]}
        _set(cfg, key, flat[key])
    return {"model_type": model_type, "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "keys": keys}


def cmd_rl_config(args: argparse.Namespace) -> None:
    import copy

    import yaml

    from .admission import load_policy

    standard = getattr(args, "standard", None)
    radius = getattr(args, "turn_footprint_radius", None)
    if radius is not None and not (math.isfinite(float(radius)) and float(radius) > 0):
        # experiment 3A (review fixes): refused before apply_standard, which takes an explicit 0 for no option given
        # (0.0 == False) and would fill an _fp standard's 0.25 in its place
        raise SystemExit(f"--turn-footprint-radius must be a positive finite number, not {radius!r}")
    if standard:  # round 12: a named standard configuration fixes the recipe's options
        apply_standard(args, standard)
    if not getattr(args, "ppo_template", None):
        raise SystemExit("rl-config needs --ppo-template (or a --standard that names one)")
    model_type = getattr(args, "model_type", None)
    stack = getattr(args, "stack", None) or ("q38" if model_type in Q38_STACK_TYPES else "8b")
    if stack not in STACKS:
        raise SystemExit(f"--stack must be one of {STACKS}, not {stack!r}")
    if model_type in Q38_STACK_TYPES and stack != "q38":  # its jobs run vagen.q38.ppo_v1, which reads a V1 config
        raise SystemExit(f"a {model_type} start model trains with the V1 entry (vagen/models/model_types.py): "
                         f"--stack q38, not {stack!r}")
    if stack == "q38":  # the Qwen3.8-27B rerun: refused options before anything is written
        q38_check_options(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(Path(args.rows))
    # Which tasks RL trains on (the train split of rows.jsonl).  Round 10: the formal tasks only (the first run with
    # reserve rows in rows.jsonl took them too).  Round 11 (the user's decision): the tasks RL trains on must be ones
    # SFT never trained on; SFT takes the formal tasks, so RL takes the reserve tasks (--task-pool reserve, rows
    # from sft-frames --include-reserve).  Validation rows stay formal.
    pool = getattr(args, "task_pool", None) or "formal"
    if pool not in TASK_POOLS:
        raise SystemExit(f"--task-pool must be one of {TASK_POOLS}")
    parts = {"train": [r for r in rows if r.get("split") == "train" and bool(r.get("reserve")) == (pool == "reserve")],
             "val": [r for r in rows if r.get("split") == "val" and not r.get("reserve")]}
    reserve_skipped = sum(1 for r in rows if r.get("reserve") and r.get("split") in ("train", "val")) \
        if pool == "formal" else 0
    # the checks of the rule: no train task in the SFT conversations, none flagged by the depth check
    sft_ids = set()
    for path in getattr(args, "sft_trajectories", None) or []:
        sft_ids |= {r["source_task_id"] for r in _jsonl(Path(path))}
    overlap = sorted(r["task_id"] for r in parts["train"] if r["task_id"] in sft_ids)
    if overlap:
        raise SystemExit(f"{len(overlap)} RL train tasks are in the SFT conversations (e.g. {overlap[:3]}): RL must "
                         "train on tasks SFT never trained on")
    flagged = set(json.loads(Path(args.exclude).read_text())["flagged_task_ids"]) if getattr(args, "exclude", None) \
        else set()
    flagged_in_train = sorted(r["task_id"] for r in parts["train"] if r["task_id"] in flagged)
    if flagged_in_train:
        raise SystemExit(f"{len(flagged_in_train)} RL train tasks are flagged by the depth check (e.g. "
                         f"{flagged_in_train[:3]})")
    val_tasks = getattr(args, "val_tasks", None)
    if val_tasks:  # round 10: the fixed validation set (evaluation.py select --split val), published rows
        parts["val"] = [dict(r, split="val") for r in _jsonl(Path(val_tasks))]
    elif args.val_limit:  # evenly spread over the val rows (they come sorted by scene)
        step = max(1, len(parts["val"]) // args.val_limit)
        parts["val"] = parts["val"][::step][: args.val_limit]
    if not parts["train"] or not parts["val"]:
        raise SystemExit("rl-config needs train and val rows (training_prep sft-frames rows.jsonl)")
    template = yaml.safe_load(Path(args.env_template).read_text())
    history = history_overrides(getattr(args, "history_mode", None) or "none",
                                int(template["envs"][0].get("max_turns", 12)))
    history["env"].update(reward_overrides(getattr(args, "reward_preset", None) or "S0"))
    # round 12 freeze (the user's 3.1.1): the format line shows placeholder or real action names.  Default: the SFT
    # export's (admission policy); the start model's SFT conversations, when given, must show the same text, and
    # then decide it for the system prompt check too.
    from vagen.envs.active_spatial.prompt import FORMAT_EXAMPLES

    from .release import format_example as policy_format_example

    policy = load_policy()
    fmt = getattr(args, "format_example", None) or policy_format_example(policy)
    if fmt not in FORMAT_EXAMPLES:
        raise SystemExit(f"--format-example must be one of {FORMAT_EXAMPLES}, not {fmt!r}")
    tcfg = template["envs"][0]["config"]
    sft_formats = sft_format_examples([Path(p) for p in getattr(args, "sft_trajectories", None) or []],
                                      int(tcfg.get("max_actions_per_step", 5)), str(tcfg.get("action_sep", "|")))
    if sft_formats and sft_formats != [fmt]:
        raise SystemExit(f"the SFT conversations show the {sft_formats} format example, the RL config would show "
                         f"{fmt!r}: give --format-example {sft_formats[0]} (or the conversations the start model "
                         "trained on)")
    if fmt != str(tcfg.get("prompt_format_example", "placeholder")):
        history["env"]["prompt_format_example"] = fmt
    if getattr(args, "lenient_parse", False):  # round 12 freeze (the user's 3.1.2): the env's lenient reading
        history["env"]["lenient_parse"] = True
    if getattr(args, "invalid_penalty", None) is not None:  # round 12 freeze (3.1.3): the unreadable answer's penalty
        history["env"]["invalid_penalty_override"] = float(args.invalid_penalty)
    footprint = getattr(args, "turn_footprint_radius", None)
    if footprint is not None:  # experiment 3A (the user's 甲 of 2026-11-06): turns blocked by the robot-base footprint
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

        if "turn_footprint_radius" not in ActiveSpatialEnvConfig.__dataclass_fields__:
            raise SystemExit("this checkout's env has no turn_footprint_radius: the RL env would drop it")
        if not (math.isfinite(float(footprint)) and float(footprint) > 0):  # e.g. a standard's own value
            raise SystemExit(f"--turn-footprint-radius must be a positive finite number, not {footprint!r}")
        if tcfg.get("wall_model") != "thick" or not tcfg.get("enable_collision_detection", True):
            raise SystemExit("--turn-footprint-radius needs an env template with collision detection and wall_model "
                             f"thick ({args.env_template})")
        history["env"]["turn_footprint_radius"] = float(footprint)
    history["env"].update(design_variant_overrides(args, history["env"]))
    # the privileged critic (package p1tgae_fc, 2026-11-13): the train env gives each observation's ground truth
    # (env critic_features) from the train tasks' distance tables; every train task must have one
    feature_env: Dict[str, Any] = {}
    feature_tables = getattr(args, "critic_feature_tables", None)
    features_package = OPT_PACKAGES.get(getattr(args, "opt_package", None) or "", {}).get("critic_source") == "features"
    if features_package or feature_tables:
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

        if not features_package:
            raise SystemExit("--critic-feature-tables is for the privileged critic: give --opt-package p1tgae_fc")
        if not feature_tables or not Path(feature_tables).is_dir():
            raise SystemExit(f"package {args.opt_package} needs --critic-feature-tables DIR (the train tasks' "
                             f"distance tables), not {feature_tables!r}")
        if "critic_features" not in ActiveSpatialEnvConfig.__dataclass_fields__:
            raise SystemExit("this checkout's env has no critic_features: the RL env would drop it")
        missing = sorted(r["task_id"] for r in parts["train"]
                         if not (Path(feature_tables) / f"{r['task_id']}.npz").is_file())
        if missing:
            raise SystemExit(f"{len(missing)} of {len(parts['train'])} train tasks have no distance table in "
                             f"{feature_tables} (e.g. {missing[:3]}): the privileged critic needs one for each")
        feature_env = {"critic_features": True, "critic_feature_tables_dir": str(Path(feature_tables).resolve())}
    env_paths = {}
    for split, selected in parts.items():
        _write_jsonl(out / f"{split}.jsonl", selected)
        env = copy.deepcopy(template)
        entry = env["envs"][0]
        entry.update(n_envs=len(selected), seed=[0, len(selected)], seed_list=list(range(len(selected))))
        entry["config"].update(jsonl_path=str((out / f"{split}.jsonl").resolve()), gs_root=str(args.gs_root),
                               train_size=len(selected), **history["env"],
                               **(feature_env if split == "train" else {}))
        env_paths[split] = out / f"{split}_env.yaml"
        env_paths[split].write_text(yaml.safe_dump(env, sort_keys=False))
    mismatches = prompt_mismatches(dict(tcfg, prompt_format_example=fmt), [r["task_type"] for r in rows],
                                   with_format_example(policy, fmt) if sft_formats else policy)
    if mismatches:
        raise SystemExit(f"RL and SFT system prompts differ for {mismatches}")
    ppo = yaml.safe_load(Path(args.ppo_template).read_text())
    run_root = Path(args.run_root).resolve()
    model = str(Path(args.model))  # no trailing "/": verl's copy_to_local refuses one
    for key, value in {
        "data.train_files": str(env_paths["train"].resolve()), "data.val_files": str(env_paths["val"].resolve()),
        "actor_rollout_ref.model.path": model, "critic.model.path": model, "critic.model.tokenizer_path": model,
        "trainer.experiment_name": args.name, "trainer.project_name": "vagen_interiorgs_v10",
        "trainer.default_local_dir": str(run_root / "checkpoints"),
        "trainer.rollout_data_dir": str(run_root / "rollout_data"),
        "trainer.validation_data_dir": str(run_root / "validation"),
    }.items():
        _set(ppo, key, value)
    for key, value in FORMAL_PPO.items():
        _set(ppo, key, value)
    if args.trial:
        for key, value in TRIAL_PPO.items():
            _set(ppo, key, value)
    for key, value in history["ppo"].items():
        _set(ppo, key, value)
    # round 10 options: validation on the fixed set without sampling (one greedy trajectory per task, in one
    # batch), its frequency, and the critic warm-up
    if getattr(args, "greedy_val", False):
        for key, value in {"actor_rollout_ref.rollout.val_kwargs.n": 1,
                           "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
                           "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0}.items():
            _set(ppo, key, value)
    if val_tasks:
        _set(ppo, "data.val_batch_size", len(parts["val"]))
    if getattr(args, "test_freq", None):
        _set(ppo, "trainer.test_freq", int(args.test_freq))
    if getattr(args, "critic_warmup", None) is not None:
        _set(ppo, "trainer.critic_warmup", int(args.critic_warmup))
    # round 11: more tasks per update, checkpoints every N updates (all kept, the older ones cut to their weights),
    # no validation inside training (models are picked with the single-card evaluation script)
    if getattr(args, "train_batch_size", None):
        _set(ppo, "data.train_batch_size", int(args.train_batch_size))
    if getattr(args, "ppo_mini_batch_size", None):
        _set(ppo, "actor_rollout_ref.actor.ppo_mini_batch_size", int(args.ppo_mini_batch_size))
    if getattr(args, "save_freq", None):
        for key, value in {"trainer.save_freq": int(args.save_freq), "trainer.ckpt_keep_every": int(args.save_freq),
                           "trainer.ckpt_prune_to_hf": True}.items():
            _set(ppo, key, value)
    if getattr(args, "no_internal_val", False):
        _set(ppo, "trainer.val_before_train", False)
        _set(ppo, "trainer.test_freq", -1)
    if getattr(args, "seed", None) is not None:  # round 12: the run's seed, the order of the tasks (data.seed); this
        _set(ppo, "data.seed", int(args.seed))      # verl's RolloutConfig has no seed (vLLM samples with seed 0)
    package = None
    if getattr(args, "opt_package", None):  # round 11: applied last, over the sizes and the warm-up above
        if not ppo["trainer"].get("concat_multi_turn"):
            raise SystemExit("the optimisation package needs the concat history mode (--history-mode concat)")
        package = package_overrides(args.opt_package, getattr(args, "candidates", None),
                                    getattr(args, "target_groups", None), getattr(args, "rollouts", None))
        # the PPO supplement (2026-11-08): a GAE package's warm-up is part of its recipe; an explicit other
        # --critic-warmup would be overridden without a word, so it is refused (GRPO packages keep the override)
        warmup = getattr(args, "critic_warmup", None)
        if (package["algorithm.adv_estimator"] in CRITIC_ESTIMATORS and warmup is not None
                and int(warmup) != package["trainer.critic_warmup"]):
            raise SystemExit(f"--opt-package {args.opt_package} sets trainer.critic_warmup to "
                             f"{package['trainer.critic_warmup']} (the value network alone for "
                             f"{package['trainer.critic_warmup'] - 1} updates); the command gives --critic-warmup "
                             f"{warmup}: leave it out")
        for key, value in package.items():
            _set(ppo, key, value)
        if package.get("trainer.critic_source") == "features":  # the privileged critic: the env's turn limit
            _set(ppo, "trainer.feature_critic", dict(package["trainer.feature_critic"],
                                                    max_turns=int(template["envs"][0].get("max_turns", 12))))
    if getattr(args, "invalid_turn_nonpositive", False):  # round 12 freeze (the user's 3.2, vagen/invalid_turns.py)
        _set(ppo, "algorithm.invalid_turn_nonpositive_adv", True)
    if getattr(args, "rollout_log_probs", False):
        # the model line (2026-10-24): vLLM returns each sampled token's log-probability, so the trainer compares it
        # with its own (vagen/ray_trainer.py: training/rollout_probs_diff_*, and with VAGEN_D0_3_SMOKE_DIR a file per
        # update).  The update still uses the trainer's recomputed values: the rollout correction stays off
        if ((ppo.get("algorithm") or {}).get("rollout_correction") or {}).get("rollout_is") is not None:
            raise SystemExit("--rollout-log-probs is an audit: the template must not turn on the rollout correction")
        _set(ppo, "actor_rollout_ref.rollout.calculate_log_probs", True)
    if getattr(args, "logprob_tokens", False):
        # the model line (2026-10-26, the user's decision 1): with the audit, every response token's two
        # log-probabilities, its turn and its distance from the last image token, one file per update
        # (vagen/logprob_tokens.py, into VAGEN_D0_3_SMOKE_DIR/tokens); the update is unchanged
        if not getattr(args, "rollout_log_probs", False):
            raise SystemExit("--logprob-tokens needs --rollout-log-probs (the rollout's log-probabilities)")
        _set(ppo, "trainer.logprob_tokens", True)
    if getattr(args, "progress_adv_weight", None) is not None:  # round 12 freeze, V8 (vagen/split_reward_adv.py)
        if not history["env"].get("enable_potential_shaping_reward"):
            raise SystemExit("--progress-adv-weight splits the dense reward's progress part: give a dense --reward-preset")
        _set(ppo, "algorithm.progress_adv_weight", float(args.progress_adv_weight))
    turn_credit = None
    if getattr(args, "turn_credit", None):  # reward line (2026-11-01): the per-turn credit (vagen/turn_credit.py)
        from vagen.turn_credit import ESTIMATOR, settings as turn_credit_settings

        turn_credit = json.loads(args.turn_credit) if isinstance(args.turn_credit, str) else dict(args.turn_credit)
        try:
            tc_checked = turn_credit_settings({"turn_credit": turn_credit})
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"--turn-credit: {exc}")
        if history["env"].get("enable_potential_shaping_reward"):
            raise SystemExit("--turn-credit adds the progress term itself: give the S0 reward (no dense --reward-preset)")
        if tc_checked["phi_version"] == "grouped_gated":  # phase 3, experiment J (丙): V4's progress score as phi
            if history["env"].get("progress_version") != "grouped_gated" or history["env"].get("distance_tables_dir"):
                raise SystemExit("--turn-credit phi_version grouped_gated reads V4's progress score: "
                                 "--progress-version grouped_gated, no --distance-tables")
            if history["env"].get("progress_weights") or history["env"].get("milestone_rewards"):
                raise SystemExit("--turn-credit phi_version grouped_gated is V4's score: no progress weights or milestones")
            if tc_checked["stall_penalty"]:  # the stall test needs the table distance, unknown without tables
                raise SystemExit("--turn-credit phi_version grouped_gated has no table distance: no stall_penalty")
        elif history["env"].get("progress_version") != "optimal" or not history["env"].get("distance_tables_dir"):
            raise SystemExit("--turn-credit reads the table's progress score: --progress-version optimal --distance-tables DIR")
        if tc_checked["phi_version"] == "success_table":  # phase 3, experiment K (戊): phi = Phi(d, b) from the table
            from vagen.turn_credit import load_success_table

            try:
                tab = load_success_table(tc_checked["phi_table"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SystemExit(f"--turn-credit phi_table: {exc}")
            merged_env = {**tcfg, **history["env"]}
            if merged_env.get("max_episode_steps") is None:  # the env's own default is not 12 (code review)
                raise SystemExit("--turn-credit phi_table needs the env template's max_episode_steps")
            budget = int(merged_env["max_episode_steps"])
            if tab["budget"] != budget:
                raise SystemExit(f"--turn-credit phi_table is for a budget of {tab['budget']} actions, the env has {budget}")
            src = tab["source"]  # the table's d must be this env's d: the same tables and neighbour lookup
            if src.get("distance_tables_dir") and \
                    Path(src["distance_tables_dir"]).resolve() != Path(history["env"]["distance_tables_dir"]).resolve():
                raise SystemExit(f"--turn-credit phi_table was fitted on {src['distance_tables_dir']}, the env reads "
                                 f"{history['env']['distance_tables_dir']}")
            if "optimal_progress_near" in src and bool(src["optimal_progress_near"]) != \
                    bool(history["env"].get("optimal_progress_near", False)):
                raise SystemExit("--turn-credit phi_table's neighbour lookup (--table-near) differs from the env's")
            if int(merged_env.get("max_consecutive_collisions") or 0) > 0:  # that turn would keep a stale d
                raise SystemExit("--turn-credit phi_table: a collision-terminated turn keeps the previous metric; "
                                 "needs max_consecutive_collisions 0")
            turn_credit["phi_table"] = str(Path(tc_checked["phi_table"]).resolve())
        if not ppo["trainer"].get("concat_multi_turn"):
            raise SystemExit("--turn-credit needs the concat history mode (one sequence an episode, the turns' spans)")
        _set(ppo, "algorithm.adv_estimator", ESTIMATOR)
        _set(ppo, "algorithm.turn_credit", turn_credit)
    if getattr(args, "entropy_coeff", None) is not None:  # round 12 batch: the entropy bonus instead of the package's
        _set(ppo, "actor_rollout_ref.actor.entropy_coeff", float(args.entropy_coeff))
    # round 12 freeze, the weak-start stability runs (the user's 11th supplement, W3 / W4 and the checkpoint interval):
    # over the package's and the standard's values, recorded in the report
    if getattr(args, "kl_loss_coef", None) is not None:
        _set(ppo, "actor_rollout_ref.actor.kl_loss_coef", float(args.kl_loss_coef))
    if getattr(args, "actor_lr", None) is not None:
        _set(ppo, "actor_rollout_ref.actor.optim.lr", float(args.actor_lr))
    if getattr(args, "checkpoint_every", None):
        for key, value in {"trainer.save_freq": int(args.checkpoint_every),
                           "trainer.ckpt_keep_every": int(args.checkpoint_every), "trainer.ckpt_prune_to_hf": True}.items():
            _set(ppo, key, value)
    group_filter = getattr(args, "group_filter", None)
    filter_kwargs = None
    if group_filter not in (None, *GROUP_FILTERS):
        raise SystemExit(f"--group-filter must be one of {GROUP_FILTERS}, not {group_filter!r}")
    if group_filter == "keep_by_return_std":  # round 12, the dense-reward first batch (D-keep)
        if not package:
            raise SystemExit("--group-filter keep_by_return_std fills the package's batch: it needs --opt-package")
        if getattr(args, "keep_std_threshold", None) is None:
            raise SystemExit("--group-filter keep_by_return_std needs --keep-std-threshold")
        filter_kwargs = dict(package["filter.filter_kwargs"], std_threshold=float(args.keep_std_threshold),
                             std_floor=float(getattr(args, "keep_std_floor", None) or 0.0),
                             rows=str((out / "train.jsonl").resolve()))  # the filled tasks' d* in the metrics
        _set(ppo, "filter.name", group_filter)
        _set(ppo, "filter.filter_kwargs", filter_kwargs)
    if group_filter == "nominal":  # reward line (2026-11-01): the nominal batch, every outcome kept
        if not package:
            raise SystemExit("--group-filter nominal keeps the package's nominal batch: it needs --opt-package")
        filter_kwargs = dict(package["filter.filter_kwargs"])
        _set(ppo, "filter.name", group_filter)
        _set(ppo, "filter.filter_kwargs", filter_kwargs)
    curriculum = None
    if getattr(args, "curriculum_easy_updates", None):  # round 12 batch: easy tasks first (vagen/curriculum_sampler)
        max_d = int(getattr(args, "curriculum_max_dstar", None) or 3)
        easy = sum(1 for r in parts["train"] if int(r["difficulty"]["d_star"]) <= max_d)
        need = int(args.curriculum_easy_updates) * int(ppo["data"]["train_batch_size"])
        if need > easy:
            raise SystemExit(f"{args.curriculum_easy_updates} easy updates need {need} tasks with d* <= {max_d}, "
                             f"the train rows have {easy}")
        curriculum = {"class_path": "pkg://vagen.curriculum_sampler", "class_name": "DifficultyCurriculumSampler",
                      "easy_updates": int(args.curriculum_easy_updates), "max_dstar": max_d}
        _set(ppo, "data.sampler", curriculum)
        _set(ppo, "data.dataloader_num_workers", 0)  # verl's rule for a custom sampler
        curriculum = dict(curriculum, easy_tasks=easy, easy_draws=need)
    type_overrides = None
    if getattr(args, "model_type", None):  # the model line (2026-10-24): the type's own overrides, applied last
        from vagen.models.model_types import check, rl_overrides

        try:
            check(args.model_type, model)
        except ValueError as exc:
            raise SystemExit(str(exc))
        type_overrides = rl_overrides(args.model_type, int(template["envs"][0].get("max_turns", 12)),
                                      int(ppo["data"]["max_prompt_length"]))
        if type_overrides and not ppo["trainer"].get("concat_multi_turn"):
            raise SystemExit(f"the {args.model_type} overrides are for the concat history mode (--history-mode concat)")
        if "trainer.agent_loop_manager" in (type_overrides or {}) and ppo["algorithm"].get("adv_estimator") == "turn_gae":
            # the per-turn PPO's turn spans are the agent loop's (compact) response positions; this type's manager
            # expands the image tokens afterwards and does not move them (vagen/agent_loop/cambrian_concat.py)
            raise SystemExit(f"turn_gae is not supported with the {args.model_type} agent-loop manager")
        for key, value in type_overrides.items():
            _set(ppo, key, value)
    q38_report = None
    if stack == "q38":  # the Qwen3.8-27B rerun: verl's V1 config from this 8B one (q38_ppo_config)
        ppo, q38_report = q38_ppo_config(ppo, len(parts["train"]), type_overrides)
    (out / "ppo.yaml").write_text(yaml.safe_dump(ppo, sort_keys=False))
    report = {"train": len(parts["train"]), "val": len(parts["val"]), "trial": bool(args.trial),
              "model": model, "system_prompt_matches_sft": True,
              "history_mode": getattr(args, "history_mode", None) or "none", "val_tasks": val_tasks,
              "greedy_val": bool(getattr(args, "greedy_val", False)),
              "test_freq": ppo["trainer"].get("test_freq"), "critic_warmup": ppo["trainer"].get("critic_warmup"),
              "reward_preset": getattr(args, "reward_preset", None) or "S0",
              "env_reward_overrides": reward_overrides(getattr(args, "reward_preset", None) or "S0"),
              "design_variant": design_variant_overrides(args, history["env"]) or None,
              "progress_adv_weight": getattr(args, "progress_adv_weight", None),
              "train_batch_size": (ppo.get("data") or {}).get("train_batch_size"),
              "save_freq": ppo["trainer"].get("save_freq"),
              "reserve_rows_skipped": reserve_skipped,
              "task_pool": pool,
              "opt_package": getattr(args, "opt_package", None), "opt_package_overrides": package,
              "standard": standard or None, "ppo_template": str(args.ppo_template), "curriculum": curriculum,
              "format_example": fmt, "format_example_sft": sft_formats or None,
              "lenient_parse": bool(getattr(args, "lenient_parse", False)),
              "invalid_turn_nonpositive": bool(getattr(args, "invalid_turn_nonpositive", False)),
              "rollout_log_probs": bool(getattr(args, "rollout_log_probs", False)),
              "logprob_tokens": bool(getattr(args, "logprob_tokens", False)),
              "model_type": getattr(args, "model_type", None), "model_type_overrides": type_overrides,
              "invalid_penalty": (float(args.invalid_penalty) if getattr(args, "invalid_penalty", None) is not None
                                  else float(tcfg.get("invalid_format_penalty", -0.1))),
              "entropy_coeff": ((ppo.get("actor_rollout_ref") or {}).get("actor") or {}).get("entropy_coeff"),
              "entropy_coeff_override": getattr(args, "entropy_coeff", None),
              "kl_loss_coef": ((ppo.get("actor_rollout_ref") or {}).get("actor") or {}).get("kl_loss_coef"),
              "actor_lr": (((ppo.get("actor_rollout_ref") or {}).get("actor") or {}).get("optim") or {}).get("lr"),
              "weak_start_overrides": {k: getattr(args, k, None) for k in ("kl_loss_coef", "actor_lr", "checkpoint_every")
                                       if getattr(args, k, None) is not None} or None,
              "group_filter": ((ppo.get("filter") or {}).get("name") if (ppo.get("filter") or {}).get("enable") else None),
              "group_filter_kwargs": filter_kwargs,
              # the task order (data.seed); vLLM's sampling seed is 0 in this verl (RolloutConfig has no seed field)
              "seed": {"data": (ppo.get("data") or {}).get("seed"), "rollout": 0},
              "sft_tasks_checked": len(sft_ids), "sft_overlap": len(overlap),
              "depth_flagged_checked": len(flagged), "depth_flagged_in_train": len(flagged_in_train),
              "train_scenes": len({r["scene_id"] for r in parts["train"]}),
              "by_family_train": dict(Counter(r["task_type"] for r in parts["train"]))}
    if footprint is not None:  # experiment 3A: only when set, so every other report stays as it was
        report["turn_footprint_radius"] = float(footprint)
    if turn_credit is not None:  # reward line (2026-11-01); only when set (RW writes both keys always), so every
        report["turn_credit"] = turn_credit  # other report stays as it was
        report["adv_estimator"] = (ppo.get("algorithm") or {}).get("adv_estimator")
    if q38_report is not None:  # the q38 stack: only when set, so every 8B report stays as it was
        report["stack"] = "q38"
        report["q38"] = q38_report
    if feature_env:  # the privileged critic (2026-11-13): only when set, so every other report stays as it was
        report["critic_source"] = "features"
        report["critic_feature_tables"] = feature_env["critic_feature_tables_dir"]
        report["critic_feature_tables_checked"] = len(parts["train"])
        # provenance of the tables (another line's folder, read live): the hashes of its build records when present
        import hashlib

        report["critic_feature_tables_records"] = {
            name: hashlib.sha256((Path(feature_tables) / name).read_bytes()).hexdigest()
            for name in ("DONE.json", "final_ids.txt", "summary.jsonl") if (Path(feature_tables) / name).is_file()}
    (out / "rl_config_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# Round 12 batch (the user's data-scale curve): nested subsets of the train scenes that have formal SFT tasks
# (a fixed seed shuffles the sorted scene list; a subset of n scenes is the first n), each with all formal tasks of
# its scenes for SFT and the reserve tasks of its scenes for RL; and an arm with the task count of one subset spread
# over every scene (scenes and tasks separated).
SUBSET_SEED = 20261020


def scene_order(scenes: Sequence[str], seed: int = SUBSET_SEED) -> List[str]:
    import random

    order = sorted(set(scenes))
    random.Random(seed).shuffle(order)
    return order


def spread_quota(capacity: Dict[str, int], total: int, seed: int = SUBSET_SEED) -> Dict[str, int]:
    """``total`` tasks over the scenes as evenly as possible (each scene at most its ``capacity``): every scene gets
    the floor share, the remainder goes one by one to randomly chosen scenes with room left."""
    import random

    scenes = sorted(capacity)
    if total > sum(capacity.values()):
        raise ValueError(f"{total} tasks wanted, only {sum(capacity.values())} available")
    quota = {s: 0 for s in scenes}
    rng = random.Random(seed)
    left = total
    while left > 0:
        room = [s for s in scenes if quota[s] < capacity[s]]
        share = max(1, left // len(room))
        if share * len(room) <= left:
            for s in room:
                take = min(share, capacity[s] - quota[s])
                quota[s] += take
                left -= take
        else:
            for s in rng.sample(room, left):
                quota[s] += 1
            left = 0
    return quota


def build_subsets(records: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], sizes: Sequence[int],
                  equal_tasks_of: Optional[int] = None, flagged: Sequence[str] = (),
                  seed: int = SUBSET_SEED) -> Dict[str, Any]:
    """{name: {"scenes", "sft" (train records of the scenes + every val record), "rows" (reserve train rows of the
    scenes, flagged tasks left out, + every formal val row)}} for subsets s<n> of the train scenes with formal SFT
    records (nested, first n of scene_order), plus w<all> with as many train records as subset ``equal_tasks_of``,
    spread over all scenes (spread_quota; records drawn at random within a scene)."""
    import random

    train = [r for r in records if r.get("split") == "train"]
    val = sorted((r for r in records if r.get("split") == "val"), key=lambda r: r["id"])
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for r in sorted(train, key=lambda r: r["id"]):
        by_scene.setdefault(str(r["scene_id"]), []).append(r)
    order = scene_order(list(by_scene), seed)
    flagged = set(flagged)
    rows = sorted(rows, key=lambda r: r["task_id"])  # the outputs do not depend on the input order
    reserve = [r for r in rows if r.get("split") == "train" and r.get("reserve") and r["task_id"] not in flagged]
    val_rows = [r for r in rows if r.get("split") == "val" and not r.get("reserve")]
    out: Dict[str, Any] = {"seed": seed, "scene_order": order, "subsets": {}}

    def subset(name: str, scenes: Sequence[str], sft_train: List[Dict[str, Any]]) -> None:
        keep = set(scenes)
        out["subsets"][name] = {"scenes": sorted(keep), "sft": sft_train + val,
                                "rows": [r for r in reserve if str(r["scene_id"]) in keep] + val_rows,
                                "train_records": len(sft_train)}

    for n in sorted(set(int(x) for x in sizes)):
        if n > len(order):
            raise ValueError(f"subset of {n} scenes wanted, only {len(order)} train scenes have formal tasks")
        scenes = order[:n]
        subset(f"s{n:03d}", scenes, [r for s in scenes for r in by_scene[s]])
    if equal_tasks_of:
        total = out["subsets"][f"s{int(equal_tasks_of):03d}"]["train_records"]
        quota = spread_quota({s: len(v) for s, v in by_scene.items()}, total, seed)
        rng = random.Random(seed + 1)
        picked = [r for s in sorted(by_scene) for r in rng.sample(by_scene[s], quota[s])]
        subset(f"w{len(order)}", list(by_scene), picked)
        out["subsets"][f"w{len(order)}"]["per_scene"] = {"min": min(quota.values()), "max": max(quota.values())}
    return out


# Round 12 freeze (the user's 3.1.4 of 2026-10-22): the format SFT of the original 7B, which RL from the original model
# needs before it (its answers collapse into made-up action names, cluster_runs/FORMAT_DRIFT_R12.md): a few hundred
# short conversations (1-2 assistant turns) from the existing SFT conversations, shown with the format example of the
# future data (prompt.FORMAT_EXAMPLES, release.format_example), trained 1-2 epochs with a small learning rate.
FORMAT_SFT_SEED = 20261022


def format_sft_records(records: Sequence[Dict[str, Any]], policy: Dict[str, Any], count: int, val_count: int,
                       max_turns: int = 2, format_example: str = "action_names",
                       seed: int = FORMAT_SFT_SEED) -> Dict[str, Any]:
    """{"records": the chosen conversations rewritten to ``format_example``, "report"}.  A conversation qualifies
    with 1..``max_turns`` assistant turns; ``count`` train and ``val_count`` val ones are drawn at random (seed),
    spread over the task families in proportion.  The rewrite replaces the system prompt and each user turn's format
    text exactly (the texts the SFT export wrote with the policy's own format example); a conversation whose texts do
    not match is left out and counted."""
    import copy
    import random

    from .release import _format_block, format_example as policy_format_example, protocol_system_prompt

    old_example = policy_format_example(policy)
    new_policy = copy.deepcopy(policy)
    new_policy["sft"]["export"]["format_example"] = format_example
    old_blocks = {first: _format_block(policy, first) for first in (True, False)}
    new_blocks = {first: _format_block(new_policy, first) for first in (True, False)}
    skipped = Counter()

    def rewrite(rec):
        conv = copy.deepcopy(rec["conversations"])
        t = str(rec.get("task_type") or "")
        if conv[0]["role"] != "system" or conv[0]["content"] != protocol_system_prompt(t, policy):
            skipped["system_prompt_differs"] += 1
            return None
        conv[0]["content"] = protocol_system_prompt(t, new_policy)
        users = [m for m in conv if m["role"] == "user"]
        for k, m in enumerate(users):
            first = k == 0
            if not m["content"].endswith("\n" + old_blocks[first]) or m["content"].count(old_blocks[first]) != 1:
                skipped["format_text_differs"] += 1
                return None
            m["content"] = m["content"][: -len(old_blocks[first])] + new_blocks[first]
        return dict(rec, conversations=conv, format_example=format_example)

    def turns(rec):
        return sum(1 for m in rec.get("conversations") or [] if m.get("role") == "assistant")

    eligible = {"train": [], "val": []}
    for rec in records:
        if rec.get("split") in eligible and 1 <= turns(rec) <= max_turns:
            eligible[rec["split"]].append(rec)
    rng = random.Random(seed)
    chosen, by_family = [], {}
    for split, want in (("train", count), ("val", val_count)):
        pool = sorted(eligible[split], key=lambda r: str(r.get("id")))
        rng.shuffle(pool)
        families = sorted({str(r.get("task_type")) for r in pool})
        quota = {f: round(want * sum(1 for r in pool if str(r.get("task_type")) == f) / max(1, len(pool)))
                 for f in families}
        picked = []
        for f in families:  # in proportion to the family's share of the eligible conversations
            for rec in [r for r in pool if str(r.get("task_type")) == f]:
                if sum(1 for p in picked if str(p.get("task_type")) == f) >= quota[f]:
                    break
                new = rewrite(rec)
                if new is not None:
                    picked.append(new)
        for rec in pool:  # rounding: fill up to the number asked for
            if len(picked) >= want:
                break
            if all(p.get("id") != rec.get("id") for p in picked):
                new = rewrite(rec)
                if new is not None:
                    picked.append(new)
        picked = picked[:want]
        chosen += picked
        by_family[split] = dict(Counter(str(r.get("task_type")) for r in picked))
    report = {"source_records": len(records), "eligible": {k: len(v) for k, v in eligible.items()},
              "max_assistant_turns": max_turns, "train": by_family.get("train") and sum(by_family["train"].values()),
              "val": by_family.get("val") and sum(by_family["val"].values()), "by_family": by_family,
              "assistant_turns": dict(Counter(turns(r) for r in chosen)), "format_example": format_example,
              "source_format_example": old_example, "skipped": dict(skipped), "seed": seed}
    return {"records": chosen, "report": report}


def cmd_format_sft(args: argparse.Namespace) -> None:
    """Round 12 freeze: the format SFT's conversations (sft_trajectories_format.jsonl) in --out, with the source SFT
    folder's frames, split manifest and task registry linked in (a folder the chain uses with "sft_records": false)."""
    import os

    from .admission import load_policy

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # the source conversations keep the format text they were exported with (rounds 10-12: the placeholder, which the
    # policy no longer names): the policy with that example is the one their texts are checked against
    policy = load_policy()
    found = sft_format_examples([Path(args.trajectories)])
    if len(found) == 1:
        policy = with_format_example(policy, found[0])
    res = format_sft_records(_jsonl(Path(args.trajectories)), policy, int(args.count), int(args.val_count),
                             int(args.max_turns), args.format_example, int(args.seed))
    if len(res["records"]) < int(args.count) + int(args.val_count):
        raise SystemExit(f"only {len(res['records'])} conversations qualify: {res['report']}")
    _write_jsonl(out / "sft_trajectories_format.jsonl", res["records"])
    if args.link_from:
        src = Path(args.link_from).resolve()
        for name in ("frames.jsonl", "frames", "split_manifest.json", "task_registry.jsonl"):
            if (src / name).exists() and not (out / name).exists():
                os.symlink(src / name, out / name)
    report = dict(res["report"], trajectories=str(args.trajectories))
    (out / "format_sft_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_subsets(args: argparse.Namespace) -> None:
    """Round 12: the data-scale subsets as files (sft_trajectories_<name>.jsonl, rows_<name>.jsonl, subsets.json)
    in --out, with the source SFT folder's frames, split manifest and task registry linked in (an SFT folder the
    training chain can use with "sft_records": false)."""
    import os

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flagged = json.loads(Path(args.exclude).read_text())["flagged_task_ids"] if args.exclude else []
    res = build_subsets(_jsonl(Path(args.trajectories)), _jsonl(Path(args.rows)),
                        [int(x) for x in args.sizes.split(",")], args.equal_tasks_of, flagged, args.seed)
    summary = {"seed": res["seed"], "trajectories": str(args.trajectories), "rows": str(args.rows),
               "scene_order": res["scene_order"], "subsets": {}}
    for name, s in res["subsets"].items():
        _write_jsonl(out / f"sft_trajectories_{name}.jsonl", s["sft"])
        _write_jsonl(out / f"rows_{name}.jsonl", s["rows"])
        summary["subsets"][name] = {"scenes": len(s["scenes"]), "train_records": s["train_records"],
                                    "val_records": len(s["sft"]) - s["train_records"],
                                    "reserve_rows": sum(1 for r in s["rows"] if r.get("reserve")),
                                    "reserve_scenes": len({r["scene_id"] for r in s["rows"] if r.get("reserve")}),
                                    **({"per_scene": s["per_scene"]} if "per_scene" in s else {}),
                                    "scene_ids": s["scenes"]}
    if args.link_from:
        src = Path(args.link_from).resolve()
        for name in ("frames.jsonl", "frames", "split_manifest.json", "task_registry.jsonl"):
            if (src / name).exists() and not (out / name).exists():
                os.symlink(src / name, out / name)
    (out / "subsets.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({n: {k: v for k, v in s.items() if k != "scene_ids"} for n, s in summary["subsets"].items()}, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("sft-frames")
    f.add_argument("--run", required=True, help="run_all output directory (tasks.jsonl = formal tasks)")
    f.add_argument("--split-manifest", required=True)
    f.add_argument("--splits", default=",".join(SFT_SPLITS))
    f.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke tests)")
    f.add_argument("--include-reserve", action="store_true",
                   help="round 10: also the reserve tasks (tasks_reserve.jsonl), marked reserve, for the larger SFT set")
    f.add_argument("--exclude", default=None, help="depth_check_summary.json: its flagged tasks are left out")
    f.add_argument("--scenes-root", default=None,
                   help="round 11: InteriorGS scene folders; with policy blocked_turn_style \"escape\" the inserted "
                        "blocked turns (blocked move, escape, new shortest path) are planned here and their frames listed")
    f.add_argument("--env-template", default=None, help="the RL env YAML (wall model, action limit) for the planning")
    f.add_argument("--workers", type=int, default=1, help="processes for the planning (one scene each)")
    f.add_argument("--out", required=True)
    r = sub.add_parser("sft-records")
    r.add_argument("--dir", required=True)
    r.add_argument("--scenes-root", default=None,
                   help="round 10: InteriorGS scene folders (labels.json, structure.json); with it a share of the "
                        "conversations get an inserted blocked turn (policy sft.export.blocked_turn_share)")
    r.add_argument("--env-template", default=None, help="the RL env YAML whose wall model the blocked turns use")
    c = sub.add_parser("rl-config")
    c.add_argument("--rows", required=True, help="published rows with split (sft-frames rows.jsonl)")
    c.add_argument("--env-template", default=str(Path(__file__).with_name("configs") / "rl_env_v10_S0.yaml"))
    c.add_argument("--ppo-template", default=None, help="a complete PPO config (dense-score S0); a --standard names one")
    c.add_argument("--gs-root", required=True, help="scene folders with 3dgs_compressed.ply")
    c.add_argument("--model", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--run-root", required=True, help="checkpoints, rollouts and validation go below it")
    c.add_argument("--val-limit", type=int, default=0)
    c.add_argument("--val-tasks", default=None,
                   help="round 10: a tasks.jsonl of published val rows (evaluation.py select --split val) used as the "
                        "validation set instead of --val-limit rows; validated in one batch")
    c.add_argument("--greedy-val", action="store_true", help="validation without sampling: one trajectory per task")
    c.add_argument("--history-mode", choices=HISTORY_MODES, default=None,
                   help="how the input carries earlier turns: none (rounds 6-9), no_concat (per-turn samples with "
                        "the full history), concat (one sequence per episode); default none")
    c.add_argument("--test-freq", type=int, default=None, help="validate every N updates (template: 25)")
    c.add_argument("--critic-warmup", type=int, default=None, help="updates that train only the value network")
    c.add_argument("--reward-preset", choices=REWARD_PRESETS, default=None,
                   help="round 11: S0 (sparse, the default) or dense (the dense-score pilot's S5 shaping on top of S0)")
    c.add_argument("--train-batch-size", type=int, default=None, help="tasks per update (template: 12)")
    c.add_argument("--ppo-mini-batch-size", type=int, default=None, help="tasks per optimizer step (template: 8)")
    c.add_argument("--save-freq", type=int, default=None,
                   help="a checkpoint every N updates, all kept; the older ones keep only actor/huggingface")
    c.add_argument("--no-internal-val", action="store_true", help="no validation inside training")
    c.add_argument("--opt-package", choices=sorted(OPT_PACKAGES), default=None,
                   help="round 11: the optimisation package (GRPO, mixed-outcome tasks only with refill, 8 episodes "
                        "per task, larger learning rate, smaller KL penalty, entropy bonus); p1gae: p1 with GAE and "
                        "the value network, which trains alone for the first 10 updates")
    c.add_argument("--candidates", type=int, default=None, help="tasks drawn per update (package default)")
    c.add_argument("--target-groups", type=int, default=None, help="tasks trained on per update (package default)")
    c.add_argument("--rollouts", type=int, default=None, help="episodes per task (package default)")
    c.add_argument("--task-pool", choices=TASK_POOLS, default=None,
                   help="round 11: the train tasks RL takes: formal (rounds 6-10, the default) or reserve (rows from sft-frames "
                        "--include-reserve; the rule since round 11: RL trains on tasks SFT never trained on)")
    c.add_argument("--format-example", default=None, choices=["placeholder", "action_names"],
                   help="round 12 freeze: the format line shows placeholder names (action1|action2|...) or real "
                        "action names; default the SFT export's (admission policy), and it must match the SFT "
                        "conversations given with --sft-trajectories")
    c.add_argument("--lenient-parse", action="store_true",
                   help="round 12 freeze: the env reads an answer leniently when the strict reading finds no action "
                        "(spaces instead of underscores, the first complete action block, a missing opening tag)")
    c.add_argument("--invalid-turn-nonpositive", action="store_true",
                   help="round 12 freeze: the tokens of a turn the env could not read get no positive advantage "
                        "(vagen/invalid_turns.py; PPO config algorithm.invalid_turn_nonpositive_adv)")
    c.add_argument("--rollout-log-probs", action="store_true",
                   help="the model line: vLLM also returns each sampled token's log-probability, for the per-update "
                        "comparison with the trainer's (an audit; the update is unchanged)")
    c.add_argument("--logprob-tokens", action="store_true",
                   help="the model line: with --rollout-log-probs, a file per update with every response token's two "
                        "log-probabilities, turn and distance from the last image token (vagen/logprob_tokens.py)")
    c.add_argument("--model-type", default=None,
                   help="the model line: the start model's type (vagen/models/model_types.py, checked against its "
                        "config.json) and its PPO-config overrides (none for the Qwen types)")
    c.add_argument("--invalid-penalty", type=float, default=None,
                   help="round 12 freeze: the penalty of an answer the env reads no action in (default the env "
                        "template's invalid_format_penalty, -0.1); the system prompt keeps the SFT text")
    c.add_argument("--turn-footprint-radius", type=float, default=None,
                   help="experiment 3A (the user's decision of 2026-11-06): the env blocks an in-place turn when a "
                        "disc of this radius (m, 0.25) around the camera's ground point meets a wall or an object box "
                        "reaching into 0.1-1.0 m (env turn_footprint_radius; standards v10_p1_*_fp); default off")
    c.add_argument("--progress-version", default=None, choices=["grouped_gated", "grouped_soft", "optimal"],
                   help="round 12 freeze, dense-reward variant V3: the grouped progress with the hard gate (D_cond's) "
                        "or the soft gate (relation group times the frame group's continuous score); round 13 (B): "
                        "optimal, the progress from the remaining minimum number of actions (needs --distance-tables)")
    c.add_argument("--distance-tables", default=None,
                   help="round 13 (B): the directory of per-task distance tables (distance_table.py build) the "
                        "optimal progress reads")
    c.add_argument("--table-near", action="store_true",
                   help="reward line (phase 2): a pose missing from its table takes the smallest reachable distance "
                        "of its 26 neighbour cells (env optimal_progress_near, distance_table_near.py); needs "
                        "--progress-version optimal")
    c.add_argument("--critic-feature-tables", default=None,
                   help="the privileged critic (package p1tgae_fc, 2026-11-13): the directory of the train tasks' "
                        "distance tables the env's critic_features read (every train task needs one; usually from "
                        "the standard)")
    c.add_argument("--progress-weights", default=None,
                   help="round 12 freeze, variants V1 / V2: the grouped progress's frame, face and relation weights, "
                        "e.g. 0.15,0.15,0.70 (default 0.25,0.25,0.5)")
    c.add_argument("--progress-reward-scale", type=float, default=None,
                   help="round 12 freeze, variants V4 / V5: the progress reward's scale (env "
                        "potential_field_reward_scale, the dense presets' 10)")
    c.add_argument("--milestone-rewards", default=None,
                   help="round 12 freeze, variant V6: frame, face and relation rewards paid once, the first time each "
                        "group is fully met, instead of the per-turn progress reward, e.g. 0.5,0.5,1.0")
    c.add_argument("--kl-loss-coef", type=float, default=None,
                   help="round 12 freeze, weak-start run W3: the actor's KL penalty coefficient, over the package's "
                        "(p1: 0.05)")
    c.add_argument("--actor-lr", type=float, default=None,
                   help="round 12 freeze, weak-start run W4: the actor's learning rate, over the package's (p1: 2e-6)")
    c.add_argument("--checkpoint-every", type=int, default=None,
                   help="round 12 freeze, weak-start runs: checkpoints every N updates, over the standard's save_freq "
                        "(the standards: 50)")
    c.add_argument("--progress-adv-weight", type=float, default=None,
                   help="round 12 freeze, variant V8: the group comparison normalises the success part (return - "
                        "progress) and the progress part apart and adds them, the progress part weighted (e.g. 0.5); "
                        "PPO config algorithm.progress_adv_weight, vagen/split_reward_adv.py")
    c.add_argument("--turn-credit", default=None,
                   help="reward line (2026-11-01): the per-turn credit estimator with these settings, a JSON object "
                        "(scale, gamma, turn_reward, baseline batch_mean / turn_index / none, turn_weight, "
                        "episode_weight, episode_norm batch / none, neg_clip, min_positive_mass, stall_penalty; "
                        "dump_dir, dump_updates: the smoke's span check); "
                        "PPO config algorithm.adv_estimator turn_credit and algorithm.turn_credit (the q38 stack: "
                        "vagen.turn_credit), vagen/turn_credit.py; "
                        "needs the S0 reward with --progress-version optimal --distance-tables")
    c.add_argument("--sft-trajectories", nargs="*", default=None,
                   help="SFT conversation files (sft_trajectories.jsonl): no RL train task may be among their tasks")
    c.add_argument("--exclude", default=None,
                   help="depth_check_summary.json: no RL train task may be among its flagged tasks")
    c.add_argument("--trial", action="store_true", help="one card, tiny batches (TRIAL_PPO)")
    c.add_argument("--seed", type=int, default=None,
                   help="round 12: the run's seed, the order of the tasks (data.seed); the template's otherwise")
    c.add_argument("--entropy-coeff", type=float, default=None,
                   help="round 12 batch: the actor's entropy bonus, over the package's (the user's decision of "
                        "2026-10-21: the A-group and E1 reruns with 0, reported apart from the standard)")
    c.add_argument("--group-filter", choices=GROUP_FILTERS, default=None,
                   help="round 12, the dense-reward first batch: the package's group filter (mixed_outcome) or "
                        "keep_by_return_std (all-fail tasks with spread returns fill the free places)")
    c.add_argument("--keep-std-threshold", type=float, default=None,
                   help="keep_by_return_std: the smallest return spread (std, n-1) of an all-fail task it takes")
    c.add_argument("--keep-std-floor", type=float, default=None,
                   help="keep_by_return_std: the filled tasks' group comparison divides by at least this (default 0)")
    c.add_argument("--curriculum-easy-updates", type=int, default=None,
                   help="round 12 batch: the first N updates draw only tasks with d* <= --curriculum-max-dstar")
    c.add_argument("--curriculum-max-dstar", type=int, default=None, help="the easy tasks' largest d* (default 3)")
    c.add_argument("--standard", default=None,
                   help="round 12: a named standard configuration (configs/rl_standards.json) fixing the recipe's options")
    c.add_argument("--stack", choices=STACKS, default=None,
                   help="the Qwen3.8-27B rerun (docs/q38/DESIGN_ASYNC.md): q38 writes the complete config of verl's V1 "
                        "trainer (verl's default + configs/ppo_q38_delta.yaml + the run's keys) in place of the 8B "
                        "one; default q38 for a --model-type in Q38_STACK_TYPES (qwen3_5), else 8b")
    c.add_argument("--out", required=True)
    u = sub.add_parser("subsets", help="round 12: nested train-scene subsets for the data-scale curve")
    u.add_argument("--trajectories", required=True, help="the SFT conversations (sft_trajectories.jsonl, with split)")
    u.add_argument("--rows", required=True, help="published rows with reserve rows (sft-frames --include-reserve)")
    u.add_argument("--exclude", default=None, help="depth_check_summary.json: flagged tasks never reach RL")
    u.add_argument("--sizes", default="50,100,200,400")
    u.add_argument("--equal-tasks-of", type=int, default=None,
                   help="also an arm with the train records of this subset size spread over all scenes")
    u.add_argument("--seed", type=int, default=SUBSET_SEED)
    u.add_argument("--link-from", default=None, help="the source SFT folder: link its frames, manifest and registry")
    u.add_argument("--out", required=True)
    x = sub.add_parser("format-sft", help="round 12 freeze: the short format SFT conversations (the user's 3.1.4)")
    x.add_argument("--trajectories", required=True, help="the SFT conversations (sft_trajectories.jsonl, with split)")
    x.add_argument("--count", type=int, default=400, help="train conversations")
    x.add_argument("--val-count", type=int, default=40, help="val conversations (the SFT's evaluation loss)")
    x.add_argument("--max-turns", type=int, default=2, help="at most this many assistant turns")
    x.add_argument("--format-example", default="action_names", choices=["placeholder", "action_names"])
    x.add_argument("--seed", type=int, default=FORMAT_SFT_SEED)
    x.add_argument("--link-from", default=None, help="the source SFT folder: link its frames, manifest and registry")
    x.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"sft-frames": cmd_sft_frames, "sft-records": cmd_sft_records, "rl-config": cmd_rl_config,
     "subsets": cmd_subsets, "format-sft": cmd_format_sft}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
