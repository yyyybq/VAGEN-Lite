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
"""

from __future__ import annotations

import argparse
import json
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
# round-11 pilot (empty until then).  The system prompt stays the SFT one for both (its reward lines describe
# the sparse terms), so the two runs differ only in the reward.
REWARD_PRESETS = ("S0", "dense")
DENSE_S5 = {"enable_potential_shaping_reward": True, "potential_field_gamma": 0.99,
            "enable_near_success_reward": True, "near_success_bonus": 0.5, "near_success_threshold": 0.55,
            "enable_visibility_shaping_reward": True}
DENSE_ADJUST: Dict[str, Any] = {}


def reward_overrides(preset: str) -> Dict[str, Any]:
    """Env-config overrides of a reward preset."""
    if preset not in REWARD_PRESETS:
        raise ValueError(f"reward preset must be one of {REWARD_PRESETS}, not {preset!r}")
    return {} if preset == "S0" else {**DENSE_S5, **DENSE_ADJUST}


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


def prompt_mismatches(env_cfg: Dict[str, Any], task_types: Sequence[str], policy: Dict[str, Any]) -> List[str]:
    """Task types whose RL system prompt differs from the SFT conversations' (release.protocol_system_prompt)."""
    from .release import protocol_system_prompt

    return [t for t in sorted(set(task_types)) if env_system_prompt(env_cfg, t) != protocol_system_prompt(t, policy)]


def cmd_rl_config(args: argparse.Namespace) -> None:
    import copy

    import yaml

    from .admission import load_policy

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(Path(args.rows))
    # RL trains on the formal tasks of train; reserve rows (sft-frames --include-reserve) are SFT-only, as the task
    # registry says (round 10: the first RL run with reserve rows in rows.jsonl took them too)
    parts = {"train": [r for r in rows if r.get("split") == "train" and not r.get("reserve")],
             "val": [r for r in rows if r.get("split") == "val" and not r.get("reserve")]}
    reserve_skipped = sum(1 for r in rows if r.get("reserve") and r.get("split") in ("train", "val"))
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
    env_paths = {}
    for split, selected in parts.items():
        _write_jsonl(out / f"{split}.jsonl", selected)
        env = copy.deepcopy(template)
        entry = env["envs"][0]
        entry.update(n_envs=len(selected), seed=[0, len(selected)], seed_list=list(range(len(selected))))
        entry["config"].update(jsonl_path=str((out / f"{split}.jsonl").resolve()), gs_root=str(args.gs_root),
                               train_size=len(selected), **history["env"])
        env_paths[split] = out / f"{split}_env.yaml"
        env_paths[split].write_text(yaml.safe_dump(env, sort_keys=False))
    mismatches = prompt_mismatches(template["envs"][0]["config"], [r["task_type"] for r in rows], load_policy())
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
    (out / "ppo.yaml").write_text(yaml.safe_dump(ppo, sort_keys=False))
    report = {"train": len(parts["train"]), "val": len(parts["val"]), "trial": bool(args.trial),
              "model": model, "system_prompt_matches_sft": True,
              "history_mode": getattr(args, "history_mode", None) or "none", "val_tasks": val_tasks,
              "greedy_val": bool(getattr(args, "greedy_val", False)),
              "test_freq": ppo["trainer"].get("test_freq"), "critic_warmup": ppo["trainer"].get("critic_warmup"),
              "reward_preset": getattr(args, "reward_preset", None) or "S0",
              "env_reward_overrides": reward_overrides(getattr(args, "reward_preset", None) or "S0"),
              "train_batch_size": (ppo.get("data") or {}).get("train_batch_size"),
              "save_freq": ppo["trainer"].get("save_freq"),
              "reserve_rows_skipped": reserve_skipped,
              "by_family_train": dict(Counter(r["task_type"] for r in parts["train"]))}
    (out / "rl_config_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


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
    c.add_argument("--ppo-template", required=True, help="a complete PPO config (dense-score S0)")
    c.add_argument("--gs-root", required=True, help="scene folders with 3dgs_compressed.ply")
    c.add_argument("--model", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--run-root", required=True, help="checkpoints, rollouts and validation go below it")
    c.add_argument("--val-limit", type=int, default=0)
    c.add_argument("--val-tasks", default=None,
                   help="round 10: a tasks.jsonl of published val rows (evaluation.py select --split val) used as the "
                        "validation set instead of --val-limit rows; validated in one batch")
    c.add_argument("--greedy-val", action="store_true", help="validation without sampling: one trajectory per task")
    c.add_argument("--history-mode", choices=HISTORY_MODES, default="none",
                   help="how the input carries earlier turns: none (rounds 6-9), no_concat (per-turn samples with "
                        "the full history), concat (one sequence per episode)")
    c.add_argument("--test-freq", type=int, default=None, help="validate every N updates (template: 25)")
    c.add_argument("--critic-warmup", type=int, default=None, help="updates that train only the value network")
    c.add_argument("--reward-preset", choices=REWARD_PRESETS, default="S0",
                   help="round 11: S0 (sparse) or dense (the dense-score pilot's S5 shaping on top of S0)")
    c.add_argument("--train-batch-size", type=int, default=None, help="tasks per update (template: 12)")
    c.add_argument("--ppo-mini-batch-size", type=int, default=None, help="tasks per optimizer step (template: 8)")
    c.add_argument("--save-freq", type=int, default=None,
                   help="a checkpoint every N updates, all kept; the older ones keep only actor/huggingface")
    c.add_argument("--no-internal-val", action="store_true", help="no validation inside training")
    c.add_argument("--trial", action="store_true", help="one card, tiny batches (TRIAL_PPO)")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"sft-frames": cmd_sft_frames, "sft-records": cmd_sft_records, "rl-config": cmd_rl_config}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
