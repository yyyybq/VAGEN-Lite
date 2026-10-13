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
from typing import Any, Dict, List, Optional, Sequence

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


def cmd_sft_frames(args: argparse.Namespace) -> None:
    from .admission import load_policy
    from .release import finalize_task_row, registry_row, split_lookup

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.split_manifest).read_text())
    lookup = split_lookup(manifest)
    splits = tuple(s for s in args.splits.split(",") if s)
    policy = load_policy()
    rows, skipped = [], Counter()
    for raw in _jsonl(Path(args.run) / "tasks.jsonl"):
        split = lookup.get(str(raw["scene_id"]))
        if split is None:
            skipped["scene_not_in_manifest"] += 1
            continue
        if split not in splits:
            skipped[f"split_{split}"] += 1
            continue
        rows.append(finalize_task_row(raw, split))
    if args.limit:
        rows = rows[: args.limit]
    frames = sft_frames(rows, policy)
    _write_jsonl(out / "rows.jsonl", rows)
    _write_jsonl(out / "frames.jsonl", frames)
    _write_jsonl(out / "task_registry.jsonl",
                 [registry_row(r, {"sft": True, "rl": r["split"] == "train"}) for r in rows])
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    report = {"tasks": len(rows), "by_split": dict(Counter(r["split"] for r in rows)),
              "by_family": dict(Counter(r["task_type"] for r in rows)), "frames": len(frames),
              "scenes": len({r["scene_id"] for r in rows}), "skipped": dict(skipped)}
    (out / "sft_frames_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


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
    records, reasons = [], Counter()
    for row in rows:
        rec, reason = build_sft_record(row, images, policy)
        if rec is None:
            reasons[reason] += 1
            continue
        records.append(rec)
    _write_jsonl(d / "sft_trajectories.jsonl", records)
    report = {"tasks": len(rows), "records": len(records), "dropped": dict(reasons),
              "by_split": dict(Counter(r["split"] for r in records)),
              "turns": dict(Counter(r["trajectory_steps"] for r in records)),
              "images": len(images)}
    (d / "sft_records_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


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
    "trainer.ckpt_milestones": [1, 2],
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
}


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
    return P.system_prompt(format=c.prompt_format, step_translation=c.step_translation,
                           step_rotation_deg=c.step_rotation_deg, max_actions_per_step=c.max_actions_per_step,
                           action_sep=getattr(c, "action_sep", "|"), format_reward=c.format_reward,
                           invalid_format_penalty=c.invalid_format_penalty, success_reward=c.success_reward,
                           task_type=task_type, action_space=c.action_space,
                           enable_explicit_done=c.enable_explicit_done)


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
    parts = {"train": [r for r in rows if r.get("split") == "train"],
             "val": [r for r in rows if r.get("split") == "val"]}
    if args.val_limit:
        parts["val"] = parts["val"][: args.val_limit]
    if not parts["train"] or not parts["val"]:
        raise SystemExit("rl-config needs train and val rows (training_prep sft-frames rows.jsonl)")
    template = yaml.safe_load(Path(args.env_template).read_text())
    env_paths = {}
    for split, selected in parts.items():
        _write_jsonl(out / f"{split}.jsonl", selected)
        env = copy.deepcopy(template)
        entry = env["envs"][0]
        entry.update(n_envs=len(selected), seed=[0, len(selected)], seed_list=list(range(len(selected))))
        entry["config"].update(jsonl_path=str((out / f"{split}.jsonl").resolve()), gs_root=str(args.gs_root),
                               train_size=len(selected))
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
    if args.trial:
        for key, value in TRIAL_PPO.items():
            _set(ppo, key, value)
    (out / "ppo.yaml").write_text(yaml.safe_dump(ppo, sort_keys=False))
    report = {"train": len(parts["train"]), "val": len(parts["val"]), "trial": bool(args.trial),
              "model": model, "system_prompt_matches_sft": True,
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
    f.add_argument("--out", required=True)
    r = sub.add_parser("sft-records")
    r.add_argument("--dir", required=True)
    c = sub.add_parser("rl-config")
    c.add_argument("--rows", required=True, help="published rows with split (sft-frames rows.jsonl)")
    c.add_argument("--env-template", default=str(Path(__file__).with_name("configs") / "rl_env_v10_S0.yaml"))
    c.add_argument("--ppo-template", required=True, help="a complete PPO config (dense-score S0)")
    c.add_argument("--gs-root", required=True, help="scene folders with 3dgs_compressed.ply")
    c.add_argument("--model", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--run-root", required=True, help="checkpoints, rollouts and validation go below it")
    c.add_argument("--val-limit", type=int, default=0)
    c.add_argument("--trial", action="store_true", help="one card, tiny batches (TRIAL_PPO)")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"sft-frames": cmd_sft_frames, "sft-records": cmd_sft_records, "rl-config": cmd_rl_config}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
