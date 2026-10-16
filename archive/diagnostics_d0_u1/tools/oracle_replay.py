"""
Oracle Replay — feed SFT (oracle) trajectories into ActiveSpatialEnv and measure
the ceiling RL is trying to reach.

For each SFT trajectory:
  1. env.reset(seed=source_item_idx)  → same scene/task as RL sees
  2. For each assistant turn, env.step(assistant_content) → env parses
     <action>...</action>, applies actions, computes reward + scores.
  3. Capture per-step reward and per-traj terminal info["metrics"]["traj_metrics"].

Outputs:
  * Per-trajectory CSV with: sum_reward, env_final_score, env_traj_success,
    sft_final_score, episode_length, success_path (done_action / auto_term /
    max_steps / none).
  * Aggregate summary printed to stdout.

Usage:
  python tools/oracle_replay.py \
      --sft-jsonl data_gen/active_spatial_sft/output_0267_v7/sft_data.jsonl \
      --env-yaml examples/train/active_spatial/env_config_v18_potential2.yaml \
      --match-sft \
      --out oracle_replay_v18_potential2.csv

Flags:
  --match-sft   Override env step_rotation_deg=30 to match SFT (the optimizer
                planned actions assuming 30°). Strongly recommended; otherwise
                action sequences underrotate and orientation_score is biased low.
"""

import argparse
import csv
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_env_config(yaml_path: Path, overrides: Dict[str, Any]):
    """Load the env_config block from a verl-style env yaml and apply overrides."""
    with open(yaml_path, "r") as f:
        doc = yaml.safe_load(f)
    if "env1" in doc and "env_config" in doc["env1"]:
        cfg_dict = dict(doc["env1"]["env_config"])
    else:
        raise ValueError(f"Could not find env1.env_config in {yaml_path}")
    cfg_dict.update(overrides)

    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
    field_names = {f.name for f in dataclasses.fields(ActiveSpatialEnvConfig)}
    for k in list(cfg_dict.keys()):
        if k not in field_names:
            print(f"[warn] env_config field not recognized, dropping: {k}", file=sys.stderr)
            cfg_dict.pop(k)
    return ActiveSpatialEnvConfig(**cfg_dict)


def extract_assistant_msgs(traj: Dict[str, Any]) -> List[str]:
    return [m.get("content", "") for m in traj.get("conversations", [])
            if m.get("role") == "assistant"]


def replay_one(env, traj: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    idx = int(traj["source_item_idx"])
    obs, info = env.reset(seed=idx)
    actions_msgs = extract_assistant_msgs(traj)

    sum_reward = 0.0
    done = False
    last_info: Dict[str, Any] = info or {}
    n_steps_executed = 0

    for content in actions_msgs:
        if done:
            break
        obs, r, done, info = env.step(content)
        last_info = info
        sum_reward += float(r)
        n_steps_executed += 1
        if verbose:
            print(f"  step {n_steps_executed}: r={r:.4f}  cum={sum_reward:.4f}  "
                  f"score={info.get('current_potential_score', float('nan')):.3f}  done={done}")

    tm = (last_info.get("metrics") or {}).get("traj_metrics", {}) if isinstance(last_info, dict) else {}

    success_path = "none"
    if tm.get("success_by_done"):
        success_path = "done_action"
    elif tm.get("success_by_auto"):
        success_path = "auto_term"
    elif tm.get("success_by_max_steps"):
        success_path = "max_steps"

    return {
        "id": traj.get("id"),
        "source_item_idx": idx,
        "scene_id": traj.get("scene_id"),
        "task_type": traj.get("task_type"),
        "sft_initial_score": float(traj.get("initial_score", float("nan"))),
        "sft_final_score": float(traj.get("final_score", float("nan"))),
        "sft_n_steps": int(traj.get("trajectory_steps", -1)),
        "sft_n_actions": int(traj.get("total_actions", -1)),
        "env_n_steps": n_steps_executed,
        "env_sum_reward": sum_reward,
        "env_final_total_score": float(tm.get("final_score", float("nan"))),
        "env_final_pos_score": float(tm.get("final_position_score", float("nan"))),
        "env_final_ori_score": float(tm.get("final_orientation_score", float("nan"))),
        "env_best_score": float(tm.get("best_score", float("nan"))),
        "env_traj_success": bool(tm.get("success", False)),
        "env_invalid_actions": int(tm.get("invalid_action_count", -1)),
        "env_near_success_steps": int(tm.get("near_success_step_count", -1)),
        "env_near_success_bonus": float(tm.get("near_success_bonus_total", 0.0)),
        "env_collisions": int(tm.get("total_collisions", 0)),
        "success_path": success_path,
        "done_at_step": n_steps_executed if done else -1,
    }


def summarize(rows: List[Dict[str, Any]], env_yaml: str) -> None:
    if not rows:
        print("No rows to summarize.")
        return

    def col(name):
        return [r[name] for r in rows
                if r[name] is not None and not (isinstance(r[name], float) and r[name] != r[name])]

    n = len(rows)
    n_succ = sum(1 for r in rows if r["env_traj_success"])
    n_done_act = sum(1 for r in rows if r["success_path"] == "done_action")
    n_auto = sum(1 for r in rows if r["success_path"] == "auto_term")
    n_max = sum(1 for r in rows if r["success_path"] == "max_steps")
    n_none = sum(1 for r in rows if r["success_path"] == "none")

    rewards = col("env_sum_reward")
    finals = col("env_final_total_score")
    pos_finals = col("env_final_pos_score")
    ori_finals = col("env_final_ori_score")
    sft_finals = col("sft_final_score")
    bests = col("env_best_score")
    ep_lens = col("env_n_steps")

    def s(xs):
        if not xs:
            return "n/a"
        return f"mean={statistics.mean(xs):.4f}  median={statistics.median(xs):.4f}  min={min(xs):.4f}  max={max(xs):.4f}"

    print("\n" + "=" * 78)
    print(f"  ORACLE REPLAY SUMMARY  ({env_yaml})")
    print("=" * 78)
    print(f"  trajectories replayed   : {n}")
    print(f"  env_traj_success TRUE   : {n_succ} / {n}  ({100.0*n_succ/n:.1f}%)")
    print(f"    via done_action       : {n_done_act}")
    print(f"    via auto_termination  : {n_auto}")
    print(f"    via max_steps         : {n_max}")
    print(f"    not succeeded         : {n_none}")
    print()
    print(f"  env_sum_reward          : {s(rewards)}")
    print(f"  env_final_total_score   : {s(finals)}")
    print(f"  env_final_pos_score     : {s(pos_finals)}")
    print(f"  env_final_ori_score     : {s(ori_finals)}")
    print(f"  env_best_score          : {s(bests)}")
    print(f"  sft_final_score (ref)   : {s(sft_finals)}")
    print(f"  episode_length (steps)  : {s(ep_lens)}")
    print()
    deltas = [r["env_final_total_score"] - r["sft_final_score"]
              for r in rows
              if r["env_final_total_score"] == r["env_final_total_score"]
              and r["sft_final_score"] == r["sft_final_score"]]
    if deltas:
        print(f"  Δ(env_final − sft_final): mean={statistics.mean(deltas):+.4f}  "
              f"median={statistics.median(deltas):+.4f}  "
              f"min={min(deltas):+.4f}  max={max(deltas):+.4f}")
        n_drop = sum(1 for d in deltas if d < -0.05)
        print(f"  trajectories with env_final << sft_final by >0.05: {n_drop} / {len(deltas)}")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-jsonl", type=str,
                    default="data_gen/active_spatial_sft/output_0267_v7/sft_data.jsonl")
    ap.add_argument("--env-yaml", type=str,
                    default="examples/train/active_spatial/env_config_v18_potential2.yaml")
    ap.add_argument("--max-traj", type=int, default=-1,
                    help="Replay only the first N trajectories (-1 = all).")
    ap.add_argument("--match-sft", action="store_true",
                    help="Override env step_rotation_deg=30 (SFT was planned at 30°).")
    ap.add_argument("--rotation-deg", type=float, default=None)
    ap.add_argument("--translation", type=float, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV path (default: oracle_replay_<yaml_stem>.csv).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sft_path = Path(args.sft_jsonl).resolve()
    yaml_path = Path(args.env_yaml).resolve()

    overrides: Dict[str, Any] = {}
    if args.match_sft:
        overrides["step_rotation_deg"] = 30.0
    if args.rotation_deg is not None:
        overrides["step_rotation_deg"] = float(args.rotation_deg)
    if args.translation is not None:
        overrides["step_translation"] = float(args.translation)

    print(f"[oracle_replay] sft_jsonl  : {sft_path}")
    print(f"[oracle_replay] env_yaml   : {yaml_path}")
    print(f"[oracle_replay] overrides  : {overrides}")

    env_cfg = load_env_config(yaml_path, overrides)
    print(f"[oracle_replay] step_translation={env_cfg.step_translation}  "
          f"step_rotation_deg={env_cfg.step_rotation_deg}  "
          f"max_actions_per_step={env_cfg.max_actions_per_step}  "
          f"success_score_threshold={env_cfg.success_score_threshold}  "
          f"success_require_both={getattr(env_cfg, 'success_require_both', False)}  "
          f"success_reward={env_cfg.success_reward}")

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    env = ActiveSpatialEnv(env_cfg)

    with open(sft_path, "r") as f:
        sft = [json.loads(l) for l in f if l.strip()]
    if args.max_traj > 0:
        sft = sft[: args.max_traj]
    print(f"[oracle_replay] loaded {len(sft)} trajectories")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, traj in enumerate(sft):
        try:
            r = replay_one(env, traj, verbose=args.verbose)
            rows.append(r)
            if (i + 1) % 10 == 0 or args.verbose:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"[{i+1}/{len(sft)}] "
                      f"succ={sum(1 for x in rows if x['env_traj_success'])} "
                      f"avg_reward={statistics.mean(x['env_sum_reward'] for x in rows):.3f} "
                      f"({rate:.2f} traj/s, "
                      f"eta={int((len(sft)-i-1)/max(rate,1e-6))}s)")
        except Exception as e:
            print(f"[error] traj {i} (id={traj.get('id')}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    out_path = Path(args.out) if args.out else Path(f"oracle_replay_{yaml_path.stem}.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[oracle_replay] wrote {len(rows)} rows → {out_path}")
    summarize(rows, str(yaml_path.name))


if __name__ == "__main__":
    main()
