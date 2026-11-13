"""Step-5 skill-instruction diagnosis: does the executor's output change when ONLY the instruction changes?

    python scripts/skill_probe.py --fixture tests/fixtures/real_obs_turning_on_radio_311 --port 8000 \
        --instruction "Pick up the radio." --instruction "Press the power button of the radio." \
        --instruction "Open the door." --repeats 3 --horizon 32 --out docs/skill_probe/radio_311

Offline part (this script):
  * the SAME recorded observations (a fixture written by scripts/record_obs_fixture.py; a --write-synthetic fixture
    only checks the wiring) go to a running executor server (executor/openpi_server/serve_b1k_prompted.py) through
    executor/openpi_ws.py OpenPIWebsocketExecutor in "chunk" mode, i.e. the server's wire format (agent/wire.py
    msgpack + numpy; "__prompt__", "__flush__", "__return_chunk__"), and only the whitelisted policy keys
    (agent/fields.py build_observation), exactly as during an episode;
  * one fresh connection per (instruction, repeat) and a {"reset": True} before every observation, so no action
    buffer or prompt state carries over between queries;
  * the reference instruction is the server's default prompt (the trained task prompt) unless --reference;
  * noise floor: mean |chunk_a - chunk_b| between repeats of the SAME instruction on the same observation (pi0.5
    samples flow-matching noise on every call, so repeats differ); cross: the same statistic between an
    instruction and the reference (repeat r against repeat r);
  * both per action group (agent/fields.py ACTION_SLICES: base velocity, torso, left/right arm joint positions,
    left/right gripper) over the whole chunk and over its first action; ratio = cross / noise floor. An
    instruction "responds" when the ratio reaches --threshold (default 2.0) in at least one group.

Reading the result:
  * an executor trained only on full-task prompts that shows NO instruction sensitivity (all ratios ~1) cannot serve
    as a skill executor: the planner's sub-instructions would be ignored. Skills stay available: false /
    verified: false in configs/skills/*.yaml, M-skill and instruction-type recovery stay off, unless a minimal skill
    fine-tune is decided (a separate, labelled training contribution);
  * sensitivity is necessary, not sufficient: changed actions do not show that the robot performs the asked skill.

Online part (not this script): full episodes with sub-instructions on Dev instances only (public_test indices 10-19
or train instances), through the official evaluator with python -m eval.launch and a config that overrides the
planner. For example a derived config with
    executor: {skill_catalog: <copy of the catalog with the probed skill available: true>}
    providers: {probe: {kind: scripted, label: skill-probe, replies: {plan.next_subtask: [{task_complete: false,
                skill: press_button, args: [{name: object, value: radio}], target_subgoals: [], max_steps: 900,
                evidence: skill probe}]}}}
    modules: {plan: {enabled: true, impl: llm, provider: probe, prompt_version: v1}}
then python -m eval.launch --config <it> --task turning_on_radio --mode public_test --instances 10 --write-video.
The scripted provider makes the instruction the only variable; such runs are diagnoses, never results. Record the
verdict (video time stamps, official JSON) in the catalog's verified / evidence fields.

Status: written and unit-tested against a fake executor server (tests/test_ops_skill_probe.py); not yet run on 105.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Optional, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.fields import ACTION_SLICES, build_observation
from agent.interfaces import ACTION_DIM_R1PRO, EpisodeContext
from executor.openpi_ws import OpenPIWebsocketExecutor
from scripts.record_obs_fixture import load_fixture

GROUPS = {k: list(range(s.start, s.stop)) for k, s in ACTION_SLICES.items()}
GROUPS["arms"] = GROUPS["left_arm"] + GROUPS["right_arm"]
GROUPS["grippers"] = GROUPS["left_gripper"] + GROUPS["right_gripper"]
NOISE_EPS = 1e-9
CHANGE_EPS = 1e-6
META_KEYS = ("default_prompt", "checkpoint_dir", "train_config", "norm_stats_repo_id", "control_mode", "action_horizon",
             "model_action_horizon", "multi_task", "supports_prompt_override")


def observations_from_fixture(messages: Sequence[dict], max_obs: int) -> list:
    """One Observation per (message, env), whitelisted exactly as the agent server does."""
    out = []
    for i, msg in enumerate(messages):
        arrays = {k: v for k, v in msg.items() if isinstance(v, np.ndarray) and v.ndim >= 1}
        proprio = next((v for k, v in arrays.items() if k.endswith("::proprio")), None)
        if proprio is None:
            continue
        n_env = proprio.shape[0] if proprio.ndim == 2 else 1
        for e in range(n_env):
            env_obs = {k: (v[e:e + 1] if proprio.ndim == 2 and v.shape[0] == n_env else v[None]) for k, v in arrays.items()}
            out.append(build_observation(env_obs, step=i, received_at=0.0))
            if len(out) >= max_obs:
                return out
    return out


def query(params: dict, ctx: EpisodeContext, obs_list: list, instruction: str, horizon: int) -> tuple[list, list]:
    """Fresh connection; reset before every observation; returns ([chunk (H, 23)], [infer_ms])."""
    ex = OpenPIWebsocketExecutor(params, env_idx=0)
    chunks, times = [], []
    try:
        for obs in obs_list:
            ex.reset(ctx)
            c = ex.act(obs, instruction, horizon)
            chunks.append(np.asarray(c.actions, np.float64).reshape(-1, ACTION_DIM_R1PRO))
            times.append(float(c.infer_ms))
    finally:
        ex.close()
    return chunks, times


def mad_per_dim(a: np.ndarray, b: np.ndarray, first_only: bool) -> np.ndarray:
    n = min(a.shape[0], b.shape[0])
    a, b = (a[:1], b[:1]) if first_only else (a[:n], b[:n])
    return np.mean(np.abs(a - b), axis=0)


def group_means(per_dim: Optional[np.ndarray]) -> dict:
    if per_dim is None:
        return {g: None for g in GROUPS}
    return {g: float(np.mean(per_dim[idx])) for g, idx in GROUPS.items()}


def analyse(chunks: dict, reference: str, threshold: float) -> dict:
    """chunks[instruction][repeat][obs] -> (H, 23). Noise floor, cross differences and ratios per group."""
    instructions = list(chunks)
    repeats = min(len(v) for v in chunks.values())
    n_obs = min(len(r) for v in chunks.values() for r in v)
    out: dict = {"repeats": repeats, "observations": n_obs, "threshold": threshold}
    for scope, first in (("chunk", False), ("first_action", True)):
        noise_terms = [mad_per_dim(chunks[i][r1][o], chunks[i][r2][o], first)
                       for i in instructions for r1, r2 in itertools.combinations(range(repeats), 2) for o in range(n_obs)]
        noise = np.mean(noise_terms, axis=0) if noise_terms else None
        noise_g = group_means(noise)
        per_instr = {}
        for ins in instructions:
            if ins == reference:
                continue
            cross = np.mean([mad_per_dim(chunks[ins][r][o], chunks[reference][r][o], first)
                             for r in range(repeats) for o in range(n_obs)], axis=0)
            cross_g = group_means(cross)
            ratios = {}
            for g, c in cross_g.items():
                nz = noise_g[g]
                if nz is None:
                    ratios[g] = None
                elif nz < NOISE_EPS:  # deterministic in this group: any change is a response
                    ratios[g] = float("inf") if c > CHANGE_EPS else None
                else:
                    ratios[g] = c / nz
            if noise is None:
                verdict = "unmeasured (needs --repeats >= 2 for a noise floor)"
            else:
                verdict = "responds" if any(r is not None and r >= threshold for r in ratios.values()) else "no response"
            per_instr[ins] = {"cross": cross_g, "ratio": ratios, "verdict": verdict}
        out[scope] = {"noise_floor": noise_g, "instructions": per_instr}
    verdicts = [v["verdict"] for v in out["chunk"]["instructions"].values()]
    responding = [i for i, v in out["chunk"]["instructions"].items() if v["verdict"].startswith("responds")]
    if not verdicts:
        overall = "unmeasured (no instruction besides the reference)"
    elif all(v.startswith("unmeasured") for v in verdicts):
        overall = "unmeasured (needs --repeats >= 2)"
    elif responding:
        overall = f"instruction-sensitive: {len(responding)}/{len(verdicts)} instructions change the actions beyond the noise floor"
    else:
        overall = ("NO instruction sensitivity detected: the executor ignores these instructions and cannot serve as a "
                   "skill executor for them")
    out["responding"] = responding
    out["overall"] = overall
    return out


def run_probe(*, fixture: str, host: str, port: int, instructions: Sequence[str], reference: Optional[str] = None,
              repeats: int = 3, horizon: int = 32, max_obs: int = 5, threshold: float = 2.0, task: Optional[str] = None,
              recv_timeout_s: float = 120.0, connect_timeout_s: float = 60.0) -> dict:
    messages, manifest = load_fixture(fixture)
    obs_list = observations_from_fixture(messages, max_obs)
    if not obs_list:
        raise ValueError(f"fixture {fixture} holds no observation with a '<robot>::proprio' key")
    params = {"host": host, "port": port, "mode": "chunk", "allow_prompt_override": True,
              "recv_timeout_s": recv_timeout_s, "connect_timeout_s": connect_timeout_s}
    probe = OpenPIWebsocketExecutor(params)
    try:
        meta = probe.metadata
    finally:
        probe.close()
    task = task or (manifest.get("tags") or {}).get("task") or next(iter(meta.get("prompts") or {}), "unknown")
    reference = reference or meta.get("default_prompt")
    if not reference:
        raise ValueError("the executor reports no default_prompt; pass --reference")
    ordered = [reference] + [i for i in dict.fromkeys(instructions) if i != reference]
    ctx = EpisodeContext(episode_id="skill_probe", env_index=0, task_name=task, task_id=obs_list[0].task_id,
                         task_instruction=reference, goal_spec=None, skills=(), max_steps=None, wallclock_budget_s=None)
    chunks: dict = {}
    infer_ms: list = []
    t0 = time.monotonic()
    for ins in ordered:
        chunks[ins] = []
        for _ in range(repeats):
            c, ms = query(params, ctx, obs_list, ins, horizon)
            chunks[ins].append(c)
            infer_ms += ms
    result = analyse(chunks, reference, threshold)
    result.update({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixture": {"path": os.path.abspath(fixture), "synthetic": manifest.get("synthetic"),
                    "source": manifest.get("source"), "tags": manifest.get("tags"), "observations_used": len(obs_list)},
        "executor": {"host": host, "port": port, **{k: meta.get(k) for k in META_KEYS}},
        "task": task, "reference": reference, "instructions": ordered[1:], "horizon_requested": horizon,
        "chunk_len": int(chunks[reference][0][0].shape[0]),
        "queries": len(infer_ms), "infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
        "wall_s": round(time.monotonic() - t0, 2),
    })
    return result


def _f(x: Optional[float], nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}g}"


def render(res: dict) -> str:
    lines = ["# Skill-instruction probe", ""]
    fx = res["fixture"]
    if fx.get("synthetic") is not False:
        lines += [f"**{'SYNTHETIC' if fx.get('synthetic') else 'UNKNOWN-PROVENANCE'} fixture: this checks the wiring "
                  "only; it is not a diagnosis of the executor.**", ""]
    ex = res["executor"]
    lines += [f"- executor {ex['host']}:{ex['port']} checkpoint `{ex.get('checkpoint_dir')}` config `{ex.get('train_config')}`",
              f"- fixture {fx['path']} ({fx['observations_used']} observations; source: {fx.get('source')})",
              f"- reference (task prompt): \"{res['reference']}\"",
              f"- repeats {res['repeats']}, chunk length {res['chunk_len']}, threshold {res['threshold']}, "
              f"{res['queries']} queries, mean infer {_f(res['infer_ms_mean'])} ms",
              "", f"**Overall: {res['overall']}**", ""]
    groups = list(GROUPS)
    for scope in ("chunk", "first_action"):
        s = res[scope]
        lines += [f"## {scope.replace('_', ' ')}: mean |difference| per group (ratio to the noise floor)", "",
                  "| instruction | " + " | ".join(groups) + " | verdict |", "|---|" + "---:|" * len(groups) + "---|",
                  "| noise floor (same instruction) | " + " | ".join(_f(s["noise_floor"][g]) for g in groups) + " | |"]
        for ins, v in s["instructions"].items():
            cells = [f"{_f(v['cross'][g])} ({_f(v['ratio'][g], 3)})" for g in groups]
            lines.append(f"| {ins} | " + " | ".join(cells) + f" | {v['verdict']} |")
        lines.append("")
    lines += ["Units: base = normalised velocity command, torso / arms = joint positions (rad), grippers = [-1, 1].",
              "Sensitivity is necessary, not sufficient: confirm a skill with full Dev episodes (see the docstring)."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", required=True, help="fixture path (scripts/record_obs_fixture.py; .npz + .json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--instruction", action="append", default=[], help="instruction to compare with the reference")
    ap.add_argument("--instructions-file", default=None, help="one instruction per line")
    ap.add_argument("--reference", default=None, help="default: the executor's default_prompt")
    ap.add_argument("--task", default=None, help="task name for the executor's prompt table (default: fixture tag)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=32, help="actions requested per chunk (__return_chunk__)")
    ap.add_argument("--max-obs", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--recv-timeout-s", type=float, default=120.0)
    ap.add_argument("--connect-timeout-s", type=float, default=60.0)
    ap.add_argument("--out", default=None, help="output path without extension (.json and .md)")
    args = ap.parse_args(argv)
    instructions = list(args.instruction)
    if args.instructions_file:
        with open(args.instructions_file, encoding="utf-8") as f:
            instructions += [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not instructions:
        ap.error("give at least one --instruction or --instructions-file")
    if args.repeats < 1 or args.max_obs < 1 or args.horizon < 1:
        ap.error("--repeats, --max-obs and --horizon must be >= 1 (--repeats >= 2 for a noise floor)")
    res = run_probe(fixture=args.fixture, host=args.host, port=args.port, instructions=instructions,
                    reference=args.reference, repeats=args.repeats, horizon=args.horizon, max_obs=args.max_obs,
                    threshold=args.threshold, task=args.task, recv_timeout_s=args.recv_timeout_s,
                    connect_timeout_s=args.connect_timeout_s)
    text = render(res)
    print(text, end="")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        with open(args.out + ".md", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}.json and {args.out}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
