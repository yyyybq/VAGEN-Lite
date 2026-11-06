"""Experiment 3A (the user's decision 甲 of 2026-11-06, turn-mismatch RL): how often the models turn and how often their
turns and translations are blocked, for any evaluation result, by replay without rendering.

The evaluation records each turn's answer, its parsed actions and its blocked moves, but not which actions the env
executed nor which of them were blocked (the feedback of the last turn is never shown).  Every episode of a result
is replayed in the evaluation's env (the template, default configs/rl_env_v10_S0.yaml, with the result's prompt
format and env switches from its model.json, so a result evaluated with --turn-footprint-radius is replayed under
the turn rule): env.reset on the task's line in the tasks file (found by its task_id), one env.step per recorded
answer, checked turn by turn against the record (task met after the turn, blocked moves and, for a result recorded
under the turn rule, the turns blocked by the footprint, footprint_blocks: replay_same).  Per turn: the actions the
env executed (each takes one step, so the parsed actions up to the step count; the rest of a turn ended by success or
the action limit is not executed) and the blocked ones, named in the env's detailed feedback ("2 of 3 actions
executed; blocked: turn_left (blocked by tea table)."; a turn whose names do not add up to its blocked count is
counted in turns_unknown).

Per result: actions executed, the rotation (turn_left / turn_right) share, the blocked share of rotations, of
translations and of all actions (over the turns whose blocked actions are known: a turn in turns_unknown is left out
of both numerator and denominator, its actions and blocked count given apart as unknown_turn_*), and the rule
(env_switches turn_footprint_radius, None = today's rule).  An env_switches key that is not an env setting of this
checkout is refused (the env would drop it and replay another rule).  d (the remaining steps where failed episodes end)
is reward/fail_end_steps.py, which replays the same way.

    python -m data_gen.interiorgs_v10.move_metrics --result DIR [--result DIR ...] --out FILE.json
        [--tasks FILE] [--env-template YAML] [--gs-root DIR] [--workers 4]

A result folder holds results.jsonl (with trajectories), model.json and done.json; its tasks file is --tasks or the
set's tasks.jsonl next to it (DIR/../tasks.jsonl).  FILE.json (per result the summary and every episode's turns)
and FILE.md (a table).  The login node: at most 4 workers, single-threaded (OMP_NUM_THREADS=1 ...), nice -n 19.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fail_analysis import BLOCKED_RE

TEMPLATE = Path(__file__).with_name("configs") / "rl_env_v10_S0.yaml"
GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"  # the evaluation's default
ROTATIONS = ("turn_left", "turn_right")
CHUNK = 40  # episodes per job


def turn_record(parsed: Sequence[str], steps_before: int, steps_after: int, feedback: str,
                blocked_count: int) -> Tuple[List[str], Optional[List[str]]]:
    """(the actions the env executed this turn, the blocked ones as its feedback names them, or None when the names
    do not add up to the turn's blocked count).  Every executed action takes one step (env.py), so the executed ones
    are the parsed actions up to the step count."""
    executed = [str(a) for a in list(parsed)[:max(0, int(steps_after) - int(steps_before))]]
    named = BLOCKED_RE.findall(feedback.split("blocked: ", 1)[1]) if "blocked: " in (feedback or "") else []
    return executed, (named if len(named) == int(blocked_count) else None)


def env_settings(template: Any, tasks_path: Any, gs_root: str, model: Dict[str, Any]) -> Dict[str, Any]:
    """The evaluation's env settings for a result, without rendering (as reward/fail_end_steps.py builds them): the
    template, the tasks file, the scenes, the v10 success rule, the result's prompt format and env switches.  A switch
    that is not an env setting is refused."""
    import yaml

    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    fields = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    switches = dict(model.get("env_switches") or {})
    unknown = sorted(set(switches) - fields)
    if unknown:
        raise SystemExit(f"env_switches {unknown} are not env settings of this checkout (the replay would run without "
                         "them): replay from a checkout that has them")
    cfg = dict(yaml.safe_load(Path(template).read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(tasks_path), gs_root=gs_root, render_backend=None, r1_legacy_success=False)
    if model.get("prompt_format"):
        cfg["prompt_format"] = model["prompt_format"]
    cfg.update(switches)
    return {k: v for k, v in cfg.items() if k in fields}


def worker(job: Tuple[str, str, str, str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    tasks_path, template, gs_root, model_json, items = job
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    model = json.loads(Path(model_json).read_text())
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**env_settings(template, tasks_path, gs_root, model)))

    def fake_render(**_kwargs):  # no frame: never a low-information one
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render
    position: Dict[str, int] = {}  # the task's line in the tasks file (a result assembled from several evaluations)
    for i, line in enumerate(l for l in Path(tasks_path).read_text().splitlines() if l.strip()):
        position.setdefault(json.loads(line)["task_id"], i)
    out = []
    for r in items:
        rec = {"index": r.get("index"), "task_id": r.get("task_id"), "task_type": r.get("task_type"),
               "success": bool(r.get("success"))}
        if r.get("sample") is not None:
            rec["sample"] = r["sample"]
        try:
            if not r.get("trajectory"):
                raise ValueError("no trajectory recorded")
            env.reset(seed=position[r["task_id"]])
            assert env.current_item["task_id"] == r["task_id"], (env.current_item["task_id"], r["task_id"])
            met, blocked, footprint, turns = [], [], [], []
            for t in r["trajectory"]:
                before = int(env._current_step)
                _, _, done, info = env.step(t["answer"])
                tm = (info.get("metrics") or {}).get("turn_metrics") or {}
                count = int(tm.get("collision_count", 0) or 0)
                executed, names = turn_record(info.get("actions") or [], before, int(env._current_step),
                                              info.get("env_feedback") or "", count)
                metric = env._calculate_canonical_metric()
                met.append(bool(metric and metric.get("success")))
                blocked.append(count)
                turn = {"executed": executed, "blocked": names, "blocked_count": count}
                # the replay's turns blocked by the footprint (the env reports them only under the turn rule)
                footprint.append(tm.get("turn_footprint_blocks"))
                if footprint[-1] is not None:
                    turn["footprint_blocks"] = int(footprint[-1])
                turns.append(turn)
                if done:
                    break
            # a result evaluated under the turn rule records each turn's footprint_blocks (evaluation.run_episode):
            # the replay must block as many turns by the footprint in that turn
            rec["replay_same"] = (met == [bool(t["met"]) for t in r["trajectory"]]
                                  and blocked == [int(t["collisions"]) for t in r["trajectory"]]
                                  and all(footprint[k] == int(t["footprint_blocks"])
                                          for k, t in enumerate(r["trajectory"][:len(footprint)])
                                          if "footprint_blocks" in t))
            rec["turns"] = turns
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out.append(rec)
    return out


def summarize(episodes: Sequence[Dict[str, Any]], rule: Optional[float] = None) -> Dict[str, Any]:
    """Counts and shares over the replayed episodes (episodes with an error left out).  The rotation share is over every
    executed action; the blocked shares (rotations, translations, all) only over the turns whose blocked actions are
    known: a turn in turns_unknown is left out of their numerators and denominators, and its executed actions and
    blocked count are given apart (unknown_turn_*)."""
    ok = [e for e in episodes if "error" not in e]
    turns = [t for e in ok for t in e["turns"]]
    known = [t for t in turns if t["blocked"] is not None]
    unknown = [t for t in turns if t["blocked"] is None]
    executed = [a for t in turns for a in t["executed"]]
    rotations = sum(a in ROTATIONS for a in executed)
    known_executed = [a for t in known for a in t["executed"]]
    known_rotations = sum(a in ROTATIONS for a in known_executed)
    named = [a for t in known for a in t["blocked"]]
    rotations_blocked = sum(a in ROTATIONS for a in named)
    translations_blocked = len(named) - rotations_blocked
    blocked = sum(int(t["blocked_count"]) for t in known)  # = len(named): a known turn's names add up to its count
    unknown_executed = [a for t in unknown for a in t["executed"]]

    def share(n: int, d: int) -> Optional[float]:
        return round(n / d, 4) if d else None

    return {"rule": rule, "episodes": len(episodes), "errors": len(episodes) - len(ok),
            "replay_same": sum(bool(e["replay_same"]) for e in ok), "turns": len(turns),
            "turns_unknown": len(unknown),
            "actions": len(executed), "rotations": rotations, "rotation_share": share(rotations, len(executed)),
            "rotations_blocked": rotations_blocked,
            "rotation_blocked_share": share(rotations_blocked, known_rotations),
            "translations": len(executed) - rotations, "translations_blocked": translations_blocked,
            "translation_blocked_share": share(translations_blocked, len(known_executed) - known_rotations),
            "blocked": blocked, "blocked_share": share(blocked, len(known_executed)),
            "unknown_turn_actions": len(unknown_executed),
            "unknown_turn_rotations": sum(a in ROTATIONS for a in unknown_executed),
            "unknown_turn_translations": sum(a not in ROTATIONS for a in unknown_executed),
            "unknown_turn_blocked": sum(int(t["blocked_count"]) for t in unknown)}


def episodes_of(result: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in (result / "results.jsonl").read_text().splitlines() if l.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", action="append", required=True, help="an evaluation result folder (repeatable)")
    ap.add_argument("--out", required=True, help="FILE.json; FILE.md gets the table")
    ap.add_argument("--tasks", default=None, help="the set's tasks.jsonl (default: next to each result folder)")
    ap.add_argument("--env-template", default=str(TEMPLATE))
    ap.add_argument("--gs-root", default=GS_ROOT)
    ap.add_argument("--workers", type=int, default=4, help="processes (the login node: at most 4; 1: in this process)")
    args = ap.parse_args(argv)
    jobs, keys, rules, infra = [], [], {}, defaultdict(list)
    for res in [Path(p) for p in args.result]:
        if not (res / "done.json").is_file():
            raise SystemExit(f"{res}: no done.json (the evaluation is not finished)")
        tasks = Path(args.tasks) if args.tasks else res.parent / "tasks.jsonl"
        if not tasks.is_file():
            raise SystemExit(f"{res}: no tasks file {tasks} (give --tasks)")
        model = json.loads((res / "model.json").read_text())
        env_settings(args.env_template, tasks, args.gs_root, model)  # refuses switches this checkout does not know
        rules[str(res)] = (model.get("env_switches") or {}).get("turn_footprint_radius")
        rows = episodes_of(res)
        infra[str(res)] = [{"index": r.get("index"), "error": f"evaluation: {r['error']}"} for r in rows if "error" in r]
        rows = [r for r in rows if "error" not in r]
        for k in range(0, len(rows), CHUNK):
            jobs.append((str(tasks), str(args.env_template), str(args.gs_root), str(res / "model.json"), rows[k:k + CHUNK]))
            keys.append(str(res))
    per: Dict[str, List[Dict[str, Any]]] = {k: list(v) for k, v in infra.items()}
    if int(args.workers) > 1:
        with Pool(int(args.workers)) as pool:
            for key, part in zip(keys, pool.imap(worker, jobs)):
                per[key] += part
    else:
        for key, job in zip(keys, jobs):
            per[key] += worker(job)
    out = {k: summarize(v, rules[k]) for k, v in per.items()}
    Path(args.out).write_text(json.dumps({"env_template": str(args.env_template), "results": out,
                                          "episodes": per}, indent=1))
    L = ["| 结果 | 规则 | 局数 | 出错 | 重放一致 | 执行的动作 | 转身占比 | 转身被挡占比 | 平移被挡占比 | 全部被挡占比 | "
         "读不出被挡动作的轮（其动作数，不计入被挡占比） |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, s in out.items():
        rule = f"转身脚印 {s['rule']} m" if s["rule"] is not None else "旧规则"
        L.append(f"| {Path(k).parent.name}/{Path(k).name} | {rule} | {s['episodes']} | {s['errors']} | {s['replay_same']} | "
                 f"{s['actions']} | {s['rotation_share']} | {s['rotation_blocked_share']}（{s['rotations_blocked']}） | "
                 f"{s['translation_blocked_share']}（{s['translations_blocked']}） | {s['blocked_share']} | "
                 f"{s['turns_unknown']}（{s['unknown_turn_actions']}） |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
