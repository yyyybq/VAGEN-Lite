"""Reward line (the user's task of 2026-11-05, section 6.1): how evaluation episodes move, turn by turn, measured in
true remaining actions.

Every episode of an evaluation result is replayed without rendering in the evaluation's env, as reward.fail_end_steps
replays failed ones (env.reset on the task's line, one env.step per recorded answer, checked against the record), and
the pose before the first turn and after every turn is looked up in the task's depth-12 table: its remaining number of
actions ("d"; the start's is d*).  A number counts only when the table gives it ("d" when the pose's cell is in the
table with a reachable success, "near" when a neighbouring cell gives it, as fail_end_steps); otherwise the turn is
left out of the rates that need it.  Per result (one checkpoint on one evaluation set):

* error rate: the share of turns after which the remaining number grew;
* recovery rate: the share of errors after which, within the next two turns, the remaining number came back to its
  value before the error or below (success counts); errors with fewer than two turns left in their episode are
  counted apart and left out of the rate unless the episode recovered in the turn it had;
* stall rate: from the second turn on, the share of turns that left the remaining number unchanged;
* the success rate of episodes with at least one error and of episodes without;
* collision rate: blocked moves per action (the record's collisions over all actions) and per move action;
* turn share: turn_left / turn_right among all actions;
* efficiency of successful episodes: actions taken / d* (mean and median), and the record's actions to success / d*.

    python -m reward.turn_metrics --result LABEL=DIR [--result ...] --tables DIR [--tables DIR ...] --out FILE.json
        [--tasks FILE] [--workers 12]

LABEL names the run and checkpoint as RUN@STEP (e.g. rw_s0@100); a result folder holds results.jsonl, model.json and
done.json, its tasks file is --tasks or DIR/../tasks.jsonl.  Several --tables folders are searched in order (the
screened test set: eval705, then screened59).  FILE.json (per result: the rates and their counts) and FILE.md.  Many
results: a one-card sco job (reward.sco_job), not the login node.
"""

import argparse
import json
import statistics
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TEMPLATE = Path(__file__).resolve().parents[1] / "data_gen" / "interiorgs_v10" / "configs" / "rl_env_v10_S0.yaml"
GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"
TURN_ACTIONS = ("turn_left", "turn_right")


def _lookup(tables: Sequence[Any], k: int) -> Optional[int]:
    """The remaining number of a contract key from the first table that knows it (cell, then its neighbours)."""
    from data_gen.interiorgs_v10.distance_table_near import neighbour_distance

    for table in tables:
        keys = table["keys"]
        i = int(np.searchsorted(keys, k))
        if i < len(keys) and int(keys[i]) == k:
            d = int(table["dist"][i])
            if d >= 0:
                return d
    for table in tables:
        d = neighbour_distance(table, k)
        if d is not None:
            return d
    return None


def worker(args):
    """Replay one batch of a result's episodes; per episode the remaining numbers (start, then after each turn)."""
    tasks_path, template, gs_root, model_json, table_dirs, items = args
    import yaml

    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    model = json.loads(Path(model_json).read_text())
    cfg = dict(yaml.safe_load(Path(template).read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(tasks_path), gs_root=gs_root, render_backend=None, r1_legacy_success=False)
    if model.get("prompt_format"):
        cfg["prompt_format"] = model["prompt_format"]
    cfg.update(model.get("env_switches") or {})
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items()
                                                     if k in ActiveSpatialEnvConfig.__dataclass_fields__}))

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render
    caches = [DT.TableCache(Path(t)) for t in table_dirs]
    position = {}
    for i, line in enumerate(l for l in Path(tasks_path).read_text().splitlines() if l.strip()):
        position.setdefault(json.loads(line)["task_id"], i)
    out = []
    for r in items:
        try:
            tables = []
            for c in caches:
                try:
                    tables.append(c.get(r["task_id"]))
                except Exception:  # not in this folder
                    pass
            if not tables:
                out.append({"task_id": r["task_id"], "skipped": "no table"})
                continue
            env.reset(seed=position[r["task_id"]])
            assert env.current_item["task_id"] == r["task_id"], (env.current_item["task_id"], r["task_id"])
            key = lambda: int(contract_keys(*_split(env.view_engine.get_pose()))[0])  # noqa: E731
            ds = [_lookup(tables, key())]
            met, blocked = [], []
            for t in r["trajectory"]:
                _, _, done, info = env.step(t["answer"])
                tm = (info.get("metrics") or {}).get("turn_metrics") or {}
                blocked.append(int(tm.get("collision_count", 0) or 0))
                m = env._calculate_canonical_metric()
                met.append(bool(m and m.get("success")))
                ds.append(0 if met[-1] else _lookup(tables, key()))
                if done:
                    break
            same = (met == [bool(t["met"]) for t in r["trajectory"]][:len(met)] and
                    blocked == [int(t["collisions"]) for t in r["trajectory"]][:len(blocked)])
            actions = [a for t in r["trajectory"] for a in (t.get("actions") or [])]
            out.append({"task_id": r["task_id"], "task_type": r["task_type"], "d_star": int(r["d_star"]),
                        "success": bool(r["success"]), "replay_same": bool(same), "d": ds,
                        "actions": len(actions), "turn_actions": sum(a in TURN_ACTIONS for a in actions),
                        "move_actions": sum(a.startswith("move_") for a in actions),
                        "collisions": int(r.get("collisions") or 0),
                        "actions_to_success": r.get("actions_to_success")})
        except Exception as exc:
            out.append({"task_id": r.get("task_id"), "error": f"{type(exc).__name__}: {exc}"})
    return out


def summarize(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [e for e in eps if "d" in e]
    turns = errors = stall_turns = stalls = 0
    recovered = recoverable = short_errors = 0
    with_error = []
    for e in ok:
        d = e["d"]
        had = False
        for t in range(1, len(d)):
            if d[t] is None or d[t - 1] is None:
                continue
            turns += 1
            if d[t] > d[t - 1]:
                errors += 1
                had = True
                later = [x for x in d[t + 1:t + 3] if x is not None]
                back = any(x <= d[t - 1] for x in later)
                if len(d) - 1 - t >= 2 or back:
                    recoverable += 1
                    recovered += back
                else:
                    short_errors += 1
            if t >= 2:
                stall_turns += 1
                stalls += d[t] == d[t - 1]
        with_error.append(had)
    rate = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    succ = [e for e in ok if e["success"]]
    eff = [e["actions"] / e["d_star"] for e in succ if e["d_star"] > 0]
    eff_hit = [e["actions_to_success"] / e["d_star"] for e in succ if e["d_star"] > 0 and e.get("actions_to_success")]
    actions = sum(e["actions"] for e in ok)
    moves = sum(e["move_actions"] for e in ok)
    err_eps = [e for e, h in zip(ok, with_error) if h]
    clean_eps = [e for e, h in zip(ok, with_error) if not h]
    return {"episodes": len(eps), "replayed": len(ok), "errors_in_replay": len(eps) - len(ok),
            "replay_same": sum(e["replay_same"] for e in ok), "success_rate": rate(len(succ), len(ok)),
            "turns_known": turns, "error_rate": rate(errors, turns), "errors": errors,
            "recovery_rate": rate(recovered, recoverable), "errors_recoverable": recoverable,
            "errors_too_late": short_errors, "stall_rate": rate(stalls, stall_turns), "stall_turns": stall_turns,
            "episodes_with_error": len(err_eps),
            "success_with_error": rate(sum(e["success"] for e in err_eps), len(err_eps)),
            "success_without_error": rate(sum(e["success"] for e in clean_eps), len(clean_eps)),
            "collision_per_action": rate(sum(e["collisions"] for e in ok), actions),
            "collision_per_move": rate(sum(e["collisions"] for e in ok), moves),
            "turn_share": rate(sum(e["turn_actions"] for e in ok), actions),
            "success_actions_over_dstar_mean": round(statistics.mean(eff), 3) if eff else None,
            "success_actions_over_dstar_median": round(statistics.median(eff), 3) if eff else None,
            "success_to_success_over_dstar_mean": round(statistics.mean(eff_hit), 3) if eff_hit else None,
            "unknown_d_turns": sum(x is None for e in ok for x in e["d"][1:])}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--result", action="append", required=True, help="LABEL=DIR (LABEL as RUN@STEP)")
    ap.add_argument("--tables", action="append", required=True, help="table folders, searched in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--env-template", default=str(TEMPLATE))
    ap.add_argument("--gs-root", default=GS_ROOT)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    jobs, labels = [], []
    for spec in a.result:
        label, _, res = spec.partition("=")
        res = Path(res)
        if not (res / "done.json").is_file():
            raise SystemExit(f"{res}: no done.json")
        tasks = Path(a.tasks) if a.tasks else res.parent / "tasks.jsonl"
        rows = [json.loads(l) for l in (res / "results.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if "error" not in r]
        for k in range(0, len(rows), 40):
            jobs.append((str(tasks), a.env_template, a.gs_root, str(res / "model.json"), list(a.tables), rows[k:k + 40]))
            labels.append(f"{label}|{res.parent.name}")
    per = defaultdict(list)
    with Pool(max(1, a.workers)) as pool:
        for lab, part in zip(labels, pool.imap(worker, jobs)):
            per[lab] += part
    out = {lab: summarize(v) for lab, v in sorted(per.items())}
    merged_eps = defaultdict(list)  # one checkpoint over its sets (validation + test)
    for lab, v in per.items():
        merged_eps[lab.split("|")[0]] += v
    merged = {lab: summarize(v) for lab, v in sorted(merged_eps.items())}
    Path(a.out).write_text(json.dumps({"tables": a.tables, "results": out, "merged": merged,
                                       "episodes": dict(sorted(per.items()))}, indent=1))
    cols = ("success_rate", "error_rate", "recovery_rate", "stall_rate", "success_with_error", "success_without_error",
            "collision_per_action", "turn_share", "success_actions_over_dstar_mean")
    L = ["| 结果 | 局数 | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 2)]
    for lab, s in list(merged.items()) + list(out.items()):  # merged sets first, then each set
        L.append(f"| {lab} | {s['replayed']} | " + " | ".join("—" if s[c] is None else f"{s[c]}" for c in cols) + " |")
    Path(a.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
