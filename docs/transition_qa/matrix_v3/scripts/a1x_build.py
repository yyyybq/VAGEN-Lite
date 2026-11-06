"""Analysis 1, extension (the user's choice 甲 of 2026-11-06, exploratory): questions whose right answer can be "Yes".

The same failed test episodes as analysis 1 (D seed 20261019's S0 and V4 pre-registered checkpoints, m3ds1s0_300 and
m3ds1d_300, eval_test_v2_screened).  At every turn's frame, one question per primitive action, in P1's goal wording:

* goal_after_one: "Task: <task>\nIf the camera <the action, once> from the current view, will the task's requirement
  be met? Answer Yes or No."

The answers come from the evaluation's env replayed without rendering (as a1_build.py): env A replays the recorded
episode; for each turn and action, env B is reset on the task, put at the turn's start pose, steps that one action
(collisions as in the env) and is checked with the canonical success metric; the remaining distance d before and
after comes from the task's distance table.  Near the goal some actions are right ("Yes"), so the answers can tell
"knows which action meets the task but chose another" from "answers No whatever".  Writes OUT/questions/<model>.jsonl
(probe rows for `qa_probe run`, the facts under "a1x") and OUT/build.json.

    python a1x_build.py OUT [--workers 3]
"""
import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
sys.path.insert(0, str(ROOT))
M3 = ROOT / "exps" / "matrix_v3"
TEST = M3 / "eval_test_v2_screened"
TABLES = "/mnt/umm/users/yinbaiqiao/reward_runs/tables/eval_test_v2_screened"
TEMPLATE = ROOT / "data_gen" / "interiorgs_v10" / "configs" / "rl_env_v10_S0.yaml"
GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"
MODELS = ("m3ds1s0_300", "m3ds1d_300")
ACTIONS = ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right")


def make_env(model_json):
    import yaml

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    model = json.loads(Path(model_json).read_text())
    cfg = dict(yaml.safe_load(TEMPLATE.read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(TEST / "tasks.jsonl"), gs_root=GS_ROOT, render_backend=None, r1_legacy_success=False)
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
    return env


def worker(args):
    model_json, items = args
    import numpy as np

    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys
    from data_gen.interiorgs_v10.distance_table_near import neighbour_distance

    env, probe_env = make_env(model_json), make_env(model_json)
    cache = DT.TableCache(TABLES)
    position = {}
    for i, line in enumerate(l for l in (TEST / "tasks.jsonl").read_text().splitlines() if l.strip()):
        position.setdefault(json.loads(line)["task_id"], i)

    def met(e) -> bool:
        m = e._calculate_canonical_metric()
        return bool(m and m.get("success"))

    def dist(e, table) -> tuple:
        k = int(contract_keys(*_split(e.view_engine.get_pose()))[0])
        keys = table["keys"]
        i = int(np.searchsorted(keys, k))
        if i < len(keys) and int(keys[i]) == k:
            d = int(table["dist"][i])
            return (d, "d") if d >= 0 else (None, "unreached")
        nd = neighbour_distance(table, k)
        return (nd, "near") if nd is not None else (None, "missing")

    out = []
    for r in items:
        env.reset(seed=position[r["task_id"]])
        assert env.current_item["task_id"] == r["task_id"], (env.current_item["task_id"], r["task_id"])
        table = cache.get(r["task_id"])
        turns = []
        for t in r["trajectory"]:
            pose = env.view_engine.get_pose()
            s0, (d0, k0) = met(env), dist(env, table)
            singles = {}
            for a in ACTIONS:  # one action from this turn's start pose, in a second env
                probe_env.reset(seed=position[r["task_id"]])
                probe_env.view_engine.reset(pose.copy())
                _, _, _, pinfo = probe_env.step(f"<action>{a}|</action>")
                ptm = (pinfo.get("metrics") or {}).get("turn_metrics") or {}
                d1, k1 = dist(probe_env, table)
                singles[a] = {"success": met(probe_env), "blocked": int(ptm.get("collision_count", 0) or 0),
                              "d_after": d1, "kind_after": k1}
            _, _, done, info = env.step(t["answer"])
            tm = (info.get("metrics") or {}).get("turn_metrics") or {}
            turns.append({"success_before": s0, "d_before": d0, "kind_before": k0, "success_after": met(env),
                          "blocked": int(tm.get("collision_count", 0) or 0), "singles": singles})
            if done:
                break
        same = ([x["success_after"] for x in turns] == [bool(t["met"]) for t in r["trajectory"]] and
                [x["blocked"] for x in turns] == [int(t["collisions"]) for t in r["trajectory"]])
        out.append({"index": r["index"], "task_id": r["task_id"], "turns": turns, "replay_same": bool(same)})
    return out


def main() -> int:
    from data_gen.transition_qa.questions import ANSWER_YN, action_text

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "questions").mkdir(parents=True, exist_ok=True)
    build = {"tables": TABLES, "tasks": str(TEST / "tasks.jsonl"), "actions": list(ACTIONS), "models": {}}
    for name in MODELS:
        res = TEST / name
        rows = [json.loads(l) for l in (res / "results.jsonl").read_text().splitlines() if l.strip()]
        fails = [r for r in rows if not r["success"]]
        chunks = [(str(res / "model.json"), fails[i::a.workers]) for i in range(a.workers)]
        with Pool(a.workers) as pool:
            replay = {x["index"]: x for part in pool.map(worker, chunks) for x in part}
        probe, stats = [], collections.Counter()
        for r in fails:
            rp = replay[r["index"]]
            stats["episodes"] += 1
            stats["replay_same"] += rp["replay_same"]
            desc = str(r["task_description"]).strip()
            for i, (t, f) in enumerate(zip(r["trajectory"], rp["turns"])):
                chosen = list(t.get("actions") or [])
                stats["turns"] += 1
                right = [x for x in ACTIONS if f["singles"][x]["success"]]
                stats["turns_with_a_yes"] += bool(right)
                for x in ACTIONS:
                    s = f["singles"][x]
                    text = (f"Task: {desc}\nIf the camera {action_text(x, 1)} from the current view, will the task's "
                            f"requirement be met? {ANSWER_YN}")
                    facts = {"model": name, "index": r["index"], "turn": i, "action": x, "chosen": chosen,
                             "valid": bool(t.get("valid")), "first_chosen": bool(chosen) and chosen[0] == x,
                             "success_before": f["success_before"], "d_before": f["d_before"], **s,
                             "right_actions": right, "replay_same": rp["replay_same"]}
                    yes = bool(s["success"])
                    probe.append({"messages": [{"role": "user", "content": "<image>\n" + text},
                                               {"role": "assistant", "content": "Yes" if yes else "No"}],
                                  "images": [str(res / t["frame"])], "pool": "A1X", "family": r["task_type"],
                                  "task_id": r["task_id"], "scene": r["scene_id"], "split": "test", "state_id": f"t{i}",
                                  "sample_id": f"a1x_{name}_{r['index']:03d}_{i:02d}_{x}", "qclass": "forward",
                                  "qtype": "goal_after_one", "answer": "Yes" if yes else "No",
                                  "answer_value": "yes" if yes else "no", "a1x": facts})
                    stats[f"answer_{'yes' if yes else 'no'}"] += 1
                    stats["blocked_moves"] += s["blocked"] > 0
        path = out / "questions" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in probe), encoding="utf-8")
        build["models"][name] = {"result": str(res), "model": json.loads((res / "model.json").read_text()).get("model"),
                                 "probe": str(path), "questions": len(probe), **dict(stats)}
        print(name, json.dumps(build["models"][name], ensure_ascii=False), flush=True)
    (out / "build.json").write_text(json.dumps(build, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
