"""Analysis 1 of the user's instruction of 2026-11-06 (exploratory): "knows but does not act".

D seed 20261019's S0 and V4 pre-registered checkpoints (m3ds1s0_300, m3ds1d_300), their failed episodes on the
screened test set (eval_test_v2_screened): every turn's frame gets two questions in P1's goal wording
(data_gen.transition_qa.questions.build_text):

* goal_now: "Task: <task>\nIn the current view, is the task's requirement already met? Answer Yes or No."
* goal_after: "Task: <task>\nIf the camera <the actions this turn's answer chose> from the current view, will the
  task's requirement be met? Answer Yes or No." (the turn's primitive actions as runs of one action, joined by
  ", then "; a turn without a valid action gets no goal_after).

The answers come from the evaluation's env replayed without rendering, as reward.fail_end_steps does: env.reset on
the task, one env.step per recorded answer, the canonical success metric before and after each turn, checked
against the record (task met after the turn, blocked moves).  Each pose's remaining distance d (minimum actions to
success) comes from the task's distance table (reward_runs/tables/eval_test_v2_screened; a missing cell takes its
nearest neighbour's, as fail_end_steps).  Each checkpoint answers the questions of its own episodes.  Writes
OUT/<checkpoint>.jsonl (probe rows for `qa_probe run`, the turn's facts under "a1") and OUT/build.json.

    python a1_build.py OUT [--workers 2]
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


def runs_of(actions: list) -> list:
    out = []
    for a in actions:
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return out


def worker(args):
    model_json, items = args
    import numpy as np
    import yaml

    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys
    from data_gen.interiorgs_v10.distance_table_near import neighbour_distance
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
    cache = DT.TableCache(TABLES)
    position = {}
    for i, line in enumerate(l for l in (TEST / "tasks.jsonl").read_text().splitlines() if l.strip()):
        position.setdefault(json.loads(line)["task_id"], i)

    def met() -> bool:
        m = env._calculate_canonical_metric()
        return bool(m and m.get("success"))

    def dist(table) -> tuple:
        k = int(contract_keys(*_split(env.view_engine.get_pose()))[0])
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
            s0, (d0, k0) = met(), dist(table)
            _, _, done, info = env.step(t["answer"])
            tm = (info.get("metrics") or {}).get("turn_metrics") or {}
            s1, (d1, k1) = met(), dist(table)
            turns.append({"success_before": s0, "success_after": s1, "d_before": d0, "d_after": d1, "kind_before": k0,
                          "kind_after": k1, "blocked": int(tm.get("collision_count", 0) or 0)})
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
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build = {"tables": TABLES, "tasks": str(TEST / "tasks.jsonl"), "models": {}}
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
                base = {"images": [str(res / t["frame"])], "pool": "A1", "family": r["task_type"], "task_id": r["task_id"],
                        "scene": r["scene_id"], "split": "test", "state_id": f"t{i}"}
                facts = {"model": name, "index": r["index"], "turn": i, "actions": list(t.get("actions") or []),
                         "valid": bool(t.get("valid")), **f, "replay_same": rp["replay_same"]}
                qs = [("goal_now", "state", f"Task: {desc}\nIn the current view, is the task's requirement already met? {ANSWER_YN}",
                       f["success_before"])]
                if facts["actions"] and facts["valid"]:
                    phrase = ", then ".join(action_text(x, k) for x, k in runs_of(facts["actions"]))
                    qs.append(("goal_after", "forward", f"Task: {desc}\nIf the camera {phrase} from the current view, "
                                                        f"will the task's requirement be met? {ANSWER_YN}", f["success_after"]))
                    stats["mixed_turns"] += len(runs_of(facts["actions"])) > 1
                else:
                    stats["turns_without_action"] += 1
                stats["turns"] += 1
                for qtype, qclass, text, yes in qs:
                    probe.append({"messages": [{"role": "user", "content": "<image>\n" + text},
                                               {"role": "assistant", "content": "Yes" if yes else "No"}],
                                  "sample_id": f"a1_{name}_{r['index']:03d}_{i:02d}_{qtype}", "qclass": qclass, "qtype": qtype,
                                  "answer": "Yes" if yes else "No", "answer_value": "yes" if yes else "no", **base, "a1": facts})
                    stats[f"{qtype}:{'yes' if yes else 'no'}"] += 1
                if f["d_before"] is not None and f["d_after"] is not None:
                    stats["d_known"] += 1
                    stats["d_up" if f["d_after"] > f["d_before"] else "d_down" if f["d_after"] < f["d_before"] else "d_same"] += 1
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in probe), encoding="utf-8")
        build["models"][name] = {"result": str(res), "model": json.loads((res / "model.json").read_text()).get("model"),
                                 "probe": str(path), "questions": len(probe), **dict(stats)}
        print(name, json.dumps(build["models"][name], ensure_ascii=False))
    (out / "build.json").write_text(json.dumps(build, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
