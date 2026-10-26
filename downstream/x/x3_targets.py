#!/usr/bin/env python3
"""X3 (the user's decision 5 of 2026-10-26): which target the robot actually worked on in the 9 spatial tasks.

The E1 final checkpoints (200k steps) replay the spatial episodes of the fixed evaluation (downstream/x/x3_plan.json:
the same seeds, kitchens, horizons and exclusion list as eval_d6.json, 447 episodes) with a read-only recorder:
closed_loop.py calls start() right after env.reset() and end() after the episode's last step and keeps both raw
records in the episode row (key "end_state").  Recording only reads the simulator (joint positions, body positions,
contacts, the fixtures' own state getters); it never steps, resets or calls mj_forward, and end() runs after the
episode is over, so the episode runs as it would without it.  Which candidate the robot chose is decided afterwards
from the two records (decide()), so a threshold can change without running anything again:

    OpenDrawer           the door or drawer whose opening grew most (by more than OPEN_MIN of its range); target:
                         the instructed drawer ("Open the left drawer")
    TurnOffStove         the knob turned most (by more than KNOB_MIN rad); target: the instructed burner's knob
    SlideDishwasherRack  the direction the top rack moved (out / in, by more than RACK_MIN); target: the instructed one
    CoffeeSetupMug       where the mug ended: under the dispenser / in the gripper / on a counter / elsewhere
                         (untouched when it moved less than MOVED_M and is not held); target: under the dispenser
    StackBowlsCabinet    how many bowls ended inside the open cabinet; target: both.  The stacking order is kept
                         apart (RoboCasa's success accepts either order although the instruction asks for the
                         smaller bowl on top)
    NavigateKitchen      the fixture whose robot placement point (RoboCasa's compute_robot_base_placement_pose, the
                         point the task's success is measured from) is nearest the final base position, if within
                         NEAR_M; target: the instructed fixture
    DeliverStraw, PrepareCoffee, SearingMeat: recorded the same way (drawer opened, mug place, pan burner) but not in
                         the main table: all four starts succeed in none of their episodes at 200k.

    python downstream/x/x3_targets.py analyze --x3 X3_DIR --experiments EXPERIMENTS --out X3_DIR/results

`analyze` first checks that every replayed episode ended as in the original E1 evaluation (success and steps, and
the initial-state fingerprint), then writes results.json / results.md: per start and task the share of episodes
that chose the target, that chose it and succeeded, the failures split into wrong target / nothing chosen / right
target not finished (Wilson 95% intervals), and per-episode McNemar against B.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x0_qa import mcnemar_exact, wilson  # noqa: E402

MAIN_TASKS = ("OpenDrawer", "TurnOffStove", "SlideDishwasherRack", "CoffeeSetupMug", "StackBowlsCabinet",
              "NavigateKitchen")
RECORDED_ONLY = ("DeliverStraw", "PrepareCoffee", "SearingMeat")
SPATIAL = MAIN_TASKS + RECORDED_ONLY
STARTS = ("B", "S", "RS", "RD")

OPEN_MIN = 0.05   # a door or drawer was opened when its opening grew by more than 5% of its range
KNOB_MIN = 0.10   # rad (about 6 degrees): a knob was turned when it moved more than this
RACK_MIN = 0.05   # the dishwasher rack moved when its position changed by more than 5% of its range
MOVED_M = 0.02    # m: the mug was moved when its position changed by more than this
NEAR_M = 0.50     # m: NavigateKitchen, the robot is at a fixture when its base is this close to its placement point
NAV_CLASSES = ("CoffeeMachine", "Toaster", "ToasterOven", "Stove", "Stovetop", "OpenCabinet", "Microwave", "Sink",
               "Hood", "Oven", "FridgeFrenchDoor", "FridgeSideBySide", "FridgeBottomFreezer", "Dishwasher",
               "ElectricKettle", "StandMixer")  # NavigateKitchen's target classes (kitchen_navigate.py)


# ---------------------------------------------------------------- recording (RoboCasa simulator Python, read only)

def kitchen_of(env):
    inner = getattr(env, "unwrapped", env)
    return getattr(inner, "env", inner)


def _r(x, digits: int = 4) -> float:
    return round(float(x), digits)


def base_pose(k) -> list:
    """[x, y, yaw] of the mobile base (yaw as kitchen_navigate.py computes it)."""
    import numpy as np
    from robosuite.utils import transform_utils as T

    bid = k.sim.model.body_name2id("mobilebase0_base")
    pos = np.array(k.sim.data.body_xpos[bid])
    yaw = T.mat2euler(np.array(k.sim.data.body_xmat[bid]).reshape((3, 3)))[2]
    return [_r(pos[0]), _r(pos[1]), _r(yaw)]


def doors(k) -> dict:
    """Every fixture with a door state (drawers, hinge cabinets): its largest opening (0..1) and its x, y."""
    out = {}
    for name, f in k.fixtures.items():
        get = getattr(f, "get_door_state", None)
        if get is None:
            continue
        try:
            state = get(env=k)
        except Exception:
            continue
        if state:
            out[name] = {"type": type(f).__name__, "open": _r(max(float(v) for v in state.values())),
                         "xy": [_r(f.pos[0], 3), _r(f.pos[1], 3)]}
    return out


def knobs(k) -> dict:
    return {loc: _r(v) for loc, v in k.stove.get_knobs_state(env=k).items()}


def rack(k) -> float:
    joint = k.dishwasher._joint_names["rack"]
    return _r(k.dishwasher.get_joint_state(k, [joint])[joint])


def obj_xyz(k, name: str) -> list:
    return [_r(x) for x in k.sim.data.body_xpos[k.obj_body_id[name]]]


def mug_end(k, name: str = "obj") -> dict:
    """Where the mug is: under the dispenser (RoboCasa's own check), its distance to the dispenser's place site, held,
    touching a counter or the machine."""
    import numpy as np
    from robocasa.utils import object_utils as OU

    cm = k.coffee_machine
    site = k.sim.data.site_xpos[k.sim.model.site_name2id(f"{cm.naming_prefix}receptacle_place_site")]
    pos = np.array(k.sim.data.body_xpos[k.obj_body_id[name]])
    return {"xyz": [_r(x) for x in pos], "under_dispenser": bool(cm.check_receptacle_placement_for_pouring(k, name)),
            "xy_to_dispenser": _r(np.linalg.norm(pos[:2] - site[:2])), "dz_to_dispenser": _r(pos[2] - site[2]),
            "grasped": bool(OU.check_obj_grasped(k, name)), "on_counter": bool(OU.check_obj_any_counter_contact(k, name)),
            "on_machine": bool(OU.check_obj_fixture_contact(k, name, cm))}


def nav_candidates(k) -> list:
    """NavigateKitchen's possible targets (its target classes that occur once in the kitchen) and its start fixture."""
    names = [type(f).__name__ for f in k.fixtures.values()]
    out = [f for f in k.fixtures.values() if type(f).__name__ in NAV_CLASSES and names.count(type(f).__name__) == 1]
    if k.src_fixture not in out:
        out.append(k.src_fixture)
    return out


def nav_points(k, base) -> dict:
    """Per candidate fixture: its robot placement point and the final base's distance to it (and the heading cosine)."""
    from robocasa.utils import env_utils as EnvUtils

    out = {}
    for f in nav_candidates(k):
        try:
            pos, ori = EnvUtils.compute_robot_base_placement_pose(k, f)
        except Exception as exc:
            out[f.name] = {"type": type(f).__name__, "error": f"{type(exc).__name__}: {exc}"[:120]}
            continue
        out[f.name] = {"type": type(f).__name__, "point": [_r(pos[0]), _r(pos[1]), _r(ori[2])],
                       "dist": _r(math.hypot(base[0] - pos[0], base[1] - pos[1])),
                       "heading_cos": _r(math.cos(ori[2] - base[2]))}
    return out


def start(task: str, env) -> dict:
    """The candidates' state right after reset (read only)."""
    k = kitchen_of(env)
    rec = {"base": base_pose(k)}
    if task in ("OpenDrawer", "DeliverStraw"):
        rec.update(target=k.drawer.name, doors=doors(k))
        if task == "OpenDrawer":
            rec["side"] = k.drawer_side
    elif task in ("TurnOffStove", "SearingMeat"):
        rec.update(target=k.knob, knobs=knobs(k))
    elif task == "SlideDishwasherRack":
        rec.update(target="out" if k.should_pull else "in", rack=rack(k))
    elif task in ("CoffeeSetupMug", "PrepareCoffee"):
        rec.update(target="under_dispenser", mug=obj_xyz(k, "obj"))
    elif task == "StackBowlsCabinet":
        rec.update(target=k.cabinet.name, radius={b: _r(k.objects[b].horizontal_radius) for b in ("bowl1", "bowl2")})
    elif task == "NavigateKitchen":
        rec.update(target=k.target_fixture.name, src=k.src_fixture.name)
    return rec


def end(task: str, env) -> dict:
    """The candidates' state after the episode's last step (read only)."""
    from robocasa.utils import object_utils as OU

    k = kitchen_of(env)
    base = base_pose(k)
    rec = {"base": base}
    if task in ("OpenDrawer", "DeliverStraw"):
        rec["doors"] = doors(k)
        if task == "DeliverStraw":
            rec["straw"] = {"grasped": bool(OU.check_obj_grasped(k, "straw")),
                            "in_glass": bool(OU.check_obj_in_receptacle(k, "straw", "glass_cup", th=0.5)),
                            "in_drawer": bool(OU.obj_inside_of(k, "straw", k.drawer, partial_check=True))}
    elif task in ("TurnOffStove", "SearingMeat"):
        rec["knobs"] = knobs(k)
        if task == "SearingMeat":
            rec["pan_burner"] = k.stove.check_obj_location_on_stove(k, "pan", threshold=0.15)
            rec["meat_in_pan"] = bool(OU.check_obj_in_receptacle(k, "meat", "pan", th=0.07))
    elif task == "SlideDishwasherRack":
        rec["rack"] = rack(k)
    elif task in ("CoffeeSetupMug", "PrepareCoffee"):
        rec["mug"] = mug_end(k)
        if task == "PrepareCoffee":
            rec["turned_on"] = bool(k.coffee_machine._turned_on)
    elif task == "StackBowlsCabinet":
        rec["bowls"] = {"bowl1_in_cabinet": bool(OU.obj_inside_of(k, "bowl1", k.cabinet)),
                        "bowl2_in_cabinet": bool(OU.obj_inside_of(k, "bowl2", k.cabinet)),
                        "bowl1_in_bowl2": bool(OU.check_obj_in_receptacle(k, "bowl1", "bowl2")),
                        "bowl2_in_bowl1": bool(OU.check_obj_in_receptacle(k, "bowl2", "bowl1")),
                        "grasped": [b for b in ("bowl1", "bowl2") if OU.check_obj_grasped(k, b)]}
    elif task == "NavigateKitchen":
        rec["fixtures"] = nav_points(k, base)
    return rec


# ---------------------------------------------------------------- decisions (plain Python on the two records)

def angle_change(a: float, b: float) -> float:
    """b - a on the circle, in (-pi, pi]."""
    d = (b - a) % (2 * math.pi)
    return d - 2 * math.pi if d > math.pi else d


def lateral(base, xy) -> float:
    """Left (+) / right (-) offset of point xy from a base pose [x, y, yaw]."""
    dx, dy = xy[0] - base[0], xy[1] - base[1]
    return -math.sin(base[2]) * dx + math.cos(base[2]) * dy


def most_changed(before: dict, after: dict, change, minimum: float):
    """The key whose value changed most (by change(before, after)), or None when no change exceeds minimum."""
    deltas = {k: change(before[k], after[k]) for k in before if k in after}
    if not deltas:
        return None, {}
    key = max(deltas, key=lambda k: (deltas[k], k))
    return (key if deltas[key] > minimum else None), {k: round(v, 4) for k, v in deltas.items()}


def decide(task: str, start_rec: dict, end_rec: dict) -> dict:
    """{"target", "chosen" (None: nothing chosen), "correct"} plus what the choice was read from."""
    out = {"target": start_rec.get("target")}
    if task in ("OpenDrawer", "DeliverStraw"):
        before = {n: d["open"] for n, d in start_rec["doors"].items()}
        after = {n: d["open"] for n, d in end_rec["doors"].items()}
        chosen, deltas = most_changed(before, after, lambda a, b: b - a, OPEN_MIN)
        out.update(chosen=chosen, opened=deltas)
        if chosen is not None:
            d = start_rec["doors"][chosen]
            out.update(chosen_type=d["type"],
                       chosen_side="left" if lateral(start_rec["base"], d["xy"]) > 0 else "right")
        if task == "DeliverStraw":
            out["straw"] = end_rec.get("straw")
    elif task in ("TurnOffStove", "SearingMeat"):
        chosen, deltas = most_changed(start_rec["knobs"], end_rec["knobs"], lambda a, b: abs(angle_change(a, b)),
                                      KNOB_MIN)
        out.update(chosen=chosen, turned=deltas)
        if task == "SearingMeat":
            out.update(pan_burner=end_rec.get("pan_burner"), meat_in_pan=end_rec.get("meat_in_pan"))
    elif task == "SlideDishwasherRack":
        d = end_rec["rack"] - start_rec["rack"]
        out.update(chosen="out" if d > RACK_MIN else "in" if d < -RACK_MIN else None, rack_change=round(d, 4))
    elif task in ("CoffeeSetupMug", "PrepareCoffee"):
        m = end_rec["mug"]
        moved = math.dist(start_rec["mug"], m["xyz"])
        if m["under_dispenser"]:
            place = "under_dispenser"
        elif m["grasped"]:
            place = "in_gripper"
        elif moved <= MOVED_M:
            place = None  # untouched
        elif m["on_counter"]:
            place = "on_counter"
        else:
            place = "elsewhere"
        out.update(chosen=place, moved=round(moved, 4), xy_to_dispenser=m["xy_to_dispenser"])
        if task == "PrepareCoffee":
            out["turned_on"] = end_rec.get("turned_on")
    elif task == "StackBowlsCabinet":
        b = end_rec["bowls"]
        inside = int(b["bowl1_in_cabinet"]) + int(b["bowl2_in_cabinet"])
        r = start_rec["radius"]
        smaller, larger = ("bowl1", "bowl2") if r["bowl1"] <= r["bowl2"] else ("bowl2", "bowl1")
        if b[f"{smaller}_in_{larger}"]:
            order = "smaller_on_top"
        elif b[f"{larger}_in_{smaller}"]:
            order = "larger_on_top"
        else:
            order = "not_stacked"
        out.update(chosen=start_rec["target"] if inside == 2 else None, bowls_in_cabinet=inside, order=order)
    elif task == "NavigateKitchen":
        fx = {n: f for n, f in end_rec["fixtures"].items() if "dist" in f}
        near = min(fx, key=lambda n: (fx[n]["dist"], n)) if fx else None
        out.update(chosen=near if near is not None and fx[near]["dist"] <= NEAR_M else None, nearest=near,
                   nearest_dist=fx[near]["dist"] if near else None,
                   target_dist=fx.get(start_rec["target"], {}).get("dist"), at_start_fixture=near == start_rec["src"])
    else:
        raise ValueError(f"not a spatial task: {task}")
    out["correct"] = out["chosen"] is not None and out["chosen"] == out["target"]
    return out


# ---------------------------------------------------------------- analysis

def read_rows(path: Path) -> dict:
    """{(task, seed): row} of the complete episodes of an episodes.jsonl (a later row of a key wins: reruns)."""
    rows = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("status") == "complete":
            rows[(r["task"], int(r["seed"]))] = r
    return rows


def identity(replay: dict, original: dict) -> dict:
    """Episodes whose replay did not end as in the original evaluation (success, steps, initial-state fingerprint)."""
    diff, missing = [], []
    for key, r in sorted(replay.items()):
        o = original.get(key)
        if o is None:
            missing.append(list(key))
            continue
        what = [f for f in ("success", "steps") if r.get(f) != o.get(f)]
        if (r.get("init") or {}).get("qpos_sha") != (o.get("init") or {}).get("qpos_sha"):
            what.append("init")
        if what:
            diff.append({"task": key[0], "seed": key[1], "differs": what, "replay": [r.get("success"), r.get("steps")],
                         "original": [o.get("success"), o.get("steps")]})
    return {"compared": len(replay) - len(missing), "same": len(replay) - len(missing) - len(diff),
            "different": diff, "not_in_original": missing}


def episode_table(rows: dict) -> dict:
    """{(task, seed): decision + success} for the rows that carry both records."""
    out = {}
    for key, r in rows.items():
        es = r.get("end_state") or {}
        if key[0] in SPATIAL and "start" in es and "end" in es:
            try:
                out[key] = dict(decide(key[0], es["start"], es["end"]), success=bool(r["success"]))
            except (KeyError, TypeError) as exc:
                out[key] = {"error": f"{type(exc).__name__}: {exc}", "success": bool(r["success"])}
    return out


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [lo, hi]}


def summarize(table: dict, tasks=MAIN_TASKS) -> dict:
    """Per task and pooled over `tasks`: chose the target, chose it and succeeded, and the failures split."""
    out = {}
    for name, keys in [(t, [k for k in table if k[0] == t]) for t in tasks] + [("all", [k for k in table if k[0] in tasks])]:
        rows = [table[k] for k in keys if "error" not in table[k]]
        fails = [r for r in rows if not r["success"]]
        out[name] = {"episodes": len(rows), "unreadable": len(keys) - len(rows),
                     "correct": rate(sum(r["correct"] for r in rows), len(rows)),
                     "correct_and_success": rate(sum(r["correct"] and r["success"] for r in rows), len(rows)),
                     "success": rate(sum(r["success"] for r in rows), len(rows)),
                     "failures": {"n": len(fails),
                                  "wrong_target": sum(r["chosen"] is not None and not r["correct"] for r in fails),
                                  "nothing_chosen": sum(r["chosen"] is None for r in fails),
                                  "right_target_not_finished": sum(r["correct"] for r in fails)},
                     "success_without_target": sum(r["success"] and not r["correct"] for r in rows)}
    return out


def paired(table: dict, ref: dict, tasks=MAIN_TASKS) -> dict:
    """McNemar on "chose the target", episode by episode, against the reference start."""
    keys = [k for k in table if k[0] in tasks and k in ref and "error" not in table[k] and "error" not in ref[k]]
    b = sum(table[k]["correct"] and not ref[k]["correct"] for k in keys)
    c = sum(ref[k]["correct"] and not table[k]["correct"] for k in keys)
    return {"pairs": len(keys), "only_this": b, "only_ref": c, "p": round(mcnemar_exact(b, c), 4)}


def analyze(x3: Path, experiments: Path, out: Path) -> dict:
    res = {"identity": {}, "starts": {}, "recorded_only": {}, "vs_B": {}}
    tables = {}
    for s in STARTS:
        path = x3 / s / "out/episodes/episodes.jsonl"
        if not path.is_file():
            continue
        replay = read_rows(path)
        original = read_rows(experiments / f"DS-{s}-100-s1/evals/steps_200000/out/episodes/episodes.jsonl")
        res["identity"][s] = identity(replay, original)
        tables[s] = episode_table(replay)
        res["starts"][s] = summarize(tables[s])
        res["recorded_only"][s] = summarize(tables[s], RECORDED_ONLY)
    for s in tables:
        if s != "B" and "B" in tables:
            res["vs_B"][s] = paired(tables[s], tables["B"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with open(out / "episodes.jsonl", "w", encoding="utf-8") as f:
        for s, table in tables.items():
            for (t, seed), d in sorted(table.items()):
                f.write(json.dumps(dict(d, start=s, task=t, seed=seed), ensure_ascii=False) + "\n")
    (out / "results.md").write_text(markdown(res), encoding="utf-8")
    return res


def _pct(r: dict) -> str:
    if not r["n"]:
        return "—"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}%（{100 * lo:.1f}–{100 * hi:.1f}）"


def markdown(res: dict) -> str:
    lines = ["# X3 落点准确率（自动生成）", "", "## 核对：重放和原评测逐回合相同吗", "",
             "| 起点 | 比较的回合 | 相同 | 不同 | 原评测里没有 |", "| --- | ---: | ---: | ---: | ---: |"]
    for s, i in res["identity"].items():
        lines.append(f"| {s} | {i['compared']} | {i['same']} | {len(i['different'])} | {len(i['not_in_original'])} |")
    lines += ["", "## 选对目标（6 个主表任务合计；括号里是 95% 区间）", "",
              "| 起点 | 回合 | 选对目标 | 选对且成功 | 成功 | 失败里选错 | 失败里没选 | 失败里选对没做成 | 和 B 配对 p |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s, st in res["starts"].items():
        a = st["all"]
        p = res["vs_B"].get(s, {}).get("p", "—")
        lines.append(f"| {s} | {a['episodes']} | {_pct(a['correct'])} | {_pct(a['correct_and_success'])} | "
                     f"{_pct(a['success'])} | {a['failures']['wrong_target']} | {a['failures']['nothing_chosen']} | "
                     f"{a['failures']['right_target_not_finished']} | {p} |")
    lines += ["", "## 逐任务：选对目标的比例", "", "| 任务 | " + " | ".join(res["starts"]) + " |",
              "| --- |" + " ---: |" * len(res["starts"])]
    for t in MAIN_TASKS:
        lines.append(f"| {t} | " + " | ".join(_pct(st[t]["correct"]) for st in res["starts"].values()) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--x3", type=Path, required=True, help="downstream_runs/x/X3 (one folder per start)")
    a.add_argument("--experiments", type=Path, required=True, help="downstream_runs/experiments (the E1 evaluations)")
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = analyze(args.x3, args.experiments, args.out)
    print(json.dumps({s: {"identity_different": len(i["different"])} for s, i in res["identity"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
