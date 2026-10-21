"""Example scenes for the report, chosen by a fixed rule (never by eye).

A scene is (task, evaluation seed, training seed): every start meets the same initial state there (EVAL_PROTOCOL.md),
so their strips can be compared side by side.  Only scenes with a strip for every start count.
  * disagreements: for each start other than the reference, up to `per_kind` scenes where it succeeded and the
    reference failed, and as many where the reference succeeded and it failed (balanced, so the page shows both
    directions); spatial tasks first, then borderline, then the rest, each by task name and seed;
  * agreements: the first scene (same order) where every start succeeded and the first where every start failed.
"""
from __future__ import annotations

from . import stats

KLASS_RANK = {"spatial": 0, "borderline": 1, "other": 2}


def scenes(coll, cmp) -> dict:
    """scene -> {start: episode} for the scenes every start of the comparison has, with a strip each."""
    by_start = {s: {(x["task"], x["eval_seed"], x["train_seed"]): x
                    for x in stats.episodes_of(coll, s, cmp["ratio"], cmp["step"], cmp["seeds"])}
                for s in cmp["starts"]}
    if not by_start:
        return {}
    common = set.intersection(*(set(v) for v in by_start.values()))
    out = {}
    for key in common:
        eps = {s: by_start[s][key] for s in cmp["starts"]}
        if all(e.get("frames") for e in eps.values()):
            out[key] = eps
    return out


def _order(coll, key):
    task, seed, train_seed = key
    return (KLASS_RANK[coll.tasks[task].klass] if task in coll.tasks else 3, task, seed, train_seed)


def _example(coll, kind: str, title: str, key, eps) -> dict:
    task, seed, train_seed = key
    any_ep = next(iter(eps.values()))
    return {"kind": kind, "title": title, "task": task, "eval_seed": seed, "train_seed": train_seed,
            "klass": coll.tasks[task].klass if task in coll.tasks else "?",
            "instruction": any_ep.get("instruction"),
            "per_start": {s: {"success": e["success"], "steps": e["steps"], "max_steps": e["max_steps"],
                              "frames": e["frames"]} for s, e in eps.items()}}


def pick(coll, cmp, per_kind: int = 2) -> list:
    sc = scenes(coll, cmp)
    ref = cmp["reference"]
    if not sc or ref not in cmp["starts"]:
        return []
    keys = sorted(sc, key=lambda k: _order(coll, k))
    out, used = [], set()
    for s in cmp["starts"]:
        if s == ref:
            continue
        for kind, cond, title in (
                ("win", lambda e: e[s]["success"] and not e[ref]["success"], f"{s} 成功、{ref} 失败"),
                ("loss", lambda e: e[ref]["success"] and not e[s]["success"], f"{ref} 成功、{s} 失败")):
            n = 0
            for k in keys:
                if n >= per_kind:
                    break
                if k not in used and cond(sc[k]):
                    out.append(_example(coll, kind, title, k, sc[k]))
                    used.add(k)
                    n += 1
    for kind, cond, title in (("all_success", lambda e: all(x["success"] for x in e.values()), "全部起点都成功"),
                              ("all_fail", lambda e: not any(x["success"] for x in e.values()), "全部起点都失败")):
        k = next((k for k in keys if k not in used and cond(sc[k])), None)
        if k:
            out.append(_example(coll, kind, title, k, sc[k]))
    return out
