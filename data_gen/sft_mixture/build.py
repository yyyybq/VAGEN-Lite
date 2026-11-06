"""Build LLaMA-Factory datasets of the pool mixtures named in one config file.

    python -m data_gen.sft_mixture.build --config data_gen/sft_mixture/configs/CONFIG.yaml [--recipes D,E]

Per recipe, OUT/<recipe>/ gets one file per pool and split (``P0_train.jsonl`` ... ``P2_val.jsonl``), every row
with its ``pool`` tag, plus ``dataset_info.json``, ``train.yaml`` and ``recipe.json``; OUT/recipes.md is the
recipe table.  Rules:

* QA rows (P0, P1) and action rows (P2) stay separate samples; rows are copied, never merged or rewritten.
* Training rows come only from the split manifest's train scenes; validation rows from its val scenes, unless a
  pool's config names another split for them (``val_scenes``; the P0 sample export, whose validation rows are
  every 10th question of train scenes) - then recipe.json and the table say so.
* Every recipe evaluates the same validation rows of every pool (also of pools it does not train on), each as its
  own LLaMA-Factory eval dataset with ``eval_on_each_dataset: true``, so the trainer logs ``eval_P0_val_loss``,
  ``eval_P1_val_loss`` and ``eval_P2_val_loss`` and recipes can be compared on them.
* Counts per pool: ``all``, a number, ``{times: x, of: POOL}`` (x times another pool's count in the same recipe)
  or ``{share: x}`` of the recipe's ``total``; smaller than the pool means a stratified draw (P0 by question
  group, P1 by question type, P2 by task family) with the config's seed.
* ``draw_as: OTHER``: draw each pool with another recipe's seed, so a recipe that drops a pool keeps exactly the
  other recipe's rows of the pools it shares (matrix D_noP0 = D without P0).
* ``exclude_scenes: FILE`` (a JSON file with an "excluded" list of scene ids; user 2026-10-29, the scene screen):
  the rows of those scenes leave every pool, training and validation, before any count or draw.
* Action pools besides P2 (the Qwen3 matrix, user 2026-10-30, after the main line's registration of 2026-10-27):
  P2q, round 11's conversations in the new format (real action names), and P2f, the format SFT's short
  conversations; both are LF exports read like P2 (kind ``sft_trajectories``).  P2f2k (the user, 2026-11-03: B' as a
  cell after matrix v3): the main line's 2,000 format conversations without the excluded scenes, read the same way.
* ``format_example: action_names`` (or ``placeholder``) on an action pool: every row's system prompt and the
  format text of each user turn must be the text the RL env shows with that format example
  (release.protocol_system_prompt / _format_block); any other row stops the build.  Rows are still never rewritten.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

POOLS = ("P0", "P1", "P2", "P2q", "P2f", "P2f2k", "Pg", "P2t")
POOL_ZH = {"P0": "P0 仿 benchmark 的 QA", "P1": "P1 转移题", "P2": "P2 SFT 动作轨迹",
           "P2q": "P2q SFT 动作轨迹（新格式：真实动作名）", "P2f": "P2f 格式 SFT 的短对话（真实动作名）",
           "P2f2k": "P2f2k 主线的格式对话（2,000 段去掉边缘场景，真实动作名）",
           # group G of matrix v3 (2026-11-06): data_gen/transition_qa/pointing_qa.py, data_gen/sft_mixture/targets_convert.py
           "Pg": "Pg 指认题（点出 / 框出任务里的目标）", "P2t": "P2t SFT 动作轨迹（先写目标位置再动作）"}
DATASET_INFO = {"formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images", "videos": "videos"},
                "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                         "assistant_tag": "assistant", "system_tag": "system"}}
FRAME_TASK = re.compile(r"(v10_[0-9a-f]+)__[a-z]\d+\.png$")


class MixtureError(SystemExit):
    pass


# =============================================================================
# io
# =============================================================================
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def row(pool: str, source: Dict[str, Any], source_id: str, scene: str, stratum: str,
        meta: Dict[str, Any]) -> Dict[str, Any]:
    """One output row: the source conversation unchanged, the pool tag and a fixed set of metadata columns."""
    return {"messages": source["messages"], "images": list(source.get("images") or []),
            "videos": list(source.get("videos") or []), "pool": pool, "source_id": str(source_id),
            "scene": str(scene), "stratum": str(stratum), "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True)}


# =============================================================================
# pools
# =============================================================================
def load_p0(spec: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    """LF export of data_gen/interiorgs_qa: qa_<split>.jsonl, line i described by audit_<split>.jsonl line i."""
    d = Path(spec["dir"])
    records = read_jsonl(d / spec[split])
    audit = read_jsonl(d / spec[split].replace("qa_", "audit_"))
    if len(records) != len(audit):
        raise MixtureError(f"P0 {split}: {len(records)} records but {len(audit)} audit lines")
    drop = set(spec.get("exclude_groups") or [])
    out = []
    for rec, a in zip(records, audit):
        groups = [str(g) for g in a["groups"]]
        if any(g in drop for g in groups):
            continue
        out.append(row("P0", rec, f"p0_{a['qids'][0]}", a["scene_id"], groups[0],
                       {"qids": a["qids"], "groups": groups, "kind": a["kind"], "questions": len(a["qids"])}))
    return out


def load_p1(spec: Dict[str, Any], split: str, name: str = "P1") -> List[Dict[str, Any]]:
    """Question rows of the transition-question kind (P1, and Pg: the pointing questions of group G)."""
    out = []
    for r in read_jsonl(Path(spec[split])):
        out.append(row(name, r, r["sample_id"], r["scene"], r["qtype"],
                       {"qtype": r["qtype"], "qclass": r["qclass"], "family": r["family"], "task_id": r["task_id"],
                        "questions": 1}))
    return out


def load_p2(spec: Dict[str, Any], split: str, tasks: Dict[str, Dict[str, Any]], name: str = "P2") -> List[Dict[str, Any]]:
    """SFT trajectory rows (messages + images only): the task, its scene and family from the frame names."""
    out = []
    for r in read_jsonl(Path(spec[split])):
        ids = {m.group(1) for p in r.get("images") or [] for m in [FRAME_TASK.search(str(p))] if m}
        if len(ids) != 1:
            raise MixtureError(f"{name} {split}: a row whose frames name {sorted(ids)} tasks")
        tid = ids.pop()
        task = tasks.get(tid)
        if task is None:
            raise MixtureError(f"{name} {split}: task {tid} is not in task_rows")
        out.append(row(name, r, tid, task["scene_id"], task["task_type"],
                       {"task_id": tid, "family": task["task_type"],
                        "turns": sum(1 for m in r["messages"] if m["role"] == "assistant")}))
    return out


def load_pool(name: str, spec: Dict[str, Any], split: str, tasks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    kind = spec["kind"]
    if kind == "interiorgs_qa_lf":
        return load_p0(spec, split)
    if kind == "transition_qa":
        return load_p1(spec, split, name)
    if kind == "sft_trajectories":
        return load_p2(spec, split, tasks, name)
    raise MixtureError(f"pool {name}: unknown kind {kind!r}")


def check_format(name: str, split: str, rows: Sequence[Dict[str, Any]], format_example: str,
                 tasks: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Action rows show the RL env's text for ``format_example``: the system prompt of their task family and, at the
    end of each user turn, the format block (the first turn's with the example), each exactly once.  A targets format
    (group G of matrix v3) names the task's objects in its block: ``tasks`` gives them (task_rows)."""
    from data_gen.interiorgs_v10.admission import load_policy
    from data_gen.interiorgs_v10.release import _format_block, protocol_system_prompt
    from data_gen.interiorgs_v10.training_prep import with_format_example
    from vagen.envs.active_spatial.pointing import target_refs
    from vagen.envs.active_spatial.prompt import TARGET_FORMATS, targets_format_block

    policy = with_format_example(load_policy(), format_example)
    blocks = {first: _format_block(policy, first) for first in (True, False)}
    targets = format_example in TARGET_FORMATS
    if targets and not tasks:
        raise MixtureError(f"{name} {split}: the {format_example} format names each task's objects: give task_rows")
    system: Dict[str, str] = {}
    bad = collections.Counter()
    for r in rows:
        family = r["stratum"]
        if family not in system:
            system[family] = protocol_system_prompt(family, policy)
        msgs = r["messages"]
        if not msgs or msgs[0]["role"] != "system" or msgs[0]["content"] != system[family]:
            bad["system prompt"] += 1
            continue
        if targets:
            refs = target_refs(tasks[r["source_id"]])
            ex = policy["sft"]["export"]
            blocks = {first: targets_format_block(format_example, ex["max_actions_per_turn"], "|", first, refs)
                      for first in (True, False)}
        users = [m for m in msgs if m["role"] == "user"]
        if not users or not all(m["content"].endswith("\n" + blocks[k == 0]) and m["content"].count(blocks[k == 0]) == 1
                                for k, m in enumerate(users)):
            bad["format text"] += 1
    if bad:
        raise MixtureError(f"{name} {split}: {sum(bad.values())} of {len(rows)} rows do not show the {format_example} "
                           f"format example the RL env shows ({dict(bad)})")


def check_media(name: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Every image and video frame is an absolute path to an existing file (the trainer runs from another folder)."""
    paths = {p for r in rows for p in r["images"]} | {p for r in rows for v in r["videos"] for p in v}
    bad = sorted(p for p in paths if not (Path(p).is_absolute() and Path(p).is_file()))
    if bad:
        raise MixtureError(f"{name}: {len(bad)} of {len(paths)} images are not absolute paths of existing files, "
                           f"e.g. {bad[:2]}")


def check_scenes(name: str, split: str, rows: Sequence[Dict[str, Any]], allowed: set, which: str) -> None:
    bad = sorted({r["scene"] for r in rows if r["scene"] not in allowed})
    if bad:
        raise MixtureError(f"{name} {split}: {len(bad)} scenes outside the {which} scenes, e.g. {bad[:3]}")


# =============================================================================
# counts and draws
# =============================================================================
def stratified(rows: Sequence[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """``n`` rows, each stratum in proportion (largest remainder), in a seeded random order inside strata."""
    if n >= len(rows):
        return list(rows)
    strata: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["source_id"]):
        strata[r["stratum"]].append(r)
    exact = {k: n * len(v) / len(rows) for k, v in strata.items()}
    take = {k: int(math.floor(x)) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - take[k]), k))[: n - sum(take.values())]:
        take[k] += 1
    out = []
    for k in sorted(strata):
        group = list(strata[k])
        rng.shuffle(group)
        out.extend(group[: take[k]])
    return out


def resolve_counts(spec: Dict[str, Any], available: Dict[str, int]) -> Dict[str, int]:
    """Number of training rows per pool for one recipe (see the module docstring)."""
    wanted = dict(spec.get("pools") or {})
    out: Dict[str, int] = {}
    for pool, c in wanted.items():
        if pool not in POOLS:
            raise MixtureError(f"unknown pool {pool!r}")
        if c == "all":
            out[pool] = available[pool]
        elif isinstance(c, int):
            out[pool] = c
    for pool, c in wanted.items():
        if isinstance(c, dict) and "times" in c:
            ref = c["of"]
            if ref not in out:
                raise MixtureError(f"{pool}: 'of: {ref}' needs {ref}'s count to be a number or 'all'")
            out[pool] = int(round(float(c["times"]) * out[ref]))
        elif isinstance(c, dict) and "share" in c:
            if "total" not in spec:
                raise MixtureError(f"{pool}: a share needs the recipe's total")
            out[pool] = int(round(float(c["share"]) * int(spec["total"])))
        elif pool not in out:
            raise MixtureError(f"{pool}: cannot read the count {c!r}")
    for pool, n in out.items():
        if n > available[pool]:
            raise MixtureError(f"{pool}: {n} rows wanted, {available[pool]} available")
    return out


# =============================================================================
# outputs
# =============================================================================
def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {"rows": len(rows),
            "questions_or_trajectories": sum(int(json.loads(r["meta"]).get("questions", 1)) for r in rows),
            "assistant_turns": sum(1 for r in rows for m in r["messages"] if m["role"] == "assistant"),
            "images": sum(len(r["images"]) for r in rows) + sum(len(v) for r in rows for v in r["videos"])}


def train_yaml(cfg: Dict[str, Any], d: Path, train_sets: Sequence[str], val_sets: Sequence[str]) -> Dict[str, Any]:
    base = yaml.safe_load(Path(cfg["base_recipe"]).read_text())
    base.update({"dataset_dir": str(d.resolve()), "dataset": ",".join(train_sets), "output_dir": str((d / "model").resolve()),
                 "image_max_pixels": 256 * 256, "video_max_pixels": 256 * 256})
    if val_sets:
        base.update({"eval_dataset": ",".join(val_sets), "eval_on_each_dataset": True, "do_eval": True})
        base.pop("val_size", None)
    base.update(copy.deepcopy(cfg.get("train_overrides") or {}))
    return base


def build(cfg: Dict[str, Any], only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(cfg["split_manifest"]).read_text())
    scenes = {k: set(manifest[f"{k}_scenes"]) for k in ("train", "val", "test")}
    tasks = {r["task_id"]: r for r in read_jsonl(Path(cfg["task_rows"]))} if cfg.get("task_rows") else {}
    seed = int(cfg.get("seed", 0))
    pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    notes: List[str] = []
    excluded_scenes = (set(json.loads(Path(cfg["exclude_scenes"]).read_text(encoding="utf-8"))["excluded"])
                       if cfg.get("exclude_scenes") else set())
    excluded: Dict[str, Dict[str, int]] = {}
    unknown = sorted(set(cfg.get("pools") or {}) - set(POOLS))
    if unknown:
        raise MixtureError(f"unknown pools {unknown} (known: {', '.join(POOLS)})")
    for name in POOLS:
        spec = (cfg.get("pools") or {}).get(name)
        if not spec:
            continue
        tr = load_pool(name, spec, "train", tasks)
        check_scenes(name, "train", tr, scenes["train"], "train")
        va = load_pool(name, spec, "val", tasks) if spec.get("val") else []
        val_split = spec.get("val_scenes", "val")
        check_scenes(name, "val", va, scenes[val_split], val_split)
        if val_split != "val":
            notes.append(f"{name} 的验证行来自 {val_split} 场景（{spec.get('val_scenes_reason', '配置指定')}），不是 val 场景")
        if spec.get("format_example"):  # the text the RL env shows (Qwen3 matrix: real action names)
            check_format(name, "train", tr, str(spec["format_example"]), tasks)
            check_format(name, "val", va, str(spec["format_example"]), tasks)
            notes.append(f"{name} 每条对话的系统提示和每个用户回合的格式行，都和 RL 环境在格式样例 "
                         f"{spec['format_example']} 下显示的逐字相同（构建时逐条核对）")
        if excluded_scenes:
            excluded[name] = {"train": sum(r["scene"] in excluded_scenes for r in tr),
                              "val": sum(r["scene"] in excluded_scenes for r in va)}
            tr = [r for r in tr if r["scene"] not in excluded_scenes]
            va = [r for r in va if r["scene"] not in excluded_scenes]
        overlap = {r["source_id"] for r in tr} & {r["source_id"] for r in va}
        if overlap:
            raise MixtureError(f"{name}: {len(overlap)} rows in both train and val, e.g. {sorted(overlap)[:3]}")
        pools[name] = {"train": tr, "val": va}
    if excluded_scenes:
        notes.append(f"排除了 {len(excluded_scenes)} 个场景（{Path(cfg['exclude_scenes']).name}）的行：" + "；".join(
            f"{p} 训练 {c['train']}、验证 {c['val']}" for p, c in excluded.items()) + "（上表的可用条数是排除之后的）")
    # validation rows: the same for every recipe
    val_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name, n in (cfg.get("val") or {}).items():
        rows = pools[name]["val"]
        want = len(rows) if n == "all" else int(n)
        val_rows[name] = stratified(rows, want, random.Random(f"{seed}:val:{name}"))
        if cfg.get("check_media", True):
            check_media(f"{name}_val", val_rows[name])
    table = []
    for rname, rspec in (cfg.get("recipes") or {}).items():
        if only and rname not in only:
            continue
        if rspec.get("same_as"):  # same SFT data and model as another recipe (matrix E = D, RL differs)
            ref = next((t for t in table if t["recipe"] == rspec["same_as"]), None)
            if ref is None:
                raise MixtureError(f"{rname}: same_as {rspec['same_as']} must come earlier in the config")
            table.append(dict(ref, recipe=rname, description=rspec.get("description", ""),
                              output_format=rspec.get("output_format", ref.get("output_format", "")),
                              note=f"SFT 与 {rspec['same_as']} 相同，共用模型；{rspec.get('description', '')}"))
            continue
        counts = resolve_counts(rspec, {k: len(v["train"]) for k, v in pools.items()})
        entry: Dict[str, Any] = {"recipe": rname, "description": rspec.get("description", ""), "counts": counts,
                                 "output_format": rspec.get("output_format", "")}
        if not counts:
            entry["note"] = "不做 SFT"
            table.append(entry)
            continue
        d = out / rname
        d.mkdir(parents=True, exist_ok=True)
        info, train_sets, val_sets, summary = {}, [], [], {}
        for pool in POOLS:
            if counts.get(pool):
                draw = rspec.get("draw_as", rname)  # another recipe's seed: the same rows of the shared pools
                rows = stratified(pools[pool]["train"], counts[pool], random.Random(f"{seed}:{draw}:{pool}"))
                if cfg.get("check_media", True):
                    check_media(f"{rname} {pool}_train", rows)
                write_jsonl(d / f"{pool}_train.jsonl", rows)
                info[f"{pool}_train"] = dict(DATASET_INFO, file_name=f"{pool}_train.jsonl")
                train_sets.append(f"{pool}_train")
                summary[f"{pool}_train"] = summarise(rows)
        for pool, rows in val_rows.items():
            write_jsonl(d / f"{pool}_val.jsonl", rows)
            info[f"{pool}_val"] = dict(DATASET_INFO, file_name=f"{pool}_val.jsonl")
            val_sets.append(f"{pool}_val")
            summary[f"{pool}_val"] = summarise(rows)
        (d / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
        (d / "train.yaml").write_text(yaml.safe_dump(train_yaml(cfg, d, train_sets, val_sets), sort_keys=False))
        entry.update(dir=str(d), summary=summary, notes=notes)
        (d / "recipe.json").write_text(json.dumps(dict(entry, config=cfg.get("name"), seed=seed), indent=1,
                                                  ensure_ascii=False))
        table.append(entry)
    (out / "recipes.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    (out / "recipes.md").write_text(recipe_table(cfg, table, pools, val_rows, notes), encoding="utf-8")
    return {"recipes": [t["recipe"] for t in table], "pools": {k: {s: len(v[s]) for s in v} for k, v in pools.items()},
            "excluded": excluded}


def recipe_table(cfg: Dict[str, Any], table: Sequence[Dict[str, Any]], pools: Dict[str, Any],
                 val_rows: Dict[str, List[Dict[str, Any]]], notes: Sequence[str]) -> str:
    cols = [p for p in POOLS if p in pools]
    extra = "".join(clause for p, clause in (("P2q", "，P2q 同 P2（新格式）"),
                                             ("P2f", "，P2f 一条是一到两轮的短对话（格式 SFT 从 P2 的对话里截的）"),
                                             ("P2f2k", "，P2f2k 同样是一到两轮的短对话（主线 11-01 生成）"))
                    if p in cols)
    lines = [f"# 配方表：{cfg.get('name', '')}", "",
             f"由 `python -m data_gen.sft_mixture.build --config ...` 生成。数字是样本条数（一条 = 一段对话）："
             f"P0、P1 一条是一道题（P0 的视频题一条含同一段视频的几道题），P2 一条是一整条动作轨迹（多回合）{extra}。", "",
             "## 池子", ""]
    lines += ["| 池 | 可用训练条数 | 可用验证条数 | 每个配方都用的验证条数 |", "| --- | --- | --- | --- |"]
    for p in cols:
        lines.append(f"| {POOL_ZH[p]} | {len(pools[p]['train'])} | {len(pools[p]['val'])} | {len(val_rows.get(p, []))} |")
    lines += ["", "## 配方", "",
              "| 组 | 输出格式 | " + " | ".join(cols) + " | 训练条数合计 | 助手回合 | 图片（含视频帧） | 说明 |",
              "| --- | --- | " + " | ".join("---" for _ in cols) + " | --- | --- | --- | --- |"]
    for t in table:
        c = t["counts"]
        s = t.get("summary") or {}
        tr = [s[k] for k in s if k.endswith("_train")]
        lines.append(f"| {t['recipe']} | {t.get('output_format') or '–'} | " + " | ".join(str(c.get(p, 0)) for p in cols) + " | "
                     f"{sum(x['rows'] for x in tr)} | {sum(x['assistant_turns'] for x in tr)} | {sum(x['images'] for x in tr)} | "
                     f"{t.get('note') or t.get('description') or ''} |")
    if notes:
        lines += ["", "## 注意", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--recipes", default="", help="comma-separated recipe names (default: all)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    res = build(cfg, [r for r in args.recipes.split(",") if r] or None)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
