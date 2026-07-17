#!/usr/bin/env python3
"""
analyze_experiments.py — VAGEN-Lite Active Spatial RL 实验分析工具

用法:
    # 分析单个实验
    python3 scripts/analyze_experiments.py --exps v25_groupadv_100scenes_klhi

    # 分析多个实验，横向对比
    python3 scripts/analyze_experiments.py \
        --exps v24_groupadv_100scenes,v25_groupadv_100scenes_klhi,v25_groupadv_100scenes_cosine

    # 指定实验目录前缀（默认 exps/vagen_active_spatial/）
    python3 scripts/analyze_experiments.py --exps v25_klhi --expdir exps/vagen_active_spatial

    # 打印 train-time 指标
    python3 scripts/analyze_experiments.py --exps v25_klhi --train

    # 打印 val 指标（默认开启）
    python3 scripts/analyze_experiments.py --exps v25_klhi --val

    # 打印行为多样性指标（rollout_data/ 目录）
    python3 scripts/analyze_experiments.py --exps v26_klhi_lr5e7 --diversity

    # 打印 per-task-type 成功率学习曲线
    python3 scripts/analyze_experiments.py --exps v34_grpo_rewscale_clip02 --task_curve

    # 打印失败模式分布（最新 val 步骤）
    python3 scripts/analyze_experiments.py --exps v34_grpo_rewscale_clip02 --failure_modes

    # 全部输出
    python3 scripts/analyze_experiments.py \
        --exps v24_groupadv_100scenes,v25_groupadv_100scenes_klhi,v25_groupadv_100scenes_cosine \
        --train --val --summary --task_curve --failure_modes
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────── 路径配置 ───────────────────────────

DEFAULT_EXPDIR = "exps/vagen_active_spatial"
DEFAULT_N_ID   = 21    # in-domain val 环境数
DEFAULT_N_OOD  = 19    # OOD val 环境数
DEFAULT_VAL_N  = 4     # 每环境 seed 数


# ─────────────────────────── Val 解析 ───────────────────────────

def parse_val_file(fpath: str, n_id: int, n_ood: int, val_n: int) -> dict:
    """
    解析单个 validation JSONL 文件。
    返回: {id_b4, id_m4, id_w4, ood_b4, ood_m4, ood_w4, total}
    """
    with open(fpath) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    total = len(lines)
    id_entries  = lines[:n_id * val_n]
    ood_entries = lines[n_id * val_n: (n_id + n_ood) * val_n]

    def agg(entries, n):
        best_list, mean_list, worst_list = [], [], []
        for i in range(0, len(entries), n):
            chunk = entries[i:i+n]
            if not chunk:
                continue
            succs = [float(r.get("traj_success", 0)) for r in chunk]
            best_list.append(max(succs))
            mean_list.append(sum(succs) / len(succs))
            worst_list.append(min(succs))
        if not best_list:
            return None, None, None
        return (
            sum(best_list) / len(best_list),
            sum(mean_list) / len(mean_list),
            sum(worst_list) / len(worst_list),
        )

    id_b4, id_m4, id_w4   = agg(id_entries, val_n)
    ood_b4, ood_m4, ood_w4 = agg(ood_entries, val_n)
    return dict(id_b4=id_b4, id_m4=id_m4, id_w4=id_w4,
                ood_b4=ood_b4, ood_m4=ood_m4, ood_w4=ood_w4,
                total=total)


def parse_val_dir(val_dir: str, n_id: int, n_ood: int, val_n: int) -> list:
    """
    扫描 validation/ 目录下所有 *.jsonl，按 step 排序返回列表。
    每项: (step: int, metrics: dict)
    """
    results = []
    if not os.path.isdir(val_dir):
        return results
    for fname in sorted(os.listdir(val_dir), key=lambda x: int(x.rstrip(".jsonl").split(".")[0]) if x.endswith(".jsonl") else 0):
        if not fname.endswith(".jsonl"):
            continue
        step = int(fname[:-6])
        fpath = os.path.join(val_dir, fname)
        try:
            metrics = parse_val_file(fpath, n_id, n_ood, val_n)
            results.append((step, metrics))
        except Exception as e:
            print(f"  [WARN] 解析 {fpath} 失败: {e}", file=sys.stderr)
    return results


# ─────────────────────────── Train 解析 ───────────────────────────

TRAIN_KEYS = [
    "actor/entropy",
    "train/traj_success/mean",
    "critic/score/mean",
    "response_length/clip_ratio",
    "actor/lr",
    "actor/kl_loss",
    "actor/grad_norm",
    "critic/returns/mean",
]

def parse_train_log(logfile: str) -> dict:
    """
    解析 train.log，按 global_step 返回 {step: {key: value, ...}}。
    日志中数值格式为 plain float 或 np.float64(xxx) 两种，均支持。
    """
    steps = {}
    if not os.path.isfile(logfile):
        return steps
    # 通用数值匹配：np.float64(xxx) 或纯浮点数
    _val_pat = r"(np\.float64\([0-9.e+\-]+\)|[0-9.e+\-]+)"
    with open(logfile) as f:
        for line in f:
            if "training/global_step:" not in line or "actor/entropy:" not in line:
                continue
            m = re.search(r"training/global_step:(\d+)", line)
            if not m:
                continue
            step = int(m.group(1))
            row = {}
            for key in TRAIN_KEYS:
                km = re.search(
                    r"(?<![/\w])" + re.escape(key) + r":" + _val_pat,
                    line,
                )
                if km:
                    raw = km.group(1)
                    if raw.startswith("np.float64("):
                        raw = raw[11:-1]  # strip "np.float64(" and ")"
                    try:
                        row[key] = float(raw)
                    except ValueError:
                        pass
            steps[step] = row
    return steps


# ─────────────────────────── Rollout 多样性解析 ───────────────────────────

# 所有合法动作 token
ACTION_TOKENS = ["move_forward", "turn_left", "turn_right", "strafe_left", "strafe_right", "stop", "done"]
ACTION_PAT = re.compile(r"\b(" + "|".join(ACTION_TOKENS) + r")\b")


def extract_actions(output_str: str) -> list:
    """从单步 rollout 输出中提取动作序列。"""
    return ACTION_PAT.findall(output_str)


def action_entropy(counts: Counter) -> float:
    """计算动作分布的香农熵（nats）。"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)


def parse_rollout_step(fpath: str) -> Optional[dict]:
    """
    解析单个 rollout step JSONL 文件。
    返回包含多样性指标的字典，或 None（文件不存在/为空）。
    """
    if not os.path.isfile(fpath):
        return None
    try:
        with open(fpath) as f:
            items = [json.loads(l) for l in f if l.strip()]
    except Exception:
        return None
    if not items:
        return None

    all_actions: list  = []          # 所有 rollout 里的全部动作
    first_actions: list = []         # 每条轨迹的第一个动作
    traj_prefixes: list = []         # 每条轨迹前 3 个动作组成的 tuple
    n_empty = 0                      # 没有动作输出的轨迹数

    for item in items:
        out = item.get("output", "")
        acts = extract_actions(out)
        if acts:
            all_actions.extend(acts)
            first_actions.append(acts[0])
            traj_prefixes.append(tuple(acts[:3]))
        else:
            n_empty += 1

    if not all_actions:
        return None

    total_acts = len(all_actions)
    action_dist = Counter(all_actions)
    dominant_action = action_dist.most_common(1)[0][0]
    dominant_pct    = action_dist.most_common(1)[0][1] / total_acts
    act_entropy     = action_entropy(action_dist)
    first_entropy   = action_entropy(Counter(first_actions))
    n_unique_prefix = len(set(traj_prefixes))
    n_rollouts      = len(items)

    return {
        "n_rollouts":      n_rollouts,
        "n_unique_prefix": n_unique_prefix,
        "act_entropy":     act_entropy,
        "first_entropy":   first_entropy,
        "dominant_action": dominant_action,
        "dominant_pct":    dominant_pct,
        "action_dist":     dict(action_dist.most_common()),
        "n_empty":         n_empty,
    }


def parse_rollout_dir(rollout_dir: str, every: int = 20) -> list:
    """
    扫描 rollout_data/ 目录，按 step 返回多样性指标列表。
    only returns steps that are multiples of `every`, plus first and last.
    返回: [(step, metrics_dict), ...]
    """
    results = []
    if not os.path.isdir(rollout_dir):
        return results

    all_steps = []
    for fname in os.listdir(rollout_dir):
        if fname.endswith(".jsonl"):
            try:
                all_steps.append(int(fname[:-6]))
            except ValueError:
                pass
    all_steps.sort()

    if not all_steps:
        return results

    show_steps = set()
    for s in all_steps:
        if s % every == 0 or s == all_steps[0] or s == all_steps[-1]:
            show_steps.add(s)

    for step in sorted(show_steps):
        fpath = os.path.join(rollout_dir, f"{step}.jsonl")
        m = parse_rollout_step(fpath)
        if m is not None:
            results.append((step, m))

    return results


def print_diversity_table(exp_name: str, diversity_records: list):
    """打印行为多样性指标表格。"""
    if not diversity_records:
        print(f"  [INFO] {exp_name}: 无 rollout_data 可分析", file=sys.stderr)
        return

    print(f"\n{'─'*100}")
    print(f"  行为多样性: {exp_name}")
    print(f"{'─'*100}")
    header = (
        f"  {'step':>5}  {'act_H':>6}  {'1st_H':>6}  {'uniq3':>6}  "
        f"{'n_roll':>6}  {'dominant':>12}  {'dom%':>6}  {'dist (top3)'}")
    print(header)
    print(f"  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*6}  {'─'*30}")

    for step, m in diversity_records:
        top3 = list(Counter(m["action_dist"]).most_common(3))
        top3_str = "  ".join(f"{a[:5]}={v/sum(m['action_dist'].values())*100:.0f}%" for a, v in top3)
        print(
            f"  {step:>5}  "
            f"{m['act_entropy']:>6.3f}  "
            f"{m['first_entropy']:>6.3f}  "
            f"{m['n_unique_prefix']:>6}  "
            f"{m['n_rollouts']:>6}  "
            f"{m['dominant_action']:>12}  "
            f"{m['dominant_pct']*100:>5.1f}%  "
            f"{top3_str}"
        )


def get_latest_step(expdir: str) -> int:
    """从 train.log 中提取最新 global_step。"""
    logfile = os.path.join(expdir, "train.log")
    if not os.path.isfile(logfile):
        return -1
    last = -1
    with open(logfile) as f:
        for line in f:
            if "training/global_step:" in line:
                m = re.search(r"training/global_step:(\d+)", line)
                if m:
                    last = int(m.group(1))
    return last


# ─────────────────────────── Task-type 推断 ───────────────────────────

# 从 Task: 描述字符串推断 task_type（与 data_gen/config.py 对应）
_TASK_TYPE_RULES = [
    # 规则按优先级排列，先匹配先返回
    ("projective_relations",     re.compile(r"left_of view|right_of view", re.I)),
    ("equidistance",             re.compile(r"equidistant view",           re.I)),
    ("centering",                re.compile(r"center view",                re.I)),
    ("occlusion_alignment",      re.compile(r"occluded view",              re.I)),
    ("size_distance_invariance", re.compile(r"equal_size view",            re.I)),
    ("apparent_size_ordering",   re.compile(r"apparent_larger view|appears larger|looks bigger", re.I)),
    ("screen_occupancy",         re.compile(r"screen occupancy",           re.I)),
    ("fov_inclusion",            re.compile(r"fov inclusion|field.of.view inclusion", re.I)),
    # front/back/left/right view → absolute_positioning
    ("absolute_positioning",     re.compile(r"\b(front|back|left|right|side)\s+view\b", re.I)),
]

ALL_TASK_TYPES_ORDERED = [
    "absolute_positioning",
    "equidistance",
    "projective_relations",
    "centering",
    "occlusion_alignment",
    "fov_inclusion",
    "size_distance_invariance",
    "apparent_size_ordering",
    "screen_occupancy",
]

TASK_SHORT = {
    "absolute_positioning":      "abs_pos",
    "equidistance":              "equidist",
    "projective_relations":      "proj_rel",
    "centering":                 "center",
    "occlusion_alignment":       "occlusion",
    "fov_inclusion":             "fov_incl",
    "size_distance_invariance":  "size_inv",
    "apparent_size_ordering":    "size_ord",
    "screen_occupancy":          "scrn_occ",
}


def infer_task_type(task_str: str) -> str:
    """从 'Task: Move the camera to the X view of ...' 推断 task_type 字符串。"""
    for ttype, pat in _TASK_TYPE_RULES:
        if pat.search(task_str):
            return ttype
    return "unknown"


def _extract_task_str(entry_input: str) -> str:
    """从 entry['input'] 中提取 Task: 那一行。"""
    for line in entry_input.split("\n"):
        if line.strip().startswith("Task:"):
            return line.strip()
    return ""


# ─────────────────────────── Per-task-type Val 解析 ───────────────────────────

def parse_val_file_per_task(fpath: str, val_n: int, n_id: int) -> dict:
    """
    解析单个 validation JSONL，按 task_type 分别统计 success rate。
    只处理 in-domain (前 n_id * val_n 条) 数据。
    返回: {task_type: {"n": int, "succ_b4": float, "succ_m4": float}}
    """
    try:
        with open(fpath) as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except Exception:
        return {}

    id_entries = lines[: n_id * val_n]

    # 按 (env_idx, task_type) 分组，每 val_n 条为一组
    task_groups: dict = {}  # task_type -> list of group-best-success values
    task_groups_mean: dict = {}

    for group_i in range(0, len(id_entries), val_n):
        chunk = id_entries[group_i : group_i + val_n]
        if not chunk:
            continue
        # Prefer explicit task_type field (retroactively annotated or from new agent loop)
        ttype = _get_task_type(chunk[0])
        succs = [float(r.get("traj_success", 0)) for r in chunk]
        best = max(succs)
        mean = sum(succs) / len(succs)
        task_groups.setdefault(ttype, []).append(best)
        task_groups_mean.setdefault(ttype, []).append(mean)

    result = {}
    for ttype in task_groups:
        bests = task_groups[ttype]
        means = task_groups_mean[ttype]
        result[ttype] = {
            "n":       len(bests),
            "succ_b4": sum(bests) / len(bests),
            "succ_m4": sum(means) / len(means),
        }
    return result


def parse_val_dir_per_task(val_dir: str, val_n: int, n_id: int) -> list:
    """
    扫描 validation/ 目录，返回 [(step, per_task_dict), ...] 按 step 排序。
    """
    results = []
    if not os.path.isdir(val_dir):
        return results
    for fname in sorted(os.listdir(val_dir), key=lambda x: int(x[:-6]) if x.endswith(".jsonl") else 0):
        if not fname.endswith(".jsonl"):
            continue
        step = int(fname[:-6])
        fpath = os.path.join(val_dir, fname)
        per_task = parse_val_file_per_task(fpath, val_n, n_id)
        if per_task:
            results.append((step, per_task))
    return results


# ─────────────────────────── Rich 主结果表（需要新字段）───────────────────────

def _get_task_type(entry: dict) -> str:
    """
    从 entry 获取 task_type。
    优先读新日志字段 entry['task_type']（gym_agent_loop 新增）；
    否则从 entry['input'] 文本推断（旧日志兼容）。
    """
    if "task_type" in entry and entry["task_type"] not in ("", "unknown", None):
        return str(entry["task_type"])
    return infer_task_type(_extract_task_str(entry.get("input", "")))


def _compute_spl(success: bool, n_steps: int, optimal_steps: int) -> float:
    """SPL = success × optimal / max(actual, optimal)."""
    if not success:
        return 0.0
    return optimal_steps / max(n_steps, optimal_steps)


# 粗略的 optimal_steps 估算：用 initial_score 的反面（距目标越远，最优步数越多）。
# 这里用一个保守的下界：如果 initial_score 已经很高（接近成功），optimal 就小。
# 实际 optimal 需要 oracle solver；这里用简单启发式：optimal = 5 步（保守最小值）。
_HEURISTIC_OPTIMAL_STEPS = 5


def parse_val_file_rich(fpath: str, val_n: int, n_id: int) -> dict:
    """
    解析 validation JSONL 生成"丰富主结果表"指标：
      - 按 task_type 分组的 success_rate, mean_score_improvement, mean_final_score, SPL
      - 全局汇总

    字段优先级（从高到低）：
      1. 新 JSONL 字段（gym_agent_loop 升级后）：task_type, initial_score, final_score,
         score_improvement, n_primitive_steps
      2. 文本推断（旧 JSONL 兼容）：task_type from 'Task:' line；
         score_improvement = None（无法从文本推断分数）

    返回:
      {
        "has_score_fields": bool,   # 是否有新版 score 字段
        "by_task": {task_type: {"n", "succ_b4", "mean_final_score",
                                 "mean_score_improvement", "spl"}},
        "overall": {...}
      }
    """
    try:
        with open(fpath) as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except Exception:
        return {}

    id_entries = lines[: n_id * val_n]

    # Detect whether new fields are present
    sample = id_entries[0] if id_entries else {}
    has_score = "initial_score" in sample and "final_score" in sample

    # Per-group accumulation
    task_data: dict = {}  # task_type -> list of per-group dicts

    for gi in range(0, len(id_entries), val_n):
        chunk = id_entries[gi : gi + val_n]
        if not chunk:
            continue

        # Task type (new field preferred)
        ttype = _get_task_type(chunk[0])

        # Pick best seed: success first, then highest reward
        best = max(chunk, key=lambda e: (
            float(e.get("traj_success", 0)),
            float(e.get("final_score", e.get("reward", 0))),
        ))
        succs = [float(e.get("traj_success", 0)) for e in chunk]

        rec = {
            "succ_best": max(succs),
            "succ_mean": sum(succs) / len(succs),
            "final_score":        float(best.get("final_score", 0.0)) if has_score else None,
            "initial_score":      float(best.get("initial_score", 0.0)) if has_score else None,
            "score_improvement":  float(best.get("score_improvement",
                                    best.get("final_score", 0) - best.get("initial_score", 0))) if has_score else None,
            "n_steps":            int(best.get("n_primitive_steps", best.get("n_turns", 0))),
        }
        task_data.setdefault(ttype, []).append(rec)

    def _agg(recs):
        n = len(recs)
        if not n:
            return {}
        succ_b4 = sum(r["succ_best"] for r in recs) / n
        succ_m4 = sum(r["succ_mean"] for r in recs) / n
        mean_fs  = (sum(r["final_score"] for r in recs) / n) if has_score else None
        mean_si  = (sum(r["score_improvement"] for r in recs) / n) if has_score else None
        # SPL: for each group, succ_best * optimal / max(actual_steps, optimal)
        spl_vals = []
        for r in recs:
            spl_vals.append(_compute_spl(
                r["succ_best"] > 0.5,
                r["n_steps"] or _HEURISTIC_OPTIMAL_STEPS,
                _HEURISTIC_OPTIMAL_STEPS,
            ))
        mean_spl   = sum(spl_vals) / n
        mean_steps = sum(r["n_steps"] for r in recs) / n
        return {
            "n": n,
            "succ_b4": succ_b4, "succ_m4": succ_m4,
            "mean_final_score": mean_fs,
            "mean_score_improvement": mean_si,
            "spl": mean_spl,
            "mean_steps": mean_steps,
        }

    by_task = {ttype: _agg(recs) for ttype, recs in task_data.items()}
    all_recs = [r for recs in task_data.values() for r in recs]
    overall  = _agg(all_recs)
    overall["task_type"] = "ALL"

    return {
        "has_score_fields": has_score,
        "by_task": by_task,
        "overall": overall,
    }


def print_rich_result_table(exp_name: str, val_step: Optional[int], fpath: str,
                             val_n: int, n_id: int):
    """
    打印丰富主结果表：按 task_type 的 success_rate / score_improvement / SPL / mean_steps。
    当 initial_score / final_score 不在 JSONL 中时，打印提示并仅显示 success_rate。
    """
    data = parse_val_file_rich(fpath, val_n, n_id)
    if not data:
        print(f"  [WARN] {exp_name}: 无法解析 {fpath}", file=sys.stderr)
        return

    has_score = data["has_score_fields"]
    by_task   = data["by_task"]
    overall   = data["overall"]
    step_str  = str(val_step) if val_step is not None else Path(fpath).stem

    print(f"\n{'═'*90}")
    print(f"  主结果表  |  {exp_name}  |  step={step_str}  |  n_id={n_id}×{val_n}")
    if not has_score:
        print(f"  [注] initial_score/final_score 字段缺失（旧日志）。")
        print(f"       score_improvement 和精确 SPL 需在新训练后才能报告。")
    print(f"{'═'*90}")

    # Header
    col_score = has_score
    hdr_base = f"  {'task_type':<24}  {'n':>3}  {'succ@best':>9}  {'succ@mean':>9}  {'mean_steps':>10}"
    hdr_score = f"  {'final_score':>11}  {'score_imprv':>11}  {'SPL':>6}" if col_score else ""
    print(hdr_base + hdr_score)
    sep_base = f"  {'─'*24}  {'─'*3}  {'─'*9}  {'─'*9}  {'─'*10}"
    sep_scr  = f"  {'─'*11}  {'─'*11}  {'─'*6}" if col_score else ""
    print(sep_base + sep_scr)

    def _fmt_row(ttype, m):
        n        = m.get("n", 0)
        sb       = m.get("succ_b4", 0)
        sm       = m.get("succ_m4", 0)
        steps    = m.get("mean_steps", 0)
        base = f"  {ttype:<24}  {n:>3}  {sb*100:>8.1f}%  {sm*100:>8.1f}%  {steps:>10.1f}"
        if col_score:
            fs  = m.get("mean_final_score")
            si  = m.get("mean_score_improvement")
            spl = m.get("spl", 0)
            fs_s  = f"{fs:.3f}" if fs is not None else "   —  "
            si_s  = f"{si:+.3f}" if si is not None else "   —  "
            base += f"  {fs_s:>11}  {si_s:>11}  {spl:>6.3f}"
        return base

    # Per-task rows
    for ttype in ALL_TASK_TYPES_ORDERED:
        if ttype in by_task:
            print(_fmt_row(ttype, by_task[ttype]))

    # Overall row
    print(sep_base + sep_scr)
    print(_fmt_row("ALL", overall))


# ─────────────────────────── 失败模式分类 ───────────────────────────
#
# 关于 val JSONL 中实际可用的信号（经实证验证）：
#
#   字段              含义
#   ──────────────    ─────────────────────────────────────────────────────────
#   traj_success      0/1，轨迹是否成功
#   reward            累积奖励（shaping + success_reward + step_penalty）
#   output            全轮次对话拼接文本，包含每轮 user/assistant/feedback
#
# 可从 output 提取的信号：
#   "Environment Feedback:" 出现次数 + 1  →  实际 turn 数
#   "Action had no effect." 出现次数      →  boundary 撞墙 / 动作无效次数
#   <action>…</action> blob              →  动作序列（含 VALID_ACTIONS 检查）
#   "Current camera pose: [tx=…]" 匹配   →  位置轨迹（XY 平面）
#
# 注意：
#   * "collision" / "invalid" 关键词在 output 中从不出现（env 只返回
#     "Action executed." 或 "Action had no effect."）。
#   * 所有失败 episode 均跑满 turn budget（max 4 turns 实测），因此
#     "timeout" 对所有失败成立，不具区分力。
#   * 有效的区分维度是 reward 量级，以及动作有效性。
#
# 失败模式定义（基于 reward 区间 + 动作有效性，全部有实际信号依据）：
#
#   success_fast   : 成功，≤ 2 turn                   → 高效导航
#   success_std    : 成功，> 2 turns                   → 正常导航
#   near_miss      : 失败，reward 很高 (≥ near_succ_thr)
#                    → 到达 near-success 区域，shaping 积累了
#                       near_success_bonus + 大量正值 ΔΦ，但位姿分数没过 threshold
#   pos_progress   : 失败，reward ∈ [0, near_succ_thr) → 往对的方向走但差太多
#   no_progress    : 失败，reward ∈ [step_floor, 0)    → 位移后净分数不变
#                    step_floor ≈ max_turns × max_act/turn × step_penalty
#   neg_progress   : 失败，reward < step_floor          → 离目标更远（负 shaping）
#                    或频繁 "Action had no effect."（no_effect_rate 高）
#   format_error   : 失败，输出中无合法 <action> tag，或所有 action blob 中
#                    无一个 VALID_ACTIONS token
#
# 参数设计：
#   near_succ_thr  基于实验 config 的 near_success_bonus（默认 0.5）；
#                  用 near_success_bonus × 2 作为"明显进入 near-success 区"判断。
#   step_floor     = max_turns × max_actions_per_step × |step_penalty|，
#                  纯步长惩罚的理论最低 reward（方向完全中性时）。
#                  在代码里取一个保守估计（不同 exp 用同一默认值）。

FAILURE_MODES = [
    "success_fast",    # 成功且 ≤ 2 turn
    "success_std",     # 成功且 > 2 turns
    "near_miss",       # 失败但 reward 很高（到了 near-success 区域才时间到）
    "pos_progress",    # 失败但 reward > 0（在朝正确方向走）
    "no_progress",     # 失败，reward 接近 0（净 shaping ≈ 0）
    "neg_progress",    # 失败，reward 低于纯步长惩罚下界（往错的方向或无效动作多）
    "format_error",    # 失败，无合法 action 输出
]

# 默认步长惩罚理论下界（对应 4 turns × 5 actions × 0.02 = 0.40）
_DEFAULT_STEP_FLOOR = -0.42    # 略宽松，避免浮点边界
_NEAR_MISS_THR      = 1.0      # 超过此值说明模型确实进入了 near-success 区域

_VALID_ACTIONS = frozenset({
    "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right", "look_up", "look_down", "done",
})
_ACTION_BLOB_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_FB_RE          = re.compile(r"Environment Feedback:")
_NO_EFFECT_RE   = re.compile(r"Action had no effect\.")


def _parse_action_validity(output: str):
    """
    从 output 提取所有 <action> blobs，返回 (has_any_valid, n_empty_tags, all_actions)。
    has_any_valid : 至少一个 blob 含合法动作词
    n_empty_tags  : blob 内没有任何合法动作词的 tag 数量
    all_actions   : 所有合法动作 token 列表
    """
    blobs = _ACTION_BLOB_RE.findall(output)
    all_actions = []
    n_empty = 0
    for blob in blobs:
        tokens = [t.strip() for t in blob.split("|") if t.strip()]
        valid  = [t for t in tokens if t in _VALID_ACTIONS]
        if not valid:
            n_empty += 1
        all_actions.extend(valid)
    return bool(all_actions), n_empty, all_actions


def classify_failure_mode(entry: dict) -> str:
    """
    从单条 val JSONL entry 推断失败/成功模式。
    所有判断分支均基于 val JSONL 中确实存在的信号。
    """
    output  = entry.get("output", "")
    reward  = float(entry.get("reward", 0.0))
    success = float(entry.get("traj_success", 0.0)) > 0.5

    # ── 基础信号 ──────────────────────────────────────────────────────────────
    n_turns    = _FB_RE.findall(output).__len__() + 1   # 实际 turn 数
    n_no_eff   = len(_NO_EFFECT_RE.findall(output))      # 无效动作次数
    has_valid, n_empty_tags, all_acts = _parse_action_validity(output)

    # ── 成功 ─────────────────────────────────────────────────────────────────
    if success:
        return "success_fast" if n_turns <= 2 else "success_std"

    # ── 失败 ─────────────────────────────────────────────────────────────────

    # 格式错误：output 中完全没有合法动作输出
    if not has_valid:
        return "format_error"

    # "Action had no effect." 在多数 turn 都出现 → 动作被边界/约束拦截
    # （注：enable_collision_detection=False 时此信号来自其他约束，仍有效）
    no_effect_rate = n_no_eff / max(n_turns - 1, 1)   # turns-1 = feedback 次数
    if no_effect_rate >= 0.5 and n_turns >= 3:
        return "neg_progress"   # 动作频繁无效 → 等价于没有正向位移

    # 基于 reward 区间的分类
    # near_miss: reward 远超纯步长惩罚 + 有明显正 shaping（进入 near-success 区域）
    if reward >= _NEAR_MISS_THR:
        return "near_miss"

    # pos_progress: reward 为正但低于 near_miss 阈值
    if reward >= 0.0:
        return "pos_progress"

    # no_progress: reward 在理论步长惩罚范围内（净 shaping ≈ 0）
    if reward >= _DEFAULT_STEP_FLOOR:
        return "no_progress"

    # neg_progress: reward 低于纯步长惩罚下界（有负 shaping = 远离目标）
    return "neg_progress"


def parse_failure_modes_from_val(fpath: str, val_n: int, n_id: int) -> dict:
    """
    解析 validation JSONL，按 best-of-N 粒度（每组只取 mode 出现最多的）
    统计失败模式分布。
    返回: {mode: count}
    """
    try:
        with open(fpath) as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except Exception:
        return {}

    id_entries = lines[: n_id * val_n]
    mode_counts: Counter = Counter()

    for group_i in range(0, len(id_entries), val_n):
        chunk = id_entries[group_i : group_i + val_n]
        if not chunk:
            continue
        # 取该组最佳成功样本（traj_success=1 优先）；若全失败，取 reward 最高的
        best = max(chunk, key=lambda e: (float(e.get("traj_success", 0)), float(e.get("reward", 0))))
        mode = classify_failure_mode(best)
        mode_counts[mode] += 1

    return dict(mode_counts)


def parse_failure_modes_per_task(fpath: str, val_n: int, n_id: int) -> dict:
    """
    解析 validation JSONL，按 task_type × failure_mode 的矩阵。
    返回: {task_type: {mode: count}}
    """
    try:
        with open(fpath) as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except Exception:
        return {}

    id_entries = lines[: n_id * val_n]
    result: dict = {}

    for group_i in range(0, len(id_entries), val_n):
        chunk = id_entries[group_i : group_i + val_n]
        if not chunk:
            continue
        task_str = _extract_task_str(chunk[0].get("input", ""))
        ttype = infer_task_type(task_str)
        best = max(chunk, key=lambda e: (float(e.get("traj_success", 0)), float(e.get("reward", 0))))
        mode = classify_failure_mode(best)
        result.setdefault(ttype, Counter())[mode] += 1

    return result


# ─────────────────────────── 打印工具 ───────────────────────────

def fmt(v, decimals=4):
    if v is None:
        return "  —   "
    return f"{v:.{decimals}f}"


def print_task_curve_table(exp_name: str, task_val_records: list):
    """
    打印 per-task-type 成功率随训练步骤的变化表格。
    task_val_records: [(step, {task_type: {"n", "succ_b4", "succ_m4"}}), ...]
    """
    if not task_val_records:
        print(f"  [INFO] {exp_name}: 无 per-task 数据", file=sys.stderr)
        return

    # Collect which task types actually appeared
    all_types = []
    for t in ALL_TASK_TYPES_ORDERED:
        for _, per_task in task_val_records:
            if t in per_task:
                all_types.append(t)
                break

    col_w = 8
    short_cols = [TASK_SHORT.get(t, t[:8]) for t in all_types]
    header_row = f"  {'step':>5}  " + "  ".join(f"{s:>{col_w}}" for s in short_cols)

    print(f"\n{'─'*max(80, len(header_row)+4)}")
    print(f"  Per-Task Success@best-of-N 学习曲线: {exp_name}")
    print(f"{'─'*max(80, len(header_row)+4)}")
    print(f"  {'TaskTypes':<10}" + "  ".join(f"{t[:col_w]:>{col_w}}" for t in all_types))
    print(header_row)
    sep = f"  {'─'*5}  " + "  ".join("─" * col_w for _ in all_types)
    print(sep)

    for step, per_task in task_val_records:
        vals = []
        for t in all_types:
            if t in per_task:
                v = per_task[t]["succ_b4"]
                vals.append(f"{v*100:>{col_w}.1f}%")
            else:
                vals.append(f"{'—':>{col_w}}")
        print(f"  {step:>5}  " + "  ".join(vals))

    # Print n-per-task from last step
    _, last_task = task_val_records[-1]
    n_row = "  [n]  " + "  ".join(f"{last_task.get(t, {}).get('n', 0):>{col_w}}" for t in all_types)
    print(n_row)


_FAILURE_MODE_DESC = {
    "success_fast":  "成功，≤2 turn（高效导航）",
    "success_std":   "成功，>2 turns（正常导航）",
    "near_miss":     "失败，reward≥1.0（到了 near-success 区域但时间耗尽）",
    "pos_progress":  "失败，0≤reward<1.0（朝正确方向走但差距太大）",
    "no_progress":   "失败，reward≈0（净 shaping 接近 0，位移后分数不变）",
    "neg_progress":  "失败，reward＜步长惩罚下界（远离目标或动作无效率高）",
    "format_error":  "失败，output 中无合法 action（格式完全崩溃）",
}


def print_failure_mode_table(exp_name: str, mode_counts: dict, per_task_modes: dict = None):
    """
    打印失败模式分布表格。
    mode_counts: {mode: count}
    per_task_modes: {task_type: {mode: count}} (optional)

    分类信号说明（仅依赖 val JSONL 实际存在的字段）：
      reward        — cumulative reward (shaping + step_penalty + success_reward)
      n_turns       — 从 "Environment Feedback:" 出现次数推断
      n_no_effect   — "Action had no effect." 出现次数（boundary hit）
      <action> tags — 合法动作词提取
    """
    if not mode_counts:
        print(f"  [INFO] {exp_name}: 无失败模式数据", file=sys.stderr)
        return

    total = sum(mode_counts.values())
    print(f"\n{'─'*80}")
    print(f"  失败模式分布 ({exp_name}, total={total} 环境-组)")
    print(f"  信号依据: reward 区间 + action 有效性 + no-effect 率")
    print(f"{'─'*80}")

    # Overall mode distribution
    print(f"  {'Mode':<16}  {'Count':>6}  {'Pct':>7}  说明")
    print(f"  {'─'*16}  {'─'*6}  {'─'*7}  {'─'*30}")
    for mode in FAILURE_MODES:
        cnt = mode_counts.get(mode, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5)
        desc = _FAILURE_MODE_DESC.get(mode, "")
        print(f"  {mode:<16}  {cnt:>6}  {pct:>6.1f}%  {bar:<12}  {desc}")

    # Per-task breakdown — only show failure-relevant modes
    if per_task_modes:
        fail_modes = ["near_miss", "pos_progress", "no_progress", "neg_progress", "format_error"]
        mode_cols = [m for m in fail_modes if any(m in v for v in per_task_modes.values())]
        if not mode_cols:
            return
        print()
        print(f"  Per-task × Failure-mode breakdown  (best-of-N, failures only)")
        col_w = 12
        hdr = f"  {'task_type':<24}" + "".join(f"  {m[:col_w]:>{col_w}}" for m in mode_cols)
        print(hdr)
        print(f"  {'─'*24}" + "".join(f"  {'─'*col_w}" for _ in mode_cols))
        for ttype in ALL_TASK_TYPES_ORDERED:
            if ttype not in per_task_modes:
                continue
            row_counts = per_task_modes[ttype]
            row_total  = sum(row_counts.values())
            # Only print row if there's at least one failure (non-success modes)
            fail_total = sum(row_counts.get(m, 0) for m in fail_modes)
            if fail_total == 0:
                continue
            cols = []
            for m in mode_cols:
                c   = row_counts.get(m, 0)
                pct = c / row_total * 100 if row_total > 0 else 0
                cols.append(f"{c}({pct:.0f}%)")
            print(f"  {ttype:<24}" + "".join(f"  {c:>{col_w}}" for c in cols))


def print_val_table(exp_name: str, val_records: list):
    """打印单实验 val 曲线表格。"""
    print(f"\n{'─'*80}")
    print(f"  Val 曲线: {exp_name}")
    print(f"{'─'*80}")
    header = f"  {'step':>5}  {'ID_b4':>7}  {'ID_m4':>7}  {'ID_w4':>7}  {'OOD_b4':>7}  {'OOD_m4':>7}  {'OOD_w4':>7}"
    print(header)
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for step, m in val_records:
        row = (
            f"  {step:>5}  "
            f"{fmt(m['id_b4']):>7}  "
            f"{fmt(m['id_m4']):>7}  "
            f"{fmt(m['id_w4']):>7}  "
            f"{fmt(m['ood_b4']):>7}  "
            f"{fmt(m['ood_m4']):>7}  "
            f"{fmt(m['ood_w4']):>7}"
        )
        print(row)


def print_train_table(exp_name: str, train_steps: dict, every: int = 20):
    """打印 train 指标表格（每隔 every 步打印一行）。"""
    print(f"\n{'─'*100}")
    print(f"  Train 指标: {exp_name}")
    print(f"{'─'*100}")
    header = f"  {'step':>5}  {'entropy':>8}  {'t_succ%':>8}  {'crit_sc':>8}  {'resp_clip':>10}  {'grad_nrm':>9}  {'lr':>10}"
    print(header)
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*10}")
    shown_steps = sorted(s for s in train_steps if s % every == 0 or s in [1, 5])
    for step in shown_steps:
        r = train_steps[step]
        tsucc = r.get("train/traj_success/mean")
        tsucc_str = f"{tsucc*100:.2f}" if tsucc is not None else "  —  "
        row = (
            f"  {step:>5}  "
            f"{fmt(r.get('actor/entropy'), 3):>8}  "
            f"{tsucc_str:>8}  "
            f"{fmt(r.get('critic/score/mean'), 3):>8}  "
            f"{fmt(r.get('response_length/clip_ratio'), 4):>10}  "
            f"{fmt(r.get('actor/grad_norm'), 3):>9}  "
            f"{r.get('actor/lr', 0.0):>10.2e}"
        )
        print(row)


def print_comparison_table(exp_val_data: list):
    """
    跨实验横向对比表（仅 peak 指标）。
    exp_val_data: [(exp_name, val_records), ...]
    """
    print(f"\n{'═'*100}")
    print("  跨实验峰值对比")
    print(f"{'═'*100}")
    header = f"  {'实验名':<42}  {'peak_ID_b4':>10}  {'@step':>5}  {'peak_OOD_b4':>11}  {'@step':>5}  {'peak_m4':>8}  {'latest_step':>11}"
    print(header)
    print(f"  {'─'*42}  {'─'*10}  {'─'*5}  {'─'*11}  {'─'*5}  {'─'*8}  {'─'*11}")

    for exp_name, val_records, latest_step in exp_val_data:
        if not val_records:
            print(f"  {exp_name:<42}  {'no val data':>10}")
            continue

        peak_id_b4, peak_id_step = max(
            ((m["id_b4"], s) for s, m in val_records if m["id_b4"] is not None),
            default=(None, None),
        )
        peak_ood_b4, peak_ood_step = max(
            ((m["ood_b4"], s) for s, m in val_records if m.get("ood_b4") is not None),
            default=(None, None),
        )
        peak_m4 = max(
            (m["id_m4"] for _, m in val_records if m["id_m4"] is not None),
            default=None,
        )

        row = (
            f"  {exp_name:<42}  "
            f"{fmt(peak_id_b4, 4):>10}  "
            f"{str(peak_id_step) if peak_id_step else '—':>5}  "
            f"{fmt(peak_ood_b4, 4) if peak_ood_b4 is not None else '  —   ':>11}  "
            f"{str(peak_ood_step) if peak_ood_step else '—':>5}  "
            f"{fmt(peak_m4, 4):>8}  "
            f"{str(latest_step) if latest_step >= 0 else '—':>11}"
        )
        print(row)


# ─────────────────────────── 主程序 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VAGEN-Lite 实验分析工具")
    parser.add_argument(
        "--exps", required=True,
        help="逗号分隔的实验名列表（在 --expdir 下查找），如 v24_groupadv_100scenes,v25_groupadv_100scenes_klhi"
    )
    parser.add_argument(
        "--expdir", default=DEFAULT_EXPDIR,
        help=f"实验目录前缀，默认 {DEFAULT_EXPDIR}"
    )
    parser.add_argument("--n_id",  type=int, default=DEFAULT_N_ID,  help=f"ID val 环境数，默认 {DEFAULT_N_ID}")
    parser.add_argument("--n_ood", type=int, default=DEFAULT_N_OOD, help=f"OOD val 环境数，默认 {DEFAULT_N_OOD}")
    parser.add_argument("--val_n", type=int, default=DEFAULT_VAL_N, help=f"每环境 seed 数，默认 {DEFAULT_VAL_N}")
    parser.add_argument("--val",     action="store_true", default=True,  help="打印 val 曲线（默认开启）")
    parser.add_argument("--no_val",  action="store_true", help="关闭 val 曲线输出")
    parser.add_argument("--train",   action="store_true", help="打印 train 指标（默认关闭）")
    parser.add_argument("--summary", action="store_true", help="打印跨实验峰值对比表（默认开启）")
    parser.add_argument("--no_summary", action="store_true", help="关闭跨实验对比输出")
    parser.add_argument("--every",    type=int, default=20, help="train/diversity 指标每隔多少步打印一行，默认 20")
    parser.add_argument("--diversity", action="store_true", help="打印行为多样性指标（rollout_data/ 目录，默认关闭）")
    # ── 新增：per-task 学习曲线 & 失败模式 & 丰富主结果表 ─────────────────────
    parser.add_argument("--task_curve",    action="store_true",
                        help="打印 per-task-type 成功率随训练步骤变化曲线")
    parser.add_argument("--failure_modes", action="store_true",
                        help="打印失败模式分类统计（使用最新 val step）")
    parser.add_argument("--failure_step",  type=int, default=None,
                        help="指定用哪个 val step 做失败分析（默认取最新）")
    parser.add_argument("--score_metrics", action="store_true",
                        help="打印丰富主结果表（task_type × success/score_improvement/SPL/steps）。"
                             "score_improvement 和精确 SPL 需要新版 gym_agent_loop 日志字段；"
                             "旧 JSONL 也可运行，仅缺少 score 相关列。")
    args = parser.parse_args()

    show_val           = args.val and not args.no_val
    show_train         = args.train
    show_diversity     = args.diversity
    show_summary       = (args.summary or len(args.exps.split(",")) > 1) and not args.no_summary
    show_task_curve    = args.task_curve
    show_failure_modes = args.failure_modes
    show_score_metrics = args.score_metrics

    exp_names = [e.strip() for e in args.exps.split(",") if e.strip()]
    exp_val_data = []

    for exp_name in exp_names:
        expdir = os.path.join(args.expdir, exp_name)
        if not os.path.isdir(expdir):
            print(f"[WARN] 实验目录不存在: {expdir}", file=sys.stderr)
            exp_val_data.append((exp_name, [], -1))
            continue

        val_dir   = os.path.join(expdir, "validation")
        train_log = os.path.join(expdir, "train.log")

        val_records = parse_val_dir(val_dir, args.n_id, args.n_ood, args.val_n)
        latest_step = get_latest_step(expdir)

        if show_val:
            print_val_table(exp_name, val_records)

        if show_train:
            train_steps = parse_train_log(train_log)
            if train_steps:
                print_train_table(exp_name, train_steps, every=args.every)
            else:
                print(f"\n  [INFO] {exp_name}: train.log 未找到或无指标", file=sys.stderr)

        if show_diversity:
            rollout_dir = os.path.join(expdir, "rollout_data")
            diversity_records = parse_rollout_dir(rollout_dir, every=args.every)
            print_diversity_table(exp_name, diversity_records)

        # ── per-task 成功率学习曲线 ─────────────────────────────────────────
        if show_task_curve:
            task_val_records = parse_val_dir_per_task(val_dir, args.val_n, args.n_id)
            print_task_curve_table(exp_name, task_val_records)

        # ── 失败模式分析 ─────────────────────────────────────────────────────
        if show_failure_modes:
            # 选取指定 step 或最新 step 的 val 文件
            if not os.path.isdir(val_dir):
                print(f"  [WARN] {exp_name}: 无 validation/ 目录", file=sys.stderr)
            else:
                val_files = sorted(
                    [f for f in os.listdir(val_dir) if f.endswith(".jsonl")],
                    key=lambda x: int(x[:-6])
                )
                if val_files:
                    target_file = None
                    if args.failure_step is not None:
                        fname = f"{args.failure_step}.jsonl"
                        if fname in val_files:
                            target_file = os.path.join(val_dir, fname)
                        else:
                            print(f"  [WARN] step {args.failure_step} 不存在，使用最新", file=sys.stderr)
                    if target_file is None:
                        target_file = os.path.join(val_dir, val_files[-1])
                        used_step = int(val_files[-1][:-6])
                        print(f"  [失败分析使用 step={used_step}]")

                    mode_counts    = parse_failure_modes_from_val(target_file, args.val_n, args.n_id)
                    per_task_modes = parse_failure_modes_per_task(target_file, args.val_n, args.n_id)
                    print_failure_mode_table(exp_name, mode_counts, per_task_modes)

        # ── 丰富主结果表（task × success / score_improvement / SPL / steps）──
        if show_score_metrics:
            if not os.path.isdir(val_dir):
                print(f"  [WARN] {exp_name}: 无 validation/ 目录", file=sys.stderr)
            else:
                val_files = sorted(
                    [f for f in os.listdir(val_dir) if f.endswith(".jsonl")],
                    key=lambda x: int(x[:-6])
                )
                if val_files:
                    # Use specified step or latest
                    score_step = args.failure_step   # reuse --failure_step for step selection
                    target_file = None
                    if score_step is not None:
                        fname = f"{score_step}.jsonl"
                        if fname in val_files:
                            target_file = os.path.join(val_dir, fname)
                    if target_file is None:
                        target_file = os.path.join(val_dir, val_files[-1])
                        score_step  = int(val_files[-1][:-6])
                    print_rich_result_table(exp_name, score_step, target_file,
                                            args.val_n, args.n_id)

        exp_val_data.append((exp_name, val_records, latest_step))

    if show_summary:
        print_comparison_table(exp_val_data)

    print()


if __name__ == "__main__":
    main()
