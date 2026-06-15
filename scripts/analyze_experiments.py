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
    # 全部输出
    python3 scripts/analyze_experiments.py \
        --exps v24_groupadv_100scenes,v25_groupadv_100scenes_klhi,v25_groupadv_100scenes_cosine \
        --train --val --summary
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


# ─────────────────────────── 打印工具 ───────────────────────────

def fmt(v, decimals=4):
    if v is None:
        return "  —   "
    return f"{v:.{decimals}f}"


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
    args = parser.parse_args()

    show_val       = args.val and not args.no_val
    show_train     = args.train
    show_diversity = args.diversity
    show_summary   = (args.summary or len(args.exps.split(",")) > 1) and not args.no_summary

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

        exp_val_data.append((exp_name, val_records, latest_step))

    if show_summary:
        print_comparison_table(exp_val_data)

    print()


if __name__ == "__main__":
    main()
