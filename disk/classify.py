#!/usr/bin/env python3
"""Turn one xtree.tsv (disk/xtree.py) into the inventory tables of the disk line (read only).

    python disk/classify.py XTREE.tsv --out DIR

Writes DIR/top.tsv (every entry directly under the user folder: bytes, files, mtime, owner, line, kind),
DIR/lines.tsv (per line: the folders of its experiments, two levels below each line root),
DIR/checkpoints.tsv (every checkpoint-like folder: kind, bytes, mtime, line), DIR/duplicates.tsv (big files that
appear more than once: same size and name; hard links share one inode and cost nothing extra) and DIR/summary.json.

Line and kind come from path rules (LINE_RULES, KIND_RULES below); "unknown" means no rule matched and the item
has to be checked by hand ("先查清再报告").
"""
import argparse
import collections
import json
import os
import re
import time

U = "/mnt/umm/users/yinbaiqiao/"
E = U + "VAGEN-Lite/exps/vagen_active_spatial/"

# (prefix relative to U, line). The first match wins, so more specific prefixes come first.
LINE_RULES = [
    ("VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_", "主线"),
    ("VAGEN-Lite/exps/vagen_active_spatial/dense_score", "主线（旧 pilot，D7b）"),
    ("VAGEN-Lite/exps/vagen_active_spatial/R1-", "主线（R1，已归档线）"),
    ("VAGEN-Lite/exps/vagen_active_spatial/", "主线（其他实验目录）"),
    ("VAGEN-Lite/exps/", "主线（exps 其他）"),
    ("VAGEN-Lite/archive/", "整理窗口（归档区）"),
    ("VAGEN-Lite/models/", "整理窗口（固定位置 models/）"),
    ("VAGEN-Lite/third_party/starVLA/playground/Checkpoints", "整理窗口 D9（StarVLA NavigateKitchen）"),
    ("VAGEN-Lite/playground/Checkpoints", "整理窗口 D9（StarVLA NavigateKitchen）"),
    ("VAGEN-Lite/.venv-qwen3vl-runtime", "模型线（Qwen3-VL 运行环境）"),
    ("VAGEN-Lite/.venv-qwen3vl", "整理窗口 D4（旁支 venv）"),
    ("VAGEN-Lite/data_gen/active_spatial_qa/artifacts", "QA 线（R1 配对 QA 产物）"),
    ("VAGEN-Lite/", "主线代码 checkout（共用）"),
    ("VAGEN-Lite-r12dense", "主线（第二批代码副本）"),
    ("VAGEN-Lite-downstream", "下游（代码 worktree）"),
    ("downstream_runs", "下游"),
    ("envs/rlinf-starvla-rc365", "下游（环境）"),
    ("VAGEN-Lite-models", "模型线（代码 worktree）"),
    ("models_runs", "模型线"),
    ("cambrian-s", "模型线（Cambrian 源码）"),
    ("VAGEN-Lite-data", "数据线（transition-qa worktree）"),
    ("VAGEN-Lite-eval", "评测线（eval-external worktree）"),
    ("eval_external_assets", "评测线"),
    ("VAGEN-Lite-cleanup-wt", "整理窗口（worktree）"),
    ("VAGEN-Lite-disk", "磁盘线（本线）"),
    ("disk_runs", "磁盘线（本线）"),
    ("VAGEN-Lite-r4-envs", "主线（旧环境副本）"),
    ("probe_spatial/robocasa365", "下游（RoboCasa365 原始示教，借 probe_spatial 目录）"),
    ("probe_spatial", "probe_spatial 项目（不在本任务的线里）"),
    ("probe_spatial_video", "probe_spatial 项目（不在本任务的线里）"),
    ("InsScene-15K", "probe_spatial 项目（不在本任务的线里）"),
    ("tstep_aoss_reclaim", "probe_spatial 项目（不在本任务的线里）"),
    ("vagen_aoss_archive", "probe_spatial 项目（不在本任务的线里）"),
    ("InteriorGS_ply", "共用数据（主线依赖清单）"),
    ("InteriorGS_meta", "共用数据（主线依赖清单）"),
    ("InteriorGS", "共用数据（root 属主，不可读）"),
    ("gsplat_cuda_validated_20261007", "共用（主线依赖清单：渲染编译结果）"),
    ("aoss_archive_manifests", "主线（归档清单）"),
    ("robocasa365", "下游（RoboCasa365 源码）"),
    (".conda", "共用环境（conda）"),
    ("envs", "共用环境"),
    (".cache", "共用缓存（HF / pip 等）"),
    ("hf_cache", "共用缓存（hf_cache）"),
    ("nested_repos_from_VAGEN-Lite", "整理窗口 D8（从仓库移出的三个独立仓库）"),
]

KIND_RULES = [
    (r"/checkpoints/global_step_\d+$", "RL 检查点"),
    (r"/lf_[^/]+/model/checkpoint-\d+$", "SFT 检查点（LLaMA-Factory，每 epoch）"),
    (r"/lf_[^/]+/model$", "SFT 输出目录"),
    (r"/full_state/steps_\d+$", "下游全状态（续训用）"),
    (r"/full_state$", "下游全状态目录"),
    (r"/evals_src/steps_\d+$", "下游评测用权重副本"),
    (r"/run[^/]*/checkpoints$", "下游权重（可直接加载）"),
    (r"/(eval_val|eval_test|eval_)[^/]*$", "评测集与评测结果"),
    (r"/evals?[^/]*$", "评测结果"),
    (r"/rollout_data$", "训练采样记录"),
    (r"/frames$|/render", "渲染画面"),
    (r"/torch_extensions[^/]*$", "编译缓存"),
    (r"/(\.conda|envs|\.venv[^/]*|env)$", "Python 环境"),
    (r"/\.cache$|/hf_cache$|/huggingface$|/pip$", "缓存"),
    (r"/(logs?|wandb)$", "日志"),
    (r"/(tmp|_tmp|scratch)$", "临时文件"),
    (r"archive", "归档"),
]


def line_of(path):
    rel = path[len(U):] if path.startswith(U) else path
    for pre, line in LINE_RULES:
        if rel.startswith(pre):
            return line
    return "unknown"


def kind_of(path):
    for rx, kind in KIND_RULES:
        if re.search(rx, path):
            return kind
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xtree")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    with open(args.xtree) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            v = line.rstrip("\n").split("\t")
            r = dict(zip(header, v))
            for k in ("depth", "bytes", "files", "mtime", "uid", "nlink", "ino"):
                r[k] = int(r[k])
            rows.append(r)
    dirs = [r for r in rows if r["kind"] == "d"]
    files = [r for r in rows if r["kind"] == "f"]
    root = [r for r in dirs if r["depth"] == 0][0]
    fmt_t = lambda t: time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))

    # entries directly under the user folder (directories from xtree; plain files are summed as "files at top")
    top = [r for r in dirs if r["depth"] == 1]
    top_bytes = sum(r["bytes"] for r in top)
    with open(os.path.join(args.out, "top.tsv"), "w") as f:
        f.write("path\tbytes\tTB\tfiles\tmtime_dir\tuid\tline\n")
        for r in sorted(top, key=lambda r: -r["bytes"]):
            f.write("%s\t%d\t%.3f\t%d\t%s\t%d\t%s\n" % (r["path"][len(U):], r["bytes"], r["bytes"] / 1e12, r["files"],
                                                     fmt_t(r["mtime"]), r["uid"], line_of(r["path"])))

    by_line = collections.Counter()
    for r in top:
        by_line[line_of(r["path"])] += r["bytes"]
    # the shared checkout VAGEN-Lite holds several lines: split it one level down (exps by experiment folder)
    vl = [r for r in dirs if r["path"].startswith(U + "VAGEN-Lite/") and r["depth"] in (2, 4)
          and (r["depth"] == 2 and not r["path"].startswith(U + "VAGEN-Lite/exps")
               or r["depth"] == 4 and r["path"].startswith(E))]
    split = collections.Counter()
    for r in vl:
        split[line_of(r["path"])] += r["bytes"]

    # per-line experiment folders: two levels below each line root that is big enough to have been descended
    roots = [U + p for p in ("VAGEN-Lite/exps/vagen_active_spatial", "downstream_runs/experiments", "downstream_runs",
                             "models_runs", "models_runs/chains", "eval_external_assets", "VAGEN-Lite-data/exps",
                             "probe_spatial/artifacts", "probe_spatial", "InsScene-15K", "VAGEN-Lite/archive/cluster",
                             ".cache", ".conda/envs", "envs")]
    with open(os.path.join(args.out, "lines.tsv"), "w") as f:
        f.write("root\tpath\tbytes\tTB\tmtime_dir\tuid\tline\tkind\n")
        for base in roots:
            kids = [r for r in dirs if os.path.dirname(r["path"]) == base]
            for r in sorted(kids, key=lambda r: -r["bytes"]):
                f.write("%s\t%s\t%d\t%.3f\t%s\t%d\t%s\t%s\n" % (base[len(U):], r["path"][len(U):], r["bytes"], r["bytes"] / 1e12,
                                                              fmt_t(r["mtime"]), r["uid"], line_of(r["path"]), kind_of(r["path"])))

    ck = [r for r in dirs if kind_of(r["path"]) in ("RL 检查点", "SFT 检查点（LLaMA-Factory，每 epoch）", "下游全状态（续训用）",
                                                   "下游评测用权重副本", "下游权重（可直接加载）")]
    with open(os.path.join(args.out, "checkpoints.tsv"), "w") as f:
        f.write("path\tbytes\tGB\tmtime_dir\tline\tkind\n")
        for r in sorted(ck, key=lambda r: r["path"]):
            f.write("%s\t%d\t%.1f\t%s\t%s\t%s\n" % (r["path"][len(U):], r["bytes"], r["bytes"] / 1e9, fmt_t(r["mtime"]),
                                                  line_of(r["path"]), kind_of(r["path"])))

    groups = collections.defaultdict(list)
    for r in files:
        groups[(r["bytes"], os.path.basename(r["path"]))].append(r)
    dup_rows = []
    for (size, name), rs in groups.items():
        if len(rs) < 2:
            continue
        inodes = {r["ino"] for r in rs}
        extra = size * (len(inodes) - 1)
        dup_rows.append((extra, size, name, len(rs), len(inodes), [r["path"][len(U):] for r in rs]))
    dup_rows.sort(key=lambda x: -x[0])
    with open(os.path.join(args.out, "duplicates.tsv"), "w") as f:
        f.write("extra_bytes\tsize\tname\tcopies\tdistinct_inodes\tpaths\n")
        for d in dup_rows:
            f.write("%d\t%d\t%s\t%d\t%d\t%s\n" % (d[0], d[1], d[2], d[3], d[4], " | ".join(d[5])))

    summary = {"root_bytes": root["bytes"], "root_files": root["files"], "top_dirs_bytes": top_bytes,
               "top_level_files_bytes": root["bytes"] - top_bytes, "by_line_top": dict(by_line.most_common()),
               "vagen_lite_split": dict(split.most_common()), "big_files": len(files),
               "duplicate_extra_bytes": sum(d[0] for d in dup_rows), "checkpoint_dirs": len(ck),
               "checkpoint_bytes_by_kind": dict(collections.Counter({k: sum(r["bytes"] for r in ck if kind_of(r["path"]) == k)
                                                                     for k in {kind_of(r["path"]) for r in ck}}))}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
