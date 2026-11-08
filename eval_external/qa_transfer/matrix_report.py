"""Tables of the matrix-v3 QA transfer evaluation (task of 2026-11-05), from the runs' per-item lenient scores.

    python -m eval_external.qa_transfer.matrix_report --stage mv3 --batches 1 --out eval_external/results/matrix_v3_qa/batch1

Reads ROOT/runs/qa/<stage>/<model>__<bench>/{results.json, lenient.json, items_lenient.jsonl} (falling back to
stage step2 for the external reference SI-7B) and adapters/registry_matrix_v3.yaml (cells, batches, RL runs).
Writes OUT.md and OUT.json:

1. Accuracy, lenient scoring (qa_transfer/lenient.py), model x benchmark.  Proportions (every benchmark scored
   right / wrong per item) carry a Wilson 95% interval; VSI-Bench and VSTI-Bench mix multiple choice with numeric
   items scored by MRA, so their headline is not a proportion and the interval is a 95% bootstrap over items
   (resampled within question type, headline recomputed the benchmark's way).  SAT-Real circular = right in both
   option orders (SAT-Real and SAT-Real-swap).  MindCube-Tiny = EASI's 1,040 items (the public leaderboard's set),
   taken from the MindCube-Tiny-1050 run; MindCube-Tiny-1050 = MindCube's own 1,050.
2. Format: per model x benchmark the share of answers that could not be parsed at all, and of multiple-choice
   answers parsed but wrong; plus how many answers only the lenient rule could read.
3. Raw model against the public numbers (official scoring, the backend's own, which is what papers report).
4. Paired comparisons B - A on the same items, lenient scores: difference in points with a paired-bootstrap 95%
   interval, exact McNemar on the items scored right / wrong (for VSI / VSTI only their multiple-choice items), and
   the mean difference over the six benchmarks (SAT-Real, SAT-Syn, ViewSpatial, MindCube full, VSI, VSTI) with its
   bootstrap interval (each draw averages the six benchmarks' draws).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from eval_external.common.stats import wilson
from eval_external.diagnostics.compare_ownenv import mcnemar_exact
from eval_external.qa_transfer.benchmarks import MINDCUBE_TINY1050
from eval_external.qa_transfer.collect import VSTI_GROUPS
from eval_external.qa_transfer.paired import headline
from eval_external.sco.job import ROOT

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parent / "adapters" / "registry_matrix_v3.yaml"
N_BOOT = 2000
SEED = 20261106
SIX = ["SAT-Real", "SAT-Syn", "ViewSpatial", "MindCube", "VSI-Bench", "VSTI-Bench"]
# (column key, label, run benchmark, item filter, sub-item)
COLUMNS = [
    ("SAT-Real", "SAT-Real", "SAT-Real", None, None),
    ("SAT-Real-circ", "SAT-Real 轮换", "SAT-Real", "circular", None),
    ("SAT-Syn", "SAT-Syn", "SAT-Syn", None, None),
    ("ViewSpatial", "ViewSpatial", "ViewSpatial", None, None),
    ("MindCube-Tiny", "MindCube-Tiny (1,040)", "MindCube-Tiny-1050", "easi1040", None),
    ("MindCube-Tiny-1050", "MindCube-Tiny (1,050)", "MindCube-Tiny-1050", None, None),
    ("MindCube", "MindCube 全集", "MindCube", None, None),
    ("VSI-Bench", "VSI", "VSI-Bench", None, None),
    ("VSI-route", "VSI 路线规划", "VSI-Bench", None, "route_planning"),
    ("VSTI-Bench", "VSTI", "VSTI-Bench", None, None),
    ("VSTI-disp", "VSTI 相机位移", "VSTI-Bench", None, "camera_displacement"),
    ("MMSI", "MMSI", "MMSI", None, None),
]
RULES = {"VSI-Bench": "vsi8", "VSTI-Bench": "vsti5", "VSTI-Bench-prefix": "vsti5"}
# public numbers for the raw Qwen3-VL-8B-Instruct (official scoring); the first source of each is the check
PUBLIC = {
    "VSI-Bench": [("EASI 榜 (32 帧)", 55.29), ("Spatial-Interactor 论文", 55.7), ("EASI / SenseNova-SI 论文", 57.9),
                  ("Qwen3-VL 技术报告", 59.4)],
    "VSI-route": [("EASI 榜", 32.47), ("Spatial-Interactor 论文", 32.0), ("EASI / SenseNova-SI 论文", 35.05)],
    "MMSI": [("EASI 榜", 31.1), ("Spatial-Interactor 论文", 30.8)],
    "MindCube-Tiny": [("EASI 榜 (1,040 题)", 29.42), ("Spatial-Interactor 论文", 29.4)],
    "ViewSpatial": [("EASI 榜", 42.2), ("Spatial-Interactor 论文", 40.3)],
    "SAT-Real": [("Spatial-Interactor 论文", 52.7)],
    "SAT-Syn": [("Spatial-Interactor 论文 (低于二选一的随机水平 50)", 43.6)],
    "VSTI-Bench-prefix": [("Spatial-Interactor 论文 (修正前标签)", 43.3)],
    "VSTI-prefix-disp": [("Spatial-Interactor 论文 (修正前标签)", 24.2)],
}
LABEL = {"qwen3vl-8B": "原始模型 Qwen3-VL-8B", "SI-7B": "Spatial-Interactor-7B (外部参照)"}


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text())


def run_dir(stage: str, model: str, bench: str) -> Path | None:
    for st in (stage, "step2"):
        d = ROOT / "runs" / "qa" / st / f"{model}__{bench}"
        if (d / "items_lenient.jsonl").exists():
            return d
    return None


def load_items(stage: str, model: str, bench: str) -> dict[str, dict] | None:
    d = run_dir(stage, model, bench)
    if d is None:
        return None
    return {r["id"]: r for r in map(json.loads, open(d / "items_lenient.jsonl"))}


def easi_1040() -> set[str]:
    info = json.loads((ROOT / "LMUData" / f"{MINDCUBE_TINY1050}.json").read_text())
    import pandas as pd

    ids = pd.read_csv(ROOT / "LMUData" / "MindCubeBench_tiny_raw_qa.tsv", sep="\t", usecols=["index"])["index"]
    out = set(map(str, ids))
    assert len(out) == 1040 and not out & set(info["added"])
    return out


class Data:
    """Per (model, column) the item scores used everywhere: {id: (type, score, mcq)}."""

    def __init__(self, stage: str):
        self.stage = stage
        self.cache: dict = {}
        self.tiny = easi_1040()

    def items(self, model: str, col: tuple, key: str = "lenient") -> dict | None:
        ck = (model, col[0], key)
        if ck in self.cache:
            return self.cache[ck]
        _, _, bench, filt, sub = col
        out = None
        if filt == "circular":
            a, b = load_items(self.stage, model, "SAT-Real"), load_items(self.stage, model, "SAT-Real-swap")
            if a and b:
                out = {i: ("all", min(a[i][key], b[i][key]), True) for i in a if i in b}
        else:
            src = load_items(self.stage, model, bench)
            if src is None and filt == "easi1040":  # models run on the 1,040-item table only (SI-7B, step 2)
                src = load_items(self.stage, model, "MindCube-Tiny")
            if src is not None:
                out = {i: (r["type"], r[key], r["mcq"]) for i, r in src.items()
                       if (filt != "easi1040" or i in self.tiny)}
                if sub is not None:
                    keep = set(VSTI_GROUPS.get(sub, [sub]))
                    out = {i: v for i, v in out.items() if v[0] in keep}
        self.cache[ck] = out
        return out


def rule_of(col: tuple) -> str:
    return "items" if col[4] is not None else RULES.get(col[2], "items")


def score(items: dict, rule: str) -> float:
    sums, ns = defaultdict(float), defaultdict(int)
    for t, s, _ in items.values():
        sums[t] += s
        ns[t] += 1
    return 100 * headline(rule, dict(sums), dict(ns))


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def boot_draws(a: dict, b: dict | None, rule: str, tag: tuple) -> tuple[np.ndarray, np.ndarray | None]:
    """Bootstrap headline draws of A (and B on the same resampled items), resampling within question type."""
    ids = sorted(set(a) & set(b)) if b is not None else sorted(a)
    types = sorted({a[i][0] for i in ids})
    rng = _rng(*tag)
    ns = {t: sum(1 for i in ids if a[i][0] == t) for t in types}
    arr_a = {t: np.array([a[i][1] for i in ids if a[i][0] == t]) for t in types}
    arr_b = {t: np.array([b[i][1] for i in ids if a[i][0] == t]) for t in types} if b is not None else None
    sa = {t: np.empty(N_BOOT) for t in types}
    sb = {t: np.empty(N_BOOT) for t in types} if b is not None else None
    for start in range(0, N_BOOT, 200):
        k = min(200, N_BOOT - start)
        for t in types:
            idx = rng.integers(0, ns[t], size=(k, ns[t]))
            sa[t][start:start + k] = arr_a[t][idx].sum(axis=1)
            if b is not None:
                sb[t][start:start + k] = arr_b[t][idx].sum(axis=1)
    ha = 100 * headline(rule, sa, ns)
    hb = 100 * headline(rule, sb, ns) if b is not None else None
    return ha, hb


def binary(items: dict) -> bool:
    return all(s in (0.0, 1.0) for _, s, _ in items.values())


def interval(col: tuple, items: dict, model: str) -> tuple[float, float, str]:
    rule = rule_of(col)
    if binary(items) and rule == "items":
        k, n = sum(s for _, s, _ in items.values()), len(items)
        _, lo, hi = wilson(k, n)
        return 100 * lo, 100 * hi, "Wilson"
    draws, _ = boot_draws(items, None, rule, ("ci", model, col[0]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), "bootstrap"


def compare(data: Data, ma: str, mb: str, col: tuple) -> dict | None:
    a, b = data.items(ma, col), data.items(mb, col)
    if not a or not b:
        return None
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b):
        raise SystemExit(f"{ma} vs {mb} {col[0]}: item sets differ ({len(a)}, {len(b)}, common {len(ids)})")
    rule = rule_of(col)
    da, db = boot_draws(a, b, rule, ("pair", ma, mb, col[0]))
    diff = db - da
    out = {"bench": col[0], "a": round(score(a, rule), 2), "b": round(score(b, rule), 2), "n": len(ids),
           "diff": round(score(b, rule) - score(a, rule), 2),
           "ci95": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)],
           "draws": diff}
    mcq = [i for i in ids if a[i][2] and a[i][1] in (0.0, 1.0) and b[i][1] in (0.0, 1.0)]
    only_a = sum(1 for i in mcq if a[i][1] == 1 and b[i][1] == 0)
    only_b = sum(1 for i in mcq if a[i][1] == 0 and b[i][1] == 1)
    out.update({"mcnemar_items": len(mcq), "only_a": only_a, "only_b": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b),
                "mcnemar_scope": "all items" if len(mcq) == len(ids) else "multiple-choice items"})
    return out


def comparisons(reg: dict, have: set[str]) -> list[tuple[str, str, str, str]]:
    """(effect, label, A, B) the user asked for, where both models have runs."""
    m = reg["models"]
    sft = {v["cell"]: k for k, v in m.items() if v["kind"] == "sft"}
    raw = next(k for k, v in m.items() if v["kind"] == "raw")
    want = [("P0", "C − C0", sft.get("C0"), sft.get("C")), ("P0", "D − D_noP0", sft.get("D_noP0"), sft.get("D")),
            ("P1", "D − C", sft.get("C"), sft.get("D")), ("P1", "D_noP0 − C0", sft.get("C0"), sft.get("D_noP0")),
            ("动作 SFT", "C0 − 原始模型", raw, sft.get("C0"))]
    for r in reg["matrix"]["runs"]:
        if r["archived"] or r["prereg_is_sft"]:
            continue
        want.append(("RL", f"{r['prereg']} − {r['sft']} ({r['cell']} {r['side']} seed {r['seed']})", r["sft"], r["prereg"]))
    return [w for w in want if w[2] in have and w[3] in have]


def fmt(x: float | None, nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}"


def build(stage: str, batches: list[int], extra: list[str]) -> dict:
    reg = load_registry()
    models = [c for b in batches for c in reg["matrix"]["batches"][b]] + extra
    data = Data(stage)
    table, fmt_rows, ci_kind = {}, {}, {}
    for mdl in models:
        row, frow = {}, {}
        for col in COLUMNS:
            items = data.items(mdl, col)
            if not items:
                continue
            lo, hi, kind = interval(col, items, mdl)
            ci_kind[col[0]] = kind
            off = data.items(mdl, col, "official")
            row[col[0]] = {"lenient": round(score(items, rule_of(col)), 2), "ci95": [round(lo, 1), round(hi, 1)],
                           "official": round(score(off, rule_of(col)), 2), "n": len(items)}
        for bench in ("SAT-Real", "SAT-Real-swap", "SAT-Syn", "ViewSpatial", "MindCube-Tiny-1050", "MindCube",
                      "VSI-Bench", "VSTI-Bench", "MMSI", "VSTI-Bench-prefix"):
            d = run_dir(stage, mdl, bench) or (run_dir(stage, mdl, "MindCube-Tiny") if bench == "MindCube-Tiny-1050"
                                               else None)
            if d is None:
                continue
            le = json.loads((d / "lenient.json").read_text())
            frow[bench] = {"n": le["n_items"], "unparseable": le["n_unparseable"], "n_mcq": le["n_mcq"],
                           "mcq_wrong": le["n_mcq_wrong_parsed"], "status": le["status"],
                           "official": le["official"]["overall"], "lenient": le["lenient"]["overall"]}
        six = [row[b]["lenient"] for b in SIX if b in row]
        row["six_mean"] = round(sum(six) / 6, 2) if len(six) == 6 else None
        table[mdl], fmt_rows[mdl] = row, frow
    raw = next(k for k, v in reg["models"].items() if v["kind"] == "raw")
    repro = {}
    if raw in table:
        for key, refs in PUBLIC.items():
            if key in ("VSTI-Bench-prefix", "VSTI-prefix-disp"):
                col = ("VSTI-Bench-prefix", "", "VSTI-Bench-prefix", None,
                       None if key == "VSTI-Bench-prefix" else "camera_displacement")
            else:
                col = next(c for c in COLUMNS if c[0] == key)
            off = data.items(raw, col, "official")
            if not off:
                continue
            ours = score(off, rule_of(col))
            repro[key] = {"ours_official": round(ours, 2), "ours_lenient": round(score(data.items(raw, col), rule_of(col)), 2),
                          "refs": [{"source": s, "value": v, "diff": round(ours - v, 2)} for s, v in refs],
                          "over_1_5": abs(ours - refs[0][1]) > 1.5}
    pairs = []
    for effect, label, ma, mb in comparisons(reg, set(table)):
        rows = {}
        for col in COLUMNS:
            c = compare(data, ma, mb, col)
            if c:
                rows[col[0]] = c
        six = [rows[b] for b in SIX if b in rows]
        avg = None
        if len(six) == 6:
            draws = sum(r["draws"] for r in six) / 6
            avg = {"diff": round(sum(r["diff"] for r in six) / 6, 2),
                   "ci95": [round(float(np.percentile(draws, 2.5)), 2), round(float(np.percentile(draws, 97.5)), 2)]}
        for r in rows.values():
            r.pop("draws")
        pairs.append({"effect": effect, "label": label, "a": ma, "b": mb, "per_bench": rows, "six_mean": avg})
    return {"stage": stage, "batches": batches, "models": models, "table": table, "format": fmt_rows,
            "ci_kind": ci_kind, "repro": repro, "pairs": pairs,
            "registry": {"sources": reg["sources"], "generated_utc": reg["generated_utc"]}}


def label(reg: dict, code: str) -> str:
    m = reg["models"].get(code)
    if code in LABEL:
        return LABEL[code]
    if m is None:
        return code
    return f"{m['cell']} {'SFT' if m['kind'] == 'sft' else 'RL ' + m['side'] + ' s' + str(m['seed'])[-2:]} ({code})"


def markdown(res: dict) -> str:
    reg = load_registry()
    L = []
    cols = [c for c in COLUMNS if any(c[0] in res["table"][m] for m in res["models"])]
    L += ["## 1. 准确率（宽松解析，%）", "",
          "每格：分数 [95% 区间]。区间：按对错计分的 benchmark 用 Wilson 区间；VSI 和 VSTI 含 MRA 计分的数值题，"
          "总分不是比例，改用题目 bootstrap 区间（题型内重抽 2000 次）。六项平均 = SAT-Real、SAT-Syn、ViewSpatial、"
          "MindCube 全集、VSI、VSTI 的平均。", "",
          "| 模型 | " + " | ".join(c[1] for c in cols) + " | 六项平均 |",
          "|---|" + "---|" * (len(cols) + 1)]
    for m in res["models"]:
        row = res["table"][m]
        cells = []
        for c in cols:
            v = row.get(c[0])
            cells.append("—" if v is None else f"{v['lenient']:.1f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]")
        L.append(f"| {label(reg, m)} | " + " | ".join(cells) + f" | {fmt(row.get('six_mean'))} |")
    L += ["", "题数：" + "；".join(f"{c[1]} {res['table'][res['models'][0]][c[0]]['n']}" for c in cols
                                if c[0] in res["table"][res["models"][0]]), ""]
    L += ["## 2. 格式：解析不出 vs 答错", "",
          "每格：解析不出的回答占全部题的 % / 选择题里解析出但答错的 % ；括号里是只有宽松规则才认出的回答数"
          "（选项原文或其他字母写法），以及官方计分 → 宽松计分。", "",
          "| 模型 | benchmark | 题数 | 解析不出 | 答错（选择题） | 宽松才认出 | 官方 → 宽松 |", "|---|---|---|---|---|---|---|"]
    for m in res["models"]:
        for bench, f in res["format"][m].items():
            extra_read = f["status"].get("text", 0) + f["status"].get("letter", 0) + f["status"].get("number", 0)
            L.append(f"| {label(reg, m)} | {bench} | {f['n']} | {100 * f['unparseable'] / f['n']:.1f}% "
                     f"({f['unparseable']}) | {100 * f['mcq_wrong'] / max(f['n_mcq'], 1):.1f}% | {extra_read} | "
                     f"{f['official']:.1f} → {f['lenient']:.1f} |")
    if res["repro"]:
        L += ["", "## 3. 原始模型对公开数（官方计分）", "",
              "| benchmark | 我们（官方计分） | 我们（宽松） | 公开数：来源 = 值（我们 − 公开） | 与第一个参照差 > 1.5？ |",
              "|---|---|---|---|---|"]
        for k, r in res["repro"].items():
            refs = "；".join(f"{x['source']} = {x['value']}（{x['diff']:+.1f}）" for x in r["refs"])
            L.append(f"| {k} | {r['ours_official']:.1f} | {r['ours_lenient']:.1f} | {refs} | "
                     f"{'**是**' if r['over_1_5'] else '否'} |")
    if res["pairs"]:
        L += ["", "## 4. 配对比较（B − A，宽松计分，点）", "",
              "每格：差 [95% 配对 bootstrap 区间]，McNemar p（只有 A 对 / 只有 B 对；VSI、VSTI 只算选择题）。", ""]
        bcols = [c for c in cols if any(c[0] in p["per_bench"] for p in res["pairs"])]
        L += ["| 效应 | B − A | " + " | ".join(c[1] for c in bcols) + " | 六项平均差 |",
              "|---|---|" + "---|" * (len(bcols) + 1)]
        for p in res["pairs"]:
            cells = []
            for c in bcols:
                r = p["per_bench"].get(c[0])
                cells.append("—" if r is None else f"{r['diff']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] "
                                                   f"p={fmt_p(r['mcnemar_p'])} ({r['only_a']}/{r['only_b']})")
            s = p["six_mean"]
            six = "—" if s is None else f"{s['diff']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
            L.append(f"| {p['effect']} | {p['label']} | " + " | ".join(cells) + f" | {six} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="mv3")
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--extra", nargs="*", default=["SI-7B"], help="further model codes (external reference)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = build(args.stage, args.batches, args.extra)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    md = markdown(res)
    args.out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
