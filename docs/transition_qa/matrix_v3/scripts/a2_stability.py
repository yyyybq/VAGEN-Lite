"""Analysis 2 (exploratory): stability of matrix v3's 22 RL runs by whether their SFT contained P1.

With P1: D, D_noP0 (8 runs).  Without: C, Cr, B', the old E (from C0's SFT, P2q only) (14 runs).  C0's 4 runs (the
model line's) ran without the entropy and image-token lines and with KL 0.2, so they are listed apart, outside the
test.  Per run: stopped by a stop line or not and which (each chain's progress.json guard_stops), trainer crashes
(trainer_crashes; the image-token ones by their message), the start's sampling-acceptance entropy (the picked SFT's
accept_summary), and the entropy and format-error means of each 25-update window (the chain's PROGRESS.md table).
Fisher's exact test (two-sided) on stopped x P1.  Writes OUT/a2.{json,md} and OUT/a2_entropy.png.

    python a2_stability.py OUT
"""
import json
import math
import re
import sys
from pathlib import Path

M3 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3")
GROUPS = {"c": ("C", False), "cr": ("Cr", False), "b": ("B'", False), "e": ("E（旧）", False), "d": ("D", True), "dn": ("D_noP0", True)}
START = {"C": (M3 / "accept_20261103", "sft_m3c_e4_s4"), "D": (M3 / "accept_20261103", "sft_m3d_e9_s4"),
         "E（旧）": (M3 / "accept_20261103", "c0q3_sft_e8_s4"), "B'": (M3 / "m3_b_sft_accept", "sft_m3b_e10_s4"),
         "D_noP0": (M3 / "m3_dn_sft_accept", "sft_m3dn_e8_s4"), "Cr": (M3 / "m3_cr_sft_accept", "sft_m3cr_e7_s4")}
ROW = re.compile(r"^\| (\d+)-(\d+)（\d+ 次） \|")


def accept_entropy(group: str):
    d, key = START[group]
    for p in (d / "accept_summary.json", d.parent / d.name.replace("_accept", "") / "accept_summary.json"):
        if p.exists():
            s = json.loads(p.read_text())
            if key in s:
                return float(s[key]["entropy"])
    return None


def windows(progress_md: str) -> tuple:
    """(run -> [(first, last, entropy, format errors)], run -> updates done) from the chain's per-run tables of
    25-update windows and their "已完成 N 次更新" lines."""
    out, done, run = {}, {}, None
    for line in progress_md.splitlines():
        m = re.match(r"^\*\*RL ([a-z0-9]+)：\*\* 已完成 (\d+) 次更新", line)
        if m:
            run = m.group(1)
            out.setdefault(run, [])
            done[run] = int(m.group(2))
            continue
        m = ROW.match(line)
        if m and run:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            try:
                out[run].append((int(m.group(1)), int(m.group(2)), float(cells[4]), float(cells[10])))
            except (ValueError, IndexError):
                pass
    return out, done


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """[[a, b], [c, d]]: rows P1 yes / no, columns stopped / ran on."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def p(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for ch in sorted(M3.glob("m3_*_s2026*")):
        g = ch.name.split("_")[1]
        if g not in GROUPS:
            continue
        group, p1 = GROUPS[g]
        prog = json.loads((ch / "progress.json").read_text())
        win, done = windows((ch / "PROGRESS.md").read_text(encoding="utf-8"))
        for run in json.loads((ch / "plan.json").read_text())["rl_runs"]:
            stop = (prog.get("guard_stops") or {}).get(run)
            crashes = (prog.get("trainer_crashes") or {}).get(run) or []
            why = None if not stop else ("熵线" if "entropy" in stop["reason"] else "读不出线" if "unparseable" in stop["reason"]
                                         else "图像 token 崩溃线" if "image-token" in stop["reason"] else stop["reason"][:40])
            w = win.get(run, [])
            runs.append({"group": group, "p1": p1, "chain": ch.name, "run": run, "side": "V4" if run.endswith("d") else "S0",
                         "updates": done.get(run), "stopped": bool(stop), "stop_line": why,
                         "crashes": len(crashes), "image_token_crashes": sum("shape mismatch" in (c.get("crash") or "") for c in crashes),
                         "start_entropy": accept_entropy(group), "windows": w,
                         "entropy_max_window": max((x[2] for x in w), default=None)})
    assert len(runs) == 22, len(runs)
    a = sum(r["stopped"] for r in runs if r["p1"])
    b = sum(not r["stopped"] for r in runs if r["p1"])
    c = sum(r["stopped"] for r in runs if not r["p1"])
    d = sum(not r["stopped"] for r in runs if not r["p1"])
    p = fisher_two_sided(a, b, c, d)
    rep = {"runs": runs, "table": {"p1_stopped": a, "p1_ran_on": b, "no_p1_stopped": c, "no_p1_ran_on": d}, "fisher_p": p}
    md = ["# 分析 2：稳不稳，按 SFT 含不含 P1（探索性）", "", __doc__.split("\n\n")[1].replace("\n", " "), "",
          f"**被停止线停下**：含 P1 的 {a} / {a + b}，不含 P1 的 {c} / {c + d}；Fisher 精确检验（双侧）p = {p:.4f}。", "",
          "| 组 | 含 P1 | RL 数 | 被停下 | 停下的线 | 训练任务崩溃（其中图像 token） | 起点采样验收的熵 | 最高一窗的熵（各 RL 平均） |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in [v[0] for v in GROUPS.values()]:
        rs = [r for r in runs if r["group"] == g]
        lines = {}
        for r in rs:
            if r["stop_line"]:
                lines[r["stop_line"]] = lines.get(r["stop_line"], 0) + 1
        ent = rs[0]["start_entropy"]
        md.append(f"| {g} | {'是' if rs[0]['p1'] else '否'} | {len(rs)} | {sum(r['stopped'] for r in rs)} | "
                  + ("、".join(f"{k} {v}" for k, v in lines.items()) or "—")
                  + f" | {sum(r['crashes'] for r in rs)}（{sum(r['image_token_crashes'] for r in rs)}） | "
                  + (f"{ent:.3f}" if ent is not None else "—") + " | "
                  + f"{sum(r['entropy_max_window'] for r in rs) / len(rs):.3f} |")
    md += ["", "## 每个 RL", "", "| 组 | RL | 跑到第几次 | 停下的线 | 崩溃（图像 token） | 熵：第 1–25 次 → 最后一窗 | 最高一窗 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        w = r["windows"]
        md.append(f"| {r['group']} | `{r['run']}` | {r['updates']} | {r['stop_line'] or '跑满'} | {r['crashes']}（{r['image_token_crashes']}） | "
                  + (f"{w[0][2]:.3f} → {w[-1][2]:.3f}" if w else "—") + f" | {r['entropy_max_window']:.3f} |")
    md += ["", "## 熵随更新次数（每 25 次一窗，组内平均；图在 `a2_entropy.png`）", ""]
    starts = sorted({x[0] for r in runs for x in r["windows"]})
    md += ["| 组 | " + " | ".join(f"{s}–{s + 24}" for s in starts) + " |", "| --- | " + " | ".join("---" for _ in starts) + " |"]
    for g in [v[0] for v in GROUPS.values()]:
        cells = []
        for s in starts:
            vals = [x[2] for r in runs if r["group"] == g for x in r["windows"] if x[0] == s]
            cells.append(f"{sum(vals) / len(vals):.3f}（{len(vals)}）" if vals else "—")
        md.append(f"| {g} | " + " | ".join(cells) + " |")
    md += ["", "括号里是还在跑的 RL 数：被停下的 RL 停下之后就没有数了。", "",
           "C0 的 4 个 RL（模型线）不在检验里：它们跑的时候没有熵线和图像 token 崩溃线，KL 线是 0.2；种子 1 的 S0 在第 150 次被读不出线停下，其余 3 个跑满。", ""]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in runs:
            if r["windows"]:
                ax.plot([(x[0] + x[1]) / 2 for x in r["windows"]], [x[2] for x in r["windows"]],
                        color="tab:blue" if r["p1"] else "tab:red", alpha=0.6, lw=1.2)
        ax.axhline(0.3, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("update"); ax.set_ylabel("entropy (25-update mean)")
        ax.set_title("matrix v3 RL: blue = SFT with P1 (D, D_noP0), red = without (C, Cr, B', old E)")
        fig.tight_layout(); fig.savefig(out / "a2_entropy.png", dpi=120)
        md.append("![熵](a2_entropy.png)")
    except Exception as exc:  # noqa: BLE001
        md.append(f"（没画图：{type(exc).__name__}）")
    (out / "a2.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "a2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
