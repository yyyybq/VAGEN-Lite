"""The report page: one self-contained HTML file (figures and example strips embedded), Chinese text, light and dark
themes.  Written without <html>/<head>/<body> so it can be published as an Artifact as is; browsers open it as a
local file too."""
from __future__ import annotations

import base64
import html
import math
from pathlib import Path

from . import stats
from .collect import RATIO_PERCENT

E = html.escape

CSS = """
/* Layout: one reading column of a lab report; figures sit on light paper cards in both themes. */
:root {
  --bg: #f4f5f2; --surface: #ffffff; --ink: #1d242b; --muted: #5b6570; --line: #dde1dc; --accent: #1d6b72;
  --accent-soft: #e2efee; --paper: #ffffff; --paper-ink: #4a535c; --good: #2f7d4f; --bad: #b4412f; --warn-bg: #fff3d6; --warn-ink: #6b4a00;
  --c-B: #7a8291; --c-S: #3569b5; --c-RS: #d9822b; --c-RD: #2f9a6a;
  --f-body: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  --f-head: "Noto Serif SC", "Songti SC", "SimSun", serif;
  --f-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121614; --surface: #1a201d; --ink: #e5ebe7; --muted: #9aa69f; --line: #2b332f; --accent: #6fbfc4;
  --accent-soft: #1f3335; --paper: #f3f5f2; --paper-ink: #4a535c; --good: #6cc191; --bad: #ef8a74; --warn-bg: #3a3017; --warn-ink: #f4d78a;
  --c-B: #9aa1ad; --c-S: #6b9be0; --c-RS: #eba45c; --c-RD: #5cc28f; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #121614; --surface: #1a201d; --ink: #e5ebe7; --muted: #9aa69f; --line: #2b332f; --accent: #6fbfc4;
  --accent-soft: #1f3335; --paper: #f3f5f2; --paper-ink: #4a535c; --good: #6cc191; --bad: #ef8a74; --warn-bg: #3a3017; --warn-ink: #f4d78a;
  --c-B: #9aa1ad; --c-S: #6b9be0; --c-RS: #eba45c; --c-RD: #5cc28f; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 15px; line-height: 1.7; }
.wrap { max-width: 1060px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header.top { display: grid; gap: 10px; padding-bottom: 18px; border-bottom: 1px solid var(--line); }
h1, h2, h3 { font-family: var(--f-head); font-weight: 600; text-wrap: balance; line-height: 1.3; margin: 0; }
h1 { font-size: 1.9rem; letter-spacing: 0.01em; }
h2 { font-size: 1.35rem; padding-top: 6px; }
h3 { font-size: 1.05rem; }
.meta { color: var(--muted); font-size: 0.85rem; display: flex; flex-wrap: wrap; gap: 4px 18px; }
.meta code, code, .num { font-family: var(--f-data); font-size: 0.85em; }
.num, td.n { font-variant-numeric: tabular-nums; }
nav.toc { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
nav.toc a { font-size: 0.85rem; color: var(--accent); text-decoration: none; border: 1px solid var(--line);
  border-radius: 999px; padding: 2px 10px; background: var(--surface); }
nav.toc a:hover, nav.toc a:focus-visible { border-color: var(--accent); outline: none; }
.banner { background: var(--warn-bg); color: var(--warn-ink); border-radius: 8px; padding: 10px 14px; font-weight: 600; }
section { display: grid; gap: 14px; padding-top: 30px; }
section > p, .prose { max-width: 68ch; margin: 0; }
.lede { font-size: 1.02rem; }
ul.facts { margin: 0; padding-left: 1.2em; display: grid; gap: 6px; max-width: 76ch; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 6px; vertical-align: -1px; background: var(--chip); }
figure.fig { margin: 0; background: var(--paper); border: 1px solid var(--line); border-radius: 6px; padding: 10px;
  overflow-x: auto; }
figure.fig img { display: block; max-width: 100%; height: auto; margin: 0 auto; }
figure.fig figcaption { color: var(--paper-ink); font-size: 0.85rem; padding-top: 8px; max-width: 80ch; }
.empty { color: var(--muted); border: 1px dashed var(--line); border-radius: 6px; padding: 14px; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; white-space: nowrap; }
th { font-weight: 600; color: var(--muted); background: var(--accent-soft); }
td.n, th.n { text-align: right; font-family: var(--f-data); }
tr:last-child td { border-bottom: 0; }
.up { color: var(--good); } .down { color: var(--bad); }
.tag { font-size: 0.75rem; border-radius: 4px; padding: 0 6px; border: 1px solid var(--line); color: var(--muted); }
.tag.spatial { color: var(--bad); border-color: var(--bad); }
dl.glossary { display: grid; grid-template-columns: minmax(0, 13em) minmax(0, 1fr); gap: 10px 20px; margin: 0; }
dl.glossary dt { font-weight: 600; }
dl.glossary dd { margin: 0; color: var(--ink); max-width: 70ch; }
@media (max-width: 640px) { dl.glossary { grid-template-columns: minmax(0, 1fr); } dl.glossary dd { padding-bottom: 6px; } }
.example { display: grid; gap: 8px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px; }
.example .head { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline; }
.example .strip { display: grid; grid-template-columns: 4.5em minmax(0, 1fr); gap: 4px 10px; align-items: center; }
.example .strip img { width: 100%; max-width: 100%; height: auto; border-radius: 3px; background: var(--paper); }
.example .who { font-family: var(--f-data); font-size: 0.8rem; }
.ok { color: var(--good); font-weight: 600; } .fail { color: var(--bad); font-weight: 600; }
footer { color: var(--muted); font-size: 0.8rem; padding-top: 36px; }
@media (prefers-reduced-motion: no-preference) { nav.toc a { transition: border-color .15s; } }
"""

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
         'family=Noto+Sans+SC:wght@400;600&family=Noto+Serif+SC:wght@600&display=swap">')

GLOSSARY = [
    ("回合", "一次闭环评测：环境按评测种子重置出一个厨房和任务，策略每次输出 16 步动作，直到成功或到任务时限。"),
    ("成功率", "成功回合数 ÷ 回合数。一组任务的成功率把这组所有回合合在一起算（空间组 9 个任务各 50 回合，其余任务各 20 回合）。"),
    ("95% Wilson 区间", "成功率的不确定范围：回合数越少越宽。比如 20 回合里成功 10 个，区间约 30%–70%。图里的浅色带就是它。"),
    ("配对", "两个起点在完全相同的回合上比较：同一任务、同一评测种子（所以初始场景逐位相同）。配对能去掉场景难易带来的差别。"),
    ("逐任务配对检验（主要检验）", "配对单位是任务（10-22 起的规定）：每个任务算一个成功率差（某起点减 B，同一批场景），34 个任务就有 34 个差。平均差的区间是对任务重抽样 2,000 次得到的 95% 区间。"),
    ("Wilcoxon 符号秩检验", "检验这 34 个差是否整体偏向一边：既看正负，也看大小的排序。p 越小，越不像是任务间的随机起伏。"),
    ("符号检验", "只数差为正和差为负的任务个数，如果两个起点其实一样好，两者应该差不多；p 是在这个前提下出现这么不平衡（或更不平衡）的概率。"),
    ("评测噪声下限", "同一检查点用两个任务各评一遍、种子相同：两次之间的成功率差、逐任务差的平均绝对值、结果不同的回合数，都是评测本身带来的。起点之间的差小于这个下限，就分不出来。"),
    ("McNemar 精确检验（参考）", "在一个任务内部按回合配对：只看两者结果不一致的回合。逐任务表和逐任务图里用它（Holm 校正后）标出单个任务上的差别，作参考。"),
    ("结果一致性", "配对回合分成四类：两个都成功、只有这个起点成功、只有 B 成功、两个都失败。"),
    ("成功所用步数 / 时限", "只看成功的回合：用了多少控制步，除以该任务的时限。越小表示完成得越快。"),
    ("动作损失（L1）", "训练时模型预测的 16 步动作与示教动作的平均绝对误差，每 100 步记一次当步那批数据的值。同一训练种子下四个起点看到的是同一批数据，所以可以逐步相减。它衡量对示教的模仿，不等于闭环成功率。"),
    ("中间检查点", "D5 每 2.5 万步存一个检查点。学习率按总步数 20 万余弦衰减，中间检查点时学习率还高，所以它们只用来看成功率何时离开地板，不能当作更短训练的结果。"),
    ("任务分组", "单步 18 个 / 多步 16 个是 RoboCasa 的分法。空间组 9 个：主指令要靠空间关系定目标或要导航（如左右抽屉、前后灶头、杯子放到出水口下面）。边界 3 个单独列出、不算入空间组。其余 22 个。"),
]


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def pts(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}"


def pval(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def img_tag(path: Path, alt: str, mime: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:{mime};base64,{data}" alt="{E(alt)}">'


def figure(fig_path, caption: str, alt: str) -> str:
    if not fig_path:
        return f'<p class="empty">{E(caption)}（数据还没有。）</p>'
    return f'<figure class="fig">{img_tag(fig_path, alt, "image/png")}<figcaption>{E(caption)}</figcaption></figure>'


def legend(starts) -> str:
    names = {"B": "B 原始 7B", "S": "S 动作 SFT 第 9 个 epoch", "RS": "RS P-S0 第 300 次", "RD": "RD P-D 第 150 次"}
    return '<div class="legend">' + "".join(
        f'<span style="--chip: var(--c-{s}, #888)">{E(names.get(s, s))}</span>' for s in starts) + "</div>"


def group_table(cmp) -> str:
    if not cmp or not cmp["starts"]:
        return '<p class="empty">还没有可比较的评测。</p>'
    starts, ref = cmp["starts"], cmp["reference"]
    head = "".join(f'<th class="n">{E(s)} 成功率（区间）</th>' for s in starts) + "".join(
        f'<th class="n">{E(s)} − {E(ref)}：逐任务平均差（区间）</th><th class="n">任务 ↑↓ · Wilcoxon p · 符号 p</th>'
        for s in starts if s != ref)
    rows = []
    for g in stats.GROUPS:
        cells = []
        for s in starts:
            k, n = cmp["groups"][s][g]
            r, lo, hi = stats.wilson(k, n)
            cells.append(f'<td class="n">{k}/{n} = {pct(r)}<br><span class="meta">{pct(lo)}–{pct(hi)}</span></td>')
        for s in starts:
            if s == ref or s not in cmp["vs_reference"]:
                continue
            v = cmp["vs_reference"][s]["groups"][g]
            sig = v["wilcoxon_p"] < 0.05 if not math.isnan(v["wilcoxon_p"]) else False
            cls = "up" if sig and v["task_mean_diff"] > 0 else "down" if sig and v["task_mean_diff"] < 0 else ""
            cells.append(f'<td class="n {cls}">{pts(v["task_mean_diff"])}<br><span class="meta">{pts(v["task_low"])} 到 '
                         f'{pts(v["task_high"])}</span></td><td class="n">{v["tasks_up"]}↑ {v["tasks_down"]}↓<br>'
                         f'<span class="meta">{pval(v["wilcoxon_p"])} · {pval(v["sign_p"])}</span></td>')
        rows.append(f"<tr><td>{E(stats.GROUP_ZH[g])}</td>{''.join(cells)}</tr>")
    return (f'<div class="tablewrap"><table><thead><tr><th>任务组</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="meta">配对单位是任务：每个任务一个成功率差（同一批场景），差值单位百分点；区间是对任务重抽样的 95% 区间；'
            f'↑↓：差为正 / 为负的任务个数；Wilcoxon：符号秩检验；符号：只看正负个数的检验。绿色 / 红色：Wilcoxon p &lt; 0.05 '
            f'的提高 / 降低。和它比较的噪声下限见"评测噪声"一节。</p>')


def task_table(coll, cmp, order) -> str:
    if not cmp or not cmp["starts"]:
        return ""
    starts, ref = cmp["starts"], cmp["reference"]
    others = [s for s in starts if s in cmp["vs_reference"]]
    head = "".join(f'<th class="n">{E(s)}</th>' for s in starts) + "".join(
        f'<th class="n">{E(s)}−{E(ref)}</th><th class="n">Holm p</th>' for s in others)
    rows = []
    for t in order:
        info = coll.tasks[t]
        tag = {"spatial": '<span class="tag spatial">空间</span>', "borderline": '<span class="tag">边界</span>'}.get(info.klass, "")
        cells = "".join(f'<td class="n">{cmp["tasks"][s].get(t, [0, 0])[0]}/{cmp["tasks"][s].get(t, [0, 0])[1]}</td>'
                        for s in starts)
        for s in others:
            v = cmp["vs_reference"][s]["tasks"].get(t)
            if not v:
                cells += "<td></td><td></td>"
                continue
            cls = "up" if v["diff"] > 0 and v["p_holm"] < 0.05 else "down" if v["diff"] < 0 and v["p_holm"] < 0.05 else ""
            cells += f'<td class="n {cls}">{pts(v["diff"])}</td><td class="n">{pval(v["p_holm"])}</td>'
        rows.append(f"<tr><td>{E(t)} {tag}</td><td>{'单步' if info.group == 'atomic_seen' else '多步'}</td>{cells}</tr>")
    return (f'<div class="tablewrap"><table><thead><tr><th>任务</th><th>组</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def seed_tables(results) -> str:
    st = results.get("seed_tables") or {}
    if not any(st.values()):
        return '<p class="empty">第三步各格的最后一个检查点还没有评测结果。</p>'
    starts = sorted({s for v in st.values() for s in v}, key=lambda s: ("B", "S", "RS", "RD").index(s) if s in ("B", "S", "RS", "RD") else 9)
    rows = []
    for g in ("all", "spatial", "other", "atomic_seen", "composite_seen", "borderline"):
        for r in ("010", "030", "100"):
            if not st.get(r):
                continue
            cells = []
            for s in starts:
                m = st[r].get(s, {}).get(g)
                cells.append(f'<td class="n">{pct(m[0])} ± {pct(m[1]).rstrip("%") if m and not math.isnan(m[1]) else "—"}'
                             f' <span class="meta">({m[2]})</span></td>' if m and m[2] else '<td class="n">—</td>')
            rows.append(f"<tr><td>{E(stats.GROUP_ZH[g])}</td><td class='n'>{RATIO_PERCENT[r]}%</td>{''.join(cells)}</tr>")
    head = "".join(f'<th class="n">{E(s)}</th>' for s in starts)
    return (f'<div class="tablewrap"><table><thead><tr><th>任务组</th><th class="n">数据</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div><p class="meta">均值 ± 样本标准差（括号里是种子数），'
            f'第 {results["final_step"] // 1000}k 步。</p>')


def fluct_html(fl, title: str, note: str) -> str:
    """Paired differences within a start (between its checkpoints) and between starts (at one checkpoint)."""
    rows = []
    for kind, label in (("within", "同一起点"), ("between", "同一检查点")):
        for r in fl.get(kind, []):
            which = (f'{E(r["start"])} {r["from"] // 1000}k → {r["to"] // 1000}k' if kind == "within" else
                     f'{E(r["start"])} − {E(r["reference"])}，{r["step"] // 1000}k')
            rows.append(f'<tr><td>{label}</td><td>{which}</td>'
                        + "".join(f'<td class="n">{pts(r[g]["diff"])}<br><span class="meta">p {pval(r[g]["p"])}</span></td>'
                                  f'<td class="n">{pts(r[g]["task_mean_diff"])}</td>' for g in ("all", "spatial")) + "</tr>")
    if not rows:
        return ""
    return (f'<h3>{E(title)}</h3><div class="tablewrap"><table><thead><tr><th>比较</th><th>哪两个</th>'
            f'<th class="n">全部：合并差（McNemar p）</th><th class="n">全部：逐任务平均差</th>'
            f'<th class="n">空间：合并差（McNemar p）</th><th class="n">空间：逐任务平均差</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div><p class="meta">{E(note)}</p>')


def e1_html(e1) -> str:
    """E1 at 150k, 175k and 200k: the success rate at each and their mean (the main numbers, the user's decision 2
    of 2026-10-23 evening), the task-level paired difference to the reference on the mean, the end fluctuation."""
    if not e1:
        return '<p class="empty">E1 的三个检查点（15 万、17.5 万、20 万步）还没都评完。</p>'
    m, points = e1["mean"], e1["points"]
    head = "".join(f'<th class="n">{p // 1000}k</th>' for p in points)
    rows = []
    for g in ("all", "spatial"):
        for s in m["starts"]:
            v = m["groups"][s][g]
            rows.append(f'<tr><td>{E(stats.GROUP_ZH[g])}</td><td>{E(s)}</td>'
                        + "".join(f'<td class="n">{pct(x)}</td>' for x in v["rates"])
                        + f'<td class="n"><b>{pct(v["mean"])}</b></td></tr>')
    rates = (f'<h3>成功率：三个检查点和三点平均</h3><div class="tablewrap"><table><thead><tr><th>任务组</th><th>起点</th>'
             f'{head}<th class="n">三点平均</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    rows = []
    for s, v in m["vs_reference"].items():
        for g in ("all", "spatial"):
            d = v["groups"][g]
            by_step = "；".join(f'{x["step"] // 1000}k {pts(x["diff"])}（p {pval(x["p"])}）' for x in d["pooled_by_step"])
            rows.append(f'<tr><td>{E(s)} − {E(m["reference"])}</td><td>{E(stats.GROUP_ZH[g])}</td>'
                        f'<td class="n">{pts(d["task_mean_diff"])}<br><span class="meta">{pts(d["task_low"])} 到 '
                        f'{pts(d["task_high"])}</span></td><td class="n">{d["tasks_up"]} / {d["tasks_down"]}</td>'
                        f'<td class="n">{pval(d["wilcoxon_p"])}</td><td class="n">{pts(d["pooled_mean_diff"])}<br>'
                        f'<span class="meta">{E(by_step)}</span></td></tr>')
    vs = (f'<h3>与 {E(m["reference"])} 的配对差（三点平均）</h3><div class="tablewrap"><table><thead><tr><th>比较</th>'
          f'<th>任务组</th><th class="n">逐任务平均差（95% 区间）</th><th class="n">任务 ↑ / ↓</th><th class="n">Wilcoxon p</th>'
          f'<th class="n">合并差：三点平均（每点，McNemar p）</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
          f'<p class="meta">单位百分点。每个任务先算三个检查点上的配对差（同一批场景、两边都完成的回合），取平均，'
          f'再对任务做主检验；合并差和 McNemar p 只作参考。</p>') if rows else ""
    fluct = fluct_html(e1["fluctuation"], "训练末尾的起伏",
                       "单位百分点，同一批场景配对。同一起点三个检查点之间（15 万 → 17.5 万 → 20 万步，以及 15 万 → 20 万步）"
                       "的差不是起点造成的，是比较起点时的起伏尺度；起点之间的差要和它比。")
    return rates + vs + fluct + e1_band_html(e1)


def e2_html(rv) -> str:
    """E2 within each start: 10% and 30% minus 100% at the final step, same scenes (build.ratio_vs_full)."""
    if not rv:
        return ""
    rows = []
    for s, by in rv.items():
        for r, row in sorted(by.items()):
            for g in ("all", "spatial"):
                d = row[g]
                rows.append(f'<tr><td>{E(s)}</td><td>{int(r)}% − 100%</td><td>{E(stats.GROUP_ZH[g])}</td>'
                            f'<td class="n">{pct(d["rate_b"])} → {pct(d["rate_a"])}</td>'
                            f'<td class="n">{pts(d["task_mean_diff"])}<br><span class="meta">{pts(d["task_low"])} 到 '
                            f'{pts(d["task_high"])}</span></td><td class="n">{d["tasks_up"]} / {d["tasks_down"]}</td>'
                            f'<td class="n">{pval(d["wilcoxon_p"])}</td><td class="n">{pval(d["p"])}</td></tr>')
    return (f'<h3>E2：同一起点，少量数据减 100% 数据（第 20 万步）</h3><div class="tablewrap"><table><thead><tr><th>起点</th>'
            f'<th>比较</th><th>任务组</th><th class="n">成功率（100% → 少量）</th><th class="n">逐任务平均差（95% 区间）</th>'
            f'<th class="n">任务 ↑ / ↓</th><th class="n">Wilcoxon p</th><th class="n">McNemar p</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div><p class="meta">单位百分点，同一批场景配对。20 万步时 100% 数据'
            f'只过了约 0.36 个 epoch，30% 约 1.18 个，10% 约 3.6 个。</p>')


def e1_band_html(e1) -> str:
    """The E3 task rule rechecked on E1's three-point means (build.e1_band)."""
    band = e1.get("band") or []
    if not band:
        return ""
    starts = e1["mean"]["starts"]
    rows = []
    for r in sorted(band, key=lambda r: -r["mean"]):
        rows.append(f'<tr><td>{E(r["task"])}</td>' + "".join(f'<td class="n">{pct(r["rates"][s])}</td>' for s in starts)
                    + f'<td class="n"><b>{pct(r["mean"])}</b></td><td>{"是" if r["in_band"] else "否"}</td>'
                    f'<td>{"是" if r["listed"] else ""}</td></tr>')
    head = "".join(f'<th class="n">{E(s)}</th>' for s in starts)
    return (f'<h3>E3 任务清单复核</h3><div class="tablewrap"><table><thead><tr><th>空间任务</th>{head}'
            f'<th class="n">起点平均</th><th>在 10%–60%</th><th>阶段 0 清单</th></tr></thead><tbody>{"".join(rows)}</tbody>'
            f'</table></div><p class="meta">每个起点取三个检查点的成功率平均，再对起点平均；落在 10%–60% 的空间任务是 RL 的'
            f'任务（你 10-23 晚的决定 5：清单变了先报告，不自行替换）。</p>')


def stage0_html(s0) -> str:
    """Stage 0: success by checkpoint (all, spatial), the spatial tasks in the band, the last two points per start."""
    if not s0 or not s0["curve"]:
        return '<p class="empty">阶段 0 还没有评完的检查点。</p>'
    steps, starts = s0["steps"], s0["starts"]
    head = "".join(f'<th class="n">{s // 1000}k</th>' for s in steps)
    evals = s0.get("evaluations") or []
    short = {(r["start"], r["step"]) for r in evals if r["kind"] == "first" and r["missing"]}
    rows = []
    for g in ("all", "spatial"):
        for s in starts:
            cells = []
            for step in steps:
                c = s0["curve"][s].get(step)
                k, n = c[g] if c else (0, 0)
                mark = "*" if (s, step) in short else ""
                cells.append(f'<td class="n">{k}/{n} = {pct(k / n)}{mark}</td>' if n else '<td class="n">—</td>')
            rows.append(f"<tr><td>{E(stats.GROUP_ZH[g])}</td><td>{E(s)}</td>{''.join(cells)}</tr>")
    curve = (f'<h3>成功率随检查点</h3><div class="tablewrap"><table><thead><tr><th>任务组</th><th>起点</th>{head}</tr>'
             f'</thead><tbody>{"".join(rows)}</tbody></table></div>'
             + ('<p class="meta">* 该评测缺回合，见下表。</p>' if short else ""))
    rows = []
    for r in evals:
        if not r["missing"]:
            continue
        a = r["all"]
        rows.append(f'<tr><td>{E(r["start"])}</td><td class="n">{r["step"] // 1000}k{"（第二遍）" if r["kind"] == "repeat" else ""}</td>'
                    f'<td class="n">{r["episodes"]}/{r["planned"]}</td><td>{E("、".join(r["missing"]))}</td>'
                    f'<td class="n">{pct(a["rate"])}<br><span class="meta">补完后 {pct(a["low"])}–{pct(a["high"])}</span></td>'
                    f'<td>{"还在跑或排队" if r["partial"] else "已结束（这些回合重跑后仍失败）"}</td></tr>')
    status = (f'<h3>缺回合的评测</h3><div class="tablewrap"><table><thead><tr><th>起点</th><th class="n">检查点</th>'
              f'<th class="n">已完成 / 应完成</th><th>缺的回合（任务 种子）</th><th class="n">全部任务成功率</th><th>状态</th>'
              f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div><p class="meta">应完成 = 计划回合减剔除清单。'
              f'"补完后"是缺的回合全部失败到全部成功时的范围。</p>') if rows else ""
    lo, hi = (int(100 * x) for x in s0["band"])
    band = '<p class="empty">还没有四个起点都评过的检查点。</p>'
    if s0["latest_common"]:
        rows = []
        for x in s0["spatial"]:
            cells = "".join(f'<td class="n">{pct(x["rates"].get(s))}</td>' for s in starts)
            rng = "" if abs(x["high_all"] - x["low_all"]) < 1e-9 else \
                f'<br><span class="meta">{pct(x["low_all"])}–{pct(x["high_all"])}</span>'
            rows.append(f'<tr><td>{E(x["task"])}</td>{cells}<td class="n">{pct(x["mean"])}</td>'
                        f'<td class="n {"up" if x["in_band"] else ""}">{"是" if x["in_band"] else "否"}</td>'
                        f'<td class="n">{pct(x["mean_all"])}{rng}</td>'
                        f'<td class="n {"up" if x["in_band_all"] else ""}">{"是" if x["in_band_all"] else "否"}'
                        f'<br><span class="meta">{x["steps_in_band"]}/{len(x["per_step"])} 个检查点在</span></td></tr>')
        band = (f'<h3>空间任务：第 {s0["latest_common"] // 1000}k 步的成功率</h3><div class="tablewrap"><table><thead><tr>'
                f'<th>任务</th>{"".join(f"<th class=n>{E(s)}</th>" for s in starts)}<th class="n">四起点平均</th>'
                f'<th class="n">在 {lo}%–{hi}%</th><th class="n">全部检查点平均</th>'
                f'<th class="n">在 {lo}%–{hi}%（全部检查点）</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
                f'<p class="meta">每格是该起点在该任务 50 个回合上的成功率；平均是对起点取平均。全部检查点平均：每个起点每个'
                f'检查点一个成功率，再平均；小字是缺的回合补完后的范围；"n/4 个检查点在"是四起点平均落在区间内的检查点个数。</p>')
    fluct = fluct_html(s0.get("fluctuation") or {}, "检查点之间的波动和起点之间的差",
                       "单位百分点，同一批场景配对。同一起点的相邻检查点相隔 2.5 万步、学习率还高，它们之间的差不是起点"
                       "造成的，可以当作只看一个检查点时的波动尺度。")
    rows = []
    for s, r in s0["flat"].items():
        cells = ""
        for g in ("all", "spatial"):
            v = r[g]
            cells += (f'<td class="n">{pts(v["diff"])}<br><span class="meta">McNemar p {pval(v["p"])}</span></td>'
                      f'<td class="n">{pts(v["task_task_mean_diff"])}<br><span class="meta">{v["task_tasks_up"]}↑ '
                      f'{v["task_tasks_down"]}↓ · Wilcoxon p {pval(v["task_wilcoxon_p"])}</span></td>')
        rows.append(f'<tr><td>{E(s)}</td><td class="n">{r["from"] // 1000}k → {r["to"] // 1000}k</td>{cells}</tr>')
    flat = (f'<h3>最后两个检查点之差</h3><div class="tablewrap"><table><thead><tr><th>起点</th><th class="n">检查点</th>'
            f'<th class="n">全部：合并差</th><th class="n">全部：逐任务平均差</th><th class="n">空间：合并差</th>'
            f'<th class="n">空间：逐任务平均差</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="meta">同一起点后一个检查点减前一个，同一批场景配对，单位百分点。合并差把所有回合放在一起；'
            f'逐任务平均差先算每个任务的差再平均。p 大、差小说明这两个点之间分不出变化（曲线已平），'
            f'判断时和"评测噪声"一节的下限比。</p>') if rows else '<p class="empty">每个起点至少要两个检查点。</p>'
    rule = s0.get("step_rule") or {}
    rows = []
    for s, r in rule.get("starts", {}).items():
        rows.append(f'<tr><td>{E(s)}</td><td class="n">{pts(r["task_mean_diff"])}<br><span class="meta">{r["tasks_up"]}↑ '
                    f'{r["tasks_down"]}↓</span></td><td class="n">{pval(r["wilcoxon_p"])}</td>'
                    f'<td class="n {"" if r["within"] else "down"}">{"是" if r["within"] else "否"}</td>'
                    f'<td class="n">{pts(r["diff"])}</td><td class="n">{pval(r["p"])}</td></tr>')
    verdict = (f'按规则阶段 1 用 <b>{rule["steps"] // 1000}k</b> 步'
               + ("" if rule["pooled_steps"] == rule["steps"] else f'（用合并差判断是 {rule["pooled_steps"] // 1000}k，不一致）')
               if rule.get("complete") else "四个起点的 50k 和 100k 还没都评完。")
    steps_rule = (f'<h3>阶段 1 的步数规则：50k → 100k</h3><div class="tablewrap"><table><thead><tr><th>起点</th>'
                  f'<th class="n">逐任务平均差</th><th class="n">Wilcoxon p</th><th class="n">在噪声内</th>'
                  f'<th class="n">合并差</th><th class="n">McNemar p</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
                  f'</div><p class="meta">你 10-22 定的规则：四个起点的 50k 到 100k 配对差都在噪声内（p &gt; 0.05 且小于 3 个百分点）'
                  f'用 10 万步，否则 20 万。按主检验（34 个任务为配对单位）判断，合并差和 McNemar 并列参考。{verdict}</p>')
    return curve + status + band + flat + steps_rule + fluct


def example_block(ex) -> str:
    strips = []
    for s, v in ex["per_start"].items():
        res = f'<span class="{"ok" if v["success"] else "fail"}">{"成功" if v["success"] else "失败"}</span>'
        alt = "%s %s seed %s" % (s, ex["task"], ex["eval_seed"])
        strips.append(f'<div class="who">{E(s)}<br>{res}<br>{v["steps"]}/{v["max_steps"]} 步</div>'
                      f'<div>{img_tag(v["frames"], alt, "image/jpeg")}</div>')
    klass = {"spatial": '<span class="tag spatial">空间</span>', "borderline": '<span class="tag">边界</span>'}.get(ex["klass"], "")
    return (f'<article class="example"><div class="head"><h3>{E(ex["title"])}</h3><span>{E(ex["task"])} {klass}</span>'
            f'<span class="meta">评测种子 {ex["eval_seed"]} · 训练种子 {ex["train_seed"]}</span></div>'
            + (f'<p class="meta">指令：{E(ex["instruction"])}</p>' if ex.get("instruction") else "")
            + f'<div class="strip">{"".join(strips)}</div></article>')


def render(results: dict, coll, figs: dict, examples: list) -> str:
    r = results
    main = r.get("main_cmp")
    order = r.get("task_order") or []
    s0 = r.get("stage0")
    nav = [("summary", "结论")] + ([("e1", "E1 三点")] if not s0 else []) + ([("stage0", "阶段 0")] if s0 else []) + [
           ("metrics", "指标说明"), ("training", "训练损失"), ("checkpoints", "检查点曲线"),
           ("efficiency", "样本效率"), ("tasks", "逐任务"), ("agreement", "结果一致性"), ("noise", "评测噪声"),
           ("examples", "范例"), ("sources", "数据来源")]
    parts = [f"<title>RoboCasa 起点对照</title>",
             '<meta name="claude:viewport-fit-sides" content="cover">', FONTS, f"<style>{CSS}</style>",
             '<div class="wrap"><header class="top">',
             "<h1>RoboCasa 起点对照</h1>",
             f'<p class="prose lede">同一套 StarVLA 动作训练和 RoboCasa365 闭环评测下，比较四个起点权重：原始 7B（B）、'
             f'动作 SFT（S）、两种 RL（RS、RD）。所有比较都在相同场景上配对进行。</p>',
             f'<div class="meta"><span>生成于 {E(r["generated"])}</span><span>代码 <code>{E(r["code"] or "?")}</code></span>'
             f'<span>{r["counts"]["experiments"]} 个实验 · {r["counts"]["evaluations"]} 次评测 · '
             f'{r["counts"]["episodes"]:,} 个回合</span></div>']
    if r["simulated"]:
        parts.append('<div class="banner">模拟数据：本页所有数字和图都是 downstream/report/simulate.py 编出来的，'
                     '只用来预览版式和统计流程，不是任何实验结果。</div>')
    parts.append("<nav class=\"toc\">" + "".join(f'<a href="#{i}">{E(t)}</a>' for i, t in nav) + "</nav></header>")
    parts.append(f'<section id="summary"><h2>结论</h2>{legend(coll.starts())}'
                 f'<ul class="facts">{"".join(f"<li>{E(x)}</li>" for x in r["summary_lines"])}</ul>'
                 f'<h3>{E(r.get("main_label", ""))}</h3>{group_table(main)}</section>')
    if not s0:
        parts.append(f'<section id="e1"><h2>E1：第 15 万、17.5 万、20 万步三点</h2><p class="prose">100% 数据、训练种子 1。'
                     f'主数字用三个检查点的平均；三点之间的配对差是训练末尾的起伏尺度（你 10-23 晚的决定 2）。第 15 万和'
                     f' 17.5 万步的权重从全状态复制出来另评，第 20 万步由训练任务自己评；三次评测用同一批场景（944 回合）。'
                     f'</p>{e1_html(r.get("e1"))}{e2_html(r.get("e2"))}</section>')
    if s0:
        parts.append(f'<section id="stage0"><h2>阶段 0：已保留检查点</h2><p class="prose">10-22 停下的四个 D5 训练'
                     f'（100% 数据、训练种子 1）各保留 2.5 万、5 万、7.5 万、10 万步四个检查点，每个都按 950 回合协议评一遍。'
                     f'</p>{stage0_html(s0)}</section>')
    parts.append('<section id="metrics"><h2>指标说明</h2><dl class="glossary">'
                 + "".join(f"<dt>{E(a)}</dt><dd>{E(b)}</dd>" for a, b in GLOSSARY) + "</dl></section>")
    parts.append(f'<section id="training"><h2>训练损失</h2>{figure(figs.get("loss"), "上：四个起点的动作损失（滑动平均，对数轴）。下：每个起点减 B，同一步、同一批数据逐点相减后再平滑；0 线以下表示损失比 B 低。", "训练损失曲线")}</section>')
    ck_caption = ("100% 数据、训练种子 1，四个起点每 2.5 万步一个检查点的成功率：左为全部 34 个任务，右为 9 个空间任务；"
                  "浅色带为 95% Wilson 区间。中间检查点学习率仍高，只用来看成功率何时离开地板和何时变平。") if s0 else \
        "100% 数据、训练种子 1 的四个任务，每 2.5 万步一个检查点的成功率，浅色带为 95% Wilson 区间。中间检查点学习率仍高，只用来看成功率何时离开地板。"
    parts.append(f'<section id="checkpoints"><h2>检查点曲线（D5）</h2>{figure(figs.get("checkpoints"), ck_caption, "检查点成功率")}</section>')
    parts.append(f'<section id="efficiency"><h2>样本效率</h2>{figure(figs.get("learning"), "第三步：横轴是用到的目标示教比例（10%、30%、100%），纵轴是最后一个检查点的成功率，点为 3 个训练种子的均值，误差条为标准差；每个小图一个任务组。", "样本效率曲线")}{seed_tables(r)}'
                 f'{figure(figs.get("efficiency"), "成功回合用了多少步（占任务时限的比例）。箱体为四分位，横线为中位数。", "成功所用步数")}</section>')
    parts.append(f'<section id="tasks"><h2>逐任务</h2><p class="prose">{E(r.get("main_label", ""))}。</p>'
                 f'{figure(figs.get("heatmap"), "每个任务每个起点的成功回合 / 回合。任务名颜色：橙色空间组，橄榄色边界，灰色其余。", "逐任务热力图")}'
                 f'{figure(figs.get("forest"), "每个任务上某起点减 B 的配对差与 95% 区间；实心点表示 Holm 校正后 McNemar p < 0.05。", "逐任务配对差")}'
                 f'{task_table(coll, main, order)}</section>')
    parts.append(f'<section id="agreement"><h2>结果一致性</h2>{figure(figs.get("agreement"), "同一批配对回合上，某起点与 B 的结果：都成功 / 只有该起点成功 / 只有 B 成功 / 都失败。中间两段决定配对检验。", "结果一致性")}</section>')
    noise_rows = "".join(
        f'<tr><td>{E(x["exp"])} 第 {x["step"] // 1000}k 步</td><td class="n">{x["workers"][0]} → {x["workers"][1]}</td>'
        f'<td class="n">{x["successes"][0]}/{x["episodes"][0]} → {x["successes"][1]}/{x["episodes"][1]}</td>'
        f'<td class="n">{pts(x["rate_diff"])}</td><td class="n">{x["changed"]}/{x["compared"]}</td>'
        f'<td class="n">{x["seconds"][0]:.0f} → {x["seconds"][1]:.0f} 秒（{x["slowdown"]:.2f} 倍）</td></tr>'
        for x in r.get("noise", []))
    noise_table = (f'<div class="tablewrap"><table><thead><tr><th>检查点</th><th class="n">并行进程数</th><th class="n">成功</th>'
                   f'<th class="n">成功率差（百分点）</th><th class="n">结果不同的回合</th><th class="n">平均每回合</th></tr>'
                   f'</thead><tbody>{noise_rows}</tbody></table></div>') if noise_rows else ""
    floor_rows = "".join(
        f'<tr><td>{E(f["exp"])} 第 {f["step"] // 1000}k 步</td><td class="n">{pts(f["pooled_diff"])}</td>'
        f'<td class="n">{pts(f["task_mean_abs_diff"]).lstrip("+")}</td><td class="n">{pts(f["task_max_abs_diff"]).lstrip("+")}</td>'
        f'<td class="n">{pts(f["task_level"]["task_mean_diff"])}（{pts(f["task_level"]["task_low"])} 到 '
        f'{pts(f["task_level"]["task_high"])}）</td><td class="n">{f["episodes_changed"]}/{f["episodes_compared"]}</td></tr>'
        for f in r.get("noise_floor", []))
    if floor_rows:
        noise_table += (f'<h3>评测噪声下限</h3><div class="tablewrap"><table><thead><tr><th>检查点</th>'
                        f'<th class="n">合并差</th><th class="n">逐任务差的平均绝对值</th><th class="n">逐任务差的最大绝对值</th>'
                        f'<th class="n">逐任务平均差（区间）</th><th class="n">结果不同的回合</th></tr></thead><tbody>{floor_rows}'
                        f'</tbody></table></div><p class="meta">同一检查点、同一批评测种子评两遍（第二遍减第一遍，百分点）。'
                        f'权重没变，差异全部来自评测本身；起点之间的差落在这个范围里就分不出来。</p>')
    parts.append(f'<section id="noise"><h2>评测噪声与并行度</h2>{figure(figs.get("noise"), "同一检查点、同一批评测种子评两遍：左为逐任务成功率（第一遍对第二遍，两边都完成的回合），右为平均每回合用时。", "评测噪声")}{noise_table}</section>')
    ex_html = "".join(example_block(x) for x in examples) if examples else \
        '<p class="empty">还没有带画面的评测回合（每个回合的 8 帧画面条从 10-21 起开始记录）。</p>'
    parts.append(f'<section id="examples"><h2>范例</h2><p class="prose">{E(r.get("examples_rule", ""))}</p>{ex_html}</section>')
    parts.append(f'<section id="sources"><h2>数据来源</h2><ul class="facts">{"".join(f"<li>{E(x)}</li>" for x in r["sources"])}</ul></section>')
    parts.append(f'<footer>由 <code>python -m downstream.report.build</code> 生成。图中标签为英文（集群没有中文字体）。</footer></div>')
    return "\n".join(parts) + "\n"
