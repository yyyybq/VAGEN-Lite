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
    ("配对", "两个起点在完全相同的回合上比较：同一任务、同一评测种子（所以初始场景逐位相同）、同一训练种子。配对能去掉场景难易带来的差别。"),
    ("配对差（百分点）", "某起点的成功率减参照起点（B）的成功率，只算两者都有的回合。区间是配对的 95% 区间。正数表示比 B 高。"),
    ("McNemar 精确检验 p 值", "只看两者结果不一致的回合：这个起点成功、B 失败的个数，和反过来的个数。如果两个起点其实一样好，两种情况应该一样多；p 值是在这个前提下出现眼前这么不平衡（或更不平衡）的概率。越小越不像巧合。"),
    ("Holm 校正", "34 个任务各做一次检验，难免有几个碰巧很小的 p。Holm 校正按任务数抬高 p 值；逐任务图里实心点表示校正后仍小于 0.05。"),
    ("跨任务 Wilcoxon 检验", "把每个任务的配对差当作一个数，检验这些差是否整体偏向一边（多数任务都高于 B 或都低于 B），不看回合数多少。"),
    ("种子均值 ± 标准差", "第三步每格有 3 个训练种子（动作头初值和数据顺序不同）。图里的点是 3 个种子成功率的均值，误差条是样本标准差，表示训练本身的随机性。"),
    ("结果一致性", "配对回合分成四类：两个都成功、只有这个起点成功、只有 B 成功、两个都失败。中间两类的多少决定了 McNemar 检验。"),
    ("成功所用步数 / 时限", "只看成功的回合：用了多少控制步，除以该任务的时限。越小表示完成得越快。"),
    ("动作损失（L1）", "训练时模型预测的 16 步动作与示教动作的平均绝对误差，每 100 步记一次当步那批数据的值。同一训练种子下四个起点看到的是同一批数据，所以可以逐步相减。它衡量对示教的模仿，不等于闭环成功率。"),
    ("中间检查点", "D5 每 2.5 万步存一个检查点。学习率按总步数 20 万余弦衰减，中间检查点时学习率还高，所以它们只用来看成功率何时离开地板，不能当作更短训练的结果。"),
    ("评测噪声", "同一检查点用两个任务各评一遍（种子相同）的差别：成功率差和逐回合结果不同的个数。评测在 CPU 上逐位可复现，剩下的差别来自 GPU 推理等。"),
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
        f'<th class="n">{E(s)} − {E(ref)}（区间）</th><th class="n">p</th>' for s in starts if s != ref)
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
            cls = "up" if v["diff"] > 0 and v["p"] < 0.05 else "down" if v["diff"] < 0 and v["p"] < 0.05 else ""
            cells.append(f'<td class="n {cls}">{pts(v["diff"])}<br><span class="meta">{pts(v["low"])} 到 '
                         f'{pts(v["high"])}</span></td><td class="n">{pval(v["p"])}<br><span class="meta">'
                         f'{v["tasks_up"]}↑ {v["tasks_down"]}↓</span></td>')
        rows.append(f"<tr><td>{E(stats.GROUP_ZH[g])}</td>{''.join(cells)}</tr>")
    return (f'<div class="tablewrap"><table><thead><tr><th>任务组</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="meta">差值单位：百分点；p：McNemar 精确检验（配对回合）；↑↓：该组里配对差为正 / 为负的任务个数。'
            f'绿色 / 红色：p &lt; 0.05 的提高 / 降低。</p>')


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
    nav = [("summary", "结论"), ("metrics", "指标说明"), ("training", "训练损失"), ("checkpoints", "检查点曲线"),
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
    parts.append('<section id="metrics"><h2>指标说明</h2><dl class="glossary">'
                 + "".join(f"<dt>{E(a)}</dt><dd>{E(b)}</dd>" for a, b in GLOSSARY) + "</dl></section>")
    parts.append(f'<section id="training"><h2>训练损失</h2>{figure(figs.get("loss"), "上：四个起点的动作损失（滑动平均，对数轴）。下：每个起点减 B，同一步、同一批数据逐点相减后再平滑；0 线以下表示损失比 B 低。", "训练损失曲线")}</section>')
    parts.append(f'<section id="checkpoints"><h2>检查点曲线（D5）</h2>{figure(figs.get("checkpoints"), "100% 数据、训练种子 1 的四个任务，每 2.5 万步一个检查点的成功率，浅色带为 95% Wilson 区间。中间检查点学习率仍高，只用来看成功率何时离开地板。", "检查点成功率")}</section>')
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
    parts.append(f'<section id="noise"><h2>评测噪声与并行度</h2>{figure(figs.get("noise"), "同一检查点、同一批评测种子评两遍：左为逐任务成功率（第一遍对重复），右为平均每回合用时；虚线是 1.5 倍上限，不超过它的最大并行度会用于以后的评测。", "评测噪声")}{noise_table}</section>')
    ex_html = "".join(example_block(x) for x in examples) if examples else \
        '<p class="empty">还没有带画面的评测回合（每个回合的 8 帧画面条从 10-21 起开始记录）。</p>'
    parts.append(f'<section id="examples"><h2>范例</h2><p class="prose">{E(r.get("examples_rule", ""))}</p>{ex_html}</section>')
    parts.append(f'<section id="sources"><h2>数据来源</h2><ul class="facts">{"".join(f"<li>{E(x)}</li>" for x in r["sources"])}</ul></section>')
    parts.append(f'<footer>由 <code>python -m downstream.report.build</code> 生成。图中标签为英文（集群没有中文字体）。</footer></div>')
    return "\n".join(parts) + "\n"
