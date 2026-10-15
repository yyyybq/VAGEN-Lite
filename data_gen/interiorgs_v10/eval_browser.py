"""A static page for browsing evaluation trajectories (round 8): open OUT/index.html by double-click, no server.

    python -m data_gen.interiorgs_v10.eval_browser --eval EVAL --out OUT [--models NAME ...]

EVAL is an evaluation folder (evaluation.py v2: EVAL/tasks.jsonl, EVAL/<model>/results.jsonl with trajectories,
EVAL/<model>/frames/*.jpg).  OUT gets index.html, data.js (the trajectories, loaded by a <script> tag so that
file:// works) and img/ (every frame once, named by its content hash: the start frame is shared by all models).
Per task the page shows each model's turns: the frame the model saw, its answer, the actions executed, blocked
moves, which success conditions were still unmet and the progress; then the last frame and how it ended.
Filters: model and outcome, family, d* band; two models side by side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

LABELS = {"base_qwen25vl7b": "原始 Qwen2.5-VL-7B", "v10_sft_e1": "SFT 第 1 轮", "v10_sft_e2": "SFT 第 2 轮",
          "v10_sft_e3": "SFT 第 3 轮", "r1_final": "R1 最终模型"}
FAMILIES = {"absolute_positioning": "到物体距离", "equidistance": "等距", "fov_inclusion": "两物同框",
            "occlusion_alignment": "让目标被挡住", "projective_relations": "左右", "screen_occupancy": "画面高度"}
END = {"success_auto": "成功（自动结束）", "max_primitive_steps": "用满动作上限", "max_primitive_steps_success": "用满动作时成功",
       "consecutive_collision": "连续 3 次碰撞", "consecutive_invalid": "连续 3 次格式错误", "low_info": "画面太单调（连续 3 次）",
       "max_turns": "用满回合上限", "continuing": "未结束"}


def label(name: str) -> str:
    m = re.fullmatch(r"v10_rl_(\d+)", name)
    return f"RL 第 {int(m.group(1))} 次更新" if m else LABELS.get(name, name)


def model_order(name: str) -> tuple:
    m = re.fullmatch(r"v10_rl_(\d+)", name)
    d = re.fullmatch(r"D(\d+)", name)  # round 9 diagnostic runs, after the models
    order = ["base_qwen25vl7b", "r1_final", "v10_sft_e1", "v10_sft_e2", "v10_sft_e3"]
    if m:
        return (1, int(m.group(1)))
    if d:
        return (2, int(d.group(1)))
    return (0, order.index(name) if name in order else 99, name)


def _lines(user: str, *prefixes: str) -> Optional[str]:
    for line in (user or "").splitlines():
        for prefix in prefixes:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return None


class Images:
    """Copies frames into OUT/img once each, named by content hash."""

    def __init__(self, out: Path) -> None:
        self.dir = out / "img"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.seen: Dict[str, str] = {}

    def add(self, path: Optional[Path]) -> Optional[str]:
        if path is None or not path.is_file():
            return None
        data = path.read_bytes()
        key = hashlib.sha1(data).hexdigest()[:16]
        if key not in self.seen:
            (self.dir / f"{key}.jpg").write_bytes(data)
            self.seen[key] = key
        return key


def episode(row: Dict[str, Any], frames_root: Path, images: Images) -> Dict[str, Any]:
    if "error" in row:
        return {"error": row["error"]}
    turns = []
    for t in row.get("trajectory") or []:
        turns.append({"f": images.add(frames_root / t["frame"]) if t.get("frame") else None,
                      "fb": _lines(t.get("user"), "Environment Feedback:", "Feedback:"), "pose": _lines(t.get("user"), "Current camera pose:"),
                      "a": t.get("answer"), "act": t.get("actions"), "v": t.get("valid"), "c": t.get("collisions"),
                      "r": t.get("reward"), "m": t.get("met"), "g": t.get("failed_gates"), "p": t.get("progress")})
    return {"s": bool(row["success"]), "n": row["turns"], "na": row["actions"], "c": row["collisions"],
            "inv": row["invalid_turns"], "ats": row.get("actions_to_success"), "end": row.get("end_reason"),
            "start": row.get("start"), "t": turns,
            "ff": images.add(frames_root / row["final_frame"]) if row.get("final_frame") else None}


def _tasks(eval_dir: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in (eval_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def build(eval_dirs: Union[Path, Sequence[Path]], out: Path, models: Optional[Sequence[str]] = None,
          labels: Optional[Dict[str, str]] = None, defaults: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """One page over one or more evaluation folders of the same tasks (e.g. round 8's models and round 9's
    diagnostic runs); ``labels`` names runs, ``defaults`` picks the two columns shown first ({"left", "right"})."""
    dirs = [Path(eval_dirs)] if isinstance(eval_dirs, (str, Path)) else [Path(d) for d in eval_dirs]
    out = Path(out)
    tasks = _tasks(dirs[0])
    where: Dict[str, Path] = {}
    for d in dirs:
        if [t["task_id"] for t in _tasks(d)] != [t["task_id"] for t in tasks]:
            raise ValueError(f"{d} has other tasks than {dirs[0]}")
        for run in d.iterdir():
            if (run / "results.jsonl").is_file() and (not models or run.name in models):
                if run.name in where:
                    raise ValueError(f"{run.name} is in both {where[run.name]} and {d}")
                where[run.name] = d
    names = sorted(where, key=model_order)
    out.mkdir(parents=True, exist_ok=True)
    images = Images(out)
    episodes: Dict[str, Dict[int, Any]] = {}
    for name in names:
        rows = {}
        for line in (where[name] / name / "results.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[int(r["index"])] = episode(r, where[name] / name, images)
        episodes[name] = rows
    data = {
        "models": [{"name": n, "label": (labels or {}).get(n, label(n))} for n in names],
        "defaults": defaults or {},
        "families": FAMILIES, "end": END,
        "tasks": [{"i": i, "id": t["task_id"], "scene": t.get("scene_id"), "fam": t["task_type"],
                   "d": int((t.get("difficulty") or {}).get("d_star", 0)), "desc": t.get("task_description")}
                  for i, t in enumerate(tasks)],
        "episodes": {n: {str(i): e for i, e in rows.items()} for n, rows in episodes.items()},
    }
    (out / "data.js").write_text("window.EVAL_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                                 + ";\n", encoding="utf-8")
    (out / "index.html").write_text(PAGE, encoding="utf-8")
    return {"models": names, "tasks": len(tasks), "images": len(images.seen)}


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>评测轨迹浏览</title>
<style>
:root { --bg:#f7f7f5; --panel:#fff; --line:#ddd; --text:#222; --muted:#666; --ok:#1f7a3f; --bad:#b3261e; --accent:#2457a6; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; background:var(--bg); color:var(--text); }
header { padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; position:sticky; top:0; z-index:2; }
header h1 { font-size:16px; margin:0 8px 0 0; }
label { color:var(--muted); }
select { font:inherit; padding:2px 4px; }
main { display:grid; grid-template-columns:minmax(260px, 360px) 1fr; height:calc(100vh - 54px); }
#list { overflow:auto; border-right:1px solid var(--line); background:var(--panel); }
#count { padding:6px 10px; color:var(--muted); border-bottom:1px solid var(--line); }
.task { padding:6px 10px; border-bottom:1px solid #eee; cursor:pointer; }
.task:hover { background:#f0f4fb; } .task.sel { background:#e3ecfa; }
.task .meta { color:var(--muted); font-size:12px; }
.dots { font-family:monospace; font-size:12px; letter-spacing:1px; }
.ok { color:var(--ok); } .bad { color:var(--bad); }
#detail { overflow:auto; padding:12px 16px; }
.desc { font-size:15px; margin:0 0 4px; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:10px; }
.col { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:8px 10px; min-width:0; }
.col h3 { margin:0 0 6px; font-size:14px; }
.turn { display:grid; grid-template-columns:200px 1fr; gap:10px; border-top:1px solid #eee; padding:8px 0; }
.turn img, .final img { width:200px; height:200px; image-rendering:auto; border:1px solid #ccc; background:#000; }
.turn .k { color:var(--muted); }
code { background:#f2f2f2; padding:1px 4px; border-radius:3px; word-break:break-all; }
.small { font-size:12px; color:var(--muted); }
.legend { font-size:12px; color:var(--muted); }
@media (max-width: 900px) { main { grid-template-columns:1fr; height:auto; } #list { max-height:40vh; } .cols { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>评测轨迹浏览</h1>
  <span><label>筛选模型 </label><select id="fmodel"></select>
    <select id="foutcome"><option value="">成功或失败</option><option value="s">成功</option><option value="f">失败</option></select></span>
  <span><label>族 </label><select id="ffam"><option value="">全部</option></select></span>
  <span><label>最少步数 </label><select id="fband"><option value="">全部</option><option value="1-3">1–3</option><option value="4-6">4–6</option><option value="7-8">7–8</option></select></span>
  <span><label>左边 </label><select id="ma"></select> <label>右边 </label><select id="mb"></select></span>
  <span class="legend" id="legend"></span>
</header>
<main>
  <div id="list"><div id="count"></div><div id="tasks"></div></div>
  <div id="detail"><p class="small">左边选一道题。列表里每个点是一个模型（顺序同"图例"）：<span class="ok">●</span> 成功，<span class="bad">●</span> 失败。</p></div>
</main>
<script src="data.js"></script>
<script>
const D = window.EVAL_DATA, $ = id => document.getElementById(id);
const band = d => d <= 3 ? "1-3" : d <= 6 ? "4-6" : "7-8";
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
let current = null;
function opt(sel, value, text) { const o = document.createElement("option"); o.value = value; o.textContent = text; sel.appendChild(o); }
D.models.forEach(m => { opt($("fmodel"), m.name, m.label); opt($("ma"), m.name, m.label); opt($("mb"), m.name, m.label); });
Object.entries(D.families).forEach(([k, v]) => opt($("ffam"), k, v + "（" + k + "）"));
const pick = (want) => { const m = D.models.find(m => m.name === want); return m ? m.name : null; };
$("ma").value = pick((D.defaults || {}).left) || pick("v10_sft_e3") || D.models[0].name;
const rl = D.models.filter(m => m.name.startsWith("v10_rl_"));
$("mb").value = pick((D.defaults || {}).right) || (rl.length ? rl[rl.length - 1].name : D.models[Math.min(1, D.models.length - 1)].name);
$("fmodel").value = $("mb").value;
$("legend").textContent = "图例：" + D.models.map((m, i) => (i + 1) + " " + m.label).join("，");
function ep(model, i) { return (D.episodes[model] || {})[String(i)]; }
function renderList() {
  const fm = $("fmodel").value, fo = $("foutcome").value, ff = $("ffam").value, fb = $("fband").value;
  const rows = D.tasks.filter(t => {
    if (ff && t.fam !== ff) return false;
    if (fb && band(t.d) !== fb) return false;
    if (fo) { const e = ep(fm, t.i); if (!e || e.error) return false; if ((fo === "s") !== e.s) return false; }
    return true;
  });
  $("count").textContent = rows.length + " 道题（共 " + D.tasks.length + " 道）";
  $("tasks").innerHTML = rows.map(t => {
    const dots = D.models.map(m => { const e = ep(m.name, t.i); return !e ? "·" : e.error ? "?" : "<span class='" + (e.s ? "ok" : "bad") + "'>●</span>"; }).join("");
    return "<div class='task" + (current === t.i ? " sel" : "") + "' data-i='" + t.i + "'><div>#" + t.i + " " + esc(t.desc) +
      "</div><div class='meta'>" + esc(D.families[t.fam] || t.fam) + " · d* " + t.d + " · " + esc(t.scene) + "</div><div class='dots'>" + dots + "</div></div>";
  }).join("");
  document.querySelectorAll(".task").forEach(el => el.onclick = () => { current = +el.dataset.i; renderList(); renderDetail(); });
}
function gates(g) { return g == null ? "—" : g.length ? g.map(esc).join("，") : "全部满足"; }
function column(model, t) {
  const m = D.models.find(x => x.name === model), e = ep(model, t.i);
  if (!e) return "<div class='col'><h3>" + esc(m.label) + "</h3><p class='small'>没有这道题的结果</p></div>";
  if (e.error) return "<div class='col'><h3>" + esc(m.label) + "</h3><p class='bad'>运行出错：" + esc(e.error) + "</p></div>";
  let h = "<div class='col'><h3>" + esc(m.label) + "：<span class='" + (e.s ? "ok'>成功" : "bad'>失败") + "</span></h3>";
  h += "<div class='small'>" + e.n + " 个回合，" + e.na + " 个动作" + (e.ats ? "（第 " + e.ats + " 个动作时成功）" : "") +
       "，碰撞 " + e.c + " 次，格式错误 " + e.inv + " 次；结束原因：" + esc(D.end[e.end] || e.end) + "</div>";
  if (e.start) h += "<div class='small'>起点：未满足的条件 " + gates(e.start.g || e.start.failed_gates) + "；进度 " + (e.start.progress ?? "—") + "</div>";
  e.t.forEach((x, k) => {
    h += "<div class='turn'><div>" + (x.f ? "<img loading='lazy' src='img/" + x.f + ".jpg' alt='第 " + (k + 1) + " 回合的画面'>" : "") + "</div><div>" +
      "<div><b>第 " + (k + 1) + " 回合</b>" + (x.fb ? " <span class='k'>上一步反馈：</span>" + esc(x.fb) : "") + "</div>" +
      (x.pose ? "<div class='small'>位姿 " + esc(x.pose) + "</div>" : "") +
      "<div><span class='k'>回答：</span><code>" + esc(x.a) + "</code></div>" +
      "<div><span class='k'>执行：</span>" + (x.act && x.act.length ? esc(x.act.join(" → ")) : "<span class='bad'>没有解析出动作</span>") +
      (x.c ? " <span class='bad'>（" + x.c + " 个被挡住）</span>" : "") + "</div>" +
      "<div><span class='k'>之后：</span>" + (x.m ? "<span class='ok'>满足要求</span>" : "未满足：" + gates(x.g)) +
      "；进度 " + (x.p ?? "—") + "；奖励 " + x.r + "</div></div></div>";
  });
  if (e.ff) h += "<div class='turn final'><div><img loading='lazy' src='img/" + e.ff + ".jpg' alt='最后的画面'></div><div><b>最后的画面</b></div></div>";
  return h + "</div>";
}
function renderDetail() {
  if (current === null) return;
  const t = D.tasks[current];
  $("detail").innerHTML = "<p class='desc'>#" + t.i + " " + esc(t.desc) + "</p><div class='small'>" + esc(D.families[t.fam] || t.fam) +
    " · 最少步数 d* = " + t.d + " · 场景 " + esc(t.scene) + " · " + esc(t.id) + "</div><div class='cols'>" +
    column($("ma").value, t) + column($("mb").value, t) + "</div>";
}
["fmodel", "foutcome", "ffam", "fband"].forEach(id => $(id).onchange = renderList);
["ma", "mb"].forEach(id => $(id).onchange = renderDetail);
renderList();
</script>
</body>
</html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, type=Path, nargs="+", help="evaluation folders of the same tasks")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--label", action="append", default=[], metavar="NAME=TEXT", help="name a run (repeatable)")
    ap.add_argument("--left", default=None, help="the run shown in the left column first")
    ap.add_argument("--right", default=None, help="the run shown in the right column first")
    args = ap.parse_args(argv)
    labels = dict(x.split("=", 1) for x in args.label)
    defaults = {k: v for k, v in (("left", args.left), ("right", args.right)) if v}
    print(json.dumps(build(args.eval, args.out, args.models, labels, defaults), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
