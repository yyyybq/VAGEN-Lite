"""WalkerBench pilot gate G0-G3, exactly as PREREG_WB_PILOT.md section 9 (WP-WBPILOT; DECISIONS B3, B4, B5, B7,
B10, B16, B17).

    PY_EVAL -m vpd.walkerbench.pilot_gate [--pilot-root OUT/walkerbench/runs/pilot] [--baselines ...]
                                          [--smoke-verdict .../smoke_verdict.json] [--starts Q3B,Q3S,E2]
    -> PILOT_ROOT/GATE.json and GATE.md (absolute numbers with Wilson 95% CIs, floors, parse-valid %, tokens per
       image).  Exit 0 = every gate decided and passed, 1 = a gate failed, 3 = incomplete (missing evals / files).

Reads each start's pilot evals OUT/walkerbench/runs/pilot/<START>/evals/pilot__<START>__<tier>__<res>__<mode>/
(results.json, or results_invalid.json = an invalid run; per_task.jsonl), the adapters' meta.json, the authoritative
floors in baselines_s2test.json (floors.nav_ODO, floors.vis_FWD, re-run in WalkerBench's own loop) and the smoke
verdict.  Rules (section 9):
  G0 mechanism: every pilot eval valid with 0 tasks without a log (179 s2test tasks, no --limit); tokens per image
     242 at 640x400 and 1002 at 1280x800 (processor image_grid_thw on the logged JPEGs, B4); HF processor parity
     and live vLLM prompt-token parity (smoke verdict, B5) plus each eval's own prompt parity; render parity across
     PY_EVAL / PY_Q3 (B17); max_pixels 1,024,000 in vLLM and in training (B16); the three starts' 0W adapters have
     the same code sha256, and 50 / 200 use the same code and start from that start's 0W (B7); every merged eval
     (#3-#6) served the adapter.pt now on disk (results.adapter.adapter_sha256); attention sdpa everywhere (B10);
     the eval settings of section 7.1 (mm kwargs max_pixels and size.longest_edge 1,024,000); the floor file
     has warnings == [].
  G1 learnable: at tier 200 (640x400, constrained, seed 1001) at least one start has BOTH nav Wilson-95% lower
     bound > the nav ODO floor rate AND vis lower bound > the vis FWD floor rate, on the same tasks.
  G2 format: every constrained eval (#1-#4, and #6 if run) of every start: 100% of replies regex-fullmatch and 0
     tasks without a log.  The unconstrained tier-200 eval (#5) is descriptive only (parse-valid %, crashes).
  G3 not saturated: every start's tier-200 pooled success <= 80% (k/179 <= 0.8, point estimate).
Wilson lower bound with z = 1.96 (section 9 formula).  The gate never computes start-vs-start differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

from vpd import paths

Z = 1.96
N_TEST = 179
N_KIND = {"nav": 85, "vis": 94}
SATURATION = 0.80
STARTS = ("Q3B", "Q3S", "E2")
SEED = 1001
TOKENS = {"640": 242, "1280": 1002}
RES = {"640": [640, 400], "1280": [1280, 800]}
PHASE = {"0W": "0W", "50": "tier50", "200": "tier200"}
# (key, tier, res, mode, mandatory, constrained gate eval); PREREG 7.2 numbering
EVALS = (("1", "raw", "640", "empty", True), ("2", "raw", "1280", "empty", True),
         ("3", "50", "640", "empty", True), ("4", "200", "640", "empty", True),
         ("5", "200", "640", "off", True), ("6", "0W", "640", "empty", False))
SETTINGS = {"task_set": "s2test", "temperature": 0.0, "max_tokens": 512, "max_images": 80, "history_turns": 0,
            "blind": False, "lenient": False, "limit": None}


def wilson_lb(k: int, n: int, z: float = Z) -> Optional[float]:
    """PREREG section 9: (p + z^2/2n - z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)."""
    if not n:
        return None
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def wilson_ub(k: int, n: int, z: float = Z) -> Optional[float]:
    if not n:
        return None
    p = k / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def read_json(p) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def sha256_file(p) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def eval_dir(root: Path, start: str, tier: str, res: str, mode: str) -> Path:
    return Path(root) / start / "evals" / f"pilot__{start}__{tier}__{res}__{mode}"


def load_eval(d: Path) -> dict:
    """{present, valid, results, rows}: results.json (valid) or results_invalid.json; per_task.jsonl rows."""
    d = Path(d)
    r = read_json(d / "results.json")
    valid = r is not None
    if r is None:
        r = read_json(d / "results_invalid.json")
    rows = []
    if (d / "per_task.jsonl").is_file():
        rows = [json.loads(x) for x in (d / "per_task.jsonl").read_text().splitlines() if x.strip()]
    return {"dir": str(d), "present": r is not None, "valid": valid, "results": r, "rows": rows}


def kind_counts(rows: List[dict]) -> Dict[str, dict]:
    """{kind: {k, n, stems}} from per_task.jsonl rows (success 0/1; a row without a log scores 0)."""
    out = {}
    for row in rows:
        c = out.setdefault(row["kind"], {"k": 0, "n": 0, "stems": set()})
        c["n"] += 1
        c["k"] += int(row.get("success") or 0)
        c["stems"].add(row["stem"])
    return out


def block(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": None if not n else round(k / n, 4),
            "wilson95": None if not n else [round(wilson_lb(k, n), 4), round(wilson_ub(k, n), 4)]}


def floors_of(baselines: dict) -> dict:
    """The per-type floors on the same tasks: nav ODO and vis FWD (k, n, rate, stems)."""
    fl = baselines["floors"]
    pol = baselines["policies"]
    out = {}
    for kind, policy, key in (("nav", "ODO", "nav_ODO"), ("vis", "FWD", "vis_FWD")):
        per = pol[policy]["per_task"]
        stems = {s for s in per if s.split("_", 1)[0] == kind}
        k = sum(int(per[s]["success"]) for s in stems)
        out[kind] = {"policy": policy, "k": int(fl[key]["k"]), "n": int(fl[key]["n"]), "rate": float(fl[key]["rate"]),
                     "stems": stems, "recount_k": k}
    out["pooled_F"] = {k: fl["pooled_F"][k] for k in ("k", "n", "rate")}
    return out


def _chk(name, ok, detail=None):
    return {"name": name, "ok": ok, "detail": detail}


def _tri(items: List[dict]) -> str:
    """pass / fail / incomplete over checks whose ok is True / False / None (None = cannot be judged)."""
    if any(c["ok"] is False for c in items):
        return "fail"
    if any(c["ok"] is None for c in items):
        return "incomplete"
    return "pass"


def check_eval_settings(r: dict, start: str, tier: str, res: str, mode: str) -> List[str]:
    bad = []
    for k, v in SETTINGS.items():
        if r.get(k) != v:
            bad.append(f"{k} {r.get(k)!r} != {v!r}")
    if list(r.get("render_size") or []) != RES[res]:
        bad.append(f"render_size {r.get('render_size')}")
    if r.get("constrained") != mode:
        bad.append(f"constrained {r.get('constrained')!r} != {mode!r}")
    if r.get("start") != start:
        bad.append(f"start {r.get('start')!r} != {start!r}")
    if r.get("n_tasks") != N_TEST:
        bad.append(f"n_tasks {r.get('n_tasks')}")
    ad = r.get("adapter")
    if tier == "raw":
        if ad is not None or r.get("model_dir"):
            bad.append("raw eval served an adapter / merged folder")
    else:
        if not ad or ad.get("tier") != PHASE[tier] or ad.get("start") != start or ad.get("seed") != SEED:
            bad.append(f"adapter {None if not ad else {k: ad.get(k) for k in ('tier', 'start', 'seed')}}")
    mm = (r.get("server") or {}).get("mm_processor_kwargs") or {}
    if mm.get("max_pixels") != paths.MAX_PIXELS or (mm.get("size") or {}).get("longest_edge") != paths.MAX_PIXELS:
        bad.append(f"mm_processor_kwargs {mm} (B16)")
    if ((r.get("server") or {}).get("cfg") or {}).get("max_images") != SETTINGS["max_images"]:
        bad.append("server max_images")
    return bad


def evaluate(pilot_root: Path, baselines_path: Path, smoke_verdict_path: Optional[Path],
             starts=STARTS, adapters_root: Optional[Path] = None) -> dict:
    pilot_root = Path(pilot_root)
    adapters_root = Path(adapters_root or pilot_root)
    base = read_json(baselines_path)
    if base is None:
        raise SystemExit(f"cannot read {baselines_path}")
    fl = floors_of(base)
    sv = read_json(smoke_verdict_path) if smoke_verdict_path else None
    out = {"schema": "vpd.walkerbench.pilot_gate/1", "pilot_root": str(pilot_root),
           "baselines": str(baselines_path), "baselines_sha256": sha256_file(baselines_path),
           "smoke_verdict": None if smoke_verdict_path is None else str(smoke_verdict_path),
           "z": Z, "floors": {k: {kk: vv for kk, vv in v.items() if kk != "stems"} for k, v in fl.items()},
           "survey_floors": (base.get("survey_expected") or {}).get("per_kind"), "starts": {}}
    g0, g2 = [], []
    # the floor file itself (WP-WBDATA: a valid floor file has warnings == [])
    g0.append(_chk("baselines_valid", base.get("warnings") == [], {"warnings": base.get("warnings")}))
    zero_w_code = {}
    adapter_sha = {}
    for s in starts:
        st = {"evals": {}, "adapters": {}}
        out["starts"][s] = st
        # adapters (B7, B10, B16)
        ameta = {}
        for tier in ("0W", "50", "200"):
            d = adapters_root / s / "adapters" / tier
            m = read_json(d / "meta.json")
            ameta[tier] = m
            st["adapters"][tier] = None if m is None else {
                k: m.get(k) for k in ("phase", "seed", "conversations", "epochs", "optimizer_steps",
                                      "optimizer_steps_planned", "attn", "max_pixels", "code_sha256", "data_sha256",
                                      "loss_first", "loss_last", "tokens_per_s", "peak_mem_gib")}
        if all(ameta.values()):
            zero_w_code[s] = ameta["0W"].get("code_sha256")
            bad = []
            sha0 = sha256_file(adapters_root / s / "adapters" / "0W" / "adapter.pt")
            for tier, m in ameta.items():
                if m.get("start") != s or m.get("seed") != SEED or m.get("phase") != PHASE[tier]:
                    bad.append(f"{tier}: start/seed/phase {m.get('start')}/{m.get('seed')}/{m.get('phase')}")
                if m.get("attn") != "sdpa" or any(v != "sdpa" for v in (m.get("attn_sites") or {"?": None}).values()):
                    bad.append(f"{tier}: attention {m.get('attn')} {m.get('attn_sites')}")
                if m.get("max_pixels") != paths.MAX_PIXELS:
                    bad.append(f"{tier}: max_pixels {m.get('max_pixels')}")
                if not m.get("optimizer_steps") or m.get("optimizer_steps") != m.get("optimizer_steps_planned"):
                    bad.append(f"{tier}: {m.get('optimizer_steps')} of {m.get('optimizer_steps_planned')} steps")
                if m.get("code_sha256") != ameta["0W"].get("code_sha256"):
                    bad.append(f"{tier}: code sha differs from this start's 0W")
                if tier != "0W":
                    ini = m.get("init_adapter") or {}
                    if ini.get("sha256") != sha0 or sha0 is None:
                        bad.append(f"{tier}: init adapter sha {ini.get('sha256')} != 0W adapter.pt {sha0}")
            g0.append(_chk(f"adapters_{s}", not bad, bad))
        else:
            g0.append(_chk(f"adapters_{s}", None, [t for t, m in ameta.items() if m is None]))
        # evals
        for key, tier, res, mode, mand in EVALS:
            e = load_eval(eval_dir(pilot_root, s, tier, res, mode))
            r = e["results"]
            row = {"key": key, "tier": tier, "res": res, "mode": mode, "mandatory": mand, "present": e["present"],
                   "valid": e["valid"], "dir": e["dir"]}
            st["evals"][key] = row
            if not e["present"]:
                if mand:
                    g0.append(_chk(f"eval_{s}_#{key}", None, "missing"))
                    if mode != "off":
                        g2.append(_chk(f"g2_{s}_#{key}", None, "missing"))
                continue
            kc = kind_counts(e["rows"])
            pooled_k = sum(c["k"] for c in kc.values())
            pooled_n = sum(c["n"] for c in kc.values())
            row.update(pooled=block(pooled_k, pooled_n), per_kind={k: block(c["k"], c["n"]) for k, c in kc.items()},
                       tasks_without_log=r.get("tasks_without_log"), parse_valid_pct=r.get("parse_valid_pct"),
                       invalid_move_crash=r.get("invalid_move_crash"), regex=r.get("regex"),
                       tokens_per_image=(r.get("image_tokens") or {}).get("tokens_per_image"),
                       prompt_parity=((r.get("image_tokens") or {}).get("prompt_parity") or {}).get("equal"),
                       mean_steps=r.get("mean_steps"), spl_mean=r.get("spl_mean"),
                       oracle=(r.get("oracle") or {}).get("pooled"), hard_subsets=r.get("hard_subsets"),
                       seconds_per_step=r.get("seconds_per_step"), wall_seconds=r.get("wall_seconds"))
            # results.json numbers must agree with per_task.jsonl
            if (r.get("pooled") or {}).get("k") != pooled_k or (r.get("pooled") or {}).get("n") != pooled_n:
                row["count_mismatch"] = {"results": r.get("pooled"), "per_task": [pooled_k, pooled_n]}
            bad = []
            if not e["valid"]:
                bad.append("invalid run (results_invalid.json)")
            if r.get("tasks_without_log") != 0:
                bad.append(f"tasks_without_log {r.get('tasks_without_log')}")
            if row.get("count_mismatch"):
                bad.append(f"results vs per_task counts {row['count_mismatch']}")
            if row["tokens_per_image"] != [TOKENS[res]]:
                bad.append(f"tokens per image {row['tokens_per_image']} != [{TOKENS[res]}]")
            if row["prompt_parity"] is not True:
                bad.append(f"eval prompt parity {row['prompt_parity']}")
            bad += check_eval_settings(r, s, tier, res, mode)
            if tier != "raw":  # the merged weights served must be the adapter.pt now on disk (stale-eval guard)
                served = (r.get("adapter") or {}).get("adapter_sha256")
                disk = adapter_sha.setdefault((s, tier), sha256_file(adapters_root / s / "adapters" / tier
                                                                      / "adapter.pt"))
                row["adapter_sha256"] = {"served": served, "on_disk": disk}
                if disk is None or served != disk:
                    bad.append(f"served adapter sha {served} != adapters/{tier}/adapter.pt {disk}")
            g0.append(_chk(f"eval_{s}_#{key}", not bad, bad))
            if mode != "off":
                rx = r.get("regex") or {}
                ok = bool(rx.get("constrained_calls")) and rx.get("constrained_calls") == rx.get("answered") \
                    and rx.get("regex_mismatch") == 0 and r.get("tasks_without_log") == 0
                g2.append(_chk(f"g2_{s}_#{key}", ok, {"constrained_calls": rx.get("constrained_calls"),
                                                        "answered": rx.get("answered"),
                                                        "regex_mismatch": rx.get("regex_mismatch"),
                                                        "tasks_without_log": r.get("tasks_without_log")}))
    # B7: one training code for every start's 0W
    if len(zero_w_code) == len(starts):
        g0.append(_chk("code_sha_0W_equal", len(set(zero_w_code.values())) == 1 and None not in zero_w_code.values(),
                       zero_w_code))
    else:
        g0.append(_chk("code_sha_0W_equal", None, zero_w_code))
    # smoke verdict: HF parity, live vLLM parity (B5), render parity (B17)
    if sv is None:
        g0.append(_chk("smoke_verdict", None, "missing"))
    else:
        by = {c["name"]: c for c in sv.get("checks", [])}
        for name in ("hf_parity", "vllm_parity", "render_parity", "train_max_pixels"):
            c = by.get(name)
            g0.append(_chk(f"smoke_{name}", None if c is None else bool(c.get("ok")),
                           None if c is None else c.get("detail")))
        g0.append(_chk("smoke_verdict_pass", sv.get("verdict") == "PASS", sv.get("failed")))
    # G1 / G3 on tier 200 constrained (#4)
    g1_rows, g3 = {}, []
    for s in starts:
        row = out["starts"][s]["evals"]["4"]
        if not row["present"]:
            g1_rows[s] = {"ok": None, "detail": "eval #4 missing"}
            g3.append(_chk(f"g3_{s}", None, "eval #4 missing"))
            continue
        e = load_eval(row["dir"])
        kc = kind_counts(e["rows"])
        res = {}
        same = True
        for kind in ("nav", "vis"):
            c = kc.get(kind) or {"k": 0, "n": 0, "stems": set()}
            lb = wilson_lb(c["k"], c["n"])
            same_tasks = c["stems"] == fl[kind]["stems"] and c["n"] == fl[kind]["n"] == N_KIND[kind]
            same = same and same_tasks
            res[kind] = {"k": c["k"], "n": c["n"], "rate": None if not c["n"] else round(c["k"] / c["n"], 4),
                         "wilson_lb": None if lb is None else round(lb, 4), "floor_policy": fl[kind]["policy"],
                         "floor_k": fl[kind]["k"], "floor_n": fl[kind]["n"], "floor_rate": round(fl[kind]["rate"], 4),
                         "same_tasks": same_tasks, "above": lb is not None and lb > fl[kind]["rate"]}
        g1_rows[s] = {"ok": (res["nav"]["above"] and res["vis"]["above"]) if same else None, **res}
        pk, pn = row["pooled"]["k"], row["pooled"]["n"]
        g3.append(_chk(f"g3_{s}", (pk * 100 <= int(SATURATION * 100) * pn) if pn == N_TEST else None,
                       {"k": pk, "n": pn, "rate": row["pooled"]["rate"], "max_k": int(SATURATION * N_TEST)}))
    if any(v["ok"] for v in g1_rows.values()):
        g1 = "pass"
    elif all(v["ok"] is False for v in g1_rows.values()):
        g1 = "fail"
    else:
        g1 = "incomplete"
    gates = {"G0": _tri(g0), "G1": g1, "G2": _tri(g2), "G3": _tri(g3)}
    out.update(gates=gates, g0_checks=g0, g1_per_start=g1_rows, g2_checks=g2, g3_checks=g3,
               decision=decide(gates))
    # secondary, descriptive only: the unconstrained tier-200 eval (#5)
    out["secondary_unconstrained_200"] = {s: {k: out["starts"][s]["evals"]["5"].get(k) for k in
                                              ("present", "parse_valid_pct", "invalid_move_crash")} for s in starts}
    return out


def decide(g: dict) -> str:
    """PREREG section 9 decision table (all failing gates are listed)."""
    if g["G0"] == "fail":
        return "G0 不过：工程问题，修好重跑受影响的步骤，不判科学结论（第 9 节 G0；修复规则见第 10 节第 4 条）"
    if g["G0"] == "incomplete":
        return "G0 不完整：缺评测或文件，补齐后再判；不判科学结论"
    if "incomplete" in (g["G1"], g["G2"], g["G3"]):
        return "闸门不完整：缺评测或文件，补齐后再判"
    fails = []
    if g["G1"] == "fail":
        fails.append("G1 不过：≤200 条示范学不会，不进确认轮；要不要改设计由用户定，写进最终报告")
    if g["G2"] == "fail":
        fails.append("G2 不过：先按 G0 查是不是实现问题；不是就不进确认轮，交用户")
    if g["G3"] == "fail":
        fails.append("G3 不过：200 档已饱和，不按现方案进确认轮；档位调整由用户定（不自动下移）")
    if fails:
        return "；".join(fails)
    return "WalkerBench 过闸门：先跑第 11 节的功效模拟；用户批准确认轮的预算后再进确认轮"


def _pct(b: Optional[dict]) -> str:
    if not b or not b.get("n"):
        return "—"
    lo, hi = b["wilson95"]
    return f"{b['k']}/{b['n']} = {100 * b['rate']:.1f}%（{100 * lo:.1f}–{100 * hi:.1f}）"


def render_md(rep: dict) -> str:
    """GATE.md in Chinese: conclusion first, absolute numbers only (no start-vs-start differences)."""
    g = rep["gates"]
    L = [f"# WalkerBench 试点闸门结果（GATE）", "",
         f"- 结论：**{rep['decision']}**",
         f"- G0 {g['G0']}，G1 {g['G1']}，G2 {g['G2']}，G3 {g['G3']}（规则见 PREREG_WB_PILOT.md 第 9 节）",
         f"- 生成时间：{rep.get('utc')}（UTC）；Wilson z = {rep['z']}；只列绝对数字，不算起点之间的差。", "",
         "## 下限（baselines_s2test.json，重跑值）", "",
         "| 类型 | 策略 | 成功 | 率 |", "|---|---|---|---|"]
    for kind in ("nav", "vis"):
        f = rep["floors"][kind]
        L.append(f"| {kind} | {f['policy']} | {f['k']}/{f['n']} | {100 * f['rate']:.1f}% |")
    pf = rep["floors"]["pooled_F"]
    L.append(f"| 合并 F（只作参考，不卡闸门） | ODO(nav)+FWD(vis) | {pf['k']}/{pf['n']} | {100 * pf['rate']:.1f}% |")
    sf = rep.get("survey_floors") or {}
    if sf:
        L.append("")
        L.append(f"调查值（对照）：nav ODO {sf.get('ODO', {}).get('nav')}/85，vis FWD {sf.get('FWD', {}).get('vis')}/94。"
                 "和重跑值不同时以重跑值为准。")
    L += ["", "## G1 / G3：200 档（640x400，约束解码）", "",
          "| 起点 | nav 成功（95% CI） | nav 下界 > ODO？ | vis 成功（95% CI） | vis 下界 > FWD？ | 合并（95% CI） | ≤80%？ |",
          "|---|---|---|---|---|---|---|"]
    for s, row in rep["g1_per_start"].items():
        ev = rep["starts"][s]["evals"]["4"]
        if not ev["present"]:
            L.append(f"| {s} | 缺 | | | | | |")
            continue
        g3 = next(c for c in rep["g3_checks"] if c["name"] == f"g3_{s}")

        def side(kind):
            r = row[kind]
            lb = "—" if r["wilson_lb"] is None else f"{100 * r['wilson_lb']:.1f}%"
            tasks = "" if r["same_tasks"] else "；题目和下限不一致"
            return f"{'是' if r['above'] else '否'}（下界 {lb} vs {100 * r['floor_rate']:.1f}%{tasks}）"

        L.append(f"| {s} | {_pct(ev['per_kind'].get('nav'))} | {side('nav')} | {_pct(ev['per_kind'].get('vis'))} | "
                 f"{side('vis')} | {_pct(ev['pooled'])} | {'是' if g3['ok'] else ('否' if g3['ok'] is False else '?')} |")
    L += ["", "## 每次评测（绝对数字）", "",
          "| 起点 | # | 档 | 分辨率 | 解码 | 有效 | 合并成功（95% CI） | 无日志题 | 正则不匹配 / 约束调用 | 解析有效率 | 无效 MOVE 崩溃 | 每图 token |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, st in rep["starts"].items():
        for key, ev in st["evals"].items():
            if not ev["present"]:
                L.append(f"| {s} | {key} | {ev['tier']} | {ev['res']} | {ev['mode']} | 缺"
                         f"{'' if ev['mandatory'] else '（可选）'} | | | | | | |")
                continue
            rx = ev.get("regex") or {}
            L.append(f"| {s} | {key} | {ev['tier']} | {ev['res']} | {ev['mode']} | {'是' if ev['valid'] else '否'} | "
                     f"{_pct(ev['pooled'])} | {ev['tasks_without_log']} | {rx.get('regex_mismatch')} / "
                     f"{rx.get('constrained_calls')} | {ev['parse_valid_pct']}% | {ev['invalid_move_crash']} | "
                     f"{ev['tokens_per_image']} |")
    L += ["", "## 不过或无法判定的检查", ""]
    bad = [c for c in rep["g0_checks"] + rep["g2_checks"] + rep["g3_checks"] if c["ok"] is not True]
    if not bad:
        L.append("无。")
    for c in bad:
        L.append(f"- {c['name']}：{'不过' if c['ok'] is False else '无法判定'}；{json.dumps(c['detail'], ensure_ascii=False, default=str)[:400]}")
    L += ["", "说明：不约束的 200 档（#5）的解析有效率和崩溃数只作描述，不卡闸门（第 9 节 G2）。"
          "试点数据（种子 1001）不进确认分析。", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-root", type=Path, default=paths.WB_RUNS / "pilot")
    ap.add_argument("--baselines", type=Path, default=paths.WB_BASELINES / "baselines_s2test.json")
    ap.add_argument("--smoke-verdict", type=Path, default=None, help="default: the newest smoke_verdict.json")
    ap.add_argument("--starts", default=",".join(STARTS))
    ap.add_argument("--out-dir", type=Path, default=None, help="default: --pilot-root")
    args = ap.parse_args(argv)
    sv = args.smoke_verdict
    if sv is None:
        from vpd.walkerbench import pilot

        sv = pilot.latest_verdict()[0]
    import datetime

    rep = evaluate(args.pilot_root, args.baselines, sv, tuple(s for s in args.starts.split(",") if s))
    rep["utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.out_dir or args.pilot_root
    out.mkdir(parents=True, exist_ok=True)
    (out / "GATE.json").write_text(json.dumps(rep, indent=1, default=lambda o: sorted(o) if isinstance(o, set)
                                              else str(o), ensure_ascii=False) + "\n")
    (out / "GATE.md").write_text(render_md(rep), encoding="utf-8")
    print(json.dumps({"gates": rep["gates"], "decision": rep["decision"], "out": str(out)}, ensure_ascii=False))
    g = rep["gates"]
    if "fail" in g.values():
        return 1
    if "incomplete" in g.values():
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
