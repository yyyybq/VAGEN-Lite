#!/usr/bin/env python3
"""Build baseline→trained master tables and full research analysis."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
SWEEP = ROOT / "exps/unified_eval_runs/unified_h800_7b_eval_v2"
ART = ROOT / "exps/unified_eval_runs/analysis_artifacts"
EASI_RESULTS = ROOT / "easi_results"
EASI_8 = ["vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site", "blink", "3dsrbench", "embspatial"]
MAX_STEPS = 12
SUCCESS_THR = 0.65
BOOT_N = 2000
SEED = 42
SUITES = ["id_test_stratified400","ood_scene","ood_instance","ood_category","ood_template","ood_geometry"]
OOD_SUITES = SUITES[1:]
MODELS = {
    "qwen_baseline": ("qwen_pretrain_baseline", 0, "Qwen baseline"),
    "v46": ("v46_7b_nodelta_w3_rerun", 200, "v46@200"),
    "v50": ("v50_7b_w3_format001_stable", 150, "v50@150"),
    "cambrian_baseline": ("cambrian_pretrain_baseline", 0, "Cambrian-LFP baseline"),
    "c8": ("c8_fwdfirst_rewscale_lfp_server_v19", 300, "c8@300"),
}
COMPARISONS = [
    ("v46","qwen_baseline","same_line"),
    ("v50","qwen_baseline","same_line"),
    ("v50","v46","same_line"),
    ("c8","cambrian_baseline","same_line"),
    ("c8","qwen_baseline","system"),
    ("c8","v46","system"),
]
QA_MAP = {
    "qwen_baseline": "qwen_pretrain_baseline",
    "v46": "v46_step200",
    "v50": "v50_step150",
    "cambrian_baseline": "cambrian_pretrain_baseline",
    "c8": "c8_step300",
}

def load_result(exp, step, suite):
    p = SWEEP / exp / f"global_step_{step}" / suite / "model" / "results_model.json"
    return json.loads(p.read_text()) if p.exists() else None

def mean(xs):
    xs = list(xs)
    return float(sum(xs)/len(xs)) if xs else float("nan")

def episode_rows(d):
    rows=[]
    for ep in d.get("episodes", []):
        steps=int(ep.get("num_steps") or 0)
        max_s=float(ep.get("max_score") or 0.0)
        final=float(ep.get("final_score") or 0.0)
        rows.append({
            "episode_id": ep.get("episode_id"), "scene_id": ep.get("scene_id"),
            "task_type": ep.get("task_type"), "object_label": ep.get("object_label"),
            "success": bool(ep.get("success")), "final_score": final, "max_score": max_s,
            "num_steps": steps, "timeout": steps >= MAX_STEPS,
            "collisions": float(ep.get("total_collisions") or 0.0),
            "had_collision": float(ep.get("total_collisions") or 0.0) > 0,
            "stop_fail": (max_s >= SUCCESS_THR) and (final < SUCCESS_THR),
        })
    return rows

def aggregate(rows):
    if not rows: return {}
    return {
        "n": len(rows),
        "success": mean(r["success"] for r in rows),
        "score": mean(r["final_score"] for r in rows),
        "timeout": mean(r["timeout"] for r in rows),
        "collision": mean(r["had_collision"] for r in rows),
        "mean_collisions": mean(r["collisions"] for r in rows),
        "valid_action": None,
        "stop_fail": mean(r["stop_fail"] for r in rows),
        "mean_steps": mean(r["num_steps"] for r in rows),
        "mean_max_score": mean(r["max_score"] for r in rows),
    }

def bootstrap_ci(diffs, n_boot=BOOT_N, seed=SEED):
    if len(diffs)==0: return float("nan"), float("nan"), float("nan")
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(diffs),size=(n_boot,len(diffs)))
    samples=diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.quantile(samples,0.025)), float(np.quantile(samples,0.975))

def paired_stats(a_rows, b_rows):
    def key(r): return (r["episode_id"], r["scene_id"], r["task_type"], r["object_label"])
    am={key(r):r for r in a_rows}; bm={key(r):r for r in b_rows}
    keys=sorted(set(am)&set(bm))
    if not keys:
        if len(a_rows)==len(b_rows) and a_rows: pairs=list(zip(a_rows,b_rows))
        else: return {"n":0}
    else:
        pairs=[(am[k],bm[k]) for k in keys]
    succ=np.array([float(a["success"])-float(b["success"]) for a,b in pairs])
    score=np.array([a["final_score"]-b["final_score"] for a,b in pairs])
    to=np.array([float(a["timeout"])-float(b["timeout"]) for a,b in pairs])
    col=np.array([float(a["had_collision"])-float(b["had_collision"]) for a,b in pairs])
    win=sum(1 for a,b in pairs if a["success"] and not b["success"])
    loss=sum(1 for a,b in pairs if b["success"] and not a["success"])
    tie=len(pairs)-win-loss
    sd,slo,shi=bootstrap_ci(succ); scd,scolo,schi=bootstrap_ci(score)
    td,tlo,thi=bootstrap_ci(to); cd,clo,chi=bootstrap_ci(col)
    return {"n":len(pairs),"success_diff":sd,"success_ci_low":slo,"success_ci_high":shi,
            "score_diff":scd,"score_ci_low":scolo,"score_ci_high":schi,
            "timeout_diff":td,"timeout_ci_low":tlo,"timeout_ci_high":thi,
            "collision_diff":cd,"collision_ci_low":clo,"collision_ci_high":chi,
            "win":win,"tie":tie,"loss":loss,
            "both_success":sum(1 for a,b in pairs if a["success"] and b["success"]),
            "both_fail":sum(1 for a,b in pairs if (not a["success"]) and (not b["success"])),
            "pairs":pairs}

def normalize_easi_score(value):
    if not isinstance(value, (int, float)): return None
    value=float(value)
    return value/100.0 if abs(value)>1.0 else value

def load_easi8(ckpt):
    path=EASI_RESULTS/ckpt/"easi_results.json"
    if not path.exists(): return {}, None
    payload=json.loads(path.read_text())
    raw=payload.get("scores", {})
    scores={bench:normalize_easi_score(raw.get(bench)) for bench in EASI_8}
    values=[scores[b] for b in EASI_8 if scores[b] is not None]
    macro=float(np.mean(values)) if len(values)==len(EASI_8) else None
    return scores, macro

def pct(x):
    if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))): return "—"
    return f"{100.0*x:.1f}%"

def delta_pp(a,b):
    if a is None or b is None: return "—"
    return f"{100.0*(a-b):+.1f}pp"

def rel_pct(a,b):
    if a is None or b is None or not b: return "—"
    return f"{100.0*(a-b)/b:+.1f}%"

def collect_model(mid):
    exp,step,label=MODELS[mid]
    suite_data={}; missing=[]
    for suite in SUITES:
        d=load_result(exp,step,suite)
        if d is None:
            missing.append(suite); suite_data[suite]=None; continue
        rows=episode_rows(d); agg=aggregate(rows)
        overall=d.get("metrics",{}).get("overall",{})
        agg["valid_action"]=overall.get("mean_action_validity")
        if overall.get("timeout_rate") is not None: agg["timeout"]=float(overall["timeout_rate"])
        suite_data[suite]={"agg":agg,"rows":rows,"overall":overall}
    ood=[suite_data[s]["agg"]["success"] for s in OOD_SUITES if suite_data[s]]
    return {"id":mid,"label":label,"exp":exp,"step":step,"suites":suite_data,"missing":missing,
            "ood_macro": float(np.mean(ood)) if ood else None}

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text(""); return
    fields=sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in fields})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--require-complete", action="store_true"); args=ap.parse_args()
    models={mid:collect_model(mid) for mid in MODELS}
    for mid,m in models.items():
        key=QA_MAP[mid]
        m["easi_scores"],m["easi8_macro"]=load_easi8(key)
    missing_all={mid:m["missing"] for mid,m in models.items() if m["missing"]}
    missing_easi8=[mid for mid,m in models.items() if m["easi8_macro"] is None]
    if args.require_complete and (missing_all or missing_easi8):
        print("INCOMPLETE", {"navigation": missing_all, "easi8": missing_easi8}); return 2

    master_rows=[]
    for mid,m in models.items():
        row={"model":m["label"],"model_id":mid}
        for suite in SUITES:
            s=m["suites"][suite]; key="id400" if suite.startswith("id_") else suite.replace("ood_","")
            if not s:
                row[f"{key}_success"]=None; row[f"{key}_score"]=None; continue
            row[f"{key}_success"]=s["agg"]["success"]; row[f"{key}_score"]=s["agg"]["score"]
        row["ood_macro_success"]=m["ood_macro"]
        id_s=m["suites"]["id_test_stratified400"]
        if id_s:
            row["timeout"]=id_s["agg"]["timeout"]; row["collision"]=id_s["agg"]["collision"]
            row["valid_action"]=id_s["agg"]["valid_action"]; row["stop_fail"]=id_s["agg"]["stop_fail"]
        else:
            row["timeout"]=row["collision"]=row["valid_action"]=row["stop_fail"]=None
        row["easi8_macro"]=m["easi8_macro"]
        for bench in EASI_8: row[f"easi8_{bench}"]=m["easi_scores"].get(bench)
        base_id="cambrian_baseline" if mid in {"c8","cambrian_baseline"} else "qwen_baseline"
        base=models[base_id]
        if mid!=base_id and base["suites"]["id_test_stratified400"] and id_s:
            row["id400_delta_pp_vs_baseline"]=100*(id_s["agg"]["success"]-base["suites"]["id_test_stratified400"]["agg"]["success"])
            row["ood_macro_delta_pp_vs_baseline"]=None if (base["ood_macro"] is None or m["ood_macro"] is None) else 100*(m["ood_macro"]-base["ood_macro"])
        else:
            row["id400_delta_pp_vs_baseline"]=0.0 if mid==base_id else None
            row["ood_macro_delta_pp_vs_baseline"]=0.0 if mid==base_id else None
        master_rows.append(row)

    fields=["model","model_id","id400_success","id400_score","scene_success","scene_score","instance_success","instance_score","category_success","category_score","template_success","template_score","geometry_success","geometry_score","ood_macro_success","easi8_macro",*[f"easi8_{bench}" for bench in EASI_8],"timeout","collision","valid_action","stop_fail","id400_delta_pp_vs_baseline","ood_macro_delta_pp_vs_baseline"]
    ART.mkdir(parents=True, exist_ok=True)
    with (ART/"model_performance_master_table.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in master_rows: w.writerow({k:r.get(k) for k in fields})

    paired_rows=[]; per_task_rows=[]; failure_rows=[]; paired_cache={}
    for a_id,b_id,kind in COMPARISONS:
        for suite in SUITES:
            a_s=models[a_id]["suites"][suite]; b_s=models[b_id]["suites"][suite]
            if not a_s or not b_s:
                paired_rows.append({"comparison":f"{a_id}_vs_{b_id}","kind":kind,"suite":suite,"n":0,"status":"missing"}); continue
            st=paired_stats(a_s["rows"], b_s["rows"]); paired_cache[(a_id,b_id,suite)]=st
            paired_rows.append({"comparison":f"{a_id}_vs_{b_id}","kind":kind,"suite":suite,"status":"ok",
                **{k:st[k] for k in st if k!="pairs"},
                "a_success":a_s["agg"]["success"],"b_success":b_s["agg"]["success"],
                "a_score":a_s["agg"]["score"],"b_score":b_s["agg"]["score"]})
            by=defaultdict(lambda:([],[]))
            for a,b in st.get("pairs",[]):
                by[a["task_type"]][0].append(a); by[a["task_type"]][1].append(b)
            for task,(ar,br) in sorted(by.items()):
                pst=paired_stats(ar,br)
                per_task_rows.append({"comparison":f"{a_id}_vs_{b_id}","suite":suite,"task_type":task,"n":pst.get("n",0),
                    "a_success":mean(x["success"] for x in ar),"b_success":mean(x["success"] for x in br),
                    "a_score":mean(x["final_score"] for x in ar),"b_score":mean(x["final_score"] for x in br),
                    "a_timeout":mean(x["timeout"] for x in ar),"b_timeout":mean(x["timeout"] for x in br),
                    "a_collision":mean(x["had_collision"] for x in ar),"b_collision":mean(x["had_collision"] for x in br),
                    "a_stop_fail":mean(x["stop_fail"] for x in ar),"b_stop_fail":mean(x["stop_fail"] for x in br),
                    "success_diff":pst.get("success_diff"),"success_ci_low":pst.get("success_ci_low"),"success_ci_high":pst.get("success_ci_high"),
                    "win":pst.get("win"),"tie":pst.get("tie"),"loss":pst.get("loss")})
            for a,b in st.get("pairs",[]):
                tags=[]
                if a["stop_fail"]: tags.append("trained_stop_fail_after_success_thr")
                if b["stop_fail"]: tags.append("baseline_stop_fail_after_success_thr")
                if a["timeout"] and not a["success"]: tags.append("trained_timeout_fail")
                if a["had_collision"] and not a["success"]: tags.append("trained_collision_fail")
                if (not a["success"]) and a["max_score"]<0.2 and b["max_score"]<0.2: tags.append("both_never_near_goal")
                if a["success"] and not b["success"]: bucket="baseline_fail_trained_success"
                elif b["success"] and not a["success"]: bucket="baseline_success_trained_fail"
                elif a["success"] and b["success"]: bucket="both_success"
                else: bucket="both_fail"
                failure_rows.append({"comparison":f"{a_id}_vs_{b_id}","suite":suite,"task_type":a["task_type"],
                    "episode_id":a["episode_id"],"scene_id":a["scene_id"],"bucket":bucket,
                    "a_success":a["success"],"b_success":b["success"],"a_final":a["final_score"],"b_final":b["final_score"],
                    "a_max":a["max_score"],"b_max":b["max_score"],"a_timeout":a["timeout"],"b_timeout":b["timeout"],
                    "a_collisions":a["collisions"],"b_collisions":b["collisions"],"tags":"|".join(tags)})

    write_csv(ART/"paired_baseline_trained_comparisons.csv", paired_rows)
    write_csv(ART/"per_task_baseline_trained.csv", per_task_rows)
    write_csv(ART/"paired_sample_buckets.csv", failure_rows)

    def suite_all_better(a_id,b_id):
        worse=[]
        for suite in OOD_SUITES:
            a_s=models[a_id]["suites"][suite]; b_s=models[b_id]["suites"][suite]
            if not a_s or not b_s: return False, [f"missing:{suite}"]
            if a_s["agg"]["success"]+1e-9 < b_s["agg"]["success"]: worse.append(suite)
        return len(worse)==0, worse

    def ci_pos(a_id,b_id,suite):
        st=paired_cache.get((a_id,b_id,suite))
        if not st or not st.get("n"): return None
        return st["success_ci_low"]>0

    md=[]
    md.append("# Full Model Performance Analysis: Baseline → Trained")
    md.append("")
    md.append("Protocol: max_steps=12, success_thr=0.65, strafe, 256x256, temp=0.1, exclude delta_control, collision on.")
    md.append("Cambrian baseline and c8 share wrapper limits: max_model_len=16384, image limit=25.")
    md.append("")
    if missing_all or missing_easi8:
        md.append("## Incomplete evaluations")
        for mid,miss in missing_all.items(): md.append(f"- `{mid}` navigation: {', '.join(miss)}")
        for mid in missing_easi8: md.append(f"- `{mid}`: EASI-8 incomplete or missing")
        md.append("")
    md.append("## A. Master table (success rate)")
    md.append("")
    md.append("| model | ID400 | scene | instance | category | template | geometry | OOD macro | EASI-8 macro | timeout | collision | valid action |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in master_rows:
        md.append("| {model} | {id400} | {scene} | {inst} | {cat} | {tmpl} | {geo} | {macro} | {easi} | {to} | {col} | {va} |".format(
            model=r["model"], id400=pct(r.get("id400_success")), scene=pct(r.get("scene_success")),
            inst=pct(r.get("instance_success")), cat=pct(r.get("category_success")), tmpl=pct(r.get("template_success")),
            geo=pct(r.get("geometry_success")), macro=pct(r.get("ood_macro_success")), easi=pct(r.get("easi8_macro")),
            to=pct(r.get("timeout")), col=pct(r.get("collision")), va=pct(r.get("valid_action"))))
    md.append("")
    md.append("## B. Same-line training deltas")
    md.append("")
    for a_id,b_id,kind in COMPARISONS:
        if kind!="same_line": continue
        md.append(f"### {MODELS[a_id][2]} vs {MODELS[b_id][2]}")
        md.append("")
        md.append("| suite | a | b | Δpp | rel% | success 95% CI | win/tie/loss | score Δ | score CI |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for suite in SUITES:
            st=paired_cache.get((a_id,b_id,suite)); a_s=models[a_id]["suites"][suite]; b_s=models[b_id]["suites"][suite]
            if not st or not a_s or not b_s or not st.get("n"):
                md.append(f"| {suite} | — | — | — | — | — | — | — | — |"); continue
            md.append("| {suite} | {a} | {b} | {dpp} | {rel} | [{lo:.3f},{hi:.3f}] | {w}/{t}/{l} | {sd:.3f} | [{slo:.3f},{shi:.3f}] |".format(
                suite=suite, a=pct(a_s["agg"]["success"]), b=pct(b_s["agg"]["success"]),
                dpp=delta_pp(a_s["agg"]["success"], b_s["agg"]["success"]), rel=rel_pct(a_s["agg"]["success"], b_s["agg"]["success"]),
                lo=st["success_ci_low"], hi=st["success_ci_high"], w=st["win"], t=st["tie"], l=st["loss"],
                sd=st["score_diff"], slo=st["score_ci_low"], shi=st["score_ci_high"]))
        md.append("")

    md.append("## C. Training-value verdicts")
    md.append("")
    v46_all,v46_worse=suite_all_better("v46","qwen_baseline")
    easi_d=None
    if models["v46"]["easi8_macro"] is not None and models["qwen_baseline"]["easi8_macro"] is not None:
        easi_d=models["v46"]["easi8_macro"]-models["qwen_baseline"]["easi8_macro"]
    md.append("### v46 vs Qwen baseline")
    md.append(f"- All five OOD stronger: **{v46_all}** (worse/missing: {v46_worse or 'none'})")
    md.append(f"- ID400 CI>0: {ci_pos('v46','qwen_baseline','id_test_stratified400')}; scene CI>0: {ci_pos('v46','qwen_baseline','ood_scene')}")
    md.append(f"- EASI-8 macro Δ: {delta_pp(models['v46']['easi8_macro'], models['qwen_baseline']['easi8_macro'])}")
    for suite in ["id_test_stratified400","ood_scene"]:
        st=paired_cache.get(("v46","qwen_baseline",suite))
        if st and st.get("n"):
            md.append(f"- `{suite}` timeoutΔ={st['timeout_diff']:+.3f} CI[{st['timeout_ci_low']:.3f},{st['timeout_ci_high']:.3f}] collisionΔ={st['collision_diff']:+.3f} W/T/L={st['win']}/{st['tie']}/{st['loss']}")
    # stop-fail summary on ID
    id_a=models["v46"]["suites"]["id_test_stratified400"]; id_b=models["qwen_baseline"]["suites"]["id_test_stratified400"]
    if id_a and id_b:
        md.append(f"- ID400 stop_fail: baseline={id_b['agg']['stop_fail']:.3f} v46={id_a['agg']['stop_fail']:.3f}")
    if v46_all and easi_d is None:
        v46_verdict,v46_reason="部分有效","导航 ID/OOD 明显提升；EASI-8 尚未完成，暂不能判断可迁移空间表征。值得作为下一轮 Qwen 线导航 baseline。"
    elif v46_all and (ci_pos("v46","qwen_baseline","id_test_stratified400") or ci_pos("v46","qwen_baseline","ood_scene")) and easi_d<0.03:
        v46_verdict,v46_reason="部分有效","导航 ID/OOD 明显提升；EASI-8 几乎不变 → 主要学到 action policy，而非可迁移静态空间表征。值得作为下一轮 Qwen 线正式 baseline。"
    elif v46_all:
        v46_verdict,v46_reason="有效","五类 OOD 与 ID 均提升。"
    else:
        v46_verdict,v46_reason="部分有效","提升不完整；待补齐 suite 后复核。"
    md.append(f"- **Verdict: {v46_verdict}** — {v46_reason}")
    md.append("")
    md.append("### v50 vs v46 (format001)")
    v50_better=0
    for suite in SUITES:
        st=paired_cache.get(("v50","v46",suite))
        if not st or not st.get("n"): continue
        if st["success_ci_low"]>0: v50_better+=1
        md.append(f"- `{suite}`: Δsuccess={st['success_diff']:+.3f} CI[{st['success_ci_low']:.3f},{st['success_ci_high']:.3f}] timeoutΔ={st['timeout_diff']:+.3f} collisionΔ={st['collision_diff']:+.3f} valid_action a/b see overall metrics")
    if v50_better==0:
        v50_verdict,v50_reason="无效","相对 v46 无稳定正效应；format001 没有证明其额外价值。下一轮应直接采用 v46 设置。"
    else:
        v50_verdict,v50_reason="部分有效","仅少数套件弱正信号，不足以作为主线。"
    md.append(f"- **Verdict: {v50_verdict}** — {v50_reason}")
    md.append("")
    md.append("### c8 vs Cambrian-LFP baseline")
    c8_missing=models["c8"]["missing"] or models["cambrian_baseline"]["missing"]
    if c8_missing: md.append(f"- Incomplete: {c8_missing}")
    for suite in SUITES:
        st=paired_cache.get(("c8","cambrian_baseline",suite)); a_s=models["c8"]["suites"][suite]; b_s=models["cambrian_baseline"]["suites"][suite]
        if not st or not a_s or not b_s or not st.get("n"): continue
        md.append(f"- `{suite}`: {pct(b_s['agg']['success'])} → {pct(a_s['agg']['success'])} (Δ={st['success_diff']:+.3f}, CI[{st['success_ci_low']:.3f},{st['success_ci_high']:.3f}], W/T/L={st['win']}/{st['tie']}/{st['loss']})")
    md.append("")
    md.append("System-level (not causal):")
    for a_id,b_id,kind in COMPARISONS:
        if kind!="system": continue
        id_a=models[a_id]["suites"]["id_test_stratified400"]; id_b=models[b_id]["suites"]["id_test_stratified400"]
        if id_a and id_b:
            md.append(f"- {MODELS[a_id][2]} vs {MODELS[b_id][2]} ID400: {pct(id_a['agg']['success'])} vs {pct(id_b['agg']['success'])} ({delta_pp(id_a['agg']['success'], id_b['agg']['success'])})")
    st_id=paired_cache.get(("c8","cambrian_baseline","id_test_stratified400"))
    ood_pos=sum(1 for suite in OOD_SUITES if (st:=paired_cache.get(("c8","cambrian_baseline",suite))) and st.get("n") and st["success_ci_low"]>0)
    if c8_missing:
        c8_verdict,c8_reason,c8_action="待定","缺 Cambrian baseline OOD/EASI-8","暂缓扩大投入，先完成公平对比"
    elif st_id and st_id.get("success_diff",0)<0.03 and ood_pos==0:
        c8_verdict,c8_reason,c8_action="无效","同线提升很小且缺稳定支持；系统级显著低于 Qwen/v46","暂停 Cambrian 线"
    else:
        c8_verdict,c8_reason,c8_action="部分有效","同线提升有限，不支持继续大规模投入","仅保留为负结果/消融"
    md.append(f"- **Verdict: {c8_verdict}** — {c8_reason}")
    md.append(f"- **Cambrian-line decision: {c8_action}**")
    md.append("")
    md.append("## D. EASI-8 vs navigation dissociation")
    md.append("Preferred interpretation if nav rises and EASI-8 is nearly flat: (1) policy shortcut/control learning primary; (2) EASI-8 measures different readout; (3) representation–control dissociation.")
    md.append("Only interpret this section after both baseline and trained checkpoints have complete EASI-8 results.")
    md.append("")
    md.append("## E. Next experiments (max 3)")
    md.append("1. arrival-and-stop if stop_fail high — unique var: stop/arrival reward.")
    md.append("2. old vs repaired data ONLY after true repaired filtered JSONL confirmed (raw 7types is not broad repair).")
    md.append("3. minimal spatial aux on v46 settings if EASI-8 remains flat.")
    md.append("")
    md.append("Artifacts: model_performance_master_table.csv, paired_baseline_trained_comparisons.csv, per_task_baseline_trained.csv, paired_sample_buckets.csv.")
    (ART/"full_model_performance_analysis.md").write_text("\n".join(md)+"\n")
    print("WROTE", ART/"model_performance_master_table.csv")
    print("WROTE", ART/"full_model_performance_analysis.md")
    print("missing", {"navigation": missing_all, "easi8": missing_easi8})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
