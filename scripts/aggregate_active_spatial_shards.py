#!/usr/bin/env python3
"""Merge native Active Spatial result files produced by contiguous shards."""
from __future__ import annotations
import argparse, json, math, shutil
from pathlib import Path
from evaluation.metrics import format_results_table

def merge_group(items):
    items=[x for x in items if x and x.get("num_episodes",0)]
    if not items: return {"num_episodes":0,"note":"no test data"}
    n=sum(int(x["num_episodes"]) for x in items); out={"num_episodes":n}
    keys=set().union(*(x.keys() for x in items))
    for key in sorted(keys):
        if key in {"num_episodes","task_type","std_final_score","num_task_types_tested"}: continue
        vals=[(float(x[key]),int(x["num_episodes"])) for x in items if isinstance(x.get(key),(int,float))]
        if vals: out[key]=sum(v*w for v,w in vals)/sum(w for _,w in vals)
    pairs=[(float(x.get("mean_final_score",0)),int(x["num_episodes"])) for x in items]
    if pairs:
        mean=sum(v*w for v,w in pairs)/n
        var=sum((float(x.get("std_final_score",0))**2+(v-mean)**2)*w for x,(v,w) in zip(items,pairs))/n
        out["mean_final_score"]=mean; out["std_final_score"]=math.sqrt(max(0,var))
    if "task_type" in items[0]: out["task_type"]=items[0]["task_type"]
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--sweep-root",required=True)
    ap.add_argument("--experiment",required=True); ap.add_argument("--step",type=int,required=True); a=ap.parse_args()
    root=Path(a.sweep_root)/a.experiment/f"global_step_{a.step}"; files=sorted(root.glob("id_shard*/model/results_model.json"))
    if not files: raise SystemExit(f"no shard results under {root}")
    docs=[json.loads(p.read_text()) for p in files]; metrics={}
    for group in ("overall","by_task_type","by_category"):
        if group=="overall": metrics[group]=merge_group([(d.get("metrics") or {}).get(group,{}) for d in docs])
        else:
            names=set().union(*(set((d.get("metrics") or {}).get(group,{})) for d in docs))
            metrics[group]={name:merge_group([(d.get("metrics") or {}).get(group,{}).get(name,{}) for d in docs]) for name in sorted(names)}
    metrics["overall"]["task_type"]="ALL"
    metrics["overall"]["num_task_types_tested"]=sum(1 for x in metrics.get("by_task_type",{}).values() if x.get("num_episodes",0))
    metrics["summary_table"]=[]
    for name,x in [("ALL",metrics["overall"])] + list(metrics.get("by_task_type",{}).items()):
        if not x.get("num_episodes",0): continue
        metrics["summary_table"].append({"task":name,"n":x["num_episodes"],"success%":f"{x.get('success_rate',0)*100:.1f}","final_score":f"{x.get('mean_final_score',0):.3f}","improvement":f"{x.get('mean_score_improvement',0):.3f}","spl":f"{x.get('spl',0):.3f}","steps":f"{x.get('mean_steps',0):.1f}","collisions":f"{x.get('mean_collisions',0):.1f}","monotonic%":f"{x.get('monotonic_improvement_rate',0)*100:.1f}"})
    meta_path=root.parent.parent.parent/"shard_meta.json"; meta=json.loads(meta_path.read_text()) if meta_path.exists() else None; episodes=[]
    for idx,(p,d) in enumerate(zip(files,docs)):
        offset=meta["shards"][idx]["start"] if meta else 0
        for ep0 in d.get("episodes",[]):
            ep=dict(ep0); ep["episode_id"]=int(ep.get("episode_id",0))+offset; episodes.append(ep)
    episodes.sort(key=lambda x:x.get("episode_id",0)); final=root/"id_test"/"model"; final.mkdir(parents=True,exist_ok=True)
    payload={"metrics":metrics,"formatted_table":format_results_table(metrics),"episodes":episodes}
    (final/"results_model.json").write_text(json.dumps(payload,indent=2)+"\n"); shutil.copy2(files[0].parent/"eval_config.yaml",final/"eval_config.yaml")
    (final/"parallel_eval.log").write_text("Merged contiguous 8-way ID shards.\n\n"+payload["formatted_table"]+"\n")
    (root/"sharded_completion.json").write_text(json.dumps({"shards":len(files),"episodes":len(episodes),"complete":len(episodes)>0},indent=2)+"\n")
    print(payload["formatted_table"])
if __name__=="__main__": main()
