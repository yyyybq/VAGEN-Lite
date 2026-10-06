#!/usr/bin/env python3
"""Prepare contiguous ID shards and a suite config for multi-GPU eval."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True); ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--gs-root", required=True)
    ap.add_argument("--shards", type=int, default=8); args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lines = [x for x in Path(args.input_jsonl).read_text().splitlines() if x.strip()]
    n = len(lines); meta = {"input": args.input_jsonl, "total": n, "shards": []}; suites = []
    for i in range(args.shards):
        start = (n*i)//args.shards; end = (n*(i+1))//args.shards
        shard = out/f"id_shard{i}.jsonl"; shard.write_text("\n".join(lines[start:end])+"\n")
        meta["shards"].append({"index":i,"start":start,"end":end,"count":end-start})
        suites.append({"name":f"id_shard{i}","jsonl_path":str(shard),"seed_offset":start})
    cfg = yaml.safe_load(Path(args.base_config).read_text())
    cfg.setdefault("defaults",{}).setdefault("env",{})["gs_root"] = args.gs_root
    cfg["suites"] = suites
    (out/"suite_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out/"shard_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta, indent=2))
if __name__ == "__main__": main()
