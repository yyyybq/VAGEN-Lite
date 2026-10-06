#!/usr/bin/env python3
"""Create the first frozen cross table, leaving unavailable model cells explicit."""
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--qa-result",required=True); ap.add_argument("--output",required=True); ap.add_argument("--active-result",default="")
    a=ap.parse_args(); qa=json.loads(Path(a.qa_result).read_text()); active=json.loads(Path(a.active_result).read_text()) if a.active_result and Path(a.active_result).exists() else None
    rows=[]
    for model, q, act, dq, da in [("Base","NOT_RUN: model prediction bank not supplied","NOT_RUN: frozen Active run not supplied","BASELINE","BASELINE"),("QA-trained","NOT_RUN: QA-SFT canary not run","NOT_RUN: QA actor canary not run","NOT_RUN: requires QA-SFT","NOT_RUN: requires QA-SFT actor"),("Act-trained","NOT_RUN: Act checkpoint QA bank not run","NOT_RUN: see frozen Active result","NOT_RUN: requires Act checkpoint QA","NOT_RUN: requires frozen Active result" )]:
        rows.append({"model":model,"qa_test":q,"active_closed_loop":act,"qa_delta_vs_base":dq,"active_delta_vs_base":da})
    Path(a.output).write_text(json.dumps({"qa_result":qa,"active_result":active,"rows":rows},indent=2)+"\n")
    print(json.dumps({"rows":rows},indent=2))
if __name__ == "__main__": main()
