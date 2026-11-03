"""The sampled acceptance of an SFT start (format SFT 3.1 of 2026-10-23; the Qwen3 line's starts, decision 83; matrix SFT
epochs, the user's decision 3 of 2026-11-03): one line per result folder of an acceptance set -- an evaluation of the
200 reserve tasks (Q/eval_accept_reserve200/tasks.jsonl: training-split RL rows the RL never draws, 172 scenes) with 4
samples a task at temperature 0.8, top-p 0.92 and the token entropy recorded (evaluation run --samples 4
--temperature 0.8 --top-p 0.92 --entropy, plus the run's --format-example), i.e. a training chain's evaluate step with
these eval options and the set in eval_sets.

Per folder: success (episodes; tasks with at least one success of 4), unreadable turns (share of all turns, the
per-episode share averaged), their forms (evaluation.unreadable_form: no_action_tag, empty_action,
format_example_copied, space_instead_of_underscore, made_up_name, other), every name outside the action list over all
answers (turn_backward apart), answers without a closing tag, the token entropy (token-weighted mean over the recorded
answers), collision rate, failed episodes' end progress, and two pass checks:

- "pass": the starts' line (decision 83 and the user's decision 3 of 2026-11-03): entropy <= 0.06 and unreadable turns
  <= 1%;
- "pass_strict": format SFT v2's stricter line (decision 48): no name outside the list, turn_backward 0, unreadable
  turns <= 0.1%.

    python -m data_gen.interiorgs_v10.accept_summary --set SET_DIR --out OUT.json NAME [NAME ...]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from . import evaluation as EV

ENTROPY_MAX = 0.06
UNREADABLE_MAX = 0.01
STRICT_UNREADABLE_MAX = 0.001


def summarise(folder: Path) -> Dict[str, Any]:
    if not (folder / "done.json").is_file():
        return {"missing": True}
    model = json.loads((folder / "model.json").read_text())
    rows = [json.loads(l) for l in (folder / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    turns = [t for r in rows for t in (r.get("trajectory") or [])]
    invalid = [t for t in turns if not t.get("valid", True)]
    names: Counter = Counter()
    no_close = 0
    for t in turns:
        m = re.search(r"<action>(.*?)</action>", str(t.get("answer") or ""), re.S)
        if not m:
            no_close += 1
            continue
        for a in (x.strip() for x in m.group(1).split("|")):
            if a and a not in EV.VALID_ACTIONS:
                names[a] += 1
    by_task: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(bool(r.get("success")))
    per_ep = [sum(not t.get("valid", True) for t in r["trajectory"]) / len(r["trajectory"])
              for r in rows if r.get("trajectory")]
    acts = sum(int(r.get("actions") or 0) for r in rows)
    share = len(invalid) / max(1, len(turns))
    entropy = EV.token_entropy(rows)
    s = {
        "model": model.get("model"), "samples": model.get("samples"), "temperature": model.get("temperature"),
        "top_p": model.get("top_p"), "format_example": (model.get("env_switches") or {}).get("prompt_format_example"),
        "episodes": len(rows), "tasks": len(by_task),
        "success": round(sum(bool(r.get("success")) for r in rows) / max(1, len(rows)), 4),
        "tasks_any_success": round(sum(any(v) for v in by_task.values()) / max(1, len(by_task)), 4),
        "turns": len(turns), "unreadable_turns": len(invalid), "unreadable_share": round(share, 5),
        "unreadable_episode_mean": round(sum(per_ep) / max(1, len(per_ep)), 5),
        "forms": EV.unreadable_forms(rows),
        "names_outside_list": dict(names.most_common(10)), "names_outside_list_total": sum(names.values()),
        "turn_backward": names.get("turn_backward", 0), "answers_without_closed_tag": no_close,
        **entropy,
        "collision_rate": round(sum(int(r.get("collisions") or 0) for r in rows) / max(1, acts), 4),
        "progress_end_fail": EV.terminal_metrics(rows).get("progress_end_fail"),
    }
    sampled = bool(model.get("samples") and model.get("temperature"))
    s["pass"] = (sampled and entropy.get("entropy") is not None and entropy["entropy"] <= ENTROPY_MAX
                 and share <= UNREADABLE_MAX)
    s["pass_strict"] = bool(sum(names.values()) == 0 and share <= STRICT_UNREADABLE_MAX)
    if not sampled or "entropy" not in entropy:
        s["note"] = "not a sampled run with the entropy recorded (needs --samples, --temperature and --entropy)"
    return s


def line(name: str, s: Dict[str, Any]) -> str:
    if s.get("missing"):
        return f"{name}: missing (no done.json)"
    return (f"{name}: T {s['temperature']} x{s['samples']} | success {s['success']:.3f} ({s['episodes']} ep), any "
            f"{s['tasks_any_success']:.3f} ({s['tasks']} tasks) | entropy {s.get('entropy')} | unreadable "
            f"{s['unreadable_turns']}/{s['turns']} = {s['unreadable_share']:.4%} (ep mean {s['unreadable_episode_mean']:.4%}) "
            f"| forms {s['forms']} | outside list {s['names_outside_list_total']} {s['names_outside_list']} | "
            f"turn_backward {s['turn_backward']} | no closed tag {s['answers_without_closed_tag']} | collisions "
            f"{s['collision_rate']:.3f} | fail end progress {s['progress_end_fail']} | pass {s['pass']} "
            f"(strict {s['pass_strict']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="the acceptance set folder (result folders inside)")
    ap.add_argument("--out", required=True, help="OUT.json: every folder's summary")
    ap.add_argument("names", nargs="+", help="result names in the set")
    a = ap.parse_args()
    out = {n: summarise(Path(a.set) / n) for n in a.names}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for n, s in out.items():
        print(line(n, s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
