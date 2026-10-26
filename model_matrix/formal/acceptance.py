"""The formal SFT's acceptance (the model line, step 3; the user's decision 2 of 2026-10-26): the picked epoch on the
acceptance set (200 reserve tasks), two runs -- sampled (4 episodes per task, temperature 0.8, top-p 0.92: the
original acceptance) and greedy (added) -- each with the share of unreadable turns (evaluation.metrics' invalid_rate:
unreadable turns / all turns), the names outside the action table (every name in every answer's <action> block that
is not one of the 6 actions), the forms of the unreadable answers (evaluation.unreadable_forms) and the answers'
token entropy (run --entropy).  The rule, applied to both runs and reported for both: unreadable turns at most 1%,
no name outside the table.  Nothing is decided here: RL is submitted by hand after the report.

    python -m model_matrix.formal.acceptance --eval EVAL_DIR --model sft_q3formal_e7

reads EVAL_DIR/<model>_s4 and EVAL_DIR/<model>_greedy, writes EVAL_DIR/acceptance_<model>.json and .md.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

from data_gen.interiorgs_v10.evaluation import VALID_ACTIONS, _jsonl, metrics, unreadable_forms

INVALID_MAX = 0.01
RUNS = {"sampled": "_s4", "greedy": "_greedy"}


def names_outside(rows: Sequence[Dict[str, Any]]) -> Counter:
    """{name: count} of the action names outside the table, over every recorded answer."""
    out: Counter = Counter()
    for r in rows:
        if "error" in r:
            continue
        for t in r.get("trajectory") or []:
            for block in re.findall(r"<action>(.*?)</action>", str(t.get("answer") or ""), re.S):
                out.update(a.strip() for a in block.split("|") if a.strip() and a.strip() not in VALID_ACTIONS)
    return out


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    m = metrics(rows)
    outside = names_outside(rows)
    ok = [r for r in rows if "error" not in r]
    s = {"episodes": m["episodes"], "errors": m["errors"], "turns": sum(r["turns"] for r in ok),
         "success": m["success"], "invalid_rate": m["invalid_rate"], "names_outside": sum(outside.values()),
         "names_outside_top": dict(outside.most_common(10)), "unreadable_forms": unreadable_forms(rows),
         "entropy": m.get("entropy"), "entropy_tokens": m.get("entropy_tokens")}
    if "pass_at_n" in m:
        s.update(samples=m["samples"], tasks=m["tasks"], pass_at_n=m["pass_at_n"])
    s["readable_ok"] = s["invalid_rate"] is not None and s["invalid_rate"] <= INVALID_MAX
    s["names_ok"] = s["names_outside"] == 0
    s["passes"] = bool(s["readable_ok"] and s["names_ok"])
    return s


def accept(eval_dir: Path, model: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"model": model, "eval": str(eval_dir), "rule": {"invalid_max": INVALID_MAX, "names_outside": 0}}
    for mode, suffix in RUNS.items():
        d = Path(eval_dir) / f"{model}{suffix}"
        if not (d / "done.json").is_file():
            raise SystemExit(f"{d} has no complete results")
        out[mode] = summarize(_jsonl(d / "results.jsonl"))
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"验收：{model}（{eval_dir}）", "",
             "| 跑法 | 局数 | 回合数 | 成功率 | 读不出的回合比例 | 动作表外的名字 | 读不出的形式 | 熵 | 过线 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode, label in (("sampled", "采样（每题 4 局，温度 0.8）"), ("greedy", "贪心")):
        s = out[mode]
        lines.append(f"| {label} | {s['episodes']} | {s['turns']} | {fmt(s['success'])} | {fmt(s['invalid_rate'])} | "
                     f"{s['names_outside']} | {'，'.join(f'{k} {v}' for k, v in s['unreadable_forms'].items()) or '—'} | "
                     f"{fmt(s['entropy'])} | {'是' if s['passes'] else '否'} |")
    (Path(eval_dir) / f"acceptance_{model}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (Path(eval_dir) / f"acceptance_{model}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    accept(a.eval, a.model)


if __name__ == "__main__":
    main()
