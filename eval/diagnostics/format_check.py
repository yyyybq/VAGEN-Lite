"""Output-format check (docs/transition_qa step 6): the env's parser reads the same actions in both formats.

    python -m eval.diagnostics.format_check --eval EVAL_DIR [--eval ...] --out OUT_PREFIX

For every model answer in the evaluation histories (EVAL/NAME/results.jsonl, per-turn ``answer`` and the actions
the env executed, ``actions``), the answer is written in both output formats and parsed with the env's own parser
(``ActiveSpatialEnv._default_parse_func``):

  action_only   the answer with its <think>...</think> block removed;
  think_action  the answer with a <think>...</think> block in front when it has none.

A turn agrees when both versions give the same action list, the same as the actions the evaluation recorded, and
the same parse success.  Also reported per model: the share of answers the parser could not read at all.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

STRAFE = {"move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"}
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def parser():
    from vagen.envs.active_spatial.env import ActiveSpatialEnv

    stub = SimpleNamespace(_allowed_actions=set(STRAFE))
    return lambda text: ActiveSpatialEnv._default_parse_func(stub, text, action_sep="|", max_actions=5)


def as_action_only(text: str) -> str:
    return THINK.sub("", text, count=1).strip()


def as_think_action(text: str) -> str:
    return text if "<think>" in text.lower() else "<think>(reasoning)</think>" + text


def check_folder(folder: Path, parse) -> Dict[str, Any]:
    fmt = json.loads((folder / "model.json").read_text()).get("prompt_format") if (folder / "model.json").exists() else None
    c: collections.Counter = collections.Counter()
    examples: List[Dict[str, Any]] = []
    for line in (folder / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for turn in row.get("trajectory") or []:
            text = turn.get("answer")
            if text is None:
                continue
            c["turns"] += 1
            c["has_think"] += int("<think>" in text.lower())
            a, t = parse(as_action_only(text)), parse(as_think_action(text))
            recorded = [str(x) for x in turn.get("actions") or []]
            same = a["actions"] == t["actions"] == recorded and bool(a["format_correct"]) == bool(t["format_correct"])
            c["agree"] += int(same)
            c["unreadable"] += int(not t["actions"])
            if not same and len(examples) < 3:
                examples.append({"answer": text[:400], "recorded": recorded, "action_only": a["actions"],
                                 "think_action": t["actions"]})
    return {"prompt_format": fmt, **dict(c), "examples": examples}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    parse = parser()
    res: Dict[str, Any] = {}
    for ev in args.eval:
        for folder in sorted(Path(ev).glob("*/results.jsonl")):
            res[f"{Path(ev).parent.name}/{Path(ev).name}/{folder.parent.name}"] = check_folder(folder.parent, parse)
    total = collections.Counter()
    for r in res.values():
        total.update({k: v for k, v in r.items() if isinstance(v, int)})
    md = ["# 输出格式检查：两种格式读出的动作是否相同", "",
          f"共 {len(res)} 个结果目录、{total['turns']} 个回合；两种写法读出的动作序列和评测记录的完全相同：{total['agree']} 个回合"
          f"（{100.0 * total['agree'] / max(total['turns'], 1):.2f}%）。", "",
          "| 结果目录 | 模型的提示格式 | 回合 | 带思考文字的回答 | 两种写法一致 | 读不出动作 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, r in res.items():
        n = r.get("turns", 0)
        md.append(f"| {name} | {r['prompt_format']} | {n} | {r.get('has_think', 0)} | {r.get('agree', 0)} | "
                  f"{r.get('unreadable', 0)}（{100.0 * r.get('unreadable', 0) / max(n, 1):.1f}%） |")
    Path(args.out + ".json").write_text(json.dumps({"total": dict(total), "folders": res}, indent=1, ensure_ascii=False))
    Path(args.out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:3]))


if __name__ == "__main__":
    main()
