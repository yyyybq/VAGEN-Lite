"""Names outside the action table in RL training, per block of 25 updates (the user's decision 22 of 2026-10-26: no
longer a gate, but listed as a quality metric).  From an RL run's rollout dumps (RUN/rollout_data/<update>.jsonl, one
row per episode; "output" holds every answer of the episode, each followed by the next observation, which itself
quotes the format example -- so only the assistant answers are read): per block, the answers, the answers the env
could not read (evaluation.unreadable_form other than a readable one), their forms, and the names outside the table
(count and the most frequent).

    python -m model_matrix.formal.quality --run RUN_DIR [--run RUN_DIR ...] [--block 25] [--out OUT]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from data_gen.interiorgs_v10.evaluation import VALID_ACTIONS, unreadable_form

END = "<|im_end|>"
ASSISTANT = "<|im_start|>assistant"


TURN = "<|im_start|>"


def _cut(text: str) -> str:
    """An answer up to its end token or, when it was cut off without one, up to the next turn's header."""
    return text.split(END)[0].split(TURN)[0]


def answers(output: str) -> List[str]:
    """The assistant answers of one episode's output: the first one, then each one after an assistant header."""
    parts = output.split(ASSISTANT)
    out = [_cut(parts[0])] + [_cut(p.lstrip("\n")) for p in parts[1:]]
    return [a.strip() for a in out if a.strip()]


def readable(answer: str) -> bool:
    m = re.search(r"<action>(.*?)</action>", answer, re.S)
    if not m:
        return False
    acts = [a.strip() for a in m.group(1).split("|") if a.strip()]
    return bool(acts) and len(acts) <= 5 and all(a in VALID_ACTIONS for a in acts)


def block_quality(run: Path, block: int = 25) -> Dict[str, Dict]:
    files = sorted((Path(run) / "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    out: Dict[str, Dict] = {}
    for f in files:
        if not f.stem.isdigit():
            continue
        u = int(f.stem)
        lo = (u - 1) // block * block + 1
        b = out.setdefault(f"{lo}-{lo + block - 1}", {"updates": set(), "answers": 0, "unreadable": 0,
                                                       "forms": Counter(), "outside": Counter()})
        b["updates"].add(u)
        for line in f.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            for a in answers(json.loads(line).get("output") or ""):
                b["answers"] += 1
                for blk in re.findall(r"<action>(.*?)</action>", a, re.S):
                    b["outside"].update(x.strip() for x in blk.split("|") if x.strip() and x.strip() not in VALID_ACTIONS)
                if not readable(a):
                    b["unreadable"] += 1
                    b["forms"][unreadable_form(a) if unreadable_form(a) != "other" else "too_many_or_other"] += 1
    return {k: {"updates": len(v["updates"]), "answers": v["answers"], "unreadable": v["unreadable"],
                "unreadable_share": round(v["unreadable"] / v["answers"], 4) if v["answers"] else None,
                "forms": dict(v["forms"].most_common()), "names_outside": sum(v["outside"].values()),
                "names_outside_top": dict(v["outside"].most_common(5))} for k, v in out.items()}


def report(results: Dict[str, Dict[str, Dict]]) -> str:
    lines = ["| 臂 | 更新 | 回答数 | 按动作表严格读不出（含超过 5 个动作；判定用训练端自己的读不出比例） | 形式 | 表外名字（个数 / 最多的） |", "| --- | --- | --- | --- | --- | --- |"]
    for run, blocks in results.items():
        for k, b in blocks.items():
            forms = "，".join(f"{f} {n}" for f, n in b["forms"].items()) or "—"
            top = "，".join(f"`{n}` {c}" for n, c in b["names_outside_top"].items()) or "—"
            share = "—" if b["unreadable_share"] is None else f"{100 * b['unreadable_share']:.2f}%"
            lines.append(f"| {run} | {k}（{b['updates']} 次） | {b['answers']} | {share} | {forms} | "
                         f"{b['names_outside']}（{top}） |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", type=Path, required=True)
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    results = {r.name.replace("rl_run_", ""): block_quality(r, a.block) for r in a.run}
    text = report(results)
    if a.out:
        a.out.with_suffix(".json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
        a.out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
