"""Diagnostic 2: success with the episode history as it is, shuffled, or left out.

    # the evaluation (data_gen/interiorgs_v10/evaluation.py run) with one history mode; same card setup as an
    # evaluation job (render service + vLLM); everything after "--" goes to "evaluation.py run" unchanged
    python -m eval.diagnostics.history_ablation run --history shuffle -- --eval EVAL --name M_hist_shuffle --model PATH
    # the three success rates and the task-by-task differences against "normal"
    python -m eval.diagnostics.history_ablation table --eval EVAL --normal M_hist_normal --shuffle M_hist_shuffle \
        --none M_hist_none --out OUT_PREFIX

normal  every model call gets all earlier turns in order (the evaluation's default, --history);
shuffle the same turns in another order: each earlier turn keeps its own frame, text and answer, only their order
        changes (a fixed permutation per call that differs from the real order whenever there are two or more);
none    no earlier turns (--no-history).
The evaluation module is not changed: "shuffle" swaps in a policy class that reorders ``earlier`` before every
model call.  Not run on a GPU yet (2026-10-18); try one task first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MODES = ("normal", "shuffle", "none")


def permutation(n: int, key: str) -> List[int]:
    """A fixed order of n earlier turns, different from 0..n-1 when n >= 2."""
    order = list(range(n))
    if n < 2:
        return order
    random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)).shuffle(order)
    if order == list(range(n)):
        order = order[1:] + order[:1]
    return order


def shuffled_policy(base: type) -> type:
    class ShuffledHistory(base):  # type: ignore[misc, valid-type]
        def generate(self, system: str, user_text: str, image: Any, earlier: Sequence[Any] = (), **kw: Any) -> str:
            earlier = list(earlier)
            order = permutation(len(earlier), f"{user_text}|{len(earlier)}")
            return super().generate(system, user_text, image, earlier=[earlier[i] for i in order], **kw)
    return ShuffledHistory


def cmd_run(mode: str, rest: Sequence[str]) -> int:
    from data_gen.interiorgs_v10 import evaluation as EV

    if any(a in ("--history", "--no-history") for a in rest):
        raise SystemExit("give the history mode with --history normal|shuffle|none, not to evaluation.py")
    if mode == "shuffle":
        EV.VllmPolicy = shuffled_policy(EV.VllmPolicy)
    return EV.main(["run", *rest, "--no-history" if mode == "none" else "--history"])


def success(path: Path) -> Dict[int, bool]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[int(r["index"])] = bool(r["success"])
    return out


def cmd_table(args: argparse.Namespace) -> None:
    ev = Path(args.eval)
    runs = {m: success(ev / getattr(args, m) / "results.jsonl") for m in MODES if getattr(args, m)}
    base = runs.get("normal", {})
    rows, res = [], {}
    for m, s in runs.items():
        n = len(s)
        rate = sum(s.values()) / n if n else None
        both = [i for i in s if i in base]
        only_base = sum(1 for i in both if base[i] and not s[i])
        only_this = sum(1 for i in both if s[i] and not base[i])
        res[m] = {"name": getattr(args, m), "tasks": n, "success": rate, "only_normal": only_base, "only_this": only_this}
        rows.append(f"| {m} | `{getattr(args, m)}` | {n} | {100 * rate:.1f}% | "
                    + ("–" if m == "normal" else f"{only_base} / {only_this}") + " |")
    md = ["# 历史打乱：三档成功率", "", "| 历史 | 结果目录 | 题数 | 成功率 | 只有 normal 成功 / 只有这一档成功 |",
          "| --- | --- | --- | --- | --- |"] + rows
    Path(args.out + ".json").write_text(json.dumps(res, indent=1))
    Path(args.out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        head, rest = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
        ap = argparse.ArgumentParser(prog="history_ablation run")
        ap.add_argument("--history", choices=MODES, required=True)
        a = ap.parse_args(head[1:])
        return cmd_run(a.history, rest)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--eval", required=True)
    for m in MODES:
        t.add_argument(f"--{m}", default="", help=f"result folder of the '{m}' run")
    t.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cmd_table(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
