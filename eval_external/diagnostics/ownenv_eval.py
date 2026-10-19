"""Our environment's own evaluation for an eval_external model, optionally with the dialogue history shuffled or
dropped (diagnostics 3c-1 and 3c-3).  Runs inside diagnostics/ownenv_job.sh (one card: render service + vLLM).

    python -m eval_external.diagnostics.ownenv_eval --model SI-7B --name SI-7B
    python -m eval_external.diagnostics.ownenv_eval --model ours-SFT --name ours-SFT__shuffle --history-mode shuffle
    python -m eval_external.diagnostics.ownenv_eval --model ours-SFT --name ours-SFT__replace --history-mode replace

It is data_gen/interiorgs_v10/evaluation.py "run" with its defaults, i.e. the setting of the round-10 and round-11
test evaluations (the env's prompt format, greedy, history on, 12 turns, default rules; round 11 re-ran two
round-10 models with it and got the same numbers to the episode).  Two things are added around the model call:

* repetition_penalty 1.05, stated explicitly.  Those evaluations ran each model folder with its own
  generation_config.json, which for Qwen2.5-VL-7B-Instruct and for our checkpoints sets 1.05 (vLLM applies it to
  the parameters the call leaves unset); our model views carry 1.0, so the call states 1.05 for every model.
* --history-mode: normal (as evaluation.py), shuffle (every call gets the same earlier turns -- frame, text and
  answer kept together -- in a random order, seeded by the turn's text and the number of earlier turns), drop
  (every call gets no earlier turns; everything else unchanged), replace (every call with k earlier turns gets k
  turns of another task of the same scene instead -- user text, frame and the model's own answer, taken from the
  same model's normal run, see donor_pools; same count and layout, other content).  Every user turn repeats the
  task line, so no mode hides the task itself.

The 279 test tasks are a copy of interiorgs_v10_r10_20261015/eval_test/tasks.jsonl in ROOT/runs/ownenv/eval_test
(the original folder is never written to); results go to ROOT/runs/ownenv/eval_test/<name>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

from eval_external.adapters import models as M
from eval_external.common.manifest import git_state, write_manifest

ROOT = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets")
SOURCE_EVAL = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r10_20261015/eval_test")
EVAL = ROOT / "runs" / "ownenv" / "eval_test"
REPETITION_PENALTY = 1.05
WORKTREE = Path(__file__).resolve().parents[2]


def ensure_eval_copy() -> Path:
    EVAL.mkdir(parents=True, exist_ok=True)
    for name in ("tasks.jsonl", "selection.json"):
        if not (EVAL / name).exists():
            shutil.copy2(SOURCE_EVAL / name, EVAL / name)
    return EVAL


def reorder(earlier: list, mode: str, user_text: str) -> list:
    """normal / shuffle / drop.  A turn is (its user text, its frame, the model's answer) and moves as one piece:
    shuffle reorders whole turns, it never pairs a frame with another turn's text or answer."""
    if mode == "drop":
        return []
    if mode == "shuffle" and len(earlier) > 1:
        seed = int.from_bytes(hashlib.sha256(f"{len(earlier)}:{user_text}".encode()).digest()[:4], "big")
        out = list(earlier)
        random.Random(seed).shuffle(out)
        return out
    return list(earlier)


def donor_pools(tasks: list[dict], donor_rows: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """--history-mode replace.  For each task index: the (donor task index, turn) pairs whose turns replace its
    earlier turns, in order: the donor run's other tasks of the same scene (by task index, each with all its turns),
    then, only for the rare call that needs more, the other scenes' tasks (nearest index first)."""
    turns = {r["index"]: len(r["trajectory"]) for r in donor_rows if "error" not in r}
    pools = {}
    for i, t in enumerate(tasks):
        same = [j for j in sorted(turns) if j != i and tasks[j]["scene_id"] == t["scene_id"]]
        other = sorted((j for j in turns if j != i and tasks[j]["scene_id"] != t["scene_id"]),
                       key=lambda j: (abs(j - i), j))
        pools[i] = [(j, k) for j in same + other for k in range(turns[j])]
    return pools


class Replacer:
    """Gives a call with k earlier turns the first k turns of its task's donor pool, frames read from the donor
    run; logs per call how many of them came from the same scene."""

    def __init__(self, donor: str, log: Path):
        self.rows = {r["index"]: r for r in map(json.loads, open(EVAL / donor / "results.jsonl")) if "error" not in r}
        tasks = [json.loads(line) for line in open(EVAL / "tasks.jsonl")]
        self.pools = donor_pools(tasks, list(self.rows.values()))
        self.scene = {i: t["scene_id"] for i, t in enumerate(tasks)}
        self.donor, self.log, self.index = donor, log, None

    def turns(self, k: int) -> list:
        from PIL import Image

        picked = self.pools[self.index][:k]
        out = []
        for j, t in picked:
            step = self.rows[j]["trajectory"][t]
            with Image.open(EVAL / self.donor / step["frame"]) as im:
                out.append((step["user"], im.convert("RGB"), step["answer"]))
        same = sum(self.scene[j] == self.scene[self.index] for j, _ in picked)
        with open(self.log, "a") as f:
            f.write(json.dumps({"index": self.index, "k": k, "same_scene_turns": same,
                                "donors": sorted({j for j, _ in picked})}) + "\n")
        return out


_ANSWER_LINE = re.compile(r"^\s*answer\s*[:：]\s*(.+?)\s*$", re.I)


def lenient_rewrite(text: str) -> str | None:
    """Diagnostic only (--lenient; one rule for every model, decided 2026-10-19): a reply without <action> tags
    whose last non-empty line is "Answer: a|b|c" becomes "<action>a|b|c|</action>"; the env still checks every
    action name.  Any other reply is passed on unchanged (None)."""
    if "<action>" in text:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    m = _ANSWER_LINE.match(lines[-1]) if lines else None
    if not m:
        return None
    actions = [a for a in re.split(r"[|,;\s]+", m.group(1).strip().strip(".")) if re.fullmatch(r"[a-z_]+", a)]
    return "<action>" + "|".join(actions) + "|</action>" if actions else None


def patched_policy(evaluation, mode: str, lenient_log: Path | None = None, replacer: Replacer | None = None):
    base = evaluation.VllmPolicy

    class Policy(base):
        def generate(self, system, user_text, image, earlier=(), as_sft=False, seed=None):
            from vllm import SamplingParams

            params = SamplingParams(temperature=self.temperature, top_p=self.top_p, seed=seed, max_tokens=384,
                                    stop=["</action>"], include_stop_str_in_output=True,
                                    repetition_penalty=REPETITION_PENALTY)
            history = (replacer.turns(len(earlier)) if mode == "replace" and earlier
                       else reorder(list(earlier), mode, user_text))
            msgs = evaluation.chat_messages(system, history, user_text, image, evaluation._image_url, as_sft)
            out = self.llm.chat(msgs, sampling_params=params, use_tqdm=False)
            self.last_prompt_tokens = len(out[0].prompt_token_ids)
            text = out[0].outputs[0].text
            if lenient_log is not None:
                fixed = lenient_rewrite(text)
                with open(lenient_log, "a") as f:
                    f.write(json.dumps({"original": text, "rewritten": fixed}) + "\n")
                if fixed is not None:
                    text = fixed
            return text

    return Policy


def summarize(name: str) -> dict:
    """Success rate with its Wilson interval, invalid turns, collisions and the failure breakdown of one run."""
    from eval_external.common.stats import wilson
    from eval_external.diagnostics.failure_breakdown import breakdown

    rows = [json.loads(line) for line in open(EVAL / name / "results.jsonl") if line.strip()]
    ok = [r for r in rows if "error" not in r]
    k = sum(bool(r["success"]) for r in ok)
    p, lo, hi = wilson(k, len(ok))
    turns = sum(r["turns"] for r in ok) or 1
    actions = sum(r["actions"] for r in ok) or 1
    return {"name": name, "n": len(ok), "errors": len(rows) - len(ok), "success": k,
            "success_rate": round(100 * p, 2), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
            "invalid_turn_pct": round(100 * sum(r["invalid_turns"] for r in ok) / turns, 2),
            "collision_per_action": round(sum(r["collisions"] for r in ok) / actions, 4),
            "breakdown": breakdown(rows), "manifest": json.loads((EVAL / name / "manifest.json").read_text())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="eval_external model code")
    ap.add_argument("--name", required=True, help="result folder name under the eval copy")
    ap.add_argument("--history-mode", choices=("normal", "shuffle", "drop", "replace"), default="normal")
    ap.add_argument("--donor-run", help="replace: the run whose turns stand in (default: this model's normal run)")
    ap.add_argument("--lenient", action="store_true",
                    help="diagnostic: wrap an untagged last line 'Answer: a|b' in <action> tags (logged per call)")
    args = ap.parse_args()
    ev = ensure_eval_copy()
    sys.path.insert(0, str(WORKTREE))
    from data_gen.interiorgs_v10 import evaluation

    original_generate = evaluation.VllmPolicy.generate
    (ev / args.name).mkdir(parents=True, exist_ok=True)
    lenient_log = (ev / args.name / "lenient_rewrites.jsonl") if args.lenient else None
    replacer = None
    if args.history_mode == "replace":
        replacer = Replacer(args.donor_run or args.model, ev / args.name / "replace_log.jsonl")
        original_run_episode = evaluation.run_episode

        def run_episode(env, policy, index, *a, **k):  # cmd_run runs the tasks one after another
            replacer.index = index
            return original_run_episode(env, policy, index, *a, **k)

        evaluation.run_episode = run_episode
    evaluation.VllmPolicy = patched_policy(evaluation, args.history_mode, lenient_log, replacer)
    write_manifest(
        ev / args.name, models=[M.describe(args.model)],
        benchmark={"name": "Active Spatial own env, 279 test tasks (round-10 eval_test)", "items_expected": 279,
                   "source": {"tasks": str(SOURCE_EVAL / "tasks.jsonl"), "code": git_state(WORKTREE)},
                   "metric": "success rate (task met at any turn), as evaluation.py"},
        backend={"name": "data_gen/interiorgs_v10/evaluation.py run (defaults)",
                 "policy_override": "VllmPolicy.generate with repetition_penalty and history mode",
                 "replaced_generate_of": getattr(original_generate, "__qualname__", None)},
        prompt={"impl": "the env's own prompt (prompt_format of rl_env_v10_S0.yaml)", "history_mode": args.history_mode,
                "donor_run": (args.donor_run or args.model) if replacer else None,
                "lenient_answer_parsing": bool(args.lenient)},
        frames=None, decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 384, "stop": ["</action>"],
                               "repetition_penalty": REPETITION_PENALTY})
    view = str(M.view_dir(args.model))
    raise SystemExit(evaluation.main(["run", "--eval", str(ev), "--name", args.name, "--model", view]))


if __name__ == "__main__":
    main()
