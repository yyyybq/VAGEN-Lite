"""Our environment's own evaluation for an eval_external model, optionally with the dialogue history shuffled or
dropped (diagnostics 3c-1 and 3c-3).  Runs inside diagnostics/ownenv_job.sh (one card: render service + vLLM).

    python -m eval_external.diagnostics.ownenv_eval --model SI-7B --name SI-7B
    python -m eval_external.diagnostics.ownenv_eval --model ours-SFT --name ours-SFT__shuffle --history-mode shuffle

It is data_gen/interiorgs_v10/evaluation.py "run" with its defaults, i.e. the setting of the round-10 and round-11
test evaluations (the env's prompt format, greedy, history on, 12 turns, default rules; round 11 re-ran two
round-10 models with it and got the same numbers to the episode).  Two things are added around the model call:

* repetition_penalty 1.05, stated explicitly.  Those evaluations ran each model folder with its own
  generation_config.json, which for Qwen2.5-VL-7B-Instruct and for our checkpoints sets 1.05 (vLLM applies it to
  the parameters the call leaves unset); our model views carry 1.0, so the call states 1.05 for every model.
* --history-mode: normal (as evaluation.py), shuffle (every call gets the same earlier turns -- frame, text and
  answer kept together -- in a random order, seeded by the turn's text and the number of earlier turns), drop
  (every call gets no earlier turns; everything else unchanged).

The 279 test tasks are a copy of interiorgs_v10_r10_20261015/eval_test/tasks.jsonl in ROOT/runs/ownenv/eval_test
(the original folder is never written to); results go to ROOT/runs/ownenv/eval_test/<name>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
    if mode == "drop":
        return []
    if mode == "shuffle" and len(earlier) > 1:
        seed = int.from_bytes(hashlib.sha256(f"{len(earlier)}:{user_text}".encode()).digest()[:4], "big")
        out = list(earlier)
        random.Random(seed).shuffle(out)
        return out
    return list(earlier)


def patched_policy(evaluation, mode: str):
    base = evaluation.VllmPolicy

    class Policy(base):
        def generate(self, system, user_text, image, earlier=(), as_sft=False, seed=None):
            from vllm import SamplingParams

            params = SamplingParams(temperature=self.temperature, top_p=self.top_p, seed=seed, max_tokens=384,
                                    stop=["</action>"], include_stop_str_in_output=True,
                                    repetition_penalty=REPETITION_PENALTY)
            msgs = evaluation.chat_messages(system, reorder(list(earlier), mode, user_text), user_text, image,
                                            evaluation._image_url, as_sft)
            out = self.llm.chat(msgs, sampling_params=params, use_tqdm=False)
            self.last_prompt_tokens = len(out[0].prompt_token_ids)
            return out[0].outputs[0].text

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
    ap.add_argument("--history-mode", choices=("normal", "shuffle", "drop"), default="normal")
    args = ap.parse_args()
    ev = ensure_eval_copy()
    sys.path.insert(0, str(WORKTREE))
    from data_gen.interiorgs_v10 import evaluation

    original_generate = evaluation.VllmPolicy.generate
    evaluation.VllmPolicy = patched_policy(evaluation, args.history_mode)
    write_manifest(
        ev / args.name, models=[M.describe(args.model)],
        benchmark={"name": "Active Spatial own env, 279 test tasks (round-10 eval_test)", "items_expected": 279,
                   "source": {"tasks": str(SOURCE_EVAL / "tasks.jsonl"), "code": git_state(WORKTREE)},
                   "metric": "success rate (task met at any turn), as evaluation.py"},
        backend={"name": "data_gen/interiorgs_v10/evaluation.py run (defaults)",
                 "policy_override": "VllmPolicy.generate with repetition_penalty and history mode",
                 "replaced_generate_of": getattr(original_generate, "__qualname__", None)},
        prompt={"impl": "the env's own prompt (prompt_format of rl_env_v10_S0.yaml)", "history_mode": args.history_mode},
        frames=None, decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 384, "stop": ["</action>"],
                               "repetition_penalty": REPETITION_PENALTY})
    view = str(M.view_dir(args.model))
    raise SystemExit(evaluation.main(["run", "--eval", str(ev), "--name", args.name, "--model", view]))


if __name__ == "__main__":
    main()
