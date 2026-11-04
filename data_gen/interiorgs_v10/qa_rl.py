"""Qwen3 main line, the QA RL pilot (the user's instruction of 2026-11-03, section 2): P0's in-domain questions as
single-turn RL tasks, the reward scored by the P0 line's own rules (data_gen.interiorgs_qa.scoring: each benchmark's
official extraction; multiple choice 1 / 0, VSI-Bench's numeric answers by MRA in [0, 1]); an answer the benchmark's
extraction cannot read scores 0 and counts as unreadable (the env's action_is_valid, the chain's 10% line).

    python -m data_gen.interiorgs_v10.qa_rl rows --lf LF.jsonl --out ROWS.jsonl [--lf-source DIR] [--selection SEL]
    python -m data_gen.interiorgs_v10.qa_rl check --rows ROWS.jsonl --out CHECK.json
    python -m data_gen.interiorgs_v10.qa_rl rl-config --rows ROWS.jsonl --val-rows VAL.jsonl --model HF_DIR \
        --name NAME --run-root RUN --out RL [--seed S] [--candidates 64] [--target-groups 32] [--response-length 256]

rl-config writes what training_prep rl-config writes for the action env (train_env.yaml / val_env.yaml with the P0QA
env, train.jsonl / val.jsonl, ppo.yaml, rl_config_report.json), with the standard RL recipe of the Qwen3 line
(standard v10_p1_S0: the concat agent loop, package p1 -- GRPO, 8 answers a question, only mixed groups trained, lr
2e-6, KL loss 0.05, entropy bonus 0.01 --, a checkpoint every 50 updates cut to its HF weights, no validation inside
training) and the pilot's sizes: 64 candidate questions an update (the package's 24), the first 32 mixed ones trained
(its 12: the same half), one turn, at most 256 answer tokens.  verl splits an update's prompts (candidates x answers)
evenly over the agent-loop workers, and the template's 48 do not divide the pilot's 512 (the smoke of 2026-11-04 failed
on it), so the agent loop gets the most workers up to the template's that divide them (32); parallelism only.

rows: one row per LLaMA-Factory record of a P0 SFT file (the records the SFT trains on: the prompt, its images and the
gold answer exactly as the SFT shows them), matched to its question in P0's selection (by the export's audit file, row
by row) for the scoring: {"qid", "benchmark", "group", "family", "answer_type", "scene_id", "prompt", "images",
"answer", "question"}.  Records with a video, or with more than one question, are refused (the pilot leaves video out).
check: every row's gold answer must score 1 and be readable; another option letter must score 0 (multiple choice);
an empty answer must be unreadable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data_gen.interiorgs_qa import scoring as S

P0 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/interiorgs_qa_final_20261028")


def _key(rec: Dict[str, Any]) -> str:
    return json.dumps([rec["messages"], rec.get("images") or [], rec.get("videos") or []], ensure_ascii=False,
                      sort_keys=True)


def readable(question: Dict[str, Any], reply: str) -> bool:
    """Whether the benchmark's own extraction finds an answer in the reply (scoring.score's rules)."""
    bench, reply = question["benchmark"], str(reply or "")
    if question["answer_type"] == "numeric":
        try:
            float(S.vsi_fuzzy(reply))
            return True
        except (TypeError, ValueError):
            return False
    if bench == "VSI-Bench":
        return S.vsi_fuzzy(reply).upper() in tuple(S.LETTERS)
    if bench == "MMSI-Bench":
        return S.mmsi_extract(reply) is not None
    if bench == "MindCube":
        return S.mindcube_extract(reply) is not None
    if bench == "ViewSpatial-Bench":
        return S.viewspatial_extract(reply) is not None
    if bench == "BLINK":
        return S.blink_extract(reply) is not None
    return re.search(r"\b([A-H])\b", reply) is not None


def reward(row: Dict[str, Any], reply: str) -> Tuple[float, bool]:
    """(score in [0, 1], readable) of one answer to one row; an unreadable answer scores 0."""
    q = row["question"]
    if not readable(q, reply):
        return 0.0, False
    got = S.score(q, str(reply), row["family"])
    return float(got.get("accuracy", got.get("MRA", 0.0))), True


def build_rows(lf: Path, lf_source: Path, selection: Path) -> List[Dict[str, Any]]:
    src = [json.loads(l) for l in (lf_source / "qa_train.jsonl").read_text().splitlines() if l.strip()]
    audit = [json.loads(l) for l in (lf_source / "audit_train.jsonl").read_text().splitlines() if l.strip()]
    assert len(src) == len(audit), (len(src), len(audit))
    by_key = {_key(r): a for r, a in zip(src, audit)}
    questions = {}
    for l in selection.read_text().splitlines():
        if l.strip():
            q = json.loads(l)["question"]
            questions[q["qid"]] = q
    rows = []
    for l in lf.read_text().splitlines():
        if not l.strip():
            continue
        rec = json.loads(l)
        if rec.get("videos"):
            raise SystemExit("a record with a video: the pilot leaves video questions out")
        users = [m for m in rec["messages"] if m["role"] == "user"]
        answers = [m for m in rec["messages"] if m["role"] == "assistant"]
        if len(users) != 1 or len(answers) != 1:
            raise SystemExit("a record with more than one question")
        a = by_key.get(_key(rec))
        if a is None:
            raise SystemExit(f"no audit entry for a record of {lf} (not from {lf_source}?)")
        qid, family = a["qids"][0], a["families"][0]
        q = questions[qid]
        if family not in q["variants"]:
            raise SystemExit(f"{qid}: family {family} not among its variants {sorted(q['variants'])}")
        rows.append({"qid": qid, "benchmark": q["benchmark"], "group": q["group"], "family": family,
                     "answer_type": q["answer_type"], "scene_id": q["scene_id"], "prompt": users[0]["content"],
                     "images": list(rec.get("images") or []), "answer": answers[0]["content"], "question": q})
    return rows


def check(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    bad: Counter = Counter()
    examples: Dict[str, List[Any]] = {}
    n = 0
    by = Counter()
    for r in rows:
        n += 1
        by[(r["benchmark"], r["family"], r["answer_type"])] += 1
        problems = []
        value, ok = reward(r, r["answer"])
        if not ok or value < 1.0:
            problems.append("gold")
        if r["answer_type"] != "numeric":
            letters = S.LETTERS[:max(2, len(r["question"]["variants"][r["family"]].get("options") or []))]
            gold = r["question"]["variants"][r["family"]]["answer"]
            other = next(c for c in letters if c != gold)
            if reward(r, r["answer"].replace(gold, other, 1))[0] != 0.0:
                problems.append("other_letter")
        if reward(r, "")[1]:
            problems.append("empty_readable")
        for p in problems:
            bad[p] += 1
            examples.setdefault(p, []).append({"qid": r["qid"], "benchmark": r["benchmark"], "family": r["family"],
                                               "answer": r["answer"][:200]})
    return {"rows": n, "problems": dict(bad), "examples": {k: v[:10] for k, v in examples.items()},
            "by_benchmark_family_type": {" / ".join(k): v for k, v in sorted(by.items())}}


ENV_NAME = "P0QA"
PPO_TEMPLATE = Path(__file__).with_name("configs") / "ppo_template_v10.yaml"


def env_yaml(rows_path: Path, n: int, response_length: int) -> Dict[str, Any]:
    return {"envs": [{"name": ENV_NAME, "data_source": "p0_qa", "max_turns": 1, "response_length_per_turn": response_length,
                      "n_envs": n, "seed": [0, n], "seed_list": list(range(n)),
                      "config": {"jsonl_path": str(rows_path.resolve()), "image_placeholder": "<image>"}}]}


def rl_config(rows: Path, val_rows: Path, model: str, name: str, run_root: Path, out: Path, seed: Optional[int] = None,
              candidates: int = 64, target_groups: int = 32, response_length: int = 256,
              save_freq: int = 50) -> Dict[str, Any]:
    import shutil

    import yaml

    from . import training_prep as TP

    out.mkdir(parents=True, exist_ok=True)
    env_paths = {}
    counts = {}
    for split, src in (("train", rows), ("val", val_rows)):
        n = sum(1 for l in src.read_text().splitlines() if l.strip())
        if not n:
            raise SystemExit(f"rl-config: no rows in {src}")
        shutil.copyfile(src, out / f"{split}.jsonl")
        env_paths[split] = out / f"{split}_env.yaml"
        env_paths[split].write_text(yaml.safe_dump(env_yaml(out / f"{split}.jsonl", n, response_length), sort_keys=False))
        counts[split] = n
    ppo = yaml.safe_load(PPO_TEMPLATE.read_text())
    model = str(Path(model))  # no trailing "/": verl's copy_to_local refuses one
    run_root = run_root.resolve()
    for key, value in {
        "data.train_files": str(env_paths["train"].resolve()), "data.val_files": str(env_paths["val"].resolve()),
        "actor_rollout_ref.model.path": model, "critic.model.path": model, "critic.model.tokenizer_path": model,
        "trainer.experiment_name": name, "trainer.project_name": "vagen_interiorgs_v10_qa",
        "trainer.default_local_dir": str(run_root / "checkpoints"),
        "trainer.rollout_data_dir": str(run_root / "rollout_data"),
        "trainer.validation_data_dir": str(run_root / "validation"),
    }.items():
        TP._set(ppo, key, value)
    for key, value in TP.FORMAL_PPO.items():
        TP._set(ppo, key, value)
    for key, value in TP.history_overrides("concat", 1)["ppo"].items():  # standard v10_p1_S0: the concat agent loop
        TP._set(ppo, key, value)
    TP._set(ppo, "data.max_response_length", int(response_length))  # one answer, not twelve turns
    for key, value in {"trainer.save_freq": save_freq, "trainer.ckpt_keep_every": save_freq,
                       "trainer.ckpt_prune_to_hf": True, "trainer.val_before_train": False,
                       "trainer.test_freq": -1}.items():
        TP._set(ppo, key, value)
    if seed is not None:
        TP._set(ppo, "data.seed", int(seed))
    package = TP.package_overrides("p1", candidates, target_groups)
    for key, value in package.items():
        TP._set(ppo, key, value)
    # verl's agent loop chunks an update's prompts evenly over its workers (DataProto.chunk): the template's 48 suit the
    # action runs' 24 or 48 candidates x 8 answers, not the pilot's 64 x 8 = 512
    prompts = int(package["data.train_batch_size"]) * int(package["actor_rollout_ref.rollout.n"])
    most = int(ppo["actor_rollout_ref"]["rollout"]["agent"]["num_workers"])
    workers = max(w for w in range(1, most + 1) if prompts % w == 0)
    TP._set(ppo, "actor_rollout_ref.rollout.agent.num_workers", workers)
    (out / "ppo.yaml").write_text(yaml.safe_dump(ppo, sort_keys=False))
    report = {"env": ENV_NAME, "train": counts["train"], "val": counts["val"], "model": model, "seed": seed,
              "standard_like": "v10_p1_S0", "opt_package": "p1", "opt_package_overrides": package,
              "candidates": candidates, "target_groups": target_groups, "response_length": response_length,
              "agent_workers": workers, "save_freq": save_freq, "ppo_template": str(PPO_TEMPLATE), "rows": str(rows),
              "val_rows": str(val_rows)}
    (out / "rl_config_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--lf", required=True, help="a LLaMA-Factory jsonl of P0 records (e.g. the SFT control's)")
    r.add_argument("--lf-source", default=str(P0 / "lf_256"), help="the P0 export it was taken from (qa_train, audit)")
    r.add_argument("--selection", default=str(P0 / "selection/train.jsonl"))
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--rows", required=True)
    c.add_argument("--out", required=True)
    g = sub.add_parser("rl-config")
    g.add_argument("--rows", required=True, help="the training questions (qa_rl rows)")
    g.add_argument("--val-rows", required=True, help="a few questions for verl's validation files (not run)")
    g.add_argument("--model", required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--run-root", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--candidates", type=int, default=64)
    g.add_argument("--target-groups", type=int, default=32)
    g.add_argument("--response-length", type=int, default=256)
    g.add_argument("--save-freq", type=int, default=50)
    a = ap.parse_args()
    if a.cmd == "rl-config":
        rep = rl_config(Path(a.rows), Path(a.val_rows), a.model, a.name, Path(a.run_root), Path(a.out), a.seed,
                        a.candidates, a.target_groups, a.response_length, a.save_freq)
        print(json.dumps(rep, indent=1))
        return 0
    if a.cmd == "rows":
        rows = build_rows(Path(a.lf), Path(a.lf_source), Path(a.selection))
        Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(len(rows), "rows", dict(Counter(x["benchmark"] for x in rows)))
        return 0
    rows = [json.loads(l) for l in Path(a.rows).read_text().splitlines() if l.strip()]
    res = check(rows)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: res[k] for k in ("rows", "problems")}, ensure_ascii=False))
    return 0 if not res["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
