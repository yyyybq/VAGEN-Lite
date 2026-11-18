"""q38 (the Qwen3.8-27B rerun line): extract a compact golden sample of 8B RL runs for the sampler tests (T1, T2).

For each run of an 8B experiment directory (read only), every update's 192 candidate episodes as the 8B trainer saw
them, and the trainer's own console metrics of that update:

- the episodes come from ``rl_run_<run>/rollout_data/<update>.jsonl`` (every candidate, rows in the order the 8B
  trainer reduced its metrics in: after _balance_batch, RT:2800-2811, RT:3113-3114);
- the draw order of the update's 24 tasks (the 8B ``uid_draw_order``, RT:2731-2732, which the group filter takes its
  nominal batch in) is not in the dump; it is the train dataloader's: verl 0.6.1 ``create_rl_sampler`` builds a
  RandomSampler on a torch.Generator seeded with data.seed (F061 trainer/main_ppo.py:425-430), the dataloader takes
  data.train_batch_size consecutive indices with drop_last (RT:1468-1475), and dataset index i is the env seed i
  (gym_agent_dataset.py seed_list) which is row i of the task file (envs/active_spatial/env.py reset).  The script
  checks for every update that the 24 tasks of the permutation slice are exactly the tasks of the dump, each with
  rollout.n episodes, and refuses the run otherwise;
- the metrics are the last ``step:<update> - ...`` console line of the update in ``rl_run_<run>/logs/rl-*.log``,
  with the names under filter/, episode/, transition/ and train/traj_success/.

Output: ``<out>/<run>.jsonl.gz``, one JSON object per line.  The first is the header (kind "header": the run, the
source paths, seed, rollout.n, train_batch_size, adv_estimator, the filter section with its ``rows`` path, the
number of updates).  Then one line per update (kind "step"):

- ``tasks``: [[task_id, task_type, d_star], ...] in draw order (d_star = difficulty.d_star of the task file, null
  without one; keep_by_return_std reads it for filter/keep/filled_min_steps_mean);
- per episode, in dump order: ``g`` (the draw position of its task), ``score`` (the return), ``traj_success`` (0/1),
  ``final_score``, ``score_improvement``, ``invalid_action``, ``kept`` (kept_for_update, 0/1).  The float columns are
  float32 values (the 8B trainer cast them to float32 before using them, RT:2943, RT:2964; score is a float32 sum),
  written as their shortest float32 decimal: np.float32(value) gives the value back exactly;
- ``log``: {metric: value} of the console line.

Resources: one process, one thread (set the thread variables), reads each dump file once; run it under nice.
Usage:
    python scripts/q38/extract_golden.py --exp <8B experiment dir> --run qw3v4s3 --run qw3s0s3 \
        --out vagen/q38/tests/golden
"""

from __future__ import annotations

import argparse
import datetime
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

LOG_PREFIXES = ("filter/", "episode/", "transition/", "train/traj_success/")
_STEP_LINE = re.compile(r"(?:^|\s)step:(\d+) - (.*)$")
_NP_WRAP = re.compile(r"^np\.(?:float|int)\d*\((.*)\)$")


def f32(x) -> float:
    """The float32 value of ``x`` as the shortest decimal that gives it back (checked)."""
    v = np.float32(x)
    out = float(str(v))
    if np.float32(out) != v and not (np.isnan(v) and np.isnan(np.float32(out))):
        raise ValueError(f"float32 round trip failed for {x!r}")
    return out


def parse_value(text: str):
    text = text.strip()
    m = _NP_WRAP.match(text)
    if m:
        text = m.group(1)
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return float(text)


def parse_log(paths: list[str]) -> dict[int, dict]:
    """{update: {metric: value}} from the console lines (the last line of an update wins: a resumed run logs it
    again, and its dump file is the later one too)."""
    out: dict[int, dict] = {}
    for path in sorted(paths):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _STEP_LINE.search(line.rstrip("\n"))
                if not m:
                    continue
                values = {}
                for part in m.group(2).split(" - "):
                    key, sep, val = part.partition(":")
                    if not sep or not key.startswith(LOG_PREFIXES):
                        continue
                    try:
                        values[key] = parse_value(val)
                    except ValueError:
                        continue
                out[int(m.group(1))] = values
    return out


def load_config(path: str) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)

    def get(key, default=None):
        return OmegaConf.select(cfg, key, default=default)

    sampler = get("data.sampler") or {}
    filt = get("filter")
    return {
        "seed": get("data.seed"),
        "shuffle": bool(get("data.shuffle")),
        "train_batch_size": int(get("data.train_batch_size")),
        "gen_batch_size": get("data.gen_batch_size"),
        "sampler_class_path": sampler.get("class_path") if sampler else None,
        "rollout_n": int(get("actor_rollout_ref.rollout.n")),
        "adv_estimator": str(get("algorithm.adv_estimator")),
        "train_files": get("data.train_files"),
        "filter": OmegaConf.to_container(filt, resolve=False) if filt is not None else None,
    }


def load_tasks(env_yaml: str) -> tuple[list[str], dict[str, int | None], str]:
    """The task file's task ids by row and d_star by task id, after checking that dataset index i is row i."""
    from omegaconf import OmegaConf

    envs = OmegaConf.load(env_yaml)
    specs = envs.get("envs")
    if specs is None or len(specs) != 1:
        raise SystemExit(f"{env_yaml}: expected one env spec")
    spec = specs[0]
    rows = spec.config.jsonl_path
    n_envs = int(spec.n_envs)
    seed_list = list(spec.seed_list) if spec.get("seed_list") is not None else None
    if seed_list is None or [int(s) for s in seed_list[:n_envs]] != list(range(n_envs)):
        raise SystemExit(f"{env_yaml}: the dataset index is not the task row (seed_list is not 0..n_envs-1)")
    task_ids, d_star = [], {}
    with open(rows, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            tid = str(r["task_id"])
            task_ids.append(tid)
            d_star[tid] = (r.get("difficulty") or {}).get("d_star")
    if len(task_ids) < n_envs:
        raise SystemExit(f"{rows}: {len(task_ids)} rows for n_envs {n_envs}")
    return task_ids[:n_envs], d_star, rows


def draw_batches(n: int, seed: int, batch_size: int, steps: int) -> list[list[int]]:
    """The dataset indices of updates 1..steps: RandomSampler(generator seeded with ``seed``) with drop_last."""
    import torch

    torch.set_num_threads(1)
    g = torch.Generator()
    g.manual_seed(int(seed))
    per_epoch = n // batch_size
    out: list[list[int]] = []
    while len(out) < steps:
        perm = torch.randperm(n, generator=g).tolist()
        out.extend(perm[b * batch_size:(b + 1) * batch_size] for b in range(per_epoch))
    return out[:steps]


def extract_run(exp: str, run: str) -> tuple[dict, list[dict]]:
    cfg_dir = os.path.join(exp, f"rl_{run}")
    run_dir = os.path.join(exp, f"rl_run_{run}")
    cfg = load_config(os.path.join(cfg_dir, "ppo.yaml"))
    if not cfg["shuffle"] or cfg["seed"] is None or cfg["sampler_class_path"]:
        raise SystemExit(f"{run}: not the seeded RandomSampler (shuffle {cfg['shuffle']}, seed {cfg['seed']})")
    bs = cfg["train_batch_size"]
    if cfg["gen_batch_size"] not in (None, bs):
        raise SystemExit(f"{run}: gen_batch_size {cfg['gen_batch_size']} != train_batch_size {bs}")
    task_ids, d_star, rows_path = load_tasks(cfg["train_files"])
    dump_dir = os.path.join(run_dir, "rollout_data")
    steps = sorted(int(os.path.basename(p)[:-6]) for p in glob.glob(os.path.join(dump_dir, "*.jsonl"))
                   if os.path.basename(p)[:-6].isdigit())
    if steps != list(range(1, len(steps) + 1)):
        raise SystemExit(f"{run}: rollout dumps are not updates 1..N")
    batches = draw_batches(len(task_ids), cfg["seed"], bs, len(steps))
    logs = parse_log(glob.glob(os.path.join(run_dir, "logs", "rl-*.log")))
    n = cfg["rollout_n"]
    out = []
    for s in steps:
        with open(os.path.join(dump_dir, f"{s}.jsonl"), encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        drawn = [task_ids[i] for i in batches[s - 1]]
        pos = {t: j for j, t in enumerate(drawn)}
        if len(pos) != bs:
            raise SystemExit(f"{run} update {s}: a task drawn twice in one update")
        if len(rows) != bs * n:
            raise SystemExit(f"{run} update {s}: {len(rows)} episodes, expected {bs * n}")
        counts: dict[str, int] = {}
        for r in rows:
            counts[str(r["task_id"])] = counts.get(str(r["task_id"]), 0) + 1
            if int(r["step"]) != s:
                raise SystemExit(f"{run} update {s}: a row of update {r['step']}")
        if set(counts) != set(pos) or any(c != n for c in counts.values()):
            raise SystemExit(f"{run} update {s}: the dump's tasks are not the dataloader's draw")
        family = {}
        kept_of = {}
        for r in rows:
            t = str(r["task_id"])
            family.setdefault(t, str(r["task_type"]))
            k = bool(r["kept_for_update"])
            if kept_of.setdefault(t, k) != k:
                raise SystemExit(f"{run} update {s}: task {t} only partly kept")
        if s not in logs:
            raise SystemExit(f"{run}: no console line for update {s}")
        out.append({
            "kind": "step", "step": s,
            "tasks": [[t, family[t], d_star.get(t)] for t in drawn],
            "g": [pos[str(r["task_id"])] for r in rows],
            "score": [f32(r["score"]) for r in rows],
            "traj_success": [int(float(r["traj_success"]) > 0.5) for r in rows],
            "final_score": [f32(r["final_score"]) for r in rows],
            "score_improvement": [f32(r["score_improvement"]) for r in rows],
            "invalid_action": [f32(r["invalid_action"]) for r in rows],
            "kept": [int(bool(r["kept_for_update"])) for r in rows],
            "log": logs[s],
        })
    header = {
        "kind": "header", "run": run, "exp": exp, "config": os.path.join(cfg_dir, "ppo.yaml"),
        "dumps": dump_dir, "rows": rows_path, "seed": cfg["seed"], "rollout_n": n, "train_batch_size": bs,
        "adv_estimator": cfg["adv_estimator"], "filter": cfg["filter"], "updates": len(steps),
        "extracted_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "scripts/q38/extract_golden.py",
    }
    return header, out


def write_gz(path: str, lines: list[dict]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as fh:
        for obj in lines:
            fh.write((json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", required=True, help="the 8B experiment directory (read only)")
    ap.add_argument("--run", action="append", required=True, help="run name, e.g. qw3v4s3 (repeatable)")
    ap.add_argument("--out", required=True, help="output directory (vagen/q38/tests/golden)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for run in args.run:
        header, steps = extract_run(args.exp, run)
        path = os.path.join(args.out, f"{run}.jsonl.gz")
        write_gz(path, [header] + steps)
        print(f"{run}: {len(steps)} updates -> {path} ({os.path.getsize(path)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
