"""Reward line, phase 2 direction 2: the guided smoke's check (docs/reward/reports/DESIGN_JSRL_20261102.md, section 8),
on the run's rollout dumps of the first updates, replayed without rendering in the run's own train_env.yaml:

* the guide record per episode (guide_eligible / guide_k / guide_actions / guide_turns / guide_stop) against the
  rule: eligible = d* >= 6 or equidistance, k = vagen.jsrl_guide.guide_count(d*, update) for eligible tasks;
* the guide's turns are in the prompt (the dump's input): as many assistant answers there as guide_turns, the same
  for the 8 episodes of a task, and they are the guide's plan (vagen.jsrl_guide.plan / chunk / action_text on the
  replayed env);
* the return: replaying the guide's answers then the policy's (the dump's output), the policy turns' rewards sum to
  the logged score (the guide's rewards are not in it), and after the guide the table distance is d* - k.

    python -m reward.jsrl_smoke_check --run RL_RUN_DIR --env-yaml TRAIN_ENV.yaml --rows ROWS --tables DIR
        --out FILE.json [--updates 1,2,3]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ASSISTANT = "<|im_start|>assistant\n"


def answers_in(text, prompt=False):
    """The assistant answers in a dumped input (prompt=True: skip the text before the first assistant tag) or
    output (the first answer is not tagged)."""
    parts = text.split(ASSISTANT)
    parts = parts[1:] if prompt else parts
    out = []
    for p in parts:
        a = p.split("<|im_end|>")[0].strip()
        if a and not a.startswith("[Observation]"):
            out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--env-yaml", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", default="1,2,3")
    args = ap.parse_args()
    import yaml

    from data_gen.interiorgs_v10 import distance_table as DT
    from vagen import jsrl_guide as J
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    cfg_guide = J.settings({"tables": args.tables})
    rows = {}
    for line in Path(args.rows).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["task_id"]] = r
    updates = [int(u) for u in args.updates.split(",")]
    work = Path(args.out).with_suffix(".rows.jsonl")
    checks = Counter()
    problems = []
    by_task = defaultdict(list)
    for u in updates:
        for line in (Path(args.run) / "rollout_data" / f"{u}.jsonl").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                by_task[(u, e["task_id"])].append(e)
    tids = sorted({t for _, t in by_task})
    work.write_text("".join(json.dumps(rows[t]) + "\n" for t in tids))
    index = {t: i for i, t in enumerate(tids)}
    env_cfg = dict(yaml.safe_load(Path(args.env_yaml).read_text())["envs"][0]["config"])
    env_cfg.update(jsonl_path=str(work), render_backend=None)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in env_cfg.items()
                                                     if k in ActiveSpatialEnvConfig.__dataclass_fields__}))

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render
    cache = DT.TableCache(args.tables)
    for (u, tid), eps in sorted(by_task.items()):
        row = rows[tid]
        d_star = int(row["difficulty"]["d_star"])
        elig = J.eligible(row, cfg_guide)
        k = J.guide_count(d_star, u, cfg_guide["stages"]) if elig else 0
        env.reset(seed=index[tid])
        plan, _, _ = J.plan(env, cache.get(tid), k) if k else ([], [], "")
        want = [J.action_text(t) for t in J.chunk(plan, cfg_guide["max_per_turn"])]
        prefixes = set()
        for e in eps:
            checks["episodes"] += 1
            rec = (int(e.get("guide_eligible", -1)), int(e.get("guide_k", -1)), int(e.get("guide_actions", -1)),
                   int(e.get("guide_turns", -1)))
            guide = answers_in(e["input"], prompt=True)
            prefixes.add(tuple(guide))
            ok_rec = rec == (int(elig), k, len(plan), len(want)) and not e.get("guide_stop")
            ok_prompt = guide == want
            checks["record_ok"] += ok_rec
            checks["prompt_is_plan"] += ok_prompt
            checks["guided"] += bool(guide)
            env.reset(seed=index[tid])
            for a in guide:
                env.step(a)
            d_after = env.current_region_metrics.get("optimal_distance") if env_cfg.get("progress_version") == "optimal" \
                else J._distance(cache.get(tid), J.pose_key(env))
            ok_d = d_after == d_star - len(plan)
            checks["takeover_d_ok"] += ok_d
            total = 0.0
            for a in answers_in(e["output"]):
                _, r, done, _ = env.step(a)
                total += float(r)
                if done:
                    break
            ok_ret = abs(total - float(e["score"])) < 1e-4
            checks["return_ok"] += ok_ret
            if not (ok_rec and ok_prompt and ok_d and ok_ret) and len(problems) < 10:
                problems.append({"update": u, "task_id": tid, "d_star": d_star, "eligible": elig, "k": k,
                                 "record": rec, "stop": e.get("guide_stop"), "guide": guide, "want": want,
                                 "d_after": d_after, "return": total, "score": e["score"]})
        checks["tasks"] += 1
        checks["same_prefix_tasks"] += len(prefixes) == 1
    out = {"updates": updates, "checks": dict(checks), "problems": problems}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
