"""Reward line, step 2b check: the episodes whose replay under the run's own reward does not give the logged return
(replay_q3's check), one process.  For each: logged, replayed, their difference, the turns, and the recorded
answers' count, so the cause can be read (e.g. a low-information-frame penalty the fake render cannot see, which the
anchored O return cancels).

    python -m reward.explain_return_mismatch --replays OUT/replays.jsonl --run RUN --env-yaml YAML --rows ROWS
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from reward.replay_q3 import answers_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--env-yaml", required=True)
    ap.add_argument("--rows", required=True)
    args = ap.parse_args()
    import yaml

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    bad = [json.loads(l) for l in Path(args.replays).read_text().splitlines()]
    bad = [g for g in bad if not all(g.get("check", [True]))]
    want = {(g["update"], g["task_id"]) for g in bad}
    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if any(r["task_id"] == t for _, t in want):
                rows[r["task_id"]] = r
    eps = defaultdict(list)
    for u, t in sorted(want):
        for line in Path(args.run, "rollout_data", f"{u}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("task_id") == t:
                eps[(u, t)].append(r)
    tmp = Path("/tmp/rw_mismatch_rows.jsonl")
    out = []
    for (u, t), records in sorted(eps.items()):
        tmp.write_text(json.dumps(rows[t]) + "\n")
        cfg = dict(yaml.safe_load(Path(args.env_yaml).read_text())["envs"][0]["config"])
        cfg.update(jsonl_path=str(tmp), render_backend=None)
        fields = set(ActiveSpatialEnvConfig.__dataclass_fields__)
        env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in fields}))

        def fake_render(env=env, **_kwargs):
            env.last_image_std = 30.0
            env.last_low_info_frame = False
            env.consecutive_low_info_frame_count = 0
            return {"obs_str": "", "multi_modal_data": {}}

        env._render = fake_render
        for e in records:
            env.reset(seed=0)
            total, n = 0.0, 0
            answers = answers_of(e["output"])
            for a in answers:
                _, r, done, _ = env.step(a)
                total += float(r)
                n += 1
                if done:
                    break
            diff = float(e["score"]) - total
            if abs(diff) > 1e-6:
                out.append({"update": u, "task_id": t, "logged": round(float(e["score"]), 4), "replayed": round(total, 4),
                            "logged_minus_replayed": round(diff, 4), "turns_replayed": n,
                            "turns_logged": e.get("n_turns"), "answers": len(answers),
                            "success_logged": e.get("traj_success")})
    print(json.dumps({"episodes": len(out), "differences": sorted({x["logged_minus_replayed"] for x in out}),
                      "items": out}, indent=1))


if __name__ == "__main__":
    main()
