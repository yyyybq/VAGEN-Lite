"""Reward line, phase 3, experiment B (docs/reward/PREREG_P3_B.md): the check run's and the smoke's checks, offline, from
a run's own logs.

* **the prefix check** (the user's 11-07 "自己答的回合，前缀算出的概率等于训练时记的"): RUN/dagger/check.jsonl, one line
  per update, the actor's differences between an own answer's log-probabilities from its prefix sample and the row's
  old_log_probs, for samples whose prefix + answer is the whole row and for the others; plus the trainer's exact
  checks (dagger/check_*_failed in the training log: the continued positions, the tokens, the prefix's image tokens);
* **the labels against the table, independently**: every chosen turn of RUN/dagger/turns.jsonl is replayed from its
  logged pose with the tables' own legality (difficulty.FastGeometry on the scene, the code the tables were built
  with; the run judged with the env's collision detector): its start distance, its judgment (own turns: every
  executed action takes d to d - 1; replaced turns: some executed action does not), and a replaced turn's label
  (every label action legal and taking d to d - 1);
* **the loss tokens**: RUN/dagger/first_batch_rank*.json, every sample's loss tokens decode to action names and
  separators through ``</action>`` and nothing else, a label sample's tokens to its label and the end of the turn;
* **the update's numbers** from the training log: the replaced share, the hit rate, the samples, the two terms'
  gradient norms and shares before clipping, the supervised loss, the seconds per update.

    python -m reward.dagger_check --run RUN_DIR --rows ROWS.jsonl --tables TABLES --gs-root GS_ROOT --out FILE.json

Writes FILE.json and FILE.md.  Read only; one process (set the thread variables to 1 on the login node).
"""

import argparse
import contextlib
import io
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ACTION_RE = re.compile(r"^((move_forward|move_backward|move_left|move_right|turn_left|turn_right)\|)+</action>$")
LOG_KEYS = ("dagger/replaced_share", "dagger/optimal_share", "dagger/action_hit_rate", "dagger/judged_share",
            "dagger/samples", "dagger/samples_own", "dagger/samples_label", "dagger/candidates",
            "dagger/build_failed", "dagger/build_seconds", "dagger/check_positions_same_failed",
            "dagger/check_tokens_same_failed", "dagger/check_image_tokens_same_failed", "dagger/retok_differs",
            "dagger/status_pose_mismatch", "dagger/status_step_count", "dagger/status_error",
            "dagger/status_unknown_start", "dagger/status_unknown_mid", "dagger/status_no_table",
            "dagger/status_no_label", "dagger/status_no_loss_tokens", "dagger/own_unexecuted",
            "dagger/dropped_by_check", "dagger/invalid_turns", "dagger/turns", "dagger/turns_judged",
            "actor/dagger_check_whole_row_mean_abs", "actor/dagger_check_prefix_mean_abs",
            "actor/dagger_check_shape_floor_mean_abs",
            "actor/dagger_grad_mag_share_und", "actor/dagger_grad_share_und", "actor/dagger_grad_proj_share_und",
            "actor/dagger_grad_norm_und", "actor/dagger_grad_norm_policy", "actor/dagger_grad_cos_policy_und",
            "actor/dagger_loss", "actor/dagger_seconds", "actor/dagger_inserted", "actor/dagger_audit",
            "actor/dagger_dropped", "actor/pg_loss", "actor/entropy", "actor/kl_loss", "actor/grad_norm",
            "timing_s/step", "timing_s/update_actor", "timing_s/dagger_build",
            "transition/invalid_action_rate/mean")


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def log_values(run: Path, keys: Sequence[str] = LOG_KEYS) -> Dict[str, Dict[int, float]]:
    """{key: {update: value}} from the run's training logs (the same reading as reward.drift_p3)."""
    from reward import drift_p3

    logs = drift_p3.run_dir_logs(run) if (run / "logs").is_dir() else []
    out: Dict[str, Dict[int, float]] = {k: {} for k in keys}
    pats = {k: re.compile(re.escape(k) + r":(?:np\.float64\()?([-+0-9.eE]+|nan|inf)") for k in keys}
    for p in sorted(logs, key=lambda q: q.name):
        for line in p.read_text(errors="replace").splitlines():
            m = drift_p3._STEP.search(line)
            if not m:
                continue
            step = int(m.group(1))
            for k, pat in pats.items():
                v = pat.search(m.group(2))
                if v and v.group(1) not in ("nan", "inf"):
                    out[k][step] = float(v.group(1))
    return out


GROUPS = ("whole_row", "prefix", "shape_floor")


def check_lines(run: Path) -> List[Dict[str, Any]]:
    """RUN/dagger/check.jsonl, one line per update (a resumed run's repeated update: the last line counts)."""
    by_step: Dict[Any, Dict[str, Any]] = {}
    for i, line in enumerate(_jsonl(run / "dagger" / "check.jsonl")):
        by_step[line.get("step", f"line{i}")] = line
    return list(by_step.values())


def pool(lines: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"updates": len(lines)}
    for g in GROUPS + ("loss_tokens",):
        part = [l[g] for l in lines if g in l]
        n = sum(x["tokens"] for x in part)
        mean = sum((x["mean_abs"] or 0.0) * x["tokens"] for x in part) / n if n else None
        mx = max((x["max_abs"] for x in part if x["tokens"]), default=None)
        rec = {"tokens": n, "mean_abs": mean, "max_abs": mx}
        if g != "loss_tokens":
            rec["over_1e-2"] = sum(x.get("over_1e-2", 0) for x in part)
            rec["over_1e-1"] = sum(x.get("over_1e-1", 0) for x in part)
        out[g] = rec
    return out


def reading(p: Dict[str, Any], exact_failures: Dict[str, int]) -> Optional[bool]:
    """The preregistered reading (PREREG_P3_B section 3), on the pooled numbers: (1) the exact parts all equal;
    (2) the whole-row group (the same tokens as the trainer's own recompute) within 0.001 on average, i.e. the
    comparison itself is sound; (3) the prefix group's mean difference at most twice the floor's (the same sample
    beside another one in a batch) plus 0.001, and (4) at most 1% of its tokens off by more than 0.1.  None when a
    group the reading needs has no token."""
    w, q, f = p["whole_row"], p["prefix"], p["shape_floor"]
    if not (w["tokens"] and q["tokens"] and f["tokens"]):
        return None
    return (not any(exact_failures.values()) and w["mean_abs"] <= 1e-3
            and q["mean_abs"] <= 2 * f["mean_abs"] + 1e-3 and q["over_1e-1"] <= 0.01 * q["tokens"])


def prefix_check(run: Path, logs: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
    """The user's check: the actor's differences pooled over the run's updates, and the trainer's exact checks."""
    out = pool(check_lines(run))
    out["exact_failures"] = {k: int(sum(logs.get(f"dagger/check_{k}_failed", {}).values()))
                             for k in ("positions_same", "tokens_same", "image_tokens_same")}
    out["retok_differs"] = int(sum(logs.get("dagger/retok_differs", {}).values()))
    out["passed"] = reading(out, out["exact_failures"])
    return out


def _distance(tab: Dict[str, Any], P: np.ndarray, R: np.ndarray) -> Optional[int]:
    from vagen.dagger import distance

    return distance(tab, P, R)[0]


def table_check(run: Path, rows_path: Path, tables: Path, gs_root: str, limit: int = 0) -> Dict[str, Any]:
    """Every chosen turn replayed from its logged pose with FastGeometry's legality and the task's table."""
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import ACTION_INDEX, IS_TRANSLATION, FastGeometry, _split, apply_action_batch
    from data_gen.interiorgs_v10.scene import load_scene

    last: Dict[Any, Dict[str, Any]] = {}  # a resumed run logs a repeated update again: the last record counts
    for t in _jsonl(run / "dagger" / "turns.jsonl"):
        last[(t.get("step"), t.get("row"), t.get("t"))] = t
    turns = [t for t in last.values() if t.get("status") == "ok"]
    if limit > 0:
        turns = turns[:limit]
    tables_cache: Dict[str, Dict[str, Any]] = {}
    rows = {}
    with open(rows_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[str(r["task_id"])] = r
    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing = Counter()
    for t in turns:
        r = rows.get(str(t.get("task_id")))
        if r is None:
            missing["row"] += 1
            continue
        by_scene[str(r["scene_id"])].append(t)
    c: Counter = Counter()
    bad: List[Dict[str, Any]] = []

    def step(geom, P, R, a):
        p, r = apply_action_batch(P, R, a)
        r = np.array(np.broadcast_to(r, (1, 3, 3)))
        if IS_TRANSLATION[ACTION_INDEX[a]] and not bool(np.asarray(geom.transitions_legal(P, p, r))[0]):
            return P, R, True
        return p, r, False

    for scene_id, ts in sorted(by_scene.items()):
        with contextlib.redirect_stdout(io.StringIO()):
            scene = load_scene(Path(gs_root) / scene_id, scene_id, wall_model="thick")
        geom = FastGeometry.for_scene(scene)
        for t in ts:
            tid = str(t["task_id"])
            if tid not in tables_cache:
                tables_cache.clear()  # one scene's tasks at a time
                tables_cache[tid] = DT.load(DT.table_path(tables, tid))
            tab = tables_cache[tid]
            P, R = _split(np.asarray(t["pose"], float).reshape(4, 4))
            d0 = _distance(tab, P, R)
            c["turns"] += 1
            if d0 != t.get("d_before"):
                c["start_distance_differs"] += 1
                bad.append({"why": "start distance", "turn": t, "d0": d0})
                continue
            hits, d, Q, S = [], d0, P, R
            for a in t.get("executed") or []:
                Q, S, blocked = step(geom, Q, S, a)
                dn = d if blocked else _distance(tab, Q, S)
                hits.append((not blocked) and dn is not None and dn == d - 1)
                d = dn if dn is not None else d
            # the answer's form (canonical) is the agent loop's, from its text; the rest is recomputed here
            optimal = (bool(t.get("executed")) and all(hits) and not t.get("done_action") and bool(t.get("canonical"))
                       and t.get("written") is not None and list(t["written"]) == list(t["executed"]))
            key = "own" if t.get("optimal") else "replaced"
            c[f"{key}_turns"] += 1
            if optimal != bool(t.get("optimal")):
                c[f"{key}_judgment_differs"] += 1
                bad.append({"why": "judgment", "turn": t, "hits_here": hits})
            if hits != list(t.get("hits") or []):
                c["hits_differ"] += 1
            if not t.get("optimal"):
                # every label action legal and bringing the table distance down (the guide takes the smallest
                # distance below the current one, so a step that drops it by more than one, cells being merged at
                # 1 cm / 1 deg, is still the guide's; counted apart)
                ok, jump, d, Q, S = True, False, d0, P, R
                for a in t.get("label_actions") or []:
                    Q, S, blocked = step(geom, Q, S, a)
                    dn = None if blocked else _distance(tab, Q, S)
                    if dn is None or dn >= d:
                        ok = False
                        break
                    jump = jump or dn < d - 1
                    d = dn
                c["labels"] += 1
                c["label_optimal" if ok else "label_not_optimal"] += 1
                c["label_jump"] += int(ok and jump)
                if not ok:
                    bad.append({"why": "label", "turn": t})
    return {"counts": dict(c), "missing": dict(missing), "examples": bad[:20]}


def mask_check(run: Path) -> Dict[str, Any]:
    c: Counter = Counter()
    bad = []
    for p in sorted((run / "dagger").glob("first_batch_rank*.json")):
        for s in json.loads(p.read_text())["samples"]:
            c["samples"] += 1
            c[s["kind"]] += 1
            ok = bool(ACTION_RE.match(s.get("loss_decoded") or ""))
            if s["kind"] == "label":
                ok = ok and s.get("label_decoded") == (s.get("label_text") or "") + "<|im_end|>"
            c["ok" if ok else "bad"] += 1
            if not ok:
                bad.append({k: s.get(k) for k in ("kind", "label_text", "label_decoded", "loss_decoded", "uid", "t")})
    return {"counts": dict(c), "examples": bad[:20]}


def numbers(logs: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
    out = {}
    for k, v in logs.items():
        if v:
            vals = [v[s] for s in sorted(v)]
            out[k] = {"updates": len(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals),
                      "per_update": {str(s): v[s] for s in sorted(v)}}
    return out


def md(rep: Dict[str, Any]) -> str:
    pc, tc, mc, nb = rep["prefix_check"], rep["table_check"], rep["mask_check"], rep["numbers"]
    f = lambda x: "—" if x is None else (f"{x:.2e}" if isinstance(x, float) and abs(x) < 1e-2 else  # noqa: E731
                                         f"{x:.4f}" if isinstance(x, float) else str(x))
    lines = [f"# 实验 B 核对：{rep['run']}", "", "## 前缀概率核对", "",
             "| 组 | token | 平均 \\|差\\| | 最大 \\|差\\| | > 0.01 | > 0.1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for g in GROUPS:
        r = pc[g]
        lines.append(f"| {g} | {r['tokens']} | {f(r['mean_abs'])} | {f(r['max_abs'])} | {r['over_1e-2']} | {r['over_1e-1']} |")
    lt = pc["loss_tokens"]
    lines += [f"| 只算损失 token（自己答的） | {lt['tokens']} | {f(lt['mean_abs'])} | {f(lt['max_abs'])} | | |", "",
              f"- 精确部分（训练器逐个样本）：{pc['exact_failures']}；重新分词不同 {pc['retok_differs']}",
              f"- 按预登记的读法：{'过' if pc['passed'] else '没过' if pc['passed'] is False else '样本不够，判不了'}", "",
              "## 标签对表（FastGeometry 独立重放）", "", f"- {tc['counts']}；缺行 {tc['missing']}", "",
              "## 损失 token（第一批解码）", "", f"- {mc['counts']}", "", "## 每次更新的数（中位数，最小–最大）", ""]
    for k in ("dagger/replaced_share", "dagger/action_hit_rate", "dagger/samples", "dagger/samples_own",
              "actor/dagger_grad_mag_share_und", "actor/dagger_grad_norm_und", "actor/dagger_grad_norm_policy",
              "actor/dagger_grad_cos_policy_und", "actor/dagger_loss", "actor/dagger_seconds", "timing_s/step",
              "timing_s/update_actor", "transition/invalid_action_rate/mean", "actor/entropy"):
        if k in nb:
            lines.append(f"- {k}：{f(nb[k]['median'])}（{f(nb[k]['min'])}–{f(nb[k]['max'])}，{nb[k]['updates']} 次）")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True,
                    help="an RL run's folder (logs/, dagger/); several: each, then the prefix check pooled over all")
    ap.add_argument("--rows", required=True, help="the runs' task rows (task_id, scene_id)")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--gs-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="at most this many turns in the table check (0: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite")
    reps, all_lines, exact = [], [], Counter()
    for r in a.run:
        run = Path(r)
        logs = log_values(run)
        rep = {"run": str(run), "prefix_check": prefix_check(run, logs),
               "table_check": table_check(run, Path(a.rows), Path(a.tables), a.gs_root, a.limit),
               "mask_check": mask_check(run), "numbers": numbers(logs)}
        reps.append(rep)
        all_lines += check_lines(run)
        exact.update(rep["prefix_check"]["exact_failures"])
    pooled = pool(all_lines)
    pooled["exact_failures"] = dict(exact)
    pooled["passed"] = reading(pooled, dict(exact))
    full = {"runs": reps, "pooled_prefix_check": pooled}
    out.write_text(json.dumps(full, indent=1) + "\n")
    f = lambda x: "—" if x is None else f"{x:.2e}"  # noqa: E731
    text = "\n".join(md(rep) for rep in reps) + (
        f"\n# 合并 {len(reps)} 个运行的前缀概率核对\n\n" + "".join(
            f"- {g}：{pooled[g]['tokens']} 个 token，平均 |差| {f(pooled[g]['mean_abs'])}，最大 {f(pooled[g]['max_abs'])}，"
            f"> 0.1 的 {pooled[g]['over_1e-1']} 个\n" for g in GROUPS)
        + f"- 精确部分：{dict(exact)}\n- 按预登记的读法：{pooled['passed']}\n")
    out.with_suffix(".md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
