"""E3VS scoring of one eval run and the pilot gate (PREREG_E3VS_PILOT sections 6, 11, 12).

score(run_dir) reads RUN/episodes.jsonl + RUN/manifest.json and the split's episodes (answerable viewpoints):
  - primary success: the final pose (at STOP, or after the last of the 25 decisions) within 0.25 m AND <= 20 deg
    of any answerable viewpoint (geometry.success); k / n over the expected episodes, Wilson 95%, and a
    scene-cluster bootstrap (10,000 resamples of the scenes with replacement, seed 0, percentile 2.5 / 97.5);
  - secondary: STOP-required success, stop rate, collision rate (skipped moves / all decisions), mean decisions,
    SPL against the BFS optimum under the same collision model (S * l / max(l, p), l = opt_len non-stop actions,
    p = the episode's non-stop decisions incl. skipped moves; episodes without a BFS optimum are left out and
    counted), distance and view angle to the nearest answerable viewpoint at the final pose, per question type;
  - parse: lenient-valid (the official parser did not fall back) and regex-valid (strict fullmatch of
    make_sft.REGEX) over all decisions, empty replies, length-truncated replies;
  - validity: every expected episode logged with status ok (else results_invalid.json);
  - the gate floor (floors JSON gate_floor: rate, seq, k / n), as a reference.
Writes RUN/results.json (or results_invalid.json) and RUN/per_episode.jsonl.

gate(root) for the pilot (OUT/e3vs/runs/pilot/<START>/evals/pilot__<START>__<tier>__<mode>/results.json):
  E0 mechanism: parity (every OUT/e3vs/parity/parity_*.json min PSNR >= 40), occupancy validation pass, all 12
     required evals valid with all 377 test episodes, every one with an audit.json whose image tokens are
     exactly [258] (a missing / failed / empty audit is an E0 problem, B4), every one run with the frozen
     collider on the validated grids (manifest collides == vpd.e3vs.occupancy:load_collider and each scene's
     occ_sha256 == validation.json's), the floor computed under the frozen collider (floors_occ.json collides)
     on the validated grids (OUT/e3vs/prep_state/floors == validation.json grids_sha256, stamped by the prep
     job), and the 0W adapters' code_sha256 equal across starts.  Provenance (provenance()): every 0W / 100
     eval's adapter_sha256 == that start's adapters/<tier>/adapter.pt (raw: no adapter), the 100 adapter
     initialised from this start's 0W (meta init_adapter.sha256), the SFT data stamped with the validated grids
     (prep_state/sft == grids_sha256), and the SFT manifests' prompt system_sha256 == every eval's.
     HF-vs-server prompt-token parity (audit prompt_parity, B5) is reported per eval, descriptively;
  E1 learnable: at least one start's 100/u Wilson lower bound > the floor rate (floors_occ.json gate_floor);
  E2 format: every start's 100/c run has regex-valid 100% and 0 episodes without a log;
  E3 not saturated: every start's 100/u success < 0.85.
The gate never compares starts.  Unconstrained parse rates are descriptive (B3).

    python -m vpd.e3vs.summary score --run RUN_DIR [--floors F] [--opt-len F]
    python -m vpd.e3vs.summary gate [--root OUT/e3vs/runs/pilot] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from vpd import paths
from vpd.e3vs import EPISODE_COUNTS, load_episodes, repo_dir, sha256_file
from vpd.e3vs import geometry as G
from vpd.e3vs.floors import wilson

N_BOOT = 10_000
BOOT_SEED = 0
SATURATION = 0.85
PILOT_STARTS = ("Q3B", "Q3S", "E2")
REQUIRED_EVALS = (("raw", "u"), ("0W", "u"), ("100", "u"), ("100", "c"))
PARITY_MIN_PSNR = 40.0
EXPECTED_IMAGE_TOKENS = 258
FROZEN_COLLIDES = "vpd.e3vs.occupancy:load_collider"
PILOT_SEED = 1001
PILOT_TIER = "100"


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": round(k / n, 6) if n else None, "wilson95": [round(lo, 6), round(hi, 6)]}


def scene_bootstrap(scenes: list[str], hits: list[bool], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Percentile CI of the pooled rate when whole scenes are resampled with replacement."""
    if not scenes:
        return {"n_boot": n_boot, "seed": seed, "ci95": [None, None], "n_scenes": 0}
    ids = sorted(set(scenes))
    pos = {s: i for i, s in enumerate(ids)}
    k = np.zeros(len(ids))
    n = np.zeros(len(ids))
    for s, h in zip(scenes, hits):
        k[pos[s]] += bool(h)
        n[pos[s]] += 1
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    r = k[draw].sum(axis=1) / n[draw].sum(axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"n_boot": n_boot, "seed": seed, "n_scenes": len(ids), "ci95": [round(float(lo), 6), round(float(hi), 6)]}


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def episode_metrics(log: dict, ep: dict, opt_len) -> dict:
    final = np.asarray(log["poses"][-1], dtype=np.float64)
    ok = G.success(final, ep["goals"])
    steps = log["steps"]
    non_stop = sum(s["action"] != "stop" for s in steps)
    _, dist, ang = G.nearest_goal(final, ep["goals"])
    if opt_len is None:
        spl = None
    elif not ok:
        spl = 0.0
    else:
        den = max(float(opt_len), float(non_stop))
        spl = 1.0 if den == 0 else float(opt_len) / den
    return {"episode_index": log["episode_index"], "scene_id": log["scene_id"], "question_type": ep["question_type"],
            "success": bool(ok), "stop_success": bool(ok and log["stopped"]), "stopped": bool(log["stopped"]),
            "decisions": len(steps), "non_stop": non_stop, "collisions": sum(bool(s["collision"]) for s in steps),
            "lenient_errors": sum(bool(s["error"]) for s in steps), "regex_ok": sum(bool(s["regex_ok"]) for s in steps),
            "empty": sum(bool(s.get("empty")) for s in steps),
            "length_cut": sum(s.get("finish_reason") == "length" for s in steps),
            "opt_len": opt_len, "spl": spl, "final_dist_m": round(dist, 4), "final_angle_deg": round(ang, 3)}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 6) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 6) if xs else None


def load_floor(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    f = json.loads(Path(path).read_text())
    g = f["gate_floor"]
    return {"path": str(path), "rate": g["rate"], "k": g["k"], "n": g["n"], "seq": g["seq"],
            "collides": f.get("collides"), "definition": g.get("definition")}


def score(run_dir: Path, repo: Path | None = None, floors: Path | None = None, opt_len: Path | None = None,
          n_boot: int = N_BOOT, seed: int = BOOT_SEED, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    split = man["split"]
    eps = {e["episode_index"]: e for e in load_episodes(split, Path(repo) if repo else repo_dir())}
    expected = sorted(eps) if man.get("episode_indices") in (None, "all") else list(man["episode_indices"])
    logs = {}
    dup = 0
    for r in read_jsonl(run_dir / "episodes.jsonl"):
        dup += r["episode_index"] in logs
        logs[r["episode_index"]] = r
    opt = {int(k): v for k, v in json.loads(Path(opt_len).read_text()).items()} if opt_len else {}
    rows, failed, missing = [], [], []
    for i in expected:
        lg = logs.get(i)
        if lg is None:
            missing.append(i)
            continue
        if lg.get("status") != "ok":
            failed.append(i)
            continue
        rows.append(episode_metrics(lg, eps[i], opt.get(i) if opt_len else None))
    n = len(expected)
    k = sum(r["success"] for r in rows)  # missing / failed episodes count as failures
    dec = sum(r["decisions"] for r in rows)
    spl_rows = [r for r in rows if r["opt_len"] is not None]
    by_qt = {}
    for qt in sorted({eps[i]["question_type"] for i in expected}):
        ids = [i for i in expected if eps[i]["question_type"] == qt]
        kk = sum(r["success"] for r in rows if r["question_type"] == qt)
        by_qt[qt] = rate(kk, len(ids))
    scene_of = [eps[i]["scene_id"] for i in expected]
    hit = {r["episode_index"]: r["success"] for r in rows}
    res = {
        "schema": "vpd.e3vs.summary/1", "eval": man.get("eval"), "start": man.get("start"), "tier": man.get("tier"),
        "mode": man.get("mode"), "split": split, "n_expected": n, "n_logged": len(logs), "n_ok": len(rows),
        "episodes_without_log": len(missing), "episodes_failed": len(failed), "duplicate_logs": dup,
        "missing": missing[:50], "failed": failed[:50],
        "pooled": rate(k, n),
        "bootstrap_scene": scene_bootstrap(scene_of, [hit.get(i, False) for i in expected], n_boot, seed),
        "stop_required": rate(sum(r["stop_success"] for r in rows), n),
        "stop_rate": rate(sum(r["stopped"] for r in rows), n),
        "collision_rate": {"collisions": sum(r["collisions"] for r in rows), "decisions": dec,
                           "rate": round(sum(r["collisions"] for r in rows) / dec, 6) if dec else None},
        "decisions_mean": _mean([r["decisions"] for r in rows]),
        "spl": {"mean": _mean([r["spl"] for r in spl_rows]), "n": len(spl_rows),
                "n_without_optimum": sum(1 for i in expected if opt_len and opt.get(i) is None),
                "source": None if not opt_len else str(opt_len)},
        "final_dist_m": {"mean": _mean([r["final_dist_m"] for r in rows]),
                         "median": _median([r["final_dist_m"] for r in rows])},
        "final_angle_deg": {"mean": _mean([r["final_angle_deg"] for r in rows]),
                            "median": _median([r["final_angle_deg"] for r in rows])},
        "parse": {"decisions": dec,
                  "lenient_valid": dec - sum(r["lenient_errors"] for r in rows),
                  "lenient_valid_rate": round(1 - sum(r["lenient_errors"] for r in rows) / dec, 6) if dec else None,
                  "regex_valid": sum(r["regex_ok"] for r in rows),
                  "regex_valid_rate": round(sum(r["regex_ok"] for r in rows) / dec, 6) if dec else None,
                  "empty": sum(r["empty"] for r in rows), "length_cut": sum(r["length_cut"] for r in rows),
                  "constrained": man.get("mode") == "c"},
        "per_question_type": by_qt,
        "floor": load_floor(floors),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    res["valid"] = not missing and not failed and dup == 0 and n > 0
    if write:
        with open(run_dir / "per_episode.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        good, bad = run_dir / "results.json", run_dir / "results_invalid.json"
        (good if res["valid"] else bad).write_text(json.dumps(res, indent=1) + "\n")
    return res


# ------------------------------------------------------------------------------------------------------------
# gate
# ------------------------------------------------------------------------------------------------------------
def eval_dir(root: Path, start: str, tier: str, mode: str) -> Path:
    return Path(root) / start / "evals" / f"pilot__{start}__{tier}__{mode}"


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _audit_problem(au) -> str | None:
    if au is None:
        return "no audit.json (image tokens unmeasured, B4)"
    if au.get("error"):
        return f"image-token audit failed: {str(au['error'])[:200]}"
    if not au.get("tokens_per_image"):
        return "audit.json has no image token count"
    if au["tokens_per_image"] != [EXPECTED_IMAGE_TOKENS]:
        return f"image tokens {au['tokens_per_image']} != [{EXPECTED_IMAGE_TOKENS}]"
    return None


def _manifest_problem(man, grid_sha: dict) -> str | None:
    if man is None:
        return "no manifest.json"
    if man.get("collides") != FROZEN_COLLIDES:
        return f"collider {man.get('collides')!r}, not the frozen {FROZEN_COLLIDES}"
    sc = man.get("scenes") or {}
    bad = sorted(s for s, e in sc.items() if not e.get("occ_sha256") or e.get("occ_sha256") != grid_sha.get(s))
    if not sc or bad:
        return f"{len(bad)} of {len(sc)} scenes not on the validated occupancy grids (e.g. {bad[:3]})"
    return None


def provenance(root: Path, starts, manifests: dict, sft_dir: Path, prep_state: Path, val: dict | None
               ) -> tuple[list[str], dict]:
    """E0 provenance of the trained and evaluated weights and of the SFT data:
    - each 0W / 100 eval's manifest adapter_sha256 == this start's adapters/<tier>/adapter.pt; raw evals no adapter;
    - adapters/100/meta.json init_adapter.sha256 == this start's adapters/0W/adapter.pt;
    - OUT/e3vs/prep_state/sft (the prep job's stamp) == validation.json grids_sha256;
    - the SFT manifests' prompt system_sha256 (warm-up and demos_100) == every eval manifest's.
    `manifests` maps (start, tier, mode) -> the eval manifest (None = not loaded / missing).  Returns
    (problems, descriptive facts)."""
    probs, facts = [], {"adapter_sha256": {}, "sft_system_sha256": None, "sft_stamp": None}
    stamp = Path(prep_state) / "sft"
    key = stamp.read_text().strip() if stamp.is_file() else None
    facts["sft_stamp"] = key
    if not key or not val or key != val.get("grids_sha256"):
        probs.append(f"SFT data not stamped with the validated grids ({stamp}: {key!r} vs validation "
                     f"{(val or {}).get('grids_sha256')!r})")
    sft_sys = set()
    for f in (Path(sft_dir) / "warmup_manifest.json",
              Path(sft_dir) / f"seed{PILOT_SEED}" / f"demos_{PILOT_TIER}_manifest.json"):
        m = _load(f)
        s = ((m or {}).get("prompt") or {}).get("system_sha256")
        if not s:
            probs.append(f"{f}: missing, or no prompt system_sha256")
        else:
            sft_sys.add(s)
    if len(sft_sys) > 1:
        probs.append(f"SFT manifests disagree on the system prompt: {sorted(sft_sys)}")
    facts["sft_system_sha256"] = sorted(sft_sys)
    for s in starts:
        ad = Path(root) / s / "adapters"
        pts = {t: (sha256_file(ad / t / "adapter.pt") if (ad / t / "adapter.pt").is_file() else None)
               for t in ("0W", PILOT_TIER)}
        facts["adapter_sha256"][s] = pts
        init = ((_load(ad / PILOT_TIER / "meta.json") or {}).get("init_adapter") or {})
        if not pts["0W"] or init.get("sha256") != pts["0W"]:
            probs.append(f"{s}: adapter {PILOT_TIER} not initialised from this start's 0W (init_adapter sha256 "
                         f"{init.get('sha256')!r} vs 0W {pts['0W']!r})")
        for (st, tier, mode), man in manifests.items():
            if st != s or man is None:
                continue
            tag = f"{s} {tier}/{mode}"
            a = man.get("adapter")
            if tier == "raw":
                if a is not None:
                    probs.append(f"{tag}: raw eval carries an adapter ({a.get('meta')})")
            elif not a or not a.get("adapter_sha256") or a.get("adapter_sha256") != pts.get(tier):
                probs.append(f"{tag}: eval adapter_sha256 {(a or {}).get('adapter_sha256')!r} != "
                             f"adapters/{tier}/adapter.pt {pts.get(tier)!r}")
            sys_sha = (man.get("prompt") or {}).get("system_sha256")
            if sft_sys and (not sys_sha or {sys_sha} != sft_sys):
                probs.append(f"{tag}: eval system prompt sha256 {sys_sha!r} != the SFT data's {sorted(sft_sys)}")
    return probs, facts


def gate(root: Path | None = None, starts=PILOT_STARTS, floors: Path | None = None, parity_dir: Path | None = None,
         validation: Path | None = None, n_test: int = EPISODE_COUNTS["test"], prep_state: Path | None = None,
         sft_dir: Path | None = None) -> dict:
    root = Path(root) if root else paths.E3VS_RUNS / "pilot"
    floors = Path(floors) if floors else paths.E3VS_OUT / "floors" / "floors_occ.json"
    parity_dir = Path(parity_dir) if parity_dir else paths.E3VS_PARITY
    validation = Path(validation) if validation else paths.E3VS_OCC / "validation.json"
    prep_state = Path(prep_state) if prep_state else paths.E3VS_OUT / "prep_state"
    sft_dir = Path(sft_dir) if sft_dir else paths.E3VS_SFT
    fl = load_floor(floors)
    val = _load(validation)
    grid_sha = {s: r.get("sha256") for s, r in ((val or {}).get("scenes") or {}).items()}
    runs, e0, audits, mans = {}, [], {}, {}
    for s in starts:
        for tier, mode in REQUIRED_EVALS:
            d = eval_dir(root, s, tier, mode)
            r = _load(d / "results.json")
            runs[(s, tier, mode)] = r
            tag = f"{s} {tier}/{mode}"
            if r is None:
                e0.append(f"{tag}: no valid results.json")
            elif r["n_expected"] != n_test or r["split"] != "test":
                e0.append(f"{tag}: {r['n_expected']} {r['split']} episodes, not {n_test} test")
            au = _load(d / "audit.json")
            prob = _audit_problem(au)
            if prob:
                e0.append(f"{tag}: {prob}")
            pp = (au or {}).get("prompt_parity") or {}
            audits[f"{s}__{tier}__{mode}"] = {"tokens_per_image": (au or {}).get("tokens_per_image"),
                                              "prompt_parity_equal": pp.get("equal"), "prompt_parity_n": pp.get("n")}
            if r is not None:
                mans[(s, tier, mode)] = _load(d / "manifest.json")
                prob = _manifest_problem(mans[(s, tier, mode)], grid_sha)
                if prob:
                    e0.append(f"{tag}: {prob}")
    par = sorted(parity_dir.glob("parity_*.json")) if parity_dir.is_dir() else []
    if not par:
        e0.append("no parity file")
    for p in par:
        x = json.loads(p.read_text())
        if not x.get("pass"):
            e0.append(f"{p.name}: min PSNR {(x.get('official') or {}).get('min_psnr')} < {PARITY_MIN_PSNR}")
    if not val or not val.get("pass"):
        e0.append(f"occupancy validation missing or failed ({validation})")
    shas = {s: (_load(root / s / "adapters" / "0W" / "meta.json") or {}).get("code_sha256") for s in starts}
    if None in shas.values() or len(set(shas.values())) != 1:
        e0.append(f"0W code_sha256 not equal across starts: {shas}")
    if fl is None:
        e0.append(f"no floors file {floors}")
    else:
        if fl.get("collides") != FROZEN_COLLIDES:
            e0.append(f"floor computed with collider {fl.get('collides')!r}, not the frozen {FROZEN_COLLIDES} (B8)")
        stamp = prep_state / "floors"
        key = stamp.read_text().strip() if stamp.is_file() else None
        if not key or not val or key != val.get("grids_sha256"):
            e0.append(f"floor not stamped with the validated grids ({stamp}: {key!r} vs validation "
                      f"{(val or {}).get('grids_sha256')!r})")
    prov_probs, prov = provenance(root, starts, mans, sft_dir, prep_state, val)
    e0 += prov_probs

    per = {}
    for s in starts:
        u, c = runs.get((s, "100", "u")), runs.get((s, "100", "c"))
        per[s] = {
            "100u": None if u is None else {**u["pooled"], "parse_lenient": u["parse"]["lenient_valid_rate"]},
            "100c": None if c is None else {"regex_valid_rate": c["parse"]["regex_valid_rate"],
                                            "episodes_without_log": c["episodes_without_log"],
                                            "episodes_failed": c["episodes_failed"], **c["pooled"]},
            "raw_u": None if runs.get((s, "raw", "u")) is None else runs[(s, "raw", "u")]["pooled"],
            "0W_u": None if runs.get((s, "0W", "u")) is None else runs[(s, "0W", "u")]["pooled"]}
    floor_rate = None if fl is None else fl["rate"]
    e1 = [s for s in starts if per[s]["100u"] and floor_rate is not None
          and per[s]["100u"]["wilson95"][0] > floor_rate]
    e2_fail = [s for s in starts if not per[s]["100c"] or per[s]["100c"]["regex_valid_rate"] != 1.0
               or per[s]["100c"]["episodes_without_log"] != 0 or per[s]["100c"]["episodes_failed"] != 0]
    e3_fail = [s for s in starts if not per[s]["100u"] or per[s]["100u"]["rate"] is None
               or per[s]["100u"]["rate"] >= SATURATION]
    res = {"E0": {"pass": not e0, "problems": e0},
           "E1": {"pass": bool(e1), "starts_above_floor": e1, "floor": fl},
           "E2": {"pass": not e2_fail, "failing_starts": e2_fail},
           "E3": {"pass": not e3_fail, "failing_starts": e3_fail, "threshold": SATURATION},
           "per_start": per, "audit": audits, "provenance": prov,
           "prompt_parity_all_equal": (all(a["prompt_parity_equal"] is True for a in audits.values())
                                       if audits else None),
           "root": str(root), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "the gate never compares starts; unconstrained parse rates are descriptive (DECISIONS B3)"}
    if not res["E0"]["pass"]:
        res["decision"] = "E0 failed: fix the implementation, rerun the affected parts (no scientific conclusion)"
    elif not res["E1"]["pass"]:
        res["decision"] = "E1 failed: not learnable with <= 100 demos; to the user"
    elif not res["E2"]["pass"]:
        res["decision"] = "E2 failed: check the implementation first (E0), else to the user"
    elif not res["E3"]["pass"]:
        res["decision"] = "E3 failed: saturated; to the user, no automatic tier change"
    else:
        res["decision"] = "E0-E3 pass: E3VS may go to the full run (budget needs the user's OK)"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("score", "gate"))
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--floors", type=Path, default=None)
    ap.add_argument("--opt-len", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if a.run is None:
            ap.error("score needs --run")
        res = score(a.run, a.repo, a.floors, a.opt_len)
        print(json.dumps({k: res[k] for k in ("eval", "valid", "pooled", "stop_rate", "parse")}))
        return 0 if res["valid"] else 3
    res = gate(a.root, floors=a.floors)
    out = a.out or (Path(a.root) if a.root else paths.E3VS_RUNS / "pilot") / "gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, out)
    print(json.dumps({k: res[k]["pass"] for k in ("E0", "E1", "E2", "E3")} | {"decision": res["decision"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
