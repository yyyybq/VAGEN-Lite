"""q38 base zero-shot: progress while the chain runs, the pass checks and the numbers once its jobs ended.

    PYTHONPATH=. python -B vagen/q38/base_zs/check.py [--easi]      # read-only; one PASS / FAIL line per check

Plan A (q38/runs/q38-base-zs): for each of q38base (greedy) and q38base_t08 (one sampled pass) on val 413, test 292
and the screened test's 59 new tasks; the screened 674 composed (val's 382 + test's 233 + the 59); P0; the probe.
Plan B (--easi): the 9 EASI-8 jobs.  The 8B base's numbers are printed beside as a reference (not a pass line).
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

from data_gen.interiorgs_v10 import evaluation as E

U = Path("/mnt/umm/users/yinbaiqiao")
RUN = U / "q38/runs/q38-base-zs"
MODEL = str(U / "q38/models/Qwen3.8-27B")
SETS = {"val": ("q38bz_eval_val_v2", 413), "test": ("q38bz_eval_test_v2", 292),
        "new59": ("q38bz_eval_test_v2_screened_new59", 59)}
SCREENED = {"val": U / "VAGEN-Lite-data/exps/matrix_inputs_20261029/eval_val_v2_screened/tasks.jsonl",
            "test": U / "VAGEN-Lite-data/exps/matrix_inputs_20261029/eval_test_v2_screened/tasks.jsonl"}
REF_8B = U / "models_runs/eval_sets"  # eval_{val,test}_v2/m_qwen3vl8b_base_t08
REF_8B_P0 = U / "VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_q3_20261027/p0qa_20261103/p0/q3base/summary.json"
REF_8B_PROBE = U / "VAGEN-Lite-data/exps/diagnostics_20261101/probe_v3g_q3/base_qwen3.json"
REF_8B_EASI = U / "VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_q3_20261027/easi8"
EASI_JOBS = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site_image", "site_video", "blink",
             "3dsrbench", "embspatial")
INVALID_MAX = 0.05  # unreadable turns / turns (8B base: 0.0 on val and test with the same format example)
GPU = 0.9
SAMPLED = {"greedy": False, "samples": 1, "temperature": 0.8, "top_p": 0.92, "sample_seed": 20261016, "entropy": True}
GREEDY = {"greedy": True, "samples": 1, "temperature": 0.0, "top_p": 1.0, "sample_seed": None, "entropy": False}
COMMON = {"version": "interiorgs_v10_eval_v2", "prompt_format": "no_think", "max_turns": 12, "history": True,
          "sft_text": False, "env_switches": {"prompt_format_example": "action_names"}, "model_type": "qwen3_5",
          "chat_template_kwargs": {"enable_thinking": False}, "llm_kwargs": {"generation_config": "vllm"},
          "cuda_visible_devices": "1", "gpu_memory_utilization": GPU}


def load(path: Path, lines: bool = False):
    if not path.is_file():
        return [] if lines else {}
    text = path.read_text()
    return [json.loads(x) for x in text.splitlines() if x.strip()] if lines else json.loads(text)


def exit_status(job: str, mode: str):
    return load(RUN / "eval_runs" / job / "logs" / f"{mode}_exit.json").get("exit_status")


def action(checks, record, result: str, kind: dict):
    out_rows = {}
    for key, (set_name, n) in SETS.items():
        ev = RUN / "sets" / set_name
        out = ev / result
        pre = f"{result} {key}: "
        model, rows, done = load(out / "model.json"), load(out / "results.jsonl", True), load(out / "done.json")
        ok = [r for r in rows if "error" not in r]
        src = load(ev / "source.json")
        checks[pre + "0 task set copy unchanged"] = hashlib.sha256((ev / "tasks.jsonl").read_bytes()).hexdigest() == \
            src.get("sha256")
        if not done:
            record[pre.strip(": ")] = {"progress": f"{len(rows)}/{n}", "errors": [r["error"][:200] for r in rows
                                                                                 if "error" in r][:3]}
            checks[pre + "1 done"] = False
            continue
        checks[pre + f"1 done.json {n} episodes, indices 0..{n - 1} once, no error row"] = (
            done.get("episodes") == n and sorted(r["index"] for r in rows) == list(range(n)) and len(ok) == n)
        checks[pre + "2 job exit 0"] = exit_status(f"{key}__{result}", "eval") == 0
        want = {**COMMON, **kind, "model": MODEL}
        checks[pre + "3 model.json: the base 27B, qwen3_5, thinking off, vllm defaults, card 1 at 0.9, rules"] = all(
            model.get(k) == v for k, v in want.items())
        answers = [str(t.get("answer") or "") for r in ok for t in r.get("trajectory") or []]
        checks[pre + "4 no thinking text in the answers"] = bool(answers) and not any(
            "<think>" in a or "</think>" in a for a in answers)
        m = E.metrics(ok)
        checks[pre + f"5 unreadable turns / turns <= {INVALID_MAX}"] = (m.get("invalid_rate") or 0) <= INVALID_MAX
        secs = (done["seconds"] - model["load_seconds"]) / n if model.get("load_seconds") is not None else None
        record[pre.strip(": ")] = {"success": m.get("success"), "invalid_rate": m.get("invalid_rate"),
                                   "entropy": m.get("entropy"), "success_by_family": m.get("success_by_family"),
                                   "mean_turns": round(statistics.mean(r["turns"] for r in ok), 2),
                                   "unreadable_forms": E.unreadable_forms(ok),
                                   "sec_per_task": round(secs, 2) if secs else None,
                                   "load_seconds": model.get("load_seconds")}
        out_rows[key] = ok
    if set(out_rows) == set(SETS):
        both = out_rows["val"] + out_rows["test"]
        record[f"{result} all_705"] = {"success": E.metrics(both).get("success"),
                                       "invalid_rate": E.metrics(both).get("invalid_rate")}
        by_id = {r["task_id"]: r for r in both + out_rows["new59"]}
        ids = [t["task_id"] for k in ("val", "test") for t in load(SCREENED[k], True)]
        found = [by_id[i] for i in ids if i in by_id]
        checks[f"{result} screened: all 674 tasks found (382 val + 233 test + 59 new)"] = len(found) == len(ids) == 674
        sv = [by_id[t["task_id"]] for t in load(SCREENED["val"], True) if t["task_id"] in by_id]
        st = [by_id[t["task_id"]] for t in load(SCREENED["test"], True) if t["task_id"] in by_id]
        record[f"{result} screened_674"] = {"success": E.metrics(found).get("success"),
                                            "val_382": E.metrics(sv).get("success"),
                                            "test_292": E.metrics(st).get("success")}


def qa(checks, record):
    out = RUN / "qa" / "p0" / "q38base"
    done, s = load(out / "done.json"), load(out / "summary.json")
    checks["p0: done, job exit 0"] = bool(done) and exit_status("qa_p0__q38base", "qa") == 0
    if s:
        meta = s.get("meta") or {}
        ref = load(REF_8B_P0).get("meta") or {}
        checks["p0: 5,590 questions, the 8B's request count, qwen3_5 thinking off, eager, 0.9"] = (
            meta.get("questions") == 5590 and meta.get("requests") == ref.get("requests", 20082)
            and meta.get("model") == MODEL and meta.get("chat_template_kwargs") == {"enable_thinking": False}
            and meta.get("llm_kwargs") == {"generation_config": "vllm", "enforce_eager": True}
            and meta.get("gpu_memory_utilization") == GPU)
        preds = load(out / "predictions.jsonl", True)
        checks["p0: every request answered, no thinking text"] = (
            len(preds) == meta.get("requests") and all("reply" in p for p in preds)
            and not any("<think>" in (p.get("reply") or "") or "</think>" in (p.get("reply") or "") for p in preds))
        video = [p for p in preds if p.get("media") == "video"]
        checks["p0: video questions ran with their frames (prompt tokens > 1,000)"] = bool(video) and all(
            (p.get("prompt_tokens") or 0) > 1000 for p in video)
        record["p0"] = {"official": (s["summary"].get("res256/orig") or {}).get("official"),
                        "all_orig": (s["summary"].get("res256/orig") or {}).get("all"),
                        "all_h1": (s["summary"].get("res256/h1") or {}).get("all"),
                        "empty_replies": sum(not (p.get("reply") or "").strip() for p in preds),
                        "seconds": meta.get("seconds"),
                        "ref_8B_all_orig": ((load(REF_8B_P0).get("summary") or {}).get("res256/orig") or {}).get("all")}
    else:
        checks["p0: summary.json"] = False
    out = RUN / "qa" / "probe" / "q38base"
    done, res = load(out / "done.json"), load(out / "q38base.json")
    answers = load(out / "q38base.jsonl", True)
    checks["probe: done, job exit 0"] = bool(done) and exit_status("qa_probe__q38base", "qa") == 0
    checks["probe: 838 answers, qwen3_5 thinking off, 0.9, images on, no thinking text"] = (
        len(answers) == 838 and res.get("model_type") == "qwen3_5" and res.get("images") is True
        and res.get("chat_template_kwargs") == {"enable_thinking": False} and res.get("gpu_memory_utilization") == GPU
        and not any("</think>" in (a.get("output") or "") for a in answers))
    if res:
        ref = load(REF_8B_PROBE)
        record["probe"] = {"all": res.get("all"), "by_qtype_goal": {k: (res.get("qtype") or {}).get(k)
                                                                    for k in ("goal_now", "goal_after")},
                           "load_seconds": res.get("load_seconds"), "answer_seconds": res.get("answer_seconds"),
                           "ref_8B_all": ref.get("all")}


def easi(checks, record):
    scores = {}
    for b in EASI_JOBS:
        done = load(RUN / "qa" / b / "q38base" / "done.json")
        checks[f"easi {b}: done (full run, no --limit), qwen3_5"] = bool(done) and done.get("limit") is None and \
            done.get("model_type") == "qwen3_5"
        scores[b] = done.get("score")
        ref = load(REF_8B_EASI / b / "q3base" / "done.json")
        record[f"easi {b}"] = {"score": done.get("score"), "ref_8B_base": ref.get("score")}
    record["easi scores"] = scores


def main() -> int:
    checks, record = {}, {}
    if "--easi" in sys.argv[1:]:
        easi(checks, record)
    else:
        action(checks, record, "q38base", GREEDY)
        action(checks, record, "q38base_t08", SAMPLED)
        qa(checks, record)
        for key in ("val", "test"):  # reference only
            rr = load(REF_8B / f"eval_{key}_v2" / "m_qwen3vl8b_base_t08" / "results.jsonl", True)
            if rr:
                record[f"ref_8B_base_t08 {key}"] = {"success": E.metrics(rr).get("success"),
                                                    "invalid_rate": E.metrics(rr).get("invalid_rate")}
    print(json.dumps(record, indent=1, ensure_ascii=False, default=str))
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print("ALL PASS" if checks and all(checks.values()) else "NOT ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
