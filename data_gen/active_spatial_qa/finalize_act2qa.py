#!/usr/bin/env python3
"""Freeze eligible RGB bank and write the Act->QA table. Reuses qa_eval.parse_answer."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_eval import parse_answer, answer_metrics
from PIL import Image
import numpy as np
from image_contract_v2 import image_stats
from data_gen.active_spatial_qa.generate_paired_qa import _load_scene_context
from data_gen.active_spatial_qa.image_contract_v2 import classify_scene_pose


def _load_jsonl(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _pose_close(a, b, atol=1e-4):
    if a is None or b is None:
        return False
    try:
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return bool(aa.shape == bb.shape == (4, 4) and np.isfinite(aa).all()
                    and np.isfinite(bb).all() and np.allclose(aa, bb, atol=atol, rtol=0))
    except Exception:
        return False


def freeze(args):
    out = Path(args.output_dir)
    rows = _load_jsonl(args.rendered_bank)
    errors = _load_jsonl(args.errors) if args.errors else []
    frozen = []
    stats = {
        "candidates": len(rows),
        "rendered": 0,
        "decode_valid": 0,
        "pose_matched": 0,
        "observable": 0,
        "eligible": 0,
        "yes_eligible": 0,
        "no_eligible": 0,
        "errors": len(errors),
        "render_errors": errors,
    }
    diagnostics = []
    contexts = {}
    for rec in rows:
        render = rec.get("render") or {}
        img_path = render.get("image_path") or rec.get("public_observation", {}).get("image_path")
        decode_valid = False
        checked_stats = None
        size = None
        if img_path:
            try:
                im = Image.open(img_path)
                im.load()
                size = list(im.size)
                checked_stats = image_stats(im)
                decode_valid = True
            except Exception as exc:
                rec = dict(rec)
                rec["decode_error"] = str(exc)
        pose_matched = _pose_close(render.get("pose_c2w"), rec.get("state_pose_c2w"))
        nonblank = bool(checked_stats and checked_stats['nonblank'])
        observable = decode_valid and nonblank and rec.get("observability_validity") == "valid"
        scene_root = getattr(args, 'scene_root', '')
        scene = rec.get('scene_id')
        if scene_root and rec.get('state_pose_c2w'):
            if scene not in contexts:
                contexts[scene] = _load_scene_context(scene_root, scene)
            context = contexts[scene]
            position = [r[3] for r in rec['state_pose_c2w'][:3]]
            legality = classify_scene_pose(position, *context) if context else {'status': 'coordinate_unconfirmed', 'reasons': ['scene_context_unavailable']}
        else:
            legality = {'status': 'coordinate_unconfirmed', 'reasons': ['freeze_requires_scene_root']}
        supervision_verified = rec.get('task_type') != 'screen_occupancy' or rec.get('supervision_contract_status') == 'verified'
        eligible = observable and pose_matched and size == [512, 512] and legality.get('status') == 'legal_indoor' and rec.get('label_validity') == 'valid' and rec.get('private_answer') in ('Yes', 'No') and supervision_verified
        rec = dict(rec)
        rec["decode_valid"] = decode_valid
        rec["pose_matched"] = pose_matched
        rec["eligible"] = eligible
        if eligible:
            rec["observability_validity"] = "valid"
            rec["public_observation"] = dict(
                rec.get("public_observation") or {},
                image_path=str(Path(img_path).resolve()),
            )
            frozen.append(rec)
        audit = ((rec.get("_audit") or {}).get("predicate") or {}).get("details") or {}
        visual = audit.get("visual_bbox_metrics") or {}
        objects = visual.get("objects") or []
        diagnostics.append({
            "sample_id": rec.get("sample_id"),
            "private_answer": rec.get("private_answer"),
            "parent_goal_id": rec.get("parent_goal_id"),
            "eligible": eligible,
            "decode_valid": decode_valid,
            "pose_matched": pose_matched,
            "observable": observable,
            "image_size": size,
            "rgb_stats": render.get("rgb_stats"),
            "independent_image_quality": checked_stats,
            "scene_pose_legality": legality,
            "supervision_verified": supervision_verified,
            "center_uv": objects[0].get("center_uv") if objects else None,
            "visual_occupancy": visual.get("visual_occupancy"),
            "target_occupancy": visual.get("target_occupancy"),
            "area_ratio": objects[0].get("area_ratio") if objects else None,
            "question": rec.get("question"),
        })
        stats["rendered"] += 1
        stats["decode_valid"] += int(decode_valid)
        stats["pose_matched"] += int(pose_matched)
        stats["observable"] += int(observable)
        stats["eligible"] += int(eligible)
        if eligible:
            if rec.get("private_answer") == "Yes":
                stats["yes_eligible"] += 1
            elif rec.get("private_answer") == "No":
                stats["no_eligible"] += 1
    stats["yes_no_center_only_risk"] = _center_only_risk(diagnostics)
    stats["prompt_leak"] = _prompt_leak(frozen or rows)
    (out / "rgb_bank_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out / "rgb_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    with (out / "frozen_visual_manifest.jsonl").open("w") as f:
        for rec in frozen:
            f.write(json.dumps(rec, default=str) + "\n")
    ids = [r["sample_id"] for r in frozen]
    (out / "eligible_ids.json").write_text(
        json.dumps({"eligible_ids": ids, "n": len(ids), "yes": stats["yes_eligible"], "no": stats["no_eligible"]}, indent=2) + "\n"
    )
    print(json.dumps(stats, indent=2))
    return stats


def _center_only_risk(diags):
    yes = [d for d in diags if d.get("private_answer") == "Yes" and d.get("center_uv")]
    no = [d for d in diags if d.get("private_answer") == "No" and d.get("center_uv")]

    def dist(uv):
        return math.hypot(float(uv[0]) - 256.0, float(uv[1]) - 256.0)

    return {
        "yes_center_mean": sum(dist(d["center_uv"]) for d in yes) / len(yes) if yes else None,
        "no_center_mean": sum(dist(d["center_uv"]) for d in no) / len(no) if no else None,
        "yes_occ_mean": sum((d.get("visual_occupancy") or 0) for d in yes) / len(yes) if yes else None,
        "no_occ_mean": sum((d.get("visual_occupancy") or 0) for d in no) / len(no) if no else None,
        "note": "If Yes/No differ only by center distance and occupancy overlap is large, visual separability is weak.",
    }


def _prompt_leak(rows):
    leaks = []
    for r in rows:
        q = r.get("question") or ""
        public = json.dumps(r.get("public_observation") or {})
        if r.get("private_answer") and r.get("private_answer") in q.replace("Answer Yes or No.", ""):
            leaks.append({"sample_id": r.get("sample_id"), "kind": "label_in_question"})
        if "bbox_min" in q or "occupancy_ratio" in public or "score" in public:
            leaks.append({"sample_id": r.get("sample_id"), "kind": "geometry_in_public"})
    return {"n": len(leaks), "items": leaks[:10]}


def _read_cfg(path):
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {"raw_unreadable": True, "path": str(path)}


def identities(args):
    out = Path(args.output_dir)
    base_cfg = _read_cfg(Path(args.base) / "config.json") if args.base else None
    act_cfg = _read_cfg(Path(args.act) / "config.json") if args.act else None
    train_hit = None
    if args.train_config and Path(args.train_config).exists():
        text = Path(args.train_config).read_text()
        found = re.findall(r"path:\s*(Qwen/[^\s]+)", text)
        train_hit = found[0] if found else None
    pair_ok = False
    reason = []
    if base_cfg and act_cfg and not act_cfg.get("raw_unreadable"):
        pair_ok = (
            base_cfg.get("model_type") == act_cfg.get("model_type")
            and base_cfg.get("hidden_size") == act_cfg.get("hidden_size")
            and base_cfg.get("num_hidden_layers") == act_cfg.get("num_hidden_layers")
            and train_hit == "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        if not pair_ok:
            reason.append("config fields or train init path mismatch")
    else:
        reason.append("act config missing/unreadable")
    rec = {
        "base": {
            "path": args.base,
            "config": {k: (base_cfg or {}).get(k) for k in ("architectures", "model_type", "hidden_size", "num_hidden_layers", "transformers_version")},
        },
        "act": {
            "path": args.act,
            "source": args.act_source,
            "config": {k: (act_cfg or {}).get(k) for k in ("architectures", "model_type", "hidden_size", "num_hidden_layers", "transformers_version")},
        },
        "train_init_path": train_hit,
        "strict_pair": pair_ok,
        "pair_label": "STRICT_MATCH" if pair_ok else "INTERFACE_CANARY_ONLY",
        "reason": reason,
    }
    (out / "model_identities.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


def _metrics(bank_rows, pred_path):
    pred = {}
    for r in _load_jsonl(pred_path):
        pred[str(r.get("sample_id"))] = parse_answer(r.get("prediction", r.get("response", "")))
    golds = [r["private_answer"] for r in bank_rows]
    parsed = [pred.get(r["sample_id"]) for r in bank_rows]
    return {
        **answer_metrics(golds, parsed),
        "always_yes": (sum(g == "Yes" for g in golds) / len(bank_rows)) if bank_rows else None,
        "always_no": (sum(g == "No" for g in golds) / len(bank_rows)) if bank_rows else None,
        "predictions": len(pred),
    }


def table(args):
    out = Path(args.output_dir)
    bank = _load_jsonl(args.bank)
    ident = json.loads((out / "model_identities.json").read_text()) if (out / "model_identities.json").exists() else {}
    base_m = _metrics(bank, args.base_pred) if args.base_pred and Path(args.base_pred).exists() else None
    act_m = _metrics(bank, args.act_pred) if args.act_pred and Path(args.act_pred).exists() else None
    strict = bool(ident.get("strict_pair"))
    delta = None
    if base_m and act_m and strict:
        delta = {k: act_m[k] - base_m[k] if act_m[k] is not None and base_m[k] is not None else None
                 for k in ("accuracy", "balanced_accuracy", "yes_accuracy", "no_accuracy", "format_failure_rate")}
    result = {
        "pair_label": ident.get("pair_label", "UNKNOWN"),
        "strict_pair": strict,
        "base": base_m,
        "act": act_m,
        "delta": delta,
        "always_yes": (base_m or act_m or {}).get("always_yes"),
        "always_no": (base_m or act_m or {}).get("always_no"),
        "n": len(bank),
        "task_types": dict(Counter(r.get("task_type") for r in bank)),
    }
    (out / "act2qa_table.json").write_text(json.dumps(result, indent=2) + "\n")

    def fmt(m, key):
        if not m:
            return "NOT_RUN"
        return "N/A" if m[key] is None else "%.3f" % m[key]

    lines = [
        "# ACT2QA_TABLE",
        "",
        "pair_label: %s" % result["pair_label"],
        "n_eligible: %s" % result["n"],
        "task_types: %s" % result["task_types"],
        "",
        "| Model | QA Acc | Balanced Acc | Yes Acc | No Acc | Format Fail |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| Base | %s | %s | %s | %s | %s |" % (fmt(base_m, "accuracy"), fmt(base_m, "balanced_accuracy"), fmt(base_m, "yes_accuracy"), fmt(base_m, "no_accuracy"), fmt(base_m, "format_failure_rate")),
        "| Act-trained | %s | %s | %s | %s | %s |" % (fmt(act_m, "accuracy"), fmt(act_m, "balanced_accuracy"), fmt(act_m, "yes_accuracy"), fmt(act_m, "no_accuracy"), fmt(act_m, "format_failure_rate")),
    ]
    if delta:
        lines.append("| d Act->QA | " + " | ".join("N/A" if delta[k] is None else "%+.3f" % delta[k]
                     for k in ("accuracy", "balanced_accuracy", "yes_accuracy", "no_accuracy", "format_failure_rate")) + " |")
    elif act_m and base_m and not strict:
        lines.append("| d Act->QA | INTERFACE_CANARY_ONLY | INTERFACE_CANARY_ONLY | INTERFACE_CANARY_ONLY | INTERFACE_CANARY_ONLY | INTERFACE_CANARY_ONLY |")
    else:
        lines.append("| d Act->QA | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |")
    lines += [
        "",
        "always-Yes baseline: %s" % result.get("always_yes"),
        "always-No baseline: %s" % result.get("always_no"),
        "",
        "Single task only: screen_occupancy. Do not treat this as a multi-task result.",
    ]
    (out / "ACT2QA_TABLE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--rendered-bank", required=True)
    f.add_argument("--errors", default="")
    f.add_argument("--output-dir", required=True)
    f.add_argument("--scene-root", default="")
    i = sub.add_parser("identities")
    i.add_argument("--output-dir", required=True)
    i.add_argument("--base", default="")
    i.add_argument("--act", default="")
    i.add_argument("--act-source", default="")
    i.add_argument("--train-config", default="")
    t = sub.add_parser("table")
    t.add_argument("--output-dir", required=True)
    t.add_argument("--bank", required=True)
    t.add_argument("--base-pred", default="")
    t.add_argument("--act-pred", default="")
    args = ap.parse_args()
    if args.cmd == "freeze":
        freeze(args)
    elif args.cmd == "identities":
        identities(args)
    else:
        table(args)


if __name__ == "__main__":
    main()
