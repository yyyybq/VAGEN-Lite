#!/usr/bin/env python3
"""Export inspectable paired spatial supervision; never load or train a model.

All input fields are preserved in audit/input_manifest.jsonl. Only messages and
images enter training-format JSONL. Human-pending samples can be candidates but
never approved training data. Recipes deliberately have do_train: false.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import yaml

SCHEMA = "active_spatial_qa_production_export_v1"
SPLITS = ("train", "val", "test")
REVIEW_LABELS = ("通过", "坏图", "目标不明确", "题意不明确", "标签错误", "证据不足")


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, values):
    with Path(path).open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def finite_json(value):
    """Keep audit records JSON portable, explicitly encode nonfinite numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_number": str(value)}
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def goal_id(row):
    return str(row.get("goal_id") or row.get("parent_goal_id") or row.get("source_task_id") or "")


def image_path(row, base):
    name = row.get("public_observation", {}).get("image_path") or row.get("image_path")
    if not isinstance(name, str) or not name:
        raise ValueError("missing_image_path")
    path = Path(name)
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def human_review(row):
    review = row.get("human_review") or row.get("semantic_review") or {}
    status = str(review.get("status", "pending"))
    # An assistant review can never authorize the human-approved partition.
    if review.get("reviewer_type") in {"assistant", "model", "automatic"}:
        return False, review
    accepted = status in {"PASS", "passed", "通过", "accepted"}
    complete = bool(str(review.get("reviewer") or "").strip()) and (review.get("reason") is None or isinstance(review["reason"], str))
    return accepted and complete, review


def initial_human_review(row, image_sha, cache):
    """Restore only complete user-supplied reviews bound by the apply tool.

    This checks recorded provenance, not a reviewer's real-world identity.
    Unreviewed rows and incomplete/unbound human claims remain pending.
    """
    pending = {"status": "待审", "reviewer": "", "reason": "", "reviewed_at": None}
    review, application = row.get("human_review") or {}, row.get("human_review_application") or {}
    if review.get("status") == "待审" or not review:
        return pending, "pending"
    try:
        if review.get("status") not in REVIEW_LABELS or review.get("reviewer_type") != "human":
            raise ValueError("not_an_explicit_human_label")
        if not all(isinstance(review.get(key), str) and review[key].strip() for key in ("reviewer", "reviewed_at")):
            raise ValueError("incomplete_human_review")
        if review.get("reason") is not None and not isinstance(review["reason"], str):
            raise ValueError("review_reason_not_text")
        stamp = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("review_time_missing_timezone")
        if application.get("schema_version") != "active_spatial_human_review_application_v1":
            raise ValueError("missing_review_application_provenance")
        if not image_sha or application.get("reviewed_image_sha256") != image_sha:
            raise ValueError("reviewed_image_sha_changed")
        def bound_file(path_key, sha_key):
            path = Path(application[path_key]).resolve(strict=True)
            expected = application[sha_key]
            key = (str(path), expected)
            if key not in cache:
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError(path_key + "_sha_mismatch")
                cache[key] = raw
            return cache[key]
        bound_file("source_manifest", "source_manifest_sha256")
        body = json.loads(bound_file("preserved_review_copy", "review_file_sha256"))
        if body.get("schema_version") != "active_spatial_human_review_v1" or body.get("manifest_sha256") != application["source_manifest_sha256"]:
            raise ValueError("review_source_manifest_binding_mismatch")
        matches = [record for record in body.get("records", []) if record.get("sample_id") == row["sample_id"]]
        if len(matches) != 1:
            raise ValueError("human_record_missing_or_duplicated")
        original = matches[0]
        if original.get("scene_id") != row.get("scene_id") or original.get("goal_id") != goal_id(row):
            raise ValueError("human_record_identity_mismatch")
        for key in ("status", "reviewer", "reason", "reviewed_at"):
            if review.get(key) != original.get(key):
                raise ValueError("human_record_changed:" + key)
        return {key: (review.get("reason") or "") if key == "reason" else review[key] for key in pending}, "verified_application_provenance"
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return pending, "pending_unverified_human_claim:" + str(exc)


def validate_messages(record):
    if set(record) != {"messages", "images"}:
        raise ValueError("Training record must contain only messages/images")
    turns = record["messages"]
    if turns and turns[0]["role"] == "system":
        turns = turns[1:]
    if not turns or len(turns) % 2:
        raise ValueError("Incomplete user/assistant conversation")
    for index, turn in enumerate(turns):
        if set(turn) != {"role", "content"}:
            raise ValueError("Unexpected model-input field")
        if turn["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Nonalternating turns")
        if not isinstance(turn["content"], str) or not turn["content"].strip():
            raise ValueError("Empty message")
    if sum(turn["content"].count("<image>") for turn in record["messages"]) != len(record["images"]):
        raise ValueError("Image-token count mismatch")
    for name in record["images"]:
        with Image.open(name) as img:
            img.verify()


def candidate_reasons(row):
    reasons = []
    if row.get("task_type") not in {"projective_relations", "fov_inclusion"}:
        reasons.append("task_family_not_released")
    if row.get("candidate_qa_allowed", row.get("qa_sft_allowed")) is not True:
        reasons.append("candidate_qa_disallowed")
    if row.get("assistant_review", {}).get("status") != "passed":
        reasons.append("assistant_review_not_passed")
    if row.get("label_validity") != "valid" or row.get("private_answer") not in {"Yes", "No"}:
        reasons.append("invalid_label")
    obs = row.get("public_observation", {})
    if row.get("observability_validity") != "valid":
        reasons.append("invalid_observability")
    if obs.get("observation_config") != "single_image" or obs.get("history"):
        reasons.append("not_single_image")
    geometry = row.get("geometry") or row.get("position_validation") or {}
    if geometry.get("status") != "legal_indoor":
        reasons.append("legal_indoor_unverified")
    question = row.get("question")
    if not isinstance(question, str) or not question.strip() or "<image>" in question:
        reasons.append("invalid_question")
    if not goal_id(row):
        reasons.append("missing_goal_id")
    return reasons


def resolve_splits(source_rows, path, seed):
    scene_set = {str(row["scene_id"]) for row in source_rows}
    lookup = {}
    if path:
        document = json.loads(Path(path).read_text())
        for scene, split in document.get("scene_splits", {}).items():
            if split not in SPLITS:
                raise ValueError("Unknown partition in scene_splits")
            lookup[str(scene)] = split
        for split in SPLITS:
            for scene in document.get(split + "_scenes", []):
                if str(scene) in lookup and lookup[str(scene)] != split:
                    raise ValueError("Scene appears in multiple frozen partitions")
                lookup[str(scene)] = split
        if scene_set - lookup.keys():
            raise ValueError("Manifest scene missing from frozen scene split")
    elif all(row.get("split") in SPLITS for row in source_rows):
        for row in source_rows:
            scene, split = str(row["scene_id"]), row["split"]
            if scene in lookup and lookup[scene] != split:
                raise ValueError("Same scene appears in multiple input partitions")
            lookup[scene] = split
    else:
        if any(row.get("split") in SPLITS for row in source_rows):
            raise ValueError("Partial split assignments are ambiguous")
        ordered = sorted(scene_set)
        random.Random(seed).shuffle(ordered)
        lookup = {scene: "train" for scene in ordered}
        if len(ordered) >= 2:
            lookup[ordered[-1]] = "val"
        if len(ordered) >= 3:
            lookup[ordered[-2]] = "test"
    for row in source_rows:
        if row.get("split") in SPLITS and row["split"] != lookup[str(row["scene_id"])]:
            raise ValueError("Input split conflicts with frozen split")
    return lookup


def get_token_counter(args):
    root = Path(args.tokenizer or args.model) if (args.tokenizer or args.model) else None
    token_file = root / "tokenizer.json" if root and root.is_dir() else root
    if not token_file or not token_file.is_file():
        return None, {"status": "unavailable", "reason": "No local tokenizer.json supplied; character counts are not tokens"}
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(token_file))
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids), {
            "status": "available", "path": str(token_file.resolve()), "sha256": sha(token_file),
            "scope": "Exact assistant-content text tokens only; excludes chat-template delimiters, image expansion and trainer masking; effective loss tokens not measured",
        }
    except Exception as exc:
        return None, {"status": "unavailable", "reason": f"Tokenizer loading failed: {type(exc).__name__}: {exc}"}


def supervision_counts(records, counter):
    texts = [turn["content"] for row in records for turn in row["messages"] if turn["role"] == "assistant"]
    lengths = [len(text) for text in texts]
    result = {"examples": len(records), "images": sum(len(row["images"]) for row in records),
              "assistant_turns": len(texts), "assistant_content_characters": sum(lengths),
              "assistant_content_character_lengths": lengths, "effective_loss_tokens": None}
    if counter:
        tokens = [counter(text) for text in texts]
        result.update(assistant_content_tokens=sum(tokens), assistant_content_token_lengths=tokens)
    else:
        result.update(assistant_content_tokens=None, assistant_content_token_lengths=None)
    return result


def validate_action(row, base, goals, scene_splits, converter):
    reasons = []
    gid, scene = goal_id(row), str(row.get("scene_id", ""))
    matched = goals.get((scene, gid))
    if not matched:
        return None, ["goal_not_in_current_qa_manifest"]
    if row.get("task_type") != matched["task_type"]:
        reasons.append("task_family_mismatch")
    if row.get("split") in SPLITS and row["split"] != scene_splits[scene]:
        reasons.append("action_scene_split_conflict")
    if row.get("success") is not True or row.get("score_contract", {}).get("success_source") != "canonical_gates":
        reasons.append("canonical_success_unverified")
    trace, primitive = row.get("trajectory_trace", []), row.get("primitive_trajectory_trace", [])
    if not trace or trace[0].get("canonical_success") is not False or trace[-1].get("canonical_success") is not True:
        reasons.append("not_false_to_true")
    if len(trace) < 2 or [part.get("frame_index") for part in trace] != list(range(len(trace))):
        reasons.append("frame_order_invalid")
    if not primitive or [part.get("primitive_index") for part in primitive] != list(range(len(primitive))):
        reasons.append("primitive_order_invalid")
    actions = [action for part in trace[1:] for action in part.get("actions_from_previous", [])]
    if len(actions) != row.get("total_actions") or actions != [part.get("action_from_previous") for part in primitive[1:]]:
        reasons.append("action_sequence_mismatch")
    if not actions:
        reasons.append("no_actions")
    for part in trace[1:]:
        replay = part.get("runtime_reward_trace", {})
        if not replay or replay.get("identity", {}).get("task_id") != gid or replay.get("identity", {}).get("scene_id") != scene:
            reasons.append("runtime_replay_identity_unverified")
            break
        if replay.get("components", {}).get("collision", {}).get("events", 1) != 0:
            reasons.append("collision_detected_or_unverified")
    geometry = row.get("geometry_trace", [])
    if len(geometry) != len(primitive) or not geometry or any(item.get("status") != "legal_indoor" for item in geometry):
        reasons.append("all_primitive_positions_not_verified_indoor")
    elif any(item.get("primitive_index") != index or item.get("pose_c2w") != primitive[index].get("pose_c2w") for index, item in enumerate(geometry)):
        reasons.append("geometry_not_bound_to_primitive_pose")
    if row.get("assistant_review", {}).get("status") != "passed":
        reasons.append("action_assistant_review_not_passed")
    if reasons:
        return None, sorted(set(reasons))
    try:
        converter._validate_record(row, base)
        record = converter._convert_record(row, base, strip_think=True)
        validate_messages(record)
        supervised = [action for turn in record["messages"] if turn["role"] == "assistant"
                      for action in re.search(r"<action>(.*?)</action>", turn["content"], re.S).group(1).split("|") if action]
        if supervised != actions:
            return None, ["supervised_action_sequence_mismatch"]
        for name in row.get("image_paths", []) + row.get("primitive_image_paths", []):
            path = Path(name) if Path(name).is_absolute() else base / name
            with Image.open(path) as img:
                img.verify()
        return record, []
    except Exception as exc:
        return None, [f"action_record_invalid:{type(exc).__name__}:{exc}"]


def make_review(out, reviews, source_sha, failure_summary):
    data = json.dumps({"schema_version": SCHEMA, "manifest_sha256": source_sha,
                       "rows": reviews, "review_labels": REVIEW_LABELS, "generation_failures": failure_summary}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    page = REVIEW_HTML.replace("__DATA__", data)
    (out / "review.html").write_text(page, encoding="utf-8")


def copy_inspection_assets(row, source_base, out):
    """Inspection graphics are always separate from original model images."""
    result, errors = {}, []
    for field in ("overlay_path", "camera_map_path"):
        name = row.get("inspection", {}).get(field)
        if not name:
            continue
        try:
            path = Path(name)
            path = (path if path.is_absolute() else source_base / path).resolve(strict=True)
            if path.suffix.lower() == ".svg":
                from xml.etree import ElementTree
                ElementTree.parse(path)
            else:
                with Image.open(path) as image:
                    image.verify()
            target = "review_assets/inspection_" + sha(path) + path.suffix.lower()
            if not (out / target).exists():
                shutil.copyfile(path, out / target)
            result[field] = target
        except Exception as exc:
            errors.append({"field": field, "error": f"{type(exc).__name__}: {exc}"})
    return result, errors


def review_objects(row):
    """Prefer the audited A/B selection over guessing from original object order."""
    projections = row.get("metrics", {}).get("independent", {}).get("projections")
    projections = projections or row.get("planning_audit", {}).get("projections")
    if projections:
        source = projections
    else:
        source = row.get("goal", {}).get("target_object", {}).get("objects", [])
    result = []
    for index, obj in enumerate(source):
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("label") or obj.get("name") or "未知名称")
        oid = str(obj.get("object_id", obj.get("id", obj.get("ins_id", ""))))
        role = str(obj.get("role") or ("AB"[index] if index < 2 else index + 1))
        result.append({"role": role, "label": label, "instance_id": oid,
                       "category_filter": "category:" + label.casefold(),
                       "instance_filter": "instance:" + str(row["scene_id"]) + "/" + oid if oid else None})
    return result


def reason_values(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [reason for item in value for reason in reason_values(item)]
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    return []


def sample_failure_reasons(row, automatic_reasons, linked_failures):
    result = ["自动准入：" + reason for reason in automatic_reasons]
    for key in ("failure_reason", "failure_reasons", "exclusion_reasons", "rejection_reasons"):
        result += ["自动记录：" + reason for reason in reason_values(row.get(key))]
    for failure in linked_failures:
        result += ["生成记录：" + reason for reason in reason_values(failure.get("reason") or failure.get("failure_reason") or failure.get("error_category"))]
    review = row.get("assistant_review", {})
    if review.get("status") not in {None, "pending", "passed"}:
        categories = []
        for key in ("category", "categories", "rejection_category", "reject_category", "failure_category", "failure_reasons", "labels"):
            categories += reason_values(review.get(key))
        if not categories:
            prose = reason_values(review.get("reason") or review.get("reasons"))
            categories = [label for label in REVIEW_LABELS[1:] if any(label in reason for reason in prose)]
            categories = categories or prose or [str(review["status"])]
        result += ["助手审阅：" + category for category in categories]
    return sorted(set(result))


def load_generation_failures(source, explicit_paths, sample_ids, out):
    paths = [Path(name).resolve(strict=True) for name in explicit_paths]
    if not paths:
        paths = [path for path in (source.parent / "all_failures.jsonl", source.parent / "failures.jsonl") if path.is_file()][:1]
    linked, retained, seen, sources = defaultdict(list), [], Counter(), []
    for path in dict.fromkeys(paths):
        source_rows = rows(path)
        digest = sha(path)
        archive = "generation_failures_" + digest[:16] + ".jsonl"
        shutil.copyfile(path, out / "audit" / archive)
        sources.append({"path": str(path), "sha256": digest, "records": len(source_rows), "archived": "audit/" + archive})
        occurrences = Counter()
        for row in source_rows:
            identity = json.dumps(finite_json(row), sort_keys=True, ensure_ascii=False)
            occurrences[identity] += 1
            if occurrences[identity] <= seen[identity]:
                continue
            retained.append(row)
            sid = row.get("sample_id")
            if sid in sample_ids:
                linked[sid].append(row)
        seen |= occurrences
    def summarize(events):
        reasons, phases, scenes, groups = Counter(), Counter(), Counter(), Counter()
        for event in events:
            values = reason_values(event.get("reason") or event.get("failure_reason") or event.get("error_category")) or ["未提供原因"]
            reasons.update(values)
            phases[str(event.get("phase") or "未提供阶段")] += 1
            scenes[str(event.get("scene_id") or "未提供场景")] += 1
            for reason in values:
                groups[(str(event.get("scene_id") or "未提供场景"), str(event.get("phase") or "未提供阶段"), reason)] += 1
        return {"events": len(events), "by_reason": dict(reasons), "by_phase": dict(phases), "by_scene": dict(scenes),
                "groups": [{"scene_id": scene, "phase": phase, "reason": reason, "events": count}
                           for (scene, phase, reason), count in sorted(groups.items())]}
    unmatched = [event for event in retained if event.get("sample_id") not in sample_ids]
    not_rendered = [event for event in unmatched if event.get("render_requested") is False or event.get("phase") in {"position", "sampling", "actual_production_gate_probe", "planning"}]
    result = {"sources": sources, "all_events": summarize(retained),
              "unmatched_events": summarize(unmatched), "explicitly_not_rendered_events": summarize(not_rendered),
              "linked_manifest_samples": len(linked), "unmatched_events_without_sample_id": sum(not event.get("sample_id") for event in unmatched),
              "count_unit": "failure/probe records, not unique generated samples; a record can have several reasons",
              "overlap_policy": "Repeated records within one file retain multiplicity; overlapping files retain the maximum multiplicity per identical record",
              "display_policy": "No image cards are created from failure logs; unmatched records are statistics only"}
    write_json(out / "generation_failure_statistics.json", finite_json(result))
    return linked, finite_json(result)


REVIEW_HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Active Spatial 配对 QA 审阅</title><style>
body{font-family:system-ui,sans-serif;background:#f3f5f8;color:#182331;margin:0}header{padding:20px 24px;background:#14273f;color:white}h1{font-size:24px;margin:0 0 8px}p{line-height:1.5}.toolbar{position:sticky;top:0;background:#fff;padding:12px 24px;border-bottom:1px solid #bbc5d1;z-index:2;display:flex;gap:10px;flex-wrap:wrap;align-items:center}select,input,button,textarea{font:inherit;padding:8px;border:1px solid #bcc6d1;border-radius:5px}button{cursor:pointer;background:#e4eefb}main{padding:20px}.pair{margin-bottom:24px}.pair h2{font-size:16px;font-weight:600;word-break:break-word}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}.card{background:white;border:1px solid #cbd5e1;border-radius:8px;padding:14px;min-width:0}.card img{width:100%;height:320px;object-fit:contain;background:#101820}.meta{font-size:12px;color:#5a6574;overflow-wrap:anywhere}.question{font-size:17px;min-height:48px}.details{background:#f3f5f8;padding:10px;font-size:12px;white-space:pre-wrap;overflow:auto;max-height:340px}.reviewline{display:flex;gap:8px;margin-top:10px}textarea{width:100%;box-sizing:border-box;margin-top:8px;resize:vertical}#counts{font-size:13px}.badge{padding:2px 6px;background:#edf2f7;border-radius:4px}#notice{padding:0 24px;color:#7b4100}.empty{padding:30px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:7px;border-bottom:1px solid #ddd}.assist{color:#6c5a39}.flag{color:#932626}.hidden{display:none!important}@media(max-width:650px){main{padding:10px}.toolbar{position:relative;padding:10px}.card img{height:280px}}
</style><header><h1>Active Spatial 配对 QA 审阅</h1><div>模型视图仅有原图和问题。切换检查视图可看参考答案、投影框、相机位置与人工审阅工具。</div></header>
<div class="toolbar"><label><input type="checkbox" id="evidence">检查视图</label><div id="inspection-controls" class="hidden" style="display:flex;gap:10px;flex-wrap:wrap;align-items:center"><label>场景 <select id="scene"></select></label><label>任务 <select id="task"></select></label><label>对象 <select id="object"></select></label><label>失败原因 <select id="failure"></select></label><label>划分 <select id="split"></select></label><label>答案 <select id="answer"></select></label><label>人工结论 <select id="status"></select></label><label>审阅人 <input id="reviewer" placeholder="填写真实审阅人" size="16"></label><button id="save">保存审阅 JSON</button><button id="import">导入审阅 JSON</button><input type="file" id="file" accept="application/json" hidden><span id="counts"></span></div></div>
<p id="notice" class="hidden">所有未人工审阅记录为“待审”。每条结论需填写审阅人及理由；本地缓存仅保存在当前浏览器，交付时请保存 JSON。</p><section id="generation-failures" class="hidden" style="margin:16px 24px;padding:14px;background:white;border:1px solid #cbd5e1;border-radius:8px"></section><main id="root"></main>
<script id="payload" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('payload').textContent),key='qa-review:'+D.manifest_sha256;let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}const labels=['待审',...D.review_labels];const el=id=>document.getElementById(id);const option=(value,text)=>{let n=document.createElement('option');n.value=value;n.textContent=text??value;return n};for(const [id,field]of [['scene','scene_id'],['task','task_type'],['split','split'],['answer','private_answer']]){el(id).append(option('','全部'));for(const v of [...new Set(D.rows.map(r=>r[field]??''))].sort())el(id).append(option(v,v||'无'));}el('status').append(option('','全部'));for(const l of labels)el('status').append(option(l));
const objectOptions=new Map();for(const r of D.rows)for(const o of r.objects||[]){objectOptions.set(o.category_filter,'类别：'+o.label);if(o.instance_filter)objectOptions.set(o.instance_filter,'实例：'+r.scene_id+' / '+o.label+' #'+o.instance_id)}el('object').append(option('','全部对象（角色 A / B）'));for(const[v,label]of [...objectOptions].sort((a,b)=>a[1].localeCompare(b[1])))el('object').append(option(v,label));
const failureOptions=new Set(D.rows.flatMap(r=>r.failure_reasons||[]));for(const reason of Object.keys(D.generation_failures?.unmatched_events?.by_reason||{}))failureOptions.add('生成记录：'+reason);el('failure').append(option('','全部'),option('__any__','有失败或拒绝原因'),option('__none__','未记录失败原因'));for(const reason of [...failureOptions].sort())el('failure').append(option(reason,reason.length>110?reason.slice(0,107)+'…':reason));let shownCount=D.rows.length;
const initialReviews=new Map(D.rows.map(r=>[r.sample_id,r.initial_human_review||{status:'待审',reviewer:'',reason:'',reviewed_at:null}]));const state=id=>saved[id]||initialReviews.get(id)||{status:'待审',reviewer:'',reason:'',reviewed_at:null};function cache(){try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){el('notice').textContent='浏览器不能写本地缓存，请及时保存 JSON。';}}function text(tag,content,cls){const n=document.createElement(tag);n.textContent=content;if(cls)n.className=cls;return n}
function update(id,status,reason){const old=state(id);saved[id]={status,reviewer:el('reviewer').value.trim()||old.reviewer,reason,reviewed_at:status==='待审'?null:new Date().toISOString()};cache();counts();}
function counts(){const finished=D.rows.filter(r=>state(r.sample_id).status!=='待审').length;el('counts').textContent=`当前 ${shownCount} / ${D.rows.length} 条 · 人工已记 ${finished} · 待审 ${D.rows.length-finished}`;}
function failureMatch(reasons){const value=el('failure').value;return !value||(value==='__any__'?reasons.length>0:value==='__none__'?reasons.length===0:reasons.includes(value));}
function generationStats(inspect){const panel=el('generation-failures');panel.className=inspect?'':'hidden';panel.replaceChildren();if(!inspect)return;const summary=D.generation_failures||{};panel.append(text('h2','未纳入图像清单的生成记录'));if(!summary.sources?.length){panel.append(text('p','本次未加载生成失败日志。'));return}panel.append(text('p',`共 ${summary.unmatched_events?.events||0} 个未关联图像清单的失败/探测事件，其中 ${summary.explicitly_not_rendered_events?.events||0} 个记录在渲染请求之前。这里仅统计日志，不创建图片样例。`));panel.append(text('p','下表仅按场景、失败原因筛选；同一事件可有多个原因，计数不等于独立样本数。','meta'));const groups=(summary.unmatched_events?.groups||[]).filter(g=>(!el('scene').value||g.scene_id===el('scene').value)&&failureMatch(['生成记录：'+g.reason]));const table=document.createElement('table');const head=document.createElement('tr');for(const title of ['场景','阶段','原因','事件数'])head.append(text('th',title));table.append(head);for(const g of groups){const line=document.createElement('tr');for(const value of [g.scene_id,g.phase,g.reason,g.events])line.append(text('td',String(value)));table.append(line)}if(groups.length)panel.append(table);else panel.append(text('p','当前条件没有匹配的生成记录。'));}
function render(){
 const inspect=el('evidence').checked;
 el('inspection-controls').className=inspect?'':'hidden';el('notice').className=inspect?'':'hidden';el('root').replaceChildren();generationStats(inspect);
 const selected=D.rows.filter(r=>!inspect||(['scene','task','split','answer'].every((id,i)=>!el(id).value||String(r[['scene_id','task_type','split','private_answer'][i]]??'')===el(id).value)&&(!el('status').value||state(r.sample_id).status===el('status').value)&&(!el('object').value||(r.objects||[]).some(o=>o.category_filter===el('object').value||o.instance_filter===el('object').value))&&failureMatch(r.failure_reasons||[])));shownCount=selected.length;
 const groups=new Map();for(const r of selected){const k=inspect?r.scene_id+' / '+r.goal_id+' / '+(r.pair_id||''):'model';if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r)}
 for(const[k,rs]of groups){
  const group=text('section','','pair');if(inspect)group.append(text('h2',k));const cards=text('div','','cards');
  for(const r of rs){
   const c=text('article','','card');if(inspect)c.append(text('div',r.sample_id+' · '+r.split,'meta'));
   if(r.review_image){const im=document.createElement('img');im.src=r.review_image;im.alt='场景观察图';im.loading='lazy';c.append(im)}else c.append(text('p','图像不可用','flag'));
   c.append(text('p',r.question||'问题缺失','question'));
   if(inspect){
    c.append(text('div','参考答案：'+(r.private_answer??'无'),'badge'));
    c.append(text('p',(r.objects||[]).map(o=>o.role+'：'+o.label+' #'+o.instance_id).join('；'),'meta'));
    if(r.failure_reasons?.length)c.append(text('p',r.failure_reasons.join('；'),'flag'));
    c.append(text('p','助手审阅：'+(r.assistant_review?.status||'pending')+'；候选导出：'+(r.candidate_exported?'是':'否'),'assist'));
    for(const[field,title]of [['overlay_path','投影框（仅检查用）'],['camera_map_path','房间边界、相机与目标（仅检查用）']]){
     if(r.inspection_assets?.[field]){c.append(text('p',title,'meta'));const im=document.createElement('img');im.src=r.inspection_assets[field];im.alt=title;im.loading='lazy';c.append(im)}
    }
    c.append(text('pre',JSON.stringify(r.evidence,null,2),'details'));
    const line=text('div','','reviewline'),sel=document.createElement('select');for(const l of labels)sel.append(option(l));sel.value=state(r.sample_id).status;
    const why=document.createElement('textarea');why.placeholder='人工结论理由（必填）';why.rows=2;why.value=state(r.sample_id).reason;
    sel.addEventListener('change',()=>update(r.sample_id,sel.value,why.value));why.addEventListener('input',()=>update(r.sample_id,sel.value,why.value));line.append(sel,text('span',state(r.sample_id).reviewer||'尚未指定审阅人','meta'));c.append(line,why);
   }
   cards.append(c);
  }
  group.append(cards);el('root').append(group);
 }
 if(!selected.length)el('root').append(text('div','此筛选条件下没有样本。','empty'));counts();
}
for(const id of ['scene','task','object','failure','split','answer','status','evidence'])el(id).addEventListener('change',render);el('save').onclick=()=>{const records=D.rows.map(r=>({sample_id:r.sample_id,scene_id:r.scene_id,goal_id:r.goal_id,...state(r.sample_id)}));const incomplete=records.filter(r=>r.status!=='待审'&&(!r.reviewer.trim()||!r.reason.trim()));if(incomplete.length){alert(`${incomplete.length} 条人工结论缺少审阅人或理由，请补齐后保存。`);return}const body={schema_version:'active_spatial_human_review_v1',manifest_sha256:D.manifest_sha256,exported_at:new Date().toISOString(),records};const url=URL.createObjectURL(new Blob([JSON.stringify(body,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='human_review_'+D.manifest_sha256.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};el('import').onclick=()=>el('file').click();el('file').onchange=async()=>{try{const v=JSON.parse(await el('file').files[0].text());if(v.manifest_sha256!==D.manifest_sha256)throw Error('来源 manifest 不一致');const ids=new Set(D.rows.map(r=>r.sample_id));for(const r of v.records){if(!ids.has(r.sample_id)||!labels.includes(r.status))throw Error('样本或标签无效');if(r.status!=='待审'&&(!r.reviewer?.trim()||!r.reason?.trim()))throw Error('审阅人或理由缺失')}for(const r of v.records)saved[r.sample_id]={status:r.status,reviewer:r.reviewer||'',reason:r.reason||'',reviewed_at:r.reviewed_at||null};cache();render()}catch(e){alert('导入失败：'+e.message)}};render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trajectory", action="append", default=[])
    parser.add_argument("--failures", action="append", default=[],
                        help="Generation failure JSONL; otherwise use sibling all_failures.jsonl or failures.jsonl")
    parser.add_argument("--split-manifest")
    parser.add_argument("--repo-root", default="/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
    parser.add_argument("--model", help="Existing local model directory, read metadata only")
    parser.add_argument("--tokenizer", help="Existing local tokenizer directory or tokenizer.json")
    parser.add_argument("--allow-pending-assistant-review", action="store_true", help="Review-only mode: no candidate or approved training exports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, out = Path(args.manifest).resolve(strict=True), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    source_rows = rows(source)
    if not source_rows:
        raise ValueError("Empty source manifest")
    ids = [row.get("sample_id") for row in source_rows]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing/duplicate sample_id")
    if any(not row.get("scene_id") for row in source_rows):
        raise ValueError("Missing scene_id")
    splits = resolve_splits(source_rows, args.split_manifest, args.seed)
    out.mkdir(parents=True)
    (out / "audit").mkdir()
    (out / "review_assets").mkdir()
    shutil.copyfile(source, out / "audit" / "input_manifest.jsonl")
    linked_failures, generation_failures = load_generation_failures(source, args.failures, set(ids), out)
    counter, token_report = get_token_counter(args)
    datasets = {(level, kind, split): [] for level in ("candidate", "approved") for kind in ("qa", "action") for split in SPLITS}
    audit, reviews, excluded, approved_excluded = [], [], Counter(), Counter()
    hashes, goals, identities, observed_goal_states = {}, {}, set(), {}
    human_provenance_cache = {}
    global_goal_scenes = {}
    for row in source_rows:
        sid, scene, gid = row["sample_id"], str(row["scene_id"]), goal_id(row)
        split, reasons = splits[scene], candidate_reasons(row)
        if gid:
            if gid in global_goal_scenes and global_goal_scenes[gid] != scene:
                raise ValueError("A goal_id is reused across scenes")
            global_goal_scenes[gid] = scene
            identity = (scene, gid)
            if identity in goals and goals[identity]["task_type"] != row.get("task_type"):
                raise ValueError("Same goal_id has conflicting task families")
            goals[identity] = row
        image_audit, review_image, record = {}, None, None
        try:
            path = image_path(row, source.parent)
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                image_audit = {"path": str(path), "width": img.width, "height": img.height, "mode": img.mode, "sha256": sha(path)}
            digest = image_audit["sha256"]
            if digest in hashes and hashes[digest] != split:
                raise ValueError("LEAKAGE: same image content appears across partitions")
            hashes[digest] = split
            goal_state = (scene, gid, digest)
            if goal_state in observed_goal_states:
                if observed_goal_states[goal_state] != row.get("private_answer"):
                    raise ValueError("CONTRADICTION: identical goal and image have different labels")
                reasons.append("duplicate_goal_image")
            else:
                observed_goal_states[goal_state] = row.get("private_answer")
            review_image = "review_assets/" + digest + path.suffix.lower()
            if not (out / review_image).exists():
                shutil.copyfile(path, out / review_image)
            if not reasons:
                record = {"messages": [{"role": "user", "content": "<image>\n" + row["question"]},
                                        {"role": "assistant", "content": row["private_answer"]}], "images": [str(path)]}
                validate_messages(record)
        except Exception as exc:
            if "LEAKAGE:" in str(exc) or "CONTRADICTION:" in str(exc):
                raise
            reasons.append(f"unreadable_image_or_record:{type(exc).__name__}:{exc}")
        if args.allow_pending_assistant_review:
            reasons.append("review_only_mode")
        is_candidate = record is not None and not reasons
        initial_human, human_binding = initial_human_review(row, image_audit.get("sha256"), human_provenance_cache)
        human_passed = initial_human["status"] == "通过"
        passed = human_passed and row.get("qa_sft_allowed") is True
        if is_candidate:
            datasets["candidate", "qa", split].append(record)
            if passed:
                datasets["approved", "qa", split].append(record)
            else:
                approved_excluded["qa_human_review_pending_or_incomplete" if not human_passed else "qa_sft_not_explicitly_allowed"] += 1
        else:
            excluded.update(reasons)
        audit.append({"sample_id": sid, "scene_id": scene, "goal_id": gid, "split": split, "kind": "qa",
                      "candidate": is_candidate, "approved": is_candidate and passed, "exclusion_reasons": reasons, "image": image_audit,
                      "private_answer": row.get("private_answer"), "task_type": row.get("task_type"), "human_review_binding": human_binding})
        inspection_assets, inspection_errors = copy_inspection_assets(row, source.parent, out)
        failure_reasons = sample_failure_reasons(row, reasons, linked_failures[sid])
        failure_reasons += ["检查文件缺失：" + error["field"] for error in inspection_errors]
        reviews.append({"sample_id": sid, "scene_id": scene, "goal_id": gid, "pair_id": row.get("pair_id"),
                        "task_type": row.get("task_type"), "split": split, "question": row.get("question"),
                        "private_answer": row.get("private_answer"), "review_image": review_image,
                        "assistant_review": finite_json(row.get("assistant_review", {})), "candidate_exported": is_candidate,
                        "initial_human_review": initial_human, "human_review_binding": human_binding,
                        "inspection_assets": inspection_assets, "inspection_asset_errors": inspection_errors,
                        "objects": review_objects(row), "failure_reasons": sorted(set(failure_reasons)),
                        "evidence": finite_json({key: value for key, value in row.items() if key not in {"question", "public_observation"}})})
    action_input_counts = Counter()
    if args.trajectory:
        spec = importlib.util.spec_from_file_location("spatial_sft_converter", Path(args.repo_root) / "data_gen/active_spatial_sft/convert_to_qwen25vl_sft.py")
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        for name in args.trajectory:
            path = Path(name).resolve(strict=True)
            for row in rows(path):
                identity = (str(row.get("scene_id")), goal_id(row), str(row.get("id")))
                if identity in identities:
                    raise ValueError("Duplicate source Action identity")
                identities.add(identity)
                action_input_counts["total"] += 1
                record, reasons = validate_action(row, path.parent, goals, splits, converter)
                if args.allow_pending_assistant_review:
                    reasons.append("review_only_mode")
                passed, _ = human_review(row)
                split = splits.get(str(row.get("scene_id")))
                is_candidate = record is not None and not reasons
                if is_candidate:
                    for name in record["images"]:
                        digest = sha(name)
                        if digest in hashes and hashes[digest] != split:
                            raise ValueError("LEAKAGE: same Action image content appears across partitions")
                        hashes[digest] = split
                    datasets["candidate", "action", split].append(record)
                    if passed:
                        datasets["approved", "action", split].append(record)
                    else:
                        approved_excluded["action_human_review_pending_or_incomplete"] += 1
                else:
                    excluded.update("action:" + reason for reason in reasons)
                audit.append({"sample_id": row.get("id"), "scene_id": row.get("scene_id"), "goal_id": goal_id(row),
                              "kind": "action", "split": split, "source": str(path), "candidate": is_candidate,
                              "approved": is_candidate and passed, "exclusion_reasons": reasons})
    dataset_info, count_report = {}, {}
    for (level, kind, split), records in datasets.items():
        stem = ("candidate_" if level == "candidate" else "") + kind + "_" + split
        write_rows(out / (stem + ".jsonl"), records)
        dataset_info[stem] = {"file_name": stem + ".jsonl", "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
                              "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}
        count_report[stem] = supervision_counts(records, counter)
    write_json(out / "dataset_info.json", dataset_info)
    split_doc = {"schema_version": SCHEMA, "seed": args.seed, "split_unit": "scene", **{split + "_scenes": sorted(scene for scene, part in splits.items() if part == split) for split in SPLITS}}
    write_json(out / "split_manifest.json", split_doc)
    write_rows(out / "audit" / "export_records.jsonl", finite_json(audit))
    write_json(out / "human_review.json", {"schema_version": "active_spatial_human_review_v1", "manifest_sha256": sha(source),
        "records": [{"sample_id": row["sample_id"], "scene_id": row["scene_id"], "goal_id": row["goal_id"], **row["initial_human_review"]} for row in reviews]})
    modes = {"QA-only": ("qa",), "Action-only": ("action",), "mixed": ("qa", "action")}
    configs = {}
    for mode, kinds in modes.items():
        missing = [f"{kind}/{split}: no human-approved samples" for kind in kinds for split in ("train", "val") if not datasets["approved", kind, split]]
        if "qa" in kinds:
            missing += [f"qa/{split}: both Yes and No required" for split in ("train", "val")
                        if {record["messages"][-1]["content"] for record in datasets["approved", "qa", split]} != {"Yes", "No"}]
        if not args.model or not (Path(args.model) / "config.json").is_file():
            missing.append("local model configuration not supplied or absent")
        elif json.loads((Path(args.model) / "config.json").read_text()).get("model_type") != "qwen2_5_vl":
            missing.append("recipe is validated only for local Qwen2.5-VL metadata")
        cfg = {"model_name_or_path": str(Path(args.model).resolve()) if args.model else "SET_LOCAL_QWEN2_5_VL_MODEL",
               "stage": "sft", "do_train": False, "do_eval": False, "finetuning_type": "full", "template": "qwen2_vl",
               "freeze_vision_tower": True, "use_fast_tokenizer": True, "image_max_pixels": 768 * 768, "image_min_pixels": 32 * 32,
               "dataset_dir": str(out), "dataset": ",".join(kind + "_train" for kind in kinds),
               "eval_dataset": ",".join(kind + "_val" for kind in kinds), "train_on_prompt": False, "mask_history": False, "cutoff_len": 16384,
               "output_dir": str(out / (mode + "_model")), "overwrite_output_dir": False, "seed": args.seed,
               "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 1e-5,
               "num_train_epochs": 3, "bf16": True, "gradient_checkpointing": True, "flash_attn": "sdpa", "report_to": "none"}
        if len(kinds) == 2:
            cfg.update(mix_strategy="interleave_over", interleave_probs="0.5,0.5")
        (out / (mode + ".yaml")).write_text("# Read-only recipe; no training was run. See export_summary.json for release blockers.\n" + yaml.safe_dump(cfg, sort_keys=False))
        configs[mode] = {"do_train": False, "approved_data_ready": not missing, "blocked_reasons": missing,
                         "sampling_probabilities": {kind: 1 / len(kinds) for kind in kinds},
                         "note": "Sampling probabilities count examples, not supervised tokens or effective loss weight"}
    # Reopen exported records and decode all referred images without model APIs.
    validated_records = 0
    for name in dataset_info:
        for record in rows(out / (name + ".jsonl")):
            validate_messages(record)
            validated_records += 1
    load_report = {"status": "PASS", "records_loaded": validated_records, "training_record_keys": ["messages", "images"],
                   "model_loaded": False, "model_forward": False, "training_run": False, "image_integrity_checked": True,
                   "scene_disjoint": True, "image_sha_cross_split_disjoint": True,
                   "scope": "Read-only JSONL and PIL image verification; no processor, trainer, image-token expansion or loss-mask validation"}
    write_json(out / "readonly_load_validation.json", load_report)
    make_review(out, reviews, sha(source), generation_failures)
    pair_labels = defaultdict(set)
    for row in audit:
        if row["kind"] == "qa" and row["candidate"]:
            pair_labels[(row["scene_id"], row["goal_id"])].add(row["private_answer"])
    summary = {"schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
               "source_manifest": str(source), "source_manifest_sha256": sha(source), "input_rows": len(source_rows),
               "input_field_preservation": "Exact source file copied to audit/input_manifest.jsonl; all metadata excluded from model messages except explicit question text",
               "review_only": args.allow_pending_assistant_review, "counts": count_report, "tokenizer": token_report,
               "review_rows": len(reviews), "review_rows_with_failure_reasons": sum(bool(row["failure_reasons"]) for row in reviews),
               "generation_failure_statistics": generation_failures,
               "candidate_unique_image_sha256": len({row.get("image", {}).get("sha256") for row in audit if row["kind"] == "qa" and row["candidate"]}),
               "candidate_goal_groups": len(pair_labels), "candidate_goals_with_both_labels": sum(labels == {"Yes", "No"} for labels in pair_labels.values()),
               "qa_label_counts": {level: {split: dict(Counter(record["messages"][-1]["content"] for record in datasets[level, "qa", split])) for split in SPLITS} for level in ("candidate", "approved")},
               "candidate_exclusions": dict(excluded), "approved_exclusions": dict(approved_excluded), "action_input_counts": dict(action_input_counts),
               "action_gap": None if any(datasets["candidate", "action", split] for split in SPLITS) else "No current-goal trajectory passed replay, full indoor-position, image, ordered-action and assistant-review checks; Action candidates are empty",
               "configs": configs, "readonly_load_validation": load_report,
               "human_review_status": "verified_user_records_preserved_otherwise_pending",
               "initial_human_review_counts": dict(Counter(row["initial_human_review"]["status"] for row in reviews))}
    write_json(out / "export_summary.json", summary)
    print(json.dumps({"output": str(out), "counts": {key: value["examples"] for key, value in count_report.items()}, "action_gap": summary["action_gap"]}, indent=2))


if __name__ == "__main__":
    main()
