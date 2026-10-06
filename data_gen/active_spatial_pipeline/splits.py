"""Dataset identities and disjoint, explicitly indexed Active Spatial splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def content_key(row):
    # Ignore aliases, split names and paraphrases: these are not new episodes.
    fields = ("scene_id", "task_type", "init_camera", "target_object", "target_region", "task_params")
    payload = {key: row.get(key) for key in fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_key(row):
    identity = row.get("source_identity") or {}
    value = identity.get("original_source_key") or row.get("source_key")
    if value is not None:
        return str(value)
    source = row.get("source_jsonl") or row.get("source_path")
    index = row.get("source_row_index", row.get("source_index"))
    if source is not None and index is not None:
        return f"{source}:{index}"
    return None


def assert_disjoint(train, evaluation):
    overlap = {content_key(r) for r in train} & {content_key(r) for r in evaluation}
    sources = {source_key(r) for r in train if source_key(r)} & {source_key(r) for r in evaluation if source_key(r)}
    if overlap or sources:
        raise ValueError(f"train/eval overlap: {len(overlap)} episode fingerprints, {len(sources)} sources")


def select_rows(rows, include=(), exclude=()):
    include, exclude = set(include or []), set(exclude or [])
    return [r for r in rows if (not include or r.get("task_type") in include) and r.get("task_type") not in exclude]


def split_training_rows(rows, *, train_size=None, test_size=19, validation=None,
                        train_include=(), train_exclude=(), val_include=(), val_exclude=(), delta_min=0):
    """Reserve the holdout BEFORE filtering; never refill from training rows."""
    if test_size < 0 or (train_size is not None and train_size < 0):
        raise ValueError("split sizes must be nonnegative")
    if train_size is None:
        train_size = len(rows) if validation is not None else len(rows) - test_size
    if train_size < 0 or train_size > len(rows):
        raise ValueError("train_size exceeds available rows")
    if validation is None:
        if train_size + test_size > len(rows):
            raise ValueError("train_size + test_size exceeds dataset length")
        validation = rows[train_size:train_size + test_size]
    train = select_rows(rows[:train_size], train_include, train_exclude)
    val = select_rows(validation, val_include, val_exclude)
    if delta_min and sum(r.get("task_type") == "delta_control" for r in val) < delta_min:
        raise ValueError("ID delta quota exceeds reserved holdout; supply an independent ID_VAL_JSONL")
    if not train or (test_size > 0 and not val):
        raise ValueError("empty training or requested validation split after filtering")
    assert_disjoint(train, val)
    return train, val


def write_rows(path, rows):
    """Never replace a previously frozen manifest with different content."""
    path = Path(path)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"refusing to replace existing manifest: {path}; use a new experiment directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def object_records(row):
    target = row.get("target_object") or {}
    if isinstance(target, list):
        return [o for o in target if isinstance(o, dict)]
    if isinstance(target, dict):
        if isinstance(target.get("objects"), list):
            return [o for o in target["objects"] if isinstance(o, dict)]
        return [target.get("primary", target)]
    return []


def categories(row):
    result = {str(o["label"]).strip().lower() for o in object_records(row) if o.get("label")}
    if not result:
        result = {s.strip().lower() for s in str(row.get("object_label", "")).split("+") if s.strip()}
    return result


def instances(row):
    result = set()
    for obj in object_records(row):
        identity = obj.get("id")
        if identity is None and obj.get("center") is not None:
            identity = json.dumps({k: obj.get(k) for k in ("label", "center", "bbox_min", "bbox_max")}, sort_keys=True)
        if identity is not None:
            result.add((str(row.get("scene_id")), str(identity)))
    return result


def id_candidates(train, candidates):
    """ID holds out episodes while retaining training scenes, tasks and categories."""
    scenes = {r.get("scene_id") for r in train}
    tasks = {r.get("task_type") for r in train}
    labels = set().union(*(categories(r) for r in train))
    keys = {content_key(r) for r in train}
    sources = {source_key(r) for r in train if source_key(r)}
    return [r for r in candidates if r.get("scene_id") in scenes and r.get("task_type") in tasks
            and bool(categories(r)) and categories(r) <= labels and content_key(r) not in keys
            and (source_key(r) is None or source_key(r) not in sources)]
