"""Offline task metadata: task id -> name, task prompt, BDDL goal spec.

Sources (all public task metadata, legal policy inputs):
  * B100_task_misc.csv from the BEHAVIOR data root (2026-challenge-task-instances/metadata/), the
    same file the official evaluator uses to map names to the ``task_id`` it sends
    (OmniGibson/omnigibson/eval/utils/eval_utils.py _load_challenge_task_misc);
  * the NL task description for the strong model: tasks.jsonl ({"task_index", "task"} lines from the
    demo dataset's meta/tasks.jsonl), else the task name (the executor's own prompt for the task is
    taken from the executor server's metadata, see agent/server.py, EpisodeContext.executor_prompt);
  * the BDDL problem file from bddl3/bddl/activity_definitions/<task>/problem0.bddl;
  * fallback snapshots in configs/tasks/ (task_catalog_2026.tsv: id, name, max_steps; tasks_2026.jsonl:
    NL instructions), used only for what the official files do not provide (see configs/tasks/README.md).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from typing import Mapping, Optional

from agent.bddl import load_goal_spec
from agent.interfaces import GoalSpec


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclasses.dataclass
class TaskMetadata:
    id_to_name: Mapping[int, str]
    prompts: Mapping[str, str]
    bddl_root: Optional[str]
    max_steps_by_name: Mapping[str, int] = dataclasses.field(default_factory=dict)
    sources: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def max_steps(self, task_name: str) -> Optional[int]:
        return self.max_steps_by_name.get(task_name)

    def name(self, task_id: int) -> str:
        if task_id not in self.id_to_name:
            raise KeyError(f"unknown task_id {task_id}; metadata has {len(self.id_to_name)} tasks")
        return self.id_to_name[task_id]

    def prompt(self, task_name: str) -> str:
        """NL task description for the strong model (tasks.jsonl), else the task name in words."""
        return self.prompts.get(task_name, task_name.replace("_", " "))

    def goal_spec(self, task_name: str) -> Optional[GoalSpec]:
        if not self.bddl_root:
            return None
        try:
            return load_goal_spec(self.bddl_root, task_name)
        except FileNotFoundError:
            return None


def load_task_misc(csv_path: str) -> dict[int, str]:
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["Task ID"])] = row["Task"]
    return out


def load_catalog(tsv_path: str) -> tuple[dict[int, str], dict[str, int]]:
    """configs/tasks/task_catalog_2026.tsv -> (task_id -> name, name -> max_steps)."""
    ids, steps = {}, {}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids[int(row["task_id"])] = row["task"]
            steps[row["task"]] = int(row["max_steps"])
    return ids, steps


def _resolve(path: Optional[str]) -> str:
    if not path:
        return ""
    path = os.path.expandvars(os.path.expanduser(path))
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_tasks_jsonl(path: str, id_to_name: Mapping[int, str]) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec.get("task_index")
            text = rec.get("task")
            name = rec.get("task_name") or (id_to_name.get(int(idx)) if idx is not None else None)
            if name and text:
                out[name] = text
    return out


def load_task_metadata(cfg: Mapping) -> TaskMetadata:
    """cfg keys: task_misc_csv (official), catalog_tsv (fallback snapshot), tasks_jsonl, bddl_root,
    prompts (inline overrides), id_to_name (inline, tests only). Official files win over snapshots."""
    id_to_name: dict[int, str] = {}
    max_steps: dict[str, int] = {}
    sources: dict[str, str] = {}
    catalog = _resolve(cfg.get("catalog_tsv"))
    if catalog and os.path.exists(catalog):
        cat_ids, max_steps = load_catalog(catalog)
        id_to_name.update(cat_ids)
        sources["id_to_name"] = sources["max_steps"] = catalog
    path = _resolve(cfg.get("task_misc_csv"))
    if path and os.path.exists(path):
        official = load_task_misc(path)
        mismatch = {k: (id_to_name.get(k), v) for k, v in official.items() if k in id_to_name and id_to_name[k] != v}
        if mismatch:
            raise ValueError(f"official {path} disagrees with the catalog snapshot {catalog}: {mismatch}")
        id_to_name.update(official)
        sources["id_to_name"] = path
    if cfg.get("id_to_name"):
        id_to_name.update({int(k): v for k, v in dict(cfg["id_to_name"]).items()})
        sources["id_to_name"] = "inline"
    prompts: dict[str, str] = {}
    jpath = _resolve(cfg.get("tasks_jsonl"))
    if jpath and os.path.exists(jpath):
        prompts.update(load_tasks_jsonl(jpath, id_to_name))
        sources["prompts"] = jpath
    prompts.update(dict(cfg.get("prompts") or {}))
    bddl_root = _resolve(cfg.get("bddl_root")) or None
    return TaskMetadata(id_to_name=id_to_name, prompts=prompts, bddl_root=bddl_root,
                        max_steps_by_name=max_steps, sources=sources)
