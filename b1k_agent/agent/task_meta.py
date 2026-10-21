"""Offline task metadata: task id -> name, task prompt, BDDL goal spec.

Sources (all public task metadata, legal policy inputs):
  * B100_task_misc.csv from the BEHAVIOR data root (2026-challenge-task-instances/metadata/), the
    same file the official evaluator uses to map names to the ``task_id`` it sends
    (OmniGibson/omnigibson/eval/utils/eval_utils.py _load_challenge_task_misc);
  * the task prompt: the executor's own default prompt when it reports one (so that L0 sends the
    exact text the executor was trained with), else a tasks.jsonl ({"task_index", "task"} lines
    from the demo dataset's meta/tasks.jsonl), else the task name;
  * the BDDL problem file from bddl3/bddl/activity_definitions/<task>/problem0.bddl.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from typing import Mapping, Optional

from agent.bddl import load_goal_spec
from agent.interfaces import GoalSpec


@dataclasses.dataclass
class TaskMetadata:
    id_to_name: Mapping[int, str]
    prompts: Mapping[str, str]
    bddl_root: Optional[str]

    def name(self, task_id: int) -> str:
        if task_id not in self.id_to_name:
            raise KeyError(f"unknown task_id {task_id}; metadata has {len(self.id_to_name)} tasks")
        return self.id_to_name[task_id]

    def prompt(self, task_name: str, executor_default: Optional[str] = None) -> str:
        if executor_default:
            return executor_default
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
    """cfg keys: task_misc_csv, tasks_jsonl (optional), bddl_root (optional), id_to_name (optional
    inline mapping, used only by tests)."""
    id_to_name: dict[int, str] = {}
    if cfg.get("id_to_name"):
        id_to_name.update({int(k): v for k, v in dict(cfg["id_to_name"]).items()})
    path = os.path.expandvars(os.path.expanduser(cfg.get("task_misc_csv") or ""))
    if path and os.path.exists(path):
        id_to_name.update(load_task_misc(path))
    prompts: dict[str, str] = {}
    jpath = os.path.expandvars(os.path.expanduser(cfg.get("tasks_jsonl") or ""))
    if jpath and os.path.exists(jpath):
        prompts.update(load_tasks_jsonl(jpath, id_to_name))
    prompts.update(dict(cfg.get("prompts") or {}))
    bddl_root = os.path.expandvars(os.path.expanduser(cfg.get("bddl_root") or "")) or None
    return TaskMetadata(id_to_name=id_to_name, prompts=prompts, bddl_root=bddl_root)
