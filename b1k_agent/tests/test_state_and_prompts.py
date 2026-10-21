"""StateView is read-only for modules (red line 2/3); prompts are per module and per version (red line 5)."""

import ast
import hashlib
import os

import numpy as np
import pytest

from agent.interfaces import EpisodeContext, Subtask
from agent.prompts import DEFAULT_PROMPT_ROOT, PromptLoader
from agent.state import EpisodeState, StateView
from conftest import ROOT


def ctx():
    return EpisodeContext(episode_id="e", env_index=0, task_name="t", task_id=0, task_instruction="do it",
                          goal_spec=None, skills=(), max_steps=100, wallclock_budget_s=100.0)


def test_state_view_rejects_writes():
    st = EpisodeState(ctx=ctx())
    st.retries["x"] = 1
    v = StateView(st, "plan")
    with pytest.raises(AttributeError):
        v.current_subtask = Subtask(subtask_id="a", skill="s", instruction="i")
    with pytest.raises(TypeError):
        v.retries["y"] = 2  # mapping proxy
    assert isinstance(v.completed, tuple)
    with pytest.raises(AttributeError):
        v.start_subtask  # write methods are hidden
    with pytest.raises(AttributeError):
        v.scratch  # other modules' scratch is not reachable
    st.last_action = np.zeros(23, np.float32)
    with pytest.raises(ValueError):
        v.last_action[0] = 1.0


def test_scratch_is_private_per_slot():
    st = EpisodeState(ctx=ctx())
    a, b = StateView(st, "plan"), StateView(st, "check")
    a.module_scratch()["k"] = 1
    assert "k" not in b.module_scratch()


def test_prompt_loader_is_bound_to_its_module():
    loader = PromptLoader("plan")
    assert "SYSTEM" in loader.load("v1") and "USER" in loader.load("v1")
    for bad in ("../check/v1", "..", "v1/../../check/v1"):
        with pytest.raises(ValueError):
            loader.load(bad)
    assert list(loader.loaded) == [os.path.join("plan", "v1.txt")]


def test_prompt_texts_are_not_shared_between_modules():
    seen = {}
    for module in sorted(os.listdir(DEFAULT_PROMPT_ROOT)):
        d = os.path.join(DEFAULT_PROMPT_ROOT, module)
        for f in sorted(os.listdir(d)):
            text = open(os.path.join(d, f), encoding="utf-8").read()
            h = hashlib.sha256(text.encode()).hexdigest()
            assert h not in seen or seen[h] == module, f"{module}/{f} duplicates {seen.get(h)}"
            seen[h] = module


def test_modules_never_build_prompt_loaders_themselves():
    """Only the registry binds a PromptLoader to a module's own folder."""
    bad = []
    for root, _, files in os.walk(os.path.join(ROOT, "modules")):
        for f in files:
            if f.endswith(".py"):
                tree = ast.parse(open(os.path.join(root, f), encoding="utf-8").read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "PromptLoader":
                        bad.append(os.path.join(root, f))
    assert not bad
