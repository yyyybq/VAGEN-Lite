"""Isolation of data/ and diagnostics/ by static analysis (complements tests/test_import_guards.py).

* data/ may be imported by runtime modules (e.g. M-memory reading ExperienceRecords), so it follows the
  runtime rule: no simulator package, no diagnostics/, and no eval/ (agent-side code never imports eval);
* diagnostics/ reads simulator truth only in replay_truth.py main(); every other file is pure Python;
* every data/ and diagnostics/ module imports without a simulator.
"""

import ast
import importlib
import os

import pytest

from conftest import ROOT

SIM = {"omnigibson", "isaacsim", "omni", "pxr", "bddl", "carb", "torch", "omegaconf", "hydra"}


def files(pkg):
    base = os.path.join(ROOT, pkg)
    return sorted(os.path.join(base, f) for f in os.listdir(base) if f.endswith(".py"))


def imports(path, tree=None):
    tree = tree or ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node, a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node, node.module


def test_data_follows_the_runtime_rule():
    bad = [f"{os.path.basename(p)}:{n.lineno} {m}" for p in files("data") for n, m in imports(p)
           if m.split(".")[0] in SIM | {"diagnostics", "eval"}]
    assert not bad, bad


def test_diagnostics_simulator_imports_only_in_replay_main():
    bad = []
    for p in files("diagnostics"):
        tree = ast.parse(open(p, encoding="utf-8").read())
        allowed = set()
        if os.path.basename(p) == "replay_truth.py":
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            allowed = {id(n) for n in ast.walk(main)}
        bad += [f"{os.path.basename(p)}:{n.lineno} {m}" for n, m in imports(p, tree)
                if m.split(".")[0] in SIM and id(n) not in allowed]
    assert not bad, bad


@pytest.mark.parametrize("name", ["data", "data.schema", "data.distill", "data.rounds", "diagnostics",
                                  "diagnostics.truth", "diagnostics.join", "diagnostics.label_distill",
                                  "diagnostics.replay_truth"])
def test_importable_without_simulator(name):
    importlib.import_module(name)
