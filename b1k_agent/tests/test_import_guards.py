"""Red lines 2 and 4 by static analysis.

* runtime packages (agent, modules, executor, providers, eval) never import diagnostics/ or any
  simulator package (omnigibson, isaacsim, omni, pxr, bddl) -- the agent physically cannot read truth;
* module packages never import each other;
* agent / modules / executor / providers never import eval (official results cannot flow back);
* runtime code does not touch well-known simulator-truth accessors.
"""

import ast
import os

import pytest

from conftest import ROOT

RUNTIME = ("agent", "modules", "executor", "providers", "eval")
SIM_PACKAGES = {"omnigibson", "isaacsim", "omni", "pxr", "bddl", "diagnostics", "carb"}
TRUTH_IDENTIFIERS = {
    "object_scope", "get_goal_option_satisfaction", "get_position_orientation", "_scene_prim",
    "check_goal", "seg_instance", "seg_semantic", "seg_instance_id", "robot_pos",
}
# openpi is the executor's own library (not simulator truth); only the executor server script uses it.
ALLOWED = {os.path.join("executor", "openpi_server", "serve_b1k_prompted.py"): {"openpi"}}


def py_files(pkg):
    base = os.path.join(ROOT, pkg)
    if not os.path.isdir(base):
        return []
    out = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(root, f))
    return out


def imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                yield arg.value


@pytest.mark.parametrize("pkg", RUNTIME)
def test_runtime_never_imports_simulator_or_diagnostics(pkg):
    bad = []
    for path in py_files(pkg):
        rel = os.path.relpath(path, ROOT)
        for mod in imports(path):
            top = mod.split(".")[0]
            if top in SIM_PACKAGES:
                bad.append(f"{rel}: import {mod}")
            if top == "openpi" and "openpi" not in ALLOWED.get(rel, set()):
                bad.append(f"{rel}: import {mod} (openpi only allowed in the executor server script)")
    assert not bad, "\n".join(bad)


def test_modules_do_not_import_each_other():
    bad = []
    base = os.path.join(ROOT, "modules")
    for pkg in sorted(os.listdir(base)):
        if not os.path.isdir(os.path.join(base, pkg)) or pkg.startswith("__"):
            continue
        for path in py_files(os.path.join("modules", pkg)):
            for mod in imports(path):
                parts = mod.split(".")
                if parts[0] == "modules" and len(parts) > 1 and parts[1] != pkg:
                    bad.append(f"{os.path.relpath(path, ROOT)}: import {mod}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("pkg", ("agent", "modules", "executor", "providers"))
def test_agent_side_never_imports_eval(pkg):
    bad = [f"{os.path.relpath(p, ROOT)}: {m}" for p in py_files(pkg) for m in imports(p) if m.split(".")[0] == "eval"]
    assert not bad, "\n".join(bad)


def test_loop_does_not_import_module_packages():
    path = os.path.join(ROOT, "agent", "loop.py")
    assert not [m for m in imports(path) if m.startswith("modules")]


@pytest.mark.parametrize("pkg", RUNTIME)
def test_runtime_has_no_truth_accessors(pkg):
    bad = []
    for path in py_files(pkg):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
            if name in TRUTH_IDENTIFIERS:
                bad.append(f"{os.path.relpath(path, ROOT)}:{node.lineno} {name}")
    assert not bad, "\n".join(bad)
