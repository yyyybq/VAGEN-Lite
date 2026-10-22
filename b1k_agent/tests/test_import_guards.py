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


def _package_of(path):
    return os.path.relpath(path, ROOT).split(os.sep)[:-1]  # the directory is the package (also for __init__.py)


def imports(path, pkg=None):
    """Absolute names of everything a file imports: plain and relative imports (resolved against the
    file's package), every name of a from-import (``from modules import plan`` -> modules.plan), and
    constant-argument importlib.import_module / import_module / __import__ calls."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    pkg = list(pkg) if pkg is not None else _package_of(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else list(pkg)
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                yield base
            for a in node.names:
                if a.name != "*" and base:
                    yield f"{base}.{a.name}"
        elif isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fname in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield arg.value


def dynamic_imports(path):
    """import_module / __import__ calls whose target is not a constant (cannot be checked statically)."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fname in ("import_module", "__import__") and not (
                    node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                yield node.lineno


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


def test_import_scanner_sees_indirect_forms(tmp_path):
    src = ("from ..check import llm\nfrom .. import check\nfrom modules import recover\n"
           "import importlib\nimportlib.import_module('modules.goal')\n__import__('omnigibson')\n")
    path = tmp_path / "probe.py"
    path.write_text(src, encoding="utf-8")
    found = set(imports(str(path), pkg=("modules", "plan")))
    assert {"modules.check", "modules.check.llm", "modules.recover", "modules.goal", "omnigibson"} <= found


@pytest.mark.parametrize("pkg", ("modules", "executor", "providers"))
def test_no_unverifiable_dynamic_imports(pkg):
    bad = [f"{os.path.relpath(p, ROOT)}:{ln}" for p in py_files(pkg) for ln in dynamic_imports(p)]
    assert not bad, "dynamic imports cannot be checked by these guards:\n" + "\n".join(bad)


def test_modules_do_not_reach_into_state_view_internals():
    """StateView keeps the real EpisodeState in a slot; a module that reads it could write the loop's state."""
    bad = []
    for path in py_files("modules"):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("_state", "_slot", "__getattribute__"):
                bad.append(f"{os.path.relpath(path, ROOT)}:{node.lineno} .{node.attr}")
            elif isinstance(node, ast.Constant) and node.value in ("_state", "_slot"):
                bad.append(f"{os.path.relpath(path, ROOT)}:{node.lineno} '{node.value}'")
    assert not bad, "\n".join(bad)
