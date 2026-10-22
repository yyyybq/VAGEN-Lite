"""modules/skill and modules/distill are packages, not loop slots: the registry still builds the nine slots, and a
cross-package reference to an impl that does not exist yet fails with a clear error. Every module package has a
README (red line 9)."""

import importlib
import os

import pytest
from conftest import ROOT

from agent.config import load_config
from agent.interfaces import SLOT_PROTOCOLS
from agent.registry import build_modules, load_impl_spec


@pytest.mark.parametrize("pkg", ["skill", "distill"])
def test_non_slot_packages_have_empty_impls(pkg):
    mod = importlib.import_module(f"modules.{pkg}")
    assert mod.IMPLS == {} and pkg not in SLOT_PROTOCOLS
    with pytest.raises(KeyError, match=rf"modules.{pkg} has no impl 'small_vlm' \(available: \[\]\)"):
        load_impl_spec(pkg, "small_vlm")


def test_registry_builds_nine_slots_and_rejects_missing_distill_impl():
    cfg = load_config(os.path.join(ROOT, "configs", "L0_exec_only.yaml"))
    ms = build_modules(cfg["modules"], providers={})
    assert set(ms.slots()) == set(SLOT_PROTOCOLS)
    bad = dict(cfg["modules"], check={"enabled": True, "impl": "distill:small_vlm"})
    with pytest.raises(KeyError, match="modules.distill has no impl"):
        build_modules(bad, providers={})
    with pytest.raises(ValueError, match="unknown module slots"):
        build_modules(dict(cfg["modules"], skill={"enabled": True}), providers={})


@pytest.mark.parametrize("pkg", sorted(list(SLOT_PROTOCOLS) + ["skill", "distill"]))
def test_every_module_package_has_a_readme(pkg):
    text = open(os.path.join(ROOT, "modules", pkg, "README.md"), encoding="utf-8").read()
    for heading in ("## Purpose", "## Status", "## Source paper", "## Interface", "## When the loop calls it",
                    "## Implementations", "## Config", "## Known failure modes", "## Enable condition"):
        assert heading in text, f"modules/{pkg}/README.md lacks {heading!r}"


def test_executor_readme_exists():
    text = open(os.path.join(ROOT, "executor", "README.md"), encoding="utf-8").read()
    assert "## Status" in text and "Galaxea G0.5" in text
