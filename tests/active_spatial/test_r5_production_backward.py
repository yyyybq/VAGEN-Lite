from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "active_spatial_r5_production_backward.py"


def _module():
    spec = importlib.util.spec_from_file_location("active_spatial_r5_production_backward", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_source_has_no_optimizer_or_checkpoint_write_call() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"step", "save_checkpoint", "save_pretrained", "torch_save"}:
            forbidden.append((node.func.attr, node.lineno))
    assert forbidden == []


def test_optimizer_step_guard_fails_closed() -> None:
    module = _module()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    calls = {"actor": 0}
    module._forbid_optimizer_step(optimizer, "actor", calls)
    with pytest.raises(RuntimeError, match="forbidden optimizer.step"):
        optimizer.step()
    assert calls == {"actor": 1}
    assert parameter.item() == 1.0


def test_module_classification_uses_original_fsdp_fqns() -> None:
    module = _module()
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter._fqns = ["model.language_model.layers.7.self_attn.q_proj.weight"]
    assert module._module_key("_flat_param", parameter) == "language.layer.07"
    parameter._fqns = ["pretrained_model.model.visual.blocks.3.mlp.up_proj.weight"]
    assert module._module_key("_flat_param", parameter) == "vision.block.03"


def test_recursive_device_copy_preserves_structure() -> None:
    module = _module()
    value = {"pixels": [torch.tensor([1.0])], "grid": (torch.tensor([2]), None)}
    copied = module._recursive_to(value, torch.device("cpu"))
    assert copied["pixels"][0].device.type == "cpu"
    assert copied["grid"][0].tolist() == [2]
    assert copied["grid"][1] is None
