"""CPU tests of the step-4 preparation in downstream/rlinf: the startup hooks (site/sitecustomize.py: post-import
patches, the fixed counter region order, the gradient norms by group) and the 7B FSDP overrides file."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_site():
    """The hook module under another name (the test interpreter keeps its own sitecustomize)."""
    spec = importlib.util.spec_from_file_location("ds_rl_site", ROOT / "rlinf/site/sitecustomize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_post_import_hook_runs_once_after_the_module_is_executed(tmp_path, monkeypatch):
    site = load_site()
    pkg = tmp_path / "dsfakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "target.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    finder = site.install({"dsfakepkg.target": lambda m: seen.append(m.VALUE)})
    try:
        import dsfakepkg.target  # noqa: F401
        importlib.import_module("dsfakepkg.target")
        assert seen == [1]
        already = []
        site.install({"dsfakepkg.target": lambda m: already.append(m.__name__)})  # imported before: patched at once
        assert already == ["dsfakepkg.target"]
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, site.PostImportHook)]
        for name in ("dsfakepkg.target", "dsfakepkg"):
            sys.modules.pop(name, None)


def test_counter_patch_keeps_first_seen_order(tmp_path, monkeypatch):
    site = load_site()
    log = tmp_path / "hooks.jsonl"
    monkeypatch.setenv("DOWNSTREAM_RL_HOOK_LOG", str(log))

    class Counter:
        def get_reset_regions(self, geoms):
            return list(set(geoms))

    module = types.SimpleNamespace(Counter=Counter, __name__="robocasa.models.fixtures.counter")
    site.patch_counter(module)
    geoms = ["c", "a", "b", "a", "c"]
    assert Counter().get_reset_regions(geoms) == ["c", "a", "b"]
    assert Counter.get_reset_regions.downstream_ordered_set
    site.patch_counter(module)  # twice: still one wrapper
    assert Counter().get_reset_regions(geoms) == ["c", "a", "b"]
    assert '"robocasa.models.fixtures.counter"' in log.read_text()


def test_group_grad_norms_split_actor_and_value_head():
    torch = pytest.importorskip("torch")
    site = load_site()
    a = torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.tensor([3.0, 4.0])
    v = torch.nn.Parameter(torch.zeros(4))
    v.grad = torch.ones(4)
    empty = torch.nn.Parameter(torch.zeros(0))
    empty.grad = torch.zeros(0)
    named = [("_fsdp_wrapped_module.starvla_model.layer.weight", a), ("value_head.mlp.weight", v),
             ("no_grad", torch.nn.Parameter(torch.zeros(3))), ("shard_elsewhere", empty)]
    n = site.group_grad_norms(named)
    assert n["actor"] == pytest.approx(5.0) and n["value_head"] == pytest.approx(2.0)
    assert n["total"] == pytest.approx(29 ** 0.5)
    two_ranks = site.group_grad_norms(named, all_reduce=lambda t: t.mul_(2))  # the same shard on two ranks
    assert two_ranks["actor"] == pytest.approx(5.0 * 2 ** 0.5)
    none = site.group_grad_norms([], all_reduce=lambda t: t.add_(1))  # no gradient here, others still reduce
    assert none["actor"] == pytest.approx(1.0) and none["value_head"] == pytest.approx(1.0)


def test_fsdp_manager_patch_logs_then_steps(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    site = load_site()
    grad_log = tmp_path / "grad.jsonl"
    monkeypatch.setenv("DOWNSTREAM_GRAD_LOG", str(grad_log))
    calls = []

    class Manager:
        optimizer_steps = 0

        def __init__(self):
            p = torch.nn.Parameter(torch.zeros(1))
            p.grad = torch.tensor([2.0])
            self.model = torch.nn.Module()
            self.model.value_head = torch.nn.Linear(1, 1, bias=False)
            self.model.value_head.weight.grad = torch.tensor([[1.0]])
            self.model.register_parameter("w", p)

        def optimizer_step(self):
            calls.append("step")
            return 1.0, [0.1]

    module = types.SimpleNamespace(FSDPModelManager=Manager, __name__="rlinf.hybrid_engines.fsdp.fsdp_model_manager")
    site.patch_fsdp_manager(module)
    site.patch_fsdp_manager(module)  # not wrapped twice
    assert Manager().optimizer_step() == (1.0, [0.1]) and calls == ["step"]
    import json

    row = json.loads(grad_log.read_text().splitlines()[0])
    assert row["actor"] == pytest.approx(2.0) and row["value_head"] == pytest.approx(1.0) and row["optimizer_step"] == 1


def test_overrides_file_holds_the_three_smoke_settings():
    lines = [x.strip() for x in (ROOT / "rlinf/fsdp_7b.overrides").read_text().splitlines()
             if x.strip() and not x.strip().startswith("#")]
    assert lines == ["actor.fsdp_config.use_orig_params=true",
                     "++actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="
                     "[Qwen2_5_VLDecoderLayer,Qwen2_5_VLVisionBlock]",
                     "actor.fsdp_config.gradient_checkpointing=true"]
