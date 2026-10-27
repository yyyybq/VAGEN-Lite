"""T2's LoRA (downstream/train/lora.py) on a tiny model with the Qwen2.5-VL module names (CPU)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import lora as L  # noqa: E402


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, 4), nn.Linear(d, 4)
        self.o_proj = nn.Linear(d, d, bias=False)


class Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(d, 12, bias=False), nn.Linear(d, 12, bias=False)
        self.down_proj = nn.Linear(12, d, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn, self.mlp = Attn(d), Mlp(d)

    def forward(self, x):
        a = self.self_attn
        x = x + a.o_proj(a.q_proj(x)) + torch.cat([a.k_proj(x), a.v_proj(x)], -1).sum(-1, keepdim=True)
        return x + self.mlp(x)


class Toy(nn.Module):
    """qwen_vl_interface.model.model.{visual, language_model}, qwen_vl_interface.model.lm_head, action_model."""

    def __init__(self, d=8):
        super().__init__()
        qwen = nn.Module()
        qwen.model = nn.Module()
        qwen.model.model = nn.Module()
        vis = nn.Module()
        vis.blocks = nn.ModuleList([Mlp(d)])  # the vision tower has gate/up/down_proj too: never wrapped
        vis.merger = nn.Linear(d, d)
        lm = nn.Module()
        lm.embed_tokens = nn.Embedding(10, d)
        lm.layers = nn.ModuleList([Layer(d), Layer(d)])
        qwen.model.model.visual, qwen.model.model.language_model = vis, lm
        qwen.model.lm_head = nn.Linear(d, 10, bias=False)
        self.qwen_vl_interface = qwen
        self.action_model = nn.Sequential(nn.Linear(d, 6), nn.ReLU(), nn.Linear(6, 3))

    def forward(self, tokens):
        q = self.qwen_vl_interface.model
        x = q.model.language_model.embed_tokens(tokens)
        x = q.model.visual.merger(q.model.visual.blocks[0](x))
        for layer in q.model.language_model.layers:
            x = layer(x)
        return self.action_model(x), q.lm_head(x)


def toy(seed=0):
    torch.manual_seed(seed)
    return Toy()


def test_wrap_takes_exactly_the_language_model_projections_and_changes_nothing_at_first():
    m, tokens = toy(), torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = [t.clone() for t in m(tokens)]
    info = L.wrap(m, rank=2, alpha=4, dropout=0.0, seed=1)
    assert info["wrapped"] == 2 * 7 and info["per_target"] == {t: 2 for t in L.TARGETS}
    assert isinstance(m.qwen_vl_interface.model.model.language_model.layers[1].mlp.down_proj, L.LoRALinear)
    assert not isinstance(m.qwen_vl_interface.model.model.visual.blocks[0].gate_proj, L.LoRALinear)
    with torch.no_grad():
        after = m(tokens)
    assert all(torch.equal(a, b) for a, b in zip(before, after))  # B = 0: the very same outputs
    lin = m.qwen_vl_interface.model.model.language_model.layers[0].self_attn.k_proj
    # rank x (in + out) per projection: q 16+16, k 16+8, v 16+8, o 16+16, gate 16+24, up 16+24, down 24+16 = 232
    assert info["lora_params"] == 2 * 232
    assert lin.lora_A.shape == (2, 8) and lin.lora_B.shape == (4, 2) and float(lin.lora_B.abs().sum()) == 0
    m2 = toy()
    L.wrap(m2, rank=2, alpha=4, dropout=0.0, seed=1)
    assert torch.equal(m2.qwen_vl_interface.model.model.language_model.layers[0].self_attn.k_proj.lora_A, lin.lora_A)
    with pytest.raises(ValueError):
        L.wrap(nn.Sequential(nn.Linear(2, 2)))


def test_freeze_and_groups_train_only_lora_and_the_action_head():
    m = toy()
    L.wrap(m, rank=2, alpha=4, seed=0)
    info = L.freeze(m)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(L.is_lora(n) or n.startswith("action_model.") for n in trainable)
    assert info["trainable_params"] == sum(p.numel() for n, p in m.named_parameters() if n in trainable)
    groups = L.param_groups(m, 1e-4, 2e-4)
    assert [g["name"] for g in groups] == ["lora", "action_model"] and [g["lr"] for g in groups] == [1e-4, 2e-4]
    assert sum(len(g["params"]) for g in groups) == len(trainable)
    m.qwen_vl_interface.model.lm_head.weight.requires_grad = True
    with pytest.raises(ValueError):
        L.param_groups(m, 1e-4, 1e-4)


def test_merged_state_dict_is_a_plain_checkpoint_with_the_same_outputs():
    m, tokens = toy(), torch.tensor([[4, 5, 6, 7]])
    plain_keys = set(toy().state_dict())
    L.wrap(m, rank=2, alpha=4, dropout=0.1, seed=0)
    for mod in m.modules():
        if isinstance(mod, L.LoRALinear):
            nn.init.normal_(mod.lora_B, std=0.5)  # as if trained
    m.eval()  # no dropout: the server runs the model in eval mode
    with torch.no_grad():
        want = m(tokens)
    full = m.state_dict()  # outside merged(): the full state keeps A and B apart
    assert any(L.is_lora(k) for k in full) and set(full) - plain_keys == {k for k in full if L.is_lora(k)}
    with L.merged(m) as n:
        merged = m.state_dict()
    assert n == 14 and set(merged) == plain_keys
    assert not any(isinstance(mod, L.LoRALinear) and mod.merge_on_save for mod in m.modules())
    plain = toy(seed=123)
    plain.load_state_dict(merged)  # strict: exactly the plain names
    with torch.no_grad():
        got = plain(tokens)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(want, got))
    k = "qwen_vl_interface.model.model.language_model.layers.0.self_attn.q_proj."
    lin = m.qwen_vl_interface.model.model.language_model.layers[0].self_attn.q_proj
    assert torch.allclose(merged[k + "weight"], full[k + "weight"] + lin.scale * full[k + "lora_B"] @ full[k + "lora_A"])
    assert torch.equal(full[k + "weight"], lin.weight.detach())  # the base weight itself is never changed


def test_bf16_weights_stay_bf16_when_merged():
    m = toy()
    m.to(torch.bfloat16)
    L.wrap(m, rank=2, alpha=4, seed=0)
    lin = m.qwen_vl_interface.model.model.language_model.layers[0].mlp.up_proj
    assert lin.lora_A.dtype == torch.bfloat16 and lin.lora_B.dtype == torch.bfloat16
    with L.merged(m):
        sd = m.state_dict()
    assert sd["qwen_vl_interface.model.model.language_model.layers.0.mlp.up_proj.weight"].dtype == torch.bfloat16
