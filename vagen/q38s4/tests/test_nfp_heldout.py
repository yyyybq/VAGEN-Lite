"""q38s4 tools.nfp_heldout: the head read and checked bit for bit, the random control, the tap against training's
NFPTap, the summary by hand."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vagen.q38s4 import nfp as N
from vagen.q38s4.tools import nfp_heldout as H
from vagen.q38s4.tools.strip_head import HEAD_KEYS, INDEX
from vagen.q38s4.tests.tiny import build, cpu_gdn, needs_models, processor, video_batch  # noqa: F401 (fixture)

SMOKE_COPY = Path("/mnt/umm/users/yinbaiqiao/q38/s4/runs/smoke/s1v2_nfp/ckpt_bf16/checkpoint-50")


def _head_tensors(dtype=torch.bfloat16, seed=0, d_in=8, d_out=4, hidden=None):
    torch.manual_seed(seed)
    h = N.NFPHead(d_in, d_out, hidden, dtype=torch.float32)
    with torch.no_grad():
        for p in h.parameters():
            p.add_(torch.randn_like(p))  # biases not zero, so a skipped bias would show
    return {H.PREFIX + k: v.to(dtype).contiguous() for k, v in h.state_dict().items()}


def _write_copy(d: Path, tensors: dict, other: dict = None) -> Path:
    """A two-shard copy: other tensors in the first shard, the given ones in the second, with an index."""
    d.mkdir(parents=True)
    other = other or {"model.x": torch.zeros(3)}
    save_file(dict(other), str(d / "model-00001-of-00002.safetensors"))
    save_file(dict(tensors), str(d / "model-00002-of-00002.safetensors"))
    wm = {**{k: "model-00001-of-00002.safetensors" for k in other},
          **{k: "model-00002-of-00002.safetensors" for k in tensors}}
    (d / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return d


# ------------------------------------------------------------------------------------------------ the head
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_read_and_load_the_head_bit_for_bit(tmp_path, dtype):
    t = _head_tensors(dtype, hidden=6)
    d = _write_copy(tmp_path / "c", t)
    got = H.read_head(d)
    assert sorted(got) == sorted(HEAD_KEYS) and all(H.same_bits(got[k], t[k]) for k in t)
    head = H.head_from_tensors(got, in_dim=8, out_dim=4)
    assert head.fc1.weight.dtype == dtype and head.fc1.weight.shape == (6, 8) and head.fc2.weight.shape == (4, 6)
    x = torch.randn(5, 8, dtype=dtype)
    ref = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(
        x, t[H.PREFIX + "fc1.weight"], t[H.PREFIX + "fc1.bias"])), t[H.PREFIX + "fc2.weight"], t[H.PREFIX + "fc2.bias"])
    assert torch.equal(head(x), ref)


def test_a_head_that_is_not_the_file_stops_the_run(tmp_path):
    """What a loader that skipped the keys would leave: a fresh head.  One flipped value is enough too."""
    t = _head_tensors()
    fresh = N.NFPHead(8, 4, dtype=torch.bfloat16)
    with pytest.raises(SystemExit, match="differ from the file"):
        H.compare_head(fresh, t)
    head = H.head_from_tensors(t)
    H.compare_head(head, t)
    with torch.no_grad():
        head.fc2.bias[0] += 1
    with pytest.raises(SystemExit, match=r"nfp_head\.fc2\.bias"):
        H.compare_head(head, t)
    # same values in another dtype are not the same bits
    with pytest.raises(SystemExit, match="differ"):
        H.compare_head(H.head_from_tensors(t).float(), t)


def test_same_bits_tells_signed_zeros_and_nans_apart():
    assert not H.same_bits(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert H.same_bits(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
    assert not H.same_bits(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))
    assert not H.same_bits(torch.zeros(2, 3), torch.zeros(3, 2))


def test_refusals(tmp_path):
    t = _head_tensors()
    with pytest.raises(SystemExit, match="C-noNFP"):
        H.read_head(_write_copy(tmp_path / "nohead", {"model.y": torch.zeros(2)}))
    with pytest.raises(SystemExit, match="expected exactly"):
        H.read_head(_write_copy(tmp_path / "extra", {**t, H.PREFIX + "fc3.weight": torch.zeros(2)}))
    with pytest.raises(SystemExit, match="expected exactly"):
        H.read_head(_write_copy(tmp_path / "less", {k: v for k, v in t.items() if not k.endswith("fc2.bias")}))
    mixed = dict(t)
    mixed[H.PREFIX + "fc1.bias"] = mixed[H.PREFIX + "fc1.bias"].float()
    with pytest.raises(SystemExit, match="several dtypes"):
        H.head_from_tensors(mixed)
    with pytest.raises(SystemExit, match="vision width"):
        H.head_from_tensors(t, in_dim=8, out_dim=5)
    with pytest.raises(SystemExit, match="neither"):
        H.read_head(tmp_path)


def test_single_file_copy(tmp_path):
    t = _head_tensors()
    (tmp_path / "s").mkdir()
    save_file({**t, "model.x": torch.zeros(1)}, str(tmp_path / "s" / H.SINGLE))
    assert all(H.same_bits(H.read_head(tmp_path / "s")[k], t[k]) for k in t)


def test_loading_info_check():
    H.check_loading({"missing_keys": [], "unexpected_keys": list(reversed(HEAD_KEYS)), "mismatched_keys": []})
    for bad in ({"missing_keys": ["model.a"], "unexpected_keys": list(HEAD_KEYS)},
                {"unexpected_keys": list(HEAD_KEYS) + ["model.b"]},
                {"unexpected_keys": list(HEAD_KEYS)[:3]},
                {"unexpected_keys": list(HEAD_KEYS), "mismatched_keys": [("model.c", (1,), (2,))]}):
        with pytest.raises(SystemExit, match="model load"):
            H.check_loading(bad)


def test_random_head_is_trainings_init_with_seed_0():
    like = H.head_from_tensors(_head_tensors(hidden=6))
    a, b = H.random_head(like, 0), H.random_head(like, 0)
    torch.manual_seed(0)
    ref = N.NFPHead(8, 4, 6, dtype=torch.float32).to(torch.bfloat16)
    for x, y, z in zip(a.state_dict().values(), b.state_dict().values(), ref.state_dict().values()):
        assert H.same_bits(x, y) and H.same_bits(x, z)
    assert a.fc1.weight.dtype == torch.bfloat16 and a.fc1.weight.shape == (6, 8)
    assert float(a.fc1.bias.abs().max()) == 0 and float(a.fc1.weight.float().abs().max()) <= 0.04 + 1e-3
    assert not torch.equal(H.random_head(like, 1).fc1.weight, a.fc1.weight)


# ------------------------------------------------------------------------------------------------ tiny model
@needs_models
def test_hf_copy_with_head_loads_and_the_head_comes_from_the_file(tmp_path, cpu_gdn):
    """save_pretrained of a tiny qwen3_5 with lf_nfp's head (sharded): the load reports exactly the head keys as
    unexpected (check_loading passes), the model has no head, and read_head + head_from_tensors give the saved one."""
    from transformers import Qwen3_5ForConditionalGeneration

    m = build(seed=0, untied=True)
    head = N.attach_head(m)
    with torch.no_grad():
        head.fc2.bias.add_(0.5)
    m.save_pretrained(tmp_path / "c", max_shard_size="200KB", safe_serialization=True)
    m2, info = Qwen3_5ForConditionalGeneration.from_pretrained(tmp_path / "c", output_loading_info=True)
    H.check_loading(info)
    assert not hasattr(m2, "nfp_head")
    got = H.head_from_tensors(H.read_head(tmp_path / "c"), in_dim=m2.config.text_config.hidden_size,
                              out_dim=m2.config.vision_config.hidden_size)
    for k, v in head.state_dict().items():
        assert H.same_bits(got.state_dict()[k], v)


@needs_models
def test_tap_equals_trainings_tap_on_the_same_forward(cpu_gdn):
    """The trained head's and the random head's sums equal nfp.NFPTap's (training's computation), the copy baseline
    too; per-unit means add up to the sums; a QA video is refused."""
    proc = processor()
    model = build(0)
    head = N.attach_head(model)
    rand = H.random_head(head, 0)
    batch = {k: v for k, v in video_batch(proc, (8,)).items() if k != "labels"}  # 4 units x 4 tokens
    want = {}
    for name, h in (("head", head), ("random", rand)):
        tap = N.NFPTap(model, h)
        tap.arm(torch.tensor([True]))
        with torch.no_grad():
            model(**batch)
        want[name] = tap.collect()
        tap.remove()
    tap = H.HeldoutTap(model, {"head": head, "random": rand})
    tap.arm(torch.tensor([True]))
    with torch.no_grad():
        model(**batch, logits_to_keep=1)
    got = tap.out
    tap.disarm()
    assert got["units"] == 4 and got["tokens_per_unit"] == 4 and got["pairs"] == 12 == want["head"].n_pairs
    for name in ("head", "random"):
        assert got[name]["mse_sum"] == pytest.approx(float(want[name].mse_sum), rel=1e-6)
        assert got[name]["cos_sum"] == pytest.approx(float(want[name].cos_sum), rel=1e-6)
    assert got["copy"]["cos_sum"] == pytest.approx(want["head"].copy_cos_sum, rel=1e-6)
    assert got["copy"]["mse_sum"] == pytest.approx(want["head"].copy_mse_sum, rel=1e-6)
    for name in ("head", "random", "copy"):
        assert len(got[name]["unit_cos"]) == 3
        assert sum(got[name]["unit_cos"]) * 4 == pytest.approx(got[name]["cos_sum"], rel=1e-5)
    assert got["head"]["cos_sum"] != got["random"]["cos_sum"]
    tap.arm(torch.tensor([False]))
    with pytest.raises(ValueError, match="one video, an NFP one"), torch.no_grad():
        model(**batch)
    tap.disarm()
    tap.remove()


# ------------------------------------------------------------------------------------------------ summary
def _rec(video, source, pairs, copy, head, rand, mse=(1.0, 2.0, 3.0)):
    return {"video": video, "source": source, "pairs": pairs,
            "copy": {"cos_sum": copy * pairs, "mse_sum": mse[0] * pairs},
            "head": {"cos_sum": head * pairs, "mse_sum": mse[1] * pairs},
            "random": {"cos_sum": rand * pairs, "mse_sum": mse[2] * pairs}}


def test_summary_by_hand():
    recs = [_rec("a1", "a", 10, 0.5, 0.3, 0.9), _rec("a1", "a", 30, 0.5, 0.4, 0.9),
            _rec("b1", "b", 20, 0.2, 0.2, 0.7), _rec("b2", "b", 40, 0.4, 0.2, 0.6)]
    s = H.summarize(recs, draws=500, seed=0)
    P = 100
    copy = (5 + 15 + 4 + 16) / P
    head = (3 + 12 + 4 + 8) / P
    rand = (9 + 27 + 14 + 24) / P
    assert s["pairs"] == P and s["rows"] == 4 and s["videos"] == 3
    assert s["copy"]["cos"] == pytest.approx(copy) and s["head"]["cos"] == pytest.approx(head)
    assert s["r"]["head"] == pytest.approx((copy - head) / copy) and s["r"]["random"] == pytest.approx((copy - rand) / copy)
    assert s["r_mse"]["head"] == pytest.approx(-1.0) and s["head"]["mse"] == pytest.approx(2.0)
    assert s["row_mean_r"]["head"] == pytest.approx(np.mean([0.4, 0.2, 0.0, 0.5]))
    a = s["by_source"]["a"]
    assert a["videos"] == 1 and a["pairs"] == 40 and a["r"]["head"] == pytest.approx((0.5 - 15 / 40) / 0.5)
    assert s["verdict_1"] == {"r": s["r"]["head"], "threshold": 0.2, "learned": True, "rule": s["verdict_1"]["rule"]}
    lo, hi = s["bootstrap"]["head"]["lo"], s["bootstrap"]["head"]["hi"]
    assert lo <= s["r"]["head"] <= hi and s["bootstrap"]["videos"] == 3
    assert H.summarize(recs, draws=500, seed=0)["bootstrap"] == s["bootstrap"]  # seeded


def test_bootstrap_collapses_when_every_video_is_alike_and_verdict_threshold():
    recs = [_rec(f"v{i}", "x", 10, 0.5, 0.42, 0.9) for i in range(6)]
    s = H.summarize(recs, draws=200)
    assert s["bootstrap"]["head"]["lo"] == pytest.approx(s["bootstrap"]["head"]["hi"]) == pytest.approx(0.16)
    assert s["verdict_1"]["learned"] is False
    s = H.summarize([_rec(f"v{i}", "x", 10, 0.5, 0.375, 0.9) for i in range(6)], draws=10)
    assert s["verdict_1"]["r"] == 0.25 and s["verdict_1"]["learned"] is True


def test_source_of():
    assert H.source_of("/mnt/x/data/VSI-590K/scannetppv2/abc/v.mp4") == "scannetppv2"
    assert H.source_of("/mnt/x/data/VSI-590K/adt/v.mp4") == "adt"
    assert H.source_of("/elsewhere/proc/v.mp4") == "proc"


# ------------------------------------------------------------------------------------------------ command line
def test_main_refuses_an_existing_out_and_a_copy_without_head(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        H.main(["--model", str(tmp_path), "--run", str(tmp_path), "--tokenized", str(tmp_path),
                "--out", str(tmp_path / "out")])
    nohead = _write_copy(tmp_path / "nohead", {"model.y": torch.zeros(2)})
    with pytest.raises(SystemExit, match="C-noNFP"):
        H.main(["--model", str(nohead), "--run", str(tmp_path), "--tokenized", str(tmp_path),
                "--out", str(tmp_path / "out2")])
    assert not (tmp_path / "out2").exists()


@pytest.mark.skipif(not (SMOKE_COPY / INDEX).is_file(), reason="smoke bf16 copy not readable")
def test_reads_the_head_of_the_smoke_copy():
    t = H.read_head(SMOKE_COPY)
    assert {k: tuple(v.shape) for k, v in t.items()} == {
        "nfp_head.fc1.weight": (5120, 5120), "nfp_head.fc1.bias": (5120,),
        "nfp_head.fc2.weight": (1152, 5120), "nfp_head.fc2.bias": (1152,)}
    assert {v.dtype for v in t.values()} == {torch.bfloat16}
    head = H.head_from_tensors(t, in_dim=5120, out_dim=1152)
    assert float(head.fc1.bias.float().abs().max()) > 0  # trained 50 steps: biases moved from 0
