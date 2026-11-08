"""WP-TRAIN tests: epochs rule, history windows, blind, reply labels, response-only loss == HF loss, init-adapter
round trip, a tiny end-to-end training (inline == DataLoader), dry run + parity check, merge of a tiny model,
the train probe (train step + load-only) on a tiny model.

Login node, 1 core, no GPU.  Two ways to run it (starVLA has no pytest, DECISIONS A3 forbids installing it):
    cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD
    nice -n 19 $PY_Q3 -m pytest -q -p no:cacheprovider vpd/tests/test_train.py      # eval-external-q3, tf 4.57.6
    nice -n 19 $PY_TRAIN vpd/tests/test_train.py                                     # starVLA, tf 4.57.0 (runner)
The tiny models are random Qwen3VLForConditionalGeneration built from Q3S's config shrunk to 2 layers / hidden 64;
the vocab stays 151936 where the real tokenizer and images go through it (2048 for the merge test).  They are saved
with safetensors.save_file + config.save_pretrained (model.save_pretrained imports deepspeed under starVLA, which
fails without CUDA_HOME).  The merge test needs save_pretrained and runs under PY_Q3 only (merge.py's interpreter).
Scratch under eval_external_assets/vpd/scratch/train/ (env VPD_TRAIN_TEST_DIR overrides), never /tmp; removed at
the end.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_WT = Path(__file__).resolve().parents[2]
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

try:
    import pytest
except ImportError:  # starVLA: the standalone runner at the bottom
    pytest = None

from vpd import paths, starts  # noqa: E402
from vpd.train import n2_train_vpd as T  # noqa: E402
from vpd.walkerbench import history  # noqa: E402


# ------------------------------------------------------------------------------------------------ helpers
class Skip(Exception):
    pass


def skip(msg: str):
    if pytest is not None:
        pytest.skip(msg)
    raise Skip(msg)


@contextlib.contextmanager
def raises(exc, match: str = None):
    try:
        yield
    except exc as e:  # noqa: PERF203
        if match and not re.search(match, str(e)):
            raise AssertionError(f"{e!r} does not match {match!r}") from e
        return
    raise AssertionError(f"{getattr(exc, '__name__', exc)} not raised")


def _torch():
    try:
        import torch
    except ImportError:
        skip("no torch")
    torch.set_num_threads(1)
    return torch


def _transformers():
    _torch()
    try:
        import transformers
    except ImportError:
        skip("no transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        skip("transformers without Qwen3-VL")
    return transformers


def _make_workdir() -> Path:
    base = Path(os.environ.get("VPD_TRAIN_TEST_DIR", paths.scratch("train")))
    d = base / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    return d


if pytest is not None:
    @pytest.fixture(scope="module")
    def workdir():
        d = _make_workdir()
        yield d
        shutil.rmtree(d)  # only what this test module created


def _image(path: Path, seed: int, size=(640, 400)) -> str:
    from PIL import Image

    rnd = random.Random(seed)
    im = Image.frombytes("RGB", size, bytes(rnd.getrandbits(8) for _ in range(size[0] * size[1] * 3)))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    return str(path)


def _reply(t: int) -> str:
    acts = ['MOVE(1)', 'ROTATION(90, 0)', 'MOVE(2)', 'STOP("ARRIVED")']
    return json.dumps({"THOUGHT": "", "ACTION": acts[t % len(acts)]})


def _demo(rid: str, turns: int, img_dir: Path = None, seed: int = 0) -> dict:
    """system + turns x (user text+'<image>', assistant reply); images written when img_dir is given."""
    msgs = [{"role": "system", "content": "You are a test walker. Reply with one JSON action."}]
    imgs = []
    for t in range(turns):
        msgs.append({"role": "user", "content": f"Step {t + 1}. Available Moves: [{{\"id\": 1}}, {{\"id\": 2}}]. "
                                                f"Decide your next action.<image>"})
        msgs.append({"role": "assistant", "content": _reply(t)})
        imgs.append(_image(img_dir / rid / f"turn_{t:02d}.jpg", seed * 100 + t) if img_dir else f"/x/{rid}/{t}.jpg")
    return {"id": rid, "task_type": "nav", "messages": msgs, "images": imgs, "n_turns": turns, "size": [640, 400]}


_DEMOS: dict = {}


def _demo_file(workdir: Path) -> Path:
    """3 demos (2, 3 and 1 turns) with real 640x400 JPEGs, written once per workdir."""
    if workdir not in _DEMOS:
        recs = [_demo("d_a", 2, workdir / "img", 1), _demo("d_b", 3, workdir / "img", 2),
                _demo("d_c", 1, workdir / "img", 3)]
        f = workdir / "demos3.jsonl"
        f.write_text("".join(json.dumps(r) + "\n" for r in recs))
        _DEMOS[workdir] = f
    return _DEMOS[workdir]


def _tiny_config(vocab: int = None):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(starts.resolve("Q3S")["path"])
    tc, vc = cfg.text_config, cfg.vision_config
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16).items():
        setattr(tc, k, v)
    if vocab:
        tc.vocab_size = vocab
    tc.rope_scaling = {**(tc.rope_scaling or {}), "mrope_section": [2, 3, 3]}
    for k, v in dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                     deepstack_visual_indexes=[1]).items():
        setattr(vc, k, v)
    cfg.tie_word_embeddings = False
    return cfg


def _tiny_model(vocab: int = None, dtype=None, seed: int = 0):
    torch = _torch()
    _transformers()
    from transformers import Qwen3VLForConditionalGeneration

    torch.manual_seed(seed)
    return Qwen3VLForConditionalGeneration(_tiny_config(vocab)).to(dtype or torch.float32).eval()


_TINY: dict = {}


def _tiny_saved(workdir: Path, vocab: int = None) -> Path:
    """A tiny bf16 model folder (safetensors.save_file + config.save_pretrained), once per (workdir, vocab)."""
    key = (workdir, vocab)
    if key not in _TINY:
        torch = _torch()
        from safetensors.torch import save_file

        d = workdir / f"tiny_{vocab or 'full'}"
        d.mkdir(parents=True, exist_ok=True)
        model = _tiny_model(vocab, torch.bfloat16)
        model.config.save_pretrained(str(d))
        save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()},
                  str(d / "model.safetensors"), metadata={"format": "pt"})
        _TINY[key] = d
    return _TINY[key]


def _lora():
    import lora  # downstream/train, on sys.path via n2_train_vpd (B1)

    return lora


# ------------------------------------------------------------------------------------- pure python tests
def test_epochs_rule():
    assert T.epochs_for("auto", 50)[0] == 5
    assert T.epochs_for("auto", 200)[0] == 3
    assert T.epochs_for("auto", 20)[0] == 12
    assert T.epochs_for("auto", 483)[0] == 3
    e, rule = T.epochs_for("auto", 25)
    assert e == 10 and rule.startswith("auto") and "N=25" in rule
    assert T.epochs_for("4", 50) == (4, "fixed: 4")
    with raises(ValueError):
        T.epochs_for("0", 10)


def test_epoch_keys_match_original_order():
    """The DataLoader key list is the original per-epoch rng.shuffle order (n2_train.py lines 182-186)."""
    rng = random.Random(1001)
    want = []
    for epoch in range(3):
        order = list(range(7))
        rng.shuffle(order)
        want += [(epoch, k) for k in order]
    assert T.epoch_keys(7, 3, 1001) == want


def test_samples_full_history_and_blind():
    rec = _demo("x", 6)
    (s,) = T.samples(rec)
    assert s["target"] == "all" and s["images"] == rec["images"] and s["messages"] == rec["messages"]
    for k in (0, 2):
        for b in T.samples(rec, k, blind=True):
            assert b["images"] == []
            assert all(T.IMAGE not in m["content"] for m in b["messages"])
    (b,) = T.samples(rec, 0, blind=True)  # only the placeholder goes; the text stays
    assert [m["content"] for m in b["messages"]] == [m["content"].replace(T.IMAGE, "") for m in rec["messages"]]
    assert rec["messages"][1]["content"].endswith(T.IMAGE)  # the record itself is untouched
    bad = dict(rec, images=rec["images"][:-1])
    with raises(ValueError, match="placeholders"):
        T.samples(bad)


def test_history_windows_match_history_module():
    rec = _demo("x", 6)
    msgs = rec["messages"]
    for k in (1, 2, 3):
        got = T.samples(rec, k)
        want = list(history.train_windows(msgs, k))
        assert len(got) == len(want) == 6
        for s, (ctx, i) in zip(got, want):
            assert s["turn"] == i and s["target"] == "last"
            assert s["messages"][:-1] == list(ctx) and s["messages"][-1] is msgs[i]
            assert history.is_clean_window(s["messages"][:-1], k)
            users = [m for m in ctx if m["role"] == "user"]
            assert len(s["images"]) == len(users) <= k + 1
            want_imgs = [rec["images"][(msgs.index(m) - 1) // 2] for m in users]
            assert s["images"] == want_imgs
    # contexts of the eval prompts: the same windows without the reply
    for (i, ctx, imgs), s in zip(T.prefix_contexts(rec, 2), T.samples(rec, 2)):
        assert i == s["turn"] and ctx == s["messages"][:-1] and imgs == s["images"]
    full = T.prefix_contexts(rec, 0)
    assert [len(x[2]) for x in full] == [1, 2, 3, 4, 5, 6]


def test_reply_spans_basic():
    ids = [9, 1, 2, 5, 5, 3, 7, 1, 2, 6, 3]
    assert T.reply_spans(ids, [1, 2], 3) == [(3, 6), (9, 11)]


# ------------------------------------------------------------------------------------ processor tests
def test_encode_labels_and_blind(workdir):
    _transformers()
    proc = starts.load_processor("Q3S")
    rec = json.loads(_demo_file(workdir).read_text().splitlines()[1])  # d_b, 3 turns
    inputs, labels, spans, text = T.encode(proc, T.samples(rec)[0])
    T.check_spans(rec["id"], rec["messages"], spans)
    assert len(spans) == 3
    tok = proc.tokenizer
    ids = inputs["input_ids"][0]
    for (a, b), m in zip(spans, [m for m in rec["messages"] if m["role"] == "assistant"]):
        assert tok.decode(ids[a:b]) == m["content"] + T.END
        assert bool((labels[0, a:b] == ids[a:b]).all())
    assert int((labels != -100).sum()) == sum(b - a for a, b in spans)
    assert inputs["image_grid_thw"].tolist() == [[1, 24, 40]] * 3
    img_id = T.image_token_id(proc)
    assert int((ids == img_id).sum()) == 3 * 240
    # history window: only the last reply is trained
    s = T.samples(rec, 1)[2]
    _, lab, sp, _ = T.encode(proc, s)
    a, b = sp[-1]
    assert int((lab != -100).sum()) == b - a and len(sp) == 2
    # blind: no image tokens, no pixels, same replies
    bi, bl, bs, _ = T.encode(proc, T.samples(rec, blind=True)[0])
    assert bi.get("pixel_values") is None and int((bi["input_ids"][0] == img_id).sum()) == 0
    assert int((bl != -100).sum()) == int((labels != -100).sum()) and len(bs) == 3


def test_response_only_loss_equals_hf_loss(workdir):
    """plan 3.4 item 2: CE on logits_to_keep=idx equals the HF labels= loss within 1e-4 (tiny fp32 model, real
    processor and images)."""
    torch = _torch()
    _transformers()
    proc = starts.load_processor("Q3S")
    model = _tiny_model()
    rec = json.loads(_demo_file(workdir).read_text().splitlines()[0])  # d_a, 2 turns, 2 images
    for s in (T.samples(rec)[0], T.samples(rec, 1)[1]):
        inputs, labels, _, _ = T.encode(proc, s)
        with torch.no_grad():
            hf = float(model(**inputs, labels=labels).loss)
            ours = float(T.response_loss(model, inputs, labels))
            n = int((labels != -100).sum())
            ours_sum = float(T.response_loss(model, inputs, labels, reduction="sum"))
        assert abs(hf - ours) < 1e-4, (hf, ours)
        assert abs(ours_sum / n - ours) < 1e-4


def test_init_adapter_roundtrip(workdir):
    torch = _torch()
    lora = _lora()
    m1 = _tiny_model(2048)
    info = lora.wrap(m1, 16, 32, 0.05, seed=1)
    assert info["wrapped"] == 14  # 2 layers x 7 projections (252 on the 36-layer 8B)
    g = torch.Generator().manual_seed(5)
    for m in m1.modules():
        if isinstance(m, lora.LoRALinear):
            m.lora_B.data = torch.randn(m.lora_B.shape, generator=g)
    d = workdir / "ad_roundtrip"
    d.mkdir()
    state = T.adapter_state(m1)
    torch.save(state, d / "adapter.pt")
    (d / "meta.json").write_text(json.dumps({"rank": 16, "alpha": 32.0, "start": "Q3S"}))

    def fresh(rank=16):
        m = _tiny_model(2048, seed=3)
        lora.wrap(m, rank, 32, 0.05, seed=2)
        for x in m.modules():
            if isinstance(x, lora.LoRALinear):
                x.lora_A.data, x.lora_B.data = x.lora_A.data.float(), x.lora_B.data.float()
        return m

    m2 = fresh()
    got = T.load_init_adapter(m2, d, 16, 32.0)
    assert got["modules"] == 14 and got["sha256"] == T.sha256_file(d / "adapter.pt") and got["start"] == "Q3S"
    s2 = T.adapter_state(m2)
    assert set(s2) == set(state)
    for n in state:
        assert torch.equal(s2[n]["A"], state[n]["A"]) and torch.equal(s2[n]["B"], state[n]["B"])
    assert all(p.dtype == torch.float32 for n, p in m2.named_parameters() if lora.is_lora(n))
    with raises(ValueError, match="shape|A "):
        T.load_init_adapter(fresh(8), d, 8, 32.0)
    with raises(ValueError, match="alpha"):
        T.load_init_adapter(fresh(), d, 16, 16.0)
    d2 = workdir / "ad_missing"
    d2.mkdir()
    torch.save({k: v for k, v in list(state.items())[1:]}, d2 / "adapter.pt")
    with raises(ValueError, match="names differ"):
        T.load_init_adapter(fresh(), d2)


def test_train_end_to_end_tiny(workdir):
    """A tiny CPU training through main(): inline and DataLoader give the same losses and adapter; meta records
    start/code/path/attn/epochs rule/code sha; init-adapter + history-k + skip work."""
    torch = _torch()
    _transformers()
    tiny = _tiny_saved(workdir)
    data = _demo_file(workdir)
    common = ["--start", "Q3S", "--model-path", str(tiny), "--data", str(data), "--device", "cpu", "--threads", "1",
              "--accum", "2"]
    base = common + ["--epochs", "2", "--seed", "1001", "--phase", "t"]
    outs = {}
    for w in (0, 1):
        out = workdir / f"run_w{w}"
        assert T.main(base + ["--out", str(out), "--num-workers", str(w)]) == 0
        outs[w] = out
    logs = {w: [json.loads(x) for x in (o / "log.jsonl").read_text().splitlines()] for w, o in outs.items()}
    assert [r["loss"] for r in logs[0]] == [r["loss"] for r in logs[1]]
    assert len(logs[0]) == 3  # ceil(3 demos x 2 epochs / 2)
    assert {"step", "epoch", "phase", "loss", "lr", "tokens", "seconds", "max_memory_allocated_gib"} <= set(logs[0][0])
    a0, a1 = (torch.load(o / "adapter.pt") for o in outs.values())
    assert set(a0) == set(a1) and len(a0) == 14
    assert all(torch.equal(a0[n]["B"], a1[n]["B"]) for n in a0)
    assert any(float(a0[n]["B"].abs().sum()) > 0 for n in a0)  # it trained
    meta = json.loads((outs[0] / "meta.json").read_text())
    e = starts.resolve("Q3S")
    assert meta["start"] == "Q3S" and meta["code"] == e["code"] and meta["path"] == str(tiny)
    assert meta["registry_path"] == e["path"] and meta["attn"] == "sdpa"
    assert set(meta["attn_sites"].values()) == {"sdpa"}
    assert meta["epochs"] == 2 and meta["epochs_rule"] == "fixed: 2" and meta["lora"]["wrapped"] == 14
    assert len(meta["code_sha256"]) == 64 and meta["max_pixels"] == paths.MAX_PIXELS
    assert meta["optimizer_steps"] == 3 and meta["skipped"] == [] and meta["max_len"] > 0
    # 0W -> next tier: init adapter, K=1 windows (1 image: about 330 tokens, 2 images: about 600), and the demos
    # with a 2-image window (d_a, d_b) skipped by --max-seq-tokens 450
    out2 = workdir / "run_init"
    rc = T.main(common + ["--epochs", "2", "--seed", "7", "--out", str(out2), "--num-workers", "0",
                          "--init-adapter", str(outs[0]), "--history-k", "1", "--max-seq-tokens", "450"])
    m2 = json.loads((out2 / "meta.json").read_text())
    assert rc == 0
    assert m2["init_adapter"]["sha256"] == T.sha256_file(outs[0] / "adapter.pt")
    assert m2["history_k"] == 1 and 0 < m2["max_len"] <= 450
    assert sorted(s["id"] for s in m2["skipped"]) == ["d_a", "d_a", "d_b", "d_b"]
    assert all(s["tokens"] > 450 for s in m2["skipped"])
    assert m2["optimizer_steps"] == 1 and m2["micro_steps"] == 2  # d_c in each of the 2 epochs


def test_dry_run_and_parity(workdir):
    _transformers()
    data = _demo_file(workdir)
    out = workdir / "dry"
    assert T.main(["--start", "Q3S", "--data", str(data), "--out", str(out), "--dry-run", "--limit", "3",
                   "--expect-grid", "24x40"]) == 0
    dry = json.loads((out / "dry_run.json").read_text())
    assert [p["id"] for p in dry["parity"]] == ["d_a", "d_b", "d_c"]
    assert dry["max_pixels"] == paths.MAX_PIXELS and dry["processor_dir"] == str(starts.processor_dir("Q3S"))
    per = {d["id"]: d for d in dry["per_conversation"]}
    assert per["d_b"]["images"] == 3 and per["d_b"]["image_tokens"] == 720 and per["d_b"]["grid_set"] == [[1, 24, 40]]
    assert [len(p["prefixes"]) for p in dry["parity"]] == [2, 3, 1]
    assert dry["parity"][0]["prefixes"][1]["n_images"] == 2
    assert (out / "dry_run_px0.pt").is_file() and (out / "dry_run_sample.txt").is_file()
    from vpd.train import parity_check

    assert parity_check.main(["--dry-run-json", str(out / "dry_run.json")]) == 0
    assert json.loads((out / "parity.json").read_text())["verdict"] == "PASS"
    bad = dict(dry)
    bad["parity"] = [dict(dry["parity"][0], input_ids_sha256="0" * 64)] + dry["parity"][1:]
    (out / "dry_bad.json").write_text(json.dumps(bad))
    assert parity_check.main(["--dry-run-json", str(out / "dry_bad.json"), "--out", str(out / "p_bad.json")]) == 1
    rep = json.loads((out / "p_bad.json").read_text())
    assert rep["verdict"] == "FAIL" and "d_a" in rep["problems"] and rep["fallback"]
    with raises(AssertionError, match="grid"):
        T.main(["--start", "Q3S", "--data", str(data), "--out", str(workdir / "dry2"), "--dry-run", "--limit", "1",
                "--expect-grid", "50x80"])


def test_merge_tiny(workdir):
    """merge.py on a tiny model: merged weights = base + scale*B A (bf16, as merge_adapter), count asserted,
    config without attn keys and bf16 at all levels, greedy generation config, processor files copied; refusals."""
    torch = _torch()
    _transformers()
    if "starVLA" in sys.executable:
        skip("merge.py runs with PY_Q3 (save_pretrained imports deepspeed under starVLA)")
    from vpd.train import merge as M

    lora = _lora()
    tiny = _tiny_saved(workdir, 2048)
    model, _ = starts.load_model("Q3S", device="cpu", attn="sdpa", path=tiny, log=None)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    info = lora.wrap(model, 16, 32, 0.05, seed=1)
    g = torch.Generator().manual_seed(9)
    for m in model.modules():
        if isinstance(m, lora.LoRALinear):
            m.lora_B.data = torch.randn(m.lora_B.shape, generator=g).to(m.lora_B.dtype) * 0.1
    ad = workdir / "ad_merge"
    ad.mkdir()
    state = T.adapter_state(model)
    torch.save(state, ad / "adapter.pt")
    meta = {"start": "Q3S", "rank": 16, "alpha": 32.0, "lora": info}
    (ad / "meta.json").write_text(json.dumps(meta))
    out = workdir / "merged"
    rep = M.merge("Q3S", ad, out=out, model_path=tiny, min_free_gb=0, _allow_shared_out=True)
    assert rep["merged"] == 14 and rep["scale"] == 2.0 and Path(rep["out"]) == out
    from transformers import AutoModelForImageTextToText

    merged = AutoModelForImageTextToText.from_pretrained(str(out), torch_dtype=torch.bfloat16)
    mp = dict(merged.named_parameters())
    for n, ab in state.items():
        want = base[n + ".weight"] + (ab["B"].float() @ ab["A"].float() * 2.0).to(torch.bfloat16)
        assert torch.equal(mp[n + ".weight"], want), n
    untouched = "model.language_model.embed_tokens.weight"
    assert torch.equal(mp[untouched], base[untouched])
    cfg = json.loads((out / "config.json").read_text())
    for holder in (cfg, cfg["text_config"], cfg["vision_config"]):
        assert holder.get("dtype") == "bfloat16"
        assert not any(k in holder for k in M.ATTN_KEYS)
    gen = json.loads((out / "generation_config.json").read_text())
    assert gen["do_sample"] is False and gen["repetition_penalty"] == 1.0
    for f in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja"):
        assert (out / f).is_file() and not (out / f).is_symlink(), f
    assert json.loads((out / "vpd_merge.json").read_text())["merged"] == 14
    # refusals
    with raises(SystemExit, match="refused"):
        M.choose_out(Path("/mnt/umm/users/yinbaiqiao/x"), "r")
    with raises(SystemExit, match="not empty"):
        M.merge("Q3S", ad, out=out, model_path=tiny, min_free_gb=0, _allow_shared_out=True)
    with raises(SystemExit, match="GB free"):
        M.choose_out(workdir / "nowhere", "r", min_free_gb=1e12, allow_shared=True)
    (ad / "meta.json").write_text(json.dumps(dict(meta, start="E2")))
    with raises(SystemExit, match="trained on"):
        M.merge("Q3S", ad, out=workdir / "m2", model_path=tiny, min_free_gb=0, _allow_shared_out=True)
    (ad / "meta.json").write_text(json.dumps(dict(meta, lora=dict(info, wrapped=252))))
    with raises(AssertionError, match="252"):
        M.merge("Q3S", ad, out=workdir / "m3", model_path=tiny, min_free_gb=0, _allow_shared_out=True)


def test_probe_longest():
    from vpd.train import wb_train_probe as P

    recs = [_demo(f"r{i}", t) for i, t in enumerate([3, 9, 1, 9, 5])]
    got = P.longest(recs, 3)
    assert [r["id"] for r in got] == ["r1", "r3", "r4"]


def test_probe_tiny_cpu(workdir):
    """wb_train_probe on the tiny model (CPU): one optimizer step on the 2 longest demos; --load-only on a
    verl-like export (top-level attn eager, float32 sub-configs) comes up sdpa everywhere (B10)."""
    _torch()
    _transformers()
    from vpd.train import wb_train_probe as P

    tiny = _tiny_saved(workdir)
    verl = workdir / "tiny_verl"
    verl.mkdir()
    c = json.loads((tiny / "config.json").read_text())
    c["attn_implementation"] = "eager"
    for sub in ("text_config", "vision_config"):
        c[sub]["dtype"] = "float32"
    (verl / "config.json").write_text(json.dumps(c, indent=1))
    (verl / "model.safetensors").symlink_to(tiny / "model.safetensors")
    out = workdir / "probe"
    rc = P.main(["--start", "Q3S", "--model-path", str(tiny), "--demos", str(_demo_file(workdir)), "--n", "2",
                 "--out", str(out), "--device", "cpu", "--num-workers", "0", "--threads", "1",
                 "--load-only", "Q3RD", "--load-only-path", str(verl)])
    assert rc == 0
    pt = json.loads((out / "probe_train.json").read_text())
    assert pt["optimizer_steps"] == 1 and pt["demos"] == ["d_a", "d_b"] and pt["images"] == [2, 3]
    assert set(pt["attn_sites"].values()) == {"sdpa"} and pt["tokens"] > 0
    pl = json.loads((out / "probe_load_Q3RD.json").read_text())
    assert pl["config_file_attn"] == "eager" and pl["verdict"] == "PASS"
    assert set(pl["attn_sites"].values()) == {"sdpa"} and pl["dtype"] == "bfloat16"


# ------------------------------------------------------------------------------- standalone runner
def _run_standalone() -> int:
    d = _make_workdir()
    fails = 0
    try:
        for name, fn in list(globals().items()):
            if not name.startswith("test_") or not callable(fn):
                continue
            kw = {"workdir": d} if "workdir" in inspect.signature(fn).parameters else {}
            t = time.time()
            try:
                fn(**kw)
                status = "PASS"
            except Skip as e:
                status = f"SKIP ({e})"
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                status, fails = "FAIL", fails + 1
            print(f"{name}: {status} ({time.time() - t:.1f} s)", flush=True)
    finally:
        shutil.rmtree(d)
    print(f"{'FAILED' if fails else 'OK'}: {fails} failure(s)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
