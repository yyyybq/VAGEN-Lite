"""WP-CORE tests: start table, starts loaders, WalkerBench sandbox, history window, action regex.

Login node, 1 core, no GPU (PY_Q3 has xgrammar 0.1.25 and transformers 4.57.6):
    cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
      nice -n 19 $PY_Q3 -m pytest -q -p no:cacheprovider vpd/tests/test_core.py
Scratch files go under eval_external_assets/vpd/scratch/core/ (env VPD_CORE_TEST_DIR overrides), never /tmp, and
are removed at the end.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vpd import paths, starts  # noqa: E402
from vpd.walkerbench import action_regex as AR  # noqa: E402
from vpd.walkerbench import history, sandbox  # noqa: E402

EXPECTED_STARTS = {"Q3B", "Q3S", "E2", "Bp", "Q3RS150", "Q3RD", "QRL-s1", "QRL-s2", "QACTL", "Q3SG"}
EXPECTED_CODES = {"Q3B": "qwen3vl-8B", "Q3S": "c0q3_sft_e8", "E2": "sft_m3e2_e10", "Bp": "sft_m3b_e10",
                  "Q3RS150": "c0q3_s1s0_150", "Q3RD": "c0q3_s1d_300", "QRL-s1": None, "QRL-s2": None,
                  "QACTL": None, "Q3SG": "sft_q3genc0_e8"}
ROOT_ONLY = {"Q3B"}  # blobs are root-only: directory existence only
# codes that are not (yet) in eval_external/adapters/registry_matrix_v3.yaml (bf4b2553): Q3SG (genc0, appended
# 2026-11-13, DECISIONS G20) is served under its code from a merged folder and read by path; no eval view
NOT_IN_EVAL_REGISTRY = {"Q3SG"}


def _torch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    return torch


@pytest.fixture(scope="module")
def workdir():
    base = Path(os.environ.get("VPD_CORE_TEST_DIR", paths.scratch("core")))
    d = base / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d)  # only what this test created


# ---------------------------------------------------------------- (a) start table ---------------------------------

def test_starts_table_names_codes_family():
    table = starts.load_table()
    assert set(table) == EXPECTED_STARTS
    for name, e in table.items():
        assert e["family"] == "qwen3"
        assert e["code"] == EXPECTED_CODES[name]
        assert os.path.isabs(e["path"])
        assert starts.resolve(name)["name"] == name
    with pytest.raises(KeyError):
        starts.resolve("nope")


@pytest.mark.parametrize("name", sorted(EXPECTED_STARTS))
def test_starts_paths_exist(name):
    p = Path(starts.resolve(name)["path"])
    assert p.is_dir(), p
    if name in ROOT_ONLY:
        return
    cfg = json.loads((p / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3VLForConditionalGeneration"]
    assert (p / "preprocessor_config.json").exists()


def test_starts_match_eval_registry():
    yaml = pytest.importorskip("yaml")
    reg_file = paths.EVAL_EXTERNAL / "adapters" / "registry_matrix_v3.yaml"
    reg = yaml.safe_load(reg_file.read_text())["models"]
    for name, e in starts.load_table().items():
        if e["code"] is None:
            continue
        if name in NOT_IN_EVAL_REGISTRY:  # once registered, drop it from NOT_IN_EVAL_REGISTRY (then checked below)
            assert e["code"] not in reg, name
            continue
        assert reg[e["code"]]["path"] == e["path"], name
        assert reg[e["code"]]["family"] == "qwen3_vl"


def test_processor_dir_prefers_view():
    for name in ("Q3B", "Q3S", "E2"):  # the pilot starts have views
        assert starts.processor_dir(name) == paths.MODEL_VIEWS / starts.resolve(name)["code"]
    for name in ("QRL-s1", "QACTL"):  # no code: the checkpoint itself
        assert starts.processor_dir(name) == Path(starts.resolve(name)["path"])


def test_norm_attn():
    assert starts.norm_attn("fa2") == "flash_attention_2"
    assert starts.norm_attn("sdpa") == "sdpa"
    with pytest.raises(ValueError):
        starts.norm_attn("flash")


def test_load_processor_max_pixels():
    """B16: the training processor gets max_pixels 1,024,000; 640x400 -> grid [1,24,40] (240 tokens), 1280x800
    unchanged -> [1,50,80] (1000 tokens), larger images are capped."""
    _torch()
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("transformers without Qwen3-VL")
    from PIL import Image

    proc = starts.load_processor("Q3S")
    assert proc.vpd_max_pixels == paths.MAX_PIXELS
    assert starts.processor_max_pixels(proc) == paths.MAX_PIXELS
    ip = proc.image_processor
    grids = {}
    for w, h in ((640, 400), (1280, 800), (2560, 1600)):
        out = ip(images=[Image.new("RGB", (w, h))], return_tensors="pt")
        grids[(w, h)] = out["image_grid_thw"].tolist()[0]
    assert grids[(640, 400)] == [1, 24, 40]
    assert grids[(1280, 800)] == [1, 50, 80]
    t, gh, gw = grids[(2560, 1600)]
    assert gh * gw * 16 * 16 <= paths.MAX_PIXELS


def _tiny_qwen3vl(dst: Path, *, verl_like: bool) -> Path:
    """Save a tiny random Qwen3-VL (float32) shaped like the real config; verl_like adds the RL-export quirks
    (top-level attn_implementation eager, float32 sub-config dtypes)."""
    torch = _torch()
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("transformers without Qwen3-VL")
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(starts.resolve("Q3S")["path"])
    tc, vc = cfg.text_config, cfg.vision_config
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, vocab_size=2048).items():
        setattr(tc, k, v)
    tc.rope_scaling = {**(tc.rope_scaling or {}), "mrope_section": [2, 3, 3]}
    for k, v in dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                     deepstack_visual_indexes=[1]).items():
        setattr(vc, k, v)
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(cfg).to(torch.float32)
    model.save_pretrained(str(dst), safe_serialization=True)
    if verl_like:
        c = json.loads((dst / "config.json").read_text())
        c["attn_implementation"] = "eager"
        for sub in ("text_config", "vision_config"):
            c[sub]["dtype"] = "float32"
        (dst / "config.json").write_text(json.dumps(c, indent=1))
    return dst


def test_load_model_tiny_verl_export(workdir):
    """B10: a verl-style export (eager + float32 sub-configs) loads as sdpa + bf16 everywhere; processor from the
    start's view because the weight folder has no processor files."""
    torch = _torch()
    d = _tiny_qwen3vl(workdir / "tiny_verl", verl_like=True)
    model, proc = starts.load_model("Q3S", device="cpu", attn="sdpa", path=d, log=None)
    info = model.vpd_load_info
    assert info["attn"] == "sdpa"
    assert set(starts.REQUIRED_SITES) <= set(info["attn_sites"])
    assert set(info["attn_sites"].values()) == {"sdpa"}
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert info["processor_dir"] == str(starts.processor_dir("Q3S"))
    assert info["path"] == str(d)


def test_check_attn_catches_mismatch(workdir):
    _torch()
    d = workdir / "tiny_verl"
    if not d.exists():
        _tiny_qwen3vl(d, verl_like=True)
    model, _ = starts.load_model("Q3S", device="cpu", attn="eager", path=d, log=None)
    assert set(starts.attn_report(model).values()) == {"eager"}
    with pytest.raises(AssertionError):
        starts.check_attn(model, "sdpa")
    starts.check_attn(model, "eager")
    # a vision tower left on another implementation is caught although the top level is right
    model.model.visual.config._attn_implementation = "sdpa"
    assert model.config._attn_implementation == "eager"
    with pytest.raises(AssertionError, match="visual"):
        starts.check_attn(model, "eager")


# ---------------------------------------------------------------- (b) sandbox -------------------------------------

def _fake_tasks(workdir: Path, n: int = 3) -> Path:
    src = sorted((paths.REPO_WB / "tasks" / "standard").glob("nav_*.json"))[:n]
    assert src, "no WalkerBench standard tasks"
    d = workdir / "s2fake" / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    for p in src:
        shutil.copy2(p, d / p.name)
    return d


def test_make_sandbox(workdir):
    tasks = _fake_tasks(workdir)
    copy, diff = sandbox.make_sandbox(workdir / "sbx", tasks, render_size=(640, 400))
    assert copy == workdir / "sbx" / "WalkerBench"
    text = (copy / "config" / "settings.py").read_text()
    assert "RENDER_OUTPUT_SIZE: Tuple[int, int] = (640, 400)" in text
    assert "(1280, 800)  # Agent" not in text
    assert f'PANORAMAS_DIR = Path("{paths.PANOS}")' in text
    assert 'TASKS_DIR = BASE_DIR / "tasks" / "vpd_s2fake"' in text
    # exactly the three rewrites, one line each
    minus = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    plus = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert len(minus) == len(plus) == 3, diff
    assert sorted(p.name for p in (copy / "tasks" / "vpd_s2fake").glob("*.json")) == \
        sorted(p.name for p in tasks.glob("*.json"))
    assert not (copy / ".git").exists()
    meta = json.loads((copy / sandbox.MARKER).read_text())
    assert meta["render_size"] == [640, 400] and meta["n_tasks"] == 3
    assert meta["commit"] and meta["commit"].startswith(paths.WB_COMMIT)
    # the source checkout is untouched
    assert "(1280, 800)" in (paths.REPO_WB / "config" / "settings.py").read_text()


def test_make_sandbox_settings_import(workdir):
    """The patched settings module really yields the new size and dirs (needs python-dotenv, as WalkerBench)."""
    pytest.importorskip("dotenv")
    tasks = _fake_tasks(workdir)
    copy, _ = sandbox.make_sandbox(workdir / "sbx_imp", tasks, render_size=(1280, 800),
                                  exclude=sandbox.LEAN_EXCLUDE)
    code = ("import json; from config.settings import settings, TASKS_DIR, PANORAMAS_DIR; "
            "print(json.dumps([list(settings.RENDER_OUTPUT_SIZE), str(TASKS_DIR), str(PANORAMAS_DIR)]))")
    env = {**os.environ, "PYTHONPATH": str(copy), "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-c", code], cwd=copy, env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    size, tdir, pdir = json.loads(out.stdout.strip().splitlines()[-1])
    assert size == [1280, 800]
    assert Path(tdir).resolve() == (copy / "tasks" / "vpd_s2fake").resolve()
    assert pdir == str(paths.PANOS)


def test_make_sandbox_replace_and_refuse(workdir):
    tasks = _fake_tasks(workdir)
    dst = workdir / "sbx_re"
    first, _ = sandbox.make_sandbox(dst, tasks, render_size=(640, 400), exclude=sandbox.LEAN_EXCLUDE)
    for rel in sandbox.LEAN_EXCLUDE:
        assert not (first / rel).exists(), rel
    assert (first / "tasks" / "standard").is_dir() and (first / "data" / "cache.db").exists()
    copy, _ = sandbox.make_sandbox(dst, tasks, render_size=(320, 200), exclude=sandbox.LEAN_EXCLUDE)  # earlier sandbox is replaced
    assert "= (320, 200)" in (copy / "config" / "settings.py").read_text()
    foreign = workdir / "foreign"
    (foreign / "WalkerBench").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        sandbox.make_sandbox(foreign, tasks)
    with pytest.raises(FileNotFoundError):
        sandbox.make_sandbox(workdir / "x", workdir / "no_tasks")


def test_patch_settings_requires_one_hit_each():
    old = (paths.REPO_WB / "config" / "settings.py").read_text()
    new, _ = sandbox.patch_settings(old, "vpd_x", (640, 400), paths.PANOS)
    with pytest.raises(RuntimeError):
        sandbox.patch_settings(new, "vpd_x", (640, 400), paths.PANOS)  # size already rewritten: 0 hits
    with pytest.raises(RuntimeError):
        sandbox.patch_settings(old + "\nTASKS_DIR = 1\n", "vpd_x", (640, 400), paths.PANOS)  # 2 hits
    assert sandbox.parse_size("640x400") == (640, 400)
    with pytest.raises(ValueError):
        sandbox.parse_size("640*400")


# ---------------------------------------------------------------- (c) history -------------------------------------

def _chat(n_msgs: int) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(1, n_msgs):
        msgs.append({"role": "user" if i % 2 else "assistant", "content": f"m{i}"})
    return msgs


def test_window_k2():
    chat = _chat(13)  # s + 6 (u, a)
    assert history.role_pattern(chat) == "s" + "ua" * 6
    ctx = chat[:12]  # step-6 context: ends with the 6th user turn
    w = history.window(ctx, 2)
    assert history.role_pattern(w) == "suauau"
    assert sum(m["role"] == "user" for m in w) == 3
    assert w[0] is chat[0] and w[-1] is chat[11]
    assert history.is_clean_window(w, 2)
    six = chat[:6]  # len == 2k+2: already a full window
    assert history.window(six, 2) is six


def test_window_k0_identity_and_short():
    chat = _chat(13)
    assert history.window(chat, 0) is chat
    short = chat[:4]  # s,u,a,u
    assert history.window(short, 2) is short
    with pytest.raises(ValueError):
        history.window(chat, -1)


def test_train_windows():
    chat = _chat(13)
    out = list(history.train_windows(chat, 2))
    assert [i for _, i in out] == [2, 4, 6, 8, 10, 12]
    pats = [history.role_pattern(c) for c, _ in out]
    assert pats == ["su", "suau", "suauau", "suauau", "suauau", "suauau"]
    assert all(history.is_clean_window(c, 2) for c, _ in out)
    full = list(history.train_windows(chat, 0))
    assert [len(c) for c, _ in full] == [2, 4, 6, 8, 10, 12]
    assert not history.is_clean_window(chat[:6] + [chat[6]], 2)  # ends with assistant
    assert not history.is_clean_window([chat[0]] + chat[2:6], 2)  # orphan assistant first (the stock trim)


# ---------------------------------------------------------------- (d) action regex --------------------------------

def _r(action: str, thought: str = "") -> str:
    return json.dumps({"THOUGHT": thought, "ACTION": action})


@pytest.mark.parametrize("n", range(0, 9))
def test_action_regex_accepts(n):
    rx = re.compile(AR.action_regex(n))
    ok = [f"MOVE({i})" for i in range(1, n + 1)] + ["ROTATION(244.4, 0)", "ROTATION(360, -85)",
                                                    "ROTATION(0, 85)", "ROTATION(12, -3.5)", 'STOP("ARRIVED")']
    for a in ok:
        assert rx.fullmatch(_r(a)), a
    assert AR.reply(AR.STOP_ACTION) == _r('STOP("ARRIVED")')


@pytest.mark.parametrize("n", range(0, 9))
def test_action_regex_rejects(n):
    rx = re.compile(AR.action_regex(n))
    bad = [_r(f"MOVE({n + 1})"), _r("MOVE(0)"), _r("ROTATION(361, 0)"), _r("ROTATION(90, 86)"),
           _r("ROTATION(90, -86)"), _r("ROTATION(90.25, 0)"), _r("ROTATION(90,0)"), _r('STOP("arrived")'),
           _r("MOVE(1)", "go"), _r("MOVE(1)") + " ", _r("MOVE(1)") + "\n", _r("MOVE(1)") + "x",
           json.dumps({"ACTION": "MOVE(1)", "THOUGHT": ""})]
    for s in bad:
        assert not rx.fullmatch(s), s


def test_action_regex_thought():
    rx = re.compile(AR.action_regex(3, tmax=32))
    assert rx.fullmatch(_r("MOVE(2)", "turn left at the corner"))
    assert rx.fullmatch(_r("MOVE(2)"))
    assert not rx.fullmatch(_r("MOVE(2)", "x" * 33))
    assert not rx.fullmatch(_r("MOVE(2)", 'say "hi"'))  # json escapes the quote -> backslash: excluded
    assert not rx.fullmatch(_r("MOVE(4)", "ok"))
    with pytest.raises(ValueError):
        AR.action_regex(-1)


@pytest.mark.parametrize("tmax", [0, 32])
def test_action_regex_xgrammar(tmax):
    xgr = pytest.importorskip("xgrammar")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    for n in range(0, 9):
        g = xgr.Grammar.from_regex(AR.action_regex(n, tmax))
        assert g is not None
