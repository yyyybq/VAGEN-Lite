"""CPU tests of vagen.q38.sft (the 27B SFT path with LLaMA-Factory main): the data conversion, the recipe, the bf16
copy, the launcher's callback, the FSDP wrap classes, and one record through LLaMA-Factory against Qwen3.8's processor
(skipped when the model folder or llamafactory is not available).

Run (login node):
  cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38 && DISABLE_VERSION_CHECK=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD nice -n 19 /mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python -B -m pytest \
    vagen/q38/tests/test_sft_lf.py -q -x -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vagen.q38.sft import lf_data, to_bf16
from vagen.q38.sft.lf_launcher import Q38SFTCallback, bf16_command, bf16_mode

REPO = Path(__file__).resolve().parents[3]
MODEL = Path("/mnt/umm/users/yinbaiqiao/q38/models/Qwen3.8-27B")
ACCEL = REPO / "vagen/q38/configs/sft/accelerate_fsdp_qwen35.yaml"
SYSTEM = "You are an agent.\nActions: move_forward.\n"


def rec(user1=" [Initial Observation]:\n<image>\nTask: go.\n", a1="<action>move_forward|</action>", images=1):
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user1},
                         {"role": "assistant", "content": a1 + "\n"},
                         {"role": "user", "content": "After that:\n<image>\n"},
                         {"role": "assistant", "content": "<action>turn_left|</action>"}],
            "images": [f"/x/{i}.png" for i in range(images + 1)]}


# ------------------------------------------------------------------------------------------------------ conversion
def test_strip_record_strips_every_message_and_counts():
    out, changed = lf_data.strip_record(rec(), "t")
    assert [m["content"] for m in out["messages"]] == [
        SYSTEM.strip(), "[Initial Observation]:\n<image>\nTask: go.", "<action>move_forward|</action>",
        "After that:\n<image>", "<action>turn_left|</action>"]
    assert changed == {"system/trailing": 1, "system/any": 1, "user/leading": 1, "user/trailing": 1, "user/any": 1,
                       "assistant/trailing": 1, "assistant/any": 1}
    assert out["images"] == ["/x/0.png", "/x/1.png"]


@pytest.mark.parametrize("bad, why", [
    (lambda r: r["messages"].pop(), "roles"),
    (lambda r: r["images"].pop(), "placeholders"),
    (lambda r: r["messages"][2].update(content="<think>x</think><action>a|</action>"), "thinking"),
    (lambda r: r["messages"][2].update(content="  \n"), "empty"),
    (lambda r: r["messages"][2].update(content="<image>"), "placeholder"),
])
def test_strip_record_refuses(bad, why):
    r = rec()
    bad(r)
    with pytest.raises(ValueError, match=why):
        lf_data.strip_record(r, "t")


def make_src(tmp: Path, n_train=10, n_val=4) -> Path:
    src = tmp / "src"
    src.mkdir()
    cfg = dict(lf_data.RECIPE_8B, model_name_or_path="/old/8b", dataset_dir=str(src), dataset="trajectory_train",
               eval_dataset="trajectory_val", template="qwen3_vl_nothink", output_dir=str(src / "model"),
               do_eval=True, eval_strategy="steps", eval_steps=200, logging_steps=5, report_to="none")
    (src / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    info = {n: {"file_name": f"{n}.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"}}
            for n in ("trajectory_train", "trajectory_val", "qa_train")}
    (src / "dataset_info.json").write_text(json.dumps(info))
    for name, n in (("trajectory_train", n_train), ("trajectory_val", n_val)):
        (src / f"{name}.jsonl").write_text("".join(json.dumps(rec(a1=f"<action>move_forward|</action> {i}")) + "\n" for i in range(n)))
    (src / "qa_train.jsonl").write_text("")
    return src


def test_build_formal_keeps_the_recipe(tmp_path):
    src = make_src(tmp_path)
    m = lf_data.build(src, tmp_path / "out", "/m/27b")
    out = tmp_path / "out"
    cfg = yaml.safe_load((out / "train.yaml").read_text())
    assert cfg["template"] == "qwen3_8" and cfg["enable_thinking"] is False and cfg["model_name_or_path"] == "/m/27b"
    assert cfg["output_dir"] == str(out / "model") and cfg["dataset_dir"] == str(out)
    assert {k: cfg[k] for k in lf_data.RECIPE_8B} == lf_data.RECIPE_8B
    assert set(m["recipe_changes"]) == {"model_name_or_path", "template", "enable_thinking", "dataset_dir",
                                        "output_dir", "ddp_timeout"}
    assert sorted(json.loads((out / "dataset_info.json").read_text())) == ["trajectory_train", "trajectory_val"]
    assert m["datasets"]["trajectory_train"]["records"] == 10 and m["datasets"]["trajectory_train"]["selected_lines"] == "all"
    first = json.loads((out / "trajectory_train.jsonl").read_text().splitlines()[0])
    assert first["messages"][0]["content"] == SYSTEM.strip()


def test_build_smoke_subset_is_seeded(tmp_path):
    src = make_src(tmp_path)
    a = lf_data.build(src, tmp_path / "a", "/m", "smoke", {"trajectory_train": 4, "trajectory_val": 2}, seed=42)
    b = lf_data.build(src, tmp_path / "b", "/m", "smoke", {"trajectory_train": 4, "trajectory_val": 2}, seed=42)
    assert a["datasets"]["trajectory_train"]["selected_lines"] == b["datasets"]["trajectory_train"]["selected_lines"]
    assert len(a["datasets"]["trajectory_train"]["selected_lines"]) == 4
    cfg = yaml.safe_load((tmp_path / "a" / "train.yaml").read_text())
    assert cfg["num_train_epochs"] == 1 and cfg["logging_steps"] == 1 and cfg["eval_strategy"] == "epoch"


def test_build_refuses_another_recipe_and_a_used_folder(tmp_path):
    src = make_src(tmp_path)
    cfg = yaml.safe_load((src / "train.yaml").read_text())
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x").write_text("x")
    with pytest.raises(SystemExit):
        lf_data.build(src, tmp_path / "used", "/m")
    cfg["learning_rate"] = 2e-5
    (src / "train.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="not the 8B recipe"):
        lf_data.build(src, tmp_path / "o", "/m")


# ------------------------------------------------------------------------------------------------------ bf16 copy
def fake_ckpt(d: Path, shards=2) -> dict:
    from safetensors.torch import save_file

    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    ts = {"a.weight": torch.randn(64, 32, generator=g), "b.weight": torch.randn(128, generator=g) * 1e-30,
          "c.A_log": torch.randn(8, generator=g), "d.idx": torch.arange(5), "e.frozen": torch.randn(16, 4, generator=g).bfloat16()}
    names = list(ts)
    if shards == 1:
        save_file(ts, str(d / "model.safetensors"), metadata={"format": "pt"})
    else:
        wm = {}
        for i, part in enumerate((names[:2], names[2:])):
            f = f"model-{i + 1:05d}-of-00002.safetensors"
            save_file({k: ts[k] for k in part}, str(d / f), metadata={"format": "pt"})
            wm.update({k: f for k in part})
        (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    (d / "trainer_state.json").write_text("{}")
    return ts


@pytest.mark.parametrize("shards", [1, 2])
def test_bf16_copy_sibling(tmp_path, shards):
    from safetensors.torch import load_file

    ts = fake_ckpt(tmp_path / "model" / "checkpoint-7", shards)
    dst = tmp_path / "model_bf16" / "checkpoint-7"
    m = to_bf16.convert(tmp_path / "model" / "checkpoint-7", dst, max_shard_gb=1e-6)
    assert m["verify"]["ok"] and m["counts"] == {"cast_f32_to_bf16": 2, "kept_fp32": 1, "kept": 2}
    assert m["fp32_modules"] == ["A_log"]
    assert (dst / "config.json").is_file() and (dst / "trainer_state.json").is_file() and (dst / "bf16_manifest.json").is_file()
    assert not Path(str(dst) + ".partial").exists()
    got = {}
    for f in to_bf16.weight_files(dst):
        got.update(load_file(str(dst / f)))
    assert set(got) == set(ts)
    assert got["a.weight"].dtype == torch.bfloat16 and torch.equal(got["a.weight"], ts["a.weight"].bfloat16())
    assert got["c.A_log"].dtype == torch.float32 and torch.equal(got["c.A_log"], ts["c.A_log"])
    assert torch.equal(got["d.idx"], ts["d.idx"]) and torch.equal(got["e.frozen"], ts["e.frozen"])
    assert m["bf16_weight_bytes"] < m["source_weight_bytes"]
    assert (tmp_path / "model" / "checkpoint-7" / to_bf16.weight_files(tmp_path / "model" / "checkpoint-7")[0]).exists()


def test_bf16_copy_in_place_replaces_fp32(tmp_path):
    ck = tmp_path / "checkpoint-3"
    ts = fake_ckpt(ck, 2)
    m = to_bf16.convert(ck, None, in_place=True)
    assert m["verify"]["ok"] and m["fp32_deleted"]
    assert not (ck / ".bf16_tmp").exists()
    from safetensors.torch import load_file

    got = {}
    for f in to_bf16.weight_files(ck):
        got.update(load_file(str(ck / f)))
    assert got["c.A_log"].dtype == torch.float32 and torch.equal(got["c.A_log"], ts["c.A_log"])
    assert got["a.weight"].dtype == torch.bfloat16


def test_bf16_copy_no_fp32_modules_casts_everything(tmp_path):
    """fp32_modules=() is the converter before question 4b: every float32 tensor, A_log included, goes to bf16."""
    from safetensors.torch import load_file

    ts = fake_ckpt(tmp_path / "src", 2)
    dst = tmp_path / "dst"
    m = to_bf16.convert(tmp_path / "src", dst, fp32_modules=())
    assert m["verify"]["ok"] and m["counts"] == {"cast_f32_to_bf16": 3, "kept_fp32": 0, "kept": 2}
    got = {}
    for f in to_bf16.weight_files(dst):
        got.update(load_file(str(dst / f)))
    assert torch.equal(got["c.A_log"], ts["c.A_log"].bfloat16())


def test_bf16_verify_catches_a_cast_fp32_module(tmp_path):
    from safetensors.torch import save_file

    ts = fake_ckpt(tmp_path / "src", 1)
    dst = tmp_path / "dst"
    dst.mkdir()
    allcast = {k: (v.bfloat16() if v.dtype == torch.float32 else v) for k, v in ts.items()}
    save_file(allcast, str(dst / "model.safetensors"))
    r = to_bf16.verify(tmp_path / "src", dst)
    assert not r["ok"] and any("c.A_log" in p for p in r["problems"])
    assert to_bf16.verify(tmp_path / "src", dst, fp32_modules=())["ok"]


def test_bf16_cli_fp32_modules(tmp_path):
    fake_ckpt(tmp_path / "src", 1)
    to_bf16.main(["convert", str(tmp_path / "src"), str(tmp_path / "d1")])
    to_bf16.main(["convert", str(tmp_path / "src"), str(tmp_path / "d2"), "--fp32-modules", ""])
    m1 = json.loads((tmp_path / "d1" / "bf16_manifest.json").read_text())
    m2 = json.loads((tmp_path / "d2" / "bf16_manifest.json").read_text())
    assert m1["fp32_modules"] == ["A_log"] and m1["counts"]["kept_fp32"] == 1
    assert m2["fp32_modules"] == [] and m2["counts"]["kept_fp32"] == 0


def test_bf16_verify_catches_a_wrong_tensor(tmp_path):
    from safetensors.torch import save_file

    ts = fake_ckpt(tmp_path / "src", 1)
    dst = tmp_path / "dst"
    dst.mkdir()
    bad = {k: (v.bfloat16() if v.dtype == torch.float32 and "A_log" not in k else v) for k, v in ts.items()}
    bad["a.weight"] = bad["a.weight"] + 1
    save_file(bad, str(dst / "model.safetensors"))
    r = to_bf16.verify(tmp_path / "src", dst)
    assert not r["ok"] and any("a.weight" in p for p in r["problems"])


# ------------------------------------------------------------------------------------------------------ callback
def test_callback_writes_steps_and_starts_no_copy_when_off(tmp_path, monkeypatch):
    monkeypatch.setenv("VAGEN_Q38_SFT_BF16", "off")
    monkeypatch.setenv("VAGEN_Q38_SFT_HOST_MEM_S", "0")
    cb = Q38SFTCallback(metrics_dir=tmp_path / "m", rank=0, local_rank=0)
    args = SimpleNamespace(output_dir=str(tmp_path / "model"))
    (tmp_path / "model" / "checkpoint-2").mkdir(parents=True)
    st = SimpleNamespace(global_step=1, epoch=0.5, max_steps=2)
    cb.on_train_begin(args, st, None)
    for s in (1, 2):
        st.global_step = s
        cb.on_step_begin(args, st, None)
        cb.on_step_end(args, st, None)
    cb.on_save(args, st, None)
    cb.on_train_end(args, st, None)
    assert cb.finish() and cb.copies == []
    lines = [json.loads(l) for l in (tmp_path / "m" / "rank0.jsonl").read_text().splitlines()]
    assert [l["event"] for l in lines] == ["train_begin", "step", "step", "save", "train_end"]
    assert lines[1]["step_s"] is not None and "used_gb" in lines[1]


def test_bf16_mode_and_command(monkeypatch):
    monkeypatch.setenv("VAGEN_Q38_SFT_BF16", "nope")
    with pytest.raises(ValueError):
        bf16_mode()
    c = bf16_command(Path("/r/lf/model/checkpoint-154"), "sibling")
    assert c[-2:] == ["/r/lf/model/checkpoint-154", "/r/lf/model_bf16/checkpoint-154"]
    assert bf16_command(Path("/r/lf/model/checkpoint-154"), "in_place")[-1] == "--in-place"


def test_folder_settings_reach_the_launcher(tmp_path, monkeypatch):
    from vagen.q38.sft.lf_launcher import apply_folder_settings

    src = make_src(tmp_path)
    lf_data.build(src, tmp_path / "out", "/m", bf16_copy="in_place")
    assert json.loads((tmp_path / "out" / "q38_sft.json").read_text()) == {"bf16_copy": "in_place", "host_mem_s": 10}
    for v in ("VAGEN_Q38_SFT_BF16", "VAGEN_Q38_SFT_HOST_MEM_S", "VAGEN_Q38_SFT_METRICS_DIR"):
        monkeypatch.delenv(v, raising=False)
    apply_folder_settings(["launcher", str(tmp_path / "out" / "train.yaml")])
    assert os.environ["VAGEN_Q38_SFT_BF16"] == "in_place" and os.environ["VAGEN_Q38_SFT_HOST_MEM_S"] == "10"
    monkeypatch.setenv("VAGEN_Q38_SFT_BF16", "off")  # a variable already set wins
    apply_folder_settings(["launcher", str(tmp_path / "out" / "train.yaml")])
    assert bf16_mode() == "off"
    with pytest.raises(ValueError):
        lf_data.build(src, tmp_path / "bad", "/m", bf16_copy="keep")


# ------------------------------------------------------------------------------------------------------ FSDP config
def test_accelerate_wrap_classes_exist():
    import torch.nn as nn
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    cfg = yaml.safe_load(ACCEL.read_text())
    classes = cfg["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"].split(",")
    assert classes == ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock", "Qwen3_5VisionPatchEmbed",
                       "Qwen3_5VisionPatchMerger", "Embedding"]
    for c in classes[:-1]:
        assert hasattr(mq, c), c
    assert hasattr(nn, "Embedding")
    assert cfg["num_processes"] == 8 and cfg["fsdp_config"]["fsdp_version"] == 1


# ------------------------------------------------------------------------------------------------------ one record end to end
@pytest.mark.skipif(not (MODEL / "tokenizer.json").is_file(), reason="Qwen3.8-27B not readable")
def test_one_record_lf_equals_processor(tmp_path):
    pytest.importorskip("llamafactory")
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
    from PIL import Image

    from vagen.q38.sft.check_lf_qwen35 import LFEncoder, check_record

    paths = []
    for i in range(2):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (256, 256), (40 * i, 90, 160)).save(p)
        paths.append(str(p))
    src = rec()
    src["images"] = paths
    conv, _ = lf_data.strip_record(src, "t")
    r = check_record(LFEncoder(str(MODEL)), conv, src, prefix=True)
    r.pop("_hf")
    assert r["ok"], r
    assert r["image_tokens_per_image_lf"] == [64, 64] and r["supervised_spans"] == 2


# ------------------------------------------------------------------------------------------------------ OOM fix (2026-11-20)
# The 27B smoke (q38-smoke-27b-sft) ran out of GPU memory in the first micro-batch backward: transformers' Trainer runs
# the micro-batches before a step's last under FSDP1's no_sync unless accelerate's GradientState.plugin_kwargs has
# sync_each_batch.  The folder's train.yaml gets accelerator_config.gradient_accumulation_kwargs.sync_each_batch, the
# launcher sets expandable segments and logs both; these tests check the path from the yaml to what the trainer reads.
SYNC = {"gradient_accumulation_kwargs": {"sync_each_batch": True}}


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", **extra)
    return env


def test_alloc_conf_is_set_before_torch_is_imported():
    """`-m vagen.q38.sft.lf_launcher` imports the vagen packages, then this module: the variable is in the environment
    when torch is first imported, and torch reads it."""
    import subprocess
    import sys

    code = (
        "import importlib.abc, json, os, sys\n"
        "seen = {}\n"
        "class Hook(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'torch' and 'torch' not in seen:\n"
        "            seen['torch'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Hook())\n"
        "import vagen.q38.sft.lf_launcher as L\n"
        "print(json.dumps({'at_torch_import': seen.get('torch', 'never'), 'set': L.ALLOC_CONF_SET,"
        " 'settings': L.alloc_settings()}))\n")
    out = subprocess.run([sys.executable, "-B", "-c", code], env=_clean_env(), capture_output=True, text=True,
                         timeout=600, cwd=str(REPO))
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["at_torch_import"] == "expandable_segments:True" and r["set"] == "expandable_segments:True"
    assert r["settings"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert r["settings"]["torch"] == "expandable_segments:True"  # torch's own reading of the variable


@pytest.mark.parametrize("var", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
def test_alloc_conf_keeps_a_value_already_set(var):
    from vagen.q38.sft.lf_launcher import ALLOC_CONF, set_alloc_conf

    env = {var: "expandable_segments:False"}
    assert set_alloc_conf(env) is None and env == {var: "expandable_segments:False"}
    env = {"PYTORCH_CUDA_ALLOC_CONF": ""}
    assert set_alloc_conf(env) == ALLOC_CONF and env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_only_qwen3_5_runs_the_q38_launcher():
    """The launcher change reaches no other model type: model_types names vagen.q38.sft.lf_launcher for qwen3_5 only,
    and no file outside vagen/q38 (and the model-type tests) names the module."""
    import re

    from vagen.models import model_types as M

    assert [k for k, t in M.TYPES.items() if t.lf_launcher == "vagen.q38.sft.lf_launcher"] == ["qwen3_5"]
    assert {k: t.lf_launcher for k, t in M.TYPES.items() if k != "qwen3_5"} == {
        "qwen2_5_vl": "llamafactory.launcher", "qwen3_vl": "model_matrix.sft.lf_launcher",
        "cambrian_qwen": "model_matrix.sft.lf_launcher_cambrian"}
    pat = re.compile(r"q38[./]sft[./]lf_launcher")
    allowed = {"vagen/models/model_types.py", "model_matrix/tests/test_model_types.py"}
    hits = []
    for top in ("vagen", "model_matrix", "data_gen", "examples", "scripts"):
        for p in (REPO / top).rglob("*"):
            if p.suffix not in (".py", ".sh", ".yaml", ".yml") or not p.is_file():
                continue
            rel = p.relative_to(REPO).as_posix()
            if rel.startswith("vagen/q38/") or rel in allowed:
                continue
            if pat.search(p.read_text(errors="ignore")):
                hits.append(rel)
    assert hits == []


def test_lf_parser_passes_sync_each_batch(tmp_path):
    """lf_data build --set accelerator_config=... -> train.yaml -> LLaMA-Factory's parser -> TrainingArguments'
    AcceleratorConfig (what Trainer.create_accelerator_and_postprocess gives GradientAccumulationPlugin)."""
    pytest.importorskip("llamafactory")
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
    from llamafactory.hparams.parser import _parse_train_args

    src = make_src(tmp_path)
    m = lf_data.build(src, tmp_path / "out", "/m/27b", "smoke", {"trajectory_train": 4, "trajectory_val": 2},
                      sets={"accelerator_config": lf_data.parse_value(
                          "{gradient_accumulation_kwargs: {sync_each_batch: true}}")})
    assert m["recipe_changes"]["accelerator_config"] == [None, SYNC]
    cfg = yaml.safe_load((tmp_path / "out" / "train.yaml").read_text())
    assert cfg["accelerator_config"] == SYNC
    cfg.update(bf16=False, use_cpu=True)  # the login node has no GPU; neither touches accelerator_config
    _, _, targs, _, _ = _parse_train_args(cfg)
    assert targs.accelerator_config.gradient_accumulation_kwargs == {"sync_each_batch": True}


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)

    def forward(self, x, labels=None):
        return {"loss": ((self.lin(x).squeeze(-1) - labels) ** 2).mean()}


@pytest.fixture
def fresh_accelerate_state():
    from accelerate.state import AcceleratorState, GradientState, PartialState

    def reset():
        AcceleratorState._reset_state(True)
        GradientState._reset_state()
        PartialState._reset_state()

    reset()
    yield
    reset()


@pytest.mark.parametrize("accel, want", [(SYNC, True), (None, False)])
def test_callback_logs_what_the_trainer_reads(tmp_path, monkeypatch, fresh_accelerate_state, accel, want):
    """A real transformers Trainer (CPU, 2 micro-batches per step): the callback's train_begin line holds the
    gradient accumulation kwargs and GradientState.plugin_kwargs['sync_each_batch'] the trainer reads, and the first
    step has one substep line (the micro-batch before the step's last)."""
    from transformers import Trainer, TrainingArguments

    monkeypatch.setenv("VAGEN_Q38_SFT_BF16", "off")
    monkeypatch.setenv("VAGEN_Q38_SFT_HOST_MEM_S", "0")
    g = torch.Generator().manual_seed(0)
    data = [{"x": torch.randn(4, generator=g), "labels": torch.tensor(0.5)} for _ in range(8)]
    kw = {} if accel is None else {"accelerator_config": accel}
    args = TrainingArguments(output_dir=str(tmp_path / "model"), per_device_train_batch_size=1,
                             gradient_accumulation_steps=2, max_steps=2, use_cpu=True, report_to=[],
                             save_strategy="no", logging_strategy="no", disable_tqdm=True, **kw)
    cb = Q38SFTCallback(metrics_dir=tmp_path / "m", rank=0, local_rank=0)
    seen = []

    class Spy(Trainer):  # what Trainer._run_epoch reads, at the time it reads it
        def training_step(self, *a, **k):
            seen.append(self.accelerator.gradient_state.plugin_kwargs.get("sync_each_batch", False))
            return super().training_step(*a, **k)

    Spy(model=_Tiny(), args=args, train_dataset=data, callbacks=[cb]).train()
    lines = [json.loads(l) for l in (tmp_path / "m" / "rank0.jsonl").read_text().splitlines()]
    begin = lines[0]
    assert begin["event"] == "train_begin"
    assert begin["grad_acc"]["sync_each_batch"] is want and seen == [want] * 4
    assert begin["grad_acc"]["plugin_kwargs"].get("sync_each_batch", False) is want
    assert begin["grad_acc"]["gradient_accumulation_steps"] == 2
    if accel is not None:
        assert begin["grad_acc"]["args_gradient_accumulation_kwargs"]["sync_each_batch"] is True
    assert begin["cuda_alloc"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert [l["event"] for l in lines] == ["train_begin", "substep", "step", "step", "train_end"]
    sub = lines[1]
    assert sub["step"] == 1 and sub["micro_batch"] == 1 and "allocated_gib" in sub and "reserved_gib" in sub
    assert lines[2]["micro_batches"] == 2 and "allocated_gib" in lines[2] and "reserved_gib" in lines[2]


class _FakeCudaMemory:
    def __init__(self):
        self.calls = []

    def _record_memory_history(self, enabled="all", **kw):
        self.calls.append(("record", enabled, kw.get("stacks"), kw.get("max_entries")))

    def _dump_snapshot(self, path):
        self.calls.append(("dump", Path(path).name))
        Path(path).write_bytes(b"snapshot")


def _fake_cuda(monkeypatch):
    fake = _FakeCudaMemory()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory", fake, raising=False)
    for f in ("max_memory_allocated", "max_memory_reserved", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, f, lambda *a, **k: 3 << 30)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    return fake


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_mem_history_step1_only_when_asked(tmp_path, monkeypatch, flag):
    monkeypatch.setenv("VAGEN_Q38_SFT_BF16", "off")
    monkeypatch.setenv("VAGEN_Q38_SFT_HOST_MEM_S", "0")
    if flag is None:
        monkeypatch.delenv("VAGEN_Q38_SFT_MEM_HISTORY", raising=False)
    else:
        monkeypatch.setenv("VAGEN_Q38_SFT_MEM_HISTORY", flag)
    fake = _fake_cuda(monkeypatch)
    for rank in (0, 1):
        cb = Q38SFTCallback(metrics_dir=tmp_path / f"m{rank}", rank=rank, local_rank=rank)
        args = SimpleNamespace(output_dir=str(tmp_path / "model"))
        st = SimpleNamespace(global_step=0, epoch=0.0, max_steps=3)
        cb.on_train_begin(args, st, None)
        for s in (1, 2, 3):
            cb.on_step_begin(args, st, None)
            cb.on_substep_end(args, st, None)
            st.global_step = s
            cb.on_step_end(args, st, None)
        ev = [json.loads(l) for l in (tmp_path / f"m{rank}" / f"rank{rank}.jsonl").read_text().splitlines()]
        on = flag == "1" and rank == 0
        assert [e["event"] for e in ev] == ["train_begin", "substep", "step"] + (["mem_history"] if on else []) + ["step", "step"]
        assert ev[2]["allocated_gib"] == 3.0 and ev[2]["reserved_gib"] == 3.0 and ev[2]["micro_batches"] == 2
        assert (tmp_path / f"m{rank}" / "mem_history_rank0_step1.pickle").is_file() is on
        if on:
            assert ev[3]["path"].endswith("mem_history_rank0_step1.pickle")
    want = [("record", "all", "python", 100000), ("dump", "mem_history_rank0_step1.pickle"), ("record", None, None, None)]
    assert fake.calls == (want if flag == "1" else [])


def test_oom_line_and_history(tmp_path, monkeypatch):
    monkeypatch.setenv("VAGEN_Q38_SFT_BF16", "off")
    monkeypatch.setenv("VAGEN_Q38_SFT_HOST_MEM_S", "0")
    monkeypatch.setenv("VAGEN_Q38_SFT_MEM_HISTORY", "1")
    fake = _fake_cuda(monkeypatch)
    cb = Q38SFTCallback(metrics_dir=tmp_path / "m", rank=0, local_rank=0)
    args = SimpleNamespace(output_dir=str(tmp_path / "model"))
    cb.on_train_begin(args, SimpleNamespace(global_step=0, epoch=0.0, max_steps=3), None)
    cb.on_exception(ValueError("not an OOM"))
    assert cb.mem_history
    cb.on_exception(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.37 GiB."))
    ev = [json.loads(l) for l in (tmp_path / "m" / "rank0.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ev] == ["train_begin", "oom"]
    assert ev[1]["mem_history"].endswith("mem_history_rank0_oom.pickle") and ev[1]["allocated_gib"] == 3.0
    assert (tmp_path / "m" / "mem_history_rank0_oom.pickle").is_file() and not cb.mem_history
    assert fake.calls[-2:] == [("dump", "mem_history_rank0_oom.pickle"), ("record", None, None, None)]


def test_main_records_an_oom_and_reraises(tmp_path, monkeypatch):
    import sys
    import types

    from vagen.q38.sft import lf_launcher as L

    monkeypatch.setenv("VAGEN_Q38_SFT_BF16", "off")
    monkeypatch.setenv("VAGEN_Q38_SFT_HOST_MEM_S", "0")
    monkeypatch.setenv("VAGEN_Q38_SFT_METRICS_DIR", str(tmp_path / "m"))
    monkeypatch.delenv("VAGEN_Q38_SFT_MEM_HISTORY", raising=False)

    def run_exp(callbacks):
        cb = callbacks[0]
        cb.on_train_begin(SimpleNamespace(output_dir=str(tmp_path / "model")),
                          SimpleNamespace(global_step=0, epoch=0.0, max_steps=1), None)
        raise torch.OutOfMemoryError("CUDA out of memory.")

    tuner = types.ModuleType("llamafactory.train.tuner")
    tuner.run_exp = run_exp
    monkeypatch.setitem(sys.modules, "llamafactory.train.tuner", tuner)
    monkeypatch.setattr(sys, "argv", ["lf_launcher"])
    with pytest.raises(torch.OutOfMemoryError):
        L.main()
    ev = [json.loads(l) for l in (tmp_path / "m" / "rank0.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ev] == ["train_begin", "oom"] and ev[1]["mem_history"] is None


def test_folder_setting_mem_history(tmp_path, monkeypatch):
    from vagen.q38.sft.lf_launcher import apply_folder_settings, _env_on

    (tmp_path / "train.yaml").write_text("{}")
    (tmp_path / "q38_sft.json").write_text(json.dumps({"bf16_copy": "sibling", "host_mem_s": 10, "mem_history": True}))
    for v in ("VAGEN_Q38_SFT_BF16", "VAGEN_Q38_SFT_HOST_MEM_S", "VAGEN_Q38_SFT_MEM_HISTORY"):
        monkeypatch.delenv(v, raising=False)
    apply_folder_settings(["launcher", str(tmp_path / "train.yaml")])
    assert _env_on("VAGEN_Q38_SFT_MEM_HISTORY")
