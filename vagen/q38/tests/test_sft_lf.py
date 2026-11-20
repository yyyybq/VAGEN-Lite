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
    assert m["verify"]["ok"] and m["counts"] == {"cast_f32_to_bf16": 3, "kept": 2}
    assert (dst / "config.json").is_file() and (dst / "trainer_state.json").is_file() and (dst / "bf16_manifest.json").is_file()
    assert not Path(str(dst) + ".partial").exists()
    got = {}
    for f in to_bf16.weight_files(dst):
        got.update(load_file(str(dst / f)))
    assert set(got) == set(ts)
    assert got["a.weight"].dtype == torch.bfloat16 and torch.equal(got["a.weight"], ts["a.weight"].bfloat16())
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
    assert torch.equal(got["c.A_log"], ts["c.A_log"].bfloat16())


def test_bf16_verify_catches_a_wrong_tensor(tmp_path):
    from safetensors.torch import save_file

    ts = fake_ckpt(tmp_path / "src", 1)
    dst = tmp_path / "dst"
    dst.mkdir()
    bad = {k: (v.bfloat16() if v.dtype == torch.float32 else v) for k, v in ts.items()}
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
