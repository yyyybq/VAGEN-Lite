"""The matrix-v3 registry: Qwen3-VL models load with their family's interpreter, batches as the user planned."""

import yaml

from eval_external.adapters import models as M


def test_registry_includes_matrix_models():
    reg = M.load_registry()
    assert reg["models"]["qwen3vl-8B"]["family"] == "qwen3_vl"
    assert M.python_for("qwen3vl-8B", reg).endswith("envs/eval-external-q3/bin/python")
    assert M.python_for("base-7B", reg).endswith("envs/eval-external/bin/python")
    assert str(M.source_dir("sft_m3c_e4", reg)).endswith("matrix_v3/sft/C/model/checkpoint-1204")


def test_batches_and_prereg():
    m = yaml.safe_load(M.REGISTRY.with_name("registry_matrix_v3.yaml").read_text())["matrix"]
    assert [len(m["batches"][b]) for b in (1, 2, 3)] == [5, 14, 10]
    same = [r["run"] for r in m["runs"] if r["prereg_is_sft"]]
    assert same == ["m3cs1d"]                       # stopped before its first checkpoint: its SFT start
    assert not any(c.startswith("m3e") for b in m["batches"].values() for c in b)  # old E archived
