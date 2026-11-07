"""reward.verdict_p3a2 on synthetic answers (temp folders only)."""

import json

import pytest

from reward import verdict_p3a2 as v
from reward.tests.test_compare_qa import _easi_model


def _x0(tmp_path, right):
    qs = [{"id": f"x{i}", "kind": "left_right", "options": ["l", "r"], "answer": "A"} for i in range(100)]
    (tmp_path / "x0.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    for diag, models in right.items():
        d = tmp_path / diag / "downstream_qa"
        d.mkdir(parents=True, exist_ok=True)
        for m, k in models.items():
            (d / f"x0_{m}.jsonl").write_text("".join(json.dumps({"id": q["id"], "reply": "A" if i < k else "B"}) + "\n"
                                                    for i, q in enumerate(qs)))
    return tmp_path / "x0.jsonl"


def _emb(tmp_path, scores):
    for m, s in scores.items():
        _easi_model(tmp_path / "easi", m, {"embspatial": s})


def test_x0_passes_only_when_all_positive_and_mean_ge_3(tmp_path):
    x0 = _x0(tmp_path, {"a": {"rw_g_250": 53, "rw_s0_150": 50, "rw_s2s0_300": 50},
                        "a2": {"rw_s2g_150": 54, "rw_s3g_300": 53, "rw_s3s0_300": 50}})
    base = [1.0] * 30 + [0.0] * 70
    _emb(tmp_path, {"rw_s2g_150": [1.0] * 40 + [0.0] * 60, "rw_s2s0_300": base,
                    "rw_s3g_300": [1.0] * 31 + [0.0] * 69, "rw_s3s0_300": base})
    res = v.run(x0, [tmp_path / "a", tmp_path / "a2"], tmp_path / "easi")
    assert res["x0"]["all_positive"] and abs(res["x0"]["mean"] - 10 / 3) < 1e-9 and res["x0"]["passes"]
    e = res["embspatial"]["pairs"]
    assert e["seed 20261021"]["p"] < 0.05 and e["seed 20261023"]["p"] > 0.05 and not res["embspatial"]["holds"]
    v.write_md(res, tmp_path / "o.md")
    assert "没过" not in (tmp_path / "o.md").read_text().split("## 2.2")[0]


def test_x0_one_negative_fails(tmp_path):
    x0 = _x0(tmp_path, {"a": {"rw_g_250": 60, "rw_s0_150": 50, "rw_s2s0_300": 50},
                        "a2": {"rw_s2g_150": 60, "rw_s3g_300": 49, "rw_s3s0_300": 50}})
    base = [1.0] * 30 + [0.0] * 70
    _emb(tmp_path, {m: base for m in ("rw_s2g_150", "rw_s2s0_300", "rw_s3g_300", "rw_s3s0_300")})
    res = v.run(x0, [tmp_path / "a", tmp_path / "a2"], tmp_path / "easi")
    assert res["x0"]["mean"] > 3 and not res["x0"]["passes"]


def test_answers_must_be_in_exactly_one_diag(tmp_path):
    x0 = _x0(tmp_path, {"a": {"rw_g_250": 50, "rw_s0_150": 50, "rw_s2s0_300": 50, "rw_s2g_150": 50},
                        "a2": {"rw_s2g_150": 50, "rw_s3g_300": 50, "rw_s3s0_300": 50}})
    with pytest.raises(SystemExit):
        v.find_answers([tmp_path / "a", tmp_path / "a2"], "rw_s2g_150")
