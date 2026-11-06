"""The P2q -> P2t conversion of group G (matrix v3): the converted text is the RL env's text for a targets format,
and the env judges each label right on the frame it labels.  Reads the matrix's D mixture, the round-11 SFT data and
InteriorGS on the cluster; skipped elsewhere."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from data_gen.sft_mixture import targets_convert as TCV

SFT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft")
P2Q = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/D/P2q_train.jsonl")
GS = Path("/mnt/umm/users/yinbaiqiao/InteriorGS_ply")
FRAME = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))


@pytest.mark.skipif(not (P2Q.exists() and (SFT / "rows.jsonl").exists() and GS.exists()), reason="cluster data only")
def test_converted_rows_are_the_env_text_and_its_truth(tmp_path):
    from data_gen.interiorgs_v10 import rl_checks as RC
    from data_gen.interiorgs_v10 import training_prep as TP

    rows = []
    with open(P2Q, encoding="utf-8") as fh:  # two families with two targets and one with one, multi-turn rows
        seen = set()
        for line in fh:
            r = json.loads(line)
            if r["stratum"] in seen or sum(m["role"] == "assistant" for m in r["messages"]) < 2:
                continue
            seen.add(r["stratum"])
            rows.append(r)
            if len(rows) == 3:
                break
    src = tmp_path / "P2q_train.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    TCV.main(["--rows", str(SFT / "rows.jsonl"), "--frames", str(SFT / "frames.jsonl"), "--in", str(src),
              "--out-dir", str(tmp_path / "out"), "--split", "train"])
    tasks = {r["task_id"]: r for r in TCV.read_jsonl(SFT / "rows.jsonl")}
    cfg = RC.env_config(Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml", "http://127.0.0.1:1/render")
    cfg["gs_root"] = str(GS)
    with mock.patch("vagen.envs.active_spatial.env.ActiveSpatialEnv._render_image", lambda self: FRAME.copy()):
        for fmt in ("targets_pixel", "targets_norm1000"):
            conv = TCV.read_jsonl(tmp_path / "out" / fmt / "P2t_train.jsonl")
            assert [r["source_id"] for r in conv] == [r["source_id"] for r in rows]
            for old, new in zip(rows, conv):
                item = tasks[new["source_id"]]
                env, _ = RC.make_env([item], tmp_path / f"{fmt}_{item['task_id']}.jsonl",
                                     dict(cfg, prompt_format_example=fmt))
                obs, _ = env.reset(seed=0)
                users = [m["content"] for m in new["messages"] if m["role"] == "user"]
                answers = [m["content"] for m in new["messages"] if m["role"] == "assistant"]
                assert new["messages"][0]["content"] == env.system_prompt()
                assert users[0] == obs["obs_str"]
                # the first label is right on the frame it labels, and the action is the old one
                old_action = [m["content"] for m in old["messages"] if m["role"] == "assistant"][0]
                assert answers[0].endswith("</targets>" + old_action) and answers[0].startswith("<targets>")
                _, _, _, info = env.step(answers[0])
                tm = info["metrics"]["turn_metrics"]
                assert (tm["targets_ok"], tm["pointing_score"]) == (True, 1.0), answers[0]
                for u in users[1:]:  # later turns: the format block without the example
                    assert u.endswith("You can take up to 5 actions per step.\n") and "Example:" not in u


@pytest.mark.skipif(not ((SFT / "rows.jsonl").exists() and GS.exists()), reason="cluster data only")
def test_occluders_follow_the_rows_structure_sign(tmp_path):
    """A scene whose rows ask the env for structure Y sign +1 (the pre-launch review: 4 scenes would mirror their
    walls without it): scene_occluders with the rows' sign equals the RL env's occluders, without it it differs."""
    import numpy as np

    from data_gen.interiorgs_v10 import rl_checks as RC
    from data_gen.interiorgs_v10 import training_prep as TP
    from vagen.envs.active_spatial import pointing as PT

    scene = "0491_840935"
    rows = [r for r in TCV.read_jsonl(SFT / "rows.jsonl") if r["scene_id"] == scene]
    sign = PT.scene_structure_sign(rows)
    assert sign == 1.0
    cfg = RC.env_config(Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml", "http://127.0.0.1:1/render")
    cfg["gs_root"] = str(GS)
    with mock.patch("vagen.envs.active_spatial.env.ActiveSpatialEnv._render_image", lambda self: FRAME.copy()):
        env, _ = RC.make_env(rows[:1], tmp_path / "rows.jsonl", cfg)
        env.reset(seed=0)
        theirs = env._visibility_occluders()
    ours, plain = PT.scene_occluders(str(GS), scene, "thick", sign), PT.scene_occluders(str(GS), scene, "thick")

    def arrays(o):
        return {k: np.asarray(v) for k, v in vars(o).items() if isinstance(v, np.ndarray)}

    a, b, c = arrays(ours), arrays(theirs), arrays(plain)
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    assert not all(a[k].shape == c[k].shape and np.array_equal(a[k], c[k]) for k in a)

