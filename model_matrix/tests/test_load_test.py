"""Tests of the model line's step-1 load test helpers (model_matrix/step1/load_test.py), CPU only."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("load_test", HERE / "step1" / "load_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_case_keeps_the_image_where_the_user_turn_has_it(tmp_path):
    from PIL import Image

    L = _load()
    img = tmp_path / "s0.png"
    Image.new("RGB", (256, 256), (10, 20, 30)).save(img)
    rec = {"id": "r1", "conversations": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "[Initial Observation]:\n<image>\nTask: go."},
        {"role": "assistant", "content": "<action>move_right|</action>"}],
        "conversation_image_paths": [str(img)]}
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"id": "other", "conversations": []}) + "\n" + json.dumps(rec) + "\n")
    got, messages, image, reference = L.load_case(str(path), "r1")
    assert got["id"] == "r1" and reference == "<action>move_right|</action>" and image.size == (256, 256)
    assert messages[0] == {"role": "system", "content": "SYS"}
    # the same split as the agent loop's convert_obs_to_content: text, one image, text
    assert messages[1]["content"] == [{"type": "text", "text": "[Initial Observation]:\n"}, {"type": "image"},
                                      {"type": "text", "text": "\nTask: go."}]
    with pytest.raises(SystemExit):
        L.load_case(str(path), "missing")


def test_allowed_actions_are_the_v10_envs():
    pytest.importorskip("vagen.envs.active_spatial.env")
    L = _load()
    names = L.allowed_actions()
    assert {"move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"} <= names
    assert "look_down" not in names and "look_up" not in names


def test_strict_read_is_the_envs_reading():
    pytest.importorskip("vagen.envs.active_spatial.env")
    L = _load()
    ok = L.strict_read("<action>move_forward|turn_left|</action>")
    assert ok["readable"] is True and ok["actions"] == ["move_forward", "turn_left"] and ok["reader"] == "env"
    for bad in ("<action>turn left|</action>", "<action></action>", "move forward", "<action>look_down|</action>"):
        r = L.strict_read(bad)
        assert r["readable"] is False and r["actions"] == [], bad
