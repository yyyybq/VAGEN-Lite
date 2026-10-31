"""N series episodes (downstream/n/n_episode.py, 2026-10-30): the upstream conversation text (needs the main
checkout), the strict reading of replies, the feedback, the socket framing, and the turn loop on a stand-in scene
(blocked moves, success mid-turn, three unreadable turns, the step budget)."""
import socket
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
sys.path.insert(0, str(ROOT / "tests"))
import n_env as N  # noqa: E402
import n_episode as E  # noqa: E402
from test_n_gen import isolate_main_modules  # noqa: E402

UPSTREAM = (N.MAIN / "vagen/envs/active_spatial/prompt.py").is_file() and E.VAL_ENV.is_file()


@pytest.fixture(autouse=True)
def isolated():  # the main checkout's vagen (n_env) and the worktree's (other tests) kept apart
    yield from isolate_main_modules()


def test_feedback_reads_like_the_env():
    assert E.turn_feedback(["move_forward", "move_backward", "turn_left"], [("move_backward", "blocked by wall")],
                           True) == "2 of 3 actions executed; blocked: move_backward (blocked by wall)."
    assert E.turn_feedback(["turn_left"], [], True) == "Action executed."
    assert E.turn_feedback(["move_left"], [], False) == "Action had no effect."


def test_chat_puts_the_image_where_the_placeholder_was():
    msgs = E.chat("SYS", "[Initial Observation]:\n<image>\nCurrent camera pose: [..]")
    assert msgs[0] == {"role": "system", "content": [{"type": "text", "text": "SYS"}]}
    assert msgs[1]["content"] == [{"type": "text", "text": "[Initial Observation]:\n"}, {"type": "image"},
                                  {"type": "text", "text": "\nCurrent camera pose: [..]"}]


def test_socket_framing_round_trip():
    a, b = socket.socketpair()
    big = {"x": "y" * 300000, "n": [1, 2, 3]}
    t = threading.Thread(target=lambda: E.send_msg(a, big))
    t.start()
    assert E.recv_msg(b) == big
    t.join()
    a.close()
    b.close()


@pytest.mark.skipif(not UPSTREAM, reason="needs the main checkout")
def test_upstream_text_and_reading():
    cfg, entry = E.upstream_config()
    assert cfg.action_space == "strafe" and cfg.max_actions_per_step == 5 and cfg.max_episode_steps == 12
    assert int(entry["max_turns"]) == 12
    system = E.system_text(cfg, "projective_relations")
    assert "move_forward" in system and "0.30 meters" in system
    first = E.user_text(cfg, True, (1.0, 2.0, 0.0), "Move so that the mug appears to the left of the kettle in your view.")
    later = E.user_text(cfg, False, (1.0, 2.0, 0.0), "TASK", "Action executed.")
    assert "<image>" in first and "Current camera pose: [tx=1.0000, ty=2.0000, tz=1.5000" in first
    assert "Move so that the mug appears" in first and "Action executed." in later and first != later
    allowed = E.allowed_actions(cfg)
    assert E.parse("<action>move_forward|turn_left|</action>", cfg, allowed) == (["move_forward", "turn_left"], "")
    assert E.parse("I would go forward", cfg, allowed)[0] == []
    assert E.parse("<action>look_up|</action>", cfg, allowed) == ([], "unknown_action_name")
    six = "<action>" + "move_left|" * 6 + "</action>"
    assert len(E.parse(six, cfg, allowed)[0]) == 5


class FakeCfg:
    max_consecutive_invalid_actions, max_episode_steps, max_actions_per_step = 3, 12, 5


class FakeScene:
    """A wall at x >= 0.65; success once x >= 0.6 and the heading is 0."""

    def __init__(self):
        self.start = (0.0, 0.0, 0.0)

    def render(self, pose):
        import numpy as np

        return np.zeros((4, 4, 3), "uint8"), None

    def try_action(self, pose, action):
        new = N.apply_action(pose, action)
        return (pose, "wall") if new[0] >= 0.65 else (new, None)

    def owner(self, hit):
        return hit

    def judge(self, item, pose, render=True):
        ok = pose[0] >= 0.6 - 1e-9 and abs(pose[2]) < 1e-9
        return {"success": ok, "visible": {"a": 0.1, "b": 0.1},
                "metric": {"relation_margin_px": 20.0, "shaping_score": 0.2 + max(0.0, pose[0])},
                "lateral": {"angle_deg": 5.0}}


INST = {"scene_id": "T_s1", "item": {"task_type": "projective_relations", "task_description": "TASK"},
        "a": {"key": "a"}, "b": {"key": "b"}}


def run(monkeypatch, replies):
    calls = []
    monkeypatch.setattr(E, "system_text", lambda cfg, t: "SYS")
    monkeypatch.setattr(E, "user_text", lambda cfg, first, pose, task, fb="": f"{first}|{fb}")
    monkeypatch.setattr(E, "allowed_actions", lambda cfg: set(N.ACTIONS))
    monkeypatch.setattr(N, "box_judge", lambda item, pose: {"success": False, "metric": {"shaping_score": 0.1},
                                                            "lateral": {}})
    monkeypatch.setattr(N, "measurements", lambda item, j: {"relation_margin_px": j["metric"]["relation_margin_px"]})

    def parse(text, cfg, allowed):
        acts = [a for a in text.split(",") if a in N.ACTIONS]
        return (acts, "") if acts else ([], "missing_action_tag")
    monkeypatch.setattr(E, "parse", parse)

    def ask(msgs, images, n):
        calls.append(msgs[1]["content"][0]["text"])
        return replies[min(len(calls) - 1, len(replies) - 1)]
    rec, frames = E.run_episode(FakeScene(), INST, ask, FakeCfg(), {"max_turns": 12, "response_length_per_turn": 384})
    return rec, frames, calls


def test_success_mid_turn_stops_the_turn(monkeypatch):
    rec, frames, calls = run(monkeypatch, ["move_forward,move_forward,turn_left"])
    assert rec["success"] and rec["end"] == "success" and rec["steps"] == 2 and rec["turns"] == 1
    assert rec["turn_log"][0]["executed"] == ["move_forward", "move_forward"]  # turn_left never ran
    assert rec["max_shaping_score"] == pytest.approx(0.8)  # the best judge along the way (x = 0.6)
    assert rec["instance_id"] == "T_s1" and rec["task_type"] == "projective_relations"
    assert rec["final_judge"]["measurements"] == {"relation_margin_px": 20.0}


def test_a_wall_blocks_the_feedback_says_so_and_three_bad_replies_end_it(monkeypatch):
    # heading 20 degrees: three forward moves reach x = 0.85 > the wall at 0.65, never the success (heading 0)
    rec, _, calls = run(monkeypatch, ["turn_left,move_forward,move_forward,move_forward", "nonsense", "nonsense",
                                      "nonsense"])
    t1 = rec["turn_log"][0]
    assert t1["executed"] == ["turn_left", "move_forward", "move_forward", "move_forward"]
    assert t1["blocked"] == [("move_forward", "blocked by wall")] and rec["blocked"] == 1
    assert calls[1] == "False|3 of 4 actions executed; blocked: move_forward (blocked by wall)."
    assert rec["end"] == "invalid" and rec["invalid_turns"] == 3 and rec["turns"] == 4 and not rec["success"]
    assert calls[3] == "False|Action had no effect."


def test_step_budget_ends_the_episode(monkeypatch):
    rec, _, _ = run(monkeypatch, ["move_left,move_right,move_left,move_right,move_left"])
    assert rec["end"] == "steps" and rec["steps"] == 12 and not rec["success"]
