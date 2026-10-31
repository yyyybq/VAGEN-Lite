"""N series episodes (downstream/n/n_episode.py, 2026-10-30): the upstream conversation text (needs the main
checkout), the strict reading of replies, the feedback, the socket framing, and the turn loop on a stand-in scene
(blocked moves, success mid-turn, three unreadable turns, the step budget)."""
import json
import os
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


def test_each_episode_runs_in_its_own_process_late_ones_are_killed_and_dead_ones_recorded(tmp_path):
    """2026-10-31: env.close() can hang for good after an episode, so each episode is a forked process that writes its
    record and leaves with os._exit; the parent kills one past the time limit and records a missing record."""
    import os
    import time

    def run_one(job):
        inst, out = job
        if inst["instance_id"] == "hangs":
            time.sleep(60)
        if inst["instance_id"] == "dies":
            raise RuntimeError("simulator crashed")
        return {"instance_id": inst["instance_id"], "success": True, "seconds": 0.1, "pid": os.getpid()}

    todo = [{"instance_id": k, "scene_id": "T_s1", "task_type": "fov_inclusion"} for k in ("ok", "hangs", "dies", "ok2")]
    t0 = time.time()
    recs = {r["instance_id"]: r for r in E.run_all(todo, tmp_path, workers=3, timeout=3, run=run_one, poll=0.1)}
    assert time.time() - t0 < 30 and set(recs) == {"ok", "hangs", "dies", "ok2"}
    assert recs["ok"]["success"] is True and recs["ok"]["pid"] != os.getpid() and recs["ok"]["pid"] != recs["ok2"]["pid"]
    assert recs["hangs"]["error"] == "no record: killed after 3 s" and recs["hangs"]["task_type"] == "fov_inclusion"
    assert recs["dies"]["error"] == "no record: its process ended with code 1"
    assert not list((tmp_path / "rows").glob("*"))  # every record moved into the caller's hands


SERVER_STUB = r"""import socket, sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", port))
s.listen(4); s.settimeout(1); end = time.time() + 30
while time.time() < end:
    try:
        s.accept()[0].close()
    except OSError:
        pass
"""
EPISODE_STUB = r"""import json, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
(out / "argv.json").write_text(json.dumps(sys.argv[1:]))
with open(out / "episodes.jsonl", "a") as f:
    f.write(json.dumps({"instance_id": "a", "success": True}) + "\n")
    f.write(json.dumps({"instance_id": "b", "success": False}) + "\n")
"""


def test_episode_job_takes_an_n1_instance_file(tmp_path):
    """run_n_episodes.sh with N1's instance file: all instances (LIMIT 0), the given port, no smoke page; run as a
    normal user it leaves ownership alone (as root it hands OUT back to OWNER_UID)."""
    import shutil
    import subprocess

    from test_run_probe_episode import SIM_ENV, free_port

    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/eval").mkdir(parents=True)
    shutil.copy(ROOT / "n/run_n_episodes.sh", wt / "downstream/n/run_n_episodes.sh")
    (wt / "downstream/n/n_policy_server.py").write_text(SERVER_STUB)
    (wt / "downstream/n/n_episode.py").write_text(EPISODE_STUB)
    (wt / "downstream/n/n_smoke.py").write_text("raise SystemExit('the smoke page must not be made for N1')\n")
    (wt / "downstream/eval/sim_env.sh").write_text(SIM_ENV)
    gen = tmp_path / "instances_eval.jsonl"
    gen.write_text('{"instance_id": "a"}\n{"instance_id": "b"}\n')
    out = tmp_path / "out"
    port = free_port()
    env = dict(os.environ, N_VLM_PY=sys.executable, N_PORT=str(port), OWNER_UID="12345", OWNER_GID="12345")
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n_episodes.sh"), str(out), str(gen), "Q3B", "3", "0"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0, res.stdout + res.stderr
    argv = json.loads((out / "argv.json").read_text())
    assert argv[argv.index("--instances") + 1] == str(gen) and argv[argv.index("--limit") + 1] == "0"
    assert argv[argv.index("--port") + 1] == str(port) and argv[argv.index("--workers") + 1] == "3"
    log = (out / "n_episodes.log").read_text()
    assert "server ready (Q3B on" in log and "1 successes of 2" in log and "page rc" not in log
    assert (out / "episodes.jsonl").stat().st_uid == os.getuid()  # not root: ownership untouched
