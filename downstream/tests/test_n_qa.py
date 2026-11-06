"""N series "can it answer, can it do" (the user's task book of 2026-11-05, item 7; downstream/n/n_qa.py,
run_n_qa_truth.sh, run_n_qa_ask.sh): the one-step truth, the three kinds of question, the samples, the analysis and the
two job scripts (CPU, no model, no simulator)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
sys.path.insert(0, str(ROOT / "x"))
import n_env as N  # noqa: E402
import n_qa as Q  # noqa: E402


class FakeScene:
    def __init__(self, blocked=("move_left",)):
        self.blocked, self.posed = set(blocked), []

    def set_pose(self, pose):
        self.posed.append(tuple(pose))

    def try_action(self, pose, action):
        if action in self.blocked:
            return tuple(pose), "geom_wall"
        return N.apply_action(pose, action), None

    def owner(self, hit):
        return "wall"

    def close(self):
        pass


def test_truth_builds_one_scene_per_robot_setting_like_the_episodes(monkeypatch, tmp_path):
    made = []

    def scene(task, seed, hide_robot=None):
        made.append((task, seed, hide_robot))
        return FakeScene(blocked=())

    monkeypatch.setattr(N, "Scene", scene)
    monkeypatch.setattr(N, "box_judge", lambda item, pose: {"metric": {"shaping_score": 0.1}})
    rows = [{"instance_id": f"T_s7_{k}", "scene_id": "T_s7", "task": "T", "seed": "7", "item": {},
             "start_pose": [0, 0, 0], **({"hide_robot": True} if k == "b" else {})} for k in "abc"]
    rows.append(dict(rows[0], instance_id="U_s1_a", scene_id="U_s1"))
    Q.write_jsonl(tmp_path / "i.jsonl", rows)
    assert Q.main(["truth", "--instances", str(tmp_path / "i.jsonl"), "--scene", "T_s7", "--out", str(tmp_path / "t")]) == 0
    assert made == [("T", 7, False), ("T", 7, True)]  # n_episode's default: the robot shown
    assert [r["instance_id"] for r in Q.read_jsonl(tmp_path / "t/T_s7.jsonl")] == ["T_s7_a", "T_s7_b", "T_s7_c"]
    assert Q.main(["truth", "--instances", str(tmp_path / "i.jsonl"), "--scene", "none", "--out", str(tmp_path / "t")]) == 2


def test_one_step_scores_each_action_and_keeps_blocked_ones_at_the_start(monkeypatch):
    monkeypatch.setattr(N, "box_judge", lambda item, pose: {"metric": {"shaping_score": 0.5 + 0.1 * pose[0]}})
    inst = {"instance_id": "T_s1_equidistance", "item": {}, "start_pose": [0.0, 0.0, 0.0]}
    s = FakeScene()
    t = Q.one_step(s, inst)
    assert t["start_shaping"] == pytest.approx(0.5)
    assert t["actions"]["move_forward"] == {"blocked": False, "by": None, "shaping": pytest.approx(0.53)}
    assert t["actions"]["move_backward"]["shaping"] == pytest.approx(0.47)
    assert t["actions"]["move_left"] == {"blocked": True, "by": "wall", "shaping": pytest.approx(0.5)}
    assert s.posed[-1] == (0.0, 0.0, 0.0) and s.posed.count((0.0, 0.0, 0.0)) == 7  # back to the start each time


def inst(iid, task_type, objs, marks, visible, desc="Move so that the bowl is left of the cup.", pose=(0.0, 0.0, 0.0),
         actions=4, near=()):
    return {"instance_id": iid, "scene_id": iid.rsplit("_", 1)[0], "task_type": task_type, "objects": objs,
            "marks": {"start": marks}, "start_judge": {"visible": visible}, "description": desc,
            "start_pose": list(pose), "shortest": {"actions": actions, "path": ["move_backward"] * actions},
            "near_threshold": list(near), "images": {"start": f"images/{iid}_start.png"}}


def box(key, name, x0):
    return {"key": key, "name": name, "kind": "object", "bbox_min": [x0, -0.1, 0.8], "bbox_max": [x0 + 0.2, 0.1, 1.0],
            "center": [x0 + 0.1, 0.0, 0.9]}


def fixtures(tmp_path):
    a, b = box("obj_a", "bowl", 1.0), box("obj_b", "cup", 2.0)
    insts = [
        inst("K_s1_equidistance", "equidistance", [a, b], {"a": [50.0, 100.0], "b": [150.0, 110.0]},
             {"obj_a": 0.01, "obj_b": 0.02}),
        inst("K_s2_fov_inclusion", "fov_inclusion", [a, dict(b, name="bowl")], {"a": [10.0, 9.0], "b": [200.0, 9.0]},
             {"obj_a": 0.01, "obj_b": 0.02}),  # two bowls: no object question
        inst("K_s3_absolute_positioning", "absolute_positioning", [a], {"a": [100.0, 100.0]}, {"obj_a": 0.05}),
        inst("K_s4_equidistance", "equidistance", [a, box("obj_b", "cup", 1.1)], {"a": [120.0, 9.0], "b": [140.0, 9.0]},
             {"obj_a": 0.01, "obj_b": 0.0001}),  # the cup hardly visible
    ]
    gen = tmp_path / "gen"
    (gen / "images").mkdir(parents=True)
    for i in insts:
        (gen / f"images/{i['instance_id']}_start.png").write_bytes(b"png")
    truth = {
        "K_s1_equidistance": {"start_shaping": 0.1, "actions": {a_: {"blocked": False, "shaping": s} for a_, s in zip(
            N.ACTIONS, (0.5, 0.1, 0.1, 0.1, 0.12, 0.1))}},
        "K_s3_absolute_positioning": {"start_shaping": 0.4, "actions": {a_: {"blocked": a_ == "move_left", "shaping": 0.41}
                                                                        for a_ in N.ACTIONS}},
    }
    return insts, gen, truth


def test_three_kinds_of_question_and_their_rules(tmp_path):
    insts, gen, truth = fixtures(tmp_path)
    qs, skipped = Q.make_questions(insts, Q.image_paths(insts, [gen]), truth, seed=1)
    kinds = {(q["instance_id"], q["kind"]) for q in qs}
    assert kinds == {("K_s1_equidistance", "left_right"), ("K_s1_equidistance", "near_far"),
                     ("K_s1_equidistance", "best_action")}
    by = {q["kind"]: q for q in qs}
    lr = by["left_right"]
    subj, other = lr["numbers"]["subject"], lr["numbers"]["other"]
    want = "left" if subj == "bowl" else "right"  # the bowl's centre (x 50) is left of the cup's (x 150)
    assert lr["options"]["AB".index(lr["answer"])] == want and lr["prompt"].endswith(Q.ANSWER_TEXT)
    assert f"is the {subj} to the left or to the right of the {other}?" in lr["prompt"]
    nf = by["near_far"]
    assert nf["options"]["AB".index(nf["answer"])] == "the bowl" and nf["numbers"]["distance_m"] == {"bowl": 1.0, "cup": 2.0}
    ba = by["best_action"]
    assert ba["options"]["AB".index(ba["answer"])] == "move forward" and ba["numbers"]["best"] == "move_forward"
    assert ba["numbers"]["other"] in ("move_backward", "move_left", "move_right", "turn_left", "turn_right")
    assert "The robot's task: Move so that the bowl is left of the cup." in ba["prompt"]
    assert skipped["object kinds: not two visible, differently named objects"] == 3
    assert skipped["best_action: no action rises enough or none clearly worse"] == 1  # K_s3: +0.01 only
    assert skipped["best_action: no truth"] == 2
    assert Q.footprint_distance((1.1, 0.0, 0.0), box("o", "x", 1.0)) == 0.0  # inside the footprint


def test_samples_take_turns_over_kinds_and_task_types():
    qs = [{"id": f"{t}{k}{i}", "kind": k, "task_type": t} for k in Q.KINDS for t in ("p", "q") for i in range(3)]
    got = Q.pick_samples(qs, 2 * len(Q.KINDS), seed=0)
    assert [q["kind"] for q in got] == list(Q.KINDS) * 2
    assert {q["task_type"] for q in got if q["kind"] == "near_far"} == {"p", "q"}
    assert len(Q.pick_samples(qs, 100)) == len(qs)


def test_readings_and_the_analysis_of_answer_against_do():
    opts = ["the bowl", "the cup"]
    assert Q.parse_strict("B") == "B" and Q.parse_strict("the cup") is None
    assert Q.parse_lenient("the cup", opts) == "B" and Q.parse_lenient("It is the bowl.", opts) == "A"
    assert Q.parse_lenient("bowl or cup", ["bowl", "cup"]) is None
    qs = [{"id": f"I{k}::near_far", "instance_id": f"I{k}", "kind": "near_far", "options": opts, "answer": "A"}
          for k in range(6)]
    insts = {f"I{k}": {"shortest": {"actions": 2 + k}, "near_threshold": ["x"] if k < 2 else []} for k in range(6)}
    ans = {"M": {f"I{k}::near_far": ("A" if k < 4 else "B") for k in range(6)},
           "W": {f"I{k}::near_far": ("B" if k < 4 else "A") for k in range(6)}}
    eps = {"M": {f"I{k}": {"success": k in (0, 1, 4), "max_shaping_score": 0.1 * k} for k in range(6)},
           "W": {f"I{k}": {"success": k == 2, "max_shaping_score": 0.2} for k in range(6)}}
    rep = Q.analyze(qs, ans, eps, insts)
    row = rep["models"]["M"]["strict/near_far"]
    assert row["correct"] == 4 and row["n"] == 6 and row["unparsed"] == 0
    g = row["groups"]["all"]
    assert g["right"]["n"] == 4 and g["right"]["success"] == 2 and g["wrong"]["success"] == 1
    assert g["right"]["max_shaping"] == pytest.approx(0.15)
    assert row["groups"]["length 2-3"]["right"]["n"] == 2 and row["groups"]["near threshold"]["right"]["n"] == 2
    pair = rep["pairs"]["strict/near_far/M>W"]  # M right, W wrong on I0-I3: M succeeds alone on I0, I1; W on I2
    assert pair == {"n": 4, "a_succeeds_b_fails": 2, "b_succeeds_a_fails": 1, "p": pytest.approx(1.0)}
    assert Q.binom_two_sided(0, 6) == pytest.approx(2 / 64)
    assert "## strict/near_far" in Q.markdown(rep)


def test_samples_page_marks_the_objects(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    insts, gen, truth = fixtures(tmp_path)
    for i in insts:
        Image.new("RGB", (256, 256), (120, 130, 140)).save(gen / f"images/{i['instance_id']}_start.png")
    qs, _ = Q.make_questions(insts, Q.image_paths(insts, [gen]), truth, seed=1)
    Q.write_jsonl(tmp_path / "q.jsonl", qs)
    Q.write_jsonl(tmp_path / "i.jsonl", insts)
    assert Q.main(["samples", "--questions", str(tmp_path / "q.jsonl"), "--instances", str(tmp_path / "i.jsonl"),
                   "--n", "3", "--out", str(tmp_path / "page.html")]) == 0
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert page.count("<section>") == 3 and "data:image/png;base64," in page and "正确答案" in page
    assert len(Q.read_jsonl(tmp_path / "page.jsonl")) == 3


TRUTH_STUB = r"""import json, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == "truth":
    sid, out = a[a.index("--scene") + 1], Path(a[a.index("--out") + 1])
    if sid == "bad":
        sys.exit(3)
    (out / f"{sid}.jsonl").write_text(json.dumps({"instance_id": sid + "_x"}) + "\n")
else:
    t, o = Path(a[a.index("--truth") + 1]), Path(a[a.index("--out") + 1])
    o.write_text("".join(p.read_text() for p in sorted(t.glob("*.jsonl"))))
"""


def test_truth_job_runs_each_scene_once_and_lists_the_failed(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/eval").mkdir(parents=True)
    (wt / "downstream/n/run_n_qa_truth.sh").write_text((ROOT / "n/run_n_qa_truth.sh").read_text())
    (wt / "downstream/n/n_qa.py").write_text(TRUTH_STUB)
    (wt / "downstream/eval/sim_env.sh").write_text('SIM_PY=python3\nsim() { shift; "$@"; }\n')
    inst = tmp_path / "i.jsonl"
    inst.write_text("".join(json.dumps({"scene_id": s}) + "\n" for s in ("A_s1", "B_s2", "bad", "A_s1")))
    out = tmp_path / "out"
    (out / "truth").mkdir(parents=True)
    (out / "truth/B_s2.jsonl").write_text('{"instance_id": "old"}\n')  # done before: kept
    env = dict(os.environ, N_QA_SIM_PY=sys.executable)
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n_qa_truth.sh"), str(inst), str(out), "2", "30"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (out / "scenes.txt").read_text().split() == ["A_s1", "B_s2", "bad"]
    assert [json.loads(x)["instance_id"] for x in (out / "truth.jsonl").read_text().splitlines()] == ["A_s1_x", "old"]
    assert (out / "failed.txt").read_text().split() == ["bad", "rc=3"]
    assert "done: 2 instances, 1 scenes failed" in (out / "truth.log").read_text()


ASK_STUB = r"""import json, sys
a = sys.argv[1:]
m, spec = a[a.index("--model") + 1], a[a.index("--ask") + 1]
q, out = spec.split("=")
with open(out, "a") as f:
    for line in open(q):
        f.write(json.dumps({"id": json.loads(line)["id"], "reply": "A"}) + "\n")
"""


def test_ask_job_runs_the_family_and_counts_the_answers(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/x").mkdir(parents=True)
    (wt / "downstream/n/run_n_qa_ask.sh").write_text((ROOT / "n/run_n_qa_ask.sh").read_text())
    (wt / "downstream/x/q0_ask.py").write_text(ASK_STUB)
    out = tmp_path / "qa"
    out.mkdir()
    Q.write_jsonl(out / "questions.jsonl", [{"id": "q1"}, {"id": "q2"}])
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n_qa_ask.sh"), str(out), "qwen3"], capture_output=True,
                         text=True, env=dict(os.environ, N_VLM_PY=sys.executable), timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert sorted(p.name for p in (out / "answers").iterdir()) == ["Q3B.jsonl", "Q3RD.jsonl", "Q3RS.jsonl", "Q3S.jsonl"]
    assert "Q3RD: rc=0, 2 answers" in (out / "ask_qwen3.log").read_text()
    bad = subprocess.run(["bash", str(wt / "downstream/n/run_n_qa_ask.sh"), str(out), "other"], capture_output=True,
                         text=True, timeout=60)
    assert bad.returncode == 2


class GridScene:
    """A robot that may not go beyond x = 0.25 (so moving forward from the start is blocked)."""

    def __init__(self):
        self.pose, self.checked = (0.0, 0.0, 0.0), 0

    def set_pose(self, pose):
        self.pose = tuple(pose)

    def collision(self):
        self.checked += 1
        return "wall" if self.pose[0] > 0.25 else None


def test_search_keeps_every_edge_and_costs_count_to_the_nearest_success():
    s = GridScene()
    states, edges, capped = Q.search(s, (0.0, 0.0, 0.0), max_depth=4, max_states=10000)
    k0 = N.pose_key((0.0, 0.0, 0.0))
    assert not capped and edges[k0]["move_forward"] is None and s.pose == (0.0, 0.0, 0.0)
    assert edges[k0]["move_left"] == N.pose_key((0.0, 0.3, 0.0))
    inst = {"instance_id": "I", "start_pose": [0.0, 0.0, 0.0]}
    r = Q.first_actions(states, edges, capped, inst, lambda pose: pose[1] >= 0.55, max_depth=4)
    c = r["cost"]
    assert r["start_dist"] == 2 and c["move_left"] == 2
    assert c["move_forward"] == 3 and r["blocked"] == ["move_forward"]  # a blocked first action wastes one
    assert c["move_backward"] == 3 and c["move_right"] == 4
    assert r["goal_poses"] > 0 and r["poses"] == len(states)
    assert r["exact_up_to"] == 4  # not capped: every pose shallower than max_depth expanded
    states, edges, capped = Q.search(GridScene(), (0.0, 0.0, 0.0), max_depth=4, max_states=5)
    assert capped  # stopped while expanding the start: only the start's own edges are sure
    assert Q.first_actions(states, edges, capped, inst, lambda pose: True)["exact_up_to"] == 0
    states, edges, capped = Q.search(GridScene(), (0.0, 0.0, 0.0), max_depth=4, max_states=12)
    assert capped and Q.exact_depth(states, edges, capped) == 1


def test_first_action_asks_a_shortest_start_against_one_at_least_two_longer():
    import collections
    import random
    inst = {"instance_id": "I", "description": "Move so that the bowl is left of the cup.",
            "shortest": {"path": ["move_left", "move_left"], "actions": 2}}
    cost = {"move_forward": 4, "move_backward": 3, "move_left": 3, "move_right": 5, "turn_left": None, "turn_right": 4}
    skipped = collections.Counter()
    q = Q.first_action_question(inst, {"instance_id": "I"}, {"cost": cost, "blocked": ["move_forward"]},
                                random.Random(0), skipped)
    assert q["numbers"]["optimal"] == ["move_backward", "move_left"] and q["numbers"]["best"] == "move_left"
    assert q["options"]["AB".index(q["answer"])] == "move left"
    # a free action 5 >= 3 + 2 or with no success reachable through it comes before the blocked move_forward
    assert q["numbers"]["other"] in ("move_right", "turn_left") and q["numbers"]["other_kind"] == "longer"
    assert "in the fewest actions?" in q["prompt"]
    picked = {Q.first_action_question(inst, {}, {"cost": cost, "blocked": ["move_forward"]}, random.Random(s),
                                      collections.Counter())["numbers"]["other"] for s in range(20)}
    assert picked == {"move_right", "turn_left"}
    near = dict(cost, move_right=4, turn_left=4)
    assert Q.first_action_question(inst, {}, {"cost": near}, random.Random(0), skipped) is None
    only_blocked = Q.first_action_question(inst, {}, {"cost": near, "blocked": ["move_forward"]}, random.Random(0), skipped)
    assert only_blocked["numbers"]["other"] == "move_forward"  # one step worse, but blocked (the 11-06 task book)
    assert only_blocked["numbers"]["other_kind"] == "blocked" and only_blocked["numbers"]["other_at_least"] == 4
    assert Q.first_action_question(inst, {}, {"cost": {a: None for a in cost}}, random.Random(0), skipped) is None
    assert skipped == {"first_action: no action clearly worse": 1, "first_action: no success reachable": 1}


def test_first_action_trusts_a_capped_search_only_up_to_its_exact_depth():
    import collections
    import random
    inst = {"instance_id": "I", "description": "d", "shortest": {"path": ["move_left"] * 3, "actions": 3}}
    cost = {"move_forward": 9, "move_backward": 6, "move_left": 3, "move_right": None, "turn_left": 5, "turn_right": 4}
    skipped = collections.Counter()
    # exact up to 3: 4, 5, 6, 9 and None only mean "at least 4", one short of 3 + 2
    assert Q.first_action_question(inst, {}, {"cost": cost, "exact_up_to": 3, "capped": True},
                                   random.Random(0), skipped) is None
    # exact up to 4: turn_right's 4 is exact (one longer), the others at least 5
    q = Q.first_action_question(inst, {}, {"cost": cost, "exact_up_to": 4, "capped": True}, random.Random(0), skipped)
    assert q["numbers"]["other"] in ("move_forward", "move_backward", "move_right", "turn_left")
    assert q["numbers"]["other_at_least"] == 5 and q["numbers"]["exact_up_to"] == 4
    # the shortest way (3) beyond exact_up_to + 1 = 2 may itself be shorter: no question
    assert Q.first_action_question(inst, {}, {"cost": cost, "exact_up_to": 1, "capped": True},
                                   random.Random(0), skipped) is None
    # a row without exact_up_to from a capped search is trusted nowhere
    assert Q.first_action_question(inst, {}, {"cost": cost, "capped": True}, random.Random(0), skipped) is None
    assert skipped == {"first_action: no action clearly worse": 1,
                       "first_action: shortest way beyond the search's exact depth": 2}


def test_samples_can_be_counted_per_kind():
    qs = [{"id": f"{t}{k}{i}", "kind": k, "task_type": t} for k in Q.KINDS for t in ("p", "q") for i in range(8)]
    got = Q.pick_by_kind(qs, {"left_right": "5", "near_far": "5", "first_action": "10"})
    assert [q["kind"] for q in got] == ["left_right"] * 5 + ["near_far"] * 5 + ["first_action"] * 10
    assert {q["task_type"] for q in got if q["kind"] == "first_action"} == {"p", "q"}


def test_object_kinds_can_be_asked_on_the_success_view(tmp_path):
    a, b = box("obj_a", "bowl", 1.0), box("obj_b", "cup", 2.0)
    i = inst("K_s5_equidistance", "equidistance", [a, b], {}, {})
    i["marks"]["shortest"] = {"a": [200.0, 100.0], "b": [100.0, 100.0]}
    i["shortest"] = {"pose": [1.4, 0.0, 0.0], "actions": 3, "path": ["move_forward"] * 3,
                     "judge": {"visible": {"obj_a": 0.02, "obj_b": 0.02}}}
    i["images"]["shortest"] = "images/K_s5_equidistance_shortest.png"
    gen = tmp_path / "gen"
    (gen / "images").mkdir(parents=True)
    for tag in ("start", "shortest"):
        (gen / f"images/K_s5_equidistance_{tag}.png").write_bytes(b"png")
    qs, skipped = Q.make_questions([i], Q.image_paths([i], [gen]), {}, seed=1, object_image="shortest",
                                   object_images=Q.image_paths([i], [gen], "shortest"),
                                   kinds=("left_right", "near_far", "best_action"))
    by = {q["kind"]: q for q in qs}
    assert set(by) == {"left_right", "near_far"} and {q["image_tag"] for q in qs} == {"shortest"}
    assert by["left_right"]["image_path"].endswith("_shortest.png")
    assert by["near_far"]["numbers"]["distance_m"] == {"bowl": 0.2, "cup": 0.6}  # from the success pose x = 1.4
    assert skipped == {"best_action: no truth": 1}
    start_only, _ = Q.make_questions([i], Q.image_paths([i], [gen]), {}, seed=1, kinds=("left_right",))
    assert start_only == []  # nothing visible at the start


def test_truth_job_also_runs_the_search_mode(tmp_path):
    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/eval").mkdir(parents=True)
    (wt / "downstream/n/run_n_qa_truth.sh").write_text((ROOT / "n/run_n_qa_truth.sh").read_text())
    (wt / "downstream/n/n_qa.py").write_text(TRUTH_STUB.replace('a[0] == "truth"', 'a[0] in ("truth", "paths")'))
    (wt / "downstream/eval/sim_env.sh").write_text('SIM_PY=python3\nsim() { shift; "$@"; }\n')
    inst = tmp_path / "i.jsonl"
    inst.write_text("".join(json.dumps({"scene_id": s}) + "\n" for s in ("A_s1", "bad")))
    out = tmp_path / "out"
    env = dict(os.environ, N_QA_SIM_PY=sys.executable)
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n_qa_truth.sh"), str(inst), str(out), "2", "30", "paths"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert [json.loads(x)["instance_id"] for x in (out / "paths.jsonl").read_text().splitlines()] == ["A_s1_x"]
    assert (out / "failed_paths.txt").read_text().split() == ["bad", "rc=3"] and not (out / "failed.txt").exists()
    assert "done: 1 instances, 1 scenes failed" in (out / "paths.log").read_text()
    bad = subprocess.run(["bash", str(wt / "downstream/n/run_n_qa_truth.sh"), str(inst), str(out), "2", "30", "x"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert bad.returncode == 2


def test_sample_note_explains_the_first_action_distractor():
    q = {"kind": "first_action", "numbers": {"fewest": 3, "exact_up_to": 5, "optimal": ["move_left", "turn_left"],
                                             "other": "move_forward", "other_kind": "blocked", "other_at_least": 4}}
    note = Q.sample_note(q)
    assert "最短 3 步" in note and "5 步以内" in note and "move_left、turn_left" in note
    assert "move_forward：起点处被挡（原地不动，至少 4 步）" in note
    q["numbers"].update(other="move_right", other_kind="longer", other_at_least=6)
    assert "move_right：走它至少要 6 步）" in Q.sample_note(q)
    assert Q.sample_note({"kind": "left_right", "numbers": {}}) == ""
