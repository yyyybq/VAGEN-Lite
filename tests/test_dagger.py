"""vagen.dagger (the reward line, phase 3, experiment B) on synthetic inputs: temp folders only, CPU only."""

import asyncio
import json
import re
import types

import numpy as np
import pytest
import torch

from vagen import dagger as D

# a camera looking along +y on the floor plane (x, y); the contract keys are (x, y, yaw)
R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
START = np.eye(4)
START[:3, :3] = R0


class Geom:  # every move legal except into x < -0.1 (a wall on the left)
    def transitions_legal(self, P0, P1, R1):
        return np.asarray(P1)[:, 0] > -0.1


class Detector:
    def check_collision(self, position, previous_position):
        return types.SimpleNamespace(has_collision=bool(np.asarray(position)[0] <= -0.1))


def _table(depth=4):
    """A real-shaped table from the tables' own graph code: success = two steps forward of the start."""
    from data_gen.interiorgs_v10.difficulty import _split, apply_action_batch, contract_keys
    from data_gen.interiorgs_v10.distance_table import backward_distances, reachable_graph

    keys, P, R, depth_of, edges = reachable_graph(Geom(), START, depth)
    p, r = _split(START)
    for _ in range(2):
        p, r = apply_action_batch(p, r, "move_forward")
    goal = int(contract_keys(p, r)[0])
    dist = backward_distances(len(keys), edges, keys == goal)
    order = np.argsort(keys)
    return {"keys": keys[order], "dist": dist[order], "depth_of": depth_of[order]}


def _pose(actions):
    from data_gen.interiorgs_v10.difficulty import _split, apply_action_batch

    p, r = _split(START)
    for a in actions:
        p, r = apply_action_batch(p, r, a)
    return p, np.array(np.broadcast_to(r, (1, 3, 3)))


# ------------------------------------------------------------------------------------------ settings

def test_settings_defaults_and_refusals(tmp_path):
    s = D.settings({"tables": "/t"})
    assert s == {**D.DEFAULTS, "tables": "/t"} and s["lam"] == 0.5 and s["cap"] == 96 and s["turns"] == "all"
    assert D.settings(json.dumps({"tables": "/t", "lam": 1})) ["lam"] == 1.0
    for bad in ({}, {"tables": "/t", "lr": 1}, {"tables": "/t", "lam": 0}, {"tables": "/t", "cap": 0},
                {"tables": "/t", "turns": "some"}, {"tables": "/t", "max_per_turn": 6}, {"tables": "/t", "cap": 1.5},
                {"tables": "/t", "audit_only": "yes"}, {"tables": "/t", "lam": True}, {"tables": "/t", "enable": True}):
        with pytest.raises(ValueError):
            D.settings(bad)
    with pytest.raises(ValueError):  # a folder without tables
        D.ppo_block({"tables": str(tmp_path)}, model="/m", run_root=tmp_path / "run")
    (tmp_path / "t1.npz").write_bytes(b"x")
    block = D.ppo_block({"tables": str(tmp_path), "cap": 16}, model="/m", run_root=tmp_path / "run")
    assert block["enable"] is True and block["cap"] == 16 and block["tokenizer"] == "/m"
    assert block["log_dir"] == str((tmp_path / "run").resolve() / "dagger")
    cfg = {"actor_rollout_ref": {"actor": {"dagger": block}}}
    assert D.from_config(cfg) == block
    assert D.from_config({"actor_rollout_ref": {"actor": {}}}) is None
    assert D.from_config({"actor_rollout_ref": {"actor": {"dagger": dict(block, enable=False)}}}) is None
    from omegaconf import OmegaConf

    assert D.from_config(OmegaConf.create(cfg)) == block


# ------------------------------------------------------------------------------------------ turns

def test_executed_actions():
    allowed = ["move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"]
    assert D.executed_actions(["move_forward", "look_up", "turn_left"], allowed, 2) == (["move_forward", "turn_left"],
                                                                                       False)
    assert D.executed_actions(["move_forward", "move_forward", "move_forward"], allowed, 1) == (["move_forward"], False)
    assert D.executed_actions(["move_forward", "done"], allowed, 1) == (["move_forward"], True)
    assert D.executed_actions(["move_forward", "done"], allowed, 0) == ([], False)  # stopped before (success)
    assert D.executed_actions([], allowed, 0) == ([], False)


def test_judge_and_plan_on_a_table():
    tab = _table()
    P, R = _pose([])
    assert D.distance(tab, P, R) == (2, False)
    det = Detector()
    ok = D.judge(tab, P, R, ["move_forward", "move_forward"], det)
    assert ok["status"] == "ok" and ok["hits"] == [True, True] and ok["d_after"] == 0
    wrong = D.judge(tab, P, R, ["move_forward", "turn_left"], det)
    assert wrong["hits"] == [True, False] and wrong["d_after"] == 2
    blocked = D.judge(tab, P, R, ["move_left"], det)  # into the wall: the pose stays, never optimal
    assert blocked["hits"] == [False] and blocked["blocked"] == 1 and blocked["d_after"] == 2
    steps, stop = D.plan(tab, P, R, 5, det, None)
    assert steps == ["move_forward", "move_forward"] and stop == ""  # a label may end the task
    assert D.chunk(["move_forward", "move_forward", "turn_left", "move_forward"] + ["turn_right"] * 6, 5) == [
        ["move_forward", "move_forward"], ["turn_left"], ["move_forward"], ["turn_right"] * 5, ["turn_right"]]
    assert D.action_text(["turn_left", "turn_left"]) == "<action>turn_left|turn_left|</action>"
    # a pose missing from the table but next to one of its cells takes the neighbour's distance
    from data_gen.interiorgs_v10.distance_table_near import neighbour_distance  # noqa: F401

    P2 = P + np.array([[0.004, 0.0, 0.0]])  # 0.4 cm off: the 1 cm cell rounds to the start's
    assert D.distance(tab, P2, R) == (2, False)
    P3 = P + np.array([[0.012, 0.0, 0.0]])  # the next cell over: only a neighbour is in the table
    d3, near3 = D.distance(tab, P3, R)
    assert near3 and d3 == 2
    far = D.judge(tab, P + np.array([[5.0, 5.0, 0.0]]), R, [], det)
    assert far["status"] == "unknown_start"
    # a cell in the table with no success reachable (-1) also takes its neighbours, as distance_table_near does
    from data_gen.interiorgs_v10.difficulty import contract_keys
    from data_gen.interiorgs_v10.distance_table_near import pack, unpack

    k0 = int(contract_keys(P, R)[0])
    kx, ky, kyaw = unpack(k0)
    keys = np.append(tab["keys"], pack(kx + 1, ky, kyaw))
    dist = np.append(np.where(tab["keys"] == k0, -1, tab["dist"]), 7)
    order = np.argsort(keys)
    assert D.distance({"keys": keys[order], "dist": dist[order]}, P, R) == (7, True)


def test_canonical_answer():
    assert D.canonical_answer("<action>move_forward|turn_left|</action><|im_end|>", ["move_forward", "turn_left"])
    assert D.canonical_answer("<action>move_forward|</action>", ["move_forward"])
    for text in ("<action>move_forward</action>", "<action>Move_Forward|</action>", "x<action>move_forward|</action>",
                 "<action>move_forward|</action><action>turn_left|</action>", "<action>move_forward|</action>\n",
                 "<action> move_forward|</action>", "<action>|move_forward|</action>"):
        assert not D.canonical_answer(text, D.written_actions(text)), text
    assert not D.canonical_answer("garbage", None)


class Tok:
    """A word-level stand-in for the Qwen tokenizer: action names, tags, '|' and the end of a turn are tokens."""
    PIECES = ["<|im_end|>", "</action>", "<action>", "|", "move_forward", "move_backward", "move_left", "move_right",
              "turn_left", "turn_right"]

    def __init__(self):
        self.vocab = {p: i + 10 for i, p in enumerate(self.PIECES)}
        self.inv = {i: p for p, i in self.vocab.items()}
        self.pat = re.compile("|".join(re.escape(p) for p in self.PIECES) + "|.", re.S)

    def encode(self, text, add_special_tokens=False):
        out = []
        for m in self.pat.findall(text):
            if m not in self.vocab:
                self.vocab[m] = len(self.vocab) + 10
                self.inv[self.vocab[m]] = m
            out.append(self.vocab[m])
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv[int(i)] for i in ids)

    def convert_tokens_to_ids(self, tok):
        return self.vocab[tok]


def test_loss_tokens():
    tok = Tok()
    ids = tok.encode("<action>move_forward|move_forward|</action><|im_end|>")
    m = D.loss_tokens(tok, ids)
    assert tok.decode([ids[j] for j in m]) == "move_forward|move_forward|</action>"
    assert D.loss_tokens(tok, tok.encode("no tags here")) == []
    noisy = tok.encode("x<action>turn_left|</action>y<action>move_left|</action>")
    assert tok.decode([noisy[j] for j in D.loss_tokens(tok, noisy)]) == "turn_left|</action>"
    chars = list("<action>a|</action>")  # a character-level tokenizer: every character of the span, nothing else

    class CharTok:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chars[i] for i in ids)

    got = D.loss_tokens(CharTok(), list(range(len(chars))))
    assert "".join(chars[j] for j in got) == "a|</action>"


class Inner:
    """What the labeller reads of ActiveSpatialEnv."""

    def __init__(self, task_id, pose):
        self.current_item = {"task_id": task_id}
        self._allowed_actions = ["move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"]
        self.collision_detector = Detector()
        self.config = types.SimpleNamespace(enable_collision_detection=True)
        self._current_step = 0
        self.pose = pose
        self.view_engine = types.SimpleNamespace(get_pose=lambda: self.pose)


def _env_step(inner, actions, executed):
    from data_gen.interiorgs_v10.difficulty import _split, apply_action_batch, to_c2w

    p, r = _split(inner.pose)
    for a in executed:
        q, s = apply_action_batch(p, r, a)
        if not (a.startswith("move_") and q[0, 0] <= -0.1):
            p, r = q, np.array(np.broadcast_to(s, (1, 3, 3)))
        inner._current_step += 1
    inner.pose = to_c2w(p, r)[0]
    return {"actions": actions, "metrics": {"turn_metrics": {"action_is_valid": bool(actions)}}}


def test_after_turn_records(tmp_path):
    from data_gen.interiorgs_v10.distance_table import save

    save(tmp_path / "task1.npz", _table())
    cfg = D.settings({"tables": str(tmp_path)})
    tok = Tok()
    end = tok.convert_tokens_to_ids("<|im_end|>")

    def run(actions, executed, answer, start=None, budget=None):
        inner = Inner("task1", (START if start is None else start).copy())
        if budget is not None:
            inner._max_episode_steps = budget
        env = types.SimpleNamespace(_inner=inner)
        pre = D.pre_turn(env)
        info = _env_step(inner, actions, executed)
        ids = tok.encode(answer) + [end]
        return asyncio.run(D.after_turn(cfg, env, pre, info, (0, len(ids)), ids, tok, 0))

    good = run(["move_forward"], ["move_forward"], "<action>move_forward|</action>")
    assert good["status"] == "ok" and good["optimal"] and good["retok_same"] and "label_ids" not in good
    assert good["mask"] == [1, 2, 3] and len(good["pose"]) == 16 and good["written_executed"]
    assert good["own_ids"] == tok.encode("<action>move_forward|</action>") + [end]
    # one step from the goal, two forwards written, the env stops at the goal after the first (success mid-turn):
    # every executed action was optimal, but the answer also wrote one the env skipped, so it is not kept as its own
    # label (its loss would teach the overshoot); the guide's label from the same pose is the one forward
    from data_gen.interiorgs_v10.difficulty import to_c2w

    one_off = to_c2w(*_pose(["move_forward"]))[0]
    over = run(["move_forward", "move_forward"], ["move_forward"], "<action>move_forward|move_forward|</action>",
               start=one_off)
    assert over["status"] == "ok" and over["hits"] == [True] and over["written"] == ["move_forward", "move_forward"]
    assert not over["written_executed"] and not over["optimal"] and over["label_actions"] == ["move_forward"]
    six = run(["move_forward"] * 5, ["move_forward"] * 2, "<action>" + "move_forward|" * 6 + "</action>")
    assert not six["optimal"]  # the parser keeps five, the env ran two: never kept
    # the env read and executed it, but not in the labels' form: replaced (a capital letter here)
    caps = run(["move_forward"], ["move_forward"], "<action>Move_forward|</action>")
    assert caps["hits"] == [True] and not caps["canonical"] and not caps["optimal"]
    assert caps["label_text"] == "<action>move_forward|move_forward|</action>"
    # one step left in the budget: the label is at most one action
    short = run(["turn_left"], ["turn_left"], "<action>turn_left|</action>", budget=1)
    assert short["label_actions"] == ["move_forward"]
    bad = run(["turn_left"], ["turn_left"], "<action>turn_left|</action>")
    assert bad["status"] == "ok" and not bad["optimal"] and bad["label_actions"] == ["move_forward", "move_forward"]
    assert bad["label_text"] == "<action>move_forward|move_forward|</action>"
    assert tok.decode(bad["label_ids"]) == bad["label_text"] + "<|im_end|>"
    assert tok.decode([bad["label_ids"][j] for j in bad["mask"]]) == "move_forward|move_forward|</action>"
    invalid = run([], [], "garbage")
    assert invalid["status"] == "ok" and not invalid["optimal"] and not invalid["valid"]
    assert invalid["label_actions"] == ["move_forward", "move_forward"]
    # the env executed fewer actions than the answer wrote (the step budget, say): judged on what ran, replaced
    odd = run(["move_forward", "move_forward"], ["move_forward"], "<action>move_forward|move_forward|</action>")
    assert odd["status"] == "ok" and odd["n_steps"] == 1 and odd["executed"] == ["move_forward"]
    assert not odd["optimal"] and odd["label_actions"] == ["move_forward", "move_forward"]
    assert D.written_actions("x<action> Move_Forward |turn_left||</action>y") == ["move_forward", "turn_left"]
    assert D.written_actions("no block") is None and D.written_actions("<action>move_left|") is None
    nt = asyncio.run(D.after_turn(cfg, types.SimpleNamespace(_inner=Inner("other", START.copy())),
                                  {"pose": START, "step": 0}, {}, (0, 1), [end], tok, 3))
    assert nt["status"] == "no_table" and nt["t"] == 3
    assert asyncio.run(D.after_turn(cfg, object(), {"pose": START, "step": 0}, {}, (0, 1), [end], tok, 0)
                       )["status"] == "no_table"  # an env without a task
    inner = Inner("task1", START.copy())
    inner.view_engine = None  # reading the pose after the turn fails
    broken = asyncio.run(D.after_turn(cfg, types.SimpleNamespace(_inner=inner), {"pose": START, "step": 0}, {},
                                      (0, 1), [end], tok, 0))
    assert broken["status"] == "error" and "AttributeError" in broken["error"]  # recorded, not raised


# ------------------------------------------------------------------------------------------ samples

VS, PAD, VE = 1, 2, 3


def _mrope(tokens):
    """Qwen-like M-RoPE rows: text +1 in all three; an image's pad tokens on a 2 x 2 grid, then past its extent."""
    rows, cur, i = [[], [], []], 0, 0
    while i < len(tokens):
        if tokens[i] == PAD:
            n = 0
            while i + n < len(tokens) and tokens[i + n] == PAD:
                n += 1
            for k in range(n):
                rows[0].append(cur)
                rows[1].append(cur + k // 2)
                rows[2].append(cur + k % 2)
            cur += 2
            i += n
        else:
            for r in rows:
                r.append(cur)
            cur += 1
            i += 1
    return rows


def _row():
    """An episode row: prompt (left-padded) with one image, turn 0 (own answer), an observation with an image, turn 1."""
    n_prompt, n_resp = 16, 24
    prompt = [40, 41, VS, PAD, PAD, PAD, PAD, VE, 42, 43]
    turn0 = [50, 51, 52]
    obs = [60, VS, PAD, PAD, PAD, PAD, VE, 61]
    turn1 = [70, 71, 72]
    resp = turn0 + obs + turn1
    ids = [0] * (n_prompt - len(prompt)) + prompt + resp + [0] * (n_resp - len(resp))
    att = [0] * (n_prompt - len(prompt)) + [1] * (len(prompt) + len(resp)) + [0] * (n_resp - len(resp))
    valid = [t for t, a in zip(ids, att) if a]
    m = _mrope(valid)
    pos = torch.ones(4, len(ids), dtype=torch.long)
    vi = [i for i, a in enumerate(att) if a]
    pos[0, vi] = torch.arange(len(vi))
    for r in range(3):
        pos[r + 1, vi] = torch.tensor(m[r])
    row = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att), "position_ids": pos,
           "responses": torch.tensor(ids[n_prompt:]), "old_log_probs": torch.randn(n_resp),
           "multi_modal_inputs": {"pixel_values": torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3),
                                  "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]])}}
    spans = {"t0": (0, 3), "t1": (len(turn0) + len(obs), len(resp))}
    return row, spans


TOKENS = {"vision_start": VS, "image_pad": PAD}


def test_build_sample_own_and_label():
    row, spans = _row()
    own0 = D.build_sample(row, {"t": 0, "span": list(spans["t0"]), "optimal": True, "mask": [1],
                                "own_ids": [50, 51, 52]}, TOKENS, 2)
    assert own0["checks"] == {"positions_same": True, "tokens_same": True, "image_tokens_same": True}
    wrong = D.build_sample(row, {"t": 0, "span": list(spans["t0"]), "optimal": True, "mask": [1],
                                 "own_ids": [50, 51]}, TOKENS, 2)
    assert wrong["checks"]["tokens_same"] is False  # the agent loop's answer is not the row's tokens at the span
    assert own0["input_ids"][0, -3:].tolist() == [50, 51, 52] and own0["responses"][0].tolist() == [50, 51, 52]
    assert own0["meta"]["images"] == 1 and own0["multi_modal_inputs"][0]["pixel_values"].shape == (16, 3)
    assert torch.equal(own0["old_lp"][0], row["old_log_probs"][0:3]) and own0["mask"][0].tolist() == [0, 1, 0]
    assert not own0["meta"]["whole_row"]
    own1 = D.build_sample(row, {"t": 1, "span": list(spans["t1"]), "optimal": True, "mask": [0, 1],
                                "own_ids": [70, 71, 72]}, TOKENS, 2)
    assert own1["checks"]["positions_same"] and own1["checks"]["image_tokens_same"]
    assert own1["meta"]["images"] == 2 and own1["multi_modal_inputs"][0]["pixel_values"].shape == (32, 3)
    assert own1["meta"]["whole_row"]  # prefix + own answer = the whole row
    # the sample's tokens and positions are the row's valid ones, in order
    valid = row["attention_mask"].bool()
    assert torch.equal(own1["input_ids"][0], row["input_ids"][valid])
    assert torch.equal(own1["position_ids"][0], row["position_ids"][:, valid])
    lab = D.build_sample(row, {"t": 1, "span": list(spans["t1"]), "optimal": False, "label_ids": [80, 81, 82, 83, 84],
                               "mask": [1, 2, 3], "label_text": "x"}, TOKENS, 2)
    assert lab["kind"] == "label" and "old_lp" not in lab and lab["responses"][0].tolist() == [80, 81, 82, 83, 84]
    last = own1["position_ids"][0, :, -4]  # the last prefix token's positions, then +1 a label token in every row
    assert torch.equal(lab["position_ids"][0, :, -5:], last[:, None] + torch.arange(1, 6)[None, :])
    assert lab["mask"][0].tolist() == [0, 1, 1, 1, 0]
    with pytest.raises(ValueError):
        D.build_sample(row, {"t": 2, "span": [20, 30], "optimal": True, "mask": [0]}, TOKENS, 2)
    bad_pixels = dict(row, multi_modal_inputs={"pixel_values": row["multi_modal_inputs"]["pixel_values"][:16],
                                               "image_grid_thw": torch.tensor([[1, 2, 2]])})
    s = D.build_sample(bad_pixels, {"t": 0, "span": list(spans["t0"]), "optimal": True, "mask": [1],
                                    "own_ids": [50, 51, 52]}, TOKENS, 2)
    assert s["checks"]["image_tokens_same"] is False  # 4 pad tokens, 4 patches: counted, not hidden


class FakeBatch:
    def __init__(self, rows, turns, uids):
        self.batch = {k: torch.stack([r[k] for r in rows]) for k in ("input_ids", "attention_mask", "position_ids",
                                                                       "responses", "old_log_probs")}
        nt = {"uid": np.array(uids, dtype=object), "dagger_turns": np.empty(len(rows), dtype=object),
              "multi_modal_inputs": np.empty(len(rows), dtype=object)}
        nt["dagger_turns"][:] = turns
        nt["multi_modal_inputs"][:] = [r["multi_modal_inputs"] for r in rows]
        self.non_tensor_batch = nt
        self.meta_info = {}

    def __len__(self):
        return len(self.batch["input_ids"])


def _ep(spans, optimal0=True, own_ids=(50, 51, 52)):
    t0 = {"t": 0, "span": list(spans["t0"]), "status": "ok", "optimal": optimal0, "mask": [1], "executed": ["a"],
          "hits": [optimal0], "valid": True, "retok_same": True, "own_ids": list(own_ids), "written": ["a"],
          "written_executed": True}
    if not optimal0:
        t0.update(label_ids=[80, 81], mask=[0], label_text="x")
    t1 = {"t": 1, "span": list(spans["t1"]), "status": "ok", "optimal": False, "label_ids": [80, 81, 82],
          "mask": [1], "executed": [], "hits": [], "valid": False}
    return {"task_id": "k", "turns": [t0, t1, {"t": 2, "status": "unknown_mid"}]}


def test_build_samples_pick_and_attach():
    row, spans = _row()
    batch = FakeBatch([row] * 4, [_ep(spans), _ep(spans, False), _ep(spans), _ep(spans)], ["u1", "u1", "u2", "u2"])
    cfg = D.settings({"tables": "/t", "cap": 5})
    samples, m = D.build_samples(batch, cfg, 7, 2, TOKENS, 2)
    assert len(samples) == 4 and m["dagger/samples"] == 4.0 and m["dagger/candidates"] == 8.0
    assert m["dagger/turns"] == 12.0 and m["dagger/turns_judged"] == 8.0 and m["dagger/status_unknown_mid"] == 4.0
    assert m["dagger/optimal_share"] == 3 / 8 and m["dagger/replaced_share"] == 5 / 8 and m["dagger/invalid_turns"] == 4
    assert m["dagger/check_positions_same_failed"] == 0.0 and m["dagger/check_image_tokens_same_failed"] == 0.0
    again, _ = D.build_samples(batch, cfg, 7, 2, TOKENS, 2)  # the same pick from the same seed and update
    assert [(s["meta"]["uid"], s["meta"]["t"], s["kind"]) for s in again] == \
           [(s["meta"]["uid"], s["meta"]["t"], s["kind"]) for s in samples]
    only = D.build_samples(batch, D.settings({"tables": "/t", "turns": "replaced", "cap": 96}), 7, 2, TOKENS, 2)[0]
    assert {s["kind"] for s in only} == {"label"} and len(only) == 4
    D.attach(batch, samples, 2)
    field = batch.non_tensor_batch["dagger_samples"]
    assert [len(x) if x else 0 for x in field] == [2, 0, 2, 0] and batch.meta_info["dagger_samples"] == 4
    with pytest.raises(ValueError):
        D.attach(batch, samples[:3], 2)
    D.detach(batch)
    assert "dagger_samples" not in batch.non_tensor_batch and "dagger_samples" not in batch.meta_info
    with pytest.raises(RuntimeError):  # the switch on but no records from the agent loop: never plain RL unnoticed
        D.build_samples(types.SimpleNamespace(non_tensor_batch={}, batch={}), cfg, 1, 2, TOKENS, 2)
    empty = types.SimpleNamespace(non_tensor_batch={"dagger_turns": np.array([None, None], dtype=object)}, batch={})
    with pytest.raises(RuntimeError):
        D.build_samples(empty, cfg, 1, 2, TOKENS, 2)
    out_all = FakeBatch([row], [{"task_id": "k", "turns": [{"t": 0, "status": "pose_mismatch"}]}], ["u"])
    with pytest.raises(RuntimeError):  # every turn left out, whatever the reason
        D.build_samples(out_all, cfg, 1, 1, TOKENS, 2)
    # a sample failing an exact check is counted and left out (an own answer whose ids are not the row's)
    odd = FakeBatch([row] * 2, [_ep(spans, own_ids=(9, 9)), _ep(spans)], ["u1", "u2"])
    got, mo = D.build_samples(odd, D.settings({"tables": "/t", "turns": "all", "cap": 96}), 3, 1, TOKENS, 2)
    assert mo["dagger/check_tokens_same_failed"] == 1.0 and mo["dagger/dropped_by_check"] == 1.0 and len(got) == 3
    # no turn judged because no table was found anywhere: the run stops instead of training plain RL unnoticed
    lost = FakeBatch([row] * 2, [{"task_id": "k", "turns": [{"t": 0, "status": "no_table"}]}] * 2, ["u1", "u2"])
    with pytest.raises(RuntimeError):
        D.build_samples(lost, cfg, 1, 2, TOKENS, 2)
    over = FakeBatch([row], [{"task_id": "k", "turns": [dict(_ep(spans)["turns"][0], optimal=False, written=["a", "a"],
                                                              written_executed=False, label_ids=[80], mask=[0])]}], ["u"])
    assert D.build_samples(over, cfg, 1, 1, TOKENS, 2)[1]["dagger/own_unexecuted"] == 1.0


# ------------------------------------------------------------------------------------------ the mixer

def test_pack_pair():
    row, spans = _row()
    a = D.build_sample(row, {"t": 0, "span": list(spans["t0"]), "optimal": True, "mask": [1],
                             "own_ids": [50, 51, 52]}, TOKENS, 2)
    b = D.build_sample(row, {"t": 1, "span": list(spans["t1"]), "optimal": False, "label_ids": [80, 81, 82, 83],
                             "mask": [1], "label_text": "x"}, TOKENS, 2)
    mb, n_a = D.pack_pair(a, b)
    la, lb = a["input_ids"].shape[1], b["input_ids"].shape[1]
    assert n_a == 3 and mb["input_ids"].shape == (2, max(la, lb)) and mb["responses"].shape == (2, 4)
    assert torch.equal(mb["input_ids"][0, -la:], a["input_ids"][0]) and int(mb["attention_mask"][0].sum()) == la
    assert torch.equal(mb["position_ids"][1, :, -lb:], b["position_ids"][0])
    mm = mb["multi_modal_inputs"][0]  # one dict, row 0's images first
    assert torch.equal(mm["pixel_values"], torch.cat([a["multi_modal_inputs"][0]["pixel_values"],
                                                      b["multi_modal_inputs"][0]["pixel_values"]]))
    assert mm["image_grid_thw"].shape[0] == a["meta"]["images"] + b["meta"]["images"]


class ToyActor:
    """The actor's forward on a toy model: log-probabilities of the last len(responses) tokens."""

    def __init__(self, vocab=100):
        torch.manual_seed(0)
        self.actor_module = torch.nn.Embedding(vocab, vocab)
        self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
        self.scaler = None
        self.use_ulysses_sp = False

    def _forward_micro_batch(self, mb, temperature, calculate_entropy=False):
        ids = mb["input_ids"]
        logits = self.actor_module(ids) / temperature
        lp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:, None])[..., 0]
        return None, lp[:, -mb["responses"].shape[1]:]


def _samples(actor, row, spans):
    own = D.build_sample(row, {"t": 1, "span": list(spans["t1"]), "optimal": True, "mask": [1, 2],
                               "own_ids": [70, 71, 72]}, TOKENS, 2)
    with torch.no_grad():  # the "recorded" log-probabilities: the same forward, so the check must read 0
        own["old_lp"] = actor._forward_micro_batch(own, 1.0)[1].float()
    lab = D.build_sample(row, {"t": 0, "span": list(spans["t0"]), "optimal": False, "label_ids": [80, 81, 82],
                               "mask": [1, 2], "label_text": "x"}, TOKENS, 2)
    for s in (own, lab):  # the toy model takes no pixels
        s["multi_modal_inputs"] = [{}]
    return [own, lab]


def _data(samples):
    field = np.empty(2, dtype=object)
    field[:] = [samples, None]
    return types.SimpleNamespace(non_tensor_batch={"dagger_samples": field},
                                 meta_info={"dagger_samples": len(samples), "temperature": 1.0, "dagger_step": 5})


@pytest.mark.parametrize("audit", [True, False])
def test_mixer_adds_or_only_measures(tmp_path, audit):
    actor = ToyActor()
    row, spans = _row()
    samples = _samples(actor, row, spans)
    mixer = D.DaggerMixer({"tables": "/t", "lam": 0.5, "audit_only": audit, "log_dir": str(tmp_path)}, actor)
    p = next(actor.actor_module.parameters())
    policy = torch.full_like(p, 0.01)
    p.grad = policy
    # the expected supervised gradient: lam x the token mean of the NLL over the loss tokens
    ref = ToyActor()
    ref.actor_module.load_state_dict(actor.actor_module.state_dict())
    n_tok = sum(float(s["mask"].sum()) for s in samples)
    loss = sum(-(ref._forward_micro_batch(s, 1.0)[1] * s["mask"]).sum() for s in samples) * 0.5 / n_tok
    loss.backward()
    sup = next(ref.actor_module.parameters()).grad
    mixer.take_batch(_data(samples))
    out = mixer.before_optimizer_step()
    if audit:
        assert p.grad is policy and torch.equal(p.grad, torch.full_like(p, 0.01))  # dropped after the measurement
        assert out["actor/dagger_inserted"] == 0.0 and out["actor/dagger_audit"] == 1.0
    else:
        assert torch.allclose(p.grad, policy.new_full(p.shape, 0.01) + sup, atol=1e-6)
        assert out["actor/dagger_inserted"] == 1.0
    want = float(sup.norm()) / (float(sup.norm()) + float(policy.norm()))
    assert out["actor/dagger_grad_mag_share_und"] == pytest.approx(want, rel=1e-5)
    assert out["actor/dagger_samples"] == 2.0 and out["actor/dagger_samples_own"] == 1.0
    assert out["actor/dagger_check_whole_row_max_abs"] == pytest.approx(0.0, abs=1e-6)
    check = json.loads((tmp_path / "check.jsonl").read_text().splitlines()[0])
    assert check["whole_row"]["tokens"] == 3 and check["prefix"]["tokens"] == 0 and check["loss_tokens"]["tokens"] == 2
    assert check["step"] == 5
    # the floor (each sample again beside another in a two-row batch) only in the measure-only mode; the toy model
    # has no attention, so a correct packing changes nothing
    assert check["shape_floor"]["tokens"] == (6 if audit else 0)
    if audit:
        assert check["shape_floor"]["max_abs"] == pytest.approx(0.0, abs=1e-6)
    steps = json.loads((tmp_path / "steps.jsonl").read_text().splitlines()[0])
    assert steps["step"] == 5 and "nan" not in json.dumps(out).lower()
    first = json.loads((tmp_path / "first_batch_rank0_step0005.json").read_text())["samples"]
    assert [s["kind"] for s in first] == ["own", "label"] and first[1]["loss_ids"] == [81, 82]
    # a second optimizer step in the same update gets nothing (no metrics either); the end of the update resets
    assert mixer.before_optimizer_step() == {}
    end = mixer.end_of_update()
    assert end["actor/dagger_update_opt_steps"] == 2.0 and end["actor/dagger_batches"] == float(not audit)


def test_mixer_skips_together_on_uneven_or_none(tmp_path):
    actor = ToyActor()
    row, spans = _row()
    samples = _samples(actor, row, spans)
    mixer = D.DaggerMixer({"tables": "/t", "log_dir": str(tmp_path)}, actor)
    p = next(actor.actor_module.parameters())
    p.grad = torch.zeros_like(p)
    data = _data(samples)
    data.meta_info["dagger_samples"] = 3  # this rank holds 2, the trainer said 3 over 1 rank
    mixer.take_batch(data)
    out = mixer.before_optimizer_step()
    assert out["actor/dagger_inserted"] == 0.0 and out["actor/dagger_dropped"] == 1.0
    mixer.end_of_update()
    mixer.take_batch(_data([]))
    assert mixer.before_optimizer_step()["actor/dagger_dropped"] == 0.0
