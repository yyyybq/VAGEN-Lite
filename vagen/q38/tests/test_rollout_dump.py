"""T3 (DESIGN_ASYNC 5.1): the rollout dump.

Three synthetic groups go through the sampler (in-memory TransferQueue, a fake tokenizer) into
``rollout_data/<s>.jsonl`` and ``rollout_data/filter/<s>.jsonl``; read back: the 8B dump's 16 keys and the q38 keys
on every row, only candidate episodes (no padding rows), kept_for_update by group, the decision table; the training
chain's readers (data_gen/interiorgs_v10/train_chain.py family_success, block_stats, Chain.signal_check, imported
read only) read the dumps.  A writer that raises half way leaves no partial file.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampler_fixtures as fx  # noqa: E402

from vagen.q38 import rollout_dump as rd  # noqa: E402

GROUPS = 3
N = 4
FILTER = {"enable": True, "name": "mixed_outcome", "filter_kwargs": {"target_groups": 2}}
# group 0 mixed, group 1 all success, group 2 mixed: target 2 -> nominal (0, 1), refill 2
OUTCOMES = [([5.0, -0.1, 5.0, -0.2], [1, 0, 1, 0]), ([5.0] * 4, [1] * 4), ([4.9, -0.3, -0.1, 5.0], [1, 0, 0, 1])]
PROMPT = [1, 3, 10, 11, 11, 11, 12, 4, 2]  # <|im_start|>user\n<image>hi<|im_end|>
RESPONSE = [5, 6, 7, 2, 13, 8]  # <action>move_forward|</action><|im_end|>\n<|endoftext|>


def _dump_steps(monkeypatch, directory: str, steps=(1, 2, 3)) -> dict[int, list[str]]:
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler(filter_cfg=FILTER, candidates=GROUPS, n=N, rollout_data_dir=directory,
                              model_path="unused-fake-tokenizer")
    sampler._tokenizer = fx.FakeTokenizer()
    out = {}
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=GROUPS)
    for s in steps:
        nxt = fx.dispatch(fake, draw_batch=s, global_steps=s, n_groups=GROUPS)
        for j, u in enumerate(w):
            fx.finish_group(fake, u, *OUTCOMES[j], global_steps=s, versions=(s - 1, s - 1),
                            task_id=f"task{s}-{j}", task_type=("fov_inclusion", "equidistance", "occlusion")[j],
                            prompt_ids=PROMPT, response_ids=RESPONSE)
        batch, _m = sampler.sample(global_steps=s, partition_id="train", batch_size=GROUPS)
        fake.kv_clear(batch.keys, "train")
        sampler.clear_adv_scale()
        out[s] = w
        w = nxt
    return out


def test_dump_rows_and_decision_table(monkeypatch):
    with fx.work_dir("q38-dump") as d:
        drawn = _dump_steps(monkeypatch, d)
        for s, uids in drawn.items():
            rows = rd.read_jsonl(os.path.join(d, f"{s}.jsonl"))
            assert len(rows) == GROUPS * N  # every candidate, nothing else
            for r in rows:
                assert set(rd.KEYS_8B) <= set(r), sorted(set(rd.KEYS_8B) - set(r))
                assert set(rd.Q38_KEYS) <= set(r)
                assert list(r)[:5] == list(rd.BASE_KEYS)  # the 8B key order starts as RT:1698-1704
                assert list(r).index("kept_for_update") == len(rd.KEYS_8B) - 1
                assert r["step"] == s and r["draw_batch"] == s - 1 and r["uid"] in uids
                assert r["gts"] is None
                assert r["input"] == "<|im_start|>user\n<image>hi<|im_end|>"
                assert r["output"] == "<action>move_forward|</action><|im_end|>\n<|endoftext|>"  # specials kept
                assert (r["min_global_steps"], r["max_global_steps"]) == (s - 1, s - 1)
            by_group = {}
            for r in rows:
                by_group.setdefault(r["uid"], set()).add(r["kept_for_update"])
            assert {u: v for u, v in by_group.items()} == {uids[0]: {True}, uids[1]: {False}, uids[2]: {True}}
            assert [r["score"] for r in rows if r["uid"] == uids[2]] == [
                float(torch.tensor(x, dtype=torch.float32)) for x in OUTCOMES[2][0]]
            table = rd.read_jsonl(os.path.join(d, "filter", f"{s}.jsonl"))
            assert [t["uid"] for t in table] == uids
            assert [t["kind"] for t in table] == ["mixed", "all_success", "mixed"]
            assert [t["selected"] for t in table] == [True, False, True]
            assert [t["adv_scale"] for t in table] == [1.0, None, 1.0]
            assert all(set(rd.GROUP_KEYS) <= set(t) for t in table)
            assert table[1]["return_std"] == 0.0 and table[0]["return_std"] > 0
            assert [t["task_id"] for t in table] == [f"task{s}-{j}" for j in range(GROUPS)]
            assert not [f for f in os.listdir(d) if f.endswith(".tmp")]


def test_training_chain_reads_the_dumps(monkeypatch):
    from data_gen.interiorgs_v10 import train_chain as tc

    with fx.work_dir("q38-dump") as d:
        run_dir = Path(d) / "run"
        rollout = run_dir / "rollout_data"
        rollout.mkdir(parents=True)
        _dump_steps(monkeypatch, str(rollout))
        fam = tc.family_success(rollout, block=3)
        assert fam["1-3"]["fov_inclusion"] == {"success": 0.5, "episodes": 12}
        assert fam["1-3"]["equidistance"] == {"success": 1.0, "episodes": 12}
        log = "\n".join(f"step:{s} - transition/invalid_action_rate/mean:0.1 - filter/mixed/kept:2" for s in (1, 2, 3))
        blocks = tc.block_stats(rollout, log, block=3)["1-3"]
        assert blocks["updates"] == 3 and blocks["episodes"] == 36 and blocks["tasks"] == 9
        assert blocks["mixed_share"] == round(6 / 9, 4) and blocks["kept"] == 2.0
        stub = types.SimpleNamespace(plan={"rl": {"signal_check": {"at": 3, "mixed_min": 0.5}}}, state={},
                                     save=lambda: None, sft_model=lambda: "sft")
        tc.Chain.signal_check(stub, {"r": {"run_dir": run_dir}})
        assert stub.state["signal_checks"][0]["ok"] is True
        assert stub.state["signal_checks"][0]["runs"]["r"]["episodes"] == 36
        assert not any((rollout / "filter").glob("*.tmp"))


def test_no_partial_file_when_the_writer_raises(monkeypatch):
    with fx.work_dir("q38-dump") as d:
        rows = [{"input": "a", "output": "b", "score": 1.0}]
        rd.write_step_dump(d, 5, rows, [{"uid": "u"}])
        before = open(os.path.join(d, "5.jsonl"), encoding="utf-8").read()

        class Unprintable:
            def __str__(self):
                raise RuntimeError("cannot serialise")

        with pytest.raises(RuntimeError, match="cannot serialise"):
            rd.write_step_dump(d, 6, [{"x": Unprintable()}], [{"uid": "u"}])
        assert not os.path.exists(os.path.join(d, "6.jsonl"))
        assert not os.path.exists(os.path.join(d, "filter", "6.jsonl"))

        def broken_fsync(fd):
            raise OSError("disk went away")

        monkeypatch.setattr(rd.os, "fsync", broken_fsync)
        with pytest.raises(OSError, match="disk went away"):
            rd.write_step_dump(d, 5, [{"input": "new"}], [{"uid": "v"}])
        monkeypatch.undo()
        assert open(os.path.join(d, "5.jsonl"), encoding="utf-8").read() == before  # the old file is intact
        leftovers = [p for p in Path(d).rglob("*.tmp")]
        assert not leftovers, leftovers


def test_episode_rows_order_and_checks():
    rows = rd.episode_rows(
        7, ["in"], ["out"], [1.5], [{"traj_success": 1.0, "task_id": "t", "score": 99}], [True],
        [{"uid": "u", "draw_batch": 6, "draw_index": 0, "min_global_steps": 5, "max_global_steps": 6}])
    assert list(rows[0]) == ["input", "output", "gts", "score", "step", "traj_success", "task_id",
                             "kept_for_update", *rd.Q38_KEYS]
    assert rows[0]["score"] == 1.5  # a reward_extra_info key never overwrites a base key
    with pytest.raises(ValueError, match="scores"):
        rd.episode_rows(1, ["a", "b"], ["x", "y"], [1.0], [{}, {}], [True, True], [{}, {}])


def test_decode_rows_padded_and_unpadded():
    tok = fx.FakeTokenizer()
    assert rd.decode_rows(tok, [torch.tensor([0, 0, 1, 4])], padded=True, side="left") == ["<|im_start|>hi"]
    assert rd.decode_rows(tok, [torch.tensor([5, 2, 0, 0])], padded=True, side="right") == ["<action><|im_end|>"]
    # unpadded rows are decoded as they are: a pad id inside or at the end stays
    assert rd.decode_rows(tok, [[4, 0]], padded=False) == ["hi<pad>"]
    assert rd.decode_rows(tok, [[10, 11, 11, 12, 13, 10, 11, 12]]) == ["<image>\n<image>"]
    assert rd.decode_rows(tok, [[10, 11, 12]], replace_image_tokens=False) == [
        "<|vision_start|><|image_pad|><|vision_end|>"]
    assert rd.unpad([0, 1, 0], 0, "left") == [1, 0] and rd.unpad([0, 1, 0], None, "right") == [0, 1, 0]
