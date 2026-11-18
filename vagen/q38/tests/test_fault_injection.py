"""Smoke S5 fault injection (vagen/q38/fault_injection.py; DESIGN_ASYNC 5.2 S5), on the CPU.

One test group per injection, plus the switch itself:
  - the switch: off by default (absent / enable false); malformed sections refused; the contract keeps it off outside
    a smoke run (EXPECTED vagen.fault_injection.enable, FORBIDDEN fault_injection_config which cannot be overridden,
    FORBIDDEN stuck_warn_below_floor);
  - (1) env.step raises at turn T of the hit episodes: the episode ends as done (termination env_error, reward 0 for
    that turn), vagen_stats.step_exceptions = 1 and the sampler's metrics count it;
  - (2) generate raises at turn T: the exception leaves the agent loop (AgentLoopWorkerTQ marks the group failure; the
    sampler's raise with the uid is tests/test_sampler.py test_failure_policy_raise; S3b saw it in a real run);
  - (3) one <|image_pad|> more than the images: "prompt" (in the first image run) and "answer" (a stray pad, the 8B
    drift).  Recorded here: what the vLLM server receives after verl's dedup, and the error AgentLoopWorker's
    post-processing raises (verl's _compute_position_ids -> transformers get_rope_index), checked against the
    training chain's rule 91 (data_gen/interiorgs_v10/train_chain.py VISION_TOKEN_CRASHES);
  - (4) the sampler's stuck warning at the S5 thresholds (min_s 1, factor 0, repeat_s 30): warnings counted in
    filter/step/stuck_warnings, the step returns, the next step goes on; the text matches train_chain SAMPLER_STUCK.

Run (login node, DESIGN_ASYNC 5 rules):
  cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38 && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=$PWD nice -n 19 /mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python -B -m pytest \
    vagen/q38/tests/test_fault_injection.py -q -x -p no:cacheprovider
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import uuid

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampler_fixtures as fx  # noqa: E402

from vagen.q38 import contract as C  # noqa: E402
from vagen.q38 import fault_injection as fi  # noqa: E402
from vagen.q38.candidate_metrics import stats_metrics  # noqa: E402
from vagen.q38.gym_agent_loop_v1 import VAGEN_STATS_KEYS, GymAgentLoopV1  # noqa: E402
from vagen.q38.tests import test_agent_loop as AL  # noqa: E402
from vagen.q38.tests import test_contract as TC  # noqa: E402
from vagen.q38.tests import test_sampler as TS  # noqa: E402

try:
    from data_gen.interiorgs_v10 import train_chain
except Exception:  # pragma: no cover - the chain is in the same worktree
    train_chain = None


def rule_91(line: str) -> bool:
    assert train_chain is not None, "data_gen.interiorgs_v10.train_chain not importable"
    return any(p.search(line) for p in train_chain.VISION_TOKEN_CRASHES)


@pytest.fixture(autouse=True)
def _fake_env_registered(monkeypatch):
    monkeypatch.setitem(GymAgentLoopV1._env_classes, "FakeEnv", AL.FakeEnv)
    AL.FakeEnv.closed = []
    yield


def fi_config(**section):
    return AL.make_config(vagen={"fault_injection": {"enable": True, **section}})


def run_row(model, server, cfg, session_id=0, uid=None, **row_kw):
    """One episode with the row's session_id / uid set (AL.run_episode always uses session 0, uid{seed})."""
    kwargs = AL.row(**row_kw)
    kwargs["session_id"] = session_id
    if uid is not None:
        kwargs["uid"] = uid

    async def go():
        loop = AL.make_loop(model, server, cfg)
        out = await loop.run(dict(AL.SAMPLING), **kwargs)
        return loop, out

    loop, out = asyncio.run(go())
    return loop, out, kwargs


# ------------------------------------------------------------------------------------------------------------
# the switch
# ------------------------------------------------------------------------------------------------------------


def test_off_by_default():
    assert fi.plan_of(AL.make_config()) is None  # no key
    assert fi.plan_of(AL.make_config(vagen={"fault_injection": None})) is None
    assert fi.plan_of(AL.make_config(vagen={"fault_injection": {"enable": False}})) is None
    assert fi.plan_of({"vagen": {"fault_injection": {"enable": False, "stuck_warn": False,
                                                      "env_step_exception": None}}}) is None
    # the smoke example (and so every config built from it) has no section
    assert fi.section_of(OmegaConf.load(TC.SMOKE)) is None


@pytest.mark.parametrize("section,needle", [
    ({"enable": False, "env_step_exception": {"turn": 2}}, "set while enable is not true"),
    ({"enable": False, "stuck_warn": True}, "set while enable is not true"),
    ({"enable": True}, "no injection is set"),
    ({"enable": "yes", "stuck_warn": True}, "must be true or false"),
    ({"enable": True, "env_step": {"turn": 2}}, "unknown keys"),
    ({"enable": True, "env_step_exception": {"turn": 0}}, "turn must be an integer >= 1"),
    ({"enable": True, "env_step_exception": {"every_n_groups": 0}}, "every_n_groups must be an integer >= 1"),
    ({"enable": True, "env_step_exception": {"turn": 2, "where": "prompt"}}, "unknown keys"),
    ({"enable": True, "env_step_exception": 3}, "must be a mapping"),
    ({"enable": True, "image_pad_extra": {"where": "middle"}}, "where must be one of"),
    ({"enable": True, "image_pad_extra": {"where": "prompt", "turn": 2}}, "turn must be 1"),
    ({"enable": True, "stuck_warn": "yes"}, "stuck_warn must be true or false"),
])
def test_malformed_sections(section, needle):
    _plan, problems = fi.parse(section)
    assert any(needle in p for p in problems), problems
    with pytest.raises(ValueError, match=re.escape(needle)):
        fi.plan_of({"vagen": {"fault_injection": section}})


def test_plan_and_group_selection():
    plan = fi.plan_of(fi_config(env_step_exception={"every_n_groups": 4, "session": 0, "turn": 2},
                                image_pad_extra={"where": "answer", "turn": 3}, stuck_warn=True))
    assert plan.env_step_exception == fi.Target(4, 0, 2) and plan.generate_exception is None
    assert plan.image_pad_extra == fi.Target(1, 0, 3, "answer") and plan.stuck_warn
    uids = [uuid.UUID(int=i).hex for i in range(4000)]
    hit = [u for u in uids if fi.group_hit(u, 4)]
    assert 0.2 < len(hit) / len(uids) < 0.3  # about one group in four
    assert all(fi.group_hit(u, 4) for u in hit) and all(fi.group_hit(u, 1) for u in uids)  # stable, 1 = all
    u = hit[0]
    assert set(plan.for_episode(u, 0)) == {"env_step_exception", "image_pad_extra"}
    assert set(plan.for_episode(u, 1)) == set()  # another session of a hit group
    miss = next(x for x in uids if not fi.group_hit(x, 4))
    assert set(plan.for_episode(miss, 0)) == {"image_pad_extra"}  # every_n_groups 1


def test_insert_helpers():
    assert fi.insert_extra_pad_in_prompt([1, 9, 9, 2], 9) == ([1, 9, 9, 9, 2], 2)
    with pytest.raises(ValueError, match="needs an image"):
        fi.insert_extra_pad_in_prompt([1, 2], 9)
    stop = {7}
    assert fi.insert_stray_pad_in_answer([1, 2, 7], [-.1, -.2, -.3], 9, stop, 10) == ([1, 2, 9, 7], [-.1, -.2, -.2, -.3])
    assert fi.insert_stray_pad_in_answer([1, 2], [-.1, -.2], 9, stop, 10) == ([1, 2, 9], [-.1, -.2, -.2])
    assert fi.insert_stray_pad_in_answer([1, 2, 7], [-.1, -.2, -.3], 9, stop, 3) == ([1, 9, 7], [-.1, -.1, -.3])
    assert fi.insert_stray_pad_in_answer([1, 2, 3], [], 9, stop, 3) == ([1, 2, 9], [])  # no logprobs


class ContractTest(TC.Base):
    """The contract keeps injection off outside a smoke run."""

    S5 = {"enable": True, "env_step_exception": {"every_n_groups": 4, "session": 0, "turn": 2}, "stuck_warn": True}

    def s5(self, section=None, name="q38-smoke-4b-s5a", allow=(C.FI_ENABLE_KEY,)):
        cfg = self.base()
        TC.put(cfg, "trainer.experiment_name", name)
        TC.put(cfg, "vagen.fault_injection", dict(self.S5 if section is None else section))
        TC.put(cfg, "vagen.stuck_warn", {"min_s": 1, "factor": 0, "repeat_s": 30})
        TC.put(cfg, "vagen.contract.allow_override", list(allow))
        return cfg

    def test_s5_config_passes_with_its_override(self):
        report = self.evaluate(self.s5())
        self.assertTrue(report["passed"], C.summary(report))
        self.assertEqual(report["overridden"], [C.FI_ENABLE_KEY])
        self.assertEqual(report["info"]["vagen.fault_injection"]["enable"], True)

    def test_the_base_config_has_it_off(self):
        report = self.evaluate(self.base())
        self.assertTrue(report["passed"], C.summary(report))
        entry = next(e for e in report["expected"] if e["key"] == C.FI_ENABLE_KEY)
        self.assertTrue(entry["ok"] and entry["actual"] == "<absent>")
        cfg = self.base()
        TC.put(cfg, "vagen.fault_injection", {"enable": False})
        self.assertTrue(self.evaluate(cfg)["passed"])

    def test_enable_needs_the_override(self):
        self.assertViolates(self.s5(allow=()), C.FI_ENABLE_KEY)

    def test_only_in_a_smoke_run_and_not_overridable(self):
        self.assertViolates(self.s5(name="q38-main-v4"), "fault_injection_config")
        cfg = self.s5(name="q38-main-v4", allow=(C.FI_ENABLE_KEY, "fault_injection_config"))
        report = self.assertViolates(cfg, "fault_injection_config")
        self.assertTrue(any("cannot be overridden" in p for p in report["problems"]), report["problems"])
        self.assertIn("fault_injection_config", C.NOT_OVERRIDABLE)

    def test_malformed_section(self):
        self.assertViolates(self.s5({"enable": True, "env_step_exception": {"turn": 0}}), "fault_injection_config")
        cfg = self.base()
        TC.put(cfg, "vagen.fault_injection", {"enable": False, "generate_exception": {"turn": 1}})
        self.assertViolates(cfg, "fault_injection_config")

    def test_stuck_warn_floor(self):
        cfg = self.base()
        TC.put(cfg, "vagen.stuck_warn", {"min_s": 1, "factor": 0, "repeat_s": 30})
        self.assertViolates(cfg, "stuck_warn_below_floor")
        for key, value in (("min_s", 59), ("factor", 0.5), ("repeat_s", 10)):
            with self.subTest(key=key):
                cfg = self.base()
                TC.put(cfg, f"vagen.stuck_warn.{key}", value)
                self.assertViolates(cfg, "stuck_warn_below_floor")
        cfg = self.base()
        TC.put(cfg, "vagen.stuck_warn", {"min_s": 60, "factor": 1, "repeat_s": 60})
        self.assertNotViolates(cfg, "stuck_warn_below_floor")
        # the S5 switch allows it; injection on without the stuck_warn switch does not
        self.assertNotViolates(self.s5(), "stuck_warn_below_floor")
        self.assertViolates(self.s5({k: v for k, v in self.S5.items() if k != "stuck_warn"}), "stuck_warn_below_floor")


# ------------------------------------------------------------------------------------------------------------
# (1) env.step exception, (2) generate exception
# ------------------------------------------------------------------------------------------------------------


def test_off_leaves_the_episode_unchanged():
    model = AL.default_model()
    _l0, out0 = AL.run_episode(model, AL.FakeServer(model.tokenizer), AL.make_config(), seed=2, done_at=3)
    cfg = AL.make_config(vagen={"fault_injection": {"enable": False}})
    loop, out1 = AL.run_episode(model, AL.FakeServer(model.tokenizer), cfg, seed=2, done_at=3)
    assert loop.fault_plan is None and loop.episode.faults == {}
    assert out1.prompt_ids == out0.prompt_ids and out1.response_ids == out0.response_ids
    assert out1.extra_fields["vagen_stats"]["termination"] == "success"
    assert tuple(out1.extra_fields["vagen_stats"]) == VAGEN_STATS_KEYS  # the same keys with or without injection


def test_env_step_exception_ends_the_episode_as_done_and_is_counted():
    model = AL.default_model()
    cfg = fi_config(env_step_exception={"every_n_groups": 1, "session": 0, "turn": 2})
    loop, out, _ = run_row(model, AL.FakeServer(model.tokenizer), cfg, seed=4, done_at=6)
    stats = out.extra_fields["vagen_stats"]
    assert stats["step_exceptions"] == 1 and stats["termination"] == "env_error"
    assert stats["env_turns"] == 2 and stats["gen_turns"] == 2  # ended on the turn that raised
    assert AL.FakeEnv.closed[-1].actions == [AL.CLEAN_ANSWERS[0]]  # the env saw turn 1 only
    assert out.reward_score == pytest.approx(-0.1)  # turn 1's reward + 0 for the failed turn (GL:451-459)
    assert out.extra_fields["reward_extra_info"]["traj_success"] == 0.0
    m = stats_metrics([stats])
    assert m["vagen/env/step_exceptions"] == 1 and m["vagen/env/termination/env_error"] == 1
    # another session of the same group is not hit
    _l, out2, _ = run_row(model, AL.FakeServer(model.tokenizer), cfg, session_id=1, seed=4, done_at=6)
    assert out2.extra_fields["vagen_stats"]["step_exceptions"] == 0
    assert out2.extra_fields["vagen_stats"]["termination"] == "success"
    # an episode that ends before the turn is not hit
    _l, out3, _ = run_row(model, AL.FakeServer(model.tokenizer), cfg, seed=4, done_at=1)
    assert out3.extra_fields["vagen_stats"]["step_exceptions"] == 0


def test_injection_needs_uid_and_session():
    model = AL.default_model()
    cfg = fi_config(env_step_exception={"turn": 2})
    kwargs = AL.row(seed=1, done_at=3)
    del kwargs["uid"]

    async def go():
        return await AL.make_loop(model, AL.FakeServer(model.tokenizer), cfg).run(dict(AL.SAMPLING), **kwargs)

    with pytest.raises(ValueError, match="no uid / session_id"):
        asyncio.run(go())


def test_generate_exception_leaves_the_loop_with_the_uid():
    model = AL.default_model()
    cfg = fi_config(generate_exception={"every_n_groups": 1, "session": 0, "turn": 2})
    server = AL.FakeServer(model.tokenizer)
    with pytest.raises(fi.FaultInjected, match=r"generate_exception: generate raised on purpose \(uid=uidx7 session=0 "
                                               r"turn=2"):
        run_row(model, server, cfg, uid="uidx7", seed=7, done_at=5)
    assert len(server.calls) == 1  # turn 2 never reached the server
    assert len(AL.FakeEnv.closed) == 1  # the env is closed on the way out


# ------------------------------------------------------------------------------------------------------------
# (3) one <|image_pad|> more than the images
# ------------------------------------------------------------------------------------------------------------


def _pads(ids, pad):
    return sum(1 for t in ids if t == pad)


def _postprocess_error(model, cfg, out, kwargs):
    """AgentLoopWorkerTQ._agent_loop_postprocess's multimodal inputs and position ids (AL.tq_fields), inside a
    coroutine as in the worker: the exception as the session task reports it (None if it passes)."""

    async def post():
        return AL.tq_fields(model, cfg, out, kwargs)

    try:
        asyncio.run(post())
    except Exception as e:  # noqa: BLE001 - recorded
        return e
    return None


@pytest.mark.parametrize("name", ["qwen35_4b", "qwen38_27b"])
def test_image_pad_extra_in_the_prompt(name):
    """'prompt': the first image run has 65 pads.  verl's vLLM server collapses runs of pads to one before vLLM
    (vllm_async_server.py:643), so the rollout gets the same prompt as without the injection and goes on; the
    post-processing (verl _compute_position_ids -> transformers get_rope_index) raises the M-RoPE shape mismatch,
    which is rule 91's first pattern."""
    from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens

    model = AL.load_model(name)
    pad = AL.image_token_id(model)
    cfg = fi_config(image_pad_extra={"every_n_groups": 1, "session": 0, "turn": 1, "where": "prompt"})
    s0, s1 = AL.FakeServer(model.tokenizer), AL.FakeServer(model.tokenizer)
    _l, base = AL.run_episode(model, s0, AL.make_config(), seed=3, done_at=3)
    loop, out, kwargs = run_row(model, s1, cfg, seed=3, done_at=3)
    n_images = len(loop.episode.images)
    assert _pads(out.prompt_ids, pad) == AL.IMAGE_TOKENS_256PX + 1
    assert _pads(out.prompt_ids + out.response_ids, pad) == AL.IMAGE_TOKENS_256PX * n_images + 1
    for c0, c1 in zip(s0.calls, s1.calls, strict=True):  # what vLLM receives is unchanged
        assert qwen2_5_vl_dedup_image_tokens(c1["prompt_ids"], model.processor) == \
            qwen2_5_vl_dedup_image_tokens(c0["prompt_ids"], model.processor)
    assert _postprocess_error(model, AL.make_config(), base, AL.row(seed=3, done_at=3)) is None
    err = _postprocess_error(model, cfg, out, kwargs)
    assert isinstance(err, RuntimeError), repr(err)
    n = len(out.prompt_ids) + len(out.response_ids)
    line = f"{type(err).__name__}: {err}"
    assert line == (f"RuntimeError: shape mismatch: value tensor of shape [3, {n - 1}] cannot be broadcast to "
                    f"indexing result of shape [3, {n}]"), line
    assert rule_91(line)


@pytest.mark.parametrize("name", ["qwen35_4b", "qwen38_27b"])
def test_image_pad_extra_in_the_answer(name):
    """'answer': a stray pad in the answer of turn 1 (the 8B drift).  It survives the dedup, so vLLM gets one pad
    more than images from turn 2 on (what vLLM 0.28 does with it is for the S5 run to show); the post-processing
    raises StopIteration in get_rope_index (the grid iterator runs out), which a coroutine turns into
    "RuntimeError: coroutine raised StopIteration": rule 91 does NOT match it."""
    from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens

    model = AL.load_model(name)
    pad = AL.image_token_id(model)
    cfg = fi_config(image_pad_extra={"every_n_groups": 1, "session": 0, "turn": 1, "where": "answer"})
    server = AL.FakeServer(model.tokenizer)
    loop, out, kwargs = run_row(model, server, cfg, seed=3, done_at=3)
    assert _pads(out.response_ids[: loop.episode.turn_spans[0][1]], pad) == 1
    assert out.response_logprobs is not None and len(out.response_logprobs) == len(out.response_ids)
    deduped = [_pads(qwen2_5_vl_dedup_image_tokens(c["prompt_ids"], model.processor), pad) for c in server.calls]
    assert deduped == [c["n_images"] + (1 if i else 0) for i, c in enumerate(server.calls)]
    err = _postprocess_error(model, cfg, out, kwargs)
    line = f"{type(err).__name__}: {err}"
    assert line == "RuntimeError: coroutine raised StopIteration", line
    assert isinstance(err.__cause__, StopIteration)
    assert not rule_91(line)


# ------------------------------------------------------------------------------------------------------------
# (4) the stuck warning at the S5 thresholds
# ------------------------------------------------------------------------------------------------------------


def test_stuck_warning_at_s5_thresholds_counts_and_goes_on(monkeypatch, caplog):
    fake = fx.FakeTQ().install(monkeypatch)
    sampler = fx.make_sampler(stuck_warn={"min_s": 1, "factor": 0, "repeat_s": 30})
    w = fx.dispatch(fake, draw_batch=0, global_steps=1, n_groups=24)
    TS._finish_batch(fake, w[:20], 1, (0, 0))
    clock = {"t": 0.0, "polls": 0}
    sampler._clock = lambda: clock["t"]

    def poll(partition_id, last):
        clock["t"] += 4.0
        clock["polls"] += 1
        if clock["polls"] == 15:  # 60 s: the last groups finish
            TS._finish_batch(fake, w[20:], 1, (0, 0))
        return last

    sampler._wait_for_next_poll = poll
    logging.getLogger("vagen.q38.sampler").propagate = True
    with caplog.at_level(logging.WARNING, logger="vagen.q38.sampler"):
        _batch, m = TS._train_step(fake, sampler, 1)
    warned = [r.getMessage() for r in caplog.records if "has waited" in r.getMessage()]
    assert len(warned) == 2  # at 4 s (threshold 1 s) and 36 s (repeat 30 s)
    assert m["filter/step/stuck_warnings"] == 2 and m["filter/step/wait_s"] == 60.0
    assert all(train_chain.SAMPLER_STUCK.search(x) for x in warned), warned
    # the next step: its batch is already there, no warning, training goes on
    b1 = fx.dispatch(fake, draw_batch=1, global_steps=2, n_groups=24)
    TS._finish_batch(fake, b1, 2, (1, 1))
    fx.dispatch(fake, draw_batch=2, global_steps=2, n_groups=24)
    _batch2, m2 = TS._train_step(fake, sampler, 2)
    assert m2["filter/step/stuck_warnings"] == 0 and m2["filter/step/draw_batch"] == 1
