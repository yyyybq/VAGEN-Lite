"""T2 (DESIGN_ASYNC 5.1): the candidate metrics against the 8B logs.

For every update of the golden 8B runs (vagen/q38/tests/golden, scripts/q38/extract_golden.py), the 192 candidate
episodes in the 8B dump order go through ``candidate_metrics``; every episode/*, train/traj_success/* and
transition/* value of the 8B console line must come out (pass line: within 1e-6; with the rows in the 8B order they
are expected to be bit-identical, which is checked too), and no other name of those families may appear.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampler_fixtures as fx  # noqa: E402

from vagen.q38.candidate_metrics import (  # noqa: E402
    TERMINATION_REASONS,
    TRANSITION_METRICS,
    candidate_metrics,
    columns,
    stats_metrics,
)

FAMILIES = ("episode/", "transition/", "train/traj_success/")
RUNS = [r for r in fx.GOLDEN_RUNS if os.path.exists(fx.golden_path(r))]


@pytest.mark.parametrize("run", RUNS)
def test_golden_candidate_metrics(run):
    _header, steps = fx.load_golden(run)
    inexact = []
    for step in steps:
        _uids, returns, extra, _draw = fx.golden_episodes(step)
        ours = candidate_metrics(extra, returns)
        logged = {k: v for k, v in step["log"].items() if k.startswith(FAMILIES)}
        assert logged, f"{run} update {step['step']}: no candidate metrics in the 8B line"
        ours_f = {k: v for k, v in ours.items() if k.startswith(FAMILIES)}
        assert set(ours_f) == set(logged), (
            f"{run} update {step['step']}: names differ: only ours {sorted(set(ours_f) - set(logged))}, "
            f"only 8B {sorted(set(logged) - set(ours_f))}"
        )
        for k, v in logged.items():
            assert math.isclose(float(ours_f[k]), float(v), rel_tol=1e-6, abs_tol=1e-9), (
                f"{run} update {step['step']} {k}: ours {ours_f[k]!r}, 8B {v!r}"
            )
            if float(ours_f[k]) != float(v):
                inexact.append((step["step"], k, ours_f[k], v))
    assert not inexact, f"{run}: {len(inexact)} values within 1e-6 but not bit-identical, e.g. {inexact[:5]}"


def test_golden_spot_value():
    """DESIGN_ASYNC 5.1 T2's example: update 100 of qw3v4s3, success 0.5208333."""
    if "qw3v4s3" not in RUNS:
        pytest.skip("golden run qw3v4s3 not extracted")
    _header, steps = fx.load_golden("qw3v4s3")
    step = next(s for s in steps if s["step"] == 100)
    _uids, returns, extra, _draw = fx.golden_episodes(step)
    m = candidate_metrics(extra, returns)
    assert m["episode/success_rate"] == 0.5208333333333334 == 100 / 192
    assert m["episode/num_episodes"] == 192
    assert m["transition/final_score/mean"] == 0.5493907332420349
    assert m["transition/reward_max"] == 6.352476596832275


def test_float32_cast_and_names():
    """Values are cast to float32 before they are reduced (RT:2943); only the listed names present become
    transition metrics; ints stay ints."""
    extra = {"traj_success": [1.0, 0.0, True, 0], "final_score": [0.1, 0.2, 0.3, 0.4],
             "task_type": ["a", "b", "a", "b"], "not_a_metric": [1, 2, 3, 4]}
    returns = [5.0, -0.1, 0.30000001192092896, 2.0]
    m = candidate_metrics(extra, returns)
    arr = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert m["transition/final_score/mean"] == float(arr.mean())
    assert m["transition/final_score/std"] == float(arr.std())
    assert m["transition/final_score/count"] == 4 and isinstance(m["transition/final_score/count"], int)
    assert m["episode/success_count"] == 2.0 and m["episode/success_rate"] == 0.5
    assert m["transition/reward_min"] == float(np.float32(-0.1))
    assert not any("not_a_metric" in k for k in m)
    assert len(TRANSITION_METRICS) == 26
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in m.values())
    assert candidate_metrics({}, []) == {}


def test_columns_keeps_shared_keys_in_order():
    infos = [{"traj_success": 1.0, "task_id": "a", "x": 1}, {"traj_success": 0.0, "task_id": "b"}]
    cols = columns(infos)
    assert list(cols) == ["traj_success", "task_id"]
    assert cols["task_id"] == ["a", "b"]
    assert columns([]) == {}


def test_stats_metrics():
    stats = [
        {"step_exceptions": 1, "render_s": [0.1, 0.2], "gen_s": [1.0, 2.0], "endoftext_stops": 0, "turns_resumed": 1},
        {"step_exceptions": 0, "render_s": [0.3], "gen_s": [3.0], "endoftext_turns": 2, "resumed_turns": 0},
        None,
    ]
    m = stats_metrics(stats)
    assert m["vagen/env/episodes_with_stats"] == 2
    assert m["vagen/env/step_exceptions"] == 1
    assert m["vagen/gen/endoftext_stops"] == 2
    assert m["vagen/gen/turns_resumed"] == 1
    assert m["vagen/env/render_s_max"] == 0.3
    assert m["vagen/env/render_s_p50"] == pytest.approx(0.2)
    assert m["vagen/gen/turn_gen_s_p50"] == pytest.approx(2.0)
    assert m["vagen/gen/turn_gen_s_p95"] == pytest.approx(float(np.percentile([1.0, 2.0, 3.0], 95)))
    assert stats_metrics([None, None]) == {"vagen/env/episodes_with_stats": 0}
    assert not any(k.startswith(("vagen/env/termination", "vagen/gen/num_preempted", "vagen/gen/length_stops"))
                   for k in m)  # not carried, not logged


def _q38_stats(termination: str, preempted: int, **kw) -> dict:
    """One episode's vagen_stats as the q38 agent loop writes it (gym_agent_loop_v1.py vagen_stats)."""
    stats = {"step_exceptions": 0, "render_s": [0.2, 0.3], "turn_gen_s": [1.5, 2.5], "turn_max_tokens": [384, 384],
             "turn_spans": [[0, 10], [40, 52]], "endoftext_stops": 0, "length_stops": 0, "turns_resumed": 0,
             "num_preempted": preempted, "gen_turns": 2, "env_turns": 2, "termination": termination,
             "render_url_refreshed": 0}
    stats.update(kw)
    return stats


def test_stats_metrics_of_the_q38_agent_loop():
    stats = [_q38_stats("success", 0), _q38_stats("max_turns", 2, length_stops=1, turn_gen_s=[4.0]),
             _q38_stats("success", -1, step_exceptions=1, render_url_refreshed=1), _q38_stats("", -1),
             _q38_stats("odd_reason", 1, render_url_refreshed=1)]
    m = stats_metrics(stats)
    assert m["vagen/gen/length_stops"] == 1 and m["vagen/env/step_exceptions"] == 1
    assert m["vagen/env/render_url_refreshed"] == 2
    # stats written before the counter existed (a trajectory restored from an older checkpoint) still aggregate
    old = [{k: v for k, v in _q38_stats("success", 0).items() if k != "render_url_refreshed"}]
    assert "vagen/env/render_url_refreshed" not in stats_metrics(old)
    assert m["vagen/gen/num_preempted"] == 3  # the -1 (not reported) episodes left out
    assert m["vagen/gen/turn_gen_s_max"] == 4.0
    assert m["vagen/env/termination/success"] == 2 and m["vagen/env/termination/max_turns"] == 1
    assert m["vagen/env/termination/unset"] == 1 and m["vagen/env/termination/odd_reason"] == 1
    for reason in TERMINATION_REASONS:  # every known reason always, so the names stay the same from step to step
        assert f"vagen/env/termination/{reason}" in m
    assert m["vagen/env/termination/obs_overflow"] == 0
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in m.values())
    assert "vagen/gen/num_preempted" not in stats_metrics([_q38_stats("success", -1)])


def test_stats_keys_follow_the_agent_loop():
    """Every key of the q38 agent loop's vagen_stats is aggregated or deliberately ignored (catches a rename)."""
    try:
        from vagen.q38.gym_agent_loop_v1 import VAGEN_STATS_KEYS
    except Exception as e:  # noqa: BLE001 - the loop's imports (vLLM, verl agent loop) may be missing on a node
        pytest.skip(f"vagen.q38.gym_agent_loop_v1 does not import here: {e!r}")
    from vagen.q38 import candidate_metrics as cm

    handled = {n for _c, names, _k in cm._COUNTERS for n in names} | {n for names, _k in cm._DURATIONS for n in names}
    handled |= {cm._PREEMPTED[0], cm._TERMINATION[0]} | set(cm.IGNORED_STATS)
    assert set(VAGEN_STATS_KEYS) <= handled, sorted(set(VAGEN_STATS_KEYS) - handled)
    assert set(_q38_stats("success", 0)) == set(VAGEN_STATS_KEYS)
