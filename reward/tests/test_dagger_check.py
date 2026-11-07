"""reward.dagger_check on a fake run folder (temp folders only)."""

import json

from reward import dagger_check as C


def _run(tmp_path, prefix_mean=0.002, prefix_over1=0, failed=0, whole_mean=0.0001, floor_mean=0.002):
    run = tmp_path / "rl_run_x"
    (run / "logs").mkdir(parents=True)
    (run / "dagger").mkdir()
    lines = [f"step:{s} - dagger/replaced_share:0.6 - actor/dagger_grad_mag_share_und:{0.2 + 0.01 * s} - "
             f"dagger/check_positions_same_failed:{failed if s == 1 else 0} - timing_s/step:70.5" for s in (1, 2, 3)]
    (run / "logs" / "rl-20261107-worker-0.log").write_text("\n".join(lines) + "\n")
    check = {"whole_row": {"tokens": 100, "mean_abs": whole_mean, "max_abs": 0.02, "over_1e-2": 3, "over_1e-1": 0},
             "prefix": {"tokens": 400, "mean_abs": prefix_mean, "max_abs": 0.05, "over_1e-2": 9,
                        "over_1e-1": prefix_over1},
             "shape_floor": {"tokens": 500, "mean_abs": floor_mean, "max_abs": 0.04, "over_1e-2": 8, "over_1e-1": 0},
             "loss_tokens": {"tokens": 200, "mean_abs": 0.001, "max_abs": 0.03}}
    # updates 1 and 2, and update 2 again (a resumed run): counted once
    (run / "dagger" / "check.jsonl").write_text("".join(json.dumps(dict(check, step=s)) + "\n" for s in (1, 2, 2)))
    first = {"samples": [
        {"kind": "own", "loss_decoded": "move_forward|move_forward|</action>"},
        {"kind": "label", "label_text": "<action>turn_left|</action>", "label_decoded": "<action>turn_left|</action><|im_end|>",
         "loss_decoded": "turn_left|</action>"},
        {"kind": "label", "label_text": "<action>turn_left|</action>", "label_decoded": "<action>turn_left|</action><|im_end|>",
         "loss_decoded": "<action>turn_left|</action>"}]}
    (run / "dagger" / "first_batch_rank0.json").write_text(json.dumps(first))
    return run


def test_log_values_and_numbers(tmp_path):
    run = _run(tmp_path)
    logs = C.log_values(run)
    assert logs["dagger/replaced_share"] == {1: 0.6, 2: 0.6, 3: 0.6}
    nb = C.numbers(logs)
    assert nb["actor/dagger_grad_mag_share_und"]["median"] == 0.22 and nb["timing_s/step"]["updates"] == 3


def test_prefix_check_reading(tmp_path):
    ok = C.prefix_check(_run(tmp_path / "a"), C.log_values(_run(tmp_path / "a2")))
    assert ok["passed"] is True and ok["prefix"]["tokens"] == 800 and ok["updates"] == 2
    assert ok["whole_row"]["mean_abs"] == 0.0001 and ok["shape_floor"]["tokens"] == 1000
    far = _run(tmp_path / "b", prefix_mean=0.006)  # 0.006 > 2 x 0.002 + 0.001
    assert C.prefix_check(far, C.log_values(far))["passed"] is False
    unsound = _run(tmp_path / "e", whole_mean=0.002)  # the same tokens as the recompute differ: the comparison is off
    assert C.prefix_check(unsound, C.log_values(unsound))["passed"] is False
    tails = _run(tmp_path / "c", prefix_over1=10)  # 20 of 800 tokens off by more than 0.1
    assert C.prefix_check(tails, C.log_values(tails))["passed"] is False
    exact = _run(tmp_path / "d", failed=1)
    rep = C.prefix_check(exact, C.log_values(exact))
    assert rep["passed"] is False and rep["exact_failures"]["positions_same"] == 1


def test_mask_check(tmp_path):
    rep = C.mask_check(_run(tmp_path))
    assert rep["counts"] == {"samples": 3, "own": 1, "label": 2, "ok": 2, "bad": 1}
    assert rep["examples"][0]["loss_decoded"] == "<action>turn_left|</action>"
