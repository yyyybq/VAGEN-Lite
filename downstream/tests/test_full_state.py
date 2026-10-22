"""CPU tests of the stage-1 pieces: full-state folders and the segmented job (downstream/train/full_state.py), the
segment settings and batch hashes of train_entry.py, the resume-test comparison (resume_check.py) and the
experiment options of make_experiment.py."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import full_state as FS  # noqa: E402
import resume_check  # noqa: E402
import train_entry as te  # noqa: E402


def state(root: Path, step: int, complete=True):
    d = root / f"steps_{step}"
    d.mkdir(parents=True)
    (d / "pytorch_model").mkdir()
    (d / "pytorch_model/blob").write_bytes(b"x" * 10)
    if complete:
        (d / FS.STATE_FILE).write_text(json.dumps({"completed_steps": step, "max_train_steps": 300}))
    return d


def test_latest_skips_unfinished_saves_and_prune_keeps_one(tmp_path):
    root = tmp_path / "full_state"
    state(root, 100)
    keep = state(root, 200)
    state(root, 300, complete=False)  # a kill during the save: no state file
    (root / "steps_400.partial").mkdir()
    path, meta = FS.latest(root)
    assert path == keep and meta["completed_steps"] == 200
    assert [s for s, _ in FS.complete_states(root)] == [100, 200]
    assert FS.dir_bytes(keep) == 10 + len((keep / FS.STATE_FILE).read_text())
    gone = FS.prune(root, keep)
    assert sorted(gone) == ["steps_100", "steps_400.partial"] and keep.is_dir()
    assert (root / "steps_300").is_dir()  # an incomplete steps_N (no state file, no .partial) is left alone
    assert FS.latest(tmp_path / "missing") is None


def test_data_position():
    assert FS.data_position(0, 566000) == (0, 0)
    assert FS.data_position(25012, 566000) == (0, 25012)
    assert FS.data_position(566005, 566000) == (1, 5)
    assert FS.data_position(7, 0) == (0, 0)


def exp(tmp_path, steps=200000, points=(50000, 100000, 150000)):
    e = tmp_path / "DS-B-100-s1"
    e.mkdir(parents=True)
    (e / "experiment.json").write_text(json.dumps({"steps": steps, "eval_points": list(points) + [steps]}))
    return e


def episodes(path: Path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_points_checkpoints_and_eval_modes(tmp_path):
    e = exp(tmp_path)
    assert FS.eval_points(e) == [50000, 100000, 150000, 200000]
    assert FS.ckpt(e, 50000) == e / "run/checkpoints/steps_50000_pytorch_model.pt"
    assert FS.ckpt(e, 200000) == e / "run/final_model/pytorch_model.pt"
    assert FS.eval_mode(e, 50000) == "injob"
    s = FS.summary(e, 50000)
    s.parent.mkdir(parents=True)
    s.write_text(json.dumps({"per_task": {}}))
    assert FS.eval_mode(e, 50000) == "done"
    (e / "injob_eval.json").write_text(json.dumps({"mode": "final_only"}))
    assert FS.eval_mode(e, 100000) == "shards" and FS.eval_mode(e, 200000) == "injob"
    short = exp(tmp_path / "x", steps=100000, points=(50000, 150000))
    assert FS.eval_points(short) == [50000, 100000]  # points beyond the run's steps are dropped


def test_speed_rule_compares_with_the_24_process_figures(tmp_path):
    e = exp(tmp_path)
    eps = e / "evals/steps_50000/out/episodes"
    # 100 steps at 24 processes would take 21.6 + 19.6 = 41.2 s; 1.6x slower is 65.92 s
    episodes(eps, [{"status": "complete", "steps": 100, "wall_seconds": 65.92, "reset_seconds": 30.0}] * 3
             + [{"status": "error", "steps": 0, "wall_seconds": 0.0}])
    out = FS.speed(e, 50000)
    assert out["episodes"] == 3 and out["ratio"] == pytest.approx(1.6) and out["mode"] == "final_only"
    assert out["seconds_per_step_96"] == pytest.approx(0.3592)
    assert json.loads((e / "injob_eval.json").read_text())["mode"] == "final_only"
    episodes(eps, [{"status": "complete", "steps": 100, "wall_seconds": 41.2, "reset_seconds": 21.6}])
    assert FS.speed(e, 50000)["mode"] == "final_only"  # decided once, by the first evaluation
    e2 = exp(tmp_path / "y")
    episodes(e2 / "evals/steps_50000/out/episodes",
             [{"status": "complete", "steps": 100, "wall_seconds": 60.0, "reset_seconds": 21.6}])
    assert FS.speed(e2, 50000)["mode"] == "all"  # 1.46x


def test_segment_settings_must_divide_by_the_save_interval():
    s = te.segment_settings({"save_interval": 25000, "full_state_interval": 25000,
                             "eval_points": [100000, 50000], "stop_at": 50000})
    assert s == {"full_every": 25000, "eval_points": [50000, 100000], "stop_at": 50000, "digests": False}
    assert te.segment_settings({"save_interval": 200000}) == {"full_every": 0, "eval_points": [], "stop_at": 0,
                                                              "digests": False}
    with pytest.raises(ValueError):
        te.segment_settings({"save_interval": 25000, "full_state_interval": 25000, "eval_points": [60000]})
    with pytest.raises(ValueError):
        te.segment_settings({"save_interval": 25000, "stop_at": 10000})


def test_batch_digest_depends_on_actions_and_instructions():
    np = pytest.importorskip("numpy")
    a = [{"action": np.zeros((16, 12)), "lang": "open the drawer", "image": [1]}]
    b = [{"action": np.zeros((16, 12)), "lang": "open the drawer", "image": [2]}]
    c = [{"action": np.ones((16, 12)), "lang": "open the drawer"}]
    d = [{"action": np.zeros((16, 12)), "lang": "close the drawer"}]
    assert te.batch_digest(a) == te.batch_digest(b)  # images do not enter the hash
    assert len({te.batch_digest(a), te.batch_digest(c), te.batch_digest(d)}) == 3


def write_run(run: Path, metric_rows, digest_rows, states=(), launches=()):
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metric_rows))
    (run / "batch_digests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in digest_rows))
    (run / "full_state_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in states))
    (run / "launches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in launches))


def m(step, loss):
    return {"step": step, "action_dit_loss": loss, "lr": [1e-5 * (1 - step / 10)]}


def test_resume_check_on_a_good_and_a_bad_resume(tmp_path):
    a_metrics = [m(s, 1.0 / s) for s in range(1, 11)]
    a_digests = [{"batch": n, "step": n, "digest": f"d{n}"} for n in range(1, 11)]
    # B: killed after step 7 (full state at 5), resumed from 5 to the end
    b_metrics = [m(s, 1.0 / s + 1e-4) for s in range(1, 8)] + [{"resumed_from_step": 5}] + \
        [m(s, 1.0 / s + 2e-4) for s in range(6, 11)]
    b_digests = [{"batch": n, "step": n, "digest": f"d{n}"} for n in range(1, 8)] + \
        [{"batch": n, "step": n, "digest": f"d{n}"} for n in range(6, 11)]
    lr5 = [1e-5 * (1 - 5 / 10)]
    write_run(tmp_path / "RESUME-A/run", a_metrics, a_digests, states=[{"completed_steps": 5, "bytes": 100}])
    write_run(tmp_path / "RESUME-B/run", b_metrics, b_digests, states=[{"completed_steps": 5, "bytes": 100}],
              launches=[{"resumed_from": {"step": 5, "batches": 5, "skip": 5, "load_seconds": 3.0,
                                          "lr_after_load": lr5, "lr_saved": lr5}}])
    out = resume_check.check(tmp_path)
    assert out["data_continues"] and out["schedule_continues"] and out["kill_effective"]
    assert not out["segment_stopped"]  # only one resume here
    assert out["b_killed_after_step"] == 7 and out["b_resumed_from"] == [5] and out["redrawn_batches"] == 2
    assert out["loss_diff_before_kill"]["n"] == 5 and out["loss_diff_after_resume"]["n"] == 5
    assert out["loss_diff_after_resume"]["max_abs"] == pytest.approx(2e-4)
    # a resume that restarted the data from the beginning
    bad = tmp_path / "bad"
    write_run(bad / "RESUME-A/run", a_metrics, a_digests)
    write_run(bad / "RESUME-B/run", b_metrics, [{"batch": n, "step": n, "digest": f"d{n}"} for n in range(1, 8)]
              + [{"batch": n, "step": n, "digest": f"d{n - 5}"} for n in range(6, 11)])
    out = resume_check.check(bad)
    assert not out["data_continues"] and out["batch_mismatch_count"] == 5


def test_build_config_writes_the_full_state_settings(tmp_path):
    import yaml

    import make_experiment

    recipe = yaml.safe_load((ROOT / "train/recipe.yaml").read_text())
    data = tmp_path / "data"
    data.mkdir()
    (data / "split_manifest.json").write_text(json.dumps({"datasets": [{"name": "A"}]}))
    (data / "mixture_spec.json").write_text(json.dumps([["A/lerobot", 1.0, "panda_omron_robocasa365"]]))
    cfg = make_experiment.build_config(recipe, exp_dir=tmp_path / "e", exp_id="DS-B-100-s1", start_path="/s",
                                       data_dir=data, seed=1, steps=100000, save_interval=25000, log_every=100,
                                       full_state_interval=25000, eval_points=[50000, 100000, 150000])
    assert cfg["trainer"]["full_state_interval"] == 25000 and cfg["trainer"]["eval_points"] == [50000]
    plain = make_experiment.build_config(recipe, exp_dir=tmp_path / "e", exp_id="x", start_path="/s", data_dir=data,
                                         seed=1, steps=10, save_interval=10, log_every=1)
    assert "full_state_interval" not in plain["trainer"]


def test_resume_check_tells_an_effective_kill_and_a_segment_end(tmp_path):
    a_metrics = [m(s, 1.0 / s) for s in range(1, 11)]
    a_digests = [{"batch": n, "step": n, "digest": f"d{n}"} for n in range(1, 11)]
    write_run(tmp_path / "RESUME-A/run", a_metrics, a_digests)
    # good: killed after 7 (states at 5), resumed 5 -> stop at 8, resumed 8 -> 10
    good = [m(s, 1.0 / s) for s in range(1, 8)] + [{"resumed_from_step": 5}] + [m(s, 1.0 / s) for s in range(6, 9)] + \
        [{"resumed_from_step": 8}] + [m(s, 1.0 / s) for s in range(9, 11)]
    write_run(tmp_path / "B2/run", good, a_digests)
    out = resume_check.check(tmp_path, "RESUME-A", "B2")
    assert out["kill_effective"] and out["segment_stopped"] and out["second_launch_last_step"] == 8
    # the first test of 2026-10-22: the kill missed the workers (first launch went on to 10), the stop launch found
    # its step reached and left at once (no rows), the last launch resumed from the same state
    missed = [m(s, 1.0 / s) for s in range(1, 11)] + [{"resumed_from_step": 8}, {"resumed_from_step": 8}] + \
        [m(s, 1.0 / s) for s in range(9, 11)]
    write_run(tmp_path / "B1/run", missed, a_digests)
    out = resume_check.check(tmp_path, "RESUME-A", "B1")
    assert out["data_continues"] and not out["kill_effective"] and not out["segment_stopped"]
    assert resume_check.main([str(tmp_path), "RESUME-A", "B2"]) == 0 and (tmp_path / "resume_check_B2.json").is_file()
