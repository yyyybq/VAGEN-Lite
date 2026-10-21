import datetime
import json
import os

from eval_external.adapters import models as M
from eval_external.diagnostics.ownenv_eval import donor_pools
from eval_external.qa_transfer import collect as C
from eval_external.qa_transfer import pixel_log


def test_donor_pools_same_scene_first():
    tasks = [{"scene_id": s} for s in ("a", "b", "a", "a", "c")]
    rows = [{"index": i, "trajectory": [{}] * n} for i, n in enumerate((2, 3, 1, 2, 4))]
    pools = donor_pools(tasks, rows)
    assert pools[0][:3] == [(2, 0), (3, 0), (3, 1)]       # same scene a: tasks 2 then 3, every turn
    assert pools[0][3:6] == [(1, 0), (1, 1), (1, 2)]      # then the nearest other scene's task
    assert pools[4][0] == (3, 0)                          # scene c alone: nearest other index first
    assert all(j != i for i, p in pools.items() for j, _ in p)


def test_pixel_log_summary(tmp_path):
    f = tmp_path / "px.jsonl"
    f.write_text(json.dumps({"image_grid_thw": [[1, 20, 30]], "video_grid_thw": []}) + "\n" +
                 json.dumps({"image_grid_thw": [], "video_grid_thw": [[16, 10, 10]]}) + "\n")
    s = pixel_log.summarize(f)
    assert s["calls"] == 2
    assert s["pixels_per_image_or_frame"] == {"min": 100 * 196, "median": 100 * 196, "max": 600 * 196}
    assert s["images_or_frames_per_call"] == {"min": 1, "median": 16.5, "max": 32}
    assert s["visual_tokens_per_call"]["max"] == 400 and s["visual_tokens_per_call"]["min"] == 150
    assert pixel_log.summarize(tmp_path / "missing.jsonl") is None


def test_checks_flag_short_runs_and_skips(tmp_path, monkeypatch):
    def run(name, hours, log=""):
        rd = tmp_path / name
        (rd / "job").mkdir(parents=True)
        start = datetime.datetime(2026, 10, 19, tzinfo=datetime.timezone.utc)
        (rd / "manifest.json").write_text(json.dumps({"utc": start.isoformat()}))
        (rd / "outputs.json").write_text("{}")
        end = start.timestamp() + hours * 3600
        os.utime(rd / "outputs.json", (end, end))
        (rd / "job" / "run.log").write_text(log)
        return rd

    monkeypatch.setattr(C, "QA_RUNS", tmp_path)
    run("step1/base-7B__VSI-Bench", 4.0)
    ok = C.checks(run("a", 3.0), 5130, 5130, "VSI-Bench")
    assert ok["predictions_equal_items"] and not ok["short_run"] and not ok["backend_skipped"]
    assert ok["duration_s"] == 3 * 3600 and ok["base7b_duration_s"] == 4 * 3600
    short = C.checks(run("b", 0.1, "Model x Dataset VSI-Bench_32frame combination failed: OOM"), 0, 5130, "VSI-Bench")
    assert short["short_run"] and short["backend_skipped"] and not short["predictions_equal_items"]


def _ckpt(tmp_path, code="ours-X", pixels=None):
    ckpt = tmp_path / "exps" / "run" / "ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["Qwen2_5_VLForConditionalGeneration"]}))
    (ckpt / "preprocessor_config.json").write_text(json.dumps({"max_pixels": 401408, "min_pixels": 3136}))
    entry = {"source": "local", "path": "run/ckpt", "family": "qwen2_5_vl", "params": "7B"}
    if pixels:
        entry["pixels"] = pixels
    return {"hf_home": str(tmp_path / "hf"), "root": str(tmp_path / "root"), "vagen_exps": str(tmp_path / "exps"),
            "models": {code: entry}}


def test_own_pixel_view_keeps_published_limits(tmp_path):
    reg = _ckpt(tmp_path, "m-ownpx", pixels="own")
    view = M.build_view("m-ownpx", reg)
    assert json.loads((view / "preprocessor_config.json").read_text())["max_pixels"] == 401408
    assert json.loads((view / "config.json").read_text())["dtype"] == "bfloat16"


def test_esi_action_rule():
    from eval_external.closed_loop.esi_format_check import action_ok, classify

    p_obj = 'ACTIONS:\n  pick up <object>\n  place <object> on top of <object>\n  put back <object>\n  move_forward | stop'
    assert action_ok("turn_left", p_obj) and action_ok("stop", p_obj)
    assert action_ok("pick up lemon to the left of B", p_obj) and not action_ok("pick up ", p_obj)
    assert not action_ok("<action>move_forward|</action>", p_obj) and not action_ok("forward", p_obj)
    p_named = "4. Available pickup actions: pick_up_obj_near_tennis ball, pick_up_obj_near_conch."
    assert action_ok("pick_up_obj_near_conch", p_named) and not action_ok("pick up conch", p_named)
    p_list = "You can take the following actions:\n  put_on_slope: place the mango on the slope\n  move_forward\n\nx"
    assert action_ok("put_on_slope", p_list) and not action_ok("put_on_table", p_list)
    p_choice = '{\n  "action": "<action_name>",\n  "choice_letter": "<A or B>",\n  "confidence": <float>\n}'
    assert classify({"action": "turn_up", "choice_letter": "B", "confidence": 0.4}, p_choice) == []
    assert classify({"action": "turn_up", "answer": "B", "confidence": 0.4}, p_choice) == ["no_answer"]
    assert classify({"action": "move_left", "answer": "not sure", "confidence": 0.2}, p_obj) == []
    assert classify({"raw_response": "x", "error": "JSON parse failed", "action": "move_forward"}, p_obj) == [
        "no_json", "no_answer", "no_confidence"]


def test_pool_slots_fill_h800_first():
    from eval_external.sco import job as J

    assert J.POOL_LIMITS == {"h800": 4, "zoetrope": 4} and J.MAX_ACTIVE == 8
    assert J.free_pool({"h800": 3, "zoetrope": 0}) == "h800"
    assert J.free_pool({"h800": 4, "zoetrope": 0}) == "zoetrope"
    assert J.free_pool({"h800": 4, "zoetrope": 3}) == "zoetrope"
    assert J.free_pool({"h800": 4, "zoetrope": 4}) is None
    assert J.active_by_pool([{"pool": "h800"}, {"pool": "zoetrope"}, {"pool": "h800"}, {}]) == {
        "h800": 3, "zoetrope": 1}  # a record without a pool was submitted to h800


def test_yield_to_own_waiting_8_card_jobs():
    from eval_external.sco import job as J
    from eval_external.sco import queue as Q

    def j(name, pool, state, gpus, user):
        return {"display_name": name, "state": state, "resource_pool": {"name": pool}, "ownership": {"user_id": user},
                "roles": [{"resource_spec": [{"limits": {"nvidia.com/gpu": str(gpus)}}]}]}

    jobs = [j("evalext-step2-mmsi-x", "h800", "RUNNING", 1, "me"),
            j("interiorgs-v10-rl-rl-run-s100d", "zoetrope", "STARTING", 8, "me"),   # RL training, waiting, 8 cards
            j("bq_tp_mmxbf16_t500_l20_p12_r2", "h800", "STARTING", 8, "me"),         # 8 cards but not RL: no yield
            j("vln_train", "h800", "STARTING", 8, "someone"),                        # someone else's
            j("interiorgs-v10-renderer-rl-run-s100d", "h800", "STARTING", 1, "me")]  # one card
    assert Q.yield_pools(jobs) == {"zoetrope"}
    assert Q.pool_waiting(jobs) == {"h800": 3, "zoetrope": 1}
    assert Q.yield_pools([j("interiorgs-v10-rl-rl-run-x", "h800", "RUNNING", 8, "me")]) == set()  # started
    assert J.free_pool({"h800": 4, "zoetrope": 0}, exclude={"zoetrope"}) is None
    assert J.free_pool({"h800": 3, "zoetrope": 0}, exclude={"zoetrope"}) == "h800"


def test_waiting_jobs_move_to_the_less_crowded_pool(tmp_path):
    from eval_external.sco import queue as Q

    t0 = datetime.datetime(2026, 10, 20, 7, 0, tzinfo=datetime.timezone.utc)
    r = tmp_path / "submission.json"
    r.write_text(json.dumps({"utc": t0.isoformat(), "pool": "zoetrope"}))
    later = t0.timestamp() + Q.MOVE_AFTER_S + 1
    w = {"h800": 11, "zoetrope": 36}
    assert Q.move_target(r, "STARTING", w, later) == "h800"
    assert Q.move_target(r, "RUNNING", w, later) is None
    assert Q.move_target(r, "STARTING", w, t0.timestamp() + 60) is None         # not waited long enough
    assert Q.move_target(r, "STARTING", {"h800": 40, "zoetrope": 36}, later) is None  # other pool not better
    for k in range(Q.MAX_MOVES):
        (tmp_path / f"submission_pt-{k}_deleted.json").write_text("{}")
    assert Q.move_target(r, "STARTING", w, later) is None                       # moved often enough


def test_done_label_reads_any_results_file(tmp_path):
    from eval_external.sco import queue as Q

    (tmp_path / "results.json").write_text(json.dumps([{"model": "m"}]))  # the ESI format check writes a list
    assert Q.done_label(tmp_path) == "done"
    (tmp_path / "results.json").write_text(json.dumps({"checks": {"short_run": True}}))
    assert Q.done_label(tmp_path) == "done_flagged: short run"


def test_queue_replace_waits_for_normal_run(tmp_path, monkeypatch):
    from eval_external.sco import queue as Q

    monkeypatch.setattr(Q, "RUNS", tmp_path)
    it = Q.item("step2", "ours-SFT", "OwnEnv-replace")
    assert it.run_dir.name == "ours-SFT__replace" and it.command[-1] == "replace" and not it.prepared()
    (tmp_path / "ownenv" / "eval_test" / "ours-SFT").mkdir(parents=True)
    (tmp_path / "ownenv" / "eval_test" / "ours-SFT" / "results.json").write_text("{}")
    assert it.prepared()
