import json

from eval_external.closed_loop.walkerbench import run as R

FAIL = "All API retries failed, using fallback."


def _log(path, reasons):
    with open(path, "w") as f:
        f.write(json.dumps({"event": "session_start"}) + "\n")
        for r in reasons:
            f.write(json.dumps({"event": "action", "reason": r}) + "\n")


def test_count_steps_and_validity(tmp_path):
    _log(tmp_path / "a.jsonl", [None, "Failed to parse response, using fallback.", f"{FAIL} {R.OVERFLOW_TAG}"])
    _log(tmp_path / "b.jsonl", [None, FAIL])
    c = R.count_steps(sorted(tmp_path.glob("*.jsonl")))
    assert c == {"steps_logged": 5, "unparseable_steps": 1, "api_failed_steps": 2, "context_overflow_steps": 1,
                 "api_failed_tasks": 2}
    assert not R.is_valid({**c, "server_alive_at_end": True})  # b's unanswered step was not overflow
    assert R.is_valid({**c, "api_failed_steps": 1, "server_alive_at_end": True})
    assert not R.is_valid({**c, "api_failed_steps": 1, "server_alive_at_end": False})


def test_overflow_from_run_log(tmp_path):
    over = "Error code: 400 - This model's maximum context length is 128000 tokens and your request has ..."
    other = "Error code: 400 - At most 64 image(s) may be provided in one prompt."
    lines = [f"API call failed (attempt 1/3): {over}", f"API call failed (attempt 3/3): {over}",
             f"API call failed (attempt 3/3): {other}", "API call failed (attempt 2/3): x", "unrelated"]
    (tmp_path / "run.log").write_text("\n".join(lines) + "\n")
    assert R.overflow_from_run_log(tmp_path / "run.log") == {"last_attempt_failures": 2, "overflow": 1}


def test_rescore_accepts_only_all_overflow(tmp_path):
    over = ("API call failed (attempt 3/3): Error code: 400 - The decoder prompt (length 128800) is longer than the "
            "maximum model length of 128000.")
    run = tmp_path / "run"
    (run / "WalkerBench" / "logs").mkdir(parents=True)
    (run / "job").mkdir()
    _log(run / "WalkerBench" / "logs" / "t1.jsonl", [None, FAIL, FAIL])
    (run / "results_invalid.json").write_text(json.dumps({"server_alive_at_end": True, "api_failed_steps": 2}))
    (run / "job" / "run.log").write_text(over + "\n" + over + "\n")
    assert R.rescore(run)
    res = json.loads((run / "results.json").read_text())
    assert res["context_overflow_steps"] == 2 and res["api_failed_tasks"] == 1 and res["rescored"]["all_overflow"]
    (run / "results.json").unlink()
    (run / "job" / "run.log").write_text(over + "\nAPI call failed (attempt 3/3): Connection refused\n")
    assert not R.rescore(run) and not (run / "results.json").exists()
