from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vagen.q38s4.tools import gen_length as GL

SENTINEL = 0.987654321  # a score value no output may carry
SCORE_KEY = "sentinel_score_key"
RUN_MODEL = "evalprep__c_nfp_500_nohead"  # lmms-eval's folder for .../evalprep/c_nfp_500_nohead
BASE_MODEL = "models__Qwen3.8-27B"


def samples(folder: Path, stamp: str, task: str, tokens, model: str = RUN_MODEL, extra_scores: bool = True) -> Path:
    d = folder / model
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{stamp}_samples_{task}.jsonl"
    lines = []
    for i, t in enumerate(tokens):
        rec = {"doc_id": i, "token_counts": [{"output_tokens": t}], "resps": [["x"]]}
        if extra_scores:
            rec.update({"accuracy": {"total": 1.0, "overall": SENTINEL}, SCORE_KEY: SENTINEL, "exact_match": SENTINEL})
        lines.append(json.dumps(rec))
    f.write_text("\n".join(lines) + "\n")
    return f


def clock(seconds: int) -> str:
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def log(folder: Path, stamp: str, seconds, tokens, wall=None, n=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"eval_qwen3_vl_{stamp}.log"
    body = "loading\n"
    if wall is not None:  # as tqdm writes it: partial bars, then the finished one twice (ANSI codes after it)
        bar = "█" * 10
        body += f"Model Responding:  50%|{bar[:5]}     | {n // 2}/{n} [00:01<00:01,  1.00it/s]\r"
        body += f"Model Responding: 100%|{bar}| {n}/{n} [{clock(wall)}<00:00,  3.71it/s]\x1b[32m2026-11-24\x1b[0m\n"
        body += f"Model Responding: 100%|{bar}| {n}/{n} [{clock(wall)}<00:00,  1.57it/s]\n"
    if seconds is not None:
        body += (f"INFO Metric summary - Total elapsed time: {seconds}s, Total gen tokens: {tokens}, "
                 f"Avg speed: 1.0 tokens/s\n")
    body += f"|overall|{SENTINEL}|\n"  # a results table line in the log is never read
    f.write_text(body)
    return f


def test_row_counts_and_projection(tmp_path):
    qa, base = tmp_path / "qa", tmp_path / "base"
    run = qa / "mmsi_bench" / "c_nfp_500"
    samples(run, "20261124_120000", "mmsi_bench", [100, 200, 300, 2048, 2048] * 10)  # 50 questions
    log(run, "20261124_120000", 500.0, 33280, wall=800, n=50)
    b = base / "mmsi_bench" / "q38base"
    samples(b, "20261120_000000", "mmsi_bench", [500] * 50 + [1000] * 1950, model=BASE_MODEL)
    log(b, "20261120_000000", 60265.5, 1000000, wall=60387, n=2000)
    r = GL.job_row(qa, "c_nfp_500", "mmsi_bench", base, "q38base")
    assert r["n"] == 50 and r["tasks"] == 1 and r["tokens_mean"] == 939.2 and r["tokens_max"] == 2048
    assert r["tokens_median"] == 300 and r["tokens_p90"] == 2048.0 and r["at_cap"] == 0.4 and r["cap"] == 2048
    assert r["gen_s_per_q"] == 10.0 and r["tok_per_s"] == 66.6 and r["gen_seconds"] == 500.0
    assert r["wall_seconds"] == 800.0 and r["wall_n"] == 50 and r["s_per_q"] == r["wall_s_per_q"] == 16.0
    assert r["s_per_q_from"] == "wall"
    assert r["full_n"] == 2000 and r["base_tokens_mean"] == round((500 * 50 + 1000 * 1950) / 2000, 1)
    assert r["base_same_n"] == 50 and r["base_same_tokens_mean"] == 500.0 and r["len_ratio"] == round(939.2 / 500, 2)
    assert r["projected_h"] == round(16.0 * 2000 / 3600, 2) == 8.89 and r["over_limit"] is True
    assert r["base_h"] == round(60387 / 3600, 2) and r["base_gen_h"] == round(60265.5 / 3600, 2)


def test_under_the_line_and_generation_only_fallback(tmp_path):
    qa, base = tmp_path / "qa", tmp_path / "base"
    run = qa / "viewspatial" / "n"
    samples(run, "a", "viewspatial", [10] * 50)
    log(run, "a", 100.0, 500)  # no finished progress bar: generation-only seconds
    samples(base / "viewspatial" / "q38base", "a", "viewspatial", [10] * 500, model=BASE_MODEL)
    r = GL.job_row(qa, "n", "viewspatial", base, "q38base")
    assert r["wall_s_per_q"] is None and r["s_per_q"] == 2.0 and r["s_per_q_from"] == "generation only"
    assert r["projected_h"] == round(2.0 * 500 / 3600, 2) and r["over_limit"] is False
    assert "base_h" not in r and "base_gen_h" not in r  # the base folder has no log
    assert "(gen)" in GL.table([r])


def test_blink_pools_its_sub_tasks(tmp_path):
    """blink is 14 lmms-eval tasks with one samples file each and --limit applies per task: pool the newest run's."""
    qa, base = tmp_path / "qa", tmp_path / "base"
    run = qa / "blink" / "n"
    samples(run, "20261124_100000", "blink_art_style", [999] * 50)  # an older run: ignored
    samples(run, "20261124_110000", "blink_art_style", [2] * 50)
    samples(run, "20261124_110000", "blink_counting", [4] * 50)
    samples(run, "20261124_110000", "blink_visual_similarity", [1024] * 50)
    log(run, "20261124_110000", 150.0, 51500, wall=300, n=150)
    b = base / "blink" / "q38base"
    samples(b, "20261120_200942", "blink_art_style", [2] * 50 + [10] * 67, model=BASE_MODEL)
    samples(b, "20261120_200942", "blink_counting", [6] * 120, model=BASE_MODEL)
    samples(b, "20261120_200942", "blink_visual_similarity", [8] * 135, model=BASE_MODEL)
    log(b, "20261120_120342", 1000.0, 5000, wall=1211, n=372)
    r = GL.job_row(qa, "n", "blink", base, "q38base")
    assert r["tasks"] == 3 and r["n"] == 150 and all("20261124_110000_samples_blink_" in s for s in r["samples"])
    assert r["tokens_mean"] == round((2 + 4 + 1024) / 3, 1) and r["tokens_max"] == 1024
    assert r["at_cap"] == round(50 / 150, 3) and r["s_per_q"] == 2.0 and r["gen_s_per_q"] == 1.0
    assert r["full_n"] == 117 + 120 + 135 and r["base_tokens_mean"] == round((100 + 670 + 720 + 1080) / 372, 1)
    assert r["base_same_n"] == 150 and r["base_same_tokens_mean"] == round((2 * 50 + 6 * 50 + 8 * 50) / 150, 1)
    assert r["projected_h"] == round(2.0 * 372 / 3600, 2) and r["base_h"] == round(1211 / 3600, 2)
    assert "tasks_not_in_base" not in r


def test_task_missing_from_base_and_two_model_folders(tmp_path):
    qa, base = tmp_path / "qa", tmp_path / "base"
    run = qa / "blink" / "n"
    samples(run, "s", "blink_counting", [3] * 5)
    samples(run, "s", "blink_jigsaw", [3] * 5)
    log(run, "s", 10.0, 30, wall=20, n=10)
    samples(base / "blink" / "q38base", "s", "blink_counting", [3] * 9, model=BASE_MODEL)
    r = GL.job_row(qa, "n", "blink", base, "q38base")
    assert r["tasks_not_in_base"] == ["blink_jigsaw"] and r["base_same_n"] == 5 and r["full_n"] == 9
    samples(run, "s", "blink_counting", [3] * 5, model="ckpt_bf16__checkpoint-500")
    with pytest.raises(SystemExit, match="2 model folders"):
        GL.job_row(qa, "n", "blink", base, "q38base")


def test_newest_log_with_a_summary(tmp_path):
    run = tmp_path / "qa" / "embspatial" / "n"
    samples(run, "20261124_110000", "embspatial", [7] * 50)
    log(run, "20261124_100000", 50.0, 250, wall=60, n=50)
    log(run, "20261124_110000", 70.0, 350, wall=3725, n=50)
    log(run, "20261124_120000", None, None)  # a newer log without a summary (a retry that died)
    r = GL.job_row(tmp_path / "qa", "n", "embspatial", tmp_path / "nobase", "q38base")
    assert r["tokens_mean"] == 7 and r["gen_seconds"] == 70.0 and r["gen_s_per_q"] == 1.4
    assert r["wall_seconds"] == 3725.0 and r["s_per_q"] == 74.5  # 1:02:05
    assert "full_n" not in r and "projected_h" not in r and "over_limit" not in r  # no base folder


def test_missing(tmp_path):
    qa = tmp_path / "qa"
    assert GL.job_row(qa, "n", "blink", tmp_path, "b")["missing"] == "no samples file"
    samples(qa / "blink" / "n", "a", "blink_counting", [1] * 3)
    assert GL.job_row(qa, "n", "blink", tmp_path, "b")["missing"] == "no Metric summary in the logs"
    log(qa / "blink" / "n", "a", None, None, wall=5, n=3)
    assert GL.job_row(qa, "n", "blink", tmp_path, "b")["missing"] == "no Metric summary in the logs"


def test_main_reads_no_score(tmp_path, capsys):
    qa, base = tmp_path / "qa", tmp_path / "base"
    for i, bench in enumerate(GL.EASI_JOBS):
        if bench == "embspatial":
            continue  # one job not run: a "missing" row
        run = qa / bench / "c_nfp_500"
        samples(run, "s", bench, [i + 1] * 50)
        log(run, "s", 25.0 * (i + 1), 50 * (i + 1), wall=50 * (i + 1), n=50)
        samples(base / bench / "q38base", "s", bench, [1] * 100, model=BASE_MODEL)
        log(base / bench / "q38base", "s", 3600.0, 100, wall=4000, n=100)
    # files a score could be read from: never opened
    for name in ("results.json", "easi_results.json", "done.json"):
        (qa / "blink" / "c_nfp_500" / name).write_text(json.dumps({"overall": SENTINEL}))
    (qa / "blink" / "c_nfp_500" / RUN_MODEL / "s_results.json").write_text(json.dumps({"overall": SENTINEL}))
    out = tmp_path / "length.json"
    assert GL.main(["--qa", str(qa), "--name", "c_nfp_500", "--base-qa", str(base), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    got = json.loads(out.read_text())
    assert [r["benchmark"] for r in got["rows"]] == list(GL.EASI_JOBS) and got["limit_h"] == 3.0
    assert got["rows"][GL.EASI_JOBS.index("embspatial")]["missing"] == "no samples file"
    assert got["rows"][0]["s_per_q"] == 1.0 and got["rows"][0]["projected_h"] == round(1.0 * 100 / 3600, 2)
    assert got["rows"][0]["base_h"] == round(4000 / 3600, 2)
    for text in (out.read_text(), printed):
        assert str(SENTINEL) not in text and SCORE_KEY not in text and "accuracy" not in text
    assert printed.count("\n| ") == len(GL.EASI_JOBS)  # one line a job after the header and the rule line
    assert all(line.count("|") == len(GL.COLUMNS) + 1 for line in printed.strip().splitlines())


def test_reads_only_its_files(tmp_path, monkeypatch):
    """gen_length opens nothing but samples files, logs (and folders' listings)."""
    qa, base = tmp_path / "qa", tmp_path / "base"
    run = qa / "blink" / "n"
    samples(run, "s", "blink_counting", [3] * 50)
    log(run, "s", 10.0, 150, wall=20, n=50)
    (run / "done.json").write_text("{}")
    (run / RUN_MODEL / "s_results.json").write_text("{}")
    samples(base / "blink" / "q38base", "s", "blink_counting", [3] * 60, model=BASE_MODEL)
    (base / "blink" / "q38base" / "done.json").write_text("{}")
    opened = []
    real_read_text = Path.read_text
    real_open = open

    def read_text(self, *a, **k):
        opened.append(self.name)
        return real_read_text(self, *a, **k)

    def fake_open(file, *a, **k):
        opened.append(os.path.basename(str(file)))
        return real_open(file, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr("builtins.open", fake_open)
    r = GL.job_row(qa, "n", "blink", base, "q38base")
    assert r["full_n"] == 60
    assert opened and all(n.endswith("_samples_blink_counting.jsonl") or n.startswith("eval_") for n in opened)
