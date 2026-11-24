from __future__ import annotations

import json
import os
from pathlib import Path

from vagen.q38s4.tools import gen_length as GL

SENTINEL = 0.987654321  # a score value no output may carry
SCORE_KEY = "sentinel_score_key"


def samples(folder: Path, stamp: str, task: str, tokens, extra_scores: bool = True) -> Path:
    d = folder / "models__Qwen3.8-27B"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{stamp}_samples_{task}.jsonl"
    lines = []
    for i, t in enumerate(tokens):
        rec = {"doc_id": i, "token_counts": [{"output_tokens": t, "input_tokens": 999}], "resps": [["x"]]}
        if extra_scores:
            rec.update({"accuracy": {"total": 1.0, "overall": SENTINEL}, SCORE_KEY: SENTINEL, "exact_match": SENTINEL})
        lines.append(json.dumps(rec))
    f.write_text("\n".join(lines) + "\n")
    return f


def log(folder: Path, stamp: str, seconds, tokens) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"eval_qwen3_vl_{stamp}.log"
    body = "loading\n"
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
    log(run, "20261124_120000", 500.0, 33280)
    b = base / "mmsi_bench" / "q38base"
    samples(b, "20261120_000000", "mmsi_bench", [1000] * 2000)
    log(b, "20261120_000000", 60265.5, 1000000)
    r = GL.job_row(qa, "c_nfp_500", "mmsi_bench", base, "q38base")
    assert r["n"] == 50 and r["tokens_mean"] == 939.2 and r["tokens_max"] == 2048 and r["tokens_median"] == 300
    assert r["tokens_p90"] == 2048.0 and r["at_cap"] == 0.4 and r["cap"] == 2048
    assert r["s_per_q"] == 10.0 and r["tok_per_s"] == 66.6 and r["gen_seconds"] == 500.0
    assert r["full_n"] == 2000 and r["base_tokens_mean"] == 1000.0
    assert r["projected_h"] == round(10.0 * 2000 / 3600, 2) == 5.56 and r["over_limit"] is True
    assert r["base_h"] == round(60265.5 / 3600, 2)


def test_under_the_line(tmp_path):
    qa, base = tmp_path / "qa", tmp_path / "base"
    run = qa / "viewspatial" / "n"
    samples(run, "a", "viewspatial", [10] * 50)
    log(run, "a", 100.0, 500)
    samples(base / "viewspatial" / "q38base", "a", "viewspatial", [10] * 500)
    r = GL.job_row(qa, "n", "viewspatial", base, "q38base")
    assert r["projected_h"] == round(2.0 * 500 / 3600, 2) and r["over_limit"] is False
    assert "base_h" not in r  # the base folder has no log


def test_newest_samples_and_log(tmp_path):
    run = tmp_path / "qa" / "blink" / "n"
    samples(run, "20261124_100000", "blink", [5] * 50)
    samples(run, "20261124_110000", "blink", [7] * 50)
    log(run, "20261124_100000", 50.0, 250)
    log(run, "20261124_110000", 70.0, 350)
    log(run, "20261124_120000", None, None)  # a newer log without a summary (a retry that died)
    r = GL.job_row(tmp_path / "qa", "n", "blink", tmp_path / "nobase", "q38base")
    assert r["tokens_mean"] == 7 and r["samples"].endswith("20261124_110000_samples_blink.jsonl")
    assert r["gen_seconds"] == 70.0 and r["s_per_q"] == 1.4
    assert "full_n" not in r and "projected_h" not in r and "over_limit" not in r  # no base folder


def test_missing(tmp_path):
    qa = tmp_path / "qa"
    assert GL.job_row(qa, "n", "blink", tmp_path, "b")["missing"] == "no samples file"
    samples(qa / "blink" / "n", "a", "blink", [1] * 3)
    assert GL.job_row(qa, "n", "blink", tmp_path, "b")["missing"] == "no Metric summary in the logs"
    log(qa / "blink" / "n", "a", None, None)
    assert GL.job_row(qa, "n", "blink", tmp_path, "b")["missing"] == "no Metric summary in the logs"


def test_main_reads_no_score(tmp_path, capsys):
    qa, base = tmp_path / "qa", tmp_path / "base"
    for i, bench in enumerate(GL.EASI_JOBS):
        if bench == "embspatial":
            continue  # one job not run: a "missing" row
        run = qa / bench / "c_nfp_500"
        samples(run, "s", bench, [i + 1] * 50)
        log(run, "s", 25.0 * (i + 1), 50 * (i + 1))
        samples(base / bench / "q38base", "s", bench, [1] * 100)
        log(base / bench / "q38base", "s", 3600.0, 100)
    # files a score could be read from: never opened
    for name in ("results.json", "easi_results.json", "done.json"):
        (qa / "blink" / "c_nfp_500" / name).write_text(json.dumps({"overall": SENTINEL}))
    out = tmp_path / "length.json"
    assert GL.main(["--qa", str(qa), "--name", "c_nfp_500", "--base-qa", str(base), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    got = json.loads(out.read_text())
    assert [r["benchmark"] for r in got["rows"]] == list(GL.EASI_JOBS) and got["limit_h"] == 3.0
    assert got["rows"][GL.EASI_JOBS.index("embspatial")]["missing"] == "no samples file"
    assert got["rows"][0]["s_per_q"] == 0.5 and got["rows"][0]["projected_h"] == round(0.5 * 100 / 3600, 2)
    for text in (out.read_text(), printed):
        assert str(SENTINEL) not in text and SCORE_KEY not in text and "accuracy" not in text
    assert printed.count("\n| ") == len(GL.EASI_JOBS)  # one line a job after the header and the rule line


def test_reads_only_its_files(tmp_path, monkeypatch):
    """gen_length opens nothing but samples files, logs (and folders' listings)."""
    qa, base = tmp_path / "qa", tmp_path / "base"
    run = qa / "blink" / "n"
    samples(run, "s", "blink", [3] * 50)
    log(run, "s", 10.0, 150)
    (run / "done.json").write_text("{}")
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
    GL.job_row(qa, "n", "blink", base, "q38base")
    assert opened and all(n.endswith("_samples_blink.jsonl") or n.startswith("eval_") for n in opened)
