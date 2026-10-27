"""X4 follow-up: chat conversion, the answer spans, the analysis (CPU)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x"))
import x4_ppl as P  # noqa: E402


def test_images_stay_in_place_in_the_chat_messages():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "[Initial Observation]:\n<image>\npose"},
            {"role": "assistant", "content": "<action>turn_left|</action>"},
            {"role": "user", "content": "<image>a<image>"}]
    chat = P.to_chat(msgs)
    assert chat[0] == {"role": "system", "content": [{"type": "text", "text": "sys"}]}
    assert chat[1]["content"] == [{"type": "text", "text": "[Initial Observation]:\n"}, {"type": "image"},
                                  {"type": "text", "text": "\npose"}]
    assert chat[3]["content"] == [{"type": "image"}, {"type": "text", "text": "a"}, {"type": "image"}]


def test_answer_spans_cover_each_answer_and_its_end_token():
    s, e, a, n = P.IM_START, P.IM_END, P.ASSISTANT, P.NEWLINE
    ids = [s, 9, n, 5, e, n, s, 8, n, 7, e, n, s, a, n, 1, 2, e, n, s, 8, n, 3, e, n, s, a, n, 4, e]
    assert P.answer_spans(ids) == [(15, 17), (28, 29)]
    assert P.answer_spans([s, a, n, 4, 4]) == [(3, 4)]  # cut off: runs to the last token
    assert P.answer_spans([1, 2, 3]) == []


def write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_analysis_pairs_trajectories_and_compares_damages(tmp_path):
    d = tmp_path / "ppl"
    d.mkdir()
    base = [{"i": i, "tokens": 10, "nll_sum": 2.0, "correct": 8, "turns": 2, "turns_exact": 1} for i in range(6)]
    write(d / "orig-B.jsonl", base + [{"i": 99, "skipped": "20000 tokens"}])
    write(d / "E1-B-200k.jsonl", [dict(r, nll_sum=12.0, correct=1, turns_exact=0) for r in base])
    write(d / "orig-S.jsonl", [dict(r, nll_sum=1.0, correct=10, turns_exact=2) for r in base])
    write(d / "E1-S-200k.jsonl", [dict(r, nll_sum=31.0, correct=0, turns_exact=0) for r in base[:5]])  # one missing
    res = P.analyze(d)
    assert res["common_trajectories"] == 5
    m = res["models"]["orig-S"]
    assert m["loss"] == 0.1 and abs(m["perplexity"] - 1.10517) < 1e-4 and m["top1"] == 1.0 and m["exact"] == 1.0
    assert res["models"]["E1-B-200k"]["loss"] == 1.2 and res["models"]["orig-B"]["exact"] == 0.5
    assert res["e1_vs_start"]["B"]["mean_diff"] == 1.0 and res["e1_vs_start"]["B"]["higher"] == 5
    assert res["e1_vs_start"]["S"]["mean_diff"] == 3.0
    assert res["damage_vs_B"]["S"]["mean_diff"] == 2.0 and res["damage_vs_B"]["S"]["higher"] == 5
    md = (d / "results/results.md").read_text(encoding="utf-8")
    assert "| orig-S | 0.100" in md and "| S | 5 | +3.000" in md


def test_run_script_scores_on_one_gpu_then_analyzes(tmp_path):
    fake = tmp_path / "fake.py"
    fake.write_text("import sys, json, pathlib\na = sys.argv[1:]\n"
                    "if a[1] == 'score':\n"
                    "    out = pathlib.Path(a[a.index('--out') + 1]); out.write_text(json.dumps({'i': 0, 'tokens': 4, 'nll_sum': 1.0, 'correct': 2, 'turns': 1, 'turns_exact': 0}) + '\\n')\n"
                    "    print(json.dumps({'scored': 1, 'model': a[a.index('--model') + 1]}))\n"
                    "else:\n    pathlib.Path(a[a.index('--dir') + 1], 'analyzed').write_text('yes')\n")
    wrapper = tmp_path / "py.sh"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    wrapper.chmod(0o755)
    env = dict(os.environ, X4_VLM_PY=str(wrapper), X4_PY=str(wrapper))
    res = subprocess.run(["bash", str(ROOT / "x/run_x4_ppl.sh"), "orig-S", "/models/S", str(tmp_path / "ppl")],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "ppl/orig-S.jsonl").is_file() and (tmp_path / "ppl/analyzed").read_text() == "yes"
    assert '"model": "/models/S"' in (tmp_path / "ppl/orig-S.log").read_text()
