"""run_qa_paths.sh (the main line's request of 2026-11-06, approved by the user: X0 / G0 asked to Qwen3 folders given
by path): every model is asked X0 and G0 with --model-path and its own answer files, a missing folder is logged and
skipped, and the analysis sets every model against the base's answers of the 2026-10-30 batch."""
import os
import subprocess
from pathlib import Path

WT = Path(__file__).resolve().parents[2]


def fake_python(tmp_path: Path) -> Path:
    calls = tmp_path / "calls.txt"
    py = tmp_path / "py.sh"
    py.write_text("#!/usr/bin/env bash\n"
                  f"echo \"$*\" >> {calls}\n"
                  "for a in \"$@\"; do case $a in *=*answers/*_Q3_*.jsonl) f=${a#*=}; echo '{}' > \"$f\";; esac; done\n")
    py.chmod(0o755)
    return py


def test_each_model_is_asked_by_path_and_set_against_the_base(tmp_path):
    qa, x0 = tmp_path / "qa", tmp_path / "x0"
    (qa / "g0").mkdir(parents=True)
    (qa / "answers").mkdir()
    x0.mkdir()
    for f in [x0 / "questions.jsonl", qa / "g0/questions.jsonl", qa / "answers/x0_Q3B.jsonl",
              qa / "answers/g0_Q3B.jsonl"]:
        f.write_text("{}\n")
    w1 = tmp_path / "w1"
    w1.mkdir()
    models = tmp_path / "models.txt"
    models.write_text(f"# name path\nC-SFT {w1}\nGone {tmp_path / 'missing'}\n")
    out = tmp_path / "out"
    env = dict(os.environ, QA_VLM_PY=str(fake_python(tmp_path)), QA_X0=str(x0))
    res = subprocess.run(["bash", str(WT / "downstream/x/run_qa_paths.sh"), str(out), str(models), str(qa)],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 1  # the missing folder
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    ask = [c for c in calls if "q0_ask.py" in c]
    assert len(ask) == 1
    assert f"--model Q3B --model-path {w1}" in ask[0]
    assert f"{x0}/questions.jsonl={out}/answers/x0_Q3_C-SFT.jsonl" in ask[0]
    assert f"{qa}/g0/questions.jsonl={out}/answers/g0_Q3_C-SFT.jsonl" in ask[0]
    x0a = [c for c in calls if "x0b_qa.py analyze" in c][0]
    assert f"Q3B={qa}/answers/x0_Q3B.jsonl" in x0a and f"Q3_C-SFT={out}/answers/x0_Q3_C-SFT.jsonl" in x0a
    assert "--reference Q3B" in x0a and "Gone" not in x0a
    g0a = [c for c in calls if "g0_qa.py analyze" in c][0]
    assert f"Q3B={qa}/answers/g0_Q3B.jsonl" in g0a and f"Q3_C-SFT={out}/answers/g0_Q3_C-SFT.jsonl" in g0a
    log = (out / "qa_paths.log").read_text()
    assert "Gone: no folder" in log and "x0 1 models, g0 1 models" in log


def test_it_stops_when_the_base_answers_are_missing(tmp_path):
    models = tmp_path / "models.txt"
    models.write_text("")
    env = dict(os.environ, QA_VLM_PY=str(fake_python(tmp_path)), QA_X0=str(tmp_path / "x0"))
    res = subprocess.run(["bash", str(WT / "downstream/x/run_qa_paths.sh"), str(tmp_path / "out"), str(models),
                          str(tmp_path / "qa")], capture_output=True, text=True, env=env)
    assert res.returncode == 2
    assert not (tmp_path / "calls.txt").exists()
