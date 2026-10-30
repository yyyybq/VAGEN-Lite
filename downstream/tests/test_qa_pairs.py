"""qa_pairs.py (2026-10-30): X0b-2's loosest reading, RL against SFT paired by question on each set, models whose
replies are missing are left out, and the CLI writes its JSON (tiny stand-in sets in tmp_path)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x"))
import qa_pairs as P  # noqa: E402
import x0_qa  # noqa: E402


def test_loosest_reading():
    opts = ["left", "right"]
    assert x0_qa.parse_letter("<B. right>") is None  # why the other readings miss Q3B's replies
    assert P.read_loosest("<B. right>", opts) == "B"
    assert P.read_loosest("<answer>A. left</answer>", opts) == "A"
    assert P.read_loosest("<answer>Right</answer>", opts) == "B"  # the lenient reading
    assert P.read_loosest("<answer>turn_right</answer>", opts) is None
    assert P.read_loosest(None, None) is None


def write(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in items))


def make(tmp_path):
    qa, x0 = tmp_path / "qa", tmp_path / "x0"
    write(x0 / "questions.jsonl", [{"id": "q1", "answer": "A"}, {"id": "q2", "answer": "B"}, {"id": "q3", "answer": "A"}])
    write(x0 / "answers_S.jsonl", [{"id": "q1", "reply": "A"}, {"id": "q2", "reply": "B"}, {"id": "q3", "reply": "B"}])
    write(x0 / "answers_RS.jsonl", [{"id": "q1", "reply": "A"}, {"id": "q2", "reply": "A"}, {"id": "q3", "reply": "A."}])
    write(qa / "answers/x0_Q3S.jsonl", [{"id": "q1", "reply": "B"}, {"id": "q2", "reply": "B"}, {"id": "q3", "reply": "B"}])
    write(qa / "answers/x0_Q3RD.jsonl", [{"id": "q1", "reply": "A"}, {"id": "q2", "reply": "B"}, {"id": "q3", "reply": "A"}])
    opts = ["left", "right"]
    write(qa / "x0b2/questions.jsonl", [{"id": "t1", "answer": "A", "options": opts},
                                        {"id": "t2", "answer": "B", "options": opts}])
    write(qa / "answers/x0b2_Q3B.jsonl", [{"id": "t1", "reply": "<A. left>"}, {"id": "t2", "reply": "<B. right>"}])
    write(qa / "answers/x0b2_Q3S.jsonl", [{"id": "t1", "reply": "<answer>left</answer>"},
                                          {"id": "t2", "reply": "<answer>turn_right</answer>"}])
    write(qa / "g0/questions.jsonl", [{"id": "g1", "kind": "existence", "options": ["yes", "no"], "answer": "A"},
                                      {"id": "g2", "kind": "counting", "options": None, "answer": "2"}])
    write(qa / "answers/g0_Q3S.jsonl", [{"id": "g1", "reply": "A. yes"}, {"id": "g2", "reply": "two"}])
    write(qa / "answers/g0_Q3RD.jsonl", [{"id": "g1", "reply": "B"}, {"id": "g2", "reply": "2"}])
    return qa, x0


def test_pairs_by_question_and_missing_models(tmp_path):
    qa, x0 = make(tmp_path)
    res = P.compute(qa, x0)
    rs = res["rl_vs_sft"]["X0"]["RS-S"]
    assert (rs["n"], rs["correct"], rs["reference_correct"], rs["only_model"], rs["only_reference"]) == (3, 2, 2, 1, 1)
    q3 = res["rl_vs_sft"]["X0"]["Q3RD-Q3S"]
    assert (q3["correct"], q3["reference_correct"], q3["only_model"], q3["only_reference"]) == (3, 1, 2, 0)
    assert abs(q3["diff_points"] - 200.0 / 3) < 1e-9 and q3["p"] == x0_qa.mcnemar_exact(2, 0)
    g0 = res["rl_vs_sft"]["G0"]["Q3RD-Q3S"]
    assert (g0["correct"], g0["reference_correct"]) == (1, 2)  # "A. yes" and "two" read right
    assert "RD-S" not in res["rl_vs_sft"]["X0"] and "X0b-1" not in res["rl_vs_sft"]  # no replies, no pair
    loose = res["loosest"]
    assert loose["accuracy"] == {"Q3B": {"n": 2, "correct": 2}, "Q3S": {"n": 2, "correct": 1}}
    assert (loose["Q3S-Q3B"]["only_model"], loose["Q3S-Q3B"]["only_reference"]) == (0, 1)


def test_cli_writes_json(tmp_path):
    qa, x0 = make(tmp_path)
    out = tmp_path / "pairs.json"
    assert P.main(["--qa", str(qa), "--x0", str(x0), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["rl_vs_sft"]["X0"]["RS-S"]["n"] == 3
