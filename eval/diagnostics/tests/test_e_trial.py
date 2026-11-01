"""eval/diagnostics/e_trial.py: group E's one-card trial (free_think, a 6,144-token budget per episode)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval.diagnostics import e_trial as ET


def _tasks(path, families=("a", "b", "c"), per=5):
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"task_id": f"{f}{i}", "task_type": f} for f in families for i in range(per)]
    (path / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_subset_and_launch(tmp_path):
    _tasks(tmp_path / "ev")
    rows = ET.subset(tmp_path / "ev", 2)
    assert [r["task_type"] for r in rows].count("a") == 2 and len(rows) == 6
    assert rows == ET.subset(tmp_path / "ev", 2)                      # a fixed draw
    launch = ET.prepare(tmp_path / "run", tmp_path / "ev", "/models/sft e8", per_family=2, pool="h800").read_text()
    assert "--prompt-format free_think --format-example action_names" in launch
    assert "--budget 6144" in launch and "timeout --signal=INT 45m" in launch
    assert "--temperature 0.8 --top-p 0.92" in launch and "'/models/sft e8'" in launch
    assert f"PY={ET.QWEN3_PYTHON}" in launch
    prepared = json.loads((tmp_path / "run" / "job" / "prepared.json").read_text())
    assert (prepared["pool"], prepared["gpus"], prepared["tasks"], prepared["budget"]) == ("h800", 1, 6, 6144)
    assert len((tmp_path / "run" / "eval" / "tasks.jsonl").read_text().splitlines()) == 6


def test_budget_policy_gives_each_turn_what_is_left(tmp_path):
    pytest.importorskip("vllm")
    from data_gen.interiorgs_v10 import evaluation as EV

    seen = []

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    class LLM:
        def get_tokenizer(self):
            return Tok()

        def chat(self, messages, sampling_params, use_tqdm):
            seen.append(sampling_params.max_tokens)
            return [SimpleNamespace(prompt_token_ids=[0] * 7, outputs=[SimpleNamespace(
                text="<think>x</think><action>w</action>", token_ids=[1, 2, 3], finish_reason="stop")])]

    log = tmp_path / "turns.jsonl"
    cls = ET.budget_policy(EV.VllmPolicy, 10, log)
    policy = cls.__new__(cls)
    policy.llm, policy.temperature, policy.top_p, policy.cambrian = LLM(), 0.8, 0.92, None
    old = EV.chat_messages
    EV.chat_messages = lambda *a: []
    try:
        policy.generate("s", "u", None)
        policy.generate("s", "u", None, earlier=[("u", None, "one two three four")])
        policy.generate("s", "u", None, earlier=[("u", None, " ".join(["w"] * 12))])
    finally:
        EV.chat_messages = old
    assert seen == [10, 6, 1]                                          # the budget left, at least one token
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c["used_before"] for c in calls] == [0, 4, 12] and calls[0]["tokens"] == 3 and calls[0]["prompt_tokens"] == 7


def test_report(tmp_path):
    run = tmp_path / "run"
    _tasks(tmp_path / "ev", per=2)
    ET.prepare(run, tmp_path / "ev", "/m", per_family=1, pool="h800")
    out = run / "eval" / ET.NAME
    out.mkdir(parents=True)

    def row(i, turns, invalid, success):
        traj = [{"answer": "<think>t</think><action>w</action>" if k >= invalid else "garbled", "valid": k >= invalid}
                for k in range(turns)]
        return {"index": i, "task_type": "a", "d_star": 2, "success": success, "actions_to_success": 2 if success else None,
                "turns": turns, "actions": turns, "collisions": 0, "invalid_turns": invalid, "end_reason": "max_turns",
                "trajectory": traj}

    rows = [row(0, 2, 0, True), row(1, 3, 1, False)]
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    calls = [{"tokens": n, "finish": f} for n, f in ((100, "stop"), (200, "stop"), (6144, "length"), (50, "stop"),
                                                    (60, "stop"), (999, "stop"))]  # the last: an episode cut off
    (run / "turns.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    rep = ET.report(run)
    assert (rep["episodes"], rep["turns"], rep["unreadable_turn_share"]) == (2, 5, 0.2)
    assert rep["tokens_per_turn"]["n"] == 5 and rep["tokens_per_turn"]["max"] == 6144
    assert rep["tokens_per_turn_unreadable"]["mean"] == 6144 and rep["turns_at_limit"] == 1
    assert rep["tokens_per_episode"]["max"] == 6254 and rep["episodes_using_budget"] == 1
    assert rep["unreadable_episode_mean"] == round((0 + 1 / 3) / 2, 4)
    assert rep["verdict"] == "under the 20% line"                   # exactly 20% is not over it
    assert (run / "report.md").read_text().startswith("# E 组试跑")
    rows.append(row(2, 1, 1, False))                                 # one more unreadable turn: 2 of 6
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ET.report(run)["verdict"] == "over the 20% line: stop E and ask"
