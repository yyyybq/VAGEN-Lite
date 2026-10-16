import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from data_gen.interiorgs_qa.capability.evaluate import (
    binomial_upper_tail, build_requests, evaluate_questions, load_questions, parse_letter,
    processor_input, prompt_text, run_inference, wilson,
)


def question(tmp_path, qid="q1", *, options=None, answer="A", group="距离", task="absolute_positioning"):
    files = []
    for i, color in enumerate(("red", "green")):
        p = tmp_path / f"{qid}-{i}.png"; Image.new("RGB", (8, 8), color).save(p); files.append(str(p))
    return {"qid": qid, "group": group, "task_type": task, "scene_id": "scene_a", "parent_task_id": "PRIVATE_TASK",
            "question": "Which object is nearer?", "options": options or ["the table", "the chair"],
            "answer": answer, "answer_key": "PRIVATE_ANSWER", "images": files,
            "evidence_zh": ["PRIVATE_GEOMETRY"], "source": {"secret": "PRIVATE_SOURCE"}, "human_review": "pending"}


def test_requests_preserve_image_order_and_do_not_contain_gt(tmp_path):
    q = question(tmp_path)
    requests = build_requests([q]); assert len(requests) == 2
    assert requests[0]["files"] == q["images"]
    assert requests[1]["files"] == [] and requests[1]["mode"] == "text_only"
    assert requests[0]["text"] == requests[1]["text"] == prompt_text(q)
    for secret in ("PRIVATE_ANSWER", "PRIVATE_TASK", "PRIVATE_GEOMETRY", "PRIVATE_SOURCE"):
        assert secret not in json.dumps(requests)
    q["answer"] = "B"; q["answer_key"] = "different"
    assert build_requests([q]) == requests


def test_processor_inputs_and_one_model_load_for_both_modes(tmp_path, monkeypatch):
    q = question(tmp_path); calls = {"processor": 0, "model": 0, "generate": 0}; captured = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured.append(messages)
            return json.dumps(messages)
    processor = Processor()
    def from_pretrained(model):
        calls["processor"] += 1; return processor
    class Model:
        def __init__(self, **kwargs):
            calls["model"] += 1; assert kwargs["limit_mm_per_prompt"] == {"image": 2}
        def generate(self, inputs, params):
            calls["generate"] += 1
            assert params.temperature == 0.0
            assert "multi_modal_data" not in inputs[1]
            images = inputs[0]["multi_modal_data"]["image"]
            assert images[0].getpixel((0, 0)) == (255, 0, 0)
            assert images[1].getpixel((0, 0)) == (0, 128, 0)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="A")], prompt_token_ids=[1, 2]) for _ in inputs]
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=from_pretrained)))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Model, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    results = run_inference("test-only-model", build_requests([q]))
    assert calls == {"processor": 1, "model": 1, "generate": 1}
    assert len(results) == 2 and all(x["reply"] == "A" for x in results)
    assert [x["type"] for x in captured[0][0]["content"]] == ["image", "image", "text"]
    assert [x["type"] for x in captured[1][0]["content"]] == ["text"]
    assert "image" not in captured[1][0]["content"][0]["text"]


@pytest.mark.parametrize("reply,expected", [("A", "A"), (" D\n", "D"), ("a", None), ("(A)", None),
    ("A. table", None), ("Answer: A", None), ("`A`", None), ("AB", None), ("E", None), ("", None)])
def test_strict_letter(reply, expected):
    assert parse_letter(reply, 4) == expected


def test_statistics_count_answers_and_stratify_chance(tmp_path):
    qs = [question(tmp_path, "q1", answer="A"), question(tmp_path, "q2", answer="B"),
          question(tmp_path, "q3", options=["a", "b", "c", "d"], answer="D")]
    predictions = [{"qid": q["qid"], "mode": mode, "reply": q["answer"] if mode == "with_images" else "A"}
                   for mode in ("with_images", "text_only") for q in qs]
    result = evaluate_questions(qs, predictions); group = result["by_group"]["距离"]
    assert result["requests"] == 6 and result["questions"] == 3
    assert group["with_images"]["correct"] == 3 and group["text_only"]["correct"] == 1
    assert math.isclose(group["chance_accuracy"], (0.5 + 0.5 + 0.25) / 3)
    assert group["majority_baseline"] == {"letter": "A", "correct": 1, "questions": 3, "accuracy": 1/3}
    tests = group["text_only_binomial_tests"]
    assert [(t["option_count"], t["questions"]) for t in tests] == [(2, 2), (4, 1)]
    assert math.isclose(tests[0]["p_one_sided"], 0.75)
    assert result["model_used_for_review"] is False and result["model_used_for_filtering"] is False
    with pytest.raises(ValueError): evaluate_questions(qs, predictions[:-1])
    with pytest.raises(ValueError): evaluate_questions(qs, predictions + [predictions[0]])


def test_exact_tail_wilson_and_text_shortcut_hint(tmp_path):
    assert math.isclose(binomial_upper_tail(4, 4, 0.5), 0.0625)
    assert binomial_upper_tail(0, 5, 0.25) == 1.0
    assert wilson(0, 0) is None
    assert wilson(0, 10)[0] == 0.0
    assert math.isclose(wilson(10, 10)[1], 1.0)
    qs = [question(tmp_path, f"q{i}") for i in range(20)]
    ps = [{"qid": q["qid"], "mode": mode, "reply": "A"} for mode in ("with_images", "text_only") for q in qs]
    assert evaluate_questions(qs, ps)["by_group"]["距离"]["text_only_binomial_tests"][0]["investigate_text_shortcut"]


def test_loader_rejects_missing_images_and_non_letter_answer(tmp_path):
    q = question(tmp_path); path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(q) + "\n"); assert load_questions(path)[0]["qid"] == "q1"
    q["answer"] = "AB"; path.write_text(json.dumps(q) + "\n")
    with pytest.raises(ValueError): load_questions(path)
    q["answer"] = "A"; q["images"] = [str(tmp_path / "missing.png")]; path.write_text(json.dumps(q) + "\n")
    with pytest.raises(ValueError): load_questions(path)
