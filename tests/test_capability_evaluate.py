import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from data_gen.interiorgs_qa.capability.evaluate import (
    binomial_upper_tail, build_requests, evaluate_questions, load_questions, parse_letter,
    markdown, processor_input, prompt_text, run_inference, wilson,
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
    q["source_answer_key"] = "PRIVATE_SOURCE_ROLE"; q["presented_answer_key"] = "PRIVATE_PRESENTED_ROLE"
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


def test_actual_kinds_split_two_choice_distance_from_four_choice_occlusion(tmp_path):
    qs = []
    # All source tasks put their second object closer; mention-order changes
    # balance the answer as read in the actual question, not the source roles.
    for i, (letter, presented) in enumerate(zip(['A', 'B', 'B', 'A'],
                                               ['a_closer', 'b_closer', 'a_closer', 'b_closer'])):
        q = question(tmp_path, f'near{i}', answer=letter, group='遮挡判断', task='occlusion_alignment')
        q.update(kind='occlusion_closer', answer_key=presented, source_answer_key='b_closer')
        qs.append(q)
    for i, (letter, key) in enumerate([('A', 'fully_hidden'), ('C', 'partly_hidden')]):
        q = question(tmp_path, f'hidden{i}', answer=letter, options=['fully', 'partly', 'clear', 'outside'],
                     group='遮挡判断', task='occlusion_alignment')
        q.update(kind='occlusion_state', answer_key=key, source_answer_key=key)
        qs.append(q)
    predictions = [{'qid': q['qid'], 'mode': mode,
                    'reply': q['answer'] if mode == 'with_images' else 'A'}
                   for mode in ('with_images', 'text_only') for q in qs]
    result = evaluate_questions(qs, predictions)
    assert result['by_group']['遮挡判断']['questions'] == 6
    assert result['by_task_type']['occlusion_alignment']['questions'] == 6
    assert set(result['by_kind']) == {'occlusion_closer', 'occlusion_state'}
    near, hidden = (result['by_kind'][kind] for kind in ['occlusion_closer', 'occlusion_state'])
    assert near['questions'] == 4 and near['chance_accuracy'] == .5
    assert hidden['questions'] == 2 and hidden['chance_accuracy'] == .25
    assert near['chance_expected_correct'] == 2 and hidden['chance_expected_correct'] == .5
    assert near['majority_baseline']['correct'] == 2
    assert near['majority_semantic_baseline']['correct'] == 2
    assert near['majority_semantic_baseline']['questions'] == 4
    assert near['presented_answer_key_counts'] == {'a_closer': 2, 'b_closer': 2}
    assert near['source_answer_key_counts'] == {'b_closer': 4}
    assert 'source_majority_semantic_baseline' not in near
    assert near['source_answer_key_fields'] == {'source_answer_key': 4}
    tests = [row['text_only_binomial_tests'][0] for row in [near, hidden]]
    assert [test['option_count'] for test in tests] == [2, 4]
    assert all(test['multiple_comparison_scope'] == 'question_kinds' for test in tests)
    assert all(test['multiple_comparison_threshold'] == .025 for test in tests)
    report = markdown(result)
    assert '两目标谁更近（遮挡任务来源）' in report and '双图遮挡状态判断' in report
    assert 'occlusion_closer' not in report and 'occlusion_state' not in report
    assert '2／4（先提到的目标更近）' in report
    assert '0.50／2（25.0%）' in report
    assert '不是模型成绩' in report


def test_semantic_field_precedence_missing_denominator_and_legacy_fallback(tmp_path):
    first, second, third = [question(tmp_path, f'fallback{i}') for i in range(3)]
    first.update(kind='occlusion_closer', answer_key='b_closer', presented_answer_key='a_closer',
                 source_answer_key='b_closer')
    second.update(kind='occlusion_closer', answer_key='a_closer')
    third.update(kind='occlusion_closer'); del third['answer_key']
    qs = [first, second, third]
    predictions = [{'qid': q['qid'], 'mode': mode, 'reply': 'A'}
                   for mode in ('with_images', 'text_only') for q in qs]
    row = evaluate_questions(qs, predictions)['by_kind']['occlusion_closer']
    assert row['questions'] == 3 and row['majority_baseline']['questions'] == 3
    assert row['presented_answer_key_counts'] == {'a_closer': 2}
    assert row['source_answer_key_counts'] == {'a_closer': 1, 'b_closer': 1}
    assert row['presented_answer_key_fields'] == {'presented_answer_key': 1, 'answer_key': 1, 'missing': 1}
    assert row['source_answer_key_fields'] == {'source_answer_key': 1, 'answer_key': 1, 'missing': 1}
    assert row['majority_semantic_baseline']['correct'] == 2
    assert row['majority_semantic_baseline']['questions'] == 2
    third_only = evaluate_questions([third], [p for p in predictions if p['qid'] == third['qid']])
    assert third_only['all']['majority_semantic_baseline'] is None
    assert '未记录答案含义' in markdown(third_only)
