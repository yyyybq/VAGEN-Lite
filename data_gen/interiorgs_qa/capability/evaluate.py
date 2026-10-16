"""Inspect image and text-only performance; never review, filter or train questions.

python -m data_gen.interiorgs_qa.capability.evaluate --questions questions.jsonl --model MODEL --out OUT
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Sequence

LETTERS = "ABCDEFGH"


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("题库为空或题号重复")
    for row in rows:
        if (not isinstance(row["options"], list) or not 2 <= len(row["options"]) <= len(LETTERS)
                or not all(isinstance(x, str) for x in row["options"])):
            raise ValueError(f"选项数不支持：{row['qid']}")
        if not isinstance(row["answer"], str) or len(row["answer"]) != 1 or row["answer"] not in LETTERS[:len(row["options"])]:
            raise ValueError(f"答案不属于本题选项：{row['qid']}")
        if not 1 <= len(row["images"]) <= 2:
            raise ValueError("本轮能力题必须提供一张或两张图")
        for image in row["images"]:
            if not Path(image).is_absolute() or not Path(image).is_file():
                raise ValueError(f"原图不存在或不是绝对路径：{image}")
    return rows


def prompt_text(question: dict[str, Any]) -> str:
    """The public question and choices only; no answer/evidence/source fields."""
    return (question["question"] + "\n" +
            "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(question["options"])) +
            "\nAnswer with the option's letter only.")


def build_requests(questions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for mode in ("with_images", "text_only"):
        for q in questions:
            requests.append({"qid": q["qid"], "group": q["group"], "task_type": q["task_type"],
                             "mode": mode, "text": prompt_text(q),
                             "files": list(q["images"]) if mode == "with_images" else []})
    return requests


def processor_input(request: dict[str, Any], processor: Any) -> dict[str, Any]:
    """No image token, image file or multi_modal_data at all for the text-only arm."""
    from PIL import Image

    if request["mode"] == "text_only":
        content = [{"type": "text", "text": request["text"]}]
        result = {}
    else:
        content = [{"type": "image"} for _ in request["files"]] + [{"type": "text", "text": request["text"]}]
        images = []
        for name in request["files"]:
            with Image.open(name) as image:
                images.append(image.convert("RGB").copy())
        result = {"multi_modal_data": {"image": images}}
    result["prompt"] = processor.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=False, add_generation_prompt=True)
    return result


def run_inference(model: str, requests: Sequence[dict[str, Any]], *, max_model_len: int = 8192,
                  max_tokens: int = 8, seed: int = 0) -> list[dict[str, Any]]:
    """Load one processor and one model, then run both arms in the same process."""
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(model)
    model_instance = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=0.85,
                         seed=seed, limit_mm_per_prompt={"image": 2})
    inputs = [processor_input(request, processor) for request in requests]
    outputs = model_instance.generate(inputs, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    if len(outputs) != len(requests):
        raise RuntimeError("模型输出数量与请求数量不一致")
    return [{"qid": request["qid"], "group": request["group"], "task_type": request["task_type"],
             "mode": request["mode"], "reply": result.outputs[0].text,
             "prompt_tokens": len(result.prompt_token_ids)} for request, result in zip(requests, outputs)]


def parse_letter(reply: str, n_options: int) -> str | None:
    """Only a standalone uppercase option letter is an accepted answer."""
    reply = str(reply).strip()
    return reply if re.fullmatch("[" + LETTERS[:n_options] + "]", reply) else None


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def binomial_upper_tail(correct: int, total: int, probability: float) -> float:
    """Exact one-sided P(X >= correct), for independent equal-chance trials."""
    if not 0 <= correct <= total or not 0 < probability < 1:
        raise ValueError("二项检验参数无效")
    if correct == 0:
        return 1.0
    values = [math.exp(math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1)
                       + k * math.log(probability) + (total - k) * math.log1p(-probability))
              for k in range(correct, total + 1)]
    return min(1.0, math.fsum(values))


def _distribution(questions: Sequence[dict[str, Any]], outcomes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(questions)
    counts = Counter(q["answer"] for q in questions)
    majority_letter, majority_n = max(sorted(counts.items()), key=lambda item: item[1])
    result: dict[str, Any] = {"questions": count, "answer_letter_counts": dict(sorted(counts.items())),
        "chance_accuracy": sum(1 / len(q["options"]) for q in questions) / count,
        "majority_baseline": {"letter": majority_letter, "correct": majority_n, "questions": count,
                              "accuracy": majority_n / count}}
    for mode in ("with_images", "text_only"):
        values = [r for r in outcomes if r["mode"] == mode]
        correct = sum(r["correct"] for r in values)
        result[mode] = {"correct": correct, "questions": len(values), "accuracy": correct / len(values),
                        "invalid_letter_answers": sum(r["parsed_answer"] is None for r in values),
                        "wilson_95": wilson(correct, len(values))}
    # Mixed 2/3/4-choice questions are not a single binomial experiment.
    # Test equal-option-count strata separately, keeping their denominators explicit.
    tests = []
    for choices in sorted({len(q["options"]) for q in questions}):
        ids = {q["qid"] for q in questions if len(q["options"]) == choices}
        values = [r for r in outcomes if r["mode"] == "text_only" and r["qid"] in ids]
        correct = sum(r["correct"] for r in values)
        tests.append({"option_count": choices, "questions": len(values), "correct": correct,
                      "chance": 1 / choices, "p_one_sided": binomial_upper_tail(correct, len(values), 1 / choices)})
    result["text_only_binomial_tests"] = tests
    return result


def evaluate_questions(questions: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]],
                       *, alpha: float = 0.05) -> dict[str, Any]:
    by_id = {q["qid"]: q for q in questions}
    if len(by_id) != len(questions) or not questions:
        raise ValueError("题库必须非空且题号唯一")
    expected = {(qid, mode) for qid in by_id for mode in ("with_images", "text_only")}
    got = [(row["qid"], row["mode"]) for row in predictions]
    if len(got) != len(set(got)) or set(got) != expected:
        raise ValueError("每道题必须恰有有图和纯文字各一次输出")
    outcomes = []
    for row in predictions:
        q = by_id[row["qid"]]
        parsed = parse_letter(row["reply"], len(q["options"]))
        outcomes.append({**row, "parsed_answer": parsed, "correct": parsed == q["answer"]})
    groups = {}
    for group in sorted({q["group"] for q in questions}):
        qs = [q for q in questions if q["group"] == group]
        ids = {q["qid"] for q in qs}
        groups[group] = _distribution(qs, [r for r in outcomes if r["qid"] in ids])
    tests = [test for group in groups.values() for test in group["text_only_binomial_tests"]]
    threshold = alpha / max(1, len(tests))
    for test in tests:
        test["multiple_comparison_threshold"] = threshold
        test["investigate_text_shortcut"] = test["p_one_sided"] < threshold
    by_task = {}
    for task in sorted({q["task_type"] for q in questions}):
        qs = [q for q in questions if q["task_type"] == task]; ids = {q["qid"] for q in qs}
        by_task[task] = _distribution(qs, [r for r in outcomes if r["qid"] in ids])
    return {"schema_version": "interiorgs_capability_model_check_v1", "questions": len(questions),
            "requests": len(predictions), "all": _distribution(questions, outcomes), "by_group": groups,
            "by_task_type": by_task, "predictions": outcomes,
            "interpretation_zh": "这些结果只用于发现文字捷径等明显问题，不替代人工审核，不筛题、不放行训练数据。",
            "binomial_test_zh": "按相同选项数分别检验纯文字答对数是否高于随机水平；能力类别之间用检验次数校正提示阈值。小样本提示不证明文字泄漏。",
            "model_used_for_review": False, "model_used_for_filtering": False}


def markdown(summary: dict[str, Any]) -> str:
    n = summary["questions"]
    lines = [f"本次对同一批{n}道样题分别进行了有图与纯文字检查，共获得{summary['requests']}条回答；结果只供查找问题，不作为人工审核结论。",
             "威尔逊区间是在样本数量有限时给准确率标注的不确定范围；这里使用95%的区间。随机水平按每题选项数计算，常选答案基线表示始终回答本组最常见选项字母的成绩。",
             "", "| 能力类别 | 有图答对／题数 | 纯文字答对／题数 | 随机水平 | 常选字母答对／题数 |",
             "|---|---:|---:|---:|---:|"]
    for group, row in summary["by_group"].items():
        v, t, m = row["with_images"], row["text_only"], row["majority_baseline"]
        lines.append(f"| {group} | {v['correct']}／{v['questions']} | {t['correct']}／{t['questions']} | {row['chance_accuracy']:.1%} | {m['correct']}／{m['questions']}（{m['letter']}） |")
    lines.extend(["", summary["binomial_test_zh"], "详细区间、各选项数量、格式不合规回答和检验结果见同目录summary.json。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    questions = load_questions(args.questions)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("请用新的空输出目录，不覆盖已有模型检查结果")
    args.out.mkdir(parents=True, exist_ok=True)
    predictions = run_inference(args.model, build_requests(questions), max_model_len=args.max_model_len,
                                max_tokens=args.max_tokens, seed=args.seed)
    summary = evaluate_questions(questions, predictions)
    summary["model"] = args.model
    summary["questions_sha256"] = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    summary["inference"] = {"temperature": 0.0, "max_tokens": args.max_tokens,
                             "max_model_len": args.max_model_len, "seed": args.seed, "model_loads": 1}
    (args.out / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in summary.pop("predictions")))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.out / "summary.md").write_text(markdown(summary))
    print(json.dumps({"questions": len(questions), "requests": len(predictions), "model": args.model,
                      "model_loads": 1, "model_used_for_review": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
