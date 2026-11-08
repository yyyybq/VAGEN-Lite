"""process_docs of sat_swap_ee: every SAT item with its options in reversed order."""

from __future__ import annotations


def reverse_options(doc: dict) -> dict:
    answers = list(doc["answers"])
    return {"answers": answers[::-1], "correct_answer_idx": len(answers) - 1 - int(doc["correct_answer_idx"])}


def process_docs(dataset):
    return dataset.map(reverse_options)
