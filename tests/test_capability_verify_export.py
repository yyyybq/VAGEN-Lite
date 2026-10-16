import copy

import pytest

from data_gen.interiorgs_qa.capability.verify_export import (
    validate_records, verify_encoded, verify_public_projection, verify_image_bindings,
)


def record(images=None):
    images = images or ["/tmp/raw_a.png", "/tmp/raw_b.png"]
    return {"messages": [{"role": "user", "content": "<image>\n" * len(images) + "Which object is nearer?\nA. First.\nB. Second.\nC. Same distance."},
                         {"role": "assistant", "content": "C"}], "images": images, "videos": []}


class Tokenizer:
    def convert_tokens_to_ids(self, text):
        return {"<|image_pad|>": 0}[text]

    def decode(self, values, skip_special_tokens=False):
        lookup = {0: "<|image_pad|>", 1: "<|vision_start|>", 2: "C", 3: "<|im_end|>", 4: "\n", 5: "user question"}
        return "".join(lookup[x] for x in values)


def encoded():
    return {"input_ids": [1, 0, 1, 0, 5, 2, 3, 4],
            "labels": [-100, -100, -100, -100, -100, 2, 3, 4],
            "images": record()["images"], "videos": []}


def test_export_accepts_one_and_two_images_without_mutating():
    rows = [record(), record(["/tmp/raw_a.png"])]
    original = copy.deepcopy(rows)
    validate_records(rows)
    assert rows == original


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(evidence_zh=["private"]),
    lambda r: r["messages"][0].update(content=r["messages"][0]["content"] + " camera_position=[0,0,1.5]"),
    lambda r: r["messages"][1].update(content="C because it looks equal"),
    lambda r: r["messages"][1].update(content="D"),
    lambda r: r.update(videos=["unused.mp4"]),
    lambda r: r.update(images=["/tmp/overlay.png", "/tmp/raw_b.png"]),
    lambda r: r.update(images=["/tmp/raw_a.png"]),
])
def test_export_rejects_private_or_malformed_inputs(mutation):
    row = record()
    mutation(row)
    with pytest.raises(ValueError):
        validate_records([row])


def test_empty_export_does_not_pass():
    with pytest.raises(ValueError, match="empty"):
        validate_records([])


def test_public_projection_preserves_order_and_excludes_private_fields():
    row = record()
    q = {"question": "Which object is nearer?", "options": ["First.", "Second.", "Same distance."],
         "answer": "C", "images": row["images"], "evidence_zh": ["private geometry"], "source": {"c2w": [1, 2, 3]}}
    verify_public_projection([row], [q])
    bad = copy.deepcopy(row)
    bad["images"].reverse()
    with pytest.raises(ValueError, match="projection"):
        verify_public_projection([bad], [q])


def test_only_answer_end_and_newline_are_supervised():
    check = verify_encoded(encoded(), record(), Tokenizer())
    assert check["supervised_decoded"] == "C<|im_end|>\n"
    assert check["ignored_label_tokens"] == 5
    assert check["supervised_token_count"] == 3


def test_image_receipt_hashes_are_checked_in_order():
    images = record()["images"]
    evidence = {images[0]: {"sha256": "original_a"}, images[1]: {"sha256": "original_b"}}
    question = {"images": images, "source": {"image_sha256": ["original_a", "original_b"]}}
    verify_image_bindings([question], evidence)
    question["source"]["image_sha256"].reverse()
    with pytest.raises(ValueError, match="ordered image hashes"):
        verify_image_bindings([question], evidence)


def test_missing_image_receipt_does_not_pass():
    images = record()["images"]
    evidence = {path: {"sha256": "hash"} for path in images}
    with pytest.raises(ValueError, match="ordered image hashes"):
        verify_image_bindings([{"images": images}], evidence)


def test_supervised_prompt_is_detected():
    row = encoded()
    row["labels"][4] = 5
    with pytest.raises(ValueError, match="more than answer"):
        verify_encoded(row, record(), Tokenizer())


def test_loader_image_reordering_is_detected():
    row = encoded()
    row["images"] = list(reversed(row["images"]))
    with pytest.raises(ValueError, match="Image order"):
        verify_encoded(row, record(), Tokenizer())


def test_ignored_answer_does_not_pass():
    row = encoded()
    row["labels"] = [-100] * len(row["labels"])
    with pytest.raises(ValueError, match="No supervised"):
        verify_encoded(row, record(), Tokenizer())
