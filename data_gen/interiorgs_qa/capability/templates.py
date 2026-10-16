"""Equivalent English capability questions with answer-independent randomization."""
from __future__ import annotations

import hashlib
import json
import random
import re


TEMPLATE_VERSION = "v2"
SYMMETRIC_KINDS = frozenset(("equidistance_closer", "occlusion_closer"))


TEMPLATES = {
    "absolute_distance": (
        "About how far are you from {a}, measured horizontally to its nearest side?",
        "What is your approximate horizontal distance to the nearest side of {a}?",
        "Which distance best describes how far the nearest side of {a} is from you along the ground?",
    ),
    "equidistance_closer": (
        "Which is closer to you, {a} or {b}, measuring horizontally to their nearest sides?",
        "Compare your horizontal distances to the nearest sides of {a} and {b}. Which is closer?",
        "Are you nearer to {a}, nearer to {b}, or about equally far from their nearest sides along the ground?",
    ),
    "occlusion_closer": (
        "Which is closer to you, {a} or {b}, measured horizontally to their nearest sides?",
        "Of {a} and {b}, whose nearest side is closer to you along the ground?",
        "Compare your horizontal distances to {a} and {b}: which nearest side is closer?",
    ),
    "projective_direction": (
        "From your viewpoint, in which main direction is {a} relative to {b}?",
        "Looking from your current position, which direction best describes {a} relative to {b}?",
        "In this view, is {a} mainly to the left, to the right, in front of, or behind {b}?",
    ),
    "fov_complete": (
        "Are {a} and {b} both fully inside the image, without being cut off by its edges?",
        "Which of {a} and {b} fit completely within the image boundaries?",
        "For {a} and {b}, which are fully in the picture rather than cropped by an image edge?",
    ),
    "occlusion_state": (
        "Use the first image to identify {a}. In the second image, is {a} completely hidden by {b}, partly hidden by {b}, not hidden by {b}, or outside the image?",
        "Use the first image to identify {a}. Looking only at the second viewpoint, how much does {b} block {a}, or is {a} outside the image?",
        "Find {a} in the first image, then describe its visibility in the second image relative to {b}. This asks about the shown views, not a future action.",
    ),
}


def _identity(identity):
    if isinstance(identity, dict):
        for key, value in identity.items():
            if re.search(r"answer|correct|label|ground.?truth|branch_intent", str(key), re.I):
                raise ValueError("sampling identity must not contain an answer or label")
            _identity(value)
    elif isinstance(identity, (list, tuple)):
        for value in identity:
            _identity(value)
    elif isinstance(identity, str):
        if re.search(r"(?:answer|correct|label|ground.?truth)\s*[=:]", identity, re.I):
            raise ValueError("sampling identity must not contain an answer or label")
    elif identity is not None and not isinstance(identity, (int, float, bool)):
        raise ValueError("sampling identity must be JSON-compatible")


def _mention_roles(kind, identity, seed):
    """One answer-independent draw, separate from wording/option-order RNG."""
    material = json.dumps({"seed": seed, "identity": identity, "kind": kind,
                           "purpose": "symmetric_mention_order_v2"},
                          sort_keys=True, ensure_ascii=False, allow_nan=False)
    rng = random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest(), "big"))
    return ["b", "a"] if rng.randrange(2) else ["a", "b"]


def render_draft(draft: dict, identity, seed: int) -> dict:
    """Use only opaque state identity, kind and seed to choose wording/order.

    ``options`` contains strings; parallel ``option_keys`` retain semantic keys.
    ``answer`` is one letter. ``source_answer_key`` keeps the formal object roles;
    returned answer/option keys refer to the presented mention order for the
    two symmetric nearer-object kinds only. No source draft is mutated.
    No correctness, geometric measurements or answer values enter the RNG seed.
    """
    _identity(identity)
    kind, fields = draft["kind"], dict(draft["fields"])
    if kind not in TEMPLATES:
        raise ValueError("template kind is not enabled")
    keys = list(draft["option_keys"])
    if len(keys) != len(set(keys)) or draft["answer_key"] not in keys:
        raise ValueError("invalid option keys or answer key")
    source_answer_key = answer_key = draft["answer_key"]
    mentioned_roles = ["a", "b"] if "b" in fields else ["a"]
    if kind in SYMMETRIC_KINDS:
        mentioned_roles = _mention_roles(kind, identity, seed)
        if mentioned_roles == ["b", "a"]:
            fields["a"], fields["b"] = fields["b"], fields["a"]
            mapping = {"a_closer": "b_closer", "b_closer": "a_closer", "same": "same"}
            keys = [mapping[key] for key in keys]
            answer_key = mapping[answer_key]
    material = json.dumps({"seed": seed, "identity": identity, "kind": kind}, sort_keys=True, ensure_ascii=False, allow_nan=False)
    rng = random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest(), "big"))
    template_index = rng.randrange(len(TEMPLATES[kind]))
    if kind == "absolute_distance":
        values = fields["distance_options_m"]
        if keys != [f"distance_{i}" for i in range(4)] or len(values) != 4 or values != sorted(set(values)):
            raise ValueError("distance options must be the four original increasing choices")
        texts = {key: f"{value:g} meters" for key, value in zip(keys, values)}
    else:
        texts = {
            "a_closer": "Closer to " + fields["a"] + ".",
            "b_closer": "Closer to " + fields["b"] + ".",
            "same": "They are about equally far away.",
            "left": "To its left.", "right": "To its right.",
            "front": "In front of it (closer to you).", "behind": "Behind it (farther from you).",
            "both": "Both are fully inside the image.",
            "a_only": "Only " + fields["a"] + " is fully inside the image.",
            "b_only": "Only " + fields["b"] + " is fully inside the image.",
            "neither": "Neither is fully inside the image.",
            "fully_hidden": "Completely hidden behind " + fields["b"] + ".",
            "partly_hidden": "Partly hidden behind " + fields["b"] + ".",
            "not_hidden": "Not hidden by " + fields["b"] + ".",
            "outside": "Outside the second image's field of view.",
        }
        rng.shuffle(keys)
    options = [texts[key] for key in keys]
    return {"question": TEMPLATES[kind][template_index].format(**fields) + " Answer with one option letter only.",
            "options": options, "option_keys": keys, "answer": chr(65+keys.index(answer_key)),
            "source_answer_key": source_answer_key, "mentioned_object_roles": mentioned_roles,
            "answer_key": answer_key, "template_id": f"{kind}_{TEMPLATE_VERSION}_{template_index}"}
