"""Is a raw model answer usable at all?  One rule for every model and backend, independent of the backend's own
scoring, so that the "unparseable answer" share in our tables means the same thing everywhere.

- Multiple choice: usable if the answer names exactly one option letter (A., (A), "A", "Answer: A", <answer>A</answer>)
  or contains exactly one option's text.
- Numeric: usable if the answer contains a number (digits or a number word up to twenty).
"""

from __future__ import annotations

import re

WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_TAGS = re.compile(r"<think>.*?</think>", re.S | re.I)
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def _core(text: str) -> str:
    """The part of a reply that holds the answer: inside <answer> if present, else the reply without <think>."""
    text = text or ""
    m = _ANSWER_TAG.findall(text)
    if m:
        return m[-1]
    return _TAGS.sub(" ", text)


def mcq_letter(text: str, n_options: int, options: list[str] | None = None) -> str | None:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options]
    core = _core(text).strip()
    m = re.fullmatch(r"[\s\(\[]*([A-Z])[\s\)\]\.:]*", core)
    if m and m.group(1) in letters:
        return m.group(1)
    found = set(re.findall(rf"(?:^|[\s\(\[:])([{letters}])(?=$|[\s\)\]\.,:])", core))
    if len(found) == 1:
        return found.pop()
    if options:
        hits = [letters[i] for i, o in enumerate(options[:n_options])
                if o and re.sub(r"^[A-Z][\.\)]\s*", "", o).strip().lower() in core.lower()]
        if len(hits) == 1:
            return hits[0]
    return None


def number(text: str) -> float | None:
    core = _core(text)
    m = re.search(r"-?\d+(?:\.\d+)?", core.replace(",", ""))
    if m:
        return float(m.group(0))
    for w in re.findall(r"[a-z]+", core.lower()):
        if w in WORDS:
            return float(WORDS[w])
    return None
