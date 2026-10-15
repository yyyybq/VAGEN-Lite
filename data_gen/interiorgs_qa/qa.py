"""Question records, phrasing families and token counts.

Every question carries several phrasings of the same content (user prompt, section 5.7):

* ``orig``     the original benchmark's wording and option layout (evaluation group 1; also one of the
               training phrasings);
* ``t1``-``t3`` training phrasings: different sentence structures and option layouts, options in random order;
* ``h1``       a held-out phrasing family never used for training (evaluation group 2).

A question kind defines its families with ``Phrasing`` objects; ``build_variants`` renders all of them with
the option order drawn per family from the question's own seed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LETTERS = "ABCDEFGH"
TRAIN_FAMILIES = ("orig", "t1", "t2", "t3")
HELD_OUT_FAMILIES = ("h1",)


# =============================================================================
# Option layouts ("styles")
# =============================================================================
def _opts(options: Sequence[str], fmt: str, sep: str) -> str:
    return sep.join(fmt.format(L=LETTERS[i], o=o) for i, o in enumerate(options))


# version 1 (2026-10-14): the 240 sample questions and the first bank trials were rendered with these; kept so
# that render() can recover a stored question's wording and re-render it with the current layouts
LEGACY_STYLES: Dict[str, Callable[[str, Sequence[str]], str]] = {
    # BLINK prompt field: "...\nSelect from the following choices.\n(A) x\n(B) y"
    "blink": lambda q, o: q + "\nSelect from the following choices.\n" + _opts(o, "({L}) {o}", "\n"),
    "blink_options": lambda q, o: q + " Select from the following options.\n" + _opts(o, "({L}) {o}", "\n"),
    # VSI-Bench doc_to_text (pre_prompt "These are frames of a video.")
    "vsi_mc": lambda q, o: "These are frames of a video.\n" + q + "\nOptions:\n" + _opts(o, "{L}. {o}", "\n")
    + "\nAnswer with the option's letter from the given choices directly.",
    "vsi_na": lambda q, o: "These are frames of a video.\n" + q + "\nPlease answer the question using a single word or phrase.",
    # MMSI-Bench question field: "...\nOptions: A: x, B: y, C: z, D: w"
    "mmsi": lambda q, o: q + "\nOptions: " + _opts(o, "{L}: {o}", ", "),
    # MindCube question field: "... A. x B. y"
    "mindcube": lambda q, o: q + " " + _opts(o, "{L}. {o}", " "),
    # 3DSRBench (VLMEvalKit tsv: question + A/B/C/D columns, VLMEvalKit MCQ prompt)
    "3dsr": lambda q, o: q + "\nOptions:\n" + _opts(o, "{L}. {o}", "\n")
    + "\nPlease select the correct answer from the options above.",
    # ViewSpatial-Bench: question + choices "A. x\nB. y"
    "viewspatial": lambda q, o: q + "\n" + _opts(o, "{L}. {o}", "\n"),
    # EmbSpatial-Bench: question + answer_options (generation protocol; layout not published -> plain)
    "embspatial": lambda q, o: q + "\nOptions:\n" + _opts(o, "({L}) {o}", "\n") + "\nAnswer with the option's letter.",
    # training layouts
    "t_paren": lambda q, o: q + "\n" + _opts(o, "({L}) {o}", "\n") + "\nAnswer with the letter.",
    "t_dot": lambda q, o: q + "\nChoices: " + _opts(o, "{L}. {o}", "; ") + ".\nRespond with one letter.",
    "t_colon": lambda q, o: q + "\n" + _opts(o, "{L}: {o}", "\n"),
    # held-out layout
    "h_inline": lambda q, o: q + " Pick one -- " + _opts(o, "option {L} is \"{o}\"", ", ") + ". Give only the option letter.",
    # numeric answers
    "num": lambda q, o: q + "\nAnswer with a single number.",
    "num_h": lambda q, o: q + " Reply with just the number, no units.",
}


# version 2 (2026-10-15, user decision 1): every "orig" layout is the prompt scripts/easi_eval.py actually sends
# (EASI v0.2.2 runner -> lmms-eval tasks in third_party/lmms-eval/lmms_eval/tasks: vsibench_multiimage, mmsi_bench,
# mindcube_tiny, viewspatial, blink, 3dsrbench_circular, embspatial), answer-format instruction included; the held-out
# wording ends with one uniform instruction; the training wordings are unchanged.
PROMPT_VERSION = 2
MINDCUBE_HEADER = ("[Task]\nYour task is to analyze the spatial arrangement of objects in the scene by examining the provided "
                   "images, which show the scene from different viewpoints.\n[Answer Instruction]\nYou only need to provide "
                   "*ONE* correct answer selecting from the options listed below. For example, if you think the correct answer "
                   "is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be "
                   "'<answer>A. Above</answer>'.\n\n[Question]\n")
HELD_OUT_MC = "Answer with the option's letter only."
HELD_OUT_NUM = "Answer with a number only."


def _blink_note(o: Sequence[str]) -> str:
    # lmms-eval blink: pre_prompt "Note: You only need to respond with {} without providing any additional
    # information.\n" formatted with the choice letters, followed by BLINK's own prompt field
    return ("Note: You only need to respond with " + ", ".join(LETTERS[i] for i in range(len(o)))
            + " without providing any additional information.\n")


STYLES: Dict[str, Callable[[str, Sequence[str]], str]] = {
    # BLINK (blink_doc_to_text): note + prompt field "...\nSelect from the following choices.\n(A) x\n(B) y"
    "blink": lambda q, o: _blink_note(o) + q + "\nSelect from the following choices.\n" + _opts(o, "({L}) {o}", "\n"),
    "blink_options": lambda q, o: _blink_note(o) + q + " Select from the following options.\n" + _opts(o, "({L}) {o}", "\n"),
    # VSI-Bench (vsibench_doc_to_text, default kwargs): the same as version 1
    "vsi_mc": lambda q, o: "These are frames of a video.\n" + q + "\nOptions:\n" + _opts(o, "{L}. {o}", "\n")
    + "\nAnswer with the option's letter from the given choices directly.",
    "vsi_na": lambda q, o: "These are frames of a video.\n" + q + "\nPlease answer the question using a single word or phrase.",
    # MMSI-Bench (msr_doc_to_text): question field with the options inline, then the post_prompt
    "mmsi": lambda q, o: q + "\nOptions: " + _opts(o, "{L}: {o}", ", ")
    + "\nAnswer with the option's letter from the given choices directly. Enclose the option's letter within ``.",
    # MindCube (mindcube_doc_to_text returns the dataset's input_prompt): one fixed header, then the question
    "mindcube": lambda q, o: MINDCUBE_HEADER + q + " " + _opts(o, "{L}. {o}", " "),
    # 3DSRBench (3dsrbench_circular doc_to_text, default post_prompt)
    "3dsr": lambda q, o: "Question: " + q + "\nOptions:\n" + "".join(f"{LETTERS[i]}. {x}\n" for i, x in enumerate(o))
    + "Please select the correct answer from the options above. \n",
    # ViewSpatial-Bench (viewspatial_doc_to_text): choices field is "A. x\nB. y"
    "viewspatial": lambda q, o: "Question: " + q + "\nChoices: " + _opts(o, "{L}. {o}", "\n")
    + "\nReply only to the corresponding option.\nAnswer:",
    # EmbSpatial-Bench (embspatial_doc_to_text): fixed note, question, "A) x" lines
    "embspatial": lambda q, o: ("Note: You only need to respond with A, B, C, D without providing any additional "
                                "information.\n") + q + "\n" + _opts(o, "{L}) {o}", "\n"),
    # training layouts (unchanged)
    "t_paren": lambda q, o: q + "\n" + _opts(o, "({L}) {o}", "\n") + "\nAnswer with the letter.",
    "t_dot": lambda q, o: q + "\nChoices: " + _opts(o, "{L}. {o}", "; ") + ".\nRespond with one letter.",
    "t_colon": lambda q, o: q + "\n" + _opts(o, "{L}: {o}", "\n"),
    # held-out layout: the uniform instruction replaces version 1's "Give only the option letter."
    "h_inline": lambda q, o: q + " Pick one -- " + _opts(o, "option {L} is \"{o}\"", ", ") + ".\n" + HELD_OUT_MC,
    # numeric answers
    "num": lambda q, o: q + "\nAnswer with a single number.",
    "num_h": lambda q, o: q + "\n" + HELD_OUT_NUM,
}


@dataclasses.dataclass
class Phrasing:
    question: Any                         # str.format template (or a list of them) over the question's fields
    style: str
    option_map: Optional[Dict[str, str]] = None   # canonical option key -> text template (None: use keys)


def render_options(keys: Sequence[str], phr: Phrasing, fields: Dict[str, Any]) -> List[str]:
    if phr.option_map is None:
        return [str(k).format(**fields) for k in keys]
    return [phr.option_map.get(k, k).format(**fields) for k in keys]


def build_variants(families: Dict[str, Phrasing], fields: Dict[str, Any], option_keys: Optional[Sequence[str]],
                   answer_key: Any, seed: str, keep_orig_order: Optional[Sequence[str]] = None,
                   numeric: bool = False) -> Dict[str, Dict[str, Any]]:
    """Render every family.  ``option_keys``: canonical options (the answer among them); each family shuffles
    them with its own seeded RNG unless ``keep_orig_order`` fixes the original benchmark's order."""
    out = {}
    for fam, phr in families.items():
        rng = random.Random(f"{seed}|{fam}")
        qt = phr.question
        if isinstance(qt, (list, tuple)):  # several sentence templates within one family: pick one
            qt = rng.choice(list(qt))
        q = qt.format(**fields)
        if numeric or option_keys is None:
            text = STYLES[phr.style](q, [])
            out[fam] = {"family": fam, "text": text, "options": None, "answer": str(answer_key), "style": phr.style,
                        "prompt_version": PROMPT_VERSION}
            continue
        keys = list(option_keys)
        if fam == "orig" and keep_orig_order is not None:
            keys = list(keep_orig_order)
        else:
            rng.shuffle(keys)
        opts = render_options(keys, phr, fields)
        idx = keys.index(answer_key)
        out[fam] = {"family": fam, "text": STYLES[phr.style](q, opts), "options": opts,
                    "option_keys": [str(k) for k in keys], "answer": LETTERS[idx], "answer_text": opts[idx],
                    "style": phr.style, "prompt_version": PROMPT_VERSION}
    return out


def _stem(text: str, style: str, options: Sequence[str], styles: Dict[str, Callable]) -> Optional[str]:
    """The question sentence inside a rendered prompt (the layout is applied around it)."""
    mark = "\x00STEM\x00"
    pre, suf = styles[style](mark, list(options or [])).split(mark)
    if text.startswith(pre) and text.endswith(suf) and len(text) >= len(pre) + len(suf):
        return text[len(pre):len(text) - len(suf)]
    return None


def render(q: Dict[str, Any], fam: str, options: Optional[Sequence[str]] = None) -> str:
    """The prompt of one wording of a stored question in the current layouts (version 2), whichever version it was
    stored with; ``options`` re-renders it with another option order (circular evaluation)."""
    v = q["variants"][fam]
    opts = list(options) if options is not None else list(v.get("options") or [])
    if v.get("prompt_version") == PROMPT_VERSION and options is None:
        return v["text"]
    styles = STYLES if v.get("prompt_version") == PROMPT_VERSION else LEGACY_STYLES
    stem = _stem(v["text"], v["style"], v.get("options") or [], styles)
    if stem is None:
        raise ValueError(f"cannot recover the question of {q.get('qid')} / {fam} (style {v['style']})")
    return STYLES[v["style"]](stem, opts)


def answer_target(q: Dict[str, Any], fam: str, letter: Optional[str] = None,
                  options: Optional[Sequence[str]] = None) -> str:
    """The training answer, written the way the wording's instruction asks for it: MindCube's
    '<answer>A. text</answer>', MMSI's letter in backticks, ViewSpatial's option line 'A. text' (the form of its
    answer field), otherwise the option letter; numeric questions: the number."""
    v = q["variants"][fam]
    if q["answer_type"] == "numeric" or not v.get("options"):
        return str(v["answer"])
    letter = letter or v["answer"]
    opts = list(options) if options is not None else list(v["options"])
    text = opts[LETTERS.index(letter)]
    style = v["style"]
    if style == "mindcube":
        return f"<answer>{letter}. {text}</answer>"
    if style == "mmsi":
        return f"`{letter}`"
    if style == "viewspatial":
        return f"{letter}. {text}"
    return letter


# =============================================================================
# Records
# =============================================================================
def qid(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


@dataclasses.dataclass
class Question:
    qid: str
    group: str
    subtasks: List[str]
    benchmark: str
    scene_id: str
    split: str
    media: str                     # "image" | "multi_image" | "video"
    views: List[str]               # view ids, in the order the model sees them (images: extra["images"])
    variants: Dict[str, Dict[str, Any]]
    answer: Any                    # canonical answer (option key or number)
    answer_type: str               # "mc" | "yesno" | "numeric" | "order"
    gt: Dict[str, Any]             # definition, differs_from_original, evidence values
    objects: List[Dict[str, Any]]  # objects the question names: id, label, name, role
    review: Dict[str, Any] = dataclasses.field(default_factory=dict)  # overlays and notes for the review page
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)
    tokens: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# =============================================================================
# Token counts (Qwen2.5-VL)
# =============================================================================
TOKENIZER = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/"
                 "r1_canonical_dev_eval32_20260915/model_restore/qwen25vl7b_pretrained_cc594898/tokenizer.json")
_TOK = None


def text_tokens(text: str) -> int:
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer

        _TOK = Tokenizer.from_file(str(TOKENIZER))
    return len(_TOK.encode(text, add_special_tokens=False).ids)


def smart_resize(h: int, w: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 28 * 28 * 1280):
    """Qwen2.5-VL image resizing (qwen_vl_utils.smart_resize)."""
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = math.floor(h / beta / factor) * factor
        wb = math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def image_tokens(size: int) -> int:
    """LLM tokens of one size x size image: (H/28) * (W/28) after smart_resize (patch 14, 2x2 merge)."""
    h, w = smart_resize(size, size)
    return (h // 28) * (w // 28)


def video_tokens(n_frames: int, size: int) -> int:
    """Frames given as one video: temporal patch 2 merges frame pairs."""
    return int(math.ceil(n_frames / 2)) * image_tokens(size)


CHAT_OVERHEAD = 21  # Qwen2-VL ChatML: system prompt, role tags, <|im_end|> markers (measured on the template)
PER_IMAGE_SPECIAL = 2  # <|vision_start|> <|vision_end|>


def count_tokens(q: Question, size: int, family: str = "orig") -> Dict[str, int]:
    v = q.variants[family]
    n_img = len(q.views)
    if q.media == "video":
        vis = video_tokens(n_img, size)
        specials = PER_IMAGE_SPECIAL
    else:
        vis = n_img * image_tokens(size)
        specials = n_img * PER_IMAGE_SPECIAL
    t_in = text_tokens(v["text"])
    t_out = text_tokens(str(v["answer"]))
    return {"visual": vis, "text_prompt": t_in, "answer": t_out,
            "total": CHAT_OVERHEAD + specials + vis + t_in + t_out + 1}


# ----------------------------------------------------------------------------------------------------------------
# replaced questions: later runs regenerate some groups/subtasks; the report and the review site drop the old ones
# ----------------------------------------------------------------------------------------------------------------
def parse_exclude(spec: str) -> List[Tuple[str, ...]]:
    """'RUN_SUFFIX:GROUP[:SUBTASK],qid:QID,...' -> rules (RUN_SUFFIX matches the end of the run directory path)."""
    return [tuple(x.split(":")) for x in spec.split(",") if x.strip()]


def excluded(run: Any, q: Dict[str, Any], rules: Sequence[Tuple[str, ...]]) -> bool:
    for rule in rules:
        if rule[0] == "qid":
            if q["qid"] == rule[1]:
                return True
            continue
        if not str(run).rstrip("/").endswith(rule[0]) or q["group"] != rule[1]:
            continue
        if len(rule) < 3 or q["subtasks"][0] == rule[2]:
            return True
    return False
