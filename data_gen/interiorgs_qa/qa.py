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


STYLES: Dict[str, Callable[[str, Sequence[str]], str]] = {
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
            out[fam] = {"family": fam, "text": text, "options": None, "answer": str(answer_key), "style": phr.style}
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
                    "style": phr.style}
    return out


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
