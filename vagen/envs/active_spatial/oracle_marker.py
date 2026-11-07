"""Oracle marker (the user's instruction of 2026-11-07, item 6): the task's targets' truth boxes drawn into the frame
the model sees.  The env config switch ``oracle_marker`` (default off) turns it on for evaluation
(``evaluation run --oracle-marker``, item 6a's check and G-oracle's oracle mode) and for RL (rl-config
``--oracle-marker``, item 6c); the SFT frames of G-oracle are drawn by the same function.

The boxes are pointing.truth's, the truth of group G's labels and pointing reward: per object of the task text (A, B
in the text's order), the projected 3D box clipped to the frame, only when the object is clearly visible from the pose
(the success metrics' face rule with the scene's occluders); an object not clearly visible gets no box.  The style is
fixed (STYLE) and drawn by P0's data_gen.interiorgs_qa.draw.draw_boxes: A red, B blue, a 2 px outline on the
256 x 256 frame, the letter on a filled tag above the box's top-left corner.  The boxes are drawn on a copy after the
env's image-quality check, so the "nearly uniform view" stop reads the unmarked frame; success and rewards never read
the image."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

STYLE: Dict[str, Any] = {
    "boxes": "vagen.envs.active_spatial.pointing.truth: per object of the task text, the projected 3D box clipped to "
             "the frame, only when the object is clearly visible (face rule with the scene's occluders)",
    "colors": {"A": [230, 30, 40], "B": [30, 110, 230]},
    "outline_px": "max(2, frame width // 128): 2 on the 256 x 256 frame",
    "label": "the slot letter, white DejaVu Sans Bold max(18, width // 22) px on a tag of the box's colour above its "
             "top-left corner (else inside or below it)",
    "draw": "data_gen.interiorgs_qa.draw.draw_boxes",
    "frame": [256, 256],
}


def boxes(item: Dict[str, Any], c2w: Any, occ: Any) -> List[Tuple[str, List[float]]]:
    """(letter, [x1, y1, x2, y2] in the 256 x 256 frame's pixels) of the targets clearly visible from ``c2w``."""
    from .pointing import truth

    return [(r["letter"], list(r["box"])) for r in truth(item, c2w, occ) if r.get("box")]


def draw(image: Any, marks: Sequence[Tuple[str, Sequence[float]]]) -> Any:
    """A copy of ``image`` with ``marks`` drawn in STYLE (boxes in 256-frame pixels, scaled to the image)."""
    from data_gen.interiorgs_qa.draw import draw_boxes

    from .pointing import FRAME_SIZE

    out = image.copy().convert("RGB")
    if marks:
        k = out.size[0] / float(FRAME_SIZE[0])
        draw_boxes(out, [tuple(b) for _, b in marks], [l for l, _ in marks],
                   colors=[tuple(STYLE["colors"][l]) for l, _ in marks], k=k)
    return out


def mark(image: Any, item: Dict[str, Any], c2w: Any, occ: Any) -> Any:
    return draw(image, boxes(item, c2w, occ))
