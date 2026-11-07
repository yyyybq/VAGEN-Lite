"""Self mode of G-oracle's evaluation (the user's instruction of 2026-11-07, item 6c; ``evaluation run --self-marker``):
before every action the same model is asked to box each target of the task, with Pg's own question ("Draw a box
around {ref}. ...", data_gen.transition_qa.pointing_qa.QUESTION, 0-1000 coordinates; one user turn with the frame and no
system prompt, as the pointing probe asks), its boxes are drawn into the frame in the oracle marker's style
(oracle_marker.draw: A red, B blue), and the action call sees that frame.  An answer without four numbers draws no box
for its target.  Each turn records the answers, the boxes read, the truth (pointing.truth) and the IoU of each box with
its target's true box (None when the target has no true box: not clearly visible)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

FRAME = 256.0


def questions(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(letter, question) per object of the task text, A, B in the text's order."""
    from data_gen.transition_qa.pointing_qa import QUESTION
    from vagen.envs.active_spatial.pointing import SLOT_LETTERS, target_refs

    return [(l, QUESTION[("box", "norm1000")].format(ref=ref)) for l, ref in zip(SLOT_LETTERS, target_refs(item))]


def read_box(text: str) -> Optional[List[float]]:
    """The box an answer gives, in the 256-frame's pixels (0-1000 coordinates scaled, corners ordered, clipped to the
    frame); None without four numbers or with an empty box."""
    from eval.diagnostics.qa_probe import _clean, read_coordinates

    nums, _ = read_coordinates(_clean(text), 4)
    if len(nums) < 4:
        return None
    s = FRAME / 1000.0
    x1, x2 = sorted((nums[0] * s, nums[2] * s))
    y1, y2 = sorted((nums[1] * s, nums[3] * s))
    x1, y1, x2, y2 = (min(max(v, 0.0), FRAME) for v in (x1, y1, x2, y2))
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)] if x2 > x1 and y2 > y1 else None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mark_turn(ask: Callable[[str, Any], str], image: Any, item: Dict[str, Any], truth_rows: Sequence[Dict[str, Any]]
              ) -> Tuple[Any, List[Dict[str, Any]]]:
    """(the frame with the model's own boxes, per target {letter, answer, box, truth_box, iou})."""
    from vagen.envs.active_spatial.oracle_marker import draw

    truth = {r["letter"]: r.get("box") for r in truth_rows}
    marks, rec = [], []
    for letter, q in questions(item):
        answer = ask(q, image)
        box = read_box(answer)
        tb = truth.get(letter)
        rec.append({"letter": letter, "answer": answer, "box": box, "truth_box": tb,
                    "iou": round(iou(box, tb), 4) if (box and tb) else (0.0 if tb else None)})
        if box:
            marks.append((letter, box))
    return draw(image, marks), rec


def summary(turns: Sequence[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Over an episode's turns: questions, boxes read, the mean IoU and the share with IoU >= 0.5 over the targets with
    a true box, and the boxes drawn for targets without one."""
    recs = [r for t in turns for r in t]
    vis = [r for r in recs if r["truth_box"]]
    return {"questions": len(recs), "boxes": sum(r["box"] is not None for r in recs), "visible": len(vis),
            "iou_mean": round(sum(r["iou"] for r in vis) / len(vis), 4) if vis else None,
            "iou_ge_05": sum(r["iou"] >= 0.5 for r in vis),
            "boxes_without_truth": sum(r["box"] is not None and not r["truth_box"] for r in recs)}
