"""Marks drawn INTO model-input images (BLINK-style points and boxes) and picking points on surfaces."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

RED = (230, 30, 40)
COLORS = [(230, 30, 40), (30, 110, 230), (20, 160, 70), (230, 160, 0)]


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_points(img, points: Sequence[Tuple[float, float]], labels: Sequence[str], color=RED) -> None:
    """BLINK relative depth / correspondence: a circle per point with its label beside it."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    s = img.size[0]
    r = max(6, s // 50)
    f = _font(max(12, s // 22))
    for (x, y), lab in zip(points, labels):
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=max(2, s // 170))
        tx = x + r + 2 if x < s - 4 * r else x - r - 2 - f.getlength(lab)
        ty = y - r - 2 if y > 3 * r else y + r
        d.rectangle([tx - 1, ty, tx + f.getlength(lab) + 1, ty + f.size + 2], fill=(255, 255, 255))
        d.text((tx, ty), lab, fill=color, font=f)


def draw_boxes(img, boxes: Sequence[Tuple[float, float, float, float]], labels: Sequence[str],
               colors: Optional[Sequence[Tuple[int, int, int]]] = None) -> None:
    """BLINK object localisation: labelled rectangles."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    s = img.size[0]
    f = _font(max(12, s // 22))
    colors = colors or COLORS
    for k, (b, lab) in enumerate(zip(boxes, labels)):
        c = colors[k % len(colors)]
        d.rectangle(list(b), outline=c, width=max(2, s // 128))
        tx, ty = b[0] + 3, max(0, b[1] - f.size - 4) if b[1] > f.size + 4 else b[1] + 3
        d.rectangle([tx - 1, ty, tx + f.getlength(lab) + 2, ty + f.size + 2], fill=c)
        d.text((tx, ty), lab, fill=(255, 255, 255), font=f)


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(r):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


def flat_surface_points(mask: np.ndarray, depth: np.ndarray, erode_px: int, max_std: float = 0.03,
                        border: int = 16, rng=None, n: int = 1) -> List[Tuple[int, int]]:
    """Pixels deep inside ``mask`` (away from object edges) where depth is locally smooth (7x7 std below
    ``max_std`` m), away from the image border."""
    m = erode(mask, erode_px)
    s = mask.shape[0]
    m[:border] = m[-border:] = False
    m[:, :border] = m[:, -border:] = False
    ys, xs = np.nonzero(m)
    if not len(xs):
        return []
    order = np.arange(len(xs))
    if rng is not None:
        rng.shuffle(order)
    out = []
    for i in order[:400]:
        y, x = int(ys[i]), int(xs[i])
        win = depth[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        if float(win.std()) <= max_std:
            out.append((x, y))
            if len(out) >= n:
                break
    return out
