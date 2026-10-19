"""Marks drawn INTO model-input images (BLINK-style points and boxes) and picking points on surfaces."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

RED = (230, 30, 40)
COLORS = [(230, 30, 40), (30, 110, 230), (20, 160, 70), (230, 160, 0)]


def _font_files() -> List[str]:
    """DejaVu Sans Bold: matplotlib's copy first (it is in the shared conda env, so the SCO workers have it
    too; their containers have no system DejaVu, and until 2026-10-19 the marks there fell back to PIL's
    10 px default font), then the system copies."""
    import importlib.util
    import os

    files = []
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.origin:
        files.append(os.path.join(os.path.dirname(spec.origin), "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf"))
    return files + ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


@lru_cache(maxsize=None)
def _font(size: int):
    from PIL import ImageFont

    for name in _font_files():
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    warnings.warn("DejaVuSans-Bold.ttf not found; marks use PIL's built-in font")
    return ImageFont.load_default(size)


def mark_style(size: int) -> Tuple[int, int, int]:
    """Circle radius, font size and circle line width for marks drawn into a ``size``-pixel image.

    Marks are drawn after the image is resized (``ViewBank.save_marked``), never shrunk with it: a 512 image
    keeps radius 10, font 23, width 3; a 256 image gets radius 7, font 18, width 2, so its capital letters are
    about 13 px tall (the 512 marks shrunk to 256 were about 8 px and the user could not read them, 2026-10-19).
    """
    return max(7, size // 50), max(18, size // 22), max(2, size // 170)


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place(cands, s: int, taken) -> Tuple[float, float, float, float]:
    """The first label rectangle (x0, y0, x1, y1) that lies inside the image and overlaps nothing in ``taken``;
    if none does, the first one moved inside the image."""
    for rect in cands:
        if rect[0] >= 0 and rect[1] >= 0 and rect[2] <= s and rect[3] <= s and not any(_overlaps(rect, t) for t in taken):
            return rect
    x0, y0, x1, y1 = cands[0]
    dx = max(0.0, -x0) - max(0.0, x1 - s)
    dy = max(0.0, -y0) - max(0.0, y1 - s)
    return x0 + dx, y0 + dy, x1 + dx, y1 + dy


def draw_points(img, points: Sequence[Tuple[float, float]], labels: Sequence[str], color=RED, k: float = 1.0) -> None:
    """BLINK relative depth / correspondence: a circle per point with its label beside it.

    ``points`` are in render pixels and ``k`` maps them to this image (0.5 for the 256 copy of a 512 render).
    A label goes to the upper right of its circle (the side with room first), else to another corner, so that
    it stays inside the image and covers no other circle or label."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    s = img.size[0]
    r, fs, w = mark_style(s)
    f = _font(fs)
    pts = [(x * k, y * k) for x, y in points]
    rings = [(x - r - 1, y - r - 1, x + r + 1, y + r + 1) for x, y in pts]
    labels_done: List[Tuple[float, float, float, float]] = []
    for i, ((x, y), lab) in enumerate(zip(pts, labels)):
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=w)
        tw = f.getlength(lab)
        xs = [x + r + 2, x - r - 2 - tw] if x < s - 4 * r else [x - r - 2 - tw, x + r + 2]
        ys = [y - r - 2, y + r] if y > 3 * r else [y + r, y - r - 2]
        cands = [(tx - 1, ty, tx + tw + 1, ty + fs + 2) for ty in ys for tx in xs]
        rect = _place(cands, s, labels_done + [g for j, g in enumerate(rings) if j != i])
        labels_done.append(rect)
        d.rectangle(list(rect), fill=(255, 255, 255))
        d.text((rect[0] + 1, rect[1]), lab, fill=color, font=f)


def draw_boxes(img, boxes: Sequence[Tuple[float, float, float, float]], labels: Sequence[str],
               colors: Optional[Sequence[Tuple[int, int, int]]] = None, k: float = 1.0) -> None:
    """BLINK object localisation: labelled rectangles (``boxes`` in render pixels, ``k`` as in ``draw_points``).
    A label sits above the box's top-left corner, else inside it or below it, without covering another label."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    s = img.size[0]
    _, fs, _ = mark_style(s)
    f = _font(fs)
    colors = colors or COLORS
    labels_done: List[Tuple[float, float, float, float]] = []
    for i, (b, lab) in enumerate(zip(boxes, labels)):
        b = [v * k for v in b]
        c = colors[i % len(colors)]
        d.rectangle(b, outline=c, width=max(2, s // 128))
        tw = f.getlength(lab)
        tx = b[0] + 3
        ys = ([max(0, b[1] - fs - 4)] if b[1] > fs + 4 else []) + [b[1] + 3, b[3] + 3, b[3] - fs - 5]
        rect = _place([(tx - 1, ty, tx + tw + 2, ty + fs + 2) for ty in ys], s, labels_done)
        labels_done.append(rect)
        d.rectangle(list(rect), fill=c)
        d.text((rect[0] + 1, rect[1]), lab, fill=(255, 255, 255), font=f)


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
