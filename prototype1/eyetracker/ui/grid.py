from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[int, int]

@dataclass(frozen=True)
class GridSpec:
    cols: int = 5
    rows: int = 5
    margin_ratio: float = 0.08  # 8% screen margin on each side

def make_grid(screen_w: int, screen_h: int, spec: GridSpec = GridSpec()) -> List[Point]:
    """Return 2D pixel coordinates of a cols×rows grid within screen, with outer margins."""
    mx = int(round(spec.margin_ratio * screen_w))
    my = int(round(spec.margin_ratio * screen_h))
    w = screen_w - 2 * mx
    h = screen_h - 2 * my
    xs = [mx + int(round(i * (w / (spec.cols - 1)))) for i in range(spec.cols)]
    ys = [my + int(round(j * (h / (spec.rows - 1)))) for j in range(spec.rows)]
    return [(x, y) for y in ys for x in xs]

def _serpentine_indices(cols: int, rows: int) -> List[int]:
    """Row-major traversal that alternates direction each row (short saccades)."""
    order: List[int] = []
    for r in range(rows):
        row_indices = [r * cols + c for c in range(cols)]
        if r % 2 == 1:
            row_indices.reverse()
        order.extend(row_indices)
    return order

def latin_like_order(cols: int, rows: int, seed: int = 17) -> List[int]:
    """
    Reproducible pseudo-random traversal that avoids long saccades:
    - serpentine backbone
    - per-row cyclic shift with seeded offsets
    - start row chosen by seed to vary vertical entry point
    """
    base = _serpentine_indices(cols, rows)
    rng = random.Random(seed)

    # split rows, apply a cyclic shift per row
    per_row = [base[r * cols:(r + 1) * cols] for r in range(rows)]
    for r in range(rows):
        shift = rng.randint(0, max(0, cols - 1))
        per_row[r] = per_row[r][shift:] + per_row[r][:shift]
        # maintain serpentine reversal semantics (already in base)
    # stitch back together
    stitched = [idx for row in per_row for idx in row]

    # rotate rows as a block to vary vertical start
    start_row = rng.randrange(rows)
    k = start_row * cols
    stitched = stitched[k:] + stitched[:k]
    return stitched

def sequence(screen_w: int, screen_h: int, spec: GridSpec = GridSpec(), seed: int = 17) -> List[Point]:
    """Grid points in a latin-like order (reproducible via seed)."""
    pts = make_grid(screen_w, screen_h, spec)
    order = latin_like_order(spec.cols, spec.rows, seed=seed)
    return [pts[i] for i in order]
