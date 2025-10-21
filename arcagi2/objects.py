"""Object perception utilities for ARC tasks."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from .grid import Coordinate, Grid


@dataclass(frozen=True)
class Object:
    """Connected set of coloured pixels."""

    pixels: frozenset[Coordinate]
    colors: Counter
    bounding_box: tuple[int, int, int, int]
    anchor: Coordinate

    def translated(self, dx: int, dy: int) -> "Object":
        moved = frozenset((x + dx, y + dy) for x, y in self.pixels)
        left, top, right, bottom = self.bounding_box
        bbox = (left + dx, top + dy, right + dx, bottom + dy)
        ax, ay = self.anchor
        return Object(moved, self.colors, bbox, (ax + dx, ay + dy))


def _neighbors(x: int, y: int) -> Iterable[Coordinate]:
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1


def extract_objects(grid: Grid, background: int | None = None) -> list[Object]:
    """Return connected components from ``grid``.

    ``background`` may contain a colour that should be ignored when forming objects.
    """

    visited: set[Coordinate] = set()
    objects: list[Object] = []
    bg = background
    for (x, y), value in grid.cells():
        if (x, y) in visited:
            continue
        if bg is not None and value == bg:
            visited.add((x, y))
            continue
        queue: deque[Coordinate] = deque([(x, y)])
        pixels: set[Coordinate] = set()
        counter: Counter = Counter()
        left = right = x
        top = bottom = y
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            cv = grid.get(cx, cy)
            if bg is not None and cv == bg:
                continue
            pixels.add((cx, cy))
            counter[cv] += 1
            left = min(left, cx)
            right = max(right, cx)
            top = min(top, cy)
            bottom = max(bottom, cy)
            for nx, ny in _neighbors(cx, cy):
                if 0 <= nx < grid.width and 0 <= ny < grid.height and (nx, ny) not in visited:
                    queue.append((nx, ny))
        if pixels:
            anchor = (left, top)
            objects.append(
                Object(
                    pixels=frozenset(pixels),
                    colors=counter,
                    bounding_box=(left, top, right + 1, bottom + 1),
                    anchor=anchor,
                )
            )
    return objects


def object_grid(grid: Grid, obj: Object, fill: int | None = None) -> Grid:
    left, top, right, bottom = obj.bounding_box
    width = right - left
    height = bottom - top
    rows = [[fill if fill is not None else grid.most_common_color() for _ in range(width)] for _ in range(height)]
    for x, y in obj.pixels:
        rows[y - top][x - left] = grid.get(x, y)
    return Grid.from_list(rows)


def translate(grid: Grid, obj: Object, dx: int, dy: int, background: int) -> Grid:
    canvas = [[background for _ in range(grid.width)] for _ in range(grid.height)]
    for (x, y), value in grid.cells():
        if (x, y) not in obj.pixels:
            canvas[y][x] = value
    for x, y in obj.pixels:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            canvas[ny][nx] = grid.get(x, y)
    return Grid.from_list(canvas)


def centroid(obj: Object) -> tuple[float, float]:
    xs = [x for x, _ in obj.pixels]
    ys = [y for _, y in obj.pixels]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def objects_match(a: Object, b: Object) -> bool:
    return a.colors == b.colors and len(a.pixels) == len(b.pixels)


def translate_vector(a: Object, b: Object) -> tuple[int, int]:
    ax, ay = centroid(a)
    bx, by = centroid(b)
    return round(bx - ax), round(by - ay)


def difference_objects(source: Sequence[Object], target: Sequence[Object]) -> list[tuple[Object, Object, tuple[int, int]]]:
    matches: list[tuple[Object, Object, tuple[int, int]]] = []
    for obj in source:
        best: tuple[Object, tuple[int, int]] | None = None
        for candidate in target:
            if objects_match(obj, candidate):
                dx, dy = translate_vector(obj, candidate)
                best = (candidate, (dx, dy))
                break
        if best:
            matches.append((obj, best[0], best[1]))
    return matches
