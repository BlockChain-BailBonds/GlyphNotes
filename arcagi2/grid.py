"""Core grid utilities for ARC style reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


Coordinate = tuple[int, int]


@dataclass(frozen=True)
class Grid:
    """Immutable wrapper around a 2D grid of colour indices."""

    data: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        height = len(self.data)
        if height == 0:
            raise ValueError("Grid cannot be empty")
        width = len(self.data[0])
        for row in self.data:
            if len(row) != width:
                raise ValueError("Grid rows must be equal length")
            for value in row:
                if not isinstance(value, int):
                    raise TypeError("Grid values must be integers")
                if value < 0:
                    raise ValueError("Grid values must be non-negative")

    @staticmethod
    def from_list(data: Sequence[Sequence[int]]) -> "Grid":
        """Create a grid from a 2D list or tuple."""

        return Grid(tuple(tuple(int(v) for v in row) for row in data))

    def to_list(self) -> list[list[int]]:
        """Return a mutable list representation of the grid."""

        return [list(row) for row in self.data]

    @property
    def height(self) -> int:
        return len(self.data)

    @property
    def width(self) -> int:
        return len(self.data[0])

    def __iter__(self) -> Iterator[tuple[int, ...]]:  # pragma: no cover - delegate
        return iter(self.data)

    def cells(self) -> Iterator[tuple[Coordinate, int]]:
        """Iterate over coordinates and values."""

        for y, row in enumerate(self.data):
            for x, value in enumerate(row):
                yield (x, y), value

    def get(self, x: int, y: int) -> int:
        return self.data[y][x]

    def set(self, x: int, y: int, value: int) -> "Grid":
        rows = [list(row) for row in self.data]
        rows[y][x] = value
        return Grid.from_list(rows)

    def map_colors(self, mapping: dict[int, int], default: int | None = None) -> "Grid":
        """Map colours according to ``mapping`` leaving others untouched."""

        rows = []
        for row in self.data:
            rows.append([
                mapping.get(value, default if default is not None else value)
                for value in row
            ])
        return Grid.from_list(rows)

    def transpose(self) -> "Grid":
        return Grid(tuple(tuple(row[y] for row in self.data) for y in range(self.width)))

    def mirror_horizontal(self) -> "Grid":
        return Grid(tuple(tuple(reversed(row)) for row in self.data))

    def mirror_vertical(self) -> "Grid":
        return Grid(tuple(reversed(self.data)))

    def rotate_right(self) -> "Grid":
        return Grid(tuple(tuple(self.data[self.height - 1 - y][x] for y in range(self.height)) for x in range(self.width)))

    def crop(self, left: int, top: int, right: int, bottom: int) -> "Grid":
        rows = [row[left:right] for row in self.data[top:bottom]]
        return Grid(tuple(tuple(row) for row in rows))

    def pad(self, padding: int, value: int = 0) -> "Grid":
        width = self.width + 2 * padding
        top_bottom = tuple((value,) * width for _ in range(padding))
        body = []
        for row in self.data:
            body.append(tuple([value] * padding + list(row) + [value] * padding))
        return Grid(top_bottom + tuple(body) + top_bottom)

    def replace_color(self, target: int, replacement: int) -> "Grid":
        return self.map_colors({target: replacement})

    def count(self, value: int) -> int:
        return sum(1 for _, cell in self.cells() if cell == value)

    def colors(self) -> set[int]:
        return {cell for _, cell in self.cells()}

    def most_common_color(self) -> int:
        counts: dict[int, int] = {}
        for _, value in self.cells():
            counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get)

    def difference(self, other: "Grid") -> list[tuple[Coordinate, int, int]]:
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grid sizes differ")
        diff: list[tuple[Coordinate, int, int]] = []
        for (coord, value_self), (_, value_other) in zip(self.cells(), other.cells()):
            if value_self != value_other:
                diff.append((coord, value_self, value_other))
        return diff

    def equals(self, other: "Grid") -> bool:
        return self.data == other.data

    def flatten(self) -> tuple[int, ...]:
        return tuple(value for _, value in self.cells())

    def resize(self, scale_x: int, scale_y: int) -> "Grid":
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("Scale must be positive")
        rows = []
        for row in self.data:
            expanded_row = []
            for value in row:
                expanded_row.extend([value] * scale_x)
            for _ in range(scale_y):
                rows.append(tuple(expanded_row))
        return Grid(tuple(rows))

    def paste(self, other: "Grid", offset: Coordinate) -> "Grid":
        ox, oy = offset
        rows = [list(row) for row in self.data]
        for (x, y), value in other.cells():
            tx, ty = x + ox, y + oy
            if 0 <= tx < self.width and 0 <= ty < self.height:
                rows[ty][tx] = value
        return Grid.from_list(rows)

    def bounding_box(self, color_filter: Iterable[int] | None = None) -> tuple[int, int, int, int] | None:
        colors = set(color_filter) if color_filter is not None else None
        coords = [coord for coord, value in self.cells() if colors is None or value in colors]
        if not coords:
            return None
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return min(xs), min(ys), max(xs) + 1, max(ys) + 1

    def subgrid(self, bbox: tuple[int, int, int, int]) -> "Grid":
        left, top, right, bottom = bbox
        return self.crop(left, top, right, bottom)

    def render(self) -> str:
        palette = "0123456789"
        return "\n".join("".join(palette[value] for value in row) for row in self.data)

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return self.render()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Grid(width={self.width}, height={self.height})"


def grid_from_data(data: Sequence[Sequence[int]]) -> Grid:
    return Grid.from_list(data)


def ensure_same_shape(grids: Sequence[Grid]) -> bool:
    widths = {grid.width for grid in grids}
    heights = {grid.height for grid in grids}
    return len(widths) == 1 and len(heights) == 1
