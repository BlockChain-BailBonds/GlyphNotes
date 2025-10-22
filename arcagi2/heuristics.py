"""Heuristic tools used by the ARC-AGI 2 solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .grid import Grid
from .objects import extract_objects, difference_objects

Transform = Callable[[Grid], Grid]
TrainingPair = tuple[Grid, Grid]


@dataclass(frozen=True)
class HeuristicResult:
    name: str
    transform: Transform
    confidence: float
    rationale: str


class Heuristic:
    name: str = "heuristic"

    def infer(self, pairs: Sequence[TrainingPair]) -> HeuristicResult | None:
        raise NotImplementedError

    def _success(self, transform: Transform, confidence: float, rationale: str) -> HeuristicResult:
        return HeuristicResult(self.name, transform, confidence, rationale)


class IdentityHeuristic(Heuristic):
    name = "identity"

    def infer(self, pairs: Sequence[TrainingPair]) -> HeuristicResult | None:
        for src, dst in pairs:
            if not src.equals(dst):
                return None
        def transform(grid: Grid) -> Grid:
            return grid
        return self._success(transform, 0.2, "All examples preserve the grid exactly.")


class ConstantFillHeuristic(Heuristic):
    name = "constant-fill"

    def infer(self, pairs: Sequence[TrainingPair]) -> HeuristicResult | None:
        colour = None
        for _, dst in pairs:
            colours = dst.colors()
            if len(colours) != 1:
                return None
            candidate = next(iter(colours))
            if colour is None:
                colour = candidate
            elif colour != candidate:
                return None
        if colour is None:
            return None
        def transform(grid: Grid) -> Grid:
            return Grid(tuple(tuple(colour for _ in range(grid.width)) for _ in range(grid.height)))
        return self._success(transform, 0.1, f"Outputs collapse to a single colour {colour}.")


class ColourMappingHeuristic(Heuristic):
    name = "colour-mapping"

    def infer(self, pairs: Sequence[TrainingPair]) -> HeuristicResult | None:
        mapping: dict[int, int] = {}
        for src, dst in pairs:
            if src.width != dst.width or src.height != dst.height:
                return None
            for (_, value_src), (_, value_dst) in zip(src.cells(), dst.cells()):
                existing = mapping.get(value_src)
                if existing is None:
                    mapping[value_src] = value_dst
                elif existing != value_dst:
                    return None
        if not mapping:
            return None
        def transform(grid: Grid) -> Grid:
            return grid.map_colors(mapping)
        return self._success(transform, 0.4, f"Pixel-wise colour remapping {mapping} fits all examples.")


class BackgroundColourHeuristic(Heuristic):
    name = "background-colour"

    def infer(self, pairs: Sequence[TrainingPair]) -> HeuristicResult | None:
        bg = None
        fg = None
        for src, dst in pairs:
            if src.width != dst.width or src.height != dst.height:
                return None
            if len(dst.colors()) != 2:
                return None
            src_bg = src.most_common_color()
            dst_bg = dst.most_common_color()
            if bg is None:
                bg = (src_bg, dst_bg)
            elif bg != (src_bg, dst_bg):
                return None
            for (_, value_src), (_, value_dst) in zip(src.cells(), dst.cells()):
                if value_src != src_bg:
                    if fg is None:
                        fg = value_dst
                    elif fg != value_dst:
                        return None
        if bg is None or fg is None:
            return None
        src_bg, dst_bg = bg
        def transform(grid: Grid) -> Grid:
            mapping = {src_bg: dst_bg}
            return grid.map_colors(mapping, default=fg)
        return self._success(transform, 0.35, f"Map background {src_bg}->{dst_bg} while foreground becomes {fg}.")


class TranslationHeuristic(Heuristic):
    name = "object-translation"

    def infer(self, pairs: Sequence[TrainingPair]) -> HeuristicResult | None:
        vector = None
        background = None
        for src, dst in pairs:
            background_src = src.most_common_color()
            background_dst = dst.most_common_color()
            if background is None:
                background = (background_src, background_dst)
            elif background != (background_src, background_dst):
                return None
            src_objs = extract_objects(src, background_src)
            dst_objs = extract_objects(dst, background_dst)
            if len(src_objs) != 1 or len(dst_objs) != 1:
                return None
            match = difference_objects(src_objs, dst_objs)
            if not match:
                return None
            _, _, (dx, dy) = match[0]
            if vector is None:
                vector = (dx, dy)
            elif vector != (dx, dy):
                return None
        if vector is None or background is None:
            return None
        src_bg, dst_bg = background
        def transform(grid: Grid) -> Grid:
            objs = extract_objects(grid, src_bg)
            if len(objs) != 1:
                return grid
            obj = objs[0]
            canvas = [[dst_bg for _ in range(grid.width)] for _ in range(grid.height)]
            for (x, y), value in grid.cells():
                if (x, y) not in obj.pixels:
                    canvas[y][x] = value
            for x, y in obj.pixels:
                nx, ny = x + vector[0], y + vector[1]
                if 0 <= nx < grid.width and 0 <= ny < grid.height:
                    canvas[ny][nx] = grid.get(x, y)
            return Grid.from_list(canvas)
        return self._success(transform, 0.5, f"Single foreground object translates by {vector}.")


class ScaleReplicationHeuristic(Heuristic):
    name = "scale-replication"

    def infer(self, pairs: Sequence[TrainingPair]) -> HeuristicResult | None:
        scale_x = scale_y = None
        for src, dst in pairs:
            if dst.width % src.width != 0 or dst.height % src.height != 0:
                return None
            candidate_x = dst.width // src.width
            candidate_y = dst.height // src.height
            expanded = src.resize(candidate_x, candidate_y)
            if not expanded.equals(dst):
                return None
            if scale_x is None:
                scale_x = candidate_x
                scale_y = candidate_y
            elif (scale_x, scale_y) != (candidate_x, candidate_y):
                return None
        if scale_x is None or scale_y is None:
            return None
        def transform(grid: Grid) -> Grid:
            return grid.resize(scale_x, scale_y)
        return self._success(transform, 0.3, f"Grids upscale by ({scale_x}, {scale_y}).")


DEFAULT_HEURISTICS: tuple[Heuristic, ...] = (
    IdentityHeuristic(),
    ConstantFillHeuristic(),
    ColourMappingHeuristic(),
    BackgroundColourHeuristic(),
    TranslationHeuristic(),
    ScaleReplicationHeuristic(),
)
