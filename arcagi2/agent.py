"""Agentic orchestration for solving ARC-AGI 2 tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from .grid import Grid
from .heuristics import DEFAULT_HEURISTICS, Heuristic, HeuristicResult, TrainingPair


@dataclass(frozen=True)
class SolutionCandidate:
    heuristic: str
    outputs: tuple[Grid, ...]
    confidence: float
    rationale: str

    def serialise(self) -> dict:
        return {
            "heuristic": self.heuristic,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "outputs": [grid.to_list() for grid in self.outputs],
        }


@dataclass(frozen=True)
class Task:
    training: tuple[TrainingPair, ...]
    tests: tuple[Grid, ...]

    @staticmethod
    def from_dict(payload: dict) -> "Task":
        training_pairs = []
        for entry in payload.get("training", []):
            training_pairs.append(
                (
                    Grid.from_list(entry["input"]),
                    Grid.from_list(entry["output"]),
                )
            )
        tests = [Grid.from_list(entry["input"]) for entry in payload.get("test", [])]
        return Task(tuple(training_pairs), tuple(tests))


class ArcSolverAgent:
    """Multi-tool agent that evaluates heuristics sequentially."""

    def __init__(self, heuristics: Iterable[Heuristic] | None = None) -> None:
        self.heuristics = tuple(heuristics or DEFAULT_HEURISTICS)

    def analyse(self, task: Task) -> list[SolutionCandidate]:
        candidates: list[SolutionCandidate] = []
        for heuristic in self.heuristics:
            result = self._apply_heuristic(heuristic, task.training, task.tests)
            if result:
                candidates.append(result)
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def solve(self, payload: dict | Task) -> dict:
        task = payload if isinstance(payload, Task) else Task.from_dict(payload)
        candidates = self.analyse(task)
        return {
            "candidates": [candidate.serialise() for candidate in candidates],
        }

    def _apply_heuristic(
        self,
        heuristic: Heuristic,
        training: Sequence[TrainingPair],
        tests: Sequence[Grid],
    ) -> SolutionCandidate | None:
        result: HeuristicResult | None = heuristic.infer(training)
        if result is None:
            return None
        outputs = tuple(result.transform(grid) for grid in tests)
        return SolutionCandidate(
            heuristic=result.name,
            outputs=outputs,
            confidence=result.confidence,
            rationale=result.rationale,
        )

    @staticmethod
    def load_task(path: str) -> Task:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return Task.from_dict(payload)


__all__ = ["ArcSolverAgent", "SolutionCandidate", "Task"]
