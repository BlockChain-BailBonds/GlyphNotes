"""Command line interface for the ARC-AGI 2 solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agent import ArcSolverAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC-AGI 2 heuristic agent")
    parser.add_argument("task", type=Path, help="Path to an ARC style JSON task")
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        help="How many solution candidates to print",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine readable JSON",
    )
    return parser


def format_candidate(index: int, candidate: dict[str, Any]) -> str:
    header = f"[{index}] heuristic={candidate['heuristic']} confidence={candidate['confidence']:.2f}"
    lines = [header, f"    rationale: {candidate['rationale']}"]
    for grid_index, grid in enumerate(candidate["outputs"], start=1):
        rows = ["".join(str(pixel) for pixel in row) for row in grid]
        lines.append(f"    output[{grid_index}]\n        " + "\n        ".join(rows))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    agent = ArcSolverAgent()
    payload = json.loads(args.task.read_text())
    result = agent.solve(payload)
    candidates = result.get("candidates", [])
    top = max(0, args.top)
    if args.json:
        if top:
            result["candidates"] = candidates[:top]
        print(json.dumps(result, indent=2))
        return 0
    if not candidates:
        print("No heuristics produced a candidate solution.")
        return 1
    for index, candidate in enumerate(candidates[:top], start=1):
        print(format_candidate(index, candidate))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
