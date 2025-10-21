# ARC-AGI 2 Solver Agent

This document describes the agentic system added to GlyphNotes for solving Abstraction and Reasoning Corpus (ARC-AGI 2) tasks.

## Overview

The solver follows a tool-based agent architecture. Each tool encapsulates a heuristic transformation that can be inferred from the training examples of an ARC puzzle. The orchestrator evaluates each tool, keeps the ones that explain the training pairs, and produces ranked solution candidates for the test grids.

### Components

- **`Grid`** – Immutable utility class that wraps ARC grids and offers transformations such as colour mapping, resizing, mirroring, and bounding box extraction.
- **`objects` module** – Connected component analysis used to reason about discrete objects in a grid, including centroid calculations and translations.
- **`heuristics` module** – Library of inference tools (identity, constant fill, colour remapping, background swapping, translation, and scale replication). Each heuristic decides whether it can explain the training examples and provides a transform for new grids.
- **`agent` module** – Orchestrates heuristics, turning successful inferences into ranked `SolutionCandidate` entries.
- **`cli`** – Command line utility to run the solver on JSON task files with optional JSON output.

### Usage

```bash
python -m arcagi2.cli path/to/task.json --top 3
```

Use `--json` to emit machine-readable output suitable for pipelines.

### Extending

To add new heuristics, implement the `Heuristic` interface, provide the inference logic, and inject the heuristic into `DEFAULT_HEURISTICS` or pass a custom list when instantiating `ArcSolverAgent`.

### Testing

`pytest` exercises the main heuristics with curated micro tasks.
