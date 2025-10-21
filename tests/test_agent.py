from arcagi2.agent import ArcSolverAgent, Task
from arcagi2.grid import Grid


def build_task(training, tests):
    return Task(
        training=tuple((Grid.from_list(src), Grid.from_list(dst)) for src, dst in training),
        tests=tuple(Grid.from_list(test) for test in tests),
    )


def test_identity_heuristic():
    task = build_task(
        training=[(
            [[1, 1], [1, 0]],
            [[1, 1], [1, 0]],
        )],
        tests=[[[1, 0], [0, 1]]],
    )
    agent = ArcSolverAgent()
    candidates = agent.analyse(task)
    identity = next(c for c in candidates if c.heuristic == "identity")
    assert identity.outputs[0].equals(Grid.from_list([[1, 0], [0, 1]]))


def test_constant_fill_heuristic():
    task = build_task(
        training=[(
            [[1]],
            [[3]],
        )],
        tests=[[[0, 0], [0, 0]]],
    )
    agent = ArcSolverAgent()
    candidates = agent.analyse(task)
    fill_candidate = next(c for c in candidates if c.heuristic == "constant-fill")
    assert fill_candidate.outputs[0].equals(Grid.from_list([[3, 3], [3, 3]]))


def test_colour_mapping_heuristic():
    task = build_task(
        training=[(
            [[1, 2]],
            [[2, 3]],
        )],
        tests=[[[1, 2]]],
    )
    agent = ArcSolverAgent()
    candidates = agent.analyse(task)
    mapping_candidate = next(c for c in candidates if c.heuristic == "colour-mapping")
    assert mapping_candidate.outputs[0].equals(Grid.from_list([[2, 3]]))


def test_translation_heuristic():
    task = build_task(
        training=[(
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
        )],
        tests=[[[0, 0, 0], [2, 0, 0], [0, 0, 0]]],
    )
    agent = ArcSolverAgent()
    candidates = agent.analyse(task)
    translation_candidate = next(c for c in candidates if c.heuristic == "object-translation")
    assert translation_candidate.outputs[0].equals(Grid.from_list([[0, 0, 0], [0, 0, 0], [0, 2, 0]]))


def test_scale_replication_heuristic():
    task = build_task(
        training=[(
            [[1, 2], [3, 4]],
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
        )],
        tests=[[[1, 2], [3, 4]]],
    )
    agent = ArcSolverAgent()
    candidates = agent.analyse(task)
    scale_candidate = next(c for c in candidates if c.heuristic == "scale-replication")
    assert scale_candidate.outputs[0].equals(
        Grid.from_list([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    )
