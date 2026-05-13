from xsmith.domain.goal import Goal, Goals
from xsmith.domain.progress import Progress


def _g(s, d, f="a.py"):
    return Goal(file=f, src=s, dst=d)


def test_goals_set_ops():
    a = Goals.from_iterable([_g(1, 2), _g(1, 3)])
    b = Goals.from_iterable([_g(1, 3), _g(1, 4)])
    assert len(a | b) == 3
    assert len(a & b) == 1
    assert len(a - b) == 1
    assert _g(1, 2) in (a - b)


def test_progress_delta_and_update():
    p = Progress(all=Goals.from_iterable([_g(1, 2), _g(1, 3), _g(2, 3)]))
    assert p.fraction == 0.0

    new = Goals.from_iterable([_g(1, 2), _g(1, 3)])
    delta = p.delta(new)
    assert len(delta) == 2

    added = p.update(new)
    assert len(added) == 2
    assert p.fraction == 2 / 3
    assert len(p.missing) == 1

    # Re-applying the same goals yields zero delta.
    added2 = p.update(new)
    assert len(added2) == 0


def test_empty_all_fraction_is_zero():
    p = Progress()
    assert p.fraction == 0.0
