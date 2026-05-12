from xsmith.domain.coverage import Branch, BranchSet, CoverageMap


def _b(s, d, f="a.py"):
    return Branch(file=f, src=s, dst=d)


def test_branchset_set_ops():
    a = BranchSet.from_iterable([_b(1, 2), _b(1, 3)])
    b = BranchSet.from_iterable([_b(1, 3), _b(1, 4)])
    assert len(a | b) == 3
    assert len(a & b) == 1
    assert len(a - b) == 1
    assert _b(1, 2) in (a - b)


def test_coverage_map_delta_and_update():
    cm = CoverageMap(total=BranchSet.from_iterable([_b(1, 2), _b(1, 3), _b(2, 3)]))
    assert cm.fraction == 0.0

    new = BranchSet.from_iterable([_b(1, 2), _b(1, 3)])
    delta = cm.delta(new)
    assert len(delta) == 2

    added = cm.update(new)
    assert len(added) == 2
    assert cm.fraction == 2 / 3
    assert len(cm.uncovered) == 1

    # Re-applying the same branches yields zero delta.
    added2 = cm.update(new)
    assert len(added2) == 0


def test_empty_total_fraction_is_zero():
    cm = CoverageMap()
    assert cm.fraction == 0.0
