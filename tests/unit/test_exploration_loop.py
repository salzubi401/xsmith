from __future__ import annotations

from xsmith.agents.base import AgentUsage
from xsmith.domain.budget import Budget
from xsmith.domain.coverage import Branch, BranchSet, CoverageMap
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase as XTestCase
from xsmith.execution.runner import TestRunResult as XTestRunResult
from xsmith.exploration.explorer import ExplorationLoop


def _branch(src: int, dst: int) -> Branch:
    return Branch(file="target_pkg/sample_mod.py", src=src, dst=dst)


class ScriptedStrategy:
    def __init__(self, codes: list[str]):
        self.codes = codes
        self.calls: list[tuple[int, int]] = []

    async def propose(self, *, target, coverage, history):
        self.calls.append((len(history), len(coverage.covered)))
        idx = len(self.calls) - 1
        return (
            XTestCase(code=self.codes[idx], rationale=f"case {idx}"),
            AgentUsage(tokens_in=idx + 1, tokens_out=idx + 2, cost_usd=0.01),
        )


class ScriptedRunner:
    def __init__(self, branch_sets: list[BranchSet]):
        self.branch_sets = branch_sets
        self.calls = 0

    async def run(self, test_case, target):
        covered = self.branch_sets[self.calls]
        self.calls += 1
        return XTestRunResult(outcome="pass", branches_covered=covered, duration_s=0.01)


async def test_exploration_loop_records_iterations_and_stops_when_covered():
    branches = BranchSet.from_iterable([_branch(1, 2), _branch(1, 3)])
    target = Target(
        target_id="t",
        module_path="sample_mod",
        source="def f(x): return x",
        branches=branches,
    )
    strategy = ScriptedStrategy(["def test_one(): pass", "def test_two(): pass"])
    runner = ScriptedRunner(
        [
            BranchSet.from_iterable([_branch(1, 2)]),
            BranchSet.from_iterable([_branch(1, 2), _branch(1, 3)]),
        ]
    )
    records = []
    loop = ExplorationLoop(
        strategy=strategy,
        runner=runner,
        on_iteration=lambda rec: records.append(rec),
    )
    budget = Budget(exec_remaining=5)

    result = await loop.explore(
        target=target,
        budget=budget,
        initial_coverage=CoverageMap(total=branches),
    )

    assert runner.calls == 2
    assert len(result.iterations) == 2
    assert len(records) == 2
    assert result.covered_count == 2
    assert result.total_count == 2
    assert result.coverage_fraction == 1.0
    assert budget.exec_remaining == 3
    assert budget.tokens_in_used == 3
    assert budget.tokens_out_used == 5
    assert budget.usd_used == 0.02
    assert strategy.calls == [(0, 0), (1, 1)]


async def test_exploration_loop_records_empty_strategy_result_without_running():
    target = Target(
        target_id="t",
        module_path="sample_mod",
        source="def f(x): return x",
        branches=BranchSet.from_iterable([_branch(1, 2)]),
    )
    strategy = ScriptedStrategy([""])
    runner = ScriptedRunner([])
    budget = Budget(exec_remaining=1)

    result = await ExplorationLoop(strategy=strategy, runner=runner).explore(
        target=target,
        budget=budget,
        initial_coverage=CoverageMap(),
    )

    assert runner.calls == 0
    assert len(result.iterations) == 1
    assert result.iterations[0].run_result.outcome == "error"
    assert "empty test case" in result.iterations[0].run_result.stderr
    assert result.total_count == 1
    assert budget.exec_remaining == 0
