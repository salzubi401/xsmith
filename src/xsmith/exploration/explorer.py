"""ExplorationLoop — depends only on the Strategy + Runner Protocols.

For one Target:

  while not budget.exhausted:
      test_case, agent_usage = strategy.propose(target, coverage, history)
      run_result               = runner.run(test_case, target)
      delta                    = coverage.update(run_result.branches_covered)
      record iteration; append to history; consume one execution

Termination:
  - budget exhausted (default), OR
  - all branches covered (early stop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from xsmith.agents.base import AgentUsage
from xsmith.domain.budget import Budget
from xsmith.domain.coverage import BranchSet, CoverageMap
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase, TestResult
from xsmith.execution.runner import TestRunner, TestRunResult
from xsmith.strategies.base import GenerationStrategy


@dataclass
class IterationRecord:
    iteration: int
    test_case: TestCase
    run_result: TestRunResult
    new_branches: BranchSet
    coverage_after: int
    coverage_total: int
    agent_usage: AgentUsage


@dataclass
class TargetExplorationResult:
    target: Target
    iterations: list[IterationRecord] = field(default_factory=list)
    final_coverage: CoverageMap | None = None

    @property
    def covered_count(self) -> int:
        return len(self.final_coverage.covered) if self.final_coverage else 0

    @property
    def total_count(self) -> int:
        return len(self.final_coverage.total) if self.final_coverage else 0

    @property
    def coverage_fraction(self) -> float:
        return self.final_coverage.fraction if self.final_coverage else 0.0


class ExplorationLoop:
    """Drive a single target's exploration with injected strategy + runner."""

    def __init__(
        self,
        *,
        strategy: GenerationStrategy,
        runner: TestRunner,
        on_iteration: Callable[[IterationRecord], Awaitable[None] | None] | None = None,
    ):
        self.strategy = strategy
        self.runner = runner
        self.on_iteration = on_iteration

    async def explore(
        self,
        *,
        target: Target,
        budget: Budget,
        initial_coverage: CoverageMap | None = None,
    ) -> TargetExplorationResult:
        coverage = initial_coverage or CoverageMap(total=target.branches)
        if coverage.total == BranchSet() and target.branches != BranchSet():
            coverage = CoverageMap(total=target.branches, covered=coverage.covered)

        history: list[TestResult] = []
        result = TargetExplorationResult(target=target)
        i = 0
        while not budget.exhausted:
            if len(coverage.total) > 0 and len(coverage.uncovered) == 0:
                break
            i += 1
            test_case, agent_usage = await self.strategy.propose(
                target=target, coverage=coverage, history=history
            )
            budget.record_usage(
                tokens_in=agent_usage.tokens_in,
                tokens_out=agent_usage.tokens_out,
                tokens_cache_read=agent_usage.tokens_cache_read,
                tokens_cache_creation=agent_usage.tokens_cache_creation,
                usd=agent_usage.cost_usd,
            )

            if not test_case.code:
                run_result = TestRunResult(
                    outcome="error",
                    stderr="empty test case (strategy failed)",
                )
                new = BranchSet()
            else:
                run_result = await self.runner.run(test_case, target)
                new = coverage.update(run_result.branches_covered)

            test_result = TestResult(
                test_case=test_case,
                outcome=run_result.outcome,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                duration_s=run_result.duration_s,
                branches_covered=run_result.branches_covered,
                new_branches_covered=new,
            )
            history.append(test_result)

            rec = IterationRecord(
                iteration=i,
                test_case=test_case,
                run_result=run_result,
                new_branches=new,
                coverage_after=len(coverage.covered),
                coverage_total=len(coverage.total),
                agent_usage=agent_usage,
            )
            result.iterations.append(rec)
            if self.on_iteration is not None:
                maybe = self.on_iteration(rec)
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[func-returns-value]

            budget.consume_execution()

        result.final_coverage = coverage
        return result
