"""Explorer — depends only on the Strategy + Evaluator Protocols.

For one Target:

  while not budget.exhausted:
      candidate, agent_usage = strategy.propose(target, progress, history)
      evaluation             = evaluator.evaluate(candidate, target)
      new_goals              = progress.update(evaluation.goals_hit)
      record Step; append to history; consume one step

Termination:
  - budget exhausted (default), OR
  - all goals hit (early stop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from xsmith.agents.base import AgentUsage
from xsmith.domain.budget import Budget
from xsmith.domain.candidate import Candidate
from xsmith.domain.evaluation import Evaluation
from xsmith.domain.goal import Goals
from xsmith.domain.progress import Progress
from xsmith.domain.target import Target
from xsmith.execution.evaluator import Evaluator
from xsmith.strategies.base import Strategy


@dataclass
class Step:
    iteration: int
    candidate: Candidate
    evaluation: Evaluation
    new_goals: Goals
    hit_after: int
    total: int
    agent_usage: AgentUsage


@dataclass
class ExplorationResult:
    target: Target
    steps: list[Step] = field(default_factory=list)
    final_progress: Progress | None = None

    @property
    def hit_count(self) -> int:
        return len(self.final_progress.hit) if self.final_progress else 0

    @property
    def total_count(self) -> int:
        return len(self.final_progress.all) if self.final_progress else 0

    @property
    def fraction(self) -> float:
        return self.final_progress.fraction if self.final_progress else 0.0


class Explorer:
    """Drive a single target's exploration with injected strategy + evaluator."""

    def __init__(
        self,
        *,
        strategy: Strategy,
        evaluator: Evaluator,
        on_step: Callable[[Step], Awaitable[None] | None] | None = None,
    ):
        self.strategy = strategy
        self.evaluator = evaluator
        self.on_step = on_step

    async def run(
        self,
        *,
        target: Target,
        budget: Budget,
        initial_progress: Progress | None = None,
    ) -> ExplorationResult:
        progress = initial_progress or Progress(all=target.goals)
        if progress.all == Goals() and target.goals != Goals():
            progress = Progress(all=target.goals, hit=progress.hit)

        history: list[Evaluation] = []
        result = ExplorationResult(target=target)
        i = 0
        while not budget.exhausted:
            if len(progress.all) > 0 and len(progress.missing) == 0:
                break
            i += 1
            candidate, agent_usage = await self.strategy.propose(
                target=target, progress=progress, history=history
            )
            budget.record_usage(
                tokens_in=agent_usage.tokens_in,
                tokens_out=agent_usage.tokens_out,
                tokens_cache_read=agent_usage.tokens_cache_read,
                tokens_cache_creation=agent_usage.tokens_cache_creation,
                usd=agent_usage.cost_usd,
            )

            if not candidate.code:
                evaluation = Evaluation(
                    candidate=candidate,
                    outcome="error",
                    stderr="empty candidate (strategy failed)",
                )
                new_goals = Goals()
            else:
                evaluation = await self.evaluator.evaluate(candidate, target)
                new_goals = progress.update(evaluation.goals_hit)

            history.append(evaluation)

            step = Step(
                iteration=i,
                candidate=candidate,
                evaluation=evaluation,
                new_goals=new_goals,
                hit_after=len(progress.hit),
                total=len(progress.all),
                agent_usage=agent_usage,
            )
            result.steps.append(step)
            if self.on_step is not None:
                maybe = self.on_step(step)
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[func-returns-value]

            budget.consume_step()

        result.final_progress = progress
        return result


async def explore(
    *,
    target: Target,
    strategy: Strategy,
    evaluator: Evaluator,
    budget: Budget,
    on_step: Callable[[Step], Awaitable[None] | None] | None = None,
    initial_progress: Progress | None = None,
) -> ExplorationResult:
    """Sugar for `Explorer(strategy=..., evaluator=..., on_step=...).run(...)`.

    Single-target by design; for multi-target orchestration use the CLI or
    call `explore()` in a loop.
    """
    return await Explorer(
        strategy=strategy, evaluator=evaluator, on_step=on_step
    ).run(target=target, budget=budget, initial_progress=initial_progress)
