from __future__ import annotations

from xsmith.agents.base import AgentUsage
from xsmith.domain.budget import Budget
from xsmith.domain.candidate import Candidate
from xsmith.domain.evaluation import Evaluation
from xsmith.domain.goal import Goal, Goals
from xsmith.domain.progress import Progress
from xsmith.domain.target import Target
from xsmith.exploration.explorer import Explorer


def _goal(src: int, dst: int) -> Goal:
    return Goal(file="target_pkg/sample_mod.py", src=src, dst=dst)


class ScriptedStrategy:
    def __init__(self, codes: list[str]):
        self.codes = codes
        self.calls: list[tuple[int, int]] = []

    async def propose(self, *, target, progress, history):
        self.calls.append((len(history), len(progress.hit)))
        idx = len(self.calls) - 1
        return (
            Candidate(code=self.codes[idx], rationale=f"case {idx}"),
            AgentUsage(tokens_in=idx + 1, tokens_out=idx + 2, cost_usd=0.01),
        )


class ScriptedEvaluator:
    def __init__(self, goal_sets: list[Goals]):
        self.goal_sets = goal_sets
        self.calls = 0

    async def evaluate(self, candidate, target):
        goals_hit = self.goal_sets[self.calls]
        self.calls += 1
        return Evaluation(
            candidate=candidate,
            outcome="pass",
            goals_hit=goals_hit,
            duration_s=0.01,
        )

    async def enumerate_goals(self, target):
        return Goals()


async def test_explorer_records_steps_and_stops_when_all_hit():
    goals = Goals.from_iterable([_goal(1, 2), _goal(1, 3)])
    target = Target(
        target_id="t",
        module_path="sample_mod",
        source="def f(x): return x",
        goals=goals,
    )
    strategy = ScriptedStrategy(["def test_one(): pass", "def test_two(): pass"])
    evaluator = ScriptedEvaluator(
        [
            Goals.from_iterable([_goal(1, 2)]),
            Goals.from_iterable([_goal(1, 2), _goal(1, 3)]),
        ]
    )
    records = []
    explorer = Explorer(
        strategy=strategy,
        evaluator=evaluator,
        on_step=lambda step: records.append(step),
    )
    budget = Budget(steps=5)

    result = await explorer.run(
        target=target,
        budget=budget,
        initial_progress=Progress(all=goals),
    )

    assert evaluator.calls == 2
    assert len(result.steps) == 2
    assert len(records) == 2
    assert result.hit_count == 2
    assert result.total_count == 2
    assert result.fraction == 1.0
    assert budget.steps == 3
    assert budget.tokens_in_used == 3
    assert budget.tokens_out_used == 5
    assert budget.usd_used == 0.02
    assert strategy.calls == [(0, 0), (1, 1)]


async def test_explorer_records_empty_strategy_result_without_evaluating():
    target = Target(
        target_id="t",
        module_path="sample_mod",
        source="def f(x): return x",
        goals=Goals.from_iterable([_goal(1, 2)]),
    )
    strategy = ScriptedStrategy([""])
    evaluator = ScriptedEvaluator([])
    budget = Budget(steps=1)

    result = await Explorer(strategy=strategy, evaluator=evaluator).run(
        target=target,
        budget=budget,
        initial_progress=Progress(),
    )

    assert evaluator.calls == 0
    assert len(result.steps) == 1
    assert result.steps[0].evaluation.outcome == "error"
    assert "empty candidate" in result.steps[0].evaluation.stderr
    assert result.total_count == 1
    assert budget.steps == 0
