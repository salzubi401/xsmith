from __future__ import annotations

import pytest

from xsmith.agents.base import AgentUsage
from xsmith.agents.generator import GeneratorAgent
from xsmith.agents.scorer import ScorerAgent
from xsmith.domain.candidate import Candidate
from xsmith.domain.goal import Goal, Goals
from xsmith.domain.target import Target


class FakeRunner:
    def __init__(self, submission):
        self.submission = submission
        self.calls = []

    async def run(self, *, options, user_prompt):
        self.calls.append((options, user_prompt))
        return self.submission, AgentUsage(tokens_in=11, tokens_out=7)


def _target() -> Target:
    return Target(
        target_id="t",
        module_path="pkg.sample",
        source="def f(x):\n    return x\n",
    )


def _missing() -> Goals:
    return Goals.from_iterable([Goal(file="sample.py", src=1, dst=2)])


@pytest.mark.asyncio
async def test_generator_agent_returns_candidate_from_submit_payload():
    runner = FakeRunner({"code": "def test_x():\n    assert True\n", "rationale": "smoke"})
    agent = GeneratorAgent(variant_idx=0, model="model-name", runner=runner)

    candidate, usage = await agent.propose(target=_target(), missing=_missing(), history=[])

    assert candidate == Candidate(code="def test_x():\n    assert True\n", rationale="smoke")
    assert usage.tokens_in == 11
    options, prompt = runner.calls[0]
    assert options.allowed_tools == [
        "mcp__xsmith__view_progress",
        "mcp__xsmith__view_history",
        "mcp__xsmith__submit_candidate",
    ]
    assert options.setting_sources == []
    assert "module_path: pkg.sample" in prompt
    assert "sample.py:1->2" in prompt


@pytest.mark.asyncio
async def test_generator_agent_rejects_missing_or_empty_code():
    for submission in (None, {"code": "", "rationale": "empty"}):
        agent = GeneratorAgent(
            variant_idx=0,
            model="model-name",
            runner=FakeRunner(submission),
        )
        candidate, _ = await agent.propose(
            target=_target(),
            missing=_missing(),
            history=[],
        )
        assert candidate is None


@pytest.mark.asyncio
async def test_scorer_agent_returns_score_with_custom_gamma():
    runner = FakeRunner({"immediate_goals": 2, "future_value": 6})
    agent = ScorerAgent(model="model-name", runner=runner, gamma=0.25)

    score, usage = await agent.score(
        target=_target(),
        missing=_missing(),
        candidate=Candidate(code="def test_x(): pass", rationale="try it"),
    )

    assert score is not None
    assert score.q == 3.5
    assert usage.tokens_out == 7
    options, prompt = runner.calls[0]
    assert options.allowed_tools == ["mcp__xsmith__submit_score"]
    assert options.setting_sources == []
    assert "future_value" in prompt
    assert "def test_x(): pass" in prompt


@pytest.mark.asyncio
async def test_scorer_agent_rejects_invalid_submit_payload():
    invalid_payloads = [
        None,
        {"immediate_goals": 1},
        {"immediate_goals": -1, "future_value": 0},
        {"immediate_goals": 0, "future_value": 11},
    ]

    for payload in invalid_payloads:
        agent = ScorerAgent(
            model="model-name",
            runner=FakeRunner(payload),
        )
        score, _ = await agent.score(
            target=_target(),
            missing=_missing(),
            candidate=Candidate(code="def test_x(): pass"),
        )
        assert score is None
