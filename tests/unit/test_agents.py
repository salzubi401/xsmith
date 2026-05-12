from __future__ import annotations

import pytest

from xsmith.agents.base import AgentUsage
from xsmith.agents.scorer import QValueScorerAgent
from xsmith.agents.test_generator import TestGeneratorAgent as XTestGeneratorAgent
from xsmith.domain.coverage import Branch, BranchSet
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase as XTestCase


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


def _uncovered() -> BranchSet:
    return BranchSet.from_iterable([Branch(file="sample.py", src=1, dst=2)])


@pytest.mark.asyncio
async def test_generator_agent_returns_test_case_from_submit_payload():
    runner = FakeRunner({"code": "def test_x():\n    assert True\n", "rationale": "smoke"})
    agent = XTestGeneratorAgent(variant_idx=0, model="model-name", runner=runner)

    test_case, usage = await agent.propose(target=_target(), uncovered=_uncovered(), history=[])

    assert test_case == XTestCase(code="def test_x():\n    assert True\n", rationale="smoke")
    assert usage.tokens_in == 11
    options, prompt = runner.calls[0]
    assert options.allowed_tools == [
        "mcp__xsmith__view_coverage",
        "mcp__xsmith__view_history",
        "mcp__xsmith__submit_test",
    ]
    assert options.setting_sources == []
    assert "module_path: pkg.sample" in prompt
    assert "sample.py:1->2" in prompt


@pytest.mark.asyncio
async def test_generator_agent_rejects_missing_or_empty_code():
    for submission in (None, {"code": "", "rationale": "empty"}):
        agent = XTestGeneratorAgent(
            variant_idx=0,
            model="model-name",
            runner=FakeRunner(submission),
        )
        test_case, _ = await agent.propose(
            target=_target(),
            uncovered=_uncovered(),
            history=[],
        )
        assert test_case is None


@pytest.mark.asyncio
async def test_scorer_agent_returns_score_with_custom_gamma():
    runner = FakeRunner({"immediate_branches": 2, "future_value": 6})
    agent = QValueScorerAgent(model="model-name", runner=runner, gamma=0.25)

    score, usage = await agent.score(
        target=_target(),
        uncovered=_uncovered(),
        candidate=XTestCase(code="def test_x(): pass", rationale="try it"),
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
        {"immediate_branches": 1},
        {"immediate_branches": -1, "future_value": 0},
        {"immediate_branches": 0, "future_value": 11},
    ]

    for payload in invalid_payloads:
        agent = QValueScorerAgent(
            model="model-name",
            runner=FakeRunner(payload),
        )
        score, _ = await agent.score(
            target=_target(),
            uncovered=_uncovered(),
            candidate=XTestCase(code="def test_x(): pass"),
        )
        assert score is None
