"""GenerationStrategy Protocol — the only contract ExplorationLoop knows."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from xsmith.agents.base import AgentUsage
from xsmith.domain.coverage import CoverageMap
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase, TestResult


@runtime_checkable
class GenerationStrategy(Protocol):
    """Propose the next TestCase for a target given current state.

    Implementations may consult `coverage` (what's already covered) and
    `history` (what has already been tried, with outcomes). They return one
    `TestCase` per call, plus telemetry about LLM usage during proposal.
    """

    async def propose(
        self,
        *,
        target: Target,
        coverage: CoverageMap,
        history: list[TestResult],
    ) -> tuple[TestCase, AgentUsage]: ...
