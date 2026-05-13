"""Strategy Protocol — the only contract the Explorer knows."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from xsmith.agents.base import AgentUsage
from xsmith.domain.candidate import Candidate
from xsmith.domain.evaluation import Evaluation
from xsmith.domain.progress import Progress
from xsmith.domain.target import Target


@runtime_checkable
class Strategy(Protocol):
    """Propose the next Candidate for a target given current state.

    Implementations may consult `progress` (which goals are already hit) and
    `history` (what has already been tried, with outcomes). They return one
    `Candidate` per call, plus telemetry about LLM usage during proposal.
    """

    async def propose(
        self,
        *,
        target: Target,
        progress: Progress,
        history: list[Evaluation],
    ) -> tuple[Candidate, AgentUsage]: ...
