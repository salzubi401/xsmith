"""CovQValueStrategy — curiosity-driven test selection.

Algorithm per `propose()`:

  1. Spawn K TestGeneratorAgent calls IN PARALLEL (asyncio.gather), each
     with a different diversity variant.
  2. Drop any candidates the generator failed to submit.
  3. Score every successful candidate with QValueScorerAgent (also parallel).
  4. Q = immediate_branches + γ · future_value
  5. Return argmax-Q candidate. Aggregate AgentUsage across all calls.

Ties on Q are broken by:
  - higher immediate_branches first
  - then shorter code (a tiebreaker that favors readability)

If all K generators fail, we return a sentinel TestCase with empty code so
the caller can detect failure. (TestResult outcome will then be "error".)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from xsmith.agents.base import AgentUsage
from xsmith.agents.scorer import QScore, QValueScorerAgent
from xsmith.agents.test_generator import TestGeneratorAgent
from xsmith.domain.coverage import CoverageMap
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase, TestResult


def _sum_usages(usages: list[AgentUsage]) -> AgentUsage:
    return AgentUsage(
        tokens_in=sum(u.tokens_in for u in usages),
        tokens_out=sum(u.tokens_out for u in usages),
        tokens_cache_read=sum(u.tokens_cache_read for u in usages),
        tokens_cache_creation=sum(u.tokens_cache_creation for u in usages),
        cost_usd=sum(u.cost_usd for u in usages),
        duration_ms=max((u.duration_ms for u in usages), default=0),
        num_turns=sum(u.num_turns for u in usages),
    )


@dataclass
class _Scored:
    candidate: TestCase
    score: QScore


class CovQValueStrategy:
    def __init__(
        self,
        *,
        model: str,
        k: int = 5,
        gamma: float = 0.5,
        max_turns_gen: int = 8,
        max_turns_score: int = 3,
    ):
        self.model = model
        self.k = k
        self.gamma = gamma
        self.max_turns_gen = max_turns_gen
        self.max_turns_score = max_turns_score

    def _make_generators(self) -> list[TestGeneratorAgent]:
        return [
            TestGeneratorAgent(
                variant_idx=i, model=self.model, max_turns=self.max_turns_gen
            )
            for i in range(self.k)
        ]

    def _make_scorer(self) -> QValueScorerAgent:
        return QValueScorerAgent(
            model=self.model, max_turns=self.max_turns_score, gamma=self.gamma
        )

    async def propose(
        self,
        *,
        target: Target,
        coverage: CoverageMap,
        history: list[TestResult],
    ) -> tuple[TestCase, AgentUsage]:
        uncovered = coverage.uncovered

        gens = self._make_generators()
        gen_tasks = [
            gen.propose(target=target, uncovered=uncovered, history=history)
            for gen in gens
        ]
        gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)

        candidates: list[TestCase] = []
        gen_usages: list[AgentUsage] = []
        for r in gen_results:
            if isinstance(r, BaseException):
                continue
            tc, usage = r
            gen_usages.append(usage)
            if tc is not None:
                candidates.append(tc)

        if not candidates:
            # All K failed → return an empty test case so the loop records
            # an error and consumes one execution.
            return (
                TestCase(code="", rationale="all generators failed"),
                _sum_usages(gen_usages),
            )

        scorer = self._make_scorer()
        score_tasks = [
            scorer.score(target=target, uncovered=uncovered, candidate=c)
            for c in candidates
        ]
        score_results = await asyncio.gather(*score_tasks, return_exceptions=True)

        scored: list[_Scored] = []
        score_usages: list[AgentUsage] = []
        for c, sr in zip(candidates, score_results):
            if isinstance(sr, BaseException):
                # Treat scoring failure as Q=0 so we don't drop the candidate
                # entirely — it might still be the only viable option.
                scored.append(_Scored(c, QScore(immediate_branches=0, future_value=0, gamma=self.gamma)))
                continue
            score, usage = sr
            score_usages.append(usage)
            if score is None:
                scored.append(_Scored(c, QScore(immediate_branches=0, future_value=0, gamma=self.gamma)))
            else:
                scored.append(_Scored(c, score))

        # argmax on Q, with tiebreakers
        scored.sort(
            key=lambda s: (-s.score.q, -s.score.immediate_branches, len(s.candidate.code))
        )
        best = scored[0]

        all_usage = _sum_usages(gen_usages + score_usages)
        # Stash the Q value in the candidate rationale prefix for later inspection.
        annotated = TestCase(
            code=best.candidate.code,
            rationale=(
                f"[Q={best.score.q:.2f} imm={best.score.immediate_branches} "
                f"fut={best.score.future_value}] {best.candidate.rationale}"
            ),
        )
        return annotated, all_usage
