"""Evaluator Protocol — the contract every execution backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from xsmith.domain.candidate import Candidate
from xsmith.domain.evaluation import Evaluation
from xsmith.domain.goal import Goals
from xsmith.domain.target import Target


@runtime_checkable
class Evaluator(Protocol):
    """Anything that knows how to evaluate a Candidate against a Target."""

    async def evaluate(self, candidate: Candidate, target: Target) -> Evaluation:
        """Run `candidate` against `target` and return an `Evaluation`."""
        ...

    async def enumerate_goals(self, target: Target) -> Goals:
        """Return the universe of recognizable goals for `target`.

        For the coverage instance, implementations run a no-op import-only
        test under coverage.py with `branch=True` and report executed+missing.
        """
        ...
