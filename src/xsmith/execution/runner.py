"""TestRunner Protocol + the shared result type produced by all runners."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from xsmith.domain.coverage import BranchSet
from xsmith.domain.target import Target
from xsmith.domain.test_case import Outcome, TestCase


class TestRunResult:
    """Outcome of executing one TestCase against one Target."""

    def __init__(
        self,
        *,
        outcome: Outcome,
        stdout: str = "",
        stderr: str = "",
        duration_s: float = 0.0,
        branches_covered: BranchSet | None = None,
    ):
        self.outcome = outcome
        self.stdout = stdout
        self.stderr = stderr
        self.duration_s = duration_s
        self.branches_covered = branches_covered or BranchSet()


@runtime_checkable
class TestRunner(Protocol):
    """Anything that knows how to execute a TestCase against a Target."""

    async def run(self, test_case: TestCase, target: Target) -> TestRunResult: ...

    async def discover_branches(self, target: Target) -> BranchSet:
        """Return the universe of executable branch arcs for `target`.

        Implementations typically run a no-op import-only test under
        coverage.py with `branch=True` and report executed+missing.
        """
        ...
