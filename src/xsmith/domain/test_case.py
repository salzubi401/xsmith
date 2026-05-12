"""TestCase = a generated pytest-style test; TestResult = the outcome of running it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from xsmith.domain.coverage import BranchSet

Outcome = Literal["pass", "fail", "error"]


class TestCase(BaseModel):
    """A self-contained Python test script proposed by the generator."""

    code: str
    rationale: str = ""
    """Free-form explanation from the LLM of which branches this aims to hit."""


class TestResult(BaseModel):
    """The outcome of executing a TestCase against a Target."""

    test_case: TestCase
    outcome: Outcome
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    branches_covered: BranchSet = Field(default_factory=BranchSet)
    new_branches_covered: BranchSet = Field(default_factory=BranchSet)
    """Subset of `branches_covered` that was *newly* covered (delta vs prior)."""
