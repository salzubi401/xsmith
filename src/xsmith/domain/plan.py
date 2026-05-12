"""A sequence of TestCases (rarely used externally — kept for future strategies)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from xsmith.domain.test_case import TestCase


class TestPlan(BaseModel):
    cases: list[TestCase] = Field(default_factory=list)
