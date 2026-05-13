"""Evaluation — the outcome of running a Candidate against a Target.

A single type returned by every Evaluator. The Explorer separately tracks
which goals were *newly* hit (the delta vs. running Progress) on each Step,
so Evaluation just reports the raw `goals_hit` observed during execution.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from xsmith.domain.candidate import Candidate, Outcome
from xsmith.domain.goal import Goals


class Evaluation(BaseModel):
    """The outcome of running one Candidate against one Target."""

    candidate: Candidate
    outcome: Outcome
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    goals_hit: Goals = Field(default_factory=Goals)
