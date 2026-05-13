"""Candidate — a single artifact proposed by a strategy for evaluation.

In the coverage instance, a `Candidate` is a self-contained pytest-style
Python test script (`code`). In other applications it could be an adversarial
prompt, an HTTP request body, or any other payload the Evaluator knows how to
run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Outcome = Literal["pass", "fail", "error"]


class Candidate(BaseModel):
    """An artifact proposed by a strategy, to be handed to an Evaluator."""

    code: str
    rationale: str = ""
    """Free-form explanation from the LLM of which goals this aims to hit."""
