"""Budget = the resource envelope for a single target exploration."""

from __future__ import annotations

from pydantic import BaseModel


class Budget(BaseModel):
    """Tracks remaining step count, tokens, and dollars.

    The exploration loop's primary termination condition is `steps`: each
    call to the evaluator counts as one step. Tokens and USD are *recorded*
    for telemetry but don't gate iteration unless `enforce_cost` is True.
    """

    steps: int
    tokens_in_used: int = 0
    tokens_out_used: int = 0
    tokens_cache_read_used: int = 0
    tokens_cache_creation_used: int = 0
    usd_used: float = 0.0
    enforce_cost: bool = False
    max_usd: float | None = None

    def consume_step(self) -> None:
        self.steps -= 1

    def record_usage(
        self,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_cache_read: int = 0,
        tokens_cache_creation: int = 0,
        usd: float = 0.0,
    ) -> None:
        self.tokens_in_used += tokens_in
        self.tokens_out_used += tokens_out
        self.tokens_cache_read_used += tokens_cache_read
        self.tokens_cache_creation_used += tokens_cache_creation
        self.usd_used += usd

    @property
    def exhausted(self) -> bool:
        if self.steps <= 0:
            return True
        if self.enforce_cost and self.max_usd is not None and self.usd_used >= self.max_usd:
            return True
        return False
