"""Output JSON schema — stable across runs."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AgentUsageRecord(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cache_read: int = 0
    tokens_cache_creation: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    num_turns: int = 0


class StepResult(BaseModel):
    iteration: int
    outcome: str
    duration_s: float
    new_goals: int
    hit_after: int
    total: int
    candidate_rationale: str
    candidate_code: str
    agent_usage: AgentUsageRecord


class TargetResult(BaseModel):
    target_id: str
    module_path: str
    steps: list[StepResult] = Field(default_factory=list)
    final_hit: int = 0
    final_total: int = 0
    final_fraction: float = 0.0


class RunResult(BaseModel):
    """Top-level result for a single `xsmith explore` invocation."""

    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    benchmark: str
    model: str
    k: int
    gamma: float
    step_budget: int
    targets: list[TargetResult] = Field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.agent_usage.cost_usd for t in self.targets for s in t.steps)

    @property
    def total_tokens_in(self) -> int:
        return sum(s.agent_usage.tokens_in for t in self.targets for s in t.steps)

    @property
    def total_tokens_out(self) -> int:
        return sum(s.agent_usage.tokens_out for t in self.targets for s in t.steps)

    @property
    def total_cache_read(self) -> int:
        return sum(s.agent_usage.tokens_cache_read for t in self.targets for s in t.steps)

    @property
    def total_cache_creation(self) -> int:
        return sum(s.agent_usage.tokens_cache_creation for t in self.targets for s in t.steps)
