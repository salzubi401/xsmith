"""Convert Explorer output → RunResult → JSON on disk."""

from __future__ import annotations

import json
from pathlib import Path

from xsmith.agents.base import AgentUsage
from xsmith.exploration.explorer import ExplorationResult, Step
from xsmith.results.schema import (
    AgentUsageRecord,
    RunResult,
    StepResult,
    TargetResult,
)


def _agent_usage_to_record(u: AgentUsage) -> AgentUsageRecord:
    return AgentUsageRecord(
        tokens_in=u.tokens_in,
        tokens_out=u.tokens_out,
        tokens_cache_read=u.tokens_cache_read,
        tokens_cache_creation=u.tokens_cache_creation,
        cost_usd=u.cost_usd,
        duration_ms=u.duration_ms,
        num_turns=u.num_turns,
    )


def _step_to_record(step: Step) -> StepResult:
    return StepResult(
        iteration=step.iteration,
        outcome=step.evaluation.outcome,
        duration_s=step.evaluation.duration_s,
        new_goals=len(step.new_goals),
        hit_after=step.hit_after,
        total=step.total,
        candidate_rationale=step.candidate.rationale,
        candidate_code=step.candidate.code,
        agent_usage=_agent_usage_to_record(step.agent_usage),
    )


def to_target_result(out: ExplorationResult) -> TargetResult:
    return TargetResult(
        target_id=out.target.target_id,
        module_path=out.target.module_path,
        steps=[_step_to_record(s) for s in out.steps],
        final_hit=out.hit_count,
        final_total=out.total_count,
        final_fraction=out.fraction,
    )


def write_run(run: RunResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = run.model_dump()
    body["total_cost_usd"] = run.total_cost_usd
    body["total_tokens_in"] = run.total_tokens_in
    body["total_tokens_out"] = run.total_tokens_out
    body["total_cache_read"] = run.total_cache_read
    body["total_cache_creation"] = run.total_cache_creation
    p.write_text(json.dumps(body, indent=2, default=str))
