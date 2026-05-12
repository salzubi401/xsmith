"""Convert ExplorationLoop output → RunResult → JSON on disk."""

from __future__ import annotations

import json
from pathlib import Path

from xsmith.agents.base import AgentUsage
from xsmith.exploration.explorer import IterationRecord, TargetExplorationResult
from xsmith.results.schema import (
    AgentUsageRecord,
    IterationResult,
    RunResult,
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


def _iteration_to_record(rec: IterationRecord) -> IterationResult:
    return IterationResult(
        iteration=rec.iteration,
        outcome=rec.run_result.outcome,
        duration_s=rec.run_result.duration_s,
        new_branches=len(rec.new_branches),
        coverage_after=rec.coverage_after,
        coverage_total=rec.coverage_total,
        test_rationale=rec.test_case.rationale,
        test_code=rec.test_case.code,
        agent_usage=_agent_usage_to_record(rec.agent_usage),
    )


def to_target_result(out: TargetExplorationResult) -> TargetResult:
    return TargetResult(
        target_id=out.target.target_id,
        module_path=out.target.module_path,
        iterations=[_iteration_to_record(r) for r in out.iterations],
        final_covered=out.covered_count,
        final_total=out.total_count,
        final_fraction=out.coverage_fraction,
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
