import json

from xsmith.domain.budget import Budget
from xsmith.domain.coverage import Branch, BranchSet
from xsmith.domain.target import Target
from xsmith.results.schema import (
    AgentUsageRecord,
    IterationResult,
    RunResult,
    TargetResult,
)


def test_target_serialization_roundtrip():
    t = Target(
        target_id="x", module_path="a.b",
        source="def f(): pass",
        branches=BranchSet.from_iterable([Branch(file="a.py", src=1, dst=2)]),
    )
    js = t.model_dump_json()
    parsed = Target.model_validate_json(js)
    assert parsed.target_id == "x"
    assert len(parsed.branches) == 1


def test_budget_consumes_and_exhausts():
    b = Budget(exec_remaining=2)
    assert not b.exhausted
    b.consume_execution()
    b.consume_execution()
    assert b.exec_remaining == 0
    assert b.exhausted


def test_budget_cost_gating_only_when_enforced():
    b = Budget(exec_remaining=10, enforce_cost=True, max_usd=1.0)
    b.record_usage(usd=2.0)
    assert b.exhausted

    b2 = Budget(exec_remaining=10, enforce_cost=False, max_usd=1.0)
    b2.record_usage(usd=2.0)
    assert not b2.exhausted


def test_run_result_totals():
    usage = AgentUsageRecord(
        tokens_in=10, tokens_out=20, tokens_cache_read=5,
        tokens_cache_creation=3, cost_usd=0.01,
    )
    itr = IterationResult(
        iteration=1, outcome="pass", duration_s=0.1, new_branches=2,
        coverage_after=2, coverage_total=10,
        test_rationale="ok", test_code="def test_x(): pass",
        agent_usage=usage,
    )
    tr = TargetResult(target_id="t", module_path="m", iterations=[itr, itr])
    rr = RunResult(
        benchmark="x", model="m", k=5, gamma=0.5, exec_budget=24,
        targets=[tr],
    )
    assert rr.total_cost_usd == 0.02
    assert rr.total_tokens_in == 20
    assert rr.total_cache_read == 10

    # JSON dump round-trips through pydantic (computed properties not in dump).
    body = rr.model_dump()
    assert body["benchmark"] == "x"
    assert json.dumps(body, default=str)
