import json

from xsmith.domain.budget import Budget
from xsmith.domain.goal import Goal, Goals
from xsmith.domain.target import Target
from xsmith.results.schema import (
    AgentUsageRecord,
    RunResult,
    StepResult,
    TargetResult,
)


def test_target_serialization_roundtrip():
    t = Target(
        target_id="x", module_path="a.b",
        source="def f(): pass",
        goals=Goals.from_iterable([Goal(file="a.py", src=1, dst=2)]),
    )
    js = t.model_dump_json()
    parsed = Target.model_validate_json(js)
    assert parsed.target_id == "x"
    assert len(parsed.goals) == 1


def test_budget_consumes_and_exhausts():
    b = Budget(steps=2)
    assert not b.exhausted
    b.consume_step()
    b.consume_step()
    assert b.steps == 0
    assert b.exhausted


def test_budget_cost_gating_only_when_enforced():
    b = Budget(steps=10, enforce_cost=True, max_usd=1.0)
    b.record_usage(usd=2.0)
    assert b.exhausted

    b2 = Budget(steps=10, enforce_cost=False, max_usd=1.0)
    b2.record_usage(usd=2.0)
    assert not b2.exhausted


def test_run_result_totals():
    usage = AgentUsageRecord(
        tokens_in=10, tokens_out=20, tokens_cache_read=5,
        tokens_cache_creation=3, cost_usd=0.01,
    )
    step = StepResult(
        iteration=1, outcome="pass", duration_s=0.1, new_goals=2,
        hit_after=2, total=10,
        candidate_rationale="ok", candidate_code="def test_x(): pass",
        agent_usage=usage,
    )
    tr = TargetResult(target_id="t", module_path="m", steps=[step, step])
    rr = RunResult(
        benchmark="x", model="m", k=5, gamma=0.5, step_budget=24,
        targets=[tr],
    )
    assert rr.total_cost_usd == 0.02
    assert rr.total_tokens_in == 20
    assert rr.total_cache_read == 10

    # JSON dump round-trips through pydantic (computed properties not in dump).
    body = rr.model_dump()
    assert body["benchmark"] == "x"
    assert json.dumps(body, default=str)
