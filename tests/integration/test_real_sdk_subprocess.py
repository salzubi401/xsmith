from __future__ import annotations

import os

import pytest

from xsmith.agents.generator import GeneratorAgent
from xsmith.config import load_settings
from xsmith.domain.target import Target
from xsmith.execution.subprocess import SubprocessEvaluator


SIMPLE_TARGET = """\
def classify(value):
    if value is None:
        return "missing"
    if value > 0:
        return "positive"
    return "other"
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_sdk_generated_candidate_runs_with_subprocess_evaluator(monkeypatch):
    settings = load_settings()
    api_key = os.environ.get("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
    if not api_key:
        pytest.skip("requires ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)

    target = Target(
        target_id="simple/classify",
        module_path="sample_target",
        source=SIMPLE_TARGET,
    )
    evaluator = SubprocessEvaluator(timeout_s=20)
    missing = await evaluator.enumerate_goals(target)
    assert len(missing) > 0

    agent = GeneratorAgent(
        variant_idx=0,
        model=os.environ.get("XSMITH_INTEGRATION_MODEL") or settings.MODEL,
        max_turns=6,
    )
    candidate, _ = await agent.propose(target=target, missing=missing, history=[])

    assert candidate is not None
    assert candidate.code.strip()

    result = await evaluator.evaluate(candidate, target)

    assert result.outcome == "pass", (
        "generated candidate should pass under SubprocessEvaluator\n"
        f"code:\n{candidate.code}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert len(result.goals_hit) > 0
