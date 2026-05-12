from __future__ import annotations

import os

import pytest

from xsmith.agents.test_generator import TestGeneratorAgent as XTestGeneratorAgent
from xsmith.config import load_settings
from xsmith.domain.target import Target
from xsmith.execution.subprocess_runner import SubprocessRunner


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
async def test_real_sdk_generated_test_runs_with_subprocess_runner(monkeypatch):
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
    runner = SubprocessRunner(timeout_s=20)
    uncovered = await runner.discover_branches(target)
    assert len(uncovered) > 0

    agent = XTestGeneratorAgent(
        variant_idx=0,
        model=os.environ.get("XSMITH_INTEGRATION_MODEL") or settings.MODEL,
        max_turns=6,
    )
    candidate, _ = await agent.propose(target=target, uncovered=uncovered, history=[])

    assert candidate is not None
    assert candidate.code.strip()

    result = await runner.run(candidate, target)

    assert result.outcome == "pass", (
        "generated test should pass under SubprocessRunner\n"
        f"code:\n{candidate.code}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert len(result.branches_covered) > 0
