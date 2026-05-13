from __future__ import annotations

import pytest

from xsmith.domain.candidate import Candidate
from xsmith.domain.goal import Goals
from xsmith.domain.target import Target
from xsmith.execution.docker import _rewrite_imports as docker_rewrite_imports
from xsmith.execution.subprocess import (
    SubprocessEvaluator,
    _rewrite_imports as subprocess_rewrite_imports,
)


TARGET_SOURCE = """\
def classify(value):
    if value > 0:
        return "positive"
    return "non-positive"
"""


def _target() -> Target:
    return Target(
        target_id="classify",
        module_path="sample_mod",
        source=TARGET_SOURCE,
    )


@pytest.mark.asyncio
async def test_subprocess_evaluator_executes_candidate_and_reports_goals_hit():
    evaluator = SubprocessEvaluator(timeout_s=10)
    target = _target()

    discovered = await evaluator.enumerate_goals(target)
    assert len(discovered) >= 2

    result = await evaluator.evaluate(
        Candidate(
            code=(
                "from sample_mod import classify\n\n"
                "def test_positive():\n"
                "    assert classify(2) == 'positive'\n"
            ),
            rationale="cover positive branch",
        ),
        target,
    )

    assert result.outcome == "pass"
    assert len(result.goals_hit) > 0
    assert len(result.goals_hit - discovered) == 0


@pytest.mark.asyncio
async def test_subprocess_evaluator_distinguishes_failures_from_errors():
    evaluator = SubprocessEvaluator(timeout_s=10)
    target = _target()

    failed = await evaluator.evaluate(
        Candidate(
            code=(
                "from sample_mod import classify\n\n"
                "def test_wrong_assertion():\n"
                "    assert classify(2) == 'non-positive'\n"
            ),
        ),
        target,
    )
    assert failed.outcome == "fail"

    errored = await evaluator.evaluate(
        Candidate(code="import definitely_missing_module\n\n\ndef test_never_runs():\n    pass\n"),
        target,
    )
    assert errored.outcome == "error"
    assert "ModuleNotFoundError" in (errored.stdout + errored.stderr)


def test_rewrite_imports_handles_module_heads_only():
    code = """\
from sample_mod import classify
import sample_mod as sm
import sample_mod
import sample_mod.extra
text = "import sample_mod"
"""
    expected = """\
from target_pkg.sample_mod import classify
import target_pkg.sample_mod as sm
import target_pkg.sample_mod
import sample_mod.extra
text = "import sample_mod"
"""

    assert subprocess_rewrite_imports(code, "sample_mod", "target_pkg.sample_mod") == expected
    assert docker_rewrite_imports(code, "sample_mod", "target_pkg.sample_mod") == expected


@pytest.mark.asyncio
async def test_subprocess_evaluator_timeout_reports_error():
    evaluator = SubprocessEvaluator(timeout_s=0.1)
    result = await evaluator.evaluate(
        Candidate(code="def test_hangs():\n    while True:\n        pass\n"),
        _target(),
    )

    assert result.outcome == "error"
    assert "timeout after" in result.stderr
    assert result.goals_hit == Goals()
