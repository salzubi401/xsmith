from __future__ import annotations

import pytest

from xsmith.domain.coverage import BranchSet
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase as XTestCase
from xsmith.execution.docker_runner import _rewrite_imports as docker_rewrite_imports
from xsmith.execution.subprocess_runner import (
    SubprocessRunner,
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
async def test_subprocess_runner_executes_test_and_reports_target_branches():
    runner = SubprocessRunner(timeout_s=10)
    target = _target()

    discovered = await runner.discover_branches(target)
    assert len(discovered) >= 2

    result = await runner.run(
        XTestCase(
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
    assert len(result.branches_covered) > 0
    assert len(result.branches_covered - discovered) == 0


@pytest.mark.asyncio
async def test_subprocess_runner_distinguishes_failures_from_errors():
    runner = SubprocessRunner(timeout_s=10)
    target = _target()

    failed = await runner.run(
        XTestCase(
            code=(
                "from sample_mod import classify\n\n"
                "def test_wrong_assertion():\n"
                "    assert classify(2) == 'non-positive'\n"
            ),
        ),
        target,
    )
    assert failed.outcome == "fail"

    errored = await runner.run(
        XTestCase(code="import definitely_missing_module\n\n\ndef test_never_runs():\n    pass\n"),
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
async def test_subprocess_runner_timeout_reports_error():
    runner = SubprocessRunner(timeout_s=0.1)
    result = await runner.run(
        XTestCase(code="def test_hangs():\n    while True:\n        pass\n"),
        _target(),
    )

    assert result.outcome == "error"
    assert "timeout after" in result.stderr
    assert result.branches_covered == BranchSet()
