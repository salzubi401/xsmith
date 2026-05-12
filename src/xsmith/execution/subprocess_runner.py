"""SubprocessRunner — run a generated test locally via `python -m coverage`.

Layout written per call into a fresh tempdir:

    <tmp>/
      target_pkg/__init__.py        # makes the module importable
      target_pkg/<module>.py        # target.source verbatim
      tests/test_generated.py       # candidate.code verbatim
      .coveragerc                   # branch=True, source=target_pkg

Then we run:

    python -m coverage run --rcfile=.coveragerc -m pytest tests -q
    python -m coverage json --rcfile=.coveragerc -o coverage.json

…and parse coverage.json into a BranchSet keyed by the relative target path.

The Target.module_path's last dotted segment is used as the module filename;
the leading segments are flattened into a single package (we don't try to
preserve arbitrary nesting — `foo.bar.baz` is imported as the module `baz`
inside package `target_pkg`). The generated test must import using the
`module_path` we tell it, so we rewrite `module_path` references in the test
to `target_pkg.<leaf>` before running.

This is a deliberate simplification: the runner exists for development and
tests. Real benchmark runs use the Docker runner.
"""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
import time
from pathlib import Path

from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase
from xsmith.execution.coverage_parser import parse_all_branches, parse_executed_branches
from xsmith.execution.runner import TestRunResult

_COVERAGERC = """\
[run]
branch = True
source = target_pkg

[report]
show_missing = False
"""


class SubprocessRunner:
    def __init__(self, *, python: str | None = None, timeout_s: float = 30.0):
        self.python = python or sys.executable
        self.timeout_s = timeout_s

    async def run(self, test_case: TestCase, target: Target) -> TestRunResult:
        leaf = target.module_path.rsplit(".", 1)[-1]
        rewritten_code = _rewrite_imports(test_case.code, target.module_path, f"target_pkg.{leaf}")

        with tempfile.TemporaryDirectory(prefix="xsmith-run-") as tmp:
            root = Path(tmp)
            pkg = root / "target_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / f"{leaf}.py").write_text(target.source)

            # Write any extra sibling files alongside, in case the module imports them.
            for rel, content in target.extra_files.items():
                p = pkg / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)

            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_generated.py").write_text(rewritten_code)

            (root / ".coveragerc").write_text(_COVERAGERC)

            outcome, stdout, stderr, duration = await self._run_pytest(root)

            covered = _BranchSetEmpty()
            try:
                cov_json = root / "coverage.json"
                if cov_json.exists():
                    target_rel = f"target_pkg/{leaf}.py"
                    covered = parse_executed_branches(
                        cov_json.read_text(),
                        file_filter={target_rel},
                    )
            except Exception as e:  # noqa: BLE001  surfaced via stderr
                stderr = f"{stderr}\n[coverage_parser] {e}"

            return TestRunResult(
                outcome=outcome,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration,
                branches_covered=covered,
            )

    async def discover_branches(self, target: Target):
        """Run a no-op import-only test and return the *universe* of branches
        (executed + missing) for the target file.

        Used by the exploration loop to initialize a CoverageMap's `total`.
        """
        from xsmith.domain.coverage import BranchSet

        leaf = target.module_path.rsplit(".", 1)[-1]
        import_test = (
            f"import target_pkg.{leaf} as _m\n"
            "def test_import():\n"
            "    assert _m is not None\n"
        )
        tc = TestCase(code=import_test, rationale="discover branches")
        # Reuse run() but parse all branches instead of just executed.
        with tempfile.TemporaryDirectory(prefix="xsmith-disc-") as tmp:
            root = Path(tmp)
            pkg = root / "target_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / f"{leaf}.py").write_text(target.source)
            for rel, content in target.extra_files.items():
                p = pkg / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_discover.py").write_text(import_test)
            (root / ".coveragerc").write_text(_COVERAGERC)

            await self._run_pytest(root)

            cov_json = root / "coverage.json"
            if not cov_json.exists():
                return BranchSet()
            target_rel = f"target_pkg/{leaf}.py"
            return parse_all_branches(cov_json.read_text(), file_filter={target_rel})

    async def _run_pytest(self, root: Path):
        start = time.monotonic()
        cmd_run = [
            self.python,
            "-m",
            "coverage",
            "run",
            "--rcfile=.coveragerc",
            "-m",
            "pytest",
            "tests",
            "-q",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd_run,
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ("error", "", f"timeout after {self.timeout_s}s", time.monotonic() - start)

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        rc = proc.returncode
        # pytest: 0=pass, 1=test failure, others=error
        if rc == 0:
            outcome = "pass"
        elif rc == 1:
            outcome = "fail"
        else:
            outcome = "error"

        # Generate JSON report even on failure (we still want the partial coverage).
        cmd_json = [
            self.python,
            "-m",
            "coverage",
            "json",
            "--rcfile=.coveragerc",
            "-o",
            "coverage.json",
        ]
        json_proc = await asyncio.create_subprocess_exec(
            *cmd_json,
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await json_proc.communicate()

        duration = time.monotonic() - start
        return outcome, stdout, stderr, duration


def _BranchSetEmpty():
    from xsmith.domain.coverage import BranchSet

    return BranchSet()


_IMPORT_PATTERNS = [
    # `from <module_path> import ...`
    (re.compile(r"\bfrom\s+{m}\b"), "from {r}"),
    # `import <module_path>`
    (re.compile(r"\bimport\s+{m}\b"), "import {r}"),
    # `import <module_path> as X`  — handled by the second pattern already
]


def _rewrite_imports(code: str, original: str, replacement: str) -> str:
    """Rewrite `module_path` references in a test to point at the sandboxed package.

    Only rewrites `from X import ...` and `import X` head forms — submodule
    references (`X.sub`) are not rewritten because we don't sandbox submodules.
    """
    safe_original = re.escape(original)
    out = re.sub(
        rf"(?m)^(\s*)from\s+{safe_original}\b(?!\.)",
        rf"\1from {replacement}",
        code,
    )
    out = re.sub(
        rf"(?m)^(\s*)import\s+{safe_original}\b(?!\.)",
        rf"\1import {replacement}",
        out,
    )
    return out
