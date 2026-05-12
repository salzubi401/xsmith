"""DockerTestRunner — execute generated tests inside a Docker container.

Adapted from the upstream `runner/docker_coverage.py` protocol:

  1. Write a self-contained pytest test script into a host tempdir.
  2. Write the target module + .coveragerc into the same tempdir.
  3. `docker run --rm -v <tmp>:/work <image> bash -c '...'`
  4. Inside the container: `python -m coverage run --branch -m pytest -q tests`,
     then emit a separator and cat `coverage.json` to stdout.
  5. Host parses everything after the separator as coverage JSON.

The runner is async at the public API level but uses a thread pool under the
hood (subprocess.run is sync). Docker's daemon serializes work anyway, so
async-over-thread is the right pattern here.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from xsmith.domain.coverage import Branch, BranchSet
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase
from xsmith.execution.runner import TestRunResult

_SEPARATOR = "===XSMITH_COVERAGE_JSON_START==="

_COVERAGERC = """\
[run]
branch = True
source = target_pkg

[report]
show_missing = False
"""


class DockerTestRunner:
    """Run a candidate test inside an isolated Docker container."""

    def __init__(
        self,
        *,
        image: str = "xsmith-runner:latest",
        python_bin: str = "python",
        working_dir: str = "/work",
        timeout_s: float = 60.0,
        extra_env: dict[str, str] | None = None,
        docker_bin: str = "docker",
    ):
        self.image = image
        self.python_bin = python_bin
        self.working_dir = working_dir
        self.timeout_s = timeout_s
        self.extra_env = extra_env or {}
        self.docker_bin = docker_bin

    async def run(self, test_case: TestCase, target: Target) -> TestRunResult:
        return await asyncio.to_thread(self._run_sync, test_case, target)

    async def discover_branches(self, target: Target) -> BranchSet:
        leaf = target.module_path.rsplit(".", 1)[-1]
        import_test = (
            f"import target_pkg.{leaf} as _m\n"
            "def test_import():\n"
            "    assert _m is not None\n"
        )
        tc = TestCase(code=import_test, rationale="discover branches")
        return await asyncio.to_thread(self._discover_sync, tc, target)

    def _discover_sync(self, test_case: TestCase, target: Target) -> BranchSet:
        leaf = target.module_path.rsplit(".", 1)[-1]
        tmp = Path(tempfile.mkdtemp(prefix="xsmith-disc-"))
        try:
            pkg = tmp / "target_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / f"{leaf}.py").write_text(target.source)
            for rel, content in target.extra_files.items():
                p = pkg / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            tests_dir = tmp / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_discover.py").write_text(test_case.code)
            (tmp / ".coveragerc").write_text(_COVERAGERC)

            inner = (
                f"cd {self.working_dir} && "
                f"{self.python_bin} -m coverage run --rcfile=.coveragerc "
                f"-m pytest tests -q 2>&1; "
                f"echo '{_SEPARATOR}'; "
                f"{self.python_bin} -m coverage json --rcfile=.coveragerc "
                f"-o /tmp/cov.json 2>/dev/null && cat /tmp/cov.json 2>/dev/null"
            )
            cmd = [
                self.docker_bin, "run", "--rm",
                "--network=none",
                "--entrypoint", "bash",
                "-v", f"{tmp}:{self.working_dir}",
                self.image,
                "-c", inner,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return BranchSet()
            stdout = proc.stdout or ""
            if _SEPARATOR in stdout:
                _, after = stdout.split(_SEPARATOR, 1)
                json_start = after.find("{")
                cov_json_str = after[json_start:].strip() if json_start >= 0 else ""
            else:
                cov_json_str = ""
            if not cov_json_str:
                return BranchSet()
            try:
                data = json.loads(cov_json_str)
            except json.JSONDecodeError:
                return BranchSet()
            target_rel = f"target_pkg/{leaf}.py"
            branches: list[Branch] = []
            for path, payload in (data.get("files", {}) or {}).items():
                if target_rel not in path:
                    continue
                for key in ("executed_branches", "missing_branches"):
                    for pair in payload.get(key, []) or []:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            branches.append(Branch(file=path, src=int(pair[0]), dst=int(pair[1])))
            return BranchSet.from_iterable(branches)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _run_sync(self, test_case: TestCase, target: Target) -> TestRunResult:
        leaf = target.module_path.rsplit(".", 1)[-1]
        rewritten = _rewrite_imports(test_case.code, target.module_path, f"target_pkg.{leaf}")

        tmp = Path(tempfile.mkdtemp(prefix="xsmith-docker-"))
        try:
            pkg = tmp / "target_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / f"{leaf}.py").write_text(target.source)
            for rel, content in target.extra_files.items():
                p = pkg / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)

            tests_dir = tmp / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_generated.py").write_text(rewritten)

            (tmp / ".coveragerc").write_text(_COVERAGERC)

            env_args: list[str] = []
            for k, v in self.extra_env.items():
                env_args.extend(["-e", f"{k}={v}"])

            inner = (
                f"cd {self.working_dir} && "
                f"{self.python_bin} -m coverage run --rcfile=.coveragerc "
                f"-m pytest tests -q 2>&1; "
                f"rc=$?; "
                f"echo '{_SEPARATOR}'; "
                f"{self.python_bin} -m coverage json --rcfile=.coveragerc "
                f"-o /tmp/cov.json 2>/dev/null && cat /tmp/cov.json 2>/dev/null; "
                f"exit $rc"
            )

            cmd = [
                self.docker_bin,
                "run",
                "--rm",
                "--network=none",
                "--entrypoint",
                "bash",
                "-v",
                f"{tmp}:{self.working_dir}",
                *env_args,
                self.image,
                "-c",
                inner,
            ]

            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout_s
                )
            except subprocess.TimeoutExpired:
                return TestRunResult(
                    outcome="error",
                    stderr=f"docker timeout after {self.timeout_s}s",
                    duration_s=time.monotonic() - start,
                )

            duration = time.monotonic() - start
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            if _SEPARATOR in stdout:
                before, after = stdout.split(_SEPARATOR, 1)
                test_output = before.strip()
                json_start = after.find("{")
                cov_json_str = after[json_start:].strip() if json_start >= 0 else ""
            else:
                test_output = stdout
                cov_json_str = ""

            target_rel = f"target_pkg/{leaf}.py"
            covered = _parse_executed(cov_json_str, target_rel)

            rc = proc.returncode
            if rc == 0:
                outcome = "pass"
            elif rc == 1:
                outcome = "fail"
            else:
                outcome = "error"

            return TestRunResult(
                outcome=outcome,
                stdout=test_output,
                stderr=stderr,
                duration_s=duration,
                branches_covered=covered,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _parse_executed(cov_json_str: str, target_rel: str) -> BranchSet:
    if not cov_json_str:
        return BranchSet()
    try:
        data = json.loads(cov_json_str)
    except json.JSONDecodeError:
        return BranchSet()
    files = data.get("files", {}) or {}
    branches: list[Branch] = []
    for path, payload in files.items():
        if target_rel not in path:
            continue
        for pair in payload.get("executed_branches", []) or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                branches.append(Branch(file=path, src=int(pair[0]), dst=int(pair[1])))
    return BranchSet.from_iterable(branches)


def _rewrite_imports(code: str, original: str, replacement: str) -> str:
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
