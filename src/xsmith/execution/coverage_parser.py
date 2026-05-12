"""Parse coverage.py JSON reports into a `BranchSet`.

coverage.py >=6 emits per-file `executed_branches` (and `missing_branches`) as
lists of `[src_line, dst_line]` pairs when `branch=True` is set. We only care
about files we asked it to track; the caller passes that filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xsmith.domain.coverage import Branch, BranchSet


def parse_executed_branches(
    coverage_json: str | dict[str, Any],
    *,
    file_filter: set[str] | None = None,
) -> BranchSet:
    """Return all *executed* branches across the report.

    `file_filter` is a set of file paths (as they appear under `files:` keys in
    the JSON) to include. If None, include all files.
    """
    data = json.loads(coverage_json) if isinstance(coverage_json, str) else coverage_json
    files = data.get("files", {}) or {}
    branches: list[Branch] = []
    for path, payload in files.items():
        if file_filter is not None and path not in file_filter:
            continue
        executed = payload.get("executed_branches", []) or []
        for pair in executed:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            src, dst = int(pair[0]), int(pair[1])
            branches.append(Branch(file=path, src=src, dst=dst))
    return BranchSet.from_iterable(branches)


def parse_all_branches(
    coverage_json: str | dict[str, Any],
    *,
    file_filter: set[str] | None = None,
) -> BranchSet:
    """Return *executed + missing* — the universe of measurable branches."""
    data = json.loads(coverage_json) if isinstance(coverage_json, str) else coverage_json
    files = data.get("files", {}) or {}
    branches: list[Branch] = []
    for path, payload in files.items():
        if file_filter is not None and path not in file_filter:
            continue
        for key in ("executed_branches", "missing_branches"):
            for pair in payload.get(key, []) or []:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                src, dst = int(pair[0]), int(pair[1])
                branches.append(Branch(file=path, src=src, dst=dst))
    return BranchSet.from_iterable(branches)


def parse_from_file(path: str | Path, *, file_filter: set[str] | None = None) -> BranchSet:
    return parse_executed_branches(Path(path).read_text(), file_filter=file_filter)
