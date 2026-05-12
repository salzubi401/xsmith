from __future__ import annotations

from xsmith.execution.coverage_parser import (
    parse_all_branches,
    parse_executed_branches,
    parse_from_file,
)


COVERAGE_JSON = {
    "files": {
        "target_pkg/sample.py": {
            "executed_branches": [[1, 2], [1, 3], ["bad"], [4, 5, 6]],
            "missing_branches": [[2, 4], [1, 3]],
        },
        "target_pkg/other.py": {
            "executed_branches": [[10, 11]],
            "missing_branches": [[10, 12]],
        },
    }
}


def _keys(branches):
    return sorted(b.key() for b in branches)


def test_parse_executed_branches_filters_files_and_ignores_malformed_pairs():
    branches = parse_executed_branches(
        COVERAGE_JSON,
        file_filter={"target_pkg/sample.py"},
    )

    assert _keys(branches) == [
        "target_pkg/sample.py:1->2",
        "target_pkg/sample.py:1->3",
    ]


def test_parse_all_branches_combines_executed_and_missing_without_duplicates():
    branches = parse_all_branches(
        COVERAGE_JSON,
        file_filter={"target_pkg/sample.py"},
    )

    assert _keys(branches) == [
        "target_pkg/sample.py:1->2",
        "target_pkg/sample.py:1->3",
        "target_pkg/sample.py:2->4",
    ]


def test_parse_from_file_reads_coverage_json(tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text(
        '{"files": {"target_pkg/sample.py": {"executed_branches": [[3, 4]]}}}'
    )

    branches = parse_from_file(path, file_filter={"target_pkg/sample.py"})

    assert _keys(branches) == ["target_pkg/sample.py:3->4"]
