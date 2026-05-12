"""Target = the module under test, plus enough metadata for the runner to execute it."""

from __future__ import annotations

from pydantic import BaseModel, Field

from xsmith.domain.coverage import BranchSet


class Target(BaseModel):
    """A single unit of code we want to generate tests for.

    `module_path` is a dotted module path (e.g. `mypkg.utils.parse`) that the
    generated test will import. `source` is the on-disk text of the module
    (so the LLM can see it without filesystem access). `entrypoints` is the
    list of top-level callables / classes worth exercising — informational.
    `branches` is the universe of executable branch arcs against which
    coverage will be measured.
    """

    target_id: str
    module_path: str
    source: str
    entrypoints: list[str] = Field(default_factory=list)
    branches: BranchSet = Field(default_factory=BranchSet)
    extra_files: dict[str, str] = Field(default_factory=dict)
    """Optional sibling files keyed by relative path, written into the sandbox."""

    def short_summary(self) -> str:
        return f"Target(id={self.target_id}, module={self.module_path}, branches={len(self.branches)})"
