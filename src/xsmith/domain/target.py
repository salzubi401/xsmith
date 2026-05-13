"""Target = the unit being explored, plus enough metadata to evaluate it."""

from __future__ import annotations

from pydantic import BaseModel, Field

from xsmith.domain.goal import Goals


class Target(BaseModel):
    """A single unit we want to explore.

    `module_path` is a dotted module path (e.g. `mypkg.utils.parse`) that the
    candidate will import (used by the coverage instance). `source` is the
    on-disk text of the module (so the LLM can see it without filesystem
    access). `entrypoints` is the list of top-level callables / classes worth
    exercising — informational. `goals` is the universe of recognizable goals
    for this target; in the coverage instance, the executable branch arcs.
    """

    target_id: str
    module_path: str
    source: str
    entrypoints: list[str] = Field(default_factory=list)
    goals: Goals = Field(default_factory=Goals)
    extra_files: dict[str, str] = Field(default_factory=dict)
    """Optional sibling files keyed by relative path, written into the sandbox."""

    def short_summary(self) -> str:
        return f"Target(id={self.target_id}, module={self.module_path}, goals={len(self.goals)})"
