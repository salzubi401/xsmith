"""Coverage primitives.

A `Branch` identifies a `(file, source_line, dest_line)` triple as reported by
coverage.py's branch tracking. `BranchSet` is a frozen set of branches.
`CoverageMap` is the mutable running record of covered branches for a target.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field


class Branch(BaseModel):
    """A single branch arc (file:src_line -> file:dst_line)."""

    model_config = {"frozen": True}

    file: str
    src: int
    dst: int

    def key(self) -> str:
        return f"{self.file}:{self.src}->{self.dst}"


class BranchSet(BaseModel):
    """An immutable-ish set of branches; comparable for set ops."""

    branches: frozenset[Branch] = Field(default_factory=frozenset)

    @classmethod
    def from_iterable(cls, items: Iterable[Branch]) -> "BranchSet":
        return cls(branches=frozenset(items))

    def __or__(self, other: "BranchSet") -> "BranchSet":
        return BranchSet(branches=self.branches | other.branches)

    def __sub__(self, other: "BranchSet") -> "BranchSet":
        return BranchSet(branches=self.branches - other.branches)

    def __and__(self, other: "BranchSet") -> "BranchSet":
        return BranchSet(branches=self.branches & other.branches)

    def __len__(self) -> int:
        return len(self.branches)

    def __contains__(self, b: Branch) -> bool:
        return b in self.branches

    def __iter__(self):
        return iter(self.branches)


class CoverageMap(BaseModel):
    """Running coverage for a target. Mutable via `update`."""

    covered: BranchSet = Field(default_factory=BranchSet)
    total: BranchSet = Field(default_factory=BranchSet)

    @property
    def uncovered(self) -> BranchSet:
        return self.total - self.covered

    @property
    def fraction(self) -> float:
        if len(self.total) == 0:
            return 0.0
        return len(self.covered) / len(self.total)

    def delta(self, new_covered: BranchSet) -> BranchSet:
        """Branches that would be newly covered if `new_covered` were applied."""
        return new_covered - self.covered

    def update(self, new_covered: BranchSet) -> BranchSet:
        """Merge `new_covered` into `self.covered`; return the delta added."""
        added = self.delta(new_covered)
        self.covered = self.covered | added
        return added
