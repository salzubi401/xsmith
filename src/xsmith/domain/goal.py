"""Goal primitives — the unit of progress in an exploration.

A `Goal` is anything we can recognize when achieved. In the coverage instance
it's a `(file, src_line, dst_line)` branch arc reported by coverage.py; in
other applications it could be a unique attack outcome, a distinct API
response shape, or a covered behavioral spec.

`Goals` is a set of goals with set-algebra operators.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field


class Goal(BaseModel):
    """A single goal. In the coverage instance, a (file, src, dst) branch arc."""

    model_config = {"frozen": True}

    file: str
    src: int
    dst: int

    def key(self) -> str:
        return f"{self.file}:{self.src}->{self.dst}"


class Goals(BaseModel):
    """An immutable-ish set of goals; comparable for set ops."""

    items: frozenset[Goal] = Field(default_factory=frozenset)

    @classmethod
    def from_iterable(cls, items: Iterable[Goal]) -> "Goals":
        return cls(items=frozenset(items))

    def __or__(self, other: "Goals") -> "Goals":
        return Goals(items=self.items | other.items)

    def __sub__(self, other: "Goals") -> "Goals":
        return Goals(items=self.items - other.items)

    def __and__(self, other: "Goals") -> "Goals":
        return Goals(items=self.items & other.items)

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, g: Goal) -> bool:
        return g in self.items

    def __iter__(self):
        return iter(self.items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Goals):
            return NotImplemented
        return self.items == other.items

    def __hash__(self) -> int:
        return hash(self.items)
