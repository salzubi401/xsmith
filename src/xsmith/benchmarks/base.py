"""Benchmark Protocol — anything that yields a list of Targets."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from xsmith.domain.target import Target


@runtime_checkable
class Benchmark(Protocol):
    name: str

    def load(self, *, max_targets: int | None = None) -> list[Target]: ...
