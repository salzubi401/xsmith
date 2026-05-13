"""Progress — the running record of which goals have been hit so far.

`Progress.all` is the universe of goals that could be hit for this target;
`Progress.hit` is the subset already achieved; `Progress.missing` is the
complement. `update(new_hit)` merges and returns the delta (newly-hit goals).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from xsmith.domain.goal import Goals


class Progress(BaseModel):
    """Running progress for a target. Mutable via `update`."""

    hit: Goals = Field(default_factory=Goals)
    all: Goals = Field(default_factory=Goals)

    @property
    def missing(self) -> Goals:
        return self.all - self.hit

    @property
    def fraction(self) -> float:
        if len(self.all) == 0:
            return 0.0
        return len(self.hit) / len(self.all)

    def delta(self, new_hit: Goals) -> Goals:
        """Goals that would be newly hit if `new_hit` were applied."""
        return new_hit - self.hit

    def update(self, new_hit: Goals) -> Goals:
        """Merge `new_hit` into `self.hit`; return the delta added."""
        added = self.delta(new_hit)
        self.hit = self.hit | added
        return added
