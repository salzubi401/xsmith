"""QValueScorerAgent — produces Q-score components for a candidate test."""

from __future__ import annotations

from pydantic import BaseModel, Field

from xsmith.agents.base import AgentRunner, AgentUsage
from xsmith.agents.isolation import build_options
from xsmith.agents.prompts import SCORER_SYSTEM
from xsmith.agents.tools import ScorerState, build_scorer_tools
from xsmith.domain.coverage import BranchSet
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase


class QScore(BaseModel):
    immediate_branches: int = Field(ge=0)
    future_value: int = Field(ge=0, le=10)
    gamma: float = 0.5

    @property
    def q(self) -> float:
        return float(self.immediate_branches) + self.gamma * float(self.future_value)


def _scorer_user_prompt(target: Target, uncovered: BranchSet, candidate: TestCase) -> str:
    branch_lines = sorted(b.key() for b in uncovered)
    branches_text = "\n".join(branch_lines) if branch_lines else "(none — all covered)"
    return f"""\
# Target module
module_path: {target.module_path}

```python
{target.source}
```

# Currently uncovered branches
{branches_text}

# Candidate test
Rationale: {candidate.rationale}

```python
{candidate.code}
```

Estimate `immediate_branches` and `future_value`, then call `submit_score`.
"""


class QValueScorerAgent:
    """Wrap AgentRunner to produce a QScore for a single candidate."""

    def __init__(
        self,
        *,
        runner: AgentRunner | None = None,
        model: str,
        max_turns: int = 3,
        gamma: float = 0.5,
    ):
        self.runner = runner or AgentRunner(submit_tool_name="submit_score")
        self.model = model
        self.max_turns = max_turns
        self.gamma = gamma

    async def score(
        self,
        *,
        target: Target,
        uncovered: BranchSet,
        candidate: TestCase,
    ) -> tuple[QScore | None, AgentUsage]:
        state = ScorerState()
        server = build_scorer_tools(state)
        options = build_options(
            system_prompt=SCORER_SYSTEM,
            mcp_server=server,
            allowed_tool_names=["submit_score"],
            model=self.model,
            max_turns=self.max_turns,
        )
        prompt = _scorer_user_prompt(target, uncovered, candidate)
        submission, usage = await self.runner.run(options=options, user_prompt=prompt)
        if submission is None:
            return None, usage
        try:
            score = QScore(
                immediate_branches=submission["immediate_branches"],
                future_value=submission["future_value"],
                gamma=self.gamma,
            )
        except (KeyError, ValueError, TypeError):
            return None, usage
        return score, usage
