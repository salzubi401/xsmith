"""ScorerAgent — produces Q-score components for a candidate."""

from __future__ import annotations

from pydantic import BaseModel, Field

from xsmith.agents.base import AgentRunner, AgentUsage
from xsmith.agents.isolation import build_options
from xsmith.agents.prompts import SCORER_SYSTEM
from xsmith.agents.tools import ScorerState, build_scorer_tools
from xsmith.domain.candidate import Candidate
from xsmith.domain.goal import Goals
from xsmith.domain.target import Target


class QScore(BaseModel):
    immediate_goals: int = Field(ge=0)
    future_value: int = Field(ge=0, le=10)
    gamma: float = 0.5

    @property
    def q(self) -> float:
        return float(self.immediate_goals) + self.gamma * float(self.future_value)


def _scorer_user_prompt(target: Target, missing: Goals, candidate: Candidate) -> str:
    goal_lines = sorted(g.key() for g in missing)
    goals_text = "\n".join(goal_lines) if goal_lines else "(none — all hit)"
    return f"""\
# Target module
module_path: {target.module_path}

```python
{target.source}
```

# Currently missing goals
{goals_text}

# Candidate
Rationale: {candidate.rationale}

```python
{candidate.code}
```

Estimate `immediate_goals` and `future_value`, then call `submit_score`.
"""


class ScorerAgent:
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
        missing: Goals,
        candidate: Candidate,
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
        prompt = _scorer_user_prompt(target, missing, candidate)
        submission, usage = await self.runner.run(options=options, user_prompt=prompt)
        if submission is None:
            return None, usage
        try:
            score = QScore(
                immediate_goals=submission["immediate_goals"],
                future_value=submission["future_value"],
                gamma=self.gamma,
            )
        except (KeyError, ValueError, TypeError):
            return None, usage
        return score, usage
