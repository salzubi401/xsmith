"""GeneratorAgent — proposes a single Candidate via the SDK."""

from __future__ import annotations

from xsmith.agents.base import AgentRunner, AgentUsage
from xsmith.agents.isolation import build_options
from xsmith.agents.prompts import generator_system_prompt
from xsmith.agents.tools import GeneratorState, build_generator_tools
from xsmith.domain.candidate import Candidate
from xsmith.domain.evaluation import Evaluation
from xsmith.domain.goal import Goals
from xsmith.domain.target import Target


def _generator_user_prompt(target: Target, missing: Goals) -> str:
    goal_lines = sorted(g.key() for g in missing)
    goals_text = "\n".join(goal_lines) if goal_lines else "(all hit)"
    return f"""\
# Target module
module_path: {target.module_path}

```python
{target.source}
```

# Missing goals (preview)
{goals_text}

Use `view_progress` and `view_history` if helpful, then call
`submit_candidate` EXACTLY ONCE with your final candidate.
"""


class GeneratorAgent:
    """One agent = one diversity variant = one candidate per call."""

    def __init__(
        self,
        *,
        variant_idx: int,
        model: str,
        runner: AgentRunner | None = None,
        max_turns: int = 8,
    ):
        self.variant_idx = variant_idx
        self.model = model
        self.runner = runner or AgentRunner(submit_tool_name="submit_candidate")
        self.max_turns = max_turns

    async def propose(
        self,
        *,
        target: Target,
        missing: Goals,
        history: list[Evaluation],
    ) -> tuple[Candidate | None, AgentUsage]:
        state = GeneratorState(missing=missing, history=history)
        server = build_generator_tools(state)
        system_prompt = generator_system_prompt(self.variant_idx)
        options = build_options(
            system_prompt=system_prompt,
            mcp_server=server,
            allowed_tool_names=["view_progress", "view_history", "submit_candidate"],
            model=self.model,
            max_turns=self.max_turns,
        )
        prompt = _generator_user_prompt(target, missing)
        submission, usage = await self.runner.run(options=options, user_prompt=prompt)
        if submission is None:
            return None, usage
        code = submission.get("code", "")
        rationale = submission.get("rationale", "")
        if not code:
            return None, usage
        return Candidate(code=code, rationale=rationale), usage
