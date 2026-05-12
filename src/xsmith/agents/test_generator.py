"""TestGeneratorAgent — proposes a single TestCase via the SDK."""

from __future__ import annotations

from xsmith.agents.base import AgentRunner, AgentUsage
from xsmith.agents.isolation import build_options
from xsmith.agents.prompts import generator_system_prompt
from xsmith.agents.tools import GeneratorState, build_generator_tools
from xsmith.domain.coverage import BranchSet
from xsmith.domain.target import Target
from xsmith.domain.test_case import TestCase, TestResult


def _generator_user_prompt(target: Target, uncovered: BranchSet) -> str:
    branch_lines = sorted(b.key() for b in uncovered)
    branches_text = "\n".join(branch_lines) if branch_lines else "(all covered)"
    return f"""\
# Target module
module_path: {target.module_path}

```python
{target.source}
```

# Uncovered branches (preview)
{branches_text}

Use `view_coverage` and `view_history` if helpful, then call `submit_test`
EXACTLY ONCE with your final pytest-style test script.
"""


class TestGeneratorAgent:
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
        self.runner = runner or AgentRunner(submit_tool_name="submit_test")
        self.max_turns = max_turns

    async def propose(
        self,
        *,
        target: Target,
        uncovered: BranchSet,
        history: list[TestResult],
    ) -> tuple[TestCase | None, AgentUsage]:
        state = GeneratorState(uncovered=uncovered, history=history)
        server = build_generator_tools(state)
        system_prompt = generator_system_prompt(self.variant_idx)
        options = build_options(
            system_prompt=system_prompt,
            mcp_server=server,
            allowed_tool_names=["view_coverage", "view_history", "submit_test"],
            model=self.model,
            max_turns=self.max_turns,
        )
        prompt = _generator_user_prompt(target, uncovered)
        submission, usage = await self.runner.run(options=options, user_prompt=prompt)
        if submission is None:
            return None, usage
        code = submission.get("code", "")
        rationale = submission.get("rationale", "")
        if not code:
            return None, usage
        return TestCase(code=code, rationale=rationale), usage
