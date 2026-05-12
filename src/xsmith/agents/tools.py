"""In-process MCP tools for the agents.

Each agent gets a freshly-built MCP server because the tools close over
agent-local state (current uncovered branches, recent history, captured
submission). This means tools can't be reused across concurrent agents —
build them per-call.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from xsmith.domain.coverage import BranchSet
from xsmith.domain.test_case import TestResult


# ----- Generator-side state ------------------------------------------------


class GeneratorState:
    """Per-call closure container for a generator agent.

    `submission` is set when the model calls `submit_test`. The AgentRunner
    reads it once the SDK conversation ends.
    """

    def __init__(self, uncovered: BranchSet, history: list[TestResult]):
        self.uncovered = uncovered
        self.history = history
        self.submission: dict[str, Any] | None = None


def build_generator_tools(state: GeneratorState):
    """Build the four tools that close over `state` and the MCP server."""

    @tool(
        "view_coverage",
        "List currently uncovered branch arcs as 'file:src->dst' strings.",
        {},
    )
    async def view_coverage(args: dict[str, Any]) -> dict[str, Any]:
        items = sorted(b.key() for b in state.uncovered)
        if not items:
            text = "All branches are already covered."
        else:
            text = "Uncovered branches:\n" + "\n".join(items)
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "view_history",
        "Return the last N test attempts with outcomes and new-coverage counts.",
        {"limit": int},
    )
    async def view_history(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit", 5))
        items = state.history[-limit:] if limit > 0 else state.history
        if not items:
            text = "No prior attempts."
        else:
            lines = []
            for i, r in enumerate(items):
                lines.append(
                    f"[{i}] outcome={r.outcome} new_branches={len(r.new_branches_covered)} "
                    f"rationale={r.test_case.rationale[:120]!r}"
                )
            text = "Recent attempts (oldest first):\n" + "\n".join(lines)
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "submit_test",
        "Submit your final test. Call EXACTLY ONCE. Provide a rationale and "
        "the complete pytest-style script as `code`.",
        {"rationale": str, "code": str},
    )
    async def submit_test(args: dict[str, Any]) -> dict[str, Any]:
        state.submission = {
            "rationale": str(args.get("rationale", "")),
            "code": str(args.get("code", "")),
        }
        return {"content": [{"type": "text", "text": "Submission received."}]}

    server = create_sdk_mcp_server(
        name="xsmith",
        version="0.1.0",
        tools=[view_coverage, view_history, submit_test],
    )
    return server


# ----- Scorer-side state ---------------------------------------------------


class ScorerState:
    def __init__(self):
        self.submission: dict[str, Any] | None = None


def build_scorer_tools(state: ScorerState):
    @tool(
        "submit_score",
        "Submit Q-score components: immediate_branches and future_value.",
        {"immediate_branches": int, "future_value": int},
    )
    async def submit_score(args: dict[str, Any]) -> dict[str, Any]:
        state.submission = {
            "immediate_branches": int(args.get("immediate_branches", 0)),
            "future_value": int(args.get("future_value", 0)),
        }
        return {"content": [{"type": "text", "text": "Score received."}]}

    server = create_sdk_mcp_server(
        name="xsmith",
        version="0.1.0",
        tools=[submit_score],
    )
    return server
