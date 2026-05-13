"""In-process MCP tools for the agents.

Each agent gets a freshly-built MCP server because the tools close over
agent-local state (current missing goals, recent history, captured
submission). This means tools can't be reused across concurrent agents —
build them per-call.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from xsmith.domain.evaluation import Evaluation
from xsmith.domain.goal import Goals


# ----- Generator-side state ------------------------------------------------


class GeneratorState:
    """Per-call closure container for a generator agent.

    `submission` is set when the model calls `submit_candidate`. The
    AgentRunner reads it once the SDK conversation ends.
    """

    def __init__(self, missing: Goals, history: list[Evaluation]):
        self.missing = missing
        self.history = history
        self.submission: dict[str, Any] | None = None


def build_generator_tools(state: GeneratorState):
    """Build the tools that close over `state` and the MCP server."""

    @tool(
        "view_progress",
        "List currently missing goals as 'file:src->dst' strings (for the "
        "coverage instance, these are the uncovered branch arcs).",
        {},
    )
    async def view_progress(args: dict[str, Any]) -> dict[str, Any]:
        items = sorted(g.key() for g in state.missing)
        if not items:
            text = "All goals already hit."
        else:
            text = "Missing goals:\n" + "\n".join(items)
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "view_history",
        "Return the last N candidate attempts with outcomes and goals-hit counts.",
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
                    f"[{i}] outcome={r.outcome} goals_hit={len(r.goals_hit)} "
                    f"rationale={r.candidate.rationale[:120]!r}"
                )
            text = "Recent attempts (oldest first):\n" + "\n".join(lines)
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "submit_candidate",
        "Submit your final candidate. Call EXACTLY ONCE. Provide a rationale "
        "and the artifact as `code` (for the coverage instance, a complete "
        "pytest-style test script).",
        {"rationale": str, "code": str},
    )
    async def submit_candidate(args: dict[str, Any]) -> dict[str, Any]:
        state.submission = {
            "rationale": str(args.get("rationale", "")),
            "code": str(args.get("code", "")),
        }
        return {"content": [{"type": "text", "text": "Submission received."}]}

    server = create_sdk_mcp_server(
        name="xsmith",
        version="0.2.0",
        tools=[view_progress, view_history, submit_candidate],
    )
    return server


# ----- Scorer-side state ---------------------------------------------------


class ScorerState:
    def __init__(self):
        self.submission: dict[str, Any] | None = None


def build_scorer_tools(state: ScorerState):
    @tool(
        "submit_score",
        "Submit Q-score components: immediate_goals and future_value.",
        {"immediate_goals": int, "future_value": int},
    )
    async def submit_score(args: dict[str, Any]) -> dict[str, Any]:
        state.submission = {
            "immediate_goals": int(args.get("immediate_goals", 0)),
            "future_value": int(args.get("future_value", 0)),
        }
        return {"content": [{"type": "text", "text": "Score received."}]}

    server = create_sdk_mcp_server(
        name="xsmith",
        version="0.2.0",
        tools=[submit_score],
    )
    return server
