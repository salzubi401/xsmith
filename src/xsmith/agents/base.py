"""AgentRunner — the single abstraction over `claude-agent-sdk`.

Both `GeneratorAgent` and `ScorerAgent` go through this. Replacing
`claude-agent-sdk` with the plain `anthropic` SDK later means rewriting this
one file.

The pattern:
  1. Build per-call state + an MCP server whose tools close over that state.
  2. Build `ClaudeAgentOptions` via `build_options()` (locked-down).
  3. `async with ClaudeSDKClient(options) as client:` connect.
  4. `await client.query(user_prompt)` send the user turn.
  5. Iterate `client.receive_response()` to drain messages, collecting:
       - any `ToolUseBlock`s whose name matches the submit tool — store args
       - the final `ResultMessage` — pull cost + usage out of it
  6. Return `(submission_args, AgentUsage)` to caller.

If the model never calls the submit tool within `max_turns`, we return
`(None, usage)` and let the caller decide whether to retry or fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    ToolUseBlock,
)


@dataclass(frozen=True)
class AgentUsage:
    """Cost & token telemetry for one agent invocation."""

    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cache_read: int = 0
    tokens_cache_creation: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    num_turns: int = 0

    @classmethod
    def zero(cls) -> "AgentUsage":
        return cls()


def _usage_from_result(rm: ResultMessage) -> AgentUsage:
    u = rm.usage or {}
    return AgentUsage(
        tokens_in=int(u.get("input_tokens", 0) or 0),
        tokens_out=int(u.get("output_tokens", 0) or 0),
        tokens_cache_read=int(u.get("cache_read_input_tokens", 0) or 0),
        tokens_cache_creation=int(u.get("cache_creation_input_tokens", 0) or 0),
        cost_usd=float(rm.total_cost_usd or 0.0),
        duration_ms=int(rm.duration_ms or 0),
        num_turns=int(rm.num_turns or 0),
    )


class AgentRunner:
    """Drive a single ClaudeSDKClient turn and harvest a submit-tool call."""

    def __init__(self, *, submit_tool_name: str, server_name: str = "xsmith"):
        self.submit_tool_name = submit_tool_name
        self.server_name = server_name

    async def run(
        self,
        *,
        options: ClaudeAgentOptions,
        user_prompt: str,
    ) -> tuple[dict[str, Any] | None, AgentUsage]:
        """Send `user_prompt` to a fresh SDK client and return (submission, usage).

        `submission` is the input args dict from the first ToolUseBlock whose
        name matches `mcp__<server>__<submit_tool_name>`. If the model never
        called the submit tool, `submission` is None.
        """
        full_name = f"mcp__{self.server_name}__{self.submit_tool_name}"
        submission: dict[str, Any] | None = None
        usage = AgentUsage.zero()

        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock) and block.name == full_name:
                            if submission is None:  # first submit wins
                                submission = dict(block.input or {})
                elif isinstance(msg, ResultMessage):
                    usage = _usage_from_result(msg)

        return submission, usage
