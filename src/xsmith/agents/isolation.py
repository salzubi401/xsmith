"""Build a `ClaudeAgentOptions` that is fully isolated from the user's
local Claude Code configuration.

Critical: `setting_sources=[]` means we DO NOT inherit:
- ~/.claude/settings.json
- ./.claude/settings.json
- ./.claude/settings.local.json

…nor the hooks, skills, or permissions those files configure. The library
must behave identically regardless of who runs it.
"""

from __future__ import annotations

from claude_agent_sdk import ClaudeAgentOptions, McpSdkServerConfig


def build_options(
    *,
    system_prompt: str,
    mcp_server: McpSdkServerConfig,
    allowed_tool_names: list[str],
    model: str,
    max_turns: int,
    server_name: str = "xsmith",
) -> ClaudeAgentOptions:
    """Return a locked-down `ClaudeAgentOptions` for an in-process MCP server.

    `allowed_tool_names` is the bare tool names (e.g. `["view_progress",
    "submit_candidate"]`). They get namespaced to `mcp__<server>__<name>` so
    the SDK can route them to our in-process server only.
    """
    namespaced = [f"mcp__{server_name}__{name}" for name in allowed_tool_names]
    return ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        mcp_servers={server_name: mcp_server},
        allowed_tools=namespaced,
        setting_sources=[],
        permission_mode="acceptEdits",
        max_turns=max_turns,
    )
