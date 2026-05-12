from __future__ import annotations

from xsmith.agents.isolation import build_options
from xsmith.agents.tools import ScorerState, build_scorer_tools


def test_build_options_namespaces_tools_and_blocks_setting_sources():
    server = build_scorer_tools(ScorerState())

    options = build_options(
        system_prompt="system",
        mcp_server=server,
        allowed_tool_names=["submit_score"],
        model="model-name",
        max_turns=3,
        server_name="sandbox",
    )

    assert options.model == "model-name"
    assert options.system_prompt == "system"
    assert options.max_turns == 3
    assert options.mcp_servers == {"sandbox": server}
    assert options.allowed_tools == ["mcp__sandbox__submit_score"]
    assert options.setting_sources == []
    assert options.permission_mode == "acceptEdits"
