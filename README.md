# xsmith

Curiosity-driven Python test generation. A clean reimplementation of
the CovQValue algorithm from
[amayuelas/qcurious-tester](https://github.com/amayuelas/qcurious-tester),
built on the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk).

## What it does

For a given Python module:

1. Spawns K (default 5) `TestGeneratorAgent`s in parallel, each with a
   different diversity directive (typical / edge / error / deep / adversarial).
2. Scores every candidate test with a `QValueScorerAgent`:
   `Q = immediate_branches + γ · future_value`  (γ = 0.5).
3. Picks the argmax-Q candidate, runs it under `coverage.py`, updates the
   covered-branch set.
4. Repeats until the per-target execution budget is exhausted.

All LLM I/O is structured via in-process MCP tools (`submit_test`,
`submit_score`), not regex parsing. The SDK is locked down with
`setting_sources=[]`, so user-local `~/.claude` config doesn't leak in.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.11+
- `ANTHROPIC_API_KEY` for non-trivial runs (the SDK shells out to the
  Claude Code CLI under the hood)
- Docker (optional — only for `--runner docker`)

## Install

```bash
uv sync
uv pip install -e .
cp .env.example .env  # then add your ANTHROPIC_API_KEY
```

## Run

```bash
# Smoke test (local subprocess runner, no Docker required)
uv run xsmith explore --benchmark repo_explore --targets 1 --budget 5 \
  --output results/run.json

# Larger run, Docker sandbox
docker build -f Dockerfile.runner -t xsmith-runner:latest .
uv run xsmith explore --benchmark repo_explore --targets 10 --budget 24 \
  --runner docker --output results/full.json
```

See `xsmith explore --help` for all flags.

## Tests

```bash
uv run pytest tests/unit -v
uv run pytest tests/integration -v   # requires ANTHROPIC_API_KEY
```

## Layout

```
src/xsmith/
  domain/        # Pydantic types: Target, CoverageMap, TestCase, Budget
  agents/        # ClaudeSDK agents + isolation + MCP tools
  strategies/    # GenerationStrategy Protocol + CovQValueStrategy
  execution/     # TestRunner Protocol + Subprocess and Docker runners
  exploration/   # ExplorationLoop (depends only on the two Protocols)
  benchmarks/    # RepoExploreBench, TestGenEvalBench
  results/       # JSON schema + writer
  cli.py         # `xsmith explore` Typer entry point
  config.py      # pydantic-settings, env-loaded
```

The two extensibility seams worth knowing:

- **`GenerationStrategy` Protocol** (`strategies/base.py`) — drop in a new
  strategy without touching the loop.
- **`TestRunner` Protocol** (`execution/runner.py`) — drop in a new sandbox
  (e.g. Firecracker, gVisor) without touching anything else.

`agents/base.py` is the single place `claude-agent-sdk` is imported.
