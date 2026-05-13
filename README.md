# xsmith

A generic exploration framework — give it a `Target`, an `Evaluator`, and a
`Strategy`, and it iteratively proposes candidates, runs them, and tracks
which goals have been hit. The reference instance is curiosity-driven
**Python test generation** with branch coverage as the goal set, but the
same loop is intended for red-teaming, API exploration, agent benchmark
creation, and prompt optimization.

Built on the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk).

## What it does (coverage instance)

For a given Python module:

1. Spawns K (default 5) `GeneratorAgent`s in parallel, each with a different
   diversity directive (typical / edge / error / deep / adversarial).
2. Scores every candidate with a `ScorerAgent`:
   `Q = immediate_goals + γ · future_value`  (γ = 0.5).
3. Picks the argmax-Q candidate, runs it under `coverage.py`, updates the
   hit goals.
4. Repeats until the per-target step budget is exhausted.

All LLM I/O is structured via in-process MCP tools (`submit_candidate`,
`submit_score`), not regex parsing. The SDK is locked down with
`setting_sources=[]`, so user-local `~/.claude` config doesn't leak in.

## API at a glance

Everything you'd want is re-exported from the top-level `xsmith` package:

```python
from xsmith import (
    Explorer, explore,                              # the loop
    Target, Goal, Goals, Progress,                  # what we're exploring
    Candidate, Evaluation, Outcome,                 # one round-trip
    Budget,                                         # resource envelope
    Strategy, Evaluator,                            # extension seams (Protocols)
    QValueStrategy,                                 # the reference strategy
    SubprocessEvaluator, DockerEvaluator,           # coverage-instance evaluators
    Benchmark, AgentUsage, Step, ExplorationResult, # supporting types
)
```

### Data flow

```
   ┌──────────┐                                        ┌──────────┐
   │  Target  │                                        │ Progress │
   │  (goals) │                                        │ hit/all  │
   └────┬─────┘                                        └────┬─────┘
        │                                                   ▲
        │     ┌─────────────┐    ┌────────────────┐         │
        ├────►│  Strategy   ├───►│   Candidate    │         │
        │     │  .propose() │    │ (code+rationale)         │
   ┌────┴──┐  └─────────────┘    └────────┬───────┘         │
   │history│                              │                 │
   │(Evals)│                              ▼                 │
   └───▲───┘                     ┌────────────────┐         │
       │                         │   Evaluator    │         │
       │                         │  .evaluate()   │         │
       │                         └────────┬───────┘         │
       │                                  ▼                 │
       │                         ┌────────────────┐         │
       └─────────────────────────┤   Evaluation   │─goals_hit
                                 │ outcome/output │  .update()
                                 └────────────────┘         │
                                                            │
                            one Step = candidate + evaluation + new_goals
                            (collected into ExplorationResult.steps)
```

### Domain types

| Type | What it is |
|---|---|
| `Goal` | A single discoverable unit. In the coverage instance: a `(file, src_line, dst_line)` branch arc. `Goal.key()` returns `"file:src->dst"`. |
| `Goals` | An immutable set of `Goal`s with `\|`, `-`, `&`, `len()`, `in`, `iter()`. Build with `Goals.from_iterable(...)`. |
| `Target` | The thing being explored. Fields: `target_id`, `module_path`, `source`, `goals` (the universe), plus optional `entrypoints` and `extra_files`. |
| `Progress` | Running state. Fields: `hit: Goals`, `all: Goals`. Properties: `missing`, `fraction`. Methods: `update(new_hit)` merges and returns the delta. |
| `Candidate` | An artifact a strategy proposes. Fields: `code`, `rationale`. (For the coverage instance, `code` is a pytest script.) |
| `Evaluation` | The result of running a candidate. Fields: `candidate`, `outcome` (`"pass"/"fail"/"error"`), `stdout`, `stderr`, `duration_s`, `goals_hit`. |
| `Budget` | Resource envelope. Fields: `steps` (the primary termination lever), token counters, `max_usd`, `enforce_cost`. Methods: `consume_step()`, `record_usage(...)`, `exhausted` property. |
| `AgentUsage` | Per-call LLM telemetry. Fields: `tokens_in/out`, `tokens_cache_read/creation`, `cost_usd`, `duration_ms`, `num_turns`. |

### Extension seams (Protocols)

| Protocol | Contract | Default impl(s) |
|---|---|---|
| `Strategy` | `async propose(*, target, progress, history) -> (Candidate, AgentUsage)` | `QValueStrategy` — K parallel generators + Q-value argmax. |
| `Evaluator` | `async evaluate(candidate, target) -> Evaluation` + `async enumerate_goals(target) -> Goals` | `SubprocessEvaluator` (local), `DockerEvaluator` (sandboxed). |

Writing a new application = implementing one or both of these.

### The loop

| | |
|---|---|
| `Explorer(strategy, evaluator, on_step=None)` | The loop object. `.run(target, budget, initial_progress=None)` returns an `ExplorationResult`. |
| `explore(*, target, strategy, evaluator, budget, on_step=None, initial_progress=None)` | One-shot async helper that wraps `Explorer(...).run(...)`. |
| `Step` | One iteration: `iteration`, `candidate`, `evaluation`, `new_goals`, `hit_after`, `total`, `agent_usage`. |
| `ExplorationResult` | Final output: `target`, `steps: list[Step]`, `final_progress`, plus `.hit_count`, `.total_count`, `.fraction`. |

## Toy example

A complete runnable script. We give xsmith a small `fizzbuzz` function and
ask it to hit all 4 branches.

```python
# toy.py — run with: ANTHROPIC_API_KEY=... uv run python toy.py
import asyncio
from xsmith import (
    Explorer, Budget, QValueStrategy, SubprocessEvaluator, Target,
)

TARGET_SRC = """
def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
"""

async def main():
    # 1. Describe what we're exploring.
    target = Target(
        target_id="toy/fizzbuzz",
        module_path="fizzbuzz_mod",
        source=TARGET_SRC,
    )

    # 2. Pick an evaluator. SubprocessEvaluator runs candidates locally under
    #    coverage.py and reports executed branch arcs as `Goals`.
    evaluator = SubprocessEvaluator(timeout_s=15)

    # 3. Enumerate the universe of goals (one-time, per target).
    target.goals = await evaluator.enumerate_goals(target)
    print(f"universe: {len(target.goals)} goals")
    for g in sorted(target.goals, key=lambda g: (g.src, g.dst)):
        print(f"  {g.key()}")

    # 4. Pick a strategy. QValueStrategy spawns K agents per step and picks the
    #    argmax-Q candidate.
    strategy = QValueStrategy(model="claude-sonnet-4-6", k=5, gamma=0.5)

    # 5. Drive the loop. on_step fires after each iteration.
    explorer = Explorer(
        strategy=strategy,
        evaluator=evaluator,
        on_step=lambda s: print(
            f"step {s.iteration}: outcome={s.evaluation.outcome} "
            f"+{len(s.new_goals)} new  hit={s.hit_after}/{s.total} "
            f"cost=${s.agent_usage.cost_usd:.4f}"
        ),
    )
    result = await explorer.run(target=target, budget=Budget(steps=3))

    print(f"\nfinal: {result.hit_count}/{result.total_count} "
          f"({result.fraction:.0%})")

asyncio.run(main())
```

Expected output (numbers vary by model + run):

```
universe: 7 goals
  target_pkg/fizzbuzz_mod.py:2->3
  target_pkg/fizzbuzz_mod.py:2->4
  target_pkg/fizzbuzz_mod.py:4->5
  target_pkg/fizzbuzz_mod.py:4->6
  target_pkg/fizzbuzz_mod.py:6->7
  target_pkg/fizzbuzz_mod.py:6->8
  target_pkg/fizzbuzz_mod.py:8->...
step 1: outcome=pass +5 new  hit=5/7 cost=$0.0421
step 2: outcome=pass +2 new  hit=7/7 cost=$0.0388

final: 7/7 (100%)
```

**What happened, line by line:**

- `Target` packages the source the LLM sees and the dotted path the
  generated candidate will import.
- `enumerate_goals` runs a no-op import-only test under `coverage.py
  --branch=True` to learn the universe of arcs (executed + missing).
- `QValueStrategy.propose` (called once per step) spawns 5 `GeneratorAgent`s
  in parallel, drops failures, scores survivors with a `ScorerAgent`,
  returns the argmax-Q `Candidate`.
- `SubprocessEvaluator.evaluate` writes the candidate to a tempdir, runs
  `pytest -q` under coverage, parses `coverage.json`, returns an
  `Evaluation` with `outcome` and `goals_hit`.
- The `Explorer` updates `Progress` with the newly-hit goals and records a
  `Step`. The loop stops early once `progress.missing` is empty (so step 3
  never fires).

## Extending xsmith

The two protocols are the only thing the loop knows about. Implement them
for a non-coverage application.

### Custom Strategy

```python
from xsmith import Strategy, Target, Progress, Evaluation, Candidate, AgentUsage

class GreedyStrategy:
    """Always proposes the same candidate. Useful for tests + baselines."""

    def __init__(self, code: str):
        self.code = code

    async def propose(
        self, *, target: Target, progress: Progress, history: list[Evaluation],
    ) -> tuple[Candidate, AgentUsage]:
        return Candidate(code=self.code, rationale="fixed"), AgentUsage()
```

Any class with that signature satisfies `Strategy` (it's a `Protocol`).

### Custom Evaluator

```python
from xsmith import Evaluator, Candidate, Target, Evaluation, Goals

class EchoEvaluator:
    """Trivial evaluator: every candidate 'passes' and hits no goals.
    
    Real applications: replace the body with an HTTP call, an LLM judge, a
    shell command, etc., and translate the result into Goals."""

    async def evaluate(self, candidate: Candidate, target: Target) -> Evaluation:
        return Evaluation(candidate=candidate, outcome="pass", goals_hit=Goals())

    async def enumerate_goals(self, target: Target) -> Goals:
        return Goals()  # the target already knows its goals
```

The `Strategy` Protocol is where new applications usually plug in (different
prompts, different scoring). The `Evaluator` Protocol is where you change
how candidates get executed (different sandbox, different signal).

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.11+
- `ANTHROPIC_API_KEY` for non-trivial runs (the SDK shells out to the
  Claude Code CLI under the hood)
- Docker (optional — only for `--evaluator docker`)

## Install

```bash
uv sync
uv pip install -e .
cp .env.example .env  # then add your ANTHROPIC_API_KEY
```

## CLI

```bash
# Smoke test (local subprocess evaluator, no Docker required)
uv run xsmith explore --benchmark repo_explore --targets 1 --budget 5 \
  --output results/run.json

# Larger run, Docker sandbox
docker build -f Dockerfile.runner -t xsmith-runner:latest .
uv run xsmith explore --benchmark repo_explore --targets 10 --budget 24 \
  --evaluator docker --output results/full.json
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
  domain/        # Pydantic types: Target, Progress, Goal/Goals, Candidate, Evaluation, Budget
  agents/        # ClaudeSDK agents + isolation + MCP tools
  strategies/    # Strategy Protocol + QValueStrategy
  execution/     # Evaluator Protocol + Subprocess and Docker evaluators
  exploration/   # Explorer (depends only on the two Protocols)
  benchmarks/    # RepoExploreBench, TestGenEvalBench
  results/       # JSON schema + writer
  cli.py         # `xsmith explore` Typer entry point
  config.py      # pydantic-settings, env-loaded
```

`agents/base.py` is the single place `claude-agent-sdk` is imported —
replacing the SDK later means rewriting that one file.
