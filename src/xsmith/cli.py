"""`xsmith explore` — single CLI entry point."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated

import typer

from xsmith.benchmarks.repo_explore import RepoExploreBench
from xsmith.benchmarks.testgeneval import TestGenEvalBench
from xsmith.config import Settings, load_settings
from xsmith.domain.budget import Budget
from xsmith.domain.coverage import CoverageMap
from xsmith.execution.docker_runner import DockerTestRunner
from xsmith.execution.runner import TestRunner
from xsmith.execution.subprocess_runner import SubprocessRunner
from xsmith.exploration.explorer import ExplorationLoop
from xsmith.results.schema import RunResult
from xsmith.results.writer import to_target_result, write_run
from xsmith.strategies.cov_qvalue import CovQValueStrategy

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _root() -> None:
    """xsmith — curiosity-driven test generation."""


def _make_runner(runner_kind: str, settings: Settings) -> TestRunner:
    if runner_kind == "subprocess":
        return SubprocessRunner(timeout_s=settings.SUBPROCESS_TIMEOUT_S)
    if runner_kind == "docker":
        return DockerTestRunner(
            image=settings.DOCKER_IMAGE,
            timeout_s=settings.DOCKER_TIMEOUT_S,
        )
    raise typer.BadParameter(f"unknown runner: {runner_kind!r}")


def _make_benchmark(name: str):
    if name == "repo_explore":
        return RepoExploreBench()
    if name == "testgeneval":
        return TestGenEvalBench()
    raise typer.BadParameter(f"unknown benchmark: {name!r}")


@app.command()
def explore(
    benchmark: Annotated[
        str, typer.Option("--benchmark", "-b", help="repo_explore | testgeneval")
    ] = "repo_explore",
    targets: Annotated[int, typer.Option("--targets", "-t", help="Max targets to explore")] = 1,
    budget: Annotated[int | None, typer.Option("--budget", help="Per-target execution budget")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="Output JSON path")] = "results/run.json",
    runner: Annotated[
        str, typer.Option("--runner", help="subprocess (local) | docker")
    ] = "subprocess",
    model: Annotated[str | None, typer.Option("--model", help="Override model")] = None,
    k: Annotated[int | None, typer.Option("--k", help="Override K candidates per iteration")] = None,
    gamma: Annotated[float | None, typer.Option("--gamma", help="Override Q-value gamma")] = None,
    max_usd: Annotated[float | None, typer.Option("--max-usd", help="Stop a target if cost exceeds this")] = None,
) -> None:
    """Run curiosity-driven test exploration on a benchmark."""
    settings = load_settings()
    if not settings.ANTHROPIC_API_KEY:
        typer.echo("[warning] ANTHROPIC_API_KEY not set; the SDK will likely fail.", err=True)

    # Surface the key into the env so the SDK's underlying CLI sees it.
    if settings.ANTHROPIC_API_KEY and "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY

    eff_model = model or settings.MODEL
    eff_k = k or settings.K
    eff_gamma = gamma if gamma is not None else settings.GAMMA
    eff_budget = budget if budget is not None else settings.EXEC_BUDGET

    bench = _make_benchmark(benchmark)
    test_runner = _make_runner(runner, settings)
    strategy = CovQValueStrategy(
        model=eff_model,
        k=eff_k,
        gamma=eff_gamma,
        max_turns_gen=settings.MAX_TURNS_GEN,
        max_turns_score=settings.MAX_TURNS_SCORE,
    )

    typer.echo(
        f"[xsmith] benchmark={benchmark} runner={runner} model={eff_model} "
        f"K={eff_k} gamma={eff_gamma} budget={eff_budget} targets={targets}"
    )

    asyncio.run(
        _run(
            bench=bench,
            test_runner=test_runner,
            strategy=strategy,
            targets=targets,
            exec_budget=eff_budget,
            max_usd=max_usd if max_usd is not None else settings.MAX_USD,
            output=output,
            benchmark_name=benchmark,
            model=eff_model,
            k=eff_k,
            gamma=eff_gamma,
        )
    )


async def _run(
    *,
    bench,
    test_runner: TestRunner,
    strategy: CovQValueStrategy,
    targets: int,
    exec_budget: int,
    max_usd: float | None,
    output: str,
    benchmark_name: str,
    model: str,
    k: int,
    gamma: float,
) -> None:
    target_list = bench.load(max_targets=targets)
    typer.echo(f"[xsmith] loaded {len(target_list)} target(s)")

    run = RunResult(
        benchmark=benchmark_name,
        model=model,
        k=k,
        gamma=gamma,
        exec_budget=exec_budget,
    )

    for tgt in target_list:
        typer.echo(f"[xsmith] target {tgt.target_id} — discovering branches…")
        universe = await test_runner.discover_branches(tgt)
        tgt.branches = universe
        typer.echo(f"[xsmith] target {tgt.target_id} — {len(universe)} branches total")

        budget_obj = Budget(
            exec_remaining=exec_budget,
            enforce_cost=max_usd is not None,
            max_usd=max_usd,
        )
        coverage = CoverageMap(total=universe)
        loop = ExplorationLoop(strategy=strategy, runner=test_runner)

        def _print_progress(rec):
            typer.echo(
                f"  iter {rec.iteration:>2}: outcome={rec.run_result.outcome} "
                f"new={len(rec.new_branches)} cov={rec.coverage_after}/{rec.coverage_total} "
                f"cost=${rec.agent_usage.cost_usd:.4f}"
            )

        loop.on_iteration = _print_progress  # type: ignore[assignment]

        out = await loop.explore(target=tgt, budget=budget_obj, initial_coverage=coverage)
        run.targets.append(to_target_result(out))
        typer.echo(
            f"[xsmith] target {tgt.target_id} done — "
            f"covered {out.covered_count}/{out.total_count} "
            f"({out.coverage_fraction:.1%})"
        )

    write_run(run, output)
    typer.echo(f"[xsmith] wrote {output} — total_cost_usd=${run.total_cost_usd:.4f}")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
