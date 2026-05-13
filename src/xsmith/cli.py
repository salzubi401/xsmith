"""`xsmith explore` — single CLI entry point."""

from __future__ import annotations

import asyncio
import os
from typing import Annotated

import typer

from xsmith.benchmarks.repo_explore import RepoExploreBench
from xsmith.benchmarks.testgeneval import TestGenEvalBench
from xsmith.config import Settings, load_settings
from xsmith.domain.budget import Budget
from xsmith.domain.progress import Progress
from xsmith.execution.docker import DockerEvaluator
from xsmith.execution.evaluator import Evaluator
from xsmith.execution.subprocess import SubprocessEvaluator
from xsmith.exploration.explorer import Explorer
from xsmith.results.schema import RunResult
from xsmith.results.writer import to_target_result, write_run
from xsmith.strategies.qvalue import QValueStrategy

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _root() -> None:
    """xsmith — curiosity-driven exploration framework."""


def _make_evaluator(evaluator_kind: str, settings: Settings) -> Evaluator:
    if evaluator_kind == "subprocess":
        return SubprocessEvaluator(timeout_s=settings.SUBPROCESS_TIMEOUT_S)
    if evaluator_kind == "docker":
        return DockerEvaluator(
            image=settings.DOCKER_IMAGE,
            timeout_s=settings.DOCKER_TIMEOUT_S,
        )
    raise typer.BadParameter(f"unknown evaluator: {evaluator_kind!r}")


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
    budget: Annotated[int | None, typer.Option("--budget", help="Per-target step budget")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="Output JSON path")] = "results/run.json",
    evaluator: Annotated[
        str, typer.Option("--evaluator", help="subprocess (local) | docker")
    ] = "subprocess",
    model: Annotated[str | None, typer.Option("--model", help="Override model")] = None,
    k: Annotated[int | None, typer.Option("--k", help="Override K candidates per iteration")] = None,
    gamma: Annotated[float | None, typer.Option("--gamma", help="Override Q-value gamma")] = None,
    max_usd: Annotated[float | None, typer.Option("--max-usd", help="Stop a target if cost exceeds this")] = None,
) -> None:
    """Run curiosity-driven exploration on a benchmark."""
    settings = load_settings()
    if not settings.ANTHROPIC_API_KEY:
        typer.echo("[warning] ANTHROPIC_API_KEY not set; the SDK will likely fail.", err=True)

    # Surface the key into the env so the SDK's underlying CLI sees it.
    if settings.ANTHROPIC_API_KEY and "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY

    eff_model = model or settings.MODEL
    eff_k = k or settings.K
    eff_gamma = gamma if gamma is not None else settings.GAMMA
    eff_budget = budget if budget is not None else settings.STEP_BUDGET

    bench = _make_benchmark(benchmark)
    ev = _make_evaluator(evaluator, settings)
    strategy = QValueStrategy(
        model=eff_model,
        k=eff_k,
        gamma=eff_gamma,
        max_turns_gen=settings.MAX_TURNS_GEN,
        max_turns_score=settings.MAX_TURNS_SCORE,
    )

    typer.echo(
        f"[xsmith] benchmark={benchmark} evaluator={evaluator} model={eff_model} "
        f"K={eff_k} gamma={eff_gamma} budget={eff_budget} targets={targets}"
    )

    asyncio.run(
        _run(
            bench=bench,
            evaluator=ev,
            strategy=strategy,
            targets=targets,
            step_budget=eff_budget,
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
    evaluator: Evaluator,
    strategy: QValueStrategy,
    targets: int,
    step_budget: int,
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
        step_budget=step_budget,
    )

    for tgt in target_list:
        typer.echo(f"[xsmith] target {tgt.target_id} — enumerating goals…")
        universe = await evaluator.enumerate_goals(tgt)
        tgt.goals = universe
        typer.echo(f"[xsmith] target {tgt.target_id} — {len(universe)} goals total")

        budget_obj = Budget(
            steps=step_budget,
            enforce_cost=max_usd is not None,
            max_usd=max_usd,
        )
        progress = Progress(all=universe)
        explorer = Explorer(strategy=strategy, evaluator=evaluator)

        def _print_progress(step):
            typer.echo(
                f"  step {step.iteration:>2}: outcome={step.evaluation.outcome} "
                f"new={len(step.new_goals)} hit={step.hit_after}/{step.total} "
                f"cost=${step.agent_usage.cost_usd:.4f}"
            )

        explorer.on_step = _print_progress  # type: ignore[assignment]

        out = await explorer.run(target=tgt, budget=budget_obj, initial_progress=progress)
        run.targets.append(to_target_result(out))
        typer.echo(
            f"[xsmith] target {tgt.target_id} done — "
            f"hit {out.hit_count}/{out.total_count} "
            f"({out.fraction:.1%})"
        )

    write_run(run, output)
    typer.echo(f"[xsmith] wrote {output} — total_cost_usd=${run.total_cost_usd:.4f}")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
