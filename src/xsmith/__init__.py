"""xsmith — generic exploration framework."""

__version__ = "0.2.0"

from xsmith.agents.base import AgentUsage
from xsmith.benchmarks.base import Benchmark
from xsmith.domain.budget import Budget
from xsmith.domain.candidate import Candidate, Outcome
from xsmith.domain.evaluation import Evaluation
from xsmith.domain.goal import Goal, Goals
from xsmith.domain.progress import Progress
from xsmith.domain.target import Target
from xsmith.execution.docker import DockerEvaluator
from xsmith.execution.evaluator import Evaluator
from xsmith.execution.subprocess import SubprocessEvaluator
from xsmith.exploration.explorer import (
    ExplorationResult,
    Explorer,
    Step,
    explore,
)
from xsmith.strategies.base import Strategy
from xsmith.strategies.qvalue import QValueStrategy

__all__ = [
    "Explorer",
    "explore",
    "ExplorationResult",
    "Step",
    "Target",
    "Goal",
    "Goals",
    "Progress",
    "Candidate",
    "Evaluation",
    "Outcome",
    "Budget",
    "Evaluator",
    "SubprocessEvaluator",
    "DockerEvaluator",
    "Strategy",
    "QValueStrategy",
    "Benchmark",
    "AgentUsage",
]
