"""Tests for CovQValueStrategy with a StubAgentRunner — no real LLM calls.

The stub replaces `AgentRunner.run` so the generator/scorer agents return
canned tool args. We test:
  - K parallel candidates are gathered
  - argmax picks the highest Q score
  - tiebreakers behave as documented
  - per-iteration AgentUsage is aggregated
  - all-failure case returns an empty TestCase
"""

from __future__ import annotations

import pytest

from xsmith.agents.base import AgentRunner, AgentUsage
from xsmith.agents.scorer import QValueScorerAgent
from xsmith.agents.test_generator import TestGeneratorAgent as XTestGeneratorAgent
from xsmith.domain.coverage import Branch, BranchSet, CoverageMap
from xsmith.domain.target import Target
from xsmith.strategies.cov_qvalue import CovQValueStrategy


def _make_target():
    return Target(
        target_id="t",
        module_path="x.y",
        source="def f(): pass",
        branches=BranchSet.from_iterable([Branch(file="a.py", src=1, dst=2)]),
    )


class CannedRunner(AgentRunner):
    """Returns scripted submissions. `fixed_idx` (if set) overrides the counter,
    so we can identify which generator a call came from (each generator owns its
    own runner). Without `fixed_idx`, falls back to a per-runner counter
    (useful for the scorer, which is created once and called K times)."""

    def __init__(self, submit_tool_name: str, plan, fixed_idx: int | None = None):
        super().__init__(submit_tool_name=submit_tool_name)
        self.plan = plan
        self.fixed_idx = fixed_idx
        self.calls = 0

    async def run(self, *, options, user_prompt):
        idx = self.fixed_idx if self.fixed_idx is not None else self.calls
        self.calls += 1
        return self.plan(idx)


@pytest.mark.asyncio
async def test_argmax_selects_highest_q(monkeypatch):
    """5 generators submit. Scorer assigns Q=[0, 4, 2, 6, 1] → pick idx=3."""
    target = _make_target()
    coverage = CoverageMap(total=target.branches)

    def gen_plan(idx):
        return (
            {"code": f"def test_x_{idx}(): pass", "rationale": f"r{idx}"},
            AgentUsage(tokens_in=10, tokens_out=5, cost_usd=0.001),
        )

    score_table = [(0, 0), (4, 0), (2, 0), (6, 0), (1, 0)]

    def score_plan(idx):
        imm, fut = score_table[idx]
        return (
            {"immediate_branches": imm, "future_value": fut},
            AgentUsage(tokens_in=20, tokens_out=10, cost_usd=0.002),
        )

    def fake_gen_init(self, *, variant_idx, model, runner=None, max_turns=8):
        self.variant_idx = variant_idx
        self.model = model
        self.runner = CannedRunner("submit_test", gen_plan, fixed_idx=variant_idx)
        self.max_turns = max_turns

    monkeypatch.setattr(XTestGeneratorAgent, "__init__", fake_gen_init)

    def fake_score_init(self, *, runner=None, model, max_turns=3, gamma=0.5):
        self.runner = CannedRunner("submit_score", score_plan)
        self.model = model
        self.max_turns = max_turns
        self.gamma = gamma

    monkeypatch.setattr(QValueScorerAgent, "__init__", fake_score_init)

    strategy = CovQValueStrategy(model="claude-sonnet-4-6", k=5, gamma=0.5)
    test_case, usage = await strategy.propose(target=target, coverage=coverage, history=[])

    # Winner is idx=3 (Q=6)
    assert "test_x_3" in test_case.code
    assert "Q=6.00" in test_case.rationale
    # Usage = K gens + 5 scorers = 5*(.001) + 5*(.002) = .015
    assert usage.cost_usd == pytest.approx(5 * 0.001 + 5 * 0.002)


@pytest.mark.asyncio
async def test_tiebreaker_prefers_immediate(monkeypatch):
    """Two candidates have Q=4 (4+0 vs 0+8). Higher immediate wins."""
    target = _make_target()
    coverage = CoverageMap(total=target.branches)

    def gen_plan(idx):
        return (
            {"code": f"def test_t_{idx}(): pass", "rationale": ""},
            AgentUsage(),
        )

    # idx 0: imm=4 fut=0  → Q=4, imm=4
    # idx 1: imm=0 fut=8  → Q=4, imm=0
    score_table = [(4, 0), (0, 8), (0, 0), (0, 0), (0, 0)]

    def score_plan(idx):
        imm, fut = score_table[idx]
        return ({"immediate_branches": imm, "future_value": fut}, AgentUsage())

    def fake_gen_init(self, *, variant_idx, model, runner=None, max_turns=8):
        self.variant_idx = variant_idx
        self.model = model
        self.runner = CannedRunner("submit_test", gen_plan, fixed_idx=variant_idx)
        self.max_turns = max_turns

    def fake_score_init(self, *, runner=None, model, max_turns=3, gamma=0.5):
        self.runner = CannedRunner("submit_score", score_plan)
        self.model = model
        self.max_turns = max_turns
        self.gamma = gamma

    monkeypatch.setattr(XTestGeneratorAgent, "__init__", fake_gen_init)
    monkeypatch.setattr(QValueScorerAgent, "__init__", fake_score_init)

    strategy = CovQValueStrategy(model="claude-sonnet-4-6", k=5, gamma=0.5)
    tc, _ = await strategy.propose(target=target, coverage=coverage, history=[])
    assert "test_t_0" in tc.code
    assert "imm=4" in tc.rationale


@pytest.mark.asyncio
async def test_all_generators_fail_returns_empty(monkeypatch):
    target = _make_target()
    coverage = CoverageMap(total=target.branches)

    def gen_plan(idx):
        return (None, AgentUsage())  # no submission

    def fake_gen_init(self, *, variant_idx, model, runner=None, max_turns=8):
        self.variant_idx = variant_idx
        self.model = model
        self.runner = CannedRunner("submit_test", gen_plan, fixed_idx=variant_idx)
        self.max_turns = max_turns

    monkeypatch.setattr(XTestGeneratorAgent, "__init__", fake_gen_init)

    strategy = CovQValueStrategy(model="claude-sonnet-4-6", k=3, gamma=0.5)
    tc, _ = await strategy.propose(target=target, coverage=coverage, history=[])
    assert tc.code == ""
    assert "failed" in tc.rationale.lower()
