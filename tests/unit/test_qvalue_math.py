import pytest

from xsmith.agents.scorer import QScore


def test_q_value_default_gamma():
    s = QScore(immediate_goals=3, future_value=4)
    assert s.gamma == 0.5
    assert s.q == 3 + 0.5 * 4


def test_q_value_custom_gamma():
    s = QScore(immediate_goals=2, future_value=10, gamma=0.9)
    assert s.q == pytest.approx(2 + 9.0)


def test_q_value_bounds_validated():
    with pytest.raises(Exception):
        QScore(immediate_goals=-1, future_value=0)
    with pytest.raises(Exception):
        QScore(immediate_goals=0, future_value=11)
    with pytest.raises(Exception):
        QScore(immediate_goals=0, future_value=-1)


def test_q_zero_components():
    s = QScore(immediate_goals=0, future_value=0)
    assert s.q == 0.0
