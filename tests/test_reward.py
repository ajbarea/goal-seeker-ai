"""Reward function: the progress reward and Ivan's potential-based shaping."""

import pytest

from common.config import RLConfig
from common.rl_utils import calculate_distance, calculate_potential, calculate_reward

TARGET = [1.0, 0.0]
GAMMA = 0.99


def test_potential_is_negative_distance_to_target():
    assert calculate_potential([0.0, 0.0], TARGET) == -1.0
    assert calculate_potential([1.0, 0.0, 0.5], TARGET) == 0.0  # z is ignored


def test_reaching_the_target_returns_the_target_reward():
    reward = calculate_reward(
        [1.0, 0.05],
        TARGET,
        previous_distance=0.3,
        target_threshold=0.1,
        previous_position=[0.7, 0.0],
        discount_factor=GAMMA,
        use_potential_shaping=True,
    )
    assert reward == RLConfig.TARGET_REACHED_REWARD


def test_moving_closer_beats_moving_away():
    closer = calculate_reward([0.5, 0.0], TARGET, previous_distance=0.6)
    away = calculate_reward([0.3, 0.0], TARGET, previous_distance=0.6)
    assert closer > away


@pytest.mark.parametrize(
    ("previous", "current"),
    [([0.0, 0.0], [0.2, 0.0]), ([0.5, 0.0], [0.3, 0.0]), ([0.2, 0.2], [0.2, 0.2])],
)
def test_shaping_adds_gamma_phi_next_minus_phi(previous, current):
    prev_d = calculate_distance(previous, TARGET)
    base = calculate_reward(current, TARGET, previous_distance=prev_d)
    shaped = calculate_reward(
        current,
        TARGET,
        previous_distance=prev_d,
        previous_position=previous,
        discount_factor=GAMMA,
        use_potential_shaping=True,
    )
    expected_f = GAMMA * calculate_potential(current, TARGET) - calculate_potential(
        previous, TARGET
    )
    assert shaped == pytest.approx(base + expected_f)


def test_shaping_is_off_without_its_inputs():
    base = calculate_reward([0.5, 0.0], TARGET, previous_distance=0.6)
    no_position = calculate_reward(
        [0.5, 0.0],
        TARGET,
        previous_distance=0.6,
        discount_factor=GAMMA,
        use_potential_shaping=True,
    )
    no_gamma = calculate_reward(
        [0.5, 0.0],
        TARGET,
        previous_distance=0.6,
        previous_position=[0.4, 0.0],
        use_potential_shaping=True,
    )
    flag_off = calculate_reward(
        [0.5, 0.0],
        TARGET,
        previous_distance=0.6,
        previous_position=[0.4, 0.0],
        discount_factor=GAMMA,
    )
    assert base == no_position == no_gamma == flag_off


def test_collision_is_penalised():
    clear = calculate_reward(
        [0.5, 0.0], TARGET, previous_distance=0.6, left_sensor=0, right_sensor=0
    )
    hit = calculate_reward(
        [0.5, 0.0], TARGET, previous_distance=0.6, left_sensor=1000, right_sensor=0
    )
    assert clear - hit == pytest.approx(RLConfig.COLLISION_PENALTY)
