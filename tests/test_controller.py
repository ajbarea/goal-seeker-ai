"""Supervisor-side training controller, driven by a stand-in for the Webots driver."""

import logging

import pytest

from common.config import RLConfig
from common.rl_utils import calculate_distance
from controllers.driver.q_learning_controller import QLearningController


class FakeEmitter:
    def __init__(self):
        self.sent = []

    def send(self, data: bytes) -> None:
        self.sent.append(data.decode("utf-8"))


class FakeDriver:
    def __init__(self):
        self.emitter = FakeEmitter()
        self.target_position = [0.0, 0.0]
        self.robot_position = None

    def set_target_position(self, position):
        self.target_position = list(position[:2])

    def reset_robot_position(self, position):
        self.robot_position = list(position)

    def step(self, _time_step):
        return 0


@pytest.fixture
def controller():
    return QLearningController(FakeDriver(), logging.getLogger("test"))


def test_new_episode_starts_shaping_from_the_start_position(controller):
    controller.start_new_episode()
    start = controller.start_positions[controller.current_start_index]
    assert controller.previous_position == start[:2]

    # Wander far away, then begin the next episode.
    controller.calculate_reward([5.0, 5.0, 0.0])
    controller.start_new_episode()
    start = controller.start_positions[controller.current_start_index]
    assert controller.previous_position == start[:2]


def test_first_step_after_reset_has_no_teleport_bonus(controller, monkeypatch):
    """Step one's shaping measures the move from the start, not from where the
    previous episode ended."""
    monkeypatch.setattr(RLConfig, "USE_POTENTIAL_SHAPING", True)
    controller.start_new_episode()
    controller.calculate_reward([5.0, 5.0, 0.0])  # end of a bad episode
    controller.start_new_episode()

    start = controller.start_positions[controller.current_start_index][:2]
    target = controller.driver.target_position
    reward = controller.calculate_reward(start + [0.0])  # robot has not moved

    controller.previous_position = start
    controller.previous_distance_to_target = calculate_distance(start, target)
    same_without_history = controller.calculate_reward(start + [0.0])
    assert reward == pytest.approx(same_without_history)


def test_shaping_uses_the_dqn_gamma_when_the_dqn_trains(controller, monkeypatch):
    monkeypatch.setattr(RLConfig, "USE_POTENTIAL_SHAPING", True)
    monkeypatch.setattr(RLConfig, "GAMMA", 0.5)
    monkeypatch.setattr(RLConfig, "DISCOUNT_FACTOR", 0.9)

    def reward_with(using_dqn: bool) -> float:
        controller.using_dqn = using_dqn
        controller.driver.target_position = [1.0, 0.0]
        controller.previous_position = [0.0, 0.0]
        controller.previous_distance_to_target = 1.0
        return controller.calculate_reward([0.5, 0.0, 0.0])

    tabular, dqn = reward_with(False), reward_with(True)
    # Only gamma differs: F = gamma * (-0.5) + 1.0
    assert tabular - dqn == pytest.approx((0.9 - 0.5) * -0.5)


def test_training_step_sends_the_reward_to_the_robot(controller):
    controller.start_learning("dqn")
    assert "training_mode:dqn" in controller.driver.emitter.sent
    start = controller.start_positions[controller.current_start_index]
    controller.manage_training_step(list(start[:2]) + [0.0])
    assert any(m.startswith("reward:") for m in controller.driver.emitter.sent)


class FakeField:
    def __init__(self, position):
        self.position = list(position)

    def getSFVec3f(self):
        return list(self.position)

    def setSFVec3f(self, value):
        self.position = list(value)


def test_evaluation_episodes_start_shaping_from_the_start_position(
    controller, monkeypatch
):
    import controllers.driver.q_learning_controller as qlc

    seen = []
    real = qlc.calculate_reward

    def spy(*args, **kwargs):
        seen.append(kwargs.get("previous_position"))
        return real(*args, **kwargs)

    monkeypatch.setattr(qlc, "calculate_reward", spy)
    target = controller.target_positions[controller.current_target_index]
    controller.driver.translation_field = FakeField(list(target[:2]) + [0.0])
    controller.previous_position = [9.0, 9.0]  # left over from training

    controller.run_evaluation_episodes()

    start = controller.start_positions[controller.current_start_index]
    assert seen, "evaluation computed no reward"
    assert all(p == start[:2] for p in seen)
