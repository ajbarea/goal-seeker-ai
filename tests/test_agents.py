"""Tabular Q-learning and DQN agents, run on CPU without Webots."""

import random

import pytest
import torch

from common.config import RLConfig
from controllers.slave.dqn_agent import DQNAgent
from controllers.slave.q_learning_agent import QLearningAgent

STATE = (2, 1, 0, 0, 1)
NEXT_STATE = (1, 1, 0, 0, 1)


@pytest.fixture
def q_agent():
    return QLearningAgent()


@pytest.fixture
def dqn_agent():
    torch.manual_seed(0)
    random.seed(0)
    return DQNAgent(state_dim=5, action_dim=5, device="cpu")


def test_the_agents_save_to_different_files():
    paths = {QLearningAgent.model_path, QLearningAgent.best_model_path}
    paths |= {DQNAgent.model_path, DQNAgent.best_model_path}
    assert len(paths) == 4
    assert QLearningAgent.model_path.endswith(".pkl")
    assert DQNAgent.model_path.endswith(".pt")


def test_q_update_moves_the_value_towards_the_reward(q_agent):
    q_agent.update_q_table(STATE, 0, 10.0, NEXT_STATE)
    assert q_agent.q_table[STATE][0] > 0.0
    assert q_agent.q_table[STATE][1:] == [0.0] * 4


def test_q_table_round_trips(q_agent, tmp_path):
    q_agent.update_q_table(STATE, 2, 5.0, NEXT_STATE)
    path = tmp_path / "q.pkl"
    assert q_agent.save_q_table(str(path))
    other = QLearningAgent()
    assert other.load_q_table(str(path))
    assert other.q_table == q_agent.q_table


def test_dqn_trains_once_the_buffer_holds_a_batch(dqn_agent):
    for _ in range(RLConfig.BATCH_SIZE - 1):
        dqn_agent.update_q_table(STATE, 0, 1.0, NEXT_STATE)
    assert dqn_agent.td_losses == []
    dqn_agent.update_q_table(STATE, 0, 1.0, NEXT_STATE)
    assert len(dqn_agent.td_losses) == 1


def test_dqn_learns_which_action_pays(dqn_agent):
    for _ in range(600):
        action = random.randrange(5)
        reward = 1.0 if action == 3 else -1.0
        dqn_agent.update_q_table(STATE, action, reward, NEXT_STATE)
    dqn_agent.set_eval_mode(True)
    assert dqn_agent.choose_action(STATE) == 3


def test_dqn_eval_mode_is_greedy(dqn_agent):
    dqn_agent.set_eval_mode(True)
    steps = dqn_agent.steps
    picks = {dqn_agent.choose_action(STATE) for _ in range(20)}
    assert len(picks) == 1
    assert dqn_agent.steps == steps  # epsilon schedule untouched


def test_dqn_checkpoint_round_trips(dqn_agent, tmp_path):
    for _ in range(RLConfig.BATCH_SIZE):
        dqn_agent.update_q_table(STATE, 1, 2.0, NEXT_STATE)
    path = tmp_path / "dqn.pt"
    assert dqn_agent.save_q_table(str(path))

    other = DQNAgent(state_dim=5, action_dim=5, device="cpu")
    assert other.load_q_table(str(path))
    x = torch.tensor([STATE], dtype=torch.float32)
    assert torch.equal(other.online(x), dqn_agent.online(x))
    assert other.steps == dqn_agent.steps


def test_q_learning_cannot_load_a_dqn_checkpoint(dqn_agent, tmp_path):
    """Why the files must differ: each format is unreadable to the other agent."""
    path = tmp_path / "model"
    dqn_agent.save_q_table(str(path))
    assert QLearningAgent().load_q_table(str(path)) is False
