import random
import collections
from typing import List, Tuple, Deque, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from common.config import RLConfig
from q_learning_agent import QLearningAgent
from common.rl_utils import get_discrete_state as util_get_discrete_state

StateType = Tuple[int, int, int, int, int]
BatchType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _state_to_tensor(state: StateType, device: torch.device) -> torch.Tensor:
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = RLConfig.BUFFER_SIZE):
        self.buffer: Deque[Tuple[StateType, int, float, StateType, bool]] = (
            collections.deque(maxlen=capacity)
        )

    def push(self, s: StateType, a: int, r: float, s2: StateType, d: bool) -> None:
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size: int = RLConfig.BATCH_SIZE) -> BatchType:
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN agent wrapper that mimics QLearningAgent interface for drop-in use."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        max_speed: float = 10.0,
    ):
        self.device = torch.device(device)
        self.online = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=RLConfig.LR)
        self.buffer = ReplayBuffer()
        self.q_table = self.buffer.buffer  # alias for compatibility
        self.steps = 0
        self.exec_agent = QLearningAgent(
            learning_rate=RLConfig.LEARNING_RATE,
            min_learning_rate=RLConfig.MIN_LEARNING_RATE,
            discount_factor=RLConfig.DISCOUNT_FACTOR,
            min_discount_factor=RLConfig.MIN_DISCOUNT_FACTOR,
            exploration_rate=RLConfig.EXPLORATION_RATE,
            max_speed=max_speed,
        )

    def update_q_table(
        self, state: StateType, action: int, reward: float, next_state: StateType
    ) -> None:
        if state is None or next_state is None:
            return
        self.buffer.push(state, action, reward, next_state, False)
        self.optimize()

    def choose_action(
        self, state: StateType, current_distance: Optional[float] = None
    ) -> int:
        return self.select_action(state)

    def choose_best_action(
        self, state: StateType, current_distance: Optional[float] = None
    ) -> int:
        with torch.no_grad():
            x = _state_to_tensor(state, self.device)
            return int(self.online(x).argmax())

    def execute_action(
        self, action: int, state: Optional[StateType] = None
    ) -> List[float]:
        return self.exec_agent.execute_action(action, state)

    def save_q_table(self, filepath: str) -> bool:
        # TODO: Implement persistent storage for saving the Q-table.
        return True

    def load_q_table(self, filepath: str) -> bool:
        # TODO: Implement persistent storage for loading the Q-table.
        return True

    def get_discrete_state(
        self,
        position: List[float],
        target_position: List[float],
        orientation: float,
        left_sensor: float,
        right_sensor: float,
        wheel_velocities: List[float],
    ) -> Optional[StateType]:
        return util_get_discrete_state(
            position,
            target_position,
            orientation,
            left_sensor,
            right_sensor,
            wheel_velocities,
        )

    def select_action(self, state: StateType) -> int:
        eps = RLConfig.EPS_END + (RLConfig.EPS_START - RLConfig.EPS_END) * max(
            0, (1 - self.steps / RLConfig.EPS_DECAY)
        )
        self.steps += 1
        if random.random() < eps:
            return random.randrange(self.online.net[-1].out_features)
        with torch.no_grad():
            x = _state_to_tensor(state, self.device)
            return int(self.online(x).argmax())

    def optimize(self) -> None:
        if len(self.buffer) < RLConfig.BATCH_SIZE:
            return
        s, a, r, s2, d = self.buffer.sample()
        s, a, r, s2, d = [t.to(self.device) for t in (s, a, r, s2, d)]
        q_values = self.online(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            tgt_q = r + RLConfig.GAMMA * self.target(s2).max(1)[0] * (1 - d)
        loss = nn.functional.mse_loss(q_values, tgt_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.steps % RLConfig.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())
