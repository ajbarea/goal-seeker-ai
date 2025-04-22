"""Q-learning agent: state discretization, action selection, and Q-table updates."""

import random
import pickle
import os
from typing import Dict, List, Tuple, Optional
from common.rl_utils import get_discrete_state
from common.config import RLConfig, get_logger

# Set up logger
logger = get_logger(__name__)


class QLearningAgent:
    """Manage Q-learning logic and table."""

    # Constants for action indices
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    BACKWARD = 3
    STOP = 4

    def __init__(
        self,
        learning_rate: float = 0.1,
        min_learning_rate: float = 0.03,
        discount_factor: float = 0.9,
        min_discount_factor: float = 0.7,
        exploration_rate: float = 0.3,
        max_speed: float = 10.0,
        angle_bins: int = 8,
    ):
        """Init agent params and load Q-table."""
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        self.min_discount_factor = min_discount_factor
        self.exploration_rate = exploration_rate
        self.max_speed = max_speed
        self.angle_bins = angle_bins

        self.q_table: Dict[Tuple, List[float]] = {}
        self.total_updates = 0

        self.td_errors: List[float] = []
        self.learning_rates: List[float] = []
        self.discount_factors: List[float] = []

        try:
            self.load_q_table(RLConfig.Q_TABLE_PATH)
        except Exception as e:
            logger.warning(f"Could not load Q-table: {e}")

    def get_discrete_state(
        self,
        position: List[float],
        target_position: List[float],
        orientation: float,
        left_sensor: float,
        right_sensor: float,
        wheel_velocities: List[float],
    ) -> Optional[Tuple]:
        """Wrap get_discrete_state from rl_utils."""
        return get_discrete_state(
            position,
            target_position,
            orientation,
            left_sensor,
            right_sensor,
            wheel_velocities,
            self.angle_bins,
        )

    def choose_action(self, state: Tuple, current_distance: float = None) -> int:
        """Epsilon-greedy action choice."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5

        allow_stop = (
            current_distance is not None
            and current_distance <= RLConfig.TARGET_THRESHOLD
        )
        action_indices = [0, 1, 2, 3]
        if allow_stop:
            action_indices.append(4)

        if random.random() < self.exploration_rate:
            if random.random() < 0.5 and 0 in action_indices:
                return 0
            return random.choice(action_indices)

        q_values = self.q_table[state]
        filtered_q = [(i, q_values[i]) for i in action_indices]
        max_q_value = max(q for i, q in filtered_q)
        best_actions = [i for i, q in filtered_q if q == max_q_value]
        return random.choice(best_actions)

    def choose_best_action(self, state: Tuple, current_distance: float = None) -> int:
        """Greedy action choice without exploration."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5

            distance_bin, angle_bin, left_obstacle, right_obstacle, is_moving = state

            if left_obstacle and right_obstacle:
                return self.BACKWARD
            elif left_obstacle:
                return self.TURN_RIGHT
            elif right_obstacle:
                return self.TURN_LEFT
            elif angle_bin < self.angle_bins // 2:
                return self.TURN_RIGHT
            else:
                return self.TURN_LEFT

        allow_stop = (
            current_distance is not None
            and current_distance <= RLConfig.TARGET_THRESHOLD
        )
        action_indices = [0, 1, 2, 3]
        if allow_stop:
            action_indices.append(4)

        q_values = self.q_table[state]
        filtered_q = [(i, q_values[i]) for i in action_indices]
        max_q_value = max(q for i, q in filtered_q)
        best_actions = [i for i, q in filtered_q if q == max_q_value]
        return random.choice(best_actions)

    def update_q_table(
        self, state: Tuple, action: int, reward: float, next_state: Tuple
    ) -> None:
        """Apply Q-learning update to the table."""
        if state is None or next_state is None:
            return

        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 5

        self.total_updates += 1

        adaptive_learning_rate = max(
            self.min_learning_rate,
            self.learning_rate
            * (
                RLConfig.LEARNING_RATE_DECAY_BASE
                ** (self.total_updates / RLConfig.LEARNING_RATE_DECAY_DENOM)
            ),
        )

        adaptive_discount = max(
            self.min_discount_factor,
            self.discount_factor * (0.9995 ** (self.total_updates / 5000)),
        )

        self.learning_rates.append(adaptive_learning_rate)
        self.discount_factors.append(adaptive_discount)

        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])

        td_error = reward + adaptive_discount * next_max_q - current_q
        self.td_errors.append(td_error)

        new_q = current_q + adaptive_learning_rate * td_error
        self.q_table[state][action] = max(-50.0, min(50.0, new_q))

    def execute_action(self, action: int) -> List[float]:
        """Return motor speeds for given action."""
        if action == self.FORWARD:
            return [self.max_speed, self.max_speed]
        elif action == self.TURN_LEFT:
            return [self.max_speed / 2, -self.max_speed / 2]
        elif action == self.TURN_RIGHT:
            return [-self.max_speed / 2, self.max_speed / 2]
        elif action == self.BACKWARD:
            return [-self.max_speed, -self.max_speed]
        elif action == self.STOP:
            return [0.0, 0.0]
        return [0.0, 0.0]

    def save_q_table(self, filepath: str) -> bool:
        """Save Q-table to file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(self.q_table, f)

            logger.info(f"Q-table saved to {filepath} with {len(self.q_table)} states")
            return True
        except Exception as e:
            logger.error(f"Error saving Q-table: {e}")
            return False

    def load_q_table(self, filepath: str) -> bool:
        """Load Q-table from file or init empty."""
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    self.q_table = pickle.load(f)

                logger.info(
                    f"Q-table loaded from {filepath} with {len(self.q_table)} states"
                )
                return True
            else:
                logger.warning(
                    f"Q-table file {filepath} not found. Starting with empty Q-table."
                )
                self.q_table = {}
                return False
        except Exception as e:
            logger.error(f"Error loading Q-table: {e}")
            self.q_table = {}
            return False
