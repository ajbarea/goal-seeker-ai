"""Implement Q-learning agent with adaptive parameters and persistence."""

import random
import pickle
import os
from typing import Dict, List, Tuple, Optional
from common.rl_utils import get_discrete_state
from common.config import RLConfig, get_logger

# Configure module-level logger
logger = get_logger("agent")


class QLearningAgent:
    """Implement Q-learning logic and Q-table updates."""

    # Action constants
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
        """Initialize Q-learning agent with parameters and Q-table.

        Args:
            learning_rate (float): Initial learning rate for Q-value updates.
            min_learning_rate (float): Minimum learning rate.
            discount_factor (float): Discount factor for future rewards.
            min_discount_factor (float): Minimum discount factor.
            exploration_rate (float): Probability of random action selection.
            max_speed (float): Maximum robot speed.
            angle_bins (int): Number of bins for angle discretization.
        """
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
        self.rng = random.Random()

    def get_discrete_state(
        self,
        position: List[float],
        target_position: List[float],
        orientation: float,
        left_sensor: float,
        right_sensor: float,
        wheel_velocities: List[float],
    ) -> Optional[Tuple]:
        """Discretize raw observations into state bins.

        Args:
            position (List[float]): Current [x, y] position.
            target_position (List[float]): Target [x, y] position.
            orientation (float): Robot orientation in radians.
            left_sensor (float): Left sensor reading.
            right_sensor (float): Right sensor reading.
            wheel_velocities (List[float]): Wheel velocities [left, right].

        Returns:
            Optional[Tuple]: Discrete state tuple or None if invalid input.
        """
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
        """Select an action using epsilon-greedy policy.

        Args:
            state (Tuple): Current discrete state.
            current_distance (float, optional): Distance to target.

        Returns:
            int: Selected action index.
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5

        allow_stop = (
            current_distance is not None
            and current_distance <= RLConfig.TARGET_THRESHOLD
        )
        action_indices = [0, 1, 2, 3]
        if allow_stop:
            action_indices.append(4)

        if self.rng.random() < self.exploration_rate:
            if self.rng.random() < 0.5 and 0 in action_indices:
                return 0
            return self.rng.choice(action_indices)

        q_values = self.q_table[state]
        filtered_q = [(i, q_values[i]) for i in action_indices]
        max_q_value = max(q for i, q in filtered_q)
        best_actions = [i for i, q in filtered_q if q == max_q_value]
        return self.rng.choice(best_actions)

    def choose_best_action(self, state: Tuple, current_distance: float = None) -> int:
        """Select the best action using Q-values."""
        if state not in self.q_table:
            # Initialize unseen state with zero Q-values
            self.q_table[state] = [0.0] * 5

        allow_stop = (
            current_distance is not None
            and current_distance <= RLConfig.TARGET_THRESHOLD
        )
        action_indices = [0, 1, 2, 3]
        if allow_stop:
            action_indices.append(4)

        # Q‑value based decision (no blending)
        q_values = self.q_table[state]
        filtered_q = [(i, q_values[i]) for i in action_indices]
        max_q_value = max(q for _, q in filtered_q)
        best_actions = [i for i, q in filtered_q if q == max_q_value]
        return self.rng.choice(best_actions)

    def update_q_table(
        self, state: Tuple, action: int, reward: float, next_state: Tuple
    ) -> None:
        """Update Q-table using temporal difference (TD) learning.

        Args:
            state (Tuple): Current discrete state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (Tuple): Next discrete state."""
        if state is None or next_state is None:
            return

        if state not in self.q_table:
            self.q_table[state] = [0.0] * 5
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 5

        self.total_updates += 1

        # Apply collision penalty if state indicates obstacle contact
        # state format: (distance_bin, angle_bin, left_obs, right_obs, velocity_state)
        _, _, left_obs, right_obs, _ = state
        if (
            left_obs >= RLConfig.COLLISION_SENSOR_THRESHOLD
            or right_obs >= RLConfig.COLLISION_SENSOR_THRESHOLD
        ):
            reward -= RLConfig.COLLISION_PENALTY

        # Compute adaptive learning rate influenced by state distance bin
        distance_bin = state[0]
        distance_factor = max(0.8, 2.0 - (distance_bin * 0.2))

        adaptive_learning_rate = max(
            self.min_learning_rate,
            self.learning_rate
            * distance_factor
            * (
                RLConfig.LEARNING_RATE_DECAY_BASE
                ** (self.total_updates / RLConfig.LEARNING_RATE_DECAY_DENOM)
            ),
        )

        # Compute adaptive discount factor based on proximity to goal
        discount_distance_factor = min(1.0, 0.7 + (distance_bin * 0.05))
        adaptive_discount = max(
            self.min_discount_factor,
            self.discount_factor
            * discount_distance_factor
            * (0.9995 ** (self.total_updates / 5000)),
        )

        self.learning_rates.append(adaptive_learning_rate)
        self.discount_factors.append(adaptive_discount)

        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])

        td_error = reward + adaptive_discount * next_max_q - current_q
        self.td_errors.append(td_error)

        new_q = current_q + adaptive_learning_rate * td_error
        self.q_table[state][action] = max(-50.0, min(50.0, new_q))

    def execute_action(self, action: int, state: Tuple = None) -> List[float]:
        """Convert action index to motor speed commands.

        Args:
            action (int): Action index.
            state (Tuple, optional): Current discrete state.

        Returns:
            List[float]: Motor speed commands [left, right]."""
        # Default speeds
        forward_speed = self.max_speed
        turn_speed = self.max_speed / 2

        # Adjust speeds based on distance if state is available
        if state is not None:
            distance_bin = state[0]

            if distance_bin < 2:  # Very close to target
                forward_speed = self.max_speed * 0.6  # More precise movements
                turn_speed = self.max_speed * 0.4
            elif distance_bin < 4:  # Medium distance
                forward_speed = self.max_speed * 0.8
                turn_speed = self.max_speed * 0.6
            else:  # Far away
                forward_speed = self.max_speed  # Full speed
                turn_speed = self.max_speed * 0.7

        if action == self.FORWARD:
            return [forward_speed, forward_speed]
        elif action == self.TURN_LEFT:
            return [turn_speed, -turn_speed]
        elif action == self.TURN_RIGHT:
            return [-turn_speed, turn_speed]
        elif action == self.BACKWARD:
            return [-forward_speed * 0.7, -forward_speed * 0.7]
        elif action == self.STOP:
            return [0.0, 0.0]
        return [0.0, 0.0]

    def save_q_table(self, filepath: str) -> bool:
        """Save the Q-table to a file.

        Args:
            filepath (str): Path to save the Q-table.

        Returns:
            bool: True if save was successful, False otherwise."""
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
        """Load the Q-table from a file or initialize an empty table.

        Args:
            filepath (str): Path to load the Q-table from.

        Returns:
            bool: True if load was successful, False otherwise."""
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
