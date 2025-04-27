"""Reinforcement learning utilities for state discretization, reward computation, and progress visualization."""

import math
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional
from common.config import RLConfig, get_logger

logger = get_logger(__name__)


def get_discrete_state(
    position: List[float],
    target_position: List[float],
    orientation: float,
    left_sensor: float,
    right_sensor: float,
    wheel_velocities: List[float],
    angle_bins: int = 8,
) -> Optional[Tuple]:
    """Compute a discrete state tuple for Q‑learning."""
    if not position or not target_position:
        return None

    # Compute Euclidean distance and relative angle to the target
    distance = calculate_distance(position, target_position)
    dx, dy = target_position[0] - position[0], target_position[1] - position[1]
    rel_angle = normalize_angle(math.atan2(dy, dx) - orientation)

    # Convert continuous observations into discrete state bins
    distance_bin = discretize_distance(distance)
    angle_bin = int((rel_angle + math.pi) / (2 * math.pi / angle_bins)) % angle_bins
    left_obs, right_obs = discretize_sensor(left_sensor), discretize_sensor(
        right_sensor
    )
    velocity_state = discretize_velocity(wheel_velocities)

    return (distance_bin, angle_bin, left_obs, right_obs, velocity_state)


def discretize_distance(distance: float) -> int:
    """Discretize continuous distance into bins 0–6."""
    if distance < 0.1:
        return 0
    elif distance < 0.25:
        return 1
    elif distance < 0.5:
        return 2
    elif distance < 0.75:
        return 3
    elif distance < 1.25:
        return 4
    elif distance < 2.0:
        return 5
    else:
        return 6


def discretize_sensor(sensor_value: float) -> int:
    """Discretize sensor reading into states 0–3 based on thresholds."""
    if sensor_value < 100:
        return 0
    elif sensor_value < 400:
        return 1
    elif sensor_value < 700:
        return 2
    else:
        return 3


def discretize_velocity(wheel_velocities: List[float]) -> int:
    """Determine discrete motion state (0–4) from wheel velocities."""
    left_vel = wheel_velocities[0]
    right_vel = wheel_velocities[1]
    avg_speed = (abs(left_vel) + abs(right_vel)) / 2
    is_turning = left_vel * right_vel < 0

    if is_turning:
        return 4
    elif avg_speed < 0.1:
        return 0
    elif left_vel > 0 and right_vel > 0:
        return 2 if avg_speed > 5.0 else 1
    elif left_vel < 0 and right_vel < 0:
        return 3
    else:
        return 0


def calculate_potential(
    position: List[float],
    target_position: List[float],
) -> float:
    """Potential function for reward shaping: negative distance to goal."""
    return -calculate_distance(position[:2], target_position)


def calculate_reward(
    current_position: List[float],
    target_position: List[float],
    previous_distance: Optional[float] = None,
    target_threshold: float = 0.1,
    left_sensor: Optional[float] = None,
    right_sensor: Optional[float] = None,
    previous_position: Optional[List[float]] = None,
    discount_factor: Optional[float] = None,
    use_potential_shaping: bool = False,
) -> float:
    """Compute the reward with optional potential-based shaping."""
    current_distance = calculate_distance(current_position[:2], target_position)

    if current_distance < target_threshold:
        return RLConfig.TARGET_REACHED_REWARD

    if previous_distance is None:
        base_reward = 0.0
    else:
        # reward is proportional to distance improvement
        distance_improvement = previous_distance - current_distance
        if distance_improvement > 0:
            base_reward = 10.0 * distance_improvement
        else:
            base_reward = -8.0 * abs(distance_improvement)
        # include per-step penalty
        base_reward -= RLConfig.STEP_PENALTY

    # Apply collision penalty if sensors indicate obstacle contact
    if left_sensor is not None and right_sensor is not None:
        left_obs = discretize_sensor(left_sensor)
        right_obs = discretize_sensor(right_sensor)
        if (
            left_obs >= RLConfig.COLLISION_SENSOR_THRESHOLD
            or right_obs >= RLConfig.COLLISION_SENSOR_THRESHOLD
        ):
            base_reward -= RLConfig.COLLISION_PENALTY

    # --- Potential-based reward shaping ---
    if use_potential_shaping and previous_position is not None and discount_factor is not None:
        phi_prev = calculate_potential(previous_position, target_position)
        phi_curr = calculate_potential(current_position, target_position)
        shaped_reward = base_reward + discount_factor * phi_curr - phi_prev
        return shaped_reward
    # --- End shaping ---

    return base_reward


def get_action_name(action: int) -> str:
    """Return the descriptive name for a given action index."""
    action_names = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "BACKWARD", "STOP"]
    if 0 <= action < len(action_names):
        return action_names[action]
    return f"UNKNOWN_ACTION({action})"


def calculate_distance(p1: List[float], p2: List[float]) -> float:
    """Compute the Euclidean distance between two 2D points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [–π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def plot_q_learning_progress(
    rewards: List[float],
    window: int = 20,
    short_window: int = 5,
    ema_span: int = 20,
    title: str = "Q‑Learning Progress",
    filename: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Plot reward history with moving averages and exponential moving average."""
    if not rewards:
        logger.warning("No rewards to plot")
        return

    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(
        episodes, rewards, color="lightblue", alpha=0.4, label="Reward per Episode"
    )

    if len(rewards) >= short_window:
        ma_s = np.convolve(rewards, np.ones(short_window) / short_window, mode="valid")
        ma_s_x = list(range(short_window, len(rewards) + 1))
        plt.plot(
            ma_s_x, ma_s, color="green", linewidth=2, label=f"{short_window}-Episode MA"
        )

    if len(rewards) >= window:
        ma_l = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ma_l_x = list(range(window, len(rewards) + 1))
        plt.plot(ma_l_x, ma_l, color="red", linewidth=2, label=f"{window}-Episode MA")

    cumavg = list(np.cumsum(rewards) / np.arange(1, len(rewards) + 1))
    plt.plot(
        episodes, cumavg, color="orange", linestyle="--", label="Cumulative Average"
    )

    if ema_span > 1 and len(rewards) > 0:
        alpha = 2.0 / (ema_span + 1)
        ema = [rewards[0]]
        for r in rewards[1:]:
            ema.append(alpha * r + (1 - alpha) * ema[-1])
        plt.plot(
            episodes, ema, color="purple", linestyle=":", label=f"{ema_span}-Span EMA"
        )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()

    if filename:
        save_dir = save_dir or "."
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{filename}.png")
        plt.savefig(path)

    plt.show()
    plt.close()


def get_nearby_states(state: Tuple) -> List[Tuple]:
    """Return neighboring states for state blending."""
    distance_bin, angle_bin, left_obstacle, right_obstacle, is_moving = state
    nearby_states = []

    # Add states with similar distance
    for d_offset in [-1, 1]:
        new_distance = distance_bin + d_offset
        if 0 <= new_distance <= 6:  # Valid distance bin range
            nearby_states.append(
                (new_distance, angle_bin, left_obstacle, right_obstacle, is_moving)
            )

    # Add states with similar angle
    max_angle_bin = 8 - 1  # Assuming 8 angle bins (0-7)
    for a_offset in [-1, 1]:
        new_angle = (angle_bin + a_offset) % (max_angle_bin + 1)  # Wrap around
        nearby_states.append(
            (distance_bin, new_angle, left_obstacle, right_obstacle, is_moving)
        )

    return nearby_states


def is_similar_position(
    pos1: List[float], pos2: List[float], threshold: float = 0.05
) -> bool:
    """Determine whether two positions are within a given threshold."""
    return calculate_distance(pos1, pos2) < threshold


def get_position_cluster(
    positions: List[List[float]], new_pos: List[float], threshold: float = 0.05
) -> int:
    """Assign a new position to an existing cluster or return –1 if none match."""
    for i, pos in enumerate(positions):
        if is_similar_position(pos, new_pos, threshold):
            return i
    return -1  # No matching cluster
