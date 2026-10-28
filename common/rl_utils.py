"""Reinforcement learning utilities for state discretization, reward computation, and progress visualization."""

import math
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional
from common.config import RLConfig, get_logger, RobotConfig

logger = get_logger(__name__)


def get_discrete_state(
    position: List[float],
    target_position: List[float],
    orientation: float,
    left_sensor: float,
    right_sensor: float,
    wheel_velocities: List[float],
    angle_bins: Optional[int] = None,
) -> Optional[Tuple[int, int, int, int, int]]:
    """Compute discrete state tuple for Q-learning.

    Discretize position, orientation, sensor readings, and wheel velocities into state bins.

    Args:
        position (List[float]): Current [x, y] position.
        target_position (List[float]): Target [x, y] position.
        orientation (float): Heading angle in radians.
        left_sensor (float): Left sensor reading.
        right_sensor (float): Right sensor reading.
        wheel_velocities (List[float]): Wheel velocities [left, right].
        angle_bins (Optional[int]): Number of angle discretization bins.

    Returns:
        Optional[Tuple[int, int, int, int, int]]: (distance_bin, angle_bin, left_obs, right_obs, velocity_state) or None if invalid input.
    """
    angle_bins = angle_bins or RLConfig.ANGLE_BINS

    if not position or not target_position:
        return None

    # Compute continuous metrics for discretization
    distance = calculate_distance(position, target_position)
    dx, dy = target_position[0] - position[0], target_position[1] - position[1]
    rel_angle = normalize_angle(math.atan2(dy, dx) - orientation)

    # Convert observations into discrete bins
    distance_bin = discretize_distance(distance)
    angle_bin = int((rel_angle + math.pi) / (2 * math.pi / angle_bins)) % angle_bins
    left_obs, right_obs = discretize_sensor(left_sensor), discretize_sensor(
        right_sensor
    )
    velocity_state = discretize_velocity(wheel_velocities)

    return (distance_bin, angle_bin, left_obs, right_obs, velocity_state)


def discretize_distance(distance: float) -> int:
    """Discretize distance into integer bins based on RLConfig thresholds.

    Args:
        distance (float): Continuous distance value.

    Returns:
        int: Discrete distance bin.
    """
    for i, threshold in enumerate(RLConfig.DISTANCE_BINS):
        if distance < threshold:
            return i
    return len(RLConfig.DISTANCE_BINS)


def discretize_sensor(sensor_value: float) -> int:
    """Convert sensor reading into discrete states 0–3.

    Args:
        sensor_value (float): Raw sensor measurement.

    Returns:
        int: Discrete sensor state.
    """
    if sensor_value < 100:
        return 0
    elif sensor_value < 400:
        return 1
    elif sensor_value < 700:
        return 2
    else:
        return 3


def discretize_velocity(wheel_velocities: List[float]) -> int:
    """Determine discrete motion state from wheel velocities.

    Args:
        wheel_velocities (List[float]): [left_vel, right_vel] velocities.

    Returns:
        int: Motion state (0: stopped, 1: slow forward, 2: fast forward, 3: backward, 4: turning).
    """
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


def calculate_reward(
    current_position: List[float],
    target_position: List[float],
    previous_distance: Optional[float] = None,
    target_threshold: float = 0.1,
    left_sensor: Optional[float] = None,
    right_sensor: Optional[float] = None,
) -> float:
    """Compute reward from progress, step penalty, and collisions.

    Args:
        current_position (List[float]): Current [x, y] position.
        target_position (List[float]): Target [x, y] position.
        previous_distance (Optional[float]): Distance from previous step.
        target_threshold (float): Distance threshold for success.
        left_sensor (Optional[float]): Left sensor reading.
        right_sensor (Optional[float]): Right sensor reading.

    Returns:
        float: Calculated reward value.
    """
    current_distance = calculate_distance(current_position[:2], target_position)

    if is_target_reached(current_distance, target_threshold):
        return RLConfig.TARGET_REACHED_REWARD

    if previous_distance is None:
        return 0.0

    # Potential-based shaping reward: previous_distance - γ * current_distance
    shaping_delta = previous_distance - RLConfig.DISCOUNT_FACTOR * current_distance
    # max possible movement per step: speed * timestep (ms->s)
    max_delta = RobotConfig.MAX_SPEED * (RobotConfig.TIME_STEP / 1000.0)
    if max_delta <= 0:
        max_delta = 1.0
    normalized = shaping_delta / max_delta

    # Scale up the reward for making progress
    reward = max(-1.0, min(1.0, normalized)) * RLConfig.REWARD_SHAPING_SCALE

    # Add a small positive reward for being on the right track (heading toward goal)
    if shaping_delta > 0:
        reward += 0.1

    # include per-step penalty
    reward -= RLConfig.STEP_PENALTY

    # Apply collision penalty if sensors indicate obstacle contact
    if left_sensor is not None and right_sensor is not None:
        left_obs = discretize_sensor(left_sensor)
        right_obs = discretize_sensor(right_sensor)
        if (
            left_obs >= RLConfig.COLLISION_SENSOR_THRESHOLD
            or right_obs >= RLConfig.COLLISION_SENSOR_THRESHOLD
        ):
            reward -= RLConfig.COLLISION_PENALTY

    return reward


def get_action_name(action: int) -> str:
    """Get human-readable action name from index.

    Args:
        action (int): Action index.

    Returns:
        str: Action name or 'UNKNOWN_ACTION' if invalid.
    """
    action_names = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "BACKWARD", "STOP"]
    if 0 <= action < len(action_names):
        return action_names[action]
    return f"UNKNOWN_ACTION({action})"


def calculate_distance(p1: List[float], p2: List[float]) -> float:
    """Compute Euclidean distance between two 2D points.

    Args:
        p1 (List[float]): First [x, y] point.
        p2 (List[float]): Second [x, y] point.

    Returns:
        float: Euclidean distance.
    """
    # Using NumPy’s optimized hypot for efficient Euclidean distance
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def normalize_angle(angle: float) -> float:
    """Normalize angle to range [-π, π].

    Args:
        angle (float): Angle in radians.

    Returns:
        float: Normalized angle.
    """
    if abs(angle % (2 * math.pi) - math.pi) < 1e-10:
        return -math.pi

    angle = angle % (2 * math.pi)

    # Convert from [0, 2π] to [-π, π]
    if angle > math.pi:
        angle -= 2 * math.pi

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
    """Plot Q-learning reward history with moving averages.

    Args:
        rewards (List[float]): Reward per episode.
        window (int): Window size for long moving average.
        short_window (int): Window size for short moving average.
        ema_span (int): Span for exponential moving average.
        title (str): Plot title.
        filename (Optional[str]): Filename (without extension) to save plot.
        save_dir (Optional[str]): Directory to save plot image.

    Returns:
        None
    """
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


def get_nearby_states(
    state: Tuple[int, int, int, int, int],
) -> List[Tuple[int, int, int, int, int]]:
    """Get neighboring states for state blending.

    Args:
        state (Tuple[int, int, int, int, int]): Base discrete state.

    Returns:
        List[Tuple[int, int, int, int, int]]: Nearby state tuples.
    """
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
    """Check if two positions are within threshold distance.

    Args:
        pos1 (List[float]): First [x, y] position.
        pos2 (List[float]): Second [x, y] position.
        threshold (float): Distance threshold.

    Returns:
        bool: True if distance < threshold.
    """
    return calculate_distance(pos1, pos2) < threshold


def get_position_cluster(
    positions: List[List[float]], new_pos: List[float], threshold: float = 0.05
) -> int:
    """Find cluster index for a new position or return -1.

    Args:
        positions (List[List[float]]): Existing position clusters.
        new_pos (List[float]): New [x, y] position.
        threshold (float): Similarity distance threshold.

    Returns:
        int: Matching cluster index or -1 if none.
    """
    for i, pos in enumerate(positions):
        if is_similar_position(pos, new_pos, threshold):
            return i
    return -1  # No matching cluster


def is_target_reached(
    distance: float, threshold: float = RLConfig.TARGET_THRESHOLD
) -> bool:
    """Return True if distance is below the goal threshold."""
    return distance < threshold


def log_goal_reached(
    distance: float,
    mode_name: str,
    logger,
    threshold: float = RLConfig.TARGET_THRESHOLD,
) -> bool:
    """
    Check if distance < threshold, log a goal‑reached message, and return True.
    """
    if is_target_reached(distance, threshold):
        logger.info(f"🎯 Target reached in {mode_name} mode!")
        return True
    return False
