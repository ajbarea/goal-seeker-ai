"""Configuration and logging setup for reinforcement learning and simulation."""

import logging
import os
from typing import List
from datetime import datetime

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
Q_TABLE_PATH = os.path.join(DATA_DIR, "q_table.pkl")


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: str = "logs",
) -> logging.Logger:
    """Create and configure a logger with console output and optional file logging.

    Args:
        name (str): Logger identifier.
        level (int): Minimum logging level.
        log_to_file (bool): If True, write logs to a file.
        log_dir (str): Directory path for log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # If a file path is passed as name, derive the logger name from its basename
    if isinstance(name, str) and name.endswith(".py"):
        name = os.path.splitext(os.path.basename(name))[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate log entries
    if logger.handlers:
        logger.handlers.clear()

    # Initialize console handler for output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Set message format for log records
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name.split('.')[-1]}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(
    name: str, level: int = logging.INFO, log_to_file: bool = False
) -> logging.Logger:
    """Get a configured logger instance.

    Get a logger by name with optional file output.

    Args:
        name (str): Logger identifier.
        level (int): Minimum logging level.
        log_to_file (bool): If True, write logs to a file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return setup_logger(name, level, log_to_file)


class RLConfig:
    """Reinforcement learning hyperparameters and limits."""

    # Core RL hyperparameters
    LEARNING_RATE: float = 0.1
    MIN_LEARNING_RATE: float = 0.01
    DISCOUNT_FACTOR: float = 0.99
    MIN_DISCOUNT_FACTOR: float = 0.9

    # Exploration parameters
    EXPLORATION_RATE: float = 1.0
    MIN_EXPLORATION_RATE: float = 0.005
    EXPLORATION_DECAY: float = 0.995
    # Exponential decay schedule:
    # epsilon = min + (max-min) * exp(-decay_rate * episode)
    EXPLORATION_DECAY_RATE: float = 0.01

    # Learning rate parameters
    LEARNING_RATE_DECAY_BASE: float = 0.999
    LEARNING_RATE_DECAY_DENOM: int = 10000

    # Episode limits
    MAX_EPISODES: int = 100  # Fallback maximum if convergence not reached
    MAX_STEPS_PER_EPISODE: int = 1500

    # Early stopping parameters
    ENABLE_EARLY_STOPPING: bool = True
    MIN_EPISODES: int = 20  # Minimum episodes before checking for convergence
    CONVERGENCE_WINDOW: int = 10  # Number of episodes to check for stable performance
    SUCCESS_RATE_THRESHOLD: float = 0.75  # Success rate threshold for convergence
    REWARD_IMPROVEMENT_THRESHOLD: float = (
        0.03  # Minimum improvement percentage to continue training
    )
    MAX_CONVERGENCE_ATTEMPTS: int = (
        3  # Number of times to confirm convergence before stopping
    )

    # Action persistence parameters
    ACTION_PERSISTENCE_INITIAL: int = 3
    ACTION_PERSISTENCE_MIN: int = 1
    ACTION_PERSISTENCE_DECAY: float = 0.95

    # Target and reward configuration
    TARGET_THRESHOLD: float = 0.15
    TARGET_REACHED_REWARD: float = 200.0  # Reward for reaching the target

    # Negative reward applied each step
    STEP_PENALTY: float = 0.02
    # Reward shaping scale factor
    REWARD_SHAPING_SCALE: float = 3.0

    # Collision penalty parameters
    COLLISION_PENALTY: float = 1.0  # Penalty for hitting obstacles
    COLLISION_SENSOR_THRESHOLD: int = 3  # Discrete sensor state indicating collision

    # Protocol prefix for action commands to the slave controller
    ACTION_COMMAND_PREFIX: str = "exec_action:"

    # Discretization settings
    DISTANCE_BINS: List[float] = [0.1, 0.25, 0.5, 0.75, 1.25, 2.0]
    ANGLE_BINS: int = 8

    # Toggle between table-based Q-learning and DQN
    USE_DQN: bool = False

    # DQN hyperparameters
    BUFFER_SIZE: int = 10000
    BATCH_SIZE: int = 64
    GAMMA: float = 0.99
    LR: float = 1e-3
    TARGET_UPDATE: int = 1000
    EPS_START: float = 1.0
    EPS_END: float = 0.05
    EPS_DECAY: int = 10000


class RobotConfig:
    """Physical parameters and start/target positions for the robot."""

    # Physical parameters of the robot
    MAX_SPEED: float = 10.0
    TIME_STEP: int = 64
    DEFAULT_POSITION: List[float] = [0.0, 0.0, 0.0]

    # Positions used as targets during training
    TARGET_POSITIONS: List[List[float]] = [[0.62, -0.61]]

    # Initial positions for training episodes
    START_POSITIONS: List[List[float]] = [[0.0, 0.0, 0.0]]


class SimulationConfig:
    """Simulation settings and logging options."""

    # Logging levels and output options
    LOG_LEVEL_DRIVER: str = "INFO"
    LOG_LEVEL_SLAVE: str = "INFO"
    LOG_TO_FILE: bool = True
    DETAILED_LOG_FREQ: int = 75

    # Frequency for position updates
    POSITION_UPDATE_FREQ: int = 50

    # Detailed messaging configuration
    ENABLE_DETAILED_LOGGING: bool = True

    # Timeout for goal seeking (seconds)
    GOAL_SEEKING_TIMEOUT: int = 500

    # Stuck detection threshold
    STUCK_THRESHOLD: int = 3
