"""
Configuration module for robot learning parameters and simulation settings.
"""

import logging
import os
from datetime import datetime

# Data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
Q_TABLE_PATH = os.path.join(DATA_DIR, "q_table.pkl")
PLOT_DIR = DATA_DIR


def setup_logger(name, level=logging.INFO, log_to_file=False, log_dir="logs"):
    """
    Configure and return a logger with the specified name and level.

    Args:
        name (str): Logger name, typically __name__ of the calling module
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file (bool): Whether to also log to a file
        log_dir (str): Directory for log files if log_to_file is True

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates when reconfiguring
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler with a formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optionally add file handler
    if log_to_file:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name.split('.')[-1]}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name, level=logging.INFO, log_to_file=False):
    """
    Get a logger with the given name. Convenience function.

    Args:
        name (str): Logger name
        level (int): Logging level
        log_to_file (bool): Whether to log to a file

    Returns:
        logging.Logger: Configured logger
    """
    return setup_logger(name, level, log_to_file)


class RLConfig:
    """Reinforcement learning configuration parameters."""

    # Core RL parameters
    LEARNING_RATE = 0.1
    MIN_LEARNING_RATE = 0.03
    DISCOUNT_FACTOR = 0.95
    MIN_DISCOUNT_FACTOR = 0.7
    EXPLORATION_RATE = 0.4
    MIN_EXPLORATION_RATE = 0.05
    EXPLORATION_DECAY = 0.985
    LEARNING_RATE_DECAY_BASE = 0.9995
    LEARNING_RATE_DECAY_DENOM = 20000

    # Episode parameters
    MAX_EPISODES = 100
    MAX_STEPS_PER_EPISODE = 600

    # Action parameters - adjust for smoother control
    ACTION_PERSISTENCE_INITIAL = 3
    ACTION_PERSISTENCE_MIN = 1
    ACTION_PERSISTENCE_DECAY = 0.95

    # Target and reward parameters
    TARGET_THRESHOLD = 0.15

    # Small negative reward for each step
    STEP_PENALTY = 0.1

    # Command protocol for sending actions to slave
    ACTION_COMMAND_PREFIX = "exec_action:"

    # Default path for Q-table
    Q_TABLE_PATH: str = Q_TABLE_PATH


class RobotConfig:
    """Robot configuration parameters."""

    # Robot physical parameters
    MAX_SPEED = 10.0
    TIME_STEP = 64
    DEFAULT_POSITION = [0.0, 0.0, 0.0]

    # Target positions for training
    # TARGET_POSITIONS = [[0.62, -0.61], [0.5, 0.5], [-0.5, 0.5]]
    TARGET_POSITIONS = [[0.62, -0.61]]

    # Starting positions for training
    # START_POSITIONS = [[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [-0.3, 0.0, 0.0]]
    START_POSITIONS = [[0.0, 0.0, 0.0]]


class SimulationConfig:
    """Simulation and logging configuration."""

    # Logging parameters
    LOG_LEVEL_DRIVER = "INFO"
    LOG_LEVEL_SLAVE = "INFO"
    LOG_TO_FILE = True

    # Reporting frequencies
    POSITION_UPDATE_FREQ = 5

    # File paths
    Q_TABLE_PATH = Q_TABLE_PATH
    PLOT_DIR = PLOT_DIR

    # Message protocol configuration
    ENABLE_DETAILED_LOGGING = True

    # Goal seeking parameters
    GOAL_SEEKING_TIMEOUT = 500  # Timeout for goal seeking in seconds
