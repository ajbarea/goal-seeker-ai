"""Configuration and logging setup for reinforcement learning and simulation."""

import logging
import os
from datetime import datetime

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
Q_TABLE_PATH = os.path.join(DATA_DIR, "q_table.pkl")
BEST_Q_TABLE_PATH = os.path.join(DATA_DIR, "best_q_table.pkl")


def setup_logger(name, level=logging.INFO, log_to_file=False, log_dir="logs"):
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


def get_logger(name, level=logging.INFO, log_to_file=False):
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
    LEARNING_RATE = 0.2
    MIN_LEARNING_RATE = 0.03
    DISCOUNT_FACTOR = 0.98
    MIN_DISCOUNT_FACTOR = 0.7
    EXPLORATION_RATE = 0.3
    MIN_EXPLORATION_RATE = 0.05
    EXPLORATION_DECAY = 0.99
    LEARNING_RATE_DECAY_BASE = 0.9995
    LEARNING_RATE_DECAY_DENOM = 20000

    # Episode limits
    MAX_EPISODES = 300
    MAX_STEPS_PER_EPISODE = 1000

    # Action persistence parameters
    ACTION_PERSISTENCE_INITIAL = 3
    ACTION_PERSISTENCE_MIN = 1
    ACTION_PERSISTENCE_DECAY = 0.95

    # Target and reward configuration
    TARGET_THRESHOLD = 0.15
    TARGET_REACHED_REWARD = 75  # Reward for reaching the target

    # Negative reward applied each step
    STEP_PENALTY = 0.1

    # Collision penalty parameters
    COLLISION_PENALTY = 2.5  # Penalty for hitting obstacles
    COLLISION_SENSOR_THRESHOLD = 3  # Discrete sensor state indicating collision

    # Protocol prefix for action commands to the slave controller
    ACTION_COMMAND_PREFIX = "exec_action:"

    # Stuck detection and recovery
    POSITION_MEMORY_SIZE = 20  # Number of previous positions to remember
    POSITION_MEMORY_THRESHOLD = 0.05  # Distance threshold to consider positions similar
    STUCK_POSITION_PENALTY = (
        -2.0
    )  # Penalty for revisiting positions where robot got stuck

    # State blending
    ENABLE_STATE_BLENDING = True
    STATE_BLENDING_FACTOR = 0.3  # Weight given to nearby states

    # Dynamic approach strategies
    PRECISION_APPROACH_DISTANCE = 0.8  # Distance to switch to precision approach


class RobotConfig:
    """Physical parameters and start/target positions for the robot."""

    # Physical parameters of the robot
    MAX_SPEED = 10.0
    TIME_STEP = 64
    DEFAULT_POSITION = [0.0, 0.0, 0.0]

    # Positions used as targets during training
    TARGET_POSITIONS = [[0.62, -0.61]]

    # Initial positions for training episodes
    START_POSITIONS = [[0.0, 0.0, 0.0]]


class SimulationConfig:
    """Simulation settings and logging options."""

    # Logging levels and output options
    LOG_LEVEL_DRIVER = "INFO"
    LOG_LEVEL_SLAVE = "INFO"
    LOG_TO_FILE = True

    # Frequency for position updates
    POSITION_UPDATE_FREQ = 5

    # Detailed messaging configuration
    ENABLE_DETAILED_LOGGING = True

    # Timeout for goal seeking (seconds)
    GOAL_SEEKING_TIMEOUT = 500

    # Stuck detection threshold
    STUCK_THRESHOLD = 3
