"""RL & simulation configuration and logger setup."""

import logging
import os
from datetime import datetime

# Data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
Q_TABLE_PATH = os.path.join(DATA_DIR, "q_table.pkl")
PLOT_DIR = DATA_DIR


def setup_logger(name, level=logging.INFO, log_to_file=False, log_dir="logs"):
    """Initialize a logger with console output and optional file logging.

    Args:
        name (str): Logger identifier.
        level (int): Minimum log level.
        log_to_file (bool): Write logs to file if True.
        log_dir (str): Directory for log files.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to prevent duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Define log message format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If log_to_file:
    if log_to_file:
        # Ensure log_dir exists
        os.makedirs(log_dir, exist_ok=True)

        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name.split('.')[-1]}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name, level=logging.INFO, log_to_file=False):
    """Shortcut to setup_logger with optional file output."""
    return setup_logger(name, level, log_to_file)


class RLConfig:
    """Common RL hyperparameters."""

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
    """Robot physical parameters and start/target positions."""

    # Robot physical parameters
    MAX_SPEED = 10.0
    TIME_STEP = 64
    DEFAULT_POSITION = [0.0, 0.0, 0.0]

    # Target positions for training
    TARGET_POSITIONS = [[0.62, -0.61]]

    # Starting positions for training
    START_POSITIONS = [[0.0, 0.0, 0.0]]


class SimulationConfig:
    """Simulation settings and logging options."""

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
