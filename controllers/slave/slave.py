"""Provide robot control for obstacle avoidance, learning, and goal seeking."""

from controller import AnsiCodes, Robot  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import logging
import math
from collections import deque, defaultdict
from common.config import (
    RLConfig,
    RobotConfig,
    SimulationConfig,
    get_logger,
    DATA_DIR,
    Q_TABLE_PATH,
    BEST_Q_TABLE_PATH,
)
from common.rl_utils import calculate_distance
from q_learning_agent import QLearningAgent

# Configure module-level logger
logger = get_logger(
    __file__,
    level=getattr(logging, SimulationConfig.LOG_LEVEL_SLAVE, "INFO"),
)


class Enumerate(object):
    """Provide a simple enumeration from space-separated names."""

    def __init__(self, names):
        for number, name in enumerate(names.split()):
            setattr(self, name, number)


class Slave(Robot):
    """Control robot modes for obstacle avoidance, learning, and goal seeking."""

    Mode = Enumerate("STOP MOVE_FORWARD AVOID_OBSTACLES TURN SEEK_GOAL LEARN")
    timeStep = RobotConfig.TIME_STEP
    maxSpeed = RobotConfig.MAX_SPEED
    mode = Mode.AVOID_OBSTACLES
    motors = []
    distanceSensors = []

    def boundSpeed(self, speed):
        """Clamp speed to allowed range."""
        return max(-self.maxSpeed, min(self.maxSpeed, speed))

    def normalize_angle(self, angle):
        """Normalize angle to range [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def __init__(self):
        """Initialize device interfaces, sensors, and Q-learning agent."""
        super(Slave, self).__init__()

        self.robot_name = self.getName()
        logger.info(f"Initializing robot: {self.robot_name}")

        self.world_time_step = int(self.getBasicTimeStep())
        self.timeStep = self.world_time_step

        try:
            custom_data = self.getCustomData()
            if custom_data and custom_data.strip():
                logger.info(f"Found custom data: {custom_data}")
        except Exception:
            pass

        self.mode = self.Mode.AVOID_OBSTACLES
        self.camera = self.getDevice("camera")
        self.camera.enable(4 * self.timeStep)
        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.timeStep)
        self.motors.append(self.getDevice("left wheel motor"))
        self.motors.append(self.getDevice("right wheel motor"))
        self.motors[0].setPosition(float("inf"))
        self.motors[1].setPosition(float("inf"))
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        for dsnumber in range(0, 2):
            self.distanceSensors.append(self.getDevice("ds" + str(dsnumber)))
            self.distanceSensors[-1].enable(self.timeStep)

        self.gps = None
        try:
            gps_device = self.getDevice("gps")
            self.gps = gps_device
            self.gps.enable(self.timeStep)
        except Exception:
            logger.info("Fallback to supervisor position updates when GPS unavailable")
            self.gps = None

        self.position = [0, 0]
        self.orientation = 0.0

        self.q_agent = QLearningAgent(
            learning_rate=RLConfig.LEARNING_RATE,
            min_learning_rate=RLConfig.MIN_LEARNING_RATE,
            discount_factor=RLConfig.DISCOUNT_FACTOR,
            min_discount_factor=RLConfig.MIN_DISCOUNT_FACTOR,
            exploration_rate=RLConfig.EXPLORATION_RATE,
            max_speed=self.maxSpeed,
        )

        self.learning_active = False
        self.target_position = None
        self.last_reward = 0
        self.current_state = None
        self.last_action = None

        self.rewards_history = []
        self.target_reached_reported = False

        self.action_persistence = 0
        self.action_persistence_duration = RLConfig.ACTION_PERSISTENCE_INITIAL
        self.current_persistent_action = None
        self.rng = random.Random()

        # Position history tracking
        self.position_history = deque(maxlen=RLConfig.POSITION_MEMORY_SIZE)
        self.stuck_positions = defaultdict(int)  # Position -> count of times stuck
        self.recovery_attempts = 0
        self.last_recovery_time = 0

        logger.info(f"Slave robot initialization complete: {self.robot_name}")

    def run(self):
        """Process messages and update control modes each timestep."""
        while True:
            if self.receiver.getQueueLength() > 0:
                message = self.receiver.getString()
                self.receiver.nextPacket()

                if message.startswith("position:"):
                    try:
                        coords = message[9:].split(",")
                        if len(coords) == 2:
                            new_position = [float(coords[0]), float(coords[1])]
                            self.position = new_position
                    except ValueError:
                        logger.error("Invalid position data")

                elif message.startswith("reward:"):
                    try:
                        reward = float(message[7:])
                        self.last_reward = reward
                        self.rewards_history.append(reward)

                        if (
                            self.learning_active
                            and self.current_state is not None
                            and self.last_action is not None
                        ):
                            next_state = self.get_discrete_state()
                            self.q_agent.update_q_table(
                                self.current_state, self.last_action, reward, next_state
                            )
                            self.current_state = next_state

                            if abs(reward) > 10:
                                self.action_persistence = max(
                                    0, self.action_persistence - 3
                                )
                    except ValueError:
                        logger.error("Invalid reward value")

                elif message == "start_learning":
                    self.mode = self.Mode.LEARN
                    self.current_state = None
                    logger.info("Entering learning mode")

                elif message == "send q_table":
                    self.send_q_table()

                elif message == "plot_learning":
                    self.plot_rewards()

                elif message == "load_q_table":
                    try:
                        self.q_agent.load_q_table(Q_TABLE_PATH)
                    except Exception as e:
                        logger.error(f"Error loading Q-table: {e}")

                elif message == "save_q_table":
                    try:
                        self.q_agent.save_q_table(Q_TABLE_PATH)
                        logger.info(f"Q-table saved to {Q_TABLE_PATH}")
                    except Exception as e:
                        logger.error(f"Error saving Q-table: {e}")

                elif message == "save_best_q_table":
                    try:
                        self.q_agent.save_q_table(BEST_Q_TABLE_PATH)
                        logger.info(f"Best Q-table saved to {BEST_Q_TABLE_PATH}")
                    except Exception as e:
                        logger.error(f"Error saving best Q-table: {e}")

                elif message == "load_best_q_table":
                    try:
                        self.q_agent.load_q_table(BEST_Q_TABLE_PATH)
                        logger.info(f"Best Q-table loaded from {BEST_Q_TABLE_PATH}")
                    except Exception as e:
                        logger.error(f"Error loading best Q-table: {e}")

                elif message == "clear_q_table":
                    self.q_agent.q_table.clear()
                    logger.info("Cleared Q‑table for new training")

                elif message == "reposition":
                    # Adjust orientation when repositioning at mid-range distances

                    # Calculate direct vector to target
                    if self.target_position and self.position:
                        # Simple direct turn in a random direction to break symmetry
                        turn_left = self.rng.random() > 0.5

                        # Execute a series of movements to reorient
                        if turn_left:
                            # Turn left
                            self.motors[0].setVelocity(self.maxSpeed * 0.6)
                            self.motors[1].setVelocity(-self.maxSpeed * 0.6)
                        else:
                            # Turn right
                            self.motors[0].setVelocity(-self.maxSpeed * 0.6)
                            self.motors[1].setVelocity(self.maxSpeed * 0.6)

                        # Turn for a short duration
                        turn_duration = 7
                        for _ in range(turn_duration):
                            if self.step(self.timeStep) == -1:
                                break

                        # Then move forward a bit
                        self.motors[0].setVelocity(self.maxSpeed * 0.7)
                        self.motors[1].setVelocity(self.maxSpeed * 0.7)

                        for _ in range(8):
                            if self.step(self.timeStep) == -1:
                                break

                        # Return to goal seeking
                        if self.mode == self.Mode.SEEK_GOAL:
                            logger.info("Goal seeking in progress")
                        else:
                            self.mode = self.Mode.SEEK_GOAL
                    else:
                        # Fallback if no position data
                        logger.info("Repositioning failed, no position data")

                elif message == "randomize":
                    # Randomize behavior to recover from a stuck state

                    # Reset persistence counter
                    self.action_persistence = 0
                    self.current_persistent_action = None

                    # Reverse for a short distance
                    self.motors[0].setVelocity(-self.maxSpeed * 0.7)
                    self.motors[1].setVelocity(-self.maxSpeed * 0.7)
                    for _ in range(10):
                        if self.step(self.timeStep) == -1:
                            break

                    # Rotate randomly to explore new paths
                    if self.rng.random() < 0.5:
                        # Turn left
                        self.motors[0].setVelocity(self.maxSpeed * 0.8)
                        self.motors[1].setVelocity(-self.maxSpeed * 0.8)
                    else:
                        # Turn right
                        self.motors[0].setVelocity(-self.maxSpeed * 0.8)
                        self.motors[1].setVelocity(self.maxSpeed * 0.8)

                    for _ in range(self.rng.randint(8, 15)):  # Random turn duration
                        if self.step(self.timeStep) == -1:
                            break

                    # Return to goal seeking if that was the previous mode
                    if self.mode == self.Mode.SEEK_GOAL:
                        self.mode = self.Mode.SEEK_GOAL
                    else:
                        self.mode = self.Mode.AVOID_OBSTACLES

                else:
                    if message.startswith(RLConfig.ACTION_COMMAND_PREFIX):
                        logger.debug(f"Received action command: {message}")
                    elif (
                        not message.startswith("reward:")
                        and not message.startswith("seek goal:")
                        and not message.startswith("position:")
                        and not message.startswith("exploration:")
                        and not message.startswith("persistence:")
                    ):
                        logger.info(
                            f"Received command: {AnsiCodes.RED_FOREGROUND}{message}{AnsiCodes.RESET}"
                        )

                    if message.startswith("learn:"):
                        coords = message[6:].split(",")
                        if len(coords) == 2:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                self.target_position = [x, y]
                                self.learning_active = True
                                self.mode = self.Mode.LEARN
                                logger.info(f"Learning to reach target at ({x}, {y})")
                            except ValueError:
                                logger.error("Invalid coordinates for learning target")

                    elif message.startswith("seek goal:"):
                        coords = message[10:].split(",")
                        if len(coords) == 2:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                self.target_position = [x, y]
                                self.mode = self.Mode.SEEK_GOAL
                                self.learning_active = False
                                self.action_persistence = 0
                                self.current_persistent_action = None
                                self.target_reached_reported = False

                                # Get current position to compare with target
                                current_position = self.position
                                current_distance = calculate_distance(
                                    current_position, [x, y]
                                )

                                # Add a small delay before allowing target reached detection
                                # to prevent false positive at initialization
                                self.goal_seek_start_time = self.getTime()
                                self.allow_target_detection = False

                                logger.info(
                                    f"Seeking goal at ({x}, {y}) from distance {current_distance:.2f}"
                                )
                            except ValueError:
                                logger.error("Invalid coordinates for goal")

                    elif message.startswith("exploration:"):
                        try:
                            new_rate = float(message[12:])
                            self.q_agent.exploration_rate = new_rate
                        except ValueError:
                            logger.error("Invalid exploration rate")

                    elif message.startswith("persistence:"):
                        try:
                            new_persistence = int(message[12:])
                            self.action_persistence_duration = new_persistence
                            self.action_persistence = 0
                            logger.debug(
                                f"Action persistence updated to {new_persistence} steps"
                            )
                        except ValueError:
                            logger.error("Invalid persistence value")

                    elif message == "stop learn":
                        self.learning_active = False
                        self.mode = self.Mode.AVOID_OBSTACLES
                        logger.info("Learning mode stopped")
                    elif message == "learn":
                        self.learning_active = True
                        self.mode = self.Mode.LEARN
                        if (
                            not hasattr(self, "learn_mode_reported")
                            or not self.learn_mode_reported
                        ):
                            logger.info("Learning mode activated")
                            self.learn_mode_reported = True
                    elif message == "avoid obstacles":
                        self.mode = self.Mode.AVOID_OBSTACLES
                    elif message == "move forward":
                        self.mode = self.Mode.MOVE_FORWARD
                    elif message == "stop":
                        # Check if we're near the target before stopping
                        if (
                            self.mode == self.Mode.SEEK_GOAL
                            and self.target_position
                            and calculate_distance(self.position, self.target_position)
                            < RLConfig.TARGET_THRESHOLD
                            and not self.target_reached_reported
                        ):
                            logger.info("🎯 Target reached in SEEK_GOAL mode!")
                            self.target_reached_reported = True

                        self.mode = self.Mode.STOP
                        self.handle_reset()
                    elif message == "turn":
                        self.mode = self.Mode.TURN

            delta = (
                self.distanceSensors[0].getValue() - self.distanceSensors[1].getValue()
            )
            speeds = [0.0, 0.0]
            if self.mode == self.Mode.AVOID_OBSTACLES:
                speeds[0] = self.boundSpeed(self.maxSpeed / 2 + 0.1 * delta)
                speeds[1] = self.boundSpeed(self.maxSpeed / 2 - 0.1 * delta)
                left_sensor = self.distanceSensors[0].getValue()
                right_sensor = self.distanceSensors[1].getValue()
                if left_sensor > 800 and right_sensor > 800:
                    speeds = [-self.maxSpeed / 2, -self.maxSpeed / 2]
                elif left_sensor > 800:
                    speeds = [self.maxSpeed / 2, -self.maxSpeed / 3]
                elif right_sensor > 800:
                    speeds = [-self.maxSpeed / 3, self.maxSpeed / 2]

            elif self.mode == self.Mode.MOVE_FORWARD:
                speeds[0] = self.maxSpeed
                speeds[1] = self.maxSpeed
                left_sensor = self.distanceSensors[0].getValue()
                right_sensor = self.distanceSensors[1].getValue()
                if left_sensor > 800 or right_sensor > 800:
                    speeds = [0.0, 0.0]

            elif self.mode == self.Mode.TURN:
                speeds[0] = self.maxSpeed / 2
                speeds[1] = -self.maxSpeed / 2

            elif self.mode == self.Mode.STOP:
                speeds = [0.0, 0.0]

            elif self.mode == self.Mode.SEEK_GOAL and self.target_position:
                # Execute Q-learning policy for goal seeking
                state = self.get_discrete_state()
                current_distance = calculate_distance(
                    self.position, self.target_position
                )

                # Enable target detection after a short delay to prevent false detection at initialization
                if hasattr(self, "goal_seek_start_time") and not getattr(
                    self, "allow_target_detection", True
                ):
                    elapsed_time = self.getTime() - self.goal_seek_start_time
                    if elapsed_time > 1.0:
                        self.allow_target_detection = True
                        logger.debug(
                            "Target detection enabled after initialization delay"
                        )

                # Check if we've reached the target
                if current_distance < RLConfig.TARGET_THRESHOLD and getattr(
                    self, "allow_target_detection", True
                ):
                    if not self.target_reached_reported:
                        logger.info("🎯 Target reached in SEEK_GOAL mode!")
                        self.target_reached_reported = True
                    else:
                        speeds = [0.0, 0.0]
                else:
                    if 0.45 <= current_distance <= 0.55:
                        # Handle mid-range distances where the robot may stall
                        if not hasattr(self, "medium_distance_counter"):
                            self.medium_distance_counter = 0

                        self.medium_distance_counter += 1

                        # After being stuck at this distance for some time, try a more direct approach
                        if self.medium_distance_counter > 50:
                            # Calculate direct vector to target
                            dx = self.target_position[0] - self.position[0]
                            dy = self.target_position[1] - self.position[1]

                            # Apply direct vector movement for challenging mid-range positions
                            if abs(dx) > abs(dy):
                                # Move primarily along x-axis
                                if dx > 0:
                                    speeds = [self.maxSpeed * 0.7, self.maxSpeed * 0.7]
                                else:
                                    speeds = [
                                        -self.maxSpeed * 0.7,
                                        -self.maxSpeed * 0.7,
                                    ]
                            else:
                                # Move primarily along y-axis
                                if dy > 0:
                                    speeds = [self.maxSpeed * 0.7, self.maxSpeed * 0.7]
                                else:
                                    speeds = [
                                        -self.maxSpeed * 0.7,
                                        -self.maxSpeed * 0.7,
                                    ]

                            # Reset counter after applying direct movement
                            if self.medium_distance_counter > 70:
                                logger.debug(
                                    f"Breaking out of difficult distance region ({current_distance:.2f})"
                                )
                                self.medium_distance_counter = 0
                        else:
                            # Standard Q-learning based movement
                            action = self.q_agent.choose_best_action(
                                state, current_distance
                            )
                            speeds = self.q_agent.execute_action(action, state)
                    else:
                        # Reset counter when not in the difficult distance range
                        if hasattr(self, "medium_distance_counter"):
                            self.medium_distance_counter = 0

                        # Standard Q-learning based movement
                        action = self.q_agent.choose_best_action(
                            state, current_distance
                        )
                        speeds = self.q_agent.execute_action(action, state)

                # Update position tracking for stuck detection
                self.update_position_history()

            elif self.mode == self.Mode.LEARN:
                position = None
                if self.gps:
                    try:
                        position = self.gps.getValues()
                        if position and len(position) >= 2:
                            position = position[:2]
                    except Exception:
                        position = None

                if position is None:
                    position = self.position

                if position:
                    self.position = position

                if self.mode == self.Mode.LEARN and self.learning_active:
                    if self.current_state is None:
                        self.current_state = self.get_discrete_state()

                    current_distance = None
                    if self.target_position and self.position:
                        try:
                            current_distance = calculate_distance(
                                self.position, self.target_position
                            )
                        except Exception:
                            current_distance = None

                    if self.action_persistence == 0:
                        action = self.q_agent.choose_action(
                            self.current_state, current_distance
                        )
                        self.action_persistence = self.action_persistence_duration
                        self.current_persistent_action = action
                    else:
                        action = self.current_persistent_action
                        self.action_persistence -= 1

                    speeds = self.q_agent.execute_action(action, self.current_state)
                    self.last_action = action

            self.motors[0].setVelocity(speeds[0])
            self.motors[1].setVelocity(speeds[1])

            if self.step(self.timeStep) == -1:
                if self.learning_active and len(self.rewards_history) > 0:
                    self.save_learning_progress()
                    self.plot_rewards()
                logger.info("Robot controller exiting")
                self.q_agent.save_q_table(Q_TABLE_PATH)
                break

    def get_discrete_state(self):
        """Return discrete state tuple for RL agent."""
        if not self.position or not self.target_position:
            return None
        left_wheel_velocity = self.motors[0].getVelocity()
        right_wheel_velocity = self.motors[1].getVelocity()
        wheel_velocities = [left_wheel_velocity, right_wheel_velocity]
        left_sensor_value = self.distanceSensors[0].getValue()
        right_sensor_value = self.distanceSensors[1].getValue()

        # Check if current position is in stuck history
        current_position_key = self.get_position_key(self.position)
        if (
            current_position_key in self.stuck_positions
            and self.stuck_positions[current_position_key] > 2
        ):
            # Apply penalty in state to avoid revisiting stuck positions
            stuck_penalty = min(self.stuck_positions[current_position_key] * 0.5, 3.0)
            logger.debug(
                f"Position {current_position_key} has stuck penalty {stuck_penalty}"
            )

        return self.q_agent.get_discrete_state(
            self.position,
            self.target_position,
            self.orientation,
            left_sensor_value,
            right_sensor_value,
            wheel_velocities,
        )

    def get_position_key(self, position):
        """Convert position to grid cell key."""
        grid_size = RLConfig.POSITION_MEMORY_THRESHOLD
        return (round(position[0] / grid_size), round(position[1] / grid_size))

    def update_position_history(self):
        """Track position history and detect stuck patterns."""
        if not self.position:
            return

        current_key = self.get_position_key(self.position)

        # Add to position history
        self.position_history.append(current_key)

        # Check for repetitive patterns
        if len(self.position_history) >= 10:
            recent = list(self.position_history)[-10:]
            unique_positions = set(recent)

            # If we're cycling through the same few positions, mark them as stuck points
            if len(unique_positions) <= 3:
                for pos in unique_positions:
                    self.stuck_positions[pos] += 1
                    if (
                        self.stuck_positions[pos] >= 3
                        and self.getTime() - self.last_recovery_time > 5.0
                    ):
                        intensity = min(self.stuck_positions[pos] * 0.2, 1.0)
                        self.initiate_recovery(intensity)
                        self.last_recovery_time = self.getTime()
                        break

    def initiate_recovery(self, intensity=0.5):
        """Execute recovery strategies based on stuck history."""
        self.recovery_attempts += 1

        if self.recovery_attempts < 3:
            self.execute_simple_avoidance(intensity)
        elif self.recovery_attempts < 5:
            self.execute_perpendicular_movement(intensity)
        else:
            self.execute_random_exploration(intensity)
            # Reset recovery counter after random exploration
            if self.recovery_attempts >= 7:
                self.recovery_attempts = 0
                # Clear some stuck history to give a fresh start
                self.stuck_positions.clear()

    def execute_simple_avoidance(self, intensity):
        """Perform simple avoidance maneuver."""
        # Back up
        self.motors[0].setVelocity(-self.maxSpeed * 0.7 * intensity)
        self.motors[1].setVelocity(-self.maxSpeed * 0.7 * intensity)
        for _ in range(int(7 * intensity)):
            if self.step(self.timeStep) == -1:
                break

        # Turn in the less obstructed direction
        if self.distanceSensors[0].getValue() > self.distanceSensors[1].getValue():
            # Turn right
            self.motors[0].setVelocity(-self.maxSpeed * 0.7 * intensity)
            self.motors[1].setVelocity(self.maxSpeed * 0.7 * intensity)
        else:
            # Turn left
            self.motors[0].setVelocity(self.maxSpeed * 0.7 * intensity)
            self.motors[1].setVelocity(-self.maxSpeed * 0.7 * intensity)

        for _ in range(int(10 * intensity)):
            if self.step(self.timeStep) == -1:
                break

    def execute_perpendicular_movement(self, intensity):
        """Perform perpendicular movement relative to target."""
        if not self.target_position or not self.position:
            self.execute_random_exploration(intensity)
            return

        # Calculate vector to target
        dx = self.target_position[0] - self.position[0]
        dy = self.target_position[1] - self.position[1]

        # Calculate perpendicular direction (90 degrees rotation)
        perp_dx = -dy
        perp_dy = dx

        # Normalize
        magnitude = math.sqrt(perp_dx**2 + perp_dy**2)
        if magnitude > 0:
            perp_dx /= magnitude
            perp_dy /= magnitude

        # Determine which perpendicular direction to take (alternate based on attempts)
        if self.recovery_attempts % 2 == 0:
            perp_dx = -perp_dx
            perp_dy = -perp_dy

        # First back up a bit
        self.motors[0].setVelocity(-self.maxSpeed * 0.6 * intensity)
        self.motors[1].setVelocity(-self.maxSpeed * 0.6 * intensity)
        for _ in range(int(5 * intensity)):
            if self.step(self.timeStep) == -1:
                break

        # Then turn perpendicular
        angle_to_perp = math.atan2(perp_dy, perp_dx)

        # Simplified turn to approximate angle
        if angle_to_perp > 0:
            # Turn left
            self.motors[0].setVelocity(self.maxSpeed * 0.7 * intensity)
            self.motors[1].setVelocity(-self.maxSpeed * 0.7 * intensity)
        else:
            # Turn right
            self.motors[0].setVelocity(-self.maxSpeed * 0.7 * intensity)
            self.motors[1].setVelocity(self.maxSpeed * 0.7 * intensity)

        for _ in range(int(8 * intensity)):
            if self.step(self.timeStep) == -1:
                break

        # Move forward in new direction
        self.motors[0].setVelocity(self.maxSpeed * 0.8 * intensity)
        self.motors[1].setVelocity(self.maxSpeed * 0.8 * intensity)
        for _ in range(int(12 * intensity)):
            if self.step(self.timeStep) == -1:
                break

    def execute_random_exploration(self, intensity):
        """Perform random exploration to escape stuck situations."""
        # Backup in a random direction
        left_speed = -self.maxSpeed * (0.6 + self.rng.random() * 0.4) * intensity
        right_speed = -self.maxSpeed * (0.6 + self.rng.random() * 0.4) * intensity

        self.motors[0].setVelocity(left_speed)
        self.motors[1].setVelocity(right_speed)

        for _ in range(int(10 * intensity)):
            if self.step(self.timeStep) == -1:
                break

        # Make a random turn
        if self.rng.random() < 0.5:
            self.motors[0].setVelocity(self.maxSpeed * intensity)
            self.motors[1].setVelocity(-self.maxSpeed * intensity)
        else:
            self.motors[0].setVelocity(-self.maxSpeed * intensity)
            self.motors[1].setVelocity(self.maxSpeed * intensity)

        turn_duration = self.rng.randint(8, 15)
        for _ in range(int(turn_duration * intensity)):
            if self.step(self.timeStep) == -1:
                break

        # Move forward
        self.motors[0].setVelocity(self.maxSpeed * intensity)
        self.motors[1].setVelocity(self.maxSpeed * intensity)

        for _ in range(int(15 * intensity)):
            if self.step(self.timeStep) == -1:
                break

    def save_learning_progress(self):
        """Save learning progress to robot custom data."""
        if hasattr(self, "q_agent") and self.q_agent.q_table:
            data = f"learning_active:{self.learning_active},exploration:{self.q_agent.exploration_rate}"
            try:
                self.setCustomData(data)
                logger.info(f"Learning progress saved to robot: {data}")
            except Exception as e:
                logger.error(f"Could not save data: {e}")

    def send_q_table(self):
        """Log current Q-table size."""
        if not hasattr(self, "emitter"):
            return
        try:
            q_table_size = len(self.q_agent.q_table)
            logger.info(f"Q-table information: {q_table_size} states")
        except Exception as e:
            logger.error(f"Error with Q-table: {e}")

    def plot_rewards(self):
        """Plot and save reward history."""
        if not self.rewards_history:
            logger.warning("No rewards to plot")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards_history, label="Rewards", color="lightblue", alpha=0.3)
        if len(self.rewards_history) > 10:
            window = min(len(self.rewards_history) // 10, 50)
            window = max(window, 2)
            rewards_smoothed = np.convolve(
                self.rewards_history, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                range(len(rewards_smoothed)),
                rewards_smoothed,
                label="Smoothed Rewards",
                color="blue",
                linewidth=2,
            )

        plt.title(f"Learning Progress - {self.robot_name}")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(alpha=0.3)
        plt.legend()

        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            plot_path = os.path.join(DATA_DIR, f"{self.robot_name}_learning.png")
            plt.savefig(plot_path)
            logger.info(f"Learning plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
        plt.close()

    def handle_reset(self):
        """Stop motors and reset internal state."""
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        self.orientation = 0.0
        self.action_persistence = 0
        self.current_persistent_action = None
        for _ in range(3):
            if self.step(self.timeStep) == -1:
                break


if __name__ == "__main__":
    controller = Slave()
    controller.run()
