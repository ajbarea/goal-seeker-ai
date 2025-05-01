"""Provide robot control for obstacle avoidance, learning, and goal seeking."""

from common.rl_utils import calculate_distance, log_goal_reached
from controller import AnsiCodes, Robot  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import logging
import math
from common.config import (
    RLConfig,
    RobotConfig,
    SimulationConfig,
    get_logger,
    DATA_DIR,
    Q_TABLE_PATH,
)
from controllers.slave.q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent

# Configure module-level logger
logger = get_logger(
    __file__,
    level=getattr(logging, SimulationConfig.LOG_LEVEL_SLAVE, "INFO"),
)

# Define path for best Q-table
BEST_Q_TABLE_PATH = os.path.join(DATA_DIR, "best_q_table.pkl")


class Enumerate(object):
    """Provide a simple enumeration from space-separated names."""

    def __init__(self, names):
        for number, name in enumerate(names.split()):
            setattr(self, name, number)


class Slave(Robot):
    """Control robot modes for obstacle avoidance, learning, and goal seeking."""

    Mode = Enumerate(
        "STOP MOVE_FORWARD AVOID_OBSTACLES TURN SEEK_GOAL LEARN RANDOM_WALK"
    )
    timeStep = RobotConfig.TIME_STEP
    maxSpeed = RobotConfig.MAX_SPEED
    mode = Mode.AVOID_OBSTACLES

    def boundSpeed(self, speed):
        """Clamp speed to the allowed range.

        Args:
            speed (float): Desired motor speed.

        Returns:
            float: Clamped speed within [-maxSpeed, maxSpeed].
        """
        return max(-self.maxSpeed, min(self.maxSpeed, speed))

    def __init__(self):
        """Initialize device interfaces, sensors, and Q-learning agent."""
        super(Slave, self).__init__()
        self.motors = []
        self.distanceSensors = []

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

        # Initialize compass sensor for robot heading
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timeStep)

        # Position updates received from driver via emitter; remove GPS fallback
        self.position = [0, 0]
        self.orientation = 0.0

        # Instantiate RL agent
        self.q_agent = None
        self.set_training_mode("q_learning" if not RLConfig.USE_DQN else "dqn")

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

        logger.info(f"Slave robot initialization complete: {self.robot_name}")

    def run(self):
        """Process messages and update control modes each timestep."""
        while True:
            # update robot heading from compass
            compass_vals = self.compass.getValues()
            # compute yaw relative to north (x, z components)
            self.orientation = math.atan2(compass_vals[0], compass_vals[2])

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

                elif message == "save_q_table":
                    try:
                        self.q_agent.save_q_table(Q_TABLE_PATH)
                    except Exception as e:
                        logger.error(f"Error saving Q-table: {e}")

                elif message == "save_best_q_table":
                    try:
                        # If we have target-reaching success, save as best Q-table
                        if (
                            self.mode == self.Mode.LEARN
                            or self.mode == self.Mode.SEEK_GOAL
                        ) and self.target_reached_reported:
                            self.q_agent.save_q_table(BEST_Q_TABLE_PATH)
                            logger.info(f"Best Q-table saved to {BEST_Q_TABLE_PATH}")
                        else:
                            # Just save current as best if we don't have specific success criteria
                            self.q_agent.save_q_table(BEST_Q_TABLE_PATH)
                            logger.info(
                                f"Current Q-table saved as best to {BEST_Q_TABLE_PATH}"
                            )
                    except Exception as e:
                        logger.error(f"Error saving best Q-table: {e}")

                elif message == "load_q_table":
                    try:
                        self.q_agent.load_q_table(Q_TABLE_PATH)
                    except Exception as e:
                        logger.error(f"Error loading Q-table: {e}")

                elif message == "load_best_q_table":
                    try:
                        success = self.q_agent.load_q_table(BEST_Q_TABLE_PATH)
                        if success:
                            logger.info(f"Best Q-table loaded from {BEST_Q_TABLE_PATH}")
                        else:
                            # Fall back to regular Q-table if best doesn't exist
                            self.q_agent.load_q_table(Q_TABLE_PATH)
                            logger.info(
                                "Best Q-table not found, loaded regular Q-table instead"
                            )
                    except Exception as e:
                        logger.error(f"Error loading best Q-table: {e}")

                elif message == "clear_q_table":
                    self.q_agent.q_table.clear()
                    logger.info("Clearing Q‑table for new training...")

                elif message.startswith("training_mode:"):
                    mode = message[14:].strip()
                    self.set_training_mode(mode)

                elif message.startswith("set_eval_mode:"):
                    try:
                        eval_mode = message[14:] == "true"
                        if hasattr(self.q_agent, "set_eval_mode"):
                            self.q_agent.set_eval_mode(eval_mode)
                            logger.info(f"Set evaluation mode: {eval_mode}")
                    except Exception as e:
                        logger.error(f"Error setting eval mode: {e}")

                elif message == "get_td_loss":
                    try:
                        if hasattr(self.q_agent, "get_average_loss"):
                            avg_loss = self.q_agent.get_average_loss()
                            self.emitter.send(f"td_loss:{avg_loss}".encode("utf-8"))
                            logger.debug(f"Sent TD loss: {avg_loss}")
                    except Exception as e:
                        logger.error(f"Error getting TD loss: {e}")

                elif message.startswith("eval:"):
                    coords = message[5:].split(",")
                    if len(coords) == 2:
                        try:
                            x = float(coords[0])
                            y = float(coords[1])
                            self.target_position = [x, y]
                            self.mode = self.Mode.SEEK_GOAL
                            self.learning_active = False
                            logger.info(f"Evaluation episode for target at ({x}, {y})")
                        except ValueError:
                            logger.error("Invalid coordinates for evaluation target")

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

                    elif message == "avoid obstacles":
                        self.mode = self.Mode.AVOID_OBSTACLES
                    elif message == "move forward":
                        self.mode = self.Mode.MOVE_FORWARD
                    elif message == "stop":
                        if (
                            self.mode == self.Mode.SEEK_GOAL
                            and self.target_position
                            and log_goal_reached(
                                calculate_distance(self.position, self.target_position),
                                "SEEK_GOAL",
                                logger,
                            )
                        ):
                            self.target_reached_reported = True

                        self.mode = self.Mode.STOP
                        self.handle_reset()
                    elif message == "turn":
                        self.mode = self.Mode.TURN
                    elif message == "random walk":
                        self.mode = self.Mode.RANDOM_WALK

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
                action = self.q_agent.choose_best_action(state, current_distance)
                speeds = self.q_agent.execute_action(action, state)

            elif self.mode == self.Mode.LEARN:
                position = None

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

            elif self.mode == self.Mode.RANDOM_WALK:
                speeds = [
                    self.boundSpeed(self.rng.uniform(-self.maxSpeed, self.maxSpeed)),
                    self.boundSpeed(self.rng.uniform(-self.maxSpeed, self.maxSpeed)),
                ]

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
        """Get the discrete state tuple for the RL agent.

        Returns:
            Optional[Tuple]: Discrete state tuple or None if invalid input.
        """
        if not self.position or not self.target_position:
            return None
        left_wheel_velocity = self.motors[0].getVelocity()
        right_wheel_velocity = self.motors[1].getVelocity()
        wheel_velocities = [left_wheel_velocity, right_wheel_velocity]
        left_sensor_value = self.distanceSensors[0].getValue()
        right_sensor_value = self.distanceSensors[1].getValue()

        return self.q_agent.get_discrete_state(
            self.position,
            self.target_position,
            self.orientation,
            left_sensor_value,
            right_sensor_value,
            wheel_velocities,
        )

    def save_learning_progress(self):
        """Save the learning progress to the robot's custom data."""
        if hasattr(self, "q_agent") and self.q_agent.q_table:
            data = f"learning_active:{self.learning_active},exploration:{self.q_agent.exploration_rate}"
            try:
                self.setCustomData(data)
                logger.info(f"Learning progress saved to robot: {data}")
            except Exception as e:
                logger.error(f"Could not save data: {e}")

    def plot_rewards(self):
        """Plot and save the reward history."""
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
        """Stop the motors and reset the internal state."""
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        self.orientation = 0.0
        self.action_persistence = 0
        self.current_persistent_action = None
        for _ in range(3):
            if self.step(self.timeStep) == -1:
                break

    def set_training_mode(self, mode):
        """Set the training mode to Q-Learning or DQN.

        Args:
            mode (str): The training mode, either 'q_learning' or 'dqn'.
        """
        if mode == "q_learning":
            self.q_agent = QLearningAgent(
                learning_rate=RLConfig.LEARNING_RATE,
                min_learning_rate=RLConfig.MIN_LEARNING_RATE,
                discount_factor=RLConfig.DISCOUNT_FACTOR,
                min_discount_factor=RLConfig.MIN_DISCOUNT_FACTOR,
                exploration_rate=RLConfig.EXPLORATION_RATE,
                max_speed=self.maxSpeed,
            )
            logger.info("Training mode set to Q-Learning.")
        elif mode == "dqn":
            self.q_agent = DQNAgent(
                state_dim=5,
                action_dim=5,
                device="cpu",
                max_speed=self.maxSpeed,
            )
            logger.info("Training mode set to DQN.")
        else:
            logger.error(f"Invalid training mode: {mode}")


if __name__ == "__main__":
    controller = Slave()
    controller.run()
