"""Slave robot controller: obstacle avoidance, learning, and goal seeking."""

from controller import AnsiCodes, Robot  # type: ignore
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from common.config import RLConfig, RobotConfig, SimulationConfig, get_logger
from common.rl_utils import calculate_distance, get_action_name
from q_learning_agent import QLearningAgent

# Set up logger
logger = get_logger(
    __name__, level=getattr(logging, SimulationConfig.LOG_LEVEL_SLAVE, "INFO")
)


class Enumerate(object):
    """Simple enum creation from space-separated names."""

    def __init__(self, names):
        for number, name in enumerate(names.split()):
            setattr(self, name, number)


class Slave(Robot):
    """Main robot controller: mode handling and RL integration."""

    Mode = Enumerate("STOP MOVE_FORWARD AVOID_OBSTACLES TURN SEEK_GOAL LEARN")
    timeStep = RobotConfig.TIME_STEP
    maxSpeed = RobotConfig.MAX_SPEED
    mode = Mode.AVOID_OBSTACLES
    motors = []
    distanceSensors = []

    def boundSpeed(self, speed):
        """Clamp speed to allowed range."""
        return max(-self.maxSpeed, min(self.maxSpeed, speed))

    def __init__(self):
        """Init devices, sensors, and Q-learning agent."""
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
            logger.info("Using supervisor position updates (GPS not available)")
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

        logger.info(f"Slave robot initialization complete: {self.robot_name}")

    def run(self):
        """Per-step message handling, mode control, and motor updates."""
        logger.info("Starting slave robot control loop")

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
                        self.q_agent.load_q_table(SimulationConfig.Q_TABLE_PATH)
                        logger.info("Q-table loaded from file")
                    except Exception as e:
                        logger.error(f"Error loading Q-table: {e}")

                else:
                    if message.startswith(RLConfig.ACTION_COMMAND_PREFIX):
                        try:
                            current_action = int(message.split(":")[1])
                            action_name = get_action_name(current_action)
                        except (ValueError, IndexError):
                            logger.info(f"Received command: {message}")
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
                                logger.info(f"Seeking goal at ({x}, {y})")
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

            elif self.mode == self.Mode.SEEK_GOAL or self.mode == self.Mode.LEARN:
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
                        self.current_persistent_action = action
                        self.action_persistence = self.action_persistence_duration
                    else:
                        action = self.current_persistent_action
                        self.action_persistence -= 1

                    speeds = self.q_agent.execute_action(action)
                    self.last_action = action

                elif self.mode == self.Mode.SEEK_GOAL and self.target_position:
                    state = self.get_discrete_state()
                    current_distance = calculate_distance(
                        self.position, self.target_position
                    )

                    if current_distance < RLConfig.TARGET_THRESHOLD:
                        if not self.target_reached_reported:
                            logger.info("🎯 Target reached in SEEK_GOAL mode!")
                            self.target_reached_reported = True

                        speeds = [0.0, 0.0]
                        self.motors[0].setVelocity(0.0)
                        self.motors[1].setVelocity(0.0)
                    else:
                        if (
                            self.target_reached_reported
                            and current_distance > RLConfig.TARGET_THRESHOLD * 1.5
                        ):
                            self.target_reached_reported = False

                        action = self.q_agent.choose_best_action(
                            state, current_distance
                        )

                        if random.random() < 0.01:
                            action_name = get_action_name(action)
                            logger.debug(
                                f"Goal seeking: state={state}, action={action_name}, "
                                f"distance={current_distance:.2f}"
                            )

                        speeds = self.q_agent.execute_action(action)

                        left_sensor = self.distanceSensors[0].getValue()
                        right_sensor = self.distanceSensors[1].getValue()

                        if left_sensor > 800 and right_sensor > 800:
                            speeds = [-self.maxSpeed / 2, -self.maxSpeed / 2]
                        elif left_sensor > 800:
                            speeds = [self.maxSpeed / 2, -self.maxSpeed / 3]
                        elif right_sensor > 800:
                            speeds = [-self.maxSpeed / 3, self.maxSpeed / 2]

                else:
                    speeds[0] = self.boundSpeed(self.maxSpeed / 2 + 0.1 * delta)
                    speeds[1] = self.boundSpeed(self.maxSpeed / 2 - 0.1 * delta)

                    left_sensor = self.distanceSensors[0].getValue()
                    right_sensor = self.distanceSensors[1].getValue()

                    if left_sensor > 800 and right_sensor > 800:
                        speeds = [-self.maxSpeed / 2, -self.maxSpeed / 2]
                    elif left_sensor > 800:
                        speeds[0] = self.maxSpeed / 2
                        speeds[1] = -self.maxSpeed / 3
                    elif right_sensor > 800:
                        speeds[0] = -self.maxSpeed / 3
                        speeds[1] = self.maxSpeed / 2

            self.motors[0].setVelocity(speeds[0])
            self.motors[1].setVelocity(speeds[1])

            if self.step(self.timeStep) == -1:
                if self.learning_active and len(self.rewards_history) > 0:
                    self.save_learning_progress()
                    self.plot_rewards()

                self.q_agent.save_q_table(SimulationConfig.Q_TABLE_PATH)
                logger.info("Robot controller exiting")
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

        return self.q_agent.get_discrete_state(
            self.position,
            self.target_position,
            self.orientation,
            left_sensor_value,
            right_sensor_value,
            wheel_velocities,
        )

    def save_learning_progress(self):
        """Save learning flags to robot custom data."""
        if hasattr(self, "q_agent") and self.q_agent.q_table:
            data = f"learning_active:{self.learning_active},exploration:{self.q_agent.exploration_rate}"
            try:
                self.setCustomData(data)
                logger.info(f"Learning progress saved to robot: {data}")
            except Exception as e:
                logger.error(f"Could not save data: {e}")

    def send_q_table(self):
        """Log Q-table size."""
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
            plot_dir = SimulationConfig.PLOT_DIR
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{self.robot_name}_learning.png")
            plt.savefig(plot_path)
            logger.info(f"Learning plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
        plt.close()

    def handle_reset(self):
        """Stop motors and reset state."""
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
