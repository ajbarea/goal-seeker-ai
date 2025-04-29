"""Manage Q-learning training sessions and goal seeking control flow."""

from common.config import (
    RLConfig,
    RobotConfig,
    SimulationConfig,
)
from common.rl_utils import calculate_distance, calculate_reward


class QLearningController:
    def __init__(self, driver, logger):
        """Initialize Q-learning controller with parameters and state.

        Args:
            driver: Driver instance for robot control.
            logger: Logger instance for logging events.
        """
        self.driver = driver
        self.logger = logger

        # State tracking for training episodes
        self.episode_count = 0
        self.max_episodes = RLConfig.MAX_EPISODES
        self.training_active = False
        self.episode_step = 0
        self.max_steps = RLConfig.MAX_STEPS_PER_EPISODE

        # Reinforcement learning hyperparameters
        self.exploration_rate = RLConfig.EXPLORATION_RATE
        self.min_exploration_rate = RLConfig.MIN_EXPLORATION_RATE
        self.exploration_decay = RLConfig.EXPLORATION_DECAY

        # Episode target and start position management
        self.target_positions = RobotConfig.TARGET_POSITIONS
        self.start_positions = RobotConfig.START_POSITIONS
        self.current_target_index = 0
        self.current_start_index = 0

        # Metrics for performance and rewards
        self.successful_reaches = 0
        self.total_episodes_completed = 0
        self.rewards_history = []
        self.episode_rewards = []

        # State tracking for reward calculation
        self.previous_distance_to_target = None

        # Control state for goal-seeking phase
        self.goal_seeking_active = False
        self.goal_seeking_start_time = 0
        self.goal_reached = False

    def start_learning(self):
        """Start reinforcement learning training session."""
        self.logger.info("Starting reinforcement learning")
        self.training_active = True
        self.episode_count = 0
        self.reset_statistics()
        self.exploration_rate = RLConfig.EXPLORATION_RATE

        # Clear any old Q‑values on the slave
        self.driver.emitter.send("clear_q_table".encode("utf-8"))

        # Ensure robot is stopped before first episode
        self.driver.emitter.send("stop".encode("utf-8"))
        self.driver.step(RobotConfig.TIME_STEP)

        self.start_new_episode()

    def reset_statistics(self):
        """Clear statistics for a new training session."""
        self.logger.info("Resetting learning statistics")
        self.episode_rewards = []
        self.rewards_history = []
        self.successful_reaches = 0
        self.total_episodes_completed = 0
        self.episode_step = 0

    def calculate_reward(self, current_position):
        """Calculate reward using distance progress and penalties.

        Args:
            current_position (List[float]): Current [x, y] position of the robot.

        Returns:
            float: Calculated reward value.
        """
        reward = calculate_reward(
            current_position[:2],
            self.driver.target_position,
            self.previous_distance_to_target,
            RLConfig.TARGET_THRESHOLD,
        )

        current_distance = calculate_distance(
            current_position[:2], self.driver.target_position
        )
        self.previous_distance_to_target = current_distance

        return reward

    def manage_training_step(self, position):
        """Handle one step of the training process.

        Args:
            position (List[float]): Current [x, y, z] position of the robot."""
        if not self.training_active:
            return

        current_distance = calculate_distance(position[:2], self.driver.target_position)

        if self.previous_distance_to_target is not None:
            reward = self.calculate_reward(position)
            # send reward to slave; policy and Q-table updates are handled in slave
            self.driver.emitter.send(f"reward:{reward}".encode("utf-8"))
            self.episode_rewards.append(reward)

        if SimulationConfig.ENABLE_DETAILED_LOGGING and self.episode_step % 75 == 0:
            self.logger.info(
                f"Training: Episode {self.episode_count}, Step {self.episode_step}, Distance {current_distance:.2f}, Reward:{reward:.2f}"
            )

        # update previous distance for next reward computation
        self.previous_distance_to_target = current_distance

        # check for episode termination
        if self.check_episode_complete(current_distance):
            self.complete_episode()

    def start_new_episode(self):
        """Set up a new episode for training."""
        self.episode_count += 1
        self.episode_step = 0
        self.episode_rewards = []

        self.current_target_index = (self.current_target_index + 1) % len(
            self.target_positions
        )
        self.current_start_index = (self.current_start_index + 1) % len(
            self.start_positions
        )

        target_position = self.target_positions[self.current_target_index]
        start_position = self.start_positions[self.current_start_index]

        self.driver.set_target_position(target_position)
        self.driver.reset_robot_position(start_position)

        self.previous_distance_to_target = calculate_distance(
            start_position[:2], target_position
        )

        self.logger.info(f"Starting episode {self.episode_count}/{self.max_episodes}")

        target_msg = f"learn:{target_position[0]},{target_position[1]}"
        self.driver.emitter.send(target_msg.encode("utf-8"))

    def check_episode_complete(self, current_distance):
        """Determine if the current episode has ended.

        Args:
            current_distance (float): Distance to the target.

        Returns:
            bool: True if the episode is complete, False otherwise.
        """
        self.episode_step += 1

        if current_distance < RLConfig.TARGET_THRESHOLD:
            self.logger.info("🎯 Target reached in LEARN mode!")
            self.successful_reaches += 1
            return True

        if self.episode_step >= self.max_steps:
            self.logger.info(f"💥 Episode timed out after {self.episode_step} steps")
            return True

        return False

    def complete_episode(self):
        """Finalize the current episode and prepare for the next."""
        self.total_episodes_completed += 1

        total_reward = sum(self.episode_rewards)
        avg_reward = total_reward / max(1, len(self.episode_rewards))
        self.rewards_history.append(total_reward)

        self.logger.info(f"Episode {self.episode_count} completed")
        self.logger.info(
            f"Total reward: {total_reward:.2f}, Average reward: {avg_reward:.2f}"
        )

        success_rate = (self.successful_reaches / self.total_episodes_completed) * 100
        self.logger.info(
            f"Success rate: {success_rate:.1f}% ({self.successful_reaches}/{self.total_episodes_completed})"
        )

        new_exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )
        if new_exploration_rate != self.exploration_rate:
            self.exploration_rate = new_exploration_rate
            self.logger.info(f"Exploration rate decayed to {self.exploration_rate:.3f}")
            self.driver.emitter.send(
                f"exploration:{self.exploration_rate}".encode("utf-8")
            )

        if self.episode_count >= self.max_episodes:
            self.end_training()
            return

        self.start_new_episode()

    def end_training(self):
        """Conclude training, save results, and switch to goal seeking."""
        self.training_active = False
        self.logger.info("Training complete")

        # Request Slave to persist current and best Q-tables
        try:
            self.driver.emitter.send("save_q_table".encode("utf-8"))
            self.logger.info("Requested Slave to save current Q-table")
            self.driver.emitter.send("save_best_q_table".encode("utf-8"))
            self.logger.info("Requested Slave to save best Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table persistence: {e}")

        # Show training plot
        self.driver.plot_training_results(self.rewards_history)

        # Now load the best table and switch to goal seeking
        self.logger.info("Starting post-training goal seeking with BEST Q‑table")
        self.start_goal_seeking()

    def start_goal_seeking(self):
        """Begin goal seeking using the trained Q-policy."""
        target_position = self.target_positions[self.current_target_index]

        start_position = self.start_positions[self.current_start_index]
        self.driver.reset_robot_position(start_position)

        self.driver.set_target_position(target_position)

        self.driver.emitter.send("stop".encode("utf-8"))
        for _ in range(5):
            self.driver.step(RobotConfig.TIME_STEP)

        self.driver.clear_pending_commands()

        # Load best policy
        self.driver.emitter.send("load_best_q_table".encode("utf-8"))

        seek_message = f"seek goal:{target_position[0]},{target_position[1]}"
        self.driver.emitter.send(seek_message.encode("utf-8"))

        for _ in range(3):
            self.driver.step(RobotConfig.TIME_STEP)

        self.logger.info(f"Goal seeking started. Target: {target_position}")

        self.goal_seeking_active = True
        self.goal_seeking_start_time = self.driver.getTime()
        self.goal_reached = False

    def save_q_table(self):
        """Request the slave to persist the Q-table to disk."""
        try:
            self.driver.emitter.send("save_q_table".encode("utf-8"))
            self.logger.info("Requested slave to save Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table save: {e}")

    def load_q_table(self):
        """Request the slave to load the Q-table from storage."""
        try:
            self.driver.emitter.send("load_q_table".encode("utf-8"))
            self.logger.info("Requested slave to load Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table load: {e}")
