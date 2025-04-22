"""Q-learning manager for training and goal seeking."""

from common.config import RLConfig, RobotConfig, SimulationConfig
from common.rl_utils import calculate_distance, calculate_reward, get_action_name
import random


class QLearningController:
    def __init__(self, driver, logger):
        """Init RL parameters and state."""
        self.driver = driver
        self.logger = logger

        # Training parameters
        self.episode_count = 0
        self.max_episodes = RLConfig.MAX_EPISODES
        self.training_active = False
        self.episode_step = 0
        self.max_steps = RLConfig.MAX_STEPS_PER_EPISODE

        # Learning parameters
        self.exploration_rate = RLConfig.EXPLORATION_RATE
        self.min_exploration_rate = RLConfig.MIN_EXPLORATION_RATE
        self.exploration_decay = RLConfig.EXPLORATION_DECAY

        # Target tracking
        self.target_positions = RobotConfig.TARGET_POSITIONS
        self.start_positions = RobotConfig.START_POSITIONS
        self.current_target_index = 0
        self.current_start_index = 0

        # Performance tracking
        self.successful_reaches = 0
        self.total_episodes_completed = 0
        self.rewards_history = []
        self.episode_rewards = []

        # Action state tracking
        self.last_action = None
        self.action_counter = 0
        self.action_persistence = RLConfig.ACTION_PERSISTENCE_INITIAL
        self.previous_distance_to_target = None

        # Goal seeking state
        self.goal_seeking_active = False
        self.goal_seeking_start_time = 0
        self.goal_reached = False

    def start_learning(self):
        """Begin training episodes."""
        self.logger.info("Starting learning process")
        self.training_active = True
        self.episode_count = 0
        self.reset_statistics()
        self.exploration_rate = RLConfig.EXPLORATION_RATE

        self.load_q_table()

        self.last_action = None
        self.action_counter = 0

        self.driver.emitter.send("stop".encode("utf-8"))
        self.driver.step(RobotConfig.TIME_STEP)

        target_position = self.target_positions[self.current_target_index]
        target_msg = f"learn:{target_position[0]},{target_position[1]}"
        self.driver.emitter.send(target_msg.encode("utf-8"))
        self.driver.step(RobotConfig.TIME_STEP)

        self.driver.emitter.send("start_learning".encode("utf-8"))
        self.start_new_episode()

    def reset_statistics(self):
        """Clear session stats."""
        self.logger.info("Resetting learning statistics")
        self.episode_rewards = []
        self.rewards_history = []
        self.successful_reaches = 0
        self.total_episodes_completed = 0
        self.episode_step = 0

    def calculate_reward(self, current_position):
        """Compute reward via centralized function."""
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
        """Process one training iteration."""
        if not self.training_active:
            return

        current_distance = calculate_distance(position[:2], self.driver.target_position)

        if self.previous_distance_to_target is not None:
            reward = self.calculate_reward(position)

            self.driver.emitter.send(f"reward:{reward}".encode("utf-8"))
            self.episode_rewards.append(reward)

            if SimulationConfig.ENABLE_DETAILED_LOGGING and self.episode_step % 50 == 0:
                self.logger.info(
                    f"Training: Episode {self.episode_count}, Step {self.episode_step}, Distance {current_distance:.2f}, Reward:{reward:.2f}"
                )

        if self.action_counter <= 0 or current_distance < RLConfig.TARGET_THRESHOLD:
            action = self.choose_action(current_distance)
            self.execute_action(action)
            self.last_action = action

            if action in [1, 2]:
                self.action_counter = max(1, self.action_persistence // 2)
            elif current_distance < 0.3:
                self.action_counter = max(1, self.action_persistence // 2)
            else:
                self.action_counter = self.action_persistence
        else:
            self.action_counter -= 1

        self.previous_distance_to_target = current_distance

        if self.check_episode_complete(current_distance):
            self.complete_episode()

    def choose_action(self, current_distance):
        """Epsilon-greedy action selection."""
        allow_stop = current_distance < RLConfig.TARGET_THRESHOLD
        action_indices = [0, 1, 2, 3]
        if allow_stop:
            action_indices.append(4)
        if random.random() < self.exploration_rate:
            return random.choice(action_indices)
        else:
            if allow_stop:
                return 4
            else:
                return 0

    def execute_action(self, action):
        """Emit action to slave."""
        action_msg = f"{RLConfig.ACTION_COMMAND_PREFIX}{action}"
        self.driver.emitter.send(action_msg.encode("utf-8"))

        if SimulationConfig.ENABLE_DETAILED_LOGGING and self.episode_step % 10 == 0:
            self.logger.debug(f"Executing action: {get_action_name(action)}")

    def start_new_episode(self):
        """Initialize new episode state."""
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

        self.last_action = None
        self.action_counter = 0

        self.logger.info(f"Starting episode {self.episode_count}/{self.max_episodes}")

        target_msg = f"learn:{target_position[0]},{target_position[1]}"
        self.driver.emitter.send(target_msg.encode("utf-8"))

        self.action_persistence = max(
            RLConfig.ACTION_PERSISTENCE_MIN,
            int(
                RLConfig.ACTION_PERSISTENCE_INITIAL
                * (RLConfig.ACTION_PERSISTENCE_DECAY**self.episode_count)
            ),
        )

    def check_episode_complete(self, current_distance):
        """Return True if episode ended."""
        self.episode_step += 1

        if current_distance < RLConfig.TARGET_THRESHOLD:
            self.logger.info(f"Target reached! Distance: {current_distance:.2f}")
            self.successful_reaches += 1
            return True

        if self.episode_step >= self.max_steps:
            self.logger.info(f"Episode timed out after {self.episode_step} steps")
            return True

        return False

    def complete_episode(self):
        """Finalize episode and prepare next."""
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
        """Terminate training, save results, and switch to goal seeking."""
        self.training_active = False
        self.logger.info("Training complete")

        success_rate = (self.successful_reaches / self.total_episodes_completed) * 100
        self.logger.info(
            f"Final success rate: {success_rate:.1f}% ({self.successful_reaches}/{self.total_episodes_completed})"
        )

        self.driver.emitter.send("stop".encode("utf-8"))

        for _ in range(3):
            self.driver.step(RobotConfig.TIME_STEP)

        self.save_q_table()

        self.driver.plot_training_results(self.rewards_history)

        self.logger.info("Starting post-training goal seeking...")
        self.start_goal_seeking()

    def start_goal_seeking(self):
        """Enter goal-seeking mode using learned policy."""
        target_position = self.target_positions[self.current_target_index]

        start_position = self.start_positions[self.current_start_index]
        self.driver.reset_robot_position(start_position)

        self.driver.set_target_position(target_position)

        self.driver.emitter.send("stop".encode("utf-8"))
        for _ in range(5):
            self.driver.step(RobotConfig.TIME_STEP)

        self.driver.clear_pending_commands()

        seek_message = f"seek goal:{target_position[0]},{target_position[1]}"
        self.driver.emitter.send(seek_message.encode("utf-8"))

        for _ in range(3):
            self.driver.step(RobotConfig.TIME_STEP)

        self.logger.info(f"Goal seeking started. Target: {target_position}")

        self.goal_seeking_active = True
        self.goal_seeking_start_time = self.driver.getTime()
        self.goal_reached = False

    def save_q_table(self):
        """Request slave to save Q-table."""
        try:
            self.driver.emitter.send("save_q_table".encode("utf-8"))
            self.logger.info("Requested slave to save Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table save: {e}")

    def load_q_table(self):
        """Request slave to load Q-table."""
        try:
            self.driver.emitter.send("load_q_table".encode("utf-8"))
            self.logger.info("Requested slave to load Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table load: {e}")
