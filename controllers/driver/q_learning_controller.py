"""Manage Q-learning training sessions and goal seeking control flow."""

import math
from typing import List, Optional, Any
import logging
from common.config import (
    RLConfig,
    RobotConfig,
    SimulationConfig,
)
from common.rl_utils import calculate_distance, calculate_reward


class QLearningController:
    def __init__(self, driver: Any, logger: logging.Logger):
        """Initialize Q-learning controller with parameters and state.

        Args:
            driver: Driver instance for robot control.
            logger: Logger instance for logging events.
        """
        self.driver = driver
        self.logger = logger

        # State tracking for training episodes
        self.episode_count: int = 0
        self.max_episodes: int = RLConfig.MAX_EPISODES
        self.training_active: bool = False
        self.episode_step: int = 0
        self.max_steps: int = RLConfig.MAX_STEPS_PER_EPISODE

        # Reinforcement learning hyperparameters
        self.exploration_rate: float = RLConfig.EXPLORATION_RATE
        self.min_exploration_rate: float = RLConfig.MIN_EXPLORATION_RATE
        self.exploration_decay: float = RLConfig.EXPLORATION_DECAY

        # Episode target and start position management
        self.target_positions: List[List[float]] = RobotConfig.TARGET_POSITIONS
        self.start_positions: List[List[float]] = RobotConfig.START_POSITIONS
        self.current_target_index: int = 0
        self.current_start_index: int = 0

        # Metrics for performance and rewards
        self.successful_reaches: int = 0
        self.total_episodes_completed: int = 0
        self.rewards_history: List[float] = []
        self.episode_rewards: List[float] = []

        # Early stopping metrics
        self.convergence_window: int = RLConfig.CONVERGENCE_WINDOW
        self.convergence_counter: int = 0
        self.recent_success_rates: List[float] = []
        self.recent_rewards: List[float] = []
        self.best_success_rate: float = 0.0
        self.best_reward_avg: float = float("-inf")

        # State tracking for reward calculation
        self.previous_distance_to_target: Optional[float] = None

        # Control state for goal-seeking phase
        self.goal_seeking_active: bool = False
        self.goal_seeking_start_time: float = 0
        self.goal_reached: bool = False

        # DQN fields for eval and early stopping
        self.eval_rewards: List[float] = []
        self.next_eval_episode: int = RLConfig.EVAL_INTERVAL
        self.patience_counter: int = 0
        self.best_eval_reward: float = float("-inf")
        self.using_dqn: bool = False
        self.recent_td_losses: List[float] = []

    def start_learning(self, algorithm: str = "q_learning") -> None:
        """Start reinforcement learning training session.

        Args:
            algorithm (str): The RL algorithm to use ('q_learning' or 'dqn').
        """
        self.logger.info(f"Starting {algorithm.upper()} reinforcement learning...")
        self.training_active = True
        self.episode_count = 0
        self.reset_statistics()
        self.exploration_rate = RLConfig.EXPLORATION_RATE

        # Clear any old Q‑values on the slave
        self.driver.emitter.send("clear_q_table".encode("utf-8"))

        # Set the training mode on the slave
        self.driver.emitter.send(f"training_mode:{algorithm}".encode("utf-8"))

        # Ensure robot is stopped before first episode
        self.driver.emitter.send("stop".encode("utf-8"))
        self.driver.step(RobotConfig.TIME_STEP)

        # Track if we're using DQN for early stopping strategy
        self.using_dqn = algorithm == "dqn"
        self.next_eval_episode = RLConfig.EVAL_INTERVAL
        self.eval_rewards = []
        self.patience_counter = 0
        self.best_eval_reward = float("-inf")
        self.recent_td_losses = []

        self.start_new_episode()

    def reset_statistics(self) -> None:
        """Clear statistics for a new training session."""
        self.logger.info("Resetting learning statistics...")
        self.episode_rewards = []
        self.rewards_history = []
        self.successful_reaches = 0
        self.total_episodes_completed = 0
        self.episode_step = 0

    def calculate_reward(self, current_position: List[float]) -> float:
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

    def manage_training_step(self, position: List[float]) -> None:
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

        if (
            SimulationConfig.ENABLE_DETAILED_LOGGING
            and self.episode_step % SimulationConfig.DETAILED_LOG_FREQ == 0
        ):
            self.logger.info(
                f"Training: Episode {self.episode_count}, Step {self.episode_step}, Distance {current_distance:.2f}, Reward:{reward:.2f}"
            )

        # update previous distance for next reward computation
        self.previous_distance_to_target = current_distance

        # check for episode termination
        if self.check_episode_complete(current_distance):
            self.complete_episode()

    def start_new_episode(self) -> None:
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

    def check_episode_complete(self, current_distance: float) -> bool:
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

    def complete_episode(self) -> None:
        """Finalize the current episode and prepare for the next."""
        self.total_episodes_completed += 1

        total_reward = sum(self.episode_rewards)
        avg_reward = total_reward / max(1, len(self.episode_rewards))
        self.rewards_history.append(total_reward)

        self.logger.info(f"Episode {self.episode_count} completed")
        self.logger.info(
            f"Total reward: {total_reward:.2f}, Average reward: {avg_reward:.2f}"
        )

        # Check if it's time to run evaluation episodes
        should_run_eval = (
            self.using_dqn
            and self.episode_count >= self.next_eval_episode
            and RLConfig.ENABLE_EARLY_STOPPING
        )

        # Save a new "best" Q‑table when avg_reward improves
        if avg_reward > self.best_reward_avg:
            self.best_reward_avg = avg_reward
            self.logger.info(
                f"🎉 New best avg reward ({avg_reward:.2f}), saving best Q‑table"
            )
            self.driver.emitter.send("save_best_q_table".encode("utf-8"))

        success_rate = (self.successful_reaches / self.total_episodes_completed) * 100
        self.logger.info(
            f"Success rate: {success_rate:.1f}% ({self.successful_reaches}/{self.total_episodes_completed})"
        )

        # Run evaluation episodes if it's time
        if should_run_eval:
            self.run_evaluation_episodes()
            self.next_eval_episode = self.episode_count + RLConfig.EVAL_INTERVAL

        # Update metrics for early stopping
        if RLConfig.ENABLE_EARLY_STOPPING:
            if self.using_dqn:
                # For DQN, use evaluation metrics if available
                if self.eval_rewards:
                    avg_eval = sum(self.eval_rewards) / len(self.eval_rewards)
                    self.update_convergence_metrics(avg_eval, success_rate / 100.0)
            else:
                # For tabular Q-learning, use training metrics
                self.update_convergence_metrics(total_reward, success_rate / 100.0)

            # Check if training should stop based on convergence
            if self.check_convergence():
                self.logger.info("🚀 Training converged! Stopping early.")
                self.end_training()
                return

        # ε = ε_min + (ε_max − ε_min) * exp(−decay_rate * episode)
        # Exponential epsilon decay schedule
        epsilon_min = RLConfig.MIN_EXPLORATION_RATE
        epsilon_max = RLConfig.EXPLORATION_RATE
        decay_rate = RLConfig.EXPLORATION_DECAY_RATE
        new_rate = epsilon_min + (epsilon_max - epsilon_min) * math.exp(
            -decay_rate * self.episode_count
        )
        # clamp within [min, max]
        new_rate = max(epsilon_min, min(epsilon_max, new_rate))
        if abs(new_rate - self.exploration_rate) > 1e-6:
            self.exploration_rate = new_rate
            self.logger.info(f"Exploration rate updated to {self.exploration_rate:.3f}")
            self.driver.emitter.send(
                f"exploration:{self.exploration_rate}".encode("utf-8")
            )

        if self.episode_count >= self.max_episodes:
            self.logger.info("Reached maximum episode limit.")
            self.end_training()
            return

        self.start_new_episode()

    def run_evaluation_episodes(self) -> None:
        """Run pure greedy evaluation episodes to assess current policy."""
        self.logger.info(f"Running {RLConfig.EVAL_EPISODES} evaluation episodes...")

        # Tell the slave agent to use greedy policy (no exploration)
        self.driver.emitter.send("set_eval_mode:true".encode("utf-8"))

        eval_rewards = []
        eval_successes = 0
        eval_final_distances = []

        original_position = self.driver.translation_field.getSFVec3f()

        # Run evaluation episodes
        for i in range(RLConfig.EVAL_EPISODES):
            self.episode_rewards = []
            self.episode_step = 0

            # Use current target and reset robot
            target_position = self.target_positions[self.current_target_index]
            start_position = self.start_positions[self.current_start_index]
            self.driver.reset_robot_position(start_position)

            # Setup episode
            self.previous_distance_to_target = calculate_distance(
                start_position[:2], target_position
            )

            # Send learn command but with eval flag
            target_msg = f"eval:{target_position[0]},{target_position[1]}"
            self.driver.emitter.send(target_msg.encode("utf-8"))

            # Run steps until completion or timeout
            success = False
            for step in range(RLConfig.MAX_STEPS_PER_EPISODE):
                position = self.driver.translation_field.getSFVec3f()

                # Calculate reward but don't send to agent
                reward = self.calculate_reward(position)
                self.episode_rewards.append(reward)

                # Update previous distance for next reward calculation
                current_distance = calculate_distance(position[:2], target_position)
                self.previous_distance_to_target = current_distance

                # Check if target reached
                if current_distance < RLConfig.TARGET_THRESHOLD:
                    self.logger.info(f"🎯 Evaluation episode {i+1} reached target!")
                    success = True
                    eval_successes += 1
                    break

                # Run simulation step
                self.driver.step(RobotConfig.TIME_STEP)
                self.episode_step += 1

            # Collect episode results
            total_reward = sum(self.episode_rewards)
            eval_rewards.append(total_reward)
            eval_final_distances.append(current_distance)

            self.logger.info(
                f"Eval episode {i+1}: reward={total_reward:.2f}, success={success}, final_distance={current_distance:.2f}"
            )

        # Switch back to training mode
        self.driver.emitter.send("set_eval_mode:false".encode("utf-8"))

        # Calculate evaluation metrics
        avg_eval_reward = sum(eval_rewards) / RLConfig.EVAL_EPISODES
        eval_success_rate = (eval_successes / RLConfig.EVAL_EPISODES) * 100
        avg_final_distance = sum(eval_final_distances) / len(eval_final_distances)

        self.logger.info(
            f"Evaluation complete - Avg reward: {avg_eval_reward:.2f}, "
            f"Success rate: {eval_success_rate:.1f}%, "
            f"Avg final distance: {avg_final_distance:.2f}"
        )

        # Save the evaluation metrics
        self.eval_rewards = eval_rewards

        # Only proceed with current training if there was some success
        if eval_success_rate < 5.0 and self.episode_count > 300:
            self.logger.warning("Very low evaluation success rate. Consider restarting training.")

        # Collect TD loss if using DQN
        try:
            # Request TD loss from slave if receiver exists
            if hasattr(self.driver, "has_receiver") and self.driver.has_receiver:
                self.driver.emitter.send("get_td_loss".encode("utf-8"))
                # We'll process this in the driver's message handler
            else:
                # Without receiver, we can't get loss data - use a default
                self.logger.info("No receiver available, skipping TD loss collection")
        except Exception as e:
            self.logger.error(f"Error requesting TD loss: {e}")

        # Restore original position before continuing training
        self.driver.reset_robot_position(original_position)

    def update_convergence_metrics(
        self, total_reward: float, success_rate: float, td_loss: Optional[float] = None
    ) -> None:
        """Update metrics used to determine training convergence.

        Args:
            total_reward (float): Total reward from the episode
            success_rate (float): Success rate as a decimal (0.0-1.0)
            td_loss (Optional[float]): TD loss from the DQN agent
        """
        # Keep track of best metrics seen so far
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate

        # Add to recent metrics lists
        self.recent_rewards.append(total_reward)
        self.recent_success_rates.append(success_rate)

        # Track TD loss if provided (for DQN)
        if td_loss is not None:
            self.recent_td_losses.append(td_loss)
            # Keep only the most recent window of values
            if len(self.recent_td_losses) > RLConfig.CONVERGENCE_WINDOW:
                self.recent_td_losses.pop(0)

        # Keep only the most recent window of values
        if len(self.recent_rewards) > RLConfig.CONVERGENCE_WINDOW:
            self.recent_rewards.pop(0)
        if len(self.recent_success_rates) > RLConfig.CONVERGENCE_WINDOW:
            self.recent_success_rates.pop(0)

    def check_convergence(self) -> bool:
        """Check if training has converged based on performance metrics.

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Don't stop before minimum episodes
        if self.episode_count < RLConfig.MIN_EPISODES:
            return False

        # Need enough episodes to evaluate convergence
        if len(self.recent_rewards) < RLConfig.CONVERGENCE_WINDOW:
            return False

        # Different convergence criteria for DQN vs tabular Q-learning
        if self.using_dqn:
            return self.check_dqn_convergence()
        else:
            return self.check_tabular_convergence()

    def check_dqn_convergence(self) -> bool:
        """Check convergence specifically for DQN using evaluation metrics.

        Returns:
            bool: True if DQN training should stop, False otherwise
        """
        # Need evaluation data
        if not self.eval_rewards:
            return False

        # Minimum success rate check - if success rate is really bad, don't stop training
        if len(self.eval_rewards) >= RLConfig.EVAL_EPISODES:
            eval_success_count = sum(1 for r in self.eval_rewards if r > 150)  # Approx threshold for success
            eval_success_rate = eval_success_count / len(self.eval_rewards)

            # If success rate is below 20%, definitely don't stop yet
            if eval_success_rate < 0.2:
                self.logger.info(f"Evaluation success rate too low: {eval_success_rate:.2f}, continuing training")
                self.patience_counter = 0
                return False

        # Calculate average evaluation reward
        avg_eval = sum(self.eval_rewards) / len(self.eval_rewards)

        # Check if we've improved beyond the delta threshold
        if avg_eval > self.best_eval_reward + RLConfig.MIN_EPSILON_DELTA:
            self.best_eval_reward = avg_eval
            self.patience_counter = 0
            self.logger.info(f"New best eval reward: {avg_eval:.2f}, resetting patience")
            return False
        else:
            self.patience_counter += 1
            self.logger.info(
                f"No significant improvement: {self.patience_counter}/{RLConfig.PATIENCE}"
            )

        # Check TD loss plateau if we have loss data
        loss_plateau = False
        if len(self.recent_td_losses) >= RLConfig.CONVERGENCE_WINDOW:
            loss_start = sum(self.recent_td_losses[:10]) / 10
            loss_end = sum(self.recent_td_losses[-10:]) / 10
            loss_slope = abs(loss_end - loss_start)

            if loss_slope < RLConfig.LOSS_EPSILON:
                self.logger.info(f"TD loss has plateaued: slope={loss_slope:.5f}")
                loss_plateau = True
        else:
            # If we don't have TD loss data (no receiver), don't use it for convergence
            self.logger.debug("Insufficient TD loss data for plateau detection")

        # Return true if we've exhausted our patience
        # If we don't have TD loss data, only rely on the patience counter
        if not self.recent_td_losses:
            return self.patience_counter >= RLConfig.PATIENCE

        # Otherwise use both signals
        return (self.patience_counter >= RLConfig.PATIENCE) or (
            self.patience_counter >= RLConfig.PATIENCE // 2 and loss_plateau
        )

    def check_tabular_convergence(self) -> bool:
        """Check convergence for tabular Q-learning using original criteria.

        Returns:
            bool: True if tabular Q-learning should stop, False otherwise
        """
        # Check success rate convergence
        avg_success_rate = sum(self.recent_success_rates) / len(
            self.recent_success_rates
        )
        if avg_success_rate >= RLConfig.SUCCESS_RATE_THRESHOLD:
            self.logger.info(
                f"Success rate threshold met: {avg_success_rate:.2f} >= {RLConfig.SUCCESS_RATE_THRESHOLD}"
            )
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
            return False

        # Check reward improvement
        if len(self.rewards_history) >= self.convergence_window * 2:
            # Compare recent window to previous window
            recent_window = self.rewards_history[-self.convergence_window :]
            previous_window = self.rewards_history[
                -(self.convergence_window * 2) : -self.convergence_window
            ]

            recent_avg = sum(recent_window) / len(recent_window)
            previous_avg = sum(previous_window) / len(previous_window)

            # Calculate improvement percentage
            if previous_avg != 0:
                improvement = (recent_avg - previous_avg) / abs(previous_avg)

                # If improvement is minimal, count toward convergence
                if improvement < RLConfig.REWARD_IMPROVEMENT_THRESHOLD:
                    self.logger.info(
                        f"Reward improvement below threshold: {improvement:.2f} < {RLConfig.REWARD_IMPROVEMENT_THRESHOLD}"
                    )
                    self.convergence_counter += 1
                else:
                    # Still improving significantly, reset counter
                    self.logger.info(f"Still improving: {improvement:.2f}")
                    self.convergence_counter = 0
                    return False

        # Return true if we've confirmed convergence enough times
        return self.convergence_counter >= RLConfig.MAX_CONVERGENCE_ATTEMPTS

    def end_training(self) -> None:
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

    def start_goal_seeking(self) -> None:
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

        # Set tracking flags and record start time
        self.goal_seeking_active = True
        self.goal_seeking_start_time = self.driver.getTime()
        self.goal_reached = False

    def save_q_table(self) -> None:
        """Request the slave to persist the Q-table to disk."""
        try:
            self.driver.emitter.send("save_q_table".encode("utf-8"))
            self.logger.info("Requested slave to save Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table save: {e}")

    def load_q_table(self) -> None:
        """Request the slave to load the Q-table from storage."""
        try:
            self.driver.emitter.send("load_q_table".encode("utf-8"))
            self.logger.info("Requested slave to load Q-table")
        except Exception as e:
            self.logger.error(f"Error requesting Q-table load: {e}")
