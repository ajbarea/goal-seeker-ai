"""Manage robot simulation, reinforcement learning, goal seeking, and manual commands."""

from controller import Supervisor  # type: ignore
import logging
import os
from common.rl_utils import calculate_distance, plot_q_learning_progress
from common.config import (
    SimulationConfig,
    RobotConfig,
    RLConfig,
    get_logger,
    DATA_DIR,
)
from q_learning_controller import QLearningController


class Driver(Supervisor):
    TIME_STEP = RobotConfig.TIME_STEP

    def __init__(self):
        """Initialize devices, logger, and RL controller."""
        super(Driver, self).__init__()

        # Initialize logger and interface devices
        self.logger = get_logger(
            __name__, level=getattr(logging, SimulationConfig.LOG_LEVEL_DRIVER, "INFO")
        )
        self.emitter = self.getDevice("emitter")
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.TIME_STEP)

        # Retrieve robot node and translation field
        self.robot = self.getFromDef("ROBOT1")
        self.translation_field = self.robot.getField("translation")

        # Initialize target tracking state
        self.target_position = None
        self.previous_distance_to_target = None

        # Initialize Q-learning controller
        self.rl_controller = QLearningController(self, self.logger)

        # Initialize simulation step counter
        self.step_counter = 0

        # Log initialization status
        self.logger.info("Driver initialization complete")
        self.logger.info("Press 'I' for help")

    def run(self):
        """Run the simulation loop to handle RL training, goal seeking, and manual inputs."""
        self.display_help()
        previous_message = ""

        while True:
            # Increment simulation step counter
            self.step_counter += 1

            # Periodically send robot position updates
            if self.step_counter % SimulationConfig.POSITION_UPDATE_FREQ == 0:
                position = self.translation_field.getSFVec3f()
                pos_message = f"position:{position[0]},{position[1]}"
                self.emitter.send(pos_message.encode("utf-8"))

            # Handle active RL training steps
            if self.rl_controller.training_active:
                position = self.translation_field.getSFVec3f()
                self.rl_controller.manage_training_step(position)

            # Handle active goal seeking behavior
            elif (
                hasattr(self.rl_controller, "goal_seeking_active")
                and self.rl_controller.goal_seeking_active
            ):
                position = self.translation_field.getSFVec3f()
                self.monitor_goal_seeking(position)

            # Process manual keyboard commands
            k = self.keyboard.getKey()
            message = ""

            if k == ord("A"):
                message = "avoid obstacles"
            elif k == ord("F"):
                message = "move forward"
            elif k == ord("S"):
                message = "stop"
            elif k == ord("T"):
                message = "turn"
            elif k == ord("G"):
                position = self.translation_field.getSFVec3f()
                self.logger.info(
                    f"ROBOT1 is located at ({position[0]:.2f}, {position[1]:.2f})"
                )
            elif k == ord("R"):
                self.safely_reset_robot()
            elif k == ord("I"):
                self.display_help()
            elif k == ord("L"):
                self.rl_controller.start_learning()

            if message and message != previous_message:
                previous_message = message
                self.logger.info(f"Command: {message}")
                self.emitter.send(message.encode("utf-8"))

            if self.step(self.TIME_STEP) == -1:
                # Save Q-table and terminate simulation
                self.rl_controller.save_q_table()
                break

    def clear_pending_commands(self):
        """Advance simulation steps to clear pending commands."""
        # Just step the simulation a few times without sending commands
        for _ in range(5):
            self.step(self.TIME_STEP)
        return

    def monitor_goal_seeking(self, position):
        """Monitor goal seeking progress, detect stuck conditions, and enforce timeout."""
        if not self.target_position:
            return

        # Compute current distance to the target
        current_distance = calculate_distance(position[:2], self.target_position)

        # Check for target reached condition
        if current_distance < RLConfig.TARGET_THRESHOLD:
            if not getattr(self.rl_controller, "goal_reached", False):
                self.rl_controller.goal_reached = True

                # Send stop command
                self.emitter.send("stop".encode("utf-8"))
                self.step(self.TIME_STEP)

                # Log time taken to reach the goal
                elapsed_time = (
                    self.getTime() - self.rl_controller.goal_seeking_start_time
                )
                self.logger.info(f"Goal reached in {elapsed_time:.1f} seconds")

                # Disable goal seeking to prevent further timeout checks
                self.rl_controller.goal_seeking_active = False
                return
            else:
                # Target already reached; skip further actions
                return

        # Check for goal seeking timeout
        current_time = self.getTime()
        elapsed_time = current_time - self.rl_controller.goal_seeking_start_time

        # Periodically log goal seeking progress
        if self.step_counter % 100 == 0:
            # Avoid logging when very close to the target
            if current_distance < RLConfig.TARGET_THRESHOLD * 1.2:
                return

            self.logger.info(
                f"Goal seeking in progress - Distance: {current_distance:.2f}, "
                f"Time elapsed: {elapsed_time:.1f}s"
            )

            # Detect if robot is stuck based on minimal progress
            if hasattr(self, "last_goal_seeking_distance"):
                if abs(current_distance - self.last_goal_seeking_distance) < 0.05:
                    self.stuck_counter = getattr(self, "stuck_counter", 0) + 1
                    if (
                        self.stuck_counter >= SimulationConfig.STUCK_THRESHOLD
                    ):  # Stuck for threshold number of checks
                        self.logger.info(
                            f"🤔 Robot stuck at distance {current_distance:.2f}. Attempting recovery..."
                        )
                        # Attempt repositioning; randomize movements on repeated failures
                        if self.stuck_counter in (
                            SimulationConfig.STUCK_THRESHOLD,
                            SimulationConfig.STUCK_THRESHOLD + 2,
                        ):
                            self.emitter.send("reposition".encode("utf-8"))
                            self.stuck_counter += 1
                        else:
                            self.logger.info(
                                f"🔄 Reposition failed; issue random movement command at distance {current_distance:.2f}."
                            )
                            self.emitter.send("randomize".encode("utf-8"))
                            self.stuck_counter = 0
                else:
                    self.stuck_counter = 0

            # Update recorded distance for next check
            self.last_goal_seeking_distance = current_distance

        # Enforce maximum goal seeking duration
        if elapsed_time > SimulationConfig.GOAL_SEEKING_TIMEOUT:
            self.logger.info(
                f"💥 Mission failed! Robot got distracted and timed out after {elapsed_time:.1f} seconds."
            )
            self.rl_controller.goal_seeking_active = False
            self.emitter.send("stop".encode("utf-8"))

    def display_help(self):
        """Log available keyboard commands."""
        self.logger.info(
            "\nCommands:\n"
            " I - Display this help message\n"
            " A - Avoid obstacles mode\n"
            " F - Move forward\n"
            " S - Stop\n"
            " T - Turn\n"
            " R - Reset robot position\n"
            " G - Get (x,y) position of ROBOT1\n"
            " L - Start reinforcement learning"
        )

    def safely_reset_robot(self):
        """Reset robot to default position and resume obstacle avoidance."""
        self.emitter.send("stop".encode("utf-8"))
        self.step(self.TIME_STEP)
        self.robot.resetPhysics()
        self.translation_field.setSFVec3f(RobotConfig.DEFAULT_POSITION)
        for _ in range(5):
            self.step(self.TIME_STEP)

        # Resume obstacle avoidance mode after reset
        self.emitter.send("avoid obstacles".encode("utf-8"))
        self.step(self.TIME_STEP)

        self.logger.info(
            "Robot reset to default position and set to avoid obstacles mode"
        )

    def reset_robot_position(self, position):
        """Reset robot to specified position with random offset and reset physics."""
        # Add small random offset for variability
        import random

        random_offset_x = random.uniform(-0.03, 0.03)
        random_offset_y = random.uniform(-0.03, 0.03)
        randomized_position = [
            position[0] + random_offset_x,
            position[1] + random_offset_y,
            position[2],
        ]

        # Send stop command if not already in training mode
        if not getattr(self.rl_controller, "training_active", False):
            self.emitter.send("stop".encode("utf-8"))
            for _ in range(3):  # Multiple steps to ensure stop is processed
                self.step(self.TIME_STEP)

        # Reset orientation to upright
        rotation_field = self.robot.getField("rotation")
        if rotation_field:
            rotation_field.setSFRotation([0, 1, 0, 0])

        # Reset position and physics
        self.translation_field.setSFVec3f(randomized_position)
        self.robot.resetPhysics()

        # Reset velocities if fields exist
        try:
            velocity_field = self.robot.getField("velocity")
            if velocity_field:
                velocity_field.setSFVec3f([0, 0, 0])
            angular_velocity_field = self.robot.getField("angularVelocity")
            if angular_velocity_field:
                angular_velocity_field.setSFVec3f([0, 0, 0])
        except Exception:
            pass  # Fields might not exist

        # Give more time to stabilize
        for _ in range(5):
            self.step(self.TIME_STEP)

        # Initialize with obstacle avoidance before learning,
        # but only if not already in training mode
        if not getattr(self.rl_controller, "training_active", False):
            self.emitter.send("avoid obstacles".encode("utf-8"))
            self.step(self.TIME_STEP * 2)

        self.logger.debug(f"Robot reset to position: {randomized_position}")

    def set_target_position(self, target_position):
        """Set the target position."""
        self.target_position = target_position

    def plot_training_results(self, rewards):
        """Plot and save training reward history."""
        if not rewards:
            self.logger.warning("No rewards to plot")
            return

        try:
            # Ensure data directory exists
            os.makedirs(DATA_DIR, exist_ok=True)

            # Generate episode index list
            episodes = list(range(1, len(rewards) + 1))

            # Verify matching lengths for episodes and rewards
            if len(episodes) != len(rewards):
                self.logger.warning(
                    f"Length mismatch: episodes({len(episodes)}) != rewards({len(rewards)})"
                )
                # Truncate to shorter length to ensure they match
                min_len = min(len(episodes), len(rewards))
                episodes = episodes[:min_len]
                rewards = rewards[:min_len]

            # Generate reward progression plot
            plot_q_learning_progress(
                rewards=rewards,
                filename="training_results",
                save_dir=DATA_DIR,
            )

            self.logger.info(
                f"Training results plotted to {DATA_DIR}\\training_results.png"
            )
        except Exception as e:
            self.logger.error(f"Error plotting training results: {e}")
            self.logger.error(
                f"Episodes length: {len(episodes)}, Rewards length: {len(rewards)}"
            )


# Main entry point
if __name__ == "__main__":
    Driver().run()
