<div align="center">

# 🎯 Goal Seeker AI

### Q-learning robot for self-adaptive navigation in Webots simulation

*A reinforcement-learning agent that learns to reach goals while avoiding obstacles, with adaptive exploration, motor control, and reward shaping tuned by proximity to target.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Webots](https://img.shields.io/badge/Webots-simulator-blueviolet?style=flat-square)](https://cyberbotics.com)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=ajbarea_goal-seeker-ai&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=ajbarea_goal-seeker-ai)

</div>

---

## Project Overview

A robotics simulation demonstrating self-adaptive behavior through reinforcement learning. The system has three pieces:

- **Driver controller** — a Webots supervisor that manages the simulation, sends commands to robots, and orchestrates the RL training loop
- **Slave robot** — learns to navigate toward target positions while avoiding obstacles
- **Shared utilities** — common functions and algorithms used by both controllers

## Key Features

- **Q-learning implementation** — robot learns optimal navigation strategies through trial and error
- **Adaptive behavior** — performance improves over time as the agent accumulates experience
- **Distance-adaptive control** — motor speeds and action selection adjust based on proximity to target
- **Goal-seeking** — the robot can be directed to reach specific target positions
- **Obstacle avoidance** — integrated sensors allow detection and avoidance of obstacles
- **Performance tracking** — visual charts and statistics show learning progress
- **Multiple training scenarios** — training with different starting positions and target locations

## Project Structure

```text
goal-seeker-ai/
├── common/
│   ├── config.py                # Configuration parameters and path management
│   └── rl_utils.py              # Reinforcement Learning utilities
├── controllers/
│   ├── driver/                  # Supervisor controller
│   │   ├── driver.py            # Main supervisor code
│   │   └── q_learning_controller.py    # RL controller
│   └── slave/                   # Robot controller
│       ├── slave.py             # Robot control code
│       ├── q_learning_agent.py  # Q-learning implementation
│       └── dqn_agent.py         # Deep Q-Network implementation
├── data/                        # Saved Q-tables and learning plots
└── worlds/                      # Webots world files
```

## How to Run

1. Install Webots robotics simulator
2. Open the project world file in Webots
3. The simulation starts with the robot in obstacle-avoidance mode
4. Use keyboard commands (below) to drive the simulation

## Keyboard Controls

| Key | Action |
|-----|--------|
| **M** | Display help message |
| **A** | Switch to obstacle avoidance mode |
| **F** | Move forward |
| **S** | Stop |
| **T** | Turn |
| **R** | Reset robot position |
| **G** | Get current robot position |
| **Q** | Start Q-learning RL |
| **D** | Start DQN RL |

## Reinforcement Learning Implementation

The project uses **Q-learning**, a model-free reinforcement learning algorithm:

- **State space**: robot's position relative to target, sensor readings, and wheel velocities
- **Action space**: FORWARD, TURN_LEFT, TURN_RIGHT, BACKWARD, STOP
- **Reward function**: distance improvements, orientation, and proximity to target
- **Learning parameters**: adaptive exploration rate, learning rate, and discount factor based on distance to target

## Implementation Details

### State Representation

The robot's state is discretized into:

- Distance bins (adaptive granularity based on proximity to target)
- Angle bins (adaptive number based on distance)
- Obstacle detection (left and right sensors with graduated sensitivity)
- Velocity bins (tracking robot's current movement speed)

### Adaptive Behaviors

- **Distance-based motor control**: more precise when close to target, faster when far away
- **Exploration vs exploitation**: adaptive exploration rate that decreases over time
- **Action persistence**: dynamic action duration for smoother movement
- **Learning rate adaptation**: increases when closer to target for more precise learning
- **Discount factor adjustment**: adapts based on proximity to goal and reward significance
- **Random movements**: occasional random behaviors to escape local minima
- **Intelligent action selection**: distance-based logic chooses optimal actions when Q-table entries are new
- **Success tracking**: success rates measured across episodes to track improvement

### Visualization

Learning progress is visualized through matplotlib charts. Performance metrics include:

- Rewards over time
- Success rate (percentage of successful target reaches)
- Learning curve visualizations
- TD error tracking

## DQN Implementation Details

The project also implements Deep Q-Network (DQN) learning with the following features:

- **Network Architecture**: 2-layer neural network with ReLU activations
- **Replay Buffer**: Size 10000, stores state transitions for experience replay
- **Batch Learning**: Mini-batch size of 64 for stable training
- **Target Network**: Updated every 1000 steps for stable Q-learning
- **Exploration**: Epsilon annealed from 1.0 to 0.05 over 10000 steps
- **Optimization**: Adam optimizer with learning rate 1e-3
- **Discount Factor**: 0.99 for future reward weighting

## Dependencies

- Python 3.x
- Webots robotics simulator
- NumPy
- Matplotlib

## Running Training

1. Press **Q** to start Q-learning training, or **D** to start DQN training
2. The robot runs multiple training episodes
3. Learning statistics print to the console with emoji indicators for important events
4. After training completes, the robot automatically switches to goal-seeking mode
5. Performance plots are saved to the `data/` directory

## Performance Analysis

The repo includes tools to visualize and analyze the robot's learning performance:

- Q-table state coverage analysis
- Reward trends over time
- Success rate statistics
- Learning rate and discount factor adaptation statistics
- TD error analysis for learning quality assessment

## License

[MIT](LICENSE)

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://res.cloudinary.com/dumwa1w5x/image/upload/q_auto,f_auto,e_negate/v1779302138/brand_gwqy8l.png">
  <img src="https://res.cloudinary.com/dumwa1w5x/image/upload/q_auto,f_auto/v1779302138/brand_gwqy8l.png" alt="" height="16" />
</picture>&nbsp;&nbsp;2026 <a href="https://ajbarea.github.io/">AJ Barea</a>

</div>
