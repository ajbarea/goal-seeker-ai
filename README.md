<!-- budget: 800 words -->
<div align="center">

# 🎯 Goal Seeker AI

### A Webots robot that learns to reach a goal with tabular Q-learning or a PyTorch DQN

*A supervisor runs training episodes; the robot picks one of five moves each step and learns from a progress reward, with optional potential-based reward shaping.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%E2%80%933.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Webots](https://img.shields.io/badge/Webots-R2023b%2B-blueviolet?style=flat-square)](https://cyberbotics.com)
[![CI](https://github.com/ajbarea/goal-seeker-ai/actions/workflows/python-app.yml/badge.svg)](https://github.com/ajbarea/goal-seeker-ai/actions/workflows/python-app.yml)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=ajbarea_goal-seeker-ai&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=ajbarea_goal-seeker-ai)

</div>

---

## What main contains

Two learners share one robot, one state encoding and one reward:

- **Tabular Q-learning** (`controllers/slave/q_learning_agent.py`): a Q-table over discretized states, with a learning rate and discount factor that adapt to the distance from the goal.
- **Deep Q-Network** (`controllers/slave/dqn_agent.py`): a PyTorch network (256, 256 and 128 hidden units) trained from a replay buffer as a Double DQN, with a target network, epsilon-greedy exploration, a greedy evaluation mode and early stopping on evaluation reward.
- **Potential-based reward shaping** (Ivan Rojas, `calculate_potential` in `common/rl_utils.py`): adds F = γΦ(s′) − Φ(s), with Φ = −distance to the goal, on top of the progress reward, the form of shaping from [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf). Their policy-invariance result assumes F on every transition, terminal steps included; here the goal step pays 200 without it. `RLConfig.USE_POTENTIAL_SHAPING` turns it on or off.

The **driver** (`controllers/driver/`) is a Webots supervisor: it places the robot and the target, runs episodes, computes each step's reward and sends it to the robot. The **slave** (`controllers/slave/`) reads its sensors, encodes the state, chooses an action and updates whichever agent is active.

## Project structure

```text
goal-seeker-ai/
├── common/
│   ├── config.py                    # Hyperparameters, paths and logging
│   └── rl_utils.py                  # State encoding, reward and shaping
├── controllers/
│   ├── driver/
│   │   ├── driver.py                # Supervisor and keyboard commands
│   │   └── q_learning_controller.py # Episodes, rewards, evaluation, early stopping
│   └── slave/
│       ├── slave.py                 # Robot controller
│       ├── q_learning_agent.py      # Tabular Q-learning
│       └── dqn_agent.py             # Deep Q-Network
├── data/                            # Learned models and plots (written at run time)
├── tests/                           # pytest suite; runs without Webots
└── worlds/search-and-rescue.wbt     # The Webots world
```

## Run it

1. Install [Webots](https://cyberbotics.com) (the world is saved in R2023b, the project file in R2025a) and [uv](https://docs.astral.sh/uv/).
2. `uv sync` creates `.venv` with NumPy, Matplotlib and CPU PyTorch.
3. Point Webots at that interpreter: Preferences, General, Python command, set to the absolute path of `.venv/bin/python` (`.venv\Scripts\python.exe` on Windows). Webots otherwise runs the first `python` on `PATH` ([Webots: Using Python](https://cyberbotics.com/doc/guide/using-python)).
4. Open `worlds/search-and-rescue.wbt`. The robot starts in obstacle-avoidance mode; click the 3D view and use the keys below.

| Key | Action |
|-----|--------|
| **M** | Show help |
| **A** | Obstacle-avoidance mode |
| **F** | Move forward |
| **S** | Stop |
| **T** | Turn |
| **R** | Reset the robot's position |
| **G** | Print the robot's position |
| **Q** | Train with Q-learning |
| **D** | Train with the DQN |

Training runs episodes until early stopping or `RLConfig.MAX_EPISODES`, then switches to goal-seeking with what it learned. Each agent keeps its own model files in `data/`: `q_table.pkl` and `best_q_table.pkl` for Q-learning, `dqn_model.pt` and `best_dqn_model.pt` for the DQN. Reward plots are saved there too.

## How it learns

**State.** Five discrete values: distance to the goal (7 bins, finer near the goal), heading to the goal (8 bins), left and right obstacle sensors (4 levels each), and wheel motion (stopped, slow, fast, backward or turning).

**Actions.** Forward, turn left, turn right, backward, stop.

**Reward.** Reaching the goal (within 0.15 m) pays 200. Otherwise the reward is the step's progress, previous distance − γ × current distance, scaled by the most the robot can move in one step, clipped to ±1 and multiplied by 3, plus 0.1 when that progress is positive, minus a 0.02 step penalty. `calculate_reward` can also subtract a 1.0 collision penalty from sensor readings, but the supervisor does not pass them, so training runs without it. With shaping on, F is added with γ = `GAMMA` (0.99) for the DQN and `DISCOUNT_FACTOR` (0.99) for Q-learning, whose agent then adapts its own discount per state.

**DQN settings** (`RLConfig`): replay buffer 50,000, batch 128, γ 0.99, Adam at 5e-4, target network copied every 500 steps, epsilon from 1.0 to 0.05 over 20,000 steps, gradients clipped at norm 10. Every 20 episodes it runs 10 greedy evaluation episodes. From episode 150, after every episode it compares the latest evaluation average with its best: while fewer than 20% of evaluation episodes score above 150 it keeps training, and otherwise it stops after 5 checks in a row without a new best. The average changes only at an evaluation, so training ends about 5 episodes after the first evaluation that clears the 20% bar.

## Tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The suite covers what runs without the simulator: the state encoding, the reward and its shaping term, the supervisor's training and evaluation episodes against a stand-in driver, the robot controller against a stand-in Webots `controller` module, and both agents on CPU (Q-table updates, DQN training on a toy task, checkpoint round trips). CI runs it on pushes and pull requests to main.

## Credits

AJ Barea wrote the controllers, both agents and the reward. Ivan Rojas added potential-based reward shaping and the training run in `data/training_results.png`, made with his shaping over the earlier reward.

## License

[MIT](LICENSE)

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://res.cloudinary.com/dumwa1w5x/image/upload/q_auto,f_auto,e_negate/v1779302138/brand_gwqy8l.png">
  <img src="https://res.cloudinary.com/dumwa1w5x/image/upload/q_auto,f_auto/v1779302138/brand_gwqy8l.png" alt="" height="16" />
</picture>&nbsp;&nbsp;2026 <a href="https://ajbarea.github.io/">AJ Barea</a>

</div>
