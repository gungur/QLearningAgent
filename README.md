# FrozenLake Q-Learning Agent

## Overview

This repository contains a Q-learning implementation for solving the FrozenLake-v1 environment from OpenAI's Gymnasium. The agent learns to navigate a frozen lake to reach the goal while avoiding holes.

## Repository Structure

```
.
├── Q_learning.py        # Main Q-learning implementation
├── Q_TABLE.pkl          # Serialized Q-table and epsilon value
└── metadata.yml         # Submission metadata
```

## Requirements

- Python 3.x
- Required packages:
  - `gymnasium`
  - `numpy`
  - `collections`

Install dependencies with:
```bash
pip install gymnasium numpy
```

## Implementation Details

### Key Parameters

- **EPISODES**: 20,000 training episodes
- **LEARNING_RATE (α)**: 0.1
- **DISCOUNT_FACTOR (γ)**: 0.99
- **Initial EPSILON (ε)**: 1.0 (100% exploration)
- **EPSILON_DECAY**: 0.999 (gradually reduces exploration)

### Algorithm

The implementation follows standard Q-learning with ε-greedy exploration:

1. Initializes a Q-table with default zero values
2. For each episode:
   - Observes the current state
   - Selects an action using ε-greedy policy
   - Performs the action and observes reward and next state
   - Updates Q-values using the Bellman equation:
     ```
     Q(s,a) ← (1-α)Q(s,a) + α[r + γ·max(Q(s',a'))]
     ```
   - Decays ε after each episode

### File Descriptions

1. **Q_learning.py**: Main implementation file containing:
   - Environment setup
   - Q-learning algorithm
   - Training loop
   - Performance tracking

2. **Q_TABLE.pkl**: Serialized Q-table and final epsilon value after training:
   - Contains the learned policy
   - Can be loaded for evaluation without retraining

3. **metadata.yml**: Submission metadata including:
   - Author information
   - Creation timestamp
   - Submission status
   - Performance score

## Usage

### Training the Agent

Run the training script:
```bash
python Q_learning.py
```

The script will:
1. Train the agent for 20,000 episodes
2. Print average rewards every 100 episodes
3. Save the final Q-table to Q_TABLE.pkl

### Monitoring Progress

During training, the script outputs:
- Average reward over the last 100 episodes
- Current ε value (exploration rate)

## Results

The implementation achieved a score of 100.0, indicating successful learning of the FrozenLake environment. The final Q-table represents the learned policy for navigating the lake.

## Customization

To modify the implementation:
- Adjust hyperparameters at the top of Q_learning.py
- Change the environment by modifying the `gym.envs.make()` call
- Alter exploration strategy by modifying the ε-greedy implementation
  
