# Training an AI agent using a Double Deep Q-Network to Play Minesweeper at Expert Difficulty

![Agent playing Minesweeper](img/Minesweeper_Agent_Playing.gif "Winning at Minesweeper!")

## Overview

I trained an AI agent using a Double Deep Q-Network (DDQN) to play Minesweeper at expert-level (16×30 grid with 99 mines). Over 25,000 games, the agent achieved a win rate of 23.6%, significantly surpassing the average human win rate of around 3%. The implementation primarily uses Python, Keras, and Tensorflow. The agent employs prioritized experience replay, precise state representations, and a specialized convolutional neural network architecture designed to preserve spatial locality, demonstrating  the feasibility and technical intricacies of applying reinforcement learning to a highly combinatorial game environment.

## Reinforcement Learning with DDQN

Rather than storing every possible state–action pair in a table, deep Q-learning uses a neural network to approximate Q-values. In Minesweeper, there are far too many combinations to handle tabularly. The agent interacts with an environment that follows the OpenAI Gym philosophy—methods like `step()`, `reset()`, and `render()` provide states and rewards, without embedding any agent-specific logic.

Standard Q-learning suffers from “maximization bias,” where stochastic rewards can mislead a single network into favoring suboptimal actions. To address this, I implemented two networks:

- **Online network** selects the action with the highest estimated Q-value.
- **Target network** evaluates the chosen action’s Q-value in the next state.

By updating only the online network’s weights during training and periodically copying them to the target network, the DDQN setup reduces bias and improves stability.

## Prioritized Experience Replay

Experience replay breaks up correlations in sequential data by sampling past transitions uniformly, but this can underweight rare, high-impact experiences—namely, the final winning or losing moves. To remedy this, I adopted proportional prioritized replay, which ranks transitions by their temporal-difference error and samples accordingly. I also applied importance-sampling weights to correct for the introduced bias. This approach:

- Speeds up learning past early plateaus
- Ensures that terminal transitions receive sufficient training attention

After switching from uniform to prioritized replay, the agent escaped a stubborn 5–10 % win-rate plateau within the first day of training.

## State and Action Representation

Instead of converting the Minesweeper board into images, I represent the grid as a NumPy integer array where values 0–8 correspond to revealed tile counts, 9 denotes an unrevealed tile, and –1 represents a mine (which immediately terminates the episode). For neural network input, each tile’s value is one-hot encoded into nine channels; unrevealed tiles are all zeros. This avoids misleading numeric relationships between unrevealed and numbered tiles. To improve efficiency, both the raw integer state and its one-hot encoding are stored in each replay entry, so the encoding step is not repeated on every sample.

Actions correspond to selecting one of 480 grid positions. Already-revealed tiles are masked out using NumPy’s masked arrays, ensuring that the agent only considers valid moves under an ε-greedy policy.

## Neural Network Architecture

My initial design included convolutional layers followed by fully connected (FC) layers, mirroring classic image classifiers like VGG or AlexNet. However, the FC layers introduced spurious connections between distant tiles, which hindered learning. The final architecture consists exclusively of convolutional layers:

1. Seven Conv2D layers with 64 filters of size 3×3, each followed by ReLU activation and “same” padding.
2. A final Conv2D layer with a single 1×1 filter and linear activation, producing a 16×30×1 output.
3. A flattening step converts this output to a 480-element Q-value vector.

This design preserves locality—each tile’s Q-value depends only on its 3×3 neighborhood—while keeping the total parameter count to 226 881.

## Hyperparameter Tuning

Optimizing every parameter over a ten-day training cycle is impractical, so I focused on those with the greatest impact:

- **Epsilon decay**: A decay rate of 0.99 per episode allowed enough exploration early on, helping the agent break through an average score plateau around 130 points.
- **Play-to-train ratio**: Instead of training every four steps (as in Atari), I increased batch size to 1024 and used a 2 : 1 ratio of gameplay steps to update steps, which accelerated convergence.
- **Learning rate schedule**: Trials showed that constant rates of 0.001 or 0.004 outperformed smaller values, but 0.004 alone caused score “dropouts.” A piecewise-linear decay starting at 0.004 and lowering over stages achieved over 300 points within 30 000 episodes.

## Mine Generation Bug

Early in development, I used two different `randint` conventions: one inclusive on both ends and one exclusive on the upper bound. This mistake prevented mines from ever appearing in the bottom row or right column. The agent quickly learned this quirk, inflating its initial performance. After correcting the bug, the win rate stabilized at 23.6 %—a more realistic but disappointing result.

## Results and Future Work

Compared to heuristic solvers that achieve 37–40 % win rates on expert difficulty, the DDQN agent underperforms. Possible improvements include:

- Providing the agent with the total mine count, mimicking the information available to human players.
- Allowing the network to learn its own first move rather than hard-coding it.
- Incorporating advanced techniques such as multi-step returns or distributional Q-learning, which have proven effective in Atari benchmarks.

Despite the challenges, this project demonstrates the feasibility of applying deep reinforcement learning to a highly stochastic, combinatorial task—and highlights the importance of careful environment design, bug checking, and targeted hyperparameter exploration.
