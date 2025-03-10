import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten
from keras.optimizers import Adam
from minesweeper import Minesweeper
from DDQN import DoubleDQNAgent


def build_dqn(initial_lr):
    net = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(ROWDIM, COLDIM, 9)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(1, (1, 1), padding='same', activation='linear'),
        Flatten()
    ])
    net.compile(optimizer=Adam(learning_rate=initial_lr), loss='mse')
    return net


def format_timestamp():
    return datetime.now().strftime("%d-%b-%Y(%H:%M:%S)")


def get_avg_q_value(states, agent):
    return np.mean([np.amax(agent.online_network.predict(states), axis=1)])


def smooth_series(data, window=100):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window:] -= cumsum[:-window]
    return cumsum[window - 1:] / window


def visualize_progress(scores):
    avg_scores = smooth_series(scores, MOVING_AVE_WINDOW)
    plt.plot(scores, label='Episode Score')
    plt.plot(range(MOVING_AVE_WINDOW, len(scores) + 1), avg_scores, label=f'{MOVING_AVE_WINDOW} Ep Moving Avg')
    plt.axhline(y=SOLVE_CONDITION, color='r', linestyle='dashed', label='Goal')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title(f'Training Progress - {ENV_NAME}')
    plt.show()


def plot_q_trend(q_values):
    plt.plot(q_values, label='Avg Holdout Q')
    plt.xlabel('Epochs')
    plt.ylabel('Avg Q-Value')
    plt.title('Holdout State Q-Value Trend')
    plt.show()


ENV_NAME = 'Minesweeper'
ROWDIM, COLDIM, MINE_COUNT = 16, 30, 99
env = Minesweeper(ROWDIM, COLDIM, MINE_COUNT)
env.seed(1)

# Hyperparameters
LR_STAGES = [0.001, 0.0005, 0.00025, 0.000125, 0.0000625, 0.000025]
LR_MILESTONES = [0, 1e6, 3e6, 6e6, 10e6, 15e6]
GAMMA = 0.99
EPSILON_START, EPSILON_DECAY, EPSILON_MIN = 1.0, 0.99, 0.0
TAU = 1.0
BATCH_SIZE = 1024
MEMORY_CAP = BATCH_SIZE * 100
HOLDOUT_STATES = BATCH_SIZE
PER_ALPHA, PER_BETA_MIN, PER_BETA_MAX, PER_BETA_ANNEAL = 0.6, 0.4, 1.0, 50e6
PER_EPSILON = 0.01

agent_params = {
    'ROWDIM': ROWDIM, 'COLDIM': COLDIM,
    'LR_PIECEWISE': LR_STAGES, 'LR_DECAY_STEPS': LR_MILESTONES,
    'GAMMA': GAMMA, 'EPSILON_INITIAL': EPSILON_START,
    'EPSILON_DECAY': EPSILON_DECAY, 'EPSILON_MIN': EPSILON_MIN,
    'TAU': TAU, 'EXPERIENCE_REPLAY_BATCH_SIZE': BATCH_SIZE,
    'AGENT_MEMORY_LIMIT': MEMORY_CAP, 'NUM_HOLDOUT_STATES': HOLDOUT_STATES,
    'PER_ALPHA': PER_ALPHA, 'PER_BETA_MIN': PER_BETA_MIN,
    'PER_BETA_MAX': PER_BETA_MAX, 'PER_BETA_ANNEAL_STEPS': PER_BETA_ANNEAL,
    'PER_EPSILON': PER_EPSILON
}

# Training Configuration
trials = []
NUM_TRIALS = 1
MAX_EPISODES = 1_000_000
STEPS_PER_EP = ROWDIM * COLDIM - MINE_COUNT
SOLVE_CONDITION = 365
MOVING_AVE_WINDOW = 100
TRAIN_INTERVAL = BATCH_SIZE / 2
MIN_MEM_REQ = 2 * BATCH_SIZE
TARGET_UPDATE_INTERVAL = 80 * TRAIN_INTERVAL
HOLDOUT_UPDATE_INTERVAL = 200 * TRAIN_INTERVAL

for trial in range(NUM_TRIALS):
    net_online, net_target = build_dqn(LR_STAGES[0]), build_dqn(LR_STAGES[0])
    agent = DoubleDQNAgent(net_online, net_target, **agent_params)
    scores, holdout_q_values = [], []
    
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        for step in range(STEPS_PER_EP):
            action, net_input, _ = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done, net_input)
            state = next_state
            agent.steps += 1

            if agent.steps == HOLDOUT_STATES:
                holdout_q_values.append(get_avg_q_value(np.array(agent.holdout_states).squeeze(), agent))

            if agent.memory_length >= MIN_MEM_REQ:
                if agent.steps % TRAIN_INTERVAL == 0:
                    agent.experience_replay()
                if agent.steps % TARGET_UPDATE_INTERVAL == 0:
                    agent.update_target_network()
                if agent.steps % HOLDOUT_UPDATE_INTERVAL == 0:
                    holdout_q_values.append(get_avg_q_value(np.array(agent.holdout_states).squeeze(), agent))
                agent.update_beta()

            if done:
                break
        
        scores.append(env.score)
        if agent.memory_length >= MIN_MEM_REQ:
            agent.update_epsilon()
        
        avg_recent = np.mean(scores[-MOVING_AVE_WINDOW:])
        outcome = 'loss' if env.explosion else 'win'
        print(f'Trial {trial} Ep {episode} Score {env.score} ({outcome}) | Avg: {avg_recent:.2f}, Q: {holdout_q_values[-1]:.2f}, Epsilon: {agent.epsilon:.3f}, LR: {agent.lrate:.3E}')
        
        if len(scores) >= MOVING_AVE_WINDOW and avg_recent >= SOLVE_CONDITION:
            print(f'Trial {trial} solved in {episode} episodes!')
            agent.save_model_to_disk(ENV_NAME, str(episode), format_timestamp())
            break

    if avg_recent < SOLVE_CONDITION:
        agent.save_model_to_disk(ENV_NAME, str(episode), format_timestamp())

    trials.append(np.array(scores))
    visualize_progress(trials[trial])
    plot_q_trend(holdout_q_values)
