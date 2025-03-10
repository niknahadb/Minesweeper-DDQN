import numpy as np
from keras.models import load_model
from minesweeper import Minesweeper
from DDQN import DoubleDQNAgent
import time


def run_minesweeper(env, agent):
    win_count = 0
    for episode_idx in range(NUM_GAMES):
            state = env.reset()
            if GUI: env.render()
            for t in range(ROWDIM*COLDIM):
                action, _, valid_qvalues = agent.act(state)
                if GUI:
                    env.render(valid_qvalues)
                    time.sleep(MOVE_DELAY)
                state, reward, done = env.step(action)
                if done:
                    if GUI:
                        env.render(valid_qvalues)
                        time.sleep(MOVE_DELAY)
                        env.plot_minefield(action)
                    if not env.explosion: 
                        win_count += 1
                        result = 'win'
                    else:
                        result = 'loss'
                    print("Episode {} finished after {} moves with a score of {} ({}) WR = {:.2f}%"
                          .format(episode_idx+1, t+2, env.score, result, win_count/(episode_idx+1)*100))
                    if GUI: time.sleep(5)
                    break
    env.close()
    win_ratio = win_count / NUM_GAMES * 100
    print('The agent won {} out of {} games for a win ratio of {:.2f}%'
          .format(win_count, NUM_GAMES, win_ratio))


def init_agent():
    # Dummy params for DDQNAgent dict
    agent_kwargs = {
        'ROWDIM' : ROWDIM,
        'COLDIM' : COLDIM,
        'LR_PIECEWISE' : [1,0],
        'LR_DECAY_STEPS' : [0,1],
        'GAMMA' : 0, 
        'EPSILON_INITIAL' : 0, 
        'EPSILON_DECAY' : 0,
        'EPSILON_MIN' : 0,
        'TAU' : 1,
        'EXPERIENCE_REPLAY_BATCH_SIZE' : 0,
        'AGENT_MEMORY_LIMIT' : 1,
        'NUM_HOLDOUT_STATES' : 0,
        'PER_ALPHA' : 1,
        'PER_BETA_MIN' : 0,
        'PER_BETA_MAX' : 1,
        'PER_BETA_ANNEAL_STEPS' : 1,
        'PER_EPSILON' : 1,
        }
    agent = DoubleDQNAgent(ONLINE_NETWORK, ONLINE_NETWORK, **agent_kwargs)
    return agent
    

# game params
ROWDIM = 16
COLDIM = 30
MINE_COUNT = 99

ONLINE_NETWORK = load_model('model/Minesweeper_model_online_386924.h5')

MOVE_DELAY = 0.1
NUM_GAMES = 3 
GUI = True # Set to false to benchmark the agent rather than watch it play

agent = init_agent()
env = Minesweeper(ROWDIM, COLDIM, MINE_COUNT, gui=GUI)
env.seed(42)  
run_minesweeper(env, agent)