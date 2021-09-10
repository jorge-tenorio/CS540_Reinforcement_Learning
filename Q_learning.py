import gym
import random
import numpy as np
import time
from collections import deque
import pickle

from collections import defaultdict

EPISODES = 20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0


if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)

    Q_table = defaultdict(default_Q_value)  # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)
    for i in range(EPISODES):
        episode_reward = 0
        # PERFORM Q LEARNING
        done = False
        state = env.reset()
        while not done:
            if np.random.random() > EPSILON:
                action = np.argmax((Q_table[state, 0], Q_table[state, 1], Q_table[state, 2], Q_table[state, 3]))
            else:
                action = np.random.randint(env.action_space.n)
            new_state, episode_reward, done, _ = env.step(action)
            current_q = Q_table[state, action]
            if not done:
                poss_actions = np.max((Q_table[new_state, 0], Q_table[new_state, 1], Q_table[new_state, 2], Q_table[new_state, 3]))
                max_future_q = episode_reward + DISCOUNT_FACTOR * poss_actions - current_q
                Q_table[state, action] = current_q + LEARNING_RATE * max_future_q
            else:
                done_q = episode_reward - current_q
                Q_table[state, action] = current_q + LEARNING_RATE * done_q
            state = new_state
        EPSILON *= EPSILON_DECAY
        episode_reward_record.append(episode_reward)
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
            print("EPSILON: " + str(EPSILON))

    model_file = open('Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    model_file.close()

