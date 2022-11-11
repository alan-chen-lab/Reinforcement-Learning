# Slide Example for ESARSA (RL-Course NTNU, Saeedvand)

import gym
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# , render_mode="human") # FrozenLake-v1, CliffWalking-v0
env = gym.make('CliffWalking-v0')


def plot(rewards):
    plt.figure(2)
    plt.title('Aveage Reward ESARSA')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.plot(rewards, color='green', label='Reward')
    plt.grid(axis='x', color='0.80')
    plt.legend(title='Parameter where:')
    plt.show()


def Q_value_initialize(state, action, type=0):
    if type == 1:
        return np.ones((state, action))
    elif type == 0:
        return np.zeros((state, action))
    elif type == -1:
        return np.random.random((state, action))


def epsilon_greedy(Q, epsilon, s):
    if np.random.rand() < epsilon:
        action = np.argmax(Q[s, :]).item()
    else:
        action = env.action_space.sample()

    return action


def normalize(list):
    xmin = min(list)
    xmax = max(list)
    for i, x in enumerate(list):
        list[i] = (x-xmin) / (xmax-xmin)
    return list


def Expected_SARSA(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, EPS_DECAY, n_tests):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type=0)
    timestep_reward = []
    for episode in range(episodes):
        print(f"Episode: {episode}")
        s, info = env.reset()  # read also state

        epsilon_threshold = EPS_END + \
            (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)

        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a = epsilon_greedy(Q, epsilon_threshold, s)

            # Reward
            # -1 per step unless other reward is triggered.
            # +20 delivering passenger.
            # -10 executing “pickup” and “drop-off” actions illegally.
            s_, reward, terminated, truncated, info = env.step(a)

            # if reward == -100:
            #     reward = -5

            #s_, reward, done, info = env.step(a)
            total_reward += reward

            ################ESARSA difference################
            # no need for next action

            if terminated or truncated:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                ################ESARSA difference################
                expected_value = np.mean(Q[s_, :])

                Q[s, a] += alpha * \
                    (reward + (gamma * expected_value) - Q[s, a])

            s = s_

            if terminated or truncated:
                s, info = env.reset()

        timestep_reward.append(total_reward/t)

        print(f"Episode: {episode}, steps: {t}, reward: {total_reward}")
        print(Q[s, :])

        # print(f"Q values:\n{Q}\nTesting now:")
    # Test policy (no learning)
    if n_tests > 0:
        test_agent(Q, n_tests)

    plot(timestep_reward)
    plot(normalize(timestep_reward))
    return timestep_reward

# ----------------------------------------------------


def test_agent(Q, n_tests=0, delay=1):
    env = gym.make('CliffWalking-v0', render_mode="human")
    for testing in range(n_tests):
        print(f"Test #{testing}")
        s, info = env.reset()
        steps = 0
        while True:
            steps += 1  # to stop the loop if algorithm has not been trained
            time.sleep(delay)
            a = np.argmax(Q[s, :]).item()
            print(f"Chose action {a} for state {s}")
            s, reward, terminated, truncated, info = env.step(a)
            # time.sleep(1)

            if terminated or truncated:
                print("Finished!", reward)
                time.sleep(5)
                break


if __name__ == "__main__":
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor
    # epsilon greedy exploration-explotation (smaller more random)
    epsilon = 0.001
    episodes = 200

    EPS_START = 1  # 0.01
    EPS_END = 1
    EPS_DECAY = 10

    max_steps = 2500  # to make it infinite make sure reach objective

    timestep_reward = Expected_SARSA(
        alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, EPS_DECAY, n_tests=1)
