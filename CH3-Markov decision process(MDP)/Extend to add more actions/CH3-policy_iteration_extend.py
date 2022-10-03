# Slide Example for Extend Policy Iteration (Saeevand)
import matplotlib.pyplot as plt
import numpy as np

states = 8  # number of states
A = ['R', 'L', 'B_R', 'B_L']  # actions
actions = 4

policy = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
Terminal = ['', '', '', 'T', '', '', 'T', 'T']

# In case that reward can be different to be in one state with
# different actions we candefine them seperately, in our example both
# left and rith actions lead to same reward in individual state [State, State, action]
Reward = [[0, 0, 0, 0], [2, 2, 2, 2], [1, 1, 1, 1], [-1, -1, -1, -1],
          [3, 3, 3, 3], [-3, -3, -3, -3], [-7, -7, -7, -7], [5, 5, 5, 5]]
TransitionProbaility = [  # [Right, Left, back right, back left]
    [[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s1

    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0],
     [0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s2

    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
     [0.3, 0.7, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s3

    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s4

    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3, 0.7], [0.0, 0.0, 0.7, 0.3], [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0]],  # s5

    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],  # s6

    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s7

    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]   # s8
]

Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Value estimation of each state

# -------------------------------------

gamma = 0.9
bellman_factor = 0
delta = 0.01

for iteration in range(0, 100):
    print(str(iteration) + ': ', Value[0], Value[1], Value[2], Value[3],  Value[4],  Value[5],
          Value[6],  Value[7], 'bellman_factor (' + str(bellman_factor) + ')', sep=",    ")
    NewValue = [-1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12]

    # step1: Policy evaluation
    for i in range(states):
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbaility[i][j][a] * Value[j]
            value_temp *= gamma
            value_temp += Reward[i][a]
            NewValue[i] = round(value_temp, 2)

    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value[i]-NewValue[i]))
    Value = NewValue
    if(bellman_factor < delta):
        break

    print(str(iteration) + ': ', Value[0], Value[1], Value[2], Value[3],  Value[4],  Value[5],
          Value[6],  Value[7], 'bellman_factor (' + str(bellman_factor) + ')', sep=",    ")
    print("( " + str(iteration + 1) + " iteration )")

    # step2: Policy improment
    policy_stable = True
    for i in range(states):
        old_policy = {policy[i]}.copy()
        q_s = np.zeros(actions)
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbaility[i][j][a] * Value[j]
            value_temp *= gamma
            value_temp += Reward[i][a]
            q_s[a] = value_temp

        best_action = np.argmax(q_s)
        if(Terminal[i] != 'T'):
            policy[i] = A[best_action]
        else:
            policy[i] = 'T'

        if np.any(old_policy != policy[i]):
            policy_stable = False

    if policy_stable == False:
        break


print("Policy Iteration algoirthm's final policy is:", policy)
