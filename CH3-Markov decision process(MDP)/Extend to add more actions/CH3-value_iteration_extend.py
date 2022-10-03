# Slide Example for extend Value Iteration add back actions (Saeevand)
import matplotlib.pyplot as plt
import numpy as np

states = 8  # number of states
A = ['R', 'L', 'B_R', 'B_L']  # actions, addd back right & back left
actions = 4

# In case that reward can be different to be in one state with
# different actions we candefine them seperately, in our example both
# left and rigth actions lead to same reward in individual state [State, State, action]
Reward = [[0, 0, 0, 0], [2, 2, 0, 0], [1, 1, 0, 0], [-1, -1, 0, 0],
          [3, 3, 0, 0], [-3, -3, 0, 0], [-7, -7, 0, 0], [5, 5, 0, 0]]
# Reward = [[0, 0, 0, 0], [2, 2, 2, 2], [1, 1, 1, 1], [-1, -1, -1, -1],
#           [3, 3, 3, 3], [-3, -3, -3, -3], [-7, -7, -7, -7], [5, 5, 5, 5]]
TransitionProbaility = [  # [Right, Left, back right, back left]
    [[0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s1

    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0],
     [0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # s2

    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],   # s3
     [0.3, 0.7, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],

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

sum = []
iter = []

for iteration in range(0, 100):
    print(str(iteration) + ': ', Value[0], Value[1], Value[2], Value[3],  Value[4],  Value[5],
          Value[6],  Value[7], 'bellman_factor (' + str(bellman_factor) + ')', sep=",    ")
    NewValue = [-1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12]
    for i in range(states):
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbaility[i][j][a] * Value[j]
            value_temp *= gamma
            value_temp += Reward[i][a]
            NewValue[i] = round(max(NewValue[i], value_temp), 2)
    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value[i]-NewValue[i]))
    Value = NewValue

    sum.append(np.sum(Value))
    iter.append(iteration)

    if(bellman_factor < delta):
        break

# Determine the policy (One time iteration)
NewValue = [-1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12, -1e12]
policy = ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
Terminal = ['', '', '', 'T', '', '', 'T', 'T']
for i in range(states):
    for a in range(actions):
        value_temp = 0
        for j in range(states):
            value_temp += TransitionProbaility[i][j][a] * Value[j]
        value_temp *= gamma
        value_temp += Reward[i][a]
        if(NewValue[i] < value_temp):
            if(Terminal[i] != 'T'):
                policy[i] = A[a]
                NewValue[i] = max(NewValue[i], value_temp)
            else:
                policy[i] = 'T'

print("Policy Iteration algoirthm's final policy is:", policy)
plt.figure(2)
plt.title('converge speed')
plt.xlabel('iteration')
plt.ylabel('sum of value')
plt.plot(iter, sum, color='green', label='sum of value')
plt.grid(axis='x', color='0.80')
plt.legend()
plt.show()
