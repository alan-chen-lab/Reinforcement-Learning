## [Final Project.py](https://github.com/alan-chen-lab/Reinforcement-Learning/blob/main/gym-examples/final_project.py)
### parameters
- `wall_num`: obstacles num
- `real_size`: full map size
- `agent_sie`: agent view
- `state_mem_size`: NN state memory(1 current, 5 previous)
- `trace_size`: trace memory
- `max_steps`: to make it infinite make sure reach objective
- `epoch_size`: NN model train
- `gamma`: discount factor
- `episodes`: round

### functions introduction
- `Qlearning()`: DQN policy
- `class NN(nn.Module)`: design NN
- `class replay_memory`: record memory(s, s_next, action, reward, terminated)
- `class trace_mem`: if there are no target in th move way, then punish th opposite direction of the action value
- `class state_mem`: state memory of NN, we consider the previous states so that the agent has memory
- `def state_to_chanel(state)`: make pre-process to define each channel as 0, 1, and 2 (0 is free space, 1 is obstacle, and 2 is target) for extract 
- `def epsilon_greedy(model, epsilon, s, see_target)`: epsilon_greedy
- `def greedy(s, model, see_target)`: choose the max value of action
- `class MyDataset(Dataset)`: load dataset
- `def trainer(train_loader, model, epoch_size)`: use SGD and cauculate the loss function

***

## Gym Examples
Some simple examples of Gym environments and wrappers.
For some explanations of these examples, see the [Gym documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).
