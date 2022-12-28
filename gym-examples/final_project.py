# Slide Example for Q-LEarning (RL-Course NTNU, Saeedvand)

import os
import gym
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import gym_examples
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import copy


class MyDataset(Dataset):
    def __init__(self, data, target_model, gamma):
        self.memory = data
        self.target_model = target_model

    def __getitem__(self, idx):
        s, s_, a, reward, terminated = self.memory[idx]
        self.gamma = gamma
        self.target_model.eval()
        if terminated:
            label = reward
        else:
            label = reward+self.gamma * self.target_model(s_.unsqueeze(0).to(device)).max().item()

        label = torch.tensor(label).unsqueeze(0)
        action = torch.tensor(a).unsqueeze(0)
        return s, label, action

    def __len__(self):
        return len(self.memory)


class NN(nn.Module):
    def __init__(self ,in_dim,agent_see):
        super(NN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim*3, 64, agent_see, 1, 0),  # [64, agent_see-2, agent_see-2]
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 4),         # [64 * (agent_see-2)^2 , 4]
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def trainer(train_loader, model, epoch_size):
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    # if not os.path.isdir('./models'):
    #     os.mkdir('./models')  # Create directory of saving models.

    for epoch in range(epoch_size):
        model.train()
        loss_record = []
        if show_info:
            train_pbar = tqdm(train_loader, position=0, leave=True)
        else:
            train_pbar = train_loader
        for x, y, a in train_pbar:
            optim.zero_grad()
            pred = model(x.to(device)).gather(1, a.to(device))
            loss = criterion(y.to(device), pred)
            loss.backward()
            optim.step()
            loss_record.append(loss.detach().item())

        mean_train_loss = sum(loss_record)/len(loss_record)
        if show_info:
            train_pbar.set_description(f'Epoch [{epoch+1}/{epoch_size}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
            print(f'mean_loss = {mean_train_loss}')


def plot(rewards):
    plt.figure(2)
    plt.title('Averange Reward')
    plt.xlabel('Episode')
    plt.ylabel('Averange Reward')
    plt.plot(rewards, color='blue', label='Reward')
    plt.grid(axis='x', color='0.80')
    # plt.legend(title='Parameter where:')
    plt.show()


def normalize(list):
    xmin = min(list)
    xmax = max(list)
    for i, x in enumerate(list):
        list[i] = (x-xmin) / (xmax-xmin)
    return list


def state_to_chanel(state):
    size = state[0].size
    matrix_0 = np.zeros([size, size])
    matrix_1 = np.ones([size, size])
    matrix_2 = np.ones([size, size])*2

    chanel_0 = np.equal(matrix_0, state).astype('float32')
    chanel_1 = np.equal(matrix_1, state).astype('float32')
    chanel_2 = np.equal(matrix_2, state).astype('float32')

    chanel_0 = torch.tensor(chanel_0).unsqueeze(0)
    chanel_1 = torch.tensor(chanel_1).unsqueeze(0)
    chanel_2 = torch.tensor(chanel_2).unsqueeze(0)

    return torch.cat([chanel_0, chanel_1, chanel_2], 0)


def epsilon_greedy(model, epsilon, s, see_target):
    if np.random.rand() < epsilon:
        model.eval()
        action = model(s.unsqueeze(0).to(device))
        action.to("cpu")
        action = action.cpu().detach().numpy()
        see_target=True
        if not see_target:
            action *= 1
        action = action.argmax().item()
    else:
        action = env.action_space.sample()

    return action


def greedy(s, model, see_target):
    model.eval()
    action = model(s.unsqueeze(0).to(device))
    action.to("cpu")
    action = action.cpu().detach().numpy()
    see_target=True
    if not see_target:
        action *= 1
    action = action.argmax().item()
    return action


class trace_mem:
    def __init__(self, trace_size):
        self.temp = []
        self.size = trace_size
        self.iter = 0

    def push(self, action):
        if(self.size != 0):
            if len(self.temp) < self.size:
                self.temp.append(action)
            else:
                self.temp[self.iter] = action
            self.iter = (self.iter+1) % self.size

    def get(self):
        self.trace = np.ones((1, 4))
        for i in self.temp:
            self.trace[0][(i+2) % 4] *= 0.9
        return self.trace
    
    def get_list(self):
        self.trace_list=[]
        for i in range(self.size):
            self.trace_list.append(self.temp[(self.iter-i-1) % self.size])
        return self.trace_list
            
class state_mem:
    def __init__(self,mem_size,agent_see):
        self.temp=[]
        self.size= mem_size
        self.iter = 0
        self.agent_see=agent_see
    def push(self,state):
        if(self.size != 0):
            if len(self.temp) < self.size:
                self.temp.append(state)
            else:
                self.temp[self.iter] = state
            self.iter = (self.iter+1) % self.size
            
    def get(self):
        self.state_list=[]
        temp=torch.tensor([])
        
        for i in range(self.size):
            if i < len(self.temp):
                self.state_list.append(self.temp[(self.iter-i-1) % self.size])
            else :
                empty_state = torch.tensor(np.zeros([3,self.agent_see,self.agent_see]).astype("float32"))
                self.state_list.append(empty_state)
        x=0
        for i in self.state_list:
            x+=1
            i=i*(0.9**x)
            temp=torch.cat((temp,i),0)
        return temp
    
class replay_memory:
    def __init__(self):
        self.memory = []
        self.size = 5000
        self.mem_iter = 0

    def push(self, data):
        if len(self.memory) < self.mem_iter+1:
            self.memory.append(data)
        else:
            self.memory[self.mem_iter] = data
        self.mem_iter = (self.mem_iter+1) % self.size


def state_see_target(state):
    size = state[0].size
    temp = np.zeros(size)
    return not np.array_equal(temp, state[2])


def Qlearning(epoch_size, gamma, episodes, max_steps, EPS_START, EPS_END, EPS_DECAY, n_tests, trace_size,state_mem_size,agent_see=5):
    #n_states, n_actions = env.observation_space.n, env.action_space.n
    timestep_reward = []

    model = NN(state_mem_size,agent_see).to(device)

    # model.load_state_dict(torch.load("./model/target_model.ckpt"))  #check point
    memory = replay_memory()
    # print(s[2])
    t = 0
    for episode in range(episodes):
        print(
            f"Episode: {episode}---------------------------------------------------------------------------------------------------------")
        epsilon_threshold = EPS_END + \
            (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)
        
        if episode<50:
            epsilon_threshold=0
        
        x = 0
        # last_a=None
        trace_list = trace_mem(trace_size)
        states=state_mem(state_mem_size-1,agent_see)
        s, info = env.reset()  # read also state
        see_target = state_see_target(s)
        s = state_to_chanel(s)
        total_reward = 0
        while True:
            x += 1
            if t % 100 == 0:
                target_model = copy.deepcopy(model)
                # Save your best model
                # torch.save(target_model.state_dict(),
                #            './model/target_model.ckpt')
            t += 1
            ss = states.get()
            ss = torch.cat((s,ss),0)
            #a = epsilon_greedy(target_model, epsilon_threshold, s, see_target)
            a = epsilon_greedy(target_model, epsilon_threshold, ss, see_target)
            states.push(s)
            trace_list.push(a)
            # 150 win
            # -50 hit the wall
            # -10 move
            s_, reward, terminated, truncated, info = env.step(a)
            
            if terminated:
                reward=reward*((max_steps-x)/max_steps)+100
            '''
            if x==max_steps:
                reward=-200
            '''
            see_target = state_see_target(s_)
            total_reward += reward
            # last_a=a
            s_ = state_to_chanel(s_)
            #reward = -200 if x>max_steps else reward
            # 想辦法 記住部份路徑
            ss_ = states.get()
            ss_=torch.cat((s_,ss_),0)
            a_next = greedy(ss_, model, see_target)

            memory.push((ss, ss_, a, reward, terminated))
            temp = (ss, ss_, a, reward, terminated)
            s, a = s_, a_next
            # train
            if t > memory.size:
                # choose data
                data = []
                idxs = np.random.choice(range(len(memory.memory)), 63, True)
                for idx in idxs:
                    data.append(memory.memory[idx])
                data.append(temp)
                # train800
                train_dataset = MyDataset(data, model, gamma)
                train_loader = DataLoader(
                    train_dataset, batch_size=64, shuffle=True, pin_memory=False)
                trainer(train_loader, model, epoch_size)
            # one round
            if terminated or truncated or x >= max_steps:  # win/lose/time up
                print(f"total_reward = {total_reward}")
                s, info = env.reset()
                s = state_to_chanel(s)
                break

        timestep_reward.append(total_reward)

    # Test policy (no learning)
    if n_tests > 0:
        test_agent(model, n_tests, max_steps=max_steps, trace_size=trace_size,state_mem_size=state_mem_size,agent_see=agent_see)
    env.close()
    plot(normalize(timestep_reward))
    return timestep_reward
# ----------------------------------------------------


def test_agent(model, n_tests=0, delay=0.4, max_steps=30, trace_size=3,state_mem_size=3,agent_see=5):
    env.change_render("human")
    for testing in range(n_tests):
        print(f"Test #{testing}")
        s, info = env.reset()
        see_target = state_see_target(s)
        s = state_to_chanel(s)
        states=state_mem(state_mem_size-1,agent_see)
        # last_a=None
        x = 0
        trace_list = trace_mem(trace_size)
        total_reward = 0
        while True:
            x += 1
            time.sleep(delay)
            #a = np.argmax(Q[s, :]).item()
            ss = states.get()
            ss = torch.cat((s,ss),0)
            states.push(s)
            a = greedy(ss, model, see_target)
            
            trace_list.push(a)
            print(f"Chose action {a} for state \n{s}")
            s, reward, terminated, truncated, info = env.step(a)
            see_target = state_see_target(s)
            total_reward += reward
            # last_a=a
            s = state_to_chanel(s)
            # time.sleep(1)

            if terminated or truncated or x >= max_steps:
                print("Finished!", total_reward)
                time.sleep(2)
                break


if __name__ == "__main__":
    gamma = 0.9     # discount factor
    episodes = 2000  # round
    epoch_size = 1  # NN model

    EPS_START = 0.8
    EPS_END = 0.9
    EPS_DECAY = 100

    wall_num = 12 # obstacles num
    real_size = 9 # full map size
    agent_size = 5 # agent view

    state_mem_size = 6 # NN state memory(1 current, 5 previous)
    trace_size = 10 # trace memory

    target_move = True
    show_info = False
    max_steps = 75  # to make it infinite make sure reach objective

    env = gym.make('gym_examples/GridWorld-v0', size=real_size,
                   agent_see=agent_size, wall=wall_num, move=target_move)
    
    #env.change_render("human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestep_reward, episodes_list = Qlearning(epoch_size, gamma, episodes, max_steps,
                                EPS_START, EPS_END, EPS_DECAY, n_tests=2, trace_size=trace_size,state_mem_size=state_mem_size,agent_see=agent_size)
