# Problem: A robot needs to go from Start to Target
# Reward: +100 for target, -1 for each other step
# Q-value: for simplicity zero
# Initial Q values are all zero
import copy
from operator import index
from matplotlib.pyplot import grid
import numpy as np
import random
np.random.seed(9527)
random.seed(9527)

class Environment():
    def __init__(self):
        self.grid_world = [[  "T",  "s1",  "s2",  "s3",  "s4",  "s5"],
                           [ "s6",  "s7",  "s8",  "s9",   "W", "s10"],
                           ["s11",   "W", "s12",   "W", "s13", "s14"],
                           ["s15", "s16", "s17", "s18", "s19", "s20"]] #T: Target, W: Wall
        self.rows = len(self.grid_world)        #4
        self.cols = len(self.grid_world[0])     #6
        self.action_to_number = {"up": 0, "right":1, "down":2, "left":3}
        self.action_dict = {"up": [-1,0], "right": [0, 1], "down": [1,0], "left":[0,-1]}
        self.direction_dict = {0: "up", 1:"right", 2:"down", 3:"left"}
        self.invalid_start = ["T", "W"]
        
    def transfer_state(self, state_coordinates, action): #Input(state, action), Output(next state)
        current_state_coordinates = state_coordinates
        next_state_coordinates = state_coordinates + self.action_dict[action]
        
        if next_state_coordinates[0] < 0 or next_state_coordinates[0] >= self.rows or next_state_coordinates[1] < 0 or next_state_coordinates[1] >= self.cols:# Out of board
            return current_state_coordinates
        next_state = self.grid_world[next_state_coordinates[0]][next_state_coordinates[1]]
        if next_state == "W": #Hit the wall 
            return current_state_coordinates
        return next_state_coordinates

class Monte_Carlo():
    def __init__(self):
        self.env = Environment()
        self.Max_iteration = 10000
        self.gamma = 0.9
        self.alpha=  0.5
        self.Horizon = 10 #Max episode_length
        self.action_len=len(self.env.direction_dict)
        self.epsilon=0.2
        self.Q_values = {}
        #self.ep=self.epsilon/6000000
        self.train=False
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.Q_values[ ((row, col), act) ] = 0 #Initialize Q value (state, action) 

        self.returns_dict = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.returns_dict[ ((row, col), act) ] = [0, 0] #[Mean value, Visited count]

    def generate_initial_state(self): #Generate randomm state
        while True:
            state_row = np.random.randint(self.env.rows)
            state_col = np.random.randint(self.env.cols)
            if self.env.grid_world[state_row][state_col] in self.env.invalid_start:
                continue
            else:
                break
        return np.array([state_row, state_col])

    def generate_random_action(self):
        action = self.env.direction_dict[np.random.randint(self.action_len)]
        while (self.env.transfer_state(self.current_state_coordinates, action) == self.current_state_coordinates).all():
                action = self.env.direction_dict[np.random.randint(self.action_len)]
        return action

    def policy(self, state_coordinate): #Optimal policy, find the maximun Q(s,a) and return action  
        Q_value = []
        valid_actions = []
        state_coordinate = np.array(state_coordinate)
        indexes = []
        
        for action in self.env.action_dict.keys():
            if not (state_coordinate == self.env.transfer_state(np.array(state_coordinate), action)).all():
                valid_actions.append(action)

        for valid_action in valid_actions:
            Q_value.append(self.Q_values[(tuple(state_coordinate), valid_action)])
        max_value = max(Q_value)
        
              
        
        if self.train:
            if np.random.rand() >= self.epsilon:
                for valid_action in valid_actions:
                    if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                        indexes.append(self.env.action_to_number[valid_action])
                return self.env.direction_dict[random.choice(indexes)]  
                # indexes = [index for index,x in enumerate(Q_value) if x == max_value]
            else :
                for valid_action in valid_actions:
                    indexes.append(self.env.action_to_number[valid_action])
                return self.env.direction_dict[random.choice(indexes)]  
        else :
            for valid_action in valid_actions:
                if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                    indexes.append(self.env.action_to_number[valid_action])
            return self.env.direction_dict[random.choice(indexes)]  
    def to_T(self):
        self.current_state_coordinates = self.generate_initial_state()
        self.action = self.generate_random_action()
        self.rd_state_flag=False 
    
    
    def iter(self): #Main loop
        next_action=0
        self.train=True
        self.rd_state_flag=False
        self.current_state_coordinates = self.generate_initial_state()
        self.action = self.generate_random_action()
        
        #start train
        for iterration in range(self.Max_iteration):
            episode = []
            
            #when arrive T  -> init
            if self.rd_state_flag:
                self.to_T()  
            
            for h in range(self.Horizon): #Generate episode
                next_state_coordinates = self.env.transfer_state(self.current_state_coordinates, self.action)
                reward = -1
                if self.env.grid_world[next_state_coordinates[0]][next_state_coordinates[1]] == "T":
                    reward = 100
                    self.rd_state_flag=True
                    episode.append([self.current_state_coordinates, self.action, reward])
                    break
                episode.append([self.current_state_coordinates, self.action, reward]) 
                self.current_state_coordinates = next_state_coordinates
                self.action = self.policy(self.current_state_coordinates)
            G = 0

            for h in range(len(episode)-1, -1, -1): #Iterate through H-1, H-2,...,0
                coordinate, action, reward = episode[h]
                
            #collect Value function's parameter
                G = self.gamma*G + reward
                v=self.Q_values[(tuple(coordinate),action)]
                
            #calculate Value
                v=v+self.alpha*(G-v)
                
            #calculate mean of Value
                returns = self.returns_dict[(tuple(coordinate), action)]
                mean = returns[0]
                visited_count = returns[1]
                mean = (mean*visited_count + v)/(visited_count + 1)
                visited_count += 1
                
            #save Value
                self.returns_dict[(tuple(episode[h][0] ), episode[h][1])] = [mean, visited_count]
                self.Q_values[(tuple(episode[h][0]), episode[h][1])] = mean
            

    def render(self): #Show results
        self.train=False
        output=copy.deepcopy(self.env.grid_world)
        #output = self.env.grid_world

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] in self.env.invalid_start:
                    continue
                else:
                    action = self.policy((row,col))
                    output[row][col] = action
        
        for row in range(0, self.env.rows):
            print("-------------------------------------------------------")
            out = "| "
            for col in range(0, self.env.cols):
                out += str(output[row][col]).ljust(6) + " | "
            print(out)
        print("-------------------------------------------------------")

   

if __name__ == "__main__":
    print("Find optimal policy using Monte Carlo algorithm with exploring starts")
    monte = Monte_Carlo()
    print("Before (random policy), T = Target, W = Wall")
    monte.render()
    monte.iter()
    print("\nOptimal policy, T = Target, W = Wall")
    monte.render()
    print("\n\n\n")








