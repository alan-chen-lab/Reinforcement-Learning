# Problem: A robot needs to go from Start to Target
# Reward: +100 for target, -1 for each other step
# Q-value: for simplicity zero
# Initial Q values are all zero

from operator import index
from matplotlib.pyplot import grid
import numpy as np
import random
import copy
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
        self.action_to_number = {"up": 0, "right":1, "down":2, "left":3, "fly":4}
        self.action_dict = {"up": [-1,0], "right": [0, 1], "down": [1,0], "left":[0,-1], "fly":[0,0]}
        self.direction_dict = {0: "up", 1:"right", 2:"down", 3:"left", 4:"fly"}
        self.invalid_start = ["T", "W"]

    def transfer_state(self, state_coordinates, action): #Input(state, action), Output(next state)
        current_state_coordinates = state_coordinates
        next_state_coordinates = state_coordinates + self.action_dict[action]
        if action=="fly" :
            if (current_state_coordinates==np.array([2,4])).all():
                return np.array([0,1])
            if (current_state_coordinates==np.array([1,3])).all():
                return np.array([1,0])
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
        self.train=False
        self.gamma = 0.9
        self.alpha=0.5
        self.lamda=1
        self.epsilon=0.2
        self.Horizon = 1 #Max episode_length
        self.action_len=len(self.env.direction_dict)
        self.Q_values = {}
        self.trace = {}
        self.init_trace()
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
    def init_trace(self):
        self.trace = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.trace[ ((row, col), act) ] = 0 # eligibility trace 
                        
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
            if np.random.rand()>self.epsilon:
                for valid_action in valid_actions:
                    if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                        indexes.append(self.env.action_to_number[valid_action])
            else:
                for valid_action in valid_actions:
                    indexes.append(self.env.action_to_number[valid_action])
     
            return self.env.direction_dict[random.choice(indexes)]
        else:
            for valid_action in valid_actions:
                if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                    indexes.append(self.env.action_to_number[valid_action])
            return self.env.direction_dict[random.choice(indexes)]
    def to_T(self):
        self.init_trace()
        self.current_state_coordinates = self.generate_initial_state()
        self.action = self.generate_random_action()
        self.rd_state_flag=False
    def iter(self): #Main loop
        next_action=0
        self.train=True
        self.rd_state_flag=False
        self.current_state_coordinates = self.generate_initial_state()
        self.action = self.generate_random_action()
        # move iterration time
        for iterration in range(self.Max_iteration):
            #when arrive T -> random(state,action)
            if self.rd_state_flag:
                self.to_T()  
            #get next (state,action)
            next_state=self.env.transfer_state(self.current_state_coordinates, self.action)
            if (next_state==np.array([0,0])).all():
                self.rd_state_flag=True
            else :
                next_action=self.policy(next_state)
                
            #update all (state,action) by trace
            for row in range(self.env.rows):
                for col in range(self.env.cols):
                    if self.env.grid_world[row][col] not in self.env.invalid_start:
                        for act in self.env.direction_dict.values():
                            
            # update trace
                            if([(row,col),act]==[tuple(self.current_state_coordinates),self.action]):
                                self.trace[((row,col),act)]=self.trace[((row,col),act)]*self.gamma*self.lamda+1
                            else :
                                self.trace[((row,col),act)]=self.trace[((row,col),act)]*self.gamma*self.lamda
                                
            #collect Value function's parameter
                            v=self.Q_values[((row,col),act)]
                            n_state=self.env.transfer_state(np.array([row,col]), act)
                            if (n_state==np.array([0,0])).all():
                                reword=100
                                next_v=0
                            else :
                                reword=-1
                                n_act=self.policy(n_state)
                                next_v=self.Q_values[(tuple(n_state),n_act)]
                                
            #update Value
                            v=v+self.alpha*(reword+self.gamma*next_v-v)*self.trace[((row,col),act)]
                            
            #calculate mean of Value
                            returns = self.returns_dict[((row,col), act)]
                            mean = returns[0]
                            visited_count = returns[1]
                            if [(row,col),act]==[tuple(self.current_state_coordinates),self.action]:
                                mean = (mean*visited_count + v)/(visited_count + 1)
                                visited_count += 1
                            else :
                                if visited_count!=0:
                                    mean = (mean*visited_count + v)/(visited_count + 1)
                                else:
                                    mean=0
                                    
            #save Value
                            self.returns_dict[((row,col), act)] = [mean, visited_count]
                            self.Q_values[((row,col), act)] = self.returns_dict[((row,col), act)][0]
            
            #update(state,action)                                              
            self.action=next_action
            self.current_state_coordinates=next_state
    def render(self): #Show results
        self.train=False
        output = []
        output = copy.deepcopy(self.env.grid_world)
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


        
    








